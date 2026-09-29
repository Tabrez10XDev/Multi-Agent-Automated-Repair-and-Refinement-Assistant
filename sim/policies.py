"""
Schedulers and retention policies.

ORACLE DISCIPLINE (HANDOFF.md section 10)
-----------------------------------------
The trace contains the future: loop trip counts are sampled up front, so
Node.next_reuse_node, Node.next_reuse_gap_s and Node.critical_path_remaining are
oracle fields. Only the offline Belady baseline and the hint builder may read
them.

Enforcement here is structural rather than by convention: online schedulers are
handed ReqView objects, which simply do not carry those attributes, so an oracle
read raises AttributeError at the point of the mistake rather than silently
producing an unbeatable policy. OnlineNodeView additionally guards anything that
does get handed a raw Node, and raises OracleLeakError with a pointed message.
`assert_no_oracle_leak()` checks the views at import time.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional

from sim.workload import CODER_MODEL, CostModel, Kind

ORACLE_FIELDS = frozenset({
    "next_reuse_node", "next_reuse_gap_s", "critical_path_remaining",
})


class OracleLeakError(RuntimeError):
    """An online policy tried to read a field that encodes the future."""


class OnlineNodeView:
    """Facade over a Node that refuses oracle attributes."""
    __slots__ = ("_node",)

    def __init__(self, node):
        object.__setattr__(self, "_node", node)

    def __getattr__(self, name):
        if name in ORACLE_FIELDS:
            raise OracleLeakError(
                f"online policy read oracle field {name!r}; only Belady and the "
                f"hint builder may see the future (HANDOFF.md section 10)")
        return getattr(self._node, name)


@dataclass
class ReqView:
    """
    Everything an online scheduler is allowed to know about a queued request.

    Deliberately excludes output_tokens. In a real serving system you do not know
    how many tokens a request will emit before it emits them, so a scheduler that
    reads the true value is cheating in a way the oracle-field guard would not
    catch. Schedulers use ExecutionEstimator instead.
    """
    req_id: int
    task_id: int
    stage: str
    model: str
    enqueued_at: float
    prompt_tokens: int
    attempt_index: int
    priority: int
    deadline: Optional[float]
    est_service_s: float
    est_remaining_s: float


def assert_no_oracle_leak():
    leaked = ORACLE_FIELDS & set(ReqView.__dataclass_fields__)
    if leaked:
        raise OracleLeakError(f"ReqView exposes oracle fields: {sorted(leaked)}")


assert_no_oracle_leak()


# --------------------------------------------------------------------------
# Online execution-time estimation
# --------------------------------------------------------------------------

# Rough stage ordering of MAARA's pipeline, used to estimate remaining work.
# This is a-priori structural knowledge of the harness, not the realized path:
# it says what stages EXIST, never how many times this particular task will
# actually loop.
_STAGE_ORDER = ["parse_issue", "navigate", "evaluate_plan",
                "generate_fix", "apply_and_test", "diagnose"]


class ExecutionEstimator:
    """
    Supplies the a-priori execution-time estimates that EDF and least-slack-first
    require, without touching the trace's realized future.

    Three legitimate sources:
      1. CostModel physics applied to prompt_tokens, which is known at enqueue.
      2. Output lengths learned ONLINE from requests that have already finished,
         as a per-stage running mean. Cold-started from StageTokens priors.
      3. The empirical retry distribution from benchmark/results.jsonl, which is
         calibration data measured before the run, not knowledge of this task.

    The weak point, and it should be stated in the writeup: remaining work
    depends on how many more retry attempts a task will need, which is genuinely
    unknowable online. We use the conditional expectation given attempts so far.
    A task that is about to succeed and one that will burn all three attempts
    look identical to this estimator, which is exactly the uncertainty a real
    scheduler faces.
    """

    def __init__(self, cost: CostModel, tok, cpu, attempt_weights=(74, 13, 33)):
        self.cost = cost
        self.tok = tok
        self.cpu = cpu
        self.attempt_weights = attempt_weights
        self._out_sum = {}
        self._out_n = {}
        self._prior = {
            "parse_issue": tok.issue_output_tokens,
            "evaluate_plan": tok.eval_output_tokens,
            "generate_fix": tok.fix_output_tokens,
            "diagnose": tok.diagnose_output_tokens,
        }

    def observe(self, stage: str, output_tokens: int) -> None:
        self._out_sum[stage] = self._out_sum.get(stage, 0) + output_tokens
        self._out_n[stage] = self._out_n.get(stage, 0) + 1

    def expected_output(self, stage: str) -> float:
        if self._out_n.get(stage):
            return self._out_sum[stage] / self._out_n[stage]
        return self._prior.get(stage, 250)

    def service_s(self, model: str, prompt_tokens: int, stage: str) -> float:
        out = self.expected_output(stage)
        return (self.cost.prefill_seconds(model, prompt_tokens)
                + out * self.cost.decode_iteration_seconds(model, 1))

    def _expected_further_attempts(self, attempt_index: int) -> float:
        """E[remaining attempts | already on attempt `attempt_index`]."""
        w = self.attempt_weights
        tail = [(a, w[a - 1]) for a in range(max(attempt_index, 1), len(w) + 1)]
        total = sum(x for _, x in tail)
        if total <= 0:
            return 0.0
        return sum((a - attempt_index) * x for a, x in tail) / total

    def remaining_s(self, stage: str, attempt_index: int, model: str) -> float:
        """
        Expected work left in this task from `stage` onward, in seconds.
        Used as the critical-path estimate for slack.
        """
        try:
            idx = _STAGE_ORDER.index(stage)
        except ValueError:
            idx = 0

        est = 0.0
        # Remainder of the current pass through the pipeline.
        for s in _STAGE_ORDER[idx + 1:]:
            if s == "navigate":
                est += self.cpu.rag_index_seconds
            elif s == "apply_and_test":
                est += self.cpu.pytest_seconds
            else:
                est += self.service_s(model, self._typical_prompt(s), s)

        # Plus however many more retry loops we expect to run.
        loop = (self.service_s(model, self._typical_prompt("generate_fix"), "generate_fix")
                + self.cpu.pytest_seconds
                + self.service_s(model, self._typical_prompt("diagnose"), "diagnose"))
        est += self._expected_further_attempts(attempt_index) * loop
        return est

    def _typical_prompt(self, stage: str) -> int:
        t = self.tok
        return {
            "parse_issue": t.issue_prefix_tokens + t.issue_suffix_tokens,
            "evaluate_plan": t.eval_prompt_tokens,
            "generate_fix": t.stable_prefix_tokens + t.fix_suffix_retry_tokens,
            "diagnose": t.diagnose_prompt_tokens,
        }.get(stage, 500)


# --------------------------------------------------------------------------
# Schedulers
# --------------------------------------------------------------------------

class Scheduler:
    """Returns req_ids in the order they should be considered for admission."""
    name = "base"
    needs_deadlines = False

    def order(self, views, now: float):
        raise NotImplementedError


class FIFOScheduler(Scheduler):
    """Arrival order. The incumbent."""
    name = "fifo"

    def order(self, views, now):
        return [v.req_id for v in sorted(views, key=lambda v: v.enqueued_at)]


class StaticPriorityScheduler(Scheduler):
    """
    Fixed per-stage priority, tie-broken by arrival.

    The ordering encodes a plausible operator intuition: unblock work that is
    nearly done, and get the cheap gate out of the way. It is a strawman for
    least-slack-first precisely because it is workflow-blind -- it cannot tell a
    task on its last attempt from one that just arrived.
    """
    name = "static-priority"
    _RANK = {"apply_and_test": 0, "diagnose": 1, "generate_fix": 2,
             "evaluate_plan": 3, "parse_issue": 4}

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (self._RANK.get(v.stage, 9), v.enqueued_at))]


class EDFScheduler(Scheduler):
    """Earliest deadline first. Deadlines come from the estimator, not the trace."""
    name = "edf"
    needs_deadlines = True

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.deadline if v.deadline is not None else math.inf,
                                  v.enqueued_at))]


class LeastSlackFirstScheduler(Scheduler):
    """
    Slack = deadline - now - estimated remaining work on the task's critical path.

    The difference from EDF is the whole point of the mechanism. EDF orders by
    deadline alone, so a request with a distant deadline but a long tail of
    dependent work behind it gets deprioritised even when it is about to become
    the bottleneck. Slack accounts for that tail, and the tail comes from the
    workflow DAG.

    est_remaining_s is computed by ExecutionEstimator from stage position and the
    empirical retry distribution -- never from Node.critical_path_remaining,
    which is oracle.
    """
    name = "least-slack-first"
    needs_deadlines = True

    def order(self, views, now):
        def slack(v):
            if v.deadline is None:
                return math.inf
            return v.deadline - now - v.est_remaining_s
        return [v.req_id for v in sorted(views, key=lambda v: (slack(v), v.enqueued_at))]


SCHEDULERS = {
    s.name: s for s in
    [FIFOScheduler, StaticPriorityScheduler, EDFScheduler, LeastSlackFirstScheduler]
}
