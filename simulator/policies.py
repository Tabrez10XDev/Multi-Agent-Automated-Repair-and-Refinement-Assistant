"""
Schedulers, retention policies, and the online execution-time estimator.

WHAT THE SCHEDULER CONTROLS
---------------------------
Exactly one decision: **when a batch slot and KV space are free, which waiting
request is admitted next.**

Everything else is fixed:

  * Running requests are NEVER preempted. Once admitted, a request holds its
    slot and its KV until it finishes. A scheduler cannot evict, pause, or
    reorder work that is already running.
  * Priorities are recomputed from scratch on every engine iteration. A
    scheduler returns an ordering, not a commitment -- a request that was first
    last iteration may be third this one, because `now` advanced and slack
    shrank. Schedulers must therefore be pure functions of (views, now).
  * The scheduler does not decide whether a request *fits*. It proposes an
    order; the engine walks that order and admits what fits. A request that
    does not fit is skipped, not reordered around.
  * The scheduler cannot see the future. See ORACLE DISCIPLINE.

ORACLE DISCIPLINE (HANDOFF.md section 10)
-----------------------------------------
The trace contains the future: loop trip counts are sampled up front, so
Node.next_reuse_node, Node.next_reuse_gap_s and Node.critical_path_remaining
encode what will happen. Node.output_tokens is equally oracular for scheduling
purposes -- a real server does not know how many tokens a request will emit
before it emits them.

Enforcement is structural rather than by convention: schedulers are handed
ReqView objects, which do not carry those attributes at all, and whose
__getattr__ raises OracleLeakError with a pointed message if one is requested.
Only the offline reference policy and the hint builder may read the real Node.
"""

from __future__ import annotations

import math
import random
import statistics
from dataclasses import dataclass
from typing import Optional

from simulator.workload import CODER_MODEL, CostModel

# Fields on a Node that encode the future.
ORACLE_FIELDS = frozenset({
    "next_reuse_node", "next_reuse_gap_s", "critical_path_remaining",
})

# Additionally forbidden to schedulers. output_tokens is not "the future" in the
# trace-construction sense, but knowing a request's exact output length before
# running it is information no real scheduler has.
FORBIDDEN_FOR_SCHEDULERS = ORACLE_FIELDS | {
    "output_tokens", "node", "true_output_tokens", "expected_seconds",
}


class OracleLeakError(RuntimeError):
    """An online policy tried to read information it cannot have at run time."""


class OnlineNodeView:
    """Facade over a Node that refuses oracle attributes."""
    __slots__ = ("_node",)

    def __init__(self, node):
        object.__setattr__(self, "_node", node)

    def __getattr__(self, name):
        if name in FORBIDDEN_FOR_SCHEDULERS:
            raise OracleLeakError(
                f"online policy read forbidden field {name!r}; only the offline "
                f"reference and the hint builder may see it")
        return getattr(self._node, name)


@dataclass
class ReqView:
    """
    Everything an online scheduler is allowed to know about a queued request.

    Note what is absent: output_tokens, and anything derived from the realized
    trace. est_service_s and est_remaining_s come from ExecutionEstimator, which
    learns from completions rather than reading ahead.
    """
    req_id: int
    task_id: int
    stage: str
    model: str
    enqueued_at: float
    prompt_tokens: int
    attempt_index: int
    replan_index: int
    priority: int
    deadline: Optional[float]
    est_service_s: float
    est_remaining_s: float
    # Observable, not oracle: when the task arrived, and the SLO class its
    # client declared (None and 0 when the run has no classes).
    task_arrival_s: float = 0.0
    slo_class: Optional[str] = None
    slo_priority: int = 0
    # Observable, not oracle: GPU wall-clock (admission to completion) of the
    # task's finished requests so far. Past service, never remaining work.
    task_attained_s: float = 0.0
    # Observable, not oracle: how many of the task's requests have finished.
    # The Gittins state (DECISIONS 38); a count of the past only.
    task_requests_done: int = 0

    def __getattr__(self, name):
        # Only reached when normal dataclass lookup fails.
        if name in FORBIDDEN_FOR_SCHEDULERS:
            raise OracleLeakError(
                f"scheduler read {name!r} off a ReqView. Schedulers must work "
                f"from estimates (est_service_s / est_remaining_s), never from "
                f"true values. See ORACLE DISCIPLINE in simulator/policies.py.")
        raise AttributeError(name)


def assert_no_oracle_leak():
    leaked = FORBIDDEN_FOR_SCHEDULERS & set(ReqView.__dataclass_fields__)
    if leaked:
        raise OracleLeakError(f"ReqView exposes forbidden fields: {sorted(leaked)}")


assert_no_oracle_leak()


# --------------------------------------------------------------------------
# Execution-time estimation
# --------------------------------------------------------------------------

# Stage order of MAARA's pipeline. This is a-priori structural knowledge of the
# harness -- what stages exist -- never how many times this task will loop.
_STAGE_ORDER = ["parse_issue", "navigate", "evaluate_plan",
                "generate_fix", "apply_and_test", "diagnose"]


class ExecutionEstimator:
    """
    Supplies the a-priori execution-time estimates EDF, SJF and LSF need.

    Three legitimate information sources:
      1. CostModel physics applied to prompt_tokens, known at enqueue time.
      2. Output lengths learned ONLINE from completed requests, as a running
         mean per (stage, attempt_index). Cold-started from StageTokens priors.
      3. The empirical retry distribution from data/outcomes/results.jsonl, which is
         calibration measured before the run, not knowledge of this task.

    Keying the running mean by (stage, attempt_index) matters: a retry prompt
    carries the previous fix plus a diagnosis, so attempt 2's coder output is
    not drawn from the same distribution as attempt 1's. Falls back to the
    stage-only mean, then to the StageTokens prior.

    The honest weakness, worth stating in the writeup: remaining work depends on
    how many more retry attempts a task will need, which is genuinely unknowable
    online. We use the conditional expectation given attempts so far, so a task
    about to succeed and one that will burn all three look identical. That is
    exactly the uncertainty a real scheduler faces.
    """

    is_oracle = False
    name = "online"

    def __init__(self, cost: CostModel, tok, cpu, attempt_weights=(74, 13, 33),
                 noise_sigma: float = 0.0, seed: int = 0):
        self.cost = cost
        self.tok = tok
        self.cpu = cpu
        self.attempt_weights = attempt_weights
        self.noise_sigma = noise_sigma
        self._rng = random.Random(seed)
        self._sum = {}
        self._n = {}
        self._noise = {}
        self._prior = {
            "parse_issue": tok.issue_output_tokens,
            "evaluate_plan": tok.eval_output_tokens,
            "generate_fix": tok.fix_output_tokens,
            "diagnose": tok.diagnose_output_tokens,
        }

    # -- learning ---------------------------------------------------------

    def observe(self, stage: str, attempt_index: int, output_tokens: int) -> None:
        """Called by the engine on every completion. Updates both key levels."""
        for key in ((stage, attempt_index), (stage, None)):
            self._sum[key] = self._sum.get(key, 0) + output_tokens
            self._n[key] = self._n.get(key, 0) + 1

    def predict_output(self, stage: str, attempt_index: int = None,
                       true_output: int = None) -> float:
        """
        Predicted output length. `true_output` is accepted but DELIBERATELY
        ignored here -- only OracleEstimator uses it. There is a test asserting
        this method's result does not change when true_output changes.
        """
        for key in ((stage, attempt_index), (stage, None)):
            if self._n.get(key):
                return self._sum[key] / self._n[key]
        return float(self._prior.get(stage, 250))

    # -- noise ------------------------------------------------------------

    def noise_factor(self, req_id) -> float:
        """
        Per-request multiplicative error, drawn once and cached.

        Drawn per request rather than per call because priorities are
        recomputed every iteration: a fresh draw each time would make one
        request's estimate jitter wildly during its own queue wait, which is not
        how a real predictor behaves. A persistent per-request bias is.
        """
        if self.noise_sigma <= 0 or req_id is None:
            return 1.0
        if req_id not in self._noise:
            self._noise[req_id] = math.exp(self._rng.gauss(0.0, self.noise_sigma))
        return self._noise[req_id]

    # -- derived times ----------------------------------------------------

    def service_s(self, model: str, prompt_tokens: int, stage: str,
                  attempt_index: int = None, true_output: int = None,
                  req_id=None) -> float:
        out = self.predict_output(stage, attempt_index, true_output)
        base = (self.cost.prefill_seconds(model, prompt_tokens)
                + out * self.cost.decode_iteration_seconds(model, 1))
        return base * self.noise_factor(req_id)

    def _expected_further_attempts(self, attempt_index: int) -> float:
        """E[remaining attempts | already on attempt `attempt_index`]."""
        w = self.attempt_weights
        lo = max(attempt_index, 1)
        tail = [(a, w[a - 1]) for a in range(lo, len(w) + 1)]
        total = sum(x for _, x in tail)
        if total <= 0:
            return 0.0
        return sum((a - attempt_index) * x for a, x in tail) / total

    def _typical_prompt(self, stage: str) -> int:
        t = self.tok
        return {
            "parse_issue": t.issue_prefix_tokens + t.issue_suffix_tokens,
            "evaluate_plan": t.eval_prompt_tokens,
            "generate_fix": t.stable_prefix_tokens + t.fix_suffix_retry_tokens,
            "diagnose": t.diagnose_prompt_tokens,
        }.get(stage, 500)

    def _stage_cost(self, stage: str, model: str, attempt_index: int = None) -> float:
        if stage == "navigate":
            return self.cpu.rag_index_seconds
        if stage == "apply_and_test":
            return self.cpu.pytest_seconds
        return self.service_s(model, self._typical_prompt(stage), stage, attempt_index)

    def gittins_prior(self, cost):
        """MAARA's Gittins prior, from the fit traces (DECISIONS 38)."""
        from simulator.gittins import maara_prior
        return maara_prior(cost)

    def remaining_s(self, stage: str, attempt_index: int, model: str,
                    req_id=None, node=None) -> float:
        """
        Expected work left in this task from `stage` onward, in seconds. This is
        the DAG's remaining critical path as estimated online, and it is what
        LSF's slack and SJF's ordering are built from.
        """
        try:
            idx = _STAGE_ORDER.index(stage)
        except ValueError:
            idx = -1

        est = sum(self._stage_cost(s, model, attempt_index)
                  for s in _STAGE_ORDER[idx + 1:])

        # Plus however many more retry loops we expect to run.
        loop = (self._stage_cost("generate_fix", model, attempt_index)
                + self.cpu.pytest_seconds
                + self._stage_cost("diagnose", model, attempt_index))
        est += self._expected_further_attempts(attempt_index) * loop
        return est * self.noise_factor(req_id)

    def estimate_task_total(self, model: str = CODER_MODEL) -> float:
        """
        Estimated wall-clock for one task run ALONE, with no contention. Used to
        set deadlines, so it deliberately does not depend on current load.
        """
        return (self._stage_cost("parse_issue", model, 1)
                + self.remaining_s("parse_issue", 1, model))


class OracleEstimator(ExecutionEstimator):
    """
    Upper-bound reference only. Returns true output lengths.

    Not a policy anyone could deploy -- it is the "what if prediction were
    perfect" line on the chart. Every output using it must say so; the engine
    labels such runs estimator=oracle.
    """

    is_oracle = True
    name = "oracle"

    def predict_output(self, stage: str, attempt_index: int = None,
                       true_output: int = None) -> float:
        if true_output is None:
            return super().predict_output(stage, attempt_index)
        return float(true_output)

    def remaining_s(self, stage: str, attempt_index: int, model: str,
                    req_id=None, node=None) -> float:
        """
        True remaining critical path, when the caller supplies the node.

        Overriding predict_output alone was not enough to make this an oracle:
        remaining_s builds its estimate from _typical_prompt() and the attempt
        distribution and never consults predict_output, so an "oracle" run
        handed the scheduler byte-identical estimates to the online run -- same
        314 distinct est_remaining_s values, same mean -- and the oracle row
        came out identical to the online row on every metric. That looked like
        the finding "a perfect estimator buys nothing" when it was a plumbing
        gap.

        critical_path_remaining includes this node's own service time, so the
        work remaining AFTER it is the difference.
        """
        if node is None:
            return super().remaining_s(stage, attempt_index, model, req_id=req_id)
        return max(0.0, node.critical_path_remaining - node.expected_seconds)


ESTIMATORS = {"online": ExecutionEstimator, "oracle": OracleEstimator}


# --------------------------------------------------------------------------
# Schedulers
# --------------------------------------------------------------------------
#
# All return req_ids in the order the engine should try to admit them. Ties are
# broken by (enqueued_at, req_id) everywhere, so runs are reproducible and two
# schedulers agreeing on the primary key produce identical orders.

class Scheduler:
    name = "base"
    needs_deadlines = False

    def order(self, views, now: float):
        raise NotImplementedError

    def admit_cap(self, now: float, views, cost: CostModel, max_batch: int):
        """How many requests may run at once; None leaves it at max_batch.
        `cost` is the hardware profile (public physics, not oracle)."""
        return None


class FCFSScheduler(Scheduler):
    """
    First come, first served. The incumbent.

    Starvation: impossible. Strict arrival order means every request is admitted
    after a bounded number of earlier ones, at any load.
    """
    name = "fcfs"

    def order(self, views, now):
        return [v.req_id for v in sorted(views, key=lambda v: (v.enqueued_at, v.req_id))]


class SJFScheduler(Scheduler):
    """
    Shortest estimated remaining work of the request's TASK, first.

    Uses est_remaining_s (whole-task remaining work) rather than est_service_s
    (this request alone), so it favours tasks that are nearly done over requests
    that are merely individually cheap. That is the variant that actually moves
    mean task latency.

    Starvation: YES, and this is its known failure mode. Under sustained load a
    steady stream of nearly-finished tasks keeps jumping ahead of a task that
    still has three retry attempts to burn, deferring it indefinitely. Expect
    p99 to degrade even as p50 improves; that trade is why it is here.
    """
    name = "sjf"

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.est_remaining_s, v.enqueued_at, v.req_id))]


class EDFScheduler(Scheduler):
    """
    Earliest deadline first. Deadlines come from the estimator, not the trace.

    Starvation: under overload, yes, and badly. EDF is optimal while the system
    is feasible, but when it is not it shows the classic domino effect -- it
    keeps prioritising the most-overdue task, misses it anyway, and drags
    everything behind it down too. Loose-deadline work is deferred indefinitely
    while tighter work keeps arriving.
    """
    name = "edf"
    needs_deadlines = True

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.deadline if v.deadline is not None else math.inf,
                                  v.enqueued_at, v.req_id))]


class LeastSlackFirstScheduler(Scheduler):
    """
    Slack = deadline - now - estimated remaining work on the task's critical
    path. Lowest slack first.

    The difference from EDF is the entire mechanism. EDF orders by deadline
    alone, so a request with a distant deadline but a long tail of dependent
    work behind it stays deprioritised right up until it becomes the
    bottleneck. Slack subtracts that tail, and the tail comes from the workflow
    DAG. Two requests with the same deadline but different amounts of remaining
    work are indistinguishable to EDF and correctly ordered by LSF.

    est_remaining_s comes from ExecutionEstimator via stage position and the
    empirical retry distribution -- never from Node.critical_path_remaining,
    which is oracle.

    Starvation: same shape as EDF. A task whose deadline is far away keeps
    losing to tighter work and can be deferred indefinitely under sustained
    overload. Unlike EDF it also keeps prioritising already-late tasks (slack
    goes negative), which can pour service into work that will miss regardless
    -- worth watching in the deadline-miss numbers.
    """
    name = "lsf"
    needs_deadlines = True

    def order(self, views, now):
        def slack(v):
            if v.deadline is None:
                return math.inf
            return v.deadline - now - v.est_remaining_s
        return [v.req_id for v in sorted(
            views, key=lambda v: (slack(v), v.enqueued_at, v.req_id))]


class ProgramFCFSScheduler(Scheduler):
    """
    Oldest TASK first: requests ordered by their task's arrival, not their own.

    The honest baseline for workflow-aware scheduling. Program-level ordering
    needs no estimates, and FINDINGS 29 found it is where every gain over
    request-level FCFS came from. A slack-based rule has to beat this, not
    FCFS, to claim value from its estimates.
    """
    name = "program-fcfs"

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.task_arrival_s, v.enqueued_at, v.req_id))]


class ProgramLASScheduler(Scheduler):
    """
    Least attained service first, at the program level: requests ordered by
    how much GPU time their task has already had, then by task age. This is
    Autellix's PLAS (Luo et al., 2025), the published alternative to
    `program-fcfs` (DECISIONS 36). MAARA's pipeline is sequential apart from
    generate_fix's one-request-per-file fan-out (1-3), whose service is summed
    as PLAS sums all of a program's calls.

    Autellix discretizes attained service into multi-level queues and adds an
    anti-starvation threshold; both are omitted, so this has no parameters.
    Attained service is the admission-to-completion time of the task's
    finished requests, which the engine observes; nothing is estimated.
    """
    name = "plas"

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.task_attained_s, v.task_arrival_s,
                                  v.enqueued_at, v.req_id))]


class AdaptiveCapProgramFCFS(ProgramFCFSScheduler):
    """
    Oldest task first, plus a concurrency cap set from the hardware's batching
    law (CONWIP from manufacturing; mentor review 2026-10-08, plan Phase B).

    On a nearly serial batcher, every extra co-runner slows every running
    request almost as much as it adds throughput, so running fewer at once
    finishes older tasks sooner. The scheduling maps show it: with c = 0.841
    the best fixed cap for program-fcfs is 2-4 at operating load and 4-8 in
    deep overload, never 16. With c = 0.15 it is 8-16.

    Relative throughput at k concurrent requests is mu(k) = k / (1 + (k-1)c),
    from the cost model's decode law. The cap is the smallest k with

        mu(k) >= (1 - EPS) * mu(max_batch)

    i.e. stop adding co-runners once they buy less than EPS of the attainable
    throughput: k = 3 at c = 0.841, 14 at c = 0.15 (max_batch 16). EPS fixed
    before any run; not tuned.

    A load-dependent term (enough throughput for the measured offered work) was
    tried first and removed: measured from request arrivals, offered work is
    endogenous, because a task's next request arrives only when its last one
    finishes. A lower cap lowers the measured load, which lowers the cap; in a
    synthetic overload test it collapsed to a mean of 1.5.

    Reads nothing off the views; only the batching coefficient off the hardware
    profile.
    """
    name = "program-fcfs-cap"
    EPS = 0.05

    def admit_cap(self, now, views, cost, max_batch):
        c = cost.batch_decode_coeff
        mu = lambda k: k / (1.0 + (k - 1) * c)  # noqa: E731
        target = (1.0 - self.EPS) * mu(max_batch)
        return next(k for k in range(1, max_batch + 1) if mu(k) >= target)


class BacklogLiftedCapProgramFCFS(AdaptiveCapProgramFCFS):
    """
    `program-fcfs-cap` with the cap lifted while a full batch of requests is
    waiting (DECISIONS 35).

    The cap gives up up to EPS of throughput. While the queue is short that
    buys latency for the oldest tasks; in overload, where throughput sets the
    tail, it is a pure loss (FINDINGS 34, 37). The waiting count is read off
    the current queue, not estimated from arrivals, so unlike the load term
    removed above it does not feed back through the cap into its own input.
    Threshold (waiting >= max_batch) fixed before any run; not tuned.
    """
    name = "program-fcfs-qcap"

    def admit_cap(self, now, views, cost, max_batch):
        if len(views) >= max_batch:
            return max_batch
        return super().admit_cap(now, views, cost, max_batch)


class HintPriorityScheduler(Scheduler):
    """
    A client-declared priority, then oldest task first. Mimics NVIDIA Dynamo's
    agent hints, where the harness sets a static `priority` per request and the
    server computes no slack or remaining work. Priority here is the SLO class
    rank (0 = tightest).
    """
    name = "hint-priority"

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.slo_priority, v.task_arrival_s, v.enqueued_at, v.req_id))]


class StaticPriorityScheduler(Scheduler):
    """
    Fixed per-stage rank, tie-broken by arrival.

    Encodes a plausible operator intuition: finish what is nearly done, and get
    the cheap gate out of the way. It is a baseline for LSF precisely because it
    is workflow-blind -- rank depends only on which stage a request is, so it
    cannot tell a task on its last attempt from one that just arrived.

    Starvation: yes. Under open arrivals parse_issue (lowest rank) is deferred
    whenever any later-stage work exists, so new tasks stop entering the system
    while in-flight ones churn.
    """
    name = "static-priority"
    _RANK = {"apply_and_test": 0, "diagnose": 1, "generate_fix": 2,
             "evaluate_plan": 3, "parse_issue": 4}

    def order(self, views, now):
        return [v.req_id for v in sorted(
            views, key=lambda v: (self._RANK.get(v.stage, 9), v.enqueued_at, v.req_id))]


class GittinsScheduler(Scheduler):
    """
    Program-level Gittins index over requests completed (DECISIONS 38,
    simulator/gittins.py). Every request of a task carries the task's index;
    the highest index goes first, then the oldest task.

    The prior comes from the run's estimator (`gittins_prior`), which builds it
    from development data only, so the engine calls `bind` once at start-up.
    Unbound (no estimator, or one without a prior) it orders by fewest requests
    completed, the index's shape under a decreasing hazard.

    `GUARD_X` adds the tail guard: the single oldest task whose time in system
    exceeds GUARD_X x the prior's mean size goes first. One task, not all of
    them: promoting every overdue task is `program-fcfs` in overload.
    """
    name = "gittins"
    WEIGHT = "latency"
    GUARD_X = None

    def __init__(self):
        self.table = None
        self.mean_size = None

    def bind(self, cost, estimator):
        from simulator.gittins import GittinsTable, pooling_threshold
        prior = getattr(estimator, "gittins_prior", None)
        if prior is None:
            return
        r_counts, work, tool_s = prior(cost)
        self.table = GittinsTable(r_counts, work, tool_s, weight=self.WEIGHT,
                                  min_count=pooling_threshold(sum(r_counts.values())))
        self.mean_size = self.table.mean_size

    def _primary(self, v):
        if self.table is None:
            return v.task_requests_done
        return -self.table.index(v.task_requests_done)

    def _promoted(self, views, now):
        if self.GUARD_X is None or self.mean_size is None:
            return None
        limit = self.GUARD_X * self.mean_size
        overdue = [v for v in views if now - v.task_arrival_s > limit]
        if not overdue:
            return None
        return min(overdue, key=lambda v: (v.task_arrival_s, v.task_id)).task_id

    def order(self, views, now):
        top = self._promoted(views, now)
        return [v.req_id for v in sorted(
            views, key=lambda v: (v.task_id != top, self._primary(v),
                                  v.task_arrival_s, v.enqueued_at, v.req_id))]


class GittinsSlowdownScheduler(GittinsScheduler):
    """`gittins` with each task's finish weighted by 1 / its size: targets mean
    slowdown. A heuristic, not proven optimal (DECISIONS 38)."""
    name = "gittins-sd"
    WEIGHT = "slowdown"


class PLASGuardScheduler(GittinsScheduler):
    """PLAS (attained GPU wall time, then task age) with the same tail guard:
    the ablation that separates the guard from the index (DECISIONS 38).
    Registered only guarded (`plas-guard-x<X>`); unguarded it is `plas`."""
    name = "plas"

    def _primary(self, v):
        return v.task_attained_s


def _guarded(base, x):
    return type(f"{base.__name__}X{x}", (base,),
                {"name": f"{base.name}-guard-x{x}", "GUARD_X": float(x)})


GITTINS_SCHEDULERS = [GittinsScheduler, GittinsSlowdownScheduler] + [
    _guarded(b, x) for b in (GittinsScheduler, GittinsSlowdownScheduler,
                             PLASGuardScheduler) for x in (2, 4, 8)]


SCHEDULERS = {c.name: c for c in [
    FCFSScheduler, SJFScheduler, EDFScheduler, LeastSlackFirstScheduler,
    StaticPriorityScheduler, ProgramFCFSScheduler, HintPriorityScheduler,
    ProgramLASScheduler,
    AdaptiveCapProgramFCFS,
    BacklogLiftedCapProgramFCFS,
    *GITTINS_SCHEDULERS,
]}


# --------------------------------------------------------------------------
# Retention policies
# --------------------------------------------------------------------------
#
# Duck-typed on the engine's KVEntry (needs .key, .nbytes, .last_used,
# .evictable) so this module does not import engine.py. Phase 5 adds ski-rental
# and the offline reference here.

class RetentionPolicy:
    name = "base"

    def on_complete(self, entry, now: float) -> bool:
        """True to retain the entry after its last user finishes."""
        raise NotImplementedError

    def choose_victims(self, candidates, need_bytes, now: float):
        """Entries to evict, in order, until need_bytes is freed."""
        raise NotImplementedError

    def on_tick(self, entries, now: float):
        """
        Keys to give up voluntarily this iteration.

        Ski-rental lives here: its decision is "hold until accrued rent reaches
        the recompute cost, then let go", which has to be checked as rent
        accrues rather than only when pressure arrives.
        """
        return []


class LRURetention(RetentionPolicy):
    """
    Retain everything; evict least-recently-used under pressure.

    The incumbent: this is what vLLM's automatic prefix caching does, so it is
    the real baseline rather than a strawman. Its weakness is that it only looks
    backward -- it knows when an entry was last used, never when it will be
    wanted again.
    """
    name = "lru"

    def on_complete(self, entry, now):
        return True

    def choose_victims(self, candidates, need_bytes, now):
        victims, freed = [], 0
        for e in sorted(candidates, key=lambda e: (e.last_used, e.key)):
            if freed >= need_bytes:
                break
            victims.append(e)
            freed += e.nbytes
        return victims


class KeepAllRetention(LRURetention):
    """Never voluntarily drop. Still forced to evict when out of space."""
    name = "keep-all"


class EvictNowRetention(RetentionPolicy):
    """
    Never retain anything. The opposite extreme from keep-all.

    Every reuse becomes a recompute, so this is the upper bound on recompute
    cost and the lower bound on memory held. Useful as the floor against which
    any retention policy must justify the memory it keeps.
    """
    name = "evict-now"

    def on_complete(self, entry, now):
        return False

    def choose_victims(self, candidates, need_bytes, now):
        return sorted(candidates, key=lambda e: (e.last_used, e.key))


class SkiRentalRetention(RetentionPolicy):
    """
    Deterministic ski-rental. Hold while cheap, let go at the break-even point.

    accrued rent is in recompute-seconds (see engine._accrue_derived_rent), and
    an entry's recompute_s is what keeping it saves, so the comparison is in
    consistent units: release once rent_accrued >= recompute_s.

    WHY 2-COMPETITIVE DOES NOT APPLY HERE
    -------------------------------------
    The textbook bound assumes rent is CONSTANT per unit time. The adversary's
    only freedom is when the season ends, and paying rent up to the buy price
    before buying costs at most twice the offline optimum.

    Derived rent is not constant. Its rate is
        E.bytes x marginal victim's recompute per byte x eviction rate
    and both the marginal victim and the eviction rate move with load, so an
    adversary controlling arrivals controls the rate as well as the deadline.
    That is strictly more power than the ski-rental adversary has, and the
    2-competitive proof does not survive it.

    So the bound is not claimed. The realized ratio against the offline
    reference is measured instead, and reported as a measurement rather than a
    guarantee.
    """
    name = "ski-rental"

    def on_complete(self, entry, now):
        return True             # start renting

    def choose_victims(self, candidates, need_bytes, now):
        # Under forced pressure, give up whatever is closest to break-even
        # first: it was about to be released anyway.
        return sorted(candidates,
                      key=lambda e: (-(e.rent_accrued - e.recompute_s), e.key))

    def on_tick(self, entries, now):
        return [e.key for e in entries
                if e.evictable and e.rent_accrued >= e.recompute_s > 0]


class SkiRentalPredictionsRetention(SkiRentalRetention):
    """
    Learning-augmented ski-rental, after Purohit, Svitkina and Kumar (2018).

    Given a prediction y of how long the entry will be wanted, and a trust
    parameter lambda in (0, 1]:

        predicted FAR  (y >= break-even)   -> release early, at lambda x b
        predicted NEAR (y <  break-even)   -> hold longer, until b / lambda

    lambda near 1 trusts the prediction (better when it is right, worse when it
    is wrong); lambda near 0 ignores it and degrades towards the deterministic
    rule. That trade is the point: consistency against robustness.

    Everything is measured in accrued rent rather than seconds, because rent is
    what actually accumulates and its rate varies. The predicted gap in seconds
    is converted using the rate this entry has seen so far; before any rent has
    accrued the conversion yields zero, which reads as "predicted near" and
    holds -- the right default when there is no pressure yet.
    """
    name = "ski-rental-pred"

    def __init__(self, lam: float = 0.5):
        if not 0.0 < lam <= 1.0:
            raise ValueError("lambda must be in (0, 1]")
        self.lam = lam

    def _predicted_rent(self, e, now):
        """
        Predicted rent between now and the predicted reuse.

        Uses the entry's CURRENT rent rate, stamped by the engine. An earlier
        version averaged rent over the entry's own lifetime, which reads zero
        until the first moment of pressure -- so every entry looked "predicted
        near" and the prediction had no effect at all. The symptom was that
        hint noise changed nothing.
        """
        return (e.hint_gap_s or 0.0) * e.rent_rate

    def on_tick(self, entries, now):
        out = []
        for e in entries:
            if not e.evictable or e.recompute_s <= 0:
                continue
            b = e.recompute_s
            if e.hint_gap_s is None:
                threshold = b                      # no prediction: plain rule
            elif self._predicted_rent(e, now) >= b:
                threshold = self.lam * b           # predicted far: leave early
            else:
                threshold = b / self.lam           # predicted near: hold on
            if e.rent_accrued >= threshold:
                out.append(e.key)
        return out


class PredictedBeladyRetention(RetentionPolicy):
    """
    Workflow-aware eviction ORDER: evict whatever the workflow will want
    latest, using the DAG's reuse hints. Belady with predictions.

    This exists because of what the measurements actually showed. Ski-rental's
    premise is that HOLDING costs something a policy can avoid by releasing;
    in this regime releasing only shifts rent onto whoever still holds, so
    voluntary release cannot pay off (FINDINGS 24). What *is* real is the
    measured Ollama failure: the server discards state the workflow is about to
    reuse (FINDINGS 21). That is an ORDERING problem, not a hold-or-release
    problem, so this policy never releases voluntarily -- it only changes who
    gets evicted when eviction is forced.

    Ordering, worst-to-keep first:
      1. hinted entries whose predicted next use is FAR (gap above `far_s`),
         furthest first -- Belady's rule where we have a prediction
      2. unhinted entries, LRU among themselves
      3. hinted entries whose predicted next use is NEAR, furthest first --
         protected most, since the workflow is about to come back for them

    `far_s` is the running median of observed hints rather than a constant, so
    the boundary adapts to the workload instead of needing a tuned number. An
    unhinted entry is therefore treated as "reuse distance unknown, assume
    typical", which is the honest default.

    Prediction noise comes from HintBuilder's sigma, so sigma=0 is the
    consistency end (perfect hints) and larger sigma tests robustness --
    the Purohit trade-off, applied to ordering instead of to a threshold.
    """
    name = "pred-belady"

    def __init__(self, far_s: float = None):
        # far_s is retained for API compatibility and is unused: the ordering
        # is now a single sort by predicted next use, which is what Belady
        # actually says. The earlier tiered version put UNHINTED entries in the
        # middle, ahead of hinted-but-far ones. That was backwards here,
        # because a missing hint means the block is never reused again -- next
        # use is infinity, so it should be evicted FIRST. It cost 97.2s of
        # recompute against keep-all's 13.4s.
        self._far_s = far_s

    def on_complete(self, entry, now):
        return True                  # ordering policy: never drop on completion

    def on_tick(self, entries, now):
        return []                    # and never release voluntarily

    def choose_victims(self, candidates, need_bytes, now):
        def rank(e):
            # No prediction at all -> treat next use as infinitely far and
            # evict first. With HintBuilder(never_gap_s=...) this branch is
            # unreachable, which is the configuration that keeps the policy
            # honest (see HintBuilder).
            if e.hint_gap_s is None:
                return (0, 0.0, e.key)
            return (1, -e.hint_gap_s, e.key)

        victims, freed = [], 0
        for e in sorted(candidates, key=rank):
            if freed >= need_bytes:
                break
            victims.append(e)
            freed += e.nbytes
        return victims


class OfflineBeladyRetention(PredictedBeladyRetention):
    """
    OFFLINE REFERENCE -- Belady's rule with the TRUE next-use time.

    The least-recompute eviction order achievable under a given memory budget,
    which is the quantity a retention policy can actually win (FINDINGS 24).
    It replaces `offline-ref` as the headline reference, because `offline-ref`
    minimises recompute PLUS rent and so trades away real recompute to reduce a
    rent term the policy cannot remove -- its recompute (29.4s) came out higher
    than keep-all's (13.4s), which makes it useless as an upper bound on the
    thing being optimised.

    Not deployable: it reads `true_gap_s`. An entry with no true reuse is
    evicted first, since it will never be wanted again.
    """
    name = "offline-belady"

    def choose_victims(self, candidates, need_bytes, now):
        def rank(e):
            # Never reused again -> evict first. Otherwise furthest use first.
            if e.true_gap_s is None:
                return (0, 0.0, e.key)
            return (1, -e.true_gap_s, e.key)

        victims, freed = [], 0
        for e in sorted(candidates, key=rank):
            if freed >= need_bytes:
                break
            victims.append(e)
            freed += e.nbytes
        return victims


class OfflineReferenceRetention(RetentionPolicy):
    """
    OFFLINE REFERENCE -- not optimal, and must never be labelled as such.

    Reads true_gap_s, the real time until each entry is next matched, which no
    online policy can see. Two rules:

      * keep an entry only if the rent it will accrue before its reuse is less
        than the recompute it saves; otherwise drop it at once;
      * under forced pressure evict by Belady, farthest next use first.

    Why not "optimal": Belady is optimal for a fixed reference string with
    uniform costs and no batching. Here entry sizes differ, eviction changes
    later timing through the queue, and rent depends on load that the policy's
    own decisions perturb. This is a strong reference, not a proven lower bound.
    """
    name = "offline-ref"

    def _rent_until_reuse(self, e, now):
        """Rent this entry will accrue before its true next use, at the current
        rate. Same instantaneous-rate reasoning as the prediction policy."""
        return (e.true_gap_s or 0.0) * e.rent_rate

    def on_complete(self, entry, now):
        # Never reused: worthless from this instant.
        return entry.true_gap_s is not None

    def choose_victims(self, candidates, need_bytes, now):
        # Belady: farthest next use first, "never" farthest of all.
        return sorted(candidates,
                      key=lambda e: (-(e.true_gap_s if e.true_gap_s is not None
                                       else float("inf")), e.key))

    def on_tick(self, entries, now):
        out = []
        for e in entries:
            if not e.evictable:
                continue
            if e.true_gap_s is None:
                out.append(e.key)
            elif e.recompute_s > 0 and self._rent_until_reuse(e, now) > e.recompute_s:
                out.append(e.key)
        return out


class HintBuilder:
    """
    Turns the trace's oracle reuse time into the noisy prediction the
    learning-augmented policy is allowed to see.

    This is the ONLY place outside the offline reference that may read
    Node.next_reuse_gap_s. Noise is multiplicative lognormal, so a prediction
    is wrong by a factor rather than by an absolute amount -- the realistic
    failure mode for a duration estimate. sigma=0 means a perfect oracle hint,
    which is the consistency end of the Purohit trade-off.
    """

    def __init__(self, sigma: float = 0.0, seed: int = 0,
                 never_gap_s: float = None):
        self.sigma = sigma
        self._seed = seed
        # A block that is never reused has no true gap. Returning None for it
        # leaks an ORACLE bit: absence of a hint means "never reused again",
        # which no online predictor would know for free. With never_gap_s set,
        # every block gets a prediction and a never-reused block is predicted
        # to come back beyond the horizon -- so hint presence carries nothing.
        # Leave it None to preserve the older behaviour for ski-rental-pred,
        # whose rule is explicitly "no prediction -> use the plain threshold".
        self.never_gap_s = never_gap_s

    def hint_for(self, node, key=None):
        """
        Noisy prediction of when this block will next be wanted.

        `key` matters: node-level `next_reuse_gap_s` is derived from
        `prefix_key`, and on the trace path it is None for EVERY node -- 320 of
        320 on holdout_qwen -- so a hint was never produced and
        `ski-rental-pred` degenerated silently into deterministic ski-rental.
        The learning-augmented half of the mechanism was therefore untested on
        real data, and the two policies printed byte-identical grid rows.
        Per-block gaps (DECISIONS 21) are the usable signal.
        """
        true_gap = None
        if key is not None:
            true_gap = getattr(node, "block_reuse_gap", {}).get(key)
        if true_gap is None:
            true_gap = getattr(node, "next_reuse_gap_s", None)
        if true_gap is None:
            if self.never_gap_s is None:
                return None
            true_gap = self.never_gap_s
        if self.sigma <= 0:
            return true_gap
        return true_gap * math.exp(self._noise_for(node, key))

    def _noise_for(self, node, key):
        """
        Noise derived from (seed, block key), NOT from call order.

        COMMON RANDOM NUMBERS. A single shared RNG draws in the order hint_for
        is called, and that order follows the admission sequence, which differs
        by policy -- so the same block received a different predicted gap under
        each policy. Measured: of 12 blocks hinted under both ski-rental-pred
        and pred-belady, 0% got the same noised value. That silently turns a
        paired comparison into an unpaired one and reintroduces exactly the
        variance pairing exists to remove, which matters here because seed
        variance on this trace is +/-20% (FINDINGS 27).

        Keying on the block also makes a prediction a property of the block,
        which is what a real predictor would be.
        """
        tag = f"{self._seed}|{key if key is not None else getattr(node, 'node_id', 0)}"
        return random.Random(tag).gauss(0.0, self.sigma)


RETENTIONS = {c.name: c for c in [
    LRURetention, KeepAllRetention, EvictNowRetention, SkiRentalRetention,
    SkiRentalPredictionsRetention, OfflineReferenceRetention,
    PredictedBeladyRetention, OfflineBeladyRetention,
]}
