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
from dataclasses import dataclass
from typing import Optional

from sim.workload import CODER_MODEL, CostModel

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

    def __getattr__(self, name):
        # Only reached when normal dataclass lookup fails.
        if name in FORBIDDEN_FOR_SCHEDULERS:
            raise OracleLeakError(
                f"scheduler read {name!r} off a ReqView. Schedulers must work "
                f"from estimates (est_service_s / est_remaining_s), never from "
                f"true values. See ORACLE DISCIPLINE in sim/policies.py.")
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
      3. The empirical retry distribution from benchmark/results.jsonl, which is
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

    def remaining_s(self, stage: str, attempt_index: int, model: str,
                    req_id=None) -> float:
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


SCHEDULERS = {c.name: c for c in [
    FCFSScheduler, SJFScheduler, EDFScheduler, LeastSlackFirstScheduler,
    StaticPriorityScheduler,
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

    def __init__(self, sigma: float = 0.0, seed: int = 0):
        self.sigma = sigma
        self._rng = random.Random(seed)

    def hint_for(self, node):
        true_gap = getattr(node, "next_reuse_gap_s", None)
        if true_gap is None:
            return None
        if self.sigma <= 0:
            return true_gap
        return true_gap * math.exp(self._rng.gauss(0.0, self.sigma))


RETENTIONS = {c.name: c for c in [
    LRURetention, KeepAllRetention, EvictNowRetention, SkiRentalRetention,
    SkiRentalPredictionsRetention, OfflineReferenceRetention,
]}
