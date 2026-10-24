"""
Event-driven continuous-batching engine with a shared KV budget.

The point of this file is to be able to REACH the state "KV budget is full and
something is waiting for it". That state is the only place opportunity cost
exists, and the previous version of this simulator could never reach it because
it ran one request at a time. Everything here is arranged around making that
state real, observable, and attributable.

Structure of one iteration:
  1. release events that are due (task arrivals, CPU stages finishing)
  2. admit waiting requests, evicting retained prefixes if that is what it takes
  3. run ONE engine iteration: at most one prefill, plus a decode step for the
     rest of the active model's batch
  4. accrue rent against retained entries if anything is KV-blocked
  5. retire finished requests and advance their task DAGs

RENT
----
There is deliberately no memory-price knob. Rent is derived, not priced
(DECISIONS 13): while the KV pool is under pressure, each retained-but-
unreferenced entry is charged in proportion to its bytes times the
recompute-per-byte of what that pressure would evict, normalised by the
evictable pool. Rent is therefore in seconds, comparable to
CostModel.recompute_seconds() -- the ski-rental comparison. It is a shared
pressure signal that releasing shifts rather than removes (DECISIONS 23), so
report it as a diagnostic, not a cost to rank policies by. The Phase 1 model,
which charged only while a request was hard-blocked, is kept as
rent_model="blocking".
"""

from __future__ import annotations

import argparse
import heapq
import math
import random
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, ExecutionEstimator,
                          FCFSScheduler, LRURetention, ReqView)
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, Kind,
                          StageTokens, WorkloadConfig, describe_trace,
                          generate_trace)


@dataclass
class EngineConfig:
    kv_budget_mb: float = 2048.0
    max_batch: int = 8                  # B: concurrent requests in the running set
    # Cost of switching which model the engine is serving. 0.0 assumes both sets
    # of weights stay resident. PLACEHOLDER - a real Ollama deployment that has
    # to swap weights would pay seconds here.
    model_switch_seconds: float = 0.0

    # Task deadline = arrival + slack_factor * estimated isolated task time
    # (the task run alone, no contention). 1.5 means "we promise a task will
    # finish in 1.5x what it would take on an idle server".
    # PLACEHOLDER: not derived from any SLO anyone actually stated. Sweep it.
    slack_factor: float = 1.5

    # Per-task deadlines (FINDINGS 29 found one common budget makes EDF plain
    # oldest-task-first). None keeps the single budget above, byte-identical.
    # Otherwise a tuple of (class name, budget multiplier, weight): each task
    # draws a class, keyed on (slo_seed, task_id) so every policy sees the same
    # draw, and its budget is also scaled by its own size as visible at arrival
    # (first LLM prompt / that stage's typical prompt, clamped to [0.25, 4]).
    #   deadline = arrival + slack_factor * multiplier * size_scale * isolated
    # Classes are declared by the client, so schedulers may read them; the
    # tightest multiplier is priority 0.
    slo_classes: Optional[tuple] = None
    slo_seed: int = 0
    # What a deadline budget is a multiple of (OPEN_ISSUES 29).
    #   "estimate": the estimator's one isolated-task constant times the
    #               first-prompt size scale. On holdout_qwen that is 90.2 s x
    #               0.35-0.41 for every task, so deadlines carry almost no
    #               per-task information. Default, so committed results hold.
    #   "task":     the task's own isolated critical path (5.6-88.8 s there),
    #               times lognormal noise of sigma deadline_noise_sigma drawn per
    #               (slo_seed, task_id): the client's rough estimate of its own
    #               job. This is a deliberate SIZE SIGNAL handed to schedulers
    #               through the deadline, as a client that sizes its SLO to its
    #               request would. Set by the experimenter, never read by a
    #               policy except as the deadline itself.
    #   deadline = arrival + slack_factor * multiplier * noisy critical path
    deadline_basis: str = "estimate"
    deadline_noise_sigma: float = 0.3

    # KV is allocated in fixed-size blocks, as in vLLM's prefix cache. 16 is
    # vLLM's default. PLACEHOLDER: not measured, and it matters here because
    # the median reusable prefix in the real trace is 4 tokens -- smaller than
    # one block -- so quantisation dominates at the median.
    block_tokens: int = 16

    # "derived"  : rent_rate(E) = E.bytes x victim recompute s/byte x eviction rate
    # "blocking" : the Phase 1 model, charged only while a request is hard-blocked
    rent_model: str = "derived"

    # Prefix-cache ARCHITECTURE, not a retention policy.
    #   "block"  vLLM-style: one shared, content-keyed block cache that every
    #            request can hit, with a retention policy deciding what to keep.
    #   "ollama" what the real server does (FINDINGS 20): ONE active slot per
    #            model plus a host-RAM tier. A new prompt whose f_keep >= 0.50
    #            reuses the slot IN PLACE and the rest of the previous prompt is
    #            discarded with no saved copy; below that the displaced state is
    #            saved to host RAM and the tier is searched for a better start.
    #
    # This cannot be a RetentionPolicy: policies choose what to keep within a
    # shared cache, whereas Ollama's defining behaviour is that only one prompt
    # is resident at all and similarity decides whether the displaced one
    # survives. Modelling it as LRU would erase exactly the effect that loses
    # coder-retry caches (FINDINGS 21).
    cache_mode: str = "block"
    # Host-RAM tier, MEASURED from the server log ("size limit: 8192 MiB"). It
    # is HOST memory, so it does NOT consume kv_budget_mb -- the Ollama baseline
    # therefore gets a second tier the block-cache policies do not have. That is
    # generous to the incumbent and deliberate, but it must be stated whenever
    # the comparison is reported.
    ollama_host_mb: float = 8192.0
    ollama_keep_gate: float = 0.50

    # Smoothing constant for the observed eviction rate, seconds. PLACEHOLDER:
    # an estimator parameter, not a price -- it changes how noisily an
    # observable is measured, not what memory costs.
    ewma_tau_s: float = 30.0

    # Chunked prefill, as vLLM does it: a long prompt is prefilled a chunk at a
    # time so one 27k-token prompt cannot monopolise an iteration and stall
    # every decode behind it. 512 is vLLM's default order of magnitude.
    # PLACEHOLDER, not measured. 0 disables chunking (the Phase 1-5 behaviour).
    # Chunked is the DEFAULT here; --prefill-chunk 0 restores unchunked.
    prefill_chunk_tokens: int = 512

    # Record every time a block is needed (built, rebuilt, or hit) in
    # Engine._key_uses, for sizing a KV budget from the reusable working set
    # offline (experiments/run_retention_grid.py --headrooms). Off by default:
    # it is experimenter-side accounting, never a policy input.
    record_key_uses: bool = False

    @property
    def kv_budget_bytes(self) -> int:
        return int(self.kv_budget_mb * 1024 * 1024)

    def blocks_for(self, tokens: int) -> int:
        """Tokens rounded up to whole KV blocks."""
        if tokens <= 0:
            return 0
        return -(-int(tokens) // self.block_tokens) * self.block_tokens


class State(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"


class RentClock:
    """
    Shared cumulative rent-per-byte counter.

    Every evictable block accrues at the SAME rate per byte -- the rate depends
    on the marginal victim and the eviction rate, neither of which is per-entry
    -- so advancing one counter once per iteration replaces a sweep over every
    entry. An entry's rent is then bytes x (counter now - counter when it became
    evictable), evaluated only when someone asks.
    """
    __slots__ = ("cum",)

    def __init__(self):
        self.cum = 0.0


@dataclass
class KVEntry:
    """
    A prefix-cache entry, keyed by content. Shared: if two requests want the same
    prefix it is stored once and refcounted, which is the other half of why
    retention pays -- not just avoided recompute, but avoided duplication.
    """
    key: str
    model: str
    tokens: int
    nbytes: int
    refcount: int = 0
    last_used: float = 0.0
    created: float = 0.0
    # Rent banked from completed intervals, plus the clock reading when this
    # entry last became evictable. See the rent_accrued property below.
    rent_banked: float = 0.0
    rent_cum_at: float = 0.0
    clock: object = None
    retained: bool = True
    # Seconds of prefill needed to rebuild this entry: its "buy" cost.
    recompute_s: float = 0.0
    # How many of this entry's tokens a later call can actually match. Below
    # `tokens` whenever block quantisation rounded the entry up, and zero for a
    # tail. Accounting only -- an online policy must not read it, since at
    # creation time a head and a tail are indistinguishable.
    useful_tokens: int = 0
    # Prompt blocks left behind after the shared head: everything past the point
    # where this call's prompt diverged from its predecessor. vLLM caches them
    # like any other block; nothing will ever match them.
    is_tail: bool = False
    # Predicted seconds until this prefix is wanted again, from the workflow
    # graph via HintBuilder. Noisy on purpose. Online policies may read it.
    hint_gap_s: Optional[float] = None
    # TRUE seconds until reuse, None if never reused. ORACLE -- only the
    # offline reference may read it. An online policy that does is cheating.
    true_gap_s: Optional[float] = None
    # Rent this entry is accruing per second, as of the last iteration under
    # pressure. Policies need an instantaneous rate to convert a predicted
    # reuse TIME into the rent scale; averaging over the entry's own lifetime
    # reads zero before the first pressure and makes any prediction inert.
    rent_rate: float = 0.0

    @property
    def evictable(self) -> bool:
        return self.refcount == 0

    @property
    def rent_accrued(self) -> float:
        """Banked rent plus whatever has accrued since this entry last became
        evictable. Pinned entries do not accrue: they cannot be displaced."""
        if self.clock is None or self.refcount > 0:
            return self.rent_banked
        return self.rent_banked + self.nbytes * (self.clock.cum - self.rent_cum_at)

    @rent_accrued.setter
    def rent_accrued(self, value: float) -> None:
        """Banking a value resets the baseline, so reading it back returns it."""
        self.rent_banked = value
        if self.clock is not None:
            self.rent_cum_at = self.clock.cum

    def bank_rent(self) -> None:
        self.rent_banked = self.rent_accrued
        if self.clock is not None:
            self.rent_cum_at = self.clock.cum

    @property
    def recompute_s_per_byte(self) -> float:
        return (self.recompute_s / self.nbytes) if self.nbytes else 0.0


@dataclass
class Request:
    req_id: int
    node: object                    # simulator.workload.Node
    enqueued_at: float
    state: State = State.WAITING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    prefill_tokens: int = 0         # tokens actually prefilled (excludes a cache hit)
    prefill_remaining: int = 0      # still to prefill, when chunking
    generated: int = 0
    private_bytes: int = 0
    entry: Optional[KVEntry] = None
    held_blocks: list = field(default_factory=list)
    matched_blocks: int = 0
    prefix_hit: bool = False
    # Ollama mode only: blocks whose prefill a host-tier restore skips
    # but whose bytes must still be allocated, since a saved state has
    # to be loaded back into GPU KV.
    ollama_restored_blocks: int = 0
    _ollama_plan: tuple = ()
    kv_blocked_seconds: float = 0.0
    # Prediction recorded at enqueue, scored against the true output at
    # completion. Kept per request so estimator error is measurable per role.
    predicted_output: Optional[float] = None

    @property
    def model(self) -> str:
        return self.node.model


# Schedulers and retention policies live in simulator/policies.py. There used to be a
# second FIFOScheduler here; having two meant the CLI silently used one while
# tests exercised the other.


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, tasks, cost: CostModel, cfg: EngineConfig,
                 retention=None, scheduler=None, estimator=None, hints=None,
                 trace: bool = False):
        self.tasks = tasks
        self.cost = cost
        self.cfg = cfg
        self.retention = retention or LRURetention()
        self.scheduler = scheduler or FCFSScheduler()
        self.estimator = estimator
        self.hints = hints
        self.do_trace = trace
        self.trace_log = []
        # task_id -> absolute deadline, set once when the task arrives.
        self.task_deadline = {}
        # task_id -> whether the deadline was already out of reach when the
        # task's first GPU request was admitted: now + the oracle critical path
        # from that node > deadline. ACCOUNTING ONLY -- no policy reads it
        # (ARCHITECTURE 10). The critical path is uncontended time, so
        # "feasible" here is optimistic: a task can be feasible by this test
        # and still be impossible under contention.
        self.task_infeasible_at_first_admit = {}

        self.now = 0.0
        self.events = []            # heap of (time, seq, kind, payload)
        self._seq = 0
        self.waiting = []
        self.running = []
        self.entries = {}           # key -> KVEntry
        self._key_uses = {}         # key -> [times needed], when cfg.record_key_uses
        self._key_nbytes = {}       # key -> block bytes, same condition
        self.pinned_bytes = 0       # private (non-shared) KV of running requests
        self.active_model = None

        # Smoothed evictions per second. Bumped by 1/tau on each eviction and
        # decayed by exp(-dt/tau) each iteration, which is an EWMA of an event
        # rate. This is the only estimator knob in derived rent.
        self._evict_rate = 0.0
        self._clock = RentClock()
        # Lazy min-heap of evictable entries by last_used, so the true marginal
        # victim is available in O(log n) instead of sorting every entry.
        # Stale rows are discarded on read.
        self._lru_heap = []
        self._entry_bytes = 0
        self._evictable_bytes = 0
        self._evictable_n = 0
        self._model_blocks = {}     # model -> evictable block count
        # "sweep" reproduces the pre-Phase-5b per-entry loop, for the
        # equivalence test only. "lazy" is the default.
        self._rent_impl = "lazy"
        # Per block key, across the whole run: rent it accrued while resident,
        # and what rebuilding it costs. Hindsight cost for a key is the cheaper
        # of the two -- what an observer with perfect foresight would have paid.
        self._key_rent = {}
        self._key_recompute = {}
        self._req_id = 0
        self.done_nodes = {}        # (task_id, node_id) -> finish time
        self.task_done_at = {}
        self.requests = []
        self._released = set()

        # Keys that have existed at some point. A miss on one of these is a
        # recompute the retention policy could have avoided; a miss on a new key
        # is a cold miss nobody could have prevented.
        self.seen_keys = set()

        self._ollama_slot, self._ollama_host = {}, {}
        self.m = {
            "prefill_seconds": 0.0, "decode_seconds": 0.0, "switch_seconds": 0.0,
            "idle_seconds": 0.0, "iterations": 0,
            "hits": 0, "cold_misses": 0, "recompute_misses": 0,
            "evictions": 0, "forced_evictions": 0,
            "recompute_tokens": 0, "recompute_seconds": 0.0,
            "kv_blocked_seconds": 0.0, "blocked_events": 0,
            "total_rent": 0.0, "peak_kv_bytes": 0,
            # Time-weighted accumulators. Iteration-averaging is wrong here: a
            # 99s prefill iteration and a 0.035s decode iteration are one sample
            # each, so a plain mean reports whatever decode was doing and hides
            # the prefill entirely.
            "w_time": 0.0, "w_batch": 0.0, "w_resident": 0.0,
            # Time spent at each active-batch size, for a time-weighted p95 and
            # the fraction of time the max_batch cap was binding.
            "batch_time": {},
            # The same for resident requests (all models). This is what the
            # max_batch cap is enforced on, so it is the one that says whether
            # the cap binds; batch_time undercounts whenever both models have work.
            "resident_time": {},
            "w_kv": 0.0, "w_pinned": 0.0, "w_retained": 0.0,
            "max_batch": 0, "max_resident": 0, "peak_active_bytes": 0,
            # Splits blocking by whether retention could have done anything about
            # it. Blocked with nothing evictable means the budget was consumed by
            # ACTIVE requests -- a scheduling/admission problem, not a retention
            # one. Charging rent for that would blame the wrong mechanism.
            "blocked_with_evictable": 0, "blocked_no_evictable": 0,
            "pinned_samples": [], "retained_samples": [],
            # Absolute percentage error of predicted vs true output tokens,
            # collected per stage so a role with a badly-fitting prior is
            # visible rather than averaged away.
            "ape_by_stage": {},
            # How much freedom the scheduler actually had. A scheduler can only
            # matter when two or more requests are waiting; if the queue is
            # almost always empty, every policy produces the same run and a
            # flat comparison table means "no choice was available", not "the
            # policies are equivalent".
            "w_queue": 0.0, "max_queue": 0, "admit_calls": 0, "calls_with_choice": 0,
            "tail_entries_created": 0, "tail_evictions": 0, "w_evict_rate": 0.0,
            "voluntary_evictions": 0,
            # ACTIVITY counters. Three mechanisms in this engine can be
            # silently inert -- KV never blocks, the scheduler never has
            # a choice, rent never accrues -- and each time that happened
            # a comparison measured nothing while still printing a table.
            # These make inactivity visible in the report itself.
            "rent_ticks": 0, "rent_active_ticks": 0,
            "ollama_slot_reuse": 0, "ollama_host_restore": 0,
            "ollama_slot_partial": 0, "ollama_cold": 0,
            "ollama_discarded_blocks": 0, "ollama_host_saves": 0,
            "ollama_host_evictions": 0,
            "voluntary_releases": 0,

        }

        # Isolated task time is computed ONCE, from the estimator's priors,
        # before any learning has happened. Deadlines must be identical across
        # policy runs on the same trace or the miss rates are not comparable --
        # and if it were recomputed per arrival, a scheduler that happened to
        # finish work sooner would teach the estimator faster and hand itself
        # different deadlines.
        self._isolated_task_s = (
            self.estimator.estimate_task_total(CODER_MODEL) if self.estimator else None)
        self.task_arrival = {t.task_id: t.arrival_time for t in tasks}
        # GPU time each task's finished requests have had (PLAS, DECISIONS 36).
        self.task_attained = {}
        self.task_slo = {}          # task_id -> (class name, priority rank)
        if self.cfg.deadline_basis not in ("estimate", "task"):
            raise ValueError(f"deadline_basis {self.cfg.deadline_basis!r}")
        by_task = self.cfg.deadline_basis == "task"
        for t in tasks:
            if by_task or self._isolated_task_s is not None:
                if by_task:
                    rng = random.Random(f"size:{self.cfg.slo_seed}:{t.task_id}")
                    noise = math.exp(rng.gauss(0.0, self.cfg.deadline_noise_sigma))
                    budget = self.cfg.slack_factor * t.critical_path_total * noise
                else:
                    budget = self.cfg.slack_factor * self._isolated_task_s
                if self.cfg.slo_classes:
                    name, mult, rank = self._draw_slo(t.task_id)
                    budget *= mult * (1.0 if by_task else self._size_scale(t))
                    self.task_slo[t.task_id] = (name, rank)
                self.task_deadline[t.task_id] = t.arrival_time + budget
            self._push(t.arrival_time, "arrival", t)

    def _draw_slo(self, task_id):
        classes = self.cfg.slo_classes
        ranks = {name: i for i, (name, _, _) in
                 enumerate(sorted(classes, key=lambda c: c[1]))}
        rng = random.Random(f"slo:{self.cfg.slo_seed}:{task_id}")
        name, mult, _ = rng.choices(classes, weights=[c[2] for c in classes])[0]
        return name, mult, ranks[name]

    def _size_scale(self, task) -> float:
        """The task's size as visible at arrival: its first LLM request's prompt
        against the typical prompt for that stage. Not oracle: the first request
        arrives with the task."""
        first = next((n for n in task.nodes if n.kind is Kind.LLM), None)
        if first is None or self.estimator is None:
            return 1.0
        typical = self.estimator._typical_prompt(first.stage)
        if typical <= 0:
            return 1.0
        return min(4.0, max(0.25, first.prompt_tokens / typical))

    # -- plumbing ----------------------------------------------------------

    def _push(self, when, kind, payload):
        self._seq += 1
        heapq.heappush(self.events, (when, self._seq, kind, payload))

    def _log(self, msg):
        if self.do_trace:
            self.trace_log.append(f"[{self.now:8.3f}s] {msg}")

    @property
    def used_bytes(self) -> int:
        # _entry_bytes is maintained incrementally. Summing self.entries here
        # was O(entries) and free_bytes calls it several times per iteration;
        # at block granularity that alone dominated the run.
        return self.pinned_bytes + self._entry_bytes

    @property
    def free_bytes(self) -> int:
        return self.cfg.kv_budget_bytes - self.used_bytes

    # -- DAG advance -------------------------------------------------------

    def _task_of(self, node):
        return self.tasks[node.task_id]

    def _release_ready(self, task, just_done_id=None):
        """Enqueue any node whose dependencies are all satisfied."""
        for node in task.nodes:
            nid = (task.task_id, node.node_id)
            if nid in self.done_nodes or nid in self._released:
                continue
            if any((task.task_id, d) not in self.done_nodes for d in node.deps):
                continue
            self._released.add(nid)
            if node.kind is Kind.CPU:
                self._push(self.now + node.cpu_seconds, "cpu_done", (task, node))
                self._log(f"  cpu   t{task.task_id}/{node.stage} starts "
                          f"({node.cpu_seconds:.2f}s, holds no KV)")
            else:
                self._req_id += 1
                r = Request(req_id=self._req_id, node=node, enqueued_at=self.now)
                if self.estimator is not None:
                    # Captured at enqueue, which is when the scheduler would
                    # act on it. true_output is passed but ignored by the
                    # online estimator; only OracleEstimator reads it.
                    r.predicted_output = self.estimator.predict_output(
                        node.stage, node.attempt_index,
                        true_output=node.output_tokens)
                self.waiting.append(r)
                self.requests.append(r)
                self._log(f"  queue t{task.task_id}/{node.stage} req{r.req_id} "
                          f"({node.prompt_tokens} tok prompt)")

    def _finish_node(self, task, node):
        self.done_nodes[(task.task_id, node.node_id)] = self.now
        if all((task.task_id, n.node_id) in self.done_nodes for n in task.nodes):
            self.task_done_at[task.task_id] = self.now
        else:
            self._release_ready(task)

    # -- admission ---------------------------------------------------------

    def _match_blocks(self, r: Request):
        """
        Longest run of leading KV blocks already cached, vLLM-style.

        Each block key hashes the chain of every block before it, so a match can
        only ever be a prefix: the first differing block ends it for good. This
        is what makes a hit PARTIAL -- matched blocks skip prefill, everything
        after is computed.
        """
        if self.cfg.cache_mode == "ollama":
            return self._match_ollama(r)
        matched = 0
        for k in r.node.block_keys:
            e = self.entries.get(k)
            if e is None or e.model != r.model:
                break
            matched += 1
        return matched

    @staticmethod
    def _lcp(a, b):
        n = 0
        for x, y in zip(a, b):
            if x != y:
                break
            n += 1
        return n

    def _match_ollama(self, r: Request):
        """
        One slot per model plus a host-RAM tier, after FINDINGS 20.

        The gate is on f_keep = lcp / SLOT length, not / incoming length, and
        the constant is 0.5 -- both read out of llama.cpp's
        `if (f_keep < 0.5f) { update_cache = true; }`. Above the gate the slot
        is reused in place and its tail is dropped with nothing saved; below it
        the displaced state goes to host RAM and the tier is searched.
        """
        chain = r.node.block_keys
        slot = self._ollama_slot.get(r.model, [])
        host = self._ollama_host.setdefault(r.model, [])

        lcp_slot = self._lcp(chain, slot)
        # f_keep is the server's own bookkeeping and is computed against the
        # whole slot, which is what llama.cpp does.
        f_keep = lcp_slot / len(slot) if slot else 0.0
        # The match itself must be capped by what is ACTUALLY resident. The
        # slot is only a list of keys; eviction pressure can have taken some of
        # those blocks out of GPU KV already, and claiming them as a hit left
        # the engine dereferencing entries that no longer exist.
        resident = 0
        for k in chain[:lcp_slot]:
            e = self.entries.get(k)
            if e is None or e.model != r.model:
                break
            resident += 1
        lcp_slot = resident

        # PURE: this is called more than once per admission attempt and the
        # attempt can still fail, so it must not mutate the slot or the host
        # tier. An earlier version did, which left a phantom slot pointing at
        # blocks that were never allocated -- a KeyError two requests later.
        # All mutation happens in _commit_ollama, after admission succeeds.
        if slot and f_keep >= self.cfg.ollama_keep_gate:
            r._ollama_plan = ("reuse", lcp_slot, 0)
            return lcp_slot

        best, best_i = 0, None
        for i, saved in enumerate(host):
            n = self._lcp(chain, saved)
            if n > best:
                best, best_i = n, i
        if best_i is not None and best > lcp_slot:
            # A restore COPIES the saved state (DECISIONS 17); the host keeps it.
            #
            # Prefill is skipped for the restored prefix, but those blocks are
            # NOT resident in GPU KV -- they were evicted when the slot was
            # displaced. Loading a saved state back costs bytes, not prefill.
            # So the resident match stays `lcp_slot` (which is what drives
            # entry bookkeeping) and the extra restored blocks are recorded
            # separately for the prefill discount.
            r.ollama_restored_blocks = max(0, best - lcp_slot)
            r._ollama_plan = ("restore", lcp_slot, r.ollama_restored_blocks)
            return lcp_slot
        r._ollama_plan = ("partial" if lcp_slot > 0 else "cold", lcp_slot, 0)
        return lcp_slot

    def _commit_ollama(self, r):
        """Apply the slot/host transition, once admission has succeeded."""
        kind, lcp_slot, _restored = getattr(r, "_ollama_plan", ("cold", 0, 0))
        slot = self._ollama_slot.get(r.model, [])
        if kind == "reuse":
            # The slot's unmatched tail is destroyed outright: not saved to
            # host, not retained. This is the behaviour that loses a coder's
            # state across a retry (FINDINGS 21).
            self.m["ollama_slot_reuse"] += 1
            self._drop_ollama_tail(r.model, slot[lcp_slot:])
        else:
            if slot:
                self._host_save(r.model, slot)
            self.m["ollama_" + ("host_restore" if kind == "restore"
                                else "slot_partial" if kind == "partial"
                                else "cold")] += 1
            # Whatever the previous slot still held beyond the shared prefix is
            # no longer reachable from the slot; it lives in the host tier now.
            self._drop_ollama_tail(r.model, slot[lcp_slot:])
        self._ollama_slot[r.model] = list(r.node.block_keys)

    def _drop_ollama_tail(self, model, keys):
        """Discard slot blocks that the new prompt did not keep."""
        for k in keys:
            e = self.entries.get(k)
            if e is not None and e.evictable:
                self._evict(e, forced=False)
                self.m["ollama_discarded_blocks"] += 1

    def _host_save(self, model, keys):
        """Save a displaced slot to the host tier, oldest-out by byte budget."""
        if not keys:
            return
        host = self._ollama_host.setdefault(model, [])
        host.append(list(keys))
        self.m["ollama_host_saves"] += 1
        bt = self.cfg.block_tokens
        limit = self.cfg.ollama_host_mb * 1024 * 1024
        size = lambda ks: self.cost.kv_bytes(model, bt) * len(ks)
        total = sum(size(k) for k in host)
        while total > limit and len(host) > 1:
            total -= size(host.pop(0))
            self.m["ollama_host_evictions"] += 1

    def _try_admit(self, r: Request) -> bool:
        """
        Returns True if admitted. Reserves KV for prompt AND full output up front:
        with no preemption, a request that runs out of KV mid-decode would have
        nowhere to go, so the engine refuses to start what it cannot finish.
        """
        node = r.node
        bt = self.cfg.block_tokens
        chain = node.block_keys
        matched = self._match_blocks(r)
        matched_tokens = min(matched * bt, node.prompt_tokens)
        new_keys = chain[matched:]

        block_bytes = self.cost.kv_bytes(r.model, bt)
        # Generated tokens are not part of the prompt chain but still need KV.
        output_bytes = self.cost.kv_bytes(
            r.model, self.cfg.blocks_for(node.output_tokens))
        need = len(new_keys) * block_bytes + output_bytes

        # Check the WORST case, not the current one. With some blocks already
        # cached a request can look admissible, then become permanently
        # infeasible once they are evicted -- which surfaced as a deadlock far
        # from the cause. A request whose whole prompt cannot fit an empty
        # budget can never run, so say so now.
        worst_case = len(chain) * block_bytes + output_bytes
        if worst_case > self.cfg.kv_budget_bytes:
            raise RuntimeError(
                f"req{r.req_id} ({node.stage}) needs {worst_case/1024/1024:.0f}MB "
                f"for its full prompt ({len(chain)} blocks + output) but the whole "
                f"budget is {self.cfg.kv_budget_mb:.0f}MB. Raise kv_budget_mb.")

        if need > self.free_bytes:
            held = set(chain[:matched])
            candidates = [e for e in self.entries.values()
                          if e.evictable and e.key not in held]
            victims = self.retention.choose_victims(
                candidates, need - self.free_bytes, self.now)
            for v in victims:
                if self.free_bytes >= need:
                    break
                self._evict(v, forced=True)

            if need > self.free_bytes and held:
                # Last resort: give up the hit. The blocks this request matched
                # were excluded from eviction to protect the reuse, but under a
                # tight budget they can be the very thing blocking the
                # request's own admission -- a deadlock where a request is held
                # up by the blocks it wanted to reuse. Releasing them costs a
                # recompute; refusing to costs the whole run.
                spares = [e for e in self.entries.values()
                          if e.evictable and e.key in held]
                for v in self.retention.choose_victims(
                        spares, need - self.free_bytes, self.now):
                    if self.free_bytes >= need:
                        break
                    self._evict(v, forced=True)
                # The match is shorter now, so redo it and re-price.
                matched = self._match_blocks(r)
                matched_tokens = min(matched * bt, node.prompt_tokens)
                new_keys = chain[matched:]
                need = len(new_keys) * block_bytes + output_bytes

            for v in ():
                # Test against live free space only. An earlier version also
                # accumulated a `freed` counter and compared it to
                # `need - self.free_bytes` -- but free_bytes already grows as
                # entries are evicted, so the same bytes were counted twice and
                # the loop stopped at roughly half the required eviction.
                if self.free_bytes >= need:
                    break
                self._evict(v, forced=True)
            if need > self.free_bytes:
                return False      # still blocked: caller records this as KV pressure

        # Accounting. A block seen before and now absent is recompute the
        # retention policy could have avoided; a block never seen is a cold
        # miss nobody could have prevented.
        for k in new_keys:
            if k in self.seen_keys:
                self.m["recompute_misses"] += 1
                self.m["recompute_tokens"] += bt
                self.m["recompute_seconds"] += self.cost.recompute_seconds(r.model, bt)
            else:
                self.m["cold_misses"] += 1
        self.m["hits"] += matched
        r.prefix_hit = matched > 0
        r.matched_blocks = matched
        restored_tokens = min(
            r.ollama_restored_blocks * bt,
            max(0, node.prompt_tokens - matched_tokens))
        r.prefill_tokens = max(0, node.prompt_tokens - matched_tokens
                               - restored_tokens)
        r.prefill_remaining = r.prefill_tokens

        for k in chain[:matched]:
            self.entries[k].last_used = self.now
        if self.cfg.record_key_uses:
            for k in chain[:matched] + list(new_keys):
                self._key_uses.setdefault(k, []).append(self.now)
                self._key_nbytes[k] = block_bytes
        for k in new_keys:
            self.entries[k] = KVEntry(
                key=k, model=r.model, tokens=bt, nbytes=block_bytes,
                created=self.now, last_used=self.now,
                useful_tokens=bt,
                recompute_s=self.cost.recompute_seconds(r.model, bt),
                # Per-block gap when the loader computed one (trace path);
                # the node-level field otherwise (synthetic path, where
                # cross-task sharing cannot occur). See Node.block_reuse_gap.
                true_gap_s=node.block_reuse_gap.get(k, node.next_reuse_gap_s),
                hint_gap_s=(self.hints.hint_for(node, k) if self.hints else None))
            self.entries[k].clock = self._clock
            self.entries[k].rent_cum_at = self._clock.cum
            self._entry_bytes += block_bytes
            heapq.heappush(self._lru_heap, (self.now, k))
            self._evictable_bytes += block_bytes
            self._evictable_n += 1
            self._model_blocks[r.model] = self._model_blocks.get(r.model, 0) + 1
            self.seen_keys.add(k)
            self._key_recompute[k] = self.cost.recompute_seconds(r.model, bt)

        for k in chain:
            e = self.entries[k]
            if e.refcount == 0:
                # Becoming pinned: bank what it accrued while displaceable.
                e.bank_rent()
                self._evictable_bytes -= e.nbytes
                self._evictable_n -= 1
                self._model_blocks[e.model] = self._model_blocks.get(e.model, 1) - 1
            e.refcount += 1
        r.held_blocks = list(chain)

        if self.cfg.cache_mode == "ollama":
            self._commit_ollama(r)

        r.private_bytes = output_bytes
        self.pinned_bytes += output_bytes
        r.state = State.PREFILL
        r.started_at = self.now
        self.running.append(r)
        dl = self.task_deadline.get(node.task_id)
        if dl is not None and node.task_id not in self.task_infeasible_at_first_admit:
            self.task_infeasible_at_first_admit[node.task_id] = (
                self.now + node.critical_path_remaining > dl)

        self._log(f"  ADMIT req{r.req_id} t{node.task_id}/{node.stage} "
                  f"{matched}/{len(chain)} blocks hit  prefill={r.prefill_tokens}tok "
                  f"batch={len(self.running)} kv={self.used_bytes/1024/1024:.0f}/"
                  f"{self.cfg.kv_budget_mb:.0f}MB")
        return True

    def _evict(self, entry: KVEntry, forced: bool):
        if entry.refcount == 0:
            self._key_rent[entry.key] = (self._key_rent.get(entry.key, 0.0)
                                         + entry.rent_accrued - entry.rent_banked)
            self._evictable_bytes -= entry.nbytes
            self._evictable_n -= 1
            self._model_blocks[entry.model] = self._model_blocks.get(entry.model, 1) - 1
        self._entry_bytes -= entry.nbytes
        del self.entries[entry.key]
        self._evict_rate += 1.0 / max(1e-9, self.cfg.ewma_tau_s)
        self.m["evictions"] += 1
        if entry.is_tail:
            self.m["tail_evictions"] += 1
        if forced:
            self.m["forced_evictions"] += 1
        self._log(f"  EVICT {entry.key} ({entry.nbytes/1024/1024:.0f}MB, "
                  f"held {self.now - entry.created:.1f}s, "
                  f"rent {entry.rent_accrued:.2f}s)")

    def _miss_by_class(self) -> dict:
        out = {}
        for t in self.tasks:
            done, dl = self.task_done_at.get(t.task_id), self.task_deadline.get(t.task_id)
            if done is None or dl is None or t.task_id not in self.task_slo:
                continue
            c = out.setdefault(self.task_slo[t.task_id][0], [0, 0])
            c[0] += done > dl
            c[1] += 1
        return {k: {"miss_rate": m / n, "tasks": n} for k, (m, n) in out.items()}

    def _view(self, r: Request) -> ReqView:
        """
        Build the restricted view a scheduler is allowed to see.

        Estimates are recomputed here rather than cached, so a scheduler sees
        the estimator's CURRENT beliefs -- that is what makes online learning
        visible in the ordering. The per-request noise factor is cached inside
        the estimator, so this does not jitter.
        """
        n = r.node
        est = self.estimator
        return ReqView(
            req_id=r.req_id, task_id=n.task_id, stage=n.stage, model=n.model,
            enqueued_at=r.enqueued_at, prompt_tokens=n.prompt_tokens,
            attempt_index=n.attempt_index, replan_index=n.replan_index,
            priority=0, deadline=self.task_deadline.get(n.task_id),
            task_arrival_s=self.task_arrival.get(n.task_id, 0.0),
            task_attained_s=self.task_attained.get(n.task_id, 0.0),
            slo_class=self.task_slo.get(n.task_id, (None, 0))[0],
            slo_priority=self.task_slo.get(n.task_id, (None, 0))[1],
            # true_output and node are handed over so an ORACLE estimator can
            # use them; ExecutionEstimator ignores both by contract (there are
            # tests asserting it). Without them an oracle run produced
            # estimates identical to an online one.
            est_service_s=est.service_s(n.model, n.prompt_tokens, n.stage,
                                        n.attempt_index,
                                        true_output=n.output_tokens,
                                        req_id=r.req_id) if est else 0.0,
            est_remaining_s=est.remaining_s(n.stage, n.attempt_index, n.model,
                                            req_id=r.req_id,
                                            node=n) if est else 0.0,
        )

    def _admit_loop(self):
        """Admit as many waiting requests as fit. Returns the KV-blocked ones."""
        blocked = []
        self._last_queue = len(self.waiting)
        self.m["admit_calls"] += 1
        self.m["max_queue"] = max(self.m["max_queue"], self._last_queue)
        if self._last_queue >= 2:
            self.m["calls_with_choice"] += 1
        if not self.waiting:
            return blocked

        by_id = {r.req_id: r for r in self.waiting}
        views = [self._view(r) for r in self.waiting]
        order = self.scheduler.order(views, self.now)
        # A scheduler may also set how many requests run at once, never above
        # max_batch and never below 1 (so an idle engine always admits).
        # Capping here rather than by dropping requests from order() keeps the
        # run loop's termination logic intact.
        cap = self.cfg.max_batch
        want = self.scheduler.admit_cap(self.now, views, self.cost, self.cfg.max_batch)
        if want is not None:
            cap = max(1, min(cap, int(want)))
        self.m["w_cap"] = self.m.get("w_cap", 0.0) + cap
        self.m["cap_calls"] = self.m.get("cap_calls", 0) + 1

        for req_id in order:
            r = by_id.get(req_id)
            if r is None:
                continue
            if len(self.running) >= cap:
                break              # batch-slot limited, not memory limited
            if self._try_admit(r):
                self.waiting.remove(r)
            else:
                blocked.append(r)
        return blocked

    # -- execution ---------------------------------------------------------

    def _choose_active_model(self):
        """
        One engine serves one model's batch per iteration. Picks the model with
        the most running requests, tie-broken by who has waited longest.
        """
        counts = {}
        for r in self.running:
            counts.setdefault(r.model, []).append(r)
        if not counts:
            return None
        return max(counts, key=lambda m: (len(counts[m]),
                                          -min(x.enqueued_at for x in counts[m])))

    def _iteration(self) -> float:
        """Run one engine iteration. Returns its duration."""
        model = self._choose_active_model()
        dt = 0.0
        if self.active_model is not None and model != self.active_model:
            dt += self.cfg.model_switch_seconds
            self.m["switch_seconds"] += self.cfg.model_switch_seconds
        self.active_model = model

        batch = [r for r in self.running if r.model == model]

        # At most one prefill per iteration. A long prefill stalls the whole
        # batch -- which is precisely why a cache miss is expensive to everyone,
        # not just to the request that missed.
        prefill = next((r for r in batch if r.state is State.PREFILL), None)
        if prefill is not None:
            chunk = prefill.prefill_remaining
            if self.cfg.prefill_chunk_tokens > 0:
                chunk = min(chunk, self.cfg.prefill_chunk_tokens)
            t = self.cost.prefill_seconds(model, chunk)
            dt += t
            self.m["prefill_seconds"] += t
            prefill.prefill_remaining -= chunk
            if prefill.prefill_remaining <= 0:
                prefill.state = State.DECODE

        decoders = [r for r in batch if r.state is State.DECODE and r is not prefill]
        if decoders:
            t = self.cost.decode_iteration_seconds(model, len(decoders))
            dt += t
            self.m["decode_seconds"] += t
            for r in decoders:
                # No KV growth here: the full output was reserved at admission,
                # so the bytes are already counted. Growing again would double-count.
                r.generated += 1

        dt += self.cost.iteration_overhead_s
        self.m["iterations"] += 1
        # Two different things, both worth watching: how many requests are
        # resident (and holding KV) vs how many actually advance this iteration.
        # They diverge whenever both models have work, because one engine serves
        # one model's batch at a time. Recorded by run(), weighted by dt.
        self._last_batch = len(batch)
        self._last_resident = len(self.running)
        self.m["max_batch"] = max(self.m["max_batch"], self._last_batch)
        self.m["max_resident"] = max(self.m["max_resident"], self._last_resident)
        return dt

    def _lru_evictable(self, k=2):
        """The k least-recently-used evictable entries, cheapest first.

        Rows go stale when an entry is evicted, pinned, or touched, so they are
        validated against the live entry and discarded on the way past."""
        out = []
        while self._lru_heap and len(out) < k:
            lu, key = self._lru_heap[0]
            e = self.entries.get(key)
            if e is None or e.refcount > 0 or e.last_used != lu:
                heapq.heappop(self._lru_heap)
                continue
            out.append(e)
            heapq.heappop(self._lru_heap)
        for e in out:
            heapq.heappush(self._lru_heap, (e.last_used, e.key))
        return out

    def _decay_evict_rate(self, dt: float) -> float:
        """
        EWMA of an event rate: each eviction adds 1/tau, and the accumulator
        decays by exp(-dt/tau). A steady e evictions/second settles at e.
        """
        self._evict_rate *= math.exp(-dt / max(1e-9, self.cfg.ewma_tau_s))
        return self._evict_rate

    def _need_bytes(self, r: Request) -> int:
        """KV a request must be given to run: new prefix blocks plus private blocks."""
        node = r.node
        matched = self._match_blocks(r)
        new_blocks = max(0, len(node.block_keys) - matched)
        return (new_blocks * self.cost.kv_bytes(r.model, self.cfg.block_tokens)
                + self.cost.kv_bytes(r.model, self.cfg.blocks_for(node.output_tokens)))

    def _under_pressure(self) -> bool:
        """
        True when the budget is full enough that admitting pending work would
        require evicting something.

        Deliberately wider than "a request is hard-blocked": under LRU the
        engine almost never blocks, it just evicts, so blocking-only rent is
        far too sparse to drive a policy (HANDOFF section 9.2).
        """
        if not self.waiting:
            return False
        free = self.free_bytes
        # Only the first few candidates matter: if the queue head cannot be
        # admitted without evicting, the system is under pressure. Walking the
        # whole queue meant re-matching every waiting request's entire block
        # chain on every iteration, which at 8x context is thousands of dict
        # lookups per request per iteration.
        for r in self.waiting[:4]:
            if self._need_bytes(r) > free:
                return True
        return False

    def _accrue_rent(self, dt: float, blocked, pressure=None, evictable=None):
        """Dispatch to the configured rent model. Blocking stats are recorded
        either way so the two remain comparable."""
        if dt > 0 and blocked:
            self.m["kv_blocked_seconds"] += dt * len(blocked)
            for r in blocked:
                r.kv_blocked_seconds += dt
        if dt <= 0:
            return
        if self.cfg.rent_model == "blocking":
            self._accrue_blocking_rent(dt, blocked)
        else:
            if pressure is None:
                pressure = self._under_pressure()
            if evictable is None:
                evictable = [e for e in self.entries.values() if e.evictable]
            self.m["rent_ticks"] += 1
            if pressure and evictable:
                self.m["rent_active_ticks"] += 1
            self._accrue_derived_rent(dt, pressure, evictable)

    def _accrue_blocking_rent(self, dt: float, blocked):
        """Phase 1 model, kept selectable for comparison. Charges only while a
        request is hard-blocked, splitting the blocked time by entry size."""
        if not blocked:
            return
        evictable = [e for e in self.entries.values() if e.evictable]
        total = sum(e.nbytes for e in evictable)
        if total <= 0:
            self.m["blocked_no_evictable"] += 1
            return
        self.m["blocked_with_evictable"] += 1
        charge = dt * len(blocked)
        self.m["total_rent"] += charge
        for e in evictable:
            e.rent_accrued += charge * (e.nbytes / total)

    def _accrue_derived_rent(self, dt: float, pressure: bool, evictable):
        if self._rent_impl == "sweep":
            return self._accrue_derived_rent_sweep(dt, pressure, evictable)

        if not pressure:
            return
        if self._evictable_n < 2:
            # No other entry to be the marginal victim, so holding displaces
            # nothing. Rule 3 of HANDOFF section 13.
            self.m["blocked_no_evictable"] += 1
            return

        pair = self._lru_evictable(2)
        if len(pair) < 2:
            self.m["blocked_no_evictable"] += 1
            return
        self.m["blocked_with_evictable"] += 1
        first, second = pair

        pool = self._evictable_bytes
        if pool <= 0:
            return
        rate = self._evict_rate

        # Everyone is charged against `first`, the entry that would actually go
        # next -- rent per byte depends on the VICTIM, not on the holder, so it
        # is one number shared by every entry regardless of model. Using the
        # true LRU entry rather than a per-model stand-in is what makes this
        # exact with a mixed-model trace.
        per_byte = first.recompute_s * rate / pool

        # `first` cannot be its own victim, so it is charged against `second`.
        # Bank what it accrued at the shared rate before advancing the clock,
        # then add its own term and re-baseline so the interval is not counted
        # twice.
        first.bank_rent()
        self._clock.cum += per_byte * dt
        first_rate = first.nbytes * second.recompute_s * rate / pool
        first.rent_banked += first_rate * dt
        first.rent_cum_at = self._clock.cum

        self.m["total_rent"] += per_byte * dt * (pool - first.nbytes) + first_rate * dt

    def _accrue_derived_rent_sweep(self, dt: float, pressure: bool, evictable):
        """
        rent_rate(E) = E.bytes x (marginal victim's recompute seconds per byte)
                       x smoothed eviction rate

        Accrued rent is therefore in recompute-seconds, directly comparable to
        E's own recompute cost -- which is the ski-rental comparison in
        consistent units. Both factors are observed: the victim's cost comes
        from CostModel prefill physics, the rate is counted from the engine.
        There is no memory price anywhere.

        The marginal victim excludes E itself. Charging E at its own rate would
        make break-even exactly one time unit for every entry, which is
        degenerate and unit-dependent.
        """
        if not pressure:
            return
        # Measure the LIVE evictable set, exactly as the lazy clock does, so
        # the two are comparable entry by entry. An earlier version charged a
        # pre-admission snapshot, which sampled a different pool and made the
        # per-entry rents disagree even though the totals matched.
        evictable = [e for e in self.entries.values() if e.evictable]
        if len(evictable) < 2:
            # With no other evictable entry there is no victim, so holding E
            # displaces nothing. Rule 3 of HANDOFF section 13.
            self.m["blocked_no_evictable"] += 1
            return
        self.m["blocked_with_evictable"] += 1

        order = sorted(evictable, key=lambda e: (e.last_used, e.key))
        first, second = order[0], order[1]
        rate = self._evict_rate
        # Normalised by the evictable pool, so the charges SUM to the damage
        # actually being done -- one victim's recompute per eviction -- instead
        # of charging every entry the full amount.
        #
        # Phase 4 wrote this as E.bytes x victim_recompute_per_byte x rate,
        # which does not conserve: total rent scaled with the number of
        # entries. That was tolerable with a handful of whole-prefix entries
        # and became pathological under block granularity, where a run holds
        # thousands of 16-token blocks and total rent inflated ~100x.
        # Break-even is unaffected: rent and benefit are both linear in size,
        # so the entry size still cancels.
        pool = sum(e.nbytes for e in evictable) or 1
        for e in evictable:
            victim = second if e is first else first
            rent_rate = (e.nbytes / pool) * victim.recompute_s * rate
            e.rent_rate = rent_rate
            e.rent_banked = e.rent_accrued + rent_rate * dt
            e.rent_cum_at = self._clock.cum
            self._key_rent[e.key] = self._key_rent.get(e.key, 0.0) + rent_rate * dt
            self.m["total_rent"] += rent_rate * dt

    def _voluntary_release(self):
        """
        Let the retention policy give entries up of its own accord.

        Ski-rental needs this: its whole decision is "hold until accrued rent
        reaches the recompute cost, then let go". Without a per-iteration hook
        a policy could only ever react to pressure, never anticipate it.
        """
        release = self.retention.on_tick(list(self.entries.values()), self.now)
        # Counted separately from voluntary_evictions: a policy can ASK to
        # release a key that turns out to be pinned. The gap between the two
        # tells you whether the rule is firing at all, which is exactly what
        # was invisible when ski-rental made zero releases across 24M ticks.
        self.m["voluntary_releases"] += len(release)
        for key in release:
            e = self.entries.get(key)
            if e is not None and e.evictable:
                self.m["voluntary_evictions"] += 1
                self._evict(e, forced=False)

    def _retire(self):
        for r in list(self.running):
            if r.state is State.DECODE and r.generated >= r.node.output_tokens:
                r.state = State.DONE
                r.finished_at = self.now
                tid = r.node.task_id
                self.task_attained[tid] = (self.task_attained.get(tid, 0.0)
                                           + r.finished_at - r.started_at)
                self.running.remove(r)
                self.pinned_bytes -= r.private_bytes
                for k in r.held_blocks:
                    e = self.entries.get(k)
                    if e is None:
                        continue
                    e.refcount -= 1
                    if e.refcount == 0:
                        e.rent_cum_at = self._clock.cum
                        heapq.heappush(self._lru_heap, (self.now, e.key))
                        self._evictable_bytes += e.nbytes
                        self._evictable_n += 1
                        self._model_blocks[e.model] = self._model_blocks.get(e.model, 0) + 1
                    e.last_used = self.now
                    if e.refcount == 0 and not self.retention.on_complete(e, self.now):
                        self._evict(e, forced=False)
                self._log(f"  DONE  req{r.req_id} t{r.node.task_id}/{r.node.stage} "
                          f"({r.finished_at - r.enqueued_at:.2f}s e2e)")
                if self.estimator is not None:
                    true_out = r.node.output_tokens
                    if r.predicted_output is not None and true_out > 0:
                        ape = abs(r.predicted_output - true_out) / true_out * 100.0
                        self.m["ape_by_stage"].setdefault(r.node.stage, []).append(ape)
                    # Learn from the completion. This is the only place the
                    # online estimator ever sees a true output length.
                    self.estimator.observe(r.node.stage, r.node.attempt_index, true_out)
                self._finish_node(self._task_of(r.node), r.node)

    # -- main loop ---------------------------------------------------------

    def run(self, max_iterations=5_000_000):
        self._released = set()
        guard = 0
        while guard < max_iterations:
            guard += 1

            while self.events and self.events[0][0] <= self.now + 1e-12:
                _, _, kind, payload = heapq.heappop(self.events)
                if kind == "arrival":
                    self._log(f"ARRIVE task {payload.task_id}")
                    self._release_ready(payload)
                else:
                    task, node = payload
                    self._finish_node(task, node)

            # Snapshot BEFORE admitting. _try_admit evicts whatever it needs to
            # fit, so by the time the admit loop returns the evictable set has
            # already been drained and the cache looks empty even though those
            # entries were held for the whole interval being charged.
            pressure = self._under_pressure()
            # Building this list is itself O(entries); only the debug sweep
            # needs it.
            evictable_before = ([e for e in self.entries.values() if e.evictable]
                                if self._rent_impl == "sweep" else None)

            blocked = self._admit_loop()

            if not self.running:
                if blocked and not self.events:
                    raise RuntimeError(
                        "deadlock: requests are KV-blocked with nothing running "
                        "to free memory. This should be impossible; check reservation math.")
                if not self.events:
                    break
                nxt = self.events[0][0]
                if nxt > self.now:
                    self.m["idle_seconds"] += nxt - self.now
                    self.now = nxt
                continue

            if blocked:
                self.m["blocked_events"] += 1
                if self.do_trace and self.m["blocked_events"] <= 40:
                    r = blocked[0]
                    self._log(f"  BLOCKED {len(blocked)} req(s) on KV: "
                              f"req{r.req_id} t{r.node.task_id}/{r.node.stage} needs space; "
                              f"kv={self.used_bytes/1024/1024:.0f}/{self.cfg.kv_budget_mb:.0f}MB "
                              f"(pinned {self.pinned_bytes/1024/1024:.0f}MB, "
                              f"retained {(self.used_bytes-self.pinned_bytes)/1024/1024:.0f}MB)")

            dt = self._iteration()
            self._accrue_rent(dt, blocked, pressure, evictable_before)
            self._voluntary_release()
            self.now += dt
            self.m["w_time"] += dt
            self.m["w_batch"] += self._last_batch * dt
            bt = self.m["batch_time"]
            bt[self._last_batch] = bt.get(self._last_batch, 0.0) + dt
            self.m["w_resident"] += self._last_resident * dt
            rt = self.m["resident_time"]
            rt[self._last_resident] = rt.get(self._last_resident, 0.0) + dt
            self.m["w_kv"] += self.used_bytes * dt
            self.m["w_pinned"] += self.pinned_bytes * dt
            self.m["w_retained"] += (self.used_bytes - self.pinned_bytes) * dt
            self.m["w_queue"] += getattr(self, "_last_queue", 0) * dt
            self._decay_evict_rate(dt)
            self.m["w_evict_rate"] += self._evict_rate * dt
            self.m["peak_kv_bytes"] = max(self.m["peak_kv_bytes"], self.used_bytes)
            # Bytes no retention policy could free: running requests' private KV
            # plus the shared blocks they hold. The floor of any workable budget.
            self.m["peak_active_bytes"] = max(
                self.m["peak_active_bytes"],
                self.pinned_bytes + self._entry_bytes - self._evictable_bytes)
            self._retire()

        return self.report()

    # -- reporting ---------------------------------------------------------

    def report(self) -> dict:
        lat = [self.task_done_at[t.task_id] - t.arrival_time
               for t in self.tasks if t.task_id in self.task_done_at]
        lat.sort()

        def pct(p):
            if not lat:
                return float("nan")
            return lat[min(len(lat) - 1, int(p * len(lat)))]

        # Deadline accounting, over completed tasks only. Slack is signed:
        # positive means finished early, negative means missed and by how much.
        # Reporting the mean of the signed value rather than of |slack| keeps
        # "just barely late" distinguishable from "hours late".
        met, slacks = 0, []
        miss_infeasible = miss_feasible = 0
        for t in self.tasks:
            done = self.task_done_at.get(t.task_id)
            dl = self.task_deadline.get(t.task_id)
            if done is None or dl is None:
                continue
            slacks.append(dl - done)
            if done <= dl:
                met += 1
            elif self.task_infeasible_at_first_admit.get(t.task_id, False):
                miss_infeasible += 1
            else:
                miss_feasible += 1
        n_dl = len(slacks)

        # Time-weighted percentile of the active batch (in-flight requests).
        bt = self.m["batch_time"]
        bt_total = sum(bt.values())
        rt_total = sum(self.m["resident_time"].values())

        def batch_pct(q):
            acc = 0.0
            for b in sorted(bt):
                acc += bt[b]
                if acc >= q * bt_total:
                    return b
            return 0

        # Per-entry hindsight: for each block key, the cheaper of rebuilding it
        # and the rent it actually accrued. Summed, that is what a policy with
        # perfect foresight would have spent, so the ratio below is a
        # measurement against an achievable floor rather than against another
        # heuristic's guess.
        hindsight = sum(min(self._key_recompute.get(k, 0.0), rent)
                        for k, rent in self._key_rent.items())
        hindsight += sum(v for k, v in self._key_recompute.items()
                         if k not in self._key_rent and False)
        policy_cost = self.m["total_rent"] + self.m["recompute_seconds"]

        all_ape = [x for v in self.m["ape_by_stage"].values() for x in v]
        mape_by_stage = {stage: statistics.fmean(v)
                         for stage, v in self.m["ape_by_stage"].items() if v}

        makespan = max(self.task_done_at.values()) if self.task_done_at else 0.0
        wt = self.m["w_time"]

        def w(key):
            """Time-weighted mean of an accumulator."""
            return self.m[key] / wt if wt > 0 else 0.0

        busy = self.m["prefill_seconds"] + self.m["decode_seconds"]
        lookups = self.m["hits"] + self.m["cold_misses"] + self.m["recompute_misses"]

        return {
            "scheduler": self.scheduler.name,
            "retention": self.retention.name,
            "estimator": self.estimator.name if self.estimator else "none",
            "slack_factor": self.cfg.slack_factor,
            "isolated_task_s": self._isolated_task_s,
            "kv_budget_mb": self.cfg.kv_budget_mb,
            "max_batch": self.cfg.max_batch,
            "tasks_completed": len(lat),
            "makespan_s": makespan,
            "p50_task_latency_s": pct(0.50),
            "p95_task_latency_s": pct(0.95),
            "p99_task_latency_s": pct(0.99),
            "deadline_miss_rate": (n_dl - met) / n_dl if n_dl else float("nan"),
            # The miss rate split by whether the deadline was already out of
            # reach (uncontended critical path) at the task's first admission.
            # Only the second kind is one a scheduler could have prevented.
            "miss_rate_infeasible_at_admit": miss_infeasible / n_dl if n_dl else float("nan"),
            "miss_rate_feasible_at_admit": miss_feasible / n_dl if n_dl else float("nan"),
            "miss_rate_by_slo_class": self._miss_by_class(),
            "mean_slack_at_completion_s": (statistics.fmean(slacks) if slacks
                                           else float("nan")),
            "estimator_mape_pct": statistics.fmean(all_ape) if all_ape else float("nan"),
            "estimator_mape_by_stage": mape_by_stage,
            "rent_model": self.cfg.rent_model,
            "block_tokens": self.cfg.block_tokens,
            "ewma_tau_s": self.cfg.ewma_tau_s,
            "tail_entries_created": self.m["tail_entries_created"],
            "voluntary_evictions": self.m["voluntary_evictions"],
            "voluntary_releases_requested": self.m["voluntary_releases"],
            "cache_mode": self.cfg.cache_mode,
            "ollama_slot_reuse": self.m["ollama_slot_reuse"],
            "ollama_host_restore": self.m["ollama_host_restore"],
            "ollama_discarded_blocks": self.m["ollama_discarded_blocks"],
            "ollama_host_saves": self.m["ollama_host_saves"],
            "rent_active_tick_frac": (
                self.m["rent_active_ticks"] / self.m["rent_ticks"]
                if self.m["rent_ticks"] else 0.0),

            "tail_evictions": self.m["tail_evictions"],
            "mean_eviction_rate_per_s": w("w_evict_rate"),
            "hindsight_cost_s": hindsight,
            "policy_cost_s": policy_cost,
            "realized_ratio": (policy_cost / hindsight) if hindsight > 0 else float("nan"),
            "mean_queue_depth": w("w_queue"),
            "max_queue_depth": self.m["max_queue"],
            "frac_decisions_with_choice": (
                self.m["calls_with_choice"] / self.m["admit_calls"]
                if self.m["admit_calls"] else 0.0),
            "mean_task_latency_s": statistics.fmean(lat) if lat else float("nan"),
            "throughput_tasks_per_s": len(lat) / makespan if makespan else 0.0,
            "gpu_busy_frac": busy / makespan if makespan else 0.0,
            "mean_batch": w("w_batch"),
            "max_batch_seen": self.m["max_batch"],
            "p95_batch": batch_pct(0.95) if bt_total else 0,
            "frac_time_at_batch_cap": (bt.get(self.cfg.max_batch, 0.0) / bt_total
                                       if bt_total else 0.0),
            "mean_resident": w("w_resident"),
            "max_resident": self.m["max_resident"],
            # Mean concurrency cap the scheduler asked for, per admission call
            # (max_batch for schedulers that do not set one).
            "mean_admit_cap": (self.m["w_cap"] / self.m["cap_calls"]
                               if self.m.get("cap_calls") else float(self.cfg.max_batch)),
            "frac_time_at_resident_cap": (
                self.m["resident_time"].get(self.cfg.max_batch, 0.0) / rt_total
                if rt_total else 0.0),
            "mean_kv_util": w("w_kv") / self.cfg.kv_budget_bytes,
            "peak_kv_util": self.m["peak_kv_bytes"] / self.cfg.kv_budget_bytes,
            "prefix_hit_rate": self.m["hits"] / lookups if lookups else 0.0,
            "evictions": self.m["evictions"],
            "forced_evictions": self.m["forced_evictions"],
            "recompute_seconds": self.m["recompute_seconds"],
            "recompute_tokens": self.m["recompute_tokens"],
            # Misses on a block that HAD been seen before: recompute the
            # retention policy could have avoided. A block never seen is a cold
            # miss nobody could prevent, so these two separate "the policy was
            # wrong" from "the policy had no say".
            "retention_controlled_decisions": self.m["recompute_misses"],
            "cold_misses": self.m["cold_misses"],
            "kv_blocked_seconds": self.m["kv_blocked_seconds"],
            "blocked_iterations": self.m["blocked_events"],
            "blocked_with_evictable": self.m["blocked_with_evictable"],
            "blocked_no_evictable": self.m["blocked_no_evictable"],
            "mean_pinned_mb": w("w_pinned") / 1024 / 1024,
            "mean_retained_mb": w("w_retained") / 1024 / 1024,
            "total_rent_s": self.m["total_rent"],
            "prefill_seconds": self.m["prefill_seconds"],
            "decode_seconds": self.m["decode_seconds"],
            "idle_seconds": self.m["idle_seconds"],
        }


def scheduling_provenance(cfg: EngineConfig, estimator) -> str:
    """
    Provenance for the knobs CostModel does not own.

    Kept separate from CostModel.provenance() because these are scheduling
    policy parameters, not cost physics -- putting slack_factor on CostModel
    would imply it was measured from the hardware.
    """
    lines = ["Scheduling provenance:"]
    lines.append(f"  slack_factor         {cfg.slack_factor}  "
                 f"[PLACEHOLDER - not derived from any stated SLO]")
    if estimator is None:
        lines.append("  estimator            none")
        return "\n".join(lines)
    tag = ("  [ORACLE - upper-bound reference, not a deployable policy]"
           if estimator.is_oracle else "  [online, learns from completions]")
    lines.append(f"  estimator            {estimator.name}{tag}")
    lines.append(f"  estimator noise      sigma={estimator.noise_sigma}  "
                 f"[PLACEHOLDER - sweep this]")
    lines.append(f"  attempt distribution {estimator.attempt_weights}  "
                 f"[MEASURED from data/outcomes/results.jsonl, 120 runs]")
    lines.append(f"  isolated task time   "
                 f"{estimator.estimate_task_total(CODER_MODEL):.1f}s  [DERIVED from the above]")
    return "\n".join(lines)


def activity_summary(rep) -> dict:
    """
    Which mechanisms were actually ACTIVE in this run.

    Each of the engine's three mechanisms can be silently inert, and each one
    has already produced a comparison that measured nothing while printing a
    perfectly plausible table:

      scheduling  at max_batch=16 the queue is nearly always trivial, so the
                  scheduler has no choice to make (FINDINGS 23)
      retention   with no eviction pressure rent never accrues, so the
                  ski-rental rule never fires (OPEN_ISSUES 26)
      reuse       with nothing blocking on KV, retention cannot matter at all

    So inactivity is reported as data, not left to prose.
    """
    sched = rep.get("frac_decisions_with_choice", 0.0)
    rent = rep.get("rent_active_tick_frac", 0.0)
    rel = rep.get("voluntary_releases_requested", 0)
    inactive = []
    if sched < 0.10:
        inactive.append("scheduling")
    if rent < 0.01:
        inactive.append("rent")
    if rel == 0:
        inactive.append("voluntary-release")
    return {"sched_choice_frac": sched, "rent_active_tick_frac": rent,
            "voluntary_releases_requested": rel, "inactive": inactive}


def activity_line(rep) -> str:
    a = activity_summary(rep)
    flag = ("  ACTIVE" if not a["inactive"]
            else "  INACTIVE: " + ",".join(a["inactive"]))
    return (f"  activity   sched-choice {a['sched_choice_frac']:>6.1%}  "
            f"rent-active ticks {a['rent_active_tick_frac']:>6.1%}  "
            f"vol-releases {a['voluntary_releases_requested']:>7}{flag}")


def format_report(rep: dict, engine: "Engine") -> str:
    L = [
        "=" * 68,
        f"RUN  scheduler={rep['scheduler']}  retention={rep['retention']}  "
        f"estimator={rep['estimator']}",
        f"     B={engine.cfg.max_batch}  KV={engine.cfg.kv_budget_mb:.0f}MB  "
        f"slack_factor={rep['slack_factor']}",
        "=" * 68,
        f"  tasks completed        {rep['tasks_completed']}",
        f"  makespan               {rep['makespan_s']:.1f}s",
        f"  task latency p50/95/99 {rep['p50_task_latency_s']:.1f}s / "
        f"{rep['p95_task_latency_s']:.1f}s / {rep['p99_task_latency_s']:.1f}s",
        f"  throughput             {rep['throughput_tasks_per_s']:.4f} tasks/s",
        "",
        "  -- deadlines --------------------------------------------------",
        f"  isolated task time     {rep['isolated_task_s']:.1f}s (estimated, no contention)",
        f"  deadline miss rate     {rep['deadline_miss_rate']:.1%}",
        f"  mean slack at finish   {rep['mean_slack_at_completion_s']:+.1f}s "
        f"(positive = early)",
        "",
        "  -- scheduling headroom ----------------------------------------",
        f"  queue depth mean/max   {rep['mean_queue_depth']:.2f} / {rep['max_queue_depth']}",
        f"  decisions with choice  {rep['frac_decisions_with_choice']:.1%}  "
        f"(>=2 waiting)",
        "",
        "  -- estimator --------------------------------------------------",
        f"  output-length MAPE     {rep['estimator_mape_pct']:.1f}%",
        *(f"    {stage:<20} {v:.1f}%"
          for stage, v in sorted(rep["estimator_mape_by_stage"].items())),
        "",
        "  -- batching ---------------------------------------------------",
        f"  resident mean / max    {rep['mean_resident']:.2f} / {rep['max_resident']}  (holding KV)",
        f"  active batch mean/max  {rep['mean_batch']:.2f} / {rep['max_batch_seen']}  (advancing per iter)",
        f"  gpu busy fraction      {rep['gpu_busy_frac']:.1%}",
        f"  prefill / decode time  {rep['prefill_seconds']:.0f}s / {rep['decode_seconds']:.0f}s",
        "",
        "  -- KV cache ---------------------------------------------------",
        f"  mean / peak KV util    {rep['mean_kv_util']:.1%} / {rep['peak_kv_util']:.1%}",
        f"  prefix hit rate        {rep['prefix_hit_rate']:.1%}",
        f"  evictions (forced)     {rep['evictions']} ({rep['forced_evictions']})",
        f"  recompute paid         {rep['recompute_seconds']:.1f}s over {rep['recompute_tokens']} tokens",
        "",
        "  -- contention (the thing the old sim could not reach) ---------",
        f"  mean KV split          {rep['mean_pinned_mb']:.0f}MB active / "
        f"{rep['mean_retained_mb']:.0f}MB retained",
        f"  blocked iterations     {rep['blocked_iterations']}",
        f"    retention could help {rep['blocked_with_evictable']}",
        f"    active KV was full   {rep['blocked_no_evictable']}",
        f"  KV-blocked time        {rep['kv_blocked_seconds']:.1f}s (summed over blocked reqs)",
        f"  rent model             {rep['rent_model']}  "
        f"(block={rep['block_tokens']}tok, tau={rep['ewma_tau_s']:g}s)",
        f"  total rent charged     {rep['total_rent_s']:.1f} recompute-s",
        f"  mean eviction rate     {rep['mean_eviction_rate_per_s']:.3f}/s",
        f"  tail entries created   {rep['tail_entries_created']} "
        f"({rep['tail_evictions']} evicted)",
        "=" * 68,
        activity_line(rep),
        "=" * 68,
    ]
    if rep["blocked_iterations"] == 0:
        L.append("  NOTE: nothing ever blocked on KV. Rent is zero, so retention")
        L.append("        policy cannot matter. Lower --kv-mb or raise --batch.")
    elif rep["blocked_with_evictable"] == 0:
        L.append("  NOTE: blocking happened, but never with anything evictable, so")
        L.append("        rent is still zero. The budget is being eaten by ACTIVE")
        L.append("        requests, not by retained prefixes. Raise --kv-mb or lower")
        L.append("        --batch to move into the regime where retention matters.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv-mb", type=float, default=2048.0, help="KV budget in MB")
    ap.add_argument("--batch", type=int, default=8, help="B, max concurrent requests")
    ap.add_argument("--tasks", type=int, default=40)
    ap.add_argument("--rate", type=float, default=0.35, help="task arrivals/sec")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--homogeneous", action="store_true",
                    help="collapse all models onto the qwen cost model")
    ap.add_argument("--scheduler", choices=sorted(SCHEDULERS), default="fcfs")
    ap.add_argument("--retention", choices=sorted(RETENTIONS), default="lru")
    ap.add_argument("--estimator", choices=sorted(ESTIMATORS), default="online",
                    help="'oracle' returns true output lengths; upper-bound "
                         "reference only, labelled as such in output")
    ap.add_argument("--estimator-noise", type=float, default=0.0,
                    metavar="SIGMA",
                    help="multiply each request's estimate by lognormal(0, sigma)")
    ap.add_argument("--rent-model", choices=["derived", "blocking"], default="derived")
    ap.add_argument("--block-tokens", type=int, default=16,
                    help="KV block size (PLACEHOLDER, vLLM default)")
    ap.add_argument("--prefill-chunk", type=int, default=512,
                    help="chunked prefill size in tokens; 0 = unchunked "
                         "(PLACEHOLDER, chunked is the default)")
    ap.add_argument("--ewma-tau", type=float, default=30.0,
                    help="eviction-rate smoothing constant, seconds (PLACEHOLDER)")
    ap.add_argument("--slack-factor", type=float, default=1.5,
                    help="task deadline = arrival + this x isolated task time "
                         "(PLACEHOLDER, not derived from a stated SLO)")
    ap.add_argument("--stable-tokens", type=int, default=None,
                    help="override the measured stable-context size (the parameter "
                         "that sets the whole memory-vs-recompute scale)")
    ap.add_argument("--trace", action="store_true", help="print an event timeline")
    ap.add_argument("--trace-lines", type=int, default=60)
    args = ap.parse_args()

    cost = CostModel(homogeneous=args.homogeneous)
    tok = StageTokens.from_measured()
    if args.stable_tokens is not None:
        tok.stable_prefix_tokens = args.stable_tokens
        tok.measured_fields = tuple(f for f in tok.measured_fields
                                    if f != "stable_prefix_tokens")
    cpu = CpuStageSeconds.from_measured()
    wcfg = WorkloadConfig(n_tasks=args.tasks, arrival_rate=args.rate, seed=args.seed)
    tasks = generate_trace(wcfg, cost, tok, cpu)

    print(cost.provenance())
    print()
    print(f"stable_prefix_tokens = {tok.stable_prefix_tokens} "
          f"({'measured' if tok.measured_fields else 'PLACEHOLDER'})"
          f"  -> {cost.kv_bytes('qwen2.5-coder:7b', tok.stable_prefix_tokens)/1024/1024:.1f}MB of KV, "
          f"recompute {cost.recompute_seconds('qwen2.5-coder:7b', tok.stable_prefix_tokens):.2f}s")
    print(f"cpu stages: pytest={cpu.pytest_seconds}s rag={cpu.rag_index_seconds}s "
          f"({'measured' if cpu.measured else 'PLACEHOLDER'})")
    print()
    estimator = ESTIMATORS[args.estimator](
        cost, tok, cpu, noise_sigma=args.estimator_noise, seed=args.seed)
    print(scheduling_provenance(
        EngineConfig(kv_budget_mb=args.kv_mb, max_batch=args.batch,
                     slack_factor=args.slack_factor, rent_model=args.rent_model,
                     block_tokens=args.block_tokens, ewma_tau_s=args.ewma_tau,
                     prefill_chunk_tokens=args.prefill_chunk), estimator))
    print()
    print(describe_trace(tasks))
    print()

    eng = Engine(
        tasks, cost,
        EngineConfig(kv_budget_mb=args.kv_mb, max_batch=args.batch,
                     slack_factor=args.slack_factor, rent_model=args.rent_model,
                     block_tokens=args.block_tokens, ewma_tau_s=args.ewma_tau,
                     prefill_chunk_tokens=args.prefill_chunk),
        retention=RETENTIONS[args.retention](),
        scheduler=SCHEDULERS[args.scheduler](),
        estimator=estimator,
        trace=args.trace)
    rep = eng.run()

    if args.trace:
        print("--- event timeline (truncated) ---")
        for line in eng.trace_log[:args.trace_lines]:
            print(line)
        print(f"--- {len(eng.trace_log)} events total ---\n")

    print(format_report(rep, eng))


if __name__ == "__main__":
    main()
