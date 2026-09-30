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
There is deliberately no memory-price knob. Rent is measured, not priced: while
a request is blocked specifically because KV is full, the blocked time is
divided among the retained-but-unreferenced entries in proportion to their size.
An entry's accrued rent is therefore in seconds, directly comparable to
CostModel.recompute_seconds() -- which is exactly the ski-rental comparison.
"""

from __future__ import annotations

import argparse
import heapq
import statistics
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from sim.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, ExecutionEstimator,
                          FCFSScheduler, LRURetention, ReqView)
from sim.workload import (CODER_MODEL, CostModel, CpuStageSeconds, Kind,
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

    @property
    def kv_budget_bytes(self) -> int:
        return int(self.kv_budget_mb * 1024 * 1024)


class State(Enum):
    WAITING = "waiting"
    PREFILL = "prefill"
    DECODE = "decode"
    DONE = "done"


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
    rent_accrued: float = 0.0       # seconds of blocking attributed to this entry
    retained: bool = True

    @property
    def evictable(self) -> bool:
        return self.refcount == 0


@dataclass
class Request:
    req_id: int
    node: object                    # sim.workload.Node
    enqueued_at: float
    state: State = State.WAITING
    started_at: Optional[float] = None
    finished_at: Optional[float] = None

    prefill_tokens: int = 0         # tokens actually prefilled (excludes a cache hit)
    generated: int = 0
    private_bytes: int = 0
    entry: Optional[KVEntry] = None
    prefix_hit: bool = False
    kv_blocked_seconds: float = 0.0
    # Prediction recorded at enqueue, scored against the true output at
    # completion. Kept per request so estimator error is measurable per role.
    predicted_output: Optional[float] = None

    @property
    def model(self) -> str:
        return self.node.model


# Schedulers and retention policies live in sim/policies.py. There used to be a
# second FIFOScheduler here; having two meant the CLI silently used one while
# tests exercised the other.


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------

class Engine:
    def __init__(self, tasks, cost: CostModel, cfg: EngineConfig,
                 retention=None, scheduler=None, estimator=None,
                 trace: bool = False):
        self.tasks = tasks
        self.cost = cost
        self.cfg = cfg
        self.retention = retention or LRURetention()
        self.scheduler = scheduler or FCFSScheduler()
        self.estimator = estimator
        self.do_trace = trace
        self.trace_log = []
        # task_id -> absolute deadline, set once when the task arrives.
        self.task_deadline = {}

        self.now = 0.0
        self.events = []            # heap of (time, seq, kind, payload)
        self._seq = 0
        self.waiting = []
        self.running = []
        self.entries = {}           # key -> KVEntry
        self.pinned_bytes = 0       # private (non-shared) KV of running requests
        self.active_model = None

        self._req_id = 0
        self.done_nodes = {}        # (task_id, node_id) -> finish time
        self.task_done_at = {}
        self.requests = []
        self._released = set()

        # Keys that have existed at some point. A miss on one of these is a
        # recompute the retention policy could have avoided; a miss on a new key
        # is a cold miss nobody could have prevented.
        self.seen_keys = set()

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
            "w_kv": 0.0, "w_pinned": 0.0, "w_retained": 0.0,
            "max_batch": 0, "max_resident": 0,
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
        }

        # Isolated task time is computed ONCE, from the estimator's priors,
        # before any learning has happened. Deadlines must be identical across
        # policy runs on the same trace or the miss rates are not comparable --
        # and if it were recomputed per arrival, a scheduler that happened to
        # finish work sooner would teach the estimator faster and hand itself
        # different deadlines.
        self._isolated_task_s = (
            self.estimator.estimate_task_total(CODER_MODEL) if self.estimator else None)
        for t in tasks:
            if self._isolated_task_s is not None:
                self.task_deadline[t.task_id] = (
                    t.arrival_time + self.cfg.slack_factor * self._isolated_task_s)
            self._push(t.arrival_time, "arrival", t)

    # -- plumbing ----------------------------------------------------------

    def _push(self, when, kind, payload):
        self._seq += 1
        heapq.heappush(self.events, (when, self._seq, kind, payload))

    def _log(self, msg):
        if self.do_trace:
            self.trace_log.append(f"[{self.now:8.3f}s] {msg}")

    @property
    def used_bytes(self) -> int:
        return self.pinned_bytes + sum(e.nbytes for e in self.entries.values())

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

    def _try_admit(self, r: Request) -> bool:
        """
        Returns True if admitted. Reserves KV for prompt AND full output up front:
        with no preemption, a request that runs out of KV mid-decode would have
        nowhere to go, so the engine refuses to start what it cannot finish.
        """
        node = r.node
        key = node.prefix_key
        entry = self.entries.get(key) if key else None
        hit = entry is not None

        prefix_new_bytes = 0 if hit else self.cost.kv_bytes(r.model, node.prefix_tokens)
        private_bytes = self.cost.kv_bytes(r.model, node.suffix_tokens + node.output_tokens)
        need = prefix_new_bytes + private_bytes

        if need > self.cfg.kv_budget_bytes:
            raise RuntimeError(
                f"req{r.req_id} ({node.stage}) needs "
                f"{need/1024/1024:.0f}MB but the whole budget is "
                f"{self.cfg.kv_budget_mb:.0f}MB. Raise kv_budget_mb.")

        if need > self.free_bytes:
            # Pressure. Ask the retention policy what to give up.
            candidates = [e for e in self.entries.values() if e.evictable]
            if hit and entry in candidates:
                candidates.remove(entry)   # never evict the thing we are about to use
            victims = self.retention.choose_victims(candidates, need - self.free_bytes, self.now)
            for v in victims:
                # Test against live free space only. An earlier version also
                # accumulated a `freed` counter and compared it to
                # `need - self.free_bytes` -- but free_bytes already grows as
                # entries are evicted, so the same bytes were counted twice and
                # the loop stopped at roughly half the required eviction. That
                # produced spurious admission failures, and a hard deadlock
                # whenever nothing was running to free memory later.
                if self.free_bytes >= need:
                    break
                self._evict(v, forced=True)
            if need > self.free_bytes:
                return False      # still blocked: caller records this as KV pressure

        if hit:
            self.m["hits"] += 1
            r.prefix_hit = True
            r.prefill_tokens = node.suffix_tokens
            entry.last_used = self.now
        else:
            if key is not None and key in self.seen_keys:
                self.m["recompute_misses"] += 1
                self.m["recompute_tokens"] += node.prefix_tokens
                self.m["recompute_seconds"] += self.cost.recompute_seconds(
                    r.model, node.prefix_tokens)
            elif key is not None:
                self.m["cold_misses"] += 1
            r.prefill_tokens = node.prompt_tokens
            if key is not None:
                entry = KVEntry(key=key, model=r.model, tokens=node.prefix_tokens,
                                nbytes=self.cost.kv_bytes(r.model, node.prefix_tokens),
                                created=self.now, last_used=self.now)
                self.entries[key] = entry
                self.seen_keys.add(key)

        if entry is not None:
            entry.refcount += 1
        r.entry = entry
        r.private_bytes = private_bytes
        self.pinned_bytes += private_bytes
        r.state = State.PREFILL
        r.started_at = self.now
        self.running.append(r)

        self._log(f"  ADMIT req{r.req_id} t{node.task_id}/{node.stage} "
                  f"{'HIT ' if hit else 'MISS'} prefill={r.prefill_tokens}tok "
                  f"batch={len(self.running)} kv={self.used_bytes/1024/1024:.0f}/"
                  f"{self.cfg.kv_budget_mb:.0f}MB")
        return True

    def _evict(self, entry: KVEntry, forced: bool):
        del self.entries[entry.key]
        self.m["evictions"] += 1
        if forced:
            self.m["forced_evictions"] += 1
        self._log(f"  EVICT {entry.key} ({entry.nbytes/1024/1024:.0f}MB, "
                  f"held {self.now - entry.created:.1f}s, "
                  f"rent {entry.rent_accrued:.2f}s)")

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
            est_service_s=est.service_s(n.model, n.prompt_tokens, n.stage,
                                        n.attempt_index, req_id=r.req_id) if est else 0.0,
            est_remaining_s=est.remaining_s(n.stage, n.attempt_index, n.model,
                                            req_id=r.req_id) if est else 0.0,
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
        order = self.scheduler.order([self._view(r) for r in self.waiting], self.now)

        for req_id in order:
            r = by_id.get(req_id)
            if r is None:
                continue
            if len(self.running) >= self.cfg.max_batch:
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
            t = self.cost.prefill_seconds(model, prefill.prefill_tokens)
            dt += t
            self.m["prefill_seconds"] += t
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

    def _accrue_rent(self, dt: float, blocked):
        """
        Charge retained entries for the delay they are causing. This is the only
        place rent enters the simulation, and it comes from observed blocking.
        """
        if not blocked or dt <= 0:
            return
        self.m["kv_blocked_seconds"] += dt * len(blocked)
        for r in blocked:
            r.kv_blocked_seconds += dt

        evictable = [e for e in self.entries.values() if e.evictable]
        total = sum(e.nbytes for e in evictable)
        if total <= 0:
            # Budget is full of ACTIVE request KV. Retention had no lever here,
            # so no entry is charged.
            self.m["blocked_no_evictable"] += 1
            return
        self.m["blocked_with_evictable"] += 1
        charge = dt * len(blocked)
        self.m["total_rent"] += charge
        for e in evictable:
            e.rent_accrued += charge * (e.nbytes / total)

    def _retire(self):
        for r in list(self.running):
            if r.state is State.DECODE and r.generated >= r.node.output_tokens:
                r.state = State.DONE
                r.finished_at = self.now
                self.running.remove(r)
                self.pinned_bytes -= r.private_bytes
                if r.entry is not None:
                    r.entry.refcount -= 1
                    r.entry.last_used = self.now
                    if r.entry.refcount == 0 and not self.retention.on_complete(r.entry, self.now):
                        self._evict(r.entry, forced=False)
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
            self._accrue_rent(dt, blocked)
            self.now += dt
            self.m["w_time"] += dt
            self.m["w_batch"] += self._last_batch * dt
            self.m["w_resident"] += self._last_resident * dt
            self.m["w_kv"] += self.used_bytes * dt
            self.m["w_pinned"] += self.pinned_bytes * dt
            self.m["w_retained"] += (self.used_bytes - self.pinned_bytes) * dt
            self.m["w_queue"] += getattr(self, "_last_queue", 0) * dt
            self.m["peak_kv_bytes"] = max(self.m["peak_kv_bytes"], self.used_bytes)
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
        for t in self.tasks:
            done = self.task_done_at.get(t.task_id)
            dl = self.task_deadline.get(t.task_id)
            if done is None or dl is None:
                continue
            slacks.append(dl - done)
            if done <= dl:
                met += 1
        n_dl = len(slacks)

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
            "mean_slack_at_completion_s": (statistics.fmean(slacks) if slacks
                                           else float("nan")),
            "estimator_mape_pct": statistics.fmean(all_ape) if all_ape else float("nan"),
            "estimator_mape_by_stage": mape_by_stage,
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
            "mean_resident": w("w_resident"),
            "max_resident": self.m["max_resident"],
            "mean_kv_util": w("w_kv") / self.cfg.kv_budget_bytes,
            "peak_kv_util": self.m["peak_kv_bytes"] / self.cfg.kv_budget_bytes,
            "prefix_hit_rate": self.m["hits"] / lookups if lookups else 0.0,
            "evictions": self.m["evictions"],
            "forced_evictions": self.m["forced_evictions"],
            "recompute_seconds": self.m["recompute_seconds"],
            "recompute_tokens": self.m["recompute_tokens"],
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
                 f"[MEASURED from benchmark/results.jsonl, 120 runs]")
    lines.append(f"  isolated task time   "
                 f"{estimator.estimate_task_total(CODER_MODEL):.1f}s  [DERIVED from the above]")
    return "\n".join(lines)


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
        f"  total rent charged     {rep['total_rent_s']:.1f}s",
        "=" * 68,
    ]
    if rep["blocked_iterations"] == 0:
        L.append("  NOTE: nothing ever blocked on KV. Rent is zero, so retention")
        L.append("        policy cannot matter. Lower --kv or raise --batch.")
    elif rep["blocked_with_evictable"] == 0:
        L.append("  NOTE: blocking happened, but never with anything evictable, so")
        L.append("        rent is still zero. The budget is being eaten by ACTIVE")
        L.append("        requests, not by retained prefixes. Raise --kv or lower")
        L.append("        --batch to move into the regime where retention matters.")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kv", type=float, default=2048.0, help="KV budget in MB")
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
        EngineConfig(kv_budget_mb=args.kv, max_batch=args.batch,
                     slack_factor=args.slack_factor), estimator))
    print()
    print(describe_trace(tasks))
    print()

    eng = Engine(
        tasks, cost,
        EngineConfig(kv_budget_mb=args.kv, max_batch=args.batch,
                     slack_factor=args.slack_factor),
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
