"""
Helpers shared by the experiment drivers.

They used to live in run_retention_grid / run_scheduler_table, so scheduling_map
imported from run_scheduler_table, which imported from run_retention_grid, and
sweep_decode_coeff and reanalyze_slo reached across the same way. Drivers now
import from here; the old modules re-export the names so existing imports,
tests and scripts keep working.

The two historical `run_one` functions are different (one builds a retention
cell, the other a scheduler row), so they keep distinct names here and the old
modules alias them back.
"""

from __future__ import annotations

import hashlib
import statistics as st

from simulator.engine import Engine, EngineConfig
from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, HintBuilder,
                                SkiRentalPredictionsRetention)
from simulator.workload import Kind
from simulator.workload_sources import get_source, resolve_workload


def scale_context(tasks, factor: int):
    """
    Grow every prompt by `factor`, keeping the sharing structure intact.

    Each block becomes `factor` blocks derived from it, so calls that shared a
    block still share all of its descendants. prefix_tokens and suffix_tokens
    scale with it, which is what makes a scaled run have proportionally larger
    SHARED blocks rather than merely longer prompts.
    """
    if factor == 1:
        return tasks
    for t in tasks:
        for n in t.nodes:
            if n.kind is not Kind.LLM:
                continue
            n.prefix_tokens *= factor
            n.suffix_tokens *= factor
            n.output_tokens *= factor
            derive = (lambda k, j:
                      hashlib.sha1(f"{k}|x{j}".encode()).hexdigest()[:12])
            # The per-block reuse gaps are keyed by block key, so they must be
            # remapped with the keys. Without this they all miss at scale > 1
            # and the engine silently falls back to the node-level gap, which
            # is None for cross-run reuse -- reintroducing the bug DECISIONS 21
            # fixed, but only in the scaled cells.
            n.block_reuse_gap = {derive(k, j): g
                                 for k, g in n.block_reuse_gap.items()
                                 for j in range(factor)}
            n.block_keys = [derive(k, j)
                            for k in n.block_keys for j in range(factor)]
    return tasks


def build(scale, seed, arrival_rate, cost, tok, cpu, trace_set="original"):
    # The one seam from a trace-set NAME to Tasks. MAARA names reach the same
    # load_trace call as before; other names reach whatever adapter registered
    # them (simulator/workload_sources.py). cost/tok/cpu are unused here and
    # kept for the call signature every driver already uses.
    tasks = resolve_workload(trace_set, seed, arrival_rate)
    return scale_context(tasks, scale)


def make_estimator(trace_set, cost, tok, cpu, name="online"):
    """The estimator for a run on `trace_set`. A source may bring its own
    "online" estimator (the stock one is wired to MAARA stage names); every
    other name, and every MAARA set, gets the stock class exactly as before."""
    src = get_source(trace_set)
    if src.estimator_factory is not None and name == "online":
        return src.estimator_factory(cost, tok, cpu)
    if src.oracle_factory is not None and name == "oracle":
        return src.oracle_factory(cost, tok, cpu)
    return ESTIMATORS[name](cost, tok, cpu)


def run_retention_one(policy, sched, scale, seed, kv_mb, arrival_rate, cost, tok,
                      cpu, batch=8, lam=0.5, sigma=0.5, chunk=512,
                      trace_set="holdout_qwen", record_key_uses=False):
    tasks = build(scale, seed, arrival_rate, cost, tok, cpu, trace_set=trace_set)
    pol = (SkiRentalPredictionsRetention(lam) if policy == "ski-rental-pred"
           else RETENTIONS[policy]())
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=kv_mb, max_batch=batch,
                              prefill_chunk_tokens=chunk,
                              record_key_uses=record_key_uses),
                 retention=pol, scheduler=SCHEDULERS[sched](),
                 estimator=make_estimator(trace_set, cost, tok, cpu),
                 hints=HintBuilder(sigma=sigma, seed=seed))
    return eng, eng.run()


def run_scheduler_one(sched, estimator_name, scale, seed, kv_mb, rate, slack,
                      retention, cost, tok, cpu, trace_set, batch=16, chunk=512,
                      sigma=0.5, slo=None, deadline_basis="estimate"):
    return scheduler_engine(sched, estimator_name, scale, seed, kv_mb, rate, slack,
                            retention, cost, tok, cpu, trace_set, batch=batch,
                            chunk=chunk, sigma=sigma, slo=slo,
                            deadline_basis=deadline_basis).run()


def scheduler_engine(sched, estimator_name, scale, seed, kv_mb, rate, slack,
                     retention, cost, tok, cpu, trace_set, batch=16, chunk=512,
                     sigma=0.5, slo=None, deadline_basis="estimate"):
    """The engine run_scheduler_one runs, unrun, for callers that need per-task
    results after run() (experiments/size_breakdown.py)."""
    tasks = build(scale, seed, rate, cost, tok, cpu, trace_set=trace_set)
    est = make_estimator(trace_set, cost, tok, cpu, name=estimator_name)
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=kv_mb, max_batch=batch,
                              prefill_chunk_tokens=chunk, slack_factor=slack,
                              slo_classes=slo, slo_seed=seed,
                              deadline_basis=deadline_basis),
                 retention=RETENTIONS[retention](),
                 scheduler=SCHEDULERS[sched](),
                 estimator=est,
                 hints=HintBuilder(sigma=sigma, seed=seed))
    return eng


def calibrate_rate(target_util, scale, seed, kv_mb, cost, tok, cpu,
                   lo=0.005, hi=8.0, iters=12, trace_set="holdout_qwen",
                   batch=8, return_achieved=False):
    """Binary-search the arrival rate giving `target_util` GPU busy under
    FCFS/keep-all. Monotone in rate, so bisection is safe.

    With return_achieved, returns (rate, busy fraction measured at that rate)
    for the run manifest; None if no probe reached the target."""
    best, best_busy = hi, None
    for _ in range(iters):
        mid = (lo + hi) / 2
        try:
            _, rep = run_retention_one("keep-all", "fcfs", scale, seed, kv_mb, mid,
                                       cost, tok, cpu, batch=batch,
                                       trace_set=trace_set)
            busy = rep["gpu_busy_frac"]
        except RuntimeError:
            busy = 1.0
        if busy < target_util:
            lo = mid
        else:
            hi = mid
            best, best_busy = mid, busy
    return (best, best_busy) if return_achieved else best


# Two-sided 95% Student t critical values by degrees of freedom. stdlib has no
# t quantile and the sim extra is deliberately numpy/matplotlib only.
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
         7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
         13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
         19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042}


def paired_ci(a, b):
    """Mean of a[i] - b[i] with a two-sided 95% t interval. Paired because
    every row runs the same seeds (DECISIONS 24)."""
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    mean = st.fmean(d)
    if n < 2:
        return {"mean_diff_s": mean, "lo_s": float("nan"),
                "hi_s": float("nan"), "n": n, "significant": False}
    df = n - 1
    t = _T975.get(df) or (_T975[max(k for k in _T975 if k <= df)]
                          if df < 30 else 1.96)
    half = t * st.stdev(d) / n ** 0.5
    lo, hi = mean - half, mean + half
    return {"mean_diff_s": mean, "lo_s": lo, "hi_s": hi, "n": n,
            "significant": not (lo <= 0.0 <= hi)}


def calibrate_slack(scale, seeds, kv_mb, rate, retention, cost, tok, cpu,
                    trace_set, batch=8, lo=0.5, hi=6.0, iters=10,
                    target=(0.20, 0.30), slo=None, deadline_basis="estimate"):
    """
    Bisect slack_factor so FCFS lands in the target deadline-miss band.

    Miss rate FALLS as slack rises, so the branch is inverted relative to
    calibrate_rate. Returns (slack, achieved_miss_rate); the achieved rate is
    returned rather than asserted because a trace may have no slack value that
    lands in the band, and silently reporting a calibrated number that missed
    its target would be worse than reporting the miss.
    """
    best, best_miss = hi, None
    for _ in range(iters):
        mid = (lo + hi) / 2
        try:
            # Average over the same seeds the table will use. Calibrating on
            # one seed and reporting a multi-seed mean produced a confusing
            # mismatch: 22.2% at calibration against 12.3% in the table.
            miss = st.fmean(
                run_scheduler_one("fcfs", "online", scale, sd, kv_mb, rate, mid,
                                  retention, cost, tok, cpu, trace_set, batch=batch,
                                  slo=slo, deadline_basis=deadline_basis
                                  )["deadline_miss_rate"]
                for sd in seeds)
        except RuntimeError:
            lo = mid
            continue
        best, best_miss = mid, miss
        if miss > target[1]:
            lo = mid          # too many misses -> more slack
        elif miss < target[0]:
            hi = mid          # too few -> less slack
        else:
            return mid, miss
    return best, best_miss
