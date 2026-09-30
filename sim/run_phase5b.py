"""
Retention x scheduler comparison on a real trace.

    ./venv/bin/python -m sim.run_phase5b            # full grid
    ./venv/bin/python -m sim.run_phase5b --quick    # one util, one scale, 2 seeds

Three things this driver does that the engine cannot do alone.

LOAD BY UTILISATION, NOT ARRIVAL RATE
    The traced run was single-stream, so replaying its timestamps gives no
    contention at all, and picking an arrival rate by hand gives a different
    load at every context scale. `--target-util` binary-searches the arrival
    rate that puts the GPU at the requested busy fraction under FCFS/keep-all,
    so 1x and 16x are compared at the same pressure rather than the same rate.

CONTEXT SCALING THAT SCALES THE SHARED CONTENT
    Multiplying prompt_tokens alone would make prompts longer without making
    the SHARED part longer, which is the opposite of what the sweep is for.
    Each original KV block is replaced by `factor` derived blocks keyed off the
    original, so two calls that shared block i still share all `factor` blocks
    derived from it. Sharing structure is preserved exactly and both prefix and
    suffix token counts scale.

HINDSIGHT AS A SHARED DENOMINATOR
    Per block key, hindsight cost is min(cost to rebuild it, rent it accrues
    while held) -- what an observer with perfect foresight would have paid.
    Measured ONCE under keep-all, where every key is held so its rent is
    actually observed, and then used as the denominator for every policy. Doing
    it per-policy would make the denominator move with the numerator: evict-now
    holds nothing, accrues no rent, and would score against a denominator of
    zero.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import statistics as st

from sim.engine import Engine, EngineConfig
from sim.policies import (RETENTIONS, SCHEDULERS, ExecutionEstimator,
                          FCFSScheduler, HintBuilder, KeepAllRetention,
                          SkiRentalPredictionsRetention)
from sim.trace_loader import TRACE_SETS, load_trace
from sim.workload import (CostModel, CpuStageSeconds, Kind, StageTokens,
                          DEFAULT_BLOCK_TOKENS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ["keep-all", "lru", "evict-now", "ski-rental", "ski-rental-pred",
            "offline-ref"]


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
            n.block_keys = [
                hashlib.sha1(f"{k}|x{j}".encode()).hexdigest()[:12]
                for k in n.block_keys for j in range(factor)]
    return tasks


def build(scale, seed, arrival_rate, cost, tok, cpu, trace_set="original"):
    paths = [os.path.join(_REPO, p) for p in TRACE_SETS[trace_set]]
    tasks = load_trace(paths, arrival_rate=arrival_rate, seed=seed).tasks
    return scale_context(tasks, scale)


def run_one(policy, sched, scale, seed, kv_mb, arrival_rate, cost, tok, cpu,
            batch=16, lam=0.5, sigma=0.5, chunk=512):
    tasks = build(scale, seed, arrival_rate, cost, tok, cpu)
    pol = (SkiRentalPredictionsRetention(lam) if policy == "ski-rental-pred"
           else RETENTIONS[policy]())
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=kv_mb, max_batch=batch,
                              prefill_chunk_tokens=chunk),
                 retention=pol, scheduler=SCHEDULERS[sched](),
                 estimator=ExecutionEstimator(cost, tok, cpu),
                 hints=HintBuilder(sigma=sigma, seed=seed))
    return eng, eng.run()


def calibrate_rate(target_util, scale, seed, kv_mb, cost, tok, cpu,
                   lo=0.005, hi=8.0, iters=12):
    """Binary-search the arrival rate giving `target_util` GPU busy under
    FCFS/keep-all. Monotone in rate, so bisection is safe."""
    best = hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        try:
            _, rep = run_one("keep-all", "fcfs", scale, seed, kv_mb, mid,
                             cost, tok, cpu)
            busy = rep["gpu_busy_frac"]
        except RuntimeError:
            busy = 1.0
        if busy < target_util:
            lo = mid
        else:
            hi = mid
            best = mid
    return best


def hindsight_denominator(scale, seed, kv_mb, arrival_rate, cost, tok, cpu):
    """min(rebuild, rent-while-held) per key, measured under keep-all."""
    eng, _ = run_one("keep-all", "fcfs", scale, seed, kv_mb, arrival_rate,
                     cost, tok, cpu)
    return sum(min(eng._key_recompute.get(k, 0.0), rent)
               for k, rent in eng._key_rent.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kv", type=float, default=1024.0)
    args = ap.parse_args()

    cost = CostModel()
    tok = StageTokens.from_measured()
    cpu = CpuStageSeconds.from_measured()

    utils = [0.6] if args.quick else [0.6, 0.8]
    scales = [1] if args.quick else [1, 8, 16]
    seeds = [1, 2] if args.quick else [1, 2, 3, 4, 5]

    for scale in scales:
        # Bigger contexts need a bigger budget simply to admit one request.
        kv = args.kv * max(1, scale // 2)
        for util in utils:
            rate = calibrate_rate(util, scale, seeds[0], kv, cost, tok, cpu)
            hind = st.fmean([hindsight_denominator(scale, s, kv, rate, cost, tok, cpu)
                             for s in seeds]) or float("nan")
            print(f"=== context {scale}x | target util {util:.0%} "
                  f"(arrival {rate:.3f}/s) | KV {kv:.0f}MB | B=16 | "
                  f"{len(seeds)} seeds | chunked prefill 512 ===")
            print(f"  hindsight denominator {hind:.1f} recompute-s "
                  f"(min(rebuild, rent) per block, measured under keep-all)")
            print(f"  {'retention':<17}{'sched':<6}{'recompute':>11}{'rent':>9}"
                  f"{'p50':>8}{'p95':>9}{'ratio':>8}{'controlled':>12}")
            for pol in POLICIES:
                for sched in ("fcfs", "lsf"):
                    rs = []
                    for s in seeds:
                        try:
                            eng, rep = run_one(pol, sched, scale, s, kv, rate,
                                               cost, tok, cpu)
                            rs.append((eng, rep))
                        except RuntimeError as e:
                            print(f"  {pol:<17}{sched:<6} ERROR {str(e)[:50]}")
                            rs = []
                            break
                    if not rs:
                        continue
                    m = lambda k: st.fmean([r[k] for _, r in rs])
                    ctrl = st.fmean([e.m["recompute_misses"] for e, _ in rs])
                    cost_s = m("recompute_seconds") + m("total_rent_s")
                    ratio = cost_s / hind if hind and hind == hind else float("nan")
                    print(f"  {pol:<17}{sched:<6}{m('recompute_seconds'):>11.1f}"
                          f"{m('total_rent_s'):>9.1f}{m('p50_task_latency_s'):>8.0f}"
                          f"{m('p95_task_latency_s'):>9.0f}{ratio:>8.2f}{ctrl:>12.0f}")
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
