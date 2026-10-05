"""
Retention x scheduler comparison on a real trace.

    ./venv/bin/python -m experiments.run_retention_grid            # full grid
    ./venv/bin/python -m experiments.run_retention_grid --quick    # one util, one scale, 2 seeds

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
import json
import time
import hashlib
import os
import statistics as st

from simulator.engine import Engine, EngineConfig
from simulator.policies import (RETENTIONS, SCHEDULERS, ExecutionEstimator,
                          FCFSScheduler, HintBuilder, KeepAllRetention,
                          SkiRentalPredictionsRetention)
from simulator.trace_loader import TRACE_SETS, load_trace
from simulator.workload import (CostModel, CpuStageSeconds, Kind, StageTokens,
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
    paths = [os.path.join(_REPO, p) for p in TRACE_SETS[trace_set]]
    tasks = load_trace(paths, arrival_rate=arrival_rate, seed=seed).tasks
    return scale_context(tasks, scale)


def run_one(policy, sched, scale, seed, kv_mb, arrival_rate, cost, tok, cpu,
            batch=8, lam=0.5, sigma=0.5, chunk=512, trace_set="holdout_qwen"):
    tasks = build(scale, seed, arrival_rate, cost, tok, cpu, trace_set=trace_set)
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
                   lo=0.005, hi=8.0, iters=12, trace_set="holdout_qwen",
                   batch=8):
    """Binary-search the arrival rate giving `target_util` GPU busy under
    FCFS/keep-all. Monotone in rate, so bisection is safe."""
    best = hi
    for _ in range(iters):
        mid = (lo + hi) / 2
        try:
            _, rep = run_one("keep-all", "fcfs", scale, seed, kv_mb, mid,
                             cost, tok, cpu, batch=batch, trace_set=trace_set)
            busy = rep["gpu_busy_frac"]
        except RuntimeError:
            busy = 1.0
        if busy < target_util:
            lo = mid
        else:
            hi = mid
            best = mid
    return best


def hindsight_denominator(scale, seed, kv_mb, arrival_rate, cost, tok, cpu,
                          trace_set="holdout_qwen", batch=8):
    """min(rebuild, rent-while-held) per key, measured under keep-all."""
    eng, _ = run_one("keep-all", "fcfs", scale, seed, kv_mb, arrival_rate,
                     cost, tok, cpu, batch=batch, trace_set=trace_set)
    return sum(min(eng._key_recompute.get(k, 0.0), rent)
               for k, rent in eng._key_rent.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--kv-mb", type=float, default=1024.0,
                    help="KV budget in MB at 1x context")
    ap.add_argument("--scales", default=None,
                    help="comma-separated context scales, e.g. 1 or 1,8,16")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--seeds-16x", type=int, default=3,
                    help="fewer seeds at 16x, where each run is slowest")
    ap.add_argument("--trace-set", default="holdout_qwen",
                    help="which TRACE_SETS entry to replay. Defaults to the "
                         "held-out trace; the fit traces contain cross-config "
                         "replay and should not be used for retention claims "
                         "(OPEN_ISSUES 20).")
    ap.add_argument("--batch", type=int, default=8,
                    help="max concurrent requests. 8 matches the scheduler "
                         "table's operating point; at 16 the scheduler has a "
                         "choice on under 9%% of decisions (FINDINGS 23).")
    ap.add_argument("--out", default=None, help="JSON path (default: results/)")
    args = ap.parse_args()

    cost = CostModel()
    tok = StageTokens.from_measured()
    cpu = CpuStageSeconds.from_measured()

    utils = [0.6] if args.quick else [0.6, 0.8]
    scales = ([int(x) for x in args.scales.split(",")] if args.scales
              else ([1] if args.quick else [1, 8, 16]))
    # --seeds used to be ignored under --quick, which silently capped a run the
    # caller had asked to be larger.
    n_seeds = 2 if args.quick else args.seeds

    t0 = time.time()
    cells, total_runs = [], 0
    print(f"trace={args.trace_set}  scales={scales}  utils={utils}  "
          f"batch={args.batch}  kv={args.kv_mb:.0f}MB@1x  "
          f"seeds={n_seeds} ({args.seeds_16x} at 16x)", flush=True)

    for scale in scales:
        # Bigger contexts need a bigger budget simply to admit one request.
        kv = args.kv_mb * max(1, scale // 2)
        seeds = list(range(1, (args.seeds_16x if scale >= 16 else n_seeds) + 1))
        for util in utils:
            el = time.time() - t0
            print(f"[{el:6.0f}s] [{scale}x util {util:.0%}] calibrating "
                  f"arrival rate...", flush=True)
            rate = calibrate_rate(util, scale, seeds[0], kv, cost, tok, cpu,
                                  trace_set=args.trace_set, batch=args.batch)
            print(f"[{time.time()-t0:6.0f}s] [{scale}x util {util:.0%}] "
                  f"rate={rate:.4f}/s; measuring hindsight...", flush=True)
            hind = st.fmean([hindsight_denominator(scale, s, kv, rate, cost, tok,
                                                   cpu, trace_set=args.trace_set,
                                                   batch=args.batch)
                             for s in seeds]) or float("nan")
            print(f"=== context {scale}x | target util {util:.0%} "
                  f"(arrival {rate:.3f}/s) | KV {kv:.0f}MB | B={args.batch} | "
                  f"{len(seeds)} seeds | chunked prefill 512 ===")
            print(f"  hindsight denominator {hind:.1f} recompute-s "
                  f"(min(rebuild, rent) per block, measured under keep-all)")
            # USER-FACING metrics first. Rent is kept as an internal column
            # because it is nearly invariant to the policy in this regime and
            # the composite "recompute + rent" ratio actually INVERTS the
            # ranking -- it rates evict-now above keep-all while evict-now is
            # worse on both latency and GPU work (FINDINGS 24).
            print(f"  {'retention':<17}{'sched':<6}{'p50':>8}{'p95':>9}"
                  f"{'gpu_recomp':>12}{'| rent':>9}{'ratio':>8}{'ctrl':>7}"
                  f"{'rent-act':>10}")
            for pol in POLICIES:
                for sched in ("fcfs", "lsf"):
                    rs = []
                    for s in seeds:
                        try:
                            eng, rep = run_one(pol, sched, scale, s, kv, rate,
                                               cost, tok, cpu,
                                               trace_set=args.trace_set,
                                               batch=args.batch)
                            rs.append((eng, rep))
                            total_runs += 1
                        except RuntimeError as e:
                            print(f"  {pol:<17}{sched:<6} ERROR {str(e)[:50]}",
                                  flush=True)
                            rs = []
                            break
                    if not rs:
                        continue
                    m = lambda k: st.fmean([r[k] for _, r in rs])  # noqa: E731
                    ctrl = m("retention_controlled_decisions")
                    cost_s = m("recompute_seconds") + m("total_rent_s")
                    ratio = cost_s / hind if hind and hind == hind else float("nan")
                    cells.append({
                        "trace_set": args.trace_set, "scale": scale,
                        "util": util, "kv_mb": kv, "max_batch": args.batch,
                        "arrival_rate": rate, "seeds": len(seeds),
                        "retention": pol, "scheduler": sched,
                        "hindsight_denominator_s": hind,
                        "recompute_seconds": m("recompute_seconds"),
                        "total_rent_s": m("total_rent_s"),
                        "p50_task_latency_s": m("p50_task_latency_s"),
                        "p95_task_latency_s": m("p95_task_latency_s"),
                        "realized_ratio_vs_hindsight": ratio,
                        "retention_controlled_decisions": ctrl,
                        "cold_misses": m("cold_misses"),
                        "prefix_hit_rate": m("prefix_hit_rate"),
                        "deadline_miss_rate": m("deadline_miss_rate"),
                    })
                    print(f"  {pol:<17}{sched:<6}"
                          f"{m('p50_task_latency_s'):>8.1f}"
                          f"{m('p95_task_latency_s'):>9.1f}"
                          f"{m('recompute_seconds'):>12.1f}"
                          f"{m('total_rent_s'):>9.1f}{ratio:>8.2f}{ctrl:>7.0f}"
                          f"{m('rent_active_tick_frac'):>9.1%}", flush=True)
            print(f"  [cell done at {time.time()-t0:.0f}s]\n", flush=True)

    out = {"trace_set": args.trace_set, "scales": scales, "utils": utils,
           "max_batch": args.batch, "kv_mb_at_1x": args.kv_mb,
           "engine_runs": total_runs, "elapsed_s": round(time.time() - t0, 1),
           "cells": cells}
    path = args.out or os.path.join(
        _REPO, "results", f"retention_grid_{args.trace_set}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{total_runs} engine runs in {out['elapsed_s']:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
