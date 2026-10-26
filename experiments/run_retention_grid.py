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

RUN MANIFEST
    Each cell's arrival rate and hindsight denominator are recorded with the
    configuration they were measured under and checked against the cell's run
    config before any policy runs; the manifest is written under "manifest" in
    the results JSON. FINDINGS 23 was a rate calibrated at one batch size and
    measured at another.
"""

from __future__ import annotations

import argparse
import json
import time
import os
import statistics as st

# Shared helpers live in experiments/common.py; re-exported so existing imports
# (tests, scripts) keep working.
from experiments.common import (build, calibrate_rate, make_estimator,  # noqa: F401  re-exported for old imports
                                run_retention_one, scale_context)
from experiments.manifest import (Calibration, Manifest, bound_args,
                                  check_consistent, default_of)
from simulator.engine import EngineConfig
from simulator.trace_loader import TRACE_SET_EXCLUSIONS, TRACE_SETS
from simulator.workload import (DEFAULT_PROFILE, PROFILES, CostModel, CpuStageSeconds,
                                StageTokens, DEFAULT_BLOCK_TOKENS)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POLICIES = ["keep-all", "lru", "evict-now", "ski-rental", "ski-rental-pred",
            "offline-ref"]


run_one = run_retention_one     # historical name; run_scheduler_table has its own


def hindsight_denominator(scale, seed, kv_mb, arrival_rate, cost, tok, cpu,
                          trace_set="holdout_qwen", batch=8):
    """min(rebuild, rent-while-held) per key, measured under keep-all."""
    eng, _ = run_one("keep-all", "fcfs", scale, seed, kv_mb, arrival_rate,
                     cost, tok, cpu, batch=batch, trace_set=trace_set)
    return sum(min(eng._key_recompute.get(k, 0.0), rent)
               for k, rent in eng._key_rent.items())

# -- headroom mode (OPEN_ISSUES 27) ------------------------------------------
#
# The fixed kv_mb rule above never put retention under a choice it could get
# wrong. Headroom mode sizes the budget from what the workload actually wants to
# keep: at each context scale, the reusable working set (WS) is the peak bytes
# of blocks that will be needed again. A cell's budget is then
#
#     peak active bytes  +  headroom x WS
#
# where active bytes are those no policy can free (running requests' private
# KV plus the blocks they hold). headroom 1.0 fits everything worth keeping at
# its peak; 0.25 forces choices.
#
# Load: the arrival rate is calibrated to the target util under FCFS/keep-all
# at the full-headroom budget (active + WS), so "util 0.8" means 80% busy when
# memory is not the constraint. Every headroom at one scale runs at that rate:
# what varies is memory, not traffic. (The fixed rule's budget cannot define
# the load: at 8x it is below active + WS, keep-all thrashes, and calibration
# landed at 0.98 busy instead of 0.8. It is kept only as pass 1's start.)
#
# WS is measured under evict-now, where the engine records each time a block is
# needed (built, rebuilt or hit). A block is live from one need to the next.
# Not keep-all with unbounded memory: at 16x that holds ~10^5 blocks and the
# per-iteration rent scan makes one run take many minutes. WS uses future need
# times, which is fine: it is the experimenter sizing the cache, not a policy.

HEADROOM_POLICIES = ["keep-all", "lru", "evict-now", "ski-rental",
                     "ski-rental-pred", "offline-ref"]
HEADROOM_METRICS = ("p50_task_latency_s", "p95_task_latency_s",
                    "mean_task_latency_s", "recompute_seconds", "total_rent_s",
                    "forced_evictions", "rent_active_tick_frac",
                    "retention_controlled_decisions", "prefix_hit_rate",
                    "gpu_busy_frac", "peak_kv_util",
                    # latency / the task's own lone-machine time (engine
                    # _isolated_service_s); report-time, never seen by a policy
                    "p95_slowdown", "mean_slowdown")


def reusable_working_set(scale, seed, arrival_rate, kv_mb, cost, tok, cpu,
                         trace_set="holdout_qwen", batch=4):
    """(WS bytes, peak active bytes), measured under evict-now.

    A block is live from each time it is needed until the next; WS is the peak
    total bytes of live blocks. A block's last need ends its liveness, so
    blocks nothing will match again (tails, one-off prompts) never count."""
    eng, _ = run_one("evict-now", "fcfs", scale, seed, kv_mb, arrival_rate,
                     cost, tok, cpu, batch=batch, trace_set=trace_set,
                     record_key_uses=True)
    events = []
    for k, uses in eng._key_uses.items():
        nb = eng._key_nbytes[k]
        for a, b in zip(uses, uses[1:]):
            if b > a:
                events.append((a, nb))
                events.append((b, -nb))
    events.sort(key=lambda e: (e[0], e[1]))   # releases before acquires at a tie
    cur = peak = 0
    for _, d in events:
        cur += d
        peak = max(peak, cur)
    return peak, eng.m["peak_active_bytes"]


def _headroom_job(args):
    # The profile is optional so a run started before --profile existed keeps
    # working if its pool spawns a worker from this file.
    (pol, scale, seed, kv, rate, batch, trace_set, *rest) = args
    cost = CostModel.from_profile(rest[0] if rest else DEFAULT_PROFILE)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    try:
        _, rep = run_one(pol, "fcfs", scale, seed, kv, rate, cost, tok, cpu,
                         batch=batch, trace_set=trace_set)
    except RuntimeError as e:
        return {"error": str(e)[:200]}
    return {k: rep[k] for k in HEADROOM_METRICS}


# Grid mode runs its seeds in a pool too (DECISIONS 37, 2026-10-26): each run is
# deterministic in its arguments, so the order runs finish in cannot change a
# number. Workers return only these metrics, never the engine.
GRID_METRICS = ("recompute_seconds", "total_rent_s", "p50_task_latency_s",
                "p95_task_latency_s", "p95_slowdown", "mean_slowdown",
                "retention_controlled_decisions", "cold_misses",
                "prefix_hit_rate", "deadline_miss_rate", "rent_active_tick_frac",
                "makespan_s", "gpu_busy_frac")


def _grid_job(args):
    (pol, sched, scale, seed, kv, rate, batch, trace_set, profile) = args
    cost = CostModel.from_profile(profile)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    try:
        _, rep = run_one(pol, sched, scale, seed, kv, rate, cost, tok, cpu,
                         trace_set=trace_set, batch=batch)
    except RuntimeError as e:
        return {"error": str(e)[:200]}
    return {k: rep[k] for k in GRID_METRICS}


def _hindsight_job(args):
    (scale, seed, kv, rate, batch, trace_set, profile) = args
    cost = CostModel.from_profile(profile)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    return hindsight_denominator(scale, seed, kv, rate, cost, tok, cpu,
                                 trace_set=trace_set, batch=batch)


def headroom_main(args):
    from concurrent.futures import ProcessPoolExecutor
    from experiments.run_scheduler_table import paired_ci

    cost = CostModel.from_profile(args.profile)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    scales =[int(x) for x in (args.scales or "1,8,16").split(",")]
    heads = [float(x) for x in args.headrooms.split(",")]
    util = args.util
    seeds = list(range(1, args.seeds + 1))
    cache_mode = EngineConfig().cache_mode
    chunk = default_of(run_one, "chunk")
    manifest = Manifest.start("retention_headroom", params={
        "trace_set": args.trace_set, "scales": scales, "headrooms": heads,
        "excluded_tasks": TRACE_SET_EXCLUSIONS.get(args.trace_set, {}),
        "util": util, "max_batch": args.batch, "cache_mode": cache_mode,
        "retentions": HEADROOM_POLICIES, "scheduler": "fcfs", "seeds": seeds,
        "prefill_chunk_tokens": chunk, "hardware_profile": cost.profile,
        "cost_provenance": cost.provenance(),
        "kv_budget_rule": "peak_active_bytes + headroom x reusable working set "
                          "(both measured under evict-now at the calibrated rate)",
        "load_reference": "arrival rate calibrated to util under keep-all at the "
                          "full-headroom budget (active + WS, from a pass-1 rate at "
                          "kv_mb_at_1x * max(1, scale // 2)); every headroom at a "
                          "scale runs at that rate",
        "kv_mb_at_1x": args.kv_mb})
    t0 = time.time()
    print(f"headroom grid: trace={args.trace_set} scales={scales} "
          f"headrooms={heads} util={util} batch={args.batch} "
          f"profile={cost.profile} coeff={cost.batch_decode_coeff} "
          f"seeds={len(seeds)}", flush=True)
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for scale in scales:
            mb = 1024 * 1024
            checks = ("util", "hardware_profile", "batch_decode_coeff")
            rate_kw = {"trace_set": args.trace_set, "batch": args.batch}
            ws_seeds = seeds[:3]

            def measure_ws(r, kv_mb):
                m = [reusable_working_set(scale, sd, r, kv_mb, cost, tok, cpu,
                                          trace_set=args.trace_set, batch=args.batch)
                     for sd in ws_seeds]
                return max(x[0] for x in m), max(x[1] for x in m)

            # Pass 1, a starting point only: a short bisection at the fixed
            # rule's budget. At 8x and above that budget is below active + WS,
            # keep-all thrashes, and busy jumps past the target (it hit 0.98 at
            # 8x on the first attempt), so it cannot define the load itself.
            kv_rule = args.kv_mb * max(1, scale // 2)
            # The bracket scales with context: work per task grows with it, so
            # at 8x the util-0.8 rate (about 0.0033/s on holdout_qwen) is below
            # the 1x floor of 0.005/s, and bisection pinned to the floor at
            # busy 0.96-0.98. Same 1x bracket at scale 1.
            rate_kw.update(lo=0.005 / scale, hi=8.0 / scale)
            rate0 = calibrate_rate(util, scale, 1, kv_rule, cost, tok, cpu,
                                   iters=6, **rate_kw)
            ws0, active0 = measure_ws(rate0, kv_rule)
            # Pass 2, the load: util at the full-headroom budget, where memory
            # is not the constraint. Every headroom then runs at this rate.
            kv_ref = (active0 + ws0) / mb
            ref = {"trace_set": args.trace_set, "scale": scale, "util": util,
                   "kv_mb": kv_ref, "max_batch": args.batch,
                   "cache_mode": cache_mode, "hardware_profile": cost.profile,
                   "batch_decode_coeff": cost.batch_decode_coeff}
            rate_call = (util, scale, 1, kv_ref, cost, tok, cpu)
            rb = bound_args(calibrate_rate, *rate_call, **rate_kw)
            under = {"trace_set": rb["trace_set"], "scale": rb["scale"],
                     "util": rb["target_util"], "kv_mb": rb["kv_mb"],
                     "max_batch": rb["batch"], "cache_mode": cache_mode,
                     "hardware_profile": cost.profile,
                     "batch_decode_coeff": cost.batch_decode_coeff,
                     "seed": rb["seed"], "scheduler": "fcfs",
                     "retention": "keep-all", "prefill_chunk_tokens": chunk,
                     "pass1_rate": rate0, "pass1_kv_mb": kv_rule}
            check_consistent("arrival_rate", under, ref, checks)
            rate, busy = calibrate_rate(*rate_call, **rate_kw, iters=14,
                                        return_achieved=True)
            rate_cal = manifest.record(Calibration(
                f"arrival_rate[{scale}x]", rate, under,
                target=f"GPU busy fraction {util} under FCFS/keep-all at the "
                       f"full-headroom budget (active + WS from pass 1)",
                achieved=busy, also_check=checks))
            if busy is None or abs(busy - util) > 0.05:
                print(f"  WARNING: {scale}x calibration achieved busy {busy} "
                      f"against target {util}", flush=True)

            ws, active = measure_ws(rate, kv_ref)
            ws_under = dict(under, arrival_rate=rate, seeds=ws_seeds)
            ws_cal = manifest.record(Calibration(
                f"reusable_working_set_bytes[{scale}x]", ws, ws_under,
                target="max over seeds of peak live reusable bytes, measured "
                       "under evict-now; achieved = peak active bytes", achieved=active,
                also_check=checks + ("arrival_rate",)))
            print(f"\n=== {scale}x  rate {rate:.4f}/s (busy {busy:.2f}; pass 1 "
                  f"{rate0:.4f}/s at {kv_rule:.0f} MB)  WS {ws / mb:.0f} MB  "
                  f"active peak {active / mb:.0f} MB  [{time.time() - t0:.0f}s] ===",
                  flush=True)

            for h in heads:
                kv = (active + h * ws) / mb
                manifest.assert_consistent(dict(ref, arrival_rate=rate),
                                           calibrations=[rate_cal, ws_cal])
                jobs = [(pol, scale, sd, kv, rate, args.batch, args.trace_set,
                         args.profile)
                        for pol in HEADROOM_POLICIES for sd in seeds]
                res = list(pool.map(_headroom_job, jobs))
                by = {pol: res[i * len(seeds):(i + 1) * len(seeds)]
                      for i, pol in enumerate(HEADROOM_POLICIES)}
                cell = {"scale": scale, "headroom": h, "kv_budget_mb": kv,
                        "ws_mb": ws / mb, "active_peak_mb": active / mb,
                        "arrival_rate": rate, "util": util,
                        "max_batch": args.batch, "policies": {}}
                base = by["keep-all"]
                base_ok = all("error" not in r for r in base)
                print(f"  headroom {h:.2f}  KV {kv:7.0f} MB   "
                      f"{'retention':<16}{'p95':>8}{'Δp95 vs keep-all':>26}"
                      f"{'recomp':>9}{'Δrecomp vs keep-all':>28}{'rent-act':>9}",
                      flush=True)
                for pol, rs in by.items():
                    ok = [r for r in rs if "error" not in r]
                    if not ok:
                        cell["policies"][pol] = {"errors": len(rs)}
                        print(f"    {pol:<16} ERROR {rs[0].get('error', '')[:60]}")
                        continue
                    row = {k: st.fmean(r[k] for r in ok) for k in HEADROOM_METRICS}
                    row["errors"] = len(rs) - len(ok)
                    row["per_seed"] = {k: [r[k] for r in ok] for k in
                                       ("p95_task_latency_s", "p50_task_latency_s",
                                        "recompute_seconds", "p95_slowdown")}
                    fmt = lambda c: (f"{c['mean_diff_s']:+7.1f} [{c['lo_s']:+.1f}, "
                                     f"{c['hi_s']:+.1f}]{'*' if c['significant'] else ' '}")
                    d95 = drc = ""
                    if pol != "keep-all" and base_ok and len(ok) == len(seeds):
                        c95 = paired_ci([r["p95_task_latency_s"] for r in ok],
                                        [r["p95_task_latency_s"] for r in base])
                        crc = paired_ci([r["recompute_seconds"] for r in ok],
                                        [r["recompute_seconds"] for r in base])
                        row["p95_vs_keep_all"], row["recompute_vs_keep_all"] = c95, crc
                        d95, drc = fmt(c95), fmt(crc)
                    cell["policies"][pol] = row
                    print(f"{'':30}{pol:<16}{row['p95_task_latency_s']:8.1f}{d95:>26}"
                          f"{row['recompute_seconds']:9.1f}{drc:>28}"
                          f"{row['rent_active_tick_frac']:8.1%}", flush=True)
                cells.append(cell)
                print(f"  [cell done at {time.time() - t0:.0f}s]", flush=True)

    out = {"mode": "headroom", "cells": cells, "elapsed_s": round(time.time() - t0, 1),
           "manifest": manifest.to_dict()}
    path = args.out or os.path.join(_REPO, "results",
                                    f"retention_headroom_{args.trace_set}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nelapsed {out['elapsed_s']:.0f}s -> {path}")
    return 0


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
    ap.add_argument("--headrooms", default=None,
                    help="headroom mode: comma-separated fractions of the reusable "
                         "working set added to the active peak, e.g. 0.25,0.5,1.0. "
                         "Runs FCFS only, one util, paired CIs vs keep-all")
    ap.add_argument("--util", type=float, default=0.8, help="headroom mode only")
    ap.add_argument("--workers", type=int, default=4,
                    help="processes running seeds in parallel (both modes)")
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                    help="hardware profile the cost model is built from "
                         "(simulator/workload.py PROFILES)")
    ap.add_argument("--rate-bracket", type=float, nargs=2, default=None,
                    metavar=("LO", "HI"),
                    help="grid mode: arrival-rate bisection bracket in tasks/s "
                         "(default calibrate_rate's 0.005 8.0, sized for MAARA). "
                         "TraceLab saturates near 0.01/s, so the default's first "
                         "probes run ~400x overloaded and 12 halvings leave a "
                         "0.002/s step, about 20%% of capacity (DECISIONS 37)")
    args = ap.parse_args()
    if args.headrooms:
        return headroom_main(args)

    cost = CostModel.from_profile(args.profile)
    tok = StageTokens.from_measured()
    cpu = CpuStageSeconds.from_measured()

    utils = [0.6] if args.quick else [0.6, 0.8]
    scales = ([int(x) for x in args.scales.split(",")] if args.scales
              else ([1] if args.quick else [1, 8, 16]))
    # --seeds used to be ignored under --quick, which silently capped a run the
    # caller had asked to be larger.
    n_seeds = 2 if args.quick else args.seeds

    from concurrent.futures import ProcessPoolExecutor
    pool = ProcessPoolExecutor(max_workers=args.workers)

    t0 = time.time()
    cells, total_runs = [], 0
    print(f"trace={args.trace_set}  scales={scales}  utils={utils}  "
          f"batch={args.batch}  kv={args.kv_mb:.0f}MB@1x  "
          f"seeds={n_seeds} ({args.seeds_16x} at 16x)  profile={cost.profile}",
          flush=True)

    # Neither driver sets cache_mode, so every run here uses the engine default.
    cache_mode = EngineConfig().cache_mode
    chunk = default_of(run_one, "chunk")
    manifest = Manifest.start("retention_grid", params={
        "trace_set": args.trace_set, "scales": scales, "utils": utils,
        "kv_mb_at_1x": args.kv_mb, "kv_mb_rule": "kv_mb_at_1x * max(1, scale // 2)",
        "max_batch": args.batch, "cache_mode": cache_mode,
        "retentions": POLICIES, "schedulers": ["fcfs", "lsf"],
        "workers": args.workers,
        "estimator": "online", "seeds": n_seeds, "seeds_16x": args.seeds_16x,
        "prefill_chunk_tokens": chunk, "sigma": default_of(run_one, "sigma"),
        "lam": default_of(run_one, "lam"),
        "hardware_profile": cost.profile, "cost_provenance": cost.provenance(),
        "cost_model": f"CostModel.from_profile({cost.profile!r}); "
                      "StageTokens/CpuStageSeconds .from_measured()"})

    for scale in scales:
        # Bigger contexts need a bigger budget simply to admit one request.
        kv = args.kv_mb * max(1, scale // 2)
        seeds = list(range(1, (args.seeds_16x if scale >= 16 else n_seeds) + 1))
        for util in utils:
            # What this cell's measurement loop runs at.
            run_config = {"trace_set": args.trace_set, "scale": scale,
                          "util": util, "kv_mb": kv, "max_batch": args.batch,
                          "cache_mode": cache_mode}
            # Read from the bound call, defaults included, so an unpassed
            # default is recorded as what it is (the FINDINGS 23 failure).
            rate_call = (util, scale, seeds[0], kv, cost, tok, cpu)
            rate_kw = {"trace_set": args.trace_set, "batch": args.batch}
            if args.rate_bracket:
                rate_kw.update(lo=args.rate_bracket[0], hi=args.rate_bracket[1])
            rb = bound_args(calibrate_rate, *rate_call, **rate_kw)
            rate_under = {"trace_set": rb["trace_set"], "scale": rb["scale"],
                          "util": rb["target_util"], "kv_mb": rb["kv_mb"],
                          "max_batch": rb["batch"], "cache_mode": cache_mode,
                          "seed": rb["seed"], "scheduler": "fcfs",
                          "retention": "keep-all",
                          "prefill_chunk_tokens": chunk,
                          "rate_bracket": [rb["lo"], rb["hi"]]}
            check_consistent("arrival_rate", rate_under, run_config, ("util",))

            el = time.time() - t0
            print(f"[{el:6.0f}s] [{scale}x util {util:.0%}] calibrating "
                  f"arrival rate...", flush=True)
            rate, busy = calibrate_rate(*rate_call, **rate_kw,
                                        return_achieved=True)
            rate_cal = manifest.record(Calibration(
                "arrival_rate", rate, rate_under,
                target=f"GPU busy fraction {util} under FCFS/keep-all",
                achieved=busy, also_check=("util",)))
            if busy is None or abs(busy - util) > 0.05:
                print(f"  WARNING: calibration achieved busy {busy} against "
                      f"target {util}; check --rate-bracket", flush=True)
            run_config["arrival_rate"] = rate
            print(f"[{time.time()-t0:6.0f}s] [{scale}x util {util:.0%}] "
                  f"rate={rate:.4f}/s; measuring hindsight...", flush=True)

            # The denominator every policy in this cell is scored against is a
            # measured value too, valid only at the rate and config it ran at.
            hb = bound_args(hindsight_denominator, scale, seeds[0], kv, rate,
                            cost, tok, cpu, trace_set=args.trace_set,
                            batch=args.batch)
            hind_under = {"trace_set": hb["trace_set"], "scale": hb["scale"],
                          "kv_mb": hb["kv_mb"], "max_batch": hb["batch"],
                          "cache_mode": cache_mode,
                          "arrival_rate": hb["arrival_rate"], "seeds": seeds,
                          "scheduler": "fcfs", "retention": "keep-all",
                          "prefill_chunk_tokens": chunk}
            check_consistent("hindsight_denominator", hind_under, run_config,
                             ("arrival_rate",))
            hind = st.fmean(pool.map(_hindsight_job, [
                (scale, s, kv, rate, args.batch, args.trace_set, args.profile)
                for s in seeds])) or float("nan")
            hind_cal = manifest.record(Calibration(
                "hindsight_denominator", hind, hind_under,
                target="mean over seeds of min(rebuild, rent) per block key "
                       "under keep-all",
                achieved=None, also_check=("arrival_rate",)))
            manifest.assert_consistent(run_config,
                                       calibrations=[rate_cal, hind_cal])
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
            # Every row of the cell is queued at once; rows print in order.
            futs = {(pol, sched): [pool.submit(_grid_job, (
                        pol, sched, scale, s, kv, rate, args.batch,
                        args.trace_set, args.profile)) for s in seeds]
                    for pol in POLICIES for sched in ("fcfs", "lsf")}
            for pol in POLICIES:
                for sched in ("fcfs", "lsf"):
                    rs = [f.result() for f in futs[(pol, sched)]]
                    bad = [r["error"] for r in rs if "error" in r]
                    if bad:
                        print(f"  {pol:<17}{sched:<6} ERROR {bad[0][:50]}",
                              flush=True)
                        continue
                    total_runs += len(rs)
                    m = lambda k: st.fmean([r[k] for r in rs])  # noqa: E731
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
                        "p95_slowdown": m("p95_slowdown"),
                        "mean_slowdown": m("mean_slowdown"),
                        "realized_ratio_vs_hindsight": ratio,
                        "retention_controlled_decisions": ctrl,
                        "cold_misses": m("cold_misses"),
                        "prefix_hit_rate": m("prefix_hit_rate"),
                        "deadline_miss_rate": m("deadline_miss_rate"),
                        "makespan_s": m("makespan_s"),
                        "gpu_busy_frac": m("gpu_busy_frac"),
                        # Seed order 1..n, for paired CIs (DECISIONS 24).
                        "per_seed": {k: [r[k] for r in rs] for k in GRID_METRICS},
                    })
                    print(f"  {pol:<17}{sched:<6}"
                          f"{m('p50_task_latency_s'):>8.1f}"
                          f"{m('p95_task_latency_s'):>9.1f}"
                          f"{m('recompute_seconds'):>12.1f}"
                          f"{m('total_rent_s'):>9.1f}{ratio:>8.2f}{ctrl:>7.0f}"
                          f"{m('rent_active_tick_frac'):>9.1%}", flush=True)
            print(f"  [cell done at {time.time()-t0:.0f}s]\n", flush=True)

    pool.shutdown()
    out = {"trace_set": args.trace_set, "scales": scales, "utils": utils,
           "max_batch": args.batch, "kv_mb_at_1x": args.kv_mb,
           "engine_runs": total_runs, "elapsed_s": round(time.time() - t0, 1),
           "cells": cells, "manifest": manifest.to_dict()}
    path = args.out or os.path.join(
        _REPO, "results", f"retention_grid_{args.trace_set}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"{total_runs} engine runs in {out['elapsed_s']:.0f}s -> {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
