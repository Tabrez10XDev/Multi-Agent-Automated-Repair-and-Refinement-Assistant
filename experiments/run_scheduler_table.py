"""
Scheduler comparison on a replayed trace, with an oracle upper bound.

    python -u -m experiments.run_scheduler_table --quick          # ~1 min
    python -u -m experiments.run_scheduler_table                  # ~3 min

WHAT THIS ANSWERS
    Does workflow-aware scheduling (least-slack-first over the DAG) beat
    arrival order, and how much of the remaining gap is the output-length
    estimator rather than the scheduling rule?

HOW IT IS MADE COMPARABLE
    1. Retention is held FIXED (default lru) across every row. Otherwise a
       scheduling difference and a retention difference are indistinguishable.
    2. Arrival rate is calibrated to a target GPU utilisation, so every row
       sees the same offered load rather than the same arrival rate.
    3. `slack_factor` is calibrated so FCFS misses 20-30% of deadlines. It is a
       declared PLACEHOLDER (engine.EngineConfig), and with it set arbitrarily
       the miss-rate column is either all zeros or all ones and says nothing.
    4. The ORACLE row uses OracleEstimator: the same scheduler with perfect
       output lengths. It is an upper bound, not a deployable policy, and the
       gap between it and the online row is the estimator's cost.

Averaged over --seeds seeds (default 20). Writes results/scheduler_table.json,
with a run manifest under "manifest". Both calibrations are checked against
the run's config before any engine run (FINDINGS 23 was a rate calibrated at
one batch size and measured at another).
"""

import argparse
import json
import os
import statistics as st
import time

from experiments import run_retention_grid
from experiments.manifest import (Calibration, Manifest, bound_args,
                                  check_consistent, default_of)
from experiments.run_retention_grid import build, calibrate_rate
from simulator.engine import Engine, EngineConfig
from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, HintBuilder)
from simulator.workload import CostModel, CpuStageSeconds, StageTokens

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["fcfs", "sjf", "edf", "lsf", "static-priority"]


def run_one(sched, estimator_name, scale, seed, kv_mb, rate, slack, retention,
            cost, tok, cpu, trace_set, batch=16, chunk=512, sigma=0.5):
    tasks = build(scale, seed, rate, cost, tok, cpu, trace_set=trace_set)
    est = ESTIMATORS[estimator_name](cost, tok, cpu)
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=kv_mb, max_batch=batch,
                              prefill_chunk_tokens=chunk, slack_factor=slack),
                 retention=RETENTIONS[retention](),
                 scheduler=SCHEDULERS[sched](),
                 estimator=est,
                 hints=HintBuilder(sigma=sigma, seed=seed))
    return eng.run()


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
                    target=(0.20, 0.30)):
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
                run_one("fcfs", "online", scale, sd, kv_mb, rate, mid,
                        retention, cost, tok, cpu, trace_set, batch=batch
                        )["deadline_miss_rate"] for sd in seeds)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-set", default="holdout_qwen")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--scale", type=int, default=1)
    ap.add_argument("--util", type=float, default=0.8)
    ap.add_argument("--kv-mb", type=float, default=1024.0)
    ap.add_argument("--retention", default="lru",
                    help="held fixed across rows so scheduling is isolated")
    ap.add_argument("--batch", type=int, default=8,
                    help="max concurrent requests. 8 is deliberate: at the "
                         "engine default of 16 this trace gives the scheduler a "
                         "choice on under 9%% of admission decisions, so no "
                         "scheduling rule can distinguish itself and the oracle "
                         "row comes out identical to the online one. 8 yields "
                         "~42%% choice and lands FCFS in the 20-30%% miss band.")
    ap.add_argument("--quick", action="store_true", help="3 seeds")
    ap.add_argument("--out", default=None,
                    help="JSON path (default: results/scheduler_table_<trace>.json)")
    args = ap.parse_args()

    seeds = list(range(1, (3 if args.quick else args.seeds) + 1))
    cost, tok, cpu = CostModel(), StageTokens(), CpuStageSeconds()
    t_start = time.time()

    print(f"trace={args.trace_set}  scale={args.scale}x  util={args.util:.0%}  "
          f"kv={args.kv_mb:.0f}MB  batch={args.batch}  "
          f"retention={args.retention} (fixed)  seeds={len(seeds)}", flush=True)

    # Neither driver sets cache_mode, so every run here uses the engine default.
    cache_mode = EngineConfig().cache_mode
    chunk = default_of(run_one, "chunk")
    manifest = Manifest.start("scheduler_table", params={
        "trace_set": args.trace_set, "scale": args.scale, "util": args.util,
        "kv_mb": args.kv_mb, "max_batch": args.batch, "cache_mode": cache_mode,
        "schedulers": ORDER, "oracle_row": "lsf",
        "retention": args.retention, "estimators": ["online", "oracle"],
        "seeds": seeds, "prefill_chunk_tokens": chunk,
        "sigma": default_of(run_one, "sigma"),
        "slack_factor": "calibrated",
        "cost_model": "CostModel() defaults; StageTokens(), CpuStageSeconds()"})
    # What the measurement loop below actually runs at.
    run_config = {"trace_set": args.trace_set, "scale": args.scale,
                  "util": args.util, "kv_mb": args.kv_mb,
                  "max_batch": args.batch, "cache_mode": cache_mode,
                  "retention": args.retention}

    # Calibration configs are read from the bound call, defaults included,
    # not from what this driver believes it passes. trace_set and batch used
    # to be omitted here, so the rate was always calibrated at batch 8 on
    # holdout_qwen whatever the table ran at (DECISIONS 25).
    rate_call = (args.util, args.scale, seeds[0], args.kv_mb, cost, tok, cpu)
    rate_kw = {"trace_set": args.trace_set, "batch": args.batch}
    rb = bound_args(calibrate_rate, *rate_call, **rate_kw)
    rate_under = {"trace_set": rb["trace_set"], "scale": rb["scale"],
                  "util": rb["target_util"], "kv_mb": rb["kv_mb"],
                  "max_batch": rb["batch"], "cache_mode": cache_mode,
                  "seed": rb["seed"], "scheduler": "fcfs",
                  "retention": "keep-all",
                  "prefill_chunk_tokens":
                      default_of(run_retention_grid.run_one, "chunk")}
    # Before spending any engine time: a mismatch here is already knowable.
    check_consistent("arrival_rate", rate_under, run_config, ("util",))

    print("calibrating arrival rate...", flush=True)
    rate, busy = calibrate_rate(*rate_call, **rate_kw, return_achieved=True)
    print(f"  rate={rate:.4f}/s", flush=True)
    manifest.record(Calibration(
        "arrival_rate", rate, rate_under,
        target=f"GPU busy fraction {args.util} under FCFS/keep-all",
        achieved=busy, also_check=("util",)))
    run_config["arrival_rate"] = rate

    slack_call = (args.scale, seeds[:3], args.kv_mb, rate, args.retention,
                  cost, tok, cpu, args.trace_set)
    sb = bound_args(calibrate_slack, *slack_call, batch=args.batch)
    slack_under = {"trace_set": sb["trace_set"], "scale": sb["scale"],
                   "kv_mb": sb["kv_mb"], "max_batch": sb["batch"],
                   "cache_mode": cache_mode, "arrival_rate": sb["rate"],
                   "retention": sb["retention"], "seeds": sb["seeds"],
                   "scheduler": "fcfs", "estimator": "online",
                   "prefill_chunk_tokens": chunk}
    slack_checks = ("arrival_rate", "retention")
    check_consistent("slack_factor", slack_under, run_config, slack_checks)

    print("calibrating slack_factor for 20-30% FCFS deadline misses...", flush=True)
    slack, achieved = calibrate_slack(*slack_call, batch=args.batch)
    if achieved is None:
        print("  FAILED to measure a miss rate; aborting rather than guessing.")
        return 1
    in_band = 0.20 <= achieved <= 0.30
    print(f"  slack_factor={slack:.3f}  FCFS miss rate={achieved:.1%}"
          f"{'' if in_band else '   *** OUTSIDE the 20-30% target band ***'}",
          flush=True)
    manifest.record(Calibration(
        "slack_factor", slack, slack_under,
        target="FCFS deadline-miss rate in [0.20, 0.30]",
        achieved=achieved, also_check=slack_checks))

    manifest.assert_consistent(run_config)

    rows, results = [], []
    for sched in ORDER:
        rows.append((sched, "online", sched))
    rows.append(("lsf", "oracle", "lsf [ORACLE]"))

    print()
    print(f"  {'scheduler':<16}{'p50':>8}{'p95':>9}{'miss':>8}{'slack@done':>12}"
          f"{'MAPE':>8}{'choice':>8}", flush=True)
    for sched, est, label in rows:
        try:
            reps = [run_one(sched, est, args.scale, sd, args.kv_mb, rate, slack,
                            args.retention, cost, tok, cpu, args.trace_set,
                            batch=args.batch)
                    for sd in seeds]
        except RuntimeError as e:
            print(f"  {label:<16} ERROR {str(e)[:48]}", flush=True)
            continue
        m = lambda k: st.fmean(r[k] for r in reps)
        row = {"scheduler": sched, "estimator": est, "label": label,
               "p50_task_latency_s": m("p50_task_latency_s"),
               "p95_task_latency_s": m("p95_task_latency_s"),
               "deadline_miss_rate": m("deadline_miss_rate"),
               "mean_slack_at_completion_s": m("mean_slack_at_completion_s"),
               "estimator_mape_pct": m("estimator_mape_pct"),
               "frac_decisions_with_choice": m("frac_decisions_with_choice"),
               "seeds": len(seeds),
               # Per seed, in seed order, so rows can be compared paired.
               "per_seed_p95_s": [r["p95_task_latency_s"] for r in reps]}
        base = next((r for r in results if r["label"] == "fcfs"), None)
        if base is not None and label != "fcfs":
            row["p95_vs_fcfs"] = paired_ci(row["per_seed_p95_s"],
                                           base["per_seed_p95_s"])
        results.append(row)
        print(f"  {label:<16}{row['p50_task_latency_s']:>8.1f}"
              f"{row['p95_task_latency_s']:>9.1f}"
              f"{row['deadline_miss_rate']:>7.1%}"
              f"{row['mean_slack_at_completion_s']:>12.1f}"
              f"{row['estimator_mape_pct']:>7.1f}%"
              f"{row['frac_decisions_with_choice']:>7.1%}", flush=True)

    print("\n  paired 95% CI on the p95 difference against fcfs (same seeds):")
    for row in results:
        ci = row.get("p95_vs_fcfs")
        if ci:
            print(f"  {row['label']:<16}{ci['mean_diff_s']:>+8.2f}s  "
                  f"[{ci['lo_s']:+.2f}, {ci['hi_s']:+.2f}]  "
                  f"{'significant' if ci['significant'] else 'not significant'}",
                  flush=True)

    out = {
        "trace_set": args.trace_set, "scale": args.scale, "util": args.util,
        "kv_mb": args.kv_mb, "max_batch": args.batch,
        "retention": args.retention, "seeds": len(seeds),
        "arrival_rate": rate,
        "slack_factor": slack, "slack_factor_provenance":
            f"CALIBRATED so FCFS misses 20-30% of deadlines; achieved "
            f"{achieved:.1%}" + ("" if in_band else " (OUTSIDE target band)"),
        "fcfs_miss_rate_achieved": achieved,
        "slack_in_target_band": in_band,
        "rows": results,
        "elapsed_s": round(time.time() - t_start, 1),
        "manifest": manifest.to_dict(),
    }
    path = args.out or os.path.join(_REPO, "results",
                                    f"scheduler_table_{args.trace_set}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n  elapsed {out['elapsed_s']:.0f}s -> {path}")
    if not in_band:
        print("  NOTE: slack_factor did not reach the target band. The miss-rate")
        print("        column is still comparable across rows, but the absolute")
        print("        level is not the one that was asked for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
