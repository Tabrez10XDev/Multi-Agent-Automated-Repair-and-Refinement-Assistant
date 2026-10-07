"""
Sensitivity of the headline conclusions to batch_decode_coeff.

    ./venv/bin/python -u -m experiments.sweep_decode_coeff [--seeds 10] [--coeffs 0.05 0.15]

WHY
    CostModel.batch_decode_coeff = 0.15 is the most load-bearing unmeasured
    constant (OPEN_ISSUES 9.3): it sets how much a batch of n decodes costs
    over one, and so how much capacity contention eats. If a conclusion flips
    across plausible values, it rests on the placeholder; if not, it does not.

WHAT IS CHECKED, PER COEFFICIENT
    The arrival rate is RECALIBRATED for each coefficient, since decode cost
    changes load; comparing coefficients at one fixed rate would compare
    different utilisations.

    retention point (FINDINGS 24): holdout_qwen 1x, util 0.8, batch 4, KV 512MB
        keep-all, ski-rental, evict-now under FCFS. Does recompute stay a tiny
        share of makespan, and does ski-rental still fail to beat keep-all on
        GPU recompute and p95?
    scheduler point (FINDINGS 27): holdout_qwen 1x, util 0.8, batch 8, KV 1024MB
        Does the scheduler still almost never have a choice, and is LSF still
        indistinguishable from FCFS on p95?

    Every comparison is paired over the same seeds (DECISIONS 24), and every
    calibration is recorded in a manifest and checked (DECISIONS 25).
"""

import argparse
import json
import os
import statistics as st
import time

from experiments.manifest import Calibration, Manifest, bound_args, check_consistent
from experiments.run_retention_grid import calibrate_rate, run_one
from experiments.run_scheduler_table import paired_ci
from simulator.engine import EngineConfig
from simulator.workload import CostModel, CpuStageSeconds, StageTokens

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COEFFS = [0.05, 0.15, 0.30, 0.50]
TRACE = "holdout_qwen"
UTIL = 0.8
RETENTION_POINT = {"batch": 4, "kv_mb": 512.0}
SCHED_POINT = {"batch": 8, "kv_mb": 1024.0}
RET_POLICIES = ["keep-all", "ski-rental", "evict-now"]
BREAKDOWN = ("p50_task_latency_s", "p95_task_latency_s", "recompute_seconds",
             "prefill_seconds", "decode_seconds", "idle_seconds", "makespan_s")


def calibrated(manifest, name, coeff, point, cost, tok, cpu, cache_mode):
    """Calibrate the arrival rate for one operating point, record and check it."""
    call = (UTIL, 1, 1, point["kv_mb"], cost, tok, cpu)
    kw = {"trace_set": TRACE, "batch": point["batch"]}
    b = bound_args(calibrate_rate, *call, **kw)
    under = {"trace_set": b["trace_set"], "scale": b["scale"],
             "util": b["target_util"], "kv_mb": b["kv_mb"],
             "max_batch": b["batch"], "cache_mode": cache_mode,
             "batch_decode_coeff": coeff, "seed": b["seed"],
             "scheduler": "fcfs", "retention": "keep-all"}
    run_config = {"trace_set": TRACE, "scale": 1, "util": UTIL,
                  "kv_mb": point["kv_mb"], "max_batch": point["batch"],
                  "cache_mode": cache_mode, "batch_decode_coeff": coeff}
    checks = ("util", "batch_decode_coeff")
    check_consistent(name, under, run_config, checks)
    rate, busy = calibrate_rate(*call, **kw, return_achieved=True)
    cal = manifest.record(Calibration(
        name, rate, under, target=f"GPU busy fraction {UTIL} under FCFS/keep-all",
        achieved=busy, also_check=checks))
    manifest.assert_consistent(run_config, calibrations=[cal])
    return rate, busy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--coeffs", type=float, nargs="+", default=COEFFS)
    ap.add_argument("--ret-policies", nargs="+", default=RET_POLICIES,
                    help="retention policies at the retention point; keep-all "
                         "and ski-rental are always included")
    ap.add_argument("--out", default=os.path.join(_REPO, "results",
                                                  "decode_coeff_sweep.json"))
    args = ap.parse_args()

    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    cache_mode = EngineConfig().cache_mode
    seeds = list(range(1, args.seeds + 1))
    ret_policies = list(dict.fromkeys(["keep-all", "ski-rental"] + args.ret_policies))
    manifest = Manifest.start("decode_coeff_sweep", params={
        "trace_set": TRACE, "util": UTIL, "coeffs": args.coeffs, "seeds": seeds,
        "retention_point": RETENTION_POINT, "sched_point": SCHED_POINT,
        "ret_policies": ret_policies,
        "cache_mode": cache_mode})
    t0 = time.time()
    out_rows = []

    for coeff in args.coeffs:
        cost = CostModel(batch_decode_coeff=coeff)
        print(f"\n=== batch_decode_coeff {coeff} ===", flush=True)

        # Retention point.
        p = RETENTION_POINT
        rate, busy = calibrated(manifest, f"arrival_rate[ret,c={coeff}]", coeff,
                                p, cost, tok, cpu, cache_mode)
        ret = {}
        for pol in ret_policies:
            ret[pol] = [run_one(pol, "fcfs", 1, s, p["kv_mb"], rate, cost, tok,
                                cpu, batch=p["batch"], trace_set=TRACE)[1]
                        for s in seeds]
        mean = lambda pol, k: st.fmean(r[k] for r in ret[pol])  # noqa: E731
        recomp_share = st.fmean(r["recompute_seconds"] / r["makespan_s"]
                                for r in ret["keep-all"])
        ski_vs_keep_p95 = paired_ci([r["p95_task_latency_s"] for r in ret["ski-rental"]],
                                    [r["p95_task_latency_s"] for r in ret["keep-all"]])
        ski_vs_keep_rc = paired_ci([r["recompute_seconds"] for r in ret["ski-rental"]],
                                   [r["recompute_seconds"] for r in ret["keep-all"]])
        print(f"  retention  rate {rate:.4f}/s (busy {busy:.2f})  "
              f"keep-all recompute {mean('keep-all', 'recompute_seconds'):.1f}s "
              f"= {recomp_share:.1%} of makespan  "
              f"rent-active {mean('ski-rental', 'rent_active_tick_frac'):.1%}",
              flush=True)
        for pol in ret:
            print(f"    {pol:<11} p95 {mean(pol, 'p95_task_latency_s'):7.1f}  "
                  f"recompute {mean(pol, 'recompute_seconds'):7.1f}s", flush=True)
        print(f"    ski-rental - keep-all: p95 {ski_vs_keep_p95['mean_diff_s']:+.2f}s "
              f"[{ski_vs_keep_p95['lo_s']:+.2f}, {ski_vs_keep_p95['hi_s']:+.2f}]  "
              f"recompute {ski_vs_keep_rc['mean_diff_s']:+.2f}s "
              f"[{ski_vs_keep_rc['lo_s']:+.2f}, {ski_vs_keep_rc['hi_s']:+.2f}]",
              flush=True)

        # Scheduler point.
        p = SCHED_POINT
        srate, sbusy = calibrated(manifest, f"arrival_rate[sched,c={coeff}]",
                                  coeff, p, cost, tok, cpu, cache_mode)
        sch = {s_name: [run_one("lru", s_name, 1, s, p["kv_mb"], srate, cost,
                                tok, cpu, batch=p["batch"], trace_set=TRACE)[1]
                        for s in seeds]
               for s_name in ("fcfs", "lsf")}
        choice = st.fmean(r["frac_decisions_with_choice"] for r in sch["fcfs"])
        lsf_vs_fcfs = paired_ci([r["p95_task_latency_s"] for r in sch["lsf"]],
                                [r["p95_task_latency_s"] for r in sch["fcfs"]])
        print(f"  scheduler  rate {srate:.4f}/s (busy {sbusy:.2f})  "
              f"choice {choice:.1%}  lsf - fcfs p95 "
              f"{lsf_vs_fcfs['mean_diff_s']:+.2f}s "
              f"[{lsf_vs_fcfs['lo_s']:+.2f}, {lsf_vs_fcfs['hi_s']:+.2f}]",
              flush=True)

        out_rows.append({
            "batch_decode_coeff": coeff,
            "retention": {
                "arrival_rate": rate, "busy": busy,
                "keep_all_recompute_share_of_makespan": recomp_share,
                "ski_rental_rent_active_tick_frac":
                    mean("ski-rental", "rent_active_tick_frac"),
                "by_policy": {pol: {k: mean(pol, k) for k in BREAKDOWN}
                              for pol in ret},
                "per_seed": {pol: {k: [r[k] for r in ret[pol]] for k in BREAKDOWN}
                             for pol in ret},
                "vs_keep_all": {pol: {
                    "p95": paired_ci([r["p95_task_latency_s"] for r in ret[pol]],
                                     [r["p95_task_latency_s"] for r in ret["keep-all"]]),
                    "recompute": paired_ci([r["recompute_seconds"] for r in ret[pol]],
                                           [r["recompute_seconds"] for r in ret["keep-all"]])}
                    for pol in ret if pol != "keep-all"},
                "ski_minus_keep_p95": ski_vs_keep_p95,
                "ski_minus_keep_recompute": ski_vs_keep_rc},
            "scheduler": {
                "arrival_rate": srate, "busy": sbusy,
                "frac_decisions_with_choice": choice,
                "lsf_minus_fcfs_p95": lsf_vs_fcfs}})

    out = {"rows": out_rows, "elapsed_s": round(time.time() - t0, 1),
           "manifest": manifest.to_dict()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nelapsed {out['elapsed_s']:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
