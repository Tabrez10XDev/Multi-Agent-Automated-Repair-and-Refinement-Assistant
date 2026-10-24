"""
Scheduling regime map: where, if anywhere, does scheduling start to matter?

    ./venv/bin/python -u -m experiments.scheduling_map            # ~20-40 min
    ./venv/bin/python -u -m experiments.scheduling_map --quick    # 3 seeds, 3 loads

WHAT THIS ANSWERS
    FINDINGS 27 found no scheduling rule different from FCFS at one load. This
    sweeps load from comfortable to heavy overload, at every batch size where
    the batch cap actually binds, and reports the first load at which LSF or
    EDF separates from FCFS, with the size of the gap there.

THE LOAD AXIS
    load = arrival rate / rate(util 0.8), where rate(u) is the arrival rate
    calibrated (calibrate_rate, FCFS/keep-all) to GPU busy fraction u at the
    same batch size. 1.0 is FINDINGS 27's operating point. Points below 1 are
    calibrated directly at util 0.6, 0.7, 0.9 and land wherever their rate
    falls on this axis; points above are multiples of rate(0.8), because busy
    fraction saturates near 1 and cannot be calibrated past it. 2.2 is the
    accidental overload FINDINGS 23 measured in; 3.0 is one point beyond.

HELD FIXED
    retention (lru), KV budget, trace, hardware profile. slack_factor is
    calibrated once per batch size, at util 0.8, so FCFS misses 20-30% of
    deadlines there, then held fixed across that batch's loads: deadlines
    must mean the same thing at every load or the miss column is not
    comparable along the axis.

PER CELL
    p50, p95, deadline miss rate split into misses already infeasible at the
    task's first admission (no scheduler could have saved them) and misses
    that were feasible then. Paired 95% CIs against FCFS over the same seeds
    (DECISIONS 24). An activity line: fraction of admission decisions with a
    choice, rent-active tick fraction, forced evictions, max and p95 in-flight
    requests, fraction of time at the batch cap. A cell is flagged INACTIVE if
    FCFS had a choice on under 5% of decisions, and CAP-NEVER-BINDS if no seed
    reached max_batch.

BATCH SIZES
    A batch size whose cap never binds at any load is KV-BOUND: the KV budget,
    not max_batch, limits concurrency, so it behaves like any larger cap. It
    is kept and labelled, not dropped, because overload cells there can still
    show scheduling effects. Batch 8 and 16 are merged if identical in every
    cell.

Every calibration is recorded in a run manifest with the hardware profile and
checked before each cell runs (DECISIONS 25, 27). Writes
results/scheduling_map.json.
"""

import argparse
import json
import os
import statistics as st
import time
from concurrent.futures import ProcessPoolExecutor

from experiments.common import (calibrate_rate, calibrate_slack, paired_ci,
                                run_retention_one, run_scheduler_one)
from experiments.manifest import (Calibration, Manifest, bound_args, check_consistent,
                                  default_of)
from simulator.engine import EngineConfig
from simulator.trace_loader import TRACE_SET_EXCLUSIONS
from simulator.workload import (DEFAULT_PROFILE, PROFILES, CostModel, CpuStageSeconds,
                                StageTokens, parse_rate_overrides)
from simulator.workload_sources import check_deadline_basis, get_source

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

POLICIES = [("fcfs", "online", "fcfs"), ("sjf", "online", "sjf"),
            ("edf", "online", "edf"), ("lsf", "online", "lsf"),
            ("lsf", "oracle", "lsf-oracle"),
            ("program-fcfs", "online", "program-fcfs"),
            ("plas", "online", "plas"),
            ("hint-priority", "online", "hint-priority"),
            ("program-fcfs-cap", "online", "program-fcfs-cap"),
            ("program-fcfs-qcap", "online", "program-fcfs-qcap")]
BASELINES = ("fcfs", "program-fcfs")
CALIBRATED_UTILS = [0.6, 0.7, 0.8, 0.9]
OVERLOAD_MULTIPLES = [1.1, 1.25, 1.5, 1.75, 2.2, 3.0]
INACTIVE_CHOICE = 0.05
# --load-axis capacity (OPEN_ISSUES 30): load = arrival rate / saturation
# throughput at the same batch cap, so 1.0 means arrivals match the most the
# server can complete, at every batch size and coefficient.
CAPACITY_LOADS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 2.0]
SATURATION_RATE = 50.0     # tasks/s: every task arrives within about a second
METRICS = ("p50_task_latency_s", "p95_task_latency_s", "mean_task_latency_s",
           "deadline_miss_rate", "miss_rate_infeasible_at_admit",
           "miss_rate_feasible_at_admit", "p95_slowdown", "mean_slowdown")
# Kept per seed so any of them can be paired offline (experiments/reanalyze_slo.py).
PER_SEED = ("p50_task_latency_s", "p95_task_latency_s", "mean_task_latency_s",
            "deadline_miss_rate", "p95_slowdown", "mean_slowdown")
ACTIVITY = ("frac_decisions_with_choice", "rent_active_tick_frac",
            "forced_evictions", "max_batch_seen", "p95_batch",
            "frac_time_at_batch_cap", "max_resident", "frac_time_at_resident_cap",
            "mean_admit_cap",
            "gpu_busy_frac")


def _job(args):
    """One engine run; top level so a process pool can pickle it."""
    (sched, est, seed, rate, slack, batch, kv_mb, retention, trace_set,
     cost, tok, cpu, slo, deadline_basis) = args
    try:
        rep = run_scheduler_one(sched, est, 1, seed, kv_mb, rate, slack, retention,
                      cost, tok, cpu, trace_set, batch=batch, slo=slo,
                      deadline_basis=deadline_basis)
    except RuntimeError as e:
        return {"error": str(e)[:200]}
    out = {k: rep[k] for k in METRICS + ACTIVITY}
    out["miss_by_class"] = {c: v["miss_rate"] for c, v in
                            rep.get("miss_rate_by_slo_class", {}).items()}
    return out


def parse_slo(text):
    """'tight:1.0:0.3,loose:4.0:0.7' -> (("tight", 1.0, 0.3), ("loose", 4.0, 0.7))."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        name, mult, weight = part.split(":")
        out.append((name, float(mult), float(weight)))
    return tuple(out)


def calibrate_point(manifest, cost, tok, cpu, prof, util, batch, kv_mb,
                    trace_set, cache_mode):
    call = (util, 1, 1, kv_mb, cost, tok, cpu)
    kw = {"trace_set": trace_set, "batch": batch}
    b = bound_args(calibrate_rate, *call, **kw)
    under = {"trace_set": b["trace_set"], "scale": b["scale"],
             "util": b["target_util"], "kv_mb": b["kv_mb"],
             "max_batch": b["batch"], "cache_mode": cache_mode,
             "seed": b["seed"], "scheduler": "fcfs", "retention": "keep-all",
             "hardware_profile": prof, "batch_decode_coeff": cost.batch_decode_coeff,
             "prefill_chunk_tokens": default_of(run_retention_one, "chunk")}
    run_config = {"trace_set": trace_set, "scale": 1, "util": util,
                  "kv_mb": kv_mb, "max_batch": batch, "cache_mode": cache_mode,
                  "hardware_profile": prof,
                  "batch_decode_coeff": cost.batch_decode_coeff}
    checks = ("util", "hardware_profile", "batch_decode_coeff")
    name = f"arrival_rate[b={batch},u={util}]"
    check_consistent(name, under, run_config, checks)
    rate, busy = calibrate_rate(*call, **kw, return_achieved=True)
    cal = manifest.record(Calibration(
        name, rate, under, target=f"GPU busy fraction {util} under FCFS/keep-all",
        achieved=busy, also_check=checks))
    return rate, busy, cal


def calibrate_saturation(manifest, cost, tok, cpu, prof, batch, kv_mb, retention,
                         trace_set, cache_mode, seeds):
    """Tasks/s the server completes when every task arrives at once, under FCFS
    at this batch cap: the denominator of the capacity load axis. Mean over
    `seeds`. It includes the drain at the end of a finite burst, so it slightly
    understates steady-state capacity; the bias is the same at every batch."""
    under = {"trace_set": trace_set, "scale": 1, "kv_mb": kv_mb,
             "max_batch": batch, "cache_mode": cache_mode, "retention": retention,
             "hardware_profile": prof, "batch_decode_coeff": cost.batch_decode_coeff,
             "scheduler": "fcfs", "arrival_rate": SATURATION_RATE, "seeds": seeds}
    vals = [run_scheduler_one("fcfs", "online", 1, sd, kv_mb, SATURATION_RATE, 1.5, retention,
                    cost, tok, cpu, trace_set, batch=batch)["throughput_tasks_per_s"]
            for sd in seeds]
    sat = st.fmean(vals)
    cal = manifest.record(Calibration(
        f"saturation_throughput[b={batch}]", sat, under,
        target="tasks/s with every task arriving at once, FCFS",
        achieved=min(vals),
        also_check=("retention", "hardware_profile", "batch_decode_coeff")))
    return sat, cal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace-set", default="holdout_qwen")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--batches", type=int, nargs="+", default=[2, 4, 8, 16])
    ap.add_argument("--kv-mb", type=float, default=1024.0)
    ap.add_argument("--retention", default="lru")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch-decode-coeff", type=float, default=None,
                    help="override the profile's measured value, for a what-if run "
                         "(reported as OVERRIDDEN in the manifest's cost provenance)")
    ap.add_argument("--slo-mix", default=None,
                    help="per-task deadline classes name:multiplier:weight,...; "
                         "default: one common budget (FINDINGS 29)")
    ap.add_argument("--deadline-basis", choices=("estimate", "task"), default="estimate",
                    help="what a deadline is a multiple of: one isolated-task estimate "
                         "(default, FINDINGS 29-31) or each task's own noisy critical "
                         "path (OPEN_ISSUES 29)")
    ap.add_argument("--load-axis", choices=("busy", "capacity"), default="busy",
                    help="busy: rates calibrated to GPU busy fraction (default, "
                         "FINDINGS 29-31); capacity: fractions of saturation "
                         "throughput at each batch cap (OPEN_ISSUES 30)")
    ap.add_argument("--quick", action="store_true",
                    help="3 seeds, loads util 0.8 and 2.2x only, batch 8")
    ap.add_argument("--out", default=os.path.join(_REPO, "results", "scheduling_map.json"))
    ap.add_argument("--profile", default=DEFAULT_PROFILE, choices=sorted(PROFILES),
                    help="hardware profile the cost model is built from "
                         "(simulator/workload.py PROFILES)")
    ap.add_argument("--rates", default=None,
                    help="override per-model rates for a what-if run, "
                         "model=prefill/decode,... in tok/s (reported as OVERRIDDEN "
                         "in the manifest's cost provenance)")
    args = ap.parse_args()

    # A source that cannot support the requested deadline basis is refused
    # before any calibration, not discovered mid-run.
    check_deadline_basis(args.trace_set, args.deadline_basis)
    # Policies the source cannot run (hint-priority ranks MAARA stages on
    # TraceLab) are skipped here, before any calibration, and recorded below.
    excluded = sorted(get_source(args.trace_set).excluded_policies)
    policies = [p for p in POLICIES if p[2] not in excluded]
    slo = parse_slo(args.slo_mix)
    utils, multiples, batches = CALIBRATED_UTILS, OVERLOAD_MULTIPLES, args.batches
    seeds = list(range(1, args.seeds + 1))
    if args.quick:
        utils, multiples, batches, seeds = [0.8], [2.2], [8], [1, 2, 3]

    cost = CostModel.from_profile(
        args.profile, rate_overrides=parse_rate_overrides(args.rates),
        **({} if args.batch_decode_coeff is None
           else {"batch_decode_coeff": args.batch_decode_coeff}))
    tok, cpu = StageTokens(), CpuStageSeconds()
    prof = cost.profile
    cache_mode = EngineConfig().cache_mode
    manifest = Manifest.start("scheduling_map", params={
        "trace_set": args.trace_set, "kv_mb": args.kv_mb, "batches": batches,
        "excluded_tasks": TRACE_SET_EXCLUSIONS.get(args.trace_set, {}),
        "retention": args.retention, "seeds": seeds, "policies": [p[2] for p in policies],
        "calibrated_utils": utils, "overload_multiples_of_util_0.8": multiples,
        "load_axis": ("arrival rate / saturation throughput at the same batch "
                      "(FCFS, all tasks at once)" if args.load_axis == "capacity" else
                      "arrival rate / rate calibrated to util 0.8 at the same batch"),
        "load_axis_mode": args.load_axis,
        "capacity_loads": CAPACITY_LOADS if args.load_axis == "capacity" else None,
        "deadline_basis": args.deadline_basis,
        "hardware_profile": prof, "cost_provenance": cost.provenance(),
        "cache_mode": cache_mode, "inactive_if_fcfs_choice_below": INACTIVE_CHOICE,
        "slo_classes": slo, "slo_seed": "= engine seed",
        "deadline": (("arrival + slack_factor * multiplier * noisy critical path"
                      if slo else "arrival + slack_factor * noisy critical path")
                     if args.deadline_basis == "task" else
                     ("arrival + slack_factor * multiplier * size_scale * isolated"
                      if slo else "arrival + slack_factor * isolated")),
        # Only present when a source excludes something, so MAARA manifests
        # keep exactly today's keys.
        **({"excluded_policies": excluded,
            "excluded_policies_reason": "not meaningful on this workload source "
                                        "(workload_sources.excluded_policies)"}
           if excluded else {})})
    t0 = time.time()
    print(f"trace={args.trace_set} kv={args.kv_mb:.0f}MB retention={args.retention} "
          f"profile={prof} coeff={cost.batch_decode_coeff} seeds={len(seeds)} "
          f"batches={batches} slo={slo} deadlines={args.deadline_basis} "
          f"load-axis={args.load_axis}", flush=True)

    out_batches = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for batch in batches:
            print(f"\n=== batch {batch} ===", flush=True)
            cal, sat = {}, None
            if args.load_axis == "capacity":
                sat, sat_cal = calibrate_saturation(
                    manifest, cost, tok, cpu, prof, batch, args.kv_mb, args.retention,
                    args.trace_set, cache_mode, seeds[:3])
                base_rate = 0.8 * sat
                print(f"  saturation throughput = {sat:.4f} tasks/s; slack calibrated "
                      f"at load 0.8 = {base_rate:.4f}/s", flush=True)
            else:
                for u in sorted(set(utils) | {0.8}):
                    rate, busy, c = calibrate_point(manifest, cost, tok, cpu, prof, u,
                                                    batch, args.kv_mb, args.trace_set,
                                                    cache_mode)
                    cal[u] = (rate, busy, c)
                    print(f"  rate(util {u}) = {rate:.4f}/s  busy {busy:.2f}", flush=True)
                base_rate = cal[0.8][0]

            # Slack: once per batch, at util 0.8, then fixed for every load.
            slack_call = (1, seeds[:3], args.kv_mb, base_rate, args.retention,
                          cost, tok, cpu, args.trace_set)
            sb = bound_args(calibrate_slack, *slack_call, batch=batch,
                            deadline_basis=args.deadline_basis)
            slack_under = {"trace_set": sb["trace_set"], "scale": sb["scale"],
                           "kv_mb": sb["kv_mb"], "max_batch": sb["batch"],
                           "cache_mode": cache_mode, "arrival_rate": sb["rate"],
                           "retention": sb["retention"], "seeds": sb["seeds"],
                           "hardware_profile": prof,
                           "batch_decode_coeff": cost.batch_decode_coeff,
                           "slo_classes": slo,
                           "deadline_basis": sb["deadline_basis"]}
            slack, miss = calibrate_slack(*slack_call, batch=batch, slo=slo,
                                          deadline_basis=args.deadline_basis)
            slack_cal = manifest.record(Calibration(
                f"slack_factor[b={batch}]", slack, slack_under,
                target="FCFS deadline-miss rate in [0.20, 0.30] at util 0.8",
                achieved=miss, also_check=("retention", "hardware_profile",
                                           "batch_decode_coeff", "slo_classes",
                                           "deadline_basis")))
            print(f"  slack_factor = {slack:.3f}  (FCFS miss {miss:.1%} at util 0.8)",
                  flush=True)

            if args.load_axis == "capacity":
                fracs = [0.8] if args.quick else CAPACITY_LOADS
                points = [(None, f, (sat, None, sat_cal)) for f in fracs]
            else:
                points = [(u, 1.0, cal[u]) for u in utils] + \
                         [(0.8, m, cal[0.8]) for m in multiples]
            cells = []
            for util, mult, (urate, _, ucal) in points:
                rate = urate * mult
                load = rate / (sat if sat else base_rate)
                run_config = {"trace_set": args.trace_set, "scale": 1, "util": util,
                              "kv_mb": args.kv_mb, "max_batch": batch,
                              "cache_mode": cache_mode, "retention": args.retention,
                              "hardware_profile": prof,
                              "batch_decode_coeff": cost.batch_decode_coeff,
                              "slo_classes": slo,
                              "deadline_basis": args.deadline_basis}
                # A cell refuses to run on any calibration mismatch.
                manifest.assert_consistent(run_config, calibrations=[ucal, slack_cal])

                jobs = [(s, e, sd, rate, slack, batch, args.kv_mb, args.retention,
                         args.trace_set, cost, tok, cpu, slo, args.deadline_basis)
                        for s, e, _ in policies for sd in seeds]
                res = list(pool.map(_job, jobs))
                by_pol = {}
                for i, (_, _, label) in enumerate(policies):
                    by_pol[label] = res[i * len(seeds):(i + 1) * len(seeds)]
                cell = {"util_calibrated": util, "multiple_of_util_rate": mult,
                        "load": load, "arrival_rate": rate, "policies": {}}
                errors = {lab: sum("error" in r for r in rs) for lab, rs in by_pol.items()}
                if any(errors.values()):
                    cell["errors"] = errors
                base_ok = {bl: [r for r in by_pol[bl] if "error" not in r] for bl in BASELINES}
                for label, rs in by_pol.items():
                    ok = [r for r in rs if "error" not in r]
                    if not ok:
                        continue
                    row = {k: st.fmean(r[k] for r in ok) for k in METRICS + ACTIVITY}
                    row["max_batch_seen_max"] = max(r["max_batch_seen"] for r in ok)
                    row["max_resident_max"] = max(r["max_resident"] for r in ok)
                    row["per_seed"] = {k: [r[k] for r in ok] for k in PER_SEED}
                    if slo:
                        row["per_seed"]["miss_by_class"] = {
                            c: [r["miss_by_class"].get(c, 0.0) for r in ok] for c, _, _ in slo}
                    if slo:
                        row["miss_by_class"] = {c: st.fmean(r["miss_by_class"].get(c, 0.0) for r in ok)
                                                for c, _, _ in slo}
                    for bl in BASELINES:
                        fc = base_ok[bl]
                        if label == bl or not (len(ok) == len(fc) == len(seeds)):
                            continue
                        tag = "fcfs" if bl == "fcfs" else "program"
                        row[f"p95_vs_{tag}"] = paired_ci(
                            [r["p95_task_latency_s"] for r in ok],
                            [r["p95_task_latency_s"] for r in fc])
                        row[f"miss_vs_{tag}"] = paired_ci(
                            [r["deadline_miss_rate"] for r in ok],
                            [r["deadline_miss_rate"] for r in fc])
                    cell["policies"][label] = row
                f = cell["policies"].get("fcfs", {})
                cell["inactive"] = f.get("frac_decisions_with_choice", 0) < INACTIVE_CHOICE
                # The cap is enforced on resident requests (all models), so that
                # is what decides whether it binds. max_batch_seen counts only the
                # model served each iteration and undercounts (OPEN_ISSUES §28).
                cell["cap_never_binds"] = f.get("max_resident_max", 0) < batch
                cells.append(cell)

                flag = (" INACTIVE" if cell["inactive"] else "") + \
                       (" CAP-NEVER-BINDS" if cell["cap_never_binds"] else "")
                print(f"  load {load:4.2f} (util {util}, x{mult})  "
                      f"choice {f.get('frac_decisions_with_choice', 0):5.1%}  "
                      f"rent>0 {f.get('rent_active_tick_frac', 0):5.1%}  "
                      f"forced {f.get('forced_evictions', 0):5.1f}  "
                      f"resident max {f.get('max_resident_max', 0)}  active max/p95 "
                      f"{f.get('max_batch_seen_max', 0)}/{f.get('p95_batch', 0):.1f}  "
                      f"at-cap {f.get('frac_time_at_resident_cap', 0):5.1%}{flag}", flush=True)
                print(f"    fcfs   p50 {f.get('p50_task_latency_s', 0):7.1f}  p95 "
                      f"{f.get('p95_task_latency_s', 0):7.1f}  miss "
                      f"{f.get('deadline_miss_rate', 0):5.1%} (infeasible "
                      f"{f.get('miss_rate_infeasible_at_admit', 0):5.1%})", flush=True)
                for label, row in cell["policies"].items():
                    if label in ("lsf", "edf", "hint-priority") and "p95_vs_program" in row:
                        p, m = row["p95_vs_program"], row["miss_vs_program"]
                        print(f"    {label:<13} vs program-fcfs: p95 {p['mean_diff_s']:+7.2f} "
                              f"[{p['lo_s']:+.2f}, {p['hi_s']:+.2f}]{'*' if p['significant'] else ' '}"
                              f"  miss {m['mean_diff_s']:+6.1%} [{m['lo_s']:+.1%}, "
                              f"{m['hi_s']:+.1%}]{'*' if m['significant'] else ''}", flush=True)
                    if label == "fcfs" or "p95_vs_fcfs" not in row:
                        continue
                    p, m = row["p95_vs_fcfs"], row["miss_vs_fcfs"]
                    print(f"    {label:<10} p95 {p['mean_diff_s']:+7.2f} "
                          f"[{p['lo_s']:+.2f}, {p['hi_s']:+.2f}]{'*' if p['significant'] else ' '}"
                          f"  miss {m['mean_diff_s']:+6.1%} "
                          f"[{m['lo_s']:+.1%}, {m['hi_s']:+.1%}]{'*' if m['significant'] else ''}",
                          flush=True)
            out_batches.append({"max_batch": batch, "slack_factor": slack,
                                "slack_fcfs_miss_at_util_0.8": miss,
                                "rate_util_0.8": base_rate, "cells": cells,
                                "saturation_throughput": sat,
                                "cap_binds_somewhere": any(not c["cap_never_binds"]
                                                           for c in cells)})

    summary = summarise(out_batches)
    out = {"batches": out_batches, "summary": summary,
           "elapsed_s": round(time.time() - t0, 1), "manifest": manifest.to_dict()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\n=== summary ===")
    for line in summary["lines"]:
        print("  " + line)
    print(f"\nelapsed {out['elapsed_s']:.0f}s -> {args.out}")
    return 0


def _identical(a, b):
    """True if two batch sizes produced the same per-seed results everywhere."""
    if len(a["cells"]) != len(b["cells"]):
        return False
    for ca, cb in zip(a["cells"], b["cells"]):
        for lab, ra in ca["policies"].items():
            rb = cb["policies"].get(lab)
            if rb is None or ra["per_seed"] != rb["per_seed"]:
                return False
    return True


def summarise(batches):
    lines, crossover = [], {}
    by_b = {b["max_batch"]: b for b in batches}
    if 8 in by_b and 16 in by_b and _identical(by_b[8], by_b[16]):
        lines.append("batch 8 and 16 are identical in every cell; merged as 8/16")
        by_b[16]["merged_into"] = 8
    for b in batches:
        if b.get("merged_into"):
            continue
        mb = b["max_batch"]
        if not b["cap_binds_somewhere"]:
            b["kv_bound"] = True
            lines.append(f"batch {mb}: cap never binds at any load (KV-bound: the "
                         f"KV budget, not max_batch, limits concurrency)")
        for tag in ("fcfs", "program"):
            for pol in ("lsf", "edf", "hint-priority", "program-fcfs", "plas",
                        "program-fcfs-cap", "program-fcfs-qcap"):
                if pol == "program-fcfs" and tag == "program":
                    continue
                first = None
                for c in sorted(b["cells"], key=lambda c: c["load"]):
                    row = c["policies"].get(pol, {})
                    keys = (f"p95_vs_{tag}", f"miss_vs_{tag}")
                    if keys[0] not in row:
                        continue
                    sig = [k for k in keys if row[k]["significant"]]
                    if sig:
                        p, m = row[keys[0]], row[keys[1]]
                        first = {"load": c["load"], "metrics": sig,
                                 "p95_diff_s": p["mean_diff_s"], "p95_ci": [p["lo_s"], p["hi_s"]],
                                 "miss_diff": m["mean_diff_s"], "miss_ci": [m["lo_s"], m["hi_s"]],
                                 "inactive": c["inactive"], "cap_never_binds": c["cap_never_binds"]}
                        break
                key = f"{pol}@b{mb}" if tag == "fcfs" else f"{pol}@b{mb}_vs_program"
                crossover[key] = first
                vs = "fcfs" if tag == "fcfs" else "program-fcfs"
                if first is None:
                    lines.append(f"batch {mb}: {pol} never separates from {vs} at any load")
                else:
                    lines.append(
                        f"batch {mb}: {pol} first separates from {vs} at load {first['load']:.2f} on "
                        f"{', '.join(first['metrics'])}: p95 {first['p95_diff_s']:+.1f}s "
                        f"[{first['p95_ci'][0]:+.1f}, {first['p95_ci'][1]:+.1f}], miss "
                        f"{first['miss_diff']:+.1%} [{first['miss_ci'][0]:+.1%}, "
                        f"{first['miss_ci'][1]:+.1%}]")
    return {"crossover": crossover, "lines": lines}


if __name__ == "__main__":
    raise SystemExit(main())
