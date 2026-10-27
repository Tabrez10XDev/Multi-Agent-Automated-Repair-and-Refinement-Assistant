"""
Per-task slowdown by task size, MAARA vs TraceLab (DECISIONS 37, exploratory
amendment 2026-10-27). No prediction; development data only.

The scheduling maps store whole-run numbers (p95 over all tasks), which cannot
say WHO a rule helps. This replays two cells of each published map and keeps
every task's latency and lone-replay time, so slowdown can be read by size.

Each cell reuses the saturation throughput and slack factor stored in its map
(`tracelab/D1_dev_c0.3.json`, `plas_taskdl/S2_gpu_public.json`) rather than
recalibrating, so it is the same operating point. Golden check: each run's p95
task latency must equal the map's per-seed value; the run stops otherwise.

    ./venv/bin/python -m experiments.size_breakdown --workers 6
"""

from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

from experiments.common import scheduler_engine
from experiments.manifest import Manifest
from simulator.workload import CostModel, CpuStageSeconds, StageTokens

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROFILE = "gpu_5060ti_public"
POLICIES = ("fcfs", "program-fcfs", "plas", "lsf", "edf")
# label: (map file, trace_set, kv_mb, deadline_basis); retention and SLO mix are
# read from the map's manifest.
WORKLOADS = {
    "MAARA": ("plas_taskdl/S2_gpu_public.json", "holdout_qwen", 1024.0, "estimate"),
    "TraceLab dev": ("tracelab/D1_dev_c0.3.json", "tracelab_dev", 8192.0, "task"),
}


def _cell(results, rel, batch, load):
    with open(os.path.join(results, rel)) as fh:
        d = json.load(fh)
    b = next(x for x in d["batches"] if x["max_batch"] == batch)
    c = next(x for x in b["cells"] if abs(x["load"] - load) < 1e-9)
    prm = d["manifest"]["params"]
    return {"sat": b["saturation_throughput"], "slack": b["slack_factor"],
            "rate": c["arrival_rate"],
            "retention": prm["retention"],
            "slo": tuple(tuple(x) for x in prm["slo_classes"]),
            "p95": {p: c["policies"][p]["per_seed"]["p95_task_latency_s"]
                    for p in POLICIES}}


def _job(args):
    (wl, pol, seed, rate, slack, batch, kv, retention, trace_set, slo, basis) = args
    cost = CostModel.from_profile(PROFILE)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    eng = scheduler_engine(pol, "online", 1, seed, kv, rate, slack, retention,
                           cost, tok, cpu, trace_set, batch=batch, slo=slo,
                           deadline_basis=basis)
    rep = eng.run()
    tasks = [{"task": i, "size_s": eng._isolated_service_s(t),
              "latency_s": eng.task_done_at[t.task_id] - t.arrival_time}
             for i, t in enumerate(eng.tasks) if t.task_id in eng.task_done_at]
    return {"p95_task_latency_s": rep["p95_task_latency_s"],
            "p95_slowdown": rep["p95_slowdown"], "tasks": tasks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=os.path.join(_REPO, "results"))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--loads", type=float, nargs="+", default=[1.0, 1.5])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_REPO, "results", "tracelab",
                                                  "size_breakdown.json"))
    args = ap.parse_args()
    seeds = list(range(1, args.seeds + 1))
    manifest = Manifest.start("size_breakdown", params={
        "workloads": {k: list(v) for k, v in WORKLOADS.items()}, "policies": POLICIES,
        "batch": args.batch, "loads": args.loads, "seeds": seeds,
        "hardware_profile": PROFILE,
        "calibration": "saturation and slack read from each map, not recalibrated",
        "task_size": "Engine._isolated_service_s, the slowdown denominator"})
    t0 = time.time()
    cells = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for wl, (rel, trace_set, kv, basis) in WORKLOADS.items():
            for load in args.loads:
                c = _cell(args.results, rel, args.batch, load)
                jobs = {(pol, sd): pool.submit(_job, (
                            wl, pol, sd, c["rate"], c["slack"], args.batch, kv,
                            c["retention"], trace_set, c["slo"], basis))
                        for pol in POLICIES for sd in seeds}
                for pol in POLICIES:
                    for sd in seeds:
                        r = jobs[(pol, sd)].result()
                        want = c["p95"][pol][sd - 1]
                        if abs(r["p95_task_latency_s"] - want) > 1e-6 * max(1.0, want):
                            raise SystemExit(f"golden check failed: {wl} load {load} "
                                             f"{pol} seed {sd}: p95 "
                                             f"{r['p95_task_latency_s']} vs map {want}")
                        cells.append({"workload": wl, "load": load, "policy": pol,
                                      "seed": sd, **r})
                print(f"[{time.time() - t0:6.0f}s] {wl} load {load}: "
                      f"{len(POLICIES) * len(seeds)} runs, p95 matches the map",
                      flush=True)
    out = {"cells": cells, "elapsed_s": round(time.time() - t0, 1),
           "manifest": manifest.to_dict()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"{len(cells)} runs in {out['elapsed_s']:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
