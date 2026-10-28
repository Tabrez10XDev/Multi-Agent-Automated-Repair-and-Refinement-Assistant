"""
Score a policy against baselines on a scheduling map, with the statistics
DECISIONS 39 fixes for every result from E4 on:

- per cell, the paired 95% t interval over seeds (`paired_ci`, better =
  hi < 0, worse = lo > 0), as every map has been counted so far;
- the ratio of the policy's mean to the baseline's, with a seed-cluster
  bootstrap (`seed_bootstrap`): the mean over active cells (primary), the
  mean over all cells and the median, each with a 95% interval, and each
  cell's own ratio interval, so cells can be counted against a margin.

Active cells are those the driver did not flag inactive (FCFS had a choice in
at least 5% of decisions), which depends on FCFS's run only, not the policy.

    ./venv/bin/python -m experiments.score_sealed results/tracelab/E4_sealed2.json \\
        --policy gittins-guard-x2 --baselines fcfs program-fcfs --margin 0.05
"""

from __future__ import annotations

import argparse
import json

from experiments.common import paired_ci, seed_bootstrap

METRICS = ("p95_task_latency_s", "p95_slowdown")


def map_cells(d):
    return [dict(c, max_batch=b["max_batch"]) for b in d["batches"] for c in b["cells"]]


def score(d, policy, baseline, key, margin=0.0, reps=10_000):
    """Counts and bootstrap ratios of `policy` vs `baseline` on metric `key`.
    `beyond_margin` counts cells whose ratio interval lies wholly above
    1 + margin; `below_one` those wholly below 1."""
    cells = map_cells(d)
    a = [c["policies"][policy]["per_seed"][key] for c in cells]
    b = [c["policies"][baseline]["per_seed"][key] for c in cells]
    better = worse = 0
    for x, y in zip(a, b):
        ci = paired_ci(x, y)
        better += ci["hi_s"] < 0
        worse += ci["lo_s"] > 0
    boot = seed_bootstrap(a, b, active=[not c["inactive"] for c in cells], reps=reps)
    return {
        "cells": len(cells), "better": better, "worse": worse,
        "beyond_margin": sum(r["lo"] > 1.0 + margin for r in boot["cells"]),
        "below_one": sum(r["hi"] < 1.0 for r in boot["cells"]),
        "margin": margin, "bootstrap": boot,
        "where": [(c["max_batch"], c["load"], c["inactive"]) for c in cells]}


def _fmt(s):
    return "n/a" if s is None else f"{s['value']:.3f} [{s['lo']:.3f}, {s['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--baselines", nargs="+", default=["fcfs", "program-fcfs"])
    ap.add_argument("--metrics", nargs="+", default=list(METRICS))
    ap.add_argument("--margin", type=float, default=0.0,
                    help="count cells whose ratio interval is wholly above 1 + margin")
    ap.add_argument("--per-cell", action="store_true")
    args = ap.parse_args()
    with open(args.map) as fh:
        d = json.load(fh)
    m = d["manifest"]
    print(f"manifest {m['git_hash'][:7]} dirty {m['git_dirty']} "
          f"trace {m['params']['trace_set']} policy {args.policy}")
    for base in args.baselines:
        for key in args.metrics:
            s = score(d, args.policy, base, key, args.margin)
            bs = s["bootstrap"]
            print(f"\nvs {base}, {key}: cells {s['cells']}, active {bs['n_active']}, "
                  f"seeds {bs['n_seeds']}")
            print(f"  paired CI:  better {s['better']}, worse {s['worse']}")
            print(f"  ratio, active mean {_fmt(bs['active_mean'])}")
            print(f"  ratio, all mean    {_fmt(bs['mean'])}")
            print(f"  ratio, median      {_fmt(bs['median'])}")
            print(f"  cells wholly above {1 + args.margin:.2f}: {s['beyond_margin']}, "
                  f"wholly below 1: {s['below_one']}")
            if args.per_cell:
                for (b, load, inact), r in zip(s["where"], bs["cells"]):
                    print(f"    b{b} {load:4.2f} {'I' if inact else 'A'}  {_fmt(r)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
