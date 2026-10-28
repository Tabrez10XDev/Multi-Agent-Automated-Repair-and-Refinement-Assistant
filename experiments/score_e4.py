"""
Score DECISIONS 38's E4 hypotheses on `E4_sealed2.json`, paired 95% CI over
seeds per cell (better = hi < 0, worse = lo > 0):

    H1  gittins-guard-x2 vs program-fcfs, p95 slowdown: better in >= 15 of 30
        cells and worse in <= 2
    H2  gittins-guard-x2 vs program-fcfs, p95 latency: worse in <= 2 cells
    H3  gittins-guard-x2 vs plas-guard-x2, mean slowdown: better in >= 5 cells

Also the freeze-amendment additions: mean ratios L and D over cells with a t
interval across cells (descriptive: cells share traces and are correlated),
and the number of active cells.

    ./venv/bin/python -m experiments.score_e4 results/tracelab/E4_sealed2.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st

from experiments.common import paired_ci

CHOSEN, ABLATION, BASE = "gittins-guard-x2", "plas-guard-x2", "program-fcfs"
FREEZE = "0b2622d3e8f285aaff7fc4297a517ccacaf56a47"


def counts(cells, a, b, key):
    better = worse = 0
    rows = []
    for c in cells:
        ci = paired_ci(c["policies"][a]["per_seed"][key], c["policies"][b]["per_seed"][key])
        better += ci["hi_s"] < 0
        worse += ci["lo_s"] > 0
        rows.append((c["max_batch"], c["load"], c["inactive"], ci))
    return better, worse, rows


def ratio(cells, a, b, key):
    r = [c["policies"][a][key] / c["policies"][b][key] for c in cells]
    m, half = st.fmean(r), 2.045 * st.stdev(r) / len(r) ** 0.5   # t(29)
    return m, m - half, m + half


def score(d):
    cells = [dict(c, max_batch=b["max_batch"]) for b in d["batches"] for c in b["cells"]]
    h1 = counts(cells, CHOSEN, BASE, "p95_slowdown")
    h2 = counts(cells, CHOSEN, BASE, "p95_task_latency_s")
    h3 = counts(cells, CHOSEN, ABLATION, "mean_slowdown")
    return {
        "cells": len(cells), "active": sum(not c["inactive"] for c in cells),
        "H1": {"better": h1[0], "worse": h1[1], "holds": h1[0] >= 15 and h1[1] <= 2},
        "H2": {"worse": h2[1], "better": h2[0], "holds": h2[1] <= 2},
        "H3": {"better": h3[0], "worse": h3[1], "holds": h3[0] >= 5},
        "L": ratio(cells, CHOSEN, BASE, "p95_task_latency_s"),
        "D": ratio(cells, CHOSEN, BASE, "p95_slowdown"),
        "L_ablation": ratio(cells, ABLATION, BASE, "p95_task_latency_s"),
        "D_ablation": ratio(cells, ABLATION, BASE, "p95_slowdown"),
        "per_cell": {"H1": h1[2], "H2": h2[2], "H3": h3[2]}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    args = ap.parse_args()
    with open(args.map) as fh:
        d = json.load(fh)
    m = d["manifest"]
    print(f"manifest {m['git_hash'][:7]} dirty {m['git_dirty']} "
          f"trace {m['params']['trace_set']} freeze {FREEZE[:7]}")
    s = score(d)
    print(f"cells {s['cells']}, active {s['active']}")
    for h in ("H1", "H2", "H3"):
        x = s[h]
        print(f"{h}: better {x['better']}, worse {x['worse']} -> "
              f"{'HOLDS' if x['holds'] else 'FAILS'}")
    for k in ("L", "D", "L_ablation", "D_ablation"):
        mean, lo, hi = s[k]
        print(f"{k:11s} {mean:.3f} [{lo:.3f}, {hi:.3f}]")
    print("per cell (batch, load, inactive): H1 p95-slowdown diff | H2 p95-latency diff s "
          "| H3 mean-slowdown diff")
    for r1, r2, r3 in zip(*(s["per_cell"][h] for h in ("H1", "H2", "H3"))):
        b, load, inact, c1 = r1
        f = lambda c: (f"{c['mean_diff_s']:+8.2f} [{c['lo_s']:+8.2f}, {c['hi_s']:+8.2f}]"
                       + ("*" if c["hi_s"] < 0 or c["lo_s"] > 0 else " "))
        print(f"  b{b} {load:4.2f} {'I' if inact else 'A'}  {f(c1)} | {f(r2[3])} | {f(r3[3])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
