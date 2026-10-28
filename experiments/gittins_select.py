"""
The DECISIONS 38 dev selection rule, as applied to `D2_gittins_dev.json`.

For each policy, L is the mean over the map's cells of its mean p95 task
latency divided by `program-fcfs`'s, and D the same for p95 slowdown. The
guarded Gittins variant with the smallest max(L, D) is chosen; a tie goes to
the smaller X, then `gittins` over `gittins-sd`. The other columns count
cells where the paired 95% CI against `program-fcfs` excludes zero.

    ./venv/bin/python -m experiments.gittins_select results/tracelab/D2_gittins_dev.json
"""

from __future__ import annotations

import argparse
import json
import re

from experiments.common import paired_ci

GUARDED = re.compile(r"^(gittins|gittins-sd)-guard-x(\d+)$")


def scores(d):
    """{policy: (max(L, D), L, D, sd_better, sd_worse, lat_better, lat_worse, cells)}."""
    cells = [c for b in d["batches"] for c in b["cells"]]
    pols = [p for p in cells[0]["policies"] if p not in ("fcfs", "program-fcfs")]
    out = {}
    for p in pols:
        L, D, n = [], [], {"sd": [0, 0], "lat": [0, 0]}
        for c in cells:
            base, q = c["policies"]["program-fcfs"], c["policies"][p]
            L.append(q["p95_task_latency_s"] / base["p95_task_latency_s"])
            D.append(q["p95_slowdown"] / base["p95_slowdown"])
            for tag, key in (("sd", "p95_slowdown"), ("lat", "p95_task_latency_s")):
                ci = paired_ci(q["per_seed"][key], base["per_seed"][key])
                n[tag][0] += ci["hi_s"] < 0
                n[tag][1] += ci["lo_s"] > 0
        l, dd = sum(L) / len(L), sum(D) / len(D)
        out[p] = (max(l, dd), l, dd, *n["sd"], *n["lat"], len(cells))
    return out


def choose(sc):
    def key(p):
        m = GUARDED.match(p)
        return (sc[p][0], int(m.group(2)), m.group(1) != "gittins")
    return min((p for p in sc if GUARDED.match(p)), key=key)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("map")
    args = ap.parse_args()
    with open(args.map) as fh:
        d = json.load(fh)
    m = d["manifest"]
    print(f"manifest {m.get('git_hash', '')[:7]} dirty {m.get('git_dirty')}")
    sc = scores(d)
    print(f"{'policy':22s} {'max':>6s} {'L':>6s} {'D':>6s}  sd b/w   lat b/w  cells")
    for p, (mx, l, dd, bs, ws, bl, wl, n) in sorted(sc.items(), key=lambda r: r[1][0]):
        print(f"{p:22s} {mx:6.3f} {l:6.3f} {dd:6.3f}  {bs:2d}/{ws:<2d}    {bl:2d}/{wl:<2d}   {n}")
    print("chosen:", choose(sc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
