"""
DECISIONS 39's dev go/no-go for `kleinrock-switch`, on the three dev maps.

Point values, not bounds: the mean over active cells of the policy's mean
p95 over seeds divided by the baseline's. Cells are counted worse beyond a
margin d when the paired 95% t interval of (policy - baseline) has
lo > d x the baseline's mean (5 seeds give too few distinct bootstrap
resamples for per-cell intervals).

    K1  dev3, sealed2: P vs program-fcfs, p95 slowdown: < 1.00, <= 2 cells worse (d = 0)
    K2  dev3, sealed2: P vs fcfs, p95 latency and p95 slowdown: <= 1.05, <= 2 cells (d = 0.05)
    K3  holdout_qwen: no switch in any run; else P vs program-fcfs p95 latency
        <= 1.02, <= 2 cells (d = 0.02)

dev3 needs at least 8 active cells, or K1/K2 there are inconclusive and the
answer is no-go. sealed2 is scored on its active cells whatever their number
(DECISIONS 39 amendment). GTS is reported the same way, with no verdict.

    ./venv/bin/python -m experiments.go_e5 DEV3.json SEALED2.json MAARA.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st

from experiments.common import paired_ci, seed_bootstrap
from experiments.score_sealed import map_cells

P, GTS = "kleinrock-switch", "gts"
MIN_ACTIVE = 8
LAT, SLOW = "p95_task_latency_s", "p95_slowdown"


def check(d, policy, base, key, delta):
    cells = map_cells(d)
    active = [c for c in cells if not c["inactive"]]
    ratio = lambda c: (st.fmean(c["policies"][policy]["per_seed"][key])
                       / st.fmean(c["policies"][base]["per_seed"][key]))
    worse = sum(paired_ci(c["policies"][policy]["per_seed"][key],
                          c["policies"][base]["per_seed"][key])["lo_s"]
                > delta * st.fmean(c["policies"][base]["per_seed"][key]) for c in cells)
    return {"active": len(active), "ratio": st.fmean(map(ratio, active)) if active else None,
            "worse": worse}


def switched(d, policy=P):
    logs = [lg for c in map_cells(d)
            for lg in c["policies"][policy]["per_seed"].get("scheduler_log", [])]
    return sum(bool(lg and lg["switches"]) for lg in logs), len(logs), logs


def verdict(dev3, sealed2, maara, policy=P):
    out, ok = [], True
    for name, d, min_active in (("dev3", dev3, MIN_ACTIVE), ("sealed2", sealed2, 1)):
        for test, base, key, lim, delta in (("K1", "program-fcfs", SLOW, 1.00, 0.0),
                                            ("K2", "fcfs", LAT, 1.05, 0.05),
                                            ("K2", "fcfs", SLOW, 1.05, 0.05)):
            r = check(d, policy, base, key, delta)
            if r["active"] < min_active:
                good, why = False, "inconclusive (too few active cells)"
            else:
                strict = r["ratio"] < lim if test == "K1" else r["ratio"] <= lim
                good, why = strict and r["worse"] <= 2, ""
            ok &= good
            out.append((name, test, base, key, r, good, why))
    n_sw, n_runs, logs = switched(maara, policy) if policy == P else (0, 0, [])
    expected = sum(len(c["policies"][policy]["per_seed"][LAT]) for c in map_cells(maara))
    r = check(maara, policy, "program-fcfs", LAT, 0.02)
    # A missing or errored log must not pass as "no switch".
    good = (n_sw == 0 and n_runs == expected and all(logs)) if policy == P else False
    if policy == P and not good:
        good = r["active"] >= 1 and r["ratio"] <= 1.02 and r["worse"] <= 2
    ok &= good
    out.append(("maara", "K3", "program-fcfs", LAT, dict(r, switched=n_sw, runs=n_runs),
                good, ""))
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dev3")
    ap.add_argument("sealed2")
    ap.add_argument("maara")
    args = ap.parse_args()
    maps = []
    for p in (args.dev3, args.sealed2, args.maara):
        with open(p) as fh:
            maps.append(json.load(fh))
        m = maps[-1]["manifest"]
        print(f"{p}: {m['git_hash'][:7]} dirty {m['git_dirty']} trace {m['params']['trace_set']}")
    for pol in (P, GTS):
        ok, rows = verdict(*maps, policy=pol)
        print(f"\n{pol}" + ("" if pol == GTS else f": {'GO' if ok else 'NO-GO'}"))
        for name, test, base, key, r, good, why in rows:
            extra = (f" switched {r['switched']}/{r['runs']}" if "switched" in r else "")
            ratio = "n/a" if r["ratio"] is None else f"{r['ratio']:.3f}"
            mark = "" if pol == GTS else ("pass" if good else "FAIL")
            print(f"  {name:7s} {test} vs {base:12s} {key:19s} active {r['active']:2d} "
                  f"ratio {ratio} worse {r['worse']}{extra} {mark} {why}")
    # Power note for K2 (DECISIONS 39): dev3 active-mean bootstrap half-width,
    # x 0.5 for 20 seeds.
    cells = map_cells(maps[0])
    act = [not c["inactive"] for c in cells]
    for key in (LAT, SLOW):
        b = seed_bootstrap([c["policies"][P]["per_seed"][key] for c in cells],
                           [c["policies"]["fcfs"]["per_seed"][key] for c in cells], active=act)
        am = b["active_mean"]
        if am:
            print(f"\npower note, K2 {key}: dev3 half-width {(am['hi'] - am['lo']) / 2:.3f}, "
                  f"projected at 20 seeds {(am['hi'] - am['lo']) / 4:.3f}")
    n_sw, n_runs, logs = switched(maps[0])
    at = [lg["switches"][0]["arrived"] for lg in logs if lg and lg["switches"]]
    print(f"dev3 switches: {n_sw}/{n_runs} runs, arrivals at first switch "
          f"median {st.median(at) if at else 'n/a'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
