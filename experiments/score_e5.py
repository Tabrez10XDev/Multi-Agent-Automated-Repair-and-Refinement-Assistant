"""
Score E5 (DECISIONS 39-40) for the scored policy P (`gts` from section 40):

    K1  sealed3: P vs program-fcfs, p95 slowdown: active-mean hi < 1.00 and
        at most 2 cells beyond d = 0
    K2  sealed3: P vs fcfs, p95 latency and p95 slowdown, each: active-mean
        hi <= 1.05 and at most 2 cells beyond d = 0.05
    K3  final_qwen: P identical to program-fcfs in every seed of every cell;
        otherwise active-mean hi <= 1.02 and at most 2 cells beyond d = 0.02
    K4  sealed3: P vs program-fcfs, p95 latency: reported, no verdict

Primary: the seed-cluster bootstrap of `score_sealed` (active-cell mean,
percentile 95%). "Beyond d": a cell whose own bootstrap ratio interval has
lo > 1 + d, counted over all cells. Fewer than 8 active cells on sealed3 makes
K1/K2 inconclusive.

Sensitivity, reported beside each bound: a t(n-1) interval on the log of the
per-seed active-cell mean ratio, and the geometric mean of the active cells'
ratios. A verdict is FRAGILE if the bound test gives the other answer with
the t upper bound in place of the bootstrap one, or with the geometric mean
in place of the arithmetic.

    ./venv/bin/python -m experiments.score_e5 results/tracelab/E5_sealed3.json \\
        results/tracelab/E5_final_qwen.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics as st

from experiments.common import _T975
from experiments.score_sealed import map_cells, score

P_DEFAULT = "gts"
LAT, SLOW = "p95_task_latency_s", "p95_slowdown"
MIN_ACTIVE = 8
PER_SEED_KEYS = ("p50_task_latency_s", "p95_task_latency_s", "mean_task_latency_s",
                 "deadline_miss_rate", "p95_slowdown", "mean_slowdown")


def sensitivity(d, policy, base, key):
    act = [c for c in map_cells(d) if not c["inactive"]]
    if not act:
        return None
    n = len(act[0]["policies"][policy]["per_seed"][key])
    logs = [math.log(st.fmean(c["policies"][policy]["per_seed"][key][s]
                              / c["policies"][base]["per_seed"][key][s] for c in act))
            for s in range(n)]
    t = _T975.get(n - 1, 1.96)
    half = t * st.stdev(logs) / n ** 0.5
    ratios = [st.fmean(c["policies"][policy]["per_seed"][key])
              / st.fmean(c["policies"][base]["per_seed"][key]) for c in act]
    return {"t_lo": math.exp(st.fmean(logs) - half), "t_hi": math.exp(st.fmean(logs) + half),
            "geo_mean": math.exp(st.fmean(math.log(r) for r in ratios))}


def test(d, policy, base, key, lim, strict, delta, min_active, reps):
    s = score(d, policy, base, key, margin=delta, reps=reps)
    b = s["bootstrap"]
    sens = sensitivity(d, policy, base, key)
    am = b["active_mean"]
    if b["n_active"] < min_active or am is None:
        return {"verdict": "INCONCLUSIVE", "score": s, "sens": sens}
    ok_bound = (lambda x: x < lim) if strict else (lambda x: x <= lim)
    holds = ok_bound(am["hi"]) and s["beyond_margin"] <= 2
    alt = [ok_bound(sens["t_hi"]) and s["beyond_margin"] <= 2,
           ok_bound(am["hi"] - am["value"] + sens["geo_mean"]) and s["beyond_margin"] <= 2]
    fragile = any(a != holds for a in alt)
    return {"verdict": ("HOLDS" if holds else "FAILS") + (" (FRAGILE)" if fragile else ""),
            "score": s, "sens": sens}


def identical(d, policy, base):
    return all(c["policies"][policy]["per_seed"][k] == c["policies"][base]["per_seed"][k]
               for c in map_cells(d) for k in PER_SEED_KEYS)


def score_e5(sealed3, final, policy=P_DEFAULT, reps=10_000):
    out = {
        "K1": test(sealed3, policy, "program-fcfs", SLOW, 1.00, True, 0.0, MIN_ACTIVE, reps),
        "K2_latency": test(sealed3, policy, "fcfs", LAT, 1.05, False, 0.05, MIN_ACTIVE, reps),
        "K2_slowdown": test(sealed3, policy, "fcfs", SLOW, 1.05, False, 0.05, MIN_ACTIVE, reps),
        "K4": score(sealed3, policy, "program-fcfs", LAT, reps=reps)}
    if identical(final, policy, "program-fcfs"):
        out["K3"] = {"verdict": "HOLDS (identical to program-fcfs)", "identical": True}
    else:
        out["K3"] = dict(test(final, policy, "program-fcfs", LAT, 1.02, False, 0.02, 1, reps),
                         identical=False)
    k2 = [out["K2_latency"]["verdict"], out["K2_slowdown"]["verdict"]]
    out["headline"] = all(v.startswith("HOLDS") for v in
                          [out["K1"]["verdict"], *k2, out["K3"]["verdict"]])
    return out


def _fmt(x):
    return "n/a" if x is None else f"{x['value']:.3f} [{x['lo']:.3f}, {x['hi']:.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sealed3")
    ap.add_argument("final")
    ap.add_argument("--policy", default=P_DEFAULT)
    args = ap.parse_args()
    maps = []
    for p in (args.sealed3, args.final):
        with open(p) as fh:
            maps.append(json.load(fh))
        m = maps[-1]["manifest"]
        print(f"{p}: {m['git_hash'][:7]} dirty {m['git_dirty']} trace {m['params']['trace_set']}")
    r = score_e5(*maps, policy=args.policy)
    for k in ("K1", "K2_latency", "K2_slowdown", "K3"):
        x = r[k]
        line = f"{k:12s} {x['verdict']}"
        if "score" in x:
            s, b = x["score"], x["score"]["bootstrap"]
            sens = x["sens"]
            line += (f"  active {b['n_active']}  active-mean {_fmt(b['active_mean'])}"
                     f"  all {_fmt(b['mean'])}  median {_fmt(b['median'])}"
                     f"  beyond {s['beyond_margin']}  t [{sens['t_lo']:.3f}, {sens['t_hi']:.3f}]"
                     f"  geo {sens['geo_mean']:.3f}  paired-t better {s['better']} worse {s['worse']}")
        print(line)
    k4 = r["K4"]["bootstrap"]
    print(f"K4 latency price vs program-fcfs: active-mean {_fmt(k4['active_mean'])}  "
          f"all {_fmt(k4['mean'])}")
    print(f"HEADLINE (K1 and K2 and K3): {'HOLDS' if r['headline'] else 'DOES NOT HOLD'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
