"""
Re-read a scheduling-map JSON on the metrics p95 hides (OPEN_ISSUES §28 plan).

FINDINGS §30-31 judged every rule on p95 task latency and overall miss rate.
With 54 tasks, p95 is roughly the 3rd-worst task, and oldest-task-first is
near-optimal for it by construction. This asks the same cells what moved
elsewhere, against `program-fcfs`:

  p50     median task latency
  tight   tight-class miss rate
  macro   per-class miss averaged with equal weight per class, so a tight-class
          gain is not simply cancelled by the larger loose class

Paired 95% CIs where the JSON kept per-seed values (maps run after c40d2dd);
point estimates otherwise, labelled as such. Reads JSON only; runs nothing.

    ./venv/bin/python -m experiments.reanalyze_slo results/scheduling_map_slo.json
"""

import argparse
import json
import statistics as st

from experiments.common import paired_ci

RULES = ("lsf", "edf", "hint-priority", "lsf-oracle")
BASE = "program-fcfs"


def _macro(by_class):
    return st.fmean(by_class.values())


def cell_deltas(cell, rule):
    """{metric: (mean delta, ci or None)} for one rule against program-fcfs."""
    r, b = cell["policies"][rule], cell["policies"][BASE]
    out = {}
    ps_r, ps_b = r.get("per_seed", {}), b.get("per_seed", {})
    if "p50_task_latency_s" in ps_r and "p50_task_latency_s" in ps_b:
        ci = paired_ci(ps_r["p50_task_latency_s"], ps_b["p50_task_latency_s"])
        out["p50"] = (ci["mean_diff_s"], ci)
    else:
        out["p50"] = (r["p50_task_latency_s"] - b["p50_task_latency_s"], None)
    mr, mb = ps_r.get("miss_by_class"), ps_b.get("miss_by_class")
    if mr and mb:
        ci = paired_ci(mr["tight"], mb["tight"])
        out["tight"] = (ci["mean_diff_s"], ci)
        n = len(mr["tight"])
        mac_r = [st.fmean(mr[c][i] for c in mr) for i in range(n)]
        mac_b = [st.fmean(mb[c][i] for c in mb) for i in range(n)]
        ci = paired_ci(mac_r, mac_b)
        out["macro"] = (ci["mean_diff_s"], ci)
    elif "miss_by_class" in r:
        out["tight"] = (r["miss_by_class"]["tight"] - b["miss_by_class"]["tight"], None)
        out["macro"] = (_macro(r["miss_by_class"]) - _macro(b["miss_by_class"]), None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("json")
    ap.add_argument("--eps", type=float, default=0.02,
                    help="point-estimate threshold when there is no CI: a rate change "
                         "beyond this, or a p50 change beyond eps x base p50, counts "
                         "(default 0.02)")
    args = ap.parse_args()
    d = json.load(open(args.json))
    print(f"{args.json}  coeff="
          f"{d['manifest']['calibrations'][0]['under'].get('batch_decode_coeff')}  "
          f"git {d['manifest'].get('git_hash', '')[:7]}")

    for rule in RULES:
        tally = {m: [0, 0] for m in ("p50", "tight", "macro")}
        has_ci = False
        lines = []
        for B in d["batches"]:
            for cell in B["cells"]:
                if rule not in cell["policies"]:
                    continue
                dl = cell_deltas(cell, rule)
                base_p50 = cell["policies"][BASE]["p50_task_latency_s"]
                for m, (v, ci) in dl.items():
                    if ci is not None:
                        has_ci = True
                        hit = ci["significant"]
                    else:
                        hit = abs(v) > (args.eps * base_p50 if m == "p50" else args.eps)
                    if hit:
                        tally[m][0 if v < 0 else 1] += 1
                lines.append((B["max_batch"], cell["load"], dl))
        kind = "significant (paired 95% CI)" if has_ci else \
            f"point estimates beyond eps={args.eps}, NO CI"
        print(f"\n{rule} vs {BASE}: cells better / worse, {kind}")
        for m, (better, worse) in tally.items():
            print(f"  {m:6s} {better:3d} / {worse:3d}")
        for batch, load, dl in lines:
            fmt = lambda m, pct: (f"{dl[m][0] * (100 if pct else 1):+7.1f}"
                                  + ("*" if dl[m][1] and dl[m][1]["significant"] else " "))
            print(f"    b{batch:<2d} load {load:4.2f}  p50 {fmt('p50', False)}s  "
                  f"tight {fmt('tight', True)}pts  macro {fmt('macro', True)}pts")


if __name__ == "__main__":
    main()
