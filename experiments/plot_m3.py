"""
Report-quality figures for Milestone 3 (seven PNG + PDF files, plus a printed audit table).

    ./venv/bin/python -m experiments.plot_m3 [--results results/] [--out results/figures/m3] [--only 1,2]

Reads existing result JSONs and the holdout_qwen trace (parse only); it never
simulates and never writes outside --out. Some result JSONs are gitignored, so
a worktree can point --results at the main checkout's results/ (read-only).

Every number drawn is also printed to stdout, so the figures can be audited
against docs/FINDINGS.md. The counting rule used everywhere (FINDINGS sections
33, 40): a rule is BETTER than the baseline in a cell when the paired 95% CI on
(rule - baseline) lies below zero (hi_s < 0), WORSE when it lies above zero
(lo_s > 0), TIED otherwise. A percentage is the paired mean difference over the
baseline's mean p95 in that cell; its CI is the CI on seconds over the same
base, so it shows the same significance as the seconds CI.

LOAD AXES
    "capacity" maps (plas_taskdl, workflow_qcap, gpu_sweep): load = arrival rate
    / saturation throughput at that batch and coefficient.
    "busy" maps (the sealed maps and scheduling_map_slo_cap): load = arrival rate
    / the rate that gives 80% GPU busy at that batch (OPEN_ISSUES 30). They are
    labelled as such; the two axes are not interchangeable.

CAPTIONS (also printed, and written to results/figures/m3/README.md by hand)
"""

import argparse
import json
import math
import os
import statistics

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_RESULTS = os.path.join(_REPO, "results")
DEFAULT_OUT = os.path.join(_REPO, "results", "figures", "m3")

# --- palette ---------------------------------------------------------------
# One colour per policy, used in every figure. Orange is reserved for "worse"
# and blue for "better" (colourblind-safe pair), so no policy uses either
# outcome hue except program-fcfs, which is the blue reference everything is
# measured against.
INK, MUTED, GRID, SURFACE = "#1f1f1e", "#6b6b67", "#e4e3df", "#ffffff"
POLICY = {
    "program-fcfs": "#2a78d6",
    "fcfs": "#6b6b67",
    "sjf": "#6250d6",
    "edf": "#1baf7a",
    "lsf": "#eda100",
    "lsf-oracle": "#008300",
    "hint-priority": "#e87ba4",
    "plas": "#e34948",
    "program-fcfs-cap": "#8f6b3d",
    "program-fcfs-qcap": "#4a2f12",
    # retention policies (figure 4 only)
    "keep-all": "#2f5d62",
    "ski-rental": "#b8860b",
    "evict-now": "#7a1f5c",
}
LABEL = {"fcfs": "FCFS", "sjf": "SJF", "edf": "EDF", "lsf": "LSF", "lsf-oracle": "LSF-oracle",
         "hint-priority": "hint-priority", "plas": "PLAS", "program-fcfs": "program-fcfs",
         "program-fcfs-cap": "program-fcfs-cap", "program-fcfs-qcap": "program-fcfs-qcap",
         "keep-all": "keep-all", "ski-rental": "ski-rental", "evict-now": "evict-now"}
BETTER, TIED, WORSE = "#2f7bd9", "#c9c8c3", "#d9622b"
# Batch size is ordered, so it gets one sequential blue ramp, light to dark.
BATCH_COL = {2: "#6aa6ec", 4: "#2f7bd9", 8: "#1c5aac", 16: "#0f3a73"}
# Diverging, blue (better, lower p95) <-> orange (worse), grey midpoint.
DIVERGING = LinearSegmentedColormap.from_list(
    "better_worse", ["#0f3a73", "#2f7bd9", "#b9d4f3", "#efeeea", "#f4c9a8", "#e07a3a", "#9c3d0b"])

LOAD_CAP = "load (arrival rate / saturation throughput)"
LOAD_BUSY = "load (arrival rate / rate at 80% GPU busy)"

CAPTIONS = {
    "fig1_headline": (
        "Figure 1 (FINDINGS 33). Change in p95 task latency of program-fcfs against request-level FCFS, "
        "by load, on the sealed test sets (20 paired seeds, Mac costs, c = 0.841). The band is the paired "
        "95% CI on the difference in seconds, converted to % of FCFS's mean p95 at the same cell. Filled "
        "markers: CI entirely below zero (better); open markers: CI includes zero (n.s.). No cell is "
        "significantly worse. Load axis is the busy axis of the sealed maps."),
    "fig2_league": (
        "Figure 2 (FINDINGS 33, 33 addendum, 40). Cells in which each rule is significantly better than, "
        "worse than, or tied with program-fcfs on p95 task latency. Top: sealed set, 80 cells per rule "
        "(final_qwen + final_llama, 4 batches x 10 loads x 2 traces). For FCFS this is the reverse of "
        "'program-fcfs vs FCFS 72/0/8'. Bottom, separated: PLAS on the development set holdout_qwen "
        "(40 cells per run; S1 Mac costs, S2 simulated GPU c = 0.3). Cells share traces and seeds, so "
        "counts show consistency, not independent tests."),
    "fig3_sensitivity": (
        "Figure 3 (FINDINGS 36, 37, 38). Sensitivity of the headline to the simulator's batching coefficient "
        "c (decode step time 1 + (n-1)c), holdout_qwen, capacity axis, 20 seeds. Top: p95 change of "
        "program-fcfs vs FCFS at load 1.0 (open marker = n.s.). Bottom: cells where program-fcfs is "
        "significantly better than FCFS, of 40 (worse: 0 at every c). Left: Mac measured rates; right: G1 "
        "RTX 5060 Ti rates. This is a sensitivity analysis inside the simulator's batching law, not GPU "
        "measurements. The Mac sweep has no c = 0.03 run."),
    "fig4_retention": (
        "Figure 4 (FINDINGS 24, 35; fixed engine, DECISIONS 8 item 11). Recompute seconds (top) and p95 task "
        "latency (bottom) for keep-all, ski-rental and evict-now, FCFS scheduler, batch 4, util 0.8. All "
        "numbers are from the engine after the KV-accounting fix (DECISIONS 8 item 11). Left: 1x context, "
        "holdout_qwen (20 seeds, KV 512 MB). Middle: 1x, sealed final_qwen, three KV headrooms (20 seeds); "
        "this column is the owner-approved correction rerun, and the pre-fix numbers are kept in FINDINGS 35. "
        "Right: 8x context, holdout_qwen, three headrooms (10 seeds, simulator-only scale). Bars are seed "
        "means. Note the different y scales per column."),
    "fig5_cap": (
        "Figure 5 (FINDINGS 34, 37, 39). p95 change against program-fcfs (%), batch x load, for the "
        "concurrency cap (program-fcfs-cap) and the self-lifting cap (program-fcfs-qcap), holdout_qwen, "
        "capacity axis, 20 seeds. Left: Mac costs (c = 0.841, R1); right: simulated GPU (c = 0.3, R2). "
        "Blue = lower p95 than program-fcfs, orange = higher; hatched = n.s. Batch 2 equals program-fcfs "
        "by construction (and batch 4 for the GPU qcap). The sealed cap map of FINDINGS 34 uses the busy "
        "axis and has no qcap, so it is not drawn here."),
    "fig6_plas": (
        "Figure 6 (FINDINGS 40). Why PLAS loses. Left: isolated critical-path time of the 54 tasks of "
        "holdout_qwen (Task.critical_path_total from the trace; CV = std/mean). Task sizes vary little, "
        "so attained-service ordering has little to exploit. Right: p95 change of PLAS against "
        "program-fcfs at load 1.0, by batch, Mac costs (S1) and simulated GPU (S2), with the paired 95% CI "
        "as % of program-fcfs p95. Open bar = n.s."),
    "fig7_tracelab": (
        "Figure 7 (FINDINGS 41). TraceLab (real Claude Code/Codex sessions mapped to the simulator) against "
        "MAARA, simulated, costs from public data for a 16 GB RTX 5060 Ti (simulated, not measured), c = 0.3, "
        "capacity load axis = arrival rate / saturation throughput. Caption is completed at run time with the "
        "TraceLab file used."),
}


# --- helpers ---------------------------------------------------------------

def _load(results, rel):
    with open(os.path.join(results, rel)) as fh:
        return json.load(fh)


def _style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=7, length=2.5)
    if grid:
        ax.grid(axis=grid, color=GRID, lw=0.5)
        ax.set_axisbelow(True)


def _save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    paths = []
    for ext, kw in (("png", {"dpi": 200}), ("pdf", {})):
        p = os.path.join(out, f"{name}.{ext}")
        fig.savefig(p, bbox_inches="tight", facecolor=SURFACE, **kw)
        paths.append(p)
    plt.close(fig)
    return paths


def classify(ci):
    """'better' / 'worse' / 'tied' from a paired CI on (rule - baseline)."""
    if ci["hi_s"] < 0:
        return "better"
    if ci["lo_s"] > 0:
        return "worse"
    return "tied"


def cells_of(data):
    """(max_batch, cell) pairs, merged batches skipped."""
    return [(b["max_batch"], c) for b in data["batches"] if not b.get("merged_into")
            for c in sorted(b["cells"], key=lambda c: c["load"])]


def pct(ci, base_s):
    """(mean, lo, hi) of a seconds CI, as % of the baseline's mean p95."""
    return tuple(100.0 * ci[k] / base_s for k in ("mean_diff_s", "lo_s", "hi_s"))


def counts(data, rule, key):
    """better/worse/tied for `rule` using CI field `key` ('p95_vs_fcfs' or 'p95_vs_program')."""
    n = {"better": 0, "worse": 0, "tied": 0}
    flag_mismatch = 0
    for _, c in cells_of(data):
        ci = c["policies"][rule][key]
        k = classify(ci)
        n[k] += 1
        flag_mismatch += (k != "tied") != bool(ci["significant"])
    return n, flag_mismatch


def _pt_chip(ax, x, y, policy, size=7):
    ax.plot([x], [y], "s", ms=size, color=POLICY[policy], transform=ax.transAxes, clip_on=False)


# --- figure 1: headline ----------------------------------------------------

def fig1(results, out):
    sources = [("sealed qwen (final_qwen, 91 tasks)", "sealed/scheduling_map_final_qwen_full.json"),
               ("sealed llama (final_llama, 92 tasks)", "sealed/scheduling_map_final_llama.json")]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharey=True, facecolor=SURFACE)
    print("\n== Figure 1: program-fcfs vs FCFS, p95 change (%) [CI as % of FCFS mean p95] ==")
    for ax, (title, rel) in zip(axes, sources):
        data = _load(results, rel)
        tot = {"better": 0, "worse": 0, "tied": 0}
        print(f"-- {title}  ({rel})")
        print("batch  load   fcfs_p95_s  diff_s [lo_s, hi_s]            pct [lo, hi]            class")
        for mb in (2, 4, 8, 16):
            pts = []
            for b, c in cells_of(data):
                if b != mb:
                    continue
                ci = c["policies"]["program-fcfs"]["p95_vs_fcfs"]
                base = c["policies"]["fcfs"]["p95_task_latency_s"]
                m, lo, hi = pct(ci, base)
                k = classify(ci)
                tot[k] += 1
                pts.append((c["load"], m, lo, hi, k))
                print(f"{mb:>5} {c['load']:5.2f} {base:11.1f}  {ci['mean_diff_s']:+8.1f} "
                      f"[{ci['lo_s']:+8.1f}, {ci['hi_s']:+8.1f}]  {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  {k}")
            pts.sort()
            x = [p[0] for p in pts]
            col = BATCH_COL[mb]
            ax.fill_between(x, [p[2] for p in pts], [p[3] for p in pts], color=col, alpha=0.16, lw=0)
            ax.plot(x, [p[1] for p in pts], "-", color=col, lw=1.4, label=f"batch {mb}")
            sig = [p for p in pts if p[4] == "better"]
            ns = [p for p in pts if p[4] == "tied"]
            ax.plot([p[0] for p in sig], [p[1] for p in sig], "o", ms=4.2, color=col)
            ax.plot([p[0] for p in ns], [p[1] for p in ns], "o", ms=4.2, mfc=SURFACE, mec=col, mew=1.0)
            bad = [p for p in pts if p[4] == "worse"]
            ax.plot([p[0] for p in bad], [p[1] for p in bad], "^", ms=5, color=WORSE)
        print(f"   totals: better {tot['better']}  worse {tot['worse']}  tied {tot['tied']}")
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([0.75, 1, 1.5, 2, 3], ["0.75", "1", "1.5", "2", "3"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_xlabel(LOAD_BUSY, fontsize=7.5, color=INK)
        ax.text(0.98, 0.03, f"{tot['better']} better / {tot['worse']} worse / {tot['tied']} n.s.",
                transform=ax.transAxes, ha="right", fontsize=7, color=MUTED)
        _style(ax)
    axes[0].set_ylabel("p95 task latency vs FCFS (%)\n(negative = faster)", fontsize=7.5, color=INK)
    axes[0].legend(loc="lower left", fontsize=7, frameon=False, title="max batch", title_fontsize=7,
                   handlelength=1.4)
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4.2, label="significant (CI < 0)"),
         plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, ms=4.2, label="n.s.")]
    axes[1].legend(handles=h, loc="lower left", fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out, "fig1_headline")


# --- figure 2: league table ------------------------------------------------

def fig2(results, out):
    q = _load(results, "sealed/scheduling_map_final_qwen_full.json")
    l = _load(results, "sealed/scheduling_map_final_llama.json")
    rules = ["fcfs", "sjf", "edf", "lsf", "lsf-oracle", "hint-priority"]
    expected = {"lsf": (0, 61, 19), "lsf-oracle": (0, 63, 17), "edf": (0, 59, 21),
                "hint-priority": (0, 67, 13), "sjf": (0, 68, 12), "fcfs": (0, 72, 8)}
    print("\n== Figure 2: rule vs program-fcfs, cells better / worse / tied (p95) ==")
    rows, bad = [], []
    for r in rules:
        nq, m1 = counts(q, r, "p95_vs_program")
        nl, m2 = counts(l, r, "p95_vs_program")
        n = {k: nq[k] + nl[k] for k in nq}
        got = (n["better"], n["worse"], n["tied"])
        flag = "" if got == expected[r] else f"  MISMATCH expected {expected[r]}"
        if flag:
            bad.append((r, expected[r], got))
        print(f"sealed  {r:14s} {got[0]:3d} / {got[1]:3d} / {got[2]:3d}   (qwen {nq['better']}/{nq['worse']}/"
              f"{nq['tied']}, llama {nl['better']}/{nl['worse']}/{nl['tied']}; sig-flag mismatches {m1 + m2}){flag}")
        rows.append((r, n, "sealed"))
    for lbl, rel in (("PLAS S1 Mac", "plas_taskdl/S1_mac.json"), ("PLAS S2 GPU c=0.3", "plas_taskdl/S2_gpu_public.json")):
        n, m = counts(_load(results, rel), "plas", "p95_vs_program")
        print(f"dev     {lbl:18s} {n['better']:3d} / {n['worse']:3d} / {n['tied']:3d}   ({rel}; sig-flag mismatches {m})")
        if n["better"] != 0:
            bad.append((lbl, "0 better", n))
        rows.append((lbl, n, "dev"))
    if bad:
        print("!! DISCREPANCY vs FINDINGS:", bad)

    sealed = [r for r in rows if r[2] == "sealed"]
    sealed.sort(key=lambda r: r[1]["worse"])
    dev = [r for r in rows if r[2] == "dev"]
    fig, ax = plt.subplots(figsize=(6.8, 3.4), facecolor=SURFACE)
    ypos, y = [], 0
    ordered = sealed + [None] + dev
    labels = []
    for item in ordered:
        if item is None:
            y += 0.7
            continue
        name, n, grp = item
        total = sum(n.values())
        left = 0
        for k, col in (("better", BETTER), ("tied", TIED), ("worse", WORSE)):
            w = n[k]
            if w:
                ax.barh(y, w, left=left, color=col, height=0.62, edgecolor=SURFACE, lw=1.2)
                ax.text(left + w / 2, y, str(w), ha="center", va="center", fontsize=7,
                        color="white" if k != "tied" else INK)
            left += w
        key = "plas" if name.startswith("PLAS") else name
        labels.append((y, f"{name if grp == 'dev' else LABEL[name]}  ({n['better']} better)", key, total))
        ypos.append(y)
        y += 1
    ax.set_xlim(0, 80)
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_yticks([p[0] for p in labels], [p[1] for p in labels])
    for t, (_, _, key, _) in zip(ax.get_yticklabels(), labels):
        t.set_color(INK)
    for yy, _, key, tot in labels:  # policy colour chip left of the label
        ax.plot([-0.02], [yy], "s", ms=6.5, color=POLICY[key], transform=ax.get_yaxis_transform(),
                clip_on=False)
    ax.tick_params(axis="y", pad=14)
    sep = sealed[-1][0] and (len(sealed) - 0.5 + 0.35)
    ax.axhline(sep, color=MUTED, lw=0.7, ls=(0, (3, 3)))
    ax.text(79.5, len(sealed) + 0.22, "development set, holdout_qwen (40 cells per run, shown on the same 80 axis)",
            ha="right", va="center", fontsize=6.5, color=MUTED)
    ax.text(79.5, -0.5, "sealed set, 80 cells per rule", ha="right", va="center", fontsize=6.5, color=MUTED)
    ax.set_xlabel("cells (p95 task latency, paired 95% CI over 20 seeds)", fontsize=7.5, color=INK)
    _style(ax, grid=None)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(handles=[Patch(color=BETTER, label="rule better than program-fcfs"),
                       Patch(color=TIED, label="tied (n.s.)"),
                       Patch(color=WORSE, label="rule worse than program-fcfs")],
              loc="upper center", bbox_to_anchor=(0.45, -0.17), ncol=3, fontsize=7, frameon=False)
    fig.tight_layout()
    return _save(fig, out, "fig2_league"), bad


# --- figure 3: sensitivity -------------------------------------------------

def fig3(results, out):
    coeffs = ["0.03", "0.15", "0.3", "0.5", "0.841"]
    srcs = [("Mac measured rates", "gpu_sweep/mac_ollama_measured"),
            ("G1 RTX 5060 Ti rates (simulated)", "gpu_sweep/gpu_5060ti_g1rates")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), sharex=True, facecolor=SURFACE,
                             gridspec_kw={"height_ratios": [1.5, 1]})
    print("\n== Figure 3: program-fcfs vs FCFS across batching coefficient c ==")
    print("source          c      batch  pct@load1.0 [lo, hi]   class   | cells better/worse/tied (of 40)")
    for j, (title, d) in enumerate(srcs):
        cnt = {}
        by = {mb: [] for mb in (2, 4, 8, 16)}
        for c_ in coeffs:
            p = os.path.join(results, d, f"map_c{c_}.json")
            if not os.path.exists(p):
                print(f"{title}: map_c{c_}.json absent -> point omitted")
                continue
            data = _load(results, f"{d}/map_c{c_}.json")
            n, _ = counts(data, "program-fcfs", "p95_vs_fcfs")
            cnt[float(c_)] = n
            for mb, c in cells_of(data):
                if abs(c["load"] - 1.0) < 1e-9:
                    ci = c["policies"]["program-fcfs"]["p95_vs_fcfs"]
                    m, lo, hi = pct(ci, c["policies"]["fcfs"]["p95_task_latency_s"])
                    by[mb].append((float(c_), m, lo, hi, classify(ci)))
                    print(f"{d.split('/')[-1][:14]:15s} {c_:6s} {mb:5d}  {m:+7.1f} [{lo:+6.1f}, {hi:+6.1f}]  "
                          f"{classify(ci):6s} | {n['better']}/{n['worse']}/{n['tied']}")
        ax = axes[0, j]
        for mb in (2, 4, 8, 16):
            pts = sorted(by[mb])
            col = BATCH_COL[mb]
            x = [p[0] for p in pts]
            ax.plot(x, [p[1] for p in pts], "-", color=col, lw=1.4, label=f"batch {mb}")
            sg = [p for p in pts if p[4] == "better"]
            ns = [p for p in pts if p[4] != "better"]
            ax.plot([p[0] for p in sg], [p[1] for p in sg], "o", ms=4.2, color=col)
            ax.plot([p[0] for p in ns], [p[1] for p in ns], "o", ms=4.2, mfc=SURFACE, mec=col, mew=1.0)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_title(title, fontsize=8, color=INK, loc="left")
        ax.set_ylim(-38, 4)
        _style(ax)
        b = axes[1, j]
        xs = sorted(cnt)
        b.plot(xs, [cnt[x]["better"] for x in xs], "-o", color=POLICY["program-fcfs"], lw=1.4, ms=4.2)
        b.set_ylim(0, 40)
        b.text(0.02, 0.06, "worse cells: 0 at every c", transform=b.transAxes, fontsize=7, color=MUTED)
        for x in xs:
            b.annotate(str(cnt[x]["better"]), (x, cnt[x]["better"]), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=6.5, color=INK)
        _style(b)
        b.set_xscale("log")
        b.set_xticks([float(c) for c in coeffs], coeffs)
        b.minorticks_off()
        b.set_xlabel("batching coefficient c (decode step = 1 + (n-1)c)", fontsize=7.5, color=INK)
    axes[0, 0].set_ylabel("p95 change at load 1.0 (%)\n(negative = faster)", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("cells where program-fcfs\nis better (of 40)", fontsize=7.5, color=INK)
    axes[0, 0].legend(loc="center left", fontsize=7, frameon=False, ncol=1, title="max batch",
                      title_fontsize=7)
    axes[0, 0].text(0.0, 0.0, "", transform=axes[0, 0].transAxes)
    axes[0, 0].annotate("no c = 0.03 run on Mac", (0.03, -34), fontsize=6.5, color=MUTED, ha="left")
    h = [plt.Line2D([], [], marker="o", ls="", color=MUTED, ms=4.2, label="significant"),
         plt.Line2D([], [], marker="o", ls="", mfc=SURFACE, mec=MUTED, ms=4.2, label="n.s.")]
    axes[0, 1].legend(handles=h, loc="lower right", fontsize=7, frameon=False)
    fig.suptitle("sensitivity inside the simulator's batching law, not GPU measurements",
                 fontsize=7.5, color=MUTED, x=0.01, ha="left", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return _save(fig, out, "fig3_sensitivity")


# --- figure 4: retention ---------------------------------------------------

T975 = {9: 2.262, 19: 2.093}


def _paired(a, b):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    m = statistics.mean(d)
    h = T975[n - 1] * statistics.stdev(d) / math.sqrt(n)
    return m, m - h, m + h


def fig4(results, out):
    pols = ["keep-all", "ski-rental", "evict-now"]
    cols = []  # (title, [(group label, {pol: (recompute, p95, per_seed_rec, per_seed_p95)})])
    f24 = _load(results, "bugfix_check/finding24.json")
    row = next(r for r in f24["rows"] if abs(r["batch_decode_coeff"] - 0.841) < 1e-9)
    ret = row["retention"]
    g = {p: (ret["by_policy"][p]["recompute_seconds"], ret["by_policy"][p]["p95_task_latency_s"],
             ret["per_seed"][p]["recompute_seconds"], ret["per_seed"][p]["p95_task_latency_s"]) for p in pols}
    cols.append(("1x holdout\n512 MB (§24, fixed)", [("", g)]))
    for title, rel in (("1x sealed final_qwen (§35, fixed)", "sealed/retention_headroom_final_qwen_fixed.json"),
                       ("8x holdout_qwen (§35, fixed)", "bugfix_check/retention_headroom_8x.json")):
        d = _load(results, rel)
        grp = []
        for c in d["cells"]:
            pp = c["policies"]
            grp.append((f"{c['kv_budget_mb']:.0f} MB", {p: (pp[p]["recompute_seconds"], pp[p]["p95_task_latency_s"],
                                                             pp[p]["per_seed"]["recompute_seconds"],
                                                             pp[p]["per_seed"]["p95_task_latency_s"]) for p in pols}))
        cols.append((title, grp))
    print("\n== Figure 4: retention (seed means; paired 95% CI vs keep-all, seconds) ==")
    print("column                      KV        policy       recompute_s  vs keep-all [CI]            p95_s   vs keep-all [CI]")
    fig, axes = plt.subplots(2, 3, figsize=(6.8, 4.6), facecolor=SURFACE,
                             gridspec_kw={"width_ratios": [1, 3, 3]})
    for j, (title, groups) in enumerate(cols):
        for i, metric in enumerate(("recompute", "p95")):
            ax = axes[i, j]
            for gi, (glabel, vals) in enumerate(groups):
                for pi, p in enumerate(pols):
                    x = gi + (pi - 1) * 0.27
                    v = vals[p][i]
                    ax.bar(x, v, width=0.25, color=POLICY[p])
                    ax.text(x, v, f" {v:.0f}", ha="center", va="bottom", fontsize=5.8, color=INK,
                            rotation=90)
                if i == 0:
                    ka, ks, ke = (vals[p] for p in pols)
                    for p in pols:
                        rc = _paired(vals[p][2], vals["keep-all"][2]) if p != "keep-all" else (0, 0, 0)
                        pc = _paired(vals[p][3], vals["keep-all"][3]) if p != "keep-all" else (0, 0, 0)
                        print(f"{title.splitlines()[0][:27]:27s} {glabel:>8s}  {p:11s} {vals[p][0]:10.1f}  "
                              f"{rc[0]:+7.1f} [{rc[1]:+7.1f}, {rc[2]:+7.1f}]   {vals[p][1]:8.1f}  "
                              f"{pc[0]:+7.1f} [{pc[1]:+7.1f}, {pc[2]:+7.1f}]")
            ax.set_xticks(range(len(groups)), [gl.replace(" MB", "") for gl, _ in groups])
            ax.set_xlim(-0.6, len(groups) - 0.4)
            ax.margins(y=0.2)
            _style(ax)
            if i == 0:
                ax.set_title(title, fontsize=7.5, color=INK, loc="left")
                ax.tick_params(axis="x", labelbottom=False)
            else:
                ax.set_xlabel("KV budget (MB)" if j else "", fontsize=7, color=INK)
    axes[0, 0].set_ylabel("recompute (s)", fontsize=7.5, color=INK)
    axes[1, 0].set_ylabel("p95 task latency (s)", fontsize=7.5, color=INK)
    axes[1, 1].legend(handles=[Patch(color=POLICY[p], label=LABEL[p]) for p in pols], loc="upper center",
                      bbox_to_anchor=(0.1, -0.28), ncol=3, fontsize=7, frameon=False)
    # tight_layout lets the long column titles inflate the gaps, so set them by hand
    fig.subplots_adjust(left=0.09, right=0.99, top=0.9, bottom=0.17, wspace=0.28, hspace=0.12)
    return _save(fig, out, "fig4_retention")


# --- figure 5: cap / qcap heatmaps ----------------------------------------

def fig5(results, out):
    runs = [("Mac costs, c = 0.841 (R1)", "workflow_qcap/R1_mac.json"),
            ("simulated GPU, c = 0.3 (R2)", "workflow_qcap/R2_gpu_public.json")]
    rules = ["program-fcfs-cap", "program-fcfs-qcap"]
    grids = {}
    print("\n== Figure 5: cap / qcap p95 change vs program-fcfs (%) ==")
    for title, rel in runs:
        data = _load(results, rel)
        loads = [c["load"] for _, c in cells_of(data) if _ == 2]
        for r in rules:
            g = np.full((4, len(loads)), np.nan)
            sg = np.zeros((4, len(loads)), bool)
            n = {"better": 0, "worse": 0, "tied": 0}
            for mb, c in cells_of(data):
                i, jx = (2, 4, 8, 16).index(mb), min(range(len(loads)), key=lambda q: abs(loads[q] - c["load"]))
                ci = c["policies"][r]["p95_vs_program"]
                base = c["policies"]["program-fcfs"]["p95_task_latency_s"]
                g[i, jx] = 100 * ci["mean_diff_s"] / base
                k = classify(ci)
                sg[i, jx] = k != "tied"
                n[k] += 1
            grids[(rel, r)] = (g, sg, loads)
            print(f"-- {r} {title} ({rel}): {n['better']} better / {n['worse']} worse / {n['tied']} tied")
            print("batch\\load " + " ".join(f"{x:6.2f}" for x in loads))
            for i, mb in enumerate((2, 4, 8, 16)):
                print(f"{mb:>9} " + " ".join(f"{g[i, jx]:+6.1f}" + ("*" if sg[i, jx] else " ")
                                              for jx in range(len(loads))))
    vmax = 15.0
    allv = np.concatenate([g.ravel() for g, _, _ in grids.values()])
    print(f"colour range +-{vmax}; data range {np.nanmin(allv):+.1f} .. {np.nanmax(allv):+.1f} (* = significant)")
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.3), facecolor=SURFACE,
                             gridspec_kw={"hspace": 0.55})
    im = None
    for j, (title, rel) in enumerate(runs):
        for i, r in enumerate(rules):
            ax = axes[i, j]
            g, sg, loads = grids[(rel, r)]
            im = ax.imshow(np.clip(g, -vmax, vmax), cmap=DIVERGING, norm=norm, aspect="auto")
            for bi in range(4):
                for lj in range(len(loads)):
                    if not sg[bi, lj]:
                        ax.add_patch(plt.Rectangle((lj - .5, bi - .5), 1, 1, fill=False, hatch="////",
                                                   edgecolor="#9a9994", lw=0))
                    v = g[bi, lj]
                    txt = f"{v:+.0f}"
                    txt = "0" if txt in ("+0", "-0") else txt
                    ax.text(lj, bi, txt, ha="center", va="center", fontsize=5.8,
                            color="white" if abs(v) > 0.55 * vmax else INK,
                            fontweight="bold" if sg[bi, lj] else "normal")
            ax.set_yticks(range(4), ["2", "4", "8", "16"])
            ax.set_xticks(range(len(loads)), [f"{x:g}" for x in loads])
            ax.tick_params(labelsize=7, length=0, colors=MUTED, labelcolor=INK)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.set_title(f"{LABEL[r]}\n{title}", fontsize=7.5, color=INK, loc="left", pad=5)
            ax.plot([-0.03], [1.19], "s", ms=6, color=POLICY[r], transform=ax.transAxes, clip_on=False)
            if j == 0:
                ax.set_ylabel("max batch", fontsize=7.5, color=INK)
            if i == 1:
                ax.set_xlabel(LOAD_CAP, fontsize=7.5, color=INK)
            else:
                ax.tick_params(labelbottom=False)
    cb = fig.colorbar(im, ax=axes, shrink=0.8, pad=0.02, extend="both")
    cb.set_label("p95 vs program-fcfs (%)\nblue = faster, orange = slower", fontsize=7, color=INK)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    fig.text(0.01, -0.02, "hatched = n.s. (CI includes zero); bold = significant; colour range +-15%, nothing clipped",
             fontsize=6.5, color=MUTED)
    return _save(fig, out, "fig5_cap")


# --- figure 6: why PLAS loses ---------------------------------------------

def critical_paths():
    """Task.critical_path_total for every holdout_qwen task (parse only)."""
    from simulator import trace_loader as tl
    vals = []
    for p in tl.trace_set("holdout_qwen"):
        vals += [t.critical_path_total for t in tl.load_trace(p, arrival_rate=0.01).tasks]
    return vals


def fig6(results, out):
    v = critical_paths()
    mn, md, mx = min(v), statistics.median(v), max(v)
    cv = statistics.stdev(v) / statistics.mean(v)
    print("\n== Figure 6: holdout_qwen task critical path (Task.critical_path_total) ==")
    print(f"n {len(v)}  min {mn:.1f}  median {md:.1f}  max {mx:.1f}  mean {statistics.mean(v):.1f}  CV {cv:.2f}")
    ok = (len(v), round(mn, 1), round(md, 1), round(mx, 1), round(cv, 2)) == (54, 5.6, 24.9, 88.8, 0.61)
    print("reproduces FINDINGS 40 (54 / 5.6 / 24.9 / 88.8 / CV 0.61):", "YES" if ok else "NO  <-- MISMATCH")
    fig, (a, b) = plt.subplots(1, 2, figsize=(6.8, 3.0), facecolor=SURFACE,
                               gridspec_kw={"width_ratios": [1.1, 1]})
    a.hist(v, bins=np.arange(0, 95, 5), color="#9ba0a6", edgecolor=SURFACE, lw=0.8)
    a.axvline(md, color=INK, lw=1, ls="--")
    a.text(md + 1.5, a.get_ylim()[1] * 0.93, f"median {md:.1f} s", fontsize=7, color=INK)
    a.text(0.97, 0.72, f"54 tasks\nmin {mn:.1f} s, max {mx:.1f} s\nCV {cv:.2f}", transform=a.transAxes,
           ha="right", va="top", fontsize=7.5, color=INK)
    a.set_xlabel("task critical-path time (s)", fontsize=7.5, color=INK)
    a.set_ylabel("tasks (holdout_qwen)", fontsize=7.5, color=INK)
    a.set_title("task sizes vary little (CV well below 1)", fontsize=8, color=INK, loc="left")
    _style(a)
    print("PLAS vs program-fcfs, p95 change at load 1.0 (CI as % of program-fcfs p95)")
    runs = [("Mac costs (S1)", "plas_taskdl/S1_mac.json", 0.0), ("simulated GPU c=0.3 (S2)", "plas_taskdl/S2_gpu_public.json", 1.0)]
    w = 0.36
    for k, (lbl, rel, _) in enumerate(runs):
        data = _load(results, rel)
        for bi, mb in enumerate((2, 4, 8, 16)):
            c = next(c for b_, c in cells_of(data) if b_ == mb and abs(c["load"] - 1.0) < 1e-9)
            ci = c["policies"]["plas"]["p95_vs_program"]
            m, lo, hi = pct(ci, c["policies"]["program-fcfs"]["p95_task_latency_s"])
            sig = classify(ci) == "worse"
            x = bi + (k - 0.5) * w
            b.bar(x, m, width=w * 0.9, color=POLICY["plas"] if sig else SURFACE, edgecolor=POLICY["plas"],
                  lw=1, alpha=1.0 if k == 0 else 0.55, hatch=None if k == 0 else "///")
            b.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=INK, lw=0.8, capsize=2)
            b.text(x, hi + 3, f"{m:+.0f}", ha="center", fontsize=6.5, color=INK)
            print(f"  {lbl:24s} batch {mb:2d}: {m:+6.1f}% [{lo:+6.1f}, {hi:+6.1f}]  {classify(ci)}   "
                  f"(diff {ci['mean_diff_s']:+.1f} s)")
    b.axhline(0, color=MUTED, lw=0.8)
    b.set_xticks(range(4), ["2", "4", "8", "16"])
    b.set_xlabel("max batch (load 1.0)", fontsize=7.5, color=INK)
    b.set_ylabel("PLAS p95 vs program-fcfs (%)\n(positive = slower)", fontsize=7.5, color=INK)
    b.set_title("PLAS is slower at every batch", fontsize=8, color=INK, loc="left")
    b.set_ylim(top=165)
    b.legend(handles=[Patch(facecolor=POLICY["plas"], edgecolor=POLICY["plas"], label="Mac costs (S1)"),
                      Patch(facecolor=SURFACE, edgecolor=POLICY["plas"], hatch="///", label="simulated GPU (S2)")],
             loc="upper right", fontsize=7, frameon=False)
    _style(b)
    fig.tight_layout()
    return _save(fig, out, "fig6_plas"), ok


# --- figure 7: TraceLab vs MAARA -------------------------------------------

TRACELAB_E1 = "tracelab/E1_sealed_c0.3.json"
TRACELAB_D1 = "tracelab/D1_dev_c0.3.json"
MAARA_S2 = "plas_taskdl/S2_gpu_public.json"
_FIG7_CAPTION = (
    "Figure 7 (FINDINGS 41). TraceLab (real Claude Code/Codex sessions, one user turn = a chain of LLM rounds "
    "and tool nodes) against MAARA (holdout_qwen, 20 seeds, plas_taskdl/S2_gpu_public.json). MAARA runs at "
    "1,024 MB KV with estimate-basis deadlines, TraceLab at 8,192 MB with task-basis deadlines; the deadline "
    "basis affects LSF and EDF only. Both are "
    "simulated with costs from public data for a 16 GB RTX 5060 Ti (simulated, not measured), c = 0.3, "
    "capacity load axis = arrival rate / saturation throughput. TraceLab file: {rel} ({kind}, {n} seeds). "
    "(a) p95 change of program-fcfs vs FCFS (filled = CI below zero, open = n.s.); where batch 16 is "
    "seed-for-seed identical to batch 8 (KV budget caps concurrency) it is drawn but labelled as such. "
    "(b) Cells where PLAS, LSF, EDF are better/worse/tied vs program-fcfs, paired 95% CI; latency from "
    "stored CIs, slowdown (TraceLab only) from per-seed paired CIs; identical batch-16 cells are excluded "
    "from the counts. (c) Task size: MAARA holdout_qwen critical path vs TraceLab development-set "
    "(tracelab_dev) lone-replay time, always the development set whichever file feeds (a) and (b); "
    "CV = population std / mean; log x-axis.")
RULES7 = ("plas", "lsf", "edf")


def _dup_batches(data):
    """{batch: earlier batch it duplicates} when every per-seed p95 of every policy is identical."""
    dup = {}
    for b in data["batches"]:
        for a in data["batches"]:
            if a["max_batch"] >= b["max_batch"] or a["max_batch"] in dup:
                continue
            same = len(a["cells"]) == len(b["cells"]) and all(
                ca["load"] == cb["load"] and all(
                    ca["policies"].get(k, {}).get("per_seed", {}).get("p95_task_latency_s")
                    == v["per_seed"]["p95_task_latency_s"] for k, v in cb["policies"].items())
                for ca, cb in zip(sorted(a["cells"], key=lambda c: c["load"]),
                                  sorted(b["cells"], key=lambda c: c["load"])))
            if same:
                dup[b["max_batch"]] = a["max_batch"]
                break
    return dup


def _tracelab_iso():
    """Lone-replay (isolated service) seconds of every tracelab_dev task; engine call, development data only."""
    from simulator.engine import Engine, EngineConfig
    from simulator.workload import CostModel
    from simulator.workload_sources import resolve_workload
    cost = CostModel.from_profile("gpu_5060ti_public")
    tasks = resolve_workload("tracelab_dev", 1, 0.01)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=8192.0, max_batch=4, prefill_chunk_tokens=512))
    return [eng._isolated_service_s(t) for t in tasks]


def fig7(results, out, tracelab=None):
    from experiments.common import paired_ci
    rel = tracelab or (TRACELAB_E1 if os.path.exists(os.path.join(results, TRACELAB_E1)) else TRACELAB_D1)
    tl, ma = _load(results, rel), _load(results, MAARA_S2)
    prm = tl["manifest"]["params"]
    kind = "SEALED, scored" if prm["trace_set"] == "tracelab_sealed" else "DEVELOPMENT, not the scored result"
    nseed = len(prm["seeds"])
    print(f"\n== Figure 7: TraceLab vs MAARA ==\nTraceLab file {rel}: {kind}; trace_set {prm['trace_set']}; "
          f"{nseed} seeds; batches {[b['max_batch'] for b in tl['batches']]}")
    print(f"MAARA file {MAARA_S2}: {len(ma['manifest']['params']['seeds'])} seeds")
    runs = [(f"MAARA holdout_qwen ({len(ma['manifest']['params']['seeds'])} seeds)", ma),
            (f"TraceLab {'sealed' if prm['trace_set'] == 'tracelab_sealed' else 'dev'} ({nseed} seeds)", tl)]
    dups = {id(d): _dup_batches(d) for _, d in runs}
    fig = plt.figure(figsize=(6.8, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.5, wspace=0.28)
    # (a)
    axes = [fig.add_subplot(gs[0, 0])]
    axes.append(fig.add_subplot(gs[0, 1], sharey=axes[0]))
    print("(a) program-fcfs vs FCFS, p95 change (% of FCFS mean p95)")
    print("workload   batch  load   pct [lo, hi]            class")
    for ax, (title, d) in zip(axes, runs):
        for b in d["batches"]:
            mb = b["max_batch"]
            pts = []
            for c in sorted(b["cells"], key=lambda c: c["load"]):
                pol = c["policies"]
                if "program-fcfs" not in pol or "fcfs" not in pol:
                    continue
                ci = pol["program-fcfs"]["p95_vs_fcfs"]
                m, lo, hi = pct(ci, pol["fcfs"]["p95_task_latency_s"])
                k = classify(ci)
                pts.append((c["load"], m, lo, hi, k))
                print(f"{title.split()[0]:10s} {mb:5d} {c['load']:5.2f}  {m:+6.1f} [{lo:+6.1f}, {hi:+6.1f}]  {k}"
                      + ("  (duplicates batch %d)" % dups[id(d)][mb] if mb in dups[id(d)] else ""))
            if not pts:
                continue
            col = BATCH_COL[mb]
            same = dups[id(d)].get(mb)
            lbl = f"batch {mb} (\u2261 batch {same})" if same else f"batch {mb}"
            x = [p[0] for p in pts]
            ax.plot(x, [p[1] for p in pts], "--" if same else "-", color=col, lw=1.4, label=lbl,
                    zorder=1 if same else 2)
            sig = [p for p in pts if p[4] == "better"]
            ns = [p for p in pts if p[4] == "tied"]
            bad = [p for p in pts if p[4] == "worse"]
            ax.plot([p[0] for p in sig], [p[1] for p in sig], "o", ms=4.2, color=col, zorder=3)
            ax.plot([p[0] for p in ns], [p[1] for p in ns], "o", ms=4.2, mfc=SURFACE, mec=col, mew=1.0, zorder=3)
            ax.plot([p[0] for p in bad], [p[1] for p in bad], "^", ms=5, color=WORSE, zorder=3)
        ax.axhline(0, color=MUTED, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([0.5, 0.75, 1, 1.5, 2], ["0.5", "0.75", "1", "1.5", "2"])
        ax.minorticks_off()
        ax.set_title("(a) " + title if ax is axes[0] else title, fontsize=7.5, color=INK, loc="left")
        ax.set_xlabel(LOAD_CAP, fontsize=7, color=INK)
        _style(ax)
        ax.legend(loc="lower left", fontsize=6.3, frameon=False, handlelength=1.6)
    axes[0].set_ylabel("p95 vs FCFS (%)\n(negative = faster)", fontsize=7.5, color=INK)
    plt.setp(axes[1].get_yticklabels(), visible=False)
    axes[1].plot([], [], "o", ms=4.2, mfc=SURFACE, mec=MUTED, label="open = n.s.")
    axes[1].plot([], [], "o", ms=4.2, color=MUTED, label="filled = CI < 0")
    h, lab = axes[1].get_legend_handles_labels()
    axes[1].legend(h, lab, loc="lower left", fontsize=6.3, frameon=False, handlelength=1.6)

    # (b)
    print("(b) vs program-fcfs, cells better / worse / tied (batch-16 duplicates excluded)")
    rows = []  # (rule, label, counts, n_excluded)
    for rule in RULES7:
        for lbl, d, metric in (("MAARA latency", ma, "p95_task_latency_s"), ("TraceLab latency", tl, "p95_task_latency_s"),
                               ("TraceLab slowdown", tl, "p95_slowdown")):
            n = {"better": 0, "worse": 0, "tied": 0}
            excl = 0
            for b in d["batches"]:
                for c in b["cells"]:
                    pol = c["policies"]
                    if rule not in pol or "program-fcfs" not in pol:
                        continue
                    if b["max_batch"] in dups[id(d)]:
                        excl += 1
                        continue
                    if metric == "p95_task_latency_s":
                        ci = pol[rule]["p95_vs_program"]
                    else:
                        if "p95_slowdown" not in pol[rule]["per_seed"]:
                            continue
                        ci = paired_ci(pol[rule]["per_seed"]["p95_slowdown"], pol["program-fcfs"]["per_seed"]["p95_slowdown"])
                    n[classify(ci)] += 1
            if sum(n.values()):
                rows.append((rule, lbl, n))
                print(f"  {LABEL[rule]:5s} {lbl:18s} {n['better']:3d} / {n['worse']:3d} / {n['tied']:3d}  "
                      f"(of {sum(n.values())} cells; {excl} duplicate cells excluded)")
    ax = fig.add_subplot(gs[1, 0])
    y, ticks, last = 0.0, [], None
    for rule, lbl, n in rows:
        if last is not None and rule != last:
            y += 0.6
        last = rule
        left = 0
        for k, col in (("better", BETTER), ("tied", TIED), ("worse", WORSE)):
            if n[k]:
                ax.barh(y, n[k], left=left, color=col, height=0.7, edgecolor=SURFACE, lw=1.0)
                if n[k] >= 3:
                    ax.text(left + n[k] / 2, y, str(n[k]), ha="center", va="center", fontsize=6.5,
                            color="white" if k != "tied" else INK)
            left += n[k]
        ax.text(left + 0.8, y, f"{n['better']} / {n['worse']} / {n['tied']}", va="center", fontsize=6.3,
                color=MUTED)
        ticks.append((y, f"{LABEL[rule]} {lbl.replace('latency', 'p95')}", rule))
        y += 1
    ax.set_ylim(y - 0.4, -0.6)
    ax.set_xlim(0, 50)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    ax.tick_params(axis="y", labelsize=6.3, length=0)
    ax.set_xlabel("cells (paired 95% CI); right of bar: better / worse / tied", fontsize=7, color=INK)
    ax.set_title("(b) rule vs program-fcfs", fontsize=7.5, color=INK, loc="left")
    _style(ax, grid=None)
    ax.spines["left"].set_visible(False)
    ax.legend(handles=[Patch(color=BETTER, label="better"), Patch(color=TIED, label="tied"),
                       Patch(color=WORSE, label="worse")], loc="upper center", bbox_to_anchor=(0.5, -0.2),
              ncol=3, fontsize=6.5, frameon=False)

    # (c)
    maara = critical_paths()
    tlv = _tracelab_iso()
    print("(c) task size, seconds (CV = population std / mean)")
    print("workload                         n    mean  median     min     max   CV(pop)  CV(sample)")
    stats = []
    for lbl, v in (("MAARA holdout_qwen critical path", maara), ("TraceLab dev lone-replay", tlv)):
        cv = statistics.pstdev(v) / statistics.mean(v)
        stats.append((lbl, v, cv))
        print(f"{lbl:32s} {len(v):4d} {statistics.mean(v):7.1f} {statistics.median(v):7.1f} {min(v):7.1f} "
              f"{max(v):7.1f}   {cv:6.2f}      {statistics.stdev(v) / statistics.mean(v):6.2f}")
    ax = fig.add_subplot(gs[1, 1])
    lo, hi = min(min(maara), min(tlv)), max(max(maara), max(tlv))
    bins = np.geomspace(lo * 0.98, hi * 1.02, 22)
    for (lbl, v, cv), col, nm in zip(stats, ("#9ba0a6", "#2f5d62"), ("MAARA", "TraceLab dev")):
        w = np.full(len(v), 100.0 / len(v))
        ax.hist(v, bins=bins, weights=w, color=col, alpha=0.62, edgecolor=SURFACE, lw=0.6,
                label=f"{nm}: n {len(v)}, median {statistics.median(v):.0f} s, CV {cv:.2f}")
        ax.axvline(statistics.median(v), ymax=0.62, color=col, lw=1, ls="--")
    ax.set_xscale("log")
    ax.set_xlabel("task size: critical path / lone-replay time (s, log)", fontsize=7, color=INK)
    ax.set_ylabel("share of tasks (%)", fontsize=7.5, color=INK)
    ax.set_title("(c) task-size distribution", fontsize=7.5, color=INK, loc="left")
    ax.legend(loc="upper right", fontsize=6.3, frameon=False)
    _style(ax)
    ax.set_ylim(0, 1.6 * ax.get_ylim()[1])
    CAPTIONS[NAMES[7]] = _FIG7_CAPTION.format(rel=rel, kind=kind.split(",")[0].lower(), n=nseed)
    fig.text(0.01, -0.03, f"simulated, public-data costs for a 16 GB RTX 5060 Ti (not measured); c = 0.3; capacity load axis; "
             f"TraceLab: {rel} ({kind.split(',')[0].lower()}, {nseed} seeds)", fontsize=6, color=MUTED)
    return _save(fig, out, "fig7_tracelab")


FIGS = {1: fig1, 2: fig2, 3: fig3, 4: fig4, 5: fig5, 6: fig6, 7: fig7}
NAMES = {1: "fig1_headline", 2: "fig2_league", 3: "fig3_sensitivity", 4: "fig4_retention",
         5: "fig5_cap", 6: "fig6_plas", 7: "fig7_tracelab"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--results", default=DEFAULT_RESULTS, help="directory holding the result JSONs (read-only)")
    ap.add_argument("--out", default=DEFAULT_OUT, help="where PNG/PDF are written")
    ap.add_argument("--only", default="", help="comma list of figure numbers, e.g. 1,3")
    ap.add_argument("--tracelab", default=None, help="figure 7 TraceLab map, relative to --results "
                    f"(default {TRACELAB_E1} if it exists, else {TRACELAB_D1})")
    args = ap.parse_args()
    which = [int(x) for x in args.only.split(",") if x] or sorted(FIGS)
    for n in which:
        res = FIGS[n](args.results, args.out, args.tracelab) if n == 7 else FIGS[n](args.results, args.out)
        paths = res[0] if isinstance(res, tuple) else res
        print("\nCAPTION:", CAPTIONS[NAMES[n]])
        print("wrote", *paths, sep="\n  ")


if __name__ == "__main__":
    main()
