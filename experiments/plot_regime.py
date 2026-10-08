"""
Heatmaps and markdown tables for the regime maps.

    ./venv/bin/python -m experiments.plot_regime scheduling [results/scheduling_map.json]

Reads a map's JSON (it never reruns anything) and writes PNGs to
results/figures/regime/ plus a markdown table beside the JSON.

HOW TO READ A SCHEDULING HEATMAP
    One panel per batch size, columns = load (arrival rate / rate at util 0.8),
    rows = policy. Colour is the paired mean difference against FCFS over the
    same seeds: blue = the policy is better (lower p95 or fewer misses), red =
    worse, grey = no difference. Hatched cells are not significant (the 95% CI
    includes zero). A hollow dot marks an INACTIVE cell (FCFS had a choice on
    under 5% of decisions, so no rule had room to act). The dashed vertical
    line is the first load at which LSF separates from FCFS.
"""

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG_DIR = os.path.join(_REPO, "results", "figures", "regime")

# Diverging blue <-> red with a neutral grey midpoint (dataviz palette).
DIVERGING = LinearSegmentedColormap.from_list(
    "better_worse", ["#184f95", "#3987e5", "#9ec5f4", "#f0efec",
                     "#f2b0ae", "#e34948", "#a32a2a"])
INK, MUTED, SURFACE = "#1f1f1e", "#6b6b67", "#fcfcfb"
POLICY_ROWS = ["sjf", "edf", "lsf", "lsf-oracle"]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def _panels(data):
    """Batch entries to draw: merged duplicates are skipped."""
    return [b for b in data["batches"] if not b.get("merged_into")]


def _label(b, data):
    mb = b["max_batch"]
    merged = [x["max_batch"] for x in data["batches"] if x.get("merged_into") == mb]
    name = f"batch {mb}" + "".join(f"/{m}" for m in merged)
    if not b["cap_binds_somewhere"]:
        name += " (KV-bound)"
    return name


def scheduling_heatmap(data, metric, fname, title, unit, scale=1.0):
    panels = _panels(data)
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 1.9 * len(panels) + 1.0),
                             squeeze=False, facecolor=SURFACE)
    vals = [abs(c["policies"][p][metric]["mean_diff_s"]) * scale
            for b in panels for c in b["cells"] for p in POLICY_ROWS
            if metric in c["policies"].get(p, {})]
    vmax = max(vals) if vals else 1.0
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    im = None
    for ax, b in zip(axes[:, 0], panels):
        cells = sorted(b["cells"], key=lambda c: c["load"])
        grid = [[c["policies"].get(p, {}).get(metric, {}).get("mean_diff_s", float("nan")) * scale
                 for c in cells] for p in POLICY_ROWS]
        im = ax.imshow(grid, cmap=DIVERGING, norm=norm, aspect="auto")
        for j, c in enumerate(cells):
            for i, p in enumerate(POLICY_ROWS):
                ci = c["policies"].get(p, {}).get(metric)
                if not ci:
                    continue
                v = ci["mean_diff_s"] * scale
                if not ci["significant"]:
                    ax.add_patch(plt.Rectangle((j - .5, i - .5), 1, 1, fill=False,
                                               hatch="///", edgecolor="#c4c3bf", lw=0))
                dark = abs(v) > 0.6 * vmax
                ax.text(j, i, f"{v:+.0f}" if abs(v) >= 10 else f"{v:+.1f}",
                        ha="center", va="center", fontsize=7,
                        color="white" if dark else INK)
            if c["inactive"]:
                ax.plot(j, -0.75, "o", ms=5, mfc="none", mec=MUTED, clip_on=False)
        first = data["summary"]["crossover"].get(f"lsf@b{b['max_batch']}")
        if first:
            j = next(k for k, c in enumerate(cells) if c["load"] == first["load"])
            ax.axvline(j - 0.5, color=INK, ls="--", lw=1.2)
        ax.set_yticks(range(len(POLICY_ROWS)), POLICY_ROWS)
        ax.set_xticks(range(len(cells)), [f"{c['load']:.2f}" for c in cells])
        ax.set_ylabel(_label(b, data), color=INK, fontsize=9)
        _style(ax)
    axes[-1, 0].set_xlabel("load = arrival rate / rate at util 0.8", color=INK, fontsize=9)
    fig.suptitle(title, color=INK, fontsize=11, x=0.02, ha="left")
    fig.text(0.02, -0.03, "blue = better than FCFS, red = worse; hatched = 95% CI includes "
             "zero; hollow dot = inactive (choice < 5%); dashed = first LSF separation",
             color=MUTED, fontsize=7)
    cb = fig.colorbar(im, ax=axes[:, 0].tolist(), shrink=0.8, pad=0.02)
    cb.set_label(f"policy − FCFS ({unit})", color=INK, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=MUTED, labelcolor=INK)
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, fname)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return path


def scheduling_table(data, manifest_path):
    lines = [f"Source: `{os.path.relpath(manifest_path, _REPO)}` (manifest inside; "
             f"git {data['manifest']['git_hash'][:7]}, profile "
             f"{data['manifest']['params']['hardware_profile']}).", ""]
    for b in _panels(data):
        lines += [f"**{_label(b, data)}**, slack_factor {b['slack_factor']:.3f} "
                  f"(FCFS miss {b['slack_fcfs_miss_at_util_0.8']:.1%} at util 0.8)", "",
                  "| load | choice | at cap | in-flight max/p95 | FCFS p95 | FCFS miss (infeasible) "
                  "| LSF − FCFS p95 | EDF − FCFS p95 | LSF − FCFS miss | flags |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
        for c in sorted(b["cells"], key=lambda c: c["load"]):
            f = c["policies"]["fcfs"]

            def ci(pol, k, pct=False):
                r = c["policies"].get(pol, {}).get(k)
                if not r:
                    return "n/a"
                fmt = (lambda x: f"{x * 100:+.1f}") if pct else (lambda x: f"{x:+.1f}")
                star = "**" if r["significant"] else ""
                return f"{star}{fmt(r['mean_diff_s'])} [{fmt(r['lo_s'])}, {fmt(r['hi_s'])}]{star}"
            flags = ", ".join(x for x, on in (("inactive", c["inactive"]),
                                              ("cap never binds", c["cap_never_binds"])) if on)
            lines.append(
                f"| {c['load']:.2f} | {f['frac_decisions_with_choice']:.1%} | "
                f"{f['frac_time_at_batch_cap']:.0%} | {f['max_batch_seen_max']}/{f['p95_batch']:.1f} | "
                f"{f['p95_task_latency_s']:.0f}s | {f['deadline_miss_rate']:.0%} "
                f"({f['miss_rate_infeasible_at_admit']:.0%}) | {ci('lsf', 'p95_vs_fcfs')} | "
                f"{ci('edf', 'p95_vs_fcfs')} | {ci('lsf', 'miss_vs_fcfs', True)} pts | {flags} |")
        lines.append("")
    lines += ["Bold = paired 95% CI excludes zero. Seconds unless marked pts "
              "(percentage points of deadline-miss rate).", ""]
    lines += [f"- {x}" for x in data["summary"]["lines"]]
    return "\n".join(lines)


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "scheduling"
    if which != "scheduling":
        raise SystemExit("only 'scheduling' is implemented")
    path = sys.argv[2] if len(sys.argv) > 2 else os.path.join(_REPO, "results",
                                                              "scheduling_map.json")
    with open(path) as fh:
        data = json.load(fh)
    a = scheduling_heatmap(data, "p95_vs_fcfs", "scheduling_p95.png",
                           "Scheduling map: p95 task latency, policy − FCFS", "s")
    b = scheduling_heatmap(data, "miss_vs_fcfs", "scheduling_miss.png",
                           "Scheduling map: deadline-miss rate, policy − FCFS",
                           "percentage points", scale=100)
    table = scheduling_table(data, path)
    tpath = path.replace(".json", "_table.md")
    with open(tpath, "w") as fh:
        fh.write(table + "\n")
    print(a, b, tpath, sep="\n")


if __name__ == "__main__":
    main()
