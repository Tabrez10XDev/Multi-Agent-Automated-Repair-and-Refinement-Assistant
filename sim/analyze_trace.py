"""
Describe a recorded MAARA trace: token distributions, reuse, and the gaps
between a prefix being produced and being wanted again.

    python -m sim.analyze_trace [trace.jsonl.gz]

Writes results/trace_stats.json and figures under results/figures/.

The question this exists to answer is whether the reuse gaps in a real run
straddle the ski-rental break-even. If every gap is far below it, holding is
always right and the policy is trivial; far above, evicting is always right and
the policy is equally trivial. Only a distribution that straddles makes the
keep-or-evict decision interesting.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics as st
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sim.trace_loader import (compute_prefixes, default_prompts_path,  # noqa: E402
                              detect_cache_hits, fingerprint_config, load_rows,
                              render_prompt, segment_task_runs,
                              validate_segmentation)
from sim.workload import CostModel  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
FIGURES = os.path.join(RESULTS, "figures")

# Memory-pressure levels for the break-even band. Derived rent is
# victim_recompute x eviction_rate (HANDOFF section 13), and a single-stream
# trace has no pressure at all, so break-even cannot be computed from the trace
# -- it has to be shown as a band across assumed pressure.
PRESSURE = {"low": 0.01, "medium": 0.1, "high": 1.0}   # evictions/second


def _stats(v):
    if not v:
        return {"n": 0}
    v = sorted(v)
    mean = st.fmean(v)
    sd = st.stdev(v) if len(v) > 1 else 0.0
    return {
        "n": len(v), "mean": round(mean, 1), "p50": v[len(v) // 2],
        "p95": v[min(len(v) - 1, int(0.95 * len(v)))], "max": v[-1],
        "cov": round(sd / mean, 3) if mean else 0.0,
    }


def analyse(main_path, prompts_path=None):
    prompts_path = prompts_path or default_prompts_path(main_path)
    rows = load_rows(main_path)
    prompts = {p["call_id"]: p.get("messages", []) for p in load_rows(prompts_path)}
    blocks = segment_task_runs(rows)
    cost = CostModel()

    out = {
        "trace": os.path.relpath(main_path, _REPO),
        "validation": validate_segmentation(blocks),
        "ollama_cache": detect_cache_hits(rows, prompts),
    }

    # --- (a) per-role token distributions, (b) effect of attempt_index ----
    by_role_prompt, by_role_output = defaultdict(list), defaultdict(list)
    by_attempt = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r.get("kind") != "chat":
            continue
        role = r["agent_role"]
        by_role_prompt[role].append(r.get("prompt_tokens") or 0)
        by_role_output[role].append(r.get("output_tokens") or 0)
        by_attempt[role][r.get("attempt_index", 0)].append(
            (r.get("prompt_tokens") or 0, r.get("output_tokens") or 0))

    out["tokens_by_role"] = {
        role: {"prompt": _stats(by_role_prompt[role]),
               "output": _stats(by_role_output[role])}
        for role in sorted(by_role_prompt)}

    out["tokens_by_attempt"] = {
        role: {str(a): {"prompt": _stats([p for p, _ in v]),
                        "output": _stats([o for _, o in v])}
               for a, v in sorted(d.items())}
        for role, d in sorted(by_attempt.items())}

    # --- (c) reuse gaps, (e) reuse size, (f) config split ----------------
    end_of = {r["call_id"]: r.get("t_end") for r in rows}
    start_of = {r["call_id"]: r.get("t_start") for r in rows}
    row_of = {r["call_id"]: r for r in rows}

    gaps, gap_detail, reuse_tokens = [], [], []
    per_config = defaultdict(lambda: {"gaps": [], "reuse_tokens": [],
                                      "prompt_tokens": 0, "reusable_tokens": 0})
    total_prompt_tokens = total_reusable = 0

    for idx, block in enumerate(blocks):
        config = fingerprint_config(block) or "unknown"
        pf = compute_prefixes(block, prompts, f"r{idx}")
        block_ids = {r["call_id"] for r in block}
        traced = [(r.get("t_start"), r.get("t_end"), r["agent_role"]) for r in block]

        for r in block:
            if r.get("kind") != "chat":
                continue
            info = pf.get(r["call_id"])
            pt = r.get("prompt_tokens") or 0
            total_prompt_tokens += pt
            per_config[config]["prompt_tokens"] += pt
            if not info or not info.shares_with or info.shared_tokens <= 0:
                continue
            if info.shares_with not in block_ids:
                continue

            total_reusable += info.shared_tokens
            per_config[config]["reusable_tokens"] += info.shared_tokens
            reuse_tokens.append(info.shared_tokens)
            per_config[config]["reuse_tokens"].append(info.shared_tokens)

            t0, t1 = end_of.get(info.shares_with), start_of.get(r["call_id"])
            if t0 is None or t1 is None or t1 < t0:
                continue
            gap = t1 - t0
            gaps.append(gap)
            per_config[config]["gaps"].append(gap)

            # What filled the gap? Traced calls overlapping the window; the
            # remainder is untraced work -- overwhelmingly pytest, which is a
            # subprocess and makes no LLM or embedding call.
            filled = defaultdict(float)
            for s, e, role in traced:
                if s is None or e is None:
                    continue
                lo, hi = max(s, t0), min(e, t1)
                if hi > lo:
                    filled[role] += hi - lo
            accounted = sum(filled.values())
            gap_detail.append({
                "gap_s": round(gap, 3),
                "shared_tokens": info.shared_tokens,
                "role": r["agent_role"], "config": config,
                "attempt_index": r.get("attempt_index", 0),
                "filled_by": {k: round(v, 3) for k, v in filled.items()},
                "untraced_s": round(max(0.0, gap - accounted), 3),
            })

    out["reuse"] = {
        "calls_with_reuse": len(reuse_tokens),
        "shared_tokens": _stats(reuse_tokens),
        "total_prompt_tokens": total_prompt_tokens,
        "total_reusable_tokens": total_reusable,
        "reusable_fraction": round(total_reusable / total_prompt_tokens, 4)
        if total_prompt_tokens else 0.0,
        "prefill_seconds_saved_if_never_evicted": round(
            cost.recompute_seconds("qwen2.5-coder:7b", total_reusable), 2),
    }
    out["gaps"] = _stats([round(g, 3) for g in gaps])
    out["gap_detail_sample"] = sorted(gap_detail, key=lambda d: -d["shared_tokens"])[:15]

    untraced = [d["untraced_s"] for d in gap_detail]
    out["gap_composition"] = {
        "untraced_seconds": _stats([round(u, 3) for u in untraced]),
        "note": ("untraced time is overwhelmingly apply_and_test: pytest runs as a "
                 "subprocess and makes no LLM or embedding call, so it leaves no row"),
    }

    out["by_config"] = {
        c: {"runs": sum(1 for b in blocks if fingerprint_config(b) == c),
            "gaps": _stats([round(g, 3) for g in d["gaps"]]),
            "shared_tokens": _stats(d["reuse_tokens"]),
            "reusable_fraction": round(d["reusable_tokens"] / d["prompt_tokens"], 4)
            if d["prompt_tokens"] else 0.0}
        for c, d in sorted(per_config.items())}

    # --- (d) break-even band ---------------------------------------------
    # t* = E_recompute / (victim_recompute x eviction_rate). Assuming the
    # marginal victim is the same size as E -- true on average when entries are
    # drawn from one distribution -- the size terms cancel and t* = 1/rate.
    # That cancellation is itself the open question in HANDOFF section 13.
    med_reuse = out["reuse"]["shared_tokens"].get("p50", 0) or 0
    out["break_even"] = {
        "definition": "hold until accrued rent >= own recompute cost (HANDOFF section 13)",
        "rent_rate": "marginal_victim_recompute_s x evictions_per_second",
        "assumption": ("victim and holder are the same size, so the size terms "
                       "cancel and t* = 1 / eviction_rate, independent of entry size"),
        "median_reuse_tokens": med_reuse,
        "median_reuse_recompute_s": round(
            cost.recompute_seconds("qwen2.5-coder:7b", med_reuse), 3),
        "t_star_seconds": {k: round(1.0 / v, 2) for k, v in PRESSURE.items()},
        "eviction_rates_assumed": PRESSURE,
        "caveat": ("the traced run is single-stream with no memory pressure, so "
                   "eviction rate is zero and true break-even is infinite; the band "
                   "shows what it would be under assumed pressure"),
    }
    return out, gaps, reuse_tokens, by_role_prompt, by_role_output, by_attempt


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def _fig_gaps(gaps, stats, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    if gaps:
        ax.hist(gaps, bins=40, color="#4C72B0", edgecolor="white")
    colours = {"low": "#7f7f7f", "medium": "#DD8452", "high": "#C44E52"}
    for name, t in stats["break_even"]["t_star_seconds"].items():
        ax.axvline(t, color=colours[name], linestyle="--", linewidth=1.6,
                   label=f"break-even, {name} pressure ({t:g}s)")
    ax.set_xlabel("gap from prefix produced to prefix wanted again (s)")
    ax.set_ylabel("number of reuses")
    ax.set_title("Reuse gaps vs ski-rental break-even\n"
                 "single-stream trace: no queueing, so gaps are pure pipeline latency")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_tokens(by_prompt, by_output, path):
    roles = sorted(by_prompt)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, data, title in ((axes[0], by_prompt, "prompt tokens"),
                            (axes[1], by_output, "output tokens")):
        # tick_labels= since matplotlib 3.9; labels= was removed in 3.11.
        ax.boxplot([data[r] for r in roles], tick_labels=roles, showfliers=False)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        ax.set_ylabel("tokens")
    fig.suptitle("Token distributions by agent role")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_reuse(reuse_tokens, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    if reuse_tokens:
        ax.hist(reuse_tokens, bins=40, color="#55A868", edgecolor="white")
        ax.axvline(st.median(reuse_tokens), color="#C44E52", linestyle="--",
                   label=f"median {int(st.median(reuse_tokens))} tokens")
        ax.legend()
    ax.set_xlabel("shared prefix size per reuse (tokens, APPROXIMATE)")
    ax.set_ylabel("number of reuses")
    ax.set_title("How much context is actually reusable at HumanEval scale")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _fig_attempt(by_attempt, path):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    roles = sorted(by_attempt)
    width = 0.8 / max(1, len(roles))
    attempts = sorted({a for d in by_attempt.values() for a in d})
    for i, role in enumerate(roles):
        xs, ys = [], []
        for j, a in enumerate(attempts):
            v = by_attempt[role].get(a)
            if v:
                xs.append(j + i * width)
                ys.append(st.fmean([p for p, _ in v]))
        ax.bar(xs, ys, width=width, label=role)
    ax.set_xticks([j + 0.4 for j in range(len(attempts))])
    ax.set_xticklabels([f"attempt {a}" for a in attempts])
    ax.set_ylabel("mean prompt tokens")
    ax.set_title("Does the retry loop grow the prompt?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", nargs="?",
                    default=os.path.join(_REPO, "traces", "full_zs_4cfg.jsonl.gz"))
    ap.add_argument("--prompts", default=None)
    args = ap.parse_args()

    os.makedirs(FIGURES, exist_ok=True)
    stats, gaps, reuse, bp, bo, ba = analyse(args.trace, args.prompts)

    _fig_gaps(gaps, stats, os.path.join(FIGURES, "gap_histogram.png"))
    _fig_tokens(bp, bo, os.path.join(FIGURES, "tokens_by_role.png"))
    _fig_reuse(reuse, os.path.join(FIGURES, "reuse_sizes.png"))
    _fig_attempt(ba, os.path.join(FIGURES, "tokens_by_attempt.png"))

    stats["figures"] = sorted(
        os.path.relpath(os.path.join(FIGURES, f), _REPO)
        for f in os.listdir(FIGURES) if f.endswith(".png"))

    path = os.path.join(RESULTS, "trace_stats.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {os.path.relpath(path, _REPO)} and {len(stats['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
