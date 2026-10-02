"""
Describe a recorded MAARA trace: token distributions, reuse, and the gaps
between a prefix being produced and being wanted again.

    python -m simulator.analyze_trace [trace.jsonl.gz]

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

from simulator.trace_loader import (TRACE_SETS, compute_prefixes,  # noqa: E402
                              default_prompts_path, detect_cache_hits,
                              fingerprint_config, load_rows, render_prompt,
                              segment_task_runs, trace_set,
                              validate_segmentation)
from simulator.workload import CostModel  # noqa: E402

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(_REPO, "results")
FIGURES = os.path.join(RESULTS, "figures")

# Memory-pressure levels for the break-even band. Derived rent is
# victim_recompute x eviction_rate (HANDOFF section 13), and a single-stream
# trace has no pressure at all, so break-even cannot be computed from the trace
# -- it has to be shown as a band across assumed pressure.
PRESSURE = {"low": 0.01, "medium": 0.1, "high": 1.0}   # evictions/second

# Must match EngineConfig.block_tokens. PLACEHOLDER (vLLM default).
BLOCK_TOKENS = 16

MODEL = "qwen2.5-coder:7b"


def break_even_table(cost, shared_sizes=(4, 64, 2048), shared_over_held=0.138):
    """
    Worked example: how long is it worth holding an entry?

        rent_rate(E) = E.bytes x victim_recompute_s_per_byte x eviction_rate
        break-even   = (recompute saved by keeping E) / rent_rate(E)

    Each row is indexed by the REUSABLE prefix size, because that is what pays
    back; the three retention units differ in how much you are obliged to store
    to keep it:

        hold only the shared prefix   held = shared            (the ideal)
        hold whole context            held = shared / 0.138    (measured ratio)
        block-quantised head          held = ceil(shared/16)*16 (vLLM-like)

    Rent is charged on held bytes, benefit accrues on the reusable part. Showing
    several sizes makes the key point visible: the size CANCELS in every unit,
    since rent and benefit are both linear in tokens. What actually moves
    break-even is useful/held -- and block quantisation only bites below one
    block, which is exactly where this workload's median reuse sits.
    """
    rows = []
    for shared in shared_sizes:
        benefit_s = cost.recompute_seconds(MODEL, shared)
        held_by_unit = (
            ("hold only the shared prefix", float(shared)),
            ("hold whole context", shared / shared_over_held),
            (f"block-quantised head ({BLOCK_TOKENS}-tok blocks)",
             float(-(-shared // BLOCK_TOKENS) * BLOCK_TOKENS)),
        )
        for unit, held in held_by_unit:
            nbytes = cost.kv_bytes(MODEL, held)
            # Victim drawn from the same distribution, so its recompute per
            # byte equals ours: 1/(prefill_rate x bytes_per_token).
            s_per_byte = cost.recompute_seconds(MODEL, held) / nbytes
            row = {"reusable_tokens": shared, "unit": unit,
                   "held_tokens": round(held, 1),
                   "useful_over_held": round(shared / held, 3),
                   "held_mb": round(nbytes / 1024 / 1024, 1),
                   "benefit_s": round(benefit_s, 4), "t_star_s": {}}
            for name, rate in PRESSURE.items():
                rent_rate = nbytes * s_per_byte * rate
                row["t_star_s"][name] = round(benefit_s / rent_rate, 3) if rent_rate else None
            rows.append(row)
    return rows


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
    paths = [main_path] if isinstance(main_path, str) else list(main_path)
    prompt_paths = ([default_prompts_path(p) for p in paths] if prompts_path is None
                    else ([prompts_path] if isinstance(prompts_path, str)
                          else list(prompts_path)))
    rows = []
    for p in paths:
        rows.extend(load_rows(p))
    prompts = {}
    for pp in prompt_paths:
        if os.path.exists(pp):
            prompts.update({p["call_id"]: p.get("messages", []) for p in load_rows(pp)})
    blocks = segment_task_runs(rows)
    cost = CostModel()

    out = {
        "trace": [os.path.relpath(p, _REPO) for p in paths],
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
    # Block quantisation: a 4-token shared prefix still occupies a whole
    # 16-token block, so the entry held is larger than the part that pays back.
    ratios = [min(1.0, t / (-(-t // BLOCK_TOKENS) * BLOCK_TOKENS))
              for t in reuse_tokens if t > 0]
    block_ratio = st.median(ratios) if ratios else 1.0
    out["block_quantisation"] = {
        "block_tokens": BLOCK_TOKENS,
        "median_useful_over_held": round(block_ratio, 4),
        "note": ("median reusable prefix is smaller than one block, so most of "
                 "each retained head is padding that can never be matched"),
    }
    out["break_even_worked_example"] = break_even_table(cost)
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
    """
    Two bands, because the retention unit changes the answer.

    Reference (dashed): entry holds exactly the reusable prefix, so break-even
    is 1/eviction_rate and the entry size cancels out.

    Chosen unit (solid): KV is allocated in whole blocks, so a 4-token shared
    prefix occupies a 16-token block. Only the useful fraction pays back, and
    break-even shrinks by that ratio.
    """
    import numpy as np

    ratio = stats["block_quantisation"]["median_useful_over_held"]
    fig, ax = plt.subplots(figsize=(10, 5.6))

    # Log x-axis with log-spaced bins. Gaps span 0.01s to 40s, so a linear axis
    # collapses three quarters of the distribution into the first bin. Zero and
    # sub-10ms gaps are floored so they stay visible on a log scale.
    floor = 0.01
    vals = [max(floor, g) for g in gaps] or [floor]
    lo, hi = floor, max(max(vals) * 1.3, 1.0)
    bins = np.logspace(np.log10(lo), np.log10(hi), 45)
    ax.hist(vals, bins=bins, color="#4C72B0", edgecolor="white", linewidth=0.4)

    colours = {"low": "#7f7f7f", "medium": "#DD8452", "high": "#C44E52"}
    for name, t in stats["break_even"]["t_star_seconds"].items():
        ax.axvline(t, color=colours[name], linestyle=":", linewidth=1.4, alpha=0.8,
                   label=f"exact-prefix reference, {name} pressure ({t:g}s)")
        ax.axvline(t * ratio, color=colours[name], linestyle="-", linewidth=2.2,
                   label=f"block-quantised (chosen), {name} ({t * ratio:.2f}s)")

    ax.set_xscale("log")
    ax.set_xlim(lo, hi)
    ax.set_xlabel(f"gap from prefix produced to prefix wanted again "
                  f"(s, log; values below {floor}s floored)")
    ax.set_ylabel("number of reuses")
    ax.set_title("Reuse gaps vs ski-rental break-even\n"
                 f"solid = chosen unit ({BLOCK_TOKENS}-token blocks, median useful/held "
                 f"{ratio:.2f});  dotted = exact-prefix reference\n"
                 "single-stream trace: no queueing, so gaps are pure pipeline latency",
                 fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper center",
              bbox_to_anchor=(0.5, -0.16), frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
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
    ap.add_argument("trace", nargs="*", default=None,
                    help="explicit trace paths; overrides --set")
    ap.add_argument("--set", dest="set_name", choices=sorted(TRACE_SETS),
                    default="original",
                    help="named trace set; figures and stats are namespaced by it")
    ap.add_argument("--prompts", default=None)
    args = ap.parse_args()

    traces = args.trace or trace_set(args.set_name)
    if not traces:
        print(f"no trace files present for set {args.set_name!r}; "
              f"expected {TRACE_SETS[args.set_name]}")
        return 1

    figdir = os.path.join(FIGURES, args.set_name)
    os.makedirs(figdir, exist_ok=True)
    stats, gaps, reuse, bp, bo, ba = analyse(traces, args.prompts)
    stats["set"] = args.set_name

    _fig_gaps(gaps, stats, os.path.join(figdir, "gap_histogram.png"))
    _fig_tokens(bp, bo, os.path.join(figdir, "tokens_by_role.png"))
    _fig_reuse(reuse, os.path.join(figdir, "reuse_sizes.png"))
    _fig_attempt(ba, os.path.join(figdir, "tokens_by_attempt.png"))

    stats["figures"] = sorted(
        os.path.relpath(os.path.join(figdir, f), _REPO)
        for f in os.listdir(figdir) if f.endswith(".png"))

    path = os.path.join(RESULTS, f"trace_stats_{args.set_name}.json")
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))
    print(f"\nwrote {os.path.relpath(path, _REPO)} and {len(stats['figures'])} figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
