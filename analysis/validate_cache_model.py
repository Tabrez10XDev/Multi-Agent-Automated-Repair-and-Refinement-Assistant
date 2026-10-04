"""
Test the FROZEN Ollama cache model against a trace it was never fit to.

    python -m analysis.validate_cache_model data/traces/<holdout>.jsonl.gz

This script has NO tuning knobs, by design. About thirteen rule variants were
scored on `original` and `shared_first` while the mechanism was being found, so
agreement on those two traces stopped being evidence some time ago. The only
honest remaining test is one run of the frozen rule against a trace that played
no part in choosing it.

If the result disappoints, the correct response is to write the number down.
Changing SIM_THRESHOLD, DEFAULT_RAM_BYTES or the rule after seeing this output
converts the held-out trace into another fitting set and there is no third
chance -- collecting a fresh trace costs hours of GPU time.

Exits 0 if it ran, 2 if the trace is one of the two the model was fit to.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from simulator.ollama_cache import (DEFAULT_RAM_BYTES, FROZEN_AT,
                                    FROZEN_SCORES, SIM_THRESHOLD,
                                    OllamaPromptCache)
from simulator.trace_loader import (TRACE_SETS, default_prompts_path,
                                    fingerprint_config, load_rows,
                                    render_prompt, segment_task_runs)
from simulator.workload import CostModel

HIT_FRAC = 1 / 3          # frozen: reusing f of a prompt gives 1/(1-f) speedup
SPEEDUP = 1.5             # frozen: the trace detector's threshold

FIT_TO = {os.path.basename(p) for paths in TRACE_SETS.values() for p in paths}


def measured_hit(row, cost):
    pt = row.get("prompt_tokens") or 0
    secs = row.get("prompt_eval_duration") or 0.0
    model = row.get("model") or ""
    if pt <= 0 or secs <= 0 or model not in cost.rates:
        return None
    return (pt / secs) > SPEEDUP * cost.rates[model].prefill_rate


def degraded_parts(chats, cost):
    """
    Flag process segments whose prefill rate ran below calibration.

    A trace written by two processes can contain a stretch where the machine
    delivered well under the calibrated uncached rate. The hit detector
    thresholds at 1.5x that rate, so inside such a stretch a real cache hit can
    look like a miss and the ground truth is not trustworthy. On the
    shared_first trace this cost 19 of 21 apparent false positives, all in one
    segment whose median rate was 1.20x against the other segment's 1.73x.
    """
    part, k = [], 0
    for i, r in enumerate(chats):
        if i and (r.get("seq") or 0) <= (chats[i - 1].get("seq") or 0):
            k += 1
        part.append(k)
    ratios = {}
    for p, r in zip(part, chats):
        pt = r.get("prompt_tokens") or 0
        secs = r.get("prompt_eval_duration") or 0.0
        m = r.get("model") or ""
        if pt <= 0 or secs <= 0 or m not in cost.rates:
            continue
        ratios.setdefault(p, []).append((pt / secs) / cost.rates[m].prefill_rate)
    bad = set()
    for p, v in ratios.items():
        below = sum(1 for x in v if x < 1.0) / len(v)
        if below > 0.15:
            bad.add(p)
    return part, bad, ratios


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace", help="path to a held-out .jsonl(.gz) trace")
    ap.add_argument("--prompts", default=None, help="sidecar (default: inferred)")
    ap.add_argument("--allow-fit-trace", action="store_true",
                    help="override the guard; the result is then not a held-out test")
    args = ap.parse_args()

    base = os.path.basename(args.trace)
    if base in FIT_TO and not args.allow_fit_trace:
        print(f"REFUSED  {base} is one of the traces the model was fit to.")
        print(f"         The frozen rule already scores {FROZEN_SCORES} on those.")
        print("         Point this at a held-out trace, or pass --allow-fit-trace")
        print("         and do not call the result a validation.")
        return 2

    cost = CostModel()
    rows = load_rows(args.trace)
    prompts = {x["call_id"]: x["messages"]
               for x in load_rows(args.prompts or default_prompts_path(args.trace))}

    cfg = {}
    for block in segment_task_runs(rows):
        c = fingerprint_config(block) or "unidentified"
        for r in block:
            cfg[r["call_id"]] = c

    chats = sorted((r for r in rows if r.get("kind") == "chat"),
                   key=lambda r: r.get("t_start") or 0.0)
    part, bad_parts, ratios = degraded_parts(chats, cost)

    cache = OllamaPromptCache(cost=cost)      # frozen defaults, no arguments
    tp = fp = fn = tn = 0
    per_role, clean = {}, {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for p, r in zip(part, chats):
        act = measured_hit(r, cost)
        if act is None:
            continue
        pt = r.get("prompt_tokens") or 0
        reused = cache.access(r["model"], render_prompt(prompts.get(r["call_id"], [])), pt)
        pred = pt > 0 and (reused / pt) > HIT_FRAC
        b = "tp" if (pred and act) else "fp" if pred else "fn" if act else "tn"
        tp += b == "tp"; fp += b == "fp"; fn += b == "fn"; tn += b == "tn"
        per_role.setdefault(r["agent_role"], {"tp": 0, "fp": 0, "fn": 0, "tn": 0})[b] += 1
        if p not in bad_parts:
            clean[b] += 1

    n = tp + fp + fn + tn
    print(f"HELD-OUT VALIDATION   trace={base}")
    print(f"  rule frozen {FROZEN_AT}: threshold={SIM_THRESHOLD}, "
          f"ram={DEFAULT_RAM_BYTES / 1024 ** 3:.0f}GiB, no tuning applied")
    print(f"  fit-set scores for reference: {FROZEN_SCORES}")
    print()
    print(f"  calls scored            {n}")
    print(f"  measured hits           {tp + fn}")
    print(f"  predicted hits          {tp + fp}")
    print(f"  matched (true positive) {tp}")
    print(f"  missed  (false negative){fn:>4}")
    print(f"  spurious (false positive){fp:>3}")
    print(f"  agreement               {(tp + tn) / n:.1%}" if n else "  agreement   n/a")
    print()
    if bad_parts:
        cn = sum(clean.values())
        print(f"  WARNING: process segment(s) {sorted(bad_parts)} ran below the")
        print(f"  calibrated prefill rate, so the hit detector is unreliable there.")
        for p in sorted(bad_parts):
            v = ratios[p]
            print(f"    segment {p}: n={len(v)}, "
                  f"{100 * sum(1 for x in v if x < 1.0) / len(v):.0f}% of calls slower than cold")
        if cn:
            print(f"  Excluding them: {clean['tp']} matched of "
                  f"{clean['tp'] + clean['fn']} measured hits, "
                  f"{clean['fp']} spurious, "
                  f"agreement {(clean['tp'] + clean['tn']) / cn:.1%} (n={cn})")
        print()
    print(f"  {'role':<14}{'measured':>9}{'predicted':>10}{'tp':>5}{'fp':>5}{'fn':>5}")
    for role, d in sorted(per_role.items()):
        print(f"  {role:<14}{d['tp'] + d['fn']:>9}{d['tp'] + d['fp']:>10}"
              f"{d['tp']:>5}{d['fp']:>5}{d['fn']:>5}")
    print()
    print("  Record this number in docs/FINDINGS.md before changing anything.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
