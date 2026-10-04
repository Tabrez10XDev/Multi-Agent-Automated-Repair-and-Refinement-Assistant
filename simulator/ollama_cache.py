"""
A model of Ollama 0.33.3's prompt cache, and its validation against the traces.

STATUS: UNVALIDATED. Do not use this as a baseline yet.
Replaying the two committed traces, this rule predicts 68 of 89 measured cache
hits on `original` (no false positives) but only 7 of 78 on `shared_first`.
The asymmetry is the open question: in `shared_first` every prompt shares a
long prefix with every other by construction, so the in-place slot path fires
on almost every call and a saved exact match is never consulted. A best-match
lookup closes most of that gap but is not adopted, because its consistency
with the affinity probe rests on an untested size hypothesis.

See docs/OPEN_ISSUES.md section 15 for the candidates tried and the one probe
that would settle it. Do not tune the parameters here against these two
traces -- about thirteen variants have already been scored on them, and a
held-out trace is the only honest test left (docs/DECISIONS.md section 18).

    python -m simulator.ollama_cache          # replay both committed traces

MECHANISM, as confirmed from the server log and the affinity probe
------------------------------------------------------------------
llama-server runs with `-np 1`: ONE active slot per model, holding the token
sequence of the last prompt it served. Displaced states can be saved to a
host-RAM cache ("prompt cache update took 23.67 ms", entries logged as
"3791 tokens, 207.364 MiB").

Which of the two happens is decided by prefix similarity against the active
slot, with a threshold of 0.10 ("checking sim = 0.002 > 0.100"):

  sim >= 0.10   The slot is REUSED in place: trimmed to the common prefix and
                extended with the new prompt. Whatever the previous prompt had
                beyond that prefix is DESTROYED, not saved.
  sim <  0.10   A slot is "selected by LRU", and the displaced state is saved
                to the RAM cache. The incoming prompt may itself be restored
                from RAM if a saved state matches it.

The affinity probe measured exactly this. With A in the slot and B sharing
25%/50%/90% of it, re-sending A recomputed 76%/52%/11% of a cold prompt --
its UNSHARED fraction each time. At 5% shared, A came back in 0.3% of cold,
i.e. restored intact from RAM. A 90% match is therefore a cheap overwrite, not
a survival, which is the counter-intuitive part: being MORE similar to the
next prompt makes you more likely to be destroyed, because similarity is what
triggers in-place reuse.

WHY THIS IS WORTH MODELLING
---------------------------
It is what the real system does today, so it is the honest baseline for any
retention policy to beat -- not a strawman, and not vLLM's block cache, which
the simulator models elsewhere.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

from simulator.trace_loader import (_lcp_len, default_prompts_path,
                              fingerprint_config, load_rows, render_prompt,
                              segment_task_runs, trace_set)
from simulator.workload import CostModel

# Prompts run to tens of thousands of characters and every access compares
# against every saved state, so reuse the trace loader's binary-search LCP
# rather than a second copy of it.
common_prefix_len = _lcp_len

# llama-server's slot-reuse similarity threshold, read off the log line
# "checking sim = 0.002 > 0.100". Not a guess.
SIM_THRESHOLD = 0.10

# f_keep gate. MEASURED from server.log over 758 slot-selection events:
# f_keep >= 0.5 searched the prompt cache 0.8% of the time (n=128); f_keep <
# 0.5 searched 64.8% (n=630). So when at least half the slot is retained the
# server reuses it in place, saving nothing and searching nothing; below that
# it saves the displaced state and looks for a better starting point.
KEEP_GATE = 0.50

# Per-prompt token limit, logged as "(limits: 8192.000 MiB, N tokens, ...)".
# N tracked num_ctx across runs (4096, 8192, 16384 observed), so it caps a
# single cached prompt and is NOT a cache total. The third field ("est") is a
# derived estimate of total token capacity, not a limit.
DEFAULT_TOKEN_LIMIT = 8192

# f_sim and f_keep use DIFFERENT denominators. From one log triple:
#   slot release: n_tokens = 379
#   checking sim = 0.130 (3/23)            -> f_sim  = lcp / INCOMING tokens
#   f_sim_best = 0.130, f_keep = 0.008     -> f_keep = lcp / SLOT tokens
# f_sim picks a slot against SIM_THRESHOLD (irrelevant at -np 1, one slot);
# f_keep drives the gate above.
#
# CAVEAT: the server computes both on TOKENS; this model approximates with
# CHARACTERS scaled by each call's own chars-per-token ratio. They agree in
# aggregate but not per call, so a call with f_keep near KEEP_GATE can fall on
# the wrong side. replay() reports how many sit within +/-0.05 of the gate.

# Host prompt-cache budget. MEASURED 2026-10-02 by probes.ollama_cache_probe
# --capacity 40: 40 distinct ~3,790-token prompts were filled, then re-probed
# newest-first; 39 hit at ~0.04s and the oldest missed at 13.36s. That is
# ~39 x 207 MiB ~= 8 GiB, consistent with llama.cpp's default (no --cache-ram
# flag on the command line). Edge is +/-1 state. Capacity is modelled in bytes,
# not states, so small states are not charged as if they were large ones.
#
# Not binding at trace scale: replaying either committed trace evicts nothing,
# because trace states are 165-800 tokens (~9-45 MiB), not 3,790.
DEFAULT_RAM_BYTES = 8 * 1024 ** 3

# --------------------------------------------------------------------------
# FROZEN RULE v2 -- frozen 2026-10-04, after the mechanism was read out of
# llama.cpp's own source rather than inferred from timings.
# --------------------------------------------------------------------------
# Confirmed in llama.cpp tools/server/server-context.cpp:
#
#     const float f_sim_cur = float(lcp_len) / task.tokens.size();
#     const float f_keep    = (f_sim_best*task.tokens.size())
#                             / ret->prompt.tokens.size();
#     if (f_keep < 0.5f) { update_cache = true; }
#
# so f_sim is over the INCOMING prompt, f_keep over the CACHED/SLOT prompt, and
# the 0.5 constant is literal in the source. The independently measured gate
# from server.log (f_keep >= 0.5 searched 0.8% of 128 events, < 0.5 searched
# 64.8% of 630) agrees with it exactly.
#
# The rule as implemented in access():
#   1. One slot per model. f_keep = common prefix / slot length.
#   2. f_keep >= 0.50 -> reuse the slot in place; keep the common prefix,
#      recompute the rest, save nothing, search nothing.
#   3. f_keep <  0.50 -> save the displaced state to the host cache, then take
#      the best prefix match among the cached states if it beats the slot.
#      A restore copies, it does not consume (DECISIONS 17).
#   4. Host cache: LRU bounded in bytes (8192 MiB, logged) with a per-prompt
#      token cap tracking num_ctx. No state-level dedup exists in the binary.
#
# Frozen parameters: KEEP_GATE = 0.50, SIM_THRESHOLD = 0.10 (inert at -np 1),
#                    DEFAULT_RAM_BYTES = 8 GiB, DEFAULT_TOKEN_LIMIT = 8192,
#                    HIT_FRAC = 1/3.
#
# Fit-set scores at freeze, hits split by where the reuse came from:
#   original      84/89 hits, 1 FP, 97.8% agreement
#                 exact-prompt repeats 46/46; partial prefix reuse 38/43
#   shared_first  76/78 hits, 21 FP, 89.3% agreement
#                 exact-prompt repeats 47/47; partial prefix reuse 29/31
# The exact-repeat column is harness artefact, not workflow reuse
# (OPEN_ISSUES 20); 19 of shared_first's 21 FPs are in its degraded segment
# (FINDINGS 18). Both are reasons these traces cannot be the final word --
# the held-out trace is.
#
# v1 (commit 9911011) is superseded, not rewritten: it encoded the retracted
# "0.10 destroys the slot" rule. Kept auditable on purpose.
FROZEN_AT = "2026-10-04"
FROZEN_SCORES = {"original": (84, 89), "shared_first": (76, 78)}
FROZEN_VERSION = 2


@dataclass
class _Slot:
    text: str = ""
    tokens: int = 0


@dataclass
class OllamaPromptCache:
    """
    One active slot per model plus a shared host-RAM LRU of saved states.

    access() returns the number of prompt tokens that did NOT need recomputing.
    """
    sim_threshold: float = SIM_THRESHOLD
    keep_gate: float = KEEP_GATE
    ram_bytes: int = DEFAULT_RAM_BYTES
    token_limit: int = DEFAULT_TOKEN_LIMIT
    cost: CostModel = field(default_factory=CostModel)

    # save_on_reuse is gone. It asked whether an in-place reuse destroys the
    # displaced state; the server log answers it directly -- the save happens
    # whenever f_keep < KEEP_GATE and is skipped above it, so the behaviour is
    # a consequence of the gate rather than a free parameter.

    def __post_init__(self):
        self._slots = {}        # model -> _Slot
        self._ram = {}          # model -> list[(text, tokens)], most recent last
        self._ram_bytes_used = 0
        self._f_keeps = []
        self.stats = {"slot_reuse": 0, "slot_partial": 0, "ram_restore": 0,
                      "cold": 0, "ram_evictions": 0, "states_saved": 0,
                      "rejected_over_token_limit": 0}

    # -- RAM cache ---------------------------------------------------------

    def _state_bytes(self, model: str, tokens: int) -> int:
        return self.cost.kv_bytes(model, tokens)

    def _ram_save(self, model: str, text: str, tokens: int) -> None:
        if tokens <= 0:
            return
        if tokens > self.token_limit:
            # Logged as "cache token limit (%zu, est: %zu) reached". The limit
            # tracked num_ctx, so a prompt larger than the slot's context is
            # not cacheable.
            self.stats["rejected_over_token_limit"] += 1
            return
        size = self._state_bytes(model, tokens)
        if size > self.ram_bytes:
            return
        entries = self._ram.setdefault(model, [])
        entries.append((text, tokens))
        self._ram_bytes_used += size
        self.stats["states_saved"] += 1
        while self._ram_bytes_used > self.ram_bytes:
            for m, lst in self._ram.items():
                if lst:
                    old_text, old_tok = lst.pop(0)        # LRU
                    self._ram_bytes_used -= self._state_bytes(m, old_tok)
                    self.stats["ram_evictions"] += 1
                    break
            else:
                break

    def _ram_lookup(self, model: str, text: str):
        """Best prefix match among saved states for this model."""
        best, best_i = 0, None
        for i, (saved, _tok) in enumerate(self._ram.get(model, [])):
            n = common_prefix_len(text, saved)
            if n > best:
                best, best_i = n, i
        return best, best_i

    # -- the access path ---------------------------------------------------

    def access(self, model: str, text: str, tokens: int) -> int:
        """
        Returns the number of prompt tokens that did NOT need recomputing.

        The rule is the one the server log shows, not an inference from timings:

          f_keep >= KEEP_GATE   reuse the slot in place. Keep the common
                                prefix, recompute the rest. Save nothing (most
                                of the slot is being kept, so nothing
                                meaningful is displaced) and search nothing.
          f_keep <  KEEP_GATE   most of the slot is about to be discarded, so
                                save it to the host cache, then look for a
                                better starting point than the slot
                                ("looking for better prompt, base f_keep = X").
        """
        slot = self._slots.get(model)
        chars = max(1, len(text))
        cpt = chars / max(1, tokens)          # this call's chars per token

        lcp = common_prefix_len(text, slot.text) if (slot and slot.text) else 0
        slot_chars = max(1, len(slot.text)) if (slot and slot.text) else 1
        f_keep = (lcp / slot_chars) if (slot and slot.text) else 0.0
        self._f_keeps.append(f_keep)

        if slot is not None and slot.text and f_keep >= self.keep_gate:
            self._slots[model] = _Slot(text, tokens)
            self.stats["slot_reuse"] += 1
            return int(lcp / cpt)

        # Below the gate: save the displaced state, then search for better.
        if slot is not None and slot.text:
            self._ram_save(model, slot.text, slot.tokens)

        best_lcp, idx = self._ram_lookup(model, text)
        self._slots[model] = _Slot(text, tokens)
        if idx is not None and best_lcp > lcp:
            # A restore COPIES the saved state; it does not consume it. See
            # DECISIONS 17 -- the consuming version capped a prompt recurring
            # four times at a single hit.
            self.stats["ram_restore"] += 1
            return int(best_lcp / cpt)
        if lcp > 0:
            self.stats["slot_partial"] += 1
            return int(lcp / cpt)

        self.stats["cold"] += 1
        return 0


# --------------------------------------------------------------------------
# Validation against the measured traces
# --------------------------------------------------------------------------

def measured_hits(rows, cost, speedup=1.5):
    """Duration-based ground truth, same rule as trace_loader.detect_cache_hits."""
    out = {}
    for r in rows:
        if r.get("kind") != "chat":
            continue
        pt = r.get("prompt_tokens") or 0
        secs = r.get("prompt_eval_duration") or 0.0
        model = r.get("model") or ""
        if pt <= 0 or secs <= 0 or model not in cost.rates:
            continue
        out[r["call_id"]] = (pt / secs) > speedup * cost.rates[model].prefill_rate
    return out


def replay(trace_set_name="original", ram_bytes=DEFAULT_RAM_BYTES,
           hit_frac=1 / 3, keep_gate=KEEP_GATE):
    """
    Replay a trace and score hits, splitting them by WHERE the reuse came from.

    The split matters more than the total. `exact` hits are calls whose whole
    prompt was seen before -- on these traces that is overwhelmingly the
    harness replaying 15 tasks under 4 configs, which an operator would never
    see (OPEN_ISSUES 20). `partial` hits are genuine prefix reuse. Reporting
    one number conflates the two and is how Milestone 1's "one call in three"
    figure came about (FINDINGS 19).
    """
    cost = CostModel()
    rows, prompts = [], {}
    for full in trace_set(trace_set_name):
        rows += load_rows(full)
        for x in load_rows(default_prompts_path(full)):
            prompts[x["call_id"]] = x["messages"]

    truth = measured_hits(rows, cost)
    config_of, task_of = {}, {}
    for block in segment_task_runs(rows):
        cfg = fingerprint_config(block) or "unidentified"
        for r in block:
            config_of[r["call_id"]] = cfg
            task_of[r["call_id"]] = (r.get("task_id"), cfg)

    chats = sorted((r for r in rows if r.get("kind") == "chat"),
                   key=lambda r: r.get("t_start") or 0.0)

    cache = OllamaPromptCache(ram_bytes=ram_bytes, cost=cost, keep_gate=keep_gate)
    seen = set()                      # (model, prompt text) seen before
    tp = fp = fn = tn = 0
    kinds = {"exact": [0, 0], "partial": [0, 0], "none": [0, 0]}   # [measured, matched]
    per_role, per_config = {}, {}
    near_gate = 0
    for r in chats:
        cid = r["call_id"]
        if cid not in truth:
            continue
        text = render_prompt(prompts.get(cid, []))
        pt = r.get("prompt_tokens") or 0
        key = (r["model"], text)
        is_exact = key in seen
        seen.add(key)

        before = len(cache._f_keeps)
        reused = cache.access(r["model"], text, pt)
        if len(cache._f_keeps) > before:
            if abs(cache._f_keeps[-1] - keep_gate) <= 0.05:
                near_gate += 1

        pred = pt > 0 and (reused / pt) > hit_frac
        act = truth[cid]
        kind = "exact" if is_exact else ("partial" if reused > 0 else "none")
        kinds[kind][0] += int(act)
        kinds[kind][1] += int(act and pred)

        bucket = ("tp" if (pred and act) else "fp" if pred else
                  "fn" if act else "tn")
        tp += bucket == "tp"; fp += bucket == "fp"
        fn += bucket == "fn"; tn += bucket == "tn"
        for d, k in ((per_role, r["agent_role"]),
                     (per_config, config_of.get(cid, "unidentified"))):
            d.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})[bucket] += 1

    total = tp + fp + tn + fn
    return {
        "set": trace_set_name, "calls": total,
        "measured_hits": tp + fn, "predicted_hits": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "agreement": (tp + tn) / total if total else 0.0,
        "kinds": kinds, "near_gate": near_gate,
        "per_role": per_role, "per_config": per_config,
        "stats": dict(cache.stats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", type=float, default=KEEP_GATE,
                    help="f_keep gate (default: the measured 0.50)")
    args = ap.parse_args()
    for ts in ("original", "shared_first"):
        r = replay(ts, keep_gate=args.gate)
        print(f"=== {ts} ===")
        print(f"  calls {r['calls']}   measured hits {r['measured_hits']}   "
              f"predicted {r['predicted_hits']}   agreement {r['agreement']:.1%}")
        print(f"  tp={r['tp']} fp={r['fp']} fn={r['fn']} tn={r['tn']}")
        k = r["kinds"]
        print(f"  {'reuse source':<22}{'measured hits':>14}{'matched':>9}")
        for name, label in (("exact", "exact prompt repeat"),
                            ("partial", "partial prefix reuse"),
                            ("none", "no reuse available")):
            m, mt = k[name]
            print(f"  {label:<22}{m:>14}{mt:>9}")
        print(f"  calls with f_keep within +/-0.05 of the gate: {r['near_gate']}"
              f"  ({r['near_gate'] / max(1, r['calls']):.1%} -- char/token "
              f"approximation makes these uncertain)")
        print(f"  stats: {r['stats']}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
