"""
A model of Ollama 0.33.3's prompt cache, and its validation against the traces.

STATUS: UNVALIDATED. Do not use this as a baseline yet.
Replaying the two committed traces, the best rule tried here predicts 56 of
89 measured cache hits on `original` and 20 of 78 on `shared_first`. It has
no false positives, so what it does predict is real, but it misses more than
it finds. The mechanism below is therefore incomplete. See docs/OPEN_ISSUES.

    python -m simulator.ollama_cache_model          # replay both committed traces

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

# Host prompt-cache budget. No --cache-ram flag is present on the command line,
# so this is llama.cpp's built-in default and is NOT established. PLACEHOLDER:
# swept during validation to see which value the traces imply.
DEFAULT_RAM_BYTES = 8 * 1024 ** 3


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
    ram_bytes: int = DEFAULT_RAM_BYTES
    cost: CostModel = field(default_factory=CostModel)

    # Does an in-place slot reuse destroy the displaced state, or is the state
    # always saved to RAM with the slot path merely taking priority over a
    # lookup? The affinity probe cannot tell these apart -- in it, A and B were
    # the same length and sat in symmetric positions, so both rules predict the
    # same recompute fractions. Only the trace separates them. Left as a flag so
    # the two can be scored side by side instead of one being assumed.
    save_on_reuse: bool = False

    def __post_init__(self):
        self._slots = {}        # model -> _Slot
        self._ram = {}          # model -> list[(text, tokens)], most recent last
        self._ram_bytes_used = 0
        self.stats = {"slot_reuse": 0, "ram_restore": 0, "cold": 0,
                      "ram_evictions": 0, "states_saved": 0}

    # -- RAM cache ---------------------------------------------------------

    def _state_bytes(self, model: str, tokens: int) -> int:
        return self.cost.kv_bytes(model, tokens)

    def _ram_save(self, model: str, text: str, tokens: int) -> None:
        if tokens <= 0:
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
        slot = self._slots.get(model)
        chars = max(1, len(text))

        if slot is not None and slot.text:
            shared = common_prefix_len(text, slot.text)
            if shared / chars >= self.sim_threshold:
                # Slot reused in place, and the RAM cache is not consulted --
                # the slot path wins whether or not a better match is saved.
                if self.save_on_reuse:
                    self._ram_save(model, slot.text, slot.tokens)
                self._slots[model] = _Slot(text, tokens)
                self.stats["slot_reuse"] += 1
                return int(tokens * shared / chars)
            # Below threshold: the displaced state is saved rather than lost.
            self._ram_save(model, slot.text, slot.tokens)

        shared, idx = self._ram_lookup(model, text)
        self._slots[model] = _Slot(text, tokens)
        if idx is not None and shared / chars >= self.sim_threshold:
            # Restored: move-to-back is handled by re-saving on next displace.
            saved_text, saved_tok = self._ram[model].pop(idx)
            self._ram_bytes_used -= self._state_bytes(model, saved_tok)
            self.stats["ram_restore"] += 1
            return int(tokens * shared / chars)

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
           hit_frac=1 / 3, save_on_reuse=False):
    """
    Replay a trace's calls in wall-clock order through the cache model.

    A predicted hit is a call whose reused fraction would make it clear the
    same 1.5x speedup the measured detector uses: reusing f of the prompt gives
    a 1/(1-f) speedup, so f > 1/3.
    """
    cost = CostModel()
    rows, prompts = [], {}
    for full in trace_set(trace_set_name):
        rows += load_rows(full)
        for x in load_rows(default_prompts_path(full)):
            prompts[x["call_id"]] = x["messages"]

    truth = measured_hits(rows, cost)
    # Config is a property of a task run, not of a call, so recover it by
    # segmenting and fingerprinting, then index it back by call.
    config_of = {}
    for block in segment_task_runs(rows):
        cfg = fingerprint_config(block) or "unidentified"
        for r in block:
            config_of[r["call_id"]] = cfg

    # The cache is global and stateful, so the replay order has to be the real
    # arrival order across every task run, not per-run.
    chats = sorted((r for r in rows if r.get("kind") == "chat"),
                   key=lambda r: r.get("t_start") or 0.0)

    cache = OllamaPromptCache(ram_bytes=ram_bytes, cost=cost,
                              save_on_reuse=save_on_reuse)
    tp = fp = tn = fn = 0
    per_role, per_model, per_config = {}, {}, {}
    for r in chats:
        cid = r["call_id"]
        if cid not in truth:
            continue
        text = render_prompt(prompts.get(cid, []))
        pt = r.get("prompt_tokens") or 0
        reused = cache.access(r["model"], text, pt)
        pred = pt > 0 and (reused / pt) > hit_frac
        act = truth[cid]
        bucket = ("tp" if (pred and act) else "fp" if pred else
                  "fn" if act else "tn")
        if bucket == "tp":
            tp += 1
        elif bucket == "fp":
            fp += 1
        elif bucket == "fn":
            fn += 1
        else:
            tn += 1
        for d, k in ((per_role, r["agent_role"]), (per_model, r["model"]),
                     (per_config, config_of.get(cid, "unidentified"))):
            d.setdefault(k, {"tp": 0, "fp": 0, "fn": 0, "tn": 0})[bucket] += 1

    total = tp + fp + tn + fn
    return {
        "set": trace_set_name, "ram_gib": ram_bytes / 1024 ** 3, "calls": total,
        "measured_hits": tp + fn, "predicted_hits": tp + fp,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "agreement": (tp + tn) / total if total else 0.0,
        "per_role": per_role, "per_model": per_model, "per_config": per_config,
        "stats": dict(cache.stats),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true",
                    help="sweep the RAM budget to see which the traces imply")
    args = ap.parse_args()

    budgets = ([0.5, 1, 2, 4, 8, 16] if args.sweep else [8])
    for ts in ("original", "shared_first"):
        print(f"=== {ts} ===")
        print(f"  {'RAM':>7}  {'calls':>6}{'measured':>10}{'predicted':>11}"
              f"{'tp':>5}{'fp':>5}{'fn':>5}{'agree':>8}")
        for gib in budgets:
            r = replay(ts, ram_bytes=int(gib * 1024 ** 3))
            print(f"  {gib:>5.1f}G  {r['calls']:>6}{r['measured_hits']:>10}"
                  f"{r['predicted_hits']:>11}{r['tp']:>5}{r['fp']:>5}{r['fn']:>5}"
                  f"{r['agreement']:>7.1%}")
        r = replay(ts, ram_bytes=int(budgets[-1] * 1024 ** 3))
        print(f"  cache stats: {r['stats']}")
        for dim in ("per_role", "per_config", "per_model"):
            label = dim.split("_")[1]
            print(f"  {label:<22}{'measured':>10}{'predicted':>11}"
                  f"{'tp':>5}{'fp':>5}{'fn':>5}")
            for k, d in sorted(r[dim].items()):
                print(f"  {str(k):<22}{d['tp']+d['fn']:>10}{d['tp']+d['fp']:>11}"
                      f"{d['tp']:>5}{d['fp']:>5}{d['fn']:>5}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
