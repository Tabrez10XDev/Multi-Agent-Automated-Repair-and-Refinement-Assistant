"""
Does Ollama reuse a cached prefix between separate chat calls?

CALLS THE MODEL. Not run automatically; invoke it yourself.

    ./venv/bin/python -m sim.check_ollama_cache

RESULT (measured 2026-09-30)
---------------------------
Ollama DOES reuse cached prefixes, and prompt_eval_count hides it completely:

    /api/generate  cold                 13.400s   281 tok/s
                   identical repeat      0.040s   -> 335x faster
                   last line changed     0.074s   -> 181x faster
                   prompt_eval_count CONSTANT throughout

So every count-based conclusion drawn before this was wrong, including the
"zero cache hits" reported for both committed traces. The trace detector has
been rewritten around prefill DURATION against the calibrated uncached rate.

It also explains the anomaly that the traces could not: issue_parser prefilled
at 2881 tok/s against a calibrated 268, because consecutive issue_parser calls
share a long fixed preamble and were hitting Ollama's cache. coder and debugger
ran at ~300 tok/s -- the uncached rate -- because a call on the other model
lands between them and evicts the slot.

WHAT THIS DOES
--------------
Three calls, same model, back to back, with keep_alive so the weights stay
resident:

    1. a ~2,000 token prompt, cold
    2. the identical prompt again          -> a full-prefix hit if caching works
    3. the same prompt with only the LAST line changed
                                           -> a near-full-prefix hit

and prints prompt_eval_count and prompt_eval_duration for each. Interpretation:

    count drops on 2 and 3        Ollama reports only recomputed tokens;
                                  the trace detector is valid as written.
    count constant, duration drops  caching works but is invisible to the
                                  count; the trace detector must be rewritten
                                  around duration.
    neither drops                 no cross-call prefix reuse on this path.

Run 1 is repeated at the end to check the result is not just warm-up: if the
fourth call is as fast as the second, the speedup was never about the prefix.

Both endpoints are exercised because the harness uses /api/chat while
sim/measure_context.py calibrated with /api/generate, and they may not share a
cache path.
"""

import json
import random
import urllib.request

OLLAMA = "http://localhost:11434"
MODEL = "qwen2.5-coder:7b"
KEEP_ALIVE = "10m"

# ~2,000 tokens of plausible code context. Deterministic so the run repeats.
BODY = "\n\n".join(
    f"def function_{i}(values):\n"
    f"    \"\"\"Compute a rolling summary for bucket {i}.\"\"\"\n"
    f"    total = 0\n"
    f"    for index, value in enumerate(values):\n"
    f"        if value is None:\n"
    f"            continue\n"
    f"        total += value * {i + 1}\n"
    f"    return total / max(1, len(values))"
    for i in range(60)
)
PROMPT = f"Here is a module under review:\n\n```python\n{BODY}\n```\n"
LAST_ORIGINAL = "Summarise the module in one sentence."
LAST_CHANGED = "Summarise the module in two sentences."


def call(endpoint, prompt, label):
    if endpoint == "chat":
        payload = {"model": MODEL, "stream": False, "keep_alive": KEEP_ALIVE,
                   "options": {"num_predict": 1, "temperature": 0, "num_ctx": 8192},
                   "messages": [{"role": "user", "content": prompt}]}
        url = f"{OLLAMA}/api/chat"
    else:
        payload = {"model": MODEL, "stream": False, "raw": True,
                   "keep_alive": KEEP_ALIVE, "prompt": prompt,
                   "options": {"num_predict": 1, "temperature": 0, "num_ctx": 8192}}
        url = f"{OLLAMA}/api/generate"

    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())

    n = d.get("prompt_eval_count") or 0
    secs = (d.get("prompt_eval_duration") or 0) / 1e9
    load = (d.get("load_duration") or 0) / 1e9
    rate = n / secs if secs else float("nan")
    print(f"  {label:<34} count={n:>6}  prefill={secs:7.3f}s  "
          f"{rate:8.1f} tok/s  load={load:6.3f}s")
    return n, secs


def main():
    print(f"model={MODEL}  keep_alive={KEEP_ALIVE}  prompt ~{len(PROMPT)//4} tokens\n")
    for endpoint in ("chat", "generate"):
        print(f"--- /api/{endpoint} ---")
        # A unique preamble makes call 1 genuinely cold. Without it the first
        # /api/chat call inherited the cache left by the /api/generate block
        # above and clocked 0.040s, so the "cold" baseline was already warm and
        # the verdict came out backwards.
        nonce = f"# session {random.randint(10**9, 10**10)}\n"
        cold_prompt = nonce + PROMPT + LAST_ORIGINAL
        a = call(endpoint, cold_prompt, "1. cold (unique preamble)")
        b = call(endpoint, cold_prompt, "2. identical prompt")
        c = call(endpoint, nonce + PROMPT + LAST_CHANGED, "3. last line changed")
        d = call(endpoint, cold_prompt, "4. identical again")

        print()
        count_drops = b[0] < a[0] * 0.9
        dur_drops = b[1] < a[1] * 0.5
        tail_reuse = c[1] < a[1] * 0.5
        print(f"  count drops on an identical repeat?   "
              f"{'yes' if count_drops else 'no'}  ({a[0]} -> {b[0]})")
        print(f"  duration drops on that repeat?        "
              f"{'yes' if dur_drops else 'no'}  ({a[1]:.3f}s -> {b[1]:.3f}s)")
        print(f"  duration drops when only the tail changed? "
              f"{'yes' if tail_reuse else 'no'}  ({a[1]:.3f}s -> {c[1]:.3f}s)")
        print(f"  4th call consistent with the 2nd?     "
              f"{'yes' if abs(d[1] - b[1]) < max(0.05, 0.5 * b[1]) else 'no'}")
        print()
        if dur_drops and not count_drops:
            print("  => PREFIX CACHING IS REAL and prompt_eval_count hides it.")
            print("     Any trace detector built on the count is blind; it must")
            print("     use prefill duration against the uncached rate.")
        elif count_drops:
            print("  => prompt_eval_count reflects only recomputed tokens, so a")
            print("     count-based trace detector is valid.")
        else:
            print("  => no cross-call prefix reuse visible on this endpoint.")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
