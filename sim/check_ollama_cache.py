"""
Does Ollama reuse a cached prefix between separate chat calls?

CALLS THE MODEL. Not run automatically; invoke it yourself.

    ./venv/bin/python -m sim.check_ollama_cache

WHY THIS EXISTS
---------------
The trace-based detector infers a cache hit from prompt_eval_count dropping
below the prompt's true size. That assumes Ollama reports only the tokens it
recomputed. If it instead reports the full count while llama.cpp quietly reuses
the cached prefix, the detector sees nothing and we would wrongly conclude no
caching happens.

The duration check on the committed traces is suggestive but confounded: coder
(307 tok prompts) and debugger (790 tok) both prefill at 291-306 tok/s, which
matches the calibrated uncached rate of 268-292 -- no sign of reuse for exactly
the roles that have long shared prefixes. But issue_parser prefills at 2881
tok/s on 166-token prompts, ten times calibration, and nothing in the trace
explains it. Prompt size, role and shared-prefix length all covary there, so
the traces cannot settle the question.

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
        a = call(endpoint, PROMPT + LAST_ORIGINAL, "1. cold")
        b = call(endpoint, PROMPT + LAST_ORIGINAL, "2. identical prompt")
        c = call(endpoint, PROMPT + LAST_CHANGED, "3. last line changed")
        d = call(endpoint, PROMPT + LAST_ORIGINAL, "4. identical again (warm-up check)")

        print()
        cd = "yes" if b[0] < a[0] * 0.9 else "no"
        dd = "yes" if b[1] < a[1] * 0.7 else "no"
        print(f"  count drops on repeat?     {cd}")
        print(f"  duration drops on repeat?  {dd}")
        if cd == "no" and dd == "no":
            print("  => no cross-call prefix reuse visible on this endpoint")
        elif cd == "no":
            print("  => caching works but prompt_eval_count hides it; "
                  "the trace detector must use duration")
        else:
            print("  => prompt_eval_count reflects recomputed tokens; "
                  "the trace detector is valid")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
