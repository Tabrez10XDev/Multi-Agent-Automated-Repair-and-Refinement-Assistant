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

import argparse
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


def call(endpoint, prompt, label, quiet=False):
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
    if not quiet:
        print(f"  {label:<34} count={n:>6}  prefill={secs:7.3f}s  "
              f"{rate:8.1f} tok/s  load={load:6.3f}s")
    return n, secs


def slot_probe(endpoint="generate", k_max=8):
    """
    How many prompts does Ollama keep cached per model?

    For each K, build K distinct ~2,000-token prompts, send them all once to
    fill whatever cache exists, then send them all again in the same order. On
    the second pass every prompt is K calls old. If the server keeps N prompts,
    repeats hit while K <= N and stop hitting once K > N, because each prompt
    has been pushed out by the K-1 that followed it.

    Each prompt carries its own nonce as the FIRST token, so no two share any
    prefix at all -- otherwise a partial match would look like a slot hit.
    """
    print(f"--- slot probe on /api/{endpoint} ---")
    print(f"  {'K':>3}  {'fill p50':>10}  {'repeat p50':>11}  {'speedup':>8}  verdict")
    last_hit = 0
    for k in range(1, k_max + 1):
        prompts = [f"# probe k{k} slot {i} {random.randint(10**9, 10**10)}\n"
                   + PROMPT + LAST_ORIGINAL for i in range(k)]
        fill = [call(endpoint, p, f"k={k} fill {i}", quiet=True)[1]
                for i, p in enumerate(prompts)]
        rep = [call(endpoint, p, f"k={k} repeat {i}", quiet=True)[1]
               for i, p in enumerate(prompts)]
        f50, r50 = sorted(fill)[len(fill) // 2], sorted(rep)[len(rep) // 2]
        speedup = f50 / r50 if r50 else float("inf")
        hit = speedup > 3.0
        if hit:
            last_hit = k
        print(f"  {k:>3}  {f50:>9.3f}s  {r50:>10.3f}s  {speedup:>7.1f}x  "
              f"{'HIT' if hit else 'miss'}")
    print()
    if last_hit == 0:
        print("  => no repeat ever hit; caching is not per-prompt on this path")
    elif last_hit == k_max:
        print(f"  => still hitting at K={k_max}; N is at least {k_max}, raise k_max")
    else:
        print(f"  => repeats hit up to K={last_hit} and stop at K={last_hit + 1}, "
              f"so Ollama holds about N={last_hit} prompts per model")
    print()
    return last_hit


def capacity_probe(endpoint="generate", n=40):
    """
    How many saved prompt states does the host RAM cache hold?

    Ollama 0.33.3 runs ONE slot (-np 1) but saves displaced slot states to a
    host-RAM cache, which is why repeats kept hitting at every K up to 8. This
    finds where that cache ends.

    Send n distinct prompts once to fill it, then re-probe in REVERSE order.
    Under LRU the most recent survive, so the number of consecutive hits before
    the first miss is the capacity. Sweeping K from 1 would cost sum(2K) calls;
    this costs n + (hits) and stops as soon as it has the answer.
    """
    print(f"--- capacity probe on /api/{endpoint}, n={n} ---")
    prompts = [f"# cap {i} {random.randint(10**9, 10**10)}\n" + PROMPT + LAST_ORIGINAL
               for i in range(n)]
    fill = []
    for i, p in enumerate(prompts):
        fill.append(call(endpoint, p, f"fill {i}", quiet=True)[1])
        if i % 10 == 9:
            print(f"  filled {i+1}/{n}  (median cold {sorted(fill)[len(fill)//2]:.2f}s)",
                  flush=True)
    cold = sorted(fill)[len(fill) // 2]

    print(f"  cold prefill median {cold:.3f}s; re-probing newest-first", flush=True)
    survivors = 0
    for rank, i in enumerate(range(n - 1, -1, -1)):
        secs = call(endpoint, prompts[i], f"reprobe {i}", quiet=True)[1]
        hit = secs < cold / 3.0
        print(f"  rank {rank:>3} (prompt {i:>3})  {secs:7.3f}s  "
              f"{'HIT' if hit else 'MISS'}", flush=True)
        if not hit:
            break
        survivors += 1
    print()
    per_state_mib = 207.364 * (len(PROMPT) // 4) / 3791
    print(f"  => {survivors} of {n} states survived; the {survivors+1}th oldest was evicted")
    print(f"  at roughly {per_state_mib:.0f} MiB per state that is "
          f"~{survivors * per_state_mib / 1024:.1f} GiB of prompt cache")
    return survivors


def affinity_probe(endpoint="generate"):
    """
    Does a SIMILAR prompt overwrite the slot, or get its own state?

    Ollama reuses the active slot when prefix similarity clears 0.10
    ("checking sim = 0.002 > 0.100"), otherwise picks a slot by LRU. If a
    similar B reuses A's slot, A's state may be overwritten rather than saved,
    so A would stop hitting. Each level runs A, then B sharing that fraction of
    A's prefix, then A again.
    """
    print(f"--- affinity probe on /api/{endpoint} ---")
    print(f"  {'shared':>7}  {'A cold':>8}  {'B':>8}  {'A again':>8}  A survived?")
    for frac in (0.05, 0.25, 0.50, 0.90):
        nonce = f"# aff {random.randint(10**9, 10**10)}\n"
        a = nonce + PROMPT + LAST_ORIGINAL
        cut = int(len(a) * frac)
        b = a[:cut] + f"\n# divergent tail {random.randint(10**9, 10**10)}\n" + BODY[:400]
        t_a = call(endpoint, a, "A cold", quiet=True)[1]
        t_b = call(endpoint, b, "B", quiet=True)[1]
        t_a2 = call(endpoint, a, "A again", quiet=True)[1]
        survived = t_a2 < t_a / 3.0
        print(f"  {frac:>6.0%}  {t_a:>7.3f}s  {t_b:>7.3f}s  {t_a2:>7.3f}s  "
              f"{'yes' if survived else 'NO - overwritten'}", flush=True)
    print()
    print("  A surviving at every level => displaced states go to the RAM cache.")
    print("  A lost above 10% similarity => the slot is overwritten in place.")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", action="store_true",
                    help="also probe how many prompts are cached per model")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--capacity", type=int, default=0,
                    help="probe host prompt-cache capacity with N distinct prompts")
    ap.add_argument("--affinity", action="store_true",
                    help="probe whether a similar prompt overwrites the slot")
    ap.add_argument("--only", action="store_true",
                    help="skip the 4-call verdict and run only the chosen probes")
    args = ap.parse_args()
    print(f"model={MODEL}  keep_alive={KEEP_ALIVE}  prompt ~{len(PROMPT)//4} tokens\n")
    if args.affinity:
        affinity_probe("generate")
    if args.capacity:
        capacity_probe("generate", args.capacity)
    if args.only:
        return 0
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

    if args.slots:
        slot_probe("generate", args.k_max)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
