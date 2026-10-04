"""
Does Ollama reuse a cached prefix between separate chat calls?

CALLS THE MODEL. Not run automatically; invoke it yourself.

    ./venv/bin/python -m probes.ollama_cache_probe

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
probes/calibrate.py calibrated with /api/generate, and they may not share a
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

# ~165 tokens, matching the issue_parser prompts in the traces (median 164).
# Needed because the long and short cases disagree: the committed traces show
# ~165-token states surviving a >=10% overwrite, while the long-A affinity
# probe showed a ~3,790-token state being destroyed by one. Size is the most
# obvious variable neither experiment controlled, so this makes it the variable.
SHORT_BODY = "\n".join(
    f"def helper_{i}(x):\n    return x * {i + 1} + {i}"
    for i in range(13)
)
SHORT_PROMPT = (
    "You are an issue parser. Given the module below and a bug report, "
    "return a short structured plan naming the affected function.\n\n"
    f"```python\n{SHORT_BODY}\n```\n\n"
    "Bug report: helper_3 returns the wrong value for negative inputs.\n"
)
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


def affinity_probe(endpoint="generate", short=False, fracs=(0.05, 0.25, 0.50, 0.90)):
    """
    Does a similar prompt overwrite the slot, or is the displaced state saved?

    Each level runs A, then a full-length B sharing that fraction of A's
    prefix, then A again, and reports both recompute fractions:

        A again / A cold  ~ 0       A was saved and restored intact
        A again / A cold  ~ 1-share A's slot was reused in place and A paid
                                    for the part B did not share

    RESULT, 2026-10-04, with the full-length B (this is the trustworthy run):

                 5%     25%    50%    90%  shared
        long A    0%      0%     0%    11%  recompute
        short A   3%      4%     4%    14%  recompute

    Two conclusions, both of which overturned earlier beliefs:

    1. Size does not matter. A ~165-token A and a ~3,790-token A behave
       identically, so the "small states survive, large ones do not"
       hypothesis is dead.
    2. Crossing the 0.10 slot-similarity threshold does NOT destroy the
       displaced state. A came back intact at 25% and 50% shared, both well
       above 0.10. The boundary is somewhere between 50% and 90%.

    WITHDRAWN: an earlier version of this probe used a SHORT B, which shared
    a[:cut] with A and so only ever computed its own small divergent tail. That
    run reported A recomputing 76%/52%/11% at 25/50/90% and was read as "0.10
    destroys the slot". It was confounded, and that conclusion is retracted.
    See docs/FINDINGS.md section 19.

    llama-server's own strings are consistent with the corrected picture:
    "idle slots will be saved to prompt cache upon starting a new task",
    unconditionally, with eviction by "removing oldest entry".

    B is padded back to A's length with its own nonce after the shared prefix,
    so B's time is a second, independent check: if the slot is being reused,
    B should itself recompute about (1 - share) of a full prompt. An earlier
    version let B be a short prompt, so B took ~0.45s at every level -- the
    cost of its small divergent tail -- which looked like a RAM-cache restore
    and was not.
    """
    base = SHORT_PROMPT if short else PROMPT
    size = "SHORT (~165 tok, issue_parser scale)" if short else "LONG (~2k tok)"
    print(f"--- affinity probe on /api/{endpoint}  [A = {size}] ---")
    print(f"  {'shared':>7}  {'A cold':>9}  {'B':>9}  {'B frac':>7}  "
          f"{'A again':>9}  {'A frac':>7}  reading")
    rows = []
    filler = ((SHORT_BODY if short else BODY) + "\n") * 8
    for frac in fracs:
        a_txt = f"# aff A {random.randint(10**9, 10**10)}\n" + base + LAST_ORIGINAL
        cut = int(len(a_txt) * frac)
        tail_nonce = f"\n# aff B {random.randint(10**9, 10**10)}\n"
        pad = max(0, len(a_txt) - cut - len(tail_nonce))
        b_txt = a_txt[:cut] + tail_nonce + filler[:pad]

        t_a = call(endpoint, a_txt, "A cold", quiet=True)[1]
        t_b = call(endpoint, b_txt, "B", quiet=True)[1]
        t_a2 = call(endpoint, a_txt, "A again", quiet=True)[1]

        a_frac = t_a2 / t_a if t_a else float("nan")
        b_frac = t_b / t_a if t_a else float("nan")
        # Restored intact, or recomputed about the part B did not share?
        restored = a_frac < 0.10
        trimmed = abs(a_frac - (1.0 - frac)) < 0.15
        reading = ("restored from RAM" if restored else
                   "slot trimmed to shared prefix" if trimmed else "unclear")
        rows.append((frac, a_frac, b_frac, restored, trimmed))
        print(f"  {frac:>6.0%}  {t_a:>8.3f}s  {t_b:>8.3f}s  {b_frac:>6.0%}  "
              f"{t_a2:>8.3f}s  {a_frac:>6.0%}  {reading}", flush=True)

    print()
    # Do NOT assume 0.10 is where behaviour changes. The 2026-10-04 rerun with
    # a full-length B showed A restored intact at 5%, 25% AND 50% shared, and
    # recomputing only its unshared fraction at 90%. So the restore-to-trim
    # boundary lies somewhere above 0.50, not at 0.10, and the earlier verdict
    # text asserted a conclusion its own rows contradicted. Report the
    # transition the data shows and name what is still unpinned.
    restored = [r for r in rows if r[1] < 0.10]
    trimmed = [r for r in rows if r[1] >= 0.10]
    print(f"  restored intact (A again < 10% of cold): "
          f"{', '.join(f'{r[0]:.0%}' for r in restored) or 'none'}")
    print(f"  paid a recompute:                        "
          f"{', '.join(f'{r[0]:.0%}' for r in trimmed) or 'none'}")
    print()
    if restored and trimmed:
        lo = max(r[0] for r in restored)
        hi = min(r[0] for r in trimmed)
        tracks = all(abs(r[1] - (1.0 - r[0])) < 0.15 for r in trimmed)
        print(f"  => the restore-to-trim boundary lies between {lo:.0%} and {hi:.0%}.")
        print(f"     Below it the displaced state comes back intact from the host")
        print(f"     prompt cache. Above it the slot is reused in place and A pays")
        print(f"     {'its unshared fraction, which the numbers track.' if tracks else 'a recompute that does NOT track its unshared fraction.'}")
        if lo >= 0.10:
            print()
            print(f"     NOTE: A survived at {lo:.0%} shared, well above the 0.10 slot")
            print(f"     similarity threshold. So crossing 0.10 does NOT destroy the")
            print(f"     displaced state. llama-server logs 'idle slots will be saved")
            print(f"     to prompt cache upon starting a new task' unconditionally,")
            print(f"     which is consistent with that.")
    elif restored:
        print("  => A was restored intact at every level tried. The boundary, if")
        print("     there is one, is above the highest fraction tested.")
    elif trimmed:
        print("  => A paid a recompute at every level tried. The boundary, if")
        print("     there is one, is below the lowest fraction tested.")
    else:
        print("  => no rows; nothing to conclude.")
    print()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", action="store_true",
                    help="also probe how many prompts are cached per model")
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--capacity", type=int, default=0,
                    help="probe host prompt-cache capacity with N distinct prompts")
    ap.add_argument("--affinity", action="store_true",
                    help="probe whether a similar prompt overwrites the slot")
    ap.add_argument("--affinity-sweep", action="store_true",
                    help="sweep 60/70/80/85/95%% shared to pin the restore-to-trim "
                         "boundary; long A only, since size does not matter")
    ap.add_argument("--affinity-short", action="store_true",
                    help="same probe with a ~165-token A, matching issue_parser; "
                         "tests whether survival of an overwrite depends on size")
    ap.add_argument("--only", action="store_true",
                    help="skip the 4-call verdict and run only the chosen probes")
    args = ap.parse_args()
    print(f"model={MODEL}  keep_alive={KEEP_ALIVE}  prompt ~{len(PROMPT)//4} tokens\n")
    if args.affinity:
        affinity_probe("generate", short=False)
    if args.affinity_short:
        affinity_probe("generate", short=True)
    if args.affinity_sweep:
        affinity_probe("generate", short=False,
                       fracs=(0.60, 0.70, 0.80, 0.85, 0.95))
    if args.capacity:
        capacity_probe("generate", args.capacity)
    if args.only:
        if args.affinity and args.affinity_short:
            print("  COMPARE the two tables above. If the short A is restored")
            print("  (A frac < 10%) at 25/50/90% where the long A recomputed its")
            print("  unshared fraction, survival depends on state SIZE and the")
            print("  size-gated rule is confirmed. If the short A also recomputes")
            print("  its unshared fraction, the size hypothesis is dead and the")
            print("  bypassed exact matches in the traces need another cause.")
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
