"""
Calibration harness for the simulator.

Everything in sim/workload.py that is a *number about the real world* should come
from here, not from a guess. Run this, and it writes sim/measured.json, which
workload.py picks up automatically.

    python -m sim.measure_context              # full run
    python -m sim.measure_context --quick      # skip rate measurement

What it measures:
  1. Token counts of the REAL coder prompt, split at the stable/variable
     boundary. This is the number that sets the whole memory-vs-recompute scale.
  2. prefill_rate / decode_rate per model, from ollama's own timing counters.
  3. Wall-clock of the two non-GPU stages (pytest, RAG index build), which is
     what sets the reuse gap the retention policy is betting on.
"""

import argparse
import json
import os
import random
import subprocess
import sys
import time
import urllib.error
import urllib.request

OLLAMA = "http://localhost:11434"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "measured.json")

CODER_MODEL = "qwen2.5-coder:7b"
SENTINEL_MODEL = "llama3.2"


# Ollama's default num_ctx is 2048 and it SILENTLY TRUNCATES longer prompts.
# Without setting this explicitly, a 6000-token prompt reports ~2050 tokens and
# every rate derived from it is nonsense.
DEFAULT_NUM_CTX = 8192


def _generate(model, prompt, num_predict, timeout=600, num_ctx=DEFAULT_NUM_CTX):
    """One raw ollama call. Returns the full response dict including counters."""
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "raw": True,
        "options": {"num_predict": num_predict, "temperature": 0, "num_ctx": num_ctx},
    }).encode()
    req = urllib.request.Request(
        f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def count_tokens(model, text, num_ctx=DEFAULT_NUM_CTX):
    """Exact token count via ollama's prompt_eval_count. No local tokenizer needed."""
    return _generate(model, text, num_predict=1, num_ctx=num_ctx)["prompt_eval_count"]


def measure_chars_per_token(model, text, chunk_chars=16000):
    """
    Empirical chars/token on real code from the repo being measured.

    Used to size prompts too large to prefill directly: a 108KB test file is
    ~27k tokens, and actually prefilling that costs ~100s and 1.5GB of KV just
    to learn a number. Measuring the ratio on a chunk and scaling is accurate
    to a few percent for homogeneous text like a single test file.
    """
    chunk = text[:chunk_chars]
    n = count_tokens(model, chunk)
    return len(chunk) / n


def measure_rates(model):
    """
    prefill_rate and decode_rate in tokens/sec, from ollama's nanosecond counters.

    Prefill is measured at several prompt lengths, because a single length cannot
    distinguish a genuine throughput rate from a fixed per-call overhead. If the
    reported rate climbs steeply with length, most of what a short-prompt
    measurement captured was overhead, not prefill.

    Each prompt gets a unique random preamble: ollama does its own prefix
    caching, and a repeated prompt would otherwise report a fantasy rate.

    These are single-stream rates on whatever backend is running now. On the
    author's Mac that is Metal; treat as placeholders pending CUDA measurement.
    """
    print(f"  [{model}] warming up (loads weights into memory)...")
    _generate(model, "warmup", num_predict=8)

    prefill_points = []
    for reps in (50, 200, 700):
        nonce = f"# {random.randint(10**9, 10**10)}\n"
        prompt = nonce + ("def f(x):\n    return x + 1\n\n" * reps)
        r = _generate(model, prompt, num_predict=1)
        n = r["prompt_eval_count"]
        secs = r["prompt_eval_duration"] / 1e9
        prefill_points.append({"tokens": n, "seconds": round(secs, 3),
                               "rate": round(n / secs, 2)})
        print(f"  [{model}] prefill {n} tok in {secs:.2f}s -> {n/secs:.1f} tok/s")

    # Slope between the smallest and largest point strips the fixed overhead.
    lo, hi = prefill_points[0], prefill_points[-1]
    marginal = (hi["tokens"] - lo["tokens"]) / (hi["seconds"] - lo["seconds"])

    print(f"  [{model}] measuring decode...")
    r = _generate(model, "Write a long Python tutorial about list comprehensions.",
                  num_predict=200)
    decode_rate = r["eval_count"] / (r["eval_duration"] / 1e9)
    print(f"  [{model}] decode {r['eval_count']} tok -> {decode_rate:.1f} tok/s")

    return {
        # Marginal rate is the honest one for the cost model: recomputing an
        # evicted prefix adds tokens to an existing call, it does not pay the
        # fixed per-call overhead again.
        "prefill_rate": round(marginal, 2),
        "prefill_rate_naive_largest": hi["rate"],
        "prefill_points": prefill_points,
        "decode_rate": round(decode_rate, 2),
        "decode_sample_tokens": r["eval_count"],
    }


# --- Reconstructing the real coder prompt -----------------------------------
#
# This mirrors agents/coder.py:113-131 exactly. If that prompt changes, this
# must change with it, or the calibration silently drifts.

def build_coder_prompt_parts(relevant_code, test_files, issue, filename, source_code,
                             previous_attempt=None, diagnosis=None):
    """
    Returns (stable_prefix, variable_suffix) as they appear in the real prompt.

    The split point matters and is subtle. coder.py lays the prompt out as:
        issue fields -> test_context -> "Current code in {filename}" -> retry_context
    and orchestrator.py:160-161 rewrites relevant_code[original_file] after every
    failed attempt. So the source block is NOT stable across retries even though
    it sits in the middle of the prompt. The durable shared prefix therefore ends
    immediately before "Current code in".
    """
    test_context = ""
    if test_files:
        test_context = "\n\nHere are the actual test assertions you must satisfy:\n"
        for test_filename, test_source in test_files.items():
            test_context += f"\n{test_filename}:\n```python\n{test_source}\n```"

    stable_prefix = f"""You are a Python debugging expert.

Issue to fix:
{issue['intent']}

Expected behavior:
{issue['expected_behavior']}

Edge cases to handle:
{', '.join(issue.get('test_hints', []))}
{test_context}
"""

    if previous_attempt and diagnosis:
        retry_context = f"""
A previous fix attempt failed.

Previous fix:
```python
{previous_attempt}
```

Diagnosis of why it failed:
{diagnosis}

Use this diagnosis to guide a different approach."""
    else:
        retry_context = "This is the first attempt."

    variable_suffix = f"""
Current code in {filename}:
```python
{source_code}
```

{retry_context}
"""
    return stable_prefix, variable_suffix


# A representative parsed issue. Shape matches what agents/issue_parser.py
# returns for the default issue in orchestrator.py's __main__ block.
REPRESENTATIVE_ISSUE = {
    "intent": "Fix calculate_average, reverse_string, and count_vowels in target.py so they handle empty input and full-string/case-insensitive behavior correctly.",
    "affected_files": ["target.py"],
    "expected_behavior": "calculate_average returns 0.0 for an empty list; reverse_string returns the entire reversed string; count_vowels counts both uppercase and lowercase vowels.",
    "test_hints": [
        "empty list input to calculate_average",
        "single character string in reverse_string",
        "uppercase vowels in count_vowels",
        "empty string input",
    ],
}


def measure_prompt_sizes(model):
    """Token counts for the real coder prompt, split stable vs variable."""
    from agents.navigator import get_test_file_contents

    test_files = get_test_file_contents(REPO)
    target_path = os.path.join(REPO, "target.py")
    with open(target_path) as f:
        source_code = f.read()

    stable, variable = build_coder_prompt_parts(
        relevant_code={"target.py": source_code},
        test_files=test_files,
        issue=REPRESENTATIVE_ISSUE,
        filename="target.py",
        source_code=source_code,
    )

    # Retry-attempt version carries a previous fix + a diagnosis paragraph.
    _, variable_retry = build_coder_prompt_parts(
        relevant_code={"target.py": source_code},
        test_files=test_files,
        issue=REPRESENTATIVE_ISSUE,
        filename="target.py",
        source_code=source_code,
        previous_attempt=source_code,
        diagnosis=("The fix returned None for the empty-list case instead of 0.0, and "
                   "reverse_string still sliced only the final character. The root cause "
                   "is that the guard clause was placed after the return statement."),
    )

    print("  counting tokens (exact, via ollama prompt_eval_count)...")
    return {
        "stable_prefix_tokens": count_tokens(model, stable),
        "variable_suffix_tokens": count_tokens(model, variable),
        "variable_suffix_retry_tokens": count_tokens(model, variable_retry),
        "test_files_included": sorted(test_files.keys()),
        "stable_prefix_chars": len(stable),
        "note": ("Measured against this repo's target.py/test_target.py, which are small. "
                 "A realistic SWE-bench-scale task will have a much larger stable prefix; "
                 "re-run against a representative benchmark case before drawing conclusions."),
    }


# A realistic issue body. SWE-bench issues are paragraphs with reproduction
# steps and tracebacks, not the one-line descriptions in benchmark/cases.json.
REALISTIC_ISSUE = {
    "intent": ("Requests raises an unexpected exception when a redirect chain includes a "
               "scheme change from https to http, instead of stripping the Authorization "
               "header and continuing the redirect."),
    "affected_files": ["src/requests/sessions.py"],
    "expected_behavior": ("When following a redirect that downgrades the scheme, the "
                          "Authorization header must be removed from the outgoing request "
                          "and the redirect should proceed normally rather than raising. "
                          "Cookies set for the original host must not leak to the new host."),
    "test_hints": [
        "redirect from https to http strips Authorization",
        "redirect preserving scheme keeps Authorization",
        "cross-host redirect drops host-scoped cookies",
        "redirect chain longer than max_redirects still raises TooManyRedirects",
    ],
}


def measure_realistic_repo(repo_path, model):
    """
    Measure the stable-prefix size MAARA would build on a real repository.

    The self-healing-agent repo is toy-scale (a 268-byte target.py), so the
    480-token stable prefix measured there says more about the test fixture than
    about the design. This re-measures against a repo SWE-bench actually uses.
    """
    from agents.navigator import get_test_file_contents

    out = {"repo": os.path.basename(os.path.abspath(repo_path))}

    # 1. What MAARA does TODAY. get_test_file_contents uses os.listdir, so it
    #    only ever sees the repo root.
    root_tests = get_test_file_contents(repo_path)
    out["maara_root_only"] = {
        "files": sorted(root_tests.keys()),
        "count": len(root_tests),
        "total_chars": sum(len(v) for v in root_tests.values()),
    }

    # 2. What is actually there, recursively.
    all_tests = {}
    for root, dirs, files in os.walk(repo_path):
        dirs[:] = [d for d in dirs if not d.startswith(".") and d not in
                   ("venv", "__pycache__", "node_modules", "build", "dist")]
        for fn in files:
            if fn.endswith(".py") and (fn.startswith("test_") or fn.endswith("_tests.py")):
                p = os.path.join(root, fn)
                try:
                    with open(p, encoding="utf-8", errors="replace") as f:
                        all_tests[os.path.relpath(p, repo_path)] = f.read()
                except OSError:
                    pass
    out["recursive"] = {
        "count": len(all_tests),
        "total_chars": sum(len(v) for v in all_tests.values()),
        "largest": sorted(((len(v), k) for k, v in all_tests.items()), reverse=True)[:5],
    }

    if not all_tests:
        return out

    # Ratio measured on this repo's own test code, not assumed.
    biggest = max(all_tests.values(), key=len)
    cpt = measure_chars_per_token(model, biggest)
    out["chars_per_token"] = round(cpt, 3)

    def prefix_tokens_for(test_subset):
        """Exact where it fits in context, scaled by the measured ratio beyond."""
        stable, _ = build_coder_prompt_parts(
            relevant_code={}, test_files=test_subset, issue=REALISTIC_ISSUE,
            filename="sessions.py", source_code="")
        if len(stable) < 24000:
            return {"tokens": count_tokens(model, stable, num_ctx=16384),
                    "method": "exact", "chars": len(stable)}
        return {"tokens": int(len(stable) / cpt), "method": "scaled-by-measured-ratio",
                "chars": len(stable)}

    scenarios = {}
    # (a) One focused test file, matched to the module under repair. This is what
    #     a competent agent would put in the prompt.
    focused = None
    for name in ("tests/test_requests.py", "tests/test_utils.py"):
        if name in all_tests:
            focused = name
            break
    if focused is None:
        focused = max(all_tests, key=lambda k: len(all_tests[k]))
    scenarios["focused_single_test_file"] = dict(
        file=focused, **prefix_tokens_for({focused: all_tests[focused]}))

    # (b) A mid-sized test file, for the low end of the realistic range.
    mid = sorted(all_tests, key=lambda k: len(all_tests[k]))[len(all_tests) // 2]
    scenarios["median_test_file"] = dict(
        file=mid, **prefix_tokens_for({mid: all_tests[mid]}))

    # (c) MAARA's actual policy -- inline EVERY test file it found -- applied to
    #     the recursive set. Included to show it is not viable at real scale.
    scenarios["all_test_files"] = dict(file=f"{len(all_tests)} files",
                                       **prefix_tokens_for(all_tests))
    out["scenarios"] = scenarios
    return out


def measure_cpu_stages():
    """Wall-clock of the two non-GPU stages. These define the reuse gap."""
    out = {}

    print("  timing pytest (apply_and_test)...")
    t0 = time.time()
    subprocess.run(
        [sys.executable, "-m", "pytest", "test_target.py", "-v", "--tb=short"],
        capture_output=True, text=True, cwd=REPO, timeout=60,
    )
    out["pytest_seconds"] = round(time.time() - t0, 3)

    print("  timing RAG index build (navigate)...")
    try:
        from agents.retriever import build_index
        t0 = time.time()
        build_index(REPO)
        out["rag_index_seconds"] = round(time.time() - t0, 3)
    except Exception as e:
        out["rag_index_seconds"] = None
        out["rag_index_error"] = str(e)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="skip per-model rate measurement")
    ap.add_argument("--rates-only", action="store_true",
                    help="re-measure model rates, keep existing prompt/CPU numbers")
    ap.add_argument("--repo", help="measure stable-prefix size against a real repo "
                                   "instead of this one; keeps other measurements")
    args = ap.parse_args()

    sys.path.insert(0, REPO)

    try:
        urllib.request.urlopen(f"{OLLAMA}/api/tags", timeout=5)
    except (urllib.error.URLError, OSError):
        print("ollama is not reachable at localhost:11434 — start it and re-run.")
        return 1

    result = {"measured_at": time.strftime("%Y-%m-%d %H:%M:%S"), "backend": "mac-metal"}

    if args.repo:
        try:
            with open(OUT_PATH) as f:
                result = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        print(f"Measuring realistic repo: {args.repo}")
        result.setdefault("realistic_repos", {})
        r = measure_realistic_repo(args.repo, CODER_MODEL)
        result["realistic_repos"][r["repo"]] = r
        with open(OUT_PATH, "w") as f:
            json.dump(result, f, indent=2)
        print(json.dumps(r, indent=2))
        return 0

    if args.rates_only:
        try:
            with open(OUT_PATH) as f:
                result = json.load(f)
            result["measured_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        except (FileNotFoundError, json.JSONDecodeError):
            print("no existing measured.json to extend; run without --rates-only first.")
            return 1
    else:
        print("Measuring prompt sizes...")
        result["prompt_sizes"] = measure_prompt_sizes(CODER_MODEL)

        print("Measuring CPU stage durations...")
        result["cpu_stages"] = measure_cpu_stages()

    if not args.quick:
        print("Measuring model rates...")
        result["rates"] = {
            CODER_MODEL: measure_rates(CODER_MODEL),
            SENTINEL_MODEL: measure_rates(SENTINEL_MODEL),
        }

    with open(OUT_PATH, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {OUT_PATH}")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
