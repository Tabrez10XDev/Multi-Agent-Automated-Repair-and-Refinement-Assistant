"""
Verify a collected trace saw only its sandbox.

On 2026-09-29 a traced benchmark run retrieved simulator source as "relevant
code" for HumanEval bugs, rewrote it, and wrote it to disk. The trace collected
alongside was contaminated and discarded. This is the gate that catches that
class of failure before hours are spent on a full run.

    python -m analysis.check_trace_clean                 # newest data/traces/smoke/*.prompts.jsonl
    python -m analysis.check_trace_clean <path>          # a specific .jsonl or .jsonl.gz

Exits 0 on PASS, 1 on FAIL.
"""

import argparse
import glob
import gzip
import json
import os
import re
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Files a sandboxed task is legitimately allowed to mention.
ALLOWED_PY = {"target.py", "test_target.py"}

# Substrings that can only come from outside the sandbox. Chosen to be
# unambiguous: "tracing.py" rather than "tracing", "sim/" rather than "sim".
#
# Both current and pre-Milestone-2 directory names are listed. The old ones
# must stay: the committed traces were collected under them, so dropping them
# would make this gate silently stop detecting the contamination it exists for.
FORBIDDEN = [
    "maara/", "simulator/", "probes/", "analysis/", "experiments/", "data/",
    "archive/", "tests/", "venv/",
    "sim/", "agents/", "benchmark/", "traces/",          # pre-Milestone-2
    "tracing.py",
    "CostModel", "prefill_rate", "kv_bytes_per_token", "contextvars",
    "MAARA_TRACE", "batch_decode_coeff", "_expected_further_attempts",
]

_PY_PATH = re.compile(r"[A-Za-z0-9_./-]+\.py")

# pytest prints a session banner into test_output, which the debugger then
# quotes verbatim in its prompt:
#
#   platform darwin -- ... -- /Users/.../self-healing-agent/venv/bin/python
#   rootdir: /Users/.../self-healing-agent/benchmark/exp_sandbox
#
# Those are artefacts of running the tests, not evidence that the agent saw
# repo source, and they matched "venv/" and "benchmark/" on the first pass.
# Stripped before scanning so the check stays strict about everything else
# rather than being loosened by deleting those two markers.
_BENIGN_PATH = re.compile(
    r"\S*/(?:venv|benchmark/exp_sandbox|benchmark/sandbox)(?:/\S*)?")


def _read_rows(path):
    """Read JSONL, gz or plain, tolerating a file still being written."""
    rows = []
    opener = gzip.open if path.endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass          # truncated final line of a live file
    except EOFError:
        pass                      # gz without an end-of-stream marker yet
    return rows


def _newest_smoke():
    hits = [p for p in glob.glob(os.path.join(_REPO, "data", "traces", "smoke", "*.prompts.jsonl*"))]
    return max(hits, key=os.path.getmtime) if hits else None


def check(path):
    rows = _read_rows(path)
    if not rows:
        print(f"FAIL  no readable prompt rows in {path}")
        return False

    marker_hits = {}
    path_hits = {}
    examples = {}

    for r in rows:
        text = " ".join(m.get("content", "") for m in r.get("messages", []))
        text = _BENIGN_PATH.sub(" ", text)
        for marker in FORBIDDEN:
            if marker in text:
                marker_hits[marker] = marker_hits.get(marker, 0) + 1
                examples.setdefault(marker, (r.get("agent_role"), r.get("task_id")))
        for m in _PY_PATH.findall(text):
            name = m.lstrip("./")
            if name not in ALLOWED_PY:
                path_hits[name] = path_hits.get(name, 0) + 1

    print(f"checked {len(rows)} prompt rows in {os.path.relpath(path, _REPO)}")
    print(f"  roles: {sorted({r.get('agent_role') for r in rows})}")
    print(f"  tasks: {len(({r.get('task_id') for r in rows}))}")

    ok = True
    if marker_hits:
        ok = False
        print("\n  out-of-sandbox content found:")
        for marker, n in sorted(marker_hits.items(), key=lambda kv: -kv[1]):
            role, task = examples[marker]
            print(f"    {marker:<28} {n:>4} rows   (e.g. {role} on {task})")
    if path_hits:
        ok = False
        print("\n  .py paths outside the sandbox file set:")
        for name, n in sorted(path_hits.items(), key=lambda kv: -kv[1])[:20]:
            print(f"    {name:<40} {n:>4} rows")

    print()
    if ok:
        print("PASS  every prompt stayed inside the sandbox "
              f"({', '.join(sorted(ALLOWED_PY))})")
    else:
        print("FAIL  the agent saw files outside its sandbox. Do not use this "
              "trace; fix isolation and re-run.")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", help="prompts sidecar (.jsonl or .jsonl.gz)")
    args = ap.parse_args()

    path = args.path or _newest_smoke()
    if not path:
        print("FAIL  no trace found under data/traces/smoke/. Run the smoke run first.")
        return 1
    if not os.path.exists(path):
        print(f"FAIL  no such file: {path}")
        return 1
    return 0 if check(path) else 1


if __name__ == "__main__":
    sys.exit(main())