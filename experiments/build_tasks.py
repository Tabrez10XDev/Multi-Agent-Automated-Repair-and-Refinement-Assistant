"""
Build the repair-task set from canonical HumanEval.

    python -m experiments.build_tasks --src <HumanEval.jsonl.gz> [--count 55]

SOURCE
    https://github.com/openai/human-eval/raw/master/data/HumanEval.jsonl.gz
    sha256 b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef
    164 problems, keys: task_id, prompt, canonical_solution, entry_point, test

HOW A TASK IS BUILT
    buggy_code  the problem's own `prompt` (imports, signature, docstring)
                plus a trivial wrong body: return the first argument, or a
                default of the annotated return type.
    issue       a short bug report from the docstring's first sentence, via one
                fixed template. No model involved.
    test_code   the problem's `check()` asserts, rewritten to the harness's
                `from target import f` / `def test_N()` form.

WHY STUB BODIES RATHER THAN MUTATION
    See docs/DECISIONS.md section 20.

EVERY TASK IS VALIDATED: its tests must FAIL on buggy_code and PASS on the
canonical solution. Anything that does not is dropped and listed.
"""

import argparse
import ast
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_URL = ("https://github.com/openai/human-eval/raw/master/data/"
              "HumanEval.jsonl.gz")
# The 15 problems the cache model was fit to. Excluded so the new trace is
# genuinely held out.
FIT_SET = {f"HumanEval/{i}" for i in range(1, 16)}

_DEFAULTS = {
    "bool": "False", "int": "0", "float": "0.0", "str": '""',
    "List": "[]", "list": "[]", "Dict": "{}", "dict": "{}",
    "Tuple": "()", "tuple": "()", "Optional": "None", "Any": "None",
}


def _return_annotation(prompt, entry_point):
    m = re.search(rf"def\s+{re.escape(entry_point)}\s*\([^)]*\)\s*->\s*([^:]+):",
                  prompt, re.S)
    return m.group(1).strip() if m else None


def _first_param(prompt, entry_point):
    m = re.search(rf"def\s+{re.escape(entry_point)}\s*\(([^)]*)\)", prompt, re.S)
    if not m or not m.group(1).strip():
        return None
    first = m.group(1).split(",")[0].strip()
    return first.split(":")[0].split("=")[0].strip() or None


def stub_body(prompt, entry_point):
    """A wrong body in the style of the existing tasks: plausible, not absurd."""
    ann = _return_annotation(prompt, entry_point)
    if ann:
        head = re.split(r"[\[\s]", ann, 1)[0]
        if head in _DEFAULTS:
            return f"    return {_DEFAULTS[head]}\n"
    p = _first_param(prompt, entry_point)
    return f"    return {p}\n" if p else "    return None\n"


def issue_text(prompt, entry_point):
    doc = re.search(r'"""(.*?)"""', prompt, re.S) or re.search(r"'''(.*?)'''", prompt, re.S)
    summary = ""
    if doc:
        body = doc.group(1).strip()
        body = body.split(">>>")[0]
        for line in body.split("\n"):
            line = line.strip()
            if line:
                summary = line
                break
    summary = re.sub(r"\s+", " ", summary).rstrip(".")
    if summary:
        return (f"The {entry_point} function is not implemented correctly. "
                f"It should: {summary[0].lower() + summary[1:]}.")
    return f"The {entry_point} function returns the wrong result."


def tests_from_check(test_src, entry_point):
    """
    Rewrite HumanEval's check() into the harness's test format.

    Problems whose check() is a flat list of asserts become one test per
    assert, matching the hand-written tasks. Anything with loops or locals is
    kept whole and driven by a single test, because splitting it would change
    what is asserted.
    """
    try:
        tree = ast.parse(test_src)
    except SyntaxError:
        return None
    check = next((n for n in tree.body
                  if isinstance(n, ast.FunctionDef) and n.name == "check"), None)
    if check is None:
        return None
    arg = check.args.args[0].arg if check.args.args else "candidate"

    preamble = [ast.unparse(n) for n in tree.body
                if isinstance(n, (ast.Import, ast.ImportFrom))]
    head = f"from target import {entry_point}\n"
    if preamble:
        head = "\n".join(preamble) + "\n" + head

    body = [n for n in check.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str))]
    if body and all(isinstance(n, ast.Assert) for n in body):
        out = [head]
        for i, n in enumerate(body):
            src = ast.unparse(n)
            src = re.sub(rf"\b{re.escape(arg)}\b", entry_point, src)
            out.append(f"def test_{i}(): {src}")
        return "\n".join(out) + "\n"

    src = ast.unparse(check)
    src = re.sub(rf"\b{re.escape(arg)}\b", "_f", src)
    return f"{head}\n{src}\n\ndef test_check(): check({entry_point})\n"


def _run(target_src, test_src):
    """True if pytest passes on this (target.py, test_target.py) pair."""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "target.py"), "w").write(target_src)
        open(os.path.join(d, "test_target.py"), "w").write(test_src)
        r = subprocess.run([sys.executable, "-m", "pytest", "-x", "-q",
                            "test_target.py", "-p", "no:cacheprovider"],
                           cwd=d, capture_output=True, text=True, timeout=120)
        return r.returncode == 0


def build(src_path, count):
    raw = open(src_path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(l) for l in gzip.decompress(raw).decode().split("\n") if l.strip()]

    cands = []
    for r in rows:
        if r["task_id"] in FIT_SET:
            continue
        prompt, ep = r["prompt"], r["entry_point"]
        tests = tests_from_check(r["test"], ep)
        if not tests:
            continue
        cands.append({
            "task_id": r["task_id"],
            "issue": issue_text(prompt, ep),
            "buggy_code": prompt.rstrip("\n") + "\n" + stub_body(prompt, ep),
            "test_code": tests,
            "_canonical": prompt.rstrip("\n") + "\n" + r["canonical_solution"],
            "_sol_lines": len([l for l in r["canonical_solution"].split("\n") if l.strip()]),
        })
    return digest, cands


def select(cands, count):
    """
    HumanEval/16-30 first (already in the harness), then a difficulty spread.

    Difficulty proxy is canonical-solution length in non-blank lines. The pool
    is split into terciles by that and sampled round-robin, so the set is not
    all one-liners -- a repair benchmark made only of trivial bodies would not
    exercise the debugger loop at all.
    """
    pinned = [c for c in cands if c["task_id"] in {f"HumanEval/{i}" for i in range(16, 31)}]
    rest = sorted((c for c in cands if c not in pinned), key=lambda c: c["_sol_lines"])
    if not rest:
        return pinned[:count]
    n = len(rest)
    tiers = [rest[:n // 3], rest[n // 3:2 * n // 3], rest[2 * n // 3:]]
    out, i = list(pinned), 0
    while len(out) < count and any(tiers):
        t = tiers[i % 3]
        if t:
            out.append(t.pop(len(t) // 2))
        i += 1
    return out[:count]


def held_out_ids(path):
    """Every problem already used by a held-out task file, kept or dropped.

    Dropped problems are excluded too: they were selected and looked at, so
    reusing them would not be untouched. Only the kept ids are written to the
    file, so the dropped ones are listed explicitly below."""
    with open(path) as fh:
        return {t["task_id"] for t in json.load(fh)["tasks"]}


# Selected for humaneval_new.json and dropped by validation (DECISIONS 20).
_NEW_DROPPED = {"HumanEval/32"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--count", type=int, default=55)
    ap.add_argument("--out", default=os.path.join(_REPO, "data", "tasks",
                                                  "humaneval_new.json"))
    ap.add_argument("--exclude", action="append", default=[],
                    help="task file whose problems must not be reused; "
                         "repeatable. Use for a set that must be untouched by "
                         "every earlier trace.")
    ap.add_argument("--all", action="store_true",
                    help="take every remaining candidate instead of --count")
    args = ap.parse_args()

    digest, cands = build(args.src, args.count)
    print(f"source sha256 {digest}")
    print(f"candidates after excluding the {len(FIT_SET)} fit-set problems "
          f"and unconvertible checks: {len(cands)}")

    excluded = set()
    for p in args.exclude:
        excluded |= held_out_ids(p)
        if os.path.basename(p) == "humaneval_new.json":
            excluded |= _NEW_DROPPED
    if excluded:
        cands = [c for c in cands if c["task_id"] not in excluded]
        print(f"after excluding {len(excluded)} problems from "
              f"{len(args.exclude)} earlier task file(s): {len(cands)}")

    chosen = cands if args.all else select(cands, args.count)
    print(f"selected {len(chosen)}; validating (tests must FAIL on the stub, "
          f"PASS on the canonical solution)...", flush=True)

    kept, dropped = [], []
    for c in chosen:
        why = None
        try:
            if _run(c["buggy_code"], c["test_code"]):
                why = "stub PASSES its own tests"
            elif not _run(c["_canonical"], c["test_code"]):
                why = "canonical solution FAILS its own tests"
        except subprocess.TimeoutExpired:
            why = "timeout"
        except Exception as e:
            why = f"{type(e).__name__}: {e}"
        if why:
            dropped.append((c["task_id"], why))
        else:
            kept.append({k: v for k, v in c.items() if not k.startswith("_")})
        print(("  drop " + c["task_id"] + ": " + why) if why
              else f"  ok   {c['task_id']}", flush=True)

    out = {
        "source_url": SOURCE_URL,
        "source_sha256": digest,
        "excluded_fit_set": sorted(FIT_SET, key=lambda s: int(s.split("/")[1])),
        "excluded_earlier": sorted(excluded, key=lambda s: int(s.split("/")[1])),
        "selection": ("every remaining candidate" if args.all else
                      "HumanEval/16-30 pinned, then round-robin over terciles "
                      "of canonical-solution length"),
        "validated": "tests fail on buggy_code and pass on canonical_solution",
        "tasks": kept,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nkept {len(kept)}, dropped {len(dropped)} -> {args.out}")
    for t, w in dropped:
        print(f"  dropped {t}: {w}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
