"""
One definition of what the coding agent is allowed to see and touch.

WHY THIS EXISTS
---------------
The same exclusion list used to be written out three times -- in
orchestrator.py (which files to offer the issue parser), retriever.py (what to
index) and navigator.py (what to hand the coder). Each copy knew about
`agents/` and `benchmark/`; none knew about directories added later.

On 2026-09-29 that gap let a traced benchmark run retrieve `sim/workload.py` as
"relevant code" for a HumanEval bug, rewrite it, and write the rewrite to disk.
The simulator and tracing.py were truncated from 1,881 lines to stubs. Nothing
was lost because it was committed, but the trace collected alongside it was
contaminated and had to be discarded.

So: one list, one predicate, three call sites. Adding a directory to the repo
should never again require remembering three places.

Note this is defence in depth. The primary fix is that benchmark runs now
execute against an isolated sandbox containing only the task's own files, so
these filters have nothing to exclude in the first place. They matter for the
non-sandboxed path (`python orchestrator.py` against the repo itself).
"""

import os

# Directory prefixes the agent must never read, index, or write.
EXCLUDED_DIRS = (
    "agents/",        # the agent's own implementation
    "benchmark/",     # harness, cases, results, sandboxes
    "sim/",           # the simulator
    "tests/",         # test suite
    "traces/",        # collected trace data
    "results/",       # simulator outputs
    "venv/",
    ".git/",
    "__pycache__/",
)

# Individual files at any level the agent must never touch.
EXCLUDED_FILES = (
    "orchestrator.py",
    "agent.py",
    "tracing.py",
    "conftest.py",
)


class SandboxViolation(RuntimeError):
    """A write was attempted outside the sandbox root."""


def _is_test_file(name: str) -> bool:
    return name.startswith("test_") or name.endswith("_tests.py")


def is_agent_visible(relpath: str) -> bool:
    """
    True if the coding agent may read, index, or modify this file.

    `relpath` is relative to the repo (or sandbox) root, using forward slashes.

    Checks the basename for test-file naming rather than the whole path: the
    old orchestrator filter tested `relpath.startswith("test_")`, so
    "tests/test_provenance.py" slipped through because the path starts with
    "tests/", not "test_".
    """
    if not relpath.endswith(".py"):
        return False
    norm = relpath.replace(os.sep, "/").lstrip("./")
    if any(norm == d.rstrip("/") or norm.startswith(d) for d in EXCLUDED_DIRS):
        return False
    base = os.path.basename(norm)
    if base in EXCLUDED_FILES or norm in EXCLUDED_FILES:
        return False
    if _is_test_file(base):
        return False
    return True


def assert_within_root(root: str, path: str) -> str:
    """
    Resolve `path` and confirm it stays inside `root`. Returns the resolved path.

    Raises SandboxViolation for absolute paths, `..` traversal, and symlinks
    that escape -- realpath resolves all three before the comparison, which
    string prefix checks would not catch.
    """
    root_r = os.path.realpath(root)
    path_r = os.path.realpath(path)
    try:
        common = os.path.commonpath([root_r, path_r])
    except ValueError:
        raise SandboxViolation(
            f"refusing to write {path!r}: not comparable to root {root!r}")
    if common != root_r:
        raise SandboxViolation(
            f"refusing to write outside the sandbox\n"
            f"  root:     {root_r}\n"
            f"  resolved: {path_r}\n"
            f"This guard exists because an unguarded write overwrote sim/ and "
            f"tracing.py during a benchmark run.")
    return path_r
