"""
Isolation tests for the 2026-09-29 incident.

A traced benchmark run retrieved sim/workload.py as "relevant code" for a
HumanEval bug, rewrote it, and wrote it to disk, truncating the simulator and
tracing.py to stubs. These tests cover the three layers that now prevent it:
the shared exclusion list, the sandbox file set, and the write guard.

No LLM is called.
"""

import os

import pytest

from maara.sandbox import (EXCLUDED_DIRS, SandboxViolation, assert_within_root,
                          is_agent_visible)
from maara.agents.test_runner import apply_fix


# --- the exclusion list --------------------------------------------------

@pytest.mark.parametrize("path", [
    "sim/engine.py", "sim/workload.py", "sim/policies.py", "sim/measure_context.py",
    "tracing.py", "tests/conftest.py", "tests/test_provenance.py",
    "agents/coder.py", "benchmark/runner.py", "traces/x.py", "venv/lib/foo.py",
    "orchestrator.py", "agent.py",
])
def test_infrastructure_is_invisible_to_the_agent(path):
    assert is_agent_visible(path) is False, f"{path} must not be visible"


@pytest.mark.parametrize("path", ["target.py", "case_01.py", "src/thing.py"])
def test_task_code_is_visible(path):
    assert is_agent_visible(path) is True


def test_tests_dir_is_excluded_by_directory_not_filename():
    """
    The original filter tested relpath.startswith("test_"), so
    "tests/test_provenance.py" slipped through -- it starts with "tests/".
    Both spellings must now be excluded.
    """
    assert is_agent_visible("tests/test_provenance.py") is False
    assert is_agent_visible("tests/conftest.py") is False
    assert is_agent_visible("test_target.py") is False
    assert is_agent_visible("thing_tests.py") is False


def test_non_python_is_never_visible():
    assert is_agent_visible("README.md") is False
    assert is_agent_visible("data/calibration/measured.json") is False


def test_sim_is_actually_in_the_exclusion_list():
    """Guards against a future edit quietly dropping the directory that broke."""
    assert "sim/" in EXCLUDED_DIRS
    assert "tests/" in EXCLUDED_DIRS
    assert "traces/" in EXCLUDED_DIRS


# --- the write guard -----------------------------------------------------

def test_apply_fix_refuses_to_write_the_simulator(tmp_path):
    """The exact attack that happened: a fix targeting sim/engine.py."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "target.py").write_text("def f(): pass\n")

    victim = tmp_path / "sim" / "engine.py"
    victim.parent.mkdir()
    victim.write_text("ORIGINAL SIMULATOR CODE\n")

    with pytest.raises(SandboxViolation):
        apply_fix({"../sim/engine.py": "CLOBBERED"}, str(sandbox))

    assert victim.read_text() == "ORIGINAL SIMULATOR CODE\n", \
        "guard raised but the file was written anyway"


def test_apply_fix_refuses_absolute_paths(tmp_path):
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("SAFE\n")

    with pytest.raises(SandboxViolation):
        apply_fix({str(outside): "CLOBBERED"}, str(sandbox))
    assert outside.read_text() == "SAFE\n"


def test_apply_fix_refuses_deep_traversal(tmp_path):
    sandbox = tmp_path / "a" / "b" / "sandbox"
    sandbox.mkdir(parents=True)
    with pytest.raises(SandboxViolation):
        apply_fix({"../../../etc/passwd": "x"}, str(sandbox))


def test_apply_fix_allows_writes_inside_the_sandbox(tmp_path):
    """The guard must not break the normal path."""
    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    (sandbox / "target.py").write_text("old\n")

    backups = apply_fix({"target.py": "new\n"}, str(sandbox))

    assert (sandbox / "target.py").read_text() == "new\n"
    assert "target.py" in backups


def test_assert_within_root_returns_resolved_path(tmp_path):
    root = tmp_path / "r"
    root.mkdir()
    got = assert_within_root(str(root), str(root / "x.py"))
    assert got == os.path.realpath(str(root / "x.py"))


# --- the sandbox file set ------------------------------------------------

def test_sandbox_reproduces_the_original_available_files(tmp_path):
    """
    Comparability check. Before sim/ existed, orchestrator's available_files
    filter over the whole repo resolved to exactly ['target.py']. A sandbox of
    {target.py, test_target.py} must resolve to the same thing, or traces
    collected now are not comparable with the original 120 runs.
    """
    from maara.agents.navigator import get_file_structure, get_test_file_contents

    sandbox = tmp_path / "exp_sandbox"
    sandbox.mkdir()
    (sandbox / "target.py").write_text("def f(x):\n    return x\n")
    (sandbox / "test_target.py").write_text("def test_f(): assert True\n")

    structure = get_file_structure(str(sandbox))
    available = [f for f in structure.keys() if is_agent_visible(f)]
    assert available == ["target.py"]

    tests = get_test_file_contents(str(sandbox))
    assert sorted(tests.keys()) == ["test_target.py"]


# --- the trace-cleanliness gate ------------------------------------------

def _write_prompts(tmp_path, texts):
    import json
    p = tmp_path / "t.prompts.jsonl"
    with open(p, "w") as f:
        for i, t in enumerate(texts):
            f.write(json.dumps({
                "call_id": f"c{i}", "task_id": "HumanEval/1", "agent_role": "coder",
                "messages": [{"role": "user", "content": t}],
            }) + "\n")
    return str(p)


def test_check_passes_on_a_clean_trace(tmp_path):
    from sim.check_trace_clean import check
    path = _write_prompts(tmp_path, [
        "Current code in target.py:\n```python\ndef f(x): return x\n```",
        "Here are the test assertions:\ntest_target.py:\ndef test_f(): assert True",
    ])
    assert check(path) is True


def test_check_tolerates_the_pytest_banner(tmp_path):
    """
    Regression. pytest prints its interpreter path and rootdir into
    test_output, which the debugger quotes verbatim:

        -- /Users/x/self-healing-agent/venv/bin/python
        rootdir: /Users/x/self-healing-agent/benchmark/exp_sandbox

    Those matched "venv/" and "benchmark/" and failed a genuinely clean run.
    They are artefacts of running the tests, not leaked source.
    """
    from sim.check_trace_clean import check
    path = _write_prompts(tmp_path, [
        "Test output showing what still fails:\n"
        "platform darwin -- Python 3.13.3, pytest-9.0.3 -- "
        "/Users/x/self-healing-agent/venv/bin/python\n"
        "rootdir: /Users/x/self-healing-agent/benchmark/exp_sandbox\n"
        "test_target.py::test_empty PASSED",
    ])
    assert check(path) is True


def test_check_still_fails_on_real_leakage(tmp_path):
    """The banner fix must not blunt the actual detection."""
    from sim.check_trace_clean import check
    assert check(_write_prompts(tmp_path, [
        "Current code in sim/workload.py:\n```python\nclass CostModel:\n    pass\n```",
    ])) is False
    assert check(_write_prompts(tmp_path, [
        "def prefill_seconds(self): return self.prefill_rate",
    ])) is False
    assert check(_write_prompts(tmp_path, [
        "Current code in tracing.py: import contextvars",
    ])) is False
