"""
Tests for the prompt sidecar and the results-path override.

No LLM is called. The sidecar is what Phase 2 needs to recover reusable prefix
sizes, so the property that actually matters is that the messages land on disk
byte-identical to what was sent -- a lossy copy would silently corrupt every
prefix measurement downstream.
"""

import gzip
import json
import os

import pytest

import tracing


def _fake_response():
    return {
        "message": {"content": "ok"},
        "prompt_eval_count": 100,
        "eval_count": 50,
        "prompt_eval_duration": 1_000_000_000,
        "eval_duration": 2_000_000_000,
    }


@pytest.fixture
def traced(tmp_path, monkeypatch):
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "testrun")
    tracing._reset()
    yield
    tracing._reset()


def _main_rows(tmp_path):
    p = tmp_path / "smoke" / "testrun.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


def _prompt_rows(tmp_path):
    p = tmp_path / "smoke" / "testrun.prompts.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


MESSAGES = [
    {"role": "system", "content": "You are a Python debugging expert."},
    {"role": "user", "content": "Issue to fix:\nreverse_string is wrong\n```python\nx=1\n```"},
]


# --- sidecar -------------------------------------------------------------

def test_sidecar_stores_messages_verbatim(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="qwen", messages=MESSAGES, role="coder")
    rows = _prompt_rows(tmp_path)
    assert len(rows) == 1
    # Byte-identical, including newlines and code fences: Phase 2 measures a
    # longest common prefix on these strings.
    assert rows[0]["messages"] == MESSAGES


def test_sidecar_joins_to_main_trace_on_call_id(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="qwen", messages=MESSAGES, role="coder")
        tracing.chat(model="llama", messages=MESSAGES, role="evaluator")
    main = _main_rows(tmp_path)
    side = _prompt_rows(tmp_path)
    assert [r["call_id"] for r in main] == [r["call_id"] for r in side]
    assert {r["call_id"] for r in side} == {r["call_id"] for r in main}


def test_main_trace_stays_lean(traced, tmp_path):
    """Prompts must not leak into the main trace; it has to stay fast to load."""
    with tracing.task("t1"):
        tracing.chat(model="qwen", messages=MESSAGES, role="coder")
    row = _main_rows(tmp_path)[0]
    assert "messages" not in row
    assert "prompt" not in row


def test_sidecar_written_even_when_call_fails(tmp_path, monkeypatch):
    """The prompt exists whether or not ollama answered; losing it loses data."""
    import ollama

    def boom(model, messages, **kw):
        raise RuntimeError("ollama down")

    monkeypatch.setattr(ollama, "chat", boom)
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "testrun")
    tracing._reset()

    with pytest.raises(RuntimeError):
        with tracing.task("t1"):
            tracing.chat(model="m", messages=MESSAGES, role="coder")

    rows = _prompt_rows(tmp_path)
    assert len(rows) == 1 and rows[0]["messages"] == MESSAGES
    tracing._reset()


def test_sidecar_carries_task_id_for_grouping(traced, tmp_path):
    """Phase 2 compares prefixes within a task, so task_id must be on the row."""
    with tracing.task("HumanEval/3"):
        tracing.chat(model="qwen", messages=MESSAGES, role="coder")
    assert _prompt_rows(tmp_path)[0]["task_id"] == "HumanEval/3"


def test_sidecar_gzipped_for_full_runs(tmp_path, monkeypatch):
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.delenv("MAARA_TRACE_SMOKE", raising=False)
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "gzrun")
    tracing._reset()

    with tracing.task("t1"):
        tracing.chat(model="m", messages=MESSAGES, role="coder")
    tracing._reset()

    p = tmp_path / "gzrun.prompts.jsonl.gz"
    assert p.exists()
    with gzip.open(p, "rt") as f:
        assert json.loads(f.readline())["messages"] == MESSAGES


def test_embedding_ops_have_no_sidecar_row(traced, tmp_path):
    """Embeddings carry no chat prompt, so they must not add sidecar noise."""
    with tracing.task("t1"):
        with tracing.embedding_op("build_index", n_documents=5):
            pass
    assert _prompt_rows(tmp_path) == []
    assert len(_main_rows(tmp_path)) == 1


# --- results path --------------------------------------------------------

def test_results_path_defaults_unchanged(monkeypatch):
    import orchestrator
    monkeypatch.delenv("MAARA_RESULTS_PATH", raising=False)
    assert orchestrator.results_path() == "data/outcomes/results.jsonl"


def test_results_path_honours_env(monkeypatch, tmp_path):
    import orchestrator
    target = str(tmp_path / "traced" / "results.jsonl")
    monkeypatch.setenv("MAARA_RESULTS_PATH", target)
    assert orchestrator.results_path() == target


def test_append_result_writes_to_override_and_creates_dirs(monkeypatch, tmp_path):
    import orchestrator
    target = tmp_path / "nested" / "dir" / "results.jsonl"
    monkeypatch.setenv("MAARA_RESULTS_PATH", str(target))

    orchestrator._append_result({"task_id": "HumanEval/1", "passed": True})
    orchestrator._append_result({"task_id": "HumanEval/2", "passed": False})

    rows = [json.loads(l) for l in target.read_text().splitlines() if l.strip()]
    assert [r["task_id"] for r in rows] == ["HumanEval/1", "HumanEval/2"]


def test_append_result_leaves_primary_file_untouched(monkeypatch, tmp_path):
    """
    The whole point of the override: a traced re-run must not append duplicates
    to data/outcomes/results.jsonl, which would make pass_at_1.py double-count.
    """
    import orchestrator
    repo = os.path.dirname(os.path.abspath(orchestrator.__file__))
    primary = os.path.join(repo, "data", "outcomes", "results.jsonl")
    before = os.path.getsize(primary) if os.path.exists(primary) else None

    monkeypatch.setenv("MAARA_RESULTS_PATH", str(tmp_path / "side.jsonl"))
    orchestrator._append_result({"task_id": "x", "passed": True})

    after = os.path.getsize(primary) if os.path.exists(primary) else None
    assert before == after, "traced run must not touch the primary results file"
    assert (tmp_path / "side.jsonl").exists()
