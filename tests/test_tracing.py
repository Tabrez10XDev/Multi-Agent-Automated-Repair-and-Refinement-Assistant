"""
Tests for per-call trace logging.

No LLM is called: ollama.chat is monkeypatched with a stub that returns the
same shape ollama does. Traces go to tmp_path, never the repo's traces/.
"""

import gzip
import json
import os

import pytest

from maara import tracing


def _fake_response(prompt_tok=100, out_tok=50):
    return {
        "message": {"content": "ok"},
        "prompt_eval_count": prompt_tok,
        "eval_count": out_tok,
        "prompt_eval_duration": 1_000_000_000,   # 1.0s in ns
        "eval_duration": 2_000_000_000,          # 2.0s in ns
    }


@pytest.fixture
def traced(tmp_path, monkeypatch):
    """Tracing enabled, writing uncompressed rows into tmp_path."""
    import ollama
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append({"model": model, "messages": messages})
        return _fake_response()

    monkeypatch.setattr(ollama, "chat", fake_chat)
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "testrun")
    tracing._reset()
    yield calls
    tracing._reset()


def _rows(tmp_path):
    p = tmp_path / "smoke" / "testrun.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]


# --- pass-through behaviour ---------------------------------------------

def test_disabled_by_default_writes_nothing(tmp_path, monkeypatch):
    """An untraced run must behave exactly as before and leave no files."""
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.delenv("MAARA_TRACE", raising=False)
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    tracing._reset()

    out = tracing.chat(model="m", messages=[], role="coder")
    assert out["message"]["content"] == "ok"
    assert list(tmp_path.iterdir()) == []
    tracing._reset()


def test_returns_ollama_response_unchanged(traced, tmp_path):
    with tracing.task("t1"):
        out = tracing.chat(model="qwen", messages=[{"role": "user", "content": "x"}],
                           role="coder")
    assert out == _fake_response()


def test_exception_propagates_and_is_logged(tmp_path, monkeypatch):
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
            tracing.chat(model="m", messages=[], role="coder")

    rows = _rows(tmp_path)
    assert len(rows) == 1 and "ollama down" in rows[0]["error"]
    tracing._reset()


# --- schema --------------------------------------------------------------

REQUIRED = ["run_id", "task_id", "call_id", "agent_role", "attempt_index",
            "parent_call_ids", "prompt_tokens", "output_tokens",
            "t_start", "t_end", "prompt_eval_duration", "eval_duration", "model"]


def test_row_has_every_required_field(traced, tmp_path):
    with tracing.task("HumanEval/1"):
        tracing.set_attempt(2)
        tracing.chat(model="qwen", messages=[], role="coder")
    row = _rows(tmp_path)[0]
    for field in REQUIRED:
        assert field in row, f"missing {field}"
    assert row["task_id"] == "HumanEval/1"
    assert row["attempt_index"] == 2
    assert row["agent_role"] == "coder"
    assert row["kind"] == "chat"


def test_tokens_are_separate_and_durations_in_seconds(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="qwen", messages=[], role="coder")
    row = _rows(tmp_path)[0]
    assert row["prompt_tokens"] == 100
    assert row["output_tokens"] == 50
    # ollama reports nanoseconds; we store seconds.
    assert row["prompt_eval_duration"] == pytest.approx(1.0)
    assert row["eval_duration"] == pytest.approx(2.0)


# --- parent edges --------------------------------------------------------

def test_issue_parser_has_no_parent_on_first_call(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="qwen", messages=[], role="issue_parser",
                     parents=("evaluator",))
    assert _rows(tmp_path)[0]["parent_call_ids"] == []


def test_issue_parser_parents_the_evaluator_on_replan(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="llama", messages=[], role="evaluator")
        tracing.chat(model="qwen", messages=[], role="issue_parser",
                     parents=("evaluator",))
    ev, reparse = _rows(tmp_path)
    assert reparse["parent_call_ids"] == [ev["call_id"]]


def test_prev_attempt_parents_point_at_the_previous_attempt(traced, tmp_path):
    """
    The subtle case: on attempt 2 the coder's previous_attempt/diagnosis come
    from attempt 1, not from a coder call earlier in attempt 2. If this used the
    current last-call map, a multi-file fan-out would make the coder its own
    parent.
    """
    with tracing.task("t1"):
        tracing.set_attempt(1)
        tracing.chat(model="qwen", messages=[], role="coder")
        tracing.chat(model="qwen", messages=[], role="debugger", parents=("coder",))
        tracing.set_attempt(2)
        tracing.chat(model="qwen", messages=[], role="coder",
                     prev_attempt_parents=("coder", "debugger"))

    coder1, dbg1, coder2 = _rows(tmp_path)
    assert dbg1["parent_call_ids"] == [coder1["call_id"]]
    assert set(coder2["parent_call_ids"]) == {coder1["call_id"], dbg1["call_id"]}
    assert coder2["call_id"] not in coder2["parent_call_ids"]


def test_parents_do_not_leak_across_tasks(traced, tmp_path):
    with tracing.task("t1"):
        tracing.chat(model="llama", messages=[], role="evaluator")
    with tracing.task("t2"):
        tracing.chat(model="qwen", messages=[], role="issue_parser",
                     parents=("evaluator",))
    assert _rows(tmp_path)[1]["parent_call_ids"] == []


# --- embeddings ----------------------------------------------------------

def test_embedding_logged_separately_without_token_counts(traced, tmp_path):
    with tracing.task("t1"):
        with tracing.embedding_op("build_index", n_documents=42):
            pass
    row = _rows(tmp_path)[0]
    assert row["kind"] == "embedding"
    assert row["agent_role"] == "retriever"
    assert row["n_documents"] == 42
    # Must not be countable as decode work.
    assert row["output_tokens"] is None
    assert row["eval_duration"] is None


# --- output location -----------------------------------------------------

def test_full_runs_are_gzipped(tmp_path, monkeypatch):
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.delenv("MAARA_TRACE_SMOKE", raising=False)
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "gzrun")
    tracing._reset()

    with tracing.task("t1"):
        tracing.chat(model="m", messages=[], role="coder")
    tracing._reset()

    p = tmp_path / "gzrun.jsonl.gz"
    assert p.exists()
    with gzip.open(p, "rt") as f:
        assert json.loads(f.readline())["agent_role"] == "coder"


def test_seq_is_monotonic(traced, tmp_path):
    with tracing.task("t1"):
        for _ in range(4):
            tracing.chat(model="m", messages=[], role="coder")
    assert [r["seq"] for r in _rows(tmp_path)] == [1, 2, 3, 4]


# --- append guard --------------------------------------------------------

def test_refuses_to_reuse_an_existing_trace_file(tmp_path, monkeypatch):
    """
    MAARA_RUN_ID pins the filename, so a restarted run used to append a second
    seq sequence into the same file under the same run_id -- indistinguishable
    from the first except by seq ceasing to advance. That silently invalidated
    a whole run's reuse figures once already.
    """
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "dup")
    monkeypatch.delenv("MAARA_TRACE_APPEND", raising=False)
    tracing._reset()

    with tracing.task("t1"):
        tracing.chat(model="m", messages=[], role="coder")
    tracing._reset()
    assert (tmp_path / "smoke" / "dup.jsonl").exists()

    with pytest.raises(RuntimeError, match="already exists"):
        tracing.chat(model="m", messages=[], role="coder")
    tracing._reset()


def test_append_is_allowed_when_asked_for(tmp_path, monkeypatch):
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.setenv("MAARA_RUN_ID", "dup2")
    tracing._reset()
    with tracing.task("t1"):
        tracing.chat(model="m", messages=[], role="coder")
    tracing._reset()

    monkeypatch.setenv("MAARA_TRACE_APPEND", "1")
    with tracing.task("t2"):
        tracing.chat(model="m", messages=[], role="coder")
    tracing._reset()
    rows = [json.loads(l) for l in
            (tmp_path / "smoke" / "dup2.jsonl").read_text().splitlines() if l.strip()]
    assert len(rows) == 2


def test_a_fresh_run_id_is_unaffected(tmp_path, monkeypatch):
    import ollama
    monkeypatch.setattr(ollama, "chat", lambda model, messages, **kw: _fake_response())
    monkeypatch.setenv("MAARA_TRACE", "1")
    monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    monkeypatch.delenv("MAARA_TRACE_APPEND", raising=False)
    for rid in ("a", "b"):
        monkeypatch.setenv("MAARA_RUN_ID", rid)
        tracing._reset()
        with tracing.task("t"):
            tracing.chat(model="m", messages=[], role="coder")
        tracing._reset()
    assert (tmp_path / "smoke" / "a.jsonl").exists()
    assert (tmp_path / "smoke" / "b.jsonl").exists()
