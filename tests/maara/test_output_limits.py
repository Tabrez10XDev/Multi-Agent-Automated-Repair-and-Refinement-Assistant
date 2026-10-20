"""
Tests for per-role output-token limits (docs/DECISIONS.md §32).

No LLM is called: ollama.chat is monkeypatched with a stub that records the
kwargs each agent sends and returns a reply that agent can parse. Traces go to
tmp_path, never the repo's data/traces/.
"""

import json

import pytest

from maara import output_limits, tracing
from maara.agents import coder, debugger, evaluator, issue_parser

ISSUE = {"intent": "fix add", "affected_files": ["target.py"],
         "expected_behavior": "add returns a + b", "test_hints": ["negatives"]}
CODE = {"target.py": "def add(a, b):\n    return a - b\n"}
TESTS = {"test_target.py": "assert add(1, 2) == 3\n"}

# role -> (call the agent, reply text the agent can parse)
AGENTS = {
    "issue_parser": (
        lambda: issue_parser.parse_issue("add is wrong", ["target.py"]),
        json.dumps({"intent": "fix add", "affected_files": ["target.py"],
                    "expected_behavior": "a + b", "test_hints": ["negatives"]})),
    "coder": (
        lambda: coder.generate_fix(ISSUE, CODE, TESTS),
        "def add(a, b):\n    return a + b\n"),
    "debugger": (
        lambda: debugger.diagnose(CODE["target.py"], "def add(a, b): ...",
                                  "AssertionError"),
        json.dumps({"diagnosis": "d", "approach": "a",
                    "error_type": "logic_error"})),
    "evaluator": (
        lambda: evaluator.evaluate_plan(ISSUE, CODE, TESTS),
        "APPROVED: yes\nSCORE: 8\nFEEDBACK: ok"),
}

EXPECTED = {
    "issue_parser": {"num_predict": 4608},
    "coder": None,
    "debugger": None,
    "evaluator": None,
}


def _stub(monkeypatch, content):
    import ollama
    calls = []

    def fake_chat(model, messages, **kw):
        calls.append(kw)
        return {"message": {"content": content}, "prompt_eval_count": 10,
                "eval_count": 5, "prompt_eval_duration": 1_000_000_000,
                "eval_duration": 1_000_000_000}

    monkeypatch.setattr(ollama, "chat", fake_chat)
    return calls


@pytest.fixture(params=["untraced", "traced"])
def trace_mode(request, tmp_path, monkeypatch):
    """Both paths through tracing.chat must forward options to ollama."""
    monkeypatch.setenv("MAARA_TRACE_DIR", str(tmp_path))
    if request.param == "traced":
        monkeypatch.setenv("MAARA_TRACE", "1")
        monkeypatch.setenv("MAARA_TRACE_SMOKE", "1")
        monkeypatch.setenv("MAARA_RUN_ID", "limits")
    else:
        monkeypatch.delenv("MAARA_TRACE", raising=False)
    tracing._reset()
    yield request.param
    tracing._reset()


@pytest.mark.parametrize("role", sorted(AGENTS))
def test_each_agent_passes_its_options(role, trace_mode, monkeypatch):
    run, content = AGENTS[role]
    calls = _stub(monkeypatch, content)
    with tracing.task("t1"):
        run()
    assert len(calls) == 1
    assert "options" in calls[0], f"{role} does not pass options at all"
    assert calls[0]["options"] == EXPECTED[role]


def test_unlimited_roles_send_ollamas_default():
    """None is ollama.chat's own default, so an unlimited role's request is unchanged."""
    import inspect
    import ollama
    assert inspect.signature(ollama.Client.chat).parameters["options"].default is None
    for role in ("coder", "debugger", "evaluator", "no_such_role"):
        assert output_limits.options(role) is None


def test_limit_table_is_sane():
    roles = {"issue_parser", "coder", "debugger", "evaluator"}
    for table in (output_limits.NUM_PREDICT, output_limits.PROPOSED_NUM_PREDICT):
        assert set(table) <= roles
        for n in table.values():
            assert isinstance(n, int) and n > 0
    # A proposed limit is not applied; moving it into NUM_PREDICT is a decision.
    assert not set(output_limits.NUM_PREDICT) & set(output_limits.PROPOSED_NUM_PREDICT)


def test_issue_parser_limit_clears_every_observed_output():
    """2,286 is the largest non-runaway issue_parser output in the committed traces."""
    assert output_limits.NUM_PREDICT["issue_parser"] >= 2 * 2286
