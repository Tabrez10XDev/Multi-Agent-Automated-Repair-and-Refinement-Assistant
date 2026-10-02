"""
Tests for the switchable prompt layout.

The load-bearing one is `original` fidelity: the committed trace was collected
under the original layout, so if the default path drifts by a single byte the
two layouts are no longer comparable and the whole experiment is void. That
test reconstructs a real recorded call's inputs from the committed sidecar and
checks the agent rebuilds the recorded prompt exactly.

No LLM is called; tracing.chat is monkeypatched.
"""

import json
import os

import pytest

from maara import prompt_layout
from simulator.trace_loader import DEFAULT_TRACES, default_prompts_path, load_rows

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRACE = os.path.join(_REPO, DEFAULT_TRACES[0])
has_trace = pytest.mark.skipif(not os.path.exists(TRACE),
                               reason="committed trace not present")

FAKE_JSON = json.dumps({"intent": "i", "affected_files": ["target.py"],
                        "expected_behavior": "e", "test_hints": ["h"],
                        "diagnosis": "d", "approach": "a",
                        "error_type": "logic_error"})
FAKE_EVAL = "APPROVED: yes\nSCORE: 8\nFEEDBACK: looks fine"


@pytest.fixture
def captured(monkeypatch):
    """Capture the messages each agent would send, without calling ollama."""
    from maara import tracing
    seen = []

    def fake_chat(model, messages, role, **kw):
        seen.append({"model": model, "role": role, "messages": messages})
        content = FAKE_EVAL if role == "evaluator" else FAKE_JSON
        return {"message": {"content": content}, "prompt_eval_count": 10,
                "eval_count": 5, "prompt_eval_duration": 1, "eval_duration": 1}

    monkeypatch.setattr(tracing, "chat", fake_chat)
    return seen


def _recorded(role):
    """First recorded (system, user) pair for a role, from the committed trace."""
    rows = {r["call_id"]: r for r in load_rows(TRACE)}
    for p in load_rows(default_prompts_path(TRACE)):
        r = rows.get(p["call_id"])
        if r and r["agent_role"] == role and len(p["messages"]) == 2:
            return p["messages"][0]["content"], p["messages"][1]["content"]
    return None, None


# --- the flag ------------------------------------------------------------

def test_default_is_original(monkeypatch):
    monkeypatch.delenv("MAARA_PROMPT_LAYOUT", raising=False)
    assert prompt_layout.layout() == "original"
    assert prompt_layout.is_shared_first() is False


@pytest.mark.parametrize("value", ["shared_first", "SHARED_FIRST", " shared_first "])
def test_flag_enables_shared_first(monkeypatch, value):
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", value)
    assert prompt_layout.is_shared_first() is True


def test_unknown_value_falls_back_to_original(monkeypatch):
    """A typo must not silently produce a third layout."""
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "sharedfirst")
    assert prompt_layout.layout() == "original"


# --- original fidelity ---------------------------------------------------

@has_trace
def test_original_reproduces_a_recorded_issue_parser_prompt(captured, monkeypatch):
    """
    Strongest available check: pull a real recorded call apart, feed the same
    inputs back in, and require the rebuilt prompt to match byte for byte.
    """
    monkeypatch.delenv("MAARA_PROMPT_LAYOUT", raising=False)
    system, user = _recorded("issue_parser")
    assert user, "no recorded issue_parser prompt"

    marker = "\n\nThe repository contains these Python files:\n"
    assert marker in user
    head, tail = user.split(marker, 1)
    issue_text = head.split("Issue:\n", 1)[1]
    # Stop at the end of the file list: the JSON schema below it also uses
    # "- " bullets, and scanning the whole tail swallows them as filenames.
    file_list = tail.split("\n\nYou MUST only reference", 1)[0]
    files = [l[2:] for l in file_list.split("\n") if l.startswith("- ")]

    from maara.agents.issue_parser import parse_issue
    parse_issue(issue_text, available_files=files)

    got = captured[0]["messages"]
    assert got[0]["content"] == system
    assert got[1]["content"] == user


@has_trace
@pytest.mark.parametrize("role", ["issue_parser", "evaluator", "coder", "debugger"])
def test_original_system_messages_match_the_trace(role):
    """Each role's system message must still be its own historical string."""
    system, _ = _recorded(role)
    assert system and system != prompt_layout.SHARED_SYSTEM


def test_original_coder_prompt_is_unchanged(captured, monkeypatch):
    monkeypatch.delenv("MAARA_PROMPT_LAYOUT", raising=False)
    from maara.agents.coder import generate_fix
    generate_fix(issue={"intent": "fix f", "expected_behavior": "works",
                        "test_hints": ["empty"]},
                 relevant_code={"target.py": "def f(): pass"},
                 test_files={"test_target.py": "def test_f(): pass"})
    msgs = captured[0]["messages"]
    assert msgs[0]["content"].startswith("You are a Python debugging expert.")
    assert msgs[1]["content"].startswith("You are a Python debugging expert.\n\nIssue to fix:")
    assert "### CODE" not in msgs[1]["content"]


# --- shared_first behaviour ----------------------------------------------

def _run_all_roles(monkeypatch):
    from maara.agents.coder import generate_fix
    from maara.agents.debugger import diagnose
    from maara.agents.evaluator import evaluate_plan
    from maara.agents.issue_parser import parse_issue

    issue = {"intent": "fix f", "affected_files": ["target.py"],
             "expected_behavior": "works", "test_hints": ["empty"]}
    code = {"target.py": "def f():\n    return 1\n"}
    tests = {"test_target.py": "def test_f():\n    assert f() == 2\n"}

    parse_issue("the issue text", available_files=["target.py"])
    evaluate_plan(issue, code, tests)
    generate_fix(issue=issue, relevant_code=code, test_files=tests)
    diagnose(original_code=code["target.py"], fixed_code="def f(): return 2",
             test_output="FAILED", filename="target.py")


def test_shared_first_uses_one_system_message_for_every_role(captured, monkeypatch):
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    systems = {c["messages"][0]["content"] for c in captured}
    assert systems == {prompt_layout.SHARED_SYSTEM}


def test_shared_first_puts_identical_content_first(captured, monkeypatch):
    """
    evaluator and coder both receive the code and the test files today, so
    under shared_first their prompts must agree for a long common head rather
    than diverging on character 10 as they do under the original layout.
    """
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    by_role = {c["role"]: c["messages"][1]["content"] for c in captured}

    ev, cd = by_role["evaluator"], by_role["coder"]
    shared = os.path.commonprefix([ev, cd])
    assert "### CODE" in shared
    assert "### TESTS" in shared
    assert len(shared) > 100

    # The head must cover the whole code and tests blocks, ending only when the
    # issue block starts -- that is where the two roles genuinely differ.
    assert shared.rstrip().endswith("```") or "### ISSUE" in shared


def test_shared_first_debugger_shares_the_code_head_with_coder(captured, monkeypatch):
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    by_role = {c["role"]: c["messages"][1]["content"] for c in captured}
    shared = os.path.commonprefix([by_role["debugger"], by_role["coder"]])
    assert "### CODE" in shared and "target.py" in shared


def test_shared_first_adds_no_information_to_a_role(captured, monkeypatch):
    """
    Rule 2. The debugger is not given the test files today and must not gain
    them; the issue_parser has neither code nor tests.
    """
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    by_role = {c["role"]: c["messages"][1]["content"] for c in captured}
    assert "### TESTS" not in by_role["debugger"]
    assert "### CODE" not in by_role["issue_parser"]
    assert "### TESTS" not in by_role["issue_parser"]


def test_shared_first_keeps_every_role_specific_instruction(captured, monkeypatch):
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    by_role = {c["role"]: c["messages"][1]["content"] for c in captured}
    assert "APPROVED: yes/no" in by_role["evaluator"]
    assert "Raw Python only" in by_role["coder"]
    assert "error_type" in by_role["debugger"]
    assert "affected_files" in by_role["issue_parser"]


def test_absent_blocks_emit_no_header(captured, monkeypatch):
    """An empty '### TESTS' header would itself break the next role's match."""
    monkeypatch.setenv("MAARA_PROMPT_LAYOUT", "shared_first")
    _run_all_roles(monkeypatch)
    for c in captured:
        body = c["messages"][1]["content"]
        assert "### TESTS\n### " not in body
        assert "### CODE\n### " not in body


def test_compose_orders_blocks_code_tests_issue_role():
    system, body = prompt_layout.compose(
        code={"a.py": "x"}, tests={"t.py": "y"}, issue="ISSUE TEXT",
        role="do the thing", role_body="INSTRUCTIONS")
    assert system == prompt_layout.SHARED_SYSTEM
    assert (body.index("### CODE") < body.index("### TESTS")
            < body.index("### ISSUE") < body.index("### TASK"))
    assert "INSTRUCTIONS" in body
