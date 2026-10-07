"""
The evaluator's reply parser must not crash on a malformed reply.

FINDINGS 14b: 7 of 60 shared_first runs died with IndexError because a reply
without a SCORE: line was indexed before it was checked. No LLM is called;
tracing.chat is monkeypatched.
"""

import pytest

from maara import tracing
from maara.agents.evaluator import DEFAULT_SCORE, _parse_score, evaluate_plan


@pytest.mark.parametrize("raw,expected", [
    ("APPROVED: yes\nSCORE: 8\nFEEDBACK: fine", 8),        # the normal reply
    ("APPROVED: no\nSCORE: 7/10\nFEEDBACK: x", 7),          # accepted before
    ("APPROVED: yes\nFEEDBACK: no score line", DEFAULT_SCORE),  # used to crash
    ("APPROVED: yes\nSCORE: 8.\nFEEDBACK: x", 8),           # used to crash
    ("APPROVED: yes\nSCORE: N/A\nFEEDBACK: x", DEFAULT_SCORE),  # used to crash
    ("", DEFAULT_SCORE),                                    # used to crash
])
def test_parse_score(raw, expected):
    assert _parse_score(raw) == expected


def test_evaluate_plan_survives_a_reply_without_score(monkeypatch):
    monkeypatch.setattr(tracing, "chat", lambda **kw: {
        "message": {"content": "APPROVED: yes\nFEEDBACK: looks fine"}})
    out = evaluate_plan({"intent": "fix f", "affected_files": ["target.py"]},
                        {"target.py": "def f(): pass"}, {})
    assert out == {"approved": True, "score": DEFAULT_SCORE,
                   "feedback": "looks fine"}
