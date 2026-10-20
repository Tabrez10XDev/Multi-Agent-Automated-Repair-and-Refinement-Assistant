# agents/evaluator.py

import re

from maara import tracing
from maara import output_limits
from maara import prompt_layout

DEFAULT_SCORE = 5


def _parse_score(raw: str) -> int:
    """
    The SCORE: line's integer, or DEFAULT_SCORE if there is none.

    This used to index the first SCORE: line before checking one existed, so a
    reply without it raised IndexError and killed the task before any fix
    attempt -- 7 of 60 shared_first runs (FINDINGS 14b). Every reply the old
    parse accepted is parsed exactly as before; the fallbacks only apply where
    it would have crashed.
    """
    score_line = [l for l in raw.split("\n") if l.startswith("SCORE:")]
    if not score_line:
        return DEFAULT_SCORE
    score_raw = score_line[0].split(":")[-1].strip()
    try:
        return int(score_raw.split("/")[0].strip())
    except ValueError:
        m = re.match(r"\s*(\d+)", score_raw)
        return int(m.group(1)) if m else DEFAULT_SCORE


def evaluate_plan(issue: dict, relevant_code: dict, test_files: dict) -> dict:
    """
    Checks the plan from issue_parser before any code is generated.
    Returns: { "approved": bool, "feedback": str, "score": int }
    """

    plan_summary = f"""
Intent: {issue['intent']}
Affected files: {issue['affected_files']}
"""

    code_context = "\n\n".join(
        [f"### {fname}\n{code}" for fname, code in relevant_code.items()]
    )

    test_context = "\n\n".join(
        [f"### {fname}\n{code}" for fname, code in test_files.items()]
    )

    prompt = f"""You are a senior software engineer reviewing a proposed fix plan.

PLAN:
{plan_summary}

RELEVANT CODE:
{code_context}

TEST FILES:
{test_context}

Evaluate the plan against these three criteria:
1. Requirement traceability: does the plan address every failing test?
2. Logical consistency: are there any contradictions or circular dependencies?
3. Feasibility: is the approach implementable given the existing code structure?

Respond in this exact format:
APPROVED: yes/no
SCORE: 0-10
FEEDBACK: <one paragraph explaining what is missing or looks good>
"""

    system_msg = "You are a strict but fair plan evaluator. Be concise and specific."

    if prompt_layout.is_shared_first():
        # Code and tests move to the front, rendered exactly as the coder
        # renders them, so the two share that whole head. The issue block keeps
        # only the fields the evaluator receives today (intent, affected_files).
        system_msg, prompt = prompt_layout.compose(
            code=relevant_code, tests=test_files,
            issue=f"Intent: {issue['intent']}\nAffected files: {issue['affected_files']}",
            role="review the proposed fix plan",
            role_body="""Evaluate the plan against these three criteria:
1. Requirement traceability: does the plan address every failing test?
2. Logical consistency: are there any contradictions or circular dependencies?
3. Feasibility: is the approach implementable given the existing code structure?

Respond in this exact format:
APPROVED: yes/no
SCORE: 0-10
FEEDBACK: <one paragraph explaining what is missing or looks good>
""")

    response = tracing.chat(
        model="llama3.2",
        role="evaluator",
        options=output_limits.options("evaluator"),
        # Prompt carries issue.intent/affected_files from issue_parser and
        # relevant_code selected by retrieval (evaluator.py:11-22).
        parents=("issue_parser", "retriever"),
        messages=[
            {
                "role": "system",
                "content": system_msg
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response["message"]["content"]

    # parse the structured response
    approved = "yes" in raw.lower().split("approved:")[-1].split("\n")[0].lower()
    score = _parse_score(raw)
    feedback_line = raw.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in raw else raw

    return {
        "approved": approved,
        "score": score,
        "feedback": feedback_line
    }