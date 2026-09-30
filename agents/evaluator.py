# agents/evaluator.py

import tracing

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

    response = tracing.chat(
        model="llama3.2",
        role="evaluator",
        # Prompt carries issue.intent/affected_files from issue_parser and
        # relevant_code selected by retrieval (evaluator.py:11-22).
        parents=("issue_parser", "retriever"),
        messages=[
            {
                "role": "system",
                "content": "You are a strict but fair plan evaluator. Be concise and specific."
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
    score_line = [l for l in raw.split("\n") if l.startswith("SCORE:")]
    score_raw = score_line[0].split(":")[-1].strip()
    score = int(score_raw.split("/")[0].strip()) if score_line else 5
    feedback_line = raw.split("FEEDBACK:")[-1].strip() if "FEEDBACK:" in raw else raw

    return {
        "approved": approved,
        "score": score,
        "feedback": feedback_line
    }