import tracing
from agents import prompt_layout


def diagnose(
    original_code: str,
    fixed_code: str,
    test_output: str,
    previous_diagnoses: list = None,
    filename: str = "target.py"
) -> dict:
    """
    Looks at what was tried and why it failed.
    Returns a diagnosis and a suggested approach for the next attempt.
    """

    history = ""
    if previous_diagnoses:
        history = "Previous diagnoses that did not work:\n"
        for i, d in enumerate(previous_diagnoses, 1):
            history += f"{i}. {d}\n"
        history += "\nDo NOT suggest the same approach again.\n"

    prompt = f"""You are a Python debugging expert analyzing a failed fix attempt.

Original code:
```python
{original_code}
```

Fix that was attempted:
```python
{fixed_code}
```

Test output showing what still fails:
{test_output}
{history}

Analyze why the fix failed. Be specific and technical.
Return a JSON object with exactly these keys:
- diagnosis: one paragraph explaining exactly why the fix failed
- approach: one paragraph describing a different specific approach to try next
- error_type: one of [logic_error, syntax_error, edge_case_missed, wrong_function_modified, incomplete_fix]

Raw JSON only. No markdown. No explanation."""

    system_msg = "You are a precise debugging analyst. Return only valid JSON."

    if prompt_layout.is_shared_first():
        # The original file leads, rendered exactly as the coder renders it, so
        # the two share that head once the retry loop has rewritten
        # relevant_code from disk. The debugger has no tests or issue block:
        # it is not given them today, and rule 2 forbids adding them.
        system_msg, prompt = prompt_layout.compose(
            code={filename: original_code},
            role="diagnose the failed fix",
            role_body=f"""Fix that was attempted:
```python
{fixed_code}
```

Test output showing what still fails:
{test_output}
{history}

Analyze why the fix failed. Be specific and technical.
Return a JSON object with exactly these keys:
- diagnosis: one paragraph explaining exactly why the fix failed
- approach: one paragraph describing a different specific approach to try next
- error_type: one of [logic_error, syntax_error, edge_case_missed, \
wrong_function_modified, incomplete_fix]

Raw JSON only. No markdown. No explanation.""")

    response = tracing.chat(
        model="qwen2.5-coder:7b",
        role="debugger",
        # fixed_code comes from THIS attempt's coder; previous_diagnoses come
        # from earlier attempts' debugger calls (debugger.py:15-36).
        parents=("coder",),
        prev_attempt_parents=("debugger",),
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

    import json
    import re

    raw = response["message"]["content"].strip()
    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?\n?", "", raw)
        raw = raw.replace("```", "").strip()

    try:
        result = json.loads(raw)
        print(f"\n[Debugger] Error type: {result.get('error_type', 'unknown')}")
        print(f"[Debugger] Diagnosis: {result.get('diagnosis', '')}")
        print(f"[Debugger] Next approach: {result.get('approach', '')}")
        return result
    except json.JSONDecodeError:
        print(f"[Debugger] Warning: invalid JSON returned. Raw:\n{raw}")
        return {
            "diagnosis": raw,
            "approach": "Try a completely different implementation strategy.",
            "error_type": "unknown"
        }