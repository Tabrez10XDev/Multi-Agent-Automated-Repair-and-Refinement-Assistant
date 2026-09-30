import tracing
from agents import prompt_layout


def generate_fix(
    issue: dict,
    relevant_code: dict,
    test_files: dict = None,
    previous_attempt: str = None,
    diagnosis: str = None,
    prompting_strategy: str = "zero_shot",
    model: str = "qwen2.5-coder:7b"
) -> dict:
    """
    Takes the parsed issue + relevant code, returns fixed code.

    Supports three prompting strategies:
      - zero_shot:       no examples, direct instruction
      - few_shot:        2-3 in-context examples prepended to the prompt
      - chain_of_thought: instructs the model to reason step-by-step before writing code

    Returns { filename: fixed_code } for each affected file.
    """
    results = {}
    total_tokens = 0

    # build test context string
    test_context = ""
    if test_files:
        test_context = "\n\nHere are the actual test assertions you must satisfy:\n"
        for test_filename, test_source in test_files.items():
            test_context += f"\n{test_filename}:\n```python\n{test_source}\n```"

    FEW_SHOT_EXAMPLES = """
Examples of correct bug fixes:

Example 1:
Buggy code:
```python
def divide(a, b):
    return a / b
```
Fixed code:
```python
def divide(a, b):
    if b == 0:
        return 0.0
    return a / b
```

Example 2:
Buggy code:
```python
def first_char(s):
    return s[0]
```
Fixed code:
```python
def first_char(s):
    if not s:
        return ""
    return s[0]
```

Example 3:
Buggy code:
```python
def count_evens(nums):
    count = 0
    for n in nums:
        if n % 2 == 1:
            count += 1
    return count
```
Fixed code:
```python
def count_evens(nums):
    count = 0
    for n in nums:
        if n % 2 == 0:
            count += 1
    return count
```
"""

    COT_INSTRUCTION = """
Before writing any code, reason through the following:
1. What is each failing test actually asserting?
2. What is wrong in the current code that causes those assertions to fail?
3. What is the minimal correct fix for each issue?
4. Are there any edge cases (empty input, zero, None) that need special handling?

Write your reasoning first, then write FINAL CODE: followed by the complete fixed Python code only.
"""

    for filename, source_code in relevant_code.items():

        if previous_attempt and diagnosis:
            retry_context = f"""
A previous fix attempt failed.

Previous fix:
```python
{previous_attempt}
```

Diagnosis of why it failed:
{diagnosis}

Use this diagnosis to guide a different approach."""
        else:
            retry_context = "This is the first attempt."

        base_prompt = f"""You are a Python debugging expert.

Issue to fix:
{issue['intent']}

Expected behavior:
{issue['expected_behavior']}

Edge cases to handle:
{', '.join(issue.get('test_hints', []))}
{test_context}

Current code in {filename}:
```python
{source_code}
```

{retry_context}
"""

        if prompting_strategy == "few_shot":
            prompt = FEW_SHOT_EXAMPLES + "\n" + base_prompt
            prompt += f"\nReturn ONLY the complete fixed Python code for {filename}.\nNo explanation. No markdown fences. Raw Python only."
            system_msg = "You are a Python debugging expert. Return ONLY raw Python code. No markdown. No explanation."

        elif prompting_strategy == "chain_of_thought":
            prompt = base_prompt + COT_INSTRUCTION
            system_msg = "You are a Python debugging expert. Think step by step, then write the fixed code after FINAL CODE:."

        else:  # zero_shot
            prompt = base_prompt
            prompt += f"\nReturn ONLY the complete fixed Python code for {filename}.\nNo explanation. No markdown fences. Raw Python only."
            system_msg = "You are a Python debugging expert. Return ONLY raw Python code. No markdown. No explanation."

        if prompt_layout.is_shared_first():
            # Code and tests lead, rendered identically to the evaluator's, so
            # the two share that head. The strategy's own wording (few-shot
            # examples, chain-of-thought instruction) moves into the role
            # section, and the strategy-specific system message is replaced by
            # the common one -- otherwise the shared head would end there.
            examples = FEW_SHOT_EXAMPLES + "\n" if prompting_strategy == "few_shot" else ""
            closing = (COT_INSTRUCTION if prompting_strategy == "chain_of_thought"
                       else f"\nReturn ONLY the complete fixed Python code for "
                            f"{filename}.\nNo explanation. No markdown fences. "
                            f"Raw Python only.")
            system_msg, prompt = prompt_layout.compose(
                code={filename: source_code},
                tests=test_files,
                issue=(f"{issue['intent']}\n\nExpected behavior:\n"
                       f"{issue['expected_behavior']}\n\nEdge cases to handle:\n"
                       f"{', '.join(issue.get('test_hints', []))}"),
                role="write the fixed code",
                role_body=f"{examples}{retry_context}\n{closing}")

        print(f"\n[Coder] Strategy: {prompting_strategy} | Model: {model} | File: {filename}")

        response = tracing.chat(
            model=model,
            role="coder",
            # Stable context comes from issue_parser and retrieval; on a retry
            # the prompt also carries the previous attempt's fix and the
            # diagnosis of why it failed (coder.py:97-131).
            parents=("issue_parser", "retriever"),
            prev_attempt_parents=("coder", "debugger"),
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

        total_tokens += response.get("eval_count", 0)
        total_tokens += response.get("prompt_eval_count", 0)

        raw = response["message"]["content"].strip()

        # extract code after FINAL CODE: for chain_of_thought
        if prompting_strategy == "chain_of_thought" and "FINAL CODE:" in raw:
            raw = raw.split("FINAL CODE:")[-1].strip()

        # strip markdown fences if the model added them anyway
        if raw.startswith("```"):
            lines = raw.split("\n")
            raw = "\n".join(lines[1:-1])

        results[filename] = raw

    print(f"[Coder] Total tokens used: {total_tokens}")
    return results, total_tokens