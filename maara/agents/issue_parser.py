import json
import re

from maara import tracing
from maara import prompt_layout


def parse_issue(issue_text: str, available_files: list = None) -> dict:
    files_context = ""
    if available_files:
        files_context = f"""
The repository contains these Python files:
{chr(10).join(f'- {f}' for f in available_files)}

You MUST only reference files from this list in affected_files.
"""

    prompt = f"""You are a software engineering assistant that reads bug reports.

Analyze this GitHub issue and extract structured information.

Issue:
{issue_text}
{files_context}
Return ONLY a JSON object with these exact keys:
- intent: one sentence describing what needs to be fixed
- affected_files: list of Python filenames likely involved (only from the list above)
- expected_behavior: what the code should do when fixed
- test_hints: list of edge cases the fix must handle

Return raw JSON only. No explanation. No markdown fences."""

    system_msg = "You are a precise software engineering assistant. Return only valid JSON."

    if prompt_layout.is_shared_first():
        # Same information, reordered: issue first, then the file list and the
        # output schema. issue_parser has no code or test blocks -- it runs
        # before retrieval -- so it shares only the common system message.
        system_msg, prompt = prompt_layout.compose(
            issue=issue_text,
            role="parse the issue",
            role_body=f"""Analyze the issue above and extract structured information.
{files_context}
Return ONLY a JSON object with these exact keys:
- intent: one sentence describing what needs to be fixed
- affected_files: list of Python filenames likely involved (only from the list above)
- expected_behavior: what the code should do when fixed
- test_hints: list of edge cases the fix must handle

Return raw JSON only. No explanation. No markdown fences.""")

    print(f"\n[Issue Parser] Sending prompt to LLM...")

    response = tracing.chat(
        model="qwen2.5-coder:7b",
        role="issue_parser",
        # Resolves to no parent on the first call, since no evaluator has run
        # yet, and to the rejecting evaluator on a replan (orchestrator.py:80).
        parents=("evaluator",),
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

    raw = response["message"]["content"].strip()
    print(f"[Issue Parser] Raw response:\n{raw}\n")

    if raw.startswith("```"):
        raw = re.sub(r"```(?:json)?\n?", "", raw)
        raw = raw.replace("```", "").strip()

    try:
        parsed = json.loads(raw)

        if "test_hints" in parsed:
            normalized = []
            for hint in parsed["test_hints"]:
                if isinstance(hint, str):
                    normalized.append(hint)
                elif isinstance(hint, dict):
                    normalized.append(hint.get("description", str(hint)))
                else:
                    normalized.append(str(hint))
            parsed["test_hints"] = normalized

        if available_files and "affected_files" in parsed:
            parsed["affected_files"] = [
                f for f in parsed["affected_files"]
                if f in available_files
            ]
            if not parsed["affected_files"]:
                parsed["affected_files"] = available_files

        return parsed

    except json.JSONDecodeError:
        print(f"[Issue Parser] Warning: invalid JSON returned. Raw output:\n{raw}")
        return {
            "intent": issue_text,
            "affected_files": available_files or ["target.py"],
            "expected_behavior": "code should pass all tests",
            "test_hints": []
        }