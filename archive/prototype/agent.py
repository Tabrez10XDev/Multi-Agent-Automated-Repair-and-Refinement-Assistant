import os
import subprocess
import tempfile
import time
import ollama
from dotenv import load_dotenv

load_dotenv()


def run_code_in_sandbox(code: str, timeout: int = 10) -> dict:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(code)
        tmp_path = f.name

    try:
        result = subprocess.run(
            ["python", tmp_path],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "timed_out": False
        }
    except subprocess.TimeoutExpired:
        return {
            "stdout": "",
            "stderr": "Execution timed out.",
            "returncode": -1,
            "timed_out": True
        }
    finally:
        os.unlink(tmp_path)


def run_tests() -> dict:
    result = subprocess.run(
        ["python", "-m", "pytest", "test_target.py", "-v", "--tb=short"],
        capture_output=True,
        text=True
    )
    return {
        "output": result.stdout + result.stderr,
        "passed": result.returncode == 0
    }


def read_current_code() -> str:
    with open("target.py", "r") as f:
        return f.read()


def write_fixed_code(code: str):
    code = code.strip()
    if code.startswith("```"):
        lines = code.split("\n")
        code = "\n".join(lines[1:-1])
    with open("target.py", "w") as f:
        f.write(code)


def ask_llm_to_fix(current_code: str, test_output: str, attempt: int) -> str:
    prompt = f"""You are a Python debugging expert.

Here is the current code:
```python
{current_code}
```

Here is the test output showing what is failing:
{test_output}
This is attempt {attempt} to fix the code.
Return ONLY the complete fixed Python code, no explanation, no markdown fences.
Fix all failing tests without breaking passing ones."""

    print(f"\n--- Prompt sent to LLM (attempt {attempt}) ---")
    print(prompt)
    print("--- End of prompt ---\n")

    response = ollama.chat(
        model="qwen2.5-coder:7b",
        messages=[
            {
                "role": "system",
                "content": "You are a Python debugging expert. Return ONLY raw Python code with no markdown, no explanation, no fences."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


def self_healing_loop(max_attempts: int = 3):
    print("=== Self-Healing Agent Starting ===\n")

    for attempt in range(1, max_attempts + 1):
        print(f"--- Attempt {attempt}/{max_attempts} ---")

        test_result = run_tests()
        print(test_result["output"])

        if test_result["passed"]:
            print("✓ All tests passing! Agent succeeded.")
            return True

        current_code = read_current_code()

        print("Asking LLM to fix the code...")
        fixed_code = ask_llm_to_fix(
            current_code,
            test_result["output"],
            attempt
        )

        print("--- Code LLM returned ---")
        print(fixed_code)
        print("--- End of returned code ---\n")

        write_fixed_code(fixed_code)
        print("Fix applied. Re-running tests...\n")

    print("✗ Agent failed to fix all tests within budget.")
    return False


if __name__ == "__main__":
    self_healing_loop(max_attempts=3)