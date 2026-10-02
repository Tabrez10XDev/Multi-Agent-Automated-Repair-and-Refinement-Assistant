import os
import json
import time

import tracing
from agents.paths import is_agent_visible
from agents.issue_parser import parse_issue
from agents.navigator import navigate
from agents.coder import generate_fix
from agents.test_runner import apply_and_test
from agents.debugger import diagnose
from agents.navigator import get_file_structure
from agents.retriever import update_index
from agents.evaluator import evaluate_plan


def results_path() -> str:
    """
    Where experiment rows are appended.

    MAARA_RESULTS_PATH sends a traced re-run to its own file, so collecting a
    trace does not append duplicate rows to the primary results.jsonl and make
    analyze_results.py double-count. Unset, behaviour is unchanged.
    """
    return os.environ.get("MAARA_RESULTS_PATH") or "data/outcomes/results.jsonl"


def _append_result(experiment_log: dict) -> None:
    path = results_path()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(experiment_log) + "\n")


def run(
    issue_text: str,
    repo_path: str,
    test_file: str = "test_target.py",
    max_attempts: int = 3,
    config: str = "hetero",
    prompting_strategy: str = "zero_shot",
    task_id: str = "unknown",
    model: str = "qwen2.5-coder:7b"
):
    """Thin tracing wrapper. All behaviour lives in _run_impl, unchanged."""
    with tracing.task(task_id):
        return _run_impl(issue_text, repo_path, test_file, max_attempts,
                         config, prompting_strategy, task_id, model)


def _run_impl(
    issue_text: str,
    repo_path: str,
    test_file: str = "test_target.py",
    max_attempts: int = 3,
    config: str = "hetero",
    prompting_strategy: str = "zero_shot",
    task_id: str = "unknown",
    model: str = "qwen2.5-coder:7b"
):
    start_time = time.time()

    print("\n" + "="*50)
    print("MAARA STARTING")
    print("="*50)

    # Step 1: parse the issue
    print("\n[Step 1/6] Parsing issue...")
    structure = get_file_structure(repo_path)
    available_files = [f for f in structure.keys() if is_agent_visible(f)]
    print(f"  Available files: {available_files}")

    issue = parse_issue(issue_text, available_files=available_files)
    print(f"  Intent: {issue['intent']}")
    print(f"  Affected files: {issue['affected_files']}")

    # Step 2: navigate
    print("\n[Step 2/6] Navigating codebase...")
    nav_result = navigate(
        repo_path,
        issue_text,
        issue["affected_files"],
        use_rag=(config != "hetero_no_rag")
    )    
    relevant_code = nav_result["relevant_code"]
    collection = nav_result["collection"]
    test_files = nav_result["test_files"]

    if not relevant_code:
        print("  Navigator found no relevant files. Aborting.")
        return False

    print(f"  Test files found: {list(test_files.keys())}")

    # Step 2.5: Evaluate the plan (Sentinel gate)
    print("\n[Step 2.5/6] Evaluating plan (Sentinel)...")
    eval_result = {"approved": True, "score": None, "feedback": "", "approved_on_attempt": 1}

    if config != "hetero_no_sentinel":
        MAX_REPLAN = 2
        for replan in range(MAX_REPLAN):
            tracing.set_replan(replan + 1)
            eval_result = evaluate_plan(issue, relevant_code, test_files)
            print(f"  Score: {eval_result['score']}/10 | Approved: {eval_result['approved']}")
            print(f"  Feedback: {eval_result['feedback']}")
            if eval_result["approved"]:
                eval_result["approved_on_attempt"] = replan + 1
                break
            print(f"  Plan rejected. Replanning ({replan + 1}/{MAX_REPLAN})...")
            issue = parse_issue(
                issue_text + "\n\nPrevious plan feedback: " + eval_result["feedback"],
                available_files=available_files
            )
        if not eval_result["approved"]:
            print("  Warning: proceeding with unapproved plan after max replan attempts.")
    else:
        print("  Sentinel skipped (ablation: hetero_no_sentinel).")

    # retry loop
    previous_diagnoses = []
    previous_fix = None

    for attempt in range(1, max_attempts + 1):
        tracing.set_attempt(attempt)
        print(f"\n{'='*50}")
        print(f"ATTEMPT {attempt}/{max_attempts}")
        print("="*50)

        # Step 3: generate fix
        print("\n[Step 3/6] Generating fix...")
        diagnosis_text = previous_diagnoses[-1] if previous_diagnoses else None
        fixed_code, tokens = generate_fix(
            issue=issue,
            relevant_code=relevant_code,
            test_files=test_files,
            previous_attempt=previous_fix,
            diagnosis=diagnosis_text,
            prompting_strategy=prompting_strategy,
            model=model
        )

        # Step 4: apply and test
        print("\n[Step 4/6] Applying fix and running tests...")
        test_result = apply_and_test(fixed_code, repo_path, test_file)

        if test_result["passed"]:
            print("\n" + "="*50)
            print(f"SUCCESS on attempt {attempt}!")
            print("All tests passing.")
            print("="*50)

            if collection is not None:
                for filename in fixed_code.keys():
                    update_index(collection, repo_path, filename)

            experiment_log = {
                "configuration": config,
                "prompting_strategy": prompting_strategy,
                "task_id": task_id,
                "passed": True,
                "attempts": attempt,
                "evaluator_approved_on_attempt": eval_result.get("approved_on_attempt", None),
                "evaluator_score": eval_result.get("score", None),
                "rag_chunks_retrieved": nav_result.get("chunks_retrieved", 0),
                "latency_seconds": round(time.time() - start_time, 2),
            }
            _append_result(experiment_log)

            return True

        # Step 5: diagnose failure
        print(f"\n[Step 5/6] Tests failed. Running debugger...")
        previous_fix = list(fixed_code.values())[0] if fixed_code else ""

        original_file = issue["affected_files"][0]
        original_path = os.path.join(repo_path, original_file)

        with open(original_path, "r") as f:
            original_code = f.read()

        debug_result = diagnose(
            original_code=original_code,
            fixed_code=previous_fix,
            test_output=test_result["output"],
            previous_diagnoses=previous_diagnoses,
            filename=original_file
        )
        previous_diagnoses.append(debug_result.get("diagnosis", ""))

        with open(original_path, "r") as f:
            relevant_code[original_file] = f.read()

    print("\n" + "="*50)
    print(f"FAILED: Could not fix all tests in {max_attempts} attempts.")
    print("="*50)

    experiment_log = {
        "configuration": config,
        "prompting_strategy": prompting_strategy,
        "task_id": task_id,
        "passed": False,
        "attempts": max_attempts,
        "evaluator_approved_on_attempt": eval_result.get("approved_on_attempt", None),
        "evaluator_score": eval_result.get("score", None),
        "rag_chunks_retrieved": nav_result.get("chunks_retrieved", 0),
        "latency_seconds": round(time.time() - start_time, 2),
    }
    _append_result(experiment_log)

    return False


if __name__ == "__main__":
    issue_text = """
        In target.py only:
        - calculate_average must return 0.0 (a float) for an empty list, not raise an exception and not return None.
        - reverse_string must return the full reversed string, not just the last character.
        - count_vowels must count both uppercase and lowercase vowels (A, E, I, O, U and a, e, i, o, u).
        """

    # The demo fixture lives in its own directory so a demo run cannot write
    # to tracked repo source. repo_path IS that directory, so the agent sees
    # exactly target.py and test_target.py.
    repo_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "data", "demo_target")

    run(
        issue_text=issue_text,
        repo_path=repo_path,
        test_file="test_target.py",
        max_attempts=3,
        config="hetero",
        prompting_strategy="zero_shot",
        task_id="local/target_functions"
    )