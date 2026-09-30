import os
import sys
import json
import shutil
import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tracing
from benchmark.metrics import MetricsTracker
from agents.issue_parser import parse_issue
from agents.navigator import navigate, get_file_structure
from agents.coder import generate_fix
from agents.test_runner import apply_and_test
from agents.debugger import diagnose
from agents.retriever import update_index


CASES_DIR = os.path.join(os.path.dirname(__file__), "cases")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CASES_JSON = os.path.join(os.path.dirname(__file__), "cases.json")

# temporary working directory for each benchmark case
SANDBOX_DIR = os.path.join(os.path.dirname(__file__), "sandbox")


def setup_sandbox(case: dict) -> str:
    """
    Copy the buggy file and test file into a clean sandbox directory.
    Returns the sandbox path.
    """
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)

    # copy buggy source file
    src = os.path.join(CASES_DIR, case["buggy_file"])
    dst = os.path.join(SANDBOX_DIR, case["buggy_file"])
    shutil.copy2(src, dst)

    # copy test file
    test_src = os.path.join(CASES_DIR, case["test_file"])
    test_dst = os.path.join(SANDBOX_DIR, case["test_file"])
    shutil.copy2(test_src, test_dst)

    return SANDBOX_DIR


def run_case(case: dict, tracker: MetricsTracker, max_attempts: int = 3) -> bool:
    """Thin tracing wrapper. All behaviour lives in _run_case_impl, unchanged."""
    with tracing.task(case["id"]):
        return _run_case_impl(case, tracker, max_attempts)


def _run_case_impl(case: dict, tracker: MetricsTracker, max_attempts: int = 3) -> bool:
    """Run a single benchmark case. Returns True if solved."""
    print(f"\n{'='*50}")
    print(f"Case {case['id']} [{case['difficulty'].upper()}]")
    print(f"Issue: {case['issue']}")
    print("="*50)

    tracker.start_case()
    tokens = 0  # add this line
    sandbox = setup_sandbox(case)

    try:
        # get available files in sandbox
        available_files = [
            f for f in os.listdir(sandbox)
            if f.endswith(".py") and not f.startswith("test_")
        ]

        # parse issue
        issue = parse_issue(case["issue"], available_files=available_files)

        # navigate
        nav_result = navigate(sandbox, case["issue"], issue["affected_files"])
        relevant_code = nav_result["relevant_code"]
        test_files = nav_result["test_files"]
        collection = nav_result["collection"]

        if not relevant_code:
            print(f"[Runner] No relevant code found for {case['id']}")
            result = tracker.finish_case(
                case["id"], case["difficulty"],
                success=False, attempts=0,
                failure_reason="navigator found no relevant code"
            )
            return False

        previous_diagnoses = []
        previous_fix = None

        for attempt in range(1, max_attempts + 1):
            tracing.set_attempt(attempt)
            print(f"\n  Attempt {attempt}/{max_attempts}...")

            diagnosis_text = previous_diagnoses[-1] if previous_diagnoses else None
            fixed_code, tokens = generate_fix(
                issue=issue,
                relevant_code=relevant_code,
                test_files=test_files,
                previous_attempt=previous_fix,
                diagnosis=diagnosis_text
            )

            test_result = apply_and_test(
                fixed_code,
                sandbox,
                test_file=case["test_file"]
            )

            if test_result["passed"]:
                print(f"  ✓ Passed on attempt {attempt}")
                update_index(collection, sandbox, case["buggy_file"])
                tracker.record_tokens(tokens)        # record BEFORE finish
                tracker.finish_case(
                    case["id"], case["difficulty"],
                    success=True, attempts=attempt
                )
                return True

            previous_fix = list(fixed_code.values())[0] if fixed_code else ""

            original_path = os.path.join(sandbox, case["buggy_file"])
            with open(original_path, "r") as f:
                original_code = f.read()

            debug_result = diagnose(
                original_code=original_code,
                fixed_code=previous_fix,
                test_output=test_result["output"],
                previous_diagnoses=previous_diagnoses
            )
            previous_diagnoses.append(debug_result.get("diagnosis", ""))

            with open(original_path, "r") as f:
                relevant_code[case["buggy_file"]] = f.read()

        # all attempts exhausted
        tracker.record_tokens(tokens)            # record BEFORE finish
        tracker.finish_case(
            case["id"], case["difficulty"],
            success=False, attempts=max_attempts,
            failure_reason=f"failed after {max_attempts} attempts"
        )
        return False

    except Exception as e:
        print(f"  ✗ Error: {e}")
        tracker.finish_case(
            case["id"], case["difficulty"],
            success=False, attempts=0,
            failure_reason=str(e)
        )
        return False

    finally:
        # always clean up sandbox
        if os.path.exists(SANDBOX_DIR):
            shutil.rmtree(SANDBOX_DIR)


def run_benchmark(case_ids: list = None, max_attempts: int = 3):
    """
    Run the full benchmark or a subset of cases.
    case_ids: list of case IDs to run, e.g. ["case_01", "case_02"].
              If None, runs all cases.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)

    with open(CASES_JSON) as f:
        all_cases = json.load(f)

    if case_ids:
        cases = [c for c in all_cases if c["id"] in case_ids]
    else:
        cases = all_cases

    print(f"\nRunning {len(cases)} benchmark cases...")

    tracker = MetricsTracker()

    for case in cases:
        run_case(case, tracker, max_attempts=max_attempts)

    tracker.print_summary()

    # save results
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"run_{timestamp}.json")
    tracker.save(results_path)

    return tracker.summary()


if __name__ == "__main__":
    # run just the easy cases first to verify everything works
    run_benchmark(
        case_ids=["case_01", "case_02", "case_03", "case_04", "case_05", "case_06"],
        max_attempts=3
    )