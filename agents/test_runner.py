import subprocess
import os
import shutil

from agents.paths import assert_within_root


def apply_fix(fixed_code: dict, repo_path: str) -> dict:
    """
    Write fixed code to disk.
    First creates backups so we can restore if needed.
    Returns { filename: backup_path }
    """
    backups = {}

    for filename, code in fixed_code.items():
        # Last line of defence. Everything upstream is a filter that can be
        # bypassed by a model returning an unexpected filename; this resolves
        # the path and refuses anything outside the root, including absolute
        # paths, `..` traversal and escaping symlinks.
        filepath = assert_within_root(repo_path, os.path.join(repo_path, filename))
        backup_path = filepath + ".backup"

        # save backup before overwriting
        if os.path.exists(filepath):
            shutil.copy2(filepath, backup_path)
            backups[filename] = backup_path

        with open(filepath, "w") as f:
            f.write(code)

        print(f"[Test Runner] Applied fix to {filename}")

    return backups


def restore_backups(backups: dict, repo_path: str):
    """Restore original files if tests fail badly."""
    for filename, backup_path in backups.items():
        filepath = assert_within_root(repo_path, os.path.join(repo_path, filename))
        if os.path.exists(backup_path):
            shutil.copy2(backup_path, filepath)
            os.remove(backup_path)
            print(f"[Test Runner] Restored {filename} from backup")


def run_tests(repo_path: str, test_file: str = "test_target.py") -> dict:
    """
    Run pytest and return structured results.
    """
    try:
        result = subprocess.run(
            ["python", "-m", "pytest", test_file, "-v", "--tb=short"],
            capture_output=True,
            text=True,
            cwd=repo_path,
            timeout=30
        )
    except subprocess.TimeoutExpired:
        print("[Test Runner] pytest timed out after 30 seconds — likely infinite loop in fix")
        return {
            "passed": False,
            "output": "TimeoutExpired: pytest exceeded 30 seconds",
            "failures": ["timeout — fixed code likely contains infinite loop"]
        }

    passed = result.returncode == 0
    output = result.stdout + result.stderr

    # parse which tests failed
    failures = []
    for line in output.split("\n"):
        if "FAILED" in line:
            failures.append(line.strip())

    print(f"\n[Test Runner] {'PASSED' if passed else 'FAILED'}")
    if failures:
        for f in failures:
            print(f"  {f}")

    return {
        "passed": passed,
        "output": output,
        "failures": failures
    }


def apply_and_test(fixed_code: dict, repo_path: str, test_file: str = "test_target.py") -> dict:
    """
    Main entry point: apply the fix, run tests, return results.
    Automatically restores backup if something catastrophic happens.
    """
    backups = apply_fix(fixed_code, repo_path)

    try:
        result = run_tests(repo_path, test_file)
        # clean up backups if tests ran (even if they failed — that's expected)
        for backup_path in backups.values():
            if os.path.exists(backup_path):
                os.remove(backup_path)
        return result
    except Exception as e:
        print(f"[Test Runner] Catastrophic error, restoring backups: {e}")
        restore_backups(backups, repo_path)
        return {
            "passed": False,
            "output": str(e),
            "failures": ["catastrophic error — backups restored"]
        }