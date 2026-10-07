# experiments/collect_humaneval.py
# Runs all 6 configurations x 3 prompting strategies across 30 HumanEval tasks
# Usage: python experiments/collect_humaneval.py

import os
import sys
import json
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maara.orchestrator import results_path, run

# Tasks live in data/tasks/ rather than inline. The default reproduces the
# Milestone 1 behaviour exactly -- the original 30, truncated to the first 15 --
# so the committed traces stay replayable. MAARA_TASK_SET=new selects the
# held-out set built by experiments/build_tasks.py.
#
#   MAARA_TASK_SET=original   HumanEval/1-15   (default; the fit set)
#   MAARA_TASK_SET=original30 HumanEval/1-30
#   MAARA_TASK_SET=new        the 54 validated held-out problems
#   MAARA_TASK_SET=final      every remaining problem, untouched by any earlier
#                             trace; the sealed test set for frozen policies
_TASK_SETS = {
    "original": ("humaneval_original30.json", 15),
    "original30": ("humaneval_original30.json", None),
    "new": ("humaneval_new.json", None),
    "final": ("humaneval_final.json", None),
}


def load_tasks(name=None):
    name = name or os.environ.get("MAARA_TASK_SET", "original")
    if name not in _TASK_SETS:
        raise SystemExit(f"MAARA_TASK_SET={name!r}; expected one of "
                         f"{sorted(_TASK_SETS)}")
    fn, limit = _TASK_SETS[name]
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(repo, "data", "tasks", fn)
    with open(path) as fh:
        tasks = json.load(fh)["tasks"]
    return tasks[:limit] if limit else tasks


TASKS = load_tasks()


CONFIGURATIONS = [
    {"config": "single_agent",       "models": {"coder": "qwen2.5-coder:7b"}},
    {"config": "homo_llama3",        "models": {"coder": "llama3.2"}},
    {"config": "hetero_no_rag",      "models": {"coder": "qwen2.5-coder:7b"}},
    {"config": "hetero_no_sentinel", "models": {"coder": "llama3.2"}},
]

PROMPTING_STRATEGIES = ["zero_shot", "few_shot"]
REPO_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Each task runs against its own throwaway directory rather than the repo. The
# agent's repo_path IS this directory, so get_file_structure, build_index and
# apply_and_test can only ever see the two files the task is about.
#
# Before this, repo_path was REPO_PATH, which is how a HumanEval run came to
# retrieve and overwrite simulator/workload.py. See agents/paths.py.
SANDBOX_DIR = os.path.join(REPO_PATH, "benchmark", "exp_sandbox")


def setup_task_sandbox(task: dict) -> str:
    """
    Build a clean directory holding exactly the two files the task is about.

    That file set is not arbitrary: before sim/ existed, orchestrator's
    available_files filter over the whole repo resolved to exactly
    ['target.py'], and get_test_file_contents to exactly {'test_target.py'}.
    A sandbox containing those two files reproduces that view, so traces stay
    comparable with the original 120 runs.
    """
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)
    os.makedirs(SANDBOX_DIR)

    with open(os.path.join(SANDBOX_DIR, "target.py"), "w") as f:
        f.write(task["buggy_code"])
    with open(os.path.join(SANDBOX_DIR, "test_target.py"), "w") as f:
        f.write(task["test_code"])

    return SANDBOX_DIR


def teardown_task_sandbox():
    if os.path.exists(SANDBOX_DIR):
        shutil.rmtree(SANDBOX_DIR)


def main():
    os.makedirs(os.path.join(REPO_PATH, "benchmark"), exist_ok=True)

    # load already completed runs so we can skip them on resume
    # Honours MAARA_RESULTS_PATH for BOTH reading and writing, so pointing a
    # traced run at a fresh file also gives it an empty resume set.
    results_file = results_path()
    logged = set()
    if os.path.exists(results_file):
        with open(results_file) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    logged.add((r["configuration"], r["prompting_strategy"], r["task_id"]))
                except Exception:
                    continue
        print(f"Resuming — {len(logged)} runs already completed, skipping those.")

    # The grid is already complete in results.jsonl, so resume would skip every
    # combination and a trace run would capture nothing. This forces a re-run.
    # Note it APPENDS duplicate rows to results.jsonl.
    if os.environ.get("MAARA_IGNORE_RESUME"):
        logged = set()
        print("Resume disabled (MAARA_IGNORE_RESUME) — re-running everything in scope.")

    # back up original files
    # No backup/restore of repo files needed any more: tasks run in a sandbox
    # and the repo's own target.py / test_target.py are never written.

    # Optional scoping, for smoke runs and for collecting a trace without
    # paying for the whole grid. All unset => identical behaviour to before.
    tasks = TASKS
    configs = CONFIGURATIONS
    strategies = PROMPTING_STRATEGIES
    if os.environ.get("MAARA_LIMIT_TASKS"):
        tasks = tasks[:int(os.environ["MAARA_LIMIT_TASKS"])]
    if os.environ.get("MAARA_ONLY_CONFIG"):
        configs = [c for c in configs if c["config"] == os.environ["MAARA_ONLY_CONFIG"]]
    if os.environ.get("MAARA_ONLY_STRATEGY"):
        strategies = [s for s in strategies if s == os.environ["MAARA_ONLY_STRATEGY"]]

    total = len(tasks) * len(configs) * len(strategies)
    completed = 0

    print(f"\n{'='*60}")
    print(f"MAARA EXPERIMENT RUNNER")
    print(f"Tasks: {len(tasks)} | Configs: {len(configs)} | Strategies: {len(strategies)}")
    print(f"Total runs: {total}")
    print(f"{'='*60}\n")

    for cfg in configs:
        for strategy in strategies:
            for task in tasks:
                completed += 1
                print(f"\n[{completed}/{total}] Config: {cfg['config']} | Strategy: {strategy} | Task: {task['task_id']}")

                if (cfg["config"], strategy, task["task_id"]) in logged:
                    print(f"  Already completed — skipping.")
                    continue

                try:
                    sandbox = setup_task_sandbox(task)
                    run(
                        issue_text=task["issue"],
                        repo_path=sandbox,
                        test_file="test_target.py",
                        max_attempts=3,
                        config=cfg["config"],
                        prompting_strategy=strategy,
                        task_id=task["task_id"],
                        model=cfg["models"]["coder"]
                    )

                except Exception as e:
                    print(f"  ERROR: {e}")
                    # log the failure so it still shows in results
                    import time
                    error_log = {
                        "configuration": cfg["config"],
                        "prompting_strategy": strategy,
                        "task_id": task["task_id"],
                        "passed": False,
                        "attempts": 0,
                        "evaluator_approved_on_attempt": None,
                        "evaluator_score": None,
                        "rag_chunks_retrieved": 0,
                        "latency_seconds": 0,
                        "error": str(e)
                    }
                    with open(results_file, "a") as f:
                        f.write(json.dumps(error_log) + "\n")

                finally:
                    # always restore originals after each task
                    teardown_task_sandbox()

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS COMPLETE")
    print(f"Results saved to {results_file}")
    print(f"Run: python analysis/pass_at_1.py to see pass@1 scores")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()