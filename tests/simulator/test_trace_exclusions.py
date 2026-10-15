"""
TRACE_SET_EXCLUSIONS drops a task before the simulator sees it (DECISIONS 31).
"""

import gzip
import json
import os

import pytest

from simulator.trace_loader import TRACE_SET_EXCLUSIONS, load_trace

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_QWEN = os.path.join(_REPO, "data/traces/final_A_qwen.jsonl.gz")


def test_the_final_qwen_exclusion_is_the_runaway_call():
    # Reads the committed file directly, not through the sealed resolver: this
    # checks the data behind the exclusion, and runs no policy.
    rows = [json.loads(l) for l in gzip.open(FINAL_QWEN, "rt") if l.strip()]
    worst = max(rows, key=lambda r: r.get("output_tokens") or 0)
    assert set(TRACE_SET_EXCLUSIONS["final_qwen"]) == {worst["task_id"]}
    assert worst["output_tokens"] == 40960
    others = [r.get("output_tokens") or 0 for r in rows if r["task_id"] != worst["task_id"]]
    assert max(others) < 1000


def test_exclude_tasks_drops_every_row_of_the_task():
    path = os.path.join(_REPO, "data/traces/holdout_A_qwen2.jsonl.gz")
    full = load_trace(path).tasks
    rows = [json.loads(l) for l in gzip.open(path, "rt") if l.strip()]
    tid = rows[0]["task_id"]
    cut = load_trace(path, exclude_tasks={tid}).tasks
    assert len(cut) == len(full) - 1
