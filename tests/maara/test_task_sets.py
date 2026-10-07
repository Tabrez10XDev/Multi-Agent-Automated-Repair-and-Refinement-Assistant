"""
The task files must stay disjoint: a held-out or final set that shares a
problem with an earlier trace is not held out.

Reads data/tasks/ directly rather than importing collect_humaneval, which
pulls in the agent pipeline.
"""

import json
import os

_TASKS = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "tasks")


def _ids(name):
    with open(os.path.join(_TASKS, name)) as fh:
        return {t["task_id"] for t in json.load(fh)["tasks"]}


def test_final_set_is_untouched_by_every_earlier_trace():
    final = _ids("humaneval_final.json")
    assert final.isdisjoint(_ids("humaneval_original30.json"))
    assert final.isdisjoint(_ids("humaneval_new.json"))
    # Dropped by validation from the held-out set: looked at, so not untouched.
    assert "HumanEval/32" not in final


def test_final_set_size_and_provenance():
    with open(os.path.join(_TASKS, "humaneval_final.json")) as fh:
        d = json.load(fh)
    assert len(d["tasks"]) == 92
    assert d["source_sha256"] == ("b796127e635a67f93fb35c04f4cb03cf06f38c8072"
                                  "ee7cee8833d7bee06979ef")
    assert {"task_id", "issue", "buggy_code", "test_code"} <= set(d["tasks"][0])


def test_every_humaneval_problem_is_accounted_for():
    used = (_ids("humaneval_original30.json") | _ids("humaneval_new.json")
            | _ids("humaneval_final.json"))
    dropped = {"HumanEval/32", "HumanEval/38", "HumanEval/50"}
    assert used.isdisjoint(dropped)
    assert len(used | dropped) == 164
