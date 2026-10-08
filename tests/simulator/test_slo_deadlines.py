"""
Per-task deadlines (SLO classes) and the program-level baselines.

FINDINGS 29: with one common deadline budget, EDF is exactly oldest-task-first,
so no deadline-aware rule can show value. These pin the fix: default runs are
unchanged, every policy sees the same classes, and the baselines order by
what they claim to.
"""

import math

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, HintPriorityScheduler,
                                ProgramFCFSScheduler, ReqView)
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)

MIX = (("tight", 1.0, 0.3), ("standard", 2.0, 0.4), ("loose", 4.0, 0.3))


def engine(sched="fcfs", slo=None, seed=5, n_tasks=30, rate=0.05):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed),
                           cost, tok, cpu)
    return Engine(tasks, cost,
                  EngineConfig(kv_budget_mb=2048.0, max_batch=4, slack_factor=1.5,
                               slo_classes=slo, slo_seed=seed),
                  retention=RETENTIONS["lru"](), scheduler=SCHEDULERS[sched](),
                  estimator=ESTIMATORS["online"](cost, tok, cpu))


def test_default_deadlines_are_the_single_budget():
    eng = engine()
    budget = eng.cfg.slack_factor * eng._isolated_task_s
    for t in eng.tasks:
        assert eng.task_deadline[t.task_id] == t.arrival_time + budget
    assert eng.task_slo == {}


def test_every_policy_sees_the_same_classes_and_deadlines():
    ref = engine("fcfs", MIX)
    for sched in ("lsf", "edf", "program-fcfs", "hint-priority"):
        other = engine(sched, MIX)
        assert other.task_slo == ref.task_slo
        assert other.task_deadline == ref.task_deadline


def test_classes_follow_the_mix_and_change_the_budget():
    eng = engine(slo=MIX, n_tasks=300, rate=0.5)
    names = [c for c, _ in eng.task_slo.values()]
    assert 0.2 < names.count("tight") / len(names) < 0.4
    assert {r for _, r in eng.task_slo.values()} == {0, 1, 2}
    budgets = {t.task_id: eng.task_deadline[t.task_id] - t.arrival_time for t in eng.tasks}
    assert len({round(b, 6) for b in budgets.values()}) > 3


def test_per_class_miss_rates_add_up():
    rep = engine(slo=MIX).run()
    by = rep["miss_rate_by_slo_class"]
    missed = sum(v["miss_rate"] * v["tasks"] for v in by.values())
    total = sum(v["tasks"] for v in by.values())
    assert math.isclose(missed / total, rep["deadline_miss_rate"], abs_tol=1e-12)


def _view(rid, task_arrival, enq, prio=0):
    return ReqView(req_id=rid, task_id=rid, stage="generate_fix", model=CODER_MODEL,
                   enqueued_at=enq, prompt_tokens=100, attempt_index=1, replan_index=0,
                   priority=0, deadline=None, est_service_s=1.0, est_remaining_s=1.0,
                   task_arrival_s=task_arrival, slo_priority=prio)


def test_program_fcfs_orders_by_task_arrival_not_request_arrival():
    vs = [_view(1, task_arrival=5.0, enq=1.0), _view(2, task_arrival=1.0, enq=9.0)]
    assert ProgramFCFSScheduler().order(vs, now=10.0) == [2, 1]


def test_hint_priority_orders_by_class_then_task_age():
    vs = [_view(1, 1.0, 1.0, prio=2), _view(2, 5.0, 5.0, prio=0), _view(3, 2.0, 2.0, prio=0)]
    assert HintPriorityScheduler().order(vs, now=10.0) == [3, 2, 1]


def test_with_one_budget_edf_is_program_fcfs():
    """The FINDINGS 29 mechanism, pinned: a common budget makes EDF oldest-task-first."""
    a, b = engine("edf").run(), engine("program-fcfs").run()
    for k in ("p95_task_latency_s", "deadline_miss_rate", "makespan_s"):
        assert a[k] == b[k], k
