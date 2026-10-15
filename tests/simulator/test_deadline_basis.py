"""
Per-task deadlines and the capacity load axis (OPEN_ISSUES 29, 30).

The "estimate" basis gives every holdout task nearly the same budget
(90.2 s x 0.35-0.41), so deadlines carry no per-task information. The "task"
basis sizes each budget to the task's own critical path, with seeded noise.
"""

import math
import statistics as st

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import ESTIMATORS, RETENTIONS, SCHEDULERS
from simulator.workload import (CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)

SLO = (("tight", 1.0, 0.3), ("standard", 2.0, 0.4), ("loose", 4.0, 0.3))


def engine(basis, sched="fcfs", slo=None, sigma=0.3, seed=4):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=40, arrival_rate=0.02, seed=seed),
                           cost, tok, cpu)
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=4096.0, max_batch=4, slack_factor=2.0,
                              slo_classes=slo, slo_seed=seed, deadline_basis=basis,
                              deadline_noise_sigma=sigma),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS[sched](),
                 estimator=ESTIMATORS["online"](cost, tok, cpu))
    return eng, tasks


def budgets(eng, tasks):
    return {t.task_id: eng.task_deadline[t.task_id] - t.arrival_time for t in tasks}


def test_task_basis_tracks_each_tasks_own_size():
    eng, tasks = engine("task", sigma=0.0)
    b = budgets(eng, tasks)
    for t in tasks:
        assert math.isclose(b[t.task_id], 2.0 * t.critical_path_total, rel_tol=1e-9)


def test_task_basis_noise_is_seeded_and_identical_across_schedulers():
    e1, tasks = engine("task", sched="fcfs", slo=SLO)
    e2, _ = engine("task", sched="lsf", slo=SLO)
    assert budgets(e1, tasks) == budgets(e2, tasks)
    ratios = [budgets(e1, tasks)[t.task_id] / t.critical_path_total for t in tasks]
    assert st.pstdev(ratios) > 0      # noise and classes actually vary the budget


def test_estimate_basis_is_unchanged_by_default():
    assert EngineConfig().deadline_basis == "estimate"
    eng, tasks = engine("estimate")
    b = set(round(v, 9) for v in budgets(eng, tasks).values())
    assert len(b) == 1                # one common budget without classes


def test_unknown_basis_is_refused():
    with pytest.raises(ValueError):
        engine("oracle")


def test_saturation_throughput_is_positive_and_grows_with_efficient_batching():
    from experiments.manifest import Manifest
    from experiments.scheduling_map import calibrate_saturation
    tok, cpu = StageTokens(), CpuStageSeconds()
    out = {}
    for coeff in (0.841, 0.15):
        cost = CostModel(batch_decode_coeff=coeff)
        m = Manifest.start("test", params={})
        sat, cal = calibrate_saturation(m, cost, tok, cpu, cost.profile, 8, 1024.0,
                                        "lru", "holdout_qwen", "block", [1])
        assert sat > 0 and cal.value == sat
        out[coeff] = sat
    assert out[0.15] > out[0.841]
