"""
The activity and miss-split metrics the regime maps read.

A map cell is only interpretable if it says whether the mechanism had room to
act (choice, a binding batch cap) and whether its misses were ones a
scheduler could have prevented. These pin the arithmetic of those numbers.
"""

import math

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import ESTIMATORS, RETENTIONS, SCHEDULERS
from simulator.workload import (CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)


def run(slack=2.0, batch=4, rate=0.05, n_tasks=12, seed=5):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed),
                           cost, tok, cpu)
    eng = Engine(tasks, cost,
                 EngineConfig(kv_budget_mb=2048.0, max_batch=batch, slack_factor=slack),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS["fcfs"](),
                 estimator=ESTIMATORS["online"](cost, tok, cpu))
    return eng, eng.run()


@pytest.mark.parametrize("slack", [0.6, 1.5, 4.0])
def test_miss_split_sums_to_the_miss_rate(slack):
    _, rep = run(slack=slack)
    total = rep["miss_rate_infeasible_at_admit"] + rep["miss_rate_feasible_at_admit"]
    assert math.isclose(total, rep["deadline_miss_rate"], abs_tol=1e-12)


def test_a_deadline_shorter_than_the_work_is_infeasible_from_the_start():
    # slack below 1 x isolated task time: no task can make it even alone,
    # so every miss is already infeasible at its first admission.
    _, rep = run(slack=0.3)
    assert rep["deadline_miss_rate"] > 0
    assert rep["miss_rate_feasible_at_admit"] == 0.0


def test_batch_percentile_and_cap_fraction_are_bounded():
    eng, rep = run(batch=4, rate=0.2)
    assert 1 <= rep["p95_batch"] <= rep["max_batch_seen"] <= 4
    assert 0.0 <= rep["frac_time_at_batch_cap"] <= 1.0
    if rep["max_batch_seen"] < 4:
        assert rep["frac_time_at_batch_cap"] == 0.0
    assert math.isclose(sum(eng.m["batch_time"].values()), eng.m["w_time"], rel_tol=1e-9)


def test_a_cap_that_binds_is_reported_as_binding():
    _, rep = run(batch=1, rate=0.5)
    assert rep["max_batch_seen"] == 1
    assert rep["frac_time_at_batch_cap"] > 0.9


def test_resident_cap_metric_is_what_the_cap_is_enforced_on():
    # OPEN_ISSUES §28: the cap limits resident requests across both models, but
    # the active batch counts only the model served each iteration. The
    # resident view must never read lower, and must sum to wall time.
    eng, rep = run(batch=4, rate=0.5, n_tasks=20)
    assert rep["max_batch_seen"] <= rep["max_resident"] <= 4
    assert rep["frac_time_at_batch_cap"] <= rep["frac_time_at_resident_cap"] <= 1.0
    assert math.isclose(sum(eng.m["resident_time"].values()), eng.m["w_time"], rel_tol=1e-9)
    assert rep["max_resident"] == 4 and rep["frac_time_at_resident_cap"] > 0.0
