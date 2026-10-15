"""
The scheduler-set concurrency cap (Scheduler.admit_cap) and the adaptive-cap
program-FCFS rule built on it (plan Phase B, mentor review 2026-10-08).
"""

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import ESTIMATORS, RETENTIONS, SCHEDULERS
from simulator.workload import (CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)


def run(sched, coeff=None, batch=8, rate=0.5, n_tasks=30, seed=3):
    cost = CostModel() if coeff is None else CostModel(batch_decode_coeff=coeff)
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed),
                           cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=4096.0, max_batch=batch),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS[sched](),
                 estimator=ESTIMATORS["online"](cost, tok, cpu))
    return eng.run()


@pytest.mark.parametrize("sched", ["fcfs", "program-fcfs", "lsf"])
def test_schedulers_without_a_cap_run_at_max_batch(sched):
    rep = run(sched)
    assert rep["mean_admit_cap"] == 8.0


def test_adaptive_cap_holds_resident_requests_under_its_cap_on_a_serial_batcher():
    # c = 0.841: mu saturates near 1/c, so the cap is 3 at max_batch 8.
    rep = run("program-fcfs-cap")
    assert rep["mean_admit_cap"] == 3.0
    assert rep["max_resident"] <= 3
    # Every task still completes: a cap must never starve the run.
    assert rep["throughput_tasks_per_s"] > 0
    plain = run("program-fcfs")
    assert rep["max_resident"] <= plain["max_resident"]


def test_efficient_batching_earns_a_higher_cap():
    serial = run("program-fcfs-cap", coeff=0.841)
    efficient = run("program-fcfs-cap", coeff=0.15)
    assert efficient["mean_admit_cap"] > serial["mean_admit_cap"]


def test_the_cap_is_one_only_when_batching_buys_nothing():
    # c = 1: batching is fully serial, so one request at a time loses nothing.
    rep = run("program-fcfs-cap", coeff=1.0, n_tasks=10)
    assert rep["mean_admit_cap"] == 1.0
    assert rep["max_resident"] == 1
    assert rep["throughput_tasks_per_s"] > 0
