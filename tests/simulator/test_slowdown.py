"""
p95_slowdown / mean_slowdown: latency over the task's time ALONE on an idle
machine under the run's cost model (Engine._isolated_service_s). Computed after
the run from completed tasks; no policy can see it.
"""

import dataclasses

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import RETENTIONS, SCHEDULERS, ReqView
from simulator.workload import (CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)


def _run(n_tasks, rate, seed=4, batch=4):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed),
                           cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=4096.0, max_batch=batch),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS["fcfs"]())
    return eng, eng.run()


def test_lone_task_slowdown_is_one():
    eng, rep = _run(1, 0.1)
    assert rep["tasks_completed"] == 1
    assert rep["p95_slowdown"] == pytest.approx(1.0, abs=1e-9)
    assert rep["mean_slowdown"] == pytest.approx(1.0, abs=1e-9)


def test_lightly_loaded_slowdown_is_about_one():
    # Arrivals ~hundreds of seconds apart, so tasks barely overlap. A task can
    # land a little under 1 only by riding blocks an earlier task left warm
    # (the lone replay starts cold); it can never be far under.
    eng, rep = _run(20, 0.002)
    slow = [(eng.task_done_at[t.task_id] - t.arrival_time) / eng._isolated_service_s(t)
            for t in eng.tasks]
    assert min(slow) >= 0.9
    assert rep["mean_slowdown"] == pytest.approx(sum(slow) / len(slow))
    assert rep["p95_slowdown"] >= rep["mean_slowdown"] * 0.99


def test_heavy_load_raises_slowdown():
    _, light = _run(20, 0.002)
    _, heavy = _run(20, 2.0)
    assert heavy["p95_slowdown"] > light["p95_slowdown"] * 2


def test_slowdown_is_not_on_the_policy_view_and_is_deterministic():
    assert not any("slowdown" in f.name for f in dataclasses.fields(ReqView))
    _, a = _run(10, 0.05)
    _, b = _run(10, 0.05)
    assert repr(a) == repr(b)      # repr: the report holds NaNs, which never compare equal


def test_isolated_cache_keys_are_fixed_size_digests():
    # The memo once keyed on full node reprs: ~28 MB per TraceLab task set,
    # added per seed and load, which filled swap mid-map (DECISIONS 8 item 12).
    from simulator import engine as engine_mod
    engine_mod._ISOLATED_CACHE.clear()
    _, first = _run(6, 0.05)
    keys = list(engine_mod._ISOLATED_CACHE)
    assert keys and all(isinstance(k, bytes) and len(k) == 32 for k in keys)
    _, again = _run(6, 0.05)                 # all hits: same answer, no growth
    assert len(engine_mod._ISOLATED_CACHE) == len(keys)
    assert again["p95_slowdown"] == first["p95_slowdown"]
