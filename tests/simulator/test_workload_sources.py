"""
The workload-source registry (simulator/workload_sources.py): drivers reach
traces only through resolve_workload, so a future public-trace adapter can
register a name without a driver change, and the sealed gate still holds.
"""

import os
import sys

import pytest

from experiments.common import build, make_estimator
from simulator import workload_sources as ws
from simulator.policies import ExecutionEstimator
from simulator.trace_loader import (SEALED_TRACE_SETS, TRACE_SET_EXCLUSIONS,
                                    TRACE_SETS, SealedTraceSet, load_trace)
from simulator.workload import (CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _dummy_tasks(name, seed, arrival_rate):
    cost, tok, cpu = CostModel(), StageTokens(), CpuStageSeconds()
    return generate_trace(WorkloadConfig(n_tasks=3, arrival_rate=arrival_rate or 0.1,
                                         seed=seed), cost, tok, cpu)


@pytest.fixture
def registered():
    added = []

    def reg(key, **kw):
        added.append(key)
        return ws.register_source(key, _dummy_tasks, **kw)
    yield reg
    for k in added:
        ws.unregister_source(k)


def test_maara_name_resolves_to_the_same_tasks_as_load_trace():
    name = "holdout_qwen"
    paths = [os.path.join(REPO, p) for p in TRACE_SETS[name]]
    direct = load_trace(paths, arrival_rate=0.05, seed=3,
                        exclude_tasks=TRACE_SET_EXCLUSIONS.get(name, {})).tasks
    assert ws.resolve_workload(name, 3, 0.05) == direct
    assert build(1, 3, 0.05, None, None, None, trace_set=name) == direct


@pytest.mark.parametrize("name", sorted(SEALED_TRACE_SETS))
def test_maara_sealed_refusal_still_fires(name, monkeypatch):
    monkeypatch.delenv("MAARA_POLICIES_FROZEN", raising=False)
    with pytest.raises(SealedTraceSet):
        ws.resolve_workload(name, 1, 0.03)


def test_dummy_source_is_reachable_through_build(registered):
    registered("dummy_dev")
    tasks = build(1, 5, 0.2, None, None, None, trace_set="dummy_dev")
    assert len(tasks) == 3 and tasks[0].nodes
    assert "dummy_dev" in ws.known_sources()


def test_prefix_registration_serves_a_family(registered):
    registered("dummyfam_", prefix=True)
    assert len(ws.resolve_workload("dummyfam_dev", 1, 0.1)) == 3
    assert len(ws.resolve_workload("dummyfam_test", 1, 0.1)) == 3


def test_adapter_declared_sealed_name_uses_the_same_gate(registered, monkeypatch):
    registered("dummy_sealed", sealed=True)
    monkeypatch.delenv("MAARA_POLICIES_FROZEN", raising=False)
    with pytest.raises(SealedTraceSet, match="DECISIONS 26"):
        build(1, 1, 0.1, None, None, None, trace_set="dummy_sealed")
    monkeypatch.setenv("MAARA_POLICIES_FROZEN", "abc1234")
    assert len(build(1, 1, 0.1, None, None, None, trace_set="dummy_sealed")) == 3


def test_unknown_name_lists_the_known_ones():
    with pytest.raises(KeyError) as e:
        ws.resolve_workload("no_such_trace", 1, 0.1)
    msg = str(e.value)
    assert "no_such_trace" in msg and "holdout_qwen" in msg


def test_estimator_factory_is_used_for_its_source_only(registered):
    sentinel = object()
    registered("dummy_est", estimator_factory=lambda cost, tok, cpu: sentinel)
    cost, tok, cpu = CostModel(), StageTokens(), CpuStageSeconds()
    assert make_estimator("dummy_est", cost, tok, cpu) is sentinel
    # a named non-online estimator is still the stock one
    assert type(make_estimator("dummy_est", cost, tok, cpu, name="oracle")).__name__ \
        == "OracleEstimator"
    # MAARA sets get exactly today's estimator
    assert type(make_estimator("holdout_qwen", cost, tok, cpu)) is ExecutionEstimator


def test_unsupported_deadline_basis_is_refused_by_scheduling_map(registered, monkeypatch):
    registered("dummy_task_only", supports_deadline_basis=("task",))
    ws.check_deadline_basis("dummy_task_only", "task")
    with pytest.raises(ValueError, match="does not support deadline_basis"):
        ws.check_deadline_basis("dummy_task_only", "estimate")
    from experiments import scheduling_map
    monkeypatch.setattr(sys, "argv", ["scheduling_map", "--trace-set", "dummy_task_only"])
    with pytest.raises(ValueError, match="does not support deadline_basis"):
        scheduling_map.main()
    # MAARA supports both
    ws.check_deadline_basis("holdout_qwen", "estimate")
    ws.check_deadline_basis("holdout_qwen", "task")


def test_bad_basis_declaration_is_rejected():
    with pytest.raises(ValueError):
        ws.register_source("dummy_bad", _dummy_tasks, supports_deadline_basis=("nope",))
