"""
The program-level Gittins index and its schedulers (DECISIONS 38).

The index tests are closed-form; the scheduler tests use hand-built views; the
leak test runs the engine on synthetic traces.
"""

import dataclasses
import math

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.gittins import GittinsTable, pooling_threshold, tracelab_prior
from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, GittinsScheduler,
                                ReqView)
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, Kind,
                                StageTokens, WorkloadConfig, generate_trace)


def view(req_id, task_id, *, done=0, arrival=0.0, attained=0.0):
    return ReqView(req_id=req_id, task_id=task_id, stage="round", model=CODER_MODEL,
                   enqueued_at=arrival, prompt_tokens=100, attempt_index=done,
                   replan_index=0, priority=0, deadline=None, est_service_s=1.0,
                   est_remaining_s=1.0, task_arrival_s=arrival,
                   task_attained_s=attained, task_requests_done=done)


# --- the index ------------------------------------------------------------

def test_two_point_index_by_hand():
    """R in {1, 10} equally likely, unit work: from k=0 the best stop is after
    one request, finishing half the tasks for 2 units of expected work."""
    t = GittinsTable({1: 1, 10: 1}, [1.0], min_count=1)
    assert t.index(0) == pytest.approx(0.5)


def test_geometric_r_gives_a_constant_index():
    """Memoryless R: every task looks the same at every age, so the index is
    flat and ties fall back to task age (program-fcfs)."""
    p = 0.3
    counts = {r: 1e9 * p * (1 - p) ** (r - 1) for r in range(1, 80)}
    t = GittinsTable(counts, [1.0], min_count=1)
    assert all(t.index(k) == pytest.approx(p, rel=1e-6) for k in range(20))


def test_decreasing_hazard_gives_a_falling_index():
    """A mixture of a short and a long geometric (decreasing hazard): the
    longer a task has run, the likelier it is a long one, so fewer completed
    goes first."""
    counts = {r: 1e9 * (0.5 * 0.5 * 0.5 ** (r - 1) + 0.5 * 0.05 * 0.95 ** (r - 1))
              for r in range(1, 400)}
    t = GittinsTable(counts, [1.0], min_count=1)
    vals = [t.index(k) for k in range(20)]
    assert all(x >= y - 1e-12 for x, y in zip(vals, vals[1:])) and vals[0] > vals[19]


def test_increasing_hazard_gives_a_rising_index():
    """Uniform R: the further along, the sooner done, so most completed goes first."""
    t = GittinsTable({r: 1 for r in range(1, 11)}, [1.0], min_count=1)
    vals = [t.index(k) for k in range(9)]
    assert vals == sorted(vals) and vals[8] > vals[0]


def test_slowdown_weight_matches_latency_when_sizes_are_equal():
    t_lat = GittinsTable({5: 10}, [2.0], tool_s=1.0, weight="latency", min_count=1)
    t_sd = GittinsTable({5: 10}, [2.0], tool_s=1.0, weight="slowdown", min_count=1)
    size = 5 * 2.0 + 5 * 1.0
    for k in range(5):
        assert t_sd.index(k) == pytest.approx(t_lat.index(k) / size)


def test_thin_states_share_the_last_well_populated_index():
    t = GittinsTable({1: 100, 2: 50, 3: 5, 40: 2}, [1.0], min_count=30)
    assert t.k_tail == 1
    assert t.index(7) == t.index(1)


def test_pooling_threshold():
    assert pooling_threshold(36853) == 30
    assert pooling_threshold(15) == 3


def test_tracelab_prior_matches_the_design_numbers():
    from simulator.adapters.tracelab import load_dev_constants
    cost = CostModel.from_profile("gpu_5060ti_public")
    r_counts, work, tool = tracelab_prior(cost, load_dev_constants())
    assert work[0] == pytest.approx(17.9, abs=0.05)
    assert work[1] == pytest.approx(8.3, abs=0.05)
    assert tool == pytest.approx(8.42, abs=0.01)
    t = GittinsTable(r_counts, work, tool, min_count=30)
    # A turn with one round done is the best bet (DECISIONS 38 review table:
    # 0.0260, 0.0511, 0.0272, 0.0162 at k = 0..3).
    assert t.index(1) > t.index(2) > t.index(0) > t.index(3)
    assert t.index(1) == pytest.approx(0.0511, abs=5e-5)


# --- the schedulers -------------------------------------------------------

def _bound(name, r_counts, work, tool=0.0):
    s = SCHEDULERS[name]()

    class Est:
        def gittins_prior(self, cost):
            return r_counts, work, tool
    s.bind(None, Est())
    return s


def test_gittins_orders_by_index_then_task_age():
    s = _bound("gittins", {1: 60, 2: 20, 50: 20}, [1.0])
    vs = [view(1, 1, done=3, arrival=0.0), view(2, 2, done=0, arrival=5.0),
          view(3, 3, done=0, arrival=1.0)]
    assert s.order(vs, now=10.0) == [3, 2, 1]


def test_unbound_gittins_orders_by_fewest_completed():
    vs = [view(1, 1, done=2), view(2, 2, done=0, arrival=3.0), view(3, 3, done=1)]
    assert GittinsScheduler().order(vs, now=0.0) == [2, 3, 1]


@pytest.mark.parametrize("name", ["gittins-guard-x2", "gittins-sd-guard-x2", "plas-guard-x2"])
def test_guard_promotes_only_the_single_oldest_overdue_task(name):
    s = _bound(name, {1: 60, 2: 20, 50: 20}, [1.0])
    limit = 2 * s.mean_size
    vs = [view(1, 1, done=5, arrival=0.0, attained=99.0),       # oldest overdue
          view(2, 2, done=5, arrival=1.0, attained=99.0),       # also overdue
          view(3, 3, done=0, arrival=limit + 5.0)]              # fresh
    order = s.order(vs, now=limit + 10.0)
    assert order[0] == 1
    assert order.index(3) < order.index(2)     # the second overdue task is not lifted


def test_guard_is_inactive_before_the_limit():
    s = _bound("gittins-guard-x8", {1: 60, 2: 20, 50: 20}, [1.0])
    vs = [view(1, 1, done=5, arrival=0.0), view(2, 2, done=0, arrival=1.0)]
    assert s.order(vs, now=1.0) == _bound("gittins", {1: 60, 2: 20, 50: 20}, [1.0]).order(vs, now=1.0)


# --- engine: the state is observable and the policy reads no future --------

def _run(tasks, name, record=None):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    sched = SCHEDULERS[name]()
    if record is not None:
        inner = sched.order

        def order(views, now):
            out = inner(views, now)
            record.append((now, tuple(out)))
            return out
        sched.order = order
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=2048.0, max_batch=4),
                 retention=RETENTIONS["lru"](), scheduler=sched,
                 estimator=ESTIMATORS["online"](cost, tok, cpu))
    return eng, eng.run()


def _tasks(seed=5):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    return generate_trace(WorkloadConfig(n_tasks=10, arrival_rate=0.08, seed=seed), cost, tok, cpu)


def test_requests_done_counts_finished_requests():
    eng, rep = _run(_tasks(), "gittins")
    for t in eng.tasks:
        n_llm = sum(1 for n in t.nodes if n.kind is not Kind.CPU)
        assert eng.task_requests_done[t.task_id] == n_llm


@pytest.mark.parametrize("name", ["gittins-guard-x2", "gittins-sd", "plas-guard-x4"])
def test_future_output_lengths_do_not_change_earlier_decisions(name):
    """Two traces that differ only in the output length of each task's LAST
    LLM request: every scheduling decision before the first such request is
    admitted must be identical."""
    a = _tasks()
    b = []
    for t in a:
        last = max((n for n in t.nodes if n.kind is not Kind.CPU), key=lambda n: n.node_id)
        b.append(dataclasses.replace(t, nodes=[
            dataclasses.replace(n, output_tokens=n.output_tokens * 3 + 7) if n is last
            else dataclasses.replace(n) for n in t.nodes]))
    rec_a, rec_b = [], []
    eng_a, _ = _run(a, name, rec_a)
    _run(b, name, rec_b)
    lasts = {(t.task_id, max(n.node_id for n in t.nodes if n.kind is not Kind.CPU)) for t in a}
    first_last_start = min(r.started_at for r in eng_a.requests
                           if (r.node.task_id, r.node.node_id) in lasts)
    early_a = [x for x in rec_a if x[0] < first_last_start]
    early_b = [x for x in rec_b if x[0] < first_last_start]
    assert early_a and early_a == early_b
    assert not math.isnan(first_last_start)
