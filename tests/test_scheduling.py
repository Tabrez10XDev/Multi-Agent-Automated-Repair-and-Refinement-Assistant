"""
Phase 3 tests: schedulers, the online estimator, and the oracle guard.

No LLM is called and no trace is read; everything runs on synthetic traces.
"""

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import (ESTIMATORS, RETENTIONS, SCHEDULERS, EDFScheduler,
                          ExecutionEstimator, FCFSScheduler,
                          LeastSlackFirstScheduler, OracleEstimator,
                          OracleLeakError, ReqView, SJFScheduler,
                          StaticPriorityScheduler)
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, Kind,
                          StageTokens, WorkloadConfig, generate_trace)


def view(req_id, *, stage="generate_fix", enqueued_at=0.0, deadline=None,
         est_remaining_s=10.0, est_service_s=1.0, task_id=0, attempt_index=1):
    return ReqView(
        req_id=req_id, task_id=task_id, stage=stage, model=CODER_MODEL,
        enqueued_at=enqueued_at, prompt_tokens=100, attempt_index=attempt_index,
        replan_index=0, priority=0, deadline=deadline,
        est_service_s=est_service_s, est_remaining_s=est_remaining_s)


def build(n_tasks=8, seed=3, stable_tokens=None, rate=0.03):
    cost = CostModel()
    tok = StageTokens.from_measured()
    if stable_tokens:
        tok.stable_prefix_tokens = stable_tokens
    cpu = CpuStageSeconds.from_measured()
    tasks = generate_trace(
        WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed), cost, tok, cpu)
    return cost, tok, cpu, tasks


def run_engine(scheduler="fcfs", estimator="online", seed=3, n_tasks=8,
               kv=2048.0, batch=8, stable_tokens=None, noise=0.0):
    cost, tok, cpu, tasks = build(n_tasks=n_tasks, seed=seed,
                                  stable_tokens=stable_tokens)
    est = ESTIMATORS[estimator](cost, tok, cpu, noise_sigma=noise, seed=seed)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=kv, max_batch=batch),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS[scheduler](),
                 estimator=est)
    return eng, eng.run()


# --- each scheduler on a hand-built case with a known answer -------------

def test_fcfs_orders_by_arrival():
    vs = [view(1, enqueued_at=5.0), view(2, enqueued_at=1.0), view(3, enqueued_at=3.0)]
    assert FCFSScheduler().order(vs, now=10.0) == [2, 3, 1]


def test_sjf_orders_by_remaining_task_work():
    vs = [view(1, est_remaining_s=50.0), view(2, est_remaining_s=5.0),
          view(3, est_remaining_s=20.0)]
    assert SJFScheduler().order(vs, now=0.0) == [2, 3, 1]


def test_edf_orders_by_deadline_ignoring_remaining_work():
    # req1 has the earlier deadline but far more work left. EDF cannot see that.
    vs = [view(1, deadline=100.0, est_remaining_s=90.0),
          view(2, deadline=200.0, est_remaining_s=1.0)]
    assert EDFScheduler().order(vs, now=0.0) == [1, 2]


def test_lsf_differs_from_edf_when_remaining_work_differs():
    """
    The case that justifies the mechanism. Same deadlines would make these
    indistinguishable to EDF; here EDF prefers req1 (earlier deadline) while
    LSF prefers req2, which has only 10s of headroom against req1's 60s.
    """
    vs = [view(1, deadline=100.0, est_remaining_s=40.0),   # slack 60
          view(2, deadline=120.0, est_remaining_s=110.0)]  # slack 10
    assert EDFScheduler().order(vs, now=0.0) == [1, 2]
    assert LeastSlackFirstScheduler().order(vs, now=0.0) == [2, 1]


def test_static_priority_orders_by_stage_rank():
    vs = [view(1, stage="parse_issue"), view(2, stage="apply_and_test"),
          view(3, stage="generate_fix")]
    assert StaticPriorityScheduler().order(vs, now=0.0) == [2, 3, 1]


def test_ties_break_deterministically_by_arrival_then_id():
    vs = [view(3, enqueued_at=1.0, est_remaining_s=5.0),
          view(1, enqueued_at=1.0, est_remaining_s=5.0),
          view(2, enqueued_at=0.5, est_remaining_s=5.0)]
    assert SJFScheduler().order(vs, now=0.0) == [2, 1, 3]


@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_every_scheduler_returns_a_permutation(name):
    vs = [view(i, enqueued_at=float(i), deadline=100.0 - i,
               est_remaining_s=float(10 - i)) for i in range(1, 6)]
    order = SCHEDULERS[name]().order(vs, now=0.0)
    assert sorted(order) == [v.req_id for v in sorted(vs, key=lambda v: v.req_id)]


# --- LSF with the oracle estimator on an obvious case --------------------

def test_lsf_picks_the_tight_deadline_task():
    """One task must finish soon, the other has hours. LSF must pick the tight one."""
    tight = view(1, task_id=1, deadline=20.0, est_remaining_s=15.0)   # slack 5
    loose = view(2, task_id=2, deadline=9000.0, est_remaining_s=15.0)  # slack ~8985
    assert LeastSlackFirstScheduler().order([loose, tight], now=0.0)[0] == 1
    # And arrival order must not rescue the loose one.
    tight2 = view(1, task_id=1, deadline=20.0, est_remaining_s=15.0, enqueued_at=99.0)
    loose2 = view(2, task_id=2, deadline=9000.0, est_remaining_s=15.0, enqueued_at=0.0)
    assert LeastSlackFirstScheduler().order([loose2, tight2], now=0.0)[0] == 1


# --- estimator -----------------------------------------------------------

def test_observe_changes_the_estimate():
    cost, tok, cpu, _ = build()
    est = ExecutionEstimator(cost, tok, cpu)
    before = est.predict_output("generate_fix", 1)
    for _ in range(5):
        est.observe("generate_fix", 1, 9999)
    after = est.predict_output("generate_fix", 1)
    assert after != before
    assert after == pytest.approx(9999)


def test_estimate_is_keyed_by_attempt_index():
    """Attempt 2 prompts carry a diagnosis, so its outputs are a different
    distribution. Learning about attempt 1 must not move attempt 2."""
    cost, tok, cpu, _ = build()
    est = ExecutionEstimator(cost, tok, cpu)
    for _ in range(10):
        est.observe("generate_fix", 1, 100)
    assert est.predict_output("generate_fix", 1) == pytest.approx(100)
    # attempt 2 has no direct observations, so it falls back to the stage-level
    # mean rather than the untouched prior.
    assert est.predict_output("generate_fix", 2) == pytest.approx(100)
    for _ in range(10):
        est.observe("generate_fix", 2, 500)
    assert est.predict_output("generate_fix", 1) == pytest.approx(100)
    assert est.predict_output("generate_fix", 2) == pytest.approx(500)


def test_online_estimator_ignores_true_output():
    """
    predict_output accepts true_output so the engine can call one signature for
    both estimators. The online one must not use it -- if it did, every
    scheduler would be silently cheating.
    """
    cost, tok, cpu, _ = build()
    est = ExecutionEstimator(cost, tok, cpu)
    a = est.predict_output("generate_fix", 1, true_output=10)
    b = est.predict_output("generate_fix", 1, true_output=100000)
    assert a == b


def test_oracle_estimator_returns_true_output():
    cost, tok, cpu, _ = build()
    est = OracleEstimator(cost, tok, cpu)
    assert est.predict_output("generate_fix", 1, true_output=1234) == 1234
    assert est.is_oracle is True


def test_engine_actually_calls_observe():
    """A learning estimator that is never taught is just the prior."""
    cost, tok, cpu, tasks = build(n_tasks=6)
    est = ExecutionEstimator(cost, tok, cpu)
    seen = []
    real = est.observe
    est.observe = lambda s, a, o: (seen.append((s, a, o)), real(s, a, o))[1]

    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=2048.0, max_batch=8),
                 retention=RETENTIONS["lru"](), scheduler=FCFSScheduler(),
                 estimator=est)
    eng.run()
    assert seen, "observe() was never called"
    n_llm = sum(1 for t in tasks for n in t.nodes if n.kind is Kind.LLM)
    assert len(seen) == n_llm


def test_estimator_noise_is_stable_per_request():
    """A per-call draw would make one request's estimate jitter during its wait."""
    cost, tok, cpu, _ = build()
    est = ExecutionEstimator(cost, tok, cpu, noise_sigma=0.5, seed=1)
    a = [est.noise_factor(7) for _ in range(5)]
    assert len(set(a)) == 1
    assert est.noise_factor(8) != est.noise_factor(7)


# --- the oracle guard ----------------------------------------------------

@pytest.mark.parametrize("field", [
    "output_tokens", "critical_path_remaining", "next_reuse_gap_s",
    "next_reuse_node", "node", "expected_seconds",
])
def test_reqview_refuses_forbidden_fields(field):
    with pytest.raises(OracleLeakError):
        getattr(view(1), field)


def test_a_cheating_scheduler_fails_loudly():
    """A scheduler that tries to sort by true output length must blow up."""
    class Cheater:
        name = "cheater"

        def order(self, views, now):
            return [v.req_id for v in sorted(views, key=lambda v: v.output_tokens)]

    with pytest.raises(OracleLeakError):
        Cheater().order([view(1), view(2)], now=0.0)


def test_reqview_still_raises_attributeerror_for_typos():
    with pytest.raises(AttributeError):
        view(1).est_remaning_s          # deliberate typo


# --- engine-level invariants ---------------------------------------------

@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_no_request_is_ever_dropped(name):
    eng, rep = run_engine(scheduler=name, n_tasks=8, stable_tokens=4000)
    total_nodes = sum(len(t.nodes) for t in eng.tasks)
    assert len(eng.done_nodes) == total_nodes
    assert rep["tasks_completed"] == len(eng.tasks)
    assert not eng.waiting and not eng.running
    assert all(r.finished_at is not None for r in eng.requests)
    assert eng.pinned_bytes == 0


@pytest.mark.parametrize("name", sorted(SCHEDULERS))
def test_same_seed_gives_identical_results(name):
    _, a = run_engine(scheduler=name, seed=11)
    _, b = run_engine(scheduler=name, seed=11)
    assert a == b


def test_different_schedulers_actually_differ():
    """Guards against every scheduler silently collapsing to arrival order."""
    _, fcfs = run_engine(scheduler="fcfs", n_tasks=14, kv=1024.0, stable_tokens=4000)
    _, lsf = run_engine(scheduler="lsf", n_tasks=14, kv=1024.0, stable_tokens=4000)
    assert fcfs["scheduler"] == "fcfs" and lsf["scheduler"] == "lsf"
    assert (fcfs["p95_task_latency_s"] != lsf["p95_task_latency_s"]
            or fcfs["makespan_s"] != lsf["makespan_s"])


# --- deadlines and reported metrics --------------------------------------

def test_deadlines_are_identical_across_schedulers():
    """
    Miss rates are only comparable if every policy faced the same deadlines.
    They are set from the estimator's priors before any learning, precisely so
    a faster scheduler cannot teach itself easier deadlines.
    """
    a, _ = run_engine(scheduler="fcfs", seed=5)
    b, _ = run_engine(scheduler="lsf", seed=5)
    assert a.task_deadline == b.task_deadline


def test_report_carries_the_phase3_metrics():
    _, rep = run_engine(scheduler="lsf", n_tasks=8)
    for key in ["p50_task_latency_s", "p95_task_latency_s", "p99_task_latency_s",
                "deadline_miss_rate", "mean_slack_at_completion_s",
                "throughput_tasks_per_s", "estimator_mape_pct",
                "estimator_mape_by_stage", "scheduler", "retention", "estimator",
                "slack_factor"]:
        assert key in rep, f"missing {key}"
    assert 0.0 <= rep["deadline_miss_rate"] <= 1.0
    assert rep["estimator_mape_by_stage"]


def test_oracle_estimator_has_zero_output_length_error():
    """Sanity check on the reference: perfect prediction means zero MAPE."""
    _, rep = run_engine(scheduler="lsf", estimator="oracle", n_tasks=8)
    assert rep["estimator"] == "oracle"
    assert rep["estimator_mape_pct"] == pytest.approx(0.0, abs=1e-9)


def test_report_exposes_scheduling_headroom():
    """
    A flat scheduler comparison is ambiguous: it can mean the policies are
    equivalent, or that the queue was empty and no policy ever chose anything.
    These fields disambiguate it, so they must always be present.
    """
    _, low = run_engine(scheduler="fcfs", n_tasks=20, stable_tokens=4000)
    for key in ["mean_queue_depth", "max_queue_depth", "frac_decisions_with_choice"]:
        assert key in low
    assert 0.0 <= low["frac_decisions_with_choice"] <= 1.0


def test_heavier_load_gives_the_scheduler_more_choice():
    _, light = run_engine(scheduler="fcfs", n_tasks=20, stable_tokens=4000)
    cost, tok, cpu, tasks = build(n_tasks=20, seed=3, stable_tokens=4000, rate=0.30)
    est = ExecutionEstimator(cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=2048.0, max_batch=8),
                 retention=RETENTIONS["lru"](), scheduler=FCFSScheduler(),
                 estimator=est)
    heavy = eng.run()
    assert heavy["frac_decisions_with_choice"] > light["frac_decisions_with_choice"]
