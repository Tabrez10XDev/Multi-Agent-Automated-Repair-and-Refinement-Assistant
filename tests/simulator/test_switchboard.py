"""
`kleinrock-switch`, its Kaplan-Meier signal, and `gts` (DECISIONS 39).
"""

import math
import random

import pytest

from simulator.engine import Engine, EngineConfig
from simulator.policies import ESTIMATORS, RETENTIONS, SCHEDULERS, ReqView
from simulator.switchboard import KMRequestsEstimator, km_cv
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, StageTokens,
                                WorkloadConfig, generate_trace)


def _pop_cv(xs):
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs)) / m


# --- the estimator ----------------------------------------------------------

def test_km_without_censoring_is_the_population_cv():
    assert km_cv([1, 2, 3], []) == pytest.approx(_pop_cv([1, 2, 3]))
    assert km_cv([3] * 11 + [5] * 3 + [10], []) == pytest.approx(
        _pop_cv([3] * 11 + [5] * 3 + [10]))
    assert km_cv([], [(4, True)]) is None


def test_censoring_convention():
    # One completed task at R = 2, one in flight with 2 requests done.
    # Queued: known R >= 3, so it survives j = 1, 2 and S(2) = 1/2.
    # In a tool phase: only R >= 2 is known, so it survives j = 1 alone,
    # is not at risk at j = 2, and S(2) = 0 (the only one at risk failed).
    queued = km_cv([2], [(2, True)])
    tool = km_cv([2], [(2, False)])
    # queued: m = 2, S = [1, 1] -> mean 2, E[R^2] = 1 + 3 = 4, CV 0
    assert queued == pytest.approx(0.0)
    assert tool == pytest.approx(0.0)
    # Add a task in a tool phase at c = 3 (at risk up to j = 2), so m = 3.
    # At j = 2 three are at risk (R=2, queued c=2, tool c=3) and one fails:
    # S(2) = 2/3.
    v = km_cv([2], [(2, True), (3, False)])
    S = [1.0, 1.0, 2.0 / 3.0]
    mean = sum(S)
    m2 = sum((2 * k + 1) * s for k, s in enumerate(S))
    assert v == pytest.approx(math.sqrt(m2 - mean ** 2) / mean)


def test_km_removes_the_short_first_completion_bias():
    """Heavy-tailed R, observed mid-run: short tasks have finished, long ones
    are still running. Completed-only CV is badly biased; KM is close to the
    truth restricted to what has been observed."""
    rng = random.Random(3)
    R = [max(1, int(rng.lognormvariate(1.0, 1.1))) for _ in range(4000)]
    events, cens = [], []
    for r in R:
        age = rng.uniform(0, 60)                 # requests it could have done by now
        if r <= age:
            events.append(r)
        else:
            cens.append((int(age), True))
    m = max(events + [c for c, _ in cens])
    truth = _pop_cv([min(r, m) for r in R])
    naive = _pop_cv(events)
    est = km_cv(events, cens)
    assert abs(est - truth) < 0.1 * truth
    assert abs(naive - truth) > 2 * abs(est - truth)


# --- the switch -------------------------------------------------------------

def _feed(sched, rs):
    """Run tasks one after another through the event protocol."""
    for tid, r in enumerate(rs):
        sched.on_task_event("arrive", tid, float(tid))
        for _ in range(r):
            sched.on_task_event("enqueue", tid, float(tid))
            sched.on_task_event("request_done", tid, float(tid))
        sched.on_task_event("task_done", tid, float(tid))


def test_switch_waits_for_warmup_then_fires_on_heavy_tails():
    s = SCHEDULERS["kleinrock-switch"]()
    heavy = [1, 1, 1, 2, 1, 40] * 10
    _feed(s, heavy[:29])
    assert s.mode == "program-fcfs" and not s.switches
    _feed_more = heavy[29:]
    for tid, r in enumerate(_feed_more, start=29):
        s.on_task_event("arrive", tid, 0.0)
        for _ in range(r):
            s.on_task_event("enqueue", tid, 0.0)
            s.on_task_event("request_done", tid, 0.0)
        s.on_task_event("task_done", tid, 0.0)
    assert s.mode == "fcfs" and len(s.switches) == 1
    assert s.switches[0]["arrived"] >= 30 and s.switches[0]["cv"] >= 1.10


def test_hysteresis_holds_between_the_thresholds():
    # R with CV about 0.98: neither 0.90 nor 1.10 is crossed, from either arm.
    rs = [1, 1, 1, 1, 2, 2, 3, 5, 8, 12] * 6
    for start in ("fcfs", "program-fcfs"):
        s = SCHEDULERS["kleinrock-switch"]()
        s.mode = start
        _feed(s, rs)
        assert 0.90 < s.km.cv() < 1.10
        assert s.mode == start and not s.switches


def test_switch_never_fires_on_maara_holdout_tasks():
    from simulator.trace_loader import load_trace, resolve_trace_set
    from simulator.workload import Kind
    rs = []
    for path in resolve_trace_set("holdout_qwen"):
        lt = load_trace(path, arrival_rate=0.01, seed=1)
        rs += [sum(1 for n in t.nodes if n.kind is not Kind.CPU) for t in lt.tasks]
    for seed in range(20):
        s = SCHEDULERS["kleinrock-switch"]()
        order = rs[:]
        random.Random(seed).shuffle(order)
        _feed(s, order)
        assert not s.switches


# --- engine: golden equality and the oracle guard ---------------------------

def _run(name, seed=5, n=10):
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=n, arrival_rate=0.08, seed=seed), cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=2048.0, max_batch=4),
                 retention=RETENTIONS["lru"](), scheduler=SCHEDULERS[name](),
                 estimator=ESTIMATORS["online"](cost, tok, cpu))
    return eng.run()


def _strip(rep):
    return {k: v for k, v in rep.items() if k not in ("scheduler_log", "scheduler")}


@pytest.mark.parametrize("name", ["kleinrock-switch", "gts"])
def test_equals_program_fcfs_when_it_never_switches(name):
    # 10 tasks never reach the 30-task warm-up; gts on the MAARA prior has
    # Q = infinity. Either way the run must equal program-fcfs field for field.
    a, b = _run(name), _run("program-fcfs")
    assert _strip(a) == _strip(b)
    if name == "kleinrock-switch":
        assert a["scheduler_log"] == {"switches": [], "final_mode": "program-fcfs"}


def test_existing_policies_report_no_scheduler_log():
    assert _run("fcfs")["scheduler_log"] is None


# --- gts --------------------------------------------------------------------

def _view(req_id, task_id, *, done, arrival, enq):
    return ReqView(req_id=req_id, task_id=task_id, stage="round", model=CODER_MODEL,
                   enqueued_at=enq, prompt_tokens=100, attempt_index=done,
                   replan_index=0, priority=0, deadline=None, est_service_s=1.0,
                   est_remaining_s=1.0, task_arrival_s=arrival,
                   task_attained_s=0.0, task_requests_done=done)


def test_gts_q_is_three_on_tracelab_and_infinite_on_maara():
    from experiments.common import make_estimator
    from simulator.gittins import maara_prior
    cost = CostModel.from_profile("gpu_5060ti_public")
    tok, cpu = StageTokens.from_measured(), CpuStageSeconds.from_measured()
    s = SCHEDULERS["gts"]()
    s.bind(cost, make_estimator("tracelab_dev", cost, tok, cpu))
    assert s.q == 3

    class Est:
        def gittins_prior(self, c):
            return maara_prior(c)
    m = SCHEDULERS["gts"]()
    m.bind(cost, Est())
    assert m.q == math.inf


def test_gts_demotes_a_long_task_to_its_request_time():
    s = SCHEDULERS["gts"]()
    s.q = 3
    vs = [_view(1, 1, done=5, arrival=0.0, enq=50.0),     # old task, past Q
          _view(2, 2, done=0, arrival=40.0, enq=40.0),    # new task
          _view(3, 3, done=2, arrival=10.0, enq=45.0)]    # young: keeps its age
    assert s.order(vs, now=60.0) == [3, 2, 1]


def test_gts_without_a_prior_refuses():
    with pytest.raises(RuntimeError):
        SCHEDULERS["gts"]().bind(None, object())
