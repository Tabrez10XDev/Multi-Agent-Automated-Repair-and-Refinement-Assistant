"""
Phase 4 tests: derived rent.

rent_rate(E) = E.bytes x (marginal victim's recompute seconds per byte)
               x smoothed eviction rate

Accrued rent is in recompute-seconds, so it compares directly against an
entry's own recompute cost -- the ski-rental decision in consistent units.
No LLM is called.
"""

import math

import pytest

from simulator.engine import Engine, EngineConfig, KVEntry
from simulator.policies import ExecutionEstimator, FCFSScheduler, LRURetention
from simulator.workload import (CODER_MODEL, CostModel, CpuStageSeconds, StageTokens,
                          WorkloadConfig, generate_trace)


def build_engine(kv_mb=4096.0, batch=8, n_tasks=10, rate=0.05,
                 stable_tokens=4000, rent_model="derived", tau=30.0, seed=3,
                 rent_impl="sweep", homogeneous=False):
    """
    Defaults to the SWEEP implementation.

    The per-entry rules in docs/DECISIONS.md section 13 -- rent proportional to bytes,
    marginal victim excluding E -- are the specification, and the sweep is its
    reference implementation. The lazy clock is an optimisation that must agree
    with it, which the equivalence tests check on a real trace. Asserting the
    rules against the lazy path directly would be testing the optimisation's
    shortcut, not the rule.
    """
    cost = CostModel(homogeneous=homogeneous)
    tok = StageTokens.from_measured()
    tok.stable_prefix_tokens = stable_tokens
    cpu = CpuStageSeconds.from_measured()
    tasks = generate_trace(
        WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed), cost, tok, cpu)
    est = ExecutionEstimator(cost, tok, cpu)
    cfg = EngineConfig(kv_budget_mb=kv_mb, max_batch=batch,
                       rent_model=rent_model, ewma_tau_s=tau)
    eng = Engine(tasks, cost, cfg, retention=LRURetention(),
                 scheduler=FCFSScheduler(), estimator=est)
    eng._rent_impl = rent_impl
    return eng


def entry(key, tokens, engine, last_used=0.0, recompute_s=None):
    nbytes = engine.cost.kv_bytes(CODER_MODEL, tokens)
    return KVEntry(
        key=key, model=CODER_MODEL, tokens=tokens, nbytes=nbytes,
        last_used=last_used, created=0.0, refcount=0,
        useful_tokens=tokens,
        recompute_s=(engine.cost.recompute_seconds(CODER_MODEL, tokens)
                     if recompute_s is None else recompute_s))


# --- rent is zero without pressure ---------------------------------------

def test_no_rent_without_memory_pressure():
    """
    Rule 1 of docs/DECISIONS.md section 13. With a budget nothing can fill, no admission
    ever needs an eviction, so holding an entry displaces nothing.
    """
    eng = build_engine(kv_mb=1_000_000.0, n_tasks=8)
    rep = eng.run()
    assert rep["total_rent_s"] == 0.0
    assert rep["evictions"] == 0


def test_no_rent_when_pressure_flag_is_false():
    eng = build_engine()
    e1, e2 = entry("a", 1000, eng), entry("b", 1000, eng)
    eng.entries = {"a": e1, "b": e2}
    eng._evict_rate = 1.0
    eng._accrue_derived_rent(dt=10.0, pressure=False, evictable=[e1, e2])
    assert e1.rent_accrued == 0.0 and e2.rent_accrued == 0.0


def test_no_rent_with_nothing_else_evictable():
    """
    Rule 3: with no other evictable entry there is no marginal victim, so
    holding E displaces nothing and rent stays zero.
    """
    eng = build_engine()
    only = entry("solo", 1000, eng)
    eng.entries = {"solo": only}
    eng._evict_rate = 1.0
    eng._accrue_derived_rent(dt=10.0, pressure=True, evictable=[only])
    assert only.rent_accrued == 0.0


def test_rent_is_zero_when_eviction_rate_is_zero():
    """No evictions happening means no pressure being expressed, so no damage."""
    eng = build_engine()
    e1, e2 = entry("a", 1000, eng), entry("b", 2000, eng)
    eng.entries = {"a": e1, "b": e2}
    eng._evict_rate = 0.0
    eng._accrue_derived_rent(dt=100.0, pressure=True, evictable=[e1, e2])
    assert e1.rent_accrued == 0.0 and e2.rent_accrued == 0.0


# --- rent scales with the holder's footprint ------------------------------

def test_rent_is_proportional_to_entry_bytes():
    """
    The correction raised before Phase 4: a 2,048-token entry must accrue rent
    32x faster than a 64-token one, because it displaces 32x the memory.
    """
    eng = build_engine()
    small, big = entry("small", 64, eng), entry("big", 2048, eng)
    victim = entry("victim", 512, eng, last_used=-1.0)   # LRU, so evicted next
    eng.entries = {e.key: e for e in (small, big, victim)}
    eng._evict_rate = 0.5
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[small, big, victim])

    assert big.rent_accrued > small.rent_accrued
    assert big.rent_accrued / small.rent_accrued == pytest.approx(2048 / 64, rel=1e-6)


def test_break_even_is_independent_of_entry_size():
    """
    Both rent and benefit are linear in tokens, so entry size cancels: a
    64-token and a 2,048-token entry break even at the same moment.

    The absolute constant depends on the evictable pool, because rent is
    normalised by it so the charges sum to the damage actually done. Only the
    size-independence is asserted here; the worked example in
    analysis/analyze_trace.py pins the constants for a stated pool.
    """
    eng = build_engine()
    rate = 0.1
    t_stars = []
    for tokens in (64, 512, 2048):
        e = entry("e", tokens, eng)
        victim = entry("v", tokens, eng, last_used=-1.0)
        eng.entries = {"e": e, "v": victim}
        eng._evict_rate = rate
        eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[e, victim])
        t_stars.append(e.recompute_s / e.rent_accrued)
    assert max(t_stars) == pytest.approx(min(t_stars), rel=1e-6)


def test_holding_more_than_you_reuse_shrinks_break_even():
    """
    Correction C. An entry that stores 4 reusable tokens in a 16-token block
    pays rent on all 16 but only earns back 4, so break-even shrinks by 4/16.
    """
    eng = build_engine()
    held = entry("head", 16, eng)
    held.useful_tokens = 4
    held.recompute_s = eng.cost.recompute_seconds(CODER_MODEL, 4)
    victim = entry("v", 16, eng, last_used=-1.0)
    eng.entries = {"head": held, "v": victim}
    eng._evict_rate = 0.1
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[held, victim])

    # Same pool, same rate: only useful/held differs, so break-even must
    # shrink by exactly that ratio against a full-value entry of equal size.
    full = entry("full", 16, eng)
    v2 = entry("v2", 16, eng, last_used=-1.0)
    eng.entries = {"full": full, "v2": v2}
    eng._evict_rate = 0.1
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[full, v2])

    shrunk = held.recompute_s / held.rent_accrued
    baseline = full.recompute_s / full.rent_accrued
    assert shrunk == pytest.approx(baseline * (4 / 16), rel=1e-6)


# --- the marginal victim excludes E --------------------------------------

def test_marginal_victim_excludes_the_entry_itself():
    """
    Rule 2. If E were charged at its own rate, break-even would be exactly one
    time unit for every entry -- degenerate, and dependent on the unit chosen.
    Here the LRU-next entry is cheap per byte and the other is expensive, so
    each must be charged using the OTHER.
    """
    eng = build_engine()
    cheap = entry("cheap", 1000, eng, last_used=0.0, recompute_s=0.0)  # worthless
    rich = entry("rich", 1000, eng, last_used=5.0)                     # real value
    eng.entries = {"cheap": cheap, "rich": rich}
    eng._evict_rate = 1.0
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[cheap, rich])

    # `cheap` is LRU-next, so `rich` is charged against `cheap` -> zero rate.
    assert rich.rent_accrued == 0.0
    # `cheap` cannot be its own victim, so it is charged against `rich`.
    assert cheap.rent_accrued > 0.0


def test_victim_is_the_lru_entry():
    eng = build_engine()
    a = entry("a", 100, eng, last_used=10.0)
    b = entry("b", 100, eng, last_used=1.0)     # LRU
    c = entry("c", 100, eng, last_used=20.0)
    b.recompute_s = 0.0                          # makes the victim identifiable
    eng.entries = {e.key: e for e in (a, b, c)}
    eng._evict_rate = 1.0
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[a, b, c])
    # a and c are both charged against b, whose rate is zero.
    assert a.rent_accrued == 0.0 and c.rent_accrued == 0.0
    assert b.rent_accrued > 0.0


def test_entries_evicted_mid_iteration_are_not_charged():
    """A snapshot taken before admission can name entries that admission then
    evicted. Charging them would be charging a corpse."""
    eng = build_engine()
    gone, alive, victim = (entry("gone", 100, eng), entry("alive", 100, eng),
                           entry("v", 100, eng, last_used=-1.0))
    eng.entries = {"alive": alive, "v": victim}       # `gone` already evicted
    eng._evict_rate = 1.0
    eng._accrue_derived_rent(dt=1.0, pressure=True, evictable=[gone, alive, victim])
    assert gone.rent_accrued == 0.0
    assert alive.rent_accrued > 0.0


# --- EWMA ----------------------------------------------------------------

def test_ewma_decays_at_the_expected_rate():
    eng = build_engine(tau=30.0)
    eng._evict_rate = 1.0
    eng._decay_evict_rate(30.0)
    assert eng._evict_rate == pytest.approx(math.exp(-1.0), rel=1e-9)
    eng._decay_evict_rate(30.0)
    assert eng._evict_rate == pytest.approx(math.exp(-2.0), rel=1e-9)


def test_ewma_settles_at_the_true_rate_after_a_step_change():
    """
    A steady e evictions/second must settle at e. Feeding 2/s for many time
    constants should converge there.
    """
    eng = build_engine(tau=10.0)
    true_rate, dt, pending = 2.0, 0.01, 0.0
    for _ in range(int(80 / dt)):                 # 8 time constants
        pending += true_rate * dt
        while pending >= 1.0:                     # emit whole evictions
            eng._evict_rate += 1.0 / eng.cfg.ewma_tau_s
            pending -= 1.0
        eng._decay_evict_rate(dt)
    assert eng._evict_rate == pytest.approx(true_rate, rel=0.05)


def test_ewma_tracks_a_drop_to_zero():
    """After evictions stop, the estimate must fall away rather than latch."""
    eng = build_engine(tau=10.0)
    for _ in range(20):
        eng._evict_rate += 1.0 / eng.cfg.ewma_tau_s
    before = eng._evict_rate
    eng._decay_evict_rate(50.0)                   # 5 tau of silence
    assert eng._evict_rate < before * 0.01


def test_ewma_is_slower_with_a_longer_tau():
    fast, slow = build_engine(tau=5.0), build_engine(tau=50.0)
    fast._evict_rate = slow._evict_rate = 1.0
    fast._decay_evict_rate(10.0)
    slow._decay_evict_rate(10.0)
    assert fast._evict_rate < slow._evict_rate


# --- the Phase 3 eviction fix still holds --------------------------------

def test_eviction_frees_enough_not_half_enough():
    """
    Phase 3 regression. The loop used to compare an accumulating `freed`
    counter against `need - free_bytes`, but free_bytes already grows as
    entries are evicted, so the same bytes counted twice and it stopped at
    about half. Symptom was a deadlock with plenty evictable.
    """
    for kv in (1024.0, 2048.0, 4096.0):
        eng = build_engine(kv_mb=kv, n_tasks=12, stable_tokens=4000)
        rep = eng.run()          # would raise RuntimeError("deadlock") if broken
        assert rep["tasks_completed"] == len(eng.tasks)
        assert eng.pinned_bytes == 0


def test_engine_never_exceeds_the_budget_under_derived_rent():
    eng = build_engine(kv_mb=2048.0, n_tasks=15, stable_tokens=4000)
    eng.run()
    assert eng.m["peak_kv_bytes"] <= eng.cfg.kv_budget_bytes


# --- model selection ------------------------------------------------------

def test_blocking_rent_is_still_selectable():
    eng = build_engine(rent_model="blocking", kv_mb=1024.0, n_tasks=12)
    rep = eng.run()
    assert rep["rent_model"] == "blocking"
    assert rep["tasks_completed"] == len(eng.tasks)


def test_derived_rent_is_denser_than_blocking_rent():
    """
    The point of the rework. Under LRU the engine rarely blocks -- it just
    evicts -- so blocking-only rent is too sparse to drive a policy.
    """
    a = build_engine(rent_model="blocking", kv_mb=4096.0, n_tasks=15, stable_tokens=4000)
    b = build_engine(rent_model="derived", kv_mb=4096.0, n_tasks=15, stable_tokens=4000)
    ra, rb = a.run(), b.run()
    assert ra["total_rent_s"] == 0.0 or rb["total_rent_s"] > ra["total_rent_s"] * 10


def test_every_prompt_block_is_cached_not_only_the_shared_head():
    """
    vLLM caches every prompt block. Under block-hash keying the trailing blocks
    are ordinary entries in the chain rather than a special "tail" object --
    they simply never get matched by anyone, which is exactly the situation
    retention has to detect without being told.
    """
    eng = build_engine(kv_mb=1_000_000.0, n_tasks=8, stable_tokens=4000)
    eng.run()
    assert len(eng.seen_keys) > 100
    blocks_per_call = [len(r.node.block_keys) for r in eng.requests]
    assert min(blocks_per_call) > 1          # prompts span many blocks
    assert eng.m["cold_misses"] > 0


# --- Phase 5b step 2: lazy rent accrual ----------------------------------

def _run_with(impl, kv_mb=1024.0, n_tasks=8, seed=3, stable_tokens=2000,
              homogeneous=False):
    """
    Equivalence must hold on the MIXED-model trace, which is the real case:
    evaluate_plan runs llama3.2 while everything else runs qwen, and the two
    differ in both block size and rebuild cost.

    It does hold, because rent per byte depends on the VICTIM's rebuild cost,
    not the holder's -- one number shared by every entry whatever model it
    belongs to. An earlier lazy version approximated the victim by "the model
    with the most evictable blocks", which is what made the two disagree; it
    now uses the true LRU entry, with the LRU entry itself charged against the
    second-LRU since it cannot be its own victim.
    """
    eng = build_engine(kv_mb=kv_mb, n_tasks=n_tasks, seed=seed,
                       stable_tokens=stable_tokens, rent_impl=impl,
                       homogeneous=homogeneous)
    rep = eng.run()
    return eng, rep


def test_the_equivalence_trace_really_has_two_models():
    """Guards the test above from silently degrading into a single-model case."""
    eng, _ = _run_with("lazy")
    models = {n.model for t in eng.tasks for n in t.nodes if n.model}
    assert len(models) >= 2, models


def test_lazy_is_the_production_default():
    from simulator.workload import WorkloadConfig, generate_trace
    cost, tok, cpu = CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()
    tasks = generate_trace(WorkloadConfig(n_tasks=2, seed=1), cost, tok, cpu)
    eng = Engine(tasks, cost, EngineConfig(), retention=LRURetention(),
                 scheduler=FCFSScheduler())
    assert eng._rent_impl == "lazy"


@pytest.mark.parametrize("homogeneous", [False, True])
def test_lazy_and_sweep_agree_on_metrics_mixed_and_single_model(homogeneous):
    lazy_eng, lazy = _run_with("lazy", homogeneous=homogeneous)
    sweep_eng, sweep = _run_with("sweep", homogeneous=homogeneous)
    for key in ("recompute_seconds", "evictions", "tasks_completed",
                "prefix_hit_rate", "makespan_s", "total_rent_s"):
        assert lazy[key] == pytest.approx(sweep[key], rel=1e-9), key


def test_lazy_and_sweep_agree_on_metrics():
    """
    Equivalence. The lazy clock advances one counter per iteration instead of
    touching every entry; it must produce the same run, not merely a similar
    one. Every block is the same size and rebuild cost, so the marginal victim
    is a per-model constant and the two formulations coincide exactly.
    """
    lazy_eng, lazy = _run_with("lazy")
    sweep_eng, sweep = _run_with("sweep")

    for key in ("recompute_seconds", "evictions", "tasks_completed",
                "prefix_hit_rate", "makespan_s", "p50_task_latency_s",
                "p95_task_latency_s", "voluntary_evictions"):
        assert lazy[key] == pytest.approx(sweep[key], rel=1e-9), key
    assert lazy["total_rent_s"] == pytest.approx(sweep["total_rent_s"], rel=1e-6)


def test_lazy_and_sweep_agree_on_eviction_order():
    lazy_eng, _ = _run_with("lazy")
    sweep_eng, _ = _run_with("sweep")
    assert lazy_eng.m["cold_misses"] == sweep_eng.m["cold_misses"]
    assert lazy_eng.m["recompute_misses"] == sweep_eng.m["recompute_misses"]
    assert sorted(lazy_eng.entries) == sorted(sweep_eng.entries)


def test_lazy_and_sweep_agree_on_per_entry_rent():
    lazy_eng, _ = _run_with("lazy")
    sweep_eng, _ = _run_with("sweep")
    shared = set(lazy_eng.entries) & set(sweep_eng.entries)
    assert shared
    for k in shared:
        assert lazy_eng.entries[k].rent_accrued == pytest.approx(
            sweep_eng.entries[k].rent_accrued, rel=1e-6, abs=1e-9), k


def test_rent_accrued_setter_round_trips():
    """Tests and policies assign rent_accrued directly; banking must make the
    value read back unchanged."""
    eng = build_engine()
    e = entry("e", 1000, eng)
    e.clock = eng._clock
    eng._clock.cum = 5.0
    e.rent_accrued = 42.0
    assert e.rent_accrued == pytest.approx(42.0)
    eng._clock.cum = 9.0
    assert e.rent_accrued > 42.0          # accrues from the new baseline


def test_pinned_entries_do_not_accrue():
    """A pinned entry cannot be displaced, so it causes no damage."""
    eng = build_engine()
    e = entry("e", 1000, eng)
    e.clock = eng._clock
    e.refcount = 1
    eng._clock.cum = 100.0
    assert e.rent_accrued == 0.0


def test_used_bytes_is_maintained_incrementally():
    """
    Regression on the hot loop. used_bytes summed every entry and free_bytes
    called it several times per iteration; the profile showed 1.29e9 generator
    calls and 115s of a 167s run. It must now agree with the explicit sum.
    """
    eng, _ = _run_with("lazy", kv_mb=1024.0, n_tasks=6)
    explicit = eng.pinned_bytes + sum(e.nbytes for e in eng.entries.values())
    assert eng.used_bytes == explicit
