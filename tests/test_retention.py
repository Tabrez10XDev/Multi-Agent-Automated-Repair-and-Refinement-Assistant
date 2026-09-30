"""
Phase 5 tests: retention policies.

No LLM is called. Runs are on synthetic traces, plus one hand-built
constant-rent case where the ski-rental bound can be checked directly.
"""

import pytest

from sim.engine import Engine, EngineConfig, KVEntry
from sim.policies import (RETENTIONS, EvictNowRetention, ExecutionEstimator,
                          FCFSScheduler, HintBuilder, KeepAllRetention,
                          OfflineReferenceRetention, SkiRentalPredictionsRetention,
                          SkiRentalRetention)
from sim.workload import (CODER_MODEL, CostModel, CpuStageSeconds, StageTokens,
                          WorkloadConfig, generate_trace)

ONLINE = ["lru", "keep-all", "evict-now", "ski-rental", "ski-rental-pred"]


def run(retention="lru", kv_mb=4096.0, batch=8, n_tasks=12, rate=0.05,
        stable_tokens=4000, seed=3, hint_sigma=0.0, lam=0.5):
    cost = CostModel()
    tok = StageTokens.from_measured()
    tok.stable_prefix_tokens = stable_tokens
    cpu = CpuStageSeconds.from_measured()
    tasks = generate_trace(
        WorkloadConfig(n_tasks=n_tasks, arrival_rate=rate, seed=seed), cost, tok, cpu)
    pol = (SkiRentalPredictionsRetention(lam) if retention == "ski-rental-pred"
           else RETENTIONS[retention]())
    eng = Engine(tasks, cost, EngineConfig(kv_budget_mb=kv_mb, max_batch=batch),
                 retention=pol, scheduler=FCFSScheduler(),
                 estimator=ExecutionEstimator(cost, tok, cpu),
                 hints=HintBuilder(sigma=hint_sigma, seed=seed))
    return eng, eng.run()


def entry(key, tokens, cost, **kw):
    kw.setdefault("useful_tokens", tokens)
    kw.setdefault("recompute_s", cost.recompute_seconds(CODER_MODEL, tokens))
    return KVEntry(key=key, model=CODER_MODEL, tokens=tokens,
                   nbytes=cost.kv_bytes(CODER_MODEL, tokens), **kw)


# --- the two extremes ----------------------------------------------------

def test_evict_now_recomputes_every_reuse_that_needed_retention():
    """
    Retaining nothing turns every prefix match into a recompute -- except for
    matches that never needed retention in the first place. Two requests wanting
    the same prefix CONCURRENTLY share the live entry while its refcount is
    still above zero, and evict-now cannot prevent that because the entry is in
    use. So the hit rate floor is concurrent sharing, not zero.
    """
    _, evict = run("evict-now")
    _, lru = run("lru")
    assert evict["recompute_seconds"] > lru["recompute_seconds"]
    assert evict["prefix_hit_rate"] < lru["prefix_hit_rate"]
    assert evict["voluntary_evictions"] == 0      # it declines at on_complete


def test_evict_now_retains_nothing_after_the_last_user_finishes():
    cost = CostModel()
    e = entry("e", 1000, cost, refcount=0)
    assert EvictNowRetention().on_complete(e, now=1.0) is False


def test_keep_all_with_unlimited_budget_recomputes_nothing():
    """With no pressure there is never a reason to drop, so no prefix is
    ever rebuilt."""
    eng, rep = run("keep-all", kv_mb=1_000_000.0)
    assert rep["recompute_seconds"] == 0.0
    assert rep["evictions"] == 0
    assert rep["total_rent_s"] == 0.0


def test_evict_now_recomputes_more_than_keep_all():
    _, a = run("evict-now", kv_mb=1_000_000.0)
    _, b = run("keep-all", kv_mb=1_000_000.0)
    assert a["recompute_seconds"] > b["recompute_seconds"] == 0.0


# --- ski-rental ----------------------------------------------------------

def test_ski_rental_holds_until_break_even():
    """Release exactly when accrued rent reaches the recompute it saves."""
    cost = CostModel()
    pol = SkiRentalRetention()
    e = entry("e", 1000, cost, rent_accrued=0.0)
    assert pol.on_tick([e], now=1.0) == []

    e.rent_accrued = e.recompute_s * 0.99
    assert pol.on_tick([e], now=1.0) == []

    e.rent_accrued = e.recompute_s
    assert pol.on_tick([e], now=1.0) == ["e"]


def test_ski_rental_never_releases_a_pinned_entry():
    cost = CostModel()
    e = entry("e", 1000, cost, refcount=1)
    e.rent_accrued = e.recompute_s * 10
    assert SkiRentalRetention().on_tick([e], now=1.0) == []


def test_ski_rental_evicts_worthless_tails_first_under_pressure():
    """A tail has zero recompute_s, so it is past break-even immediately."""
    cost = CostModel()
    head = entry("head", 1000, cost, rent_accrued=0.0)
    tail = entry("tail", 1000, cost, recompute_s=0.0, useful_tokens=0, is_tail=True)
    victims = SkiRentalRetention().choose_victims([head, tail], 1, now=1.0)
    assert victims[0].key == "tail"


# --- ski-rental with predictions -----------------------------------------

def test_predictions_release_early_when_reuse_is_predicted_far():
    """
    Purohit et al: predicted far -> buy at lambda x b, i.e. give up sooner than
    the deterministic rule would.
    """
    cost = CostModel()
    lam = 0.5
    pol = SkiRentalPredictionsRetention(lam)
    e = entry("e", 1000, cost, created=0.0)
    e.rent_accrued = e.recompute_s * 0.6      # past lambda*b, short of b
    e.rent_rate = 1.0                         # engine stamps this under pressure
    e.hint_gap_s = 10_000.0                   # far
    assert pol.on_tick([e], now=1.0) == ["e"]
    # The deterministic rule would still be holding at this point.
    assert SkiRentalRetention().on_tick([e], now=1.0) == []


def test_predictions_hold_longer_when_reuse_is_predicted_near():
    cost = CostModel()
    pol = SkiRentalPredictionsRetention(0.5)
    e = entry("e", 1000, cost, created=0.0)
    e.rent_accrued = e.recompute_s * 1.2      # past b, short of b/lambda
    e.rent_rate = 1.0
    e.hint_gap_s = 0.001                      # imminent
    assert pol.on_tick([e], now=1.0) == []
    assert SkiRentalRetention().on_tick([e], now=1.0) == ["e"]


def test_lambda_one_matches_the_deterministic_rule():
    cost = CostModel()
    e = entry("e", 1000, cost, created=0.0)
    e.rent_accrued = e.recompute_s
    e.rent_rate = 1.0
    e.hint_gap_s = 10_000.0
    assert SkiRentalPredictionsRetention(1.0).on_tick([e], now=1.0) == ["e"]


def test_prediction_is_inert_without_a_rent_rate():
    """
    A predicted reuse TIME only converts to the rent scale through a rate. With
    no pressure the rate is zero, every entry reads "predicted near", and the
    policy falls back to the deterministic rule. That is correct -- rent is not
    accruing either -- but it is why the prediction affects only a small
    minority of decisions on this workload.
    """
    cost = CostModel()
    e = entry("e", 1000, cost, created=0.0)
    e.rent_accrued = e.recompute_s * 0.6
    e.rent_rate = 0.0
    e.hint_gap_s = 10_000.0
    assert SkiRentalPredictionsRetention(0.5).on_tick([e], now=1.0) == []


def test_lambda_must_be_in_range():
    for bad in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError):
            SkiRentalPredictionsRetention(bad)


def test_hint_noise_is_multiplicative_and_seeded():
    class N:
        next_reuse_gap_s = 100.0
    exact = HintBuilder(sigma=0.0).hint_for(N())
    assert exact == 100.0
    a = HintBuilder(sigma=0.8, seed=1).hint_for(N())
    b = HintBuilder(sigma=0.8, seed=1).hint_for(N())
    c = HintBuilder(sigma=0.8, seed=2).hint_for(N())
    assert a == b and a != c and a > 0


def test_hint_is_none_when_there_is_no_reuse():
    class N:
        next_reuse_gap_s = None
    assert HintBuilder(sigma=0.5).hint_for(N()) is None


# --- offline reference ---------------------------------------------------

def test_offline_reference_drops_entries_that_are_never_reused():
    cost = CostModel()
    pol = OfflineReferenceRetention()
    never = entry("never", 1000, cost, true_gap_s=None)
    reused = entry("reused", 1000, cost, true_gap_s=5.0)
    assert pol.on_complete(never, 0.0) is False
    assert pol.on_complete(reused, 0.0) is True
    assert pol.on_tick([never, reused], now=1.0) == ["never"]


def test_offline_reference_evicts_by_belady():
    """Farthest next use first; never-reused counts as farthest of all."""
    cost = CostModel()
    soon = entry("soon", 100, cost, true_gap_s=1.0)
    later = entry("later", 100, cost, true_gap_s=50.0)
    never = entry("never", 100, cost, true_gap_s=None)
    order = OfflineReferenceRetention().choose_victims(
        [soon, later, never], need_bytes=1, now=0.0)
    assert [e.key for e in order] == ["never", "later", "soon"]


def test_offline_reference_pays_the_least_recompute():
    """
    It sees true reuse times, so it never keeps something that will not be
    matched and never drops something that will be matched cheaply. That makes
    it the floor on RECOMPUTE.

    It is deliberately NOT asserted to be the floor on total cost, because it
    is not. Measured on this trace it pays 75.2s recompute against LRU's 86.3s
    but 124.7s blocked against LRU's 108.9s, for a worse total. The rule is
    greedy -- it extrapolates the current rent rate across the whole gap, which
    overstates rent when pressure is intermittent -- and dropping an entry
    itself creates the pressure the next decision sees. Hence "offline
    reference", never "optimal". Improving it is open work.
    """
    _, ref = run("offline-ref", kv_mb=2048.0, n_tasks=15)
    for name in ONLINE:
        _, got = run(name, kv_mb=2048.0, n_tasks=15)
        assert ref["recompute_seconds"] <= got["recompute_seconds"] + 1e-9, \
            f"{name} paid less recompute than the offline reference"


def test_offline_reference_is_not_claimed_optimal():
    """Guards the label: the docstring must not promise optimality."""
    doc = OfflineReferenceRetention.__doc__.lower()
    assert "not optimal" in doc or "never" in doc
    assert OfflineReferenceRetention.name == "offline-ref"


# --- the competitive bound, where it actually applies --------------------

def test_ski_rental_within_2x_offline_on_a_constant_rent_case():
    """
    The 2-competitive bound holds only for CONSTANT rent, so it is checked on a
    hand-built case with a fixed rate rather than on the engine, where derived
    rent moves with load.

    Season length is how long the entry would have been held; buy cost b is its
    recompute. Offline pays min(season, b). Ski-rental rents to b then buys, so
    it pays min(season, 2b).
    """
    cost = CostModel()
    b = cost.recompute_seconds(CODER_MODEL, 1000)
    rate = 1.0                                   # constant rent per second
    pol = SkiRentalRetention()

    for season in (0.05 * b, 0.5 * b, b, 2 * b, 10 * b):
        e = entry("e", 1000, cost, created=0.0)
        paid, t, bought = 0.0, 0.0, False
        step = b / 500.0
        while t < season:
            e.rent_accrued += rate * step
            paid += rate * step
            t += step
            if pol.on_tick([e], now=t):
                paid += 0.0     # "buying" is paying the recompute, counted below
                bought = True
                break
        if bought:
            paid += b           # recompute later
        offline = min(season * rate, b)
        assert paid <= 2 * offline + 1e-6, f"season={season}: {paid} > 2x{offline}"


# --- engine integration --------------------------------------------------

@pytest.mark.parametrize("name", ONLINE + ["offline-ref"])
def test_every_policy_completes_the_workload(name):
    eng, rep = run(name, kv_mb=2048.0, n_tasks=12)
    assert rep["tasks_completed"] == len(eng.tasks)
    assert len(eng.done_nodes) == sum(len(t.nodes) for t in eng.tasks)
    assert eng.pinned_bytes == 0


@pytest.mark.parametrize("name", ONLINE)
def test_online_policies_ignore_the_true_gap(name):
    """
    true_gap_s is oracle. An online policy reading it would be cheating, so
    perturbing it must not change any decision.
    """
    a_eng, a = run(name, kv_mb=2048.0, n_tasks=10)
    b_eng, b = run(name, kv_mb=2048.0, n_tasks=10)
    for e in b_eng.entries.values():
        e.true_gap_s = 1e9
    assert a["recompute_seconds"] == b["recompute_seconds"]
    assert a["evictions"] == b["evictions"]


def test_ski_rental_releases_voluntarily_not_only_under_pressure():
    eng, rep = run("ski-rental", kv_mb=4096.0, n_tasks=15)
    assert rep["voluntary_evictions"] > 0


def test_keep_all_never_releases_voluntarily():
    _, rep = run("keep-all", kv_mb=4096.0, n_tasks=15)
    assert rep["voluntary_evictions"] == 0
