"""
KV accounting invariants under eviction pressure.

A prefix match stops at the first missing block, but a LATER block of the same
chain can survive eviction. Admission used to re-create those surviving blocks,
overwriting the live entry without releasing its bytes, so _entry_bytes and the
evictable counters drifted upward for the rest of the run: phantom KV use that
forced extra evictions (DECISIONS 37 amendment). These tests pin the running
counters to a recount of the live entries at every rent tick.
"""

import pytest

from experiments.common import run_retention_one
from simulator.engine import Engine
from simulator.workload import CostModel, CpuStageSeconds, StageTokens


@pytest.fixture
def recount(monkeypatch):
    seen = {"ticks": 0, "bad": []}
    orig = Engine._accrue_rent

    def checked(self, dt, blocked, pressure=None, evictable=None):
        live = [e for e in self.entries.values() if e.evictable]
        seen["ticks"] += 1
        state = (self._evictable_n, self._evictable_bytes, self._entry_bytes)
        truth = (len(live), sum(e.nbytes for e in live),
                 sum(e.nbytes for e in self.entries.values()))
        if state != truth and len(seen["bad"]) < 3:
            seen["bad"].append((self.now, state, truth))
        return orig(self, dt, blocked, pressure, evictable)

    monkeypatch.setattr(Engine, "_accrue_rent", checked)
    return seen


@pytest.mark.parametrize("policy", ["ski-rental", "lru", "keep-all"])
def test_counters_match_live_entries_under_pressure(recount, policy):
    # holdout_qwen at 512 MB, batch 4: the retention operating point of
    # FINDINGS 24, where ski-rental re-created 165 surviving blocks per run.
    cost = CostModel.from_profile("mac_ollama_measured")
    eng, rep = run_retention_one(policy, "fcfs", 1, 1, 512.0, 0.05, cost,
                                 StageTokens.from_measured(),
                                 CpuStageSeconds.from_measured(),
                                 batch=4, trace_set="holdout_qwen")
    assert recount["ticks"] > 1000
    assert recount["bad"] == []
    assert rep["evictions"] > 0          # the pressure the bug needed was present


def test_no_entry_is_held_by_a_request_that_did_not_pin_it():
    cost = CostModel.from_profile("mac_ollama_measured")
    eng, _ = run_retention_one("ski-rental", "fcfs", 1, 1, 512.0, 0.05, cost,
                               StageTokens.from_measured(),
                               CpuStageSeconds.from_measured(),
                               batch=4, trace_set="holdout_qwen")
    # Every request has retired, so nothing may still be pinned.
    assert all(e.refcount == 0 for e in eng.entries.values())
