"""
The reusable working set that headroom mode sizes KV budgets from
(experiments/run_retention_grid.py, OPEN_ISSUES 27).

The recorder is experimenter-side accounting: off by default, and when on it
must log every time a block is needed, so a block that is evicted and rebuilt
still counts as wanted again.
"""

from experiments.run_retention_grid import reusable_working_set, run_one
from simulator.workload import CostModel, CpuStageSeconds, StageTokens

ARGS = (1, 1, 1024.0, 0.0265)   # scale, seed, kv_mb, arrival_rate


def _ctx():
    return CostModel(), StageTokens.from_measured(), CpuStageSeconds.from_measured()


def test_recorder_is_off_by_default():
    cost, tok, cpu = _ctx()
    eng, _ = run_one("lru", "fcfs", *ARGS, cost, tok, cpu, batch=4)
    assert eng._key_uses == {} and eng._key_nbytes == {}


def test_working_set_is_positive_and_below_everything_ever_needed():
    cost, tok, cpu = _ctx()
    ws, active = reusable_working_set(1, 1, 0.0265, 1024.0, cost, tok, cpu, batch=4)
    eng, _ = run_one("evict-now", "fcfs", *ARGS, cost, tok, cpu, batch=4,
                     record_key_uses=True)
    total = sum(eng._key_nbytes.values())
    assert 0 < ws < total
    assert active > 0
    # Under evict-now every reuse is a rebuild, so some block is needed twice.
    assert any(len(u) > 1 for u in eng._key_uses.values())
