"""
The `final` traces are the sealed test set (DECISIONS 26). No policy driver
may replay them until the policies are frozen in a commit, so the guard must
hold on every path a driver uses to resolve a trace set by name.
"""

import os

import pytest

from experiments.run_retention_grid import build
from simulator.trace_loader import (SEALED_TRACE_SETS, TRACE_SETS, SealedTraceSet,
                                    resolve_trace_set, trace_set)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_sealed_sets_are_not_reachable_as_ordinary_sets():
    assert not set(SEALED_TRACE_SETS) & set(TRACE_SETS)


@pytest.mark.parametrize("name", sorted(SEALED_TRACE_SETS))
def test_sealed_sets_refuse_without_a_freeze(name, monkeypatch):
    monkeypatch.delenv("MAARA_POLICIES_FROZEN", raising=False)
    with pytest.raises(SealedTraceSet, match="DECISIONS 26"):
        resolve_trace_set(name)
    with pytest.raises(SealedTraceSet):
        trace_set(name)
    with pytest.raises(SealedTraceSet):
        build(1, 1, 0.03, None, None, None, trace_set=name)


@pytest.mark.parametrize("name", sorted(SEALED_TRACE_SETS))
def test_sealed_sets_resolve_after_a_freeze(name, monkeypatch):
    monkeypatch.setenv("MAARA_POLICIES_FROZEN", "abc1234")
    assert resolve_trace_set(name) == SEALED_TRACE_SETS[name]


@pytest.mark.parametrize("name", sorted(SEALED_TRACE_SETS))
def test_sealed_trace_files_are_committed(name):
    for p in SEALED_TRACE_SETS[name]:
        assert os.path.exists(os.path.join(REPO, p)), p
        assert os.path.exists(os.path.join(REPO, p.replace(".jsonl.gz", ".prompts.jsonl.gz")))
