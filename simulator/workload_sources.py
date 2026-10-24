"""
Workload-source registry: one place that turns a trace-set NAME into Tasks.

Every policy driver used to reach traces through a `trace_set` string and a
hard-wired load_trace call. That made MAARA the only workload the drivers could
replay. A public trace (TraceLab, Weka, ...) has different stage names, no
sidecar prompts and its own sealing story, so it needs a seam: the adapter
registers a name here and produces simulator.workload.Task objects; the
drivers, which only ever call resolve_workload(name, ...), do not change.

WHAT AN ENTRY CARRIES
    load                  fn(name, seed, arrival_rate) -> list[Task]. Gets the
                          name so one prefix-registered adapter can serve a
                          family ("tracelab_dev", "tracelab_test").
    sealed                True makes the name a sealed test set: refused unless
                          MAARA_POLICIES_FROZEN is set, the same gate as
                          trace_loader.resolve_trace_set (DECISIONS 26). The
                          gate runs here, before `load` is called, so an
                          adapter cannot forget it.
    estimator_factory     fn(cost, tok, cpu) -> estimator, or None for today's
                          ExecutionEstimator. The online estimator is wired to
                          MAARA stage names, so a source with other stages
                          needs its own.
    supports_deadline_basis   which --deadline-basis values make sense on this
                          source. A source with no isolated-task estimate can
                          only support "task".

MAARA names (TRACE_SETS keys and the sealed names) are served by a built-in
entry that calls exactly the load_trace path the drivers used before.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional

from simulator.trace_loader import (SEALED_TRACE_SETS, TRACE_SET_EXCLUSIONS,
                                    TRACE_SETS, SealedTraceSet, load_trace,
                                    resolve_trace_set)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ALL_DEADLINE_BASES = frozenset({"estimate", "task"})


@dataclass(frozen=True)
class WorkloadSource:
    key: str                       # exact name, or prefix when prefix=True
    load: Callable                 # (name, seed, arrival_rate) -> list[Task]
    prefix: bool = False
    sealed: bool = False
    estimator_factory: Optional[Callable] = None
    supports_deadline_basis: frozenset = ALL_DEADLINE_BASES

    def matches(self, name):
        return name.startswith(self.key) if self.prefix else name == self.key


_SOURCES: list = []


def register_source(key, load, *, prefix=False, sealed=False,
                    estimator_factory=None,
                    supports_deadline_basis=ALL_DEADLINE_BASES):
    """Register a workload source under an exact name (or a name prefix).
    Re-registering a key replaces it, so an adapter module can be reloaded."""
    bases = frozenset(supports_deadline_basis)
    bad = bases - ALL_DEADLINE_BASES
    if bad:
        raise ValueError(f"unknown deadline basis {sorted(bad)}")
    src = WorkloadSource(key, load, prefix, sealed, estimator_factory, bases)
    unregister_source(key)
    _SOURCES.append(src)
    return src


def unregister_source(key):
    _SOURCES[:] = [s for s in _SOURCES if s.key != key]


def _maara_load(name, seed, arrival_rate):
    # Exactly the call build() made before the registry existed. The sealed
    # refusal lives in resolve_trace_set and still fires from here.
    paths = [os.path.join(_REPO, p) for p in resolve_trace_set(name)]
    return load_trace(paths, arrival_rate=arrival_rate, seed=seed,
                      exclude_tasks=TRACE_SET_EXCLUSIONS.get(name, {})).tasks


_MAARA = WorkloadSource("maara", _maara_load)


def known_sources():
    names = set(TRACE_SETS) | set(SEALED_TRACE_SETS)
    for s in _SOURCES:
        names.add(s.key + "*" if s.prefix else s.key)
    return sorted(names)


def get_source(name):
    """The entry serving `name`: exact registrations first, then the longest
    matching prefix, then MAARA's own sets."""
    exact = [s for s in _SOURCES if not s.prefix and s.matches(name)]
    if exact:
        return exact[0]
    pre = [s for s in _SOURCES if s.prefix and s.matches(name)]
    if pre:
        return max(pre, key=lambda s: len(s.key))
    if name in TRACE_SETS or name in SEALED_TRACE_SETS:
        return _MAARA
    raise KeyError(f"unknown workload source {name!r}; known: "
                   f"{', '.join(known_sources())}")


def _require_freeze(name):
    if not os.environ.get("MAARA_POLICIES_FROZEN"):
        raise SealedTraceSet(
            f"{name!r} is a sealed test set (DECISIONS 26). Freeze the "
            f"policies in a commit first, then set MAARA_POLICIES_FROZEN=<that commit>.")


def resolve_workload(name, seed, arrival_rate):
    """Tasks for the named source. The sealed gate runs before any loading."""
    src = get_source(name)
    if src.sealed:
        _require_freeze(name)
    return src.load(name, seed, arrival_rate)


def check_deadline_basis(name, basis):
    """Refuse a --deadline-basis the source cannot support, before any run."""
    src = get_source(name)
    if basis not in src.supports_deadline_basis:
        raise ValueError(
            f"workload source {name!r} does not support deadline_basis "
            f"{basis!r}; it supports {sorted(src.supports_deadline_basis)}")
