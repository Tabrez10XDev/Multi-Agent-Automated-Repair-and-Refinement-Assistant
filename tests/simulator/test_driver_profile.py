"""
--profile reaches every driver's run manifest (docs/GPU_TODO.md, prep work).

A GPU rerun is meant to be one flag, and a result is only as good as the
record of which costs produced it. Each driver is stopped at Manifest.start,
before any engine run, so no simulation happens here.
"""

import sys

import pytest

from experiments import (run_retention_grid, run_scheduler_table, scheduling_map,
                         sweep_decode_coeff)
from experiments.manifest import Manifest


class _Stop(Exception):
    pass


DRIVERS = [
    pytest.param(scheduling_map, [], id="scheduling_map"),
    pytest.param(run_scheduler_table, [], id="run_scheduler_table"),
    pytest.param(sweep_decode_coeff, [], id="sweep_decode_coeff"),
    pytest.param(run_retention_grid, [], id="run_retention_grid"),
    pytest.param(run_retention_grid, ["--headrooms", "0.5"], id="run_retention_grid-headroom"),
]


def _manifest_params(monkeypatch, module, argv):
    seen = {}

    def start(cls, experiment, params):
        seen.update(params)
        raise _Stop

    monkeypatch.setattr(Manifest, "start", classmethod(start))
    monkeypatch.setattr(sys, "argv", [module.__name__] + argv)
    with pytest.raises(_Stop):
        module.main()
    return seen


@pytest.mark.parametrize("module,extra", DRIVERS)
@pytest.mark.parametrize("profile", ["gpu_5060ti", "mac_ollama_measured"])
def test_profile_reaches_the_manifest(monkeypatch, module, extra, profile):
    params = _manifest_params(monkeypatch, module, ["--profile", profile] + extra)
    assert params["hardware_profile"] == profile
    assert f"hardware profile     {profile}" in params["cost_provenance"]


@pytest.mark.parametrize("module,extra", DRIVERS)
def test_default_profile_is_the_mac(monkeypatch, module, extra):
    assert _manifest_params(monkeypatch, module, extra)["hardware_profile"] \
        == "mac_ollama_measured"


def test_unknown_profile_is_refused(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["scheduling_map", "--profile", "gpu_4060"])
    with pytest.raises(SystemExit):
        scheduling_map.main()


def test_headroom_job_still_takes_the_tuple_without_a_profile(monkeypatch):
    """A run started before --profile existed sends 7-tuples to workers that
    may import this file; they must build the default profile, not crash."""
    built = []
    monkeypatch.setattr(run_retention_grid, "run_one",
                        lambda pol, sched, scale, seed, kv, rate, cost, *a, **k:
                        built.append(cost.profile) or (None, {m: 0 for m in
                                                              run_retention_grid.HEADROOM_METRICS}))
    job = ("lru", 1, 1, 1024.0, 0.05, 4, "holdout_qwen")
    run_retention_grid._headroom_job(job)
    run_retention_grid._headroom_job(job + ("gpu_5060ti",))
    assert built == ["mac_ollama_measured", "gpu_5060ti"]
