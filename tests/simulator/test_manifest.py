"""
Run-manifest tests: a calibrated value cannot be used at a configuration it
was not calibrated under.

No LLM is called and no engine is run.
"""

import pytest

from experiments.manifest import (Calibration, CalibrationMismatch, Manifest,
                                  bound_args, check_consistent)

BASE = {"trace_set": "holdout_qwen", "scale": 1, "kv_mb": 1024.0,
        "max_batch": 8, "cache_mode": "block", "util": 0.8}


def rate_cal(**under_overrides):
    return Calibration(name="arrival_rate", value=0.0343,
                       under={**BASE, **under_overrides},
                       target="GPU busy fraction 0.8 under FCFS/keep-all",
                       achieved=0.8, also_check=("util",))


def manifest_with(cal):
    m = Manifest.start("test", params=dict(BASE))
    m.record(cal)
    return m


def test_findings23_rate_calibrated_at_batch16_run_at_batch8_raises():
    # The exact retracted configuration: rate calibrated at batch 16, table
    # run at batch 8, ~2.2x overload with nothing in the output to say so.
    m = manifest_with(rate_cal(max_batch=16))
    with pytest.raises(CalibrationMismatch) as e:
        m.assert_consistent({**BASE, "max_batch": 8})
    msg = str(e.value)
    assert "max_batch=16" in msg and "max_batch=8" in msg
    assert "arrival_rate=0.0343" in msg
    assert "Recalibrate arrival_rate at max_batch=8" in msg
    assert m.verified_runs == []


def test_same_config_passes_and_is_recorded():
    m = manifest_with(rate_cal())
    m.assert_consistent(dict(BASE))
    assert m.verified_runs == [{"config": BASE, "against": ["arrival_rate"]}]


@pytest.mark.parametrize("key,other", [("trace_set", "original"),
                                       ("kv_mb", 512.0),
                                       ("cache_mode", "ollama"),
                                       ("scale", 8),
                                       ("util", 0.6)])
def test_differing_sensitive_key_raises(key, other):
    m = manifest_with(rate_cal())
    with pytest.raises(CalibrationMismatch, match=key):
        m.assert_consistent({**BASE, key: other})


def test_util_is_only_checked_where_declared():
    # slack_factor is calibrated at a given rate, not a given util; a util
    # difference alone must not trip it.
    cal = Calibration("slack_factor", 2.0, under=dict(BASE),
                      target="FCFS misses 20-30%", achieved=0.25)
    m = manifest_with(cal)
    m.assert_consistent({**BASE, "util": 0.6})


def test_key_missing_from_run_config_raises():
    m = manifest_with(rate_cal())
    run = dict(BASE)
    del run["cache_mode"]
    with pytest.raises(CalibrationMismatch, match="missing from the run config"):
        m.assert_consistent(run)


def test_recording_without_a_sensitive_key_raises():
    under = dict(BASE)
    del under["max_batch"]
    with pytest.raises(CalibrationMismatch, match="max_batch"):
        Manifest.start("test", {}).record(
            Calibration("arrival_rate", 0.03, under, "x", 0.8))


def test_assert_with_nothing_recorded_raises():
    with pytest.raises(CalibrationMismatch, match="no calibrations"):
        Manifest.start("test", {}).assert_consistent(dict(BASE))


def test_mismatch_is_not_a_runtimeerror():
    # Both drivers swallow RuntimeError from engine runs and keep going.
    assert not issubclass(CalibrationMismatch, RuntimeError)


def test_precheck_before_calibrating():
    with pytest.raises(CalibrationMismatch, match="not yet calibrated"):
        check_consistent("arrival_rate", {**BASE, "max_batch": 16},
                         dict(BASE))


def test_bound_args_sees_unpassed_defaults():
    # The §23 rate came from a default nobody passed.
    def calibrate(util, kv_mb, batch=16):
        return 0.0
    assert bound_args(calibrate, 0.8, kv_mb=1024)["batch"] == 16


def test_round_trips_through_json():
    m = manifest_with(rate_cal())
    m.assert_consistent(dict(BASE))
    back = Manifest.from_json(m.to_json())
    assert back == m
    assert isinstance(back.calibrations[0], Calibration)
    assert back.calibrations[0].also_check == ("util",)
    back.assert_consistent(dict(BASE))   # still functional after the trip


def test_git_hash_and_dirty_flag_populated():
    m = Manifest.start("test", {})
    assert isinstance(m.git_hash, str) and len(m.git_hash) == 40
    assert isinstance(m.git_dirty, bool)
    assert m.git_dirty == bool(m.git_dirty_paths)
