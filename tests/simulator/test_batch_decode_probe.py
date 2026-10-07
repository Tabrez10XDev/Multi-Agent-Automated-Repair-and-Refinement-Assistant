"""
The batch_decode_coeff probe's arithmetic, tested without a model server.

The probe itself calls Ollama and is run by the owner. What can be wrong
offline is the fit and the guard against a queueing server, and both would
produce a plausible-looking wrong coefficient, so both are pinned here.
"""

import pytest

from probes.batch_decode_probe import fit_coeff, overlap, per_token_seconds


def _times(c, ns=(1, 2, 4, 8), t1=0.035):
    return {n: t1 * (1 + (n - 1) * c) for n in ns}


@pytest.mark.parametrize("c", [0.0, 0.05, 0.15, 0.5, 1.0])
def test_fit_recovers_the_coefficient_the_simulator_assumes(c):
    fit = fit_coeff(_times(c))
    assert fit["batch_decode_coeff"] == pytest.approx(c)
    assert fit["max_abs_residual"] == pytest.approx(0.0, abs=1e-12)
    assert fit["single_stream_decode_rate"] == pytest.approx(1 / 0.035)


def test_nonlinearity_shows_up_in_the_per_n_values():
    """A saturating cost curve must not be averaged into one confident number."""
    t = {1: 1.0, 2: 1.2, 4: 1.5, 8: 1.8}
    implied = fit_coeff(t)["implied_coeff_by_n"]
    assert implied[2] > implied[4] > implied[8]
    assert fit_coeff(t)["max_abs_residual"] > 0.05


def test_fit_needs_a_baseline_and_a_batch():
    with pytest.raises(ValueError, match="n=1"):
        fit_coeff({2: 1.0, 4: 1.2})
    with pytest.raises(ValueError, match="n > 1"):
        fit_coeff({1: 1.0})


def test_overlap_tells_parallel_from_queued():
    # Four 10s requests: run together they finish in about 10s, queued in 40s.
    assert overlap([10, 10, 10, 10], 10.0) == pytest.approx(4.0)
    assert overlap([10, 20, 30, 40], 40.0) == pytest.approx(2.5)


def test_per_token_seconds_uses_ollama_counters():
    assert per_token_seconds({"eval_count": 200, "eval_duration": 7_000_000_000}) \
        == pytest.approx(0.035)
    with pytest.raises(ValueError):
        per_token_seconds({"eval_count": 0, "eval_duration": 0})
