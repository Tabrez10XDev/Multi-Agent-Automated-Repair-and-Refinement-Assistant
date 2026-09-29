"""
Guards the MEASURED vs PLACEHOLDER discipline.

Every constant the simulator relies on must declare where it came from. The
failure this prevents is quiet and expensive: a placeholder that nobody
remembers is a placeholder gets quoted in the writeup as though it were
measured. These tests fail loudly if a constant loses its label.
"""

import json
import os

from sim.workload import (CODER_MODEL, SENTINEL_MODEL, CostModel, StageTokens,
                          _MEASURED_PATH)


def test_provenance_labels_every_model_rate():
    text = CostModel().provenance()
    for model in (CODER_MODEL, SENTINEL_MODEL):
        line = next((l for l in text.splitlines() if model in l), None)
        assert line is not None, f"{model} missing from provenance()"
        assert "[MEASURED]" in line or "[PLACEHOLDER]" in line, \
            f"{model} rate has no provenance label: {line!r}"


def test_batch_decode_coeff_is_declared_placeholder():
    # Load-bearing and never measured; it decides how much throughput a blocked
    # request forgoes. If someone measures it, this test should be updated in
    # the same commit.
    text = CostModel().provenance()
    line = next((l for l in text.splitlines() if "batch_decode_coeff" in l), None)
    assert line is not None, "batch_decode_coeff missing from provenance()"
    assert "PLACEHOLDER" in line


def test_measured_json_actually_overrides_defaults():
    """The MEASURED path must work, not just be reachable in principle."""
    if not os.path.exists(_MEASURED_PATH):
        return  # nothing measured yet; defaults are correctly PLACEHOLDER
    with open(_MEASURED_PATH) as f:
        measured = json.load(f)
    if not measured.get("rates"):
        return

    cm = CostModel()
    rates = cm.rates_for(CODER_MODEL)
    assert rates.measured is True, "measured.json present but rates not marked MEASURED"
    expected = measured["rates"][CODER_MODEL]["prefill_rate"]
    assert rates.prefill_rate == expected, \
        f"prefill_rate {rates.prefill_rate} does not match measured.json {expected}"


def test_stage_tokens_records_which_fields_were_measured():
    tok = StageTokens.from_measured()
    if os.path.exists(_MEASURED_PATH):
        with open(_MEASURED_PATH) as f:
            if json.load(f).get("prompt_sizes"):
                assert "stable_prefix_tokens" in tok.measured_fields
                return
    assert tok.measured_fields == ()
