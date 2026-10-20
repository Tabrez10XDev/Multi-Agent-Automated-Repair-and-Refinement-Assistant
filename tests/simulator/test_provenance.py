"""
Guards the MEASURED vs PLACEHOLDER discipline.

Every constant the simulator relies on must declare where it came from. The
failure this prevents is quiet and expensive: a placeholder that nobody
remembers is a placeholder gets quoted in the writeup as though it were
measured. These tests fail loudly if a constant loses its label.
"""

import json
import os

from simulator.workload import (CODER_MODEL, SENTINEL_MODEL, CostModel, StageTokens,
                          _MEASURED_PATH)


def test_provenance_labels_every_model_rate():
    """
    Each model line carries TWO provenance tags now: one for the prefill and
    decode rates, one for kv_bytes_per_token. They have different origins --
    the rates were measured against ollama, while qwen's KV figure was
    confirmed against an ollama 0.33.3 prompt-cache log line and llama3.2's is
    still derived from published config.
    """
    text = CostModel().provenance()
    for model in (CODER_MODEL, SENTINEL_MODEL):
        line = next((l for l in text.splitlines() if model in l), None)
        assert line is not None, f"{model} missing from provenance()"
        assert "rates MEASURED" in line or "rates PLACEHOLDER" in line, \
            f"{model} rate has no provenance label: {line!r}"
        assert "kv MEASURED" in line or "kv DERIVED" in line, \
            f"{model} kv_bytes_per_token has no provenance label: {line!r}"


def test_kv_bytes_provenance_is_per_profile():
    """Both KV constants are confirmed against Mac server logs (DECISIONS 30).
    That confirmation is per machine: the GPU profile must still say DERIVED."""
    for profile, label in (("mac_ollama_measured", "kv MEASURED"), ("gpu_5060ti", "kv DERIVED")):
        text = CostModel.from_profile(profile).provenance()
        for model in (CODER_MODEL, SENTINEL_MODEL):
            line = next(l for l in text.splitlines() if model in l)
            assert label in line, (profile, line)


def test_batch_decode_coeff_provenance_follows_the_profile():
    # Measured on the Mac by probes/batch_decode_probe.py (DECISIONS 28); the
    # GPU profile has no measurement yet and must still say PLACEHOLDER.
    for profile, label in (("mac_ollama_measured", "MEASURED"), ("gpu_5060ti", "PLACEHOLDER")):
        text = CostModel.from_profile(profile).provenance()
        line = next((l for l in text.splitlines() if "batch_decode_coeff" in l), None)
        assert line is not None, "batch_decode_coeff missing from provenance()"
        assert f"[{label}" in line, (profile, line)


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


# --------------------------------------------------------------------------
# Hardware profiles. Every finding so far is a Mac result; these tests keep
# that visible and keep the Mac baseline from drifting.
# --------------------------------------------------------------------------

import pytest

from simulator.workload import DEFAULT_PROFILE, PROFILES, get_profile


def test_default_profile_is_the_mac_and_its_measured_fields_say_so():
    cm = CostModel()
    assert DEFAULT_PROFILE == "mac_ollama_measured"
    assert cm.profile == "mac_ollama_measured"
    for model in (CODER_MODEL, SENTINEL_MODEL):
        assert cm.rates_for(model).measured is True
    # Both confirmed against Ollama server logs on this Mac (DECISIONS 30).
    assert cm.hardware_profile.kv_measured == {CODER_MODEL, SENTINEL_MODEL}


def test_gpu_5060ti_exists_and_nothing_in_it_is_measured():
    cm = CostModel.from_profile("gpu_5060ti")
    assert all(r.measured is False for r in cm.rates.values())
    assert cm.hardware_profile.kv_measured == frozenset()
    assert cm.hardware_profile.batch_decode_coeff_measured is False
    text = cm.provenance()
    # The profile's description names both tags, so check the field lines only.
    fields = [l for l in text.splitlines() if "hardware profile" not in l]
    assert "MEASURED" not in "\n".join(fields).replace("PLACEHOLDER", "")
    assert "PLACEHOLDER" in next(l for l in text.splitlines() if "hardware profile" in l)


def test_provenance_names_the_active_profile():
    for name in PROFILES:
        text = CostModel.from_profile(name).provenance()
        line = next(l for l in text.splitlines() if "hardware profile" in l)
        assert name in line


def test_unknown_profile_fails_clearly():
    with pytest.raises(ValueError, match="unknown hardware profile 'rtx9090'"):
        CostModel(profile="rtx9090")
    with pytest.raises(ValueError, match="known profiles"):
        get_profile("rtx9090")


def test_explicit_override_is_reported_not_hidden():
    line = next(l for l in CostModel(batch_decode_coeff=0.3).provenance().splitlines()
                if "batch_decode_coeff" in l)
    assert "OVERRIDDEN from profile value 0.841" in line


def test_mac_profile_values_are_pinned():
    """
    The baseline every finding was measured against. If one of these moves,
    every number in FINDINGS moves with it, so a change must be deliberate:
    update this test in the same commit and say why in DECISIONS.
    """
    cm = CostModel()
    assert cm.batch_decode_coeff == 0.841
    assert cm.hardware_profile.batch_decode_coeff_measured is True
    assert cm.iteration_overhead_s == 0.001
    assert cm.homogeneous is False
    expected = {
        CODER_MODEL: (267.9, 28.09, 57344),
        SENTINEL_MODEL: (541.87, 59.9, 114688),
    }
    for model, (prefill, decode, kv) in expected.items():
        r = cm.rates_for(model)
        assert (r.prefill_rate, r.decode_rate, r.kv_bytes_per_token) == (prefill, decode, kv)


def test_profiles_are_independent_objects():
    """Building the placeholder profile must not mark the Mac's rates unmeasured."""
    a, b = CostModel(), CostModel.from_profile("gpu_5060ti")
    assert a.rates is not b.rates
    assert a.rates[CODER_MODEL].measured is True
    assert b.rates[CODER_MODEL].measured is False


def test_rates_measured_on_another_machine_are_refused(monkeypatch):
    """measured.json from a GPU box must not load under the Mac profile's name."""
    import simulator.workload as w
    monkeypatch.setattr(w, "MEASURED", {"backend": "cuda", "rates": {
        CODER_MODEL: {"prefill_rate": 5000.0, "decode_rate": 60.0}}})
    with pytest.raises(ValueError, match="backend 'cuda'"):
        w.CostModel()
