"""
The gpu_5060ti profile loads what the GPU box's probes write, and nothing else
(docs/GPU_TODO.md, prep item 3).
"""

import json
import os

import pytest

from probes.calibrate import calibration_path
from simulator import workload
from simulator.workload import (CODER_MODEL, GPU_BACKEND, SENTINEL_MODEL, CostModel,
                                gpu_coeff_from, gpu_rates_from)


def write(d, name, data):
    with open(os.path.join(d, name), "w") as f:
        json.dump(data, f)


def test_the_profile_reads_where_the_probes_write():
    assert workload._GPU_CAL_DIR == os.path.dirname(
        calibration_path(GPU_BACKEND, "measured.json"))


def test_an_empty_directory_gives_mac_copies_marked_placeholder(tmp_path):
    rates = gpu_rates_from(str(tmp_path))
    mac = CostModel().rates
    for name, r in rates.items():
        assert not r.measured
        assert r.prefill_rate == mac[name].prefill_rate
    assert gpu_coeff_from(str(tmp_path)) == (0.15, False)


def test_measured_models_are_loaded_and_only_they_are_marked_measured(tmp_path):
    write(tmp_path, "measured.json", {
        "backend": GPU_BACKEND,
        "rates": {CODER_MODEL: {"prefill_rate": 3000.0, "decode_rate": 90.0}}})
    rates = gpu_rates_from(str(tmp_path))
    assert rates[CODER_MODEL].measured
    assert (rates[CODER_MODEL].prefill_rate, rates[CODER_MODEL].decode_rate) == (3000.0, 90.0)
    assert not rates[SENTINEL_MODEL].measured


def test_the_coefficient_is_loaded_from_the_fit(tmp_path):
    write(tmp_path, "batch_decode.json",
          {"backend": GPU_BACKEND, "fit": {"batch_decode_coeff": 0.12}})
    assert gpu_coeff_from(str(tmp_path)) == (0.12, True)


@pytest.mark.parametrize("fn,data", [
    ("measured.json", {"backend": "mac-metal", "rates": {}}),
    ("batch_decode.json", {"backend": "mac-metal", "fit": {"batch_decode_coeff": 0.84}}),
    ("measured.json", {"rates": {}}),
])
def test_a_file_from_another_machine_is_refused(tmp_path, fn, data):
    write(tmp_path, fn, data)
    with pytest.raises(ValueError):
        gpu_rates_from(str(tmp_path)) if fn == "measured.json" else gpu_coeff_from(str(tmp_path))


def test_the_mac_profile_is_untouched():
    mac = CostModel.from_profile("mac_ollama_measured")
    assert mac.batch_decode_coeff == pytest.approx(0.841)
    assert all(r.measured for r in mac.rates.values())


def test_rate_overrides_parse_model_names_with_colons():
    assert workload.parse_rate_overrides("qwen2.5-coder:7b=3021.98/62.78,llama3.2=4001.12/98.25") == {
        CODER_MODEL: (3021.98, 62.78), SENTINEL_MODEL: (4001.12, 98.25)}
    assert workload.parse_rate_overrides(None) is None


def test_overridden_rates_are_used_and_reported_as_overridden():
    cost = CostModel.from_profile("mac_ollama_measured", batch_decode_coeff=0.3,
                                  rate_overrides={CODER_MODEL: (3000.0, 60.0)})
    assert (cost.rates[CODER_MODEL].prefill_rate, cost.rates[CODER_MODEL].decode_rate) == (3000.0, 60.0)
    assert cost.batch_decode_coeff == 0.3
    lines = cost.provenance().splitlines()
    assert "OVERRIDDEN" in next(l for l in lines if CODER_MODEL in l)
    assert "MEASURED" in next(l for l in lines if SENTINEL_MODEL in l)
    # The profile itself is untouched.
    assert CostModel.from_profile("mac_ollama_measured").rates[CODER_MODEL].prefill_rate != 3000.0


def test_an_unknown_model_override_is_refused():
    with pytest.raises(ValueError):
        CostModel.from_profile("mac_ollama_measured", rate_overrides={"gpt-x": (1.0, 1.0)})
