"""
A probe run on another machine must not overwrite the Mac's calibration.

Both probes used to hardcode backend "mac-metal" and the Mac's output paths,
so running them on the GPU box would have replaced the numbers every finding
rests on and labelled CUDA measurements as Metal.
"""

import os

import pytest

from probes.calibrate import MAC_BACKEND, REPO, calibration_path
from simulator.workload import _MEASURED_PATH

CAL = os.path.join(REPO, "data", "calibration")


def test_the_mac_keeps_the_paths_workload_loads():
    assert calibration_path(MAC_BACKEND, "measured.json") == _MEASURED_PATH
    assert calibration_path(MAC_BACKEND, "batch_decode.json") == \
        os.path.join(CAL, "batch_decode.json")


@pytest.mark.parametrize("fn", ["measured.json", "batch_decode.json"])
def test_another_backend_gets_its_own_directory(fn):
    path = calibration_path("cuda-5060ti", fn)
    assert path == os.path.join(CAL, "cuda-5060ti", fn)
    assert path != calibration_path(MAC_BACKEND, fn)


@pytest.mark.parametrize("bad", ["", "../mac", "a/b", ".hidden"])
def test_backend_names_cannot_escape_the_calibration_dir(bad):
    with pytest.raises(ValueError):
        calibration_path(bad, "measured.json")


def test_the_demo_task_is_where_the_probe_reads_it():
    from probes.calibrate import DEMO_DIR
    for fn in ("target.py", "test_target.py"):
        assert os.path.exists(os.path.join(DEMO_DIR, fn))
