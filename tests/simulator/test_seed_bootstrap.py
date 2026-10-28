"""Seed-cluster bootstrap and the general scorer (DECISIONS 39)."""

import json
import os

import pytest

from experiments.common import seed_bootstrap
from experiments.score_sealed import score

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_identical_policies_give_exactly_one_with_zero_width():
    a = [[3.0, 5.0, 7.0, 11.0], [1.0, 2.0, 4.0, 8.0]]
    out = seed_bootstrap(a, a, active=[True, False], reps=500)
    for s in (out["mean"], out["median"], out["active_mean"], *out["cells"]):
        assert s["value"] == s["lo"] == s["hi"] == 1.0


def test_known_answer_two_cells():
    # a = 2b in cell 0 and a = b/2 in cell 1, seed for seed: every resample
    # gives exactly those ratios, because the same seeds are drawn for a and b.
    b = [[1.0, 2.0, 3.0], [4.0, 6.0, 8.0]]
    a = [[2.0, 4.0, 6.0], [2.0, 3.0, 4.0]]
    out = seed_bootstrap(a, b, active=[True, False], reps=500)
    assert [c["value"] for c in out["cells"]] == [2.0, 0.5]
    assert out["mean"]["value"] == out["mean"]["lo"] == out["mean"]["hi"] == 1.25
    assert out["active_mean"]["value"] == 2.0 and out["n_active"] == 1


def test_resample_is_shared_across_cells():
    # Cell 1 is cell 0 scaled: with one resample shared by both cells, the two
    # cells' ratios move together, so the mean's interval equals each cell's.
    b = [[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]
    a = [[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]]
    out = seed_bootstrap(a, b, reps=2000)
    assert out["mean"]["lo"] == pytest.approx(out["cells"][0]["lo"])
    assert out["mean"]["hi"] == pytest.approx(out["cells"][0]["hi"])


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        seed_bootstrap([[1.0, 2.0]], [[1.0, 2.0, 3.0]])
    with pytest.raises(ValueError):
        seed_bootstrap([[1.0]], [[1.0]])


E4 = os.path.join(REPO, "results", "tracelab", "E4_sealed2.json")


@pytest.mark.skipif(not os.path.exists(E4), reason="E4 map not present")
def test_reproduces_score_e4_counts():
    from experiments import score_e4
    with open(E4) as fh:
        d = json.load(fh)
    old = score_e4.score(d)
    h1 = score(d, "gittins-guard-x2", "program-fcfs", "p95_slowdown", reps=200)
    h2 = score(d, "gittins-guard-x2", "program-fcfs", "p95_task_latency_s", reps=200)
    assert (h1["better"], h1["worse"]) == (old["H1"]["better"], old["H1"]["worse"])
    assert (h2["better"], h2["worse"]) == (old["H2"]["better"], old["H2"]["worse"])
    assert h1["bootstrap"]["mean"]["value"] == pytest.approx(old["D"][0])
    assert h1["bootstrap"]["n_active"] == old["active"]
