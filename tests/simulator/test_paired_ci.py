"""Paired confidence interval used by the scheduler table. No engine run."""

import math

from experiments.run_scheduler_table import paired_ci


def test_identical_rows_are_not_significant():
    ci = paired_ci([10.0, 12.0, 11.0], [10.0, 12.0, 11.0])
    assert ci["mean_diff_s"] == 0.0 and not ci["significant"]


def test_consistent_shift_is_significant():
    a = [10.0, 12.0, 11.0, 13.0, 9.0]
    ci = paired_ci([x - 2.0 + 0.01 * i for i, x in enumerate(a)], a)
    assert ci["hi_s"] < 0 and ci["significant"]


def test_matches_hand_computed_t_interval():
    # d = [1, 2, 3]: mean 2, sd 1, t(df=2) = 4.303, half = 4.303 / sqrt(3)
    ci = paired_ci([2.0, 4.0, 6.0], [1.0, 2.0, 3.0])
    half = 4.303 / math.sqrt(3)
    assert math.isclose(ci["lo_s"], 2 - half) and math.isclose(ci["hi_s"], 2 + half)


def test_df_between_table_entries_is_conservative():
    # df = 22 is not tabulated; it must use df = 20's larger critical value.
    d = [float(i % 3) for i in range(23)]
    ci = paired_ci(d, [0.0] * 23)
    sd = (sum((x - sum(d) / 23) ** 2 for x in d) / 22) ** 0.5
    assert math.isclose(ci["hi_s"] - ci["mean_diff_s"], 2.086 * sd / math.sqrt(23))
