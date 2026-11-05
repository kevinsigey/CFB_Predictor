"""Planted-example check for the coverage audit: 10 outcomes, 8 inside the interval."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from quantile_coverage_audit import coverage  # noqa: E402


def test_coverage_counts_closed_interval_on_planted_example():
    actual = [1, 2, 3, 4, 5, 6, 7, 8, 11, 100]  # 8 inside [0, 10], 2 outside
    lo = [0] * 10
    hi = [10] * 10
    assert coverage(actual, lo, hi) == 0.8

    # Boundary values count as inside; a single point outside moves the rate.
    assert coverage([10.0, 10.5], [0, 0], [10, 10]) == 0.5
    assert np.isnan(coverage([], [], []))
