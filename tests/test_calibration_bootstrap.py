import math

import numpy as np
import pytest

from judgekit.bootstrap import bootstrap_ci, wilson_interval
from judgekit.calibration import brier, ece_equal_mass, murphy_decomposition, reliability_curve


def test_ece_hand_computed():
    # Two equal-mass bins: [0.2, 0.4] vs outcomes [0, 1] and [0.6, 0.8] vs [0, 1].
    # Both bins miss by |0.5 - mean confidence| = 0.2.
    p = [0.2, 0.4, 0.6, 0.8]
    y = [0, 1, 0, 1]
    assert ece_equal_mass(p, y, n_bins=2) == pytest.approx(0.2)


def test_ece_zero_when_frequencies_match_exactly():
    p = [0.2] * 5 + [0.8] * 5
    y = [1, 0, 0, 0, 0, 1, 1, 1, 1, 0]
    assert ece_equal_mass(p, y, n_bins=2) == pytest.approx(0.0, abs=1e-15)


def test_reliability_curve_counts_cover_everything():
    rng = np.random.default_rng(0)
    p = rng.random(103)
    y = (rng.random(103) < p).astype(float)
    curve = reliability_curve(p, y, n_bins=10)
    assert sum(count for _, _, count in curve) == 103


def test_brier_hand_computed():
    assert brier([1.0, 0.0], [1, 0]) == pytest.approx(0.0)
    assert brier([0.5], [1]) == pytest.approx(0.25)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_murphy_decomposition_identity(seed):
    rng = np.random.default_rng(seed)
    p = rng.choice([0.1, 0.3, 0.5, 0.7, 0.9], size=200)
    y = (rng.random(200) < p).astype(float)
    reliability, resolution, uncertainty = murphy_decomposition(p, y)
    assert brier(p, y) == pytest.approx(reliability - resolution + uncertainty, abs=1e-12)


def test_murphy_perfect_forecaster():
    # Forecasts equal to outcomes: no reliability penalty, resolution equals uncertainty.
    p = [1.0, 1.0, 0.0, 0.0]
    y = [1, 1, 0, 0]
    reliability, resolution, uncertainty = murphy_decomposition(p, y)
    assert reliability == pytest.approx(0.0)
    assert resolution == pytest.approx(uncertainty)


@pytest.mark.parametrize(("successes", "n"), [(0, 10), (5, 10), (10, 10), (37, 120), (1, 3)])
def test_wilson_matches_statsmodels(successes, n):
    from statsmodels.stats.proportion import proportion_confint

    low, high = wilson_interval(successes, n)
    ref_low, ref_high = proportion_confint(successes, n, alpha=0.05, method="wilson")
    assert low == pytest.approx(float(ref_low), abs=1e-10)
    assert high == pytest.approx(float(ref_high), abs=1e-10)


def test_bootstrap_is_deterministic_and_covers_mean():
    rng = np.random.default_rng(7)
    units = rng.normal(loc=2.0, scale=1.0, size=300).tolist()

    def mean(sample: list[float]) -> float:
        return float(np.mean(sample))

    first = bootstrap_ci(mean, units, seed=11)
    second = bootstrap_ci(mean, units, seed=11)
    assert first == second
    low, high = first
    # The percentile interval brackets the sample statistic; the population
    # mean is only covered with 95% probability, which is not a test.
    assert low < float(np.mean(units)) < high
    assert high - low < 4.0 / math.sqrt(len(units))


def test_bootstrap_refuses_mostly_undefined_statistic():
    def usually_nan(sample: list[float]) -> float:
        return float("nan") if sum(sample) % 2 < 1.8 else 1.0

    low, high = bootstrap_ci(usually_nan, [1.0, 2.0, 3.0], n_resamples=200, seed=0)
    assert math.isnan(low) and math.isnan(high)


def test_bootstrap_resamples_at_item_level():
    # Each unit is a group of rows; the statistic sees whole groups only.
    units = [[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]]

    def check_groups(sample: list[list[float]]) -> float:
        assert all(group in units for group in sample)
        return float(np.mean([v for group in sample for v in group]))

    low, high = bootstrap_ci(check_groups, units, n_resamples=50, seed=0)
    assert 1.0 <= low <= high <= 3.0
