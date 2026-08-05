import math

import numpy as np
import pytest

from judgekit.agreement import (
    cohen_kappa,
    exact_agreement,
    krippendorff_alpha,
    pearson,
    rank_average,
    spearman,
    within_one,
)

SCALE = [1, 2, 3, 4, 5]


def _random_ratings(seed, n=120):
    rng = np.random.default_rng(seed)
    base = rng.integers(1, 6, size=n)
    noisy = np.clip(base + rng.integers(-1, 2, size=n), 1, 5)
    return base.tolist(), noisy.tolist()


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("weights", [None, "linear", "quadratic"])
def test_cohen_kappa_matches_sklearn(seed, weights):
    from sklearn.metrics import cohen_kappa_score

    a, b = _random_ratings(seed)
    ours = cohen_kappa(a, b, labels=SCALE, weights=weights)
    reference = cohen_kappa_score(a, b, labels=SCALE, weights=weights)
    assert ours == pytest.approx(reference, abs=1e-12)


def test_cohen_kappa_with_unused_category_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    rng = np.random.default_rng(3)
    a = rng.choice([1, 3, 5], size=80).tolist()
    b = rng.choice([1, 3, 5], size=80).tolist()
    ours = cohen_kappa(a, b, labels=SCALE, weights="quadratic")
    reference = cohen_kappa_score(a, b, labels=SCALE, weights="quadratic")
    assert ours == pytest.approx(reference, abs=1e-12)


def test_cohen_kappa_degenerate_is_nan():
    assert math.isnan(cohen_kappa([2, 2, 2], [2, 2, 2]))


def _random_reliability(seed, n_units=40, n_raters=4):
    rng = np.random.default_rng(seed)
    matrix = rng.integers(1, 6, size=(n_raters, n_units)).astype(float)
    missing = rng.random(size=matrix.shape) < 0.3
    matrix[missing] = np.nan
    return matrix


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("level", ["nominal", "interval"])
def test_krippendorff_alpha_matches_reference(seed, level):
    import krippendorff

    matrix = _random_reliability(seed)
    units = [
        [value for value in matrix[:, u] if not np.isnan(value)] for u in range(matrix.shape[1])
    ]
    ours = krippendorff_alpha(units, level=level)
    reference = krippendorff.alpha(reliability_data=matrix, level_of_measurement=level)
    assert ours == pytest.approx(float(reference), abs=1e-12)


def test_krippendorff_alpha_needs_pairable_units():
    with pytest.raises(ValueError, match="two or more"):
        krippendorff_alpha([[1.0], [2.0]])


def test_rank_average_handles_ties():
    assert rank_average([10.0, 20.0, 20.0, 30.0]).tolist() == [1.0, 2.5, 2.5, 4.0]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_spearman_matches_scipy(seed):
    from scipy.stats import spearmanr

    rng = np.random.default_rng(seed)
    a = rng.integers(1, 6, size=100).astype(float).tolist()
    b = rng.integers(1, 6, size=100).astype(float).tolist()
    assert spearman(a, b) == pytest.approx(float(spearmanr(a, b).statistic), abs=1e-10)


def test_pearson_degenerate_is_nan():
    assert math.isnan(pearson([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))


def test_exact_and_within_one():
    a = [1, 2, 3, 4, 5]
    b = [1, 3, 3, 2, 5]
    assert exact_agreement(a, b) == pytest.approx(0.6)
    assert within_one(a, b) == pytest.approx(0.8)
