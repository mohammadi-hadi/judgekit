"""Agreement between a judge and a reference rater.

Implemented from scratch on plain float arithmetic; the test suite cross-checks
every function against an independent reference (scikit-learn, scipy, and the
krippendorff package).  Nothing here calls into BLAS, so results are identical
across platforms — reductions use numpy's fixed pairwise summation and the
small confusion/coincidence matrices are accumulated in Python loops.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

import numpy as np
from numpy.typing import NDArray

Weights = Literal["linear", "quadratic"] | None
FloatArray = Sequence[float] | NDArray[np.float64]


def _as_pair(
    a: FloatArray, b: FloatArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"length mismatch: {x.shape} vs {y.shape}")
    if x.size == 0:
        raise ValueError("no ratings")
    return x, y


def confusion(
    a: FloatArray, b: FloatArray, labels: FloatArray
) -> NDArray[np.float64]:
    """Count matrix with rows indexed by ``a`` and columns by ``b``."""
    x, y = _as_pair(a, b)
    index = {label: i for i, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for u, v in zip(x.tolist(), y.tolist(), strict=True):
        if u not in index or v not in index:
            raise ValueError(f"rating pair ({u}, {v}) outside labels {list(labels)}")
        matrix[index[u], index[v]] += 1.0
    return matrix


def cohen_kappa(
    a: FloatArray,
    b: FloatArray,
    labels: FloatArray | None = None,
    weights: Weights = None,
) -> float:
    """Cohen's kappa, optionally weighted for ordinal scales.

    Disagreement weights follow scikit-learn: distances are between label
    *positions* in ``labels``, so pass the full scale (for instance
    ``[1, 2, 3, 4, 5]``) rather than letting a category that nobody used
    silently shrink the scale.  ``quadratic`` is the usual choice for graded
    scores.  Returns nan when raters use a single label, where chance-corrected
    agreement is undefined.
    """
    x, y = _as_pair(a, b)
    if labels is None:
        labels = sorted(set(x.tolist()) | set(y.tolist()))
    counts = confusion(x, y, labels)
    n = counts.sum()
    observed = counts / n
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col)

    k = len(labels)
    positions = np.arange(k, dtype=np.float64)
    delta = np.abs(positions[:, None] - positions[None, :])
    if weights is None:
        w = (delta > 0).astype(np.float64)
    elif weights == "linear":
        w = delta
    elif weights == "quadratic":
        w = delta**2
    else:
        raise ValueError(f"unknown weights {weights!r}")

    expected_disagreement = float((w * expected).sum())
    if expected_disagreement == 0.0:
        return float("nan")
    return 1.0 - float((w * observed).sum()) / expected_disagreement


def krippendorff_alpha(
    values_by_unit: Sequence[FloatArray],
    level: Literal["nominal", "interval"] = "nominal",
) -> float:
    """Krippendorff's alpha over units rated by any number of raters.

    ``values_by_unit`` holds, per unit, the ratings that were actually given;
    missing ratings are simply absent, which is the point of alpha.  Units with
    fewer than two ratings carry no agreement information and are skipped.
    """

    def distance(u: float, v: float) -> float:
        if level == "interval":
            return (u - v) ** 2
        if level == "nominal":
            return 0.0 if u == v else 1.0
        raise ValueError(f"unknown level {level!r}")

    units = [[float(v) for v in unit] for unit in values_by_unit if len(unit) >= 2]
    if not units:
        raise ValueError("no unit has two or more ratings")
    n_pairable = float(sum(len(unit) for unit in units))

    observed = 0.0
    for unit in units:
        m = len(unit)
        pair_sum = 0.0
        for i in range(m):
            for j in range(m):
                if i != j:
                    pair_sum += distance(unit[i], unit[j])
        observed += pair_sum / (m - 1)
    observed /= n_pairable

    pooled = Counter(value for unit in units for value in unit)
    expected = 0.0
    for u, n_u in pooled.items():
        for v, n_v in pooled.items():
            if u != v:
                expected += n_u * n_v * distance(u, v)
    expected /= n_pairable * (n_pairable - 1.0)

    if expected == 0.0:
        return float("nan")
    return 1.0 - observed / expected


def rank_average(values: FloatArray) -> NDArray[np.float64]:
    """Ranks starting at 1, ties sharing the average rank."""
    x = np.asarray(values, dtype=np.float64)
    order = np.argsort(x, kind="stable")
    ranks = np.empty(x.size, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < x.size:
        j = i
        while j + 1 < x.size and sorted_x[j + 1] == sorted_x[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return ranks


def pearson(a: FloatArray, b: FloatArray) -> float:
    x, y = _as_pair(a, b)
    xc = x - x.mean()
    yc = y - y.mean()
    denominator = float(np.sqrt((xc**2).sum() * (yc**2).sum()))
    if denominator == 0.0:
        return float("nan")
    return float((xc * yc).sum()) / denominator


def spearman(a: FloatArray, b: FloatArray) -> float:
    """Spearman rank correlation with average ranks for ties."""
    return pearson(rank_average(a), rank_average(b))


def exact_agreement(a: FloatArray, b: FloatArray) -> float:
    x, y = _as_pair(a, b)
    return float((x == y).mean())


def within_one(a: FloatArray, b: FloatArray) -> float:
    """Share of ratings at most one scale point apart, for integer scales."""
    x, y = _as_pair(a, b)
    return float((np.abs(x - y) <= 1.0 + 1e-12).mean())
