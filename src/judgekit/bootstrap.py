"""Confidence intervals: item-level bootstrap, plus Wilson for proportions.

Verdicts nest inside items — re-judgments, swapped presentations, k samples —
so the exchangeable unit is the item, not the verdict row.  Resampling rows
would treat correlated rows as independent evidence and hand back intervals
that are too narrow.  Every audit therefore groups its rows by item before
calling ``bootstrap_ci``.

Resampling is seeded and uses numpy's PCG64, which draws identically on every
platform, so reported intervals are reproducible to the digit.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from statistics import NormalDist
from typing import TypeVar

import numpy as np

U = TypeVar("U")

# If more than this share of resamples produce an undefined statistic, an
# interval over the survivors would be quietly conditioned on "the statistic
# happened to be defined" — refuse instead.
MAX_UNDEFINED_SHARE = 0.1


def bootstrap_ci(
    statistic: Callable[[list[U]], float],
    units: Sequence[U],
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ``statistic`` over resampled units."""
    if not units:
        raise ValueError("no units to resample")
    rng = np.random.default_rng(seed)
    n = len(units)
    draws: list[float] = []
    for _ in range(n_resamples):
        indices = rng.integers(0, n, size=n)
        value = statistic([units[i] for i in indices])
        if not math.isnan(value):
            draws.append(value)
    if len(draws) < n_resamples * (1.0 - MAX_UNDEFINED_SHARE):
        return (float("nan"), float("nan"))
    tail = (1.0 - level) / 2.0
    lower, upper = np.quantile(np.asarray(draws), [tail, 1.0 - tail])
    return float(lower), float(upper)


def wilson_interval(successes: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} outside [0, {n}]")
    z = NormalDist().inv_cdf(0.5 + level / 2.0)
    p = successes / n
    denominator = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denominator
    half = z * math.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))
