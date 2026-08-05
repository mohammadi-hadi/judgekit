"""Calibration of binary verdict confidences.

A judge that says "pass, 90% sure" is making a frequency claim: of all the
times it says 90%, about nine in ten should actually be passes.  These
functions measure how far the claims are from the frequencies.

Binning is equal-mass (same number of verdicts per bin) rather than
equal-width: judges concentrate confidence near the top of the range, and
equal-width binning leaves most bins nearly empty there, which makes ECE a
lottery over bin edges.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from judgekit.agreement import FloatArray


def _as_probability_outcomes(
    p: FloatArray, y: FloatArray
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    probs = np.asarray(p, dtype=np.float64)
    outcomes = np.asarray(y, dtype=np.float64)
    if probs.shape != outcomes.shape:
        raise ValueError(f"length mismatch: {probs.shape} vs {outcomes.shape}")
    if probs.size == 0:
        raise ValueError("no verdicts")
    if probs.min() < 0.0 or probs.max() > 1.0:
        raise ValueError("probabilities outside [0, 1]")
    if not np.all((outcomes == 0.0) | (outcomes == 1.0)):
        raise ValueError("outcomes must be 0 or 1")
    return probs, outcomes


def reliability_curve(
    p: FloatArray, y: FloatArray, n_bins: int = 10
) -> list[tuple[float, float, int]]:
    """Per equal-mass bin: (mean confidence, observed frequency, count)."""
    probs, outcomes = _as_probability_outcomes(p, y)
    order = np.argsort(probs, kind="stable")
    curve: list[tuple[float, float, int]] = []
    for indices in np.array_split(order, min(n_bins, probs.size)):
        if indices.size == 0:
            continue
        curve.append(
            (
                float(probs[indices].mean()),
                float(outcomes[indices].mean()),
                int(indices.size),
            )
        )
    return curve


def ece_equal_mass(p: FloatArray, y: FloatArray, n_bins: int = 10) -> float:
    """Expected calibration error over equal-mass bins."""
    probs, _ = _as_probability_outcomes(p, y)
    total = probs.size
    return sum(
        (count / total) * abs(frequency - confidence)
        for confidence, frequency, count in reliability_curve(p, y, n_bins)
    )


def brier(p: FloatArray, y: FloatArray) -> float:
    probs, outcomes = _as_probability_outcomes(p, y)
    return float(((probs - outcomes) ** 2).mean())


def murphy_decomposition(p: FloatArray, y: FloatArray) -> tuple[float, float, float]:
    """Brier score split into (reliability, resolution, uncertainty).

    Grouping is by distinct forecast value, where the decomposition is exact:
    ``brier == reliability - resolution + uncertainty`` to machine precision,
    and the test suite asserts exactly that.  Reliability is the calibration
    term (lower is better); resolution rewards forecasts that separate
    outcomes; uncertainty is the base-rate entropy the judge cannot change.
    """
    probs, outcomes = _as_probability_outcomes(p, y)
    n = probs.size
    base_rate = float(outcomes.mean())

    reliability = 0.0
    resolution = 0.0
    for value in np.unique(probs):
        group = outcomes[probs == value]
        frequency = float(group.mean())
        weight = group.size / n
        reliability += weight * (float(value) - frequency) ** 2
        resolution += weight * (frequency - base_rate) ** 2

    uncertainty = base_rate * (1.0 - base_rate)
    return reliability, resolution, uncertainty
