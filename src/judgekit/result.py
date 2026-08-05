"""The unit every probe returns."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeResult:
    """One measured quantity with its uncertainty and its innocent value.

    ``null_value`` is what an unbiased judge would show (0.5 for position
    preference, 0.0 for a bias correlation, 1.0 for a spread ratio).  A probe
    with ``null_value=None`` is descriptive only and never triggers.
    """

    name: str
    value: float
    ci: tuple[float, float]
    null_value: float | None
    n: int
    detail: str

    @property
    def triggered(self) -> bool | None:
        """True when the 95% interval excludes the innocent value."""
        if self.null_value is None:
            return None
        low, high = self.ci
        if math.isnan(low) or math.isnan(high) or math.isnan(self.value):
            return None
        return not low <= self.null_value <= high
