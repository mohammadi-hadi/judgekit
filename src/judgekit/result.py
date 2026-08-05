"""The unit every probe returns."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ProbeResult:
    """One measured quantity with its uncertainty and its innocent range.

    ``innocent`` is the band a defect-free judge could plausibly occupy —
    a region of practical equivalence, not a point.  A point null would be a
    straw man: with enough data, a spread ratio of 1.04 or a partial
    correlation of 0.03 excludes the point while meaning nothing in practice.
    The band widths are documented probe by probe and are deliberately part of
    the audit's opinion.  A probe with ``innocent=None`` is descriptive only
    and never triggers.
    """

    name: str
    value: float
    ci: tuple[float, float]
    innocent: tuple[float, float] | None
    n: int
    detail: str

    @property
    def triggered(self) -> bool | None:
        """True when the 95% interval lies entirely outside the innocent band."""
        if self.innocent is None:
            return None
        low, high = self.ci
        if math.isnan(low) or math.isnan(high) or math.isnan(self.value):
            return None
        return high < self.innocent[0] or low > self.innocent[1]
