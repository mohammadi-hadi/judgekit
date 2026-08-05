"""The unit every probe returns."""

from __future__ import annotations

import math
from dataclasses import dataclass


def _finite(value: float) -> float | None:
    return None if math.isnan(value) else value


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

    def to_dict(self) -> dict[str, object]:
        """JSON-safe view of the result; nan becomes null."""
        return {
            "name": self.name,
            "value": _finite(self.value),
            "ci": [_finite(self.ci[0]), _finite(self.ci[1])],
            "innocent": list(self.innocent) if self.innocent is not None else None,
            "n": self.n,
            "triggered": self.triggered,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class SkippedProbe:
    """A probe that could not run, and the data that would enable it.

    A skip is a statement about the log file, not about the judge: the audit
    surfaces it so the next run can be logged richer instead of the check
    silently disappearing.
    """

    name: str
    needs: str
