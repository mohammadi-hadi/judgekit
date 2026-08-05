"""Consistency: does the judge agree with itself?

Re-judge the same items a few times (``sample_index`` 0..k-1) and this probe
measures how often the verdicts for an item all agree.  Pairwise re-judgments
are grouped per presentation order, so order effects — measured separately by
``bias.swap_flip_rate`` — do not leak into the sampling-noise number.

The headline is the unanimity rate because it is always defined and reads
directly; ``agreement.krippendorff_alpha`` is available when a
chance-corrected coefficient is wanted instead.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from judgekit.bootstrap import bootstrap_ci
from judgekit.result import ProbeResult
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict

Group = list[object]
Unit = list[Group]


def _units_from(verdicts: Sequence[GradedVerdict | BinaryVerdict | PairwiseVerdict]) -> list[Unit]:
    """Group verdict values into re-judgment sets, clustered per item."""
    per_item: dict[str, dict[object, Group]] = {}
    for verdict in verdicts:
        if isinstance(verdict, GradedVerdict):
            group_key: object = ()
            value: object = verdict.score
        elif isinstance(verdict, BinaryVerdict):
            group_key = ()
            value = verdict.label
        else:
            group_key = verdict.swapped
            value = verdict.choice
        per_item.setdefault(verdict.item_id, {}).setdefault(group_key, []).append(value)
    units: list[Unit] = []
    for groups in per_item.values():
        multi = [group for group in groups.values() if len(group) >= 2]
        if multi:
            units.append(multi)
    return units


def _unanimity(sample: list[Unit]) -> float:
    groups = [group for unit in sample for group in unit]
    if not groups:
        return float("nan")
    return float(np.mean([1.0 if len(set(map(str, group))) == 1 else 0.0 for group in groups]))


def resample_consistency(
    verdicts: Sequence[GradedVerdict | BinaryVerdict | PairwiseVerdict], *, seed: int = 0
) -> ProbeResult | None:
    """Share of re-judged items where every re-judgment agreed exactly."""
    kinds = {verdict.kind for verdict in verdicts}
    if len(kinds) > 1:
        raise ValueError(f"pass one verdict kind at a time, got {sorted(kinds)}")
    units = _units_from(verdicts)
    if not units:
        return None

    value = _unanimity(units)
    ci = bootstrap_ci(_unanimity, units, seed=seed)
    n_groups = sum(len(unit) for unit in units)

    spread = ""
    if kinds == {"graded"}:
        sds = [
            float(np.std([float(v) for v in group], ddof=1))  # type: ignore[arg-type]
            for unit in units
            for group in unit
        ]
        spread = f"; mean within-item score spread {float(np.mean(sds)):.2f}"

    return ProbeResult(
        name="re-judgment unanimity",
        value=value,
        ci=ci,
        innocent=None,
        n=n_groups,
        detail=(
            f"all re-judgments of an item agreed in {value:.1%} of {n_groups} "
            f"re-judged sets{spread}"
        ),
    )
