"""Bias probes: the known failure modes of LLM judges, each as one number.

Every probe returns a point estimate with a 95% interval from the item-level
bootstrap, and each interval is judged against the value an unbiased judge
would show.  Verdicts for the same item (re-judgments, swapped presentations)
are resampled together, never independently.

A probe returns ``None`` when the input simply does not contain the data it
needs — no swapped pairs, no candidate lengths, no candidate models.  That is
a statement about the log file, not about the judge.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeVar

import numpy as np

from judgekit.agreement import pearson, rank_average
from judgekit.bootstrap import bootstrap_ci
from judgekit.result import ProbeResult
from judgekit.schema import GradedVerdict, PairwiseVerdict

Row = TypeVar("Row")


def _group_rows(pairs: Sequence[tuple[str, Row]]) -> list[list[Row]]:
    """Group (item_id, row) pairs into per-item units, insertion-ordered."""
    units: dict[str, list[Row]] = {}
    for item_id, row in pairs:
        units.setdefault(item_id, []).append(row)
    return list(units.values())


def _estimate(
    statistic: Callable[[list[list[Row]]], float],
    units: list[list[Row]],
    seed: int,
) -> tuple[float, tuple[float, float]]:
    value = statistic(units)
    ci = bootstrap_ci(statistic, units, seed=seed)
    return value, ci


def _pooled(units: list[list[Row]]) -> list[Row]:
    return [row for unit in units for row in unit]


def position_preference(
    pairwise: Sequence[PairwiseVerdict], *, seed: int = 0
) -> ProbeResult | None:
    """Share of decisive verdicts that went to the candidate presented first.

    An unbiased judge sits at 0.5 when pairs are shown in both orders or in
    random order.  If every pair was shown in one fixed order, 0.5 is only the
    innocent value when candidate quality is unrelated to slot assignment —
    show swapped presentations to make the probe assumption-free.
    """
    rows = [(v.item_id, v.chose_first) for v in pairwise if v.chose_first is not None]
    if not rows:
        return None
    units = _group_rows(rows)

    def share_first(sample: list[list[bool]]) -> float:
        flat = _pooled(sample)
        return sum(flat) / len(flat) if flat else float("nan")

    value, ci = _estimate(share_first, units, seed)
    ties = sum(v.choice == "tie" for v in pairwise)
    return ProbeResult(
        name="position preference",
        value=value,
        ci=ci,
        null_value=0.5,
        n=len(rows),
        detail=(
            f"chose the presented-first candidate in {value:.1%} of {len(rows)} decisive "
            f"verdicts across {len(units)} items ({ties} ties excluded)"
        ),
    )


def swap_flip_rate(pairwise: Sequence[PairwiseVerdict], *, seed: int = 0) -> ProbeResult | None:
    """Share of pairs whose winner changes when presentation order is swapped.

    Matches verdicts on (item, judge, sample) across ``swapped`` False/True.
    Any flip means presentation order, not content, decided the verdict.
    """
    by_key: dict[tuple[str, str, int], dict[bool, PairwiseVerdict]] = {}
    for verdict in pairwise:
        by_key.setdefault((verdict.item_id, verdict.judge_id, verdict.sample_index), {})[
            verdict.swapped
        ] = verdict
    rows: list[tuple[str, bool]] = []
    for (item_id, _, _), presentations in by_key.items():
        if len(presentations) == 2:
            original, swapped = presentations[False], presentations[True]
            if original.choice != "tie" and swapped.choice != "tie":
                rows.append((item_id, original.choice != swapped.choice))
    if not rows:
        return None
    units = _group_rows(rows)

    def flip_share(sample: list[list[bool]]) -> float:
        flat = _pooled(sample)
        return sum(flat) / len(flat) if flat else float("nan")

    value, ci = _estimate(flip_share, units, seed)
    return ProbeResult(
        name="swap flip rate",
        value=value,
        ci=ci,
        null_value=0.0,
        n=len(rows),
        detail=f"the winner flipped under order swap in {value:.1%} of {len(rows)} matched pairs",
    )


def identical_pair_decisiveness(
    pairwise: Sequence[PairwiseVerdict], *, seed: int = 0
) -> ProbeResult | None:
    """Non-tie rate on pairs marked ``meta={"identical": True}``.

    When both candidates are the same text, any decisive verdict is an
    artifact of position or sampling noise.  This is the cheapest position
    probe to add to a run: duplicate a few candidates and flag them.
    """
    rows = [
        (v.item_id, v.choice != "tie") for v in pairwise if v.meta.get("identical") is True
    ]
    if not rows:
        return None
    units = _group_rows(rows)

    def decisive_share(sample: list[list[bool]]) -> float:
        flat = _pooled(sample)
        return sum(flat) / len(flat) if flat else float("nan")

    value, ci = _estimate(decisive_share, units, seed)
    return ProbeResult(
        name="identical-pair decisiveness",
        value=value,
        ci=ci,
        null_value=0.0,
        n=len(rows),
        detail=(
            f"declared a winner between identical candidates in {value:.1%} of "
            f"{len(rows)} verdicts"
        ),
    )


def _partial_spearman(
    scores: Sequence[float], lengths: Sequence[float], control: Sequence[float]
) -> float:
    """Spearman correlation of scores with lengths, holding the control fixed.

    Ranks all three, removes the control's linear effect from the other two in
    closed form (single regressor, so no linear-algebra library is involved),
    then correlates the residuals.  A constant control cannot be regressed out
    and contributes nothing, so the plain correlation is returned.  When length
    is perfectly collinear with the control, no independent length variation
    remains and the result is nan: the data cannot separate verbosity from
    quality, and no verdict is the honest verdict.
    """
    rank_scores = rank_average(scores)
    rank_lengths = rank_average(lengths)
    rank_control = rank_average(control)

    centred_control = rank_control - rank_control.mean()
    control_variance = float((centred_control**2).sum())
    if control_variance == 0.0:
        return pearson(rank_scores, rank_lengths)

    def residual(ranks: np.ndarray) -> np.ndarray:
        centred = ranks - ranks.mean()
        beta = float((centred * centred_control).sum()) / control_variance
        return centred - beta * centred_control

    return pearson(residual(rank_scores), residual(rank_lengths))


def verbosity_graded(
    judge_human_pairs: Sequence[tuple[GradedVerdict, GradedVerdict]], *, seed: int = 0
) -> ProbeResult | None:
    """Partial rank correlation between score and candidate length.

    Longer answers are often genuinely better, so the raw correlation between
    score and length confounds verbosity with quality.  The human score is the
    quality control: what remains after holding it fixed is preference for
    length itself.
    """
    rows = [
        (judge.item_id, (judge.score, float(judge.candidate_len), human.score))
        for judge, human in judge_human_pairs
        if judge.candidate_len is not None
    ]
    if len(rows) < 3:
        return None
    units = _group_rows(rows)

    def partial(sample: list[list[tuple[float, float, float]]]) -> float:
        flat = _pooled(sample)
        scores = [row[0] for row in flat]
        lengths = [row[1] for row in flat]
        control = [row[2] for row in flat]
        return _partial_spearman(scores, lengths, control)

    value, ci = _estimate(partial, units, seed)
    flat = _pooled(units)
    raw = pearson(
        rank_average([row[0] for row in flat]), rank_average([row[1] for row in flat])
    )
    return ProbeResult(
        name="verbosity bias",
        value=value,
        ci=ci,
        null_value=0.0,
        n=len(rows),
        detail=(
            f"score-length rank correlation is {value:+.2f} after controlling for the human "
            f"score (raw {raw:+.2f}, n={len(rows)})"
        ),
    )


def verbosity_pairwise(
    judge_human_pairs: Sequence[tuple[PairwiseVerdict, PairwiseVerdict]], *, seed: int = 0
) -> ProbeResult | None:
    """How much more often the judge picks the longer candidate than the human does."""
    rows: list[tuple[str, tuple[bool, bool]]] = []
    for judge, human in judge_human_pairs:
        if (
            judge.a_len is None
            or judge.b_len is None
            or judge.a_len == judge.b_len
            or judge.choice == "tie"
            or human.choice == "tie"
        ):
            continue
        longer = "a" if judge.a_len > judge.b_len else "b"
        rows.append((judge.item_id, (judge.choice == longer, human.choice == longer)))
    if not rows:
        return None
    units = _group_rows(rows)

    def excess(sample: list[list[tuple[bool, bool]]]) -> float:
        flat = _pooled(sample)
        if not flat:
            return float("nan")
        judge_share = sum(row[0] for row in flat) / len(flat)
        human_share = sum(row[1] for row in flat) / len(flat)
        return judge_share - human_share

    value, ci = _estimate(excess, units, seed)
    return ProbeResult(
        name="verbosity bias (pairwise)",
        value=value,
        ci=ci,
        null_value=0.0,
        n=len(rows),
        detail=(
            f"picks the longer candidate {value:+.1%} more often than the human reference "
            f"on {len(rows)} unequal-length pairs"
        ),
    )


def self_preference(
    judge_human_pairs: Sequence[tuple[GradedVerdict, GradedVerdict]],
    judge_model: str,
    *,
    seed: int = 0,
) -> ProbeResult | None:
    """Extra generosity toward the judge's own model, human score held fixed.

    Per verdict, the judge's departure from the human score; the probe is the
    mean departure on own-model candidates minus the mean on everyone else's.
    """
    rows = [
        (judge.item_id, (judge.score - human.score, judge.candidate_model == judge_model))
        for judge, human in judge_human_pairs
        if judge.candidate_model is not None
    ]
    own = sum(is_own for _, (_, is_own) in rows)
    if own == 0 or own == len(rows):
        return None
    units = _group_rows(rows)

    def generosity_gap(sample: list[list[tuple[float, bool]]]) -> float:
        flat = _pooled(sample)
        own_deltas = [delta for delta, is_own in flat if is_own]
        other_deltas = [delta for delta, is_own in flat if not is_own]
        if not own_deltas or not other_deltas:
            return float("nan")
        return float(np.mean(own_deltas) - np.mean(other_deltas))

    value, ci = _estimate(generosity_gap, units, seed)
    return ProbeResult(
        name="self-preference",
        value=value,
        ci=ci,
        null_value=0.0,
        n=len(rows),
        detail=(
            f"scores its own model's candidates {value:+.2f} scale points above its "
            f"departure on others ({own} own vs {len(rows) - own} other verdicts)"
        ),
    )


def central_tendency(
    judge_human_pairs: Sequence[tuple[GradedVerdict, GradedVerdict]], *, seed: int = 0
) -> ProbeResult | None:
    """Ratio of judge score spread to human score spread.

    Below 1.0 the judge compresses toward the middle of the scale; above 1.0
    it sprays the extremes.
    """
    rows = [(judge.item_id, (judge.score, human.score)) for judge, human in judge_human_pairs]
    if len(rows) < 3:
        return None
    units = _group_rows(rows)

    def spread_ratio(sample: list[list[tuple[float, float]]]) -> float:
        flat = _pooled(sample)
        if len(flat) < 3:
            return float("nan")
        judge_sd = float(np.std([row[0] for row in flat], ddof=1))
        human_sd = float(np.std([row[1] for row in flat], ddof=1))
        if human_sd == 0.0:
            return float("nan")
        return judge_sd / human_sd

    value, ci = _estimate(spread_ratio, units, seed)
    if math.isnan(value):
        return None
    return ProbeResult(
        name="central tendency",
        value=value,
        ci=ci,
        null_value=1.0,
        n=len(rows),
        detail=f"judge score spread is {value:.2f}x the human spread over {len(rows)} verdicts",
    )
