"""Run every applicable check on a judge's verdicts and collect the results.

The audit takes two verdict files — the judge's and the reference (usually
human) — and produces one ``AuditReport``: agreement, calibration, bias probes
and consistency, each as a ``ProbeResult`` with a bootstrap interval.  Which
checks run is decided by what the data contains; nothing is required beyond
matching ``item_id`` values.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from judgekit import bias, consistency
from judgekit.agreement import cohen_kappa, exact_agreement, spearman, within_one
from judgekit.bootstrap import bootstrap_ci
from judgekit.calibration import brier, ece_equal_mass, murphy_decomposition, reliability_curve
from judgekit.io import align_on_items, by_kind
from judgekit.result import ProbeResult
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict, Verdict

# The finite-sample floor of equal-mass ECE at the demo's scale is a few
# hundredths even for a perfectly calibrated judge, so the innocent band
# stops at 0.05 rather than zero.
INNOCENT_ECE = (0.0, 0.05)


@dataclass(frozen=True)
class AuditReport:
    judge_id: str
    n_graded: int
    n_binary: int
    n_pairwise: int
    agreement: list[ProbeResult] = field(default_factory=list)
    calibration: list[ProbeResult] = field(default_factory=list)
    probes: list[ProbeResult] = field(default_factory=list)
    # Small data slices the figures are drawn from.
    reliability: list[tuple[float, float, int]] | None = None
    graded_scores: tuple[list[float], list[float]] | None = None

    @property
    def results(self) -> list[ProbeResult]:
        return [*self.agreement, *self.calibration, *self.probes]

    @property
    def flags(self) -> list[ProbeResult]:
        return [result for result in self.results if result.triggered is True]


def _grouped_scores(
    pairs: Sequence[tuple[Verdict, Verdict]],
    values: Sequence[tuple[float, float]],
) -> list[list[tuple[float, float]]]:
    units: dict[str, list[tuple[float, float]]] = {}
    for (judge, _), row in zip(pairs, values, strict=True):
        units.setdefault(judge.item_id, []).append(row)
    return list(units.values())


def _pooled(units: list[list[tuple[float, float]]]) -> tuple[list[float], list[float]]:
    flat = [row for unit in units for row in unit]
    return [row[0] for row in flat], [row[1] for row in flat]


def _agreement_graded(
    pairs: Sequence[tuple[GradedVerdict, GradedVerdict]], seed: int
) -> list[ProbeResult]:
    scale_min = pairs[0][0].scale_min
    scale_max = pairs[0][0].scale_max
    values = [(judge.score, human.score) for judge, human in pairs]
    units = _grouped_scores(pairs, values)
    results = []

    integral = all(
        float(judge).is_integer() and float(human).is_integer() for judge, human in values
    )
    if integral and scale_min.is_integer() and scale_max.is_integer():
        labels = [float(v) for v in range(int(scale_min), int(scale_max) + 1)]

        def kappa(sample: list[list[tuple[float, float]]]) -> float:
            judge_scores, human_scores = _pooled(sample)
            return cohen_kappa(judge_scores, human_scores, labels=labels, weights="quadratic")

        value = kappa(units)
        results.append(
            ProbeResult(
                name="quadratic-weighted kappa",
                value=value,
                ci=bootstrap_ci(kappa, units, seed=seed),
                innocent=None,
                n=len(values),
                detail=(
                    f"chance-corrected ordinal agreement with the reference "
                    f"on {len(values)} verdicts"
                ),
            )
        )

    def rho(sample: list[list[tuple[float, float]]]) -> float:
        judge_scores, human_scores = _pooled(sample)
        return spearman(judge_scores, human_scores)

    results.append(
        ProbeResult(
            name="spearman rho",
            value=rho(units),
            ci=bootstrap_ci(rho, units, seed=seed),
            innocent=None,
            n=len(values),
            detail="rank correlation with the reference scores",
        )
    )

    if integral:

        def exact(sample: list[list[tuple[float, float]]]) -> float:
            judge_scores, human_scores = _pooled(sample)
            return exact_agreement(judge_scores, human_scores)

        def close(sample: list[list[tuple[float, float]]]) -> float:
            judge_scores, human_scores = _pooled(sample)
            return within_one(judge_scores, human_scores)

        results.append(
            ProbeResult(
                name="exact agreement",
                value=exact(units),
                ci=bootstrap_ci(exact, units, seed=seed),
                innocent=None,
                n=len(values),
                detail="share of verdicts equal to the reference score",
            )
        )
        results.append(
            ProbeResult(
                name="within one point",
                value=close(units),
                ci=bootstrap_ci(close, units, seed=seed),
                innocent=None,
                n=len(values),
                detail="share of verdicts at most one scale point from the reference",
            )
        )
    return results


def _agreement_binary(
    pairs: Sequence[tuple[BinaryVerdict, BinaryVerdict]], seed: int
) -> list[ProbeResult]:
    values = [(1.0 if judge.label else 0.0, 1.0 if human.label else 0.0) for judge, human in pairs]
    units = _grouped_scores(pairs, values)

    def accuracy(sample: list[list[tuple[float, float]]]) -> float:
        judge_labels, human_labels = _pooled(sample)
        return exact_agreement(judge_labels, human_labels)

    def kappa(sample: list[list[tuple[float, float]]]) -> float:
        judge_labels, human_labels = _pooled(sample)
        return cohen_kappa(judge_labels, human_labels, labels=[0.0, 1.0])

    return [
        ProbeResult(
            name="label accuracy",
            value=accuracy(units),
            ci=bootstrap_ci(accuracy, units, seed=seed),
            innocent=None,
            n=len(values),
            detail=f"share of pass/fail labels matching the reference on {len(values)} verdicts",
        ),
        ProbeResult(
            name="cohen kappa",
            value=kappa(units),
            ci=bootstrap_ci(kappa, units, seed=seed),
            innocent=None,
            n=len(values),
            detail="chance-corrected label agreement",
        ),
    ]


def _agreement_pairwise(
    pairs: Sequence[tuple[PairwiseVerdict, PairwiseVerdict]], seed: int
) -> list[ProbeResult]:
    categories = {"a": 0.0, "b": 1.0, "tie": 2.0}
    values = [(categories[judge.choice], categories[human.choice]) for judge, human in pairs]
    units = _grouped_scores(pairs, values)

    def match(sample: list[list[tuple[float, float]]]) -> float:
        judge_choices, human_choices = _pooled(sample)
        return exact_agreement(judge_choices, human_choices)

    return [
        ProbeResult(
            name="choice agreement",
            value=match(units),
            ci=bootstrap_ci(match, units, seed=seed),
            innocent=None,
            n=len(values),
            detail=(
                f"share of presentations whose canonical choice matches the reference "
                f"on {len(values)} verdicts"
            ),
        )
    ]


def _calibration(
    pairs: Sequence[tuple[BinaryVerdict, BinaryVerdict]], seed: int
) -> tuple[list[ProbeResult], list[tuple[float, float, int]] | None]:
    confident = [
        (judge.p_positive, 1.0 if human.label else 0.0)
        for judge, human in pairs
        if judge.p_positive is not None
    ]
    if len(confident) < 20:
        return [], None
    values = [(p, y) for p, y in confident if p is not None]
    units = _grouped_scores(
        [pair for pair in pairs if pair[0].p_positive is not None], values
    )

    def ece(sample: list[list[tuple[float, float]]]) -> float:
        probabilities, outcomes = _pooled(sample)
        return ece_equal_mass(probabilities, outcomes)

    probabilities, outcomes = _pooled(units)
    reliability_probe = ProbeResult(
        name="expected calibration error",
        value=ece(units),
        ci=bootstrap_ci(ece, units, seed=seed),
        innocent=INNOCENT_ECE,
        n=len(values),
        detail="mean gap between claimed confidence and observed frequency, equal-mass bins",
    )
    rel, res, unc = murphy_decomposition(probabilities, outcomes)
    decomposition = ProbeResult(
        name="brier score",
        value=brier(probabilities, outcomes),
        ci=(float("nan"), float("nan")),
        innocent=None,
        n=len(values),
        detail=(
            f"decomposes exactly into reliability {rel:.3f} - resolution {res:.3f} "
            f"+ uncertainty {unc:.3f}"
        ),
    )
    return [reliability_probe, decomposition], reliability_curve(probabilities, outcomes)


def run_audit(
    judge: Sequence[Verdict],
    human: Sequence[Verdict],
    *,
    judge_model: str | None = None,
    seed: int = 0,
) -> AuditReport:
    """Audit one judge against one reference; see the module docstring."""
    judge_graded, judge_binary, judge_pairwise = by_kind(judge)
    human_graded, human_binary, human_pairwise = by_kind(human)
    judge_id = judge[0].judge_id if judge else "judge"

    agreement: list[ProbeResult] = []
    calibration: list[ProbeResult] = []
    probes: list[ProbeResult] = []
    reliability = None
    graded_scores = None

    graded_pairs = align_on_items(judge_graded, human_graded) if human_graded else []
    if graded_pairs:
        scales = {(v.scale_min, v.scale_max) for v, _ in graded_pairs} | {
            (v.scale_min, v.scale_max) for _, v in graded_pairs
        }
        if len(scales) > 1:
            raise ValueError(f"mixed score scales in one audit: {sorted(scales)}")
        agreement.extend(_agreement_graded(graded_pairs, seed))
        graded_scores = (
            [judge_verdict.score for judge_verdict, _ in graded_pairs],
            [human_verdict.score for _, human_verdict in graded_pairs],
        )
        for probe in (
            bias.verbosity_graded(graded_pairs, seed=seed),
            bias.self_preference(graded_pairs, judge_model, seed=seed) if judge_model else None,
            bias.central_tendency(graded_pairs, seed=seed),
        ):
            if probe is not None:
                probes.append(probe)

    binary_pairs = align_on_items(judge_binary, human_binary) if human_binary else []
    if binary_pairs:
        agreement.extend(_agreement_binary(binary_pairs, seed))
        calibration, reliability = _calibration(binary_pairs, seed)

    if judge_pairwise:
        pairwise_pairs = (
            align_on_items(judge_pairwise, human_pairwise) if human_pairwise else []
        )
        if pairwise_pairs:
            agreement.extend(_agreement_pairwise(pairwise_pairs, seed))
            verbosity = bias.verbosity_pairwise(pairwise_pairs, seed=seed)
            if verbosity is not None:
                probes.append(verbosity)
        for probe in (
            bias.position_preference(judge_pairwise, seed=seed),
            bias.swap_flip_rate(judge_pairwise, seed=seed),
            bias.identical_pair_decisiveness(judge_pairwise, seed=seed),
        ):
            if probe is not None:
                probes.append(probe)

    for kind_verdicts in (judge_graded, judge_binary, judge_pairwise):
        if kind_verdicts:
            stability = consistency.resample_consistency(kind_verdicts, seed=seed)
            if stability is not None:
                probes.append(stability)

    return AuditReport(
        judge_id=judge_id,
        n_graded=len(judge_graded),
        n_binary=len(judge_binary),
        n_pairwise=len(judge_pairwise),
        agreement=agreement,
        calibration=calibration,
        probes=probes,
        reliability=reliability,
        graded_scores=graded_scores,
    )
