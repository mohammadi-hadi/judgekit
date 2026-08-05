"""Synthetic judges with implanted pathologies.

Validation by implantation: build a world where the true pathology magnitude
is a dial we set, emit verdicts from a simulated judge with exactly that
pathology, run the real probes, and require the estimate to recover the dial.
The test suite does this for every probe, including the case that matters
most — the clean judge, on which nothing may trigger.

The same generators at fixed seeds produce the demo report, so every number
in the README comes from a judge whose ground truth is known.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict

SCALE_MIN, SCALE_MAX = 1.0, 5.0


# ---------------------------------------------------------------- graded ----


@dataclass(frozen=True)
class GradedItem:
    item_id: str
    quality: int  # the human score, 1..5
    length: int
    model: str


def graded_items(
    n_items: int = 300, seed: int = 0, *, confound_length: bool = False
) -> list[GradedItem]:
    """Items with known quality; lengths independent of quality by default.

    With ``confound_length=True`` longer answers really are better, which is
    the world where a raw score-length correlation lies and the partial one
    must not.
    """
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        quality = int(rng.integers(1, 6))
        if confound_length:
            length = int(40 * quality + rng.integers(0, 40))
        else:
            length = int(rng.integers(20, 400))
        model = ("model-a", "model-b", "judge-model")[i % 3]
        items.append(GradedItem(item_id=f"g{i}", quality=quality, length=length, model=model))
    return items


def human_graded(items: list[GradedItem]) -> list[GradedVerdict]:
    return [
        GradedVerdict(item_id=item.item_id, judge_id="human", score=float(item.quality))
        for item in items
    ]


def _clip_round(raw: float) -> float:
    return float(min(max(round(raw), SCALE_MIN), SCALE_MAX))


def _z(values: list[int]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    sd = float(array.std())
    if sd == 0.0:
        return [0.0] * len(values)
    return [float(v) for v in (array - array.mean()) / sd]


def graded_judge(
    items: list[GradedItem],
    *,
    judge_id: str = "judge",
    seed: int = 1,
    noise_sd: float = 0.4,
    verbosity: float = 0.0,
    self_preference: float = 0.0,
    shrink: float = 1.0,
    n_samples: int = 1,
) -> list[GradedVerdict]:
    """A graded judge whose defects are the keyword arguments.

    All dials at their defaults give the clean judge: quality plus a little
    noise.  ``verbosity`` adds standardized length to the perceived quality,
    ``self_preference`` adds a bonus when the candidate came from
    ``judge-model``, and ``shrink`` scales departures from the scale midpoint
    (1.0 leaves them alone; 0.5 halves them).
    """
    rng = np.random.default_rng(seed)
    z_length = _z([item.length for item in items])
    verdicts = []
    for sample in range(n_samples):
        for item, z_len in zip(items, z_length, strict=True):
            raw = float(item.quality)
            raw += verbosity * z_len
            if item.model == "judge-model":
                raw += self_preference
            raw = 3.0 + shrink * (raw - 3.0)
            raw += float(rng.normal(0.0, noise_sd))
            verdicts.append(
                GradedVerdict(
                    item_id=item.item_id,
                    judge_id=judge_id,
                    sample_index=sample,
                    score=_clip_round(raw),
                    candidate_len=item.length,
                    candidate_model=item.model,
                )
            )
    return verdicts


# -------------------------------------------------------------- pairwise ----


@dataclass(frozen=True)
class PairItem:
    item_id: str
    quality_a: float
    quality_b: float
    length_a: int
    length_b: int
    identical: bool


def pair_items(n_items: int = 200, seed: int = 0, *, n_identical: int = 0) -> list[PairItem]:
    """Pairs with continuous known qualities, so exact ties never occur.

    The last ``n_identical`` items present the same candidate twice — the
    cheapest artifact probe there is.
    """
    rng = np.random.default_rng(seed)
    items = []
    for i in range(n_items):
        identical = i >= n_items - n_identical
        quality_a = float(rng.uniform(0.0, 1.0))
        length_a = int(rng.integers(20, 400))
        if identical:
            quality_b, length_b = quality_a, length_a
        else:
            quality_b = float(rng.uniform(0.0, 1.0))
            length_b = int(rng.integers(20, 400))
        items.append(
            PairItem(
                item_id=f"p{i}",
                quality_a=quality_a,
                quality_b=quality_b,
                length_a=length_a,
                length_b=length_b,
                identical=identical,
            )
        )
    return items


def human_pairwise(items: list[PairItem]) -> list[PairwiseVerdict]:
    return [
        PairwiseVerdict(
            item_id=item.item_id,
            judge_id="human",
            choice="tie" if item.identical else ("a" if item.quality_a > item.quality_b else "b"),
        )
        for item in items
    ]


def pairwise_judge(
    items: list[PairItem],
    *,
    judge_id: str = "judge",
    seed: int = 1,
    noise_sd: float = 0.15,
    tie_margin: float = 0.1,
    position: float = 0.0,
    verbosity: float = 0.0,
    n_samples: int = 1,
) -> list[PairwiseVerdict]:
    """A pairwise judge presenting every pair in both orders.

    Per presentation the judge perceives each candidate's quality with fresh
    noise (plus ``verbosity`` times standardized length), declares a tie when
    the perceived gap is inside ``tie_margin``, and otherwise picks the
    stronger — except that with probability ``position`` it takes whatever
    was presented first.  With ``position=0`` and symmetric candidates the
    expected share of first-slot picks is exactly 0.5; with probability p of
    override it is 0.5 + p/2 when ``tie_margin`` is zero, which is the closed
    form the recovery test checks.
    """
    rng = np.random.default_rng(seed)
    z_length = _z([length for item in items for length in (item.length_a, item.length_b)])
    z_a = z_length[0::2]
    z_b = z_length[1::2]
    verdicts = []
    for sample in range(n_samples):
        for item, za, zb in zip(items, z_a, z_b, strict=True):
            for swapped in (False, True):
                perceived_a = item.quality_a + verbosity * za + float(rng.normal(0.0, noise_sd))
                perceived_b = item.quality_b + verbosity * zb + float(rng.normal(0.0, noise_sd))
                choice: Literal["a", "b", "tie"]
                if abs(perceived_a - perceived_b) < tie_margin:
                    choice = "tie"
                else:
                    choice = "a" if perceived_a > perceived_b else "b"
                if choice != "tie" and float(rng.uniform()) < position:
                    # Take the presented-first candidate: canonical A normally,
                    # canonical B when the pair is swapped.
                    choice = "b" if swapped else "a"
                verdicts.append(
                    PairwiseVerdict(
                        item_id=item.item_id,
                        judge_id=judge_id,
                        sample_index=sample,
                        choice=choice,
                        swapped=swapped,
                        a_len=item.length_a,
                        b_len=item.length_b,
                        meta={"identical": True} if item.identical else {},
                    )
                )
    return verdicts


# ---------------------------------------------------------------- binary ----


def binary_world(
    n_items: int = 1000, seed: int = 0
) -> tuple[list[float], list[BinaryVerdict]]:
    """True pass probabilities and the outcomes drawn from them."""
    rng = np.random.default_rng(seed)
    probabilities = [float(p) for p in rng.uniform(0.05, 0.95, size=n_items)]
    outcomes = [
        BinaryVerdict(item_id=f"b{i}", judge_id="human", label=bool(rng.uniform() < p))
        for i, p in enumerate(probabilities)
    ]
    return probabilities, outcomes


def binary_judge(
    probabilities: list[float],
    *,
    judge_id: str = "judge",
    temperature: float = 1.0,
) -> list[BinaryVerdict]:
    """A judge that knows each item's true pass probability.

    ``temperature=1`` reports it honestly and is calibrated by construction.
    Below 1 the probability is sharpened on the logit scale — the shape of
    overconfidence — and above 1 it is flattened toward 0.5.
    """
    verdicts = []
    for i, p in enumerate(probabilities):
        logit = math.log(p / (1.0 - p))
        claimed = 1.0 / (1.0 + math.exp(-logit / temperature))
        label = claimed >= 0.5
        verdicts.append(
            BinaryVerdict(
                item_id=f"b{i}",
                judge_id=judge_id,
                label=label,
                confidence=claimed if label else 1.0 - claimed,
            )
        )
    return verdicts
