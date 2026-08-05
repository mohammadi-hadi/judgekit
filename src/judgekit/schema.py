"""Verdict records: what a judge said about an item.

Three judging modes exist in practice, and they support different audits:

- graded:   one candidate, one numeric score on a fixed scale.  Agreement,
            verbosity and self-preference probes live here.
- binary:   pass/fail with an optional confidence.  Calibration lives here,
            because calibration is only defined when the number attached to a
            verdict claims to be a probability.
- pairwise: choose between two candidates.  Position probes live here, because
            position only exists when there are two slots.

Human labels use the same records with ``judge_id="human"``; audits take the
judge file and the human file as two inputs and align them on ``item_id``.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

MetaValue = str | int | float | bool


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    judge_id: str = "judge"
    # 0 for a single judgment; 0..k-1 when the same judge re-judged the same
    # item, which is what the consistency audit consumes.
    sample_index: int = 0
    rationale: str | None = None
    meta: dict[str, MetaValue] = Field(default_factory=dict)


class GradedVerdict(_Record):
    """A numeric score for one candidate on a fixed scale."""

    kind: Literal["graded"] = "graded"
    score: float
    scale_min: float = 1.0
    scale_max: float = 5.0
    # Length in any unit (characters, tokens) as long as it is consistent
    # across a file; only rank correlations are computed on it.
    candidate_len: int | None = None
    candidate_model: str | None = None

    @model_validator(mode="after")
    def _score_on_scale(self) -> GradedVerdict:
        if not self.scale_min < self.scale_max:
            raise ValueError(f"empty scale [{self.scale_min}, {self.scale_max}]")
        if not self.scale_min <= self.score <= self.scale_max:
            raise ValueError(
                f"score {self.score} outside scale [{self.scale_min}, {self.scale_max}]"
            )
        return self


class BinaryVerdict(_Record):
    """A pass/fail label, optionally with the judge's confidence in that label."""

    kind: Literal["binary"] = "binary"
    label: bool
    confidence: float | None = None

    @model_validator(mode="after")
    def _confidence_is_probability(self) -> BinaryVerdict:
        if self.confidence is not None and not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence {self.confidence} outside [0, 1]")
        return self

    @property
    def p_positive(self) -> float | None:
        """Probability assigned to the positive label.

        ``confidence`` is confidence in the emitted label (how people log it);
        calibration needs the probability of a fixed event.  A fail at 0.9
        confidence is a pass at probability 0.1.
        """
        if self.confidence is None:
            return None
        return self.confidence if self.label else 1.0 - self.confidence


class PairwiseVerdict(_Record):
    """A choice between candidate A and candidate B of one item.

    ``choice`` is always in canonical space: ``"a"`` means canonical candidate
    A won, no matter which slot it was shown in.  ``swapped=True`` records that
    the pair was presented in reversed order, canonical B first.  Position
    probes recover the presented winner from the two fields together.
    """

    kind: Literal["pairwise"] = "pairwise"
    choice: Literal["a", "b", "tie"]
    swapped: bool = False
    confidence: float | None = None
    a_len: int | None = None
    b_len: int | None = None
    a_model: str | None = None
    b_model: str | None = None

    @property
    def chose_first(self) -> bool | None:
        """True if the judge chose whichever candidate was presented first."""
        if self.choice == "tie":
            return None
        return (self.choice == "a") != self.swapped


Verdict = Annotated[GradedVerdict | BinaryVerdict | PairwiseVerdict, Field(discriminator="kind")]

VERDICT_ADAPTER: TypeAdapter[Verdict] = TypeAdapter(Verdict)
