"""Read, write and align verdicts stored as JSON Lines."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TypeVar

from pydantic import ValidationError

from judgekit.schema import (
    VERDICT_ADAPTER,
    BinaryVerdict,
    GradedVerdict,
    PairwiseVerdict,
    Verdict,
)

V = TypeVar("V", GradedVerdict, BinaryVerdict, PairwiseVerdict)


def load_verdicts(path: str | Path) -> list[Verdict]:
    """Load one verdict per line, failing with the line number on bad input."""
    records: list[Verdict] = []
    for lineno, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            records.append(VERDICT_ADAPTER.validate_json(line))
        except ValidationError as exc:
            raise ValueError(f"{path}, line {lineno}: {exc}") from exc
    return records


def dump_verdicts(verdicts: Iterable[Verdict], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as fh:
        for verdict in verdicts:
            fh.write(verdict.model_dump_json(exclude_none=True) + "\n")


def by_kind(
    verdicts: Iterable[Verdict],
) -> tuple[list[GradedVerdict], list[BinaryVerdict], list[PairwiseVerdict]]:
    graded: list[GradedVerdict] = []
    binary: list[BinaryVerdict] = []
    pairwise: list[PairwiseVerdict] = []
    for verdict in verdicts:
        if isinstance(verdict, GradedVerdict):
            graded.append(verdict)
        elif isinstance(verdict, BinaryVerdict):
            binary.append(verdict)
        else:
            pairwise.append(verdict)
    return graded, binary, pairwise


def align_on_items(judge: Sequence[V], human: Sequence[V]) -> list[tuple[V, V]]:
    """Pair judge verdicts with the human verdict for the same item.

    Every judge verdict for an item (including re-judgments and swapped
    presentations) is paired with that item's single human verdict.  A file
    with several human raters per item must be aggregated first;
    ``agreement.krippendorff_alpha`` is the one audit that consumes multiple
    raters directly.
    """
    reference: dict[str, V] = {}
    for verdict in human:
        if verdict.item_id in reference:
            raise ValueError(
                f"multiple human verdicts for item {verdict.item_id!r}; "
                "aggregate raters into one verdict per item first"
            )
        reference[verdict.item_id] = verdict
    return [(v, reference[v.item_id]) for v in judge if v.item_id in reference]
