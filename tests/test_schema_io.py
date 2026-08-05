import pytest
from pydantic import ValidationError

from judgekit.io import align_on_items, by_kind, dump_verdicts, load_verdicts
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict


def test_round_trip(tmp_path):
    verdicts = [
        GradedVerdict(item_id="i1", score=4.0, candidate_len=120, candidate_model="m1"),
        BinaryVerdict(item_id="i2", label=False, confidence=0.8),
        PairwiseVerdict(item_id="i3", choice="b", swapped=True, a_len=50, b_len=200),
    ]
    path = tmp_path / "verdicts.jsonl"
    dump_verdicts(verdicts, path)
    assert load_verdicts(path) == verdicts


def test_score_outside_scale_raises():
    with pytest.raises(ValidationError, match="outside scale"):
        GradedVerdict(item_id="i", score=6.0, scale_min=1.0, scale_max=5.0)


def test_unknown_field_raises():
    with pytest.raises(ValidationError):
        GradedVerdict(item_id="i", score=3.0, scores=3.0)


def test_bad_line_reports_line_number(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    good = GradedVerdict(item_id="i", score=3.0).model_dump_json()
    path.write_text(good + "\n" + '{"kind": "graded", "item_id": "j"}\n')
    with pytest.raises(ValueError, match="line 2"):
        load_verdicts(path)


@pytest.mark.parametrize(
    ("swapped", "choice", "expected"),
    [
        (False, "a", True),
        (False, "b", False),
        (True, "a", False),
        (True, "b", True),
        (False, "tie", None),
        (True, "tie", None),
    ],
)
def test_chose_first(swapped, choice, expected):
    verdict = PairwiseVerdict(item_id="i", choice=choice, swapped=swapped)
    assert verdict.chose_first is expected


def test_p_positive():
    assert BinaryVerdict(item_id="i", label=True, confidence=0.9).p_positive == 0.9
    assert BinaryVerdict(item_id="i", label=False, confidence=0.9).p_positive == pytest.approx(0.1)
    assert BinaryVerdict(item_id="i", label=True).p_positive is None


def test_align_pairs_and_skips_unmatched():
    judge = [
        GradedVerdict(item_id="i1", score=4.0),
        GradedVerdict(item_id="i1", score=5.0, sample_index=1),
        GradedVerdict(item_id="missing", score=2.0),
    ]
    human = [GradedVerdict(item_id="i1", judge_id="human", score=3.0)]
    pairs = align_on_items(judge, human)
    assert [(j.score, h.score) for j, h in pairs] == [(4.0, 3.0), (5.0, 3.0)]


def test_align_duplicate_human_raises():
    human = [
        GradedVerdict(item_id="i1", judge_id="r1", score=3.0),
        GradedVerdict(item_id="i1", judge_id="r2", score=4.0),
    ]
    with pytest.raises(ValueError, match="multiple human verdicts"):
        align_on_items([], human)


def test_by_kind_split():
    verdicts = [
        GradedVerdict(item_id="i1", score=4.0),
        BinaryVerdict(item_id="i2", label=True),
        PairwiseVerdict(item_id="i3", choice="tie"),
    ]
    graded, binary, pairwise = by_kind(verdicts)
    assert [len(graded), len(binary), len(pairwise)] == [1, 1, 1]
