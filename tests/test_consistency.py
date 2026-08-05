import pytest

from judgekit.consistency import resample_consistency
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict


def test_graded_unanimity_and_spread():
    verdicts = []
    for i in range(20):
        stable = i < 10
        for k in range(3):
            score = 3.0 if stable else float(3 + (k % 2))
            verdicts.append(GradedVerdict(item_id=f"i{i}", score=score, sample_index=k))
    result = resample_consistency(verdicts)
    assert result.value == pytest.approx(0.5)
    assert result.n == 20
    assert "spread" in result.detail


def test_pairwise_groups_by_presentation_order():
    verdicts = [
        # Same order re-judgments agree...
        PairwiseVerdict(item_id="i0", choice="a", swapped=False, sample_index=0),
        PairwiseVerdict(item_id="i0", choice="a", swapped=False, sample_index=1),
        # ...the swapped-order re-judgments do not.
        PairwiseVerdict(item_id="i0", choice="a", swapped=True, sample_index=0),
        PairwiseVerdict(item_id="i0", choice="b", swapped=True, sample_index=1),
    ]
    result = resample_consistency(verdicts)
    assert result.value == pytest.approx(0.5)
    assert result.n == 2


def test_binary_unanimity():
    verdicts = []
    for k in range(2):
        verdicts.append(BinaryVerdict(item_id="i0", label=True, sample_index=k))
        verdicts.append(BinaryVerdict(item_id="i1", label=k == 0, sample_index=k))
    result = resample_consistency(verdicts)
    assert result.value == pytest.approx(0.5)


def test_no_rejudgments_returns_none():
    verdicts = [GradedVerdict(item_id=f"i{i}", score=3.0) for i in range(5)]
    assert resample_consistency(verdicts) is None


def test_mixed_kinds_rejected():
    verdicts = [
        GradedVerdict(item_id="i0", score=3.0),
        BinaryVerdict(item_id="i0", label=True),
    ]
    with pytest.raises(ValueError, match="one verdict kind"):
        resample_consistency(verdicts)
