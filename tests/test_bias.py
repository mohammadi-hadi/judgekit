import pytest

from judgekit.bias import (
    central_tendency,
    identical_pair_decisiveness,
    position_preference,
    self_preference,
    swap_flip_rate,
    verbosity_graded,
    verbosity_pairwise,
)
from judgekit.schema import GradedVerdict, PairwiseVerdict


def _pair(item, choice, swapped=False, **kwargs):
    return PairwiseVerdict(item_id=item, choice=choice, swapped=swapped, **kwargs)


def _graded_pair(item, judge_score, human_score, **kwargs):
    judge = GradedVerdict(item_id=item, score=judge_score, scale_max=10.0, **kwargs)
    human = GradedVerdict(item_id=item, judge_id="human", score=human_score, scale_max=10.0)
    return judge, human


def test_position_preference_flags_first_slave():
    verdicts = []
    for i in range(40):
        verdicts.append(_pair(f"i{i}", "a", swapped=False))
        verdicts.append(_pair(f"i{i}", "b", swapped=True))
    result = position_preference(verdicts)
    assert result.value == pytest.approx(1.0)
    assert result.triggered is True


def test_position_preference_passes_balanced_judge():
    verdicts = [_pair(f"i{i}", "a" if i % 2 == 0 else "b") for i in range(40)]
    result = position_preference(verdicts)
    assert result.value == pytest.approx(0.5)
    assert result.triggered is False


def test_position_preference_needs_decisive_verdicts():
    assert position_preference([_pair("i0", "tie")]) is None


def test_swap_flip_rate_counts_only_matched_flips():
    verdicts = []
    for i in range(30):
        flips = i < 6
        verdicts.append(_pair(f"i{i}", "a", swapped=False))
        verdicts.append(_pair(f"i{i}", "b" if flips else "a", swapped=True))
    result = swap_flip_rate(verdicts)
    assert result.value == pytest.approx(0.2)
    assert result.n == 30
    assert result.triggered is None


def test_swap_flip_rate_zero_for_stable_judge():
    verdicts = []
    for i in range(20):
        verdicts.append(_pair(f"i{i}", "a", swapped=False))
        verdicts.append(_pair(f"i{i}", "a", swapped=True))
    result = swap_flip_rate(verdicts)
    assert result.value == pytest.approx(0.0)
    assert result.triggered is None


def test_swap_flip_rate_needs_both_orders():
    assert swap_flip_rate([_pair("i0", "a")]) is None


def test_identical_pair_decisiveness():
    verdicts = [
        _pair(f"i{i}", "a" if i < 4 else "tie", meta={"identical": True}) for i in range(10)
    ]
    result = identical_pair_decisiveness(verdicts)
    assert result.value == pytest.approx(0.4)
    assert identical_pair_decisiveness([_pair("i0", "a")]) is None


def test_verbosity_graded_flags_length_lover():
    pairs = []
    for i in range(60):
        human = float(i % 5 + 1)
        length = (i * 7) % 60 + 10
        judge_score = human + (1.0 if length > 39 else 0.0)
        pairs.append(_graded_pair(f"i{i}", judge_score, human, candidate_len=length))
    result = verbosity_graded(pairs)
    assert result.value > 0.3
    assert result.triggered is True


def test_verbosity_graded_partial_clears_quality_confound():
    # Length tracks true quality closely and the judge follows the human
    # exactly (plus a tiny wiggle unrelated to length): the raw correlation is
    # large, the partial correlation is not.
    pairs = []
    for i in range(60):
        human = float(i % 5 + 1)
        length = int(human * 10) + i % 7
        judge_score = human + 0.1 * (i % 3)
        pairs.append(_graded_pair(f"i{i}", judge_score, human, candidate_len=length))
    result = verbosity_graded(pairs)
    assert abs(result.value) < 0.15
    assert result.triggered is False
    assert "raw +0.9" in result.detail


def test_verbosity_graded_undefined_under_perfect_collinearity():
    # When length never varies independently of the human score, verbosity
    # cannot be separated from quality; the probe reports nan, not a verdict.
    pairs = []
    for i in range(60):
        human = float(i % 5 + 1)
        pairs.append(_graded_pair(f"i{i}", human, human, candidate_len=int(human * 10)))
    result = verbosity_graded(pairs)
    assert result.value != result.value  # nan
    assert result.triggered is None


def test_verbosity_graded_needs_lengths():
    pairs = [_graded_pair(f"i{i}", 3.0, 3.0) for i in range(10)]
    assert verbosity_graded(pairs) is None


def test_verbosity_pairwise_flags_longer_lover():
    pairs = []
    for i in range(40):
        human_choice = "a" if i % 2 == 0 else "b"
        judge = _pair(f"i{i}", "b", a_len=50, b_len=200)
        human = PairwiseVerdict(item_id=f"i{i}", judge_id="human", choice=human_choice)
        pairs.append((judge, human))
    result = verbosity_pairwise(pairs)
    assert result.value == pytest.approx(0.5)
    assert result.triggered is True


def test_self_preference_flags_generous_judge():
    pairs = []
    for i in range(50):
        own = i % 2 == 0
        human = float(i % 5 + 1)
        judge_score = human + (1.0 if own else 0.0)
        pairs.append(
            _graded_pair(
                f"i{i}", judge_score, human, candidate_model="own-model" if own else "other-model"
            )
        )
    result = self_preference(pairs, judge_model="own-model")
    assert result.value == pytest.approx(1.0)
    assert result.triggered is True


def test_self_preference_needs_both_groups():
    pairs = [
        _graded_pair(f"i{i}", 3.0, 3.0, candidate_model="own-model") for i in range(10)
    ]
    assert self_preference(pairs, judge_model="own-model") is None


def test_central_tendency_flags_compression():
    pairs = []
    for i in range(50):
        human = float(i % 5 + 1)
        judge_score = 3.0 + 0.5 * (human - 3.0)
        pairs.append(_graded_pair(f"i{i}", judge_score, human))
    result = central_tendency(pairs)
    assert result.value == pytest.approx(0.5, abs=0.01)
    assert result.triggered is True


def test_central_tendency_passes_identity_judge():
    pairs = [_graded_pair(f"i{i}", float(i % 5 + 1), float(i % 5 + 1)) for i in range(50)]
    result = central_tendency(pairs)
    assert result.value == pytest.approx(1.0)
    assert result.triggered is False
