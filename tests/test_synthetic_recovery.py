"""Every probe must recover the pathology that was implanted — and stay quiet
when nothing was.

These tests audit simulated judges whose defect magnitudes are dials we set.
They are the reason to believe the probes measure what they claim: detection
on the guilty judge, silence on the clean one, and monotonicity in the dial.
"""

import pytest

from judgekit.agreement import cohen_kappa, rank_average, spearman
from judgekit.bias import (
    central_tendency,
    identical_pair_decisiveness,
    position_preference,
    self_preference,
    swap_flip_rate,
    verbosity_graded,
    verbosity_pairwise,
)
from judgekit.calibration import ece_equal_mass
from judgekit.consistency import resample_consistency
from judgekit.io import align_on_items
from judgekit.synthetic import (
    binary_judge,
    binary_world,
    graded_items,
    graded_judge,
    human_graded,
    human_pairwise,
    pair_items,
    pairwise_judge,
)

ITEMS = graded_items(300, seed=0)
HUMAN = human_graded(ITEMS)
PAIRS = pair_items(200, seed=0, n_identical=20)
HUMAN_PAIRS = human_pairwise(PAIRS)


def aligned(judge_verdicts):
    return align_on_items(judge_verdicts, HUMAN)


# ----------------------------------------------------------- clean judge ----


def test_clean_judge_agrees_with_humans_and_triggers_nothing():
    clean = graded_judge(ITEMS, judge_id="clean", seed=1)
    pairs = aligned(clean)
    kappa = cohen_kappa(
        [judge.score for judge, _ in pairs],
        [human.score for _, human in pairs],
        labels=[1, 2, 3, 4, 5],
        weights="quadratic",
    )
    assert kappa > 0.7
    assert verbosity_graded(pairs).triggered is False
    assert self_preference(pairs, judge_model="judge-model").triggered is False
    assert central_tendency(pairs).triggered is False


def test_clean_pairwise_judge_sits_at_half():
    clean = pairwise_judge(PAIRS, judge_id="clean", seed=3)
    result = position_preference(clean)
    assert 0.45 < result.value < 0.55
    assert result.triggered is False


# -------------------------------------------------------------- position ----


def test_position_dial_recovered_exactly_when_ties_are_off():
    # With no tie margin the base mechanism picks first exactly half the
    # time, so an override probability of 0.4 puts the truth at 0.7.
    judge = pairwise_judge(PAIRS, seed=2, tie_margin=0.0, position=0.4)
    result = position_preference(judge)
    assert result.value == pytest.approx(0.7, abs=0.07)
    assert result.triggered is True


def test_position_estimate_is_monotone_in_the_dial():
    estimates = [
        position_preference(pairwise_judge(PAIRS, seed=4, position=dial)).value
        for dial in (0.0, 0.3, 0.6)
    ]
    assert estimates[0] < estimates[1] < estimates[2]


def test_position_bias_inflates_flip_rate_and_identical_decisiveness():
    clean = pairwise_judge(PAIRS, judge_id="clean", seed=5)
    slave = pairwise_judge(PAIRS, judge_id="slave", seed=5, tie_margin=0.0, position=1.0)
    assert swap_flip_rate(slave).value > swap_flip_rate(clean).value
    assert identical_pair_decisiveness(slave).value == pytest.approx(1.0)
    decisive_clean = identical_pair_decisiveness(clean)
    assert 0.0 < decisive_clean.value < 1.0
    assert decisive_clean.triggered is None and swap_flip_rate(clean).triggered is None


# ------------------------------------------------------------- verbosity ----


def test_verbosity_dial_is_detected_and_monotone():
    weak = verbosity_graded(aligned(graded_judge(ITEMS, seed=7, verbosity=0.4)))
    strong = verbosity_graded(aligned(graded_judge(ITEMS, seed=7, verbosity=1.2)))
    assert strong.triggered is True
    assert strong.value > weak.value > 0.0


def test_confounded_length_does_not_fool_the_partial_correlation():
    # Longer answers really are better here, and the judge is honest.  The raw
    # score-length correlation is large; the partial one must not indict.
    items = graded_items(300, seed=5, confound_length=True)
    judge = graded_judge(items, seed=6)
    pairs = align_on_items(judge, human_graded(items))
    scores = [j.score for j, _ in pairs]
    lengths = [float(j.candidate_len) for j, _ in pairs]
    assert spearman(rank_average(scores), rank_average(lengths)) > 0.5
    result = verbosity_graded(pairs)
    assert result.triggered is False


def test_pairwise_verbosity_detected_only_when_implanted():
    human_by_item = HUMAN_PAIRS
    verbose = pairwise_judge(PAIRS, seed=8, verbosity=0.5)
    clean = pairwise_judge(PAIRS, seed=8)
    verbose_pairs = align_on_items(verbose, human_by_item)
    clean_pairs = align_on_items(clean, human_by_item)
    assert verbosity_pairwise(verbose_pairs).triggered is True
    assert verbosity_pairwise(clean_pairs).triggered is False


# ------------------------------------------------- self-preference et al ----


def test_self_preference_dial_recovered():
    generous = graded_judge(ITEMS, seed=9, self_preference=1.0)
    result = self_preference(aligned(generous), judge_model="judge-model")
    assert 0.6 < result.value < 1.4
    assert result.triggered is True


def test_middle_shrink_recovered():
    middling = graded_judge(ITEMS, seed=10, shrink=0.5)
    result = central_tendency(aligned(middling))
    assert 0.35 < result.value < 0.7
    assert result.triggered is True


# ------------------------------------------------------------ calibration ---


def test_calibration_separates_honest_from_sharpened():
    probabilities, outcomes = binary_world(1000, seed=0)
    labels = [1.0 if outcome.label else 0.0 for outcome in outcomes]

    honest = [v.p_positive for v in binary_judge(probabilities)]
    sharpened = [v.p_positive for v in binary_judge(probabilities, temperature=0.5)]

    honest_ece = ece_equal_mass(honest, labels)
    sharpened_ece = ece_equal_mass(sharpened, labels)
    assert honest_ece < 0.05
    assert sharpened_ece > 0.08
    assert sharpened_ece > honest_ece


# ------------------------------------------------------------ consistency ---


def test_rejudgment_unanimity_orders_by_noise():
    quiet = graded_judge(ITEMS, seed=11, noise_sd=0.15, n_samples=3)
    loud = graded_judge(ITEMS, seed=11, noise_sd=0.8, n_samples=3)
    unanimity_quiet = resample_consistency(quiet)
    unanimity_loud = resample_consistency(loud)
    assert unanimity_quiet.value > unanimity_loud.value
    assert unanimity_quiet.n == 300
