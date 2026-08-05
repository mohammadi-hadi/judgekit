"""The audit must say which probes could not run and what would enable them."""

import json

import pytest

from judgekit.audit import run_audit
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


def _skip_names(audit):
    return {skip.name for skip in audit.skipped}


def test_graded_without_reference_reports_the_gap():
    items = graded_items(20, seed=0)
    judge = graded_judge(items, seed=1)
    audit = run_audit(list(judge), [])
    assert "graded agreement and bias probes" in _skip_names(audit)
    assert audit.agreement == []


def test_missing_candidate_len_skips_verbosity():
    items = graded_items(20, seed=0)
    judge = [v.model_copy(update={"candidate_len": None}) for v in graded_judge(items, seed=1)]
    audit = run_audit(judge, list(human_graded(items)), judge_model="judge-model")
    names = _skip_names(audit)
    assert "verbosity bias" in names
    assert all(result.name != "verbosity bias" for result in audit.results)


def test_missing_judge_model_skips_self_preference():
    items = graded_items(20, seed=0)
    judge = graded_judge(items, seed=1)
    audit = run_audit(list(judge), list(human_graded(items)))
    skip = next(s for s in audit.skipped if s.name == "self-preference")
    assert "judge_model" in skip.needs


def test_unconfident_binary_skips_calibration_with_count():
    probabilities, outcomes = binary_world(50, seed=0)
    judge = [v.model_copy(update={"confidence": None}) for v in binary_judge(probabilities)]
    audit = run_audit(judge, list(outcomes))
    skip = next(s for s in audit.skipped if s.name == "calibration (ECE, Brier)")
    assert "found 0" in skip.needs


def test_single_order_pairwise_skips_swap_and_identical():
    items = pair_items(30, seed=0)
    judge = [v for v in pairwise_judge(items, seed=1) if not v.swapped]
    audit = run_audit(judge, list(human_pairwise(items)))
    names = _skip_names(audit)
    assert "swap flip rate" in names
    assert "identical-pair decisiveness" in names


def test_single_samples_skip_unanimity():
    items = graded_items(20, seed=0)
    judge = graded_judge(items, seed=1)
    audit = run_audit(list(judge), list(human_graded(items)), judge_model="judge-model")
    assert "re-judgment unanimity" in _skip_names(audit)

    rejudged = graded_judge(items, seed=1, n_samples=3)
    audit = run_audit(list(rejudged), list(human_graded(items)), judge_model="judge-model")
    assert "re-judgment unanimity" not in _skip_names(audit)


def test_mixed_judge_ids_rejected():
    items = graded_items(10, seed=0)
    judge = [*graded_judge(items, judge_id="a", seed=1), *graded_judge(items, judge_id="b", seed=2)]
    with pytest.raises(ValueError, match="one audit reads one judge"):
        run_audit(judge, list(human_graded(items)))


def test_to_dict_is_valid_json_with_null_for_nan():
    items = graded_items(20, seed=0)
    judge = graded_judge(items, seed=1)
    audit = run_audit(list(judge), list(human_graded(items)), judge_model="judge-model")
    payload = json.loads(json.dumps(audit.to_dict()))
    assert payload["judge_id"] == "judge"
    assert payload["flags"] == [flag.name for flag in audit.flags]
    assert {skip["name"] for skip in payload["skipped"]} == _skip_names(audit)
    for section in ("agreement", "calibration", "probes"):
        for result in payload[section]:
            assert result["value"] is None or isinstance(result["value"], float)
