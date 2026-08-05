import pytest

from judgekit.audit import AuditReport, run_audit
from judgekit.demo import build_audits, summary_table
from judgekit.report import inject, probe_by_name, render_markdown
from judgekit.result import ProbeResult
from judgekit.synthetic import graded_items, graded_judge, human_graded


def _probe(name, value, innocent, ci):
    return ProbeResult(name=name, value=value, ci=ci, innocent=innocent, n=10, detail="d")


def test_render_marks_flags_and_clean_judges():
    flagged = AuditReport(
        judge_id="biased",
        n_graded=10,
        n_binary=0,
        n_pairwise=0,
        probes=[_probe("verbosity bias", 0.5, (-0.1, 0.1), (0.4, 0.6))],
    )
    clean = AuditReport(
        judge_id="clean",
        n_graded=10,
        n_binary=0,
        n_pairwise=0,
        probes=[_probe("verbosity bias", 0.02, (-0.1, 0.1), (-0.05, 0.08))],
    )
    text = render_markdown([flagged, clean], "t")
    assert "| verbosity bias | 0.500 | [0.400, 0.600] | FLAG |" in text
    assert "No probe left its innocent band." in text


def test_inject_replaces_between_markers():
    text = "before\n<!-- judgekit:demo -->\nold\n<!-- /judgekit:demo -->\nafter"
    updated = inject(text, "demo", "| new |")
    assert "old" not in updated
    assert "| new |" in updated
    assert updated.startswith("before\n<!-- judgekit:demo -->")


def test_inject_skips_markers_inside_code_fences():
    text = (
        "```\n<!-- judgekit:demo -->\nexample\n<!-- /judgekit:demo -->\n```\n"
        "<!-- judgekit:demo -->\nold\n<!-- /judgekit:demo -->"
    )
    updated = inject(text, "demo", "new")
    assert "example" in updated
    assert "old" not in updated


def test_inject_requires_markers():
    with pytest.raises(ValueError, match="no marker pair"):
        inject("nothing here", "demo", "x")


def test_mixed_scales_rejected():
    items = graded_items(20, seed=0)
    judge = graded_judge(items, seed=1)
    human = human_graded(items)
    judge[0] = judge[0].model_copy(update={"scale_max": 10.0, "score": 3.0})
    with pytest.raises(ValueError, match="mixed score scales"):
        run_audit(list(judge), list(human))


def test_demo_summary_is_deterministic_and_flags_the_diagonal():
    audits = build_audits()
    table = summary_table(audits)
    by_id = {audit.judge_id: audit for audit in audits}

    assert by_id["clean"].flags == []
    assert {flag.name for flag in by_id["first-picker"].flags} == {"position preference"}
    assert {flag.name for flag in by_id["self-server"].flags} == {"self-preference"}
    assert {flag.name for flag in by_id["middler"].flags} == {"central tendency"}
    assert {flag.name for flag in by_id["overconfident"].flags} == {
        "expected calibration error"
    }
    assert "verbosity bias" in {flag.name for flag in by_id["length-lover"].flags}

    again = summary_table(build_audits())
    assert table == again

    position = probe_by_name(by_id["first-picker"], "position preference")
    assert position.value == pytest.approx(0.675, abs=0.05)
