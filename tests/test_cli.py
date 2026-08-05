from judgekit.cli import main
from judgekit.io import dump_verdicts
from judgekit.synthetic import human_pairwise, pair_items, pairwise_judge


def _write_pairwise(tmp_path, position):
    items = pair_items(120, seed=0)
    judge = pairwise_judge(items, seed=1, position=position)
    human = human_pairwise(items)
    judge_path = tmp_path / "judge.jsonl"
    human_path = tmp_path / "human.jsonl"
    dump_verdicts(judge, judge_path)
    dump_verdicts(human, human_path)
    return judge_path, human_path


def test_report_command_writes_report(tmp_path, capsys):
    judge_path, human_path = _write_pairwise(tmp_path, position=0.0)
    out = tmp_path / "audit"
    code = main(["report", str(judge_path), "--human", str(human_path), "--out", str(out)])
    assert code == 0
    text = (out / "report.md").read_text()
    assert "position preference" in text
    assert (out / "figures" / "position.png").exists()


def test_report_fail_on_flags_exits_nonzero(tmp_path, capsys):
    judge_path, human_path = _write_pairwise(tmp_path, position=0.6)
    out = tmp_path / "audit"
    code = main(
        [
            "report",
            str(judge_path),
            "--human",
            str(human_path),
            "--out",
            str(out),
            "--fail-on-flags",
        ]
    )
    assert code == 2
    assert "FLAG position preference" in capsys.readouterr().out


def test_demo_and_inject_readme(tmp_path, capsys):
    results = tmp_path / "results"
    assert main(["demo", "--out", str(results)]) == 0
    assert (results / "report.md").exists()
    assert (results / "figures" / "position.png").exists()

    readme = tmp_path / "README.md"
    readme.write_text("intro\n<!-- judgekit:demo -->\nstale\n<!-- /judgekit:demo -->\n")
    assert main(["inject-readme", str(readme), "--results", str(results)]) == 0
    updated = readme.read_text()
    assert "| judge | implanted defect |" in updated
    assert "stale" not in updated
