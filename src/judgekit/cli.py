"""Command line interface: audit a judge, or run the built-in demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from judgekit import demo
from judgekit.audit import run_audit
from judgekit.io import load_verdicts
from judgekit.report import position_figure, reliability_figure, render_markdown


def _cmd_report(args: argparse.Namespace) -> int:
    judge = load_verdicts(args.judge)
    human = load_verdicts(args.human) if args.human else []
    audit = run_audit(judge, human, judge_model=args.judge_model, seed=args.seed)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    preamble = ""
    if position_figure([audit], figures_dir / "position.png"):
        preamble += "![position preference](figures/position.png)\n"
    if reliability_figure([audit], figures_dir / "reliability.png"):
        preamble += "![reliability](figures/reliability.png)\n"
    (out_dir / "report.md").write_text(
        render_markdown([audit], f"judge audit: {audit.judge_id}", preamble=preamble),
        encoding="utf-8",
    )

    for flag in audit.flags:
        print(f"FLAG {flag.name}: {flag.detail}")
    print(f"report written to {out_dir / 'report.md'} ({len(audit.flags)} flags)")
    if args.fail_on_flags and audit.flags:
        return 2
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    audits = demo.write_demo(Path(args.out))
    total = sum(len(audit.flags) for audit in audits)
    print(f"demo report written to {Path(args.out) / 'report.md'} ({total} flags)")
    return 0


def _cmd_inject_readme(args: argparse.Namespace) -> int:
    demo.inject_readme(Path(args.readme), Path(args.results))
    print(f"updated {args.readme} from {args.results}/summary.md")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="judgekit", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    report = commands.add_parser("report", help="audit a judge's verdicts")
    report.add_argument("judge", help="judge verdicts, one JSON object per line")
    report.add_argument("--human", help="reference verdicts aligned on item_id")
    report.add_argument("--out", default="audit", help="output directory")
    report.add_argument("--judge-model", help="model name for the self-preference probe")
    report.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    report.add_argument(
        "--fail-on-flags", action="store_true", help="exit 2 when any probe triggers"
    )
    report.set_defaults(func=_cmd_report)

    demo_cmd = commands.add_parser("demo", help="audit six synthetic judges")
    demo_cmd.add_argument("--out", default="results", help="output directory")
    demo_cmd.set_defaults(func=_cmd_demo)

    injector = commands.add_parser("inject-readme", help="refresh the README demo table")
    injector.add_argument("readme")
    injector.add_argument("--results", default="results", help="directory with summary.md")
    injector.set_defaults(func=_cmd_inject_readme)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
