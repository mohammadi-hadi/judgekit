"""Render audits as a markdown report card with figures.

Tables are deterministic text — same verdicts, same bytes, on any platform —
which is what lets CI regenerate the demo report and fail on drift.  Figures
are for humans and are not byte-stable across matplotlib versions, so they are
committed for display and never diffed.
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from judgekit.audit import AuditReport
from judgekit.result import ProbeResult

ACCENT = "#2563eb"
MUTED = "#9ca3af"


def _fmt(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.3f}"


def _verdict(result: ProbeResult) -> str:
    if result.triggered is True:
        return "FLAG"
    if result.triggered is False:
        return "ok"
    return "-"


def probe_by_name(audit: AuditReport, name: str) -> ProbeResult | None:
    for result in audit.results:
        if result.name == name:
            return result
    return None


def render_markdown(audits: list[AuditReport], title: str, preamble: str = "") -> str:
    lines = [f"# {title}", ""]
    if preamble:
        lines += [preamble.rstrip(), ""]
    for audit in audits:
        counts = ", ".join(
            f"{n} {kind}"
            for kind, n in (
                ("graded", audit.n_graded),
                ("binary", audit.n_binary),
                ("pairwise", audit.n_pairwise),
            )
            if n
        )
        lines += [f"## {audit.judge_id} ({counts})", ""]
        if audit.flags:
            lines += ["Flags:", ""]
            lines += [f"- **{flag.name}**: {flag.detail}" for flag in audit.flags]
        else:
            lines += ["No probe left its innocent band."]
        lines += [
            "",
            "| check | value | 95% CI | verdict | reading |",
            "|---|---|---|---|---|",
        ]
        for result in audit.results:
            low, high = result.ci
            interval = "n/a" if math.isnan(low) else f"[{_fmt(low)}, {_fmt(high)}]"
            lines.append(
                f"| {result.name} | {_fmt(result.value)} | {interval} "
                f"| {_verdict(result)} | {result.detail} |"
            )
        lines.append("")
        if audit.skipped:
            lines += ["Not run — a skip describes the log file, not the judge:", ""]
            lines += [f"- **{skip.name}** needs {skip.needs}" for skip in audit.skipped]
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


# ---------------------------------------------------------------- figures ---


def position_figure(audits: list[AuditReport], path: Path) -> bool:
    entries = []
    for audit in audits:
        probe = probe_by_name(audit, "position preference")
        if probe is not None:
            entries.append((audit.judge_id, probe))
    if not entries:
        return False
    figure, axis = plt.subplots(figsize=(6.4, 3.2))
    names = [name for name, _ in entries]
    values = [probe.value for _, probe in entries]
    lows = [probe.value - probe.ci[0] for _, probe in entries]
    highs = [probe.ci[1] - probe.value for _, probe in entries]
    axis.axhspan(0.45, 0.55, color=MUTED, alpha=0.25, label="innocent band")
    axis.axhline(0.5, color=MUTED, linewidth=1)
    axis.errorbar(names, values, yerr=[lows, highs], fmt="o", color=ACCENT, capsize=4)
    axis.set_ylabel("share choosing presented-first")
    axis.set_ylim(0.3, 1.0)
    axis.legend(loc="upper left", frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def reliability_figure(audits: list[AuditReport], path: Path) -> bool:
    entries = [(audit.judge_id, audit.reliability) for audit in audits if audit.reliability]
    if not entries:
        return False
    figure, axis = plt.subplots(figsize=(4.8, 4.4))
    axis.plot([0, 1], [0, 1], color=MUTED, linewidth=1, label="perfect")
    colors = [ACCENT, "#dc2626", "#059669", "#d97706"]
    for (name, curve), color in zip(entries, colors, strict=False):
        confidences = [confidence for confidence, _, _ in curve]
        frequencies = [frequency for _, frequency, _ in curve]
        axis.plot(confidences, frequencies, "o-", color=color, label=name)
    axis.set_xlabel("claimed probability")
    axis.set_ylabel("observed frequency")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


def score_distribution_figure(audit: AuditReport, path: Path) -> bool:
    if audit.graded_scores is None:
        return False
    judge_scores, human_scores = audit.graded_scores
    scale = sorted(set(human_scores) | set(judge_scores))
    judge_share = [judge_scores.count(s) / len(judge_scores) for s in scale]
    human_share = [human_scores.count(s) / len(human_scores) for s in scale]
    width = 0.38
    figure, axis = plt.subplots(figsize=(5.6, 3.2))
    positions = range(len(scale))
    axis.bar([p - width / 2 for p in positions], human_share, width, color=MUTED, label="human")
    axis.bar(
        [p + width / 2 for p in positions], judge_share, width, color=ACCENT, label=audit.judge_id
    )
    axis.set_xticks(list(positions), [f"{s:g}" for s in scale])
    axis.set_xlabel("score")
    axis.set_ylabel("share of verdicts")
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return True


# -------------------------------------------------------------- injection ---


def inject(text: str, key: str, payload: str) -> str:
    """Replace the block between ``<!-- judgekit:key -->`` markers.

    Marker pairs inside fenced code blocks are left alone, so documentation
    showing the marker syntax does not get real tables injected into it.
    """
    start = f"<!-- judgekit:{key} -->"
    end = f"<!-- /judgekit:{key} -->"
    fenced = [m.span() for m in re.finditer(r"^```.*?^```", text, re.MULTILINE | re.DOTALL)]
    pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.DOTALL)
    for match in pattern.finditer(text):
        if not any(a <= match.start() < b for a, b in fenced):
            replacement = f"{start}\n{payload.rstrip()}\n{end}"
            return text[: match.start()] + replacement + text[match.end() :]
    raise ValueError(f"no marker pair for key {key!r} outside code fences")
