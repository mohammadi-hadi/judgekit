"""judgekit: bias probes for LLM judges, each with a bootstrap confidence interval."""

from judgekit.audit import AuditReport, run_audit
from judgekit.io import dump_verdicts, load_verdicts
from judgekit.result import ProbeResult, SkippedProbe
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict

__version__ = "0.2.0"

__all__ = [
    "AuditReport",
    "BinaryVerdict",
    "GradedVerdict",
    "PairwiseVerdict",
    "ProbeResult",
    "SkippedProbe",
    "__version__",
    "dump_verdicts",
    "load_verdicts",
    "run_audit",
]
