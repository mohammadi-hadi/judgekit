"""judgekit: audit an LLM judge before you trust it."""

from judgekit.audit import AuditReport, run_audit
from judgekit.io import dump_verdicts, load_verdicts
from judgekit.result import ProbeResult
from judgekit.schema import BinaryVerdict, GradedVerdict, PairwiseVerdict

__version__ = "0.1.0"

__all__ = [
    "AuditReport",
    "BinaryVerdict",
    "GradedVerdict",
    "PairwiseVerdict",
    "ProbeResult",
    "__version__",
    "dump_verdicts",
    "load_verdicts",
    "run_audit",
]
