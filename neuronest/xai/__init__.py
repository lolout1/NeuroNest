from .base import XAIBase
from .attention import AttentionMixin
from .output import OutputMixin
from .gradient import GradientMixin
from .defect import DefectMixin
from .orchestrator import OrchestratorMixin
from .report import ReportMixin
from .cache import CacheMixin
from .renderer import render_notebook_html, render_defect_notebook_html


class XAIAnalyzer(
    XAIBase,
    AttentionMixin,
    OutputMixin,
    GradientMixin,
    DefectMixin,
    OrchestratorMixin,
    ReportMixin,
    CacheMixin,
):
    """Assembled XAI analyzer with all 7 methods, defect analysis, caching, and reporting."""

    pass


__all__ = [
    "XAIAnalyzer",
    "render_notebook_html",
    "render_defect_notebook_html",
]
