"""Backward-compatible facade — imports from neuronest.xai package."""

from neuronest.xai import XAIAnalyzer
from neuronest.xai.renderer import render_notebook_html, render_defect_notebook_html
from neuronest.xai.viz import METHOD_INFO
from neuronest.xai.hooks import (
    AttentionCaptureHook,
    HiddenStateCaptureHook,
    ActivationGradientHook,
    eager_attention as _eager_attention,
)

__all__ = [
    "XAIAnalyzer",
    "render_notebook_html",
    "render_defect_notebook_html",
    "METHOD_INFO",
    "AttentionCaptureHook",
    "HiddenStateCaptureHook",
    "ActivationGradientHook",
]
