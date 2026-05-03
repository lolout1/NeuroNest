"""Defect 3 — Vertical placement analyzer for signs and clocks.

Public surface:
    - VerticalPlacementAnalyzer  (orchestrator; mirrors BlackspotDetector contract)
    - PlacementResult, PlacementDetection  (typed result containers)

Implementation modules:
    - geometry:    pinhole intrinsics, back-projection, RANSAC plane fit
    - calibration: door / ceiling-based scale self-calibration
    - viz:         annotated visualization
    - analyzer:    end-to-end orchestrator
"""

from .analyzer import VerticalPlacementAnalyzer
from .geometry import (
    PinholeCamera,
    back_project,
    fit_floor_plane,
    point_to_plane_distance,
)

__all__ = [
    "VerticalPlacementAnalyzer",
    "PinholeCamera",
    "back_project",
    "fit_floor_plane",
    "point_to_plane_distance",
]
