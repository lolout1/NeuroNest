"""Defect 3 — Vertical placement analyzer for signs and clocks (geometry primitives)."""

from .geometry import (
    PinholeCamera,
    back_project,
    fit_floor_plane,
    point_to_plane_distance,
)

__all__ = [
    "PinholeCamera",
    "back_project",
    "fit_floor_plane",
    "point_to_plane_distance",
]
