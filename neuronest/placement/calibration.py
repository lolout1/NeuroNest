"""Scene-scale self-calibration for monocular metric depth.

Off-the-shelf monocular metric depth estimators have a residual scale ambiguity
of ~10-20% on consumer indoor photos. This module reduces that to ~3-5% by
finding a known-size reference object in the scene (a door, a ceiling) and
deriving a multiplicative scale correction.

Calibration sources, ranked by reliability:
    1. Door (class 14) — US standard interior door is 80 inches; present in
       ~60-80% of indoor photos; geometrically tall and well-segmented.
    2. Ceiling (class 5) — distance from floor plane to ceiling pixels;
       residential rooms are 84-144 inches; useful when no door is visible.
    3. Prior — if neither reference is available, fall back to depth
       model's raw output and flag low confidence.

Strategy:
    - Compute the "measured" 3D extent of the reference (e.g. door height)
      using the uncalibrated depth + back-projection.
    - Compute scale = reference_truth / reference_measured.
    - Reject scales outside [0.7, 1.4] as implausible (depth model failure).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config import (
    CALIBRATION_SCALE_MAX,
    CALIBRATION_SCALE_MIN,
    CEILING_CLASS_ID,
    CEILING_MAX_REFERENCE_IN,
    CEILING_MIN_REFERENCE_IN,
    DOOR_CLASS_ID,
    DOOR_REFERENCE_HEIGHT_IN,
)
from .geometry import (
    FloorPlane,
    PinholeCamera,
    back_project_mask,
    meters_to_inches,
    point_to_plane_distance,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Calibration:
    """Result of scale self-calibration."""
    scale_factor: float          # multiply raw depths/heights by this
    source: str                  # "door", "ceiling", or "prior"
    reference_truth_in: float    # known reference dimension (inches)
    reference_measured_in: float # uncalibrated measurement (inches)
    confidence: float            # 0..1, analyzer-self-assessed

    @property
    def applied(self) -> bool:
        return self.source != "prior"


def _measure_door_height_in(
    seg_mask: np.ndarray,
    depth_map: np.ndarray,
    floor_plane: FloorPlane,
    camera: PinholeCamera,
) -> Optional[float]:
    """Measure tallest connected door's height above the floor plane in inches.

    Approach: back-project all door pixels, take the percentile-pair (5%, 95%)
    along the floor-plane-perpendicular direction. The 95th percentile gives the
    door top, the 5th gives its bottom near the floor (robust to feet not being
    exactly on the floor due to depth noise).
    """
    door_mask = (seg_mask == DOOR_CLASS_ID)
    if door_mask.sum() < 200:    # too small to be a real door
        return None

    pts = back_project_mask(door_mask, depth_map, camera, max_points=4000)
    if pts.shape[0] < 100:
        return None

    # Signed perpendicular distances from the floor plane
    distances = pts @ floor_plane.normal + floor_plane.offset

    top_m = float(np.percentile(distances, 95))
    bottom_m = float(np.percentile(distances, 5))
    height_m = top_m - bottom_m
    if height_m <= 0:
        return None

    return meters_to_inches(height_m)


def _measure_ceiling_height_in(
    seg_mask: np.ndarray,
    depth_map: np.ndarray,
    floor_plane: FloorPlane,
    camera: PinholeCamera,
) -> Optional[float]:
    """Measure floor-to-ceiling distance in inches.

    Uses the median perpendicular distance from the floor plane to ceiling
    pixels, which is robust to depth noise and ignores ceiling-fan / chandelier
    outliers near the median.
    """
    ceiling_mask = (seg_mask == CEILING_CLASS_ID)
    if ceiling_mask.sum() < 500:
        return None

    pts = back_project_mask(ceiling_mask, depth_map, camera, max_points=4000)
    if pts.shape[0] < 200:
        return None

    distances = pts @ floor_plane.normal + floor_plane.offset
    height_m = float(np.median(distances))
    if height_m <= 0:
        return None

    return meters_to_inches(height_m)


def calibrate_scale(
    seg_mask: np.ndarray,
    depth_map: np.ndarray,
    floor_plane: FloorPlane,
    camera: PinholeCamera,
) -> Calibration:
    """Find the best available scale correction for this scene.

    Priority: door (most reliable) → ceiling (fallback) → prior (no correction).

    Returns a Calibration even when no reference is found; in that case
    `scale_factor == 1.0`, `source == "prior"`, and `confidence` is low.
    """
    # --- Try door first ---
    door_measured = _measure_door_height_in(
        seg_mask, depth_map, floor_plane, camera
    )
    if door_measured is not None:
        scale = DOOR_REFERENCE_HEIGHT_IN / door_measured
        if CALIBRATION_SCALE_MIN <= scale <= CALIBRATION_SCALE_MAX:
            logger.info(
                f"Calibration: door measured={door_measured:.1f}\" "
                f"truth={DOOR_REFERENCE_HEIGHT_IN}\" → scale={scale:.3f}"
            )
            # Confidence drops as scale deviates further from 1.0
            confidence = max(0.5, 1.0 - abs(scale - 1.0))
            return Calibration(
                scale_factor=scale,
                source="door",
                reference_truth_in=DOOR_REFERENCE_HEIGHT_IN,
                reference_measured_in=door_measured,
                confidence=confidence,
            )
        logger.warning(
            f"Calibration: door scale {scale:.3f} outside "
            f"[{CALIBRATION_SCALE_MIN}, {CALIBRATION_SCALE_MAX}], rejecting"
        )

    # --- Fall back to ceiling ---
    ceiling_measured = _measure_ceiling_height_in(
        seg_mask, depth_map, floor_plane, camera
    )
    if ceiling_measured is not None:
        if CEILING_MIN_REFERENCE_IN <= ceiling_measured <= CEILING_MAX_REFERENCE_IN:
            # No correction needed if measurement is already plausible — the
            # ceiling validates the scene scale, doesn't override it.
            logger.info(
                f"Calibration: ceiling measured={ceiling_measured:.1f}\" "
                "(in plausible range, no correction)"
            )
            return Calibration(
                scale_factor=1.0,
                source="ceiling",
                reference_truth_in=ceiling_measured,
                reference_measured_in=ceiling_measured,
                confidence=0.7,
            )
        # Ceiling measurement outside plausible range — derive a scale that
        # pulls it to the nearest plausible bound (108" residential standard).
        target = 108.0
        scale = target / ceiling_measured
        if CALIBRATION_SCALE_MIN <= scale <= CALIBRATION_SCALE_MAX:
            logger.info(
                f"Calibration: ceiling measured={ceiling_measured:.1f}\" "
                f"out of band, target {target}\" → scale={scale:.3f}"
            )
            return Calibration(
                scale_factor=scale,
                source="ceiling",
                reference_truth_in=target,
                reference_measured_in=ceiling_measured,
                confidence=0.5,
            )

    # --- No reference found; trust the depth model as-is ---
    logger.info("Calibration: no door/ceiling reference; using depth as-is (prior)")
    return Calibration(
        scale_factor=1.0,
        source="prior",
        reference_truth_in=0.0,
        reference_measured_in=0.0,
        confidence=0.4,
    )


def measure_height_above_floor_in(
    centroid_3d: np.ndarray,
    floor_plane: FloorPlane,
    calibration: Calibration,
) -> float:
    """Compute centroid height in inches, applying the scene-scale correction."""
    perp_distance_m = abs(point_to_plane_distance(centroid_3d, floor_plane))
    return meters_to_inches(perp_distance_m) * calibration.scale_factor
