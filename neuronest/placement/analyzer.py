"""End-to-end vertical placement analyzer for sign and clock instances.

Public class `VerticalPlacementAnalyzer` mirrors the contract used by
`ImprovedBlackspotDetector` and `UniversalContrastAnalyzer` so the orchestrator
in `neuronest/pipeline.py` can fan it out via the same `ThreadPoolExecutor`.

End-to-end flow on a single image:
    1. Extract sign/clock connected components from the segmentation mask.
       Short-circuit with a "skipped" result if there are none.
    2. Compute or accept a depth map from `MonocularMetricDepth`.
    3. Build pinhole intrinsics from image shape + assumed FOV.
    4. Back-project floor-prior pixels and RANSAC-fit a floor plane.
    5. Self-calibrate scale via door (or ceiling fallback).
    6. For each sign/clock instance:
       - Robust-sample depth at the centroid (median + std of patch).
       - Back-project centroid to 3D camera coordinates.
       - Compute perpendicular distance to floor plane → inches.
       - Classify severity per ADA thresholds.
    7. Render annotated visualization.
    8. Return a Dict matching the schema in
       `docs/DEFECT_3_PLACEMENT_PLAN.md` and `API_CHANGES.md`.
"""

from __future__ import annotations

import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..config import (
    ADA_PLACEMENT_HIGH_IN,
    ADA_PLACEMENT_LOW_IN,
    ADA_PLACEMENT_TOLERANCE_IN,
    CLOCK_CLASS_IDS,
    PLACEMENT_MIN_FLOOR_PIXELS,
    PLACEMENT_MIN_INSTANCE_PIXELS,
    PLACEMENT_SEVERITY_CRITICAL_IN,
    PLACEMENT_SEVERITY_HIGH_IN,
    PLACEMENT_TARGET_CLASS_IDS,
    SIGN_CLASS_IDS,
)
from ..models.depth import MonocularMetricDepth
from ..utils import prepare_display_image
from .calibration import calibrate_scale, measure_height_above_floor_in
from .geometry import (
    PinholeCamera,
    back_project,
    back_project_mask,
    extract_instances,
    fit_floor_plane,
    meters_to_inches,
    sample_depth_at,
)
from .viz import render_placement_visualization

logger = logging.getLogger(__name__)


def _classify_severity(
    height_in: float,
    low: float,
    high: float,
    tolerance: float,
) -> Tuple[str, Optional[str]]:
    """Map height to (severity_label, violation_type).

    severity:       "ok" | "medium" | "high" | "critical"
    violation_type: None | "below" | "above"
    """
    if low <= height_in <= high:
        return "ok", None
    if height_in < low:
        deviation = low - height_in
        violation = "below"
    else:
        deviation = height_in - high
        violation = "above"

    # Tolerance band: minor over-/under-shoots get nudged one tier softer
    effective_dev = max(0.0, deviation - tolerance * 0.5)

    if effective_dev > PLACEMENT_SEVERITY_CRITICAL_IN:
        return "critical", violation
    if effective_dev > PLACEMENT_SEVERITY_HIGH_IN:
        return "high", violation
    return "medium", violation


def _class_label(class_id: int) -> str:
    if class_id in SIGN_CLASS_IDS:
        return "sign"
    if class_id in CLOCK_CLASS_IDS:
        return "clock"
    return f"class_{class_id}"


def _empty_result(image_rgb: np.ndarray, reason: str) -> Dict:
    """Return a well-formed 'skipped' result without crashing the pipeline."""
    return {
        "visualization": prepare_display_image(image_rgb),
        "detections": [],
        "num_detections": 0,
        "num_violations": 0,
        "depth_map": None,
        "floor_plane": None,
        "scale_factor": 1.0,
        "calibration_source": None,
        "skipped": True,
        "reason": reason,
    }


class VerticalPlacementAnalyzer:
    """Orchestrator for Defect 3 (sign / clock vertical placement)."""

    def __init__(
        self,
        depth_model: MonocularMetricDepth,
        low_in: float = ADA_PLACEMENT_LOW_IN,
        high_in: float = ADA_PLACEMENT_HIGH_IN,
        tolerance_in: float = ADA_PLACEMENT_TOLERANCE_IN,
        sign_class_ids: Tuple[int, ...] = SIGN_CLASS_IDS,
        clock_class_ids: Tuple[int, ...] = CLOCK_CLASS_IDS,
        enable_self_calibration: bool = True,
        min_instance_pixels: int = PLACEMENT_MIN_INSTANCE_PIXELS,
    ):
        self.depth_model = depth_model
        self.low_in = low_in
        self.high_in = high_in
        self.tolerance_in = tolerance_in
        self.sign_class_ids = tuple(sign_class_ids)
        self.clock_class_ids = tuple(clock_class_ids)
        self.target_class_ids = self.sign_class_ids + self.clock_class_ids
        self.enable_self_calibration = enable_self_calibration
        self.min_instance_pixels = int(min_instance_pixels)

    def initialize(self) -> bool:
        """Lazy-init the depth model. Returns True iff depth model loaded."""
        if not self.depth_model.is_loaded:
            ok = self.depth_model.initialize()
            if not ok:
                logger.error("Placement analyzer: depth model failed to initialize")
                return False
        logger.info(
            f"Placement analyzer initialized "
            f"(thresholds: {self.low_in}\"-{self.high_in}\", "
            f"tolerance: \u00b1{self.tolerance_in}\")"
        )
        return True

    # --- Main entry point ----------------------------------------------------

    def analyze_placement(
        self,
        image_rgb: np.ndarray,
        seg_mask: np.ndarray,
        floor_prior: Optional[np.ndarray] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> Dict:
        """Analyze sign and clock placements in a single image.

        Args:
            image_rgb:    H x W x 3 uint8 image.
            seg_mask:     H x W uint8 ADE20K class map.
            floor_prior:  optional H x W bool floor mask (faster than re-deriving
                          from seg_mask). When None, computed from `seg_mask`.
            depth_map:    optional H x W float32 depth in meters. When None, the
                          analyzer runs `depth_model.estimate_depth` itself.

        Returns:
            Dict matching the schema documented in `API_CHANGES.md` and
            `docs/DEFECT_3_PLACEMENT_PLAN.md`. Always contains at least the keys
            in `_empty_result`; never raises.
        """
        t0 = time.perf_counter()
        try:
            return self._analyze_inner(image_rgb, seg_mask, floor_prior, depth_map, t0)
        except Exception as e:
            logger.error(f"Placement analysis crashed: {e}", exc_info=True)
            return _empty_result(image_rgb, f"analyzer error: {e}")

    def _analyze_inner(
        self,
        image_rgb: np.ndarray,
        seg_mask: np.ndarray,
        floor_prior: Optional[np.ndarray],
        depth_map: Optional[np.ndarray],
        t0: float,
    ) -> Dict:
        h, w = image_rgb.shape[:2]
        if seg_mask.shape[:2] != (h, w):
            raise ValueError(
                f"seg_mask shape {seg_mask.shape} != image shape ({h}, {w})"
            )

        # --- Step 1: short-circuit if no target classes present -------------
        instances = extract_instances(
            seg_mask, self.target_class_ids, self.min_instance_pixels
        )
        if not instances:
            logger.info("Placement: no sign/clock instances above min area; skipping")
            return _empty_result(image_rgb, "no sign or clock detected")

        # --- Step 2: floor mask ---------------------------------------------
        if floor_prior is None:
            from ..config import FLOOR_CLASS_IDS
            floor_mask = np.isin(seg_mask, list(FLOOR_CLASS_IDS))
        else:
            floor_mask = floor_prior.astype(bool, copy=False)
            if floor_mask.shape != (h, w):
                import cv2
                floor_mask = cv2.resize(
                    floor_mask.astype(np.uint8), (w, h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)
        if int(floor_mask.sum()) < PLACEMENT_MIN_FLOOR_PIXELS:
            logger.info(
                f"Placement: floor pixels {int(floor_mask.sum())} < "
                f"{PLACEMENT_MIN_FLOOR_PIXELS}; cannot fit plane"
            )
            return _empty_result(image_rgb, "floor not visible enough")

        # --- Step 3: depth + intrinsics -------------------------------------
        if depth_map is None:
            depth_map = self.depth_model.estimate_depth(image_rgb)
        if depth_map.shape[:2] != (h, w):
            raise ValueError(
                f"depth_map shape {depth_map.shape} != image shape ({h}, {w})"
            )
        camera = PinholeCamera.from_image_shape(h, w)

        # --- Step 4: floor plane via RANSAC ---------------------------------
        floor_pts = back_project_mask(floor_mask, depth_map, camera)
        floor_plane = fit_floor_plane(floor_pts)
        if floor_plane is None:
            logger.info("Placement: RANSAC floor fit failed")
            return _empty_result(image_rgb, "could not fit floor plane")
        logger.info(
            f"Placement: floor plane "
            f"normal={floor_plane.normal.round(3).tolist()}, "
            f"inliers={floor_plane.inlier_count}/{floor_plane.total_points}, "
            f"rms={floor_plane.rms_residual_m * 100:.1f}cm"
        )

        # --- Step 5: self-calibrate scale -----------------------------------
        if self.enable_self_calibration:
            calibration = calibrate_scale(seg_mask, depth_map, floor_plane, camera)
        else:
            from .calibration import Calibration
            calibration = Calibration(
                scale_factor=1.0, source="prior",
                reference_truth_in=0.0, reference_measured_in=0.0,
                confidence=0.5,
            )
        logger.info(
            f"Placement: scale calibration source={calibration.source} "
            f"factor={calibration.scale_factor:.3f}"
        )

        # --- Step 6: per-instance height computation ------------------------
        detections: List[Dict] = []
        for idx, inst in enumerate(instances, start=1):
            cx, cy = inst.centroid_px
            depth_at_centroid, depth_std = sample_depth_at(
                depth_map, cx, cy, window=5
            )
            if depth_at_centroid <= 0:
                logger.warning(
                    f"Placement: instance {idx} ({_class_label(inst.class_id)}) "
                    "has non-positive depth at centroid; skipping"
                )
                continue

            centroid_3d = back_project(cx, cy, depth_at_centroid, camera)
            height_in = measure_height_above_floor_in(
                centroid_3d, floor_plane, calibration
            )

            # Propagate depth uncertainty: dh/dz \u2248 1 (height ~linear in depth
            # when sign is roughly perpendicular-projected onto the plane normal),
            # then apply scale factor and convert to inches.
            height_uncertainty_in = (
                meters_to_inches(depth_std) * calibration.scale_factor
            )

            severity, violation = _classify_severity(
                height_in, self.low_in, self.high_in, self.tolerance_in
            )

            # Confidence: combines calibration confidence + relative uncertainty
            rel_uncertainty = (
                height_uncertainty_in / max(height_in, 1.0) if height_in > 0 else 1.0
            )
            confidence = float(np.clip(
                calibration.confidence * (1.0 - min(rel_uncertainty, 0.5)),
                0.0, 1.0,
            ))

            detections.append({
                "id": idx,
                "class": _class_label(inst.class_id),
                "class_id": int(inst.class_id),
                "centroid_px": [int(cx), int(cy)],
                "bbox": list(map(int, inst.bbox)),
                "area_pixels": int(inst.area_pixels),
                "height_in": round(float(height_in), 1),
                "height_in_uncertainty": round(float(height_uncertainty_in), 1),
                "severity": severity,
                "violation_type": violation,
                "calibration_source": calibration.source,
                "confidence": round(confidence, 3),
            })

        # --- Step 7: visualization ------------------------------------------
        visualization = render_placement_visualization(
            image_rgb=image_rgb,
            detections=detections,
            low_in=self.low_in,
            high_in=self.high_in,
            calibration_source=calibration.source,
            scale_factor=calibration.scale_factor,
        )
        visualization_display = prepare_display_image(visualization)

        elapsed = time.perf_counter() - t0
        num_violations = sum(1 for d in detections if d["severity"] != "ok")
        logger.info(
            f"Placement: {len(detections)} detections, "
            f"{num_violations} violations, {elapsed:.1f}s"
        )

        return {
            "visualization": visualization_display,
            "detections": detections,
            "num_detections": len(detections),
            "num_violations": num_violations,
            "depth_map": depth_map,
            "floor_plane": {
                "normal": floor_plane.normal.tolist(),
                "offset": float(floor_plane.offset),
                "inlier_count": int(floor_plane.inlier_count),
                "total_points": int(floor_plane.total_points),
                "rms_residual_m": round(float(floor_plane.rms_residual_m), 4),
            },
            "scale_factor": round(float(calibration.scale_factor), 3),
            "calibration_source": calibration.source,
            "ada_recommended_range_in": [self.low_in, self.high_in],
            "skipped": False,
        }
