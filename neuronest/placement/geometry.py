"""3D geometry primitives for monocular metric placement analysis.

This module provides the bridge from a 2D pixel + per-pixel depth in meters to
real-world 3D coordinates and the floor plane that everything is measured
against. All units are SI (meters, radians) until the analyzer converts to
inches at the very end.

The pinhole-camera model used here assumes:
- Optical center at the image center (`cx`, `cy` = `w / 2`, `h / 2`)
- Square pixels (`fx == fy`)
- No lens distortion correction

These approximations are reasonable for modern phone cameras at default zoom
and have a negligible effect compared to the dominant error source (depth model
uncertainty). When EXIF / mobile IMU data becomes available in a future
revision, `PinholeCamera.from_exif(...)` is the natural extension point.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..config import (
    DEFAULT_HORIZONTAL_FOV_DEG,
    PLACEMENT_MIN_FLOOR_PIXELS,
)

logger = logging.getLogger(__name__)


# --- Camera model -----------------------------------------------------------


@dataclass(frozen=True)
class PinholeCamera:
    """Simple pinhole camera intrinsics.

    Coordinate convention (right-handed, OpenCV-style):
        +x = right, +y = down, +z = forward (into the scene)
    """
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_image_shape(
        cls,
        height: int,
        width: int,
        horizontal_fov_deg: float = DEFAULT_HORIZONTAL_FOV_DEG,
    ) -> "PinholeCamera":
        """Construct intrinsics from image shape and an assumed horizontal FOV.

        With the standard pinhole equation, fx = (W / 2) / tan(HFOV / 2).
        Square pixels => fy = fx. Optical center at image center.
        """
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image shape: ({height}, {width})")
        if not (10.0 <= horizontal_fov_deg <= 170.0):
            logger.warning(
                f"Horizontal FOV {horizontal_fov_deg}° is outside plausible range; "
                f"falling back to {DEFAULT_HORIZONTAL_FOV_DEG}°"
            )
            horizontal_fov_deg = DEFAULT_HORIZONTAL_FOV_DEG
        half_fov = math.radians(horizontal_fov_deg) / 2.0
        fx = (width / 2.0) / math.tan(half_fov)
        return cls(
            fx=fx, fy=fx,           # square pixels
            cx=width / 2.0, cy=height / 2.0,
            width=width, height=height,
        )


# --- Back-projection --------------------------------------------------------


def back_project(
    u: np.ndarray | float,
    v: np.ndarray | float,
    z: np.ndarray | float,
    camera: PinholeCamera,
) -> np.ndarray:
    """Lift 2D pixel(s) + depth to 3D camera coordinates.

    Standard inverse pinhole projection:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        P = (X, Y, Z)

    Args:
        u, v: pixel coordinates (scalar or array of same shape).
        z:    depth in meters at those pixels (same shape).
        camera: PinholeCamera intrinsics.

    Returns:
        For scalar inputs: shape (3,) array.
        For array inputs:  shape (..., 3) array stacked along last axis.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    v_arr = np.asarray(v, dtype=np.float64)
    z_arr = np.asarray(z, dtype=np.float64)

    x = (u_arr - camera.cx) * z_arr / camera.fx
    y = (v_arr - camera.cy) * z_arr / camera.fy
    return np.stack([x, y, z_arr], axis=-1)


def back_project_mask(
    mask: np.ndarray,
    depth_map: np.ndarray,
    camera: PinholeCamera,
    max_points: int = 8000,
) -> np.ndarray:
    """Back-project all pixels under a boolean mask to a (N, 3) point cloud.

    Subsamples uniformly when more than `max_points` pixels are inside the mask
    so RANSAC stays bounded. Returns an empty (0, 3) array if the mask is empty
    or all depths are non-positive.
    """
    if mask.shape != depth_map.shape:
        raise ValueError(
            f"mask shape {mask.shape} != depth_map shape {depth_map.shape}"
        )
    ys, xs = np.where(mask)
    if ys.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    # Subsample if needed
    if ys.size > max_points:
        idx = np.random.default_rng(seed=0).choice(ys.size, max_points, replace=False)
        ys, xs = ys[idx], xs[idx]

    zs = depth_map[ys, xs]
    # Reject non-positive AND non-finite depths; inf/NaN would corrupt the
    # SVD step in fit_floor_plane.
    valid = (zs > 0) & np.isfinite(zs)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float64)
    ys, xs, zs = ys[valid], xs[valid], zs[valid]

    return back_project(xs.astype(np.float64), ys.astype(np.float64), zs, camera)


# --- Plane fitting ----------------------------------------------------------


@dataclass(frozen=True)
class FloorPlane:
    """A floor plane in 3D camera coordinates.

    Plane equation: n . p + d = 0, where n is unit-normal pointing UP (toward
    the camera in OpenCV convention, i.e. negative-y component).
    """
    normal: np.ndarray         # shape (3,), unit vector
    offset: float              # signed offset such that n . p + d = 0 on the plane
    inlier_count: int          # how many input points were on the plane
    total_points: int          # how many input points were considered
    rms_residual_m: float      # root-mean-square residual of inliers

    @property
    def inlier_fraction(self) -> float:
        return self.inlier_count / max(1, self.total_points)


def fit_floor_plane(
    points_3d: np.ndarray,
    inlier_threshold_m: float = 0.05,
    max_iterations: int = 200,
    rng_seed: int = 0,
) -> Optional[FloorPlane]:
    """Robust RANSAC plane fit.

    Args:
        points_3d:           (N, 3) point cloud in camera coordinates.
        inlier_threshold_m:  point-to-plane distance below which a point is an inlier.
        max_iterations:      number of RANSAC iterations.
        rng_seed:            for deterministic results.

    Returns:
        FloorPlane on success; None if too few points or no good fit found.

    The returned plane normal is oriented to point "upward" in image-space terms
    (negative y component in OpenCV convention). This makes
    `point_to_plane_distance` return a positive value for points above the floor.
    """
    n = points_3d.shape[0]
    if n < PLACEMENT_MIN_FLOOR_PIXELS // 4:
        logger.info(
            f"fit_floor_plane: only {n} points — below minimum, skipping"
        )
        return None

    rng = np.random.default_rng(rng_seed)
    best_inliers: Optional[np.ndarray] = None
    best_normal: Optional[np.ndarray] = None
    best_offset: float = 0.0

    for _ in range(max_iterations):
        # Sample 3 points; reject degenerate (near-collinear) triples
        idx = rng.choice(n, 3, replace=False)
        p1, p2, p3 = points_3d[idx]
        v1 = p2 - p1
        v2 = p3 - p1
        normal = np.cross(v1, v2)
        norm_mag = np.linalg.norm(normal)
        if norm_mag < 1e-9:
            continue
        normal = normal / norm_mag
        offset = -float(np.dot(normal, p1))

        # Distance of all points to this candidate plane
        residuals = np.abs(points_3d @ normal + offset)
        inlier_mask = residuals < inlier_threshold_m
        n_inliers = int(inlier_mask.sum())

        if best_inliers is None or n_inliers > int(best_inliers.sum()):
            best_inliers = inlier_mask
            best_normal = normal
            best_offset = offset

    if best_inliers is None or best_normal is None or int(best_inliers.sum()) < 50:
        logger.info("fit_floor_plane: no consensus plane found")
        return None

    # Refine using least-squares on the inliers
    inlier_pts = points_3d[best_inliers]
    centroid = inlier_pts.mean(axis=0)
    centered = inlier_pts - centroid
    # SVD: smallest singular vector is the plane normal
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    refined_normal = vh[-1]
    refined_normal = refined_normal / (np.linalg.norm(refined_normal) + 1e-12)
    refined_offset = -float(np.dot(refined_normal, centroid))

    # Orient normal so it points "up" (negative y in OpenCV convention)
    if refined_normal[1] > 0:
        refined_normal = -refined_normal
        refined_offset = -refined_offset

    residuals = inlier_pts @ refined_normal + refined_offset
    rms = float(np.sqrt(np.mean(residuals ** 2)))

    return FloorPlane(
        normal=refined_normal,
        offset=refined_offset,
        inlier_count=int(best_inliers.sum()),
        total_points=n,
        rms_residual_m=rms,
    )


def point_to_plane_distance(point_3d: np.ndarray, plane: FloorPlane) -> float:
    """Signed perpendicular distance from a 3D point to the floor plane.

    Positive when the point is above the floor (in the direction of `plane.normal`,
    which is oriented upward). For analyzer use, take `abs(...)` to get height.
    """
    return float(np.dot(plane.normal, point_3d) + plane.offset)


# --- Connected-component instance extraction --------------------------------


@dataclass(frozen=True)
class Instance2D:
    """A single connected-component instance of a target class in image space."""
    class_id: int
    centroid_px: Tuple[int, int]   # (cx, cy)
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    area_pixels: int
    mask: np.ndarray               # bool, full-image-sized


def extract_instances(
    seg_mask: np.ndarray,
    target_class_ids: Tuple[int, ...],
    min_pixels: int,
) -> list[Instance2D]:
    """Extract connected-component instances of any target class from a seg mask.

    Uses OpenCV's `connectedComponentsWithStats` per class, then filters by area.
    Centroid is the geometric pixel centroid (same as cv2 stats).

    Args:
        seg_mask:        H x W uint8 with ADE20K class IDs.
        target_class_ids: tuple of class IDs to extract.
        min_pixels:       skip components smaller than this.

    Returns:
        list of Instance2D, one per surviving connected component.
    """
    import cv2

    instances: list[Instance2D] = []
    for class_id in target_class_ids:
        class_mask = (seg_mask == class_id).astype(np.uint8)
        if class_mask.sum() < min_pixels:
            continue

        n_components, labels, stats, centroids = cv2.connectedComponentsWithStats(
            class_mask, connectivity=8
        )
        # component 0 is background; iterate 1..n
        for cid in range(1, n_components):
            area = int(stats[cid, cv2.CC_STAT_AREA])
            if area < min_pixels:
                continue
            x = int(stats[cid, cv2.CC_STAT_LEFT])
            y = int(stats[cid, cv2.CC_STAT_TOP])
            w = int(stats[cid, cv2.CC_STAT_WIDTH])
            h = int(stats[cid, cv2.CC_STAT_HEIGHT])
            cx = int(round(centroids[cid, 0]))
            cy = int(round(centroids[cid, 1]))
            instance_mask = (labels == cid)
            instances.append(Instance2D(
                class_id=class_id,
                centroid_px=(cx, cy),
                bbox=(x, y, x + w, y + h),
                area_pixels=area,
                mask=instance_mask,
            ))
    return instances


def sample_depth_at(
    depth_map: np.ndarray,
    cx: int,
    cy: int,
    window: int = 5,
) -> Tuple[float, float]:
    """Robust depth lookup at a pixel: median + std over a `window x window` patch.

    Returns (median_depth_m, std_m). Falls back to single-pixel value when the
    patch overlaps the image edge.
    """
    h, w = depth_map.shape[:2]
    half = window // 2
    y0 = max(0, cy - half); y1 = min(h, cy + half + 1)
    x0 = max(0, cx - half); x1 = min(w, cx + half + 1)
    patch = depth_map[y0:y1, x0:x1]
    if patch.size == 0:
        return float(depth_map[cy, cx]), 0.0
    return float(np.median(patch)), float(np.std(patch))


def meters_to_inches(meters: float) -> float:
    """Convert SI meters to imperial inches (1 m = 39.3700787 in)."""
    return meters * 39.37007874015748
