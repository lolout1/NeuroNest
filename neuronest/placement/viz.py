"""Annotated visualization for the vertical placement analyzer.

Renders detections as bounding boxes with severity-colored badges showing each
sign / clock's measured height in inches and ADA pass/fail status. The visual
language matches the contrast analyzer:

    critical  →  crimson    (220, 20, 60)    deviation > 6 inches
    high      →  dark orange (255, 140, 0)   deviation 3-6 inches
    medium    →  gold       (255, 215, 0)    deviation 0-3 inches
    ok        →  green      (40, 200, 80)    inside [low, high]

A bottom banner summarizes counts. A right-side legend identifies severity
colors and shows the configured ADA range and calibration source.
"""

from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np

SEVERITY_COLORS = {
    "critical": (220, 20, 60),
    "high":     (255, 140, 0),
    "medium":   (255, 215, 0),
    "ok":       (40, 200, 80),
}

LABEL_BG_ALPHA = 0.78


def _draw_outlined_text(
    image: np.ndarray,
    text: str,
    pos: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
) -> None:
    """Black outline + colored fill so labels stay readable on any background."""
    cv2.putText(image, text, pos, font, scale, (0, 0, 0),
                thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, pos, font, scale, color,
                thickness, cv2.LINE_AA)


def _draw_detection_box(
    image: np.ndarray,
    detection: Dict,
    low_in: float,
    high_in: float,
) -> None:
    x1, y1, x2, y2 = detection["bbox"]
    severity = detection["severity"]
    color = SEVERITY_COLORS.get(severity, (255, 255, 255))

    # Bounding box (slightly thicker for non-ok detections)
    box_thickness = 3 if severity != "ok" else 2
    cv2.rectangle(image, (x1, y1), (x2, y2), color, box_thickness, cv2.LINE_AA)

    # Centroid marker
    cx, cy = detection["centroid_px"]
    cv2.drawMarker(image, (cx, cy), color, markerType=cv2.MARKER_CROSS,
                   markerSize=14, thickness=2, line_type=cv2.LINE_AA)

    # Build label text
    height_in = detection["height_in"]
    uncertainty = detection["height_in_uncertainty"]
    cls = detection["class"].upper()
    if severity == "ok":
        status = "OK"
    elif detection["violation_type"] == "below":
        status = "TOO LOW"
    else:
        status = "TOO HIGH"
    label = f"{cls} #{detection['id']}  {height_in:.0f}\u00b1{uncertainty:.0f}\"  {status}"

    # Label background (semi-transparent rectangle above the bbox)
    (tw, th), baseline = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
    )
    pad = 6
    label_x = x1
    label_y = max(y1 - th - pad * 2, 4)
    bg_x1 = label_x
    bg_y1 = label_y
    bg_x2 = min(image.shape[1] - 1, label_x + tw + pad * 2)
    bg_y2 = label_y + th + pad * 2

    overlay = image.copy()
    cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), color, -1)
    cv2.addWeighted(overlay, LABEL_BG_ALPHA, image, 1 - LABEL_BG_ALPHA, 0, image)
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), (30, 30, 30), 1, cv2.LINE_AA)

    # Determine readable foreground color (dark on light bg, white on dark bg)
    fg = (30, 30, 30) if (color[0] + color[1] + color[2]) > 380 else (255, 255, 255)
    cv2.putText(
        image, label, (bg_x1 + pad, bg_y1 + th + pad),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, fg, 1, cv2.LINE_AA,
    )


def _draw_summary_banner(
    canvas_w: int,
    detections: List[Dict],
    low_in: float,
    high_in: float,
    calibration_source: str,
    scale_factor: float,
) -> np.ndarray:
    banner_h = 80
    banner = np.full((banner_h, canvas_w, 3), 35, dtype=np.uint8)

    counts = {"critical": 0, "high": 0, "medium": 0, "ok": 0}
    for d in detections:
        counts[d["severity"]] = counts.get(d["severity"], 0) + 1

    metrics = [
        ("CRITICAL", counts["critical"], SEVERITY_COLORS["critical"]),
        ("HIGH",     counts["high"],     SEVERITY_COLORS["high"]),
        ("MEDIUM",   counts["medium"],   SEVERITY_COLORS["medium"]),
        ("OK",       counts["ok"],       SEVERITY_COLORS["ok"]),
    ]
    spacing = canvas_w // (len(metrics) + 1)
    for i, (label, value, color) in enumerate(metrics, start=1):
        cx = spacing * i
        val_str = str(value)
        (tw, _), _ = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
        cv2.putText(banner, val_str, (cx - tw // 2, 48),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
        (tw2, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(banner, label, (cx - tw2 // 2, 68),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    cv2.putText(
        banner, "SIGN & CLOCK PLACEMENT", (15, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA,
    )
    cv2.putText(
        banner,
        f"ADA range: {low_in:.0f}-{high_in:.0f}\"  |  "
        f"Calibration: {calibration_source}  |  Scale: {scale_factor:.2f}",
        (15, 44),
        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (160, 160, 160), 1, cv2.LINE_AA,
    )
    return banner


def _draw_legend(image: np.ndarray, low_in: float, high_in: float) -> None:
    h, w = image.shape[:2]
    lw, lh = 240, 168
    lx = w - lw - 15
    ly = h - lh - 15

    overlay = image.copy()
    cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
    cv2.rectangle(image, (lx, ly), (lx + lw, ly + lh), (60, 60, 60), 2)

    cv2.putText(image, "Placement Severity", (lx + 10, ly + 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
    cv2.line(image, (lx + 10, ly + 28), (lx + lw - 10, ly + 28),
             (180, 180, 180), 1)

    levels = [
        ("CRITICAL", SEVERITY_COLORS["critical"], "> 6\" off ADA"),
        ("HIGH",     SEVERITY_COLORS["high"],     "3-6\" off"),
        ("MEDIUM",   SEVERITY_COLORS["medium"],   "0-3\" off"),
        ("OK",       SEVERITY_COLORS["ok"],       f"in {low_in:.0f}-{high_in:.0f}\""),
    ]
    y = ly + 50
    for label, color, desc in levels:
        cv2.rectangle(image, (lx + 12, y - 10), (lx + 30, y + 5), color, -1)
        cv2.rectangle(image, (lx + 12, y - 10), (lx + 30, y + 5), (60, 60, 60), 1)
        cv2.putText(image, label, (lx + 38, y + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.putText(image, desc, (lx + 12, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1, cv2.LINE_AA)
        y += 32

    cv2.putText(image, "ADA / dementia-design",
                (lx + 10, ly + lh - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (80, 80, 80), 1, cv2.LINE_AA)


def render_placement_visualization(
    image_rgb: np.ndarray,
    detections: List[Dict],
    low_in: float,
    high_in: float,
    calibration_source: str,
    scale_factor: float,
) -> np.ndarray:
    """Compose the full annotated placement visualization.

    Layout:
        [ summary banner          ]
        [ original image          ]
        [ with bbox + label per   ]
        [ detection + legend      ]
    """
    img_h, img_w = image_rgb.shape[:2]
    banner = _draw_summary_banner(
        img_w, detections, low_in, high_in, calibration_source, scale_factor,
    )

    annotated = image_rgb.copy()
    # Sort: ok last so violations render on top if bboxes overlap
    sorted_detections = sorted(
        detections,
        key=lambda d: {"critical": 0, "high": 1, "medium": 2, "ok": 3}.get(
            d["severity"], 4
        ),
        reverse=True,
    )
    for det in sorted_detections:
        _draw_detection_box(annotated, det, low_in, high_in)

    if detections:
        _draw_legend(annotated, low_in, high_in)

    canvas = np.full((banner.shape[0] + img_h, img_w, 3), 42, dtype=np.uint8)
    canvas[0:banner.shape[0], 0:img_w] = banner
    canvas[banner.shape[0]:, 0:img_w] = annotated
    return canvas
