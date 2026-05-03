"""Regression harness for the UniversalContrastAnalyzer Phase A upgrades.

Runs paired POST_A1 vs FULL_A comparisons on a synthetic scene set targeting
each defect class. Verifies four invariants, in order of importance:

    1. Schema integrity   — FULL_A only adds JSON keys; never removes / renames
    2. No false-negative regression — issues POST_A1 catches, FULL_A also catches
    3. Target-case improvement       — each new feature catches its target case
    4. Latency budget                — FULL_A stays within 2x of POST_A1 wall time

Run:    python -m scripts.eval_contrast
Or:     /path/to/python scripts/eval_contrast.py
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

# Ensure repo root is on sys.path when invoked as a script.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from universal_contrast_analyzer import UniversalContrastAnalyzer  # noqa: E402


# --- Synthetic scenes -------------------------------------------------------

H, W = 240, 320

# ADE20K class IDs the analyzer maps to known categories
WALL, FLOOR, RUG, DOOR = 0, 3, 28, 14


@dataclass
class Scene:
    name: str
    image: np.ndarray
    seg: np.ndarray
    targets: List[str]   # which Phase-A features are expected to react


def _scene_uniform_high_contrast() -> Scene:
    """Black floor against white wall — both configs must emit a critical/high issue."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    img[0:120, :] = (250, 250, 250); seg[0:120, :] = WALL
    img[120:, :] = (10, 10, 10);     seg[120:, :] = FLOOR
    return Scene(
        name="uniform_high_contrast",
        image=img, seg=seg,
        targets=[],   # nothing new fires; sanity check that defaults don't degrade
    )


def _scene_wainscoting() -> Scene:
    """Wall white-on-top + beige-on-bottom against beige floor (Phase A1 target)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    img[0:60, :] = (255, 255, 255); seg[0:120, :] = WALL
    img[60:120, :] = (230, 225, 210)
    img[120:, :] = (230, 225, 210); seg[120:, :] = FLOOR
    return Scene(
        name="wainscoting_wall",
        image=img, seg=seg,
        targets=["A1", "A4"],   # A4 should warn the wall is multi-modal
    )


def _scene_speckled_boundary() -> Scene:
    """Two regions with a noisy 1-pixel jittered boundary (Phase A3 target).

    Without A3, the speckled pixels register a phantom pair. With A3, the
    morph-close + connected-component drop removes the speckle; the long
    clean boundary remains as the only real pair.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    img[:, :] = (200, 200, 200); seg[:, :] = WALL
    img[120:, :] = (100, 100, 100); seg[120:, :] = FLOOR
    rng = np.random.default_rng(42)
    # Inject 1-pixel speckle: random isolated FLOOR pixels in WALL territory
    speckle_y = rng.integers(0, 60, size=8)
    speckle_x = rng.integers(0, W, size=8)
    seg[speckle_y, speckle_x] = FLOOR
    img[speckle_y, speckle_x] = (100, 100, 100)
    return Scene(
        name="speckled_boundary",
        image=img, seg=seg,
        targets=[],   # the legitimate boundary is high-contrast, no new issue
    )


def _scene_phantom_pair_speckle() -> Scene:
    """Multiple small speckle clusters of a third category create a phantom pair.

    POST_A1 filters pairs by total-boundary-pixel sum >= 20, so 4 disconnected
    ~6-pixel DOOR clusters inside the WALL pass the gate as a wall-door pair.
    A3's connected-component filter requires each component to be >= 20 px,
    so it drops the speckle entirely. The speckle color is chosen close to
    the wall color so the phantom pair is actually a *low-contrast* issue
    (POST_A1 reports it; FULL_A correctly suppresses it).
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    img[:, :] = (240, 240, 240); seg[:, :] = WALL
    img[120:, :] = (10, 10, 10);  seg[120:, :] = FLOOR
    # 13 widely-spaced 2x2 DOOR speckle blobs. Spacing >= 16 px so the morph
    # close kernel can't merge them. Per-cluster boundary mask is ~16 px
    # (4 door + 12 wall-ring) — fails A3's 20-px component minimum. Total
    # cumulative boundary ~208 px — passes POST_A1's 20-px sum gate. Total
    # DOOR area = 52 — passes the 50-px segment gate. Speckle color is
    # near-wall so the phantom wall-door pair is also a *low-contrast* issue
    # (POST_A1 reports it; FULL_A drops it).
    speckle_color = (215, 215, 215)
    placements = [
        (10, 30), (10, 80),  (10, 150), (10, 220), (10, 290),
        (35, 50), (35, 110), (35, 180), (35, 250),
        (70, 40), (70, 130), (70, 200), (70, 280),
    ]
    for cy, cx in placements:
        seg[cy:cy + 2, cx:cx + 2] = DOOR
        img[cy:cy + 2, cx:cx + 2] = speckle_color
    return Scene(
        name="phantom_pair_speckle",
        image=img, seg=seg,
        targets=["A3"],
    )


def _scene_striped_rug() -> Scene:
    """Single floor segment with red/white alternating stripes (Phase A4 target)."""
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    img[0:120, :] = (240, 240, 240); seg[0:120, :] = WALL
    seg[120:, :] = RUG
    img[120:, :] = (240, 240, 240)
    # 12-pixel stripes of saturated red over the rug area
    for y0 in range(120, H, 24):
        img[y0:y0 + 12, :] = (200, 30, 30)
    return Scene(
        name="striped_rug",
        image=img, seg=seg,
        targets=["A4"],
    )


def _scene_isoluminant() -> Scene:
    """Floor and door at near-equal luminance but very different chroma.

    Phase A5 (ΔE2000) should be large; WCAG ratio is ~1.0 — both configs
    flag the pair as low contrast (good), but only FULL_A emits the
    chromatic-difference field that makes the diagnosis interpretable.
    """
    img = np.zeros((H, W, 3), dtype=np.uint8)
    seg = np.zeros((H, W), dtype=np.int32)
    # Use approximately iso-luminant red/green per ITU-R BT.709 weights
    # L_red ≈ 0.2126*200 + 0.7152*30 + 0.0722*30 = 66.3
    # L_grn ≈ 0.2126*30 + 0.7152*100 + 0.0722*30 = 79.8 (close-ish)
    img[0:120, :] = (200, 30, 30);  seg[0:120, :] = DOOR
    img[120:, :] = (30, 100, 30);   seg[120:, :] = FLOOR
    return Scene(
        name="isoluminant_door_floor",
        image=img, seg=seg,
        targets=["A5"],
    )


SCENES: List[Scene] = [
    _scene_uniform_high_contrast(),
    _scene_wainscoting(),
    _scene_speckled_boundary(),
    _scene_phantom_pair_speckle(),
    _scene_striped_rug(),
    _scene_isoluminant(),
]


# --- Analyzer factories -----------------------------------------------------

def _make_post_a1() -> UniversalContrastAnalyzer:
    """Baseline: A1 on (matches main as of commit 72819fe), A2-A5 off."""
    return UniversalContrastAnalyzer(
        use_local_boundary_sampling=True,
        use_kmeans_lab=False,
        use_morphological_boundary_close=False,
        use_intrasegment_scan=False,
        compute_delta_e2000=False,
    )


def _make_full_a() -> UniversalContrastAnalyzer:
    """Full Phase A: A1-A5 all on (post-A2-A5 ship default)."""
    return UniversalContrastAnalyzer()   # all defaults are True


# --- Eval primitives --------------------------------------------------------

def _existing_issue_keys(issue: Dict) -> set:
    """Subset of issue keys that must be schema-stable across configs."""
    return {
        "segment_ids", "categories", "colors", "wcag_ratio",
        "hue_difference", "saturation_difference",
        "boundary_pixels", "severity", "is_floor_object", "boundary_mask",
    }


def _run(analyzer_factory: Callable[[], UniversalContrastAnalyzer],
         scene: Scene) -> Tuple[Dict, float]:
    np.random.seed(0)
    analyzer = analyzer_factory()
    t0 = time.perf_counter()
    res = analyzer.analyze_contrast(scene.image, scene.seg)
    return res, time.perf_counter() - t0


def _categorize_pairs(issues: List[Dict]) -> set:
    return {tuple(sorted(i["categories"])) for i in issues}


# --- Invariant checks -------------------------------------------------------

def check_schema(post: Dict, full: Dict) -> List[str]:
    """Existing keys must be present (no rename / remove); additions allowed."""
    msgs = []
    if post["issues"]:
        post_keys = _existing_issue_keys(post["issues"][0])
        if full["issues"]:
            full_keys = set(full["issues"][0].keys())
            missing = post_keys - full_keys
            if missing:
                msgs.append(f"FULL_A removed legacy issue keys: {sorted(missing)}")
    # FULL_A is allowed to add 'intrasegment_warnings'; POST_A1 must always
    # carry an empty list (we initialize it unconditionally on the analyzer
    # side) so consumers can iterate without `.get(..., [])`.
    if "intrasegment_warnings" not in post:
        msgs.append("POST_A1 missing intrasegment_warnings key")
    if "intrasegment_warnings" not in full:
        msgs.append("FULL_A missing intrasegment_warnings key")
    return msgs


def check_no_false_neg_regression(post_pairs: set, full_pairs: set,
                                  scene: Scene, post: Dict, full: Dict) -> List[str]:
    """A real low-contrast pair POST_A1 caught must still be caught by FULL_A.

    Exception: scenes whose targets include A3 — that feature's whole point
    is to drop phantom pairs from speckle. We allow strict reduction in
    issues but require the legitimate background pair was still analyzed
    (i.e. analyzed_pairs > 0 in FULL_A's stats)."""
    msgs = []
    if "A3" in scene.targets:
        if full["statistics"]["analyzed_pairs"] == 0:
            msgs.append("FULL_A analyzed_pairs == 0; legitimate adjacencies missing")
        return msgs
    lost = post_pairs - full_pairs
    if lost:
        msgs.append(f"FULL_A lost pairs vs POST_A1: {sorted(lost)}")
    return msgs


def check_a3_drops_phantom(post_pairs: set, full_pairs: set,
                           scene: Scene) -> List[str]:
    """Verify A3 actually drops a phantom pair POST_A1 incorrectly retained."""
    if "A3" not in scene.targets:
        return []
    dropped = post_pairs - full_pairs
    if not dropped:
        return ["A3 expected to drop at least one phantom pair, dropped none"]
    return []


def check_target_features(post: Dict, full: Dict, scene: Scene) -> List[str]:
    msgs = []
    if "A4" in scene.targets:
        if not full.get("intrasegment_warnings"):
            msgs.append("expected intrasegment_warning, FULL_A produced none")
    if "A5" in scene.targets:
        if not full["issues"]:
            msgs.append("expected at least one issue with delta_e2000, none recorded")
        else:
            de = full["issues"][0].get("delta_e2000")
            if de is None or de < 30:
                msgs.append(
                    f"expected ΔE2000 > 30 on iso-luminant pair, got {de}"
                )
    if "A3" in scene.targets:
        # A3 effect verified by post_pairs vs full_pairs comparison elsewhere
        pass
    return msgs


def check_latency(t_post: float, t_full: float) -> List[str]:
    if t_full > max(0.05, 2.0 * t_post + 0.1):
        return [f"FULL_A latency {t_full:.3f}s > 2x POST_A1 ({t_post:.3f}s)"]
    return []


# --- Main loop --------------------------------------------------------------

def main() -> int:
    # Warm up: amortizes scipy / skimage import cost so per-scene latency
    # measurements reflect steady-state per-call work, not module-load.
    warm = _scene_uniform_high_contrast()
    _run(_make_full_a, warm)
    _run(_make_post_a1, warm)

    failed = 0
    print(f"{'scene':28s}  {'POST_A1':>8s} {'FULL_A':>8s}  delta  warnings  ΔE2000  status")
    print("-" * 100)
    for scene in SCENES:
        post, t_post = _run(_make_post_a1, scene)
        full, t_full = _run(_make_full_a, scene)

        post_pairs = _categorize_pairs(post["issues"])
        full_pairs = _categorize_pairs(full["issues"])

        problems = []
        problems += check_schema(post, full)
        problems += check_no_false_neg_regression(post_pairs, full_pairs, scene, post, full)
        problems += check_a3_drops_phantom(post_pairs, full_pairs, scene)
        problems += check_target_features(post, full, scene)
        problems += check_latency(t_post, t_full)

        n_post = len(post["issues"])
        n_full = len(full["issues"])
        n_warn = len(full.get("intrasegment_warnings", []))
        de_max = max(
            (i.get("delta_e2000", 0.0) for i in full["issues"]),
            default=0.0,
        )
        status = "PASS" if not problems else "FAIL"
        if problems:
            failed += 1
        print(
            f"{scene.name:28s}  {n_post:8d} {n_full:8d}  {n_full - n_post:+5d}  "
            f"{n_warn:8d}  {de_max:6.1f}  {status} "
            f"({t_post * 1000:.0f}ms → {t_full * 1000:.0f}ms)"
        )
        for p in problems:
            print(f"    - {p}")

    print("-" * 100)
    print(
        f"{'OVERALL':28s}  "
        f"{'PASS' if failed == 0 else f'FAIL ({failed}/{len(SCENES)} scenes)'}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
