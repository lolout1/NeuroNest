"""
Universal Contrast Analyzer for detecting low contrast between ALL adjacent objects.
Optimized for Alzheimer's/dementia care environments.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from skimage.segmentation import find_boundaries
import colorsys

logger = logging.getLogger(__name__)


class UniversalContrastAnalyzer:
    """
    Analyzes contrast between ALL adjacent objects in a room.
    Ensures proper visibility for elderly individuals with Alzheimer's or dementia.
    """
    
    def __init__(
        self,
        wcag_threshold: float = 4.5,
        use_local_boundary_sampling: bool = True,
        boundary_band_pixels: int = 8,
        min_boundary_band_pixels: int = 50,
        use_kmeans_lab: bool = True,
        kmeans_k: int = 3,
        use_morphological_boundary_close: bool = True,
        boundary_close_kernel: int = 3,
        min_boundary_component_pixels: int = 20,
        use_intrasegment_scan: bool = True,
        intrasegment_delta_e_threshold: float = 35.0,
        intrasegment_min_segment_pixels: int = 5000,
        intrasegment_min_minority_share: float = 0.15,
        compute_delta_e2000: bool = True,
        compute_apca_lc: bool = True,
        compute_weber_contrast: bool = True,
    ):
        self.wcag_threshold = wcag_threshold

        # Local-boundary sampling (Phase A1).
        # Dementia-friendly contrast depends on the colors that meet AT the
        # boundary — not the segment-wide average, which gets dominated by
        # wainscoting / shadow zones / patterned surfaces and produces useless
        # midtones. When enabled, per-pair colors are sampled from a thin
        # band around each pair's boundary instead of the full segment mask.
        self.use_local_boundary_sampling = bool(use_local_boundary_sampling)
        self.boundary_band_pixels = int(boundary_band_pixels)
        self.min_boundary_band_pixels = int(min_boundary_band_pixels)
        if self.boundary_band_pixels < 1:
            raise ValueError(
                f"boundary_band_pixels must be >= 1, got {self.boundary_band_pixels}"
            )
        if self.min_boundary_band_pixels < 1:
            raise ValueError(
                f"min_boundary_band_pixels must be >= 1, got {self.min_boundary_band_pixels}"
            )

        # K-means in CIELAB (Phase A2).
        # Replaces IQR median for color extraction. CIELAB is perceptually
        # uniform, so cluster centroids correspond to colors a person would
        # actually call "the color of that region" rather than an arithmetic
        # mid-tone of two distinct populations (lit vs shadow).
        self.use_kmeans_lab = bool(use_kmeans_lab)
        self.kmeans_k = int(kmeans_k)
        if self.kmeans_k < 2:
            raise ValueError(f"kmeans_k must be >= 2, got {self.kmeans_k}")

        # Boundary morphological cleanup (Phase A3).
        # 1-pixel segmentation jitter inflates the count of "low-contrast pairs"
        # on noisy seg maps. A small morphological close + connected-component
        # filter throws out tiny speckle edges before any pair is scored.
        self.use_morphological_boundary_close = bool(use_morphological_boundary_close)
        self.boundary_close_kernel = int(boundary_close_kernel)
        self.min_boundary_component_pixels = int(min_boundary_component_pixels)
        if self.boundary_close_kernel < 1:
            raise ValueError(
                f"boundary_close_kernel must be >= 1, got {self.boundary_close_kernel}"
            )

        # Intra-segment pattern scan (Phase A4).
        # Striped rugs and patterned wallpaper are a real fall-hazard class the
        # pair-only analyzer silently misses. Within each large segment, run
        # k=2 k-means in LAB; flag a warning when the two cluster centroids are
        # ΔE > threshold AND the minority cluster covers >= min_share of pixels.
        self.use_intrasegment_scan = bool(use_intrasegment_scan)
        self.intrasegment_delta_e_threshold = float(intrasegment_delta_e_threshold)
        self.intrasegment_min_segment_pixels = int(intrasegment_min_segment_pixels)
        self.intrasegment_min_minority_share = float(intrasegment_min_minority_share)

        # CIEDE2000 chromatic difference (Phase A5).
        # Strictly additive: emitted alongside `wcag_ratio`. WCAG looks at
        # luminance only, so red/green isoluminant pairs slip through; ΔE2000
        # captures the chromatic difference WCAG ignores.
        self.compute_delta_e2000 = bool(compute_delta_e2000)

        # APCA Lc (Phase B1) — strictly additive, emitted alongside wcag_ratio.
        # Polarity-aware: distinguishes light-on-dark from dark-on-light, which
        # WCAG's symmetric ratio cannot. Severity classification UNCHANGED
        # (Phase B4 composite severity intentionally deferred — would require
        # ground-truth calibration we don't have).
        self.compute_apca_lc = bool(compute_apca_lc)

        # Weber contrast (Phase B3) — strictly additive figure-on-ground metric.
        # Smaller-area segment is treated as foreground; sign encodes whether
        # foreground is brighter (+) or darker (−) than background. Severity
        # classification UNCHANGED.
        self.compute_weber_contrast = bool(compute_weber_contrast)

        # ADE20K semantic class mappings for indoor care environments.
        # Each ID verified against ade20k_classes.py (0-indexed, 150 classes).
        # Classes not listed here default to 'unknown' and get standard 3:1 checks.
        self.semantic_classes = {
            # Floor / walking surfaces (keep in sync with config.FLOOR_CLASS_IDS)
            'floor': [3, 28],  # 3: floor, 28: rug

            # Vertical room surfaces
            'wall': [0],  # 0: wall

            # Overhead surfaces
            'ceiling': [5],  # 5: ceiling

            # Large furnishings on floors / against walls
            'furniture': [
                7,   # bed
                10,  # cabinet
                15,  # table
                19,  # chair
                23,  # sofa
                24,  # shelf
                30,  # armchair
                31,  # seat
                33,  # desk
                35,  # wardrobe, closet
                44,  # chest of drawers
                45,  # counter
                55,  # case, display case
                56,  # pool table
                62,  # bookcase
                64,  # coffee table
                69,  # bench
                73,  # kitchen island
                75,  # swivel chair
                77,  # bar
                97,  # ottoman
                99,  # buffet, sideboard
                110, # stool
                117, # cradle
            ],

            # Doors and openings
            'door': [14, 58],  # 14: door, 58: screen door

            # Windows
            'window': [8],  # 8: window

            # Stairs, steps, and associated safety elements
            'stairs': [53, 59, 95, 96, 121],
            # 53: stairs, 59: stairway, 95: bannister, 96: escalator, 121: step

            # Smaller items — trip hazards, tabletop objects
            'objects': [
                17,  # plant
                36,  # lamp
                39,  # cushion
                41,  # box
                57,  # pillow
                67,  # book
                74,  # computer
                81,  # towel
                89,  # tv
                92,  # clothes
                98,  # bottle
                108, # plaything, toy
                112, # basket
                115, # bag
                119, # ball
                120, # food
                125, # pot
                131, # blanket
                135, # vase
                137, # tray
                138, # trash can
                139, # fan
                142, # plate
                147, # glass, drinking glass
            ],

            # Built-in kitchen / bathroom fixtures
            'fixtures': [
                37,  # tub, bathtub
                47,  # sink
                49,  # fireplace
                50,  # refrigerator
                65,  # toilet
                70,  # countertop
                71,  # stove
                107, # washer
                118, # oven
                124, # microwave
                129, # dishwasher
                133, # hood, exhaust hood
                145, # shower
                146, # radiator
            ],

            # Wall hangings, window treatments, lighting, visual elements
            'decorative': [
                18,  # curtain
                22,  # painting, picture
                27,  # mirror
                40,  # base, pedestal, stand
                42,  # column, pillar
                63,  # blind, screen
                66,  # flower
                82,  # light
                85,  # chandelier
                100, # poster
                130, # screen
                132, # sculpture
                134, # sconce
                143, # monitor
                144, # bulletin board
                148, # clock
            ],
        }
        
        # Create reverse mapping for quick lookup
        self.class_to_category = {}
        for category, class_ids in self.semantic_classes.items():
            for class_id in class_ids:
                self.class_to_category[class_id] = category
    
    @staticmethod
    def _rgb_to_color_info(rgb) -> Dict:
        """Convert an RGB array to a dict with rgb list, hex string, and color name."""
        r, g, b = int(rgb[0]), int(rgb[1]), int(rgb[2])
        hex_code = f'#{r:02x}{g:02x}{b:02x}'

        h, s, v = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
        h, s, v = h * 360, s * 100, v * 100

        # Achromatic
        if s < 12:
            if v < 15:
                name = "black"
            elif v < 35:
                name = "dark gray"
            elif v < 65:
                name = "gray"
            elif v < 85:
                name = "light gray"
            else:
                name = "white"
        # Warm neutrals (brown / tan / beige family)
        # Includes desaturated warm tones AND dark warm tones (which look brown)
        elif 15 <= h < 50 and (s < 50 or v < 65):
            if v < 35:
                name = "dark brown"
            elif v < 65:
                name = "brown"
            elif v < 80:
                name = "tan"
            else:
                name = "beige"
        else:
            # Chromatic hue names
            for boundary, hue_name in [
                (15, "red"), (40, "orange"), (65, "yellow"),
                (160, "green"), (195, "teal"), (250, "blue"),
                (290, "purple"), (345, "pink"), (361, "red"),
            ]:
                if h < boundary:
                    break
            if v < 30:
                name = f"dark {hue_name}"
            elif v > 85 and s < 30:
                name = f"pale {hue_name}"
            elif v > 80:
                name = f"light {hue_name}"
            else:
                name = hue_name

        return {"rgb": [r, g, b], "hex": hex_code, "name": name}

    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG 2.0 contrast ratio between two colors"""
        def relative_luminance(rgb):
            # Normalize to 0-1
            rgb_norm = rgb / 255.0
            
            # Apply gamma correction (linearize)
            rgb_linear = np.where(
                rgb_norm <= 0.03928,
                rgb_norm / 12.92,
                ((rgb_norm + 0.055) / 1.055) ** 2.4
            )
            
            # Calculate luminance using ITU-R BT.709 coefficients
            return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])
        
        lum1 = relative_luminance(color1)
        lum2 = relative_luminance(color2)
        
        # Ensure lighter color is in numerator
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def calculate_hue_difference(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate hue difference in degrees (0-180)"""
        # Convert RGB to HSV
        hsv1 = cv2.cvtColor(color1.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hsv2 = cv2.cvtColor(color2.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        
        # Calculate circular hue difference (0-180 range in OpenCV)
        hue_diff = abs(hsv1[0] - hsv2[0])
        if hue_diff > 90:
            hue_diff = 180 - hue_diff
            
        return hue_diff
    
    def calculate_saturation_difference(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate saturation difference (0-255)"""
        hsv1 = cv2.cvtColor(color1.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hsv2 = cv2.cvtColor(color2.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        
        return abs(int(hsv1[1]) - int(hsv2[1]))
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray,
                             sample_size: int = 1000) -> np.ndarray:
        """Extract dominant color from a masked region.

        Dispatches to the perceptual k-means-in-LAB extractor (Phase A2) when
        enabled, otherwise falls back to the legacy IQR-median method.
        """
        if self.use_kmeans_lab:
            return self._extract_dominant_color_kmeans_lab(image, mask, sample_size)
        return self._extract_dominant_color_iqr(image, mask, sample_size)

    @staticmethod
    def _extract_dominant_color_iqr(image: np.ndarray, mask: np.ndarray,
                                    sample_size: int = 1000) -> np.ndarray:
        """Legacy IQR-median extractor (Phase pre-A2 baseline).

        Vectorized IQR outlier rejection — O(n) numpy. Kept as the backward-
        compatible fallback when use_kmeans_lab=False.
        """
        if not np.any(mask):
            return np.array([128, 128, 128])

        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])

        if len(masked_pixels) > sample_size:
            indices = np.random.choice(len(masked_pixels), sample_size, replace=False)
            masked_pixels = masked_pixels[indices]

        if len(masked_pixels) > 50:
            q1 = np.percentile(masked_pixels, 25, axis=0)
            q3 = np.percentile(masked_pixels, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            inlier_mask = np.all(
                (masked_pixels >= lower) & (masked_pixels <= upper), axis=1
            )
            if np.sum(inlier_mask) > 10:
                return np.median(masked_pixels[inlier_mask], axis=0).astype(int)

        return np.median(masked_pixels, axis=0).astype(int)

    def _extract_dominant_color_kmeans_lab(
        self, image: np.ndarray, mask: np.ndarray, sample_size: int = 1000,
    ) -> np.ndarray:
        """Pick the dominant cluster's centroid in CIELAB (Phase A2).

        For multi-modal regions (lit + shadowed wall, wainscoting, patterned
        floor), arithmetic-mean and IQR-median both collapse the modes into a
        midtone that doesn't exist anywhere in the region. K-means in LAB
        partitions the modes into clusters; the largest cluster's centroid is
        the color a viewer would actually point to.

        Falls back to the IQR extractor on degenerate input (too few pixels,
        scipy unavailable, all-empty clusters).
        """
        if not np.any(mask):
            return np.array([128, 128, 128])
        masked = image[mask]
        if len(masked) == 0:
            return np.array([128, 128, 128])

        if len(masked) > sample_size:
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(len(masked), sample_size, replace=False)
            masked = masked[idx]

        if len(masked) < self.kmeans_k * 5:
            return self._extract_dominant_color_iqr(image, mask, sample_size)

        try:
            from scipy.cluster.vq import kmeans2
        except ImportError:
            return self._extract_dominant_color_iqr(image, mask, sample_size)

        # OpenCV LAB (uint8): L in [0,255] (scaled from 0-100), a/b in [0,255]
        # centered at 128. Floats here so kmeans returns continuous centroids.
        lab = cv2.cvtColor(
            masked.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float64)

        # Skip k-means on near-constant input — kmeans2 emits a noisy
        # "empty cluster" warning and the median is the right answer anyway.
        if float(lab.std(axis=0).max()) < 1.0:
            return self._extract_dominant_color_iqr(image, mask, sample_size)

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centers, labels = kmeans2(
                    lab, k=self.kmeans_k, seed=0, minit="++", missing="warn"
                )
        except Exception:
            return self._extract_dominant_color_iqr(image, mask, sample_size)

        counts = np.bincount(labels, minlength=self.kmeans_k)
        if counts.max() == 0:
            return self._extract_dominant_color_iqr(image, mask, sample_size)
        dominant_idx = int(counts.argmax())

        center_lab = centers[dominant_idx].clip(0, 255).astype(np.uint8).reshape(1, 1, 3)
        center_rgb = cv2.cvtColor(center_lab, cv2.COLOR_LAB2RGB).reshape(3)
        return center_rgb.astype(int)

    def _sample_boundary_band_color(
        self,
        image: np.ndarray,
        segment_mask: np.ndarray,
        boundary_mask: np.ndarray,
        fallback_color: np.ndarray,
    ) -> np.ndarray:
        """Sample a segment's color within `boundary_band_pixels` of the pair boundary.

        The boundary mask carries pixels from both sides of the pair (see
        find_adjacent_segments). Dilating it by the band radius and intersecting
        with the segment isolates the segment's pixels nearest the boundary —
        the colors that actually determine perceived contrast at the edge.
        Falls back to the precomputed segment-mean color when the band is too
        sparse (small segment, very short shared boundary).
        """
        radius = self.boundary_band_pixels
        k = 2 * radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        band = cv2.dilate(boundary_mask.astype(np.uint8), kernel) > 0
        band &= segment_mask

        if int(band.sum()) < self.min_boundary_band_pixels:
            return fallback_color

        return self.extract_dominant_color(image, band)

    def find_adjacent_segments(self, segmentation: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """Find all pairs of adjacent segments using vectorized numpy operations.

        Uses array slicing to compare each pixel with its 8 neighbors in bulk,
        replacing the O(H*W) Python loop with 8 vectorized C-level comparisons.
        Achieves 50-200x speedup over the per-pixel Python iteration.

        Returns dict mapping (seg1_id, seg2_id) to boundary mask.
        """
        h, w = segmentation.shape
        adjacencies: Dict[Tuple[int, int], np.ndarray] = {}

        # Interior region (avoid boundary indexing issues, same as original)
        seg_interior = segmentation[1:-1, 1:-1]

        # 8-connected neighbor shifts: (dy, dx)
        shifts = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

        for dy, dx in shifts:
            # Extract neighbor slice aligned with the interior
            neighbor = segmentation[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]

            # Vectorized comparison: find all pixels where segments differ.
            # ADE20K uses 0-149 for all 150 classes (0 = wall); there is no
            # background label, so every value is a valid segment.
            diff_mask = seg_interior != neighbor

            if not np.any(diff_mask):
                continue

            # Extract segment IDs at boundary pixels
            center_ids = seg_interior[diff_mask]
            neighbor_ids = neighbor[diff_mask]

            # Canonical pair ordering (smaller ID first)
            pair_min = np.minimum(center_ids, neighbor_ids)
            pair_max = np.maximum(center_ids, neighbor_ids)

            # Get pixel coordinates in full image frame
            ys, xs = np.where(diff_mask)
            ys += 1  # offset for interior crop
            xs += 1

            # Find unique pairs and assign boundary pixels
            unique_pairs = np.unique(np.column_stack([pair_min, pair_max]), axis=0)
            for row in unique_pairs:
                pair = (int(row[0]), int(row[1]))
                pair_mask = ((pair_min == pair[0]) & (pair_max == pair[1]))
                if pair not in adjacencies:
                    adjacencies[pair] = np.zeros((h, w), dtype=bool)
                adjacencies[pair][ys[pair_mask], xs[pair_mask]] = True

        # Filter small boundaries (noise). Phase A3: when enabled, run
        # morph-close + connected-component drop so 1-pixel segmentation
        # speckle doesn't produce phantom pairs.
        cleaned: Dict[Tuple[int, int], np.ndarray] = {}
        for pair, boundary in adjacencies.items():
            if self.use_morphological_boundary_close:
                surviving = self._cleanup_boundary_mask(boundary)
            else:
                surviving = boundary if int(boundary.sum()) >= 20 else None
            if surviving is not None:
                cleaned[pair] = surviving
        return cleaned

    # Reference image area at which `min_boundary_component_pixels` was tuned
    # (the synthetic 240x320 regression scenes). Real photos are ~5-20x bigger;
    # boundary lengths scale roughly with the linear dimension (sqrt of area),
    # so the threshold scales the same way to avoid dropping legitimate small
    # contacts on high-resolution images.
    _MORPH_THRESHOLD_REFERENCE_AREA = 240 * 320

    def _scaled_min_component(self, image_shape: Tuple[int, ...]) -> int:
        h, w = image_shape[:2]
        scale = max(1.0, ((h * w) / self._MORPH_THRESHOLD_REFERENCE_AREA) ** 0.5)
        return int(self.min_boundary_component_pixels * scale)

    def _cleanup_boundary_mask(self, boundary: np.ndarray) -> Optional[np.ndarray]:
        """Morph-close a boundary mask, then drop tiny connected components.

        Returns the surviving boundary (HxW bool) or None if everything is
        speckle. Order matters: closing first knits adjacent jitter into solid
        chunks so the connected-component filter operates on coherent pieces
        rather than punishing legitimate boundaries that happen to have a few
        pixel-scale gaps. The min-component threshold scales with sqrt of the
        image area so it stays meaningful on 720x1280 photos.
        """
        k = self.boundary_close_kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        closed = cv2.morphologyEx(
            boundary.astype(np.uint8), cv2.MORPH_CLOSE, kernel
        )
        n, labels, stats, _ = cv2.connectedComponentsWithStats(closed, connectivity=8)
        keep = np.zeros_like(closed, dtype=bool)
        min_pixels = self._scaled_min_component(boundary.shape)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] >= min_pixels:
                keep |= (labels == i)
        return keep if keep.any() else None
    
    # Categories that act as discrete "figures" on a larger surface (Phase B
    # figure-ground heuristic). The remaining categories that aren't here or
    # in _GROUND_CATEGORIES (e.g. door, window, stairs, decorative) are
    # ambiguous — we fall back to area for those.
    _FIGURE_CATEGORIES = frozenset({'objects', 'furniture', 'fixtures'})
    _GROUND_CATEGORIES = frozenset({'floor', 'wall', 'ceiling'})

    @classmethod
    def _figure_ground_assignment(cls, info1: Dict, info2: Dict) -> Tuple[Dict, Dict, bool]:
        """Pick which segment is figure vs ground for Weber / APCA polarity.

        Returns (figure_info, ground_info, semantic_clear). When
        semantic_clear is False, both segments share the same role family
        (two co-planar surfaces such as wall+floor, or two figures), so the
        figure/ground sign of the resulting metric is conventional rather
        than physically meaningful — magnitude is still informative.

        The heuristic, in order:
          1. one figure-category + one ground-category   → unambiguous
          2. else, the smaller-area segment is figure    → fallback
        """
        c1, c2 = info1['category'], info2['category']
        if c1 in cls._FIGURE_CATEGORIES and c2 in cls._GROUND_CATEGORIES:
            return info1, info2, True
        if c2 in cls._FIGURE_CATEGORIES and c1 in cls._GROUND_CATEGORIES:
            return info2, info1, True
        if info1['area'] <= info2['area']:
            return info1, info2, False
        return info2, info1, False

    @staticmethod
    def _srgb_to_screen_luminance(rgb: np.ndarray) -> float:
        """sRGB → screen-Y per APCA SAPC-8 reference (gamma 2.4, no soft-clip).

        APCA's luminance formula differs from WCAG 2.x in two ways: (1) gamma
        2.4 with no piecewise-linear toe, (2) slightly different ITU-R BT.709
        weights. The numbers are taken verbatim from W3C's SAPC reference
        (https://github.com/Myndex/SAPC-APCA, public-domain algorithm).
        """
        r, g, b = (np.asarray(rgb, dtype=np.float64) / 255.0) ** 2.4
        return float(0.2126729 * r + 0.7151522 * g + 0.072175 * b)

    def calculate_apca_lc(self, fg_color: np.ndarray, bg_color: np.ndarray) -> float:
        """APCA Lc — signed perceptual lightness contrast (Phase B1).

        Caller is responsible for designating which surface is foreground
        ("text") and which is background. The sign of Lc encodes polarity:
            positive = fg darker than bg (BoW — dark figure on light ground)
            negative = fg lighter than bg (WoB — light figure on dark ground)
        Magnitude is polarity-independent contrast strength. W3C reference
        thresholds: body text |Lc| ≥ 75; environmental signage ≈ |Lc| ≥ 60.

        Returns 0.0 when below the raw-SAPC noise floor (|sapc| < loClip)
        per W3 SAPC reference, or when computation is disabled.

        Constants and the loClip=0.1 raw-SAPC zero-gate match the W3 SAPC
        reference (Andrew Somers, public-domain). Spot-checks: black on
        white returns ≈ +106; white on black returns ≈ -108.
        """
        if not self.compute_apca_lc:
            return 0.0

        BLK_THRS = 0.022
        BLK_CLMP = 1.414
        DELTA_Y_MIN = 0.0005
        LO_CLIP = 0.1                # raw-SAPC zero-gate per W3 SAPC reference
        SCALE_BOW = SCALE_WOB = 1.14
        NORM_BG, NORM_TXT = 0.56, 0.57
        REV_BG,  REV_TXT  = 0.65, 0.62
        LO_BOW_OFFSET = LO_WOB_OFFSET = 0.027

        def soft_clip(y: float) -> float:
            return y if y >= BLK_THRS else y + (BLK_THRS - y) ** BLK_CLMP

        y_t = soft_clip(self._srgb_to_screen_luminance(fg_color))
        y_b = soft_clip(self._srgb_to_screen_luminance(bg_color))
        if abs(y_b - y_t) < DELTA_Y_MIN:
            return 0.0
        if y_b > y_t:   # BoW: dark fg on light bg → positive Lc
            sapc = (y_b ** NORM_BG - y_t ** NORM_TXT) * SCALE_BOW
            return (sapc - LO_BOW_OFFSET) * 100.0 if sapc >= LO_CLIP else 0.0
        sapc = (y_b ** REV_BG - y_t ** REV_TXT) * SCALE_WOB
        return (sapc + LO_WOB_OFFSET) * 100.0 if sapc <= -LO_CLIP else 0.0

    def calculate_weber_contrast(
        self, fg_color: np.ndarray, bg_color: np.ndarray,
    ) -> float:
        """Weber contrast (L_fg − L_bg) / L_bg for figure-on-ground (Phase B3).

        Right metric for small obstacles on a uniform floor — WCAG's symmetric
        ratio ignores which surface is figure vs ground, but a dark cushion on
        a light floor is detected differently than the same colors swapped.
        Weber convention: positive when figure is brighter than background,
        negative when darker.

        Caller must designate which surface is "figure" (typically the
        smaller-area one); for pair iteration we use the smaller-area segment
        as foreground.

        Returns 0.0 when background luminance is too small to be meaningful
        (avoids division by zero on near-black backgrounds).
        """
        l_fg = self._srgb_to_screen_luminance(fg_color)
        l_bg = self._srgb_to_screen_luminance(bg_color)
        if l_bg < 1e-6:
            return 0.0
        return float((l_fg - l_bg) / l_bg)

    def calculate_delta_e2000(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """CIEDE2000 perceptual color-difference between two RGB triples (Phase A5).

        Strictly additive metric: WCAG luminance ratio misses chromatic
        contrast entirely (red/green isoluminant pairs that look indistinct
        to a normally-sighted observer pass WCAG with ratio = 1.0). ΔE2000
        captures that gap. Returns 0.0 when computation is disabled.
        """
        if not self.compute_delta_e2000:
            return 0.0
        try:
            from skimage.color import rgb2lab, deltaE_ciede2000
        except ImportError:
            return 0.0
        lab1 = rgb2lab(
            np.asarray(color1, dtype=np.float64).reshape(1, 1, 3) / 255.0
        ).reshape(3)
        lab2 = rgb2lab(
            np.asarray(color2, dtype=np.float64).reshape(1, 1, 3) / 255.0
        ).reshape(3)
        return float(deltaE_ciede2000(lab1, lab2))

    def _scan_intrasegment_pattern(
        self, image: np.ndarray, mask: np.ndarray, seg_id: int, category: str,
    ) -> Optional[Dict]:
        """Detect multi-modal patterning inside a single segment (Phase A4).

        Catches striped rugs, patterned wallpaper, and tiled floors that the
        pair-only contrast analyzer can't see because both modes live within
        one segment. Returns a warning dict on detection or None.
        """
        try:
            from scipy.cluster.vq import kmeans2
        except ImportError:
            return None
        masked = image[mask]
        if len(masked) < 200:
            return None
        if len(masked) > 2000:
            rng = np.random.default_rng(seed=0)
            idx = rng.choice(len(masked), 2000, replace=False)
            masked = masked[idx]

        lab = cv2.cvtColor(
            masked.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB
        ).reshape(-1, 3).astype(np.float64)

        if float(lab.std(axis=0).max()) < 1.0:
            return None

        import warnings
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                centers, labels = kmeans2(
                    lab, k=2, seed=0, minit="++", missing="warn"
                )
        except Exception:
            return None
        counts = np.bincount(labels, minlength=2)
        total = int(counts.sum())
        if total == 0:
            return None
        minority_share = counts.min() / total
        if minority_share < self.intrasegment_min_minority_share:
            return None

        # CIE76 ΔE between the two cluster centers — fast screening gate.
        # Wood-grain and similar low-frequency texture can clear CIE76 ≈ 25
        # while perceptually being a single surface (CIEDE2000 captures the
        # asymmetric chroma weighting CIE76 misses). Confirm with ΔE2000
        # before emitting the warning to suppress that false-positive class.
        delta_e_cie76 = float(np.linalg.norm(centers[0] - centers[1]))
        if delta_e_cie76 < self.intrasegment_delta_e_threshold:
            return None

        centers_rgb = []
        for c in centers:
            c_lab = c.clip(0, 255).astype(np.uint8).reshape(1, 1, 3)
            c_rgb = cv2.cvtColor(c_lab, cv2.COLOR_LAB2RGB).reshape(3).astype(int)
            centers_rgb.append(c_rgb)
        major_idx = int(counts.argmax())
        minor_idx = 1 - major_idx

        # ΔE2000 confirmation — must clear 0.5× the CIE76 threshold (default
        # 35 → 17.5). The two thresholds work in different LAB scales (CIE76
        # in OpenCV uint8 LAB, ΔE2000 in standard CIELAB), so this is an
        # empirical calibration, not a mathematical conversion. Reference
        # cases this discriminates correctly:
        #   - striped red/white rug      ΔE2000 ≈ 49  → KEEP (real pattern)
        #   - wainscoting white/beige    ΔE2000 ≈  9  → DROP (low-contrast
        #                                              two-region surface;
        #                                              boundary is A1's job)
        #   - wood-grain mid-frequency   ΔE2000 ≈ 10  → DROP (single
        #                                              perceptual surface)
        delta_e2000 = self.calculate_delta_e2000(
            centers_rgb[major_idx], centers_rgb[minor_idx]
        )
        if delta_e2000 < self.intrasegment_delta_e_threshold * 0.5:
            return None

        return {
            "segment_id": int(seg_id),
            "category": category,
            "delta_e": delta_e_cie76,
            "delta_e2000": float(delta_e2000),
            "minority_share": float(minority_share),
            "primary_color": self._rgb_to_color_info(centers_rgb[major_idx]),
            "secondary_color": self._rgb_to_color_info(centers_rgb[minor_idx]),
            "area_pixels": int(mask.sum()),
        }

    def is_contrast_sufficient(self, color1: np.ndarray, color2: np.ndarray,
                             category1: str, category2: str) -> Tuple[bool, str]:
        """
        Determine if contrast is sufficient based on WCAG and perceptual guidelines.
        Returns (is_sufficient, severity_if_not)
        """
        wcag_ratio = self.calculate_wcag_contrast(color1, color2)
        hue_diff = self.calculate_hue_difference(color1, color2)
        sat_diff = self.calculate_saturation_difference(color1, color2)
        
        # Critical relationships requiring highest contrast (7:1).
        # Tuples are stored in sorted order to match the lookup key
        # produced by tuple(sorted([category1, category2])).
        critical_pairs = [
            ('door', 'floor'),     # exit/entry visibility at floor level
            ('floor', 'stairs'),   # fall risk at level changes
            ('stairs', 'wall'),    # stair boundary against wall
        ]

        # High priority relationships (4.5:1)
        high_priority_pairs = [
            ('door', 'stairs'),     # stair-door transition
            ('door', 'wall'),       # door visibility in wall
            ('fixtures', 'floor'),  # bathroom fixtures near floor (tub, toilet)
            ('floor', 'furniture'), # furniture visibility on floor
            ('floor', 'objects'),   # trip hazard visibility
            ('floor', 'wall'),      # spatial orientation — room boundary
            ('furniture', 'wall'),  # furniture against wall
        ]
        
        # Check relationship type
        relationship = tuple(sorted([category1, category2]))
        
        # Determine thresholds based on relationship
        if relationship in critical_pairs:
            # Critical: require 7:1 contrast ratio
            if wcag_ratio < 7.0:
                return False, 'critical'
            if hue_diff < 30 and sat_diff < 50:
                return False, 'critical'
                
        elif relationship in high_priority_pairs:
            # High priority: require 4.5:1 contrast ratio
            if wcag_ratio < 4.5:
                return False, 'high'
            if wcag_ratio < 7.0 and hue_diff < 20 and sat_diff < 40:
                return False, 'high'
                
        else:
            # Standard: require 3:1 contrast ratio minimum
            if wcag_ratio < 3.0:
                return False, 'medium'
            if wcag_ratio < 4.5 and hue_diff < 15 and sat_diff < 30:
                return False, 'medium'
        
        return True, None
    
    # Severity color palette (colorblind-accessible)
    SEVERITY_COLORS = {
        'critical': (220, 20, 60),    # Crimson
        'high': (255, 140, 0),        # Dark orange
        'medium': (255, 215, 0),      # Gold
    }

    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Perform comprehensive contrast analysis between ALL adjacent objects.

        Args:
            image: RGB image
            segmentation: Segmentation mask with class IDs

        Returns:
            Dictionary containing analysis results and visualizations
        """
        h, w = segmentation.shape
        results = {
            'issues': [],
            'visualization': image.copy(),
            # Phase A4 — populated below when use_intrasegment_scan is on.
            # Always present as a list so consumers can iterate unconditionally.
            'intrasegment_warnings': [],
            'statistics': {
                'total_segments': 0,
                'analyzed_pairs': 0,
                'low_contrast_pairs': 0,
                'critical_issues': 0,
                'high_priority_issues': 0,
                'medium_priority_issues': 0,
                'floor_object_issues': 0,
                'intrasegment_warning_count': 0,
            }
        }

        # Get unique segments — all values 0-149 are valid ADE20K class labels
        unique_segments = np.unique(segmentation)
        results['statistics']['total_segments'] = len(unique_segments)

        # Build segment information
        segment_info = {}

        logger.info(f"Building segment information for {len(unique_segments)} segments...")

        for seg_id in unique_segments:
            mask = segmentation == seg_id
            area = np.sum(mask)

            if area < 50:  # Skip very small segments
                continue

            category = self.class_to_category.get(seg_id, 'unknown')
            color = self.extract_dominant_color(image, mask)

            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': area,
                'class_id': seg_id
            }

        # Phase A4 — intra-segment pattern scan. Runs once per qualifying
        # segment, independent of the pair-contrast loop, so a striped rug
        # produces a warning even when its boundaries pass WCAG against
        # neighboring surfaces.
        if self.use_intrasegment_scan:
            for seg_id, info in segment_info.items():
                if info['area'] < self.intrasegment_min_segment_pixels:
                    continue
                if info['category'] == 'unknown':
                    continue
                warning = self._scan_intrasegment_pattern(
                    image, info['mask'], seg_id, info['category']
                )
                if warning is not None:
                    results['intrasegment_warnings'].append(warning)
            results['statistics']['intrasegment_warning_count'] = len(
                results['intrasegment_warnings']
            )

        # Find all adjacent segment pairs
        logger.info("Finding adjacent segments...")
        adjacencies = self.find_adjacent_segments(segmentation)
        logger.info(f"Found {len(adjacencies)} adjacent segment pairs")

        # Analyze each adjacent pair
        for (seg1_id, seg2_id), boundary in adjacencies.items():
            if seg1_id not in segment_info or seg2_id not in segment_info:
                continue

            info1 = segment_info[seg1_id]
            info2 = segment_info[seg2_id]

            # Skip if both are unknown categories
            if info1['category'] == 'unknown' and info2['category'] == 'unknown':
                continue

            results['statistics']['analyzed_pairs'] += 1

            # Resolve per-pair colors. Local-boundary sampling (Phase A1) gets
            # the colors that actually meet at the edge; segment-mean is the
            # fallback for tiny shared boundaries.
            if self.use_local_boundary_sampling:
                color1 = self._sample_boundary_band_color(
                    image, info1['mask'], boundary, info1['color']
                )
                color2 = self._sample_boundary_band_color(
                    image, info2['mask'], boundary, info2['color']
                )
            else:
                color1 = info1['color']
                color2 = info2['color']

            # Check contrast sufficiency
            is_sufficient, severity = self.is_contrast_sufficient(
                color1, color2,
                info1['category'], info2['category']
            )

            if not is_sufficient:
                results['statistics']['low_contrast_pairs'] += 1

                # Calculate detailed metrics
                wcag_ratio = self.calculate_wcag_contrast(color1, color2)
                hue_diff = self.calculate_hue_difference(color1, color2)
                sat_diff = self.calculate_saturation_difference(color1, color2)

                # Check if it's a floor-object issue
                is_floor_object = (
                    (info1['category'] == 'floor' and info2['category'] in ['furniture', 'objects']) or
                    (info2['category'] == 'floor' and info1['category'] in ['furniture', 'objects'])
                )

                if is_floor_object:
                    results['statistics']['floor_object_issues'] += 1

                # Count by severity
                if severity == 'critical':
                    results['statistics']['critical_issues'] += 1
                elif severity == 'high':
                    results['statistics']['high_priority_issues'] += 1
                elif severity == 'medium':
                    results['statistics']['medium_priority_issues'] += 1

                # Phase A5 + B1 + B3 metrics — strictly additive alongside
                # wcag_ratio. None changes severity classification.
                # Figure-ground assignment drives the SIGN of weber/APCA: the
                # figure is "text"/foreground, ground is "bg". When both
                # segments are co-planar surfaces or both figures, the sign
                # is conventional (semantic_clear=False); magnitude is still
                # meaningful in both cases.
                fg_info, bg_info, semantic_clear = self._figure_ground_assignment(
                    info1, info2
                )
                fg_color = color1 if fg_info is info1 else color2
                bg_color = color2 if fg_info is info1 else color1
                issue = {
                    'segment_ids': (seg1_id, seg2_id),
                    'categories': (info1['category'], info2['category']),
                    'figure_category': fg_info['category'],
                    'ground_category': bg_info['category'],
                    'figure_ground_semantic_clear': bool(semantic_clear),
                    'colors': (
                        self._rgb_to_color_info(color1),
                        self._rgb_to_color_info(color2),
                    ),
                    'wcag_ratio': float(wcag_ratio),
                    'delta_e2000': float(self.calculate_delta_e2000(color1, color2)),
                    'apca_lc': float(self.calculate_apca_lc(fg_color, bg_color)),
                    'weber_contrast': float(
                        self.calculate_weber_contrast(fg_color, bg_color)
                    ),
                    'hue_difference': float(hue_diff),
                    'saturation_difference': float(sat_diff),
                    'boundary_pixels': int(np.sum(boundary)),
                    'severity': severity,
                    'is_floor_object': is_floor_object,
                    'boundary_mask': boundary
                }

                results['issues'].append(issue)

        # Sort issues by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2}
        results['issues'].sort(key=lambda x: severity_order.get(x['severity'], 3))

        # Build enhanced visualization
        results['visualization'] = self._create_enhanced_visualization(
            image, results['issues'], results['statistics']
        )

        logger.info(f"Contrast analysis complete: {results['statistics']['low_contrast_pairs']} issues found")

        return results

    # ── Visualization helpers ─────────────────────────────────────────

    @staticmethod
    def _draw_text_outlined(image, text, pos, scale, color,
                            thickness=1, font=cv2.FONT_HERSHEY_SIMPLEX):
        """Draw text with black outline for readability on any background."""
        cv2.putText(image, text, pos, font, scale, (0, 0, 0),
                    thickness + 2, cv2.LINE_AA)
        cv2.putText(image, text, pos, font, scale, color,
                    thickness, cv2.LINE_AA)

    def _draw_boundary_glow(self, image, boundary, severity):
        """Draw a multi-layer glow contour for a boundary mask."""
        color = self.SEVERITY_COLORS.get(severity, (255, 255, 255))
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=2)
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # Glow layers: thick + transparent → thin + opaque
        for thick, alpha in [(8, 0.15), (5, 0.25), (3, 0.45)]:
            overlay = image.copy()
            cv2.drawContours(overlay, contours, -1, color, thick, cv2.LINE_AA)
            cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
        # Solid core
        cv2.drawContours(image, contours, -1, color, 2, cv2.LINE_AA)

    def _draw_issue_markers(self, image, issues, max_markers=8):
        """Draw numbered circle markers at each issue boundary centroid."""
        for idx, issue in enumerate(issues[:max_markers], 1):
            bm = issue['boundary_mask']
            ys, xs = np.where(bm)
            if len(ys) == 0:
                continue
            cx, cy = int(np.median(xs)), int(np.median(ys))
            h, w = image.shape[:2]
            cx = np.clip(cx, 25, w - 25)
            cy = np.clip(cy, 25, h - 25)

            color = self.SEVERITY_COLORS.get(issue['severity'], (255, 255, 255))

            # White filled circle + colored border
            cv2.circle(image, (cx, cy), 18, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(image, (cx, cy), 18, color, 3, cv2.LINE_AA)

            # Number inside
            num = str(idx)
            ts = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.putText(image, num,
                        (cx - ts[0] // 2, cy + ts[1] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (30, 30, 30), 2, cv2.LINE_AA)

            # Label below marker
            cat1, cat2 = issue['categories']
            label = f"{cat1}<>{cat2} {issue['wcag_ratio']:.1f}:1"
            label_y = cy + 32
            if label_y > h - 10:
                label_y = cy - 26
            self._draw_text_outlined(image, label, (cx - 50, label_y),
                                     0.42, (255, 255, 255), 1)

    def _draw_statistics_banner(self, canvas_w, stats):
        """Return an 80-px-tall dark banner with metric blocks."""
        banner_h = 80
        banner = np.full((banner_h, canvas_w, 3), 35, dtype=np.uint8)

        metrics = [
            ("CRITICAL", stats.get('critical_issues', 0), (220, 20, 60)),
            ("HIGH", stats.get('high_priority_issues', 0), (255, 140, 0)),
            ("MEDIUM", stats.get('medium_priority_issues', 0), (255, 215, 0)),
            ("PAIRS", stats.get('analyzed_pairs', 0), (100, 220, 100)),
        ]
        spacing = canvas_w // len(metrics)

        for i, (label, value, color) in enumerate(metrics):
            cx = spacing * i + spacing // 2
            # Large number
            val_str = str(value)
            ts = cv2.getTextSize(val_str, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)[0]
            cv2.putText(banner, val_str, (cx - ts[0] // 2, 48),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            # Small label
            ts2 = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0]
            cv2.putText(banner, label, (cx - ts2[0] // 2, 68),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

        # Title on the left
        cv2.putText(banner, "CONTRAST ANALYSIS", (15, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)

        return banner

    def _draw_color_swatches(self, panel_h, issues, max_swatches=5):
        """Return a 300-px-wide right-side panel with color swatch comparisons."""
        panel_w = 300
        panel = np.full((panel_h, panel_w, 3), 42, dtype=np.uint8)

        # Panel title
        cv2.putText(panel, "Top Issues", (15, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.line(panel, (15, 38), (panel_w - 15, 38), (80, 80, 80), 1)

        y_offset = 55
        row_h = (panel_h - 70) // max(max_swatches, 1)
        row_h = min(row_h, 120)

        for idx, issue in enumerate(issues[:max_swatches], 1):
            c1 = tuple(issue['colors'][0]['rgb'])
            c2 = tuple(issue['colors'][1]['rgb'])
            sev_color = self.SEVERITY_COLORS.get(issue['severity'], (200, 200, 200))

            # Issue number + severity dot
            cv2.circle(panel, (20, y_offset + 10), 8, sev_color, -1, cv2.LINE_AA)
            cv2.putText(panel, str(idx), (16, y_offset + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # Color swatch 1
            sw = 36
            sx1 = 40
            cv2.rectangle(panel, (sx1, y_offset), (sx1 + sw, y_offset + sw), c1, -1)
            cv2.rectangle(panel, (sx1, y_offset), (sx1 + sw, y_offset + sw), (120, 120, 120), 1)

            # "vs" text
            cv2.putText(panel, "vs", (sx1 + sw + 4, y_offset + sw // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 160), 1, cv2.LINE_AA)

            # Color swatch 2
            sx2 = sx1 + sw + 28
            cv2.rectangle(panel, (sx2, y_offset), (sx2 + sw, y_offset + sw), c2, -1)
            cv2.rectangle(panel, (sx2, y_offset), (sx2 + sw, y_offset + sw), (120, 120, 120), 1)

            # WCAG ratio + PASS/FAIL
            ratio_x = sx2 + sw + 12
            ratio_str = f"{issue['wcag_ratio']:.1f}:1"
            cv2.putText(panel, ratio_str, (ratio_x, y_offset + 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            if issue['wcag_ratio'] < 4.5:
                cv2.putText(panel, "FAIL", (ratio_x, y_offset + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 60, 60), 1, cv2.LINE_AA)
            else:
                cv2.putText(panel, "PASS", (ratio_x, y_offset + 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 200, 60), 1, cv2.LINE_AA)

            # Category labels
            cat1 = issue['categories'][0][:10]
            cat2 = issue['categories'][1][:10]
            cv2.putText(panel, cat1, (sx1, y_offset + sw + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 170, 170), 1, cv2.LINE_AA)
            cv2.putText(panel, cat2, (sx2, y_offset + sw + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.33, (170, 170, 170), 1, cv2.LINE_AA)

            y_offset += row_h

        return panel

    def _draw_legend(self, image):
        """Draw a severity legend panel in the bottom-right corner."""
        h, w = image.shape[:2]
        lw, lh = 235, 170
        lx = w - lw - 15
        ly = h - lh - 15

        # Semi-transparent background
        overlay = image.copy()
        cv2.rectangle(overlay, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.82, image, 0.18, 0, image)
        cv2.rectangle(image, (lx, ly), (lx + lw, ly + lh), (60, 60, 60), 2)

        # Title
        cv2.putText(image, "Contrast Severity", (lx + 10, ly + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.line(image, (lx + 10, ly + 28), (lx + lw - 10, ly + 28), (180, 180, 180), 1)

        levels = [
            ("CRITICAL", (220, 20, 60), "< 7.0:1 (stairs/doors)"),
            ("HIGH", (255, 140, 0), "< 4.5:1 (walls/furniture)"),
            ("MEDIUM", (255, 215, 0), "< 3.0:1 (standard)"),
            ("PASS", (50, 180, 50), ">= threshold"),
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

        # Footer
        cv2.putText(image, "WCAG 2.1 Level AA", (lx + 10, ly + lh - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1, cv2.LINE_AA)

    def _create_enhanced_visualization(self, image, issues, stats):
        """Compose the full annotated contrast visualization."""
        img_h, img_w = image.shape[:2]
        panel_w = 300 if issues else 0
        banner_h = 80
        canvas_w = img_w + panel_w
        canvas_h = banner_h + img_h

        # -- Statistics banner --
        banner = self._draw_statistics_banner(canvas_w, stats)

        # -- Annotated image --
        annotated = image.copy()
        # Draw glow boundaries
        for issue in issues:
            self._draw_boundary_glow(annotated, issue['boundary_mask'], issue['severity'])
        # Draw numbered markers
        self._draw_issue_markers(annotated, issues)
        # Draw legend
        if issues:
            self._draw_legend(annotated)

        # -- Color swatch panel --
        if issues:
            swatch_panel = self._draw_color_swatches(img_h, issues)
        else:
            swatch_panel = None

        # -- Assemble canvas --
        canvas = np.full((canvas_h, canvas_w, 3), 42, dtype=np.uint8)
        canvas[0:banner_h, 0:canvas_w] = banner
        canvas[banner_h:banner_h + img_h, 0:img_w] = annotated
        if swatch_panel is not None:
            canvas[banner_h:banner_h + img_h, img_w:img_w + panel_w] = swatch_panel

        return canvas
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed text report of contrast analysis"""
        stats = results['statistics']
        issues = results['issues']
        
        report = []
        report.append("=== Universal Contrast Analysis Report ===\n")
        
        # Summary statistics
        report.append(f"Total segments analyzed: {stats['total_segments']}")
        report.append(f"Adjacent pairs analyzed: {stats['analyzed_pairs']}")
        report.append(f"Low contrast pairs found: {stats['low_contrast_pairs']}")
        report.append(f"- Critical issues: {stats['critical_issues']}")
        report.append(f"- High priority issues: {stats['high_priority_issues']}") 
        report.append(f"- Medium priority issues: {stats['medium_priority_issues']}")
        report.append(f"Floor-object contrast issues: {stats['floor_object_issues']}\n")
        
        # Detailed issues
        if issues:
            report.append("=== Contrast Issues (sorted by severity) ===\n")
            
            for i, issue in enumerate(issues[:10], 1):  # Show top 10 issues
                cat1, cat2 = issue['categories']
                wcag = issue['wcag_ratio']
                hue_diff = issue['hue_difference']
                sat_diff = issue['saturation_difference']
                severity = issue['severity'].upper()
                
                report.append(f"{i}. [{severity}] {cat1} ↔ {cat2}")
                report.append(f"   - WCAG Contrast Ratio: {wcag:.2f}:1 (minimum: 4.5:1)")
                report.append(f"   - Hue Difference: {hue_diff:.1f}° (recommended: >30°)")
                report.append(f"   - Saturation Difference: {sat_diff} (recommended: >50)")
                
                if issue['is_floor_object']:
                    report.append("   - ⚠️ Object on floor - requires high visibility!")
                
                report.append(f"   - Boundary size: {issue['boundary_pixels']} pixels")
                report.append("")
        else:
            report.append("✅ No contrast issues detected!")
        
        return "\n".join(report)
