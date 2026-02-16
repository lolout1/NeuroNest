import numpy as np
import cv2
import gc
import time
import logging
import traceback

from ade20k_classes import ADE20K_NAMES
from .viz import METHOD_INFO

logger = logging.getLogger(__name__)


class DefectMixin:
    """Defect overlay and defect-focused XAI analysis."""

    def _overlay_defects(self, blended, blackspot_mask, contrast_issues=None, seg_mask=None):
        vis = blended.copy()
        h, w = vis.shape[:2]

        if blackspot_mask is not None and blackspot_mask.any():
            bm = blackspot_mask
            if bm.shape[:2] != (h, w):
                bm = cv2.resize(
                    bm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            contours, _ = cv2.findContours(
                bm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            cv2.drawContours(vis, contours, -1, (255, 50, 50), 3)
            for c in contours:
                M = cv2.moments(c)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    cv2.putText(
                        vis, "BLACKSPOT", (max(cx - 40, 5), cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2, cv2.LINE_AA,
                    )

        if contrast_issues and seg_mask is not None:
            sm = seg_mask
            if sm.shape[:2] != (h, w):
                sm = cv2.resize(
                    sm.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST
                )
            for issue in contrast_issues[:10]:
                cats = issue.get("categories", [])
                sev = issue.get("severity", "medium")
                color = (255, 50, 50) if sev == "critical" else (255, 180, 0)
                for cat_name in cats:
                    for cid, name in enumerate(ADE20K_NAMES):
                        if cat_name.lower() in name.lower():
                            bd = cv2.Canny(
                                (sm == cid).astype(np.uint8) * 255, 50, 150
                            )
                            bd_contours, _ = cv2.findContours(
                                bd, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
                            )
                            cv2.drawContours(vis, bd_contours, -1, color, 2)
                            break

        return vis

    def run_defect_analysis(
        self, image, blackspot_mask, contrast_issues=None,
        seg_mask=None, progress_callback=None,
    ):
        logger.info("[XAI] Running defect-focused analysis...")
        t0 = time.perf_counter()

        target_class = self._dominant_class(seg_mask) if seg_mask is not None else None
        cn = self._cname(target_class) if target_class is not None else "floor"

        def _progress(frac, msg):
            if progress_callback:
                try:
                    progress_callback(frac, desc=msg)
                except Exception:
                    pass

        methods = [
            ("gradcam", self.gradcam_segmentation, {"target_class_id": target_class}),
            ("entropy", self.predictive_entropy, {}),
            ("integrated_gradients", self.integrated_gradients, {"target_class_id": target_class}),
            ("chefer", self.chefer_relevancy, {"target_class_id": target_class}),
        ]

        results = {}
        for i, (key, fn, kw) in enumerate(methods):
            _progress(i / len(methods), f"Running {key}...")
            try:
                r = fn(image, **kw)
                raw = r.get("_blended_raw")
                if raw is not None:
                    defect_vis = self._overlay_defects(
                        raw, blackspot_mask, contrast_issues, seg_mask
                    )
                    info = METHOD_INFO.get(key, {})
                    r["defect_visualization"] = self._annotate(
                        defect_vis, key,
                        f"{info.get('title', key)} | Defect Overlay",
                        f"Target: {cn} | Blackspots: red | Contrast: amber",
                    )
                results[key] = r
            except Exception as e:
                logger.error(f"[XAI-Defect] {key} failed: {e}\n{traceback.format_exc()}")
                results[key] = self._fallback(image, str(e), key)
            gc.collect()

        self.cleanup_fp32()
        elapsed = time.perf_counter() - t0
        _progress(1.0, f"Defect analysis complete ({elapsed:.0f}s)")
        logger.info(f"[XAI-Defect] Done in {elapsed:.1f}s")
        return results
