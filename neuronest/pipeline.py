import numpy as np
import cv2
import gc
import time
import logging
from typing import Dict
from concurrent.futures import ThreadPoolExecutor

from .models import EoMTSegmenter, ImprovedBlackspotDetector
from .utils import prepare_display_image
from universal_contrast_analyzer import UniversalContrastAnalyzer

logger = logging.getLogger(__name__)


class NeuroNestApp:
    def __init__(self):
        self.segmenter = EoMTSegmenter()
        self.blackspot_detector = None
        self.contrast_analyzer = UniversalContrastAnalyzer(wcag_threshold=4.5)
        self.xai_analyzer = None
        self.initialized = False

    def initialize(self):
        logger.info("Initializing NeuroNest...")
        seg_success = self.segmenter.initialize()

        blackspot_success = False
        try:
            self.blackspot_detector = ImprovedBlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
        except Exception as e:
            logger.warning(f"Could not initialize blackspot detector: {e}")

        if seg_success:
            from .xai import XAIAnalyzer

            self.xai_analyzer = XAIAnalyzer(
                eomt_model=self.segmenter.model,
                eomt_processor=self.segmenter.processor,
                blackspot_predictor=(
                    self.blackspot_detector.predictor
                    if self.blackspot_detector
                    else None
                ),
            )

        self.initialized = seg_success
        return seg_success, blackspot_success

    def analyze_image(
        self,
        image_path: str,
        blackspot_threshold: float = 0.5,
        contrast_threshold: float = 4.5,
        enable_blackspot: bool = True,
        enable_contrast: bool = True,
    ) -> Dict:
        if not self.initialized:
            return {"error": "Application not properly initialized"}
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return {"error": "Could not load image"}
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image: {image_rgb.shape}")

            results = {
                "original_image": image_rgb,
                "segmentation": None,
                "blackspot": None,
                "contrast": None,
                "statistics": {},
            }

            t0 = time.perf_counter()
            logger.info("Running segmentation...")
            seg_mask, seg_vis = self.segmenter.semantic_segmentation(image_rgb)
            results["segmentation"] = {"visualization": seg_vis, "mask": seg_mask}
            floor_prior = self.segmenter.extract_floor_areas(seg_mask)
            t_seg = time.perf_counter() - t0
            logger.info(f"Segmentation: {t_seg:.1f}s")

            t1 = time.perf_counter()
            futures = {}
            with ThreadPoolExecutor(max_workers=2) as executor:
                if enable_blackspot and self.blackspot_detector is not None:
                    futures["blackspot"] = executor.submit(
                        self.blackspot_detector.detect_blackspots,
                        image_rgb, seg_mask, floor_prior,
                    )
                if enable_contrast:
                    futures["contrast"] = executor.submit(
                        self.contrast_analyzer.analyze_contrast,
                        image_rgb, seg_mask,
                    )

                for key, future in futures.items():
                    try:
                        result = future.result(timeout=300)
                        if key == "contrast":
                            result["visualization"] = prepare_display_image(
                                result["visualization"]
                            )
                        results[key] = result
                        logger.info(f"{key} completed")
                    except Exception as e:
                        logger.error(f"{key} failed: {e}")
                        results[key] = None

            t_total = time.perf_counter() - t0
            logger.info(f"Pipeline: {t_total:.1f}s total")

            gc.collect()
            results["statistics"] = self._generate_statistics(results)
            return results

        except Exception as e:
            logger.error(f"Analysis error: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_statistics(self, results: Dict) -> Dict:
        stats = {}
        if results["segmentation"]:
            unique_classes = np.unique(results["segmentation"]["mask"])
            stats["segmentation"] = {
                "num_classes": len(unique_classes),
                "image_size": results["segmentation"]["mask"].shape,
            }
        if results["blackspot"]:
            bs = results["blackspot"]
            stats["blackspot"] = {
                "floor_area_pixels": bs["floor_area"],
                "blackspot_area_pixels": bs["blackspot_area"],
                "coverage_percentage": bs["coverage_percentage"],
                "num_detections": bs["num_detections"],
                "avg_confidence": bs["avg_confidence"],
            }
        if results["contrast"]:
            cs = results["contrast"]["statistics"]
            stats["contrast"] = {
                "total_segments": cs.get("total_segments", 0),
                "analyzed_pairs": cs.get("analyzed_pairs", 0),
                "low_contrast_pairs": cs.get("low_contrast_pairs", 0),
                "critical_issues": cs.get("critical_issues", 0),
                "high_priority_issues": cs.get("high_priority_issues", 0),
                "medium_priority_issues": cs.get("medium_priority_issues", 0),
                "floor_object_issues": cs.get("floor_object_issues", 0),
            }
        return stats
