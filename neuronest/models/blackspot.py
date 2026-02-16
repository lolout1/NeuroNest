import numpy as np
import cv2
import os
import logging
from typing import Dict, List, Optional

from detectron2.config import get_cfg
from detectron2.engine.defaults import DefaultPredictor
from detectron2 import model_zoo
from huggingface_hub import hf_hub_download

from ..config import (
    DEVICE, ENABLE_QUANTIZATION, BLACKSPOT_MODEL_REPO,
    BLACKSPOT_MODEL_FILE, FLOOR_CLASS_IDS,
)
from ..utils import prepare_display_image
from .quantization import quantize_model_int8

logger = logging.getLogger(__name__)


class ImprovedBlackspotDetector:
    def __init__(self, model_path: str = None):
        self.model_path = model_path
        self.predictor = None
        self.floor_classes = FLOOR_CLASS_IDS

    def download_model(self) -> str:
        root_path = f"./{BLACKSPOT_MODEL_FILE}"
        if os.path.exists(root_path):
            logger.info(f"Using local blackspot model: {root_path}")
            return root_path
        try:
            model_path = hf_hub_download(
                repo_id=BLACKSPOT_MODEL_REPO, filename=BLACKSPOT_MODEL_FILE
            )
            logger.info(f"Downloaded blackspot model to: {model_path}")
            return model_path
        except Exception as e:
            logger.warning(f"Could not download blackspot model: {e}")
            local_path = f"./output_floor_blackspot/{BLACKSPOT_MODEL_FILE}"
            if os.path.exists(local_path):
                logger.info(f"Using local blackspot model: {local_path}")
                return local_path
            return None

    def initialize(self, threshold: float = 0.5) -> bool:
        try:
            if self.model_path is None:
                self.model_path = self.download_model()
            if self.model_path is None:
                logger.error("No blackspot model available")
                return False
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file(
                    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
                )
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.DEVICE = DEVICE
            self.predictor = DefaultPredictor(cfg)
            if ENABLE_QUANTIZATION:
                self.predictor.model = quantize_model_int8(
                    self.predictor.model, "MaskRCNN-R50-FPN"
                )
            else:
                logger.info("INT8 quantization disabled for MaskRCNN (FP32 mode)")
            logger.info("MaskRCNN blackspot detector initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize blackspot detector: {e}")
            return False

    def is_on_floor_surface(
        self,
        blackspot_mask: np.ndarray,
        segmentation: np.ndarray,
        floor_mask: np.ndarray,
        overlap_threshold: float = 0.8,
    ) -> bool:
        if np.sum(blackspot_mask) == 0:
            return False
        overlap = blackspot_mask & floor_mask
        overlap_ratio = np.sum(overlap) / np.sum(blackspot_mask)
        if overlap_ratio < overlap_threshold:
            return False
        blackspot_pixels = segmentation[blackspot_mask]
        if len(blackspot_pixels) == 0:
            return False
        unique_classes, counts = np.unique(blackspot_pixels, return_counts=True)
        floor_pixel_count = sum(
            counts[unique_classes == cls]
            for cls in self.floor_classes
            if cls in unique_classes
        )
        floor_ratio = floor_pixel_count / len(blackspot_pixels)
        return floor_ratio > 0.7

    def filter_non_floor_blackspots(
        self,
        blackspot_masks: List[np.ndarray],
        segmentation: np.ndarray,
        floor_mask: np.ndarray,
    ) -> List[np.ndarray]:
        filtered = []
        for mask in blackspot_masks:
            if self.is_on_floor_surface(mask, segmentation, floor_mask):
                filtered.append(mask)
        return filtered

    def detect_blackspots(
        self,
        image: np.ndarray,
        segmentation: np.ndarray,
        floor_prior: Optional[np.ndarray] = None,
    ) -> Dict:
        if self.predictor is None:
            raise RuntimeError("Blackspot detector not initialized")
        original_h, original_w = image.shape[:2]
        if floor_prior is not None and floor_prior.shape != (original_h, original_w):
            floor_prior = cv2.resize(
                floor_prior.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        if segmentation.shape != (original_h, original_w):
            segmentation = cv2.resize(
                segmentation.astype(np.uint8),
                (original_w, original_h),
                interpolation=cv2.INTER_NEAREST,
            )
        try:
            outputs = self.predictor(image)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            logger.error(f"MaskRCNN prediction error: {e}")
            return self._empty_results(image)

        if len(instances) == 0:
            return self._empty_results(image)

        pred_classes = instances.pred_classes.numpy()
        pred_masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()

        blackspot_indices = pred_classes == 1
        blackspot_masks = pred_masks[blackspot_indices] if np.any(blackspot_indices) else []
        blackspot_scores = scores[blackspot_indices] if np.any(blackspot_indices) else []

        if floor_prior is not None:
            floor_mask = floor_prior
        else:
            floor_mask = np.zeros(segmentation.shape, dtype=bool)
            for cls in self.floor_classes:
                floor_mask |= segmentation == cls

        filtered_blackspot_masks = self.filter_non_floor_blackspots(
            blackspot_masks, segmentation, floor_mask
        )

        combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
        for mask in filtered_blackspot_masks:
            combined_blackspot |= mask

        visualization = self._create_visualization(image, floor_mask, combined_blackspot)
        visualization_display = prepare_display_image(visualization)

        floor_area = int(np.sum(floor_mask))
        blackspot_area = int(np.sum(combined_blackspot))
        coverage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0

        return {
            "visualization": visualization_display,
            "floor_mask": floor_mask,
            "blackspot_mask": combined_blackspot,
            "floor_area": floor_area,
            "blackspot_area": blackspot_area,
            "coverage_percentage": coverage,
            "num_detections": len(filtered_blackspot_masks),
            "avg_confidence": float(np.mean(blackspot_scores)) if len(blackspot_scores) > 0 else 0.0,
        }

    def _create_visualization(
        self, image: np.ndarray, floor_mask: np.ndarray, blackspot_mask: np.ndarray
    ) -> np.ndarray:
        vis = image.copy()
        floor_overlay = vis.copy()
        floor_overlay[floor_mask] = [0, 255, 0]
        vis = cv2.addWeighted(vis, 0.7, floor_overlay, 0.3, 0)
        vis[blackspot_mask] = [255, 0, 0]
        contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(vis, contours, -1, (255, 255, 0), 4)
        return vis

    def _empty_results(self, image: np.ndarray) -> Dict:
        empty_mask = np.zeros(image.shape[:2], dtype=bool)
        return {
            "visualization": prepare_display_image(image),
            "floor_mask": empty_mask,
            "blackspot_mask": empty_mask,
            "floor_area": 0,
            "blackspot_area": 0,
            "coverage_percentage": 0,
            "num_detections": 0,
            "avg_confidence": 0.0,
        }
