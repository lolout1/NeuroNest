import numpy as np
import cv2
import torch
import logging
from typing import Tuple
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForUniversalSegmentation

from ..config import ENABLE_QUANTIZATION, EOMT_MODEL_ID, FLOOR_CLASSES, OUTDOOR_CLASS_IDS
from ..utils import prepare_display_image
from .quantization import quantize_model_int8

logger = logging.getLogger(__name__)

try:
    from ade20k_classes import ADE20K_COLORS, ADE20K_NAMES
except ImportError:
    from neuronest.data.ade20k import ADE20K_COLORS, ADE20K_NAMES


class EoMTSegmenter:
    """Semantic segmentation via EoMT-DINOv3 (ADE20K 150-class, 59.5 mIoU)."""

    def __init__(self):
        self.processor = None
        self.model = None
        self.initialized = False

    def initialize(self, backbone: str = "dinov3") -> bool:
        try:
            logger.info(f"Loading EoMT-DINOv3 from {EOMT_MODEL_ID}...")
            self.processor = AutoImageProcessor.from_pretrained(EOMT_MODEL_ID)
            self.model = AutoModelForUniversalSegmentation.from_pretrained(EOMT_MODEL_ID)
            self.model.eval()
            if ENABLE_QUANTIZATION:
                self.model = quantize_model_int8(self.model, "EoMT-DINOv3-L")
            else:
                logger.info("INT8 quantization disabled for EoMT (FP32 mode)")
            self.initialized = True
            logger.info("EoMT-DINOv3 initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize EoMT: {e}")
            return False

    def semantic_segmentation(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if not self.initialized:
            raise RuntimeError("EoMT not initialized")
        original_h, original_w = image.shape[:2]
        logger.info(f"Processing image at {image.shape}")
        pil_image = Image.fromarray(image)
        inputs = self.processor(images=pil_image, return_tensors="pt")
        with torch.inference_mode():
            outputs = self.model(**inputs)
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(original_h, original_w)]
        )
        seg_mask = seg_maps[0].cpu().numpy().astype(np.uint8)
        vis_image = self._visualize_segmentation(image, seg_mask)
        vis_image_display = prepare_display_image(vis_image)
        return seg_mask, vis_image_display

    def _visualize_segmentation(
        self, image: np.ndarray, seg_mask: np.ndarray, alpha: float = 0.6
    ) -> np.ndarray:
        h, w = seg_mask.shape
        color_overlay = np.zeros((h, w, 3), dtype=np.uint8)
        for label_id in np.unique(seg_mask):
            if label_id < len(ADE20K_COLORS):
                if label_id in OUTDOOR_CLASS_IDS:
                    color_overlay[seg_mask == label_id] = [60, 60, 60]
                else:
                    color_overlay[seg_mask == label_id] = ADE20K_COLORS[label_id]
        vis = cv2.addWeighted(image, 1 - alpha, color_overlay, alpha, 0)
        labels, areas = np.unique(seg_mask, return_counts=True)
        min_area = h * w * 0.01
        for label_id, area in zip(labels, areas):
            if label_id in OUTDOOR_CLASS_IDS:
                continue
            if area >= min_area and label_id < len(ADE20K_NAMES):
                ys, xs = np.where(seg_mask == label_id)
                cx, cy = int(np.median(xs)), int(np.median(ys))
                name = ADE20K_NAMES[label_id].split(",")[0]
                cv2.putText(
                    vis, name, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA,
                )
                cv2.putText(
                    vis, name, (cx, cy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
                )
        return vis

    def extract_floor_areas(self, segmentation: np.ndarray) -> np.ndarray:
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        for class_ids in FLOOR_CLASSES.values():
            for class_id in class_ids:
                floor_mask |= segmentation == class_id
        return floor_mask
