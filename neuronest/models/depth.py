"""Monocular metric depth estimation.

Wraps Depth Anything V2 Metric Indoor (Small) from HuggingFace Transformers.
Outputs per-pixel depth in **meters** for indoor scenes. Used as a shared
service by the placement analyzer, and (in a future iteration) by the
contrast and blackspot pipelines for distance-aware severity weighting.

Designed to mirror the lifecycle of `EoMTSegmenter` and `ImprovedBlackspotDetector`:
explicit `initialize()` step, optional INT8 quantization, graceful failure
returning False rather than raising.
"""

from __future__ import annotations

import logging
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

from ..config import DEPTH_MODEL_ID, ENABLE_QUANTIZATION
from .quantization import quantize_model_int8

logger = logging.getLogger(__name__)


class MonocularMetricDepth:
    """Indoor monocular metric depth estimator.

    Loads `depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf`, which is
    fine-tuned on Hypersim + Virtual KITTI to output depth in physical meters
    rather than relative depth.
    """

    def __init__(self, model_id: str = DEPTH_MODEL_ID):
        self.model_id = model_id
        self.processor: Optional[AutoImageProcessor] = None
        self.model: Optional[AutoModelForDepthEstimation] = None
        self.initialized = False

    def initialize(self) -> bool:
        """Load weights + processor. Returns True on success, False otherwise.

        Quantizes nn.Linear layers to INT8 when `ENABLE_QUANTIZATION` is set,
        matching the segmenter / blackspot pattern for CPU-friendly inference.
        """
        try:
            logger.info(f"Loading metric depth model from {self.model_id}...")
            self.processor = AutoImageProcessor.from_pretrained(self.model_id)
            self.model = AutoModelForDepthEstimation.from_pretrained(self.model_id)
            self.model.eval()
            if ENABLE_QUANTIZATION:
                self.model = quantize_model_int8(self.model, "DepthAnythingV2-Metric-S")
            else:
                logger.info("INT8 quantization disabled for depth model (FP32 mode)")
            self.initialized = True
            logger.info("Metric depth model initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize metric depth model: {e}")
            self.initialized = False
            return False

    @torch.inference_mode()
    def estimate_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        """Run inference and return depth map in meters at the original resolution.

        Args:
            image_rgb: H x W x 3 uint8 RGB array.

        Returns:
            H x W float32 depth map in meters. Shape matches the input image.
        """
        if not self.initialized:
            raise RuntimeError("MonocularMetricDepth not initialized")

        original_h, original_w = image_rgb.shape[:2]
        pil_image = Image.fromarray(image_rgb)
        inputs = self.processor(images=pil_image, return_tensors="pt")

        outputs = self.model(**inputs)
        # Depth Anything V2 Metric returns `predicted_depth` of shape (1, H', W').
        predicted = outputs.predicted_depth.squeeze().cpu().numpy().astype(np.float32)

        # Resize to the original image resolution so per-pixel depth aligns
        # exactly with the segmentation mask the rest of the pipeline uses.
        if predicted.shape != (original_h, original_w):
            predicted = cv2.resize(
                predicted, (original_w, original_h),
                interpolation=cv2.INTER_LINEAR,
            )

        # Sanity clamp: indoor scenes never exceed ~30 m, never below 0.1 m.
        # Out-of-range values are usually depth-model artifacts at sky / window
        # patches that downstream geometry should discard.
        np.clip(predicted, 0.1, 30.0, out=predicted)
        return predicted

    def estimate_depth_with_uncertainty(
        self, image_rgb: np.ndarray, window: int = 5
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (depth_map, local_std_map) for downstream uncertainty propagation.

        The std map is computed as the local standard deviation of depth within
        a `window x window` box around each pixel — a cheap proxy for per-pixel
        uncertainty without needing an MC-dropout or ensemble depth estimator.
        """
        depth = self.estimate_depth(image_rgb)

        # Box-filter trick for local variance: Var(X) = E[X^2] - E[X]^2
        kernel_size = (window, window)
        mean = cv2.boxFilter(depth, ddepth=-1, ksize=kernel_size, normalize=True)
        mean_sq = cv2.boxFilter(depth * depth, ddepth=-1, ksize=kernel_size, normalize=True)
        var = np.clip(mean_sq - mean * mean, 0.0, None)
        std = np.sqrt(var)
        return depth, std

    @property
    def is_loaded(self) -> bool:
        return self.initialized
