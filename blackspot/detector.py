"""Enhanced blackspot detection with clear visualization."""

import torch
import numpy as np
import cv2
import os
import logging
from typing import Dict, Optional

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor as DetectronPredictor
from detectron2 import model_zoo

from config import DEVICE

logger = logging.getLogger(__name__)


class BlackspotDetector:
    """Manages blackspot detection with MaskRCNN - Enhanced Version"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = None

    def initialize(self, threshold: float = 0.5) -> bool:
        """Initialize MaskRCNN model"""
        try:
            cfg = get_cfg()
            cfg.merge_from_file(
                model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
            )
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # [floors, blackspot]
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
            cfg.MODEL.WEIGHTS = self.model_path
            cfg.MODEL.DEVICE = DEVICE

            self.predictor = DetectronPredictor(cfg)
            logger.info("MaskRCNN blackspot detector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize blackspot detector: {e}")
            return False

    def create_enhanced_visualizations(self, image: np.ndarray, floor_mask: np.ndarray,
                                     blackspot_mask: np.ndarray) -> Dict:
        """Create multiple enhanced visualizations of blackspot detection"""

        # 1. Pure Segmentation View (like semantic segmentation output)
        segmentation_view = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
        segmentation_view[floor_mask] = [34, 139, 34]      # Forest green for floor
        segmentation_view[blackspot_mask] = [255, 0, 0]    # Bright red for blackspots
        segmentation_view[~(floor_mask | blackspot_mask)] = [128, 128, 128]  # Gray for other areas

        # 2. High Contrast Overlay
        high_contrast_overlay = image.copy()
        # Make background slightly darker to emphasize blackspots
        high_contrast_overlay = cv2.convertScaleAbs(high_contrast_overlay, alpha=0.6, beta=0)
        # Add bright overlays
        high_contrast_overlay[floor_mask] = cv2.addWeighted(
            high_contrast_overlay[floor_mask], 0.7,
            np.full_like(high_contrast_overlay[floor_mask], [0, 255, 0]), 0.3, 0
        )
        high_contrast_overlay[blackspot_mask] = [255, 0, 255]  # Magenta for maximum visibility

        # 3. Blackspot-only View (white blackspots on black background)
        blackspot_only = np.zeros((*image.shape[:2], 3), dtype=np.uint8)
        blackspot_only[blackspot_mask] = [255, 255, 255]  # White blackspots
        blackspot_only[floor_mask & ~blackspot_mask] = [64, 64, 64]  # Dark gray for floor areas

        # 4. Side-by-side comparison
        h, w = image.shape[:2]
        side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
        side_by_side[:, :w] = image
        side_by_side[:, w:] = segmentation_view

        # Add text labels
        cv2.putText(side_by_side, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Blackspot Detection", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 5. Annotated view with bounding boxes and labels
        annotated_view = image.copy()

        # Find blackspot contours for bounding boxes
        blackspot_contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for i, contour in enumerate(blackspot_contours):
            if cv2.contourArea(contour) > 50:  # Filter small artifacts
                # Draw bounding box
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(annotated_view, (x, y), (x + w, y + h), (255, 0, 0), 2)

                # Draw contour
                cv2.drawContours(annotated_view, [contour], -1, (255, 0, 255), 2)

                # Add label
                area = cv2.contourArea(contour)
                label = f"Blackspot {i+1}: {area:.0f}px"
                cv2.putText(annotated_view, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        return {
            'segmentation_view': segmentation_view,
            'high_contrast_overlay': high_contrast_overlay,
            'blackspot_only': blackspot_only,
            'side_by_side': side_by_side,
            'annotated_view': annotated_view
        }

    def detect_blackspots(self, image: np.ndarray, floor_prior: Optional[np.ndarray] = None) -> Dict:
        """Detect blackspots with enhanced visualizations"""
        if self.predictor is None:
            raise RuntimeError("Blackspot detector not initialized")

        # Get original image dimensions
        original_h, original_w = image.shape[:2]

        # Handle floor prior shape mismatch
        processed_image = image.copy()
        if floor_prior is not None:
            prior_h, prior_w = floor_prior.shape

            # Resize floor_prior to match original image if needed
            if (prior_h, prior_w) != (original_h, original_w):
                logger.info(f"Resizing floor prior from {(prior_h, prior_w)} to {(original_h, original_w)}")
                floor_prior_resized = cv2.resize(
                    floor_prior.astype(np.uint8),
                    (original_w, original_h),
                    interpolation=cv2.INTER_NEAREST
                ).astype(bool)
            else:
                floor_prior_resized = floor_prior
        else:
            floor_prior_resized = None

        # Run detection on the processed image
        try:
            outputs = self.predictor(processed_image)
            instances = outputs["instances"].to("cpu")
        except Exception as e:
            logger.error(f"Error in MaskRCNN prediction: {e}")
            # Return empty results
            empty_mask = np.zeros(image.shape[:2], dtype=bool)
            return {
                'visualization': image,
                'floor_mask': empty_mask,
                'blackspot_mask': empty_mask,
                'floor_area': 0,
                'blackspot_area': 0,
                'coverage_percentage': 0,
                'num_detections': 0,
                'avg_confidence': 0.0,
                'enhanced_views': self.create_enhanced_visualizations(image, empty_mask, empty_mask)
            }

        # Process results
        if len(instances) == 0:
            # No detections
            combined_floor = floor_prior_resized if floor_prior_resized is not None else np.zeros(image.shape[:2], dtype=bool)
            combined_blackspot = np.zeros(image.shape[:2], dtype=bool)
            blackspot_scores = []
        else:
            pred_classes = instances.pred_classes.numpy()
            pred_masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()

            # Separate floor and blackspot masks
            floor_indices = pred_classes == 0
            blackspot_indices = pred_classes == 1

            floor_masks = pred_masks[floor_indices] if np.any(floor_indices) else []
            blackspot_masks = pred_masks[blackspot_indices] if np.any(blackspot_indices) else []
            blackspot_scores = scores[blackspot_indices] if np.any(blackspot_indices) else []

            # Combine masks
            combined_floor = np.zeros(image.shape[:2], dtype=bool)
            combined_blackspot = np.zeros(image.shape[:2], dtype=bool)

            for mask in floor_masks:
                combined_floor |= mask

            for mask in blackspot_masks:
                combined_blackspot |= mask

            # Apply floor prior if available
            if floor_prior_resized is not None:
                # Combine OneFormer floor detection with MaskRCNN floor detection
                combined_floor |= floor_prior_resized
                # Keep only blackspots that are on floors
                combined_blackspot &= combined_floor

        # Create all enhanced visualizations
        enhanced_views = self.create_enhanced_visualizations(image, combined_floor, combined_blackspot)

        # Calculate statistics
        floor_area = int(np.sum(combined_floor))
        blackspot_area = int(np.sum(combined_blackspot))
        coverage_percentage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0

        # Count individual blackspot instances
        blackspot_contours, _ = cv2.findContours(
            combined_blackspot.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        actual_detections = len([c for c in blackspot_contours if cv2.contourArea(c) > 50])

        return {
            'visualization': enhanced_views['high_contrast_overlay'],  # Main view
            'floor_mask': combined_floor,
            'blackspot_mask': combined_blackspot,
            'floor_area': floor_area,
            'blackspot_area': blackspot_area,
            'coverage_percentage': coverage_percentage,
            'num_detections': actual_detections,
            'avg_confidence': float(np.mean(blackspot_scores)) if len(blackspot_scores) > 0 else 0.0,
            'enhanced_views': enhanced_views  # All visualization options
        }
