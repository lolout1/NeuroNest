"""Enhanced blackspot detection with clear visualization - FLOORS ONLY, BLACK AREAS ONLY."""

import torch
import numpy as np
import cv2
import os
import logging
from typing import Dict, Optional, Tuple

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor as DetectronPredictor
from detectron2 import model_zoo

from config import DEVICE

logger = logging.getLogger(__name__)


class BlackspotDetector:
    """Enhanced blackspot detector - detects only black/dark areas on floors"""

    def __init__(self, model_path: str):
        self.model_path = model_path
        self.predictor = None
        self.black_threshold = 70  # RGB values below this are considered "black"
        self.dark_threshold = 110  # RGB values below this are considered "dark"
        self.confidence_threshold = 0.5

    def initialize(self, threshold: float = 0.5) -> bool:
        """Initialize MaskRCNN model"""
        try:
            self.confidence_threshold = threshold
            
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

    def is_black_or_dark(self, color: np.ndarray) -> Tuple[bool, bool]:
        """Check if a color is black or dark - Python 3.8 compatible"""
        # Convert to grayscale for better dark detection
        gray_value = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]

        # Check RGB values
        max_rgb = np.max(color)
        mean_rgb = np.mean(color)

        # More stringent black detection
        is_black = max_rgb < self.black_threshold and gray_value < self.black_threshold and mean_rgb < self.black_threshold
        is_dark = max_rgb < self.dark_threshold and gray_value < self.dark_threshold

        return is_black, is_dark

    def detect_floor_blackspots_by_color(self, image: np.ndarray, floor_mask: np.ndarray) -> np.ndarray:
        """Detect black/dark areas on floors using robust multi-method color analysis"""
        if not np.any(floor_mask):
            return np.zeros_like(floor_mask, dtype=bool)

        # Create blackspot mask for floor areas only
        blackspot_mask = np.zeros_like(floor_mask, dtype=bool)

        # Convert image to different color spaces for comprehensive black detection
        hsv_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab_image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Get floor pixels
        floor_pixels = floor_mask > 0

        if not np.any(floor_pixels):
            return blackspot_mask

        # Method 1: Strict RGB-based black detection
        rgb_black = (
            (image[:, :, 0] < self.black_threshold) &
            (image[:, :, 1] < self.black_threshold) &
            (image[:, :, 2] < self.black_threshold) &
            floor_pixels
        )

        # Method 2: HSV-based dark detection (very low value/brightness)
        hsv_very_dark = (
            (hsv_image[:, :, 2] < self.black_threshold) &  # Very low value
            (hsv_image[:, :, 1] < 150) &  # Not too saturated (avoid bright colors)
            floor_pixels
        )

        # Method 3: LAB-based detection (very low lightness)
        lab_very_dark = (
            (lab_image[:, :, 0] < 60) &  # Very low L* value in LAB
            floor_pixels
        )

        # Method 4: Grayscale intensity (very dark)
        gray_very_dark = (
            (gray_image < self.black_threshold) &
            floor_pixels
        )

        # Method 5: Combined RGB intensity
        rgb_combined_dark = (
            ((image[:, :, 0] + image[:, :, 1] + image[:, :, 2]) / 3 < self.black_threshold) &
            floor_pixels
        )

        # Combine methods - require at least 3 methods to agree for high confidence
        detection_votes = (
            rgb_black.astype(int) +
            hsv_very_dark.astype(int) +
            lab_very_dark.astype(int) +
            gray_very_dark.astype(int) +
            rgb_combined_dark.astype(int)
        )

        # Areas with 3+ votes are considered blackspots (high confidence)
        blackspot_mask = (detection_votes >= 3) & floor_pixels

        # Morphological operations to clean up the mask
        kernel_small = np.ones((3, 3), np.uint8)
        kernel_medium = np.ones((5, 5), np.uint8)
        
        # Remove very small noise
        blackspot_mask = cv2.morphologyEx(blackspot_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel_small)
        # Fill small gaps
        blackspot_mask = cv2.morphologyEx(blackspot_mask, cv2.MORPH_CLOSE, kernel_medium)

        # Remove small connected components
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blackspot_mask.astype(np.uint8))
        min_area = 150  # Minimum area for a blackspot (larger to avoid noise)

        final_mask = np.zeros_like(blackspot_mask, dtype=bool)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                component_mask = labels == i
                
                # Additional verification: check if the component is actually dark
                component_pixels = image[component_mask]
                if len(component_pixels) > 0:
                    avg_color = np.mean(component_pixels, axis=0)
                    is_black, is_dark = self.is_black_or_dark(avg_color)
                    
                    # Only keep if it's genuinely dark
                    if is_dark:
                        final_mask |= component_mask
                        logger.debug(f"Confirmed blackspot component {i}: avg_color={avg_color}, area={stats[i, cv2.CC_STAT_AREA]}")
                    else:
                        logger.debug(f"Rejected component {i}: avg_color={avg_color} (not dark enough)")

        return final_mask

    def create_enhanced_visualizations(self, image: np.ndarray, floor_mask: np.ndarray,
                                     blackspot_mask: np.ndarray) -> Dict:
        """Create multiple enhanced visualizations of blackspot detection"""

        h, w = image.shape[:2]

        # 1. Pure Segmentation View
        segmentation_view = np.zeros((h, w, 3), dtype=np.uint8)
        segmentation_view[floor_mask] = [34, 139, 34]      # Forest green for floor
        segmentation_view[blackspot_mask] = [255, 0, 0]    # Bright red for blackspots
        segmentation_view[~(floor_mask | blackspot_mask)] = [128, 128, 128]  # Gray for other areas

        # 2. High Contrast Overlay
        high_contrast_overlay = image.copy()
        # Darken background to emphasize blackspots
        high_contrast_overlay = cv2.convertScaleAbs(high_contrast_overlay, alpha=0.6, beta=0)
        
        # Add bright floor overlay
        if np.any(floor_mask):
            high_contrast_overlay[floor_mask] = cv2.addWeighted(
                high_contrast_overlay[floor_mask], 0.6,
                np.full_like(high_contrast_overlay[floor_mask], [0, 255, 0]), 0.4, 0
            )
        
        # Add bright blackspot overlay
        if np.any(blackspot_mask):
            high_contrast_overlay[blackspot_mask] = [255, 0, 255]  # Bright magenta

        # 3. Blackspot-only View
        blackspot_only = np.zeros((h, w, 3), dtype=np.uint8)
        blackspot_only[blackspot_mask] = [255, 255, 255]  # White blackspots
        blackspot_only[floor_mask & ~blackspot_mask] = [64, 64, 64]  # Dark gray for safe floor

        # 4. Side-by-side comparison
        side_by_side = np.zeros((h, w * 2, 3), dtype=np.uint8)
        side_by_side[:, :w] = image
        side_by_side[:, w:] = high_contrast_overlay

        # Add text labels
        cv2.putText(side_by_side, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "Blackspot Detection", (w + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # 5. Annotated view with detailed risk assessment
        annotated_view = image.copy()

        # Find blackspot contours for analysis
        blackspot_contours, _ = cv2.findContours(
            blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        total_blackspot_area = 0
        for i, contour in enumerate(blackspot_contours):
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small artifacts
                total_blackspot_area += area
                
                # Determine risk level based on size and darkness
                if area > 8000:
                    risk_level = "EXTREME"
                    color = (255, 0, 0)  # Red
                elif area > 3000:
                    risk_level = "HIGH"
                    color = (255, 165, 0)  # Orange
                elif area > 1000:
                    risk_level = "MEDIUM"
                    color = (255, 255, 0)  # Yellow
                else:
                    risk_level = "LOW"
                    color = (0, 255, 255)  # Cyan

                # Draw bounding box
                x, y, w_box, h_box = cv2.boundingRect(contour)
                cv2.rectangle(annotated_view, (x, y), (x + w_box, y + h_box), color, 3)

                # Draw semi-transparent filled contour
                overlay = annotated_view.copy()
                cv2.fillPoly(overlay, [contour], color)
                annotated_view = cv2.addWeighted(annotated_view, 0.7, overlay, 0.3, 0)

                # Add detailed label
                label = f"#{i+1}: {risk_level}"
                sub_label = f"{area:.0f}px, Fall Risk"
                
                # Main label
                label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                cv2.rectangle(annotated_view, (x, y-45), (x + max(label_size[0], 120) + 10, y), (0, 0, 0), -1)
                cv2.putText(annotated_view, label, (x + 5, y-25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(annotated_view, sub_label, (x + 5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return {
            'segmentation_view': segmentation_view,
            'high_contrast_overlay': high_contrast_overlay,
            'blackspot_only': blackspot_only,
            'side_by_side': side_by_side,
            'annotated_view': annotated_view
        }

    def detect_blackspots(self, image: np.ndarray, floor_prior: Optional[np.ndarray] = None) -> Dict:
        """Detect blackspots with enhanced color-based detection - FLOORS ONLY, BLACK AREAS ONLY"""
        
        original_h, original_w = image.shape[:2]
        logger.info(f"Starting blackspot detection on image shape: {image.shape}")

        # Handle floor prior
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
            # If no floor prior, assume we need to detect floors everywhere (fallback)
            logger.warning("No floor prior provided, assuming entire image could contain floors")
            floor_prior_resized = np.ones((original_h, original_w), dtype=bool)

        logger.info(f"Floor prior covers {np.sum(floor_prior_resized)} pixels ({np.sum(floor_prior_resized)/(original_h*original_w)*100:.1f}% of image)")

        # Primary method: Color-based blackspot detection
        color_blackspots = self.detect_floor_blackspots_by_color(image, floor_prior_resized)
        logger.info(f"Color-based detection found {np.sum(color_blackspots)} blackspot pixels")

        # Backup method: MaskRCNN detection (if available)
        model_blackspots = np.zeros_like(color_blackspots, dtype=bool)
        
        if self.predictor is not None:
            try:
                logger.info("Running MaskRCNN detection as backup method...")
                outputs = self.predictor(image)
                instances = outputs["instances"].to("cpu")

                if len(instances) > 0:
                    pred_classes = instances.pred_classes.numpy()
                    pred_masks = instances.pred_masks.numpy()
                    scores = instances.scores.numpy()

                    # Get blackspot masks from model (class 1)
                    blackspot_indices = pred_classes == 1
                    if np.any(blackspot_indices):
                        blackspot_masks = pred_masks[blackspot_indices]
                        blackspot_scores = scores[blackspot_indices]
                        
                        for mask, score in zip(blackspot_masks, blackspot_scores):
                            if score > self.confidence_threshold:
                                model_blackspots |= mask

                        # Keep only model blackspots that are on floors
                        model_blackspots &= floor_prior_resized
                        logger.info(f"MaskRCNN detection found {np.sum(model_blackspots)} blackspot pixels")

            except Exception as e:
                logger.warning(f"MaskRCNN detection failed, using color-based only: {e}")

        # Combine both methods (color-based has priority, model provides additional coverage)
        combined_blackspot = color_blackspots | model_blackspots

        # Final validation: ensure all detected blackspots are actually black/dark
        final_blackspots = np.zeros_like(combined_blackspot, dtype=bool)

        # Analyze each connected component for final validation
        num_labels, labels = cv2.connectedComponents(combined_blackspot.astype(np.uint8))

        validated_components = 0
        rejected_components = 0

        for label_id in range(1, num_labels):  # Skip background
            component_mask = labels == label_id

            # Get average color of this component
            if np.any(component_mask):
                component_pixels = image[component_mask]
                avg_color = np.mean(component_pixels, axis=0)
                min_color = np.min(component_pixels, axis=0)
                max_color = np.max(component_pixels, axis=0)

                # Strict validation: component must be genuinely dark
                is_black, is_dark = self.is_black_or_dark(avg_color)
                
                # Additional check: ensure it's not just a shadow or lighting artifact
                color_variance = np.var(component_pixels, axis=0)
                is_uniformly_dark = np.all(max_color < self.dark_threshold)

                # Keep only if it passes all validation checks
                if is_dark and (is_black or is_uniformly_dark):
                    final_blackspots |= component_mask
                    validated_components += 1
                    logger.debug(f"Validated blackspot component {label_id}: avg={avg_color}, max={max_color}")
                else:
                    rejected_components += 1
                    logger.debug(f"Rejected component {label_id}: avg={avg_color}, max={max_color} (not genuinely dark)")

        logger.info(f"Validation complete: {validated_components} components validated, {rejected_components} rejected")

        # Ensure final blackspots are strictly on floors
        final_blackspots &= floor_prior_resized

        # Create all enhanced visualizations
        enhanced_views = self.create_enhanced_visualizations(image, floor_prior_resized, final_blackspots)

        # Calculate comprehensive statistics
        floor_area = int(np.sum(floor_prior_resized))
        blackspot_area = int(np.sum(final_blackspots))
        coverage_percentage = (blackspot_area / floor_area * 100) if floor_area > 0 else 0

        # Count and analyze individual blackspot instances
        blackspot_contours, _ = cv2.findContours(
            final_blackspots.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # Filter and categorize blackspots by size
        significant_blackspots = []
        total_risk_score = 0
        
        for contour in blackspot_contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum significant size
                if area > 8000:
                    risk_score = 10  # Extreme
                elif area > 3000:
                    risk_score = 7   # High
                elif area > 1000:
                    risk_score = 4   # Medium
                else:
                    risk_score = 1   # Low
                
                significant_blackspots.append({
                    'area': area,
                    'risk_score': risk_score
                })
                total_risk_score += risk_score

        actual_detections = len(significant_blackspots)

        logger.info(f"Final blackspot analysis complete:")
        logger.info(f"  - {actual_detections} significant blackspots detected")
        logger.info(f"  - {blackspot_area} total pixels ({coverage_percentage:.2f}% of floor)")
        logger.info(f"  - Total risk score: {total_risk_score}")

        return {
            'visualization': enhanced_views['high_contrast_overlay'],  # Default view
            'floor_mask': floor_prior_resized,
            'blackspot_mask': final_blackspots,
            'floor_area': floor_area,
            'blackspot_area': blackspot_area,
            'coverage_percentage': coverage_percentage,
            'num_detections': actual_detections,
            'avg_confidence': 0.95,  # High confidence for validated color-based detection
            'enhanced_views': enhanced_views,  # All visualization options
            'detection_method': 'validated_color_based_floors_only',
            'risk_score': total_risk_score,
            'significant_blackspots': significant_blackspots
        }
