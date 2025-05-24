"""Enhanced blackspot detection using MaskRCNN and pixel-based methods."""

import numpy as np
import cv2
import torch
import logging
from typing import Dict, Tuple, Optional

from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor

logger = logging.getLogger(__name__)


class BlackspotDetector:
    """Detects black spots on floors using both ML and pixel-based methods"""
    
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.predictor = None
        self.initialized = False
        self.use_model = bool(model_path)
        
    def initialize(self, threshold: float = 0.5):
        """Initialize the detector"""
        if self.use_model and self.model_path:
            try:
                cfg = get_cfg()
                cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
                cfg.MODEL.WEIGHTS = self.model_path
                cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = threshold
                cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
                
                self.predictor = DefaultPredictor(cfg)
                self.initialized = True
                logger.info("MaskRCNN blackspot detector initialized")
                return True
            except Exception as e:
                logger.error(f"Failed to initialize MaskRCNN: {e}")
                self.use_model = False
        
        # Always return True for pixel-based fallback
        self.initialized = True
        return True
    
    def detect_blackspots(self, image: np.ndarray, floor_mask: np.ndarray) -> Dict:
        """Detect blackspots using both ML model and pixel-based methods"""
        h, w = image.shape[:2]
        results = {
            'blackspot_mask': np.zeros((h, w), dtype=bool),
            'enhanced_views': {},
            'num_detections': 0,
            'floor_area': np.sum(floor_mask),
            'blackspot_area': 0,
            'coverage_percentage': 0,
            'avg_confidence': 0,
            'risk_score': 0,
            'detection_method': 'none',
            'confidence_scores': {}
        }
        
        if np.sum(floor_mask) == 0:
            logger.warning("No floor area detected for blackspot analysis")
            return results
        
        # Method 1: MaskRCNN model detection
        model_mask = np.zeros((h, w), dtype=bool)
        model_confidence = 0.0
        
        if self.use_model and self.predictor is not None:
            try:
                outputs = self.predictor(image)
                instances = outputs["instances"]
                
                if len(instances) > 0:
                    masks = instances.pred_masks.cpu().numpy()
                    scores = instances.scores.cpu().numpy()
                    
                    for i, (mask, score) in enumerate(zip(masks, scores)):
                        # Check if this detection is on the floor
                        if np.sum(mask & floor_mask) > np.sum(mask) * 0.5:
                            model_mask |= mask
                            model_confidence = max(model_confidence, score)
                            logger.info(f"MaskRCNN: Detection {i} - confidence: {score:.3f}")
                
                results['confidence_scores']['maskrcnn'] = float(model_confidence)
            except Exception as e:
                logger.error(f"MaskRCNN detection failed: {e}")
        
        # Method 2: Pixel-based detection
        pixel_mask, pixel_confidence = self._detect_blackspots_pixel_based(image, floor_mask)
        results['confidence_scores']['pixel_based'] = float(pixel_confidence)
        
        # Combine both methods with OR logic
        combined_mask = model_mask | pixel_mask
        
        # Final filtering for size
        final_mask = self._filter_small_components(combined_mask, min_size=50)
        
        # Calculate statistics
        blackspot_pixels = np.sum(final_mask)
        results['blackspot_mask'] = final_mask
        results['blackspot_area'] = blackspot_pixels
        results['coverage_percentage'] = (blackspot_pixels / np.sum(floor_mask)) * 100 if np.sum(floor_mask) > 0 else 0
        
        # Count individual blackspots
        num_labels, labels = cv2.connectedComponents(final_mask.astype(np.uint8))
        results['num_detections'] = num_labels - 1  # Subtract background
        
        # Average confidence from both methods
        confidences = [c for c in [model_confidence, pixel_confidence] if c > 0]
        results['avg_confidence'] = np.mean(confidences) if confidences else 0.0
        
        # Determine detection method
        if model_mask.any() and pixel_mask.any():
            results['detection_method'] = 'combined'
        elif model_mask.any():
            results['detection_method'] = 'maskrcnn'
        elif pixel_mask.any():
            results['detection_method'] = 'pixel_based'
        
        # Risk score calculation
        if results['coverage_percentage'] > 10:
            results['risk_score'] = 10
        elif results['coverage_percentage'] > 5:
            results['risk_score'] = 8
        elif results['coverage_percentage'] > 2:
            results['risk_score'] = 6
        elif results['coverage_percentage'] > 0.5:
            results['risk_score'] = 4
        elif results['num_detections'] > 0:
            results['risk_score'] = 2
        else:
            results['risk_score'] = 0
        
        # Create enhanced visualizations
        results['enhanced_views'] = self._create_visualizations(image, final_mask, floor_mask)
        
        # Log final results
        logger.info(f"Blackspot detection complete: {results['num_detections']} spots, "
                   f"{results['coverage_percentage']:.2f}% coverage, "
                   f"method: {results['detection_method']}, "
                   f"confidence: MaskRCNN={model_confidence:.3f}, Pixel={pixel_confidence:.3f}")
        
        return results
    
    def _detect_blackspots_pixel_based(self, image: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhanced pixel-based blackspot detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Define thresholds for black detection
        very_black = 40
        black = 60
        dark = 80
        
        # Create masks for different darkness levels
        very_black_mask = (gray < very_black) & floor_mask
        black_mask = (gray < black) & floor_mask
        dark_mask = (gray < dark) & floor_mask
        
        # Check color variance to exclude grays
        b, g, r = cv2.split(image)
        color_variance = np.std([b, g, r], axis=0)
        not_gray = color_variance > 20  # Higher variance means not gray
        
        # Combine conditions: very black OR (black and colorful)
        blackspot_mask = very_black_mask | (black_mask & not_gray & floor_mask)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blackspot_mask = cv2.morphologyEx(blackspot_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        blackspot_mask = cv2.morphologyEx(blackspot_mask, cv2.MORPH_OPEN, kernel)
        
        # Calculate confidence
        if blackspot_mask.any():
            darkness_values = gray[blackspot_mask.astype(bool)]
            avg_darkness = np.mean(darkness_values)
            confidence = 1.0 - (avg_darkness / dark)
        else:
            confidence = 0.0
        
        return blackspot_mask.astype(bool), confidence
    
    def _filter_small_components(self, mask: np.ndarray, min_size: int = 50) -> np.ndarray:
        """Filter out small connected components"""
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        filtered_mask = np.zeros_like(mask, dtype=bool)
        for label_id in range(1, num_labels):
            component = labels == label_id
            if np.sum(component) >= min_size:
                filtered_mask |= component
        
        return filtered_mask
    
    def _create_visualizations(self, image: np.ndarray, blackspot_mask: np.ndarray, 
                              floor_mask: np.ndarray) -> Dict[str, np.ndarray]:
        """Create various visualization modes"""
        views = {}
        
        # High contrast overlay
        overlay = image.copy()
        overlay[blackspot_mask] = [255, 0, 0]  # Red for blackspots
        overlay[floor_mask & ~blackspot_mask] = overlay[floor_mask & ~blackspot_mask] * 0.7 + np.array([0, 255, 0]) * 0.3
        views['high_contrast_overlay'] = overlay
        
        # Side by side
        display = np.hstack([image, overlay])
        views['side_by_side'] = display
        
        # Blackspots only
        blackspot_only = np.ones_like(image) * 255
        blackspot_only[blackspot_mask] = [0, 0, 0]
        blackspot_only[floor_mask & ~blackspot_mask] = [200, 200, 200]
        views['blackspot_only'] = blackspot_only
        
        # Annotated view
        annotated = image.copy()
        contours, _ = cv2.findContours(blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(annotated, contours, -1, (255, 0, 0), 3)
        
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.putText(annotated, f"Blackspot {i+1}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        views['annotated_view'] = annotated
        
        # Segmentation view (if available)
        views['segmentation_view'] = overlay
        
        return views
