"""Enhanced blackspot detection using OneFormer segmentation and pixel-based methods."""

import numpy as np
import cv2
import torch
import logging
from typing import Dict, Tuple, Optional, Set

logger = logging.getLogger(__name__)


class BlackspotDetector:
    """Detects black spots ONLY on floors using segmentation and pixel-based methods"""
    
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.initialized = False
        
        # STRICT floor-only class IDs from ADE20K
        self.floor_class_ids = {
            3: 'floor',
            4: 'wood_floor', 
            28: 'rug',
            29: 'carpet',
            46: 'sand',  # For outdoor patios
            52: 'path',
            54: 'runway',
            78: 'mat',
            91: 'dirt_track'
        }
        
        # Classes that are NEVER floors (to double-check)
        self.non_floor_classes = {
            15: 'table', 33: 'desk', 56: 'pool_table', 64: 'coffee_table',
            5: 'ceiling', 6: 'sky', 0: 'wall', 1: 'building',
            7: 'bed', 10: 'cabinet', 19: 'chair', 23: 'sofa'
        }
        
    def initialize(self, threshold: float = 0.5):
        """Initialize the detector"""
        self.initialized = True
        logger.info("Blackspot detector initialized (enhanced floor-only mode)")
        return True
    
    def detect_blackspots(self, image: np.ndarray, floor_mask: np.ndarray = None, 
                         segmentation_mask: np.ndarray = None) -> Dict:
        """Detect blackspots ONLY on floor surfaces"""
        h, w = image.shape[:2]
        
        results = {
            'blackspot_mask': np.zeros((h, w), dtype=bool),
            'enhanced_views': {},
            'num_detections': 0,
            'floor_area': 0,
            'blackspot_area': 0,
            'coverage_percentage': 0,
            'avg_confidence': 0,
            'risk_score': 0,
            'detection_method': 'floor_only_enhanced',
            'confidence_scores': {},
            'floor_breakdown': {},
            'non_floor_blackspots_ignored': 0
        }
        
        # CRITICAL: Create strict floor mask
        if segmentation_mask is not None:
            strict_floor_mask = self._create_strict_floor_mask(segmentation_mask)
            
            # Log what we found
            for class_id in np.unique(segmentation_mask):
                if class_id in self.floor_class_ids:
                    pixels = np.sum(segmentation_mask == class_id)
                    logger.info(f"Floor type found: {self.floor_class_ids[class_id]} - {pixels} pixels")
                elif class_id in self.non_floor_classes:
                    pixels = np.sum(segmentation_mask == class_id)
                    logger.debug(f"Non-floor ignored: {self.non_floor_classes[class_id]} - {pixels} pixels")
        elif floor_mask is not None:
            strict_floor_mask = floor_mask
            logger.info(f"Using provided floor mask: {np.sum(floor_mask)} pixels")
        else:
            # Ultra-conservative fallback: only bottom 20% of image
            strict_floor_mask = np.zeros((h, w), dtype=bool)
            strict_floor_mask[int(h*0.8):, :] = True
            logger.warning(f"No segmentation available - using bottom 20% only: {np.sum(strict_floor_mask)} pixels")
        
        if np.sum(strict_floor_mask) == 0:
            logger.warning("No floor area detected - cannot detect blackspots")
            return results
        
        results['floor_area'] = np.sum(strict_floor_mask)
        
        # Detect ALL dark areas first (for comparison)
        all_blackspots = self._detect_all_blackspots(image)
        
        # Count how many blackspots we're ignoring (not on floor)
        non_floor_blackspots = all_blackspots & ~strict_floor_mask
        results['non_floor_blackspots_ignored'] = np.sum(non_floor_blackspots)
        
        if results['non_floor_blackspots_ignored'] > 0:
            logger.info(f"Ignored {results['non_floor_blackspots_ignored']} blackspot pixels on non-floor surfaces")
        
        # ONLY keep blackspots that are on the floor
        floor_only_blackspots = all_blackspots & strict_floor_mask
        
        # Enhanced filtering for floor blackspots
        blackspot_mask, confidence = self._enhance_floor_blackspots(
            image, floor_only_blackspots, strict_floor_mask
        )
        
        # Size filtering - remove tiny spots
        final_mask = self._filter_blackspots_by_size(blackspot_mask, min_dimension=50)
        
        # Calculate statistics
        blackspot_pixels = np.sum(final_mask)
        results['blackspot_mask'] = final_mask
        results['blackspot_area'] = blackspot_pixels
        results['coverage_percentage'] = (blackspot_pixels / results['floor_area']) * 100 if results['floor_area'] > 0 else 0
        results['avg_confidence'] = confidence
        
        # Count individual blackspots
        num_labels, labels = cv2.connectedComponents(final_mask.astype(np.uint8))
        results['num_detections'] = num_labels - 1
        
        # Risk score calculation
        coverage = results['coverage_percentage']
        if coverage > 15:
            results['risk_score'] = 10
        elif coverage > 10:
            results['risk_score'] = 8
        elif coverage > 5:
            results['risk_score'] = 6
        elif coverage > 2:
            results['risk_score'] = 4
        elif results['num_detections'] > 0:
            results['risk_score'] = 2
        else:
            results['risk_score'] = 0
        
        # Create visualizations
        results['enhanced_views'] = self._create_enhanced_visualizations(
            image, final_mask, strict_floor_mask, segmentation_mask, all_blackspots
        )
        
        # Floor type breakdown
        if segmentation_mask is not None:
            results['floor_breakdown'] = self._analyze_floor_types(
                segmentation_mask, final_mask
            )
        
        logger.info(f"Floor-only blackspot detection: {results['num_detections']} spots on floor, "
                   f"{results['coverage_percentage']:.2f}% floor coverage, "
                   f"ignored {results['non_floor_blackspots_ignored']} non-floor pixels")
        
        return results
    
    def _create_strict_floor_mask(self, segmentation: np.ndarray) -> np.ndarray:
        """Create a strict floor-only mask from segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        
        # Only include known floor classes
        for class_id in self.floor_class_ids.keys():
            floor_mask |= (segmentation == class_id)
        
        # Additional validation: remove any floor pixels that are too high in the image
        # (floors should generally be in the lower portion)
        h, w = floor_mask.shape
        for y in range(int(h * 0.3)):  # Top 30% of image
            if np.sum(floor_mask[y, :]) > w * 0.8:  # If most of row is "floor"
                # This might be a misclassification (e.g., ceiling classified as floor)
                logger.warning(f"Removing suspected misclassified floor at row {y}")
                floor_mask[y, :] = False
        
        return floor_mask
    
    def _detect_all_blackspots(self, image: np.ndarray) -> np.ndarray:
        """Detect ALL dark areas in the image (before floor filtering)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Multiple detection methods
        very_black = gray < 30
        dark = gray < 50
        
        # Color variance check
        b, g, r = cv2.split(image)
        color_variance = np.std([b, g, r], axis=0)
        low_variance = color_variance < 10
        
        # HSV black detection
        low_saturation = hsv[:, :, 1] < 30
        low_value = hsv[:, :, 2] < 50
        hsv_black = low_saturation & low_value
        
        # LAB detection
        lab_black = lab[:, :, 0] < 40
        
        # Combine all methods
        all_black = very_black | (dark & low_variance) | hsv_black | lab_black
        
        # Clean up with morphology
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        all_black = cv2.morphologyEx(all_black.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        all_black = cv2.morphologyEx(all_black, cv2.MORPH_OPEN, kernel)
        
        return all_black.astype(bool)
    
    def _enhance_floor_blackspots(self, image: np.ndarray, initial_mask: np.ndarray, 
                                 floor_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhance blackspot detection specifically for floor areas"""
        if not np.any(initial_mask):
            return initial_mask, 0.0
        
        # Additional validation for floor blackspots
        enhanced_mask = initial_mask.copy()
        
        # Check each connected component
        num_labels, labels = cv2.connectedComponents(initial_mask.astype(np.uint8))
        
        for label_id in range(1, num_labels):
            component = labels == label_id
            
            # Validate this is really a blackspot on floor
            component_pixels = image[component]
            if len(component_pixels) > 0:
                mean_brightness = np.mean(cv2.cvtColor(
                    component_pixels.reshape(-1, 1, 3), cv2.COLOR_RGB2GRAY
                ))
                
                # If too bright, remove it
                if mean_brightness > 60:
                    enhanced_mask[component] = False
                    logger.debug(f"Removed false blackspot with brightness {mean_brightness}")
        
        # Calculate confidence
        if np.any(enhanced_mask):
            darkness_values = cv2.cvtColor(image[enhanced_mask].reshape(-1, 1, 3), 
                                         cv2.COLOR_RGB2GRAY).flatten()
            avg_darkness = np.mean(darkness_values)
            confidence = 1.0 - (avg_darkness / 50)
        else:
            confidence = 0.0
        
        return enhanced_mask, confidence
    
    def _filter_blackspots_by_size(self, mask: np.ndarray, min_dimension: int = 50) -> np.ndarray:
        """Filter blackspots by minimum size"""
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        filtered_mask = np.zeros_like(mask, dtype=bool)
        
        for label_id in range(1, num_labels):
            component = labels == label_id
            
            y_coords, x_coords = np.where(component)
            if len(y_coords) == 0:
                continue
                
            height = np.max(y_coords) - np.min(y_coords) + 1
            width = np.max(x_coords) - np.min(x_coords) + 1
            
            if width >= min_dimension or height >= min_dimension:
                filtered_mask |= component
        
        return filtered_mask
    
    def _analyze_floor_types(self, segmentation: np.ndarray, blackspot_mask: np.ndarray) -> Dict:
        """Analyze blackspots by floor type"""
        breakdown = {}
        
        for class_id, class_name in self.floor_class_ids.items():
            floor_type_mask = segmentation == class_id
            floor_type_pixels = np.sum(floor_type_mask)
            
            if floor_type_pixels > 0:
                blackspots_on_type = blackspot_mask & floor_type_mask
                blackspot_pixels_on_type = np.sum(blackspots_on_type)
                coverage_on_type = (blackspot_pixels_on_type / floor_type_pixels) * 100
                
                breakdown[class_name] = {
                    'total_pixels': floor_type_pixels,
                    'blackspot_pixels': blackspot_pixels_on_type,
                    'coverage_percentage': coverage_on_type
                }
        
        return breakdown
    
    def _create_enhanced_visualizations(self, image: np.ndarray, blackspot_mask: np.ndarray, 
                                      floor_mask: np.ndarray, segmentation_mask: np.ndarray,
                                      all_blackspots: np.ndarray) -> Dict[str, np.ndarray]:
        """Create comprehensive visualizations"""
        views = {}
        
        # Main visualization - blackspots on floor only
        overlay = image.copy()
        overlay[blackspot_mask] = [255, 0, 0]  # Red for floor blackspots
        overlay[floor_mask & ~blackspot_mask] = overlay[floor_mask & ~blackspot_mask] * 0.8 + np.array([0, 255, 0]) * 0.2
        views['high_contrast_overlay'] = overlay
        
        # Show what we ignored (non-floor blackspots)
        ignored_viz = image.copy()
        non_floor_black = all_blackspots & ~floor_mask
        if np.any(non_floor_black):
            ignored_viz[non_floor_black] = ignored_viz[non_floor_black] * 0.5 + np.array([128, 128, 128]) * 0.5
        ignored_viz[blackspot_mask] = [255, 0, 0]  # Still show floor blackspots
        views['ignored_blackspots'] = ignored_viz
        
        # Floor types visualization
        if segmentation_mask is not None:
            floor_viz = image.copy()
            colors = [(255, 200, 200), (200, 255, 200), (200, 200, 255), 
                     (255, 255, 200), (255, 200, 255), (200, 255, 255)]
            
            for i, (class_id, class_name) in enumerate(self.floor_class_ids.items()):
                mask = segmentation_mask == class_id
                if np.any(mask):
                    color = colors[i % len(colors)]
                    floor_viz[mask] = floor_viz[mask] * 0.6 + np.array(color) * 0.4
            
            floor_viz[blackspot_mask] = [255, 0, 0]
            views['floor_types'] = floor_viz
        
        return views
