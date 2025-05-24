"""Enhanced blackspot detection using OneFormer segmentation and pixel-based methods."""

import numpy as np
import cv2
import torch
import logging
from typing import Dict, Tuple, Optional

logger = logging.getLogger(__name__)


class BlackspotDetector:
    """Detects black spots on floors using OneFormer segmentation and pixel-based methods"""
    
    def __init__(self, model_path: str = ""):
        self.model_path = model_path
        self.initialized = False
        
        # Floor-related class IDs from ADE20K (oneformer uses these)
        self.floor_class_ids = {
            3: 'floor',
            4: 'wood_floor', 
            13: 'earth',
            28: 'rug',
            29: 'field',
            52: 'path',
            53: 'stairs',
            54: 'runway',
            78: 'mat',
            91: 'dirt_track'
        }
        
    def initialize(self, threshold: float = 0.5):
        """Initialize the detector"""
        # Always return True since we use pixel-based detection
        self.initialized = True
        logger.info("Blackspot detector initialized (pixel-based mode)")
        return True
    
    def detect_blackspots(self, image: np.ndarray, floor_mask: np.ndarray = None, 
                         segmentation_mask: np.ndarray = None) -> Dict:
        """Detect blackspots using enhanced pixel-based methods with floor segmentation"""
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
            'detection_method': 'pixel_based_enhanced',
            'confidence_scores': {},
            'floor_breakdown': {}
        }
        
        # Extract floor areas from segmentation if available
        if segmentation_mask is not None:
            enhanced_floor_mask = self._extract_floor_areas_from_segmentation(segmentation_mask)
            logger.info(f"Enhanced floor extraction: {np.sum(enhanced_floor_mask)} pixels from segmentation")
        elif floor_mask is not None:
            enhanced_floor_mask = floor_mask
            logger.info(f"Using provided floor mask: {np.sum(enhanced_floor_mask)} pixels")
        else:
            # Fallback: assume bottom 30% of image is floor
            enhanced_floor_mask = np.zeros((h, w), dtype=bool)
            enhanced_floor_mask[int(h*0.7):, :] = True
            logger.info(f"Using fallback floor assumption: {np.sum(enhanced_floor_mask)} pixels")
        
        if np.sum(enhanced_floor_mask) == 0:
            logger.warning("No floor area detected for blackspot analysis")
            return results
        
        results['floor_area'] = np.sum(enhanced_floor_mask)
        
        # Enhanced blackspot detection
        blackspot_mask, confidence = self._detect_blackspots_enhanced(image, enhanced_floor_mask)
        
        # Filter blackspots by size (minimum 50x50 pixels)
        final_mask = self._filter_blackspots_by_size(blackspot_mask, min_dimension=50)
        
        # Calculate statistics
        blackspot_pixels = np.sum(final_mask)
        results['blackspot_mask'] = final_mask
        results['blackspot_area'] = blackspot_pixels
        results['coverage_percentage'] = (blackspot_pixels / results['floor_area']) * 100
        results['avg_confidence'] = confidence
        
        # Count individual blackspots
        num_labels, labels = cv2.connectedComponents(final_mask.astype(np.uint8))
        results['num_detections'] = num_labels - 1  # Subtract background
        
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
        
        # Create enhanced visualizations
        results['enhanced_views'] = self._create_enhanced_visualizations(
            image, final_mask, enhanced_floor_mask, segmentation_mask
        )
        
        # Floor breakdown if segmentation available
        if segmentation_mask is not None:
            results['floor_breakdown'] = self._analyze_floor_types(
                segmentation_mask, final_mask
            )
        
        logger.info(f"Enhanced blackspot detection complete: {results['num_detections']} spots, "
                   f"{results['coverage_percentage']:.2f}% coverage, "
                   f"confidence: {confidence:.3f}")
        
        return results
    
    def _extract_floor_areas_from_segmentation(self, segmentation: np.ndarray) -> np.ndarray:
        """Extract floor areas from OneFormer segmentation"""
        floor_mask = np.zeros_like(segmentation, dtype=bool)
        
        # Check for each floor-related class
        detected_classes = []
        for class_id, class_name in self.floor_class_ids.items():
            class_pixels = np.sum(segmentation == class_id)
            if class_pixels > 0:
                floor_mask |= (segmentation == class_id)
                detected_classes.append(f"{class_name}({class_pixels}px)")
        
        if detected_classes:
            logger.info(f"Floor classes detected: {', '.join(detected_classes)}")
        
        return floor_mask
    
    def _detect_blackspots_enhanced(self, image: np.ndarray, floor_mask: np.ndarray) -> Tuple[np.ndarray, float]:
        """Enhanced blackspot detection with multiple methods"""
        
        # Convert to different color spaces for analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Method 1: Pure black detection (very strict)
        very_black_threshold = 25
        very_black_mask = (gray < very_black_threshold) & floor_mask
        
        # Method 2: Dark detection with color variance check
        dark_threshold = 50
        dark_mask = (gray < dark_threshold) & floor_mask
        
        # Color variance check to distinguish black from dark gray
        b, g, r = cv2.split(image)
        color_variance = np.std([b, g, r], axis=0)
        is_colorful = color_variance > 15  # Areas with color variation
        is_truly_black = color_variance < 10  # Very low color variation = black/gray
        
        # Method 3: HSV-based black detection
        # Black areas have low saturation and low value
        low_saturation = hsv[:, :, 1] < 30  # Low saturation
        low_value = hsv[:, :, 2] < dark_threshold  # Low brightness
        hsv_black_mask = low_saturation & low_value & floor_mask
        
        # Method 4: LAB color space detection
        # In LAB, black areas have low L (lightness) values
        lab_dark_mask = (lab[:, :, 0] < 40) & floor_mask
        
        # Combine methods with priority
        blackspot_candidates = (
            very_black_mask |  # Highest priority: very black areas
            (dark_mask & is_truly_black) |  # Dark areas that are truly black, not gray
            (hsv_black_mask & ~is_colorful) |  # HSV black that isn't colorful
            lab_dark_mask  # LAB dark areas
        )
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        blackspot_candidates = cv2.morphologyEx(
            blackspot_candidates.astype(np.uint8), 
            cv2.MORPH_CLOSE, 
            kernel
        )
        blackspot_candidates = cv2.morphologyEx(
            blackspot_candidates, 
            cv2.MORPH_OPEN, 
            kernel
        )
        
        # Calculate confidence based on detection strength
        if np.any(blackspot_candidates):
            # Higher confidence for very black areas
            very_black_ratio = np.sum(very_black_mask) / max(1, np.sum(blackspot_candidates))
            darkness_values = gray[blackspot_candidates.astype(bool)]
            avg_darkness = np.mean(darkness_values)
            
            # Confidence calculation
            darkness_confidence = 1.0 - (avg_darkness / 50)  # Normalize by threshold
            coverage_confidence = min(1.0, np.sum(blackspot_candidates) / 1000)  # Size factor
            purity_confidence = very_black_ratio  # How much is very black vs just dark
            
            confidence = (darkness_confidence * 0.5 + 
                         coverage_confidence * 0.2 + 
                         purity_confidence * 0.3)
        else:
            confidence = 0.0
        
        return blackspot_candidates.astype(bool), confidence
    
    def _filter_blackspots_by_size(self, mask: np.ndarray, min_dimension: int = 50) -> np.ndarray:
        """Filter blackspots to only include those larger than min_dimension in width or height"""
        num_labels, labels = cv2.connectedComponents(mask.astype(np.uint8))
        
        filtered_mask = np.zeros_like(mask, dtype=bool)
        valid_spots = 0
        
        for label_id in range(1, num_labels):
            component = labels == label_id
            
            # Get bounding box to check dimensions
            y_coords, x_coords = np.where(component)
            if len(y_coords) == 0:
                continue
                
            min_y, max_y = np.min(y_coords), np.max(y_coords)
            min_x, max_x = np.min(x_coords), np.max(x_coords)
            
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            
            # Keep if either dimension is >= min_dimension
            if width >= min_dimension or height >= min_dimension:
                filtered_mask |= component
                valid_spots += 1
                logger.debug(f"Blackspot {label_id}: {width}x{height}px - KEPT")
            else:
                logger.debug(f"Blackspot {label_id}: {width}x{height}px - filtered out (too small)")
        
        logger.info(f"Size filtering: {valid_spots}/{num_labels-1} blackspots kept (>={min_dimension}px)")
        return filtered_mask
    
    def _analyze_floor_types(self, segmentation: np.ndarray, blackspot_mask: np.ndarray) -> Dict:
        """Analyze which types of floors have blackspots"""
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
                                      floor_mask: np.ndarray, segmentation_mask: np.ndarray = None) -> Dict[str, np.ndarray]:
        """Create comprehensive visualizations"""
        views = {}
        
        # High contrast overlay
        overlay = image.copy()
        overlay[blackspot_mask] = [255, 0, 0]  # Red for blackspots
        overlay[floor_mask & ~blackspot_mask] = overlay[floor_mask & ~blackspot_mask] * 0.8 + np.array([0, 255, 0]) * 0.2
        views['high_contrast_overlay'] = overlay
        
        # Side by side comparison
        views['side_by_side'] = np.hstack([image, overlay])
        
        # Blackspots only on white background
        blackspot_only = np.ones_like(image) * 255
        blackspot_only[blackspot_mask] = [0, 0, 0]  # Black spots
        blackspot_only[floor_mask & ~blackspot_mask] = [200, 200, 200]  # Gray floor areas
        views['blackspot_only'] = blackspot_only
        
        # Annotated view with labels
        annotated = image.copy()
        contours, _ = cv2.findContours(blackspot_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for i, contour in enumerate(contours):
            # Draw contour
            cv2.drawContours(annotated, [contour], -1, (255, 0, 0), 3)
            
            # Add size label
            x, y, w, h = cv2.boundingRect(contour)
            area = cv2.contourArea(contour)
            if area > 50:  # Only label significant spots
                label = f"Spot {i+1}: {w}×{h}px"
                cv2.putText(annotated, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(annotated, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
        
        views['annotated_view'] = annotated
        
        # Floor segmentation view if available
        if segmentation_mask is not None:
            seg_view = image.copy()
            
            # Color different floor types
            colors = [(255, 100, 100), (100, 255, 100), (100, 100, 255), 
                     (255, 255, 100), (255, 100, 255), (100, 255, 255)]
            
            for i, (class_id, class_name) in enumerate(self.floor_class_ids.items()):
                class_mask = segmentation_mask == class_id
                if np.any(class_mask):
                    color = colors[i % len(colors)]
                    seg_view[class_mask] = seg_view[class_mask] * 0.6 + np.array(color) * 0.4
            
            # Highlight blackspots on top
            seg_view[blackspot_mask] = [255, 0, 0]
            views['segmentation_view'] = seg_view
        else:
            views['segmentation_view'] = overlay
        
        return views
