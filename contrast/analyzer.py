"""Enhanced contrast analysis for Alzheimer's-friendly environments with strict adjacency detection."""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from scipy import ndimage
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
import colorsys
import logging

logger = logging.getLogger(__name__)


class RobustContrastAnalyzer:
    """Contrast analyzer with strict adjacency detection and Alzheimer's-optimized thresholds"""
    
    def __init__(self, wcag_threshold: float = 4.5, alzheimer_threshold: float = 7.0, 
                 color_similarity_threshold: float = 30.0, perceptual_threshold: float = 0.15):
        # Contrast thresholds
        self.wcag_threshold = wcag_threshold
        self.alzheimer_threshold = alzheimer_threshold
        
        # Color similarity thresholds (stricter for Alzheimer's)
        self.color_similarity_threshold = color_similarity_threshold
        self.perceptual_threshold = perceptual_threshold
        self.hue_difference_threshold = 30.0  # Minimum 30° hue difference as per guidelines
        self.luminance_difference_threshold = 0.2  # 20% minimum luminance difference
        
        # Comprehensive ADE20K class mappings for indoor environments
        self.semantic_classes = {
            'floor': [3, 4, 13, 28, 29, 52, 53, 54, 78, 91],  # floor, wood floor, earth, rug, field, path, stairs, runway, mat, dirt track
            'wall': [0, 1, 9, 25, 26, 43, 93, 96, 97, 102],  # wall, building, brick, house, sea(wall art), signboard, pole, wallpaper, tile wall, painting
            'ceiling': [5, 6],  # ceiling, sky(skylight)
            'furniture': {
                'seating': [10, 18, 19, 23, 30, 31, 75, 97],  # sofa, armchair, chair, couch, cushion, seat, swivel chair, ottoman
                'tables': [15, 33, 56, 64, 77],  # table, desk, pool table, coffee table, bar
                'storage': [11, 24, 35, 41, 44, 62, 112],  # cabinet, shelf, wardrobe, box, chest of drawers, bookcase, basket
                'beds': [7, 117],  # bed, cradle
            },
            'doors_windows': [8, 14, 58, 74],  # window, door, screen door, screen window
            'stairs_ramps': [53, 54, 59, 96, 121],  # stairs, runway, stairway, escalator, step
            'bathroom': [37, 47, 60, 65, 107, 145],  # bathtub, sink, river(water feature), toilet, washer, shower
            'kitchen': [45, 50, 70, 71, 118, 119, 124, 125, 129],  # counter, refrigerator, countertop, stove, oven, ball(decorative), microwave, pot, dishwasher
            'lighting': [36, 82, 85, 87, 134],  # lamp, light, chandelier, streetlight, sconce
            'textiles': [18, 39, 49, 57, 63, 81, 83, 84, 131],  # curtain, cushion, blanket, pillow, blind, towel, pillow, blanket, bedding
            'decorative': [22, 27, 66, 75, 76, 100, 132, 135],  # painting, mirror, flower, picture, poster, artwork, sculpture, vase
            'electronics': [74, 89, 130, 141, 143],  # computer, television, screen, crt screen, monitor
            'rails_barriers': [32, 38, 95],  # fence, railing, bannister
            'floors_coverings': [3, 4, 13, 28, 46, 52, 78, 91],  # floor, wood floor, earth, rug, sand, path, mat, dirt track
            'fixtures': [49, 104, 146],  # fireplace, fountain, radiator
            'outdoor_elements': [16, 17, 21, 34, 66, 72, 87, 88],  # mountain(wall art), plant, water, rock, flower, palm, plant pot, booth
        }
        
        # Flatten furniture subcategories
        self.all_classes = {}
        for category, items in self.semantic_classes.items():
            if isinstance(items, dict):
                for subcat, ids in items.items():
                    for id in ids:
                        self.all_classes[id] = f"{category}_{subcat}"
            else:
                for id in items:
                    self.all_classes[id] = category
        
        # Critical safety relationships (MUST have good contrast)
        self.critical_safety_pairs = {
            ('floor', 'stairs_ramps'), ('stairs_ramps', 'floor'),
            ('floor', 'doors_windows'), ('doors_windows', 'floor'),
            ('wall', 'doors_windows'), ('doors_windows', 'wall'),
            ('floor', 'bathroom'), ('bathroom', 'floor'),
            ('stairs_ramps', 'rails_barriers'), ('rails_barriers', 'stairs_ramps'),
        }
        
        # High priority pairs
        self.high_priority_pairs = {
            ('floor', 'furniture_seating'), ('furniture_seating', 'floor'),
            ('floor', 'furniture_tables'), ('furniture_tables', 'floor'),
            ('floor', 'furniture_storage'), ('furniture_storage', 'floor'),
            ('wall', 'furniture_seating'), ('furniture_seating', 'wall'),
            ('wall', 'furniture_tables'), ('furniture_tables', 'wall'),
            ('wall', 'decorative'), ('decorative', 'wall'),
            ('wall', 'electronics'), ('electronics', 'wall'),
        }
    
    def get_object_category(self, class_id: int) -> Optional[str]:
        """Map segmentation class to object category"""
        return self.all_classes.get(class_id)
    
    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG contrast ratio"""
        def relative_luminance(rgb):
            # Normalize to 0-1
            rgb_norm = rgb / 255.0
            # Apply gamma correction
            rgb_linear = np.where(rgb_norm <= 0.03928,
                                rgb_norm / 12.92,
                                ((rgb_norm + 0.055) / 1.055) ** 2.4)
            # Calculate luminance
            return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])

        lum1 = relative_luminance(color1)
        lum2 = relative_luminance(color2)

        # Ensure correct ratio calculation
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)
    
    def calculate_color_metrics(self, color1: np.ndarray, color2: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive color difference metrics"""
        # RGB to HSV conversion
        hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        
        # Hue difference (circular, in degrees)
        hue1 = hsv1[0] * 2  # Convert to 0-360
        hue2 = hsv2[0] * 2
        hue_diff = min(abs(hue1 - hue2), 360 - abs(hue1 - hue2))
        
        # Saturation and Value differences
        sat_diff = abs(hsv1[1] - hsv2[1]) / 255.0
        val_diff = abs(hsv1[2] - hsv2[2]) / 255.0
        
        # Luminance difference
        lum1 = self.calculate_luminance(color1)
        lum2 = self.calculate_luminance(color2)
        lum_diff = abs(lum1 - lum2)
        
        # RGB Euclidean distance
        rgb_distance = euclidean(color1, color2)
        
        return {
            'wcag_contrast': self.calculate_wcag_contrast(color1, color2),
            'hue_difference': hue_diff,
            'saturation_difference': sat_diff,
            'value_difference': val_diff,
            'luminance_difference': lum_diff,
            'rgb_distance': rgb_distance,
            'is_similar_hue': hue_diff < self.hue_difference_threshold,
            'is_low_contrast': self.calculate_wcag_contrast(color1, color2) < self.alzheimer_threshold
        }
    
    def calculate_luminance(self, color: np.ndarray) -> float:
        """Calculate relative luminance for a color"""
        rgb_norm = color / 255.0
        rgb_linear = np.where(rgb_norm <= 0.03928,
                            rgb_norm / 12.92,
                            ((rgb_norm + 0.055) / 1.055) ** 2.4)
        return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])
    
    def find_true_adjacency(self, mask1: np.ndarray, mask2: np.ndarray, 
                           min_boundary_length: int = 50) -> Tuple[bool, np.ndarray]:
        """Find true adjacency between segments with strict criteria"""
        # Direct adjacency check with minimal dilation
        kernel = np.ones((3, 3), np.uint8)
        
        # Single pixel dilation to find touching boundaries
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)
        
        # Find overlap
        boundary = dilated1 & dilated2
        
        # Remove small noise
        kernel_small = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel_small)
        
        # Check if boundary is significant
        boundary_pixels = np.sum(boundary)
        
        # Calculate shared boundary length
        if boundary_pixels > min_boundary_length:
            # Validate that this is a real boundary (not just corner touching)
            contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_boundary = np.zeros_like(boundary)
            for contour in contours:
                # Check contour length (perimeter)
                perimeter = cv2.arcLength(contour, True)
                if perimeter > 20:  # Minimum perimeter for valid boundary
                    cv2.fillPoly(valid_boundary, [contour], 1)
            
            is_adjacent = np.sum(valid_boundary) > min_boundary_length
            return is_adjacent, valid_boundary.astype(bool)
        
        return False, np.zeros_like(boundary, dtype=bool)
    
    def extract_representative_color(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract representative color using clustering to avoid outliers"""
        if not np.any(mask):
            return np.array([128, 128, 128])
        
        # Get masked pixels
        masked_pixels = image[mask]
        if len(masked_pixels) < 10:
            return np.mean(masked_pixels, axis=0).astype(int)
        
        # Sample for efficiency
        if len(masked_pixels) > 1000:
            indices = np.random.choice(len(masked_pixels), 1000, replace=False)
            masked_pixels = masked_pixels[indices]
        
        # Use clustering to find dominant color
        try:
            clustering = DBSCAN(eps=30, min_samples=20).fit(masked_pixels)
            labels = clustering.labels_
            
            # Find largest cluster
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) > 0:
                dominant_label = unique_labels[np.argmax(counts)]
                dominant_pixels = masked_pixels[labels == dominant_label]
                return np.median(dominant_pixels, axis=0).astype(int)
        except:
            pass
        
        # Fallback to median
        return np.median(masked_pixels, axis=0).astype(int)
    
    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """Analyze contrast with strict adjacency detection"""
        h, w = segmentation.shape
        
        results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'low_issues': [],
            'good_contrasts': [],
            'visualization': image.copy(),
            'boundary_mask': np.zeros((h, w), dtype=bool),
            'statistics': {},
            'detailed_analysis': []
        }
        
        # Build segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}
        
        logger.info(f"Analyzing {len(unique_segments)} unique segments")
        
        # Extract segment properties
        for seg_id in unique_segments:
            if seg_id == 0 or seg_id == 255:  # Skip background/ignore
                continue
            
            mask = segmentation == seg_id
            pixel_count = np.sum(mask)
            
            # Skip tiny segments
            if pixel_count < 100:
                continue
            
            category = self.get_object_category(seg_id)
            if category is None:
                continue
            
            # Extract color
            color = self.extract_representative_color(image, mask)
            
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': pixel_count,
                'class_id': seg_id
            }
        
        logger.info(f"Processing {len(segment_info)} valid segments")
        
        # Analyze only truly adjacent pairs
        total_pairs_checked = 0
        adjacent_pairs_found = 0
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        segment_ids = list(segment_info.keys())
        
        for i, seg_id1 in enumerate(segment_ids):
            for seg_id2 in segment_ids[i+1:]:
                total_pairs_checked += 1
                
                info1 = segment_info[seg_id1]
                info2 = segment_info[seg_id2]
                
                # Check true adjacency
                is_adjacent, boundary = self.find_true_adjacency(
                    info1['mask'], info2['mask']
                )
                
                if not is_adjacent:
                    continue
                
                adjacent_pairs_found += 1
                
                # Calculate color metrics
                metrics = self.calculate_color_metrics(info1['color'], info2['color'])
                
                # Determine if there's an issue
                severity = None
                issues = []
                
                # Check Alzheimer's criteria
                if metrics['wcag_contrast'] < self.alzheimer_threshold:
                    issues.append(f"Low contrast: {metrics['wcag_contrast']:.2f}:1 (need ≥{self.alzheimer_threshold}:1)")
                    if metrics['wcag_contrast'] < self.wcag_threshold:
                        severity = 'critical'
                    else:
                        severity = 'high'
                
                # Check hue difference
                if metrics['is_similar_hue']:
                    issues.append(f"Similar hues: {metrics['hue_difference']:.1f}° apart (need ≥30°)")
                    severity = 'critical' if severity is None else severity
                
                # Check luminance difference
                if metrics['luminance_difference'] < self.luminance_difference_threshold:
                    issues.append(f"Low luminance difference: {metrics['luminance_difference']:.2f} (need ≥0.2)")
                    severity = 'high' if severity is None else severity
                
                # Check relationship priority
                cat1, cat2 = info1['category'], info2['category']
                rel_tuple = (cat1, cat2) if (cat1, cat2) in self.critical_safety_pairs else (cat2, cat1)
                
                if rel_tuple in self.critical_safety_pairs and severity:
                    severity = 'critical'
                elif rel_tuple in self.high_priority_pairs and severity == 'medium':
                    severity = 'high'
                
                # Create issue record if problems found
                if issues:
                    issue = {
                        'categories': (cat1, cat2),
                        'metrics': metrics,
                        'boundary_pixels': np.sum(boundary),
                        'severity': severity,
                        'issues': issues
                    }
                    
                    # Visualize on boundary with appropriate tint
                    if severity == 'critical':
                        # Red tint for critical
                        tint_color = np.array([255, 200, 200])
                        results['critical_issues'].append(issue)
                        issue_counts['critical'] += 1
                    elif severity == 'high':
                        # Orange tint for high
                        tint_color = np.array([255, 220, 200])
                        results['high_issues'].append(issue)
                        issue_counts['high'] += 1
                    elif severity == 'medium':
                        # Yellow tint for medium
                        tint_color = np.array([255, 255, 200])
                        results['medium_issues'].append(issue)
                        issue_counts['medium'] += 1
                    else:
                        # Light yellow for low
                        tint_color = np.array([255, 255, 220])
                        results['low_issues'].append(issue)
                        issue_counts['low'] += 1
                    
                    # Apply tint to boundary area only
                    results['boundary_mask'][boundary] = True
                    results['visualization'][boundary] = (
                        results['visualization'][boundary] * 0.5 + tint_color * 0.5
                    ).astype(np.uint8)
                else:
                    # Good contrast
                    if metrics['wcag_contrast'] >= self.alzheimer_threshold:
                        results['good_contrasts'].append({
                            'categories': (cat1, cat2),
                            'contrast_ratio': metrics['wcag_contrast'],
                            'hue_difference': metrics['hue_difference']
                        })
        
        # Compile statistics
        results['statistics'] = {
            'total_segments': len(segment_info),
            'total_pairs_checked': total_pairs_checked,
            'adjacent_pairs': adjacent_pairs_found,
            'total_issues': sum(issue_counts.values()),
            'critical_count': issue_counts['critical'],
            'high_count': issue_counts['high'],
            'medium_count': issue_counts['medium'],
            'low_count': issue_counts['low'],
            'good_contrast_count': len(results['good_contrasts']),
            'wcag_threshold': self.wcag_threshold,
            'alzheimer_threshold': self.alzheimer_threshold
        }
        
        logger.info(f"Contrast analysis complete: {adjacent_pairs_found} adjacent pairs, "
                   f"{sum(issue_counts.values())} issues found")
        
        return results
