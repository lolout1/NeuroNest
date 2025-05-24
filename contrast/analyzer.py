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
    """Contrast analyzer with strict adjacency detection for ALL object pairs"""
    
    def __init__(self, wcag_threshold: float = 4.5, alzheimer_threshold: float = 7.0, 
                 color_similarity_threshold: float = 30.0, perceptual_threshold: float = 0.15):
        self.wcag_threshold = wcag_threshold
        self.alzheimer_threshold = alzheimer_threshold
        
        # Color similarity thresholds per guidelines
        self.color_similarity_threshold = color_similarity_threshold
        self.perceptual_threshold = perceptual_threshold
        self.hue_difference_threshold = 30.0  # Minimum 30° hue difference
        self.luminance_difference_threshold = 0.2  # 20% minimum
        self.saturation_difference_threshold = 0.2  # 20% minimum
        
        # Comprehensive ADE20K mappings for ALL indoor objects
        self.object_classes = {
            # Floors
            3: 'floor', 4: 'wood_floor', 28: 'rug', 29: 'carpet', 78: 'mat', 
            46: 'sand', 52: 'path', 54: 'runway', 91: 'dirt_track',
            
            # Walls and structural
            0: 'wall', 1: 'building', 9: 'brick_wall', 26: 'wall_art', 
            43: 'signboard', 96: 'wallpaper', 97: 'tile_wall',
            
            # Ceiling
            5: 'ceiling', 6: 'skylight',
            
            # Furniture - Seating
            10: 'sofa', 18: 'armchair', 19: 'chair', 23: 'couch', 
            30: 'cushion', 31: 'seat', 75: 'swivel_chair', 97: 'ottoman',
            
            # Furniture - Tables
            15: 'table', 33: 'desk', 56: 'pool_table', 64: 'coffee_table', 
            77: 'bar', 99: 'buffet',
            
            # Furniture - Storage
            11: 'cabinet', 24: 'shelf', 35: 'wardrobe', 41: 'box', 
            44: 'chest_of_drawers', 62: 'bookcase', 112: 'basket',
            
            # Beds
            7: 'bed', 117: 'cradle',
            
            # Doors and Windows
            8: 'window', 14: 'door', 58: 'screen_door', 74: 'window_screen',
            
            # Stairs and Rails
            53: 'stairs', 54: 'runway', 59: 'stairway', 96: 'escalator', 
            121: 'step', 32: 'fence', 38: 'railing', 95: 'bannister',
            
            # Bathroom
            37: 'bathtub', 47: 'sink', 65: 'toilet', 107: 'washer', 
            145: 'shower', 60: 'water_feature',
            
            # Kitchen
            45: 'counter', 50: 'refrigerator', 70: 'countertop', 71: 'stove', 
            118: 'oven', 124: 'microwave', 125: 'pot', 129: 'dishwasher',
            
            # Lighting
            36: 'lamp', 82: 'light', 85: 'chandelier', 87: 'streetlight', 
            134: 'sconce',
            
            # Textiles
            18: 'curtain', 39: 'cushion', 57: 'pillow', 63: 'blind', 
            81: 'towel', 131: 'blanket',
            
            # Decorative
            22: 'painting', 27: 'mirror', 66: 'flower', 100: 'poster', 
            132: 'sculpture', 135: 'vase',
            
            # Electronics
            74: 'computer', 89: 'television', 130: 'screen', 141: 'crt_screen', 
            143: 'monitor',
            
            # Plants
            17: 'plant', 72: 'palm', 87: 'potted_plant',
            
            # Fixtures
            49: 'fireplace', 104: 'fountain', 146: 'radiator'
        }
        
        # Critical safety pairs (MUST have high contrast)
        self.critical_safety_pairs = {
            ('floor', 'stairs'), ('stairs', 'floor'),
            ('floor', 'door'), ('door', 'floor'),
            ('wall', 'door'), ('door', 'wall'),
            ('floor', 'toilet'), ('toilet', 'floor'),
            ('floor', 'bathtub'), ('bathtub', 'floor'),
            ('stairs', 'railing'), ('railing', 'stairs'),
            ('floor', 'rug'), ('rug', 'floor'),
            ('floor', 'mat'), ('mat', 'floor'),
            ('wall', 'mirror'), ('mirror', 'wall'),
            ('wall', 'window'), ('window', 'wall')
        }
        
        # High priority pairs
        self.high_priority_pairs = {
            ('floor', 'furniture'), ('furniture', 'floor'),
            ('wall', 'furniture'), ('furniture', 'wall'),
            ('table', 'chair'), ('chair', 'table'),
            ('countertop', 'cabinet'), ('cabinet', 'countertop'),
            ('sofa', 'coffee_table'), ('coffee_table', 'sofa')
        }
    
    def get_object_category(self, class_id: int) -> Optional[str]:
        """Map segmentation class to object category"""
        base_name = self.object_classes.get(class_id)
        if not base_name:
            return None
            
        # Group similar objects
        if base_name in ['sofa', 'chair', 'armchair', 'cushion', 'seat', 'ottoman']:
            return 'furniture'
        elif base_name in ['table', 'desk', 'coffee_table', 'bar']:
            return 'furniture'
        elif base_name in ['cabinet', 'shelf', 'wardrobe', 'bookcase']:
            return 'furniture'
        elif base_name in ['floor', 'wood_floor', 'mat', 'dirt_track']:
            return 'floor'
        elif base_name in ['rug', 'carpet']:
            return base_name  # Keep separate for critical checking
        elif base_name in ['wall', 'building', 'brick_wall', 'wallpaper']:
            return 'wall'
        elif base_name in ['stairs', 'stairway', 'step']:
            return 'stairs'
        elif base_name in ['railing', 'bannister', 'fence']:
            return 'railing'
        elif base_name in ['sink', 'toilet', 'bathtub', 'shower']:
            return base_name  # Keep bathroom items separate
        else:
            return base_name
    
    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG contrast ratio according to guidelines"""
        def linearize_color_component(c):
            c = c / 255.0
            return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
        
        # Calculate relative luminance
        r1, g1, b1 = [linearize_color_component(c) for c in color1]
        r2, g2, b2 = [linearize_color_component(c) for c in color2]
        
        lum1 = 0.2126 * r1 + 0.7152 * g1 + 0.0722 * b1
        lum2 = 0.2126 * r2 + 0.7152 * g2 + 0.0722 * b2
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def calculate_color_metrics(self, color1: np.ndarray, color2: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive color difference metrics per guidelines"""
        # RGB to HSV conversion
        hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0]
        hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0]
        
        # Convert to standard ranges
        h1, s1, v1 = hsv1[0] * 2, hsv1[1] / 255.0, hsv1[2] / 255.0
        h2, s2, v2 = hsv2[0] * 2, hsv2[1] / 255.0, hsv2[2] / 255.0
        
        # Hue difference (circular, in degrees)
        hue_diff = min(abs(h1 - h2), 360 - abs(h1 - h2))
        
        # Saturation and Value differences
        sat_diff = abs(s1 - s2)
        val_diff = abs(v1 - v2)
        
        # Luminance difference
        lum1 = self.calculate_luminance(color1)
        lum2 = self.calculate_luminance(color2)
        lum_diff = abs(lum1 - lum2)
        
        # RGB Euclidean distance
        rgb_distance = euclidean(color1, color2)
        
        # Check similarity thresholds
        is_similar_hue = hue_diff < self.hue_difference_threshold
        is_low_lum_diff = lum_diff < self.luminance_difference_threshold
        is_low_sat_diff = sat_diff < self.saturation_difference_threshold
        
        wcag_contrast = self.calculate_wcag_contrast(color1, color2)
        
        return {
            'wcag_contrast': wcag_contrast,
            'hue_difference': hue_diff,
            'saturation_difference': sat_diff,
            'value_difference': val_diff,
            'luminance_difference': lum_diff,
            'rgb_distance': rgb_distance,
            'is_similar_hue': is_similar_hue,
            'is_low_luminance_diff': is_low_lum_diff,
            'is_low_saturation_diff': is_low_sat_diff,
            'is_low_contrast': wcag_contrast < self.alzheimer_threshold,
            'fails_wcag': wcag_contrast < self.wcag_threshold
        }
    
    def calculate_luminance(self, color: np.ndarray) -> float:
        """Calculate relative luminance for a color"""
        rgb_norm = color / 255.0
        rgb_linear = np.where(rgb_norm <= 0.03928,
                            rgb_norm / 12.92,
                            ((rgb_norm + 0.055) / 1.055) ** 2.4)
        return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])
    
    def find_true_adjacency(self, mask1: np.ndarray, mask2: np.ndarray, 
                           min_boundary_length: int = 20) -> Tuple[bool, np.ndarray]:
        """Find true adjacency between segments with strict criteria"""
        # Use minimal dilation to find touching boundaries
        kernel = np.ones((3, 3), np.uint8)
        
        # Single pixel dilation
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)
        
        # Find overlap (boundary)
        boundary = dilated1 & dilated2
        
        # Clean up noise
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel)
        
        # Check if boundary is significant
        boundary_pixels = np.sum(boundary)
        
        # Validate real boundary (not just corner touching)
        if boundary_pixels >= min_boundary_length:
            # Additional check: ensure boundary forms a line, not just scattered pixels
            contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            valid_boundary = np.zeros_like(boundary)
            for contour in contours:
                # Check if contour is significant
                area = cv2.contourArea(contour)
                perimeter = cv2.arcLength(contour, True)
                
                # A real boundary should have reasonable perimeter-to-area ratio
                if area > 10 and perimeter > 15:
                    cv2.fillPoly(valid_boundary, [contour], 1)
            
            is_adjacent = np.sum(valid_boundary) >= min_boundary_length
            return is_adjacent, valid_boundary.astype(bool)
        
        return False, np.zeros_like(boundary, dtype=bool)
    
    def extract_representative_color(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract representative color using robust method"""
        if not np.any(mask):
            return np.array([128, 128, 128])
        
        masked_pixels = image[mask]
        if len(masked_pixels) < 10:
            return np.mean(masked_pixels, axis=0).astype(int)
        
        # Sample pixels if too many
        if len(masked_pixels) > 1000:
            indices = np.random.choice(len(masked_pixels), 1000, replace=False)
            masked_pixels = masked_pixels[indices]
        
        # Use median for robustness
        return np.median(masked_pixels, axis=0).astype(int)
    
    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """Analyze contrast between ALL adjacent objects per guidelines"""
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
            'detailed_analysis': [],
            'object_summary': {}
        }
        
        # Build comprehensive segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}
        object_count = {}
        
        logger.info(f"Analyzing {len(unique_segments)} unique segments for contrast")
        
        # Extract ALL segments
        for seg_id in unique_segments:
            if seg_id == 0 or seg_id == 255:  # Skip background/ignore
                continue
            
            mask = segmentation == seg_id
            pixel_count = np.sum(mask)
            
            # Include even small segments for completeness
            if pixel_count < 50:
                continue
            
            category = self.get_object_category(seg_id)
            if category is None:
                continue
            
            # Track object types
            object_count[category] = object_count.get(category, 0) + 1
            
            # Extract color
            color = self.extract_representative_color(image, mask)
            
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': pixel_count,
                'class_id': seg_id,
                'original_class': self.object_classes.get(seg_id, f'class_{seg_id}')
            }
        
        logger.info(f"Found objects: {object_count}")
        results['object_summary'] = object_count
        
        # Analyze ALL pairs for adjacency
        total_pairs_checked = 0
        adjacent_pairs_found = 0
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        checked_pairs = set()  # Avoid duplicate checks
        
        segment_ids = list(segment_info.keys())
        
        for i, seg_id1 in enumerate(segment_ids):
            for j, seg_id2 in enumerate(segment_ids):
                if i >= j:  # Skip self and already checked pairs
                    continue
                
                pair_key = tuple(sorted([seg_id1, seg_id2]))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)
                
                total_pairs_checked += 1
                
                info1 = segment_info[seg_id1]
                info2 = segment_info[seg_id2]
                
                # Check true adjacency
                is_adjacent, boundary = self.find_true_adjacency(
                    info1['mask'], info2['mask'], min_boundary_length=20
                )
                
                if not is_adjacent:
                    continue
                
                adjacent_pairs_found += 1
                
                # Calculate comprehensive color metrics
                metrics = self.calculate_color_metrics(info1['color'], info2['color'])
                
                # Determine issues based on guidelines
                issues = []
                severity = None
                
                # Check WCAG/Alzheimer's contrast
                if metrics['wcag_contrast'] < self.alzheimer_threshold:
                    issues.append(f"Low contrast: {metrics['wcag_contrast']:.2f}:1 (need ≥{self.alzheimer_threshold}:1)")
                    if metrics['fails_wcag']:
                        severity = 'critical'
                    else:
                        severity = 'high'
                
                # Check hue difference (30° minimum)
                if metrics['is_similar_hue']:
                    issues.append(f"Similar hues: {metrics['hue_difference']:.1f}° apart (need ≥30°)")
                    severity = 'critical' if severity is None else severity
                
                # Check luminance difference (20% minimum)
                if metrics['is_low_luminance_diff']:
                    issues.append(f"Low luminance difference: {metrics['luminance_difference']:.3f} (need ≥0.2)")
                    severity = 'high' if severity is None else severity
                
                # Check saturation difference
                if metrics['is_low_saturation_diff'] and not metrics['is_similar_hue']:
                    issues.append(f"Low saturation difference: {metrics['saturation_difference']:.3f}")
                    severity = 'medium' if severity is None else severity
                
                # Check RGB distance
                if metrics['rgb_distance'] < self.color_similarity_threshold:
                    issues.append(f"Similar RGB colors: distance {metrics['rgb_distance']:.1f}")
                    severity = 'high' if severity is None else severity
                
                # Adjust severity based on object relationship
                cat1, cat2 = info1['category'], info2['category']
                orig1, orig2 = info1['original_class'], info2['original_class']
                
                # Check critical safety pairs
                for crit_pair in self.critical_safety_pairs:
                    if (cat1 in crit_pair[0] and cat2 in crit_pair[1]) or \
                       (cat2 in crit_pair[0] and cat1 in crit_pair[1]) or \
                       (orig1 in crit_pair[0] and orig2 in crit_pair[1]) or \
                       (orig2 in crit_pair[0] and orig1 in crit_pair[1]):
                        if severity:
                            severity = 'critical'
                            issues.insert(0, "SAFETY CRITICAL PAIR")
                        break
                
                # Create issue record if problems found
                if issues:
                    issue = {
                        'categories': (cat1, cat2),
                        'original_classes': (orig1, orig2),
                        'metrics': metrics,
                        'boundary_pixels': np.sum(boundary),
                        'severity': severity,
                        'issues': issues
                    }
                    
                    # Visualize boundaries with severity colors
                    if severity == 'critical':
                        tint_color = np.array([255, 0, 0])  # Red
                        results['critical_issues'].append(issue)
                        issue_counts['critical'] += 1
                    elif severity == 'high':
                        tint_color = np.array([255, 128, 0])  # Orange
                        results['high_issues'].append(issue)
                        issue_counts['high'] += 1
                    elif severity == 'medium':
                        tint_color = np.array([255, 255, 0])  # Yellow
                        results['medium_issues'].append(issue)
                        issue_counts['medium'] += 1
                    else:
                        tint_color = np.array([255, 255, 128])  # Light yellow
                        results['low_issues'].append(issue)
                        issue_counts['low'] += 1
                    
                    # Apply visualization
                    results['boundary_mask'][boundary] = True
                    # Extend boundary visualization for visibility
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    extended_boundary = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=1)
                    results['visualization'][extended_boundary.astype(bool)] = tint_color
                    
                else:
                    # Good contrast - meeting Alzheimer's standards
                    if metrics['wcag_contrast'] >= self.alzheimer_threshold and \
                       not metrics['is_similar_hue'] and \
                       not metrics['is_low_luminance_diff']:
                        results['good_contrasts'].append({
                            'categories': (cat1, cat2),
                            'original_classes': (orig1, orig2),
                            'contrast_ratio': metrics['wcag_contrast'],
                            'hue_difference': metrics['hue_difference'],
                            'luminance_difference': metrics['luminance_difference']
                        })
                
                # Add to detailed analysis
                results['detailed_analysis'].append({
                    'pair': (cat1, cat2),
                    'metrics': metrics,
                    'is_adjacent': True,
                    'boundary_size': np.sum(boundary),
                    'issues': issues,
                    'severity': severity
                })
        
        # Compile comprehensive statistics
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
            'alzheimer_threshold': self.alzheimer_threshold,
            'hue_threshold': self.hue_difference_threshold,
            'luminance_threshold': self.luminance_difference_threshold,
            'object_types_found': len(object_count),
            'compliance_rate': (len(results['good_contrasts']) / max(1, adjacent_pairs_found)) * 100
        }
        
        logger.info(f"Contrast analysis complete: {adjacent_pairs_found} adjacent pairs from "
                   f"{total_pairs_checked} total, {sum(issue_counts.values())} issues, "
                   f"{len(results['good_contrasts'])} good contrasts")
        
        return results
