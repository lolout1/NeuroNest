"""Enhanced contrast analysis for Alzheimer's-friendly environments with ALL object comparisons."""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from scipy import ndimage
from scipy.spatial.distance import euclidean
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import colorsys
import logging

logger = logging.getLogger(__name__)


class RobustContrastAnalyzer:
    """Ultra-comprehensive contrast analyzer that detects ANY similar colors in Alzheimer's environments"""

    def __init__(self, wcag_threshold: float = 4.5, alzheimer_threshold: float = 7.0, 
                 color_similarity_threshold: float = 30.0, perceptual_threshold: float = 0.15):
        # Contrast thresholds
        self.wcag_threshold = wcag_threshold
        self.alzheimer_threshold = alzheimer_threshold
        
        # Color similarity thresholds
        self.color_similarity_threshold = color_similarity_threshold
        self.perceptual_threshold = perceptual_threshold
        self.hue_similarity_threshold = 15.0
        self.saturation_similarity_threshold = 0.2
        self.value_similarity_threshold = 0.2
        
        # ADE20K class mappings for ALL indoor objects
        self.semantic_classes = {
            'floor': [3, 4, 13, 28, 78],      # floor, wood floor, rug, carpet, mat
            'wall': [0, 1, 9, 96, 97],        # wall, building, brick, wallpaper, tile wall
            'ceiling': [5],                   # ceiling
            'sofa': [10],                     # sofa
            'chair': [19],                    # chair
            'armchair': [18],                 # armchair
            'table': [15],                    # table
            'bed': [7],                       # bed
            'cabinet': [23, 24],              # cabinet, dresser
            'door': [25],                     # door
            'window': [8, 74],                # window, window frame
            'stairs': [53],                   # stairs
            'shelf': [30, 31],                # shelf, bookcase
            'desk': [33],                     # desk
            'lamp': [36],                     # lamp
            'curtain': [49, 93],              # curtain, drapes
            'refrigerator': [50],             # refrigerator
            'television': [89],               # television
            'counter': [11, 12],              # counter, countertop
            'sink': [14],                     # sink
            'toilet': [60],                   # toilet
            'bathtub': [62],                  # bathtub
            'mirror': [73],                   # mirror
            'picture': [75, 76],              # painting, picture frame
            'plant': [87, 88],                # plant, potted plant
            'pillow': [83],                   # pillow
            'blanket': [84],                  # blanket
            'towel': [85],                    # towel
        }
        
        # Critical safety relationships
        self.critical_safety_pairs = {
            ('floor', 'stairs'), ('stairs', 'floor'),
            ('floor', 'door'), ('door', 'floor'),
            ('wall', 'door'), ('door', 'wall'),
            ('floor', 'toilet'), ('toilet', 'floor'),
            ('floor', 'bathtub'), ('bathtub', 'floor'),
        }
        
        # High priority pairs
        self.high_priority_pairs = {
            ('floor', 'sofa'), ('sofa', 'floor'),
            ('floor', 'chair'), ('chair', 'floor'),
            ('floor', 'table'), ('table', 'floor'),
            ('floor', 'bed'), ('bed', 'floor'),
            ('wall', 'sofa'), ('sofa', 'wall'),
            ('wall', 'chair'), ('chair', 'wall'),
            ('wall', 'table'), ('table', 'wall'),
            ('wall', 'cabinet'), ('cabinet', 'wall'),
        }

    def get_object_category(self, class_id: int) -> Optional[str]:
        """Map segmentation class to object category"""
        for category, class_ids in self.semantic_classes.items():
            if class_id in class_ids:
                return category
        return None

    def calculate_color_similarity_score(self, color1: np.ndarray, color2: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive color similarity using multiple methods"""
        
        # 1. Euclidean distance in RGB space
        rgb_distance = euclidean(color1, color2)
        rgb_similarity = max(0, 1 - (rgb_distance / 441.67))  # 441.67 = sqrt(255^2 * 3)
        
        # 2. HSV component analysis
        hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        
        # Hue similarity (circular)
        hue1, hue2 = hsv1[0] * 2, hsv2[0] * 2  # Convert to 0-360
        hue_diff = min(abs(hue1 - hue2), 360 - abs(hue1 - hue2))
        hue_similarity = max(0, 1 - (hue_diff / 180))
        
        # Saturation and Value similarity
        sat_similarity = 1 - abs(hsv1[1] - hsv2[1]) / 255
        val_similarity = 1 - abs(hsv1[2] - hsv2[2]) / 255
        
        # 3. Combined similarity score
        combined_similarity = (
            rgb_similarity * 0.4 +
            hue_similarity * 0.3 +
            sat_similarity * 0.15 +
            val_similarity * 0.15
        )
        
        return {
            'rgb_distance': rgb_distance,
            'rgb_similarity': rgb_similarity,
            'hue_similarity': hue_similarity,
            'saturation_similarity': sat_similarity,
            'value_similarity': val_similarity,
            'combined_similarity': combined_similarity,
            'hue_difference_degrees': hue_diff
        }

    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG contrast ratio"""
        def relative_luminance(rgb):
            rgb_norm = rgb / 255.0
            rgb_linear = np.where(rgb_norm <= 0.03928,
                                rgb_norm / 12.92,
                                ((rgb_norm + 0.055) / 1.055) ** 2.4)
            return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])

        lum1 = relative_luminance(color1)
        lum2 = relative_luminance(color2)

        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)

        return (lighter + 0.05) / (darker + 0.05)

    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract dominant color from masked region"""
        if not np.any(mask):
            return np.array([128, 128, 128])

        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])

        # Use median for robustness against outliers
        return np.median(masked_pixels, axis=0).astype(int)

    def find_adjacent_segments(self, seg1_mask: np.ndarray, seg2_mask: np.ndarray,
                             min_boundary_length: int = 30) -> np.ndarray:
        """Find clean boundaries between segments"""
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(seg1_mask.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(seg2_mask.astype(np.uint8), kernel, iterations=1)

        boundary = dilated1 & dilated2

        # Remove small disconnected components
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_boundary = np.zeros_like(boundary)

        for contour in contours:
            if cv2.contourArea(contour) >= min_boundary_length:
                cv2.fillPoly(clean_boundary, [contour], 1)

        return clean_boundary.astype(bool)

    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """Perform comprehensive contrast analysis between ALL objects"""
        h, w = segmentation.shape
        
        # Initialize results with ALL required keys
        results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'low_issues': [],
            'good_contrasts': [],  # ENSURE this key exists
            'visualization': image.copy(),
            'statistics': {},
            'detailed_analysis': []
        }
        
        # Build segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}

        logger.info(f"Found {len(unique_segments)} unique segments")
        
        for seg_id in unique_segments:
            if seg_id == 0:  # Skip background
                continue
                
            mask = segmentation == seg_id
            pixel_count = np.sum(mask)
            
            # Skip very small segments
            if pixel_count < 100:
                continue
                
            category = self.get_object_category(seg_id)
            if category is None:
                continue
                
            color = self.extract_dominant_color(image, mask)
            
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': pixel_count,
                'class_id': seg_id
            }
            
        logger.info(f"Analyzing {len(segment_info)} segments for contrast")
        
        # Analyze pairs
        total_pairs_checked = 0
        adjacent_pairs = 0
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        
        segment_ids = list(segment_info.keys())
        
        for i, seg_id1 in enumerate(segment_ids):
            for j in range(i + 1, len(segment_ids)):
                seg_id2 = segment_ids[j]
                
                info1 = segment_info[seg_id1]
                info2 = segment_info[seg_id2]
                
                # Check if segments are adjacent
                boundary = self.find_adjacent_segments(info1['mask'], info2['mask'])
                is_adjacent = np.any(boundary)
                
                if is_adjacent:
                    adjacent_pairs += 1
                
                total_pairs_checked += 1
                
                # Calculate contrast and similarity
                wcag_contrast = self.calculate_wcag_contrast(info1['color'], info2['color'])
                similarity = self.calculate_color_similarity_score(info1['color'], info2['color'])
                
                # Determine issues
                issues = []
                severity = None
                
                # WCAG contrast check
                if wcag_contrast < self.alzheimer_threshold:
                    if wcag_contrast < self.wcag_threshold:
                        issues.append(f"Very low WCAG contrast: {wcag_contrast:.1f}:1")
                        severity = 'critical'
                    else:
                        issues.append(f"Low contrast for Alzheimer's: {wcag_contrast:.1f}:1")
                        severity = 'high'
                
                # Color similarity check
                if similarity['combined_similarity'] > (1 - self.perceptual_threshold):
                    issues.append(f"Colors too similar (similarity: {similarity['combined_similarity']:.3f})")
                    severity = 'critical'
                elif similarity['rgb_distance'] < self.color_similarity_threshold:
                    issues.append(f"RGB colors too close (distance: {similarity['rgb_distance']:.1f})")
                    severity = 'high' if severity != 'critical' else severity
                
                # Hue similarity check
                if similarity['hue_difference_degrees'] < self.hue_similarity_threshold:
                    issues.append(f"Similar hues: {similarity['hue_difference_degrees']:.1f}° apart")
                    severity = 'high' if severity is None else severity
                
                # Determine final severity based on relationship
                cat1, cat2 = info1['category'], info2['category']
                if (cat1, cat2) in self.critical_safety_pairs and severity:
                    severity = 'critical'
                elif (cat1, cat2) in self.high_priority_pairs and severity == 'medium':
                    severity = 'high'
                
                # Create issue or good contrast
                if issues:
                    description = f"{cat1} vs {cat2}"
                    if (cat1, cat2) in self.critical_safety_pairs:
                        description = f"SAFETY CRITICAL: {description}"
                    
                    issue = {
                        'categories': (cat1, cat2),
                        'contrast_ratio': wcag_contrast,
                        'boundary_area': np.sum(boundary),
                        'description': description,
                        'priority': severity,
                        'issues': issues
                    }
                    
                    # Visualize and categorize
                    if is_adjacent and np.any(boundary):
                        if severity == 'critical':
                            results['visualization'][boundary] = [255, 0, 0]  # Red
                            results['critical_issues'].append(issue)
                            issue_counts['critical'] += 1
                        elif severity == 'high':
                            results['visualization'][boundary] = [255, 165, 0]  # Orange
                            results['high_issues'].append(issue)
                            issue_counts['high'] += 1
                        elif severity == 'medium':
                            results['visualization'][boundary] = [255, 255, 0]  # Yellow
                            results['medium_issues'].append(issue)
                            issue_counts['medium'] += 1
                        else:
                            results['visualization'][boundary] = [255, 255, 128]  # Light yellow
                            results['low_issues'].append(issue)
                            issue_counts['low'] += 1
                    else:
                        # Add to lists even if not adjacent
                        if severity == 'critical':
                            results['critical_issues'].append(issue)
                            issue_counts['critical'] += 1
                        elif severity == 'high':
                            results['high_issues'].append(issue)
                            issue_counts['high'] += 1
                        elif severity == 'medium':
                            results['medium_issues'].append(issue)
                            issue_counts['medium'] += 1
                        else:
                            results['low_issues'].append(issue)
                            issue_counts['low'] += 1
                else:
                    # Good contrast - ALWAYS add to good_contrasts
                    if wcag_contrast >= self.alzheimer_threshold and similarity['combined_similarity'] < 0.7:
                        results['good_contrasts'].append({
                            'categories': (cat1, cat2),
                            'contrast_ratio': wcag_contrast,
                            'hue_difference': similarity['hue_difference_degrees']
                        })
                
                # Add to detailed analysis
                results['detailed_analysis'].append({
                    'categories': (cat1, cat2),
                    'wcag_contrast': wcag_contrast,
                    'similarity_metrics': similarity,
                    'is_adjacent': is_adjacent,
                    'severity': severity,
                    'issues': issues
                })
        
        # ENSURE all statistics are present
        results['statistics'] = {
            'total_segments': len(segment_info),
            'total_pairs_checked': total_pairs_checked,
            'adjacent_pairs': adjacent_pairs,
            'total_issues': sum(issue_counts.values()),
            'critical_count': issue_counts['critical'],
            'high_count': issue_counts['high'],
            'medium_count': issue_counts['medium'],
            'low_count': issue_counts['low'],
            'good_contrast_count': len(results['good_contrasts']),
            'wcag_threshold': self.wcag_threshold,
            'alzheimer_threshold': self.alzheimer_threshold
        }
        
        logger.info(f"Contrast analysis complete: {total_pairs_checked} pairs checked, "
                   f"{sum(issue_counts.values())} issues found, "
                   f"{len(results['good_contrasts'])} good contrasts")
        
        return results
