"""Improved contrast analysis for Alzheimer's-friendly environments."""

import numpy as np
import cv2
from typing import Dict


class RobustContrastAnalyzer:
    """Advanced contrast analyzer for Alzheimer's-friendly environments"""

    def __init__(self, wcag_threshold: float = 4.5):
        self.wcag_threshold = wcag_threshold

        # ADE20K class mappings for important objects
        self.semantic_classes = {
            'floor': [3, 4, 13, 28, 78],  # floor, wood floor, rug, carpet, mat
            'wall': [0, 1, 9],            # wall, building, brick
            'ceiling': [5],               # ceiling
            'furniture': [10, 19, 15, 7, 18, 23],  # sofa, chair, table, bed, armchair, cabinet
            'door': [25],                 # door
            'window': [8],                # window
            'stairs': [53],               # stairs
        }

        # Priority relationships for safety
        self.priority_relationships = {
            ('floor', 'furniture'): ('critical', 'Furniture must be clearly visible against floor'),
            ('floor', 'stairs'): ('critical', 'Stairs must have clear contrast with floor'),
            ('floor', 'door'): ('high', 'Door should be easily distinguishable from floor'),
            ('wall', 'furniture'): ('high', 'Furniture should stand out from walls'),
            ('wall', 'door'): ('high', 'Doors should be clearly visible on walls'),
            ('wall', 'window'): ('medium', 'Windows should have adequate contrast'),
            ('ceiling', 'wall'): ('low', 'Ceiling-wall contrast is less critical'),
        }

    def get_object_category(self, class_id: int) -> str:
        """Map segmentation class to object category"""
        for category, class_ids in self.semantic_classes.items():
            if class_id in class_ids:
                return category
        return 'other'

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
        """Perform comprehensive contrast analysis"""
        h, w = segmentation.shape
        results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'visualization': image.copy(),
            'statistics': {}
        }

        # Build segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}

        for seg_id in unique_segments:
            if seg_id == 0:  # Skip background
                continue

            mask = segmentation == seg_id
            if np.sum(mask) < 100:  # Skip very small segments
                continue

            category = self.get_object_category(seg_id)
            if category == 'other':
                continue

            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': self.extract_dominant_color(image, mask),
                'area': np.sum(mask)
            }

        # Analyze priority relationships
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0}

        for seg_id1, info1 in segment_info.items():
            for seg_id2, info2 in segment_info.items():
                if seg_id1 >= seg_id2:
                    continue

                # Check if this is a priority relationship
                relationship = tuple(sorted([info1['category'], info2['category']]))
                if relationship not in self.priority_relationships:
                    continue

                priority, description = self.priority_relationships[relationship]

                # Check if segments are adjacent
                boundary = self.find_adjacent_segments(info1['mask'], info2['mask'])
                if not np.any(boundary):
                    continue

                # Calculate contrast
                wcag_contrast = self.calculate_wcag_contrast(info1['color'], info2['color'])

                # Determine if there's an issue
                if wcag_contrast < self.wcag_threshold:
                    issue = {
                        'categories': (info1['category'], info2['category']),
                        'contrast_ratio': wcag_contrast,
                        'boundary_area': np.sum(boundary),
                        'description': description,
                        'priority': priority
                    }

                    # Color-code boundaries and store issues
                    if priority == 'critical':
                        results['critical_issues'].append(issue)
                        results['visualization'][boundary] = [255, 0, 0]  # Red
                        issue_counts['critical'] += 1
                    elif priority == 'high':
                        results['high_issues'].append(issue)
                        results['visualization'][boundary] = [255, 165, 0]  # Orange
                        issue_counts['high'] += 1
                    elif priority == 'medium':
                        results['medium_issues'].append(issue)
                        results['visualization'][boundary] = [255, 255, 0]  # Yellow
                        issue_counts['medium'] += 1

        # Calculate statistics
        results['statistics'] = {
            'total_segments': len(segment_info),
            'total_issues': sum(issue_counts.values()),
            'critical_count': issue_counts['critical'],
            'high_count': issue_counts['high'],
            'medium_count': issue_counts['medium'],
            'wcag_threshold': self.wcag_threshold
        }

        return results
