"""
Universal Contrast Analyzer for detecting low contrast between ALL adjacent objects.
Optimized for Alzheimer's/dementia care environments.
"""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from scipy.spatial import distance
from skimage.segmentation import find_boundaries

logger = logging.getLogger(__name__)


class UniversalContrastAnalyzer:
    """
    Analyzes contrast between ALL adjacent objects in a room.
    Ensures proper visibility for elderly individuals with Alzheimer's or dementia.
    """
    
    def __init__(self, wcag_threshold: float = 4.5):
        self.wcag_threshold = wcag_threshold
        
        # ADE20K semantic class mappings
        self.semantic_classes = {
            # Floors and ground surfaces
            'floor': [3, 4, 13, 28, 78],  # floor, wood floor, rug, carpet, mat
            
            # Walls and vertical surfaces
            'wall': [0, 1, 9],  # wall, building, brick
            
            # Ceiling
            'ceiling': [5],
            
            # Furniture
            'furniture': [10, 19, 15, 7, 18, 23, 30, 33, 34, 36, 44, 45, 57, 63, 64, 65, 75],
            # sofa, chair, table, bed, armchair, cabinet, desk, counter, stool, bench, nightstand, 
            # coffee table, ottoman, wardrobe, dresser, shelf, chest of drawers
            
            # Doors and openings
            'door': [25, 14],  # door, windowpane
            
            # Windows
            'window': [8],
            
            # Stairs and steps
            'stairs': [53, 59],  # stairs, step
            
            # Small objects that might be on floors/furniture
            'objects': [17, 20, 24, 37, 38, 39, 42, 62, 68, 71, 73, 80, 82, 84, 89, 90, 92, 93],
            # curtain, book, picture, towel, clothes, pillow, box, bag, lamp, fan, cushion,
            # basket, bottle, plate, clock, vase, tray, bowl
            
            # Kitchen/bathroom fixtures
            'fixtures': [32, 46, 49, 50, 54, 66, 69, 70, 77, 94, 97, 98, 99, 117, 118, 119, 120],
            # sink, toilet, bathtub, shower, dishwasher, oven, microwave, refrigerator,
            # stove, washer, dryer, range hood, kitchen island
        }
        
        # Create reverse mapping for quick lookup
        self.class_to_category = {}
        for category, class_ids in self.semantic_classes.items():
            for class_id in class_ids:
                self.class_to_category[class_id] = category
    
    def calculate_wcag_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate WCAG 2.0 contrast ratio between two colors"""
        def relative_luminance(rgb):
            # Normalize to 0-1
            rgb_norm = rgb / 255.0
            
            # Apply gamma correction
            rgb_linear = np.where(
                rgb_norm <= 0.03928,
                rgb_norm / 12.92,
                ((rgb_norm + 0.055) / 1.055) ** 2.4
            )
            
            # Calculate luminance using ITU-R BT.709 coefficients
            return np.dot(rgb_linear, [0.2126, 0.7152, 0.0722])
        
        lum1 = relative_luminance(color1)
        lum2 = relative_luminance(color2)
        
        # Ensure lighter color is in numerator
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def calculate_hue_difference(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate hue difference in degrees (0-180)"""
        # Convert RGB to HSV
        hsv1 = cv2.cvtColor(color1.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hsv2 = cv2.cvtColor(color2.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        
        # Calculate circular hue difference
        hue_diff = abs(hsv1[0] - hsv2[0])
        if hue_diff > 90:
            hue_diff = 180 - hue_diff
            
        return hue_diff
    
    def calculate_saturation_difference(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate saturation difference (0-255)"""
        hsv1 = cv2.cvtColor(color1.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        hsv2 = cv2.cvtColor(color2.reshape(1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV)[0, 0]
        
        return abs(int(hsv1[1]) - int(hsv2[1]))
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray, 
                             sample_size: int = 1000) -> np.ndarray:
        """Extract dominant color from masked region using robust statistics"""
        if not np.any(mask):
            return np.array([128, 128, 128])  # Default gray
        
        # Get masked pixels
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])
        
        # Sample if too many pixels (for efficiency)
        if len(masked_pixels) > sample_size:
            indices = np.random.choice(len(masked_pixels), sample_size, replace=False)
            masked_pixels = masked_pixels[indices]
        
        # Use median for robustness against outliers
        dominant_color = np.median(masked_pixels, axis=0).astype(int)
        
        return dominant_color
    
    def find_adjacent_segments(self, segmentation: np.ndarray) -> Dict[Tuple[int, int], np.ndarray]:
        """
        Find all pairs of adjacent segments and their boundaries.
        Returns dict mapping (seg1_id, seg2_id) to boundary mask.
        """
        adjacencies = {}
        
        # Find boundaries using 4-connectivity
        boundaries = find_boundaries(segmentation, mode='inner')
        
        # For each boundary pixel, check its neighbors
        h, w = segmentation.shape
        for y in range(1, h-1):
            for x in range(1, w-1):
                if boundaries[y, x]:
                    center_id = segmentation[y, x]
                    
                    # Check 4-connected neighbors
                    neighbors = [
                        segmentation[y-1, x],  # top
                        segmentation[y+1, x],  # bottom
                        segmentation[y, x-1],  # left
                        segmentation[y, x+1]   # right
                    ]
                    
                    for neighbor_id in neighbors:
                        if neighbor_id != center_id and neighbor_id != 0:  # Different segment, not background
                            # Create ordered pair (smaller id first)
                            pair = tuple(sorted([center_id, neighbor_id]))
                            
                            # Add this boundary pixel to the adjacency map
                            if pair not in adjacencies:
                                adjacencies[pair] = np.zeros((h, w), dtype=bool)
                            adjacencies[pair][y, x] = True
        
        # Filter out small boundaries (noise)
        min_boundary_pixels = 10
        filtered_adjacencies = {}
        for pair, boundary in adjacencies.items():
            if np.sum(boundary) >= min_boundary_pixels:
                filtered_adjacencies[pair] = boundary
        
        return filtered_adjacencies
    
    def is_object_on_surface(self, obj_mask: np.ndarray, surface_mask: np.ndarray, 
                           min_contact_ratio: float = 0.1) -> bool:
        """
        Determine if an object is resting on a surface (e.g., object on floor).
        Uses vertical proximity and overlap analysis.
        """
        if not np.any(obj_mask) or not np.any(surface_mask):
            return False
        
        # Find bottom edge of object
        obj_coords = np.where(obj_mask)
        if len(obj_coords[0]) == 0:
            return False
            
        obj_bottom_y = np.max(obj_coords[0])
        obj_bottom_mask = obj_mask.copy()
        obj_bottom_mask[:obj_bottom_y-5, :] = False  # Keep only bottom 5 pixels
        
        # Check for overlap with surface in the bottom region
        overlap = obj_bottom_mask & surface_mask
        
        # Calculate contact ratio
        obj_bottom_pixels = np.sum(obj_bottom_mask)
        if obj_bottom_pixels == 0:
            return False
            
        contact_ratio = np.sum(overlap) / obj_bottom_pixels
        
        return contact_ratio >= min_contact_ratio
    
    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """
        Perform comprehensive contrast analysis between ALL adjacent objects.
        
        Args:
            image: RGB image
            segmentation: Segmentation mask with class IDs
            
        Returns:
            Dictionary containing analysis results and visualizations
        """
        h, w = segmentation.shape
        results = {
            'issues': [],
            'visualization': image.copy(),
            'statistics': {
                'total_segments': 0,
                'analyzed_pairs': 0,
                'low_contrast_pairs': 0,
                'critical_issues': 0,
                'floor_object_issues': 0
            }
        }
        
        # Get unique segments
        unique_segments = np.unique(segmentation)
        unique_segments = unique_segments[unique_segments != 0]  # Remove background
        results['statistics']['total_segments'] = len(unique_segments)
        
        # Build segment information
        segment_info = {}
        floor_segments = []
        
        for seg_id in unique_segments:
            mask = segmentation == seg_id
            if np.sum(mask) < 50:  # Skip very small segments
                continue
            
            category = self.class_to_category.get(seg_id, 'unknown')
            color = self.extract_dominant_color(image, mask)
            
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': np.sum(mask),
                'class_id': seg_id
            }
            
            # Track floor segments
            if category == 'floor':
                floor_segments.append(seg_id)
        
        # Find all adjacent segment pairs
        adjacencies = self.find_adjacent_segments(segmentation)
        
        # Analyze each adjacent pair
        for (seg1_id, seg2_id), boundary in adjacencies.items():
            if seg1_id not in segment_info or seg2_id not in segment_info:
                continue
            
            info1 = segment_info[seg1_id]
            info2 = segment_info[seg2_id]
            
            results['statistics']['analyzed_pairs'] += 1
            
            # Calculate all contrast metrics
            wcag_ratio = self.calculate_wcag_contrast(info1['color'], info2['color'])
            hue_diff = self.calculate_hue_difference(info1['color'], info2['color'])
            sat_diff = self.calculate_saturation_difference(info1['color'], info2['color'])
            
            # Determine if there's insufficient contrast
            has_issue = False
            severity = 'low'
            
            # Check WCAG contrast
            if wcag_ratio < self.wcag_threshold:
                has_issue = True
                if wcag_ratio < 3.0:
                    severity = 'critical'
                elif wcag_ratio < 4.0:
                    severity = 'high'
                else:
                    severity = 'medium'
            
            # Additional checks for very similar colors
            if hue_diff < 30 and sat_diff < 50 and wcag_ratio < 7.0:
                has_issue = True
                if severity == 'low':
                    severity = 'medium'
            
            if has_issue:
                results['statistics']['low_contrast_pairs'] += 1
                
                # Determine relationship type
                is_floor_object = False
                if info1['category'] == 'floor' or info2['category'] == 'floor':
                    # Check if non-floor object is on the floor
                    if info1['category'] == 'floor':
                        floor_info, obj_info = info1, info2
                    else:
                        floor_info, obj_info = info2, info1
                    
                    if self.is_object_on_surface(obj_info['mask'], floor_info['mask']):
                        is_floor_object = True
                        results['statistics']['floor_object_issues'] += 1
                        if severity != 'critical':
                            severity = 'high'  # Elevate floor-object issues
                
                if severity == 'critical':
                    results['statistics']['critical_issues'] += 1
                
                # Record the issue
                issue = {
                    'segment_ids': (seg1_id, seg2_id),
                    'categories': (info1['category'], info2['category']),
                    'colors': (info1['color'], info2['color']),
                    'wcag_ratio': wcag_ratio,
                    'hue_difference': hue_diff,
                    'saturation_difference': sat_diff,
                    'boundary_pixels': np.sum(boundary),
                    'severity': severity,
                    'is_floor_object': is_floor_object,
                    'boundary_mask': boundary
                }
                
                results['issues'].append(issue)
                
                # Visualize on the output image
                self._visualize_issue(results['visualization'], boundary, severity)
        
        # Sort issues by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        results['issues'].sort(key=lambda x: severity_order[x['severity']])
        
        return results
    
    def _visualize_issue(self, image: np.ndarray, boundary: np.ndarray, severity: str):
        """Add visual indicators for contrast issues"""
        # Color coding by severity
        colors = {
            'critical': (255, 0, 0),     # Red
            'high': (255, 128, 0),       # Orange
            'medium': (255, 255, 0),     # Yellow
            'low': (128, 255, 128)       # Light green
        }
        
        color = colors.get(severity, (255, 255, 255))
        
        # Dilate boundary for better visibility
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(boundary.astype(np.uint8), kernel, iterations=2)
        
        # Apply color overlay
        image[dilated > 0] = color
        
        return image
    
    def generate_report(self, results: Dict) -> str:
        """Generate a detailed text report of contrast analysis"""
        stats = results['statistics']
        issues = results['issues']
        
        report = []
        report.append("=== Universal Contrast Analysis Report ===\n")
        
        # Summary statistics
        report.append(f"Total segments analyzed: {stats['total_segments']}")
        report.append(f"Adjacent pairs analyzed: {stats['analyzed_pairs']}")
        report.append(f"Low contrast pairs found: {stats['low_contrast_pairs']}")
        report.append(f"Critical issues: {stats['critical_issues']}")
        report.append(f"Floor-object contrast issues: {stats['floor_object_issues']}\n")
        
        # Detailed issues
        if issues:
            report.append("=== Contrast Issues (sorted by severity) ===\n")
            
            for i, issue in enumerate(issues, 1):
                cat1, cat2 = issue['categories']
                wcag = issue['wcag_ratio']
                severity = issue['severity'].upper()
                
                report.append(f"{i}. [{severity}] {cat1} ↔ {cat2}")
                report.append(f"   - WCAG Contrast Ratio: {wcag:.2f} (minimum: {self.wcag_threshold})")
                report.append(f"   - Hue Difference: {issue['hue_difference']:.1f}°")
                report.append(f"   - Saturation Difference: {issue['saturation_difference']}")
                
                if issue['is_floor_object']:
                    report.append("   - ⚠️ Object on floor - requires high visibility!")
                
                report.append(f"   - Boundary size: {issue['boundary_pixels']} pixels")
                report.append("")
        else:
            report.append("✅ No contrast issues detected!")
        
        return "\n".join(report)