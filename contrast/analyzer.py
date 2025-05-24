"""Ultra-robust contrast analysis for Alzheimer's-friendly environments with complete color similarity detection."""

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Set
from scipy import ndimage
from scipy.spatial.distance import euclidean, cosine
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
        self.color_similarity_threshold = color_similarity_threshold  # Euclidean distance in RGB space
        self.perceptual_threshold = perceptual_threshold  # Perceptual similarity threshold (0-1)
        self.hue_similarity_threshold = 15.0  # Degrees for hue similarity
        self.saturation_similarity_threshold = 0.2  # Saturation similarity (0-1)
        self.value_similarity_threshold = 0.2  # Value/brightness similarity (0-1)
        
        # ADE20K class mappings for ALL indoor objects
        self.semantic_classes = {
            'floor': [3, 4, 13, 28, 78],      # floor, wood floor, rug, carpet, mat
            'wall': [0, 1, 9, 96, 97],        # wall, building, brick, wallpaper, tile wall
            'ceiling': [5],                   # ceiling
            'sofa': [10, 23],                 # sofa, couch
            'chair': [19, 30, 75],            # chair, armchair, swivel chair
            'table': [15, 64],                # table, coffee table
            'bed': [7],                       # bed
            'cabinet': [10, 24, 44],          # cabinet, wardrobe, chest of drawers
            'door': [14],                     # door
            'window': [8, 58],                # window, screen door
            'stairs': [53, 59, 121],          # stairs, stairway, step
            'shelf': [24, 62],                # shelf, bookcase
            'desk': [33],                     # desk
            'lamp': [36, 82],                 # lamp, light
            'curtain': [18, 63],              # curtain, blind
            'refrigerator': [50],             # refrigerator
            'television': [89],               # television
            'counter': [45, 70],              # counter, countertop
            'sink': [47],                     # sink
            'toilet': [65],                   # toilet
            'bathtub': [37],                  # bathtub
            'mirror': [27],                   # mirror
            'picture': [22, 100],             # painting, poster
            'plant': [17, 66],                # plant, flower
            'pillow': [57],                   # pillow
            'blanket': [131],                 # blanket
            'towel': [81],                    # towel
            'rug': [28],                      # rug
            'cushion': [39],                  # cushion
            'railing': [38, 95],              # railing, bannister
            'fireplace': [49],                # fireplace
            'oven': [118],                    # oven
            'dishwasher': [129],              # dishwasher
            'microwave': [124],               # microwave
        }
        
        # Critical safety relationships (these MUST have good contrast)
        self.critical_safety_pairs = {
            ('floor', 'stairs'), ('stairs', 'floor'),
            ('floor', 'door'), ('door', 'floor'),
            ('wall', 'door'), ('door', 'wall'),
            ('floor', 'toilet'), ('toilet', 'floor'),
            ('floor', 'bathtub'), ('bathtub', 'floor'),
            ('stairs', 'railing'), ('railing', 'stairs'),
        }
        
        # High priority pairs
        self.high_priority_pairs = {
            ('floor', 'sofa'), ('sofa', 'floor'),
            ('floor', 'chair'), ('chair', 'floor'),
            ('floor', 'table'), ('table', 'floor'),
            ('floor', 'bed'), ('bed', 'floor'),
            ('floor', 'rug'), ('rug', 'floor'),
            ('wall', 'sofa'), ('sofa', 'wall'),
            ('wall', 'chair'), ('chair', 'wall'),
            ('wall', 'table'), ('table', 'wall'),
            ('wall', 'cabinet'), ('cabinet', 'wall'),
            ('wall', 'mirror'), ('mirror', 'wall'),
            ('wall', 'picture'), ('picture', 'wall'),
        }
    
    def get_object_category(self, class_id: int) -> Optional[str]:
        """Map segmentation class to object category"""
        for category, class_ids in self.semantic_classes.items():
            if class_id in class_ids:
                return category
        return None
    
    def rgb_to_lab(self, rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to LAB color space for perceptual color differences"""
        # Normalize RGB to 0-1
        rgb_norm = rgb.astype(np.float32) / 255.0
        
        # Convert to LAB using OpenCV
        lab = cv2.cvtColor(rgb_norm.reshape(1, 1, 3), cv2.COLOR_RGB2LAB)[0, 0]
        return lab
    
    def calculate_color_similarity_score(self, color1: np.ndarray, color2: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive color similarity using multiple methods"""
        
        # 1. Euclidean distance in RGB space
        rgb_distance = euclidean(color1, color2)
        rgb_similarity = max(0, 1 - (rgb_distance / 441.67))  # 441.67 = sqrt(255^2 * 3)
        
        # 2. Perceptual distance in LAB space
        lab1 = self.rgb_to_lab(color1)
        lab2 = self.rgb_to_lab(color2)
        lab_distance = euclidean(lab1, lab2)
        perceptual_similarity = max(0, 1 - (lab_distance / 100))  # Normalize LAB distance
        
        # 3. HSV component analysis
        hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0].astype(float)
        
        # Hue similarity (circular)
        hue1, hue2 = hsv1[0] * 2, hsv2[0] * 2  # Convert to 0-360
        hue_diff = min(abs(hue1 - hue2), 360 - abs(hue1 - hue2))
        hue_similarity = max(0, 1 - (hue_diff / 180))
        
        # Saturation and Value similarity
        sat_similarity = 1 - abs(hsv1[1] - hsv2[1]) / 255
        val_similarity = 1 - abs(hsv1[2] - hsv2[2]) / 255
        
        # 4. Cosine similarity of RGB vectors
        cosine_sim = cosine_similarity([color1], [color2])[0][0]
        
        # 5. Combined similarity score
        combined_similarity = (
            rgb_similarity * 0.25 +
            perceptual_similarity * 0.35 +
            hue_similarity * 0.20 +
            sat_similarity * 0.10 +
            val_similarity * 0.10
        )
        
        return {
            'rgb_distance': rgb_distance,
            'rgb_similarity': rgb_similarity,
            'perceptual_similarity': perceptual_similarity,
            'hue_similarity': hue_similarity,
            'saturation_similarity': sat_similarity,
            'value_similarity': val_similarity,
            'cosine_similarity': cosine_sim,
            'combined_similarity': combined_similarity,
            'hue_difference_degrees': hue_diff,
            'lab_distance': lab_distance
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
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray, 
                              use_clustering: bool = True) -> np.ndarray:
        """Extract dominant color from masked region with optional clustering"""
        if not np.any(mask):
            return np.array([128, 128, 128])

        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])

        if use_clustering and len(masked_pixels) > 100:
            # Sample pixels for efficiency
            sample_size = min(1000, len(masked_pixels))
            indices = np.random.choice(len(masked_pixels), sample_size, replace=False)
            sampled_pixels = masked_pixels[indices]
            
            # Use DBSCAN clustering to find dominant color cluster
            try:
                clustering = DBSCAN(eps=30, min_samples=10).fit(sampled_pixels)
                labels = clustering.labels_
                
                # Find the largest cluster
                unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                if len(unique_labels) > 0:
                    dominant_label = unique_labels[np.argmax(counts)]
                    dominant_pixels = sampled_pixels[labels == dominant_label]
                    return np.median(dominant_pixels, axis=0).astype(int)
            except:
                pass

        # Fallback to median
        return np.median(masked_pixels, axis=0).astype(int)
    
    def find_adjacent_segments(self, seg1_mask: np.ndarray, seg2_mask: np.ndarray,
                             min_boundary_length: int = 30) -> np.ndarray:
        """Find clean boundaries between segments with robust adjacency detection"""
        # Use multiple dilation levels to catch different types of adjacency
        boundaries = []
        
        # Level 1: Direct adjacency (3x3 kernel)
        kernel_small = np.ones((3, 3), np.uint8)
        dilated1_small = cv2.dilate(seg1_mask.astype(np.uint8), kernel_small, iterations=1)
        dilated2_small = cv2.dilate(seg2_mask.astype(np.uint8), kernel_small, iterations=1)
        boundary_small = dilated1_small & dilated2_small
        
        # Level 2: Near adjacency (5x5 kernel)
        kernel_medium = np.ones((5, 5), np.uint8)
        dilated1_medium = cv2.dilate(seg1_mask.astype(np.uint8), kernel_medium, iterations=1)
        dilated2_medium = cv2.dilate(seg2_mask.astype(np.uint8), kernel_medium, iterations=1)
        boundary_medium = dilated1_medium & dilated2_medium
        
        # Combine boundaries with preference for direct adjacency
        boundary = boundary_small | boundary_medium
        
        # Apply morphological closing to connect nearby boundary fragments
        kernel_close = np.ones((3, 3), np.uint8)
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_CLOSE, kernel_close)
        
        # Clean up small noise
        boundary = cv2.morphologyEx(boundary, cv2.MORPH_OPEN, kernel_small)

        # Remove small disconnected components
        contours, _ = cv2.findContours(boundary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        clean_boundary = np.zeros_like(boundary)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area >= min_boundary_length:
                cv2.fillPoly(clean_boundary, [contour], 1)
                
                # Additional check: ensure the boundary actually separates the two segments
                # by checking if pixels on either side belong to different segments
                x, y, w, h = cv2.boundingRect(contour)
                roi1 = seg1_mask[y:y+h, x:x+w]
                roi2 = seg2_mask[y:y+h, x:x+w]
                
                # If both segments have presence in this region, it's a valid boundary
                if np.any(roi1) and np.any(roi2):
                    cv2.fillPoly(clean_boundary, [contour], 1)

        return clean_boundary.astype(bool)
    
    def analyze_contrast(self, image: np.ndarray, segmentation: np.ndarray) -> Dict:
        """Perform comprehensive contrast analysis detecting ALL similar colors (not just adjacent)"""
        h, w = segmentation.shape
        
        # Initialize results with ALL required keys
        results = {
            'critical_issues': [],
            'high_issues': [],
            'medium_issues': [],
            'low_issues': [],
            'good_contrasts': [],
            'visualization': image.copy(),
            'statistics': {},
            'detailed_analysis': [],
            'similar_color_pairs': []
        }
        
        # Build segment information
        unique_segments = np.unique(segmentation)
        segment_info = {}

        logger.info(f"Found {len(unique_segments)} unique segments")
        
        for seg_id in unique_segments:
            if seg_id == 0 or seg_id == 255:  # Skip background and ignore labels
                continue
                
            mask = segmentation == seg_id
            pixel_count = np.sum(mask)
            
            # Skip very small segments
            if pixel_count < 100:
                continue
                
            category = self.get_object_category(seg_id)
            if category is None:
                continue
                
            # Extract dominant color with clustering
            color = self.extract_dominant_color(image, mask, use_clustering=True)
            
            segment_info[seg_id] = {
                'category': category,
                'mask': mask,
                'color': color,
                'area': pixel_count,
                'class_id': seg_id
            }
            
        logger.info(f"Analyzing {len(segment_info)} segments for contrast")
        
        # Analyze ALL pairs for similar colors (not just adjacent)
        total_pairs_checked = 0
        adjacent_pairs = 0
        issue_counts = {'critical': 0, 'high': 0, 'medium': 0, 'low': 0}
        similar_color_count = 0
        
        segment_ids = list(segment_info.keys())
        
        for i, seg_id1 in enumerate(segment_ids):
            for j in range(i + 1, len(segment_ids)):
                seg_id2 = segment_ids[j]
                
                info1 = segment_info[seg_id1]
                info2 = segment_info[seg_id2]
                
                total_pairs_checked += 1
                
                # Check if segments are adjacent
                boundary = self.find_adjacent_segments(info1['mask'], info2['mask'])
                is_adjacent = np.any(boundary)
                if is_adjacent:
                    adjacent_pairs += 1
                
                # Calculate contrast and similarity
                wcag_contrast = self.calculate_wcag_contrast(info1['color'], info2['color'])
                similarity = self.calculate_color_similarity_score(info1['color'], info2['color'])
                
                # Determine issues
                issues = []
                severity = None
                
                # Check for similar colors regardless of adjacency
                is_similar = False
                
                # Ultra-sensitive color similarity detection
                if similarity['rgb_distance'] < self.color_similarity_threshold:
                    issues.append(f"Very similar RGB values (distance: {similarity['rgb_distance']:.1f})")
                    is_similar = True
                    severity = 'critical'
                
                if similarity['combined_similarity'] > (1 - self.perceptual_threshold):
                    issues.append(f"Perceptually very similar ({similarity['combined_similarity']:.3f})")
                    is_similar = True
                    severity = 'critical'
                
                if similarity['hue_difference_degrees'] < self.hue_similarity_threshold and similarity['saturation_similarity'] > 0.7:
                    issues.append(f"Similar hues ({similarity['hue_difference_degrees']:.1f}° apart)")
                    is_similar = True
                    severity = 'high' if severity != 'critical' else severity
                
                if similarity['lab_distance'] < 20:
                    issues.append(f"Perceptually similar (LAB distance: {similarity['lab_distance']:.1f})")
                    is_similar = True
                    severity = 'high' if severity != 'critical' else severity
                
                # WCAG contrast check
                if wcag_contrast < self.alzheimer_threshold:
                    if wcag_contrast < self.wcag_threshold:
                        issues.append(f"Very low WCAG contrast: {wcag_contrast:.1f}:1")
                        severity = 'critical'
                    else:
                        issues.append(f"Low contrast for Alzheimer's: {wcag_contrast:.1f}:1")
                        severity = 'high' if severity != 'critical' else severity
                
                # Track similar colors
                if is_similar:
                    similar_color_count += 1
                    results['similar_color_pairs'].append({
                        'categories': (info1['category'], info2['category']),
                        'colors': (info1['color'].tolist(), info2['color'].tolist()),
                        'similarity_metrics': similarity,
                        'is_adjacent': is_adjacent
                    })
                
                # Determine final severity based on relationship
                cat1, cat2 = info1['category'], info2['category']
                if (cat1, cat2) in self.critical_safety_pairs or (cat2, cat1) in self.critical_safety_pairs:
                    if severity:
                        severity = 'critical'
                elif (cat1, cat2) in self.high_priority_pairs or (cat2, cat1) in self.high_priority_pairs:
                    if severity == 'medium':
                        severity = 'high'
                
                # Create issue or good contrast
                if issues:
                    description = f"{cat1} vs {cat2}"
                    if ((cat1, cat2) in self.critical_safety_pairs or 
                        (cat2, cat1) in self.critical_safety_pairs):
                        description = f"SAFETY CRITICAL: {description}"
                    
                    issue = {
                        'categories': (cat1, cat2),
                        'contrast_ratio': wcag_contrast,
                        'boundary_area': np.sum(boundary) if is_adjacent else 0,
                        'boundary_length': len(np.where(boundary)[0]) if is_adjacent else 0,
                        'description': description,
                        'priority': severity,
                        'issues': issues,
                        'color1': info1['color'].tolist(),
                        'color2': info2['color'].tolist(),
                        'is_adjacent': is_adjacent,
                        'similarity_score': similarity['combined_similarity']
                    }
                    
                    # Visualize on boundary if adjacent
                    if is_adjacent and np.any(boundary):
                        if severity == 'critical':
                            results['visualization'][boundary] = [255, 0, 0]  # Red
                        elif severity == 'high':
                            results['visualization'][boundary] = [255, 165, 0]  # Orange
                        elif severity == 'medium':
                            results['visualization'][boundary] = [255, 255, 0]  # Yellow
                        else:
                            results['visualization'][boundary] = [255, 255, 128]  # Light yellow
                    
                    # Add to appropriate issue list
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
                    # Good contrast - add to good_contrasts
                    if wcag_contrast >= self.alzheimer_threshold and similarity['combined_similarity'] < 0.7:
                        results['good_contrasts'].append({
                            'categories': (cat1, cat2),
                            'contrast_ratio': wcag_contrast,
                            'hue_difference': similarity['hue_difference_degrees'],
                            'color1': info1['color'].tolist(),
                            'color2': info2['color'].tolist(),
                            'is_adjacent': is_adjacent
                        })
                
                # Add to detailed analysis
                results['detailed_analysis'].append({
                    'categories': (cat1, cat2),
                    'wcag_contrast': wcag_contrast,
                    'similarity_metrics': similarity,
                    'is_adjacent': is_adjacent,
                    'boundary_pixels': np.sum(boundary) if is_adjacent else 0,
                    'severity': severity,
                    'issues': issues
                })
        
        # Compile statistics
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
            'similar_color_pairs': similar_color_count,
            'wcag_threshold': self.wcag_threshold,
            'alzheimer_threshold': self.alzheimer_threshold,
            'detection_sensitivity': 'ultra_high',
            'checks_all_pairs': True  # We check ALL pairs, not just adjacent
        }
        
        logger.info(f"Contrast analysis complete: {total_pairs_checked} pairs analyzed, "
                   f"{sum(issue_counts.values())} issues found, "
                   f"{similar_color_count} similar color pairs, "
                   f"{len(results['good_contrasts'])} good contrasts")
        
        return results
