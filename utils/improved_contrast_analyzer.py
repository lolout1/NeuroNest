# Create an improved contrast detection file: utils/improved_contrast_analyzer.py

import numpy as np
import cv2
import colorsys
from scipy import ndimage, spatial
from sklearn.cluster import DBSCAN

class ImprovedContrastAnalyzer:
    """
    Advanced contrast analyzer focused on Alzheimer's-friendly environments
    """
    
    def __init__(self, wcag_threshold=4.5):
        self.wcag_threshold = wcag_threshold
        
        # ADE20K class mappings for important objects
        self.important_classes = {
            'floor': [3, 4],  # floor, wood floor
            'wall': [0, 1],   # wall, building
            'ceiling': [5],   # ceiling
            'sofa': [10],     # sofa
            'chair': [19],    # chair
            'table': [15],    # table
            'door': [25],     # door
            'window': [8],    # window
            'stairs': [53],   # stairs
            'bed': [7],       # bed
        }
        
        # Priority relationships (high priority = more important for safety)
        self.priority_relationships = {
            ('floor', 'sofa'): 'high',
            ('floor', 'chair'): 'high', 
            ('floor', 'table'): 'high',
            ('wall', 'sofa'): 'medium',
            ('wall', 'chair'): 'medium',
            ('wall', 'door'): 'high',
            ('floor', 'stairs'): 'critical',
            ('floor', 'bed'): 'medium',
            ('wall', 'window'): 'low',
            ('ceiling', 'wall'): 'low',
        }
    
    def get_object_category(self, class_id):
        """Map segmentation class to object category"""
        for category, class_ids in self.important_classes.items():
            if class_id in class_ids:
                return category
        return 'other'
    
    def calculate_wcag_contrast(self, color1, color2):
        """Calculate WCAG contrast ratio"""
        def relative_luminance(color):
            rgb = [c / 255.0 for c in color]
            return sum(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 
                      for c in rgb) * [0.2126, 0.7152, 0.0722][i] for i, c in enumerate(rgb))
        
        lum1 = sum(self.relative_luminance_component(color1))
        lum2 = sum(self.relative_luminance_component(color2))
        
        lighter = max(lum1, lum2)
        darker = min(lum1, lum2)
        
        return (lighter + 0.05) / (darker + 0.05)
    
    def relative_luminance_component(self, color):
        """Calculate relative luminance components"""
        rgb = [c / 255.0 for c in color]
        components = []
        factors = [0.2126, 0.7152, 0.0722]
        
        for i, c in enumerate(rgb):
            if c <= 0.03928:
                components.append((c / 12.92) * factors[i])
            else:
                components.append(((c + 0.055) / 1.055) ** 2.4 * factors[i])
        
        return components
    
    def calculate_perceptual_contrast(self, color1, color2):
        """Calculate perceptual contrast including hue and saturation differences"""
        # Convert to HSV for better perceptual analysis
        hsv1 = cv2.cvtColor(np.uint8([[color1]]), cv2.COLOR_RGB2HSV)[0][0] / 255.0
        hsv2 = cv2.cvtColor(np.uint8([[color2]]), cv2.COLOR_RGB2HSV)[0][0] / 255.0
        
        # Hue difference (circular)
        hue_diff = abs(hsv1[0] - hsv2[0])
        if hue_diff > 0.5:
            hue_diff = 1 - hue_diff
        
        # Saturation difference
        sat_diff = abs(hsv1[1] - hsv2[1])
        
        # Value (brightness) difference  
        val_diff = abs(hsv1[2] - hsv2[2])
        
        # Combined perceptual score (0-1, higher is more different)
        perceptual_contrast = np.sqrt(hue_diff**2 + sat_diff**2 + val_diff**2) / np.sqrt(3)
        
        return perceptual_contrast
    
    def find_clean_boundaries(self, mask1, mask2, min_boundary_length=50):
        """Find clean boundaries between two segments"""
        # Dilate both masks slightly
        kernel = np.ones((3, 3), np.uint8)
        dilated1 = cv2.dilate(mask1.astype(np.uint8), kernel, iterations=1)
        dilated2 = cv2.dilate(mask2.astype(np.uint8), kernel, iterations=1)
        
        # Find intersection (boundary area)
        boundary = (dilated1 & dilated2).astype(bool)
        
        # Remove small disconnected boundary pieces
        labeled_boundary = ndimage.label(boundary)[0]
        for region_id in range(1, labeled_boundary.max() + 1):
            region_mask = labeled_boundary == region_id
            if np.sum(region_mask) < min_boundary_length:
                boundary[region_mask] = False
        
        return boundary
    
    def get_representative_colors(self, image, mask, n_samples=1000):
        """Get representative colors from a masked region using clustering"""
        if not np.any(mask):
            return np.array([0, 0, 0])
        
        # Sample pixels from the mask
        y_coords, x_coords = np.where(mask)
        if len(y_coords) > n_samples:
            indices = np.random.choice(len(y_coords), n_samples, replace=False)
            y_coords = y_coords[indices]
            x_coords = x_coords[indices]
        
        colors = image[y_coords, x_coords]
        
        # Use DBSCAN clustering to find dominant colors
        if len(colors) > 10:
            clustering = DBSCAN(eps=30, min_samples=5).fit(colors)
            labels = clustering.labels_
            
            # Get the largest cluster
            unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
            if len(unique_labels) > 0:
                dominant_label = unique_labels[np.argmax(counts)]
                dominant_colors = colors[labels == dominant_label]
                return np.mean(dominant_colors, axis=0).astype(int)
        
        # Fallback to mean color
        return np.mean(colors, axis=0).astype(int)
    
    def analyze_improved_contrast(self, image, segmentation):
        """
        Perform improved contrast analysis focused on important relationships
        """
        h, w = segmentation.shape
        results = {
            'critical_issues': [],
            'high_priority_issues': [],
            'medium_priority_issues': [],
            'statistics': {},
            'visualization': image.copy()
        }
        
        # Get unique segments and their categories
        unique_segments = np.unique(segmentation)
        segment_categories = {}
        segment_colors = {}
        
        for seg_id in unique_segments:
            if seg_id == 0:  # Skip background
                continue
            
            mask = segmentation == seg_id
            category = self.get_object_category(seg_id)
            segment_categories[seg_id] = category
            segment_colors[seg_id] = self.get_representative_colors(image, mask)
        
        # Analyze important relationships
        total_issues = 0
        critical_count = 0
        high_count = 0
        medium_count = 0
        
        for i, seg_id1 in enumerate(unique_segments):
            if seg_id1 == 0:
                continue
                
            category1 = segment_categories.get(seg_id1, 'other')
            if category1 == 'other':
                continue
                
            for seg_id2 in unique_segments[i+1:]:
                if seg_id2 == 0:
                    continue
                    
                category2 = segment_categories.get(seg_id2, 'other')
                if category2 == 'other':
                    continue
                
                # Check if this is an important relationship
                relationship = tuple(sorted([category1, category2]))
                priority = self.priority_relationships.get(relationship)
                
                if priority is None:
                    continue
                
                # Check if segments are adjacent
                mask1 = segmentation == seg_id1
                mask2 = segmentation == seg_id2
                boundary = self.find_clean_boundaries(mask1, mask2)
                
                if not np.any(boundary):
                    continue
                
                # Calculate contrasts
                color1 = segment_colors[seg_id1]
                color2 = segment_colors[seg_id2]
                
                wcag_contrast = self.calculate_wcag_contrast(color1, color2)
                perceptual_contrast = self.calculate_perceptual_contrast(color1, color2)
                
                # Determine if there's an issue
                wcag_issue = wcag_contrast < self.wcag_threshold
                perceptual_issue = perceptual_contrast < 0.3  # Threshold for perceptual difference
                
                if wcag_issue or perceptual_issue:
                    issue = {
                        'categories': (category1, category2),
                        'segment_ids': (seg_id1, seg_id2),
                        'wcag_contrast': wcag_contrast,
                        'perceptual_contrast': perceptual_contrast,
                        'boundary_pixels': np.sum(boundary),
                        'priority': priority
                    }
                    
                    # Color-code the boundary based on priority
                    if priority == 'critical':
                        results['critical_issues'].append(issue)
                        results['visualization'][boundary] = [255, 0, 0]  # Red
                        critical_count += 1
                    elif priority == 'high':
                        results['high_priority_issues'].append(issue)
                        results['visualization'][boundary] = [255, 128, 0]  # Orange
                        high_count += 1
                    elif priority == 'medium':
                        results['medium_priority_issues'].append(issue)
                        results['visualization'][boundary] = [255, 255, 0]  # Yellow
                        medium_count += 1
                    
                    total_issues += 1
        
        # Calculate statistics
        results['statistics'] = {
            'total_issues': total_issues,
            'critical_issues': critical_count,
            'high_priority_issues': high_count,
            'medium_priority_issues': medium_count,
            'segments_analyzed': len([cat for cat in segment_categories.values() if cat != 'other'])
        }
        
        return results

# Update your contrast detection imports and usage
class PrioritizedContrastDetector:
    """Wrapper for the improved contrast analyzer"""
    
    def __init__(self, threshold=4.5):
        self.analyzer = ImprovedContrastAnalyzer(wcag_threshold=threshold)
    
    def analyze(self, image, segmentation, threshold, highlight_color=(255, 0, 0)):
        """Analyze with improved logic"""
        results = self.analyzer.analyze_improved_contrast(image, segmentation)
        
        # Convert to format expected by original interface
        contrast_image = results['visualization']
        
        # Create a simple problem areas mask for compatibility
        problem_areas = np.any([
            contrast_image[:, :, 0] == 255,  # Any red channel highlighting
        ], axis=0)
        
        # Format statistics
        stats = results['statistics'].copy()
        stats['threshold'] = threshold
        stats['problem_count'] = stats['total_issues']
        
        # Add detailed breakdown
        if results['critical_issues']:
            stats['critical_details'] = [
                f"{issue['categories'][0]}-{issue['categories'][1]}: WCAG {issue['wcag_contrast']:.1f}:1"
                for issue in results['critical_issues']
            ]
        
        if results['high_priority_issues']:
            stats['high_priority_details'] = [
                f"{issue['categories'][0]}-{issue['categories'][1]}: WCAG {issue['wcag_contrast']:.1f}:1"
                for issue in results['high_priority_issues']
            ]
        
        return contrast_image, problem_areas, stats
