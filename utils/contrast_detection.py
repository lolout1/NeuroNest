import numpy as np
from abc import ABC, abstractmethod
import cv2
from scipy import ndimage

class ContrastDetector(ABC):
    """Base class for contrast detection between segments"""
    
    def __init__(self):
        pass
    
    @abstractmethod
    def calculate_contrast(self, color1, color2):
        """Calculate contrast between two colors"""
        pass
    
    def analyze(self, image, segmentation, threshold, highlight_color=None):
        """Analyze contrast between adjacent segments"""
        if highlight_color is None:
            highlight_color = (255, 0, 0)  # Red
        
        unique_segments = np.unique(segmentation)
        h, w = segmentation.shape
        contrast_image = image.copy()
        problem_areas = np.zeros((h, w), dtype=bool)
        
        # Calculate segment colors
        segment_colors = {}
        for segment_id in unique_segments:
            segment_mask = (segmentation == segment_id)
            if np.any(segment_mask):
                segment_colors[segment_id] = np.mean(image[segment_mask], axis=0).astype(int)
        
        # Find boundaries between segments
        contrast_values = []
        for i in range(h-1):
            for j in range(w-1):
                current_seg = segmentation[i, j]
                
                # Check right and down neighbors
                for di, dj in [(0, 1), (1, 0)]:
                    ni, nj = i + di, j + dj
                    if ni < h and nj < w:
                        neighbor_seg = segmentation[ni, nj]
                        
                        if current_seg != neighbor_seg:
                            color1 = segment_colors[current_seg]
                            color2 = segment_colors[neighbor_seg]
                            
                            contrast = self.calculate_contrast(color1, color2)
                            contrast_values.append(contrast)
                            
                            if contrast < threshold:
                                problem_areas[i, j] = True
                                contrast_image[i, j] = highlight_color
        
        # Calculate statistics
        stats = {
            "threshold": threshold,
            "problem_count": np.sum(problem_areas)
        }
        
        if contrast_values:
            stats.update({
                "min_contrast": min(contrast_values),
                "max_contrast": max(contrast_values),
                "average_contrast": sum(contrast_values) / len(contrast_values)
            })
        
        return contrast_image, problem_areas, stats
