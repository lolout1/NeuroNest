"""Base contrast detector class for compatibility."""

import numpy as np
import cv2
from abc import ABC, abstractmethod
from typing import Tuple, Dict, Any


class ContrastDetector(ABC):
    """Abstract base class for contrast detection - maintains compatibility"""
    
    def __init__(self, threshold: float = 4.5):
        self.threshold = threshold
    
    @abstractmethod
    def calculate_contrast(self, color1: np.ndarray, color2: np.ndarray) -> float:
        """Calculate contrast between two colors"""
        pass
    
    def extract_dominant_color(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Extract dominant color from masked region"""
        if not np.any(mask):
            return np.array([128, 128, 128])
        
        masked_pixels = image[mask]
        if len(masked_pixels) == 0:
            return np.array([128, 128, 128])
        
        return np.median(masked_pixels, axis=0).astype(int)
    
    def analyze(self, image, segmentation, threshold=None, highlight_color=(255, 0, 0)):
        """Basic analyze method for compatibility"""
        if threshold is None:
            threshold = self.threshold
        
        # Return compatible format
        contrast_image = image.copy()
        problem_areas = np.zeros(image.shape[:2], dtype=bool)
        statistics = {
            'threshold': threshold,
            'problem_count': 0,
            'total_segments': 0,
            'detector_type': self.__class__.__name__
        }
        
        return contrast_image, problem_areas, statistics
