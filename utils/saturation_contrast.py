import numpy as np
import colorsys
from .contrast_detector import ContrastDetector

class SaturationContrastDetector(ContrastDetector):
    """Saturation contrast detector"""
    
    def calculate_contrast(self, color1, color2):
        """Calculate saturation difference between two colors"""
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255.0, color1[1]/255.0, color1[2]/255.0)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255.0, color2[1]/255.0, color2[2]/255.0)
        
        saturation_diff = abs(hsv1[1] - hsv2[1])
        
        # Scale to 0-10 range to match WCAG scale
        return saturation_diff * 10
