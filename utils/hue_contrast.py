import numpy as np
import colorsys
from .contrast_detector import ContrastDetector

class HueContrastDetector(ContrastDetector):
    """Hue contrast detector"""
    
    def calculate_contrast(self, color1, color2):
        """Calculate hue difference between two colors"""
        hsv1 = colorsys.rgb_to_hsv(color1[0]/255.0, color1[1]/255.0, color1[2]/255.0)
        hsv2 = colorsys.rgb_to_hsv(color2[0]/255.0, color2[1]/255.0, color2[2]/255.0)
        
        hue_diff = abs(hsv1[0] - hsv2[0])
        
        # Adjust for circular nature of hue
        if hue_diff > 0.5:
            hue_diff = 1 - hue_diff
        
        # Scale to 0-10 range to match WCAG scale
        return hue_diff * 20
