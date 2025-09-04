import numpy as np
from .contrast_detector import ContrastDetector

class LuminanceContrastDetector(ContrastDetector):
    """WCAG luminance contrast detector"""
    
    def calculate_relative_luminance(self, rgb):
        """Calculate relative luminance according to WCAG 2.0"""
        rgb_normalized = np.array(rgb) / 255.0
        rgb_linear = np.where(
            rgb_normalized <= 0.03928,
            rgb_normalized / 12.92,
            ((rgb_normalized + 0.055) / 1.055) ** 2.4
        )
        return 0.2126 * rgb_linear[0] + 0.7152 * rgb_linear[1] + 0.0722 * rgb_linear[2]
    
    def calculate_contrast(self, color1, color2):
        """Calculate WCAG contrast ratio"""
        luminance1 = self.calculate_relative_luminance(color1)
        luminance2 = self.calculate_relative_luminance(color2)
        
        # Ensure luminance1 is the lighter color
        if luminance1 < luminance2:
            luminance1, luminance2 = luminance2, luminance1
        
        return (luminance1 + 0.05) / (luminance2 + 0.05)
