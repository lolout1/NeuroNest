import numpy as np
from .contrast_detector import ContrastDetector
from .luminance_contrast import LuminanceContrastDetector
from .hue_contrast import HueContrastDetector
from .saturation_contrast import SaturationContrastDetector

class CombinedContrastDetector(ContrastDetector):
    """Combined contrast detector using multiple methods"""
    
    def __init__(self):
        super().__init__()
        self.luminance_detector = LuminanceContrastDetector()
        self.hue_detector = HueContrastDetector()
        self.saturation_detector = SaturationContrastDetector()
    
    def calculate_contrast(self, color1, color2):
        """Calculate combined contrast using multiple methods"""
        luminance_contrast = self.luminance_detector.calculate_contrast(color1, color2)
        hue_contrast = self.hue_detector.calculate_contrast(color1, color2)
        saturation_contrast = self.saturation_detector.calculate_contrast(color1, color2)
        
        # Weighted average (60% luminance, 20% hue, 20% saturation)
        return 0.6 * luminance_contrast + 0.2 * hue_contrast + 0.2 * saturation_contrast
