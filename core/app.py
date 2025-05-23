"""Main application class integrating all components."""

import os
import cv2
import numpy as np
import logging
from typing import Dict

from oneformer_local import OneFormerManager
from blackspot import BlackspotDetector
from contrast import RobustContrastAnalyzer
from config import ONEFORMER_CONFIG

logger = logging.getLogger(__name__)


class NeuroNestApp:
    """Main application class integrating all components - FIXED VERSION"""

    def __init__(self):
        self.oneformer = OneFormerManager()
        self.blackspot_detector = None
        self.contrast_analyzer = RobustContrastAnalyzer()
        self.initialized = False

    def initialize(self, blackspot_model_path: str = "./blackspot/model_0004999.pth"):
        """Initialize all components"""
        logger.info("Initializing NeuroNest application...")

        # Initialize OneFormer
        oneformer_success = self.oneformer.initialize()

        # Initialize blackspot detector if model exists
        blackspot_success = False
        if os.path.exists(blackspot_model_path):
            self.blackspot_detector = BlackspotDetector(blackspot_model_path)
            blackspot_success = True
        else:
            logger.warning(f"Blackspot model not found at {blackspot_model_path}")

        self.initialized = oneformer_success
        return oneformer_success, blackspot_success

    def analyze_image(self,
                     image_path: str,
                     blackspot_threshold: float = 0.5,
                     contrast_threshold: float = 4.5,
                     enable_blackspot: bool = True,
                     enable_contrast: bool = True) -> Dict:
        """Perform complete image analysis - FIXED VERSION"""

        if not self.initialized:
            return {"error": "Application not properly initialized"}

        try:
            # Load and preprocess image
            image = cv2.imread(image_path)
            if image is None:
                return {"error": "Could not load image"}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image with shape: {image_rgb.shape}")

            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'statistics': {}
            }

            # 1. Semantic Segmentation (always performed)
            logger.info("Running semantic segmentation...")
            seg_mask, seg_visualization = self.oneformer.semantic_segmentation(image_rgb)
            logger.info(f"Segmentation mask shape: {seg_mask.shape}")

            results['segmentation'] = {
                'visualization': seg_visualization,
                'mask': seg_mask
            }

            # Extract floor areas for blackspot detection
            floor_prior = self.oneformer.extract_floor_areas(seg_mask)
            logger.info(f"Floor prior shape: {floor_prior.shape}, total floor pixels: {np.sum(floor_prior)}")

            # 2. Blackspot Detection (if enabled and model available)
            if enable_blackspot and self.blackspot_detector is not None:
                logger.info("Running blackspot detection...")
                try:
                    self.blackspot_detector.initialize(threshold=blackspot_threshold)
                    blackspot_results = self.blackspot_detector.detect_blackspots(image_rgb, floor_prior)
                    results['blackspot'] = blackspot_results
                    logger.info("Blackspot detection completed successfully")
                except Exception as e:
                    logger.error(f"Error in blackspot detection: {e}")
                    # Continue without blackspot results
                    results['blackspot'] = None

            # 3. Contrast Analysis (if enabled)
            if enable_contrast:
                logger.info("Running contrast analysis...")
                try:
                    # Use the resized image for contrast analysis to match segmentation
                    width = ONEFORMER_CONFIG["ADE20K"]["width"]
                    h, w = image_rgb.shape[:2]
                    if w != width:
                        scale = width / w
                        new_h = int(h * scale)
                        image_for_contrast = cv2.resize(image_rgb, (width, new_h))
                    else:
                        image_for_contrast = image_rgb

                    contrast_results = self.contrast_analyzer.analyze_contrast(image_for_contrast, seg_mask)
                    results['contrast'] = contrast_results
                    logger.info("Contrast analysis completed successfully")
                except Exception as e:
                    logger.error(f"Error in contrast analysis: {e}")
                    # Continue without contrast results
                    results['contrast'] = None

            # 4. Generate combined statistics
            stats = self._generate_statistics(results)
            results['statistics'] = stats

            logger.info("Image analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Error in image analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {}

        # Segmentation stats
        if results['segmentation']:
            unique_classes = np.unique(results['segmentation']['mask'])
            stats['segmentation'] = {
                'num_classes': len(unique_classes),
                'image_size': results['segmentation']['mask'].shape
            }

        # Blackspot stats
        if results['blackspot']:
            bs = results['blackspot']
            stats['blackspot'] = {
                'floor_area_pixels': bs['floor_area'],
                'blackspot_area_pixels': bs['blackspot_area'],
                'coverage_percentage': bs['coverage_percentage'],
                'num_detections': bs['num_detections'],
                'avg_confidence': bs['avg_confidence']
            }

        # Contrast stats
        if results['contrast']:
            cs = results['contrast']['statistics']
            stats['contrast'] = {
                'total_issues': cs['total_issues'],
                'critical_issues': cs['critical_count'],
                'high_priority_issues': cs['high_count'],
                'medium_priority_issues': cs['medium_count'],
                'segments_analyzed': cs['total_segments']
            }

        return stats
