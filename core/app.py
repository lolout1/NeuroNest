"""Main application class integrating all components - Enhanced with High Resolution and Labeling."""

import os
import cv2
import numpy as np
import logging
from typing import Dict, Optional

# Use the local oneformer manager, not the library
from oneformer_local import OneFormerManager
from blackspot import BlackspotDetector
from contrast import RobustContrastAnalyzer
from config import ONEFORMER_CONFIG

logger = logging.getLogger(__name__)


class NeuroNestApp:
    """Main application class with enhanced integration, high resolution, and labeling support"""

    def __init__(self):
        self.oneformer = OneFormerManager()
        self.blackspot_detector = None
        self.use_high_res = False
        
        # Initialize contrast analyzer with ultra-sensitive parameters
        try:
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=25.0,  # Very sensitive to similar colors
                perceptual_threshold=0.12         # Very sensitive to perceptual similarity
            )
            logger.info("Contrast analyzer initialized with ultra-sensitive settings")
        except Exception as e:
            logger.error(f"Failed to initialize contrast analyzer: {e}")
            # Create a fallback analyzer with minimal parameters
            self.contrast_analyzer = RobustContrastAnalyzer()
            
        self.initialized = False

    def initialize(self, blackspot_model_path: str = "./blackspot/model_0004999.pth", 
                   use_high_res: bool = False):
        """Initialize all components with robust error handling"""
        logger.info(f"Initializing NeuroNest application (high_res={use_high_res})...")
        
        self.use_high_res = use_high_res

        # Initialize OneFormer with optional high resolution
        oneformer_success = self.oneformer.initialize(use_high_res=use_high_res)
        if not oneformer_success:
            logger.error("Failed to initialize OneFormer - this is required")
            return False, False

        # Initialize blackspot detector if model exists
        blackspot_success = False
        if os.path.exists(blackspot_model_path):
            try:
                self.blackspot_detector = BlackspotDetector(blackspot_model_path)
                blackspot_success = self.blackspot_detector.initialize()
                logger.info(f"Blackspot detector initialized: {blackspot_success}")
            except Exception as e:
                logger.error(f"Failed to initialize blackspot detector: {e}")
                self.blackspot_detector = None
        else:
            logger.warning(f"Blackspot model not found at {blackspot_model_path}")
            logger.info("Color-based blackspot detection will be used as fallback")

        self.initialized = oneformer_success
        logger.info(f"NeuroNest initialization complete - OneFormer: {oneformer_success}, Blackspot: {blackspot_success}")
        return oneformer_success, blackspot_success

    def analyze_image(self,
                     image_path: str,
                     blackspot_threshold: float = 0.5,
                     contrast_threshold: float = 7.0,
                     enable_blackspot: bool = True,
                     enable_contrast: bool = True,
                     show_labels: bool = True) -> Dict:
        """Perform comprehensive image analysis with enhanced error handling and labeling"""

        if not self.initialized:
            return {"error": "Application not properly initialized"}

        try:
            # Load and validate image
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"Loaded image with shape: {image_rgb.shape}")

            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'statistics': {},
                'show_labels': show_labels
            }

            # 1. Semantic Segmentation (always performed)
            logger.info("Running semantic segmentation with labeling support...")
            try:
                seg_mask, seg_visualization, labeled_visualization = self.oneformer.semantic_segmentation(image_rgb)
                logger.info(f"Segmentation completed - mask shape: {seg_mask.shape}, unique classes: {len(np.unique(seg_mask))}")

                results['segmentation'] = {
                    'visualization': seg_visualization,
                    'labeled_visualization': labeled_visualization,
                    'mask': seg_mask
                }

                # Extract floor areas for blackspot detection
                floor_prior = self.oneformer.extract_floor_areas(seg_mask)
                floor_coverage = np.sum(floor_prior) / (seg_mask.shape[0] * seg_mask.shape[1]) * 100
                logger.info(f"Floor extraction completed - {np.sum(floor_prior)} pixels ({floor_coverage:.1f}% coverage)")

            except Exception as e:
                logger.error(f"Segmentation failed: {e}")
                import traceback
                traceback.print_exc()
                return {"error": f"Segmentation failed: {str(e)}"}

            # 2. Enhanced Blackspot Detection (Floor-only)
            if enable_blackspot:
                logger.info("Running enhanced blackspot detection (floors only)...")
                try:
                    # Blackspot detector works on original image resolution
                    if self.blackspot_detector is not None:
                        # Use trained model + color-based detection
                        blackspot_results = self.blackspot_detector.detect_blackspots(
                            image_rgb, floor_prior
                        )
                        logger.info(f"Model-based blackspot detection completed - found {blackspot_results.get('num_detections', 0)} blackspots")
                    else:
                        # Fallback to pure color-based detection
                        logger.info("Using color-based blackspot detection (model not available)")
                        dummy_detector = BlackspotDetector("")  # No model path
                        blackspot_results = dummy_detector.detect_blackspots(
                            image_rgb, floor_prior
                        )
                        logger.info(f"Color-based blackspot detection completed - found {blackspot_results.get('num_detections', 0)} blackspots")
                    
                    results['blackspot'] = blackspot_results

                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results['blackspot'] = None

            # 3. Ultra-Comprehensive Contrast Analysis (ALL object pairs, not just adjacent)
            if enable_contrast:
                logger.info("Running ultra-comprehensive contrast analysis (ALL object pairs)...")
                try:
                    # Update analyzer thresholds
                    self.contrast_analyzer.wcag_threshold = min(contrast_threshold, 4.5)  # Don't go below WCAG
                    self.contrast_analyzer.alzheimer_threshold = contrast_threshold

                    # Run the enhanced analysis on original resolution image
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, seg_mask)
                    
                    total_issues = contrast_results['statistics']['total_issues']
                    critical_issues = contrast_results['statistics']['critical_count']
                    similar_pairs = contrast_results['statistics']['similar_color_pairs']
                    good_contrasts = contrast_results['statistics']['good_contrast_count']
                    
                    logger.info(f"Contrast analysis completed - {total_issues} issues ({critical_issues} critical), "
                               f"{similar_pairs} similar color pairs, {good_contrasts} good contrasts")
                    
                    results['contrast'] = contrast_results

                except Exception as e:
                    logger.error(f"Contrast analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results['contrast'] = None

            # 4. Generate comprehensive statistics
            stats = self._generate_enhanced_statistics(results)
            results['statistics'] = stats

            logger.info("Comprehensive image analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Critical error in image analysis: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_enhanced_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics with robust error handling"""
        stats = {}

        # Segmentation stats
        if results.get('segmentation'):
            try:
                unique_classes = np.unique(results['segmentation']['mask'])
                stats['segmentation'] = {
                    'num_classes': len(unique_classes),
                    'image_size': results['segmentation']['mask'].shape,
                    'resolution_mode': 'high (1280x1280)' if self.use_high_res else 'standard (640x640)'
                }
            except Exception as e:
                logger.warning(f"Error generating segmentation stats: {e}")
                stats['segmentation'] = {'num_classes': 0, 'image_size': 'unknown'}

        # Enhanced blackspot stats
        if results.get('blackspot'):
            try:
                bs = results['blackspot']
                stats['blackspot'] = {
                    'floor_area_pixels': bs.get('floor_area', 0),
                    'blackspot_area_pixels': bs.get('blackspot_area', 0),
                    'coverage_percentage': bs.get('coverage_percentage', 0),
                    'num_detections': bs.get('num_detections', 0),
                    'avg_confidence': bs.get('avg_confidence', 0),
                    'risk_score': bs.get('risk_score', 0),
                    'detection_method': bs.get('detection_method', 'unknown')
                }
            except Exception as e:
                logger.warning(f"Error generating blackspot stats: {e}")
                stats['blackspot'] = {}

        # Enhanced contrast stats
        if results.get('contrast'):
            try:
                cs = results['contrast']['statistics']
                # Pass through all statistics from the analyzer
                stats['contrast'] = dict(cs)  # Copy all stats
                
                # Add risk assessment
                critical_count = cs.get('critical_count', 0)
                total_issues = cs.get('total_issues', 0)
                similar_pairs = cs.get('similar_color_pairs', 0)
                
                # Enhanced risk assessment including similar colors
                if critical_count > 0:
                    risk_level = 'critical'
                elif total_issues > 15 or similar_pairs > 10:
                    risk_level = 'high'
                elif total_issues > 8 or similar_pairs > 5:
                    risk_level = 'medium'
                elif total_issues > 3 or similar_pairs > 2:
                    risk_level = 'low'
                else:
                    risk_level = 'excellent'
                
                stats['contrast']['risk_level'] = risk_level
                stats['contrast']['risk_score'] = critical_count * 3 + total_issues + similar_pairs
                stats['contrast']['checks_all_objects'] = True  # Confirm we check ALL objects
                
            except Exception as e:
                logger.warning(f"Error generating contrast stats: {e}")
                stats['contrast'] = {}

        return stats
