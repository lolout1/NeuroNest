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
        
        # Initialize contrast analyzer with proper parameters
        try:
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=25.0,  # Very sensitive
                perceptual_threshold=0.12         # Very sensitive
            )
            logger.info("Contrast analyzer initialized successfully")
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
                    # Resize floor_prior to match original image dimensions if needed
                    h_orig, w_orig = image_rgb.shape[:2]
                    h_seg, w_seg = seg_mask.shape
                    
                    if (h_orig, w_orig) != (h_seg, w_seg):
                        # Resize floor prior back to original image size
                        floor_prior_original = cv2.resize(
                            floor_prior.astype(np.uint8),
                            (w_orig, h_orig),
                            interpolation=cv2.INTER_NEAREST
                        ).astype(bool)
                    else:
                        floor_prior_original = floor_prior
                    
                    if self.blackspot_detector is not None:
                        # Use trained model + color-based detection
                        blackspot_results = self.blackspot_detector.detect_blackspots(
                            image_rgb, floor_prior_original
                        )
                        logger.info(f"Model-based blackspot detection completed - found {blackspot_results.get('num_detections', 0)} blackspots")
                    else:
                        # Fallback to pure color-based detection
                        logger.info("Using color-based blackspot detection (model not available)")
                        dummy_detector = BlackspotDetector("")  # No model path
                        blackspot_results = dummy_detector.detect_blackspots(
                            image_rgb, floor_prior_original
                        )
                        logger.info(f"Color-based blackspot detection completed - found {blackspot_results.get('num_detections', 0)} blackspots")
                    
                    results['blackspot'] = blackspot_results

                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
                    import traceback
                    traceback.print_exc()
                    results['blackspot'] = None

            # 3. Ultra-Comprehensive Contrast Analysis (Adjacent objects only)
            if enable_contrast:
                logger.info("Running ultra-comprehensive contrast analysis (adjacent objects only)...")
                try:
                    # Update analyzer thresholds safely
                    if hasattr(self.contrast_analyzer, 'wcag_threshold'):
                        self.contrast_analyzer.wcag_threshold = min(contrast_threshold, 4.5)  # Don't go below WCAG
                    if hasattr(self.contrast_analyzer, 'alzheimer_threshold'):
                        self.contrast_analyzer.alzheimer_threshold = contrast_threshold

                    # Get the appropriate resolution config
                    config = ONEFORMER_CONFIG["ADE20K"]
                    if self.use_high_res:
                        target_width = config.get("high_res_width", 1280)
                        target_height = config.get("high_res_height", 1280)
                    else:
                        target_width = config["width"]
                        target_height = config.get("height", config["width"])

                    # Use the resized image for contrast analysis to match segmentation
                    h, w = image_rgb.shape[:2]
                    if w != target_width or h != target_height:
                        scale = min(target_width / w, target_height / h)
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        image_for_contrast = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                        
                        # Pad if needed
                        if new_w < target_width or new_h < target_height:
                            pad_w = target_width - new_w
                            pad_h = target_height - new_h
                            pad_left = pad_w // 2
                            pad_top = pad_h // 2
                            image_for_contrast = cv2.copyMakeBorder(
                                image_for_contrast,
                                pad_top, pad_h - pad_top,
                                pad_left, pad_w - pad_left,
                                cv2.BORDER_CONSTANT,
                                value=[0, 0, 0]
                            )
                        logger.info(f"Resized image for contrast analysis: {image_for_contrast.shape}")
                    else:
                        image_for_contrast = image_rgb

                    # Run the enhanced analysis
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_for_contrast, seg_mask)
                    
                    total_issues = contrast_results['statistics']['total_issues']
                    critical_issues = contrast_results['statistics']['critical_count']
                    good_contrasts = contrast_results['statistics']['good_contrast_count']
                    
                    logger.info(f"Contrast analysis completed - {total_issues} issues ({critical_issues} critical), {good_contrasts} good contrasts")
                    
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
                    'resolution_mode': 'high' if self.use_high_res else 'standard'
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
                
                if critical_count > 0:
                    risk_level = 'critical'
                elif total_issues > 15:
                    risk_level = 'high'
                elif total_issues > 8:
                    risk_level = 'medium'
                elif total_issues > 3:
                    risk_level = 'low'
                else:
                    risk_level = 'excellent'
                
                stats['contrast']['risk_level'] = risk_level
                stats['contrast']['risk_score'] = critical_count * 3 + total_issues
                stats['contrast']['adjacency_based'] = True  # Confirm we only check adjacent objects
                
            except Exception as e:
                logger.warning(f"Error generating contrast stats: {e}")
                stats['contrast'] = {}

        return stats
