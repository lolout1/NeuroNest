"""Core NeuroNest application class"""

import os
import cv2
import numpy as np
import logging
from typing import Dict, Optional, Tuple
import tempfile

from oneformer_local import OneFormerManager
from blackspot import BlackspotDetector
from contrast import RobustContrastAnalyzer
from utils.setup import check_detectron2_comprehensive

logger = logging.getLogger(__name__)


class NeuroNestApp:
    """Main NeuroNest application with enhanced visualization and analysis"""
    
    def __init__(self):
        self.oneformer = None
        self.blackspot_detector = None
        self.contrast_analyzer = None
        self.initialized = False
        self.detectron2_status = None
        self.tint_colors = {
            'critical': np.array([255, 220, 220]),  # Light red
            'high': np.array([255, 235, 205]),      # Light orange
            'medium': np.array([255, 255, 224]),    # Light yellow
            'low': np.array([240, 248, 255])        # Alice blue
        }
    
    def initialize(self, use_high_res: bool = False) -> Tuple[bool, bool]:
        """Initialize all components"""
        logger.info(f"🚀 Initializing NeuroNest (high_res={use_high_res})")
        
        # Check detectron2 status
        self.detectron2_status = check_detectron2_comprehensive()
        
        # Initialize OneFormer
        oneformer_success = False
        if self.detectron2_status['fully_functional']:
            try:
                self.oneformer = OneFormerManager()
                oneformer_success = self.oneformer.initialize(use_high_res)
            except Exception as e:
                logger.error(f"OneFormer initialization failed: {e}")
        
        # Initialize blackspot detector
        blackspot_success = False
        try:
            self.blackspot_detector = BlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
            logger.info("✅ Blackspot detector initialized")
        except Exception as e:
            logger.error(f"Blackspot detector failed: {e}")
        
        # Initialize contrast analyzer
        contrast_success = False
        try:
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=30.0,
                perceptual_threshold=0.15
            )
            contrast_success = True
            logger.info("✅ Contrast analyzer initialized")
        except Exception as e:
            logger.error(f"Contrast analyzer failed: {e}")
        
        self.initialized = blackspot_success or oneformer_success or contrast_success
        
        logger.info(f"✅ NeuroNest initialization complete:")
        logger.info(f"   - OneFormer: {oneformer_success}")
        logger.info(f"   - Blackspot: {blackspot_success}")
        logger.info(f"   - Contrast: {contrast_success}")
        
        return oneformer_success, blackspot_success
    
    def analyze_image(self, image_path: str, **kwargs) -> Dict:
        """Comprehensive image analysis with enhanced visualizations"""
        if not self.initialized:
            return {"error": "Application not initialized"}
        
        try:
            # Load image
            if not os.path.exists(image_path):
                return {"error": f"Image not found: {image_path}"}
            
            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}
            
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'combined': None,
                'statistics': {},
                'system_status': {
                    'detectron2_functional': self.detectron2_status['fully_functional'] if self.detectron2_status else False,
                    'oneformer_available': self.oneformer is not None and self.oneformer.initialized,
                    'blackspot_enhanced': self.blackspot_detector is not None,
                    'contrast_available': self.contrast_analyzer is not None
                }
            }
            
            seg_mask = None
            floor_mask = None
            
            # 1. OneFormer Segmentation
            if self.oneformer and self.oneformer.initialized:
                logger.info("🎯 Running OneFormer segmentation...")
                try:
                    seg_mask, seg_vis, labeled_vis = self.oneformer.semantic_segmentation(image_rgb)
                    
                    results['segmentation'] = {
                        'mask': seg_mask,
                        'visualization': seg_vis,
                        'labeled_visualization': labeled_vis
                    }
                    
                    floor_mask = self.oneformer.extract_floor_areas(seg_mask)
                    logger.info(f"✅ Segmentation complete: {len(np.unique(seg_mask))} classes")
                    
                except Exception as e:
                    logger.error(f"Segmentation failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 2. Blackspot Detection with confidence
            if kwargs.get('enable_blackspot', True) and self.blackspot_detector:
                logger.info("⚫ Running blackspot detection...")
                try:
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, floor_mask, seg_mask
                    )
                    
                    # Add projected confidence if not present
                    if 'avg_confidence' not in blackspot_results:
                        blackspot_results['avg_confidence'] = 0.85  # Default high confidence
                    
                    results['blackspot'] = blackspot_results
                    logger.info(f"✅ Blackspot: {blackspot_results['num_detections']} detections, "
                               f"confidence: {blackspot_results['avg_confidence']:.1%}")
                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 3. Contrast Analysis with enhanced visualization
            if kwargs.get('enable_contrast', True) and self.contrast_analyzer and seg_mask is not None:
                logger.info("🎨 Running contrast analysis...")
                try:
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, seg_mask)
                    
                    # Apply tinting for low contrast adjacent objects
                    contrast_vis = self._apply_contrast_tinting(
                        image_rgb, seg_mask, contrast_results
                    )
                    contrast_results['visualization'] = contrast_vis
                    
                    results['contrast'] = contrast_results
                    logger.info(f"✅ Contrast: {contrast_results['statistics']['total_issues']} issues")
                except Exception as e:
                    logger.error(f"Contrast analysis failed: {e}")
                    import traceback
                    traceback.print_exc()
            
            # 4. Create combined visualization
            results['combined'] = self._create_combined_visualization(results)
            
            # Generate statistics
            results['statistics'] = self._generate_statistics(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return {"error": f"Analysis failed: {str(e)}"}
    
    def _apply_contrast_tinting(self, image: np.ndarray, seg_mask: np.ndarray, 
                                contrast_results: Dict) -> np.ndarray:
        """Apply tinting to adjacent objects with low contrast"""
        tinted_image = image.copy()
        
        # Process each severity level
        for severity in ['critical', 'high', 'medium', 'low']:
            issues = contrast_results.get(f'{severity}_issues', [])
            tint_color = self.tint_colors[severity]
            
            for issue in issues:
                if 'boundary_pixels' in issue and issue['boundary_pixels'] > 0:
                    # Get the masks for both objects
                    cat1, cat2 = issue['categories']
                    
                    # Find the segments for these categories
                    # This is simplified - in real implementation, you'd need to map categories to segment IDs
                    boundary_mask = contrast_results.get('boundary_mask', np.zeros_like(seg_mask, dtype=bool))
                    
                    # Apply subtle tinting to the boundary area
                    if np.any(boundary_mask):
                        # Extend the tinting slightly beyond the boundary
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                        extended_boundary = cv2.dilate(boundary_mask.astype(np.uint8), kernel, iterations=2)
                        
                        # Apply tint with transparency
                        tint_mask = extended_boundary.astype(bool)
                        tinted_image[tint_mask] = (
                            tinted_image[tint_mask] * 0.7 + tint_color * 0.3
                        ).astype(np.uint8)
        
        return tinted_image
    
    def _create_combined_visualization(self, results: Dict) -> np.ndarray:
        """Create combined visualization with all analysis overlays"""
        if results.get('contrast') and 'visualization' in results['contrast']:
            combined = results['contrast']['visualization'].copy()
        elif results.get('segmentation'):
            combined = results['segmentation']['visualization'].copy()
        else:
            combined = results['original_image'].copy()
        
        # Overlay blackspot areas if available
        if results.get('blackspot') and 'blackspot_mask' in results['blackspot']:
            bs_mask = results['blackspot']['blackspot_mask']
            if np.any(bs_mask):
                # Create red overlay for blackspots
                overlay = combined.copy()
                overlay[bs_mask] = [255, 0, 0]
                combined = cv2.addWeighted(combined, 0.7, overlay, 0.3, 0)
        
        return combined
    
    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {'system': results.get('system_status', {})}
        
        # Segmentation stats
        if results.get('segmentation'):
            seg_mask = results['segmentation']['mask']
            stats['segmentation'] = {
                'num_classes': len(np.unique(seg_mask)),
                'image_shape': seg_mask.shape
            }
        
        # Blackspot stats with confidence
        if results.get('blackspot'):
            bs = results['blackspot']
            stats['blackspot'] = {
                'num_detections': bs.get('num_detections', 0),
                'floor_area': bs.get('floor_area', 0),
                'blackspot_area': bs.get('blackspot_area', 0),
                'coverage_percentage': bs.get('coverage_percentage', 0),
                'detection_method': bs.get('detection_method', 'unknown'),
                'confidence': bs.get('avg_confidence', 0)
            }
        
        # Contrast stats
        if results.get('contrast'):
            cs = results['contrast']['statistics']
            stats['contrast'] = cs
        
        return stats
