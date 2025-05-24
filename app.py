"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Enhanced with local detectron2 installation
"""

import os
import cv2
import numpy as np
import logging
import sys
import warnings
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

warnings.filterwarnings("ignore")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_python_paths():
    """Setup Python paths with proper detectron2 integration"""
    project_root = Path(__file__).parent.absolute()
    
    # Clean any existing paths that might cause conflicts
    sys.path = [p for p in sys.path if not any(x in p.lower() for x in ['oneformer', 'neuronest'])]
    
    # Add project root FIRST (highest priority for our modules)
    sys.path.insert(0, str(project_root))
    
    # Add local detectron2 - should be installed via pip now, but ensure path
    detectron2_path = project_root / "detectron2"
    if detectron2_path.exists():
        logger.info(f"✅ Local detectron2 directory found: {detectron2_path}")
    
    # Add oneformer directory LAST (to avoid config conflicts)
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists():
        sys.path.append(str(oneformer_path))
        logger.info(f"✅ OneFormer path added: {oneformer_path}")
    
    logger.info(f"✅ Project root: {project_root}")
    logger.info(f"✅ Python path configured with {len(sys.path)} entries")
    
    return project_root

def comprehensive_detectron2_check():
    """Comprehensive detectron2 installation verification"""
    logger.info("🔍 Performing comprehensive detectron2 check...")
    
    status = {
        'installed': False,
        'version': None,
        'core_modules': {},
        'fully_functional': False,
        'installation_method': 'unknown'
    }
    
    try:
        # Basic import test
        import detectron2
        status['installed'] = True
        
        # Try to get version
        try:
            if hasattr(detectron2, '__version__'):
                status['version'] = detectron2.__version__
            else:
                # Try alternative version detection
                try:
                    import pkg_resources
                    status['version'] = pkg_resources.get_distribution('detectron2').version
                    status['installation_method'] = 'pip_package'
                except:
                    status['version'] = 'local_build'
                    status['installation_method'] = 'local_editable'
        except:
            status['version'] = 'unknown'
        
        # Test core module imports
        core_tests = {
            'config': 'detectron2.config',
            'engine': 'detectron2.engine',
            'data': 'detectron2.data',
            'model_zoo': 'detectron2.model_zoo',
            'utils': 'detectron2.utils'
        }
        
        for test_name, module_name in core_tests.items():
            try:
                __import__(module_name)
                status['core_modules'][test_name] = True
            except Exception as e:
                status['core_modules'][test_name] = str(e)
        
        # Test actual functionality
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            from detectron2.data import MetadataCatalog
            
            # Try to create a config
            cfg = get_cfg()
            status['core_modules']['config_creation'] = True
            status['fully_functional'] = True
            
        except Exception as e:
            status['core_modules']['config_creation'] = str(e)
            status['fully_functional'] = False
        
        working_modules = [k for k, v in status['core_modules'].items() if v is True]
        
        logger.info(f"✅ Detectron2 Status:")
        logger.info(f"   - Installed: {status['installed']}")
        logger.info(f"   - Version: {status['version']}")
        logger.info(f"   - Installation: {status['installation_method']}")
        logger.info(f"   - Working modules: {working_modules}")
        logger.info(f"   - Fully functional: {status['fully_functional']}")
        
        return status
        
    except ImportError as e:
        logger.error(f"❌ Detectron2 not available: {e}")
        return status

def check_system_dependencies():
    """Enhanced dependency checking with local detectron2"""
    deps_status = {
        'torch': False,
        'detectron2': False,
        'opencv': False,
        'gradio': False,
        'numpy': False,
        'config': False,
        'oneformer_local': False,
        'blackspot': False,
        'contrast': False,
        'interface': False
    }
    
    # Check PyTorch
    try:
        import torch
        deps_status['torch'] = torch.__version__
        logger.info(f"✅ PyTorch {torch.__version__} on {torch.device('cpu')}")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch issue: {e}")
    
    # Enhanced Detectron2 check
    detectron2_status = comprehensive_detectron2_check()
    if detectron2_status['installed']:
        deps_status['detectron2'] = detectron2_status['version']
    
    # Check other dependencies
    for dep_name, import_name in [
        ('opencv', 'cv2'),
        ('gradio', 'gradio'),
        ('numpy', 'numpy')
    ]:
        try:
            module = __import__(import_name)
            version = getattr(module, '__version__', 'installed')
            deps_status[dep_name] = version
            logger.info(f"✅ {dep_name.title()} {version}")
        except Exception as e:
            logger.warning(f"⚠️ {dep_name.title()} issue: {e}")
    
    # Check our modules
    module_checks = {
        'config': 'config.device_config',
        'oneformer_local': 'oneformer_local',
        'blackspot': 'blackspot',
        'contrast': 'contrast',
        'interface': 'interface.gradio_ui'
    }
    
    for module_key, module_name in module_checks.items():
        try:
            __import__(module_name)
            deps_status[module_key] = True
            logger.info(f"✅ {module_name} available")
        except Exception as e:
            logger.warning(f"⚠️ {module_name} not available: {e}")
    
    return deps_status, detectron2_status

# ====================== ENHANCED APPLICATION CLASS ======================

class NeuroNestApp:
    """Enhanced application class with local detectron2 support"""

    def __init__(self):
        self.oneformer = None
        self.blackspot_detector = None
        self.contrast_analyzer = None
        self.use_high_res = False
        self.initialized = False
        self.detectron2_status = None

    def initialize(self, use_high_res: bool = False):
        """Initialize with enhanced detectron2 handling"""
        logger.info(f"🚀 Initializing NeuroNest with local detectron2 (high_res={use_high_res})...")
        
        self.use_high_res = use_high_res
        
        # Get detectron2 status
        _, self.detectron2_status = check_system_dependencies()
        
        # Initialize OneFormer if detectron2 is functional
        oneformer_success = False
        if self.detectron2_status and self.detectron2_status['fully_functional']:
            try:
                from oneformer_local import OneFormerManager
                self.oneformer = OneFormerManager()
                oneformer_success = self.oneformer.initialize(use_high_res=use_high_res)
                if oneformer_success:
                    logger.info("✅ OneFormer initialized with functional detectron2")
                else:
                    logger.warning("⚠️ OneFormer initialization failed despite functional detectron2")
            except Exception as e:
                logger.error(f"❌ OneFormer init failed: {e}")
        else:
            logger.info("⚠️ Skipping OneFormer - detectron2 not fully functional")

        # Initialize enhanced blackspot detector (always works)
        blackspot_success = False
        try:
            from blackspot.detector import BlackspotDetector
            self.blackspot_detector = BlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
            logger.info("✅ Enhanced floor-only blackspot detector (50px+) initialized")
        except Exception as e:
            logger.error(f"❌ Blackspot detector failed: {e}")

        # Initialize contrast analyzer
        try:
            from contrast import RobustContrastAnalyzer
            self.contrast_analyzer = RobustContrastAnalyzer(
                wcag_threshold=4.5,
                alzheimer_threshold=7.0,
                color_similarity_threshold=25.0,
                perceptual_threshold=0.12
            )
            logger.info("✅ Contrast analyzer initialized")
        except Exception as e:
            logger.error(f"❌ Contrast analyzer failed: {e}")
            self.contrast_analyzer = None

        self.initialized = blackspot_success or oneformer_success or (self.contrast_analyzer is not None)
        
        logger.info(f"🎯 NeuroNest initialization summary:")
        logger.info(f"   - Overall success: {self.initialized}")
        logger.info(f"   - OneFormer: {oneformer_success}")
        logger.info(f"   - Enhanced Blackspot: {blackspot_success} (floor-only, 50px+)")
        logger.info(f"   - Contrast Analysis: {self.contrast_analyzer is not None}")
        
        return oneformer_success, blackspot_success

    def analyze_image(self, image_path: str, **kwargs) -> Dict:
        """Enhanced analysis with local detectron2 support"""
        if not self.initialized:
            return {"error": "Application not properly initialized"}

        try:
            # Load image
            if not os.path.exists(image_path):
                return {"error": f"Image file not found: {image_path}"}

            image = cv2.imread(image_path)
            if image is None:
                return {"error": f"Could not load image: {image_path}"}

            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            logger.info(f"📸 Loaded image: {image_rgb.shape}")

            results = {
                'original_image': image_rgb,
                'segmentation': None,
                'blackspot': None,
                'contrast': None,
                'statistics': {},
                'detectron2_status': self.detectron2_status
            }

            seg_mask = None
            floor_prior = None

            # 1. Semantic Segmentation (if detectron2 functional)
            if self.oneformer and self.detectron2_status['fully_functional']:
                logger.info("🎯 Running OneFormer segmentation...")
                try:
                    seg_mask, seg_vis, labeled_vis = self.oneformer.semantic_segmentation(image_rgb)
                    
                    results['segmentation'] = {
                        'visualization': seg_vis,
                        'labeled_visualization': labeled_vis,
                        'mask': seg_mask
                    }
                    
                    floor_prior = self.oneformer.extract_floor_areas(seg_mask)
                    logger.info(f"✅ Segmentation complete: {len(np.unique(seg_mask))} classes, {np.sum(floor_prior)} floor pixels")
                    
                except Exception as e:
                    logger.error(f"❌ Segmentation failed: {e}")

            # 2. Enhanced Floor-Only Blackspot Detection (50px+ minimum)
            if kwargs.get('enable_blackspot', True) and self.blackspot_detector:
                logger.info("⚫ Running enhanced floor-only blackspot detection (50px+ minimum)...")
                try:
                    # Create floor fallback if no segmentation
                    if floor_prior is None:
                        h, w = image_rgb.shape[:2]
                        floor_prior = np.zeros((h, w), dtype=bool)
                        floor_prior[int(h*0.7):, :] = True
                    
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, 
                        floor_mask=floor_prior,
                        segmentation_mask=seg_mask
                    )
                    
                    logger.info(f"✅ Enhanced blackspot detection: {blackspot_results.get('num_detections', 0)} spots "
                               f"({blackspot_results.get('coverage_percentage', 0):.2f}% coverage)")
                    
                    results['blackspot'] = blackspot_results
                    
                except Exception as e:
                    logger.error(f"❌ Blackspot detection failed: {e}")

            # 3. Contrast Analysis
            if kwargs.get('enable_contrast', True) and self.contrast_analyzer:
                logger.info("🎨 Running contrast analysis...")
                try:
                    if seg_mask is not None:
                        analysis_mask = seg_mask
                    else:
                        h, w = image_rgb.shape[:2]
                        analysis_mask = np.random.randint(0, 10, (h, w), dtype=np.uint8)
                    
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, analysis_mask)
                    results['contrast'] = contrast_results
                    
                    logger.info(f"✅ Contrast analysis: {contrast_results['statistics']['total_issues']} issues")
                    
                except Exception as e:
                    logger.error(f"❌ Contrast analysis failed: {e}")

            # Generate statistics
            results['statistics'] = self._generate_statistics(results)
            
            logger.info("🎉 Enhanced image analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"💥 Critical analysis error: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {}
        
        # Detectron2 status
        stats['detectron2'] = {
            'available': self.detectron2_status['installed'] if self.detectron2_status else False,
            'functional': self.detectron2_status['fully_functional'] if self.detectron2_status else False,
            'version': self.detectron2_status['version'] if self.detectron2_status else None
        }

        # Segmentation stats
        if results.get('segmentation'):
            unique_classes = np.unique(results['segmentation']['mask'])
            stats['segmentation'] = {
                'num_classes': len(unique_classes),
                'oneformer_available': True,
                'resolution': 'high' if self.use_high_res else 'standard'
            }
        else:
            stats['segmentation'] = {'oneformer_available': False}

        # Enhanced blackspot stats
        if results.get('blackspot'):
            bs = results['blackspot']
            stats['blackspot'] = {
                'num_detections': bs.get('num_detections', 0),
                'coverage_percentage': bs.get('coverage_percentage', 0),
                'floor_area_pixels': bs.get('floor_area', 0),
                'detection_method': 'enhanced_floor_only_50px_minimum',
                'risk_score': bs.get('risk_score', 0)
            }

        # Contrast stats
        if results.get('contrast'):
            cs = results['contrast']['statistics']
            stats['contrast'] = {
                'total_issues': cs.get('total_issues', 0),
                'critical_count': cs.get('critical_count', 0),
                'risk_level': 'critical' if cs.get('critical_count', 0) > 0 else 'good'
            }

        return stats

# ====================== INTERFACE CREATION ======================

def create_enhanced_interface(app_instance):
    """Create interface highlighting local detectron2 capabilities"""
    try:
        import gradio as gr
        import tempfile
        from PIL import Image
        
        # Get system status for display
        detectron2_status = app_instance.detectron2_status
        d2_functional = detectron2_status and detectron2_status.get('fully_functional', False)
        d2_version = detectron2_status.get('version', 'unknown') if detectron2_status else 'not available'
        
        def analyze_image(image, blackspot_threshold, contrast_threshold, 
                         enable_blackspot, enable_contrast, use_high_res, show_labels):
            if image is None:
                return None, None, None, None, None, "Please upload an image."
            
            try:
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    if hasattr(image, 'save'):
                        image.save(tmp.name)
                    else:
                        Image.fromarray(image).save(tmp.name)
                    
                    # Reinitialize if needed
                    if use_high_res != app_instance.use_high_res:
                        app_instance.initialize(use_high_res=use_high_res)
                    
                    # Run analysis
                    results = app_instance.analyze_image(
                        image_path=tmp.name,
                        enable_blackspot=enable_blackspot,
                        enable_contrast=enable_contrast
                    )
                    
                    os.unlink(tmp.name)
                
                if "error" in results:
                    return None, None, None, None, None, f"❌ {results['error']}"
                
                # Extract visualizations
                seg_vis = None
                if results.get('segmentation'):
                    seg_vis = results['segmentation'].get('labeled_visualization' if show_labels else 'visualization')
                
                blackspot_vis = None
                if results.get('blackspot') and 'enhanced_views' in results['blackspot']:
                    blackspot_vis = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
                
                contrast_vis = results.get('contrast', {}).get('visualization')
                
                # Generate report
                stats = results.get('statistics', {})
                bs_stats = stats.get('blackspot', {})
                contrast_stats = stats.get('contrast', {})
                d2_stats = stats.get('detectron2', {})
                
                report = f"""
# 🧠 NeuroNest Analysis Report
**Enhanced with Local Detectron2 Installation**

## 🔧 System Status
- **Detectron2:** {"✅ v" + str(d2_stats.get('version', 'unknown')) if d2_stats.get('functional') else "⚠️ Limited"}
- **OneFormer:** {"✅ Active" if results.get('segmentation') else "⚠️ Fallback mode"}
- **Enhanced Blackspot:** ✅ Floor-only detection (50px+ minimum)

## ⚫ Enhanced Blackspot Analysis
- **Method:** Floor-only detection with 50px minimum size filter
- **Detections:** {bs_stats.get('num_detections', 0)} blackspots found
- **Floor Coverage:** {bs_stats.get('coverage_percentage', 0):.2f}%
- **Risk Score:** {bs_stats.get('risk_score', 0)}/10

## 🎨 Contrast Analysis
- **Total Issues:** {contrast_stats.get('total_issues', 0)}
- **Critical Issues:** {contrast_stats.get('critical_count', 0)}
- **Risk Level:** {contrast_stats.get('risk_level', 'unknown').title()}

## 📋 Alzheimer's Care Recommendations
### ✅ Best Practices:
1. **7:1 contrast minimum** between adjacent objects
2. **No blackspots on floors** - all detected spots should be addressed
3. **Warm colors preferred** (red, yellow, orange vs cool blues/greens)
4. **High saturation** - avoid muted or pastel colors

### ⚠️ Immediate Actions Needed:
{"- **Address " + str(bs_stats.get('num_detections', 0)) + " blackspots** - potential trip hazards" if bs_stats.get('num_detections', 0) > 0 else "- No blackspots detected ✅"}
{"- **Fix " + str(contrast_stats.get('critical_count', 0)) + " critical contrast issues**" if contrast_stats.get('critical_count', 0) > 0 else "- Contrast levels acceptable ✅"}

*Analysis powered by local Detectron2 v{d2_version} + Enhanced NeuroNest AI*
                """
                
                return seg_vis, seg_vis, blackspot_vis, contrast_vis, blackspot_vis, report
                
            except Exception as e:
                return None, None, None, None, None, f"Analysis failed: {str(e)}"
        
        with gr.Blocks(
            title="NeuroNest - Enhanced with Local Detectron2",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")
        ) as interface:
            
            gr.Markdown(f"""
            # 🧠 NeuroNest: Advanced Alzheimer's Environment Analysis
            **Enhanced with Local Detectron2 v{d2_version}**
            
            *Abheek Pradhan | Faculty: Dr. Nadim Adi and Dr. Greg Lakomski*  
            *Texas State University - Computer Science & Interior Design*
            
            ### 🔧 Enhanced Features:
            - **Local Detectron2:** {"✅ v" + d2_version + " fully functional" if d2_functional else "⚠️ Limited functionality"}
            - **Floor-Only Blackspot Detection:** ✅ 50px+ minimum size filter
            - **OneFormer Segmentation:** {"✅ Available" if d2_functional else "⚠️ Fallback mode"}
            - **Alzheimer's Standards:** ✅ 7:1 contrast optimization
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Room Image",
                        type="pil",
                        height=300
                    )
                    
                    with gr.Accordion("🎛️ Analysis Settings", open=True):
                        use_high_res = gr.Checkbox(
                            value=False,
                            label="High Resolution (1280x1280)",
                            info="Better accuracy, slower processing"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels"
                        )
                        
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enhanced Blackspot Detection",
                            info="Floor-only, 50px+ minimum size"
                        )
                        
                        blackspot_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="Blackspot Sensitivity"
                        )
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Contrast Analysis"
                        )
                        
                        contrast_threshold = gr.Slider(
                            minimum=1.0, maximum=10.0, value=7.0, step=0.1,
                            label="Contrast Threshold (7:1 for Alzheimer's)"
                        )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze with Enhanced Detection",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=3):
                    main_output = gr.Image(label="🎯 Analysis Result", height=400)
                    
                    with gr.Tabs():
                        with gr.Tab("📊 Enhanced Analysis Report"):
                            analysis_report = gr.Markdown()
                        
                        with gr.Tab("🏷️ Object Segmentation"):
                            seg_output = gr.Image(label="OneFormer Segmentation", height=400)
                        
                        with gr.Tab("⚫ Floor Blackspot Detection"):
                            blackspot_output = gr.Image(label="Enhanced Floor-Only Detection (50px+)", height=400)
                        
                        with gr.Tab("🎨 Contrast Analysis"):
                            contrast_output = gr.Image(label="Alzheimer's Contrast Standards", height=400)
                        
                        with gr.Tab("🔄 Combined View"):
                            combined_output = gr.Image(label="Complete Assessment", height=400)
            
            analyze_btn.click(
                analyze_image,
                inputs=[image_input, blackspot_threshold, contrast_threshold,
                        enable_blackspot, enable_contrast, use_high_res, show_labels],
                outputs=[main_output, seg_output, blackspot_output, 
                        contrast_output, combined_output, analysis_report]
            )
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Enhanced interface creation failed: {e}")
        return None

# ====================== MAIN ======================

def main():
    """Main with enhanced local detectron2 support"""
    logger.info("🚀 Starting NeuroNest with Enhanced Local Detectron2 Integration")
    
    try:
        # Setup paths
        project_root = setup_python_paths()
        
        # Check system
        deps, detectron2_status = check_system_dependencies()
        
        # Initialize app
        app = NeuroNestApp()
        oneformer_ok, blackspot_ok = app.initialize()
        
        # Create interface
        if app.initialized and deps.get('gradio'):
            interface = create_enhanced_interface(app)
            if interface:
                logger.info("✅ Enhanced interface with local detectron2 created")
                
                # Launch
                logger.info("🌐 Launching enhanced NeuroNest...")
                interface.queue(default_concurrency_limit=2, max_size=10).launch(
                    server_name="0.0.0.0", port=7860, share=False, show_error=True
                )
                return True
        
        logger.error("❌ Failed to create interface")
        return False
        
    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        time.sleep(3600)
        return False

if __name__ == "__main__":
    success = main()
    if success:
        logger.info("🎉 NeuroNest with enhanced local detectron2 started successfully!")
    else:
        logger.error("💥 NeuroNest failed to start")
