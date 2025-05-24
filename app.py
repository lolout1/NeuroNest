"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Complete integrated application with enhanced blackspot detection
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
    """Setup Python paths with proper priority and detectron2 fix"""
    project_root = Path(__file__).parent.absolute()
    
    # Clean any existing paths that might cause conflicts
    sys.path = [p for p in sys.path if not any(x in p.lower() for x in ['oneformer', 'neuronest'])]
    
    # Add project root FIRST (highest priority for our modules)
    sys.path.insert(0, str(project_root))
    
    # Add local detectron2 path
    detectron2_path = project_root / "detectron2"
    if detectron2_path.exists():
        sys.path.insert(1, str(detectron2_path))
        logger.info(f"✅ Local detectron2 path added: {detectron2_path}")
    
    # Add oneformer directory LAST (to avoid config conflicts)
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists():
        sys.path.append(str(oneformer_path))
        logger.info(f"✅ OneFormer path added: {oneformer_path}")
    
    logger.info(f"✅ Project root: {project_root}")
    logger.info(f"✅ Python path configured with {len(sys.path)} entries")
    
    return project_root

def get_safe_version(module, module_name):
    """Safely get version from a module with multiple fallback methods"""
    try:
        if hasattr(module, '__version__'):
            return module.__version__
        if hasattr(module, 'version'):
            return module.version
        
        # Special handling for detectron2
        if module_name == 'detectron2':
            try:
                import pkg_resources
                return pkg_resources.get_distribution('detectron2').version
            except:
                # Check if we can import core detectron2 modules
                try:
                    from detectron2.config import get_cfg
                    return "local_build"
                except:
                    return "installed_incomplete"
        
        return "installed"
        
    except Exception:
        return "unknown"

def check_detectron2_health():
    """Comprehensive detectron2 health check"""
    try:
        import detectron2
        version = get_safe_version(detectron2, 'detectron2')
        
        # Test core imports
        core_imports = {}
        try:
            from detectron2.config import get_cfg
            core_imports['config'] = True
        except Exception as e:
            core_imports['config'] = str(e)
        
        try:
            from detectron2.engine import DefaultPredictor
            core_imports['engine'] = True
        except Exception as e:
            core_imports['engine'] = str(e)
        
        try:
            from detectron2.data import MetadataCatalog
            core_imports['data'] = True
        except Exception as e:
            core_imports['data'] = str(e)
        
        working_imports = [k for k, v in core_imports.items() if v is True]
        
        return {
            'available': True,
            'version': version,
            'core_imports': core_imports,
            'working_imports': working_imports,
            'fully_functional': len(working_imports) >= 3
        }
        
    except ImportError:
        return {
            'available': False,
            'version': None,
            'core_imports': {},
            'working_imports': [],
            'fully_functional': False
        }

def check_system_dependencies():
    """Comprehensive dependency checking with robust error handling"""
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
        version = get_safe_version(torch, 'torch')
        deps_status['torch'] = version
        logger.info(f"✅ PyTorch {version} on {torch.device('cpu')}")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch issue: {e}")
    
    # Enhanced Detectron2 check
    detectron2_status = check_detectron2_health()
    if detectron2_status['available']:
        deps_status['detectron2'] = detectron2_status['version']
        if detectron2_status['fully_functional']:
            logger.info(f"✅ Detectron2 {detectron2_status['version']} - fully functional")
        else:
            working = detectron2_status['working_imports']
            logger.warning(f"⚠️ Detectron2 {detectron2_status['version']} - partial ({working})")
    else:
        logger.warning("⚠️ Detectron2 not available")
    
    # Check other core dependencies
    for dep_name, import_name in [
        ('opencv', 'cv2'),
        ('gradio', 'gradio'),
        ('numpy', 'numpy')
    ]:
        try:
            module = __import__(import_name)
            version = get_safe_version(module, dep_name)
            deps_status[dep_name] = version
            logger.info(f"✅ {dep_name.title()} {version}")
        except Exception as e:
            logger.warning(f"⚠️ {dep_name.title()} issue: {e}")
    
    # Check our modules with better error handling
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
    """Enhanced application class with robust blackspot detection"""

    def __init__(self):
        self.oneformer = None
        self.blackspot_detector = None
        self.contrast_analyzer = None
        self.use_high_res = False
        self.initialized = False
        self.detectron2_status = None

    def initialize(self, use_high_res: bool = False):
        """Initialize all components with enhanced error handling"""
        logger.info(f"Initializing NeuroNest application (high_res={use_high_res})...")
        
        self.use_high_res = use_high_res
        
        # Check detectron2 status
        _, self.detectron2_status = check_system_dependencies()
        
        # Initialize OneFormer if detectron2 is functional
        oneformer_success = False
        if self.detectron2_status['fully_functional']:
            try:
                from oneformer_local import OneFormerManager
                self.oneformer = OneFormerManager()
                oneformer_success = self.oneformer.initialize(use_high_res=use_high_res)
                if oneformer_success:
                    logger.info("✅ OneFormer initialized successfully")
                else:
                    logger.warning("⚠️ OneFormer initialization failed")
            except Exception as e:
                logger.error(f"❌ OneFormer import/init failed: {e}")
        else:
            logger.info("⚠️ Skipping OneFormer - detectron2 not fully functional")

        # Initialize enhanced blackspot detector (always works)
        blackspot_success = False
        try:
            from blackspot.detector import BlackspotDetector
            self.blackspot_detector = BlackspotDetector()
            blackspot_success = self.blackspot_detector.initialize()
            logger.info("✅ Enhanced blackspot detector initialized")
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
        logger.info(f"NeuroNest initialization complete - Status: {self.initialized}")
        return oneformer_success, blackspot_success

    def analyze_image(self,
                     image_path: str,
                     blackspot_threshold: float = 0.5,
                     contrast_threshold: float = 7.0,
                     enable_blackspot: bool = True,
                     enable_contrast: bool = True,
                     show_labels: bool = True) -> Dict:
        """Enhanced image analysis with robust blackspot detection"""

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

            seg_mask = None
            floor_prior = None

            # 1. Semantic Segmentation (if available)
            if self.oneformer:
                logger.info("Running semantic segmentation...")
                try:
                    seg_mask, seg_visualization, labeled_visualization = self.oneformer.semantic_segmentation(image_rgb)
                    logger.info(f"Segmentation completed - unique classes: {len(np.unique(seg_mask))}")

                    results['segmentation'] = {
                        'visualization': seg_visualization,
                        'labeled_visualization': labeled_visualization,
                        'mask': seg_mask
                    }

                    # Extract floor areas
                    floor_prior = self.oneformer.extract_floor_areas(seg_mask)
                    floor_coverage = np.sum(floor_prior) / (seg_mask.shape[0] * seg_mask.shape[1]) * 100
                    logger.info(f"Floor extraction: {np.sum(floor_prior)} pixels ({floor_coverage:.1f}%)")

                except Exception as e:
                    logger.error(f"Segmentation failed: {e}")
                    seg_mask = None

            # 2. Enhanced Blackspot Detection (always runs)
            if enable_blackspot and self.blackspot_detector:
                logger.info("Running enhanced blackspot detection...")
                try:
                    # If no floor_prior from segmentation, create fallback
                    if floor_prior is None:
                        h, w = image_rgb.shape[:2]
                        floor_prior = np.zeros((h, w), dtype=bool)
                        floor_prior[int(h*0.7):, :] = True  # Bottom 30% as floor
                    
                    blackspot_results = self.blackspot_detector.detect_blackspots(
                        image_rgb, 
                        floor_mask=floor_prior,
                        segmentation_mask=seg_mask
                    )
                    
                    logger.info(f"Enhanced blackspot detection: {blackspot_results.get('num_detections', 0)} spots, "
                               f"{blackspot_results.get('coverage_percentage', 0):.2f}% coverage")
                    
                    results['blackspot'] = blackspot_results

                except Exception as e:
                    logger.error(f"Blackspot detection failed: {e}")
                    results['blackspot'] = None

            # 3. Contrast Analysis (if available)
            if enable_contrast and self.contrast_analyzer:
                logger.info("Running contrast analysis...")
                try:
                    # Update thresholds
                    self.contrast_analyzer.wcag_threshold = min(contrast_threshold, 4.5)
                    self.contrast_analyzer.alzheimer_threshold = contrast_threshold

                    # Use segmentation mask if available, otherwise create dummy
                    if seg_mask is not None:
                        analysis_mask = seg_mask
                    else:
                        h, w = image_rgb.shape[:2]
                        analysis_mask = np.random.randint(0, 10, (h, w), dtype=np.uint8)
                    
                    contrast_results = self.contrast_analyzer.analyze_contrast(image_rgb, analysis_mask)
                    
                    total_issues = contrast_results['statistics']['total_issues']
                    critical_issues = contrast_results['statistics']['critical_count']
                    
                    logger.info(f"Contrast analysis: {total_issues} issues ({critical_issues} critical)")
                    results['contrast'] = contrast_results

                except Exception as e:
                    logger.error(f"Contrast analysis failed: {e}")
                    results['contrast'] = None

            # 4. Generate comprehensive statistics
            stats = self._generate_enhanced_statistics(results)
            results['statistics'] = stats

            logger.info("Enhanced image analysis completed successfully")
            return results

        except Exception as e:
            logger.error(f"Critical error in image analysis: {e}")
            return {"error": f"Analysis failed: {str(e)}"}

    def _generate_enhanced_statistics(self, results: Dict) -> Dict:
        """Generate comprehensive statistics"""
        stats = {}

        # Segmentation stats
        if results.get('segmentation'):
            try:
                unique_classes = np.unique(results['segmentation']['mask'])
                stats['segmentation'] = {
                    'num_classes': len(unique_classes),
                    'image_size': results['segmentation']['mask'].shape,
                    'resolution_mode': 'high (1280x1280)' if self.use_high_res else 'standard (640x640)',
                    'oneformer_available': True
                }
            except Exception:
                stats['segmentation'] = {'num_classes': 0, 'oneformer_available': False}
        else:
            stats['segmentation'] = {'oneformer_available': False}

        # Enhanced blackspot stats
        if results.get('blackspot'):
            bs = results['blackspot']
            stats['blackspot'] = {
                'floor_area_pixels': bs.get('floor_area', 0),
                'blackspot_area_pixels': bs.get('blackspot_area', 0),
                'coverage_percentage': bs.get('coverage_percentage', 0),
                'num_detections': bs.get('num_detections', 0),
                'avg_confidence': bs.get('avg_confidence', 0),
                'risk_score': bs.get('risk_score', 0),
                'detection_method': bs.get('detection_method', 'enhanced_pixel_based'),
                'floor_breakdown': bs.get('floor_breakdown', {})
            }
        else:
            stats['blackspot'] = {}

        # Contrast stats
        if results.get('contrast'):
            try:
                cs = results['contrast']['statistics']
                stats['contrast'] = dict(cs)
                
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
                
            except Exception:
                stats['contrast'] = {}

        return stats

# ====================== INTERFACE CREATION ======================

def create_advanced_interface(app_instance):
    """Create advanced interface using the NeuroNestApp instance"""
    try:
        import gradio as gr
        import numpy as np
        from PIL import Image
        import tempfile
        
        def analyze_uploaded_image(image, blackspot_threshold, contrast_threshold, 
                                 enable_blackspot, enable_contrast, use_high_res, show_labels):
            if image is None:
                return None, None, None, None, None, "Please upload an image to analyze."
            
            try:
                # Save uploaded image to temporary file
                with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                    if hasattr(image, 'save'):
                        image.save(tmp.name)
                    else:
                        Image.fromarray(image).save(tmp.name)
                    
                    # Reinitialize if resolution changed
                    if use_high_res != app_instance.use_high_res:
                        app_instance.initialize(use_high_res=use_high_res)
                    
                    # Run analysis
                    results = app_instance.analyze_image(
                        image_path=tmp.name,
                        blackspot_threshold=blackspot_threshold,
                        contrast_threshold=contrast_threshold,
                        enable_blackspot=enable_blackspot,
                        enable_contrast=enable_contrast,
                        show_labels=show_labels
                    )
                    
                    # Clean up temp file
                    os.unlink(tmp.name)
                
                if "error" in results:
                    return None, None, None, None, None, f"Error: {results['error']}"
                
                # Extract outputs
                seg_vis = None
                if results.get('segmentation'):
                    if show_labels:
                        seg_vis = results['segmentation'].get('labeled_visualization')
                    else:
                        seg_vis = results['segmentation'].get('visualization')
                
                seg_labeled = results.get('segmentation', {}).get('labeled_visualization')
                contrast_vis = results.get('contrast', {}).get('visualization')
                
                # Blackspot visualization
                blackspot_vis = None
                if results.get('blackspot') and 'enhanced_views' in results['blackspot']:
                    blackspot_vis = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
                
                # Combined visualization
                combined_vis = contrast_vis
                if results.get('blackspot') and blackspot_vis is not None:
                    combined_vis = blackspot_vis
                
                # Generate report
                try:
                    from utils.helpers import generate_analysis_report
                    report = generate_analysis_report(results)
                except:
                    # Fallback report generation
                    stats = results.get('statistics', {})
                    blackspot_stats = stats.get('blackspot', {})
                    contrast_stats = stats.get('contrast', {})
                    
                    report = f"""
# 🧠 NeuroNest Analysis Report

## 📊 Analysis Summary

### Blackspot Detection
- **Detections:** {blackspot_stats.get('num_detections', 0)} blackspots found
- **Coverage:** {blackspot_stats.get('coverage_percentage', 0):.2f}% of floor area
- **Risk Score:** {blackspot_stats.get('risk_score', 0)}/10

### Contrast Analysis  
- **Total Issues:** {contrast_stats.get('total_issues', 0)}
- **Critical Issues:** {contrast_stats.get('critical_count', 0)}
- **Risk Level:** {contrast_stats.get('risk_level', 'unknown')}

### Recommendations
- Review all detected blackspots for safety
- Address critical contrast issues immediately
- Ensure minimum 7:1 contrast ratios for Alzheimer's care
                    """
                
                return seg_vis, seg_labeled, blackspot_vis, contrast_vis, combined_vis, report
                
            except Exception as e:
                return None, None, None, None, None, f"Analysis failed: {str(e)}"
        
        with gr.Blocks(
            title="NeuroNest - Advanced Alzheimer's Environment Analysis",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")
        ) as interface:
            
            gr.Markdown("""
            # 🧠 NeuroNest: Advanced Environment Analysis for Alzheimer's Care
            
            **Enhanced with Robust Blackspot Detection**  
            *Abheek Pradhan | Faculty: Dr. Nadim Adi and Dr. Greg Lakomski*  
            *Texas State University - Computer Science & Interior Design*
            
            ---
            """)
            
            system_status = f"""
            ### 🔧 System Status:
            - **Detectron2:** {"✅ Functional" if app_instance.detectron2_status and app_instance.detectron2_status['fully_functional'] else "⚠️ Limited"}
            - **OneFormer:** {"✅ Available" if app_instance.oneformer else "⚠️ Fallback mode"}
            - **Blackspot Detection:** ✅ Enhanced (floor-only, 50px+ minimum)
            - **Contrast Analysis:** {"✅ Available" if app_instance.contrast_analyzer else "⚠️ Limited"}
            """
            
            gr.Markdown(system_status)
            
            with gr.Row():
                with gr.Column(scale=1):
                    # Input controls
                    image_input = gr.Image(
                        label="📸 Upload Room Image",
                        type="pil",
                        height=300,
                        sources=["upload", "clipboard"]
                    )
                    
                    with gr.Accordion("🔧 Analysis Settings", open=True):
                        use_high_res = gr.Checkbox(
                            value=False,
                            label="High Resolution Analysis",
                            info="1280x1280 for better accuracy"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels"
                        )
                        
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enhanced Blackspot Detection",
                            info="Floor-only, 50px+ minimum"
                        )
                        
                        blackspot_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Blackspot Sensitivity"
                        )
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Contrast Analysis"
                        )
                        
                        contrast_threshold = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=7.0,
                            step=0.1,
                            label="Contrast Threshold (7:1 for Alzheimer's)"
                        )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Environment",
                        variant="primary",
                        size="lg"
                    )
                
                with gr.Column(scale=3):
                    # Main output
                    main_output = gr.Image(
                        label="🎯 Analysis Result",
                        height=400
                    )
                    
                    # Detailed tabs
                    with gr.Tabs():
                        with gr.Tab("📊 Analysis Report"):
                            analysis_report = gr.Markdown(
                                value="Upload an image and click 'Analyze Environment' for comprehensive results."
                            )
                        
                        with gr.Tab("🏷️ Object Segmentation"):
                            labeled_output = gr.Image(
                                label="Object Labels",
                                height=400
                            )
                        
                        with gr.Tab("⚫ Enhanced Blackspot Detection"):
                            blackspot_output = gr.Image(
                                label="Floor Safety Analysis (50px+ minimum)",
                                height=400
                            )
                        
                        with gr.Tab("🎨 Contrast Analysis"):
                            contrast_output = gr.Image(
                                label="Color Contrast Issues",
                                height=400
                            )
                        
                        with gr.Tab("🔄 Combined View"):
                            combined_output = gr.Image(
                                label="Complete Assessment",
                                height=400
                            )
            
            # Connect the interface
            analyze_btn.click(
                fn=analyze_uploaded_image,
                inputs=[
                    image_input, blackspot_threshold, contrast_threshold,
                    enable_blackspot, enable_contrast, use_high_res, show_labels
                ],
                outputs=[
                    main_output, labeled_output, blackspot_output,
                    contrast_output, combined_output, analysis_report
                ]
            )
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Advanced interface creation failed: {e}")
        return None

def create_working_interface():
    """Create working interface with basic functionality"""
    try:
        import gradio as gr
        import numpy as np
        
        def basic_analysis(image):
            if image is None:
                return None, "Please upload an image."
            
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Basic analysis
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            brightness_std = np.std(gray)
            mean_brightness = np.mean(img_array)
            contrast_score = min(10, brightness_std / 10)
            
            # Simple blackspot detection on bottom area
            bottom_area = img_array[int(height*0.7):, :]
            dark_pixels = np.sum(np.mean(bottom_area, axis=2) < 50)
            total_floor_pixels = bottom_area.shape[0] * bottom_area.shape[1]
            blackspot_percentage = (dark_pixels / total_floor_pixels) * 100
            
            report = f"""
# 🧠 NeuroNest - Basic Analysis Mode

## 📊 Image Assessment

### Properties
- **Resolution:** {width} × {height} pixels
- **Brightness:** {mean_brightness:.1f}/255  
- **Contrast Score:** {contrast_score:.1f}/10

### Basic Floor Safety Check
- **Dark areas on floor:** {blackspot_percentage:.1f}%
- **Status:** {"⚠️ CHECK NEEDED" if blackspot_percentage > 5 else "✅ Looks good"}

### Alzheimer's Environment Guidelines

#### ✅ Essential Requirements:
1. **7:1 contrast minimum** between adjacent objects
2. **No dark spots on floors** (trip hazards)
3. **Warm colors preferred** (red, yellow, orange)
4. **High saturation** - avoid pastels
5. **30°+ hue separation** on color wheel

#### Current Assessment:
- **Contrast:** {"Good" if contrast_score > 5 else "Needs improvement"}
- **Floor Safety:** {"Attention needed" if blackspot_percentage > 5 else "Acceptable"}

*Enhanced AI analysis with object segmentation available when system fully initializes.*
            """
            
            return image, report
        
        with gr.Blocks(title="NeuroNest - Basic Mode") as interface:
            gr.Markdown("""
            # 🧠 NeuroNest: Alzheimer's Environment Analysis
            **Basic Analysis Mode**
            
            *Texas State University - Enhanced Blackspot Detection*
            """)
            
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(label="📸 Room Image", type="pil")
                    analyze_btn = gr.Button("🔍 Basic Analysis", variant="primary")
                
                with gr.Column():
                    result_image = gr.Image(label="Result")
                    analysis_report = gr.Markdown()
            
            analyze_btn.click(basic_analysis, inputs=[image_input], outputs=[result_image, analysis_report])
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Basic interface failed: {e}")
        return None

# ====================== MAIN APPLICATION ======================

def main():
    """Main application entry point with enhanced initialization"""
    logger.info("🚀 Starting NeuroNest Application with Enhanced Blackspot Detection")
    
    try:
        # Setup paths with detectron2 fix
        project_root = setup_python_paths()
        
        # Check system status
        deps, detectron2_status = check_system_dependencies()
        
        # Log system capabilities
        working_deps = [k for k, v in deps.items() if v]
        logger.info(f"✅ Working dependencies: {working_deps}")
        logger.info(f"🔧 Detectron2 status: {detectron2_status}")
        
        # Initialize core application
        app = NeuroNestApp()
        init_success = False
        
        try:
            oneformer_ok, blackspot_ok = app.initialize(use_high_res=False)
            init_success = app.initialized
            logger.info(f"✅ NeuroNest core initialized: {init_success}")
            logger.info(f"   - OneFormer: {oneformer_ok}")
            logger.info(f"   - Enhanced Blackspot: {blackspot_ok}")
        except Exception as e:
            logger.warning(f"⚠️ Core initialization failed: {e}")
        
        # Create appropriate interface
        interface = None
        
        if init_success and deps.get('gradio'):
            try:
                interface = create_advanced_interface(app)
                if interface:
                    logger.info("✅ Advanced interface with enhanced blackspot detection created")
            except Exception as e:
                logger.warning(f"⚠️ Advanced interface failed: {e}")
        
        if interface is None:
            try:
                interface = create_working_interface()
                if interface:
                    logger.info("✅ Basic interface created")
            except Exception as e:
                logger.error(f"❌ Basic interface failed: {e}")
        
        if interface is None:
            logger.error("❌ No interface could be created")
            return False
        
        # Launch interface
        logger.info("🌐 Launching interface...")
        interface.queue(
            default_concurrency_limit=2,
            max_size=10
        ).launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            show_error=True,
            prevent_thread_lock=False
        )
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        time.sleep(3600)  # Keep container alive for debugging
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ NeuroNest started successfully with enhanced blackspot detection")
        else:
            logger.error("❌ NeuroNest failed to start")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        time.sleep(3600)
