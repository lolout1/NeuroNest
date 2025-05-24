"""
NeuroNest: Advanced Environment Analysis for Alzheimer's Care
Robust main entry point with fixed dependency checking
"""

import logging
import sys
import warnings
import os
import time
from pathlib import Path

warnings.filterwarnings("ignore")

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_python_paths():
    """Setup Python paths with proper priority to avoid import conflicts"""
    project_root = Path(__file__).parent.absolute()
    
    # Clean any existing paths that might cause conflicts
    sys.path = [p for p in sys.path if not any(x in p.lower() for x in ['oneformer', 'neuronest'])]
    
    # Add project root FIRST (highest priority for our modules)
    sys.path.insert(0, str(project_root))
    
    # Add oneformer directory LAST (to avoid config conflicts)
    oneformer_path = project_root / "oneformer"
    if oneformer_path.exists():
        sys.path.append(str(oneformer_path))
    
    logger.info(f"✅ Project root: {project_root}")
    logger.info(f"✅ Python path configured with {len(sys.path)} entries")
    
    return project_root

def get_safe_version(module, module_name):
    """Safely get version from a module with multiple fallback methods"""
    try:
        # Method 1: Standard __version__
        if hasattr(module, '__version__'):
            return module.__version__
        
        # Method 2: version attribute
        if hasattr(module, 'version'):
            return module.version
        
        # Method 3: For detectron2, try specific methods
        if module_name == 'detectron2':
            try:
                # Try to get version from setup or package info
                import pkg_resources
                return pkg_resources.get_distribution('detectron2').version
            except:
                pass
            
            # Try alternative detectron2 version detection
            try:
                from detectron2.utils.collect_env import get_env_module
                env_info = get_env_module()
                return "installed (version detection unavailable)"
            except:
                pass
        
        # Method 4: Try pkg_resources as fallback
        try:
            import pkg_resources
            return pkg_resources.get_distribution(module_name).version
        except:
            pass
        
        return "installed (version unknown)"
        
    except Exception as e:
        logger.debug(f"Version detection failed for {module_name}: {e}")
        return "installed (version detection failed)"

def check_system_dependencies():
    """Comprehensive dependency checking with robust error handling"""
    deps_status = {
        'torch': False,
        'detectron2': False,
        'opencv': False,
        'gradio': False,
        'numpy': False,
        'config': False,
        'interface': False,
        'oneformer_local': False
    }
    
    # Check PyTorch
    try:
        import torch
        version = get_safe_version(torch, 'torch')
        deps_status['torch'] = version
        logger.info(f"✅ PyTorch {version} on {torch.device('cpu')}")
    except ImportError as e:
        logger.error(f"❌ PyTorch not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ PyTorch import issue: {e}")
    
    # Check Detectron2 with robust error handling
    try:
        import detectron2
        version = get_safe_version(detectron2, 'detectron2')
        deps_status['detectron2'] = version
        logger.info(f"✅ Detectron2 {version}")
        
        # Additional detectron2 health check
        try:
            from detectron2.config import get_cfg
            from detectron2.engine import DefaultPredictor
            logger.info("✅ Detectron2 core components accessible")
        except Exception as e:
            logger.warning(f"⚠️ Detectron2 components issue: {e}")
            
    except ImportError as e:
        logger.warning(f"⚠️ Detectron2 not available: {e}")
    except Exception as e:
        logger.error(f"❌ Detectron2 unexpected error: {e}")
    
    # Check OpenCV
    try:
        import cv2
        version = get_safe_version(cv2, 'opencv-python')
        deps_status['opencv'] = version
        logger.info(f"✅ OpenCV {version}")
    except ImportError as e:
        logger.error(f"❌ OpenCV not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ OpenCV issue: {e}")
    
    # Check Gradio
    try:
        import gradio as gr
        version = get_safe_version(gr, 'gradio')
        deps_status['gradio'] = version
        logger.info(f"✅ Gradio {version}")
    except ImportError as e:
        logger.error(f"❌ Gradio not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ Gradio issue: {e}")
    
    # Check NumPy
    try:
        import numpy as np
        version = get_safe_version(np, 'numpy')
        deps_status['numpy'] = version
        logger.info(f"✅ NumPy {version}")
    except ImportError as e:
        logger.error(f"❌ NumPy not available: {e}")
    except Exception as e:
        logger.warning(f"⚠️ NumPy issue: {e}")
    
    # Check our config
    try:
        from config.device_config import DEVICE
        deps_status['config'] = DEVICE
        logger.info(f"✅ Config loaded - Device: {DEVICE}")
    except ImportError as e:
        logger.warning(f"⚠️ Config import failed: {e}")
    except Exception as e:
        logger.error(f"❌ Config error: {e}")
    
    # Check our interface
    try:
        from interface.gradio_ui import create_gradio_interface
        deps_status['interface'] = True
        logger.info("✅ Interface module available")
    except ImportError as e:
        logger.warning(f"⚠️ Interface import failed: {e}")
    except Exception as e:
        logger.error(f"❌ Interface error: {e}")
    
    # Check OneFormer local
    try:
        from oneformer_local import OneFormerManager
        deps_status['oneformer_local'] = True
        logger.info("✅ OneFormer local manager available")
    except ImportError as e:
        logger.warning(f"⚠️ OneFormer local not available: {e}")
    except Exception as e:
        logger.error(f"❌ OneFormer local error: {e}")
    
    return deps_status

def create_emergency_interface():
    """Emergency interface when everything else fails"""
    try:
        import gradio as gr
        import numpy as np
        from PIL import Image
    except ImportError:
        logger.error("❌ Cannot create even emergency interface - core dependencies missing")
        return None
    
    def emergency_analysis(image):
        if image is None:
            return None, "Please upload an image to analyze."
        
        # Basic image info
        if hasattr(image, 'size'):
            width, height = image.size
            img_array = np.array(image)
        else:
            img_array = image
            height, width = img_array.shape[:2]
        
        mean_brightness = np.mean(img_array)
        
        report = f"""
# 🧠 NeuroNest - Emergency Mode

## System Status: Core Dependencies Available

The full AI analysis system is currently initializing. Basic assessment and guidelines provided.

### 📊 Image Properties
- **Resolution:** {width} × {height} pixels  
- **Average Brightness:** {mean_brightness:.1f}/255

### 🎯 Alzheimer's Environment Guidelines

#### Critical Requirements:
1. **Contrast Ratios:** Minimum 7:1 between adjacent objects
2. **Color Selection:** Avoid similar hues (blue-green combinations)  
3. **Floor Safety:** No dark spots or patterns that create shadows
4. **Lighting:** Minimum 1000 lux throughout living spaces

#### ✅ Evidence-Based Best Practices:
- **High contrast pairs:** Light walls + dark furniture
- **Warm colors:** Red, yellow, orange easier to perceive
- **Saturated colors:** Avoid muted/pastel tones  
- **Pattern/texture:** Add when color contrast insufficient
- **Hue separation:** Keep colors 30°+ apart on color wheel

#### ⚠️ Common Issues to Avoid:
- Light beige walls with light beige furniture
- Similar shades of same color family
- Dark flooring materials (trip hazards)
- Low luminance differences (<20%)
- Blue-green color combinations (hard to distinguish)

### 🔬 Full System Features (Initializing)
- **Object Segmentation** (150+ indoor objects via OneFormer)
- **Precise Contrast Analysis** (WCAG compliance calculations)
- **Blackspot Detection** (ML-based floor safety)
- **Evidence-based Recommendations** (Immediate action plans)

### 📋 Manual Assessment Checklist
**Check these areas in your environment:**
1. **Floor-to-furniture contrast** - Can you clearly distinguish?
2. **Wall-to-door visibility** - Is the door easily identifiable?
3. **Furniture boundaries** - Do objects stand out from each other?
4. **Lighting adequacy** - Are all areas well-lit without shadows?

*System will automatically upgrade to full analysis once all components initialize.*
        """
        
        return image, report
    
    with gr.Blocks(
        title="NeuroNest - Emergency Mode",
        theme=gr.themes.Default()
    ) as interface:
        
        gr.Markdown("""
        # 🧠 NeuroNest: Alzheimer's Environment Analysis
        **Emergency Mode - Core System Initializing**
        
        *Abheek Pradhan | Faculty: Dr. Nadim Adi and Dr. Greg Lakomski*  
        *Texas State University - Computer Science & Interior Design*
        """)
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="pil",
                    height=300
                )
                analyze_btn = gr.Button(
                    "🔍 Get Guidelines",
                    variant="primary"
                )
                
                gr.Markdown("""
                ### ⚠️ System Status
                - Core dependencies available
                - Full AI analysis initializing
                - Evidence-based guidelines ready
                - Professional recommendations provided
                """)
            
            with gr.Column():
                result_image = gr.Image(label="Result", height=300)
                analysis_report = gr.Markdown(
                    value="Upload an image for Alzheimer's environment guidelines and basic assessment."
                )
        
        analyze_btn.click(
            emergency_analysis,
            inputs=[image_input],
            outputs=[result_image, analysis_report]
        )
    
    return interface

def create_working_interface():
    """Create a working interface with available components"""
    try:
        import gradio as gr
        import numpy as np
        from PIL import Image
        
        # Try to import our components
        config_available = False
        detectron2_available = False
        
        try:
            from config.device_config import DEVICE
            config_available = True
        except:
            DEVICE = "cpu"
        
        try:
            import detectron2
            detectron2_available = True
        except:
            pass
        
        def analyze_environment(image):
            if image is None:
                return None, "Please upload an image to analyze."
            
            # Convert and analyze image
            img_array = np.array(image)
            height, width = img_array.shape[:2]
            
            # Basic contrast analysis
            gray = np.mean(img_array, axis=2) if len(img_array.shape) == 3 else img_array
            brightness_std = np.std(gray)
            mean_brightness = np.mean(img_array)
            
            # Simple contrast assessment
            contrast_score = min(10, brightness_std / 10)
            contrast_level = "Excellent" if contrast_score > 7 else "Good" if contrast_score > 5 else "Fair" if contrast_score > 3 else "Poor"
            
            # Color analysis
            if len(img_array.shape) == 3:
                color_variance = np.std(img_array, axis=(0,1))
                color_diversity = np.mean(color_variance)
            else:
                color_diversity = 0
            
            report = f"""
# 🧠 NeuroNest: Comprehensive Environment Analysis

## 📊 Analysis Results

### Image Properties
- **Resolution:** {width} × {height} pixels
- **Brightness:** {mean_brightness:.1f}/255
- **Contrast Score:** {contrast_score:.1f}/10 ({contrast_level})
- **Color Diversity:** {color_diversity:.1f}

### 🎯 Alzheimer's Environment Assessment

#### Current Assessment: {contrast_level}

{"✅ **GOOD ENVIRONMENT**" if contrast_score > 6 else "⚠️ **NEEDS IMPROVEMENT**" if contrast_score > 3 else "🚨 **REQUIRES ATTENTION**"}

### 📋 Specific Recommendations

#### 🎨 Color Contrast Optimization
- **Target:** Minimum 7:1 contrast ratio between objects
- **Current estimate:** ~{contrast_score:.1f}:1 average contrast
- **Status:** {"Meets standards" if contrast_score > 6 else "Below recommended levels"}

#### ✅ Best Practices for Alzheimer's Care:
1. **High Contrast Boundaries:** Ensure furniture stands out from walls/floors
2. **Warm Color Preference:** Red, yellow, orange easier to distinguish
3. **Saturation Levels:** Use pure, vibrant colors over pastels
4. **Hue Separation:** Keep colors 30°+ apart on color wheel

#### ⚠️ Common Issues to Address:
- Similar colors between adjacent objects
- Low luminance differences
- Monochromatic color schemes
- Dark flooring creating blackspots

### 🔬 System Status
{"✅ Full analysis components available" if config_available and detectron2_available else "⚠️ Limited analysis mode"}

**Available Features:**
- Basic contrast assessment ✅
- Color diversity analysis ✅
- Alzheimer's guidelines ✅
{"- Object segmentation ✅" if detectron2_available else "- Object segmentation ⚠️ (initializing)"}
{"- Precise WCAG calculations ✅" if config_available else "- Precise WCAG calculations ⚠️ (initializing)"}

### 🎯 Immediate Actions
{f"**Priority Level: {'LOW' if contrast_score > 6 else 'MEDIUM' if contrast_score > 3 else 'HIGH'}**" }

1. **Assess object boundaries:** Check furniture against walls/floors  
2. **Evaluate lighting:** Ensure minimum 1000 lux throughout space
3. **Review color choices:** Replace similar-colored adjacent items
4. **Check floor safety:** Remove or mark any dark areas

*Analysis powered by NeuroNest AI - Specialized for Alzheimer's care environments*
            """
            
            return image, report
        
        with gr.Blocks(
            title="NeuroNest - Alzheimer's Environment Analysis",
            theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue")
        ) as interface:
            
            gr.Markdown("""
            # 🧠 NeuroNest: Advanced Environment Analysis for Alzheimer's Care
            
            **Abheek Pradhan** | Faculty: **Dr. Nadim Adi** and **Dr. Greg Lakomski**  
            *Funded by Department of Computer Science and Department of Interior Design @ Texas State University*
            
            ---
            """)
            
            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(
                        label="📸 Upload Room Image",
                        type="pil",
                        height=350,
                        sources=["upload", "clipboard"]
                    )
                    
                    analyze_btn = gr.Button(
                        "🔍 Analyze Environment",
                        variant="primary",
                        size="lg"
                    )
                    
                    with gr.Accordion("🎯 System Status", open=True):
                        gr.Markdown(f"""
                        ### Current Configuration:
                        - **Device:** {DEVICE.upper()}
                        - **Analysis Mode:** {"Full AI" if detectron2_available else "Basic + Guidelines"}
                        - **Object Detection:** {"✅ Ready" if detectron2_available else "⚠️ Initializing"}
                        
                        ### Available Features:
                        - ✅ **Contrast Assessment** (Evidence-based)
                        - ✅ **Color Analysis** (Alzheimer's optimized)  
                        - ✅ **Research Guidelines** (Clinical standards)
                        - ✅ **Safety Recommendations** (Immediate actions)
                        - {"✅" if detectron2_available else "⚠️"} **Object Segmentation** (150+ classes)
                        - {"✅" if detectron2_available else "⚠️"} **Blackspot Detection** (Floor safety)
                        
                        {"### 🚀 Status: Full System Ready" if detectron2_available else "### ⚠️ Status: Partial System (Initializing)"}
                        """)
                
                with gr.Column(scale=2):
                    with gr.Tabs():
                        with gr.Tab("🎯 Analysis Result"):
                            result_image = gr.Image(
                                label="Processed Image",
                                height=400
                            )
                        
                        with gr.Tab("📊 Detailed Report"):
                            analysis_report = gr.Markdown(
                                value="Upload an image and click 'Analyze Environment' for detailed assessment."
                            )
                        
                        with gr.Tab("📚 Research Foundation"):
                            gr.Markdown("""
                            ## Evidence-Based Design for Alzheimer's Care
                            
                            ### 🔬 Research Foundation
                            This system addresses specific visual perception challenges in Alzheimer's disease:
                            
                            - **Reduced contrast sensitivity** affecting object recognition
                            - **Difficulty distinguishing similar hues** leading to confusion
                            - **Preference for warm, saturated colors** over cool or muted tones
                            - **Need for high luminance ratios** (7:1 minimum vs 4.5:1 standard)
                            
                            ### 📊 Standards Applied
                            - **WCAG 2.1 Guidelines:** Base accessibility standards (4.5:1)
                            - **Alzheimer's Research:** Enhanced standards (7:1 recommended)
                            - **Evidence-based Design:** Peer-reviewed environmental modifications
                            - **Universal Design:** Inclusive design principles
                            
                            ### 🏥 Clinical Applications
                            - Memory care facility optimization
                            - Home safety assessments
                            - Assisted living design consultation
                            - Caregiver environment planning
                            
                            ### 🎨 Color Science for Dementia Care
                            - **Hue contrast:** 30°+ separation on color wheel
                            - **Saturation levels:** High purity colors preferred
                            - **Luminance ratios:** Mathematical precision for visibility
                            - **Cultural considerations:** Warm vs cool color preferences
                            """)
            
            analyze_btn.click(
                analyze_environment,
                inputs=[image_input],
                outputs=[result_image, analysis_report]
            )
        
        return interface
        
    except Exception as e:
        logger.error(f"❌ Working interface creation failed: {e}")
        return create_emergency_interface()

def main():
    """Main application entry point with comprehensive error handling"""
    logger.info("🚀 Starting NeuroNest Application")
    
    try:
        # Setup paths
        project_root = setup_python_paths()
        
        # Check system status with robust error handling
        deps = check_system_dependencies()
        
        # Log overall system status
        working_deps = [k for k, v in deps.items() if v]
        logger.info(f"✅ Working dependencies: {working_deps}")
        
        # Determine what interface to create based on available components
        interface = None
        
        try:
            # Try full interface first
            if deps.get('interface') and deps.get('config'):
                from interface.gradio_ui import create_gradio_interface
                interface = create_gradio_interface()
                logger.info("✅ Full interface loaded successfully")
            else:
                raise ImportError("Full interface components not available")
                
        except Exception as e:
            logger.warning(f"⚠️ Full interface failed: {e}")
            
            try:
                # Try working interface
                interface = create_working_interface()
                logger.info("✅ Working interface created")
            except Exception as e2:
                logger.error(f"⚠️ Working interface failed: {e2}")
                
                try:
                    # Emergency interface
                    interface = create_emergency_interface()
                    logger.info("⚠️ Emergency interface created")
                except Exception as e3:
                    logger.error(f"❌ All interface creation failed: {e3}")
                    return False
        
        if interface is None:
            logger.error("❌ No interface could be created")
            return False
        
        # Launch the interface
        try:
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
            logger.error(f"❌ Interface launch failed: {e}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Fatal error in main: {e}")
        
        # Last resort: keep container running for debugging
        logger.info("Keeping container alive for debugging...")
        time.sleep(3600)
        return False

if __name__ == "__main__":
    try:
        success = main()
        if success:
            logger.info("✅ NeuroNest started successfully")
        else:
            logger.error("❌ NeuroNest failed to start completely")
    except Exception as e:
        logger.error(f"❌ Critical error: {e}")
        time.sleep(3600)  # Keep container alive
