"""Enhanced Gradio interface for NeuroNest"""

import gradio as gr
import numpy as np
import tempfile
import os
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional

from core.neuronest import NeuroNestApp
from utils.report_generator import ReportGenerator

logger = None


def create_interface():
    """Create the enhanced Gradio interface"""
    global logger
    import logging
    logger = logging.getLogger(__name__)
    
    # Initialize application
    app = NeuroNestApp()
    report_generator = ReportGenerator()
    
    # Initialize with standard resolution by default
    oneformer_ok, blackspot_ok = app.initialize(use_high_res=False)
    
    if not app.initialized:
        raise RuntimeError("Failed to initialize NeuroNest components")
    
    def analyze_comprehensive(image, blackspot_threshold, contrast_threshold, 
                            enable_blackspot, enable_contrast, show_labels, use_high_res):
        """Comprehensive analysis wrapper"""
        if image is None:
            return [None] * 6 + ["Please upload an image to analyze."]
        
        try:
            # Check if we need to reinitialize with different resolution
            if hasattr(app, 'use_high_res') and use_high_res != app.use_high_res:
                gr.Info(f"Reinitializing with {'high' if use_high_res else 'standard'} resolution...")
                app.initialize(use_high_res=use_high_res)
                app.use_high_res = use_high_res
            
            # Save image to temporary file
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
                if hasattr(image, 'save'):
                    image.save(tmp.name)
                else:
                    Image.fromarray(image).save(tmp.name)
                
                # Run analysis
                results = app.analyze_image(
                    image_path=tmp.name,
                    enable_blackspot=enable_blackspot,
                    enable_contrast=enable_contrast
                )
                
                os.unlink(tmp.name)
            
            if "error" in results:
                return [None] * 6 + [f"❌ Error: {results['error']}"]
            
            # Extract all visualizations
            combined_vis = results.get('combined', image)
            seg_vis = None
            seg_labeled = None
            blackspot_vis = None
            contrast_vis = None
            
            # Segmentation visualizations
            if results.get('segmentation'):
                seg_vis = results['segmentation'].get('visualization')
                
                # Show labeled version based on preference
                if show_labels:
                    seg_labeled = results['segmentation'].get('labeled_visualization')
                else:
                    seg_labeled = results['segmentation'].get('visualization')
            
            # Blackspot visualization
            if results.get('blackspot') and 'enhanced_views' in results['blackspot']:
                blackspot_vis = results['blackspot']['enhanced_views'].get('high_contrast_overlay')
            
            # Contrast visualization (with tinting)
            if results.get('contrast'):
                contrast_vis = results['contrast'].get('visualization')
            
            # Generate comprehensive report
            report = report_generator.generate_comprehensive_report(results)
            
            # Return all visualizations
            return (combined_vis, seg_vis, seg_labeled, blackspot_vis, 
                   contrast_vis, combined_vis, report)
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            import traceback
            traceback.print_exc()
            return [None] * 6 + [f"Analysis failed: {str(e)}"]
    
    # Create interface with custom CSS
    css = """
    .gradio-container {
        font-family: 'Arial', sans-serif;
    }
    .output-markdown {
        max-height: 600px;
        overflow-y: auto;
        padding: 15px;
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    .gr-button-primary {
        background-color: #ff6b35 !important;
        border-color: #ff6b35 !important;
    }
    .gr-button-primary:hover {
        background-color: #ff5722 !important;
        border-color: #ff5722 !important;
    }
    """
    
    with gr.Blocks(
        title="NeuroNest - Alzheimer's Environment Analysis",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css=css
    ) as interface:
        
        gr.Markdown("""
        # 🧠 NeuroNest: AI-Powered Alzheimer's Environment Analysis
        
        **Comprehensive Safety Assessment for Dementia-Friendly Spaces**  
        *Developed by: Abheek Pradhan | Faculty: Dr. Nadim Adi & Dr. Greg Lakomski*  
        *Texas State University - Computer Science & Interior Design Collaboration*
        
        ---
        """)
        
        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="pil",
                    height=400
                )
                
                with gr.Accordion("🎛️ Analysis Settings", open=True):
                    with gr.Group():
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enable Blackspot Detection",
                            info="Detect dark floor areas (trip hazards)"
                        )
                        
                        blackspot_threshold = gr.Slider(
                            minimum=0.1, maximum=0.9, value=0.5, step=0.05,
                            label="Blackspot Sensitivity",
                            visible=False  # Hide for simplicity
                        )
                    
                    with gr.Group():
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Enable Contrast Analysis",
                            info="Check color contrast (7:1 Alzheimer's standard)"
                        )
                        
                        contrast_threshold = gr.Slider(
                            minimum=1.0, maximum=10.0, value=7.0, step=0.1,
                            label="Contrast Threshold",
                            info="7.0 recommended for Alzheimer's care"
                        )
                    
                    with gr.Group():
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels",
                            info="Display object names on segmentation"
                        )
                        
                        use_high_res = gr.Checkbox(
                            value=False,
                            label="High Resolution Mode",
                            info="Better accuracy, slower processing"
                        )
                
                analyze_btn = gr.Button(
                    "🔍 Analyze Environment",
                    variant="primary",
                    size="lg"
                )
                
                # Status indicators
                with gr.Accordion("📊 System Status", open=False):
                    gr.Markdown(f"""
                    - **OneFormer:** {'✅ Ready' if oneformer_ok else '❌ Unavailable'}
                    - **Blackspot:** {'✅ Ready' if blackspot_ok else '❌ Unavailable'}
                    - **Contrast:** ✅ Ready
                    """)
            
            # Output Column
            with gr.Column(scale=3):
                # Tabbed interface for different views
                with gr.Tabs():
                    with gr.Tab("🔄 Combined Analysis"):
                        combined_output = gr.Image(
                            label="All Issues Highlighted",
                            height=500
                        )
                    
                    with gr.Tab("🏷️ Object Segmentation"):
                        with gr.Row():
                            seg_output = gr.Image(
                                label="Segmented View",
                                height=400
                            )
                            seg_labeled_output = gr.Image(
                                label="Labeled Objects",
                                height=400
                            )
                    
                    with gr.Tab("⚫ Blackspot Detection"):
                        blackspot_output = gr.Image(
                            label="Floor Blackspot Analysis",
                            height=500
                        )
                    
                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_output = gr.Image(
                            label="Color Contrast Issues (Tinted)",
                            height=500
                        )
                    
                    with gr.Tab("📊 Detailed Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' for detailed results.",
                            elem_classes=["output-markdown"]
                        )
        
        # Connect interface
        analyze_btn.click(
            fn=analyze_comprehensive,
            inputs=[
                image_input, blackspot_threshold, contrast_threshold,
                enable_blackspot, enable_contrast, show_labels, use_high_res
            ],
            outputs=[
                combined_output, seg_output, seg_labeled_output,
                blackspot_output, contrast_output, combined_output,
                analysis_report
            ]
        )
        
        # Add examples if available
        example_dir = Path("examples")
        if example_dir.exists():
            example_files = list(example_dir.glob("*.jpg")) + list(example_dir.glob("*.png"))
            if example_files:
                gr.Examples(
                    examples=[[str(f)] for f in example_files[:3]],
                    inputs=[image_input],
                    label="Example Images"
                )
        
        # Information section
        with gr.Accordion("ℹ️ About NeuroNest", open=False):
            gr.Markdown("""
            NeuroNest uses advanced AI to analyze indoor environments for Alzheimer's safety:
            
            - **Object Segmentation:** Identifies 150+ indoor objects and furniture
            - **Blackspot Detection:** Finds dark floor areas that may cause falls  
            - **Contrast Analysis:** Ensures 7:1 color contrast for dementia visibility
            - **Evidence-Based:** Follows clinical guidelines for dementia care
            
            All analysis is performed locally for privacy.
            """)
    
    return interface
