"""Enhanced Gradio interface with resolution options and label visualization."""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional

from core import NeuroNestApp
from utils.helpers import generate_analysis_report  # Explicit import


def create_gradio_interface():
    """Create the enhanced Gradio interface with resolution and labeling options"""

    # Initialize the application
    app = None
    
    def initialize_app(use_high_res: bool = False) -> Tuple[bool, bool]:
        """Initialize or reinitialize the app with specified resolution"""
        nonlocal app
        app = NeuroNestApp()
        return app.initialize(use_high_res=use_high_res)

    # Initialize with standard resolution by default
    oneformer_ok, blackspot_ok = initialize_app(use_high_res=False)

    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")

    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold,
                       enable_blackspot, enable_contrast, visualization_mode,
                       use_high_res, show_labels):
        """Enhanced wrapper function for comprehensive analysis"""
        if image_path is None:
            return None, None, None, None, None, "Please upload an image"

        try:
            # Check if we need to reinitialize with different resolution
            if use_high_res != app.use_high_res:
                gr.Info(f"Reinitializing with {'high' if use_high_res else 'standard'} resolution...")
                oneformer_ok, blackspot_ok = initialize_app(use_high_res=use_high_res)
                if not oneformer_ok:
                    return None, None, None, None, None, "Failed to reinitialize with new resolution"

            # Run analysis
            results = app.analyze_image(
                image_path=image_path,
                blackspot_threshold=blackspot_threshold,
                contrast_threshold=contrast_threshold,
                enable_blackspot=enable_blackspot,
                enable_contrast=enable_contrast,
                show_labels=show_labels
            )

            if "error" in results:
                return None, None, None, None, None, f"Error: {results['error']}"

            # Extract segmentation outputs
            seg_visualization = None
            seg_labeled = None
            
            if results['segmentation']:
                if show_labels:
                    seg_visualization = results['segmentation'].get('labeled_visualization')
                else:
                    seg_visualization = results['segmentation'].get('visualization')
                
                # Always get both for the dedicated tab
                seg_labeled = results['segmentation'].get('labeled_visualization')

            # Enhanced visualization selection
            main_output = seg_visualization  # Default
            blackspot_output = None
            contrast_output = None
            combined_output = None

            if results['blackspot'] and 'enhanced_views' in results['blackspot']:
                blackspot_views = results['blackspot']['enhanced_views']

                # Select blackspot visualization based on mode
                if visualization_mode == "Side by Side":
                    blackspot_output = blackspot_views.get('side_by_side')
                elif visualization_mode == "High Contrast":
                    blackspot_output = blackspot_views.get('high_contrast_overlay')
                elif visualization_mode == "Blackspots Only":
                    blackspot_output = blackspot_views.get('blackspot_only')
                elif visualization_mode == "Annotated":
                    blackspot_output = blackspot_views.get('annotated_view')
                elif visualization_mode == "Segmentation Only":
                    blackspot_output = blackspot_views.get('segmentation_view')
                else:
                    blackspot_output = blackspot_views.get('high_contrast_overlay')

            # Contrast visualization
            if results['contrast']:
                contrast_output = results['contrast'].get('visualization')

            # Combined visualization
            if results['blackspot'] and results['contrast'] and enable_blackspot and enable_contrast:
                contrast_vis = results['contrast'].get('visualization')
                if contrast_vis is not None:
                    combined_output = contrast_vis.copy()

                    # Overlay blackspot areas
                    blackspot_mask = results['blackspot'].get('blackspot_mask')
                    if blackspot_mask is not None and np.any(blackspot_mask):
                        # Resize mask if needed
                        if blackspot_mask.shape != combined_output.shape[:2]:
                            blackspot_mask_resized = cv2.resize(
                                blackspot_mask.astype(np.uint8),
                                (combined_output.shape[1], combined_output.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        else:
                            blackspot_mask_resized = blackspot_mask

                        # Create a more visible overlay
                        overlay = combined_output.copy()
                        overlay[blackspot_mask_resized] = [255, 255, 0]  # Bright yellow for blackspots
                        combined_output = cv2.addWeighted(combined_output, 0.7, overlay, 0.3, 0)

            # Determine main output based on mode and what's enabled
            if visualization_mode == "Combined Analysis" and combined_output is not None:
                main_output = combined_output
            elif visualization_mode in ["High Contrast", "Side by Side", "Blackspots Only", "Annotated"] and blackspot_output is not None:
                main_output = blackspot_output
            elif enable_contrast and contrast_output is not None and not enable_blackspot:
                main_output = contrast_output

            # Generate comprehensive report
            report = generate_analysis_report(results)

            return main_output, seg_labeled, blackspot_output, contrast_output, combined_output, report

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            return None, None, None, None, None, error_msg

    # Create the interface
    title = "🧠 NeuroNest: Comprehensive Alzheimer's Environment Analysis"
    subtitle = "**Abheek Pradhan** | Faculty: **Dr. Nadim Adi** and **Dr. Greg Lakomski**"
    funding = "Funded by Department of Computer Science and Department of Interior Design @ Texas State University"
    
    description = """
    **Advanced AI System for Creating Dementia-Friendly Environments**

    **🎯 Core Features:**
    - **Object Segmentation**: Identifies all room elements with labeled visualization
    - **⚫ Blackspot Detection**: Detects dangerous black/dark areas on floors ONLY
    - **🎨 Adjacency-Based Contrast Analysis**: Analyzes contrast between touching objects only
    - **📊 Evidence-Based Reporting**: Comprehensive safety recommendations
    - **🔍 High Resolution Support**: Optional 896x896 resolution for enhanced accuracy

    **🔬 Detection Capabilities:**
    - Labeled object identification with ADE20K classes
    - Floor-constrained blackspot detection (RGB < 110)
    - Adjacent object pair contrast analysis only
    - Multiple visualization modes for comprehensive analysis
    """

    with gr.Blocks(
        title="NeuroNest - Alzheimer's Environment Analysis",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .title-section { text-align: center; margin-bottom: 1.5rem; }
        .subtitle { font-size: 1.1em; color: #555; margin-top: 0.5rem; }
        .funding { font-size: 0.95em; color: #666; font-style: italic; margin-top: 0.3rem; }
        .analysis-section { border: 2px solid #f0f0f0; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .critical-text { color: #ff0000; font-weight: bold; }
        .high-text { color: #ff8800; font-weight: bold; }
        .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin: 10px 0; }
        .info-box { background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 10px; margin: 10px 0; }
        .success-box { background-color: #d4edda; border: 1px solid #c3e6cb; border-radius: 5px; padding: 10px; margin: 10px 0; }
        """
    ) as interface:

        # Title section with author and funding information
        with gr.Column(elem_classes=["title-section"]):
            gr.Markdown(f"# {title}")
            gr.Markdown(f'<div class="subtitle">{subtitle}</div>', elem_classes=["subtitle"])
            gr.Markdown(f'<div class="funding">{funding}</div>', elem_classes=["funding"])
        
        gr.Markdown(description)

        with gr.Row():
            # Input Column
            with gr.Column(scale=1):
                # Image upload
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="filepath",
                    height=300
                )

                # Analysis settings
                with gr.Accordion("🔧 Analysis Settings", open=True):

                    # Resolution settings
                    with gr.Group():
                        gr.Markdown("### 🔍 Resolution & Display Settings")
                        use_high_res = gr.Checkbox(
                            value=False,
                            label="Use High Resolution (896x896)",
                            info="Higher resolution provides better accuracy but requires more processing time"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels",
                            info="Display object class names on segmentation"
                        )

                    # Blackspot settings
                    with gr.Group():
                        gr.Markdown("### ⚫ Blackspot Detection (Floor-Only)")
                        if blackspot_ok:
                            gr.Markdown('<div class="success-box">✅ <strong>Blackspot Detection Ready</strong><br/>Detects genuinely black/dark areas on floors using validated color analysis.</div>')
                        else:
                            gr.Markdown('<div class="warning-box">⚠️ <strong>Using Color-Based Detection</strong><br/>Model not found, using robust color-based detection.</div>')
                        
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enable Floor Blackspot Detection",
                            interactive=True
                        )

                        blackspot_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Detection Sensitivity",
                            info="Higher = more sensitive to dark areas"
                        )

                    # Contrast settings
                    with gr.Group():
                        gr.Markdown("### 🎨 Adjacent Object Contrast Analysis")
                        gr.Markdown('<div class="info-box">🔬 <strong>Adjacency-Based Detection</strong><br/>Analyzes contrast ONLY between objects that touch each other, ensuring relevant safety assessments.</div>')
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Enable Adjacent Contrast Analysis"
                        )

                        contrast_threshold = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=7.0,
                            step=0.1,
                            label="Alzheimer's Contrast Threshold",
                            info="7:1 recommended for dementia care (vs 4.5:1 standard WCAG)"
                        )

                    # Visualization options
                    with gr.Group():
                        gr.Markdown("### 👁️ Visualization Mode")
                        visualization_mode = gr.Radio(
                            choices=[
                                "High Contrast", 
                                "Side by Side", 
                                "Blackspots Only", 
                                "Segmentation Only",
                                "Annotated",
                                "Combined Analysis"
                            ],
                            value="High Contrast",
                            label="Display Style",
                            info="Choose how to visualize the analysis results"
                        )

                # Analysis button
                analyze_button = gr.Button(
                    "🔍 Analyze Environment",
                    variant="primary",
                    size="lg"
                )

                # Info box
                with gr.Accordion("ℹ️ Technical Information", open=False):
                    gr.Markdown("""
                    **Resolution Options:**
                    - **Standard (640x640)**: Fast processing, suitable for most cases
                    - **High Resolution (896x896)**: Enhanced accuracy for detailed analysis
                    
                    **Object Labeling:**
                    - Shows ADE20K class names on detected objects
                    - 150 indoor/outdoor object classes supported
                    - Helps identify specific items in the environment
                    
                    **Blackspot Detection:**
                    - Analyzes ONLY floor surfaces (carpet, wood, tile, rugs, mats)
                    - Detects genuinely black/dark areas (RGB < 110)
                    - Uses multi-method validation for accuracy
                    - Filters out shadows and lighting artifacts
                    
                    **Contrast Analysis:**
                    - Checks ONLY adjacent object pairs (touching objects)
                    - Uses RGB, HSV, and LAB color spaces
                    - Alzheimer's-specific 7:1 threshold
                    - Prioritizes safety-critical relationships
                    
                    **Visualization Modes:**
                    - **High Contrast**: Enhanced overlays with bright highlighting
                    - **Side by Side**: Original vs analysis comparison
                    - **Blackspots Only**: Pure floor blackspot visualization
                    - **Segmentation Only**: Clean object segmentation
                    - **Annotated**: Detailed risk labels and assessments
                    - **Combined Analysis**: Both blackspots and contrast issues
                    """)

            # Output Column
            with gr.Column(scale=2):
                # Main display
                main_display = gr.Image(
                    label="🎯 Primary Analysis View",
                    height=500,
                    interactive=False
                )

                # Analysis tabs
                with gr.Tabs():
                    with gr.Tab("📊 Comprehensive Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' to see detailed results.",
                            elem_classes=["analysis-section"]
                        )

                    with gr.Tab("🏷️ Labeled Segmentation"):
                        labeled_display = gr.Image(
                            label="Object Detection with Labels",
                            height=400,
                            interactive=False
                        )

                    with gr.Tab("⚫ Blackspot Analysis"):
                        blackspot_display = gr.Image(
                            label="Floor Blackspot Detection",
                            height=400,
                            interactive=False
                        )

                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_display = gr.Image(
                            label="Adjacent Object Contrast Issues",
                            height=400,
                            interactive=False
                        )

                    with gr.Tab("🔄 Combined Analysis"):
                        combined_display = gr.Image(
                            label="Comprehensive View: All Issues",
                            height=400,
                            interactive=False
                        )

        # Status display
        with gr.Row():
            gr.Markdown("""
            <div class="info-box">
            <strong>💡 Analysis Status:</strong> Ready to analyze • 
            <strong>🔧 Resolution:</strong> <span id="res-status">Standard (640x640)</span> • 
            <strong>🏷️ Labels:</strong> <span id="label-status">Enabled</span>
            </div>
            """)

        # Connect the interface
        analyze_button.click(
            fn=analyze_wrapper,
            inputs=[
                image_input,
                blackspot_threshold,
                contrast_threshold,
                enable_blackspot,
                enable_contrast,
                visualization_mode,
                use_high_res,
                show_labels
            ],
            outputs=[
                main_display,
                labeled_display,
                blackspot_display,
                contrast_display,
                combined_display,
                analysis_report
            ]
        )

        # Example images (if available)
        example_dir = Path("examples")
        if example_dir.exists():
            examples = []
            for img_path in example_dir.glob("*.jpg"):
                if img_path.is_file():
                    examples.append([str(img_path), 0.5, 7.0, True, True, "High Contrast", False, True])
            
            for img_path in example_dir.glob("*.png"):
                if img_path.is_file():
                    examples.append([str(img_path), 0.5, 7.0, True, True, "High Contrast", False, True])

            if examples:
                gr.Examples(
                    examples=examples[:3],
                    inputs=[
                        image_input,
                        blackspot_threshold,
                        contrast_threshold,
                        enable_blackspot,
                        enable_contrast,
                        visualization_mode,
                        use_high_res,
                        show_labels
                    ],
                    outputs=[
                        main_display,
                        labeled_display,
                        blackspot_display,
                        contrast_display,
                        combined_display,
                        analysis_report
                    ],
                    fn=analyze_wrapper,
                    label="🖼️ Example Images"
                )

        # Footer with attribution
        gr.Markdown("""
        ---
        **NeuroNest Ultra** - Advanced AI for Alzheimer's-friendly environments
        
        *🔬 Features: Labeled object detection • Floor-only blackspots • Adjacent-only contrast • High resolution support*
        
        *🏥 Designed for dementia care facilities, assisted living, and home environments*
        
        <div class="success-box">
        <strong>✨ Optimal Configuration:</strong> High Resolution + Labels + Both Analyses = Maximum Safety Assessment
        </div>
        
        <div style="text-align: center; margin-top: 1rem; padding: 1rem; background-color: #f8f9fa; border-radius: 5px;">
        <strong>Research Team:</strong> Abheek Pradhan | <strong>Faculty Advisors:</strong> Dr. Nadim Adi, Dr. Greg Lakomski<br/>
        <em>Department of Computer Science & Department of Interior Design</em><br/>
        <strong>Texas State University</strong>
        </div>
        """)

    return interface
