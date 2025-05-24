"""Enhanced Gradio interface with comprehensive blackspot and contrast visualization options."""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Dict

from core import NeuroNestApp
from utils import generate_analysis_report


def create_gradio_interface():
    """Create the enhanced Gradio interface with comprehensive visualization options"""

    # Initialize the application
    app = NeuroNestApp()
    oneformer_ok, blackspot_ok = app.initialize()

    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")

    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold,
                       enable_blackspot, enable_contrast, visualization_mode):
        """Enhanced wrapper function for comprehensive analysis"""
        if image_path is None:
            return None, None, None, None, "Please upload an image"

        try:
            results = app.analyze_image(
                image_path=image_path,
                blackspot_threshold=blackspot_threshold,
                contrast_threshold=contrast_threshold,
                enable_blackspot=enable_blackspot,
                enable_contrast=enable_contrast
            )

            if "error" in results:
                return None, None, None, None, f"Error: {results['error']}"

            # Extract main segmentation output
            seg_output = results['segmentation']['visualization'] if results['segmentation'] else None

            # Enhanced visualization selection
            main_output = seg_output  # Default
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

            # Combined visualization (show both blackspots and contrast on same image)
            if results['blackspot'] and results['contrast']:
                contrast_vis = results['contrast'].get('visualization')
                if contrast_vis is not None:
                    combined_output = contrast_vis.copy()

                    # Overlay blackspot areas in bright yellow for distinction
                    blackspot_mask = results['blackspot'].get('blackspot_mask')
                    if blackspot_mask is not None and np.any(blackspot_mask):
                        # Resize mask if needed to match contrast visualization
                        if blackspot_mask.shape != combined_output.shape[:2]:
                            blackspot_mask_resized = cv2.resize(
                                blackspot_mask.astype(np.uint8),
                                (combined_output.shape[1], combined_output.shape[0]),
                                interpolation=cv2.INTER_NEAREST
                            ).astype(bool)
                        else:
                            blackspot_mask_resized = blackspot_mask

                        # Overlay blackspots in bright yellow
                        combined_output[blackspot_mask_resized] = [255, 255, 0]

            # Determine main output based on what's enabled and mode
            if visualization_mode == "Combined Analysis" and combined_output is not None:
                main_output = combined_output
            elif enable_blackspot and blackspot_output is not None and visualization_mode != "Combined Analysis":
                main_output = blackspot_output
            elif enable_contrast and contrast_output is not None and not enable_blackspot:
                main_output = contrast_output

            # Generate comprehensive report
            report = generate_analysis_report(results)

            return main_output, blackspot_output, contrast_output, combined_output, report

        except Exception as e:
            error_msg = f"Analysis failed: {str(e)}"
            return None, None, None, None, error_msg

    # Create the interface
    title = "🧠 NeuroNest: Ultra-Comprehensive Alzheimer's Environment Analysis"
    description = """
    **Advanced AI System for Creating Dementia-Friendly Environments**

    **🎯 Core Features:**
    - **Object Segmentation**: Identifies all room elements with high precision
    - **⚫ Blackspot Detection**: Detects dangerous black/dark areas on floors ONLY
    - **🎨 Ultra-Sensitive Contrast Analysis**: Finds ANY similar colors that could cause confusion
    - **📊 Evidence-Based Reporting**: Comprehensive safety recommendations

    **🔬 Detection Capabilities:**
    - Analyzes genuine black/dark areas (RGB < 110) on floor surfaces
    - Checks ALL object pairs for color similarity issues
    - Uses Alzheimer's-specific 7:1 contrast threshold
    - Provides multiple visualization modes for comprehensive analysis
    """

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .analysis-section { border: 2px solid #f0f0f0; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .critical-text { color: #ff0000; font-weight: bold; }
        .high-text { color: #ff8800; font-weight: bold; }
        .warning-box { background-color: #fff3cd; border: 1px solid #ffeaa7; border-radius: 5px; padding: 10px; margin: 10px 0; }
        .info-box { background-color: #d1ecf1; border: 1px solid #bee5eb; border-radius: 5px; padding: 10px; margin: 10px 0; }
        """
    ) as interface:

        gr.Markdown(f"# {title}")
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

                    # Blackspot settings
                    with gr.Group():
                        gr.Markdown("### ⚫ Blackspot Detection (Floor-Only)")
                        if blackspot_ok:
                            gr.Markdown('<div class="info-box">✅ <strong>Blackspot Detection Available</strong><br/>Detects genuinely black/dark areas on floors using multi-method color analysis.</div>')
                        else:
                            gr.Markdown('<div class="warning-box">⚠️ <strong>Blackspot Model Not Found</strong><br/>Color-based detection will be used as backup.</div>')
                        
                        enable_blackspot = gr.Checkbox(
                            value=True,  # Always enable since we have color-based fallback
                            label="Enable Enhanced Blackspot Detection",
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
                        gr.Markdown("### 🎨 Ultra-Sensitive Contrast Analysis")
                        gr.Markdown('<div class="info-box">🔬 <strong>Advanced Detection</strong><br/>Analyzes ALL object pairs for similar colors that could cause confusion in dementia care.</div>')
                        
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Enable Ultra-Sensitive Contrast Analysis"
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
                    **Blackspot Detection Technical Details:**
                    - Analyzes ONLY floor surfaces (carpet, wood, tile, rugs)
                    - Detects genuinely black/dark areas (RGB < 110, multiple color spaces)
                    - Uses 5-method voting system for high accuracy
                    - Validates all detections to ensure they are actually dark
                    - Filters out shadows and lighting artifacts
                    
                    **Contrast Analysis Technical Details:**
                    - Checks ALL possible object pairs in the room
                    - Detects similar colors using RGB, HSV, and perceptual analysis
                    - Uses Alzheimer's-specific 7:1 contrast threshold
                    - Prioritizes safety-critical relationships (floor-stairs, wall-door)
                    - Provides comprehensive similarity scoring
                    
                    **Visualization Modes:**
                    - **High Contrast**: Enhanced overlays with bright highlighting
                    - **Side by Side**: Original image vs analysis results
                    - **Blackspots Only**: Pure blackspot visualization on floors
                    - **Segmentation Only**: Clean object segmentation view
                    - **Annotated**: Detailed risk assessments and labels
                    - **Combined Analysis**: Both blackspots and contrast issues together
                    """)

            # Output Column
            with gr.Column(scale=2):
                # Main display
                main_display = gr.Image(
                    label="🎯 Primary Analysis View",
                    height=400,
                    interactive=False
                )

                # Analysis tabs
                with gr.Tabs():
                    with gr.Tab("📊 Comprehensive Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' to see detailed results.",
                            elem_classes=["analysis-section"]
                        )

                    with gr.Tab("⚫ Blackspot Analysis"):
                        blackspot_display = gr.Image(
                            label="Blackspot Detection (Floors Only - Genuine Black/Dark Areas)",
                            height=300,
                            interactive=False
                        )

                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_display = gr.Image(
                            label="Similar Color & Contrast Issues (All Object Pairs)",
                            height=300,
                            interactive=False
                        )

                    with gr.Tab("🔄 Combined Analysis"):
                        combined_display = gr.Image(
                            label="Comprehensive View: Blackspots + Contrast Issues",
                            height=300,
                            interactive=False
                        )

        # Connect the interface
        analyze_button.click(
            fn=analyze_wrapper,
            inputs=[
                image_input,
                blackspot_threshold,
                contrast_threshold,
                enable_blackspot,
                enable_contrast,
                visualization_mode
            ],
            outputs=[
                main_display,
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
                    examples.append([str(img_path), 0.5, 7.0, True, True, "High Contrast"])
            
            for img_path in example_dir.glob("*.png"):
                if img_path.is_file():
                    examples.append([str(img_path), 0.5, 7.0, True, True, "High Contrast"])

            if examples:
                gr.Examples(
                    examples=examples[:3],  # Show max 3 examples
                    inputs=[
                        image_input,
                        blackspot_threshold,
                        contrast_threshold,
                        enable_blackspot,
                        enable_contrast,
                        visualization_mode
                    ],
                    outputs=[
                        main_display,
                        blackspot_display,
                        contrast_display,
                        combined_display,
                        analysis_report
                    ],
                    fn=analyze_wrapper,
                    label="🖼️ Example Images"
                )

        # Footer with technical specifications
        gr.Markdown("""
            ---
            **NeuroNest Ultra** - Advanced AI for Alzheimer's-friendly environments
            
            *🔬 Features: Floor-only blackspot detection (RGB<110) • Ultra-sensitive contrast analysis • 5-method validation • Evidence-based recommendations*
            
            *🏥 Designed for dementia care facilities, assisted living, and home environments*
            """)

    return interface
