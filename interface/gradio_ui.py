"""Enhanced Gradio interface with multiple blackspot views."""

import gradio as gr
import numpy as np
from pathlib import Path
from typing import Dict

from core import NeuroNestApp
from utils import generate_analysis_report


def create_gradio_interface():
    """Create the enhanced Gradio interface with better blackspot visualization"""

    # Initialize the application
    app = NeuroNestApp()
    oneformer_ok, blackspot_ok = app.initialize()

    if not oneformer_ok:
        raise RuntimeError("Failed to initialize OneFormer")

    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold,
                       enable_blackspot, enable_contrast, blackspot_view_type):
        """Enhanced wrapper function for Gradio interface"""
        if image_path is None:
            return None, None, None, None, None, "Please upload an image"

        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast
        )

        if "error" in results:
            return None, None, None, None, None, f"Error: {results['error']}"

        # Extract outputs
        seg_output = results['segmentation']['visualization'] if results['segmentation'] else None

        # Enhanced blackspot output selection
        blackspot_output = None
        blackspot_segmentation = None
        if results['blackspot'] and 'enhanced_views' in results['blackspot']:
            views = results['blackspot']['enhanced_views']

            # Select view based on user choice
            if blackspot_view_type == "High Contrast":
                blackspot_output = views['high_contrast_overlay']
            elif blackspot_view_type == "Segmentation Only":
                blackspot_output = views['segmentation_view']
            elif blackspot_view_type == "Blackspots Only":
                blackspot_output = views['blackspot_only']
            elif blackspot_view_type == "Side by Side":
                blackspot_output = views['side_by_side']
            elif blackspot_view_type == "Annotated":
                blackspot_output = views['annotated_view']
            else:
                blackspot_output = views['high_contrast_overlay']

            # Always provide segmentation view for the dedicated tab
            blackspot_segmentation = views['segmentation_view']

        contrast_output = results['contrast']['visualization'] if results['contrast'] else None

        # Generate report
        report = generate_analysis_report(results)

        return seg_output, blackspot_output, blackspot_segmentation, contrast_output, report

    # Create the interface with enhanced controls
    title = "🧠 NeuroNest: Advanced Environment Analysis for Alzheimer's Care"
    description = """
    **Comprehensive analysis system for creating Alzheimer's-friendly environments**

    This application integrates:
    - **Semantic Segmentation**: Identifies rooms, furniture, and objects
    - **Enhanced Blackspot Detection**: Locates and visualizes dangerous black areas on floors
    - **Contrast Analysis**: Evaluates color contrast for visual accessibility
    """

    with gr.Blocks(
        title=title,
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        .main-header { text-align: center; margin-bottom: 2rem; }
        .analysis-section { border: 2px solid #f0f0f0; border-radius: 10px; padding: 1rem; margin: 1rem 0; }
        .critical-text { color: #ff0000; font-weight: bold; }
        .high-text { color: #ff8800; font-weight: bold; }
        .medium-text { color: #ffaa00; font-weight: bold; }
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
                    enable_blackspot = gr.Checkbox(
                        value=blackspot_ok,
                        label="Enable Blackspot Detection",
                        interactive=blackspot_ok
                    )

                    blackspot_threshold = gr.Slider(
                        minimum=0.1,
                        maximum=0.9,
                        value=0.5,
                        step=0.05,
                        label="Blackspot Detection Threshold",
                        visible=blackspot_ok
                    )

                    # NEW: Blackspot visualization options
                    blackspot_view_type = gr.Radio(
                        choices=["High Contrast", "Segmentation Only", "Blackspots Only", "Side by Side", "Annotated"],
                        value="High Contrast",
                        label="Blackspot Visualization Style",
                        visible=blackspot_ok
                    )

                    enable_contrast = gr.Checkbox(
                        value=True,
                        label="Enable Contrast Analysis"
                    )

                    contrast_threshold = gr.Slider(
                        minimum=1.0,
                        maximum=10.0,
                        value=4.5,
                        step=0.1,
                        label="WCAG Contrast Threshold"
                    )

                # Analysis button
                analyze_button = gr.Button(
                    "🔍 Analyze Environment",
                    variant="primary",
                    size="lg"
                )

            # Output Column
            with gr.Column(scale=2):
                # Main display (Segmentation by default)
                main_display = gr.Image(
                    label="🎯 Object Detection & Segmentation",
                    height=400,
                    interactive=False
                )

                # Enhanced analysis tabs
                with gr.Tabs():
                    with gr.Tab("📊 Analysis Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze Environment' to see results.",
                            elem_classes=["analysis-section"]
                        )

                    if blackspot_ok:
                        with gr.Tab("⚫ Blackspot Detection"):
                            blackspot_display = gr.Image(
                                label="Blackspot Analysis (Selected View)",
                                height=300,
                                interactive=False
                            )

                        with gr.Tab("🔍 Blackspot Segmentation"):
                            blackspot_segmentation_display = gr.Image(
                                label="Pure Blackspot Segmentation",
                                height=300,
                                interactive=False
                            )
                    else:
                        blackspot_display = gr.Image(visible=False)
                        blackspot_segmentation_display = gr.Image(visible=False)

                    with gr.Tab("🎨 Contrast Analysis"):
                        contrast_display = gr.Image(
                            label="Contrast Issues Visualization",
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
                blackspot_view_type
            ],
            outputs=[
                main_display,
                blackspot_display,
                blackspot_segmentation_display,
                contrast_display,
                analysis_report
            ]
        )

        # Example images (optional)
        example_dir = Path("examples")
        if example_dir.exists():
            examples = [
                [str(img), 0.5, 4.5, True, True, "High Contrast"]
                for img in example_dir.glob("*.jpg")
            ]

            if examples:
                gr.Examples(
                    examples=examples[:3],  # Show max 3 examples
                    inputs=[
                        image_input,
                        blackspot_threshold,
                        contrast_threshold,
                        enable_blackspot,
                        enable_contrast,
                        blackspot_view_type
                    ],
                    outputs=[
                        main_display,
                        blackspot_display,
                        blackspot_segmentation_display,
                        contrast_display,
                        analysis_report
                    ],
                    fn=analyze_wrapper,
                    label="🖼️ Example Images"
                )

        # Footer
        gr.Markdown("""
            ---
            **NeuroNest** - Advanced AI for Alzheimer's-friendly environments
            *Helping create safer, more accessible spaces for cognitive health*
            """)

    return interface
