"""Enhanced Gradio interface with resolution options and label visualization."""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional

from core import NeuroNestApp
from utils.helpers import generate_analysis_report

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
    """

    with gr.Blocks(
        title="NeuroNest - Alzheimer's Environment Analysis",
        theme=gr.themes.Soft(primary_hue="orange", secondary_hue="blue"),
        css="""
        /* Compact header */
        .title-section { text-align: center; margin-bottom: 0.5rem; padding: 0.5rem; }
        .subtitle { font-size: 0.9em; color: #555; margin-top: 0.2rem; }
        .funding { font-size: 0.8em; color: #666; font-style: italic; margin-top: 0.1rem; }
        
        /* Maximize image display area */
        .image-container { 
            width: 100%;
            height: calc(100vh - 250px);
            display: flex;
            justify-content: center;
            align-items: center;
            overflow: hidden;
        }
        
        .gradio-image {
            width: 100% !important;
            height: 100% !important;
            object-fit: contain !important;
            max-width: 100% !important;
            max-height: 100% !important;
        }
        
        .image-frame {
            width: 100% !important;
            height: calc(100vh - 300px) !important;
            display: flex !important;
            justify-content: center !important;
            align-items: center !important;
        }
        
        /* Make image containers fill space */
        .gr-panel {
            height: 100%;
        }
        
        div[data-testid="image"] {
            height: 100% !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
        }
        
        div[data-testid="image"] img {
            max-width: 100% !important;
            max-height: 100% !important;
            width: auto !important;
            height: auto !important;
            object-fit: contain !important;
        }
        
        /* Compact controls */
        .analysis-section { 
            border: 1px solid #f0f0f0; 
            border-radius: 5px; 
            padding: 0.5rem; 
            margin: 0.5rem 0;
            font-size: 0.9em;
        }
        
        .gr-form { gap: 0.5rem !important; }
        .gr-box { padding: 0.5rem !important; }
        
        /* Smaller text elements */
        .gr-button { padding: 0.5rem 1rem !important; font-size: 0.9em !important; }
        .gr-input-label { font-size: 0.85em !important; }
        .gr-checkbox-label { font-size: 0.85em !important; }
        
        /* Compact accordions */
        .gr-accordion { margin: 0.25rem 0 !important; }
        
        /* Tab content full height */
        .gr-tab-content {
            height: calc(100vh - 400px) !important;
            overflow: hidden !important;
        }
        
        /* Hide unnecessary padding */
        .gr-padded { padding: 0.5rem !important; }
        
        /* Info boxes compact */
        .info-box, .warning-box, .success-box {
            padding: 5px !important;
            margin: 5px 0 !important;
            font-size: 0.85em !important;
        }
        """
    ) as interface:

        # Compact title section
        with gr.Column(elem_classes=["title-section"]):
            gr.Markdown(f"""
            <h2 style="margin: 0; padding: 0;">{title}</h2>
            <div class="subtitle">{subtitle}</div>
            <div class="funding">{funding}</div>
            """)

        with gr.Row():
            # Input Column - Compact
            with gr.Column(scale=1, min_width=300):
                # Image upload - smaller
                image_input = gr.Image(
                    label="📸 Upload Room Image",
                    type="filepath",
                    height=250,  # Smaller input
                    image_mode="RGB",
                    sources=["upload", "clipboard"],
                    show_label=True
                )

                # Compact analysis settings
                with gr.Accordion("🔧 Analysis Settings", open=True):
                    # Resolution settings
                    with gr.Group():
                        use_high_res = gr.Checkbox(
                            value=False,
                            label="High Resolution (1280x1280)",
                            info="Better accuracy, slower"
                        )
                        
                        show_labels = gr.Checkbox(
                            value=True,
                            label="Show Object Labels"
                        )

                    # Blackspot settings
                    with gr.Group():
                        enable_blackspot = gr.Checkbox(
                            value=True,
                            label="Enable Blackspot Detection"
                        )

                        blackspot_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=0.9,
                            value=0.5,
                            step=0.05,
                            label="Blackspot Sensitivity"
                        )

                    # Contrast settings
                    with gr.Group():
                        enable_contrast = gr.Checkbox(
                            value=True,
                            label="Enable Contrast Analysis"
                        )

                        contrast_threshold = gr.Slider(
                            minimum=1.0,
                            maximum=10.0,
                            value=7.0,
                            step=0.1,
                            label="Contrast Threshold"
                        )

                    # Visualization options
                    visualization_mode = gr.Radio(
                        choices=[
                            "High Contrast", 
                            "Side by Side", 
                            "Combined Analysis"
                        ],
                        value="High Contrast",
                        label="Display Mode"
                    )

                # Analysis button
                analyze_button = gr.Button(
                    "🔍 Analyze",
                    variant="primary",
                    size="sm"
                )

            # Output Column - Maximized
            with gr.Column(scale=4):
                # Main display - Full size
                with gr.Column(elem_classes=["image-container"]):
                    main_display = gr.Image(
                        label="🎯 Analysis Result",
                        interactive=False,
                        show_label=True,
                        container=True,
                        elem_classes=["image-frame"]
                    )

                # Compact tabs
                with gr.Tabs():
                    with gr.Tab("📊 Report"):
                        analysis_report = gr.Markdown(
                            value="Upload an image and click 'Analyze' to see results.",
                            elem_classes=["analysis-section"]
                        )

                    with gr.Tab("🏷️ Labeled"):
                        labeled_display = gr.Image(
                            label="Labeled Segmentation",
                            interactive=False,
                            show_label=False,
                            container=True,
                            elem_classes=["image-frame"]
                        )

                    with gr.Tab("⚫ Blackspots"):
                        blackspot_display = gr.Image(
                            label="Blackspot Detection",
                            interactive=False,
                            show_label=False,
                            container=True,
                            elem_classes=["image-frame"]
                        )

                    with gr.Tab("🎨 Contrast"):
                        contrast_display = gr.Image(
                            label="Contrast Analysis",
                            interactive=False,
                            show_label=False,
                            container=True,
                            elem_classes=["image-frame"]
                        )

                    with gr.Tab("🔄 Combined"):
                        combined_display = gr.Image(
                            label="Combined Analysis",
                            interactive=False,
                            show_label=False,
                            container=True,
                            elem_classes=["image-frame"]
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
                with gr.Accordion("🖼️ Examples", open=False):
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
                        examples_per_page=3
                    )

    return interface
