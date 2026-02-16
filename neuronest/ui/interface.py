import numpy as np
import cv2
import os
import time
import logging
from typing import Dict

import gradio as gr

from ade20k_classes import ADE20K_NAMES
from ..pipeline import NeuroNestApp
from ..xai import XAIAnalyzer
from ..xai.renderer import render_notebook_html, render_defect_notebook_html
from .css import MAIN_CSS
from ..agent import create_chat_handler

logger = logging.getLogger(__name__)


def create_gradio_interface():
    app = NeuroNestApp()
    seg_ok, blackspot_ok = app.initialize()
    if not seg_ok:
        raise RuntimeError("Failed to initialize EoMT segmentation")

    SAMPLE_IMAGES = [
        img
        for img in [
            "samples/example1.png", "samples/example2.png", "samples/example3.png",
            "samples/example1_original.png", "samples/example2_original.png", "samples/example3_original.png",
        ]
        if os.path.exists(img)
    ]
    sample_images_available = len(SAMPLE_IMAGES) > 0

    # --- Wrapper functions ---

    def analyze_wrapper(image_path, blackspot_threshold, contrast_threshold, enable_blackspot, enable_contrast):
        if image_path is None:
            return None, None, None, "Please upload an image", {}
        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast,
        )
        if "error" in results:
            return None, None, None, f"Error: {results['error']}", {}
        seg_output = results["segmentation"]["visualization"] if results["segmentation"] else None
        blackspot_output = results["blackspot"]["visualization"] if results["blackspot"] else None
        contrast_output = results["contrast"]["visualization"] if results["contrast"] else None

        contrast_report = "Contrast analysis not performed."
        if results["contrast"]:
            contrast_report = app.contrast_analyzer.generate_report(results["contrast"])

        blackspot_report = "Blackspot analysis not performed."
        if results["blackspot"]:
            bs = results["blackspot"]
            blackspot_report = (
                f"**Legend:** Green = detected floor area, Red = blackspot hazards  \n"
                f"**Floor Area:** {bs['floor_area']:,} pixels  \n"
                f"**Blackspot Area:** {bs['blackspot_area']:,} pixels  \n"
                f"**Coverage:** {bs['coverage_percentage']:.2f}%  \n"
                f"**Detections:** {bs['num_detections']}  \n"
                f"**Average Confidence:** {bs['avg_confidence']:.2f}"
            )

        report = _comprehensive_report(results, contrast_report, blackspot_report)
        return seg_output, blackspot_output, contrast_output, report, results

    def xai_wrapper(image_path, method, layer, head_choice, target_class, progress=gr.Progress(track_tqdm=True)):
        if image_path is None:
            return "<p>Upload an image and click <b>Run XAI Analysis</b> to begin.</p>"
        if app.xai_analyzer is None:
            return "<p style='color:red;'>XAI analyzer not initialized.</p>"
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return "<p style='color:red;'>Could not load image.</p>"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            layer_idx = int(layer)
            head_idx = None if head_choice == "Mean (all heads)" else int(head_choice.split()[-1])
            class_id = None if target_class == "Auto (dominant)" else int(target_class.split(":")[0])

            if method == "Full Suite":
                cached = app.xai_analyzer.load_cached_from_file(image_path)
                if cached:
                    progress(1.0, desc="Loaded from cache")
                    return render_notebook_html(cached, image_rgb)

                progress(0, desc="Starting XAI Full Suite analysis...")
                results = app.xai_analyzer.run_full_analysis(
                    image_rgb, layer=layer_idx, head=head_idx,
                    target_class_id=class_id, progress_callback=progress,
                )
                try:
                    app.xai_analyzer.save_results(results, image_path=image_path)
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")

                app.xai_analyzer.cleanup_fp32()
                return render_notebook_html(results, image_rgb)
            else:
                method_map = {
                    "Self-Attention": ("self_attention_maps", {"layer": layer_idx, "head": head_idx}),
                    "Attention Rollout": ("attention_rollout", {}),
                    "GradCAM": ("gradcam_segmentation", {"target_class_id": class_id}),
                    "Predictive Entropy": ("predictive_entropy", {}),
                    "Feature PCA": ("feature_pca", {"layer": layer_idx}),
                    "Integrated Gradients": ("integrated_gradients", {"target_class_id": class_id}),
                    "Chefer Relevancy": ("chefer_relevancy", {"target_class_id": class_id}),
                }
                if method in method_map:
                    func_name, kwargs = method_map[method]
                    progress(0.2, desc=f"Running {method}...")
                    func = getattr(app.xai_analyzer, func_name)
                    result = func(image_rgb, **kwargs)
                    progress(1.0, desc=f"{method} complete")

                    if method in ("GradCAM", "Integrated Gradients", "Chefer Relevancy"):
                        app.xai_analyzer.cleanup_fp32()

                    key_map = {
                        "Self-Attention": "attention", "Attention Rollout": "rollout",
                        "GradCAM": "gradcam", "Predictive Entropy": "entropy",
                        "Feature PCA": "pca", "Integrated Gradients": "integrated_gradients",
                        "Chefer Relevancy": "chefer",
                    }
                    return render_notebook_html({key_map[method]: result}, image_rgb)

            return "<p>Unknown method selected.</p>"
        except Exception as e:
            logger.error(f"XAI analysis error: {e}")
            import traceback
            traceback.print_exc()
            return f"<p style='color:red;'><b>Error</b>: {str(e)}</p>"

    def defect_xai_wrapper(image_path, progress=gr.Progress(track_tqdm=True)):
        if image_path is None:
            return "<p>Upload an image and click <b>Run Defect XAI</b> to begin.</p>"
        if app.xai_analyzer is None:
            return "<p style='color:red;'>XAI analyzer not initialized.</p>"
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return "<p style='color:red;'>Could not load image.</p>"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cached = XAIAnalyzer.load_cached_defect_from_file(image_path)
            if cached:
                progress(1.0, desc="Loaded defect analysis from cache")
                return render_defect_notebook_html(cached, image_rgb)

            progress(0.05, desc="Running segmentation...")
            seg_mask, _ = app.segmenter.semantic_segmentation(image_rgb)
            floor_prior = app.segmenter.extract_floor_areas(seg_mask)

            blackspot_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            blackspot_count = 0
            blackspot_coverage = 0.0
            contrast_issues = []

            if app.blackspot_detector is not None:
                progress(0.15, desc="Detecting blackspots...")
                try:
                    bs = app.blackspot_detector.detect_blackspots(image_rgb, seg_mask, floor_prior)
                    blackspot_mask = bs.get("blackspot_mask", blackspot_mask)
                    blackspot_count = bs.get("num_detections", 0)
                    blackspot_coverage = bs.get("coverage_percentage", 0.0)
                except Exception as e:
                    logger.warning(f"Blackspot detection failed: {e}")

            progress(0.25, desc="Analyzing contrast...")
            try:
                contrast_result = app.contrast_analyzer.analyze_contrast(image_rgb, seg_mask)
                contrast_issues = contrast_result.get("issues", [])
            except Exception as e:
                logger.warning(f"Contrast analysis failed: {e}")

            progress(0.3, desc="Starting defect XAI analysis...")

            def defect_progress(frac, desc=""):
                progress(0.3 + frac * 0.65, desc=desc)

            results = app.xai_analyzer.run_defect_analysis(
                image_rgb, blackspot_mask=blackspot_mask,
                contrast_issues=contrast_issues, seg_mask=seg_mask,
                progress_callback=defect_progress,
            )

            try:
                app.xai_analyzer.save_defect_results(results, image_path=image_path)
            except Exception as e:
                logger.warning(f"Defect cache save failed: {e}")

            target_class = app.xai_analyzer._dominant_class(seg_mask)
            floor_name = app.xai_analyzer._cname(target_class)

            progress(1.0, desc="Complete")
            return render_defect_notebook_html(
                results, image_rgb,
                blackspot_count=blackspot_count,
                blackspot_coverage=blackspot_coverage,
                floor_class_name=floor_name,
            )
        except Exception as e:
            logger.error(f"Defect XAI error: {e}")
            import traceback
            traceback.print_exc()
            return f"<p style='color:red;'><b>Error</b>: {str(e)}</p>"

    def load_sample_gallery():
        html_parts = []
        found = 0
        for img_path in SAMPLE_IMAGES:
            if not os.path.exists(img_path):
                continue
            cached = XAIAnalyzer.load_cached_from_file(img_path)
            if cached is None:
                continue
            found += 1
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            html_parts.append(f'<h2 style="margin:32px 0 8px;color:#4f46e5;">Sample: {os.path.basename(img_path)}</h2>')
            html_parts.append(render_notebook_html(cached, img_rgb))
            html_parts.append('<hr style="margin:40px 0;border:2px solid #e5e7eb;">')

        if not html_parts:
            return (
                '<div style="text-align:center;padding:40px;color:#6b7280;">'
                "<p style='font-size:1.2em;'>No pre-computed analyses found.</p>"
                "<p>Run <b>Full Suite</b> on a sample image first — results appear here automatically.</p></div>"
            )
        return (
            f'<div style="max-width:1200px;margin:auto;">'
            f'<p style="color:#059669;font-weight:600;">{found} cached analyses available</p>'
            + "\n".join(html_parts)
            + "</div>"
        )

    # --- Build Gradio interface ---

    with gr.Blocks(
        css=MAIN_CSS,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate"),
    ) as interface:
        analysis_state = gr.State(value={})

        with gr.Column(elem_classes="container"):
            gr.HTML(_HERO_HTML)

            with gr.Tabs(elem_classes="nav-tabs"):
                _build_overview_tab()
                _build_team_tab()
                _build_demo_tab(
                    app, analyze_wrapper, SAMPLE_IMAGES,
                    sample_images_available, blackspot_ok,
                    analysis_state,
                )
                _build_xai_tab(
                    xai_wrapper, defect_xai_wrapper, load_sample_gallery,
                    SAMPLE_IMAGES, sample_images_available,
                )
                _build_chat_tab(analysis_state)
                _build_technical_tab()

    return interface


# --- Tab builders ---

_HERO_HTML = """
<div class="hero">
    <h1>NeuroNest</h1>
    <p class="hero-sub">
        Production ML system for Alzheimer's care environment analysis.
        Multi-model ensemble combining semantic segmentation, blackspot detection,
        WCAG contrast analysis, and 7 Explainable AI visualization methods.
    </p>
    <div class="metrics-row">
        <div class="metric"><span class="metric-val">98%</span><span class="metric-label">Detection Precision</span></div>
        <div class="metric"><span class="metric-val">150</span><span class="metric-label">ADE20K Classes</span></div>
        <div class="metric"><span class="metric-val">7</span><span class="metric-label">XAI Methods</span></div>
        <div class="metric"><span class="metric-val">2-3x</span><span class="metric-label">INT8 Speedup</span></div>
        <div class="metric"><span class="metric-val">WCAG</span><span class="metric-label">2.1 Level AA</span></div>
    </div>
    <div class="badge-row">
        <span class="badge">EoMT-DINOv3 ViT-Large</span>
        <span class="badge">Mask R-CNN R50-FPN</span>
        <span class="badge">PyTorch 2.5</span>
        <span class="badge">HuggingFace Transformers</span>
        <span class="badge">Detectron2</span>
        <span class="badge">Gradio 5.x</span>
    </div>
</div>"""


def _build_overview_tab():
    with gr.TabItem("Project Overview"):
        gr.Markdown("""## Problem & Motivation

Alzheimer's patients experience **visual-perceptual deficits** that make dark floor areas appear as voids
and low-contrast objects invisible. This leads to falls, reduced mobility, and decreased independence.

## Solution Architecture

| Stage | Model | Output |
|-------|-------|--------|
| 1. Semantic Segmentation | EoMT-DINOv3-Large (ViT backbone, 512x512, 24 layers) | 150-class pixel-level scene parsing |
| 2. Floor Isolation | Region filtering (80% overlap threshold) | Floor-only surface mask |
| 3. Blackspot Detection | Mask R-CNN R50-FPN (custom trained on 15,000+ samples) | Instance-level dark area detection |
| 4. Contrast Analysis | Vectorized boundary detection + WCAG 2.1 | Adjacent object-pair contrast ratios |
| 5. Explainable AI | 7 methods: attention, gradient, output-based | Model interpretability visualizations |

## Key Technical Achievements

- **98% precision** on blackspot detection via fine-tuned Mask R-CNN with active learning
- **Dynamic INT8 quantization**: ~2-3x CPU inference speedup
- **Vectorized analysis**: 50-200x faster contrast computation using numpy C-level operations
- **Concurrent pipeline**: Blackspot + contrast analysis execute in parallel (ThreadPoolExecutor)
- **7 XAI methods**: Self-attention, rollout, GradCAM, entropy, PCA, Integrated Gradients, Chefer relevancy
- **CPU-optimized**: Runs on HuggingFace Spaces free tier (2 vCPU, 16 GB RAM)
""")


def _build_team_tab():
    with gr.TabItem("Research & Team"):
        gr.Markdown("""## Research Context

**Texas State University** — C.A.D.S (Center for Analytics and Data Science) Research Initiative

## Team

| Role | Name | Contribution |
|------|------|-------------|
| **ML Engineer** | Abheek Pradhan | Model architecture, training pipeline, INT8 optimization, XAI system, deployment |
| **Frontend Developer** | Samuel Chutter | Mobile app (React Native), UI/UX design |
| **Frontend Developer** | Priyanka Karki | Web interface, data visualization |
| **Faculty Advisor** | Dr. Nadim Adi | Interior design expertise, clinical requirements |
| **Faculty Advisor** | Dr. Greg Lakomski | Computer science guidance, research methodology |

## Technology Stack

| Category | Technologies |
|----------|-------------|
| **ML Frameworks** | PyTorch 2.5, Detectron2, HuggingFace Transformers |
| **Models** | EoMT-DINOv3-Large, Mask R-CNN R50-FPN |
| **XAI Libraries** | grad-cam, captum (custom hook-based implementation) |
| **Image Processing** | OpenCV, Pillow, scikit-image, scipy |
| **Deployment** | Docker, HuggingFace Spaces, Gradio 5.x |
| **Standards** | WCAG 2.1 Level AA, ADE20K (150 classes) |

## Links

- [GitHub Repository](https://github.com/lolout1/NeuroNest)
- [HuggingFace Space](https://huggingface.co/spaces/lolout1/txstNeuroNest)
""")


def _build_demo_tab(app, analyze_wrapper, sample_images, sample_available, blackspot_ok, analysis_state):
    with gr.TabItem("Live Demo"):
        gr.Markdown("Upload a room image to detect visual hazards for Alzheimer's patients.")

        with gr.Row(visible=False):
            image_input = gr.Image(label="Upload", type="filepath")

        if sample_available:
            with gr.Column(elem_classes="sample-section"):
                gr.Markdown("**Try a sample image** or upload your own below. Processing takes ~1-2 min on CPU.")
                gr.Examples(examples=sample_images, inputs=image_input, label="", examples_per_page=3)

        with gr.Row():
            with gr.Column(scale=3):
                image_input_display = gr.Image(label="Upload Room Image", type="filepath", height=400)
                image_input.change(fn=lambda x: x, inputs=image_input, outputs=image_input_display)
            with gr.Column(scale=1):
                analyze_button = gr.Button("Analyze Environment", variant="primary", elem_classes="main-button", size="lg")
                with gr.Accordion("Settings", open=False):
                    enable_blackspot = gr.Checkbox(value=blackspot_ok, label="Blackspot Detection", interactive=blackspot_ok)
                    blackspot_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Blackspot Sensitivity", visible=blackspot_ok)
                    enable_contrast = gr.Checkbox(value=True, label="Contrast Analysis")
                    contrast_threshold = gr.Slider(minimum=3.0, maximum=7.0, value=4.5, step=0.1, label="WCAG Contrast Threshold", info="4.5:1 = WCAG AA, 7:1 = AAA")

        gr.Markdown("### Results")
        with gr.Tabs():
            with gr.TabItem("Segmentation"):
                seg_display = gr.Image(label="150-class ADE20K semantic segmentation", interactive=False, elem_classes="image-output")
            with gr.TabItem("Blackspot Detection"):
                if blackspot_ok:
                    blackspot_display = gr.Image(label="Blackspot Detection (Green = floor, Red = blackspot hazards)", interactive=False, elem_classes="image-output")
                else:
                    blackspot_display = gr.Image(visible=False)
                    gr.Markdown("Blackspot detection model not available in this deployment.")
            with gr.TabItem("Contrast Analysis"):
                contrast_display = gr.Image(label="WCAG 2.1 contrast ratio analysis", interactive=False, elem_classes="image-output")
            with gr.TabItem("Full Report"):
                analysis_report = gr.Markdown(value="Upload an image and click **Analyze Environment** to begin.", elem_classes="report-box")

        analyze_button.click(
            fn=analyze_wrapper,
            inputs=[image_input_display, blackspot_threshold, contrast_threshold, enable_blackspot, enable_contrast],
            outputs=[seg_display, blackspot_display, contrast_display, analysis_report, analysis_state],
        )


def _build_xai_tab(xai_wrapper, defect_xai_wrapper, load_sample_gallery, sample_images, sample_available):
    with gr.TabItem("Model Interpretability"):
        gr.Markdown("""**7 XAI visualization methods** for interpreting DINOv3-EoMT Vision Transformer decisions.
Includes defect-focused analysis with hazard overlays and full model understanding suite.
Analyses are **cached automatically** — re-running the same image loads instantly.""")

        with gr.Tabs():
            # Defect Analysis sub-tab
            with gr.TabItem("Defect Analysis"):
                gr.Markdown("""Runs the full detection pipeline (segmentation, blackspot, contrast) then applies
4 targeted XAI methods to explain model behavior at defect locations.
**Red contours** = blackspot hazards, **Amber contours** = contrast failures.""")

                with gr.Row(visible=False):
                    defect_hidden = gr.Image(label="Defect Image", type="filepath")

                if sample_available:
                    with gr.Column(elem_classes="sample-section"):
                        gr.Markdown("**Select a sample** or upload your own image below.")
                        gr.Examples(examples=sample_images, inputs=defect_hidden, label="", examples_per_page=6)

                with gr.Row(elem_classes="xai-controls"):
                    with gr.Column(scale=2):
                        defect_image = gr.Image(label="Upload Image", type="filepath", height=300)
                        defect_hidden.change(fn=lambda x: x, inputs=defect_hidden, outputs=defect_image)
                    with gr.Column(scale=1):
                        gr.Markdown("""**Methods applied:**
- GradCAM — floor class activation
- Predictive Entropy — boundary uncertainty
- Integrated Gradients — pixel attribution
- Chefer Relevancy — transformer attention""")
                        defect_btn = gr.Button("Run Defect XAI", variant="primary", elem_classes="main-button", size="lg")
                        gr.Markdown("<small>Full pipeline + 4 XAI methods. ~4-6 min on CPU. Cached results load instantly.</small>")

                defect_nb = gr.HTML(
                    value='<div style="text-align:center;padding:60px 20px;color:#6b7280;">'
                    '<p style="font-size:1.3em;font-weight:600;color:#dc2626;">Defect Interpretability</p>'
                    '<p style="max-width:600px;margin:12px auto;line-height:1.6;">'
                    "Upload an image and click <b>Run Defect XAI</b> to analyze model behavior at defect locations.</p>"
                    '<p style="font-size:0.85em;margin-top:16px;">Red = blackspot contours &bull; Amber = contrast failures</p></div>',
                    elem_classes="xai-report",
                )
                defect_btn.click(fn=defect_xai_wrapper, inputs=[defect_image], outputs=[defect_nb])

            # Model Understanding sub-tab
            with gr.TabItem("Model Understanding"):
                with gr.Row(visible=False):
                    xai_hidden = gr.Image(label="XAI Image", type="filepath")

                if sample_available:
                    with gr.Column(elem_classes="sample-section"):
                        gr.Markdown("**Select a sample** or upload your own image below.")
                        gr.Examples(examples=sample_images, inputs=xai_hidden, label="", examples_per_page=6)

                with gr.Row(elem_classes="xai-controls"):
                    with gr.Column(scale=2):
                        xai_image = gr.Image(label="Upload Image", type="filepath", height=300)
                        xai_hidden.change(fn=lambda x: x, inputs=xai_hidden, outputs=xai_image)
                    with gr.Column(scale=1):
                        xai_method = gr.Radio(
                            choices=["Full Suite", "Self-Attention", "Attention Rollout", "GradCAM",
                                     "Predictive Entropy", "Feature PCA", "Integrated Gradients", "Chefer Relevancy"],
                            value="Full Suite", label="XAI Method",
                        )
                        with gr.Accordion("Advanced Controls", open=False):
                            xai_layer = gr.Slider(minimum=0, maximum=23, value=19, step=1, label="Transformer Layer",
                                                  info="0=early features, 19=last encoder, 23=last decoder")
                            xai_head = gr.Dropdown(
                                choices=["Mean (all heads)"] + [f"Head {i}" for i in range(16)],
                                value="Mean (all heads)", label="Attention Head", info="Affects Self-Attention only",
                            )
                            xai_class = gr.Dropdown(
                                choices=["Auto (dominant)"] + [f"{i}: {ADE20K_NAMES[i].split(',')[0]}" for i in range(len(ADE20K_NAMES))],
                                value="Auto (dominant)", label="Target Class", info="Affects GradCAM, IG, Chefer",
                            )
                        xai_btn = gr.Button("Run XAI Analysis", variant="primary", elem_classes="main-button", size="lg")
                        gr.Markdown("<small>Full Suite: ~3-5 min on CPU. Cached results load instantly.</small>")

                xai_nb = gr.HTML(
                    value='<div style="text-align:center;padding:60px 20px;color:#6b7280;">'
                    '<p style="font-size:1.3em;font-weight:600;color:#4f46e5;">Model Understanding</p>'
                    '<p style="max-width:600px;margin:12px auto;line-height:1.6;">'
                    "Upload an image and click <b>Run XAI Analysis</b> for a notebook-style report with 7 methods.</p></div>",
                    elem_classes="xai-report",
                )
                xai_btn.click(
                    fn=xai_wrapper,
                    inputs=[xai_image, xai_method, xai_layer, xai_head, xai_class],
                    outputs=[xai_nb],
                )

            # Sample Gallery sub-tab
            with gr.TabItem("Sample Gallery"):
                gr.Markdown("""**Pre-computed analyses** for sample images.
Run Full Suite on any sample image — results are cached and displayed here automatically.""")
                gallery_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;color:#6b7280;">'
                    "<p>Click <b>Refresh Gallery</b> to load cached analyses.</p></div>"
                )
                gallery_btn = gr.Button("Refresh Gallery", variant="secondary")
                gallery_btn.click(fn=load_sample_gallery, outputs=[gallery_html])


def _build_chat_tab(analysis_state):
    with gr.TabItem("AI Assistant"):
        gr.Markdown(
            "Ask questions about the analysis results. "
            "Run an analysis in the **Live Demo** tab first for context-aware responses."
        )

        chatbot = gr.Chatbot(
            value=[],
            height=480,
            type="messages",
            placeholder="Analysis-aware AI assistant. Ask about hazards, safety, or recommendations.",
        )

        state_ref = {"value": analysis_state}

        def _get_state():
            return state_ref["value"]

        chat_handler = create_chat_handler(_get_state)

        def _respond(message, history, current_state):
            state_ref["value"] = current_state
            response = chat_handler(message, history)
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            return history, ""

        with gr.Row():
            chat_input = gr.Textbox(
                placeholder="Ask about the analysis results...",
                show_label=False,
                scale=4,
                container=False,
            )
            send_btn = gr.Button("Send", variant="primary", scale=1, min_width=80)

        gr.Examples(
            examples=[
                "What hazards did you find in this image?",
                "Is the floor safe for someone with Alzheimer's?",
                "What should I fix first?",
                "Explain the blackspot detection results",
                "How does the contrast analysis work?",
            ],
            inputs=chat_input,
            label="Example questions",
        )

        send_btn.click(
            fn=_respond,
            inputs=[chat_input, chatbot, analysis_state],
            outputs=[chatbot, chat_input],
        )
        chat_input.submit(
            fn=_respond,
            inputs=[chat_input, chatbot, analysis_state],
            outputs=[chatbot, chat_input],
        )


def _build_technical_tab():
    with gr.TabItem("Technical Details"):
        gr.Markdown("""## Model Architecture

### EoMT-DINOv3-Large (Semantic Segmentation)
| Property | Value |
|----------|-------|
| HuggingFace Model | `tue-mps/ade20k_semantic_eomt_large_512` |
| Backbone | DINOv2 ViT-Large (24 layers, 16 heads, 1024 hidden dim) |
| Architecture | EoMT (Efficient open-vocabulary Mask Transformer) |
| Input Resolution | 512 x 512 |
| Classes | 150 (ADE20K) |
| Quantization | Dynamic INT8 on nn.Linear layers |

### Mask R-CNN R50-FPN (Blackspot Detection)
| Property | Value |
|----------|-------|
| Framework | Detectron2 |
| Backbone | ResNet-50 + Feature Pyramid Network |
| Classes | 2 (blackspot, background) |
| Training Data | 15,000+ samples with active learning |
| Precision | 98% on custom floor blackspot dataset |

## XAI Methods

| # | Method | Type | Needs FP32 | Reference |
|---|--------|------|-----------|-----------|
| 1 | Self-Attention Maps | Attention | No | Vaswani et al. 2017 |
| 2 | Attention Rollout | Attention | No | Abnar & Zuidema 2020 |
| 3 | GradCAM | Gradient | Yes | Selvaraju et al. 2017 |
| 4 | Predictive Entropy | Output | No | Shannon 1948 |
| 5 | Feature PCA | Hidden State | No | SVD projection |
| 6 | Integrated Gradients | Gradient | Yes | Sundararajan et al. 2017 |
| 7 | Chefer Relevancy | Attn x Grad | Yes | Chefer et al. 2021 |

## Deployment
- **Platform**: HuggingFace Spaces (Docker SDK)
- **Hardware**: CPU free tier (2 vCPU, 16 GB RAM)
- **Docker Base**: python:3.10-slim (Debian Trixie)
- **Key Dependencies**: PyTorch 2.5.1, Transformers, Detectron2, Gradio 5.x
""")


def _comprehensive_report(results: Dict, contrast_report: str, blackspot_report: str) -> str:
    report = [f"# NeuroNest Analysis Report\n", f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n"]

    if results["segmentation"]:
        stats = results["statistics"].get("segmentation", {})
        report.append("## Object Segmentation")
        report.append(f"- **Classes detected:** {stats.get('num_classes', 'N/A')}")
        report.append(f"- **Resolution:** {stats.get('image_size', 'N/A')}")
        report.append("")

    report.append("## Blackspot Analysis")
    report.append(blackspot_report)
    report.append("")
    report.append("## Universal Contrast Analysis")
    report.append(contrast_report)
    report.append("")
    report.append("## Recommendations for Alzheimer's Care")

    has_issues = False
    if results["blackspot"] and results["statistics"]["blackspot"]["coverage_percentage"] > 0:
        has_issues = True
        report.append("\n### Blackspot Mitigation:")
        report.append("- Replace dark flooring materials with lighter alternatives")
        report.append("- Install additional lighting in affected areas")
        report.append("- Use light-colored rugs or runners to cover dark spots")
        report.append("- Add contrasting tape or markers around blackspot perimeters")

    if results["contrast"] and results["statistics"]["contrast"]["low_contrast_pairs"] > 0:
        has_issues = True
        report.append("\n### Contrast Improvements:")
        issues = results["contrast"]["issues"]
        critical = [i for i in issues if i["severity"] == "critical"]
        high = [i for i in issues if i["severity"] == "high"]
        if critical:
            report.append("\n**CRITICAL — Immediate attention required:**")
            for issue in critical[:3]:
                c1, c2 = issue["categories"]
                report.append(f"- {c1.title()} / {c2.title()}: Increase contrast to 7:1 minimum")
        if high:
            report.append("\n**HIGH PRIORITY:**")
            for issue in high[:3]:
                c1, c2 = issue["categories"]
                report.append(f"- {c1.title()} / {c2.title()}: Increase contrast to 4.5:1 minimum")
        report.append("\n**General recommendations:**")
        report.append("- Paint furniture in colors that contrast with floors/walls")
        report.append("- Add colored tape or markers to furniture edges")
        report.append("- Install LED strip lighting under furniture edges")

    if not has_issues:
        report.append("\nThis environment appears well-optimized for individuals with Alzheimer's.")
        report.append("No significant visual hazards detected.")

    return "\n".join(report)
