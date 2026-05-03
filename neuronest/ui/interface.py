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
from ..agent import create_agent_report_generator

logger = logging.getLogger(__name__)


def create_gradio_interface():
    app = NeuroNestApp()
    seg_ok, blackspot_ok, placement_ok = app.initialize()
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

    def analyze_wrapper(
        image_path,
        blackspot_threshold,
        contrast_threshold,
        enable_blackspot,
        enable_contrast,
        enable_placement,
    ):
        if image_path is None:
            return (
                None, None, None, None,
                "Please upload an image.", {}, {},
            )
        results = app.analyze_image(
            image_path=image_path,
            blackspot_threshold=blackspot_threshold,
            contrast_threshold=contrast_threshold,
            enable_blackspot=enable_blackspot,
            enable_contrast=enable_contrast,
            enable_placement=enable_placement,
        )
        if "error" in results:
            return (
                None, None, None, None,
                f"Error: {results['error']}", {}, {},
            )
        seg_output = results["segmentation"]["visualization"] if results["segmentation"] else None
        blackspot_output = results["blackspot"]["visualization"] if results["blackspot"] else None
        contrast_output = results["contrast"]["visualization"] if results["contrast"] else None
        placement_output = (
            results["placement"]["visualization"]
            if results.get("placement") and not results["placement"].get("skipped")
            else None
        )

        report = _comprehensive_report(results)
        structured = _build_structured_json(results)
        return (
            seg_output, blackspot_output, contrast_output, placement_output,
            report, structured, results,
        )

    def _build_xai_summary(results, method_name="Full Suite"):
        METHOD_NAMES = {
            "attention": "Self-Attention Maps", "rollout": "Attention Rollout",
            "gradcam": "GradCAM Segmentation", "entropy": "Predictive Entropy",
            "pca": "Feature PCA", "integrated_gradients": "Integrated Gradients",
            "chefer": "Chefer Relevancy",
        }
        lines = [f"[XAI ANALYSIS] Method: {method_name}", f"  Methods completed: {len(results)}"]
        for key, result in results.items():
            name = METHOD_NAMES.get(key, key)
            report = result.get("report", "")
            lines.append(f"  - {name}: {report}" if report else f"  - {name}: completed")
        if "entropy" in results:
            emap = results["entropy"].get("entropy_map")
            if emap is not None:
                me = float(emap.mean())
                hp = float((emap > 0.5).mean() * 100)
                lines.append(f"  Model confidence: mean entropy={me:.3f}, high-uncertainty pixels={hp:.1f}%")
                if me < 0.10:
                    lines.append("  Assessment: Very high confidence — model strongly recognizes all elements")
                elif me < 0.20:
                    lines.append("  Assessment: High confidence — clear boundaries with minor ambiguity")
                elif me < 0.35:
                    lines.append("  Assessment: Moderate confidence — some boundaries cause uncertainty")
                else:
                    lines.append("  Assessment: Low confidence — significant uncertain regions")
        return "\n".join(lines)

    def xai_wrapper(image_path, method, layer, head_choice, target_class, current_state, progress=gr.Progress(track_tqdm=True)):
        current_state = current_state or {}
        if image_path is None:
            return "<p>Upload an image and click <b>Run XAI Analysis</b> to begin.</p>", current_state
        if app.xai_analyzer is None:
            return "<p style='color:red;'>XAI analyzer not initialized.</p>", current_state
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return "<p style='color:red;'>Could not load image.</p>", current_state
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            layer_idx = int(layer)
            head_idx = None if head_choice == "Mean (all heads)" else int(head_choice.split()[-1])
            class_id = None if target_class == "Auto (dominant)" else int(target_class.split(":")[0])

            if method == "Full Suite":
                cached = app.xai_analyzer.load_cached_from_file(image_path)
                if cached:
                    progress(1.0, desc="Loaded from cache")
                    current_state["xai_report"] = _build_xai_summary(cached, "Full Suite (cached)")
                    return render_notebook_html(cached, image_rgb), current_state

                progress(0, desc="Starting XAI Full Suite analysis...")
                results = app.xai_analyzer.run_full_analysis(
                    image_rgb, layer=layer_idx, head=head_idx,
                    target_class_id=class_id, progress_callback=progress,
                )
                current_state["xai_report"] = _build_xai_summary(results, "Full Suite")
                try:
                    app.xai_analyzer.save_results(results, image_path=image_path)
                except Exception as e:
                    logger.warning(f"Cache save failed: {e}")

                app.xai_analyzer.cleanup_fp32()
                return render_notebook_html(results, image_rgb), current_state
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
                    single_results = {key_map[method]: result}
                    current_state["xai_report"] = _build_xai_summary(single_results, method)
                    return render_notebook_html(single_results, image_rgb), current_state

            return "<p>Unknown method selected.</p>", current_state
        except Exception as e:
            logger.error(f"XAI analysis error: {e}")
            import traceback
            traceback.print_exc()
            return f"<p style='color:red;'><b>Error</b>: {str(e)}</p>", current_state

    def defect_xai_wrapper(image_path, current_state, progress=gr.Progress(track_tqdm=True)):
        current_state = current_state or {}
        if image_path is None:
            return "<p>Upload an image and click <b>Run Defect XAI</b> to begin.</p>", current_state
        if app.xai_analyzer is None:
            return "<p style='color:red;'>XAI analyzer not initialized.</p>", current_state
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image is None:
                return "<p style='color:red;'>Could not load image.</p>", current_state
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            cached = XAIAnalyzer.load_cached_defect_from_file(image_path)
            if cached:
                progress(1.0, desc="Loaded defect analysis from cache")
                current_state["xai_report"] = (
                    "[DEFECT XAI ANALYSIS] Loaded from cache\n"
                    f"  Methods: {len(cached)} (GradCAM, Entropy, IG, Chefer)"
                )
                return render_defect_notebook_html(cached, image_rgb), current_state

            progress(0.05, desc="Running segmentation...")
            seg_mask, _ = app.segmenter.semantic_segmentation(image_rgb)
            floor_prior = app.segmenter.extract_floor_areas(seg_mask)

            blackspot_mask = np.zeros(image_rgb.shape[:2], dtype=bool)
            blackspot_count = 0
            blackspot_coverage = 0.0
            contrast_issues = []
            placement_detections = []
            placement_count = 0
            placement_violations = 0

            if app.blackspot_detector is not None:
                progress(0.15, desc="Detecting blackspots...")
                try:
                    bs = app.blackspot_detector.detect_blackspots(image_rgb, seg_mask, floor_prior)
                    blackspot_mask = bs.get("blackspot_mask", blackspot_mask)
                    blackspot_count = bs.get("num_detections", 0)
                    blackspot_coverage = bs.get("coverage_percentage", 0.0)
                except Exception as e:
                    logger.warning(f"Blackspot detection failed: {e}")

            progress(0.22, desc="Analyzing contrast...")
            try:
                contrast_result = app.contrast_analyzer.analyze_contrast(image_rgb, seg_mask)
                contrast_issues = contrast_result.get("issues", [])
            except Exception as e:
                logger.warning(f"Contrast analysis failed: {e}")

            if (
                app.placement_analyzer is not None
                and app.placement_analyzer.depth_model.is_loaded
            ):
                progress(0.28, desc="Analyzing sign & clock placement...")
                try:
                    placement_result = app.placement_analyzer.analyze_placement(
                        image_rgb, seg_mask, floor_prior,
                    )
                    if not placement_result.get("skipped"):
                        placement_detections = placement_result.get("detections", [])
                        placement_count = placement_result.get("num_detections", 0)
                        placement_violations = placement_result.get("num_violations", 0)
                except Exception as e:
                    logger.warning(f"Placement analysis failed: {e}")

            progress(0.32, desc="Starting defect XAI analysis...")

            def defect_progress(frac, desc=""):
                progress(0.32 + frac * 0.63, desc=desc)

            results = app.xai_analyzer.run_defect_analysis(
                image_rgb, blackspot_mask=blackspot_mask,
                contrast_issues=contrast_issues, seg_mask=seg_mask,
                placement_detections=placement_detections,
                progress_callback=defect_progress,
            )

            try:
                app.xai_analyzer.save_defect_results(results, image_path=image_path)
            except Exception as e:
                logger.warning(f"Defect cache save failed: {e}")

            target_class = app.xai_analyzer._dominant_class(seg_mask)
            floor_name = app.xai_analyzer._cname(target_class)

            defect_summary_lines = [
                "[DEFECT XAI ANALYSIS]",
                f"  Floor class targeted: {floor_name}",
                f"  Blackspot detections: {blackspot_count}",
                f"  Blackspot coverage: {blackspot_coverage:.2f}%",
                f"  Contrast failures overlaid: {len(contrast_issues)}",
                f"  Placement detections: {placement_count} "
                f"({placement_violations} violating ADA range)",
                f"  XAI methods applied: {len(results)} (GradCAM, Entropy, Integrated Gradients, Chefer Relevancy)",
            ]
            for key, result in results.items():
                report = result.get("report", "")
                if report:
                    defect_summary_lines.append(f"  - {key}: {report}")
            current_state["xai_report"] = "\n".join(defect_summary_lines)

            progress(1.0, desc="Complete")
            return render_defect_notebook_html(
                results, image_rgb,
                blackspot_count=blackspot_count,
                blackspot_coverage=blackspot_coverage,
                floor_class_name=floor_name,
            ), current_state
        except Exception as e:
            logger.error(f"Defect XAI error: {e}")
            import traceback
            traceback.print_exc()
            return f"<p style='color:red;'><b>Error</b>: {str(e)}</p>", current_state

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

    agent_report_generator = create_agent_report_generator()

    # --- Build Gradio interface ---

    with gr.Blocks(
        css=MAIN_CSS,
        theme=gr.themes.Soft(primary_hue="indigo", secondary_hue="blue", neutral_hue="slate"),
    ) as interface:
        analysis_state = gr.State(value={})

        with gr.Column(elem_classes="container"):
            gr.HTML(_HERO_HTML)

            # Tier 1: Informational tabs (lighter)
            with gr.Tabs(elem_classes="info-tabs"):
                _build_overview_tab()
                _build_team_tab()
                _build_technical_tab()

            # Tier 2: Workspace tabs (prominent)
            with gr.Tabs(elem_classes="workspace-tabs"):
                agent_report_in_full = _build_demo_tab(
                    app, analyze_wrapper, xai_wrapper, SAMPLE_IMAGES,
                    sample_images_available, blackspot_ok, placement_ok,
                    analysis_state,
                )
                _build_xai_tab(
                    xai_wrapper, defect_xai_wrapper, load_sample_gallery,
                    SAMPLE_IMAGES, sample_images_available, analysis_state,
                    agent_report_generator, agent_report_in_full,
                )

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
    <div class="try-now-row">
        <button class="try-now-btn" onclick="
            var ws = document.querySelector('.workspace-tabs');
            if (ws) {
                ws.scrollIntoView({behavior: 'smooth', block: 'start'});
                var btn = ws.querySelector('.tab-nav button');
                if (btn) setTimeout(function(){ btn.click(); }, 400);
            }
        ">Try Live Demo</button>
        <button class="try-now-btn try-now-secondary" onclick="
            var ws = document.querySelector('.workspace-tabs');
            if (ws) {
                ws.scrollIntoView({behavior: 'smooth', block: 'start'});
                var btns = ws.querySelectorAll('.tab-nav button');
                if (btns.length > 1) setTimeout(function(){ btns[1].click(); }, 400);
            }
        ">Explore Model XAI</button>
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


def _build_demo_tab(app, analyze_wrapper, xai_wrapper, sample_images, sample_available, blackspot_ok, placement_ok, analysis_state):
    with gr.TabItem("Live Demo"):
        gr.Markdown("""## How It Works

1. **Upload** a room photo (or select a sample below)
2. **Click Analyze** — the pipeline runs semantic segmentation (150-class EoMT-DINOv3), blackspot detection (Mask R-CNN), WCAG 2.1 contrast analysis, monocular metric depth + ADA sign/clock placement analysis, and optionally 7 Explainable AI methods
3. **Review results** across tabs: segmentation map, blackspot hazards, contrast issues, sign & clock placement, XAI visualizations, and a full safety report

The system identifies visual hazards that cause falls in Alzheimer's patients — dark floor areas perceived as voids (blackspots), low-contrast boundaries between surfaces, and signs or clocks placed outside the ADA-recommended 48-60 inch centroid range.""")

        with gr.Row(visible=False):
            image_input = gr.Image(label="Upload", type="filepath")

        if sample_available:
            with gr.Column(elem_classes="sample-section"):
                gr.Markdown("**Try a sample image** or upload your own below.")
                gr.Examples(examples=sample_images, inputs=image_input, label="", examples_per_page=3)

        with gr.Row():
            with gr.Column(scale=3):
                image_input_display = gr.Image(label="Upload Room Image", type="filepath", height=400)
                image_input.change(fn=lambda x: x, inputs=image_input, outputs=image_input_display)
            with gr.Column(scale=1):
                analyze_button = gr.Button("Analyze Environment", variant="primary", elem_classes="main-button", size="lg")
                enable_xai = gr.Checkbox(value=True, label="Include Explainable AI (7 methods, adds ~3-5 min)")
                with gr.Accordion("Settings", open=False):
                    enable_blackspot = gr.Checkbox(value=blackspot_ok, label="Blackspot Detection", interactive=blackspot_ok)
                    blackspot_threshold = gr.Slider(minimum=0.1, maximum=0.9, value=0.5, step=0.05, label="Blackspot Sensitivity", visible=blackspot_ok)
                    enable_contrast = gr.Checkbox(value=True, label="Contrast Analysis")
                    contrast_threshold = gr.Slider(minimum=3.0, maximum=7.0, value=4.5, step=0.1, label="WCAG Contrast Threshold", info="4.5:1 = WCAG AA, 7:1 = AAA")
                    enable_placement = gr.Checkbox(
                        value=placement_ok,
                        label="Sign & Clock Placement (ADA)",
                        interactive=placement_ok,
                        info="Adds ~3-5s on CPU. Requires monocular metric depth model.",
                    )

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
            with gr.TabItem("Sign & Clock Placement"):
                if placement_ok:
                    placement_display = gr.Image(
                        label="ADA-compliant centroid heights (48-60 in). "
                              "Color: critical/high/medium/ok severity.",
                        interactive=False, elem_classes="image-output",
                    )
                else:
                    placement_display = gr.Image(visible=False)
                    gr.Markdown(
                        "Placement analyzer not available in this deployment "
                        "(monocular metric depth model failed to load)."
                    )
            with gr.TabItem("Explainable AI"):
                xai_auto_html = gr.HTML(
                    value='<div style="text-align:center;padding:40px;color:#6b7280;">'
                    "<p>XAI results will appear here after analysis completes.</p>"
                    "<p style='font-size:0.85em;'>Uncheck <b>Include Explainable AI</b> above to skip.</p></div>",
                    elem_classes="xai-report",
                )
            with gr.TabItem("Full Report"):
                analysis_report = gr.Markdown(value="Upload an image and click **Analyze Environment** to begin.", elem_classes="report-box")
                with gr.Accordion("Structured Data (JSON)", open=False):
                    structured_json_display = gr.JSON(label="Analysis Data")
                agent_report_section = gr.HTML(value="", visible=False, elem_classes="report-box")

        def auto_xai_wrapper(image_path, enable_xai_flag, current_state, progress=gr.Progress(track_tqdm=True)):
            current_state = current_state or {}
            if not enable_xai_flag or image_path is None or app.xai_analyzer is None:
                msg = "XAI analysis skipped." if not enable_xai_flag else "No image provided."
                return (
                    f'<div style="text-align:center;padding:40px;color:#6b7280;"><p>{msg}</p></div>',
                    current_state,
                )
            return xai_wrapper(
                image_path, "Full Suite", 19, "Mean (all heads)",
                "Auto (dominant)", current_state, progress,
            )

        analyze_button.click(
            fn=analyze_wrapper,
            inputs=[
                image_input_display, blackspot_threshold, contrast_threshold,
                enable_blackspot, enable_contrast, enable_placement,
            ],
            outputs=[
                seg_display, blackspot_display, contrast_display, placement_display,
                analysis_report, structured_json_display, analysis_state,
            ],
        ).then(
            fn=auto_xai_wrapper,
            inputs=[image_input_display, enable_xai, analysis_state],
            outputs=[xai_auto_html, analysis_state],
        )

    return agent_report_section


def _build_xai_tab(xai_wrapper, defect_xai_wrapper, load_sample_gallery, sample_images, sample_available, analysis_state, agent_report_generator=None, agent_report_in_full=None):
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
                defect_btn.click(fn=defect_xai_wrapper, inputs=[defect_image, analysis_state], outputs=[defect_nb, analysis_state])

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
                    inputs=[xai_image, xai_method, xai_layer, xai_head, xai_class, analysis_state],
                    outputs=[xai_nb, analysis_state],
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

        # Agent panel — floating accordion below XAI sub-tabs
        if agent_report_generator is not None:
            with gr.Accordion("AI Agent Analysis", open=False, elem_classes="agent-panel"):
                gr.Markdown(
                    "Generate an AI-powered interpretation of your XAI results. "
                    "Run any XAI method above first, then click the button below."
                )
                agent_btn = gr.Button(
                    "Generate AI Analysis", elem_classes="agent-button", size="lg",
                )
                agent_html = gr.HTML(
                    value='<div style="text-align:center;padding:20px;color:#9ca3af;">'
                    "Run an XAI analysis above, then click <b>Generate AI Analysis</b>.</div>",
                    elem_classes="agent-output",
                )

                def _run_agent(current_state):
                    xai_text = (current_state or {}).get("xai_report", "")
                    report_html = agent_report_generator(xai_text)
                    new_state = dict(current_state or {})
                    new_state["agent_report"] = report_html
                    if agent_report_in_full is not None:
                        return (
                            report_html,
                            new_state,
                            gr.update(
                                value=f'<h2 style="color:#4f46e5;">AI Agent Interpretation</h2>{report_html}',
                                visible=True,
                            ),
                        )
                    return report_html, new_state

                if agent_report_in_full is not None:
                    agent_btn.click(
                        fn=_run_agent,
                        inputs=[analysis_state],
                        outputs=[agent_html, analysis_state, agent_report_in_full],
                    )
                else:
                    agent_btn.click(
                        fn=_run_agent,
                        inputs=[analysis_state],
                        outputs=[agent_html, analysis_state],
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


def _describe_location(cx, cy, img_w, img_h):
    """Convert centroid pixel coordinates to a human-readable region name."""
    v = "upper" if cy < img_h / 3 else ("lower" if cy > img_h * 2 / 3 else "center")
    h = "left" if cx < img_w / 3 else ("right" if cx > img_w * 2 / 3 else "center")
    if v == "center" and h == "center":
        return "center"
    if v == "center":
        return h
    if h == "center":
        return v
    return f"{v}-{h}"


def _build_structured_json(results: Dict) -> Dict:
    """Build a JSON-serializable structured analysis dict for the API."""
    data: Dict = {}

    bs = results.get("blackspot")
    if bs:
        data["blackspots"] = {
            "count": bs.get("num_detections", 0),
            "coverage_pct": round(bs.get("coverage_percentage", 0), 2),
            "avg_confidence": round(bs.get("avg_confidence", 0), 3),
            "floor_area_pixels": bs.get("floor_area", 0),
            "blackspot_area_pixels": bs.get("blackspot_area", 0),
            "detections": bs.get("detections", []),
        }

    ct = results.get("contrast")
    if ct:
        stats = ct.get("statistics", {})
        serialized = []
        for issue in ct.get("issues", []):
            serialized.append({
                "severity": issue["severity"],
                "surfaces": list(issue["categories"]),
                "wcag_ratio": round(issue["wcag_ratio"], 2),
                "colors": {
                    "surface_1": issue["colors"][0],
                    "surface_2": issue["colors"][1],
                },
                "hue_difference": round(issue["hue_difference"], 1),
                "saturation_difference": int(issue["saturation_difference"]),
                "boundary_pixels": issue["boundary_pixels"],
            })
        data["contrast"] = {
            "total_issues": stats.get("low_contrast_pairs", 0),
            "by_severity": {
                "critical": stats.get("critical_issues", 0),
                "high": stats.get("high_priority_issues", 0),
                "medium": stats.get("medium_priority_issues", 0),
            },
            "issues": serialized,
        }

    pl = results.get("placement")
    if pl:
        if pl.get("skipped"):
            data["placement"] = {
                "skipped": True,
                "reason": pl.get("reason", "unknown"),
            }
        else:
            data["placement"] = {
                "count": pl.get("num_detections", 0),
                "violations": pl.get("num_violations", 0),
                "calibration": pl.get("calibration_source"),
                "scale_factor": pl.get("scale_factor", 1.0),
                "ada_recommended_range_in": pl.get("ada_recommended_range_in"),
                "detections": pl.get("detections", []),
            }

    return data


def _comprehensive_report(results: Dict) -> str:
    """Build a user-facing markdown safety report from pipeline results."""
    lines = [
        f"# NeuroNest Safety Analysis\n",
        f"*{time.strftime('%Y-%m-%d %H:%M:%S')}*\n",
    ]

    # ── Scene overview ──
    if results.get("segmentation"):
        s = results.get("statistics", {}).get("segmentation", {})
        lines.append("## Scene Overview\n")
        lines.append(f"- **Indoor classes detected:** {s.get('num_classes', 'N/A')}")
        sz = s.get("image_size")
        if sz:
            lines.append(f"- **Resolution:** {sz[1]} \u00d7 {sz[0]}")
        lines.append("")

    # ── Blackspot hazards ──
    lines.append("---\n")
    lines.append("## Blackspot Hazards\n")
    bs = results.get("blackspot")
    if bs is None:
        lines.append("*Blackspot detection not available in this deployment.*\n")
    elif bs["num_detections"] == 0:
        lines.append("**No blackspot hazards detected.**\n")
    else:
        conf_pct = f"{bs['avg_confidence'] * 100:.0f}%"
        lines.append(
            f"**{bs['num_detections']} detected** \u00b7 "
            f"{bs['coverage_percentage']:.1f}% floor coverage \u00b7 "
            f"{conf_pct} avg confidence\n"
        )
        detections = bs.get("detections", [])
        if detections:
            img = results.get("original_image")
            img_h, img_w = img.shape[:2] if img is not None else (1, 1)
            lines.append("| # | Location | Area | Confidence |")
            lines.append("|---|----------|------|------------|")
            for d in detections:
                cx, cy = d["centroid"]
                loc = _describe_location(cx, cy, img_w, img_h)
                area = f"{d['area_pixels']:,} px"
                conf = f"{d['confidence'] * 100:.0f}%"
                lines.append(f"| {d['id']} | {loc} | {area} | {conf} |")
            lines.append("")

    # ── Contrast issues ──
    lines.append("---\n")
    lines.append("## Contrast Issues\n")
    ct = results.get("contrast")
    if ct is None:
        lines.append("*Contrast analysis not performed.*\n")
    else:
        issues = ct.get("issues", [])
        stats = ct.get("statistics", {})
        total = stats.get("low_contrast_pairs", 0)
        if total == 0:
            lines.append("**No contrast issues \u2014 environment is well optimized.**\n")
        else:
            parts = []
            for key, label in [("critical_issues", "critical"), ("high_priority_issues", "high"), ("medium_priority_issues", "medium")]:
                n = stats.get(key, 0)
                if n:
                    parts.append(f"{n} {label}")
            sep = ' \u00b7 '
            lines.append(f"**{total} issues found** \u2014 {sep.join(parts)}\n")

            for severity, heading, req in [
                ("critical", "Critical", "7.0:1"),
                ("high", "High Priority", "4.5:1"),
                ("medium", "Medium", "3.0:1"),
            ]:
                group = [i for i in issues if i["severity"] == severity]
                if not group:
                    continue
                lines.append(f"### {heading} (requires {req})\n")
                lines.append("| Boundary | Ratio | Color 1 | Color 2 |")
                lines.append("|----------|-------|---------|---------|")
                for issue in group:
                    c1_cat, c2_cat = issue["categories"]
                    ratio = f"{issue['wcag_ratio']:.1f}:1"
                    col1 = issue["colors"][0]
                    col2 = issue["colors"][1]
                    c1_str = f"{col1['name']} `{col1['hex']}`"
                    c2_str = f"{col2['name']} `{col2['hex']}`"
                    lines.append(f"| {c1_cat.title()} \u2194 {c2_cat.title()} | {ratio} | {c1_str} | {c2_str} |")
                lines.append("")

    # ── Sign & Clock Placement ──
    lines.append("---\n")
    lines.append("## Sign & Clock Placement\n")
    pl = results.get("placement")
    placement_violations = []
    if pl is None:
        lines.append("*Placement analysis not performed.*\n")
    elif pl.get("skipped"):
        reason = pl.get("reason", "unknown")
        lines.append(f"*Placement analysis skipped — {reason}.*\n")
    else:
        n_det = pl.get("num_detections", 0)
        n_vio = pl.get("num_violations", 0)
        rng = pl.get("ada_recommended_range_in") or [48, 60]
        cal = pl.get("calibration_source", "prior")
        scale = pl.get("scale_factor", 1.0)
        lines.append(
            f"**{n_det} detected** · **{n_vio} outside ADA range** "
            f"({rng[0]:.0f}\u2013{rng[1]:.0f}\") · "
            f"calibration: *{cal}* (scale {scale:.2f})\n"
        )
        detections = pl.get("detections", [])
        if detections:
            lines.append("| # | Class | Height (in) | Status | Confidence |")
            lines.append("|---|-------|-------------|--------|------------|")
            for d in detections:
                cls = d["class"]
                h = d["height_in"]
                u = d["height_in_uncertainty"]
                sev = d["severity"]
                vio = d.get("violation_type")
                if sev == "ok":
                    status = "OK"
                else:
                    arrow = "\u2193" if vio == "below" else "\u2191"
                    status = f"{sev.upper()} {arrow}"
                    placement_violations.append(d)
                conf = f"{d['confidence'] * 100:.0f}%"
                lines.append(
                    f"| {d['id']} | {cls} | {h:.1f} \u00b1 {u:.1f} | {status} | {conf} |"
                )
            lines.append("")

    # ── Recommendations ──
    lines.append("---\n")
    lines.append("## Recommendations\n")
    has_issues = False

    if bs and bs.get("num_detections", 0) > 0:
        has_issues = True
        lines.append("### Blackspot Mitigation\n")
        lines.append("- Replace dark flooring with lighter alternatives")
        lines.append("- Add lighting in affected areas")
        lines.append("- Use light-colored rugs to cover dark spots")
        lines.append("- Add contrasting tape around blackspot perimeters\n")

    if ct and stats.get("low_contrast_pairs", 0) > 0:
        has_issues = True
        lines.append("### Contrast Improvements\n")
        floor_issues = [i for i in ct.get("issues", []) if i.get("is_floor_object")]
        if floor_issues:
            lines.append("- Ensure floor-level objects visually contrast with flooring")
        lines.append("- Paint furniture in colors that contrast with floors and walls")
        lines.append("- Add colored tape or markers to furniture edges")
        lines.append("- Install LED strip lighting under furniture edges\n")

    if placement_violations:
        has_issues = True
        lines.append("### Sign & Clock Placement Adjustments\n")
        rng = pl.get("ada_recommended_range_in") or [48, 60]
        for d in placement_violations:
            cls = d["class"]
            h = d["height_in"]
            vio = d.get("violation_type")
            if vio == "below":
                target = rng[0]
                delta = target - h
                lines.append(
                    f"- Raise the {cls} (#{d['id']}) by approximately "
                    f"{delta:.0f} in to reach the ADA-recommended "
                    f"{rng[0]:.0f}\u2013{rng[1]:.0f}\" centroid range."
                )
            else:
                target = rng[1]
                delta = h - target
                lines.append(
                    f"- Lower the {cls} (#{d['id']}) by approximately "
                    f"{delta:.0f} in to reach the ADA-recommended "
                    f"{rng[0]:.0f}\u2013{rng[1]:.0f}\" centroid range."
                )
        lines.append(
            "- Where possible, mount signs at the lower end of the range "
            "(closer to 48 in) for residents who use wheelchairs or have "
            "stooped posture.\n"
        )

    if not has_issues:
        lines.append("Environment appears well-optimized for Alzheimer's care.  ")
        lines.append("No significant visual hazards detected.\n")

    return "\n".join(lines)
