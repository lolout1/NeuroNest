import re
import base64
from io import BytesIO
from PIL import Image

_NB_CSS = """<style>
.nb{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;max-width:1100px;margin:0 auto}
.nb-cell{margin:20px 0;padding:20px;background:#fafbfc;border:1px solid #e5e7eb;border-radius:12px}
.nb-cell-header{display:flex;align-items:center;gap:10px;margin-bottom:14px;flex-wrap:wrap}
.nb-num{display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;border-radius:50%;font-size:.8em;font-weight:700;color:#fff}
.nb-title{font-size:1.15em;font-weight:700;color:#1f2937}
.nb-badge{padding:2px 10px;border-radius:12px;font-size:.72em;font-weight:600;color:#fff}
.nb-what{color:#374151;font-size:.95em;margin:8px 0 4px;font-weight:500;line-height:1.5}
.nb-why{color:#6b7280;font-size:.85em;line-height:1.6;margin:4px 0 12px}
.nb-img{width:100%;border-radius:8px;border:1px solid #d1d5db;margin:10px 0}
.nb-metrics{color:#4b5563;font-size:.82em;padding:8px 12px;background:#f3f4f6;border-radius:8px;margin-top:8px;line-height:1.6;font-family:'SF Mono',Monaco,monospace}
.nb-sep{border:none;border-top:2px solid #e5e7eb;margin:28px 0}
.nb-section-header{font-size:1.3em;font-weight:800;color:#1f2937;margin:24px 0 8px;display:flex;align-items:center;gap:8px}
.nb-section-desc{color:#6b7280;font-size:.9em;margin-bottom:16px}
.nb-report{margin:24px 0;padding:24px;background:#fff;border:1px solid #e5e7eb;border-radius:12px;line-height:1.8}
.nb-report h1,.nb-report h2,.nb-report h3{color:#1f2937;margin-top:20px}
.nb-report table{width:100%;border-collapse:collapse;margin:12px 0;font-size:.9em}
.nb-report th,.nb-report td{padding:8px 12px;border:1px solid #e5e7eb;text-align:left}
.nb-report th{background:#f9fafb;font-weight:600}
.nb-report blockquote{border-left:3px solid #6366f1;padding:8px 16px;margin:12px 0;background:#f5f3ff;border-radius:0 8px 8px 0;color:#4b5563}
.nb-error{color:#dc2626;background:#fef2f2;padding:12px;border-radius:8px;border:1px solid #fecaca}
.defect-header{padding:20px;background:linear-gradient(135deg,#fef2f2,#fff7ed);border:1px solid #fecaca;border-radius:12px;margin-bottom:20px}
.defect-legend{display:flex;gap:20px;margin-top:12px;flex-wrap:wrap}
.defect-legend-item{display:flex;align-items:center;gap:6px;font-size:.85em}
.defect-swatch{width:24px;height:12px;border-radius:3px;border:1px solid #d1d5db}
@media(max-width:768px){
 .nb{padding:0 4px}.nb-cell{margin:12px 0;padding:12px}
 .nb-cell-header{gap:6px}.nb-num{width:24px;height:24px;font-size:.7em}
 .nb-title{font-size:1em}.nb-badge{font-size:.65em;padding:1px 8px}
 .nb-what{font-size:.88em}.nb-why{font-size:.8em}
 .nb-metrics{font-size:.75em;padding:6px 8px}
 .nb-section-header{font-size:1.1em}.nb-report{padding:12px}
 .nb-report table{font-size:.8em;display:block;overflow-x:auto}
 .nb-report th,.nb-report td{padding:4px 8px}
 .defect-header{padding:14px}.defect-legend{gap:10px}.defect-legend-item{font-size:.78em}
}
@media(max-width:480px){
 .nb-cell{padding:8px;margin:8px 0}.nb-title{font-size:.9em}
 .nb-why{font-size:.75em}.nb-section-header{font-size:.95em}.defect-header{padding:10px}
}
@media(prefers-color-scheme:dark){
 .nb-cell{background:#1f2937;border-color:#374151}.nb-title{color:#f3f4f6}
 .nb-what{color:#d1d5db}.nb-why{color:#9ca3af}
 .nb-metrics{background:#374151;color:#d1d5db}
 .nb-report{background:#111827;border-color:#374151}
 .nb-report h1,.nb-report h2,.nb-report h3{color:#f3f4f6}
 .nb-report th{background:#374151}
 .nb-report blockquote{background:#312e81;border-color:#6366f1;color:#c7d2fe}
 .nb-section-header{color:#f3f4f6}.nb-section-desc{color:#9ca3af}
 .defect-header{background:linear-gradient(135deg,#451a1a,#451a03);border-color:#7f1d1d}
}
</style>"""

_NB_METHODS = [
    {"key": "attention", "num": 1, "category": "Attention-Based", "cat_color": "#6366f1",
     "title": "Self-Attention Map",
     "what": "Shows where a single transformer layer focuses its spatial attention.",
     "why": "Each of the 16 attention heads learns different patterns. This map reveals the raw attention distribution before aggregation."},
    {"key": "rollout", "num": 2, "category": "Attention-Based", "cat_color": "#6366f1",
     "title": "Attention Rollout",
     "what": "Cumulative attention flow aggregated across all 20 encoder layers.",
     "why": "Multiplying attention matrices layer-by-layer reveals the model's overall spatial priority. Brighter regions are where information flows to the CLS token."},
    {"key": "gradcam", "num": 3, "category": "Gradient-Based", "cat_color": "#dc2626",
     "title": "GradCAM",
     "what": "Gradient-weighted class activation map — which internal features activate for a class.",
     "why": "Computes the gradient of the target class score w.r.t. last encoder layer activations. Red/yellow = strongest activation for the predicted class."},
    {"key": "entropy", "num": 4, "category": "Output Analysis", "cat_color": "#059669",
     "title": "Predictive Entropy",
     "what": "Per-pixel classification uncertainty — dark = confident, bright = uncertain.",
     "why": "Shannon entropy over the 150-class posterior. High entropy at boundaries reveals where the model struggles — correlating with visual transitions challenging for Alzheimer's patients."},
    {"key": "pca", "num": 5, "category": "Output Analysis", "cat_color": "#059669",
     "title": "Feature PCA",
     "what": "Learned feature structure — similar colors indicate similar internal representations.",
     "why": "Projects 1024-dim hidden states to 3 principal components (RGB). Distinct color boundaries between floor and furniture indicate strong feature-level differentiation."},
    {"key": "integrated_gradients", "num": 6, "category": "Gradient-Based", "cat_color": "#dc2626",
     "title": "Integrated Gradients",
     "what": "Principled pixel-level attribution via path integral from black baseline to input.",
     "why": "Satisfies the completeness axiom (Sundararajan 2017): attributions sum to the prediction difference. Identifies which pixels most influence the target class."},
    {"key": "chefer", "num": 7, "category": "Gradient-Based", "cat_color": "#dc2626",
     "title": "Chefer Relevancy",
     "what": "Attention x gradient propagation — transformer-specific relevance attribution.",
     "why": "Most theoretically grounded XAI for ViTs (Chefer 2021). Combines attention with gradient flow across all encoder layers. Green overlay = model-relevant regions."},
]

_DEFECT_NB_METHODS = [
    {"key": "gradcam", "num": 1, "category": "Defect Attribution", "cat_color": "#dc2626",
     "title": "GradCAM — Floor Activation",
     "what": "Which internal features activate for the floor class where blackspots are detected.",
     "why": "Highlights the model's feature-level response to the floor surface. Red contours show blackspot locations relative to activation."},
    {"key": "entropy", "num": 2, "category": "Defect Uncertainty", "cat_color": "#d97706",
     "title": "Predictive Entropy — Defect Boundary Uncertainty",
     "what": "Per-pixel classification uncertainty at defect boundaries.",
     "why": "High entropy at blackspot edges indicates the model struggles to classify those regions — the same visual ambiguity that challenges Alzheimer's patients."},
    {"key": "integrated_gradients", "num": 3, "category": "Defect Attribution", "cat_color": "#dc2626",
     "title": "Integrated Gradients — Pixel Attribution",
     "what": "Which input pixels most influence the floor class prediction at defect sites.",
     "why": "Where attributions are weak near blackspots, the model is less certain about the floor surface — matching the visual hazard for cognitively impaired individuals."},
    {"key": "chefer", "num": 4, "category": "Defect Relevance", "cat_color": "#7c3aed",
     "title": "Chefer Relevancy — Transformer Attention at Defects",
     "what": "Which regions the transformer considers relevant for floor prediction near defects.",
     "why": "Defect overlay reveals whether blackspots fall in high or low relevance zones — low relevance suggests the model may miss or misclassify the defect area."},
]


def _img_b64(img_array, quality=85):
    if img_array is None:
        return ""
    pil = Image.fromarray(img_array)
    buf = BytesIO()
    pil.save(buf, format="JPEG", quality=quality, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def _render_input_cell(original_image):
    if original_image is None:
        return ""
    b64 = _img_b64(original_image)
    return f"""
    <div class="nb-cell">
        <div class="nb-cell-header">
            <span class="nb-num" style="background:#1f2937;">In</span>
            <span class="nb-title">Input Image</span>
        </div>
        <img class="nb-img" src="data:image/jpeg;base64,{b64}" alt="Input" />
    </div>"""


def _render_method_cell(m, r):
    vis = r.get("visualization")
    report = r.get("report", "")
    is_error = "Error" in report

    parts = [f"""
    <div class="nb-cell">
        <div class="nb-cell-header">
            <span class="nb-num" style="background:{m['cat_color']};">{m['num']}</span>
            <span class="nb-title">{m['title']}</span>
            <span class="nb-badge" style="background:{m['cat_color']};">{m['category']}</span>
        </div>
        <div class="nb-what">{m['what']}</div>
        <div class="nb-why">{m['why']}</div>
    """]

    if vis is not None:
        b64 = _img_b64(vis)
        parts.append(f'<img class="nb-img" src="data:image/jpeg;base64,{b64}" alt="{m["title"]}" />')
    elif is_error:
        parts.append(f'<div class="nb-error">{report}</div>')
    else:
        parts.append('<div class="nb-error">Not computed</div>')

    if report and not is_error:
        parts.append(f'<div class="nb-metrics">{report}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def render_notebook_html(results: dict, original_image=None) -> str:
    parts = [_NB_CSS, '<div class="nb">']
    parts.append(_render_input_cell(original_image))

    cat_descs = {
        "Attention-Based": "Where does the model look? Spatial attention patterns across transformer layers.",
        "Gradient-Based": "What drives predictions? Importance attribution to input regions and internal features.",
        "Output Analysis": "How confident is the model? Prediction certainty and learned representations.",
    }

    current_cat = None
    for m in _NB_METHODS:
        if m["category"] != current_cat:
            current_cat = m["category"]
            parts.append(f'<hr class="nb-sep">')
            parts.append(
                f'<div class="nb-section-header">'
                f'<span style="color:{m["cat_color"]};">&#9679;</span> {current_cat}</div>'
            )
            parts.append(f'<div class="nb-section-desc">{cat_descs.get(current_cat, "")}</div>')

        r = results.get(m["key"], {})
        parts.append(_render_method_cell(m, r))

    full_report = results.get("report", {}).get("report", "")
    if full_report:
        rpt_html = _md_to_html(full_report)
        parts.append('<hr class="nb-sep">')
        parts.append('<div class="nb-section-header">Comprehensive Analysis Report</div>')
        parts.append(f'<div class="nb-report">{rpt_html}</div>')

    parts.append("</div>")
    return "\n".join(parts)


def render_defect_notebook_html(
    results: dict, original_image=None,
    blackspot_count: int = 0, blackspot_coverage: float = 0.0,
    floor_class_name: str = "floor",
) -> str:
    parts = [_NB_CSS, '<div class="nb">']

    parts.append(f"""
    <div class="defect-header">
        <div style="font-size:1.3em;font-weight:800;color:#991b1b;">Defect Interpretability Analysis</div>
        <div style="color:#6b7280;margin:8px 0;line-height:1.6;">
            XAI methods targeting <strong>{floor_class_name}</strong> with detected defect overlays.
        </div>
        <div style="display:flex;gap:24px;flex-wrap:wrap;margin:12px 0;">
            <div><span style="font-size:1.5em;font-weight:800;color:#dc2626;">{blackspot_count}</span>
                 <span style="color:#6b7280;font-size:.85em;"> blackspots</span></div>
            <div><span style="font-size:1.5em;font-weight:800;color:#d97706;">{blackspot_coverage:.2f}%</span>
                 <span style="color:#6b7280;font-size:.85em;"> floor coverage</span></div>
        </div>
        <div class="defect-legend">
            <div class="defect-legend-item"><div class="defect-swatch" style="background:#ff3232;"></div><span>Blackspot contours</span></div>
            <div class="defect-legend-item"><div class="defect-swatch" style="background:#ffb400;"></div><span>Contrast failures</span></div>
            <div class="defect-legend-item"><div class="defect-swatch" style="background:#22c55e;"></div><span>Floor surface</span></div>
        </div>
    </div>""")

    parts.append(_render_input_cell(original_image))

    for m in _DEFECT_NB_METHODS:
        r = results.get(m["key"], {})
        vis = r.get("defect_visualization") or r.get("visualization")
        cell_result = {"visualization": vis, "report": r.get("report", "")}
        parts.append(_render_method_cell(m, cell_result))

    parts.append("</div>")
    return "\n".join(parts)


def _md_to_html(md_text: str) -> str:
    lines = md_text.split("\n")
    html_lines = []
    in_table = False
    in_list = False
    in_blockquote = False

    for line in lines:
        stripped = line.strip()

        if in_blockquote and not stripped.startswith(">"):
            html_lines.append("</blockquote>")
            in_blockquote = False
        if in_list and not stripped.startswith("- ") and not stripped.startswith("* "):
            html_lines.append("</ul>")
            in_list = False
        if in_table and not stripped.startswith("|"):
            html_lines.append("</tbody></table>")
            in_table = False

        if not stripped:
            html_lines.append("<br>")
            continue

        if stripped.startswith("# "):
            html_lines.append(f"<h1>{stripped[2:]}</h1>")
        elif stripped.startswith("## "):
            html_lines.append(f"<h2>{stripped[3:]}</h2>")
        elif stripped.startswith("### "):
            html_lines.append(f"<h3>{stripped[4:]}</h3>")
        elif stripped.startswith("> "):
            if not in_blockquote:
                html_lines.append("<blockquote>")
                in_blockquote = True
            html_lines.append(f"<p>{_inline_md(stripped[2:])}</p>")
        elif stripped.startswith("|"):
            cells = [c.strip() for c in stripped.split("|")[1:-1]]
            if all(set(c) <= set("-: ") for c in cells):
                continue
            if not in_table:
                in_table = True
                html_lines.append("<table><thead><tr>")
                for c in cells:
                    html_lines.append(f"<th>{_inline_md(c)}</th>")
                html_lines.append("</tr></thead><tbody>")
            else:
                html_lines.append("<tr>")
                for c in cells:
                    html_lines.append(f"<td>{_inline_md(c)}</td>")
                html_lines.append("</tr>")
        elif stripped.startswith("- ") or stripped.startswith("* "):
            if not in_list:
                in_list = True
                html_lines.append("<ul>")
            html_lines.append(f"<li>{_inline_md(stripped[2:])}</li>")
        else:
            html_lines.append(f"<p>{_inline_md(stripped)}</p>")

    if in_blockquote:
        html_lines.append("</blockquote>")
    if in_list:
        html_lines.append("</ul>")
    if in_table:
        html_lines.append("</tbody></table>")

    return "\n".join(html_lines)


def _inline_md(text: str) -> str:
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`(.+?)`", r"<code>\1</code>", text)
    return text
