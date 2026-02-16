import numpy as np
import torch
import gc
import time
import logging

from ade20k_classes import ADE20K_NAMES

logger = logging.getLogger(__name__)


class ReportMixin:
    """Cross-method XAI analysis report generation."""

    def generate_xai_report(self, image, seg_mask=None, method_results=None):
        t0 = time.perf_counter()

        if seg_mask is None:
            inputs, oh, ow = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_mask = self._seg_from_outputs(outputs, oh, ow)
            del outputs
            gc.collect()

        oh, ow = seg_mask.shape[:2]
        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        info = sorted(zip(cls, cnt), key=lambda x: -x[1])

        ent_result = None
        emap = None
        me, hp = 0, 0
        if method_results and "entropy" in method_results:
            ent_result = method_results["entropy"]
            emap = ent_result.get("entropy_map")
        if emap is None:
            ent_result = self.predictive_entropy(image)
            emap = ent_result.get("entropy_map")

        if emap is not None:
            me = float(emap.mean())
            hp = float((emap > 0.5).mean() * 100)

        boundary_info = []
        if emap is not None:
            try:
                from skimage.segmentation import find_boundaries

                for cid, _ in info[:8]:
                    bd = find_boundaries(seg_mask == cid, mode="thick")
                    if bd.any():
                        be = float(emap[bd].mean())
                        interior = (seg_mask == cid) & ~bd
                        ie = float(emap[interior].mean()) if interior.any() else 0
                        boundary_info.append({
                            "name": self._cname(cid),
                            "id": int(cid),
                            "boundary_entropy": be,
                            "interior_entropy": ie,
                            "area_pct": float(cnt[list(cls).index(cid)] / total * 100),
                            "confidence_ratio": ie / (be + 1e-8),
                        })
            except ImportError:
                pass

        lines = ["# Comprehensive XAI Analysis Report\n"]

        # Scene composition
        lines += [
            "## 1. Scene Composition & Object Distribution",
            f"The model identified **{len(cls)} semantic categories** in this scene, "
            f"processed through {self._n_encoder} encoder layers and "
            f"{self._num_layers - self._n_encoder} decoder layers.\n",
        ]
        for cid, c in info[:10]:
            p = c / total * 100
            bar = "\u2588" * int(p / 4) + "\u2591" * max(0, 25 - int(p / 4))
            lines.append(f"- **{self._cname(cid)}** (ID {cid}): {p:.1f}% `{bar}`")
        if len(info) > 10:
            rest_pct = sum(c for _, c in info[10:]) / total * 100
            lines.append(f"- *{len(info)-10} more categories*: {rest_pct:.1f}% combined")

        # Confidence
        lines += [
            "\n## 2. Prediction Confidence Analysis",
            f"- **Mean entropy**: {me:.3f} (range 0-1, lower = more confident)",
            f"- **High-uncertainty pixels**: {hp:.1f}% of image",
        ]
        if me < 0.10:
            lines.append("- Assessment: **Very high confidence** — model strongly recognizes all elements")
        elif me < 0.20:
            lines.append("- Assessment: **High confidence** — clear boundaries with minor ambiguity")
        elif me < 0.35:
            lines.append("- Assessment: **Moderate confidence** — some boundaries cause uncertainty")
        else:
            lines.append("- Assessment: **Low confidence** — significant uncertain regions")

        # Boundaries
        if boundary_info:
            lines += ["\n## 3. Boundary Clarity & Transition Zones"]
            uncertain = [b for b in boundary_info if b["boundary_entropy"] > 0.25]
            clear = [b for b in boundary_info if b["boundary_entropy"] <= 0.25]

            if uncertain:
                lines.append("\n**Challenging boundaries** (high entropy at transitions):")
                for b in sorted(uncertain, key=lambda x: -x["boundary_entropy"])[:5]:
                    lines.append(
                        f"- **{b['name']}** boundaries: entropy={b['boundary_entropy']:.3f} "
                        f"(interior={b['interior_entropy']:.3f}, ratio={b['confidence_ratio']:.2f})"
                    )
                lines.append(
                    "\n> *High boundary entropy suggests the model struggles to delineate these objects "
                    "— a pattern also observed in human visual perception for Alzheimer's patients.*"
                )
            if clear:
                lines.append("\n**Well-defined boundaries** (model is confident):")
                for b in sorted(clear, key=lambda x: x["boundary_entropy"])[:3]:
                    lines.append(
                        f"- **{b['name']}**: boundary entropy={b['boundary_entropy']:.3f} "
                        f"(strong semantic contrast)"
                    )

        # Cross-method insights
        lines += ["\n## 4. Cross-Method Interpretation"]

        has = {
            k: method_results
            and k in method_results
            and "Error" not in method_results[k].get("report", "")
            for k in ["attention", "rollout", "gradcam", "integrated_gradients", "chefer", "pca"]
        }

        if has["attention"] and has["rollout"]:
            lines += [
                "\n### Attention Analysis (Self-Attention + Rollout)",
                "- **Self-Attention** shows single-layer focus — what features are processed at a depth",
                "- **Attention Rollout** aggregates across all encoder layers, revealing overall spatial priority",
                "- If rollout focuses on centers while self-attention spreads on edges, "
                "the model has learned strong object-level representations",
            ]

        if has["gradcam"] and has["integrated_gradients"]:
            lines += [
                "\n### Gradient Attribution (GradCAM + Integrated Gradients)",
                "- **GradCAM** highlights internal feature map regions (coarse resolution)",
                "- **Integrated Gradients** attributes importance to individual input pixels (full resolution)",
                "- Agreement indicates robust localization; disagreement reveals contextual cues",
            ]
        elif has["integrated_gradients"]:
            lines += [
                "\n### Gradient Attribution (Integrated Gradients)",
                "- Attributions satisfying the completeness axiom",
                "- Bright regions show pixels most influencing the target class",
            ]

        if has["chefer"]:
            lines += [
                "\n### Transformer-Specific Attribution (Chefer Relevancy)",
                "- Combines attention with gradient flow — most grounded method for ViTs",
                "- Green overlay = model-relevant regions",
            ]

        if has["pca"]:
            lines += [
                "\n### Learned Feature Structure (Feature PCA)",
                "- Projects 1024-dim hidden states to 3 principal components (RGB)",
                "- Similar colors = similar learned representations",
            ]

        # Accessibility implications
        lines += ["\n## 5. Implications for Accessibility Analysis"]
        if boundary_info:
            floor_related = [
                b for b in boundary_info
                if any(f in b["name"].lower() for f in ["floor", "carpet", "rug", "mat"])
            ]
            if floor_related:
                lines.append("\n**Floor boundary analysis:**")
                for b in floor_related:
                    if b["boundary_entropy"] > 0.3:
                        lines.append(
                            f"- **{b['name']}** has uncertain boundaries (entropy={b['boundary_entropy']:.3f}) — "
                            "suggests low contrast that may challenge Alzheimer's patients"
                        )
                    else:
                        lines.append(
                            f"- **{b['name']}** has clear boundaries (entropy={b['boundary_entropy']:.3f})"
                        )

        lines += [
            "\n> *Regions where the model shows high uncertainty or diffuse attention "
            "correlate with areas presenting visual navigation challenges for individuals "
            "with Alzheimer's-related perceptual deficits.*",
        ]

        # Method reference
        lines += [
            "\n## 6. Method Reference",
            "| Method | Type | Key Property | Reference |",
            "|--------|------|-------------|-----------|",
            "| Self-Attention | Attention | Single-layer spatial focus | Vaswani et al. (2017) |",
            "| Attention Rollout | Attention | Multi-layer cumulative focus | Abnar & Zuidema (2020) |",
            "| GradCAM | Gradient | Class-discriminative localization | Selvaraju et al. (2017) |",
            "| Predictive Entropy | Output | Per-pixel classification uncertainty | Shannon (1948) |",
            "| Feature PCA | Hidden State | Learned representation structure | — |",
            "| Integrated Gradients | Gradient | Axiom-satisfying pixel attribution | Sundararajan et al. (2017) |",
            "| Chefer Relevancy | Attn x Grad | ViT-specific relevance propagation | Chefer et al. (2021) |",
            f"\n**Architecture**: DINOv3-EoMT-Large — {self._num_layers} layers "
            f"({self._n_encoder} encoder + {self._num_layers - self._n_encoder} decoder), "
            f"{self._num_heads} heads, {self._patch_size}px patches, 150 ADE20K classes",
        ]

        vis = ent_result["visualization"] if ent_result else None
        return {"visualization": vis, "report": "\n".join(lines)}
