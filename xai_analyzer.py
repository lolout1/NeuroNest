"""
Explainable AI (XAI) analyzer for NeuroNest.

Provides 7 visualization methods for the EoMT-DINOv3 semantic segmentation model:
1. Self-Attention Maps — raw attention from specific layers/heads
2. Attention Rollout — aggregated attention across all layers
3. GradCAM — class-specific activation maps
4. Predictive Entropy — per-pixel uncertainty
5. Feature PCA — hidden state structure visualization
6. Class Saliency — gradient-based input importance
7. Chefer Relevancy — attention x gradient propagation

All methods are CPU-compatible and memory-managed for HF Spaces free tier.
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import logging
import time
from contextlib import contextmanager
from typing import Dict, Optional, Tuple
from PIL import Image

from ade20k_classes import ADE20K_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hook utilities
# ---------------------------------------------------------------------------

class AttentionCaptureHook:
    """Forward hook to capture attention weights from EomtAttention modules.

    The EoMT model uses SDPA by default which does NOT return attention weights.
    Must switch to eager attention (via `_force_eager_attention`) before forward
    pass for this hook to receive non-None values.
    """

    def __init__(self):
        self.attention_weights = None
        self._handle = None

    def hook_fn(self, module, args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.attention_weights = output[1].detach().cpu()

    def register(self, module):
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.attention_weights = None


class HiddenStateCaptureHook:
    """Forward hook to capture hidden states from transformer layers."""

    def __init__(self):
        self.hidden_state = None
        self._handle = None

    def hook_fn(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        self.hidden_state = out.detach().cpu()

    def register(self, module):
        self._handle = module.register_forward_hook(self.hook_fn)
        return self

    def remove(self):
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def clear(self):
        self.hidden_state = None


class ActivationGradientHook:
    """Captures both forward activations and backward gradients from a module."""

    def __init__(self):
        self.activation = None
        self.gradient = None
        self._fwd_handle = None
        self._bwd_handle = None

    def _fwd(self, module, inp, out):
        self.activation = out[0] if isinstance(out, tuple) else out

    def _bwd(self, module, grad_input, grad_output):
        self.gradient = grad_output[0]

    def register(self, module):
        self._fwd_handle = module.register_forward_hook(self._fwd)
        self._bwd_handle = module.register_full_backward_hook(self._bwd)
        return self

    def remove(self):
        if self._fwd_handle is not None:
            self._fwd_handle.remove()
        if self._bwd_handle is not None:
            self._bwd_handle.remove()
        self._fwd_handle = None
        self._bwd_handle = None

    def clear(self):
        self.activation = None
        self.gradient = None


# ---------------------------------------------------------------------------
# Context manager: force eager attention
# ---------------------------------------------------------------------------

@contextmanager
def _force_eager_attention(model):
    """Temporarily switch EoMT to eager attention so attention weights are returned.

    SDPA (scaled dot-product attention) is faster but returns None for attn_weights.
    Eager attention computes weights explicitly and returns them from the hook.
    """
    original = getattr(model.config, "_attn_implementation", None)
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        if original is not None:
            model.config._attn_implementation = original
        else:
            model.config._attn_implementation = "sdpa"


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

class XAIAnalyzer:
    """Explainable AI analysis for EoMT-DINOv3 segmentation model.

    All methods operate on CPU and manage memory for HF Spaces (16 GB RAM).
    Attention-based methods temporarily switch to eager attention mode.
    Gradient-based methods lazy-load a separate FP32 model copy.
    """

    def __init__(self, eomt_model, eomt_processor, blackspot_predictor=None):
        self.model = eomt_model
        self.processor = eomt_processor
        self.blackspot_predictor = blackspot_predictor
        self._model_fp32 = None

        # Probe architecture
        self._num_layers = len(self.model.layers)
        self._num_prefix_tokens = getattr(
            getattr(self.model, "embeddings", None), "num_prefix_tokens", 5
        )
        self._num_heads = getattr(self.model.config, "num_attention_heads", 16)
        self._attn_impl = getattr(self.model.config, "_attn_implementation", "unknown")

        logger.info(
            f"[XAI] Initialized: {self._num_layers} layers, {self._num_heads} heads, "
            f"{self._num_prefix_tokens} prefix tokens, attn_impl={self._attn_impl}"
        )

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray) -> Tuple[dict, int, int]:
        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        return inputs, h, w

    def _patch_grid(self, inputs: dict) -> Tuple[int, int]:
        _, _, h, w = inputs["pixel_values"].shape
        return h // 14, w // 14  # DINOv2 patch_size=14

    def _get_dominant_class(self, seg_mask: np.ndarray) -> int:
        classes, counts = np.unique(seg_mask, return_counts=True)
        return int(classes[np.argmax(counts)])

    def _class_name(self, class_id: int) -> str:
        if 0 <= class_id < len(ADE20K_NAMES):
            return ADE20K_NAMES[class_id].split(",")[0].strip()
        return f"class_{class_id}"

    def _segmentation_from_outputs(self, outputs, h: int, w: int) -> np.ndarray:
        seg_maps = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )
        return seg_maps[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _blend_heatmap(
        heatmap: np.ndarray, image: np.ndarray,
        colormap: int = cv2.COLORMAP_INFERNO, alpha: float = 0.5,
    ) -> np.ndarray:
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = heatmap.min(), heatmap.max()
        norm = (heatmap - lo) / (hi - lo + 1e-8)
        colored = cv2.applyColorMap((norm * 255).astype(np.uint8), colormap)
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        return cv2.addWeighted(image, 1 - alpha, colored_rgb, alpha, 0)

    @staticmethod
    def _add_title(image: np.ndarray, title: str) -> np.ndarray:
        h, w = image.shape[:2]
        bar = np.full((40, w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, title, (10, 28), cv2.FONT_HERSHEY_SIMPLEX,
                     0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, image])

    # ------------------------------------------------------------------
    # 1. Self-Attention Maps
    # ------------------------------------------------------------------

    def self_attention_maps(
        self, image: np.ndarray, layer: int = -1, head: Optional[int] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        idx = layer if layer >= 0 else self._num_layers + layer
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = AttentionCaptureHook()
        hook.register(self.model.layers[idx].attention)
        try:
            with _force_eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.attention_weights is None:
            return self._fallback(image, "Self-attention weights not available")

        attn = hook.attention_weights[0]  # (heads, seq, seq)
        if head is not None and 0 <= head < attn.shape[0]:
            attn_map = attn[head]
            head_label = f"Head {head}"
        else:
            attn_map = attn.mean(dim=0)
            head_label = "Mean"

        spatial = self._extract_patch_map(attn_map, gh, gw)
        vis = self._blend_heatmap(spatial, image)
        vis = self._add_title(vis, f"Self-Attention | Layer {idx} | {head_label}")

        del hook.attention_weights
        gc.collect()
        return {
            "visualization": vis,
            "report": f"Self-attention layer {idx}, {head_label}. Grid {gh}x{gw}. "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # 2. Attention Rollout
    # ------------------------------------------------------------------

    def attention_rollout(
        self, image: np.ndarray, head_fusion: str = "mean", discard_ratio: float = 0.1,
    ) -> Dict:
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hooks = [AttentionCaptureHook() for _ in range(self._num_layers)]
        for i, h in enumerate(hooks):
            h.register(self.model.layers[i].attention)

        try:
            with _force_eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        rollout = None
        captured = 0
        for h in hooks:
            if h.attention_weights is None:
                continue
            captured += 1
            attn = h.attention_weights[0]
            if head_fusion == "max":
                fused = attn.max(dim=0).values
            elif head_fusion == "min":
                fused = attn.min(dim=0).values
            else:
                fused = attn.mean(dim=0)

            fused = fused + torch.eye(fused.shape[0])
            fused = fused / fused.sum(dim=-1, keepdim=True)
            rollout = fused if rollout is None else rollout @ fused
            h.attention_weights = None

        del hooks
        gc.collect()

        if rollout is None:
            return self._fallback(image, "Attention rollout failed (no layers captured)")

        spatial = self._extract_patch_map(rollout, gh, gw)
        if discard_ratio > 0:
            thr = np.percentile(spatial, discard_ratio * 100)
            spatial = np.where(spatial > thr, spatial, 0)

        vis = self._blend_heatmap(spatial, image)
        vis = self._add_title(vis, f"Attention Rollout | {captured} Layers")

        del rollout
        gc.collect()
        return {
            "visualization": vis,
            "report": f"Attention rollout across {captured} layers ({head_fusion} fusion). "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # 3. Predictive Entropy
    # ------------------------------------------------------------------

    def predictive_entropy(self, image: np.ndarray) -> Dict:
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(**inputs)

        mask_probs = torch.sigmoid(outputs.masks_queries_logits[0])  # (Q, H, W)
        class_probs = F.softmax(outputs.class_queries_logits[0][:, :-1], dim=-1)  # (Q, C)
        num_classes = class_probs.shape[-1]

        pixel_probs = torch.einsum("qhw,qc->chw", mask_probs, class_probs)
        pixel_probs = pixel_probs / (pixel_probs.sum(dim=0, keepdim=True) + 1e-10)
        entropy = -(pixel_probs * torch.log(pixel_probs + 1e-10)).sum(dim=0)
        entropy_norm = (entropy / np.log(num_classes)).numpy()
        entropy_map = cv2.resize(entropy_norm, (ow, oh), interpolation=cv2.INTER_CUBIC)

        vis = self._blend_heatmap(entropy_map, image, cv2.COLORMAP_MAGMA, 0.55)
        vis = self._add_title(vis, "Predictive Entropy (Uncertainty)")

        mean_e = float(entropy_map.mean())
        high_pct = float((entropy_map > 0.5).mean() * 100)

        del outputs, mask_probs, class_probs, pixel_probs, entropy
        gc.collect()
        return {
            "visualization": vis,
            "entropy_map": entropy_map,
            "report": f"Entropy — mean: {mean_e:.3f}, high-uncertainty: {high_pct:.1f}%. "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # 4. Feature PCA
    # ------------------------------------------------------------------

    def feature_pca(self, image: np.ndarray, layer: int = -1) -> Dict:
        t0 = time.perf_counter()
        idx = layer if layer >= 0 else self._num_layers + layer
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = HiddenStateCaptureHook()
        hook.register(self.model.layers[idx])
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.hidden_state is None:
            return self._fallback(image, "Failed to capture hidden states")

        hidden = hook.hidden_state[0].numpy()
        n_pre = self._num_prefix_tokens
        n_pat = gh * gw
        feats = hidden[n_pre:n_pre + n_pat]

        centered = feats - feats.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        comps = U[:, :3] * S[:3]

        for c in range(3):
            lo, hi = comps[:, c].min(), comps[:, c].max()
            comps[:, c] = (comps[:, c] - lo) / (hi - lo + 1e-8) * 255

        pca_img = cv2.resize(
            comps.reshape(gh, gw, 3).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_CUBIC
        )
        vis = self._add_title(pca_img, f"Feature PCA | Layer {idx}")

        total_var = (S ** 2).sum()
        var3 = (S[:3] ** 2) / total_var * 100

        del hook.hidden_state
        gc.collect()
        return {
            "visualization": vis,
            "report": f"Feature PCA layer {idx}. Variance: PC1={var3[0]:.1f}%, "
                      f"PC2={var3[1]:.1f}%, PC3={var3[2]:.1f}%. "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # FP32 model management (for gradient methods)
    # ------------------------------------------------------------------

    def _get_fp32_model(self):
        if self._model_fp32 is not None:
            return self._model_fp32

        logger.info("[XAI] Loading FP32 model for gradient-based methods...")
        from transformers import AutoModelForUniversalSegmentation
        model_id = getattr(self.model.config, "_name_or_path",
                           "tue-mps/ade20k_semantic_eomt_large_512")
        self._model_fp32 = AutoModelForUniversalSegmentation.from_pretrained(model_id)
        self._model_fp32.eval()
        # Force eager attention so hooks can capture weights
        self._model_fp32.config._attn_implementation = "eager"
        logger.info("[XAI] FP32 model loaded (eager attention)")
        return self._model_fp32

    def cleanup_fp32(self):
        if self._model_fp32 is not None:
            del self._model_fp32
            self._model_fp32 = None
            gc.collect()
            logger.info("[XAI] FP32 model released")

    # ------------------------------------------------------------------
    # 5. GradCAM
    # ------------------------------------------------------------------

    def gradcam_segmentation(
        self, image: np.ndarray, target_class_id: Optional[int] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        fp32 = self._get_fp32_model()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = ActivationGradientHook()
        hook.register(fp32.layers[-1].mlp)

        try:
            outputs = fp32(**inputs)
            seg_mask = self._segmentation_from_outputs(outputs, oh, ow)

            if target_class_id is None:
                target_class_id = self._get_dominant_class(seg_mask)
            cname = self._class_name(target_class_id)

            # Backprop from target class mask logits
            mask_logits = outputs.masks_queries_logits[0]
            class_logits = outputs.class_queries_logits[0]
            preds = class_logits[:, :-1].argmax(dim=-1)
            queries = (preds == target_class_id).nonzero(as_tuple=True)[0]
            if len(queries) == 0:
                queries = class_logits[:, target_class_id].argmax(dim=0, keepdim=True)

            mask_logits[queries].sum().backward()

            if hook.activation is not None and hook.gradient is not None:
                weights = hook.gradient.mean(dim=-1, keepdim=True)
                cam = F.relu((hook.activation * weights).sum(dim=-1))
                spatial = cam[0, self._num_prefix_tokens:
                              self._num_prefix_tokens + gh * gw].detach().cpu().numpy()
                spatial = spatial.reshape(gh, gw)
                vis = self._blend_heatmap(spatial, image, cv2.COLORMAP_JET)
            else:
                vis = image.copy()
                cname = "N/A"
        finally:
            hook.remove()
            fp32.zero_grad()

        vis = self._add_title(vis, f"GradCAM | {cname}")
        del hook
        gc.collect()
        return {
            "visualization": vis,
            "target_class": target_class_id,
            "report": f"GradCAM for '{cname}' (ID {target_class_id}). "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # 6. Class Saliency
    # ------------------------------------------------------------------

    def class_saliency(
        self, image: np.ndarray, target_class_id: Optional[int] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        fp32 = self._get_fp32_model()
        inputs, oh, ow = self._preprocess(image)
        pv = inputs["pixel_values"].requires_grad_(True)

        outputs = fp32(pixel_values=pv)
        seg_mask = self._segmentation_from_outputs(outputs, oh, ow)

        if target_class_id is None:
            target_class_id = self._get_dominant_class(seg_mask)
        cname = self._class_name(target_class_id)

        outputs.class_queries_logits[0][:, target_class_id].sum().backward()

        saliency = pv.grad[0].abs().max(dim=0).values.detach().cpu().numpy()
        saliency = cv2.resize(saliency, (ow, oh), interpolation=cv2.INTER_CUBIC)

        vis = self._blend_heatmap(saliency, image, cv2.COLORMAP_HOT)
        vis = self._add_title(vis, f"Saliency | {cname}")

        fp32.zero_grad()
        gc.collect()
        return {
            "visualization": vis,
            "report": f"Class saliency for '{cname}' (ID {target_class_id}). "
                      f"Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # 7. Chefer Relevancy
    # ------------------------------------------------------------------

    def chefer_relevancy(
        self, image: np.ndarray, target_class_id: Optional[int] = None,
    ) -> Dict:
        t0 = time.perf_counter()
        fp32 = self._get_fp32_model()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        # Hook every attention layer for both attn weights and gradients
        layer_data = []
        handles = []
        for i in range(self._num_layers):
            data = {"attn": None, "grad": None}

            def mk_fwd(d):
                def fn(mod, args, out):
                    if isinstance(out, tuple) and len(out) >= 2 and out[1] is not None:
                        d["attn"] = out[1]
                return fn

            def mk_bwd(d):
                def fn(mod, gi, go):
                    if len(go) >= 2 and go[1] is not None:
                        d["grad"] = go[1]
                return fn

            attn_mod = fp32.layers[i].attention
            handles.append(attn_mod.register_forward_hook(mk_fwd(data)))
            handles.append(attn_mod.register_full_backward_hook(mk_bwd(data)))
            layer_data.append(data)

        try:
            outputs = fp32(**inputs)
            seg_mask = self._segmentation_from_outputs(outputs, oh, ow)

            if target_class_id is None:
                target_class_id = self._get_dominant_class(seg_mask)
            cname = self._class_name(target_class_id)

            outputs.class_queries_logits[0][:, target_class_id].sum().backward()

            n_pre = self._num_prefix_tokens
            n_pat = gh * gw
            seq = n_pre + n_pat
            R = torch.eye(seq)

            for data in layer_data:
                attn = data.get("attn")
                if attn is None:
                    continue
                attn_mean = attn[0].detach().cpu().mean(dim=0)
                grad = data.get("grad")
                if grad is not None:
                    grad_mean = grad[0].detach().cpu().mean(dim=0)
                    rel = torch.clamp(attn_mean * grad_mean, min=0)
                else:
                    rel = attn_mean

                s = min(rel.shape[0], seq)
                rel = rel[:s, :s] + torch.eye(s)
                rel = rel / (rel.sum(dim=-1, keepdim=True) + 1e-10)
                R[:s, :s] = R[:s, :s] + R[:s, :s] @ rel

            spatial = R[0, n_pre:n_pre + n_pat].numpy().reshape(gh, gw)
            vis = self._blend_heatmap(spatial, image)

            # Threshold overlay
            thr = spatial.mean()
            mask = cv2.resize((spatial > thr).astype(np.float32), (ow, oh),
                              interpolation=cv2.INTER_NEAREST)
            overlay = vis.copy()
            overlay[mask > 0.5] = (overlay[mask > 0.5] * 0.7 +
                                   np.array([0, 80, 0]) * 0.3).astype(np.uint8)
            vis = overlay

        finally:
            for h in handles:
                h.remove()
            fp32.zero_grad()

        vis = self._add_title(vis, f"Chefer Relevancy | {cname}")
        del layer_data, handles
        gc.collect()
        return {
            "visualization": vis,
            "report": f"Chefer relevancy for '{cname}' (ID {target_class_id}), "
                      f"{self._num_layers} layers. Time: {time.perf_counter()-t0:.1f}s",
        }

    # ------------------------------------------------------------------
    # Report generator
    # ------------------------------------------------------------------

    def generate_xai_report(
        self, image: np.ndarray, seg_mask: Optional[np.ndarray] = None,
    ) -> Dict:
        t0 = time.perf_counter()

        entropy_result = self.predictive_entropy(image)

        if seg_mask is None:
            inputs, oh, ow = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_mask = self._segmentation_from_outputs(outputs, oh, ow)
            del outputs
            gc.collect()

        classes, counts = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        class_info = sorted(zip(classes, counts), key=lambda x: -x[1])

        entropy_map = entropy_result.get("entropy_map")
        mean_e = float(entropy_map.mean()) if entropy_map is not None else 0
        high_pct = float((entropy_map > 0.5).mean() * 100) if entropy_map is not None else 0

        uncertain_boundaries = []
        if entropy_map is not None:
            from skimage.segmentation import find_boundaries
            for cid, cnt in class_info[:5]:
                boundary = find_boundaries(seg_mask == cid, mode="thick")
                if boundary.any():
                    be = float(entropy_map[boundary].mean())
                    if be > 0.3:
                        uncertain_boundaries.append((self._class_name(cid), be))

        lines = [
            "# Explainable AI Analysis Report\n",
            f"*Generated in {time.perf_counter()-t0:.1f}s*\n",
            "## Scene Composition",
            f"The model identified **{len(classes)} object categories**:\n",
        ]
        for cid, cnt in class_info[:8]:
            pct = cnt / total * 100
            bar = "\u2588" * int(pct / 5) + "\u2591" * (20 - int(pct / 5))
            lines.append(f"- **{self._class_name(cid)}**: {pct:.1f}% `{bar}`")

        lines += [
            "\n## Model Confidence",
            f"- **Mean uncertainty**: {mean_e:.3f} (0=certain, 1=uncertain)",
            f"- **High uncertainty regions**: {high_pct:.1f}% of image",
        ]
        if mean_e < 0.15:
            lines.append("- Model is **highly confident** in its predictions")
        elif mean_e < 0.35:
            lines.append("- Model shows **moderate confidence** overall")
        else:
            lines.append("- Model shows **significant uncertainty** — review predictions carefully")

        if uncertain_boundaries:
            lines.append("\n### Uncertain Boundaries:")
            for name, ent in sorted(uncertain_boundaries, key=lambda x: -x[1])[:5]:
                lines.append(f"- {name} boundary: entropy={ent:.3f}")

        lines += [
            "\n## Architecture",
            f"- {self._num_layers}-layer, {self._num_heads}-head Vision Transformer (DINOv3-EoMT-Large)",
            f"- 150-class ADE20K semantic segmentation at 512x512 resolution",
            "\n## Methodology",
            "- **Entropy**: Shannon entropy of per-pixel class probability distributions",
            "- **Attention Rollout**: Abnar & Zuidema (2020)",
            "- **GradCAM**: Selvaraju et al. (2017)",
            "- **Chefer Relevancy**: Chefer et al. (2021)",
            "- **Feature PCA**: Principal component analysis of intermediate representations",
            "- **Saliency**: Simonyan et al. (2014)",
        ]

        return {
            "visualization": entropy_result["visualization"],
            "report": "\n".join(lines),
        }

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_full_analysis(
        self, image: np.ndarray,
        layer: int = -1, head: Optional[int] = None,
        target_class_id: Optional[int] = None,
    ) -> Dict:
        logger.info("[XAI] Running full analysis suite...")
        t0 = time.perf_counter()
        results = {}

        # Phase 1: no-gradient methods (quantized model OK)
        logger.info("[XAI] Phase 1: attention + entropy + PCA...")
        for key, fn, kwargs in [
            ("attention", self.self_attention_maps, {"layer": layer, "head": head}),
            ("rollout",   self.attention_rollout,   {}),
            ("entropy",   self.predictive_entropy,  {}),
            ("pca",       self.feature_pca,         {"layer": layer}),
        ]:
            try:
                results[key] = fn(image, **kwargs)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}")
                results[key] = self._fallback(image, str(e))
        gc.collect()

        # Phase 2: gradient methods (FP32 model)
        logger.info("[XAI] Phase 2: gradient-based methods...")
        for key, fn, kwargs in [
            ("gradcam",  self.gradcam_segmentation, {"target_class_id": target_class_id}),
            ("saliency", self.class_saliency,       {"target_class_id": target_class_id}),
            ("chefer",   self.chefer_relevancy,      {"target_class_id": target_class_id}),
        ]:
            try:
                results[key] = fn(image, **kwargs)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}")
                results[key] = self._fallback(image, str(e))
        gc.collect()

        # Phase 3: report
        logger.info("[XAI] Phase 3: generating report...")
        try:
            results["report"] = self.generate_xai_report(image)
        except Exception as e:
            logger.error(f"[XAI] Report failed: {e}")
            results["report"] = self._fallback(image, str(e))

        elapsed = time.perf_counter() - t0
        logger.info(f"[XAI] Full analysis completed in {elapsed:.1f}s")
        return results

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _extract_patch_map(self, attn_matrix: torch.Tensor, gh: int, gw: int) -> np.ndarray:
        """Extract CLS-to-patch attention from row 0, reshape to spatial grid."""
        n_pre = self._num_prefix_tokens
        n_pat = gh * gw
        row = attn_matrix[0, n_pre:n_pre + n_pat]
        if isinstance(row, torch.Tensor):
            row = row.numpy()
        return row.reshape(gh, gw)

    @staticmethod
    def _fallback(image: np.ndarray, message: str) -> Dict:
        return {"visualization": image, "report": f"Error: {message}"}
