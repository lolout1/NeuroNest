"""
Explainable AI (XAI) analyzer for NeuroNest.

7 visualization methods for EoMT-DINOv3 semantic segmentation:
1. Self-Attention Maps   2. Attention Rollout   3. GradCAM
4. Predictive Entropy    5. Feature PCA         6. Class Saliency
7. Chefer Relevancy

CPU-compatible. Memory-managed for HF Spaces (16 GB).
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
# Hook classes
# ---------------------------------------------------------------------------

class AttentionCaptureHook:
    """Captures attention weights from EomtAttention.forward() output[1]."""
    def __init__(self):
        self.weights = None
        self._h = None

    def _fn(self, module, args, output):
        if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
            self.weights = output[1].detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class HiddenStateCaptureHook:
    """Captures hidden states (first element of layer output)."""
    def __init__(self):
        self.state = None
        self._h = None

    def _fn(self, module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        self.state = out.detach().cpu()

    def register(self, module):
        self._h = module.register_forward_hook(self._fn)
        return self

    def remove(self):
        if self._h:
            self._h.remove()
            self._h = None


class ActivationGradientHook:
    """Captures forward activations AND backward gradients."""
    def __init__(self):
        self.activation = None
        self.gradient = None
        self._fh = None
        self._bh = None

    def register(self, module):
        self._fh = module.register_forward_hook(
            lambda m, i, o: setattr(self, 'activation', o[0] if isinstance(o, tuple) else o)
        )
        self._bh = module.register_full_backward_hook(
            lambda m, gi, go: setattr(self, 'gradient', go[0])
        )
        return self

    def remove(self):
        if self._fh:
            self._fh.remove()
        if self._bh:
            self._bh.remove()
        self._fh = self._bh = None


# ---------------------------------------------------------------------------
# Context manager: force eager attention
# ---------------------------------------------------------------------------

@contextmanager
def _eager_attention(model):
    """Temporarily switch to eager attention so attention weights are returned.
    SDPA is default (fast) but returns None for attn_weights."""
    orig = getattr(model.config, "_attn_implementation", "sdpa")
    model.config._attn_implementation = "eager"
    try:
        yield
    finally:
        model.config._attn_implementation = orig


# ---------------------------------------------------------------------------
# XAI Analyzer
# ---------------------------------------------------------------------------

class XAIAnalyzer:
    """All methods CPU-safe, memory-managed, dynamically adapt to model config."""

    def __init__(self, eomt_model, eomt_processor, blackspot_predictor=None):
        self.model = eomt_model
        self.processor = eomt_processor
        self._model_fp32 = None

        # --- Probe architecture dynamically ---
        cfg = self.model.config
        self._num_layers = len(self.model.layers)
        self._num_heads = getattr(cfg, "num_attention_heads", 16)
        self._patch_size = getattr(cfg, "patch_size", 16)
        self._num_queries = getattr(cfg, "num_queries", 100)
        self._num_prefix = getattr(
            getattr(self.model, "embeddings", None), "num_prefix_tokens", 5
        )

        # Encoder/decoder split
        n_dec = getattr(cfg, "num_decoder_layers", getattr(cfg, "decoder_layers", 4))
        self._n_encoder = self._num_layers - n_dec

        logger.info(
            f"[XAI] {self._num_layers} layers ({self._n_encoder}enc+{n_dec}dec), "
            f"{self._num_heads} heads, patch={self._patch_size}, "
            f"prefix={self._num_prefix}, queries={self._num_queries}, "
            f"attn={getattr(cfg, '_attn_implementation', '?')}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image: np.ndarray):
        h, w = image.shape[:2]
        inputs = self.processor(images=Image.fromarray(image), return_tensors="pt")
        return inputs, h, w

    def _patch_grid(self, inputs: dict) -> Tuple[int, int]:
        _, _, h, w = inputs["pixel_values"].shape
        return h // self._patch_size, w // self._patch_size

    def _infer_grid(self, seq_len: int, is_decoder: bool = False) -> Tuple[int, int]:
        """Infer (gh, gw) from attention sequence length."""
        n = seq_len - self._num_prefix
        if is_decoder:
            n -= self._num_queries
        n = max(n, 1)
        side = int(round(np.sqrt(n)))
        return side, side

    def _cls_to_patches(self, attn_2d, gh: int, gw: int, is_decoder: bool = False):
        """Extract CLS→patch attention row, reshape to spatial. Returns (map, gh, gw)."""
        seq = attn_2d.shape[-1]
        n_pre = self._num_prefix
        n_expected = gh * gw
        n_avail = seq - n_pre - (self._num_queries if is_decoder else 0)

        if n_avail != n_expected:
            gh, gw = self._infer_grid(seq, is_decoder)
            n_expected = gh * gw
            logger.debug(f"[XAI] Grid inferred: {gh}x{gw} from seq_len={seq}")

        row = attn_2d[0, n_pre:n_pre + n_expected]
        if isinstance(row, torch.Tensor):
            row = row.numpy()
        return row.reshape(gh, gw), gh, gw

    def _dominant_class(self, seg_mask):
        cls, cnt = np.unique(seg_mask, return_counts=True)
        return int(cls[np.argmax(cnt)])

    def _cname(self, cid):
        return ADE20K_NAMES[cid].split(",")[0].strip() if 0 <= cid < len(ADE20K_NAMES) else f"class_{cid}"

    def _seg_from_outputs(self, outputs, h, w):
        return self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[(h, w)]
        )[0].cpu().numpy().astype(np.uint8)

    @staticmethod
    def _blend(heatmap, image, cmap=cv2.COLORMAP_INFERNO, alpha=0.5):
        h, w = image.shape[:2]
        if heatmap.shape[:2] != (h, w):
            heatmap = cv2.resize(heatmap, (w, h), interpolation=cv2.INTER_CUBIC)
        lo, hi = heatmap.min(), heatmap.max()
        norm = (heatmap - lo) / (hi - lo + 1e-8)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cmap)
        return cv2.addWeighted(image, 1 - alpha, cv2.cvtColor(cm, cv2.COLOR_BGR2RGB), alpha, 0)

    @staticmethod
    def _title(image, text):
        h, w = image.shape[:2]
        bar = np.full((40, w, 3), 30, dtype=np.uint8)
        cv2.putText(bar, text, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return np.vstack([bar, image])

    @staticmethod
    def _fallback(image, msg):
        return {"visualization": image, "report": f"Error: {msg}"}

    # ------------------------------------------------------------------
    # 1. Self-Attention Maps
    # ------------------------------------------------------------------

    def self_attention_maps(self, image, layer=-1, head=None):
        t0 = time.perf_counter()
        # Default to last ENCODER layer for clean patch-only attention
        idx = layer if layer >= 0 else self._n_encoder - 1
        idx = min(idx, self._num_layers - 1)
        is_dec = idx >= self._n_encoder

        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = AttentionCaptureHook().register(self.model.layers[idx].attention)
        try:
            with _eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.weights is None:
            return self._fallback(image, "Attention weights not captured")

        attn = hook.weights[0]
        if head is not None and 0 <= head < attn.shape[0]:
            a = attn[head]
            hl = f"Head {head}"
        else:
            a = attn.mean(dim=0)
            hl = "Mean"

        spatial, gh, gw = self._cls_to_patches(a, gh, gw, is_dec)
        vis = self._title(self._blend(spatial, image), f"Self-Attention | Layer {idx} | {hl}")

        del hook.weights
        gc.collect()
        return {"visualization": vis, "report": f"Self-attention L{idx} {hl}, grid {gh}x{gw}. {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # 2. Attention Rollout (encoder layers only)
    # ------------------------------------------------------------------

    def attention_rollout(self, image, head_fusion="mean", discard_ratio=0.1):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        # Only encoder layers — decoder layers have different seq_len due to queries
        n_enc = self._n_encoder
        hooks = [AttentionCaptureHook().register(self.model.layers[i].attention) for i in range(n_enc)]

        try:
            with _eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            for h in hooks:
                h.remove()

        rollout = None
        captured = 0
        for h in hooks:
            if h.weights is None:
                continue
            captured += 1
            attn = h.weights[0]
            if head_fusion == "max":
                fused = attn.max(dim=0).values
            elif head_fusion == "min":
                fused = attn.min(dim=0).values
            else:
                fused = attn.mean(dim=0)

            fused = fused + torch.eye(fused.shape[0])
            fused = fused / fused.sum(dim=-1, keepdim=True)

            if rollout is None:
                rollout = fused
            else:
                # Safety: ensure compatible dimensions
                if rollout.shape != fused.shape:
                    logger.warning(f"[XAI] Rollout shape mismatch: {rollout.shape} vs {fused.shape}, stopping")
                    break
                rollout = rollout @ fused
            h.weights = None

        del hooks
        gc.collect()

        if rollout is None:
            return self._fallback(image, f"Rollout failed (0/{n_enc} layers captured)")

        spatial, gh, gw = self._cls_to_patches(rollout, gh, gw, is_decoder=False)
        if discard_ratio > 0:
            thr = np.percentile(spatial, discard_ratio * 100)
            spatial = np.where(spatial > thr, spatial, 0)

        vis = self._title(self._blend(spatial, image), f"Attention Rollout | {captured} Encoder Layers")
        del rollout
        gc.collect()
        return {"visualization": vis, "report": f"Rollout {captured} encoder layers ({head_fusion}). {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # 3. Predictive Entropy
    # ------------------------------------------------------------------

    def predictive_entropy(self, image):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)

        with torch.no_grad():
            outputs = self.model(**inputs)

        mp = torch.sigmoid(outputs.masks_queries_logits[0])
        cp = F.softmax(outputs.class_queries_logits[0][:, :-1], dim=-1)
        nc = cp.shape[-1]

        pp = torch.einsum("qhw,qc->chw", mp, cp)
        pp = pp / (pp.sum(dim=0, keepdim=True) + 1e-10)
        ent = -(pp * torch.log(pp + 1e-10)).sum(dim=0)
        ent_norm = (ent / np.log(nc)).numpy()
        ent_map = cv2.resize(ent_norm, (ow, oh), interpolation=cv2.INTER_CUBIC)

        vis = self._title(self._blend(ent_map, image, cv2.COLORMAP_MAGMA, 0.55), "Predictive Entropy")
        me = float(ent_map.mean())
        hp = float((ent_map > 0.5).mean() * 100)

        del outputs, mp, cp, pp, ent
        gc.collect()
        return {"visualization": vis, "entropy_map": ent_map,
                "report": f"Entropy mean={me:.3f}, high-unc={hp:.1f}%. {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # 4. Feature PCA
    # ------------------------------------------------------------------

    def feature_pca(self, image, layer=-1):
        t0 = time.perf_counter()
        idx = layer if layer >= 0 else self._n_encoder - 1
        idx = min(idx, self._num_layers - 1)

        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = HiddenStateCaptureHook().register(self.model.layers[idx])
        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.state is None:
            return self._fallback(image, "Hidden state not captured")

        hidden = hook.state[0].numpy()  # (seq_len, hidden_dim)
        n_pre = self._num_prefix
        seq_len = hidden.shape[0]

        # Infer actual patch count from seq_len
        is_dec = idx >= self._n_encoder
        n_patches_avail = seq_len - n_pre - (self._num_queries if is_dec else 0)
        side = int(round(np.sqrt(max(n_patches_avail, 1))))
        n_use = side * side
        gh, gw = side, side

        feats = hidden[n_pre:n_pre + n_use]
        centered = feats - feats.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        comps = U[:, :3] * S[:3]

        for c in range(3):
            lo, hi = comps[:, c].min(), comps[:, c].max()
            comps[:, c] = (comps[:, c] - lo) / (hi - lo + 1e-8) * 255

        pca_img = cv2.resize(comps.reshape(gh, gw, 3).astype(np.uint8), (ow, oh), interpolation=cv2.INTER_CUBIC)
        vis = self._title(pca_img, f"Feature PCA | Layer {idx}")

        tv = (S ** 2).sum()
        v3 = (S[:3] ** 2) / tv * 100

        del hook.state
        gc.collect()
        return {"visualization": vis,
                "report": f"PCA L{idx}: PC1={v3[0]:.1f}% PC2={v3[1]:.1f}% PC3={v3[2]:.1f}%. {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # FP32 model for gradient methods
    # ------------------------------------------------------------------

    def _get_fp32(self):
        if self._model_fp32 is not None:
            return self._model_fp32
        logger.info("[XAI] Loading FP32 model for gradient methods...")
        from transformers import AutoModelForUniversalSegmentation
        mid = getattr(self.model.config, "_name_or_path", "tue-mps/ade20k_semantic_eomt_large_512")
        self._model_fp32 = AutoModelForUniversalSegmentation.from_pretrained(mid)
        self._model_fp32.eval()
        self._model_fp32.config._attn_implementation = "eager"
        logger.info("[XAI] FP32 loaded (eager attn)")
        return self._model_fp32

    def cleanup_fp32(self):
        if self._model_fp32 is not None:
            del self._model_fp32
            self._model_fp32 = None
            gc.collect()
            logger.info("[XAI] FP32 released")

    # ------------------------------------------------------------------
    # 5. GradCAM
    # ------------------------------------------------------------------

    def gradcam_segmentation(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        # Hook ENCODER layer (not decoder — avoids query token issues)
        target_layer_idx = self._n_encoder - 1
        hook = ActivationGradientHook().register(fp32.layers[target_layer_idx].mlp)

        try:
            outputs = fp32(**inputs)
            seg = self._seg_from_outputs(outputs, oh, ow)
            if target_class_id is None:
                target_class_id = self._dominant_class(seg)
            cn = self._cname(target_class_id)

            ml = outputs.masks_queries_logits[0]
            cl = outputs.class_queries_logits[0]
            preds = cl[:, :-1].argmax(dim=-1)
            qs = (preds == target_class_id).nonzero(as_tuple=True)[0]
            if len(qs) == 0:
                qs = cl[:, target_class_id].argmax(dim=0, keepdim=True)

            ml[qs].sum().backward()

            if hook.activation is not None and hook.gradient is not None:
                act = hook.activation  # (1, seq, hidden)
                grad = hook.gradient
                w = grad.mean(dim=-1, keepdim=True)
                cam = F.relu((act * w).sum(dim=-1))  # (1, seq)
                cam_np = cam[0].detach().cpu().numpy()

                # Extract patch tokens from encoder layer output
                n_pre = self._num_prefix
                seq = cam_np.shape[0]
                n_avail = seq - n_pre
                side = int(round(np.sqrt(max(n_avail, 1))))
                n_use = side * side
                spatial = cam_np[n_pre:n_pre + n_use].reshape(side, side)
                vis = self._blend(spatial, image, cv2.COLORMAP_JET)
            else:
                vis = image.copy()
                cn = "N/A"
        finally:
            hook.remove()
            fp32.zero_grad()

        vis = self._title(vis, f"GradCAM | {cn}")
        gc.collect()
        return {"visualization": vis, "target_class": target_class_id,
                "report": f"GradCAM '{cn}' (ID {target_class_id}). {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # 6. Class Saliency
    # ------------------------------------------------------------------

    def class_saliency(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)

        # Ensure only pixel_values is passed with grad enabled
        pv = inputs["pixel_values"].clone().requires_grad_(True)

        outputs = fp32(pixel_values=pv)
        seg = self._seg_from_outputs(outputs, oh, ow)
        if target_class_id is None:
            target_class_id = self._dominant_class(seg)
        cn = self._cname(target_class_id)

        score = outputs.class_queries_logits[0][:, target_class_id].sum()
        score.backward()

        if pv.grad is not None:
            sal = pv.grad[0].abs().max(dim=0).values.detach().cpu().numpy()
            sal = cv2.resize(sal, (ow, oh), interpolation=cv2.INTER_CUBIC)
            vis = self._blend(sal, image, cv2.COLORMAP_HOT)
        else:
            vis = image.copy()
            logger.warning("[XAI] Saliency: no gradient on pixel_values")

        vis = self._title(vis, f"Saliency | {cn}")
        fp32.zero_grad()
        gc.collect()
        return {"visualization": vis,
                "report": f"Saliency '{cn}' (ID {target_class_id}). {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # 7. Chefer Relevancy
    # ------------------------------------------------------------------

    def chefer_relevancy(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        # Only hook ENCODER layers for consistent seq_len
        layer_data = []
        handles = []
        for i in range(self._n_encoder):
            data = {"attn": None, "grad": None}

            def mk_fwd(d):
                def fn(m, a, o):
                    if isinstance(o, tuple) and len(o) >= 2 and o[1] is not None:
                        d["attn"] = o[1]
                return fn

            def mk_bwd(d):
                def fn(m, gi, go):
                    if len(go) >= 2 and go[1] is not None:
                        d["grad"] = go[1]
                return fn

            mod = fp32.layers[i].attention
            handles.append(mod.register_forward_hook(mk_fwd(data)))
            handles.append(mod.register_full_backward_hook(mk_bwd(data)))
            layer_data.append(data)

        try:
            outputs = fp32(**inputs)
            seg = self._seg_from_outputs(outputs, oh, ow)
            if target_class_id is None:
                target_class_id = self._dominant_class(seg)
            cn = self._cname(target_class_id)

            outputs.class_queries_logits[0][:, target_class_id].sum().backward()

            # Build relevancy matrix from encoder layers
            first_attn = next((d["attn"] for d in layer_data if d["attn"] is not None), None)
            if first_attn is None:
                return self._fallback(image, "No attention captured for Chefer")

            seq = first_attn.shape[-1]
            R = torch.eye(seq)

            for data in layer_data:
                attn = data.get("attn")
                if attn is None:
                    continue
                am = attn[0].detach().cpu().mean(dim=0)
                grad = data.get("grad")
                if grad is not None:
                    gm = grad[0].detach().cpu().mean(dim=0)
                    rel = torch.clamp(am * gm, min=0)
                else:
                    rel = am

                s = min(rel.shape[0], seq)
                rel = rel[:s, :s] + torch.eye(s)
                rel = rel / (rel.sum(dim=-1, keepdim=True) + 1e-10)
                R[:s, :s] = R[:s, :s] + R[:s, :s] @ rel

            spatial, gh, gw = self._cls_to_patches(R, gh, gw, is_decoder=False)
            vis = self._blend(spatial, image)

            # Threshold overlay
            thr = spatial.mean()
            mask = cv2.resize((spatial > thr).astype(np.float32), (ow, oh), interpolation=cv2.INTER_NEAREST)
            ov = vis.copy()
            ov[mask > 0.5] = (ov[mask > 0.5] * 0.7 + np.array([0, 80, 0]) * 0.3).astype(np.uint8)
            vis = ov

        finally:
            for h in handles:
                h.remove()
            fp32.zero_grad()

        vis = self._title(vis, f"Chefer Relevancy | {cn}")
        gc.collect()
        return {"visualization": vis,
                "report": f"Chefer '{cn}' (ID {target_class_id}), {self._n_encoder} encoder layers. {time.perf_counter()-t0:.1f}s"}

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def generate_xai_report(self, image, seg_mask=None):
        t0 = time.perf_counter()
        ent_result = self.predictive_entropy(image)

        if seg_mask is None:
            inputs, oh, ow = self._preprocess(image)
            with torch.no_grad():
                outputs = self.model(**inputs)
            seg_mask = self._seg_from_outputs(outputs, oh, ow)
            del outputs
            gc.collect()

        cls, cnt = np.unique(seg_mask, return_counts=True)
        total = seg_mask.size
        info = sorted(zip(cls, cnt), key=lambda x: -x[1])

        emap = ent_result.get("entropy_map")
        me = float(emap.mean()) if emap is not None else 0
        hp = float((emap > 0.5).mean() * 100) if emap is not None else 0

        ub = []
        if emap is not None:
            from skimage.segmentation import find_boundaries
            for cid, _ in info[:5]:
                bd = find_boundaries(seg_mask == cid, mode="thick")
                if bd.any():
                    be = float(emap[bd].mean())
                    if be > 0.3:
                        ub.append((self._cname(cid), be))

        lines = [
            "# XAI Analysis Report\n",
            "## Scene Composition",
            f"**{len(cls)} categories** detected:\n",
        ]
        for cid, c in info[:8]:
            p = c / total * 100
            bar = "\u2588" * int(p / 5) + "\u2591" * max(0, 20 - int(p / 5))
            lines.append(f"- **{self._cname(cid)}**: {p:.1f}% `{bar}`")

        lines += [
            "\n## Model Confidence",
            f"- Mean uncertainty: {me:.3f}",
            f"- High-uncertainty pixels: {hp:.1f}%",
        ]
        if me < 0.15:
            lines.append("- **Highly confident** predictions")
        elif me < 0.35:
            lines.append("- **Moderate confidence**")
        else:
            lines.append("- **Significant uncertainty** — review carefully")

        if ub:
            lines.append("\n### Uncertain Boundaries:")
            for n, e in sorted(ub, key=lambda x: -x[1])[:5]:
                lines.append(f"- {n}: entropy={e:.3f}")

        lines += [
            f"\n## Architecture",
            f"- DINOv3-EoMT-Large: {self._num_layers} layers, {self._num_heads} heads",
            f"- Patch size: {self._patch_size}, 150-class ADE20K",
            "\n## Citations",
            "- Attention Rollout: Abnar & Zuidema (2020)",
            "- GradCAM: Selvaraju et al. (2017)",
            "- Chefer Relevancy: Chefer et al. (2021)",
            "- Saliency: Simonyan et al. (2014)",
        ]
        return {"visualization": ent_result["visualization"], "report": "\n".join(lines)}

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------

    def run_full_analysis(self, image, layer=-1, head=None, target_class_id=None):
        logger.info("[XAI] Running full analysis suite...")
        t0 = time.perf_counter()
        results = {}

        logger.info("[XAI] Phase 1: attention + entropy + PCA...")
        for key, fn, kw in [
            ("attention", self.self_attention_maps, {"layer": layer, "head": head}),
            ("rollout",   self.attention_rollout,   {}),
            ("entropy",   self.predictive_entropy,  {}),
            ("pca",       self.feature_pca,         {"layer": layer}),
        ]:
            try:
                results[key] = fn(image, **kw)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}")
                results[key] = self._fallback(image, str(e))
        gc.collect()

        logger.info("[XAI] Phase 2: gradient methods...")
        for key, fn, kw in [
            ("gradcam",  self.gradcam_segmentation, {"target_class_id": target_class_id}),
            ("saliency", self.class_saliency,       {"target_class_id": target_class_id}),
            ("chefer",   self.chefer_relevancy,      {"target_class_id": target_class_id}),
        ]:
            try:
                results[key] = fn(image, **kw)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}")
                results[key] = self._fallback(image, str(e))
        gc.collect()

        logger.info("[XAI] Phase 3: report...")
        try:
            results["report"] = self.generate_xai_report(image)
        except Exception as e:
            logger.error(f"[XAI] report failed: {e}")
            results["report"] = self._fallback(image, str(e))

        logger.info(f"[XAI] Full analysis done in {time.perf_counter()-t0:.1f}s")
        return results
