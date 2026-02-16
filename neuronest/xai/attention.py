import numpy as np
import torch
import gc
import time
import logging

from .hooks import AttentionCaptureHook, eager_attention

logger = logging.getLogger(__name__)


class AttentionMixin:
    """Self-attention maps and attention rollout methods."""

    def self_attention_maps(self, image, layer=-1, head=None):
        t0 = time.perf_counter()
        idx = layer if layer >= 0 else self._n_encoder - 1
        idx = min(idx, self._num_layers - 1)
        is_dec = idx >= self._n_encoder

        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        hook = AttentionCaptureHook().register(self.model.layers[idx].attention)
        try:
            with eager_attention(self.model), torch.no_grad():
                self.model(**inputs)
        finally:
            hook.remove()

        if hook.weights is None:
            return self._fallback(
                image, "Attention weights not captured (SDPA fallback)", "attention"
            )

        attn = hook.weights[0]
        if head is not None and 0 <= head < attn.shape[0]:
            a = attn[head]
            hl = f"Head {head}"
        else:
            a = attn.mean(dim=0)
            hl = "Mean"

        spatial, gh, gw = self._cls_to_patches(a, gh, gw, is_dec)

        attn_max = float(spatial.max())
        attn_mean = float(spatial.mean())
        focus_pct = float((spatial > spatial.mean() + spatial.std()).mean() * 100)

        blended = self._blend(spatial, image)
        extra = f"Layer {idx}/{self._num_layers-1} | {hl} | Focus: {focus_pct:.0f}% above mean"
        vis = self._annotate(
            blended, "attention", f"Self-Attention | Layer {idx} | {hl}", extra
        )

        elapsed = time.perf_counter() - t0
        del hook.weights
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "report": (
                f"**Self-Attention** (Layer {idx}, {hl}): "
                f"Grid {gh}x{gw}, peak={attn_max:.4f}, mean={attn_mean:.4f}, "
                f"focused area={focus_pct:.1f}%. "
                f"{'Decoder' if is_dec else 'Encoder'} layer. {elapsed:.1f}s"
            ),
        }

    def attention_rollout(self, image, head_fusion="mean", discard_ratio=0.1):
        t0 = time.perf_counter()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

        n_enc = self._n_encoder
        hooks = [
            AttentionCaptureHook().register(self.model.layers[i].attention)
            for i in range(n_enc)
        ]

        try:
            with eager_attention(self.model), torch.no_grad():
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
                if rollout.shape != fused.shape:
                    logger.warning(
                        f"[XAI] Rollout shape mismatch: {rollout.shape} vs {fused.shape}"
                    )
                    break
                rollout = rollout @ fused
            h.weights = None

        del hooks
        gc.collect()

        if rollout is None:
            return self._fallback(
                image, f"Rollout failed (0/{n_enc} layers captured)", "rollout"
            )

        spatial, gh, gw = self._cls_to_patches(rollout, gh, gw, is_decoder=False)
        if discard_ratio > 0:
            thr = np.percentile(spatial, discard_ratio * 100)
            spatial = np.where(spatial > thr, spatial, 0)

        focus_pct = float((spatial > spatial.mean()).mean() * 100)
        blended = self._blend(spatial, image)
        extra = f"{captured}/{n_enc} layers | {head_fusion} fusion | Focused: {focus_pct:.0f}%"
        vis = self._annotate(
            blended, "rollout", f"Attention Rollout | {captured} Layers", extra
        )

        elapsed = time.perf_counter() - t0
        del rollout
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "report": (
                f"**Attention Rollout**: {captured}/{n_enc} encoder layers, "
                f"{head_fusion} head fusion, discard bottom {discard_ratio*100:.0f}%. "
                f"Focused area={focus_pct:.1f}%. {elapsed:.1f}s"
            ),
        }
