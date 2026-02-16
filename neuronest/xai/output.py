import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import time
import logging

from .hooks import HiddenStateCaptureHook

logger = logging.getLogger(__name__)


class OutputMixin:
    """Predictive entropy and feature PCA methods."""

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

        me = float(ent_map.mean())
        hp = float((ent_map > 0.5).mean() * 100)
        conf_label = "HIGH" if me < 0.15 else "MODERATE" if me < 0.35 else "LOW"

        blended = self._blend(ent_map, image, cv2.COLORMAP_MAGMA, 0.55)
        extra = f"Mean H={me:.3f} | High-unc={hp:.1f}% | Confidence: {conf_label}"
        vis = self._annotate(blended, "entropy", "Predictive Entropy", extra)

        elapsed = time.perf_counter() - t0
        del outputs, mp, cp, pp, ent
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "entropy_map": ent_map,
            "report": (
                f"**Predictive Entropy**: Mean={me:.3f}, high-uncertainty={hp:.1f}%, "
                f"confidence={conf_label}. Bright regions = class boundaries or ambiguity. "
                f"{elapsed:.1f}s"
            ),
        }

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
            return self._fallback(image, "Hidden state not captured", "pca")

        hidden = hook.state[0].numpy()
        n_pre = self._num_prefix
        seq_len = hidden.shape[0]

        is_dec = idx >= self._n_encoder
        n_patches_avail = seq_len - n_pre - (self._num_queries if is_dec else 0)
        side = int(round(np.sqrt(max(n_patches_avail, 1))))
        n_use = side * side
        gh, gw = side, side

        feats = hidden[n_pre : n_pre + n_use]
        centered = feats - feats.mean(axis=0, keepdims=True)
        U, S, _ = np.linalg.svd(centered, full_matrices=False)
        comps = U[:, :3] * S[:3]

        for c in range(3):
            lo, hi = comps[:, c].min(), comps[:, c].max()
            comps[:, c] = (comps[:, c] - lo) / (hi - lo + 1e-8) * 255

        pca_img = cv2.resize(
            comps.reshape(gh, gw, 3).astype(np.uint8),
            (ow, oh),
            interpolation=cv2.INTER_CUBIC,
        )

        tv = (S**2).sum()
        v3 = (S[:3] ** 2) / tv * 100

        title_bar = np.full((44, ow, 3), 30, dtype=np.uint8)
        cv2.putText(
            title_bar, f"Feature PCA | Layer {idx}", (12, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA,
        )
        vis = np.vstack([title_bar, pca_img])

        legend_h = 60
        legend = np.full((legend_h, ow, 3), 25, dtype=np.uint8)
        cv2.putText(
            legend,
            f"PC1(R)={v3[0]:.1f}%  PC2(G)={v3[1]:.1f}%  PC3(B)={v3[2]:.1f}%",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA,
        )
        cv2.putText(
            legend,
            "Similar colors = similar learned representations | SVD of 1024-dim features",
            (10, 45),
            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 140), 1, cv2.LINE_AA,
        )
        sw = 18
        cv2.rectangle(
            legend, (ow - 3 * (sw + 5) - 10, 5), (ow - 2 * (sw + 5) - 10, 5 + sw),
            (255, 0, 0), -1,
        )
        cv2.rectangle(
            legend, (ow - 2 * (sw + 5) - 10, 5), (ow - 1 * (sw + 5) - 10, 5 + sw),
            (0, 255, 0), -1,
        )
        cv2.rectangle(
            legend, (ow - 1 * (sw + 5) - 10, 5), (ow - 10, 5 + sw),
            (0, 0, 255), -1,
        )
        vis = np.vstack([vis, legend])

        elapsed = time.perf_counter() - t0
        del hook.state
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": pca_img,
            "report": (
                f"**Feature PCA** (Layer {idx}): "
                f"PC1={v3[0]:.1f}%, PC2={v3[1]:.1f}%, PC3={v3[2]:.1f}% variance explained. "
                f"Similar colors = similar feature representations. {elapsed:.1f}s"
            ),
        }
