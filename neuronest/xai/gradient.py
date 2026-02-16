import numpy as np
import cv2
import torch
import torch.nn.functional as F
import gc
import time
import logging

from .hooks import ActivationGradientHook

logger = logging.getLogger(__name__)


class GradientMixin:
    """GradCAM, Integrated Gradients, and Chefer Relevancy methods."""

    def gradcam_segmentation(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

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
                act = hook.activation
                grad = hook.gradient
                w = grad.mean(dim=-1, keepdim=True)
                cam = F.relu((act * w).sum(dim=-1))
                cam_np = cam[0].detach().cpu().numpy()

                n_pre = self._num_prefix
                seq = cam_np.shape[0]
                n_avail = seq - n_pre
                side = int(round(np.sqrt(max(n_avail, 1))))
                n_use = side * side
                spatial = cam_np[n_pre : n_pre + n_use].reshape(side, side)
                blended = self._blend(spatial, image, cv2.COLORMAP_JET)
                cam_max = float(spatial.max())
                hot_pct = float(
                    (spatial > spatial.mean() + spatial.std()).mean() * 100
                )
            else:
                blended = image.copy()
                cn = "N/A (no gradient)"
                cam_max = 0
                hot_pct = 0
        finally:
            hook.remove()
            fp32.zero_grad()

        extra = f"Target: {cn} (ID {target_class_id}) | Peak={cam_max:.3f} | Hot area={hot_pct:.0f}%"
        vis = self._annotate(blended, "gradcam", f"GradCAM | {cn}", extra)

        elapsed = time.perf_counter() - t0
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "target_class": target_class_id,
            "report": (
                f"**GradCAM** (target: '{cn}', ID {target_class_id}): "
                f"Encoder layer {target_layer_idx}, peak={cam_max:.3f}, "
                f"hot area={hot_pct:.1f}%. Red/yellow = strongest class activation. "
                f"{elapsed:.1f}s"
            ),
        }

    def integrated_gradients(self, image, target_class_id=None, n_steps=8):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)

        with torch.no_grad():
            outputs_ref = fp32(**inputs)
        seg = self._seg_from_outputs(outputs_ref, oh, ow)
        if target_class_id is None:
            target_class_id = self._dominant_class(seg)
        cn = self._cname(target_class_id)
        del outputs_ref
        gc.collect()

        pv = inputs["pixel_values"]
        baseline = torch.zeros_like(pv)

        ig_grads = torch.zeros_like(pv)
        for step in range(n_steps + 1):
            alpha = step / n_steps
            interp = (baseline + alpha * (pv - baseline)).detach().requires_grad_(True)

            fwd = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }
            fwd["pixel_values"] = interp

            outputs = fp32(**fwd)
            score = outputs.class_queries_logits[0][:, target_class_id].sum()
            score.backward()

            if interp.grad is not None:
                ig_grads += interp.grad.detach()

            fp32.zero_grad()
            del outputs, score, fwd, interp
            if step % 3 == 0:
                gc.collect()

        ig_attr = (pv - baseline) * ig_grads / (n_steps + 1)
        attr_map = ig_attr[0].abs().max(dim=0).values.detach().cpu().numpy()
        attr_map = cv2.resize(attr_map, (ow, oh), interpolation=cv2.INTER_CUBIC)

        attr_max = float(attr_map.max())
        attr_sum = float(ig_attr.sum())
        significant_pct = float(
            (attr_map > attr_map.mean() + attr_map.std()).mean() * 100
        )

        blended = self._blend(attr_map, image, cv2.COLORMAP_HOT)
        extra = f"Target: {cn} | {n_steps} steps | Significant={significant_pct:.0f}% | Sum={attr_sum:.2f}"
        vis = self._annotate(
            blended, "integrated_gradients", f"Integrated Gradients | {cn}", extra
        )

        elapsed = time.perf_counter() - t0
        del ig_grads, ig_attr, baseline
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "attr_map": attr_map,
            "report": (
                f"**Integrated Gradients** (target: '{cn}', ID {target_class_id}): "
                f"{n_steps} steps, peak={attr_max:.4f}, "
                f"significant area={significant_pct:.1f}%, sum={attr_sum:.2f}. "
                f"Satisfies completeness axiom (Sundararajan 2017). "
                f"Bright = pixels most influencing '{cn}' prediction. {elapsed:.1f}s"
            ),
        }

    def chefer_relevancy(self, image, target_class_id=None):
        t0 = time.perf_counter()
        fp32 = self._get_fp32()
        inputs, oh, ow = self._preprocess(image)
        gh, gw = self._patch_grid(inputs)

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

            first_attn = next(
                (d["attn"] for d in layer_data if d["attn"] is not None), None
            )
            if first_attn is None:
                return self._fallback(
                    image, "No attention captured for Chefer", "chefer"
                )

            seq = first_attn.shape[-1]
            R = torch.eye(seq)
            layers_used = 0

            for data in layer_data:
                attn = data.get("attn")
                if attn is None:
                    continue
                layers_used += 1
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
            blended = self._blend(spatial, image)

            thr = spatial.mean()
            mask = cv2.resize(
                (spatial > thr).astype(np.float32),
                (ow, oh),
                interpolation=cv2.INTER_NEAREST,
            )
            ov = blended.copy()
            ov[mask > 0.5] = (
                ov[mask > 0.5] * 0.7 + np.array([0, 80, 0]) * 0.3
            ).astype(np.uint8)
            relevance_pct = float((spatial > thr).mean() * 100)

        finally:
            for h in handles:
                h.remove()
            fp32.zero_grad()

        extra = f"Target: {cn} | {layers_used} layers | Relevant area={relevance_pct:.0f}% (green tint)"
        vis = self._annotate(ov, "chefer", f"Chefer Relevancy | {cn}", extra)

        elapsed = time.perf_counter() - t0
        gc.collect()
        return {
            "visualization": vis,
            "_blended_raw": blended,
            "report": (
                f"**Chefer Relevancy** (target: '{cn}', ID {target_class_id}): "
                f"{layers_used}/{self._n_encoder} encoder layers, "
                f"relevant area={relevance_pct:.1f}%. "
                f"Green overlay = model-relevant regions. {elapsed:.1f}s"
            ),
        }
