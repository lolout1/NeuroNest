import gc
import time
import logging
import traceback

logger = logging.getLogger(__name__)


class OrchestratorMixin:
    """Full 7-method analysis suite orchestration."""

    def run_full_analysis(
        self, image, layer=-1, head=None, target_class_id=None,
        progress_callback=None,
    ):
        logger.info("[XAI] Running full analysis suite...")
        t0 = time.perf_counter()
        results = {}

        def _progress(frac, msg):
            if progress_callback:
                try:
                    progress_callback(frac, desc=msg)
                except Exception:
                    pass
            logger.info(f"[XAI] [{frac*100:.0f}%] {msg}")

        methods = [
            ("attention", self.self_attention_maps, {"layer": layer, "head": head}),
            ("rollout", self.attention_rollout, {}),
            ("entropy", self.predictive_entropy, {}),
            ("pca", self.feature_pca, {"layer": layer}),
            ("gradcam", self.gradcam_segmentation, {"target_class_id": target_class_id}),
            ("integrated_gradients", self.integrated_gradients, {"target_class_id": target_class_id}),
            ("chefer", self.chefer_relevancy, {"target_class_id": target_class_id}),
        ]

        for i, (key, fn, kw) in enumerate(methods):
            _progress(i / len(methods), f"Running {key}...")
            try:
                results[key] = fn(image, **kw)
            except Exception as e:
                logger.error(f"[XAI] {key} failed: {e}\n{traceback.format_exc()}")
                results[key] = self._fallback(image, str(e), key)
            gc.collect()

        _progress(0.9, "Generating report...")
        try:
            results["report"] = self.generate_xai_report(image, method_results=results)
        except Exception as e:
            logger.error(f"[XAI] report failed: {e}")
            results["report"] = {
                "visualization": None,
                "report": f"Report generation failed: {e}",
            }

        elapsed = time.perf_counter() - t0
        _progress(1.0, f"Complete ({elapsed:.0f}s)")

        ok = sum(
            1
            for k in ["attention", "rollout", "entropy", "pca", "gradcam", "integrated_gradients", "chefer"]
            if k in results and "Error" not in results[k].get("report", "")
        )
        logger.info(f"[XAI] {ok}/7 methods succeeded in {elapsed:.1f}s")
        return results
