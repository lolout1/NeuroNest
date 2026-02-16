import os
import hashlib
import numpy as np
import logging
from PIL import Image

logger = logging.getLogger(__name__)

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "xai_cache",
)
_METHOD_KEYS = ["attention", "rollout", "gradcam", "entropy", "pca", "integrated_gradients", "chefer"]
_DEFECT_METHOD_KEYS = ["gradcam", "entropy", "integrated_gradients", "chefer"]


def _cache_key_from_file(image_path: str) -> str:
    h = hashlib.md5()
    try:
        with open(image_path, "rb") as f:
            h.update(f.read(256 * 1024))
    except (OSError, IOError):
        h.update(image_path.encode())
    return h.hexdigest()[:12]


def _cache_key_from_array(image: np.ndarray) -> str:
    h = hashlib.md5()
    h.update(image.tobytes()[:256 * 1024])
    return h.hexdigest()[:12]


class CacheMixin:
    """XAI result caching (content-based hashing)."""

    _CACHE_DIR = _CACHE_DIR
    _METHOD_KEYS = _METHOD_KEYS
    _DEFECT_METHOD_KEYS = _DEFECT_METHOD_KEYS

    @staticmethod
    def _cache_key_from_file(image_path: str) -> str:
        return _cache_key_from_file(image_path)

    @staticmethod
    def _cache_key_from_array(image: np.ndarray) -> str:
        return _cache_key_from_array(image)

    def save_results(self, results: dict, image_path: str = None, image: np.ndarray = None):
        if image_path:
            key = _cache_key_from_file(image_path)
        elif image is not None:
            key = _cache_key_from_array(image)
        else:
            return None

        out = os.path.join(_CACHE_DIR, key)
        os.makedirs(out, exist_ok=True)

        saved = 0
        for method_key in _METHOD_KEYS:
            r = results.get(method_key, {})
            vis = r.get("visualization")
            if vis is not None:
                try:
                    Image.fromarray(vis).save(
                        os.path.join(out, f"{method_key}.png"), optimize=True
                    )
                    saved += 1
                except Exception as e:
                    logger.warning(f"[XAI] Cache save failed for {method_key}: {e}")
            report = r.get("report", "")
            if report:
                with open(os.path.join(out, f"{method_key}.txt"), "w") as f:
                    f.write(report)

        full_report = results.get("report", {}).get("report", "")
        if full_report:
            with open(os.path.join(out, "full_report.md"), "w") as f:
                f.write(full_report)

        logger.info(f"[XAI] Cached {saved} visualizations to {out}")
        return out

    @staticmethod
    def load_cached_from_file(image_path: str):
        key = _cache_key_from_file(image_path)
        return _load_cache_dir(os.path.join(_CACHE_DIR, key))

    @staticmethod
    def load_cached_from_array(image: np.ndarray):
        key = _cache_key_from_array(image)
        return _load_cache_dir(os.path.join(_CACHE_DIR, key))

    @staticmethod
    def list_cached():
        if not os.path.isdir(_CACHE_DIR):
            return []
        return [
            os.path.join(_CACHE_DIR, d)
            for d in os.listdir(_CACHE_DIR)
            if os.path.isdir(os.path.join(_CACHE_DIR, d))
        ]

    # Defect caching

    def save_defect_results(self, results: dict, image_path: str = None, image: np.ndarray = None):
        if image_path:
            key = _cache_key_from_file(image_path)
        elif image is not None:
            key = _cache_key_from_array(image)
        else:
            return None

        out = os.path.join(_CACHE_DIR, key, "defect")
        os.makedirs(out, exist_ok=True)

        saved = 0
        for method_key in _DEFECT_METHOD_KEYS:
            r = results.get(method_key, {})
            vis = r.get("defect_visualization")
            if vis is not None:
                try:
                    Image.fromarray(vis).save(
                        os.path.join(out, f"{method_key}.png"), optimize=True
                    )
                    saved += 1
                except Exception as e:
                    logger.warning(f"[XAI] Defect cache save failed for {method_key}: {e}")
            report = r.get("report", "")
            if report:
                with open(os.path.join(out, f"{method_key}.txt"), "w") as f:
                    f.write(report)

        logger.info(f"[XAI] Cached {saved} defect visualizations to {out}")
        return out

    @staticmethod
    def load_cached_defect_from_file(image_path: str):
        key = _cache_key_from_file(image_path)
        defect_dir = os.path.join(_CACHE_DIR, key, "defect")
        if not os.path.isdir(defect_dir):
            return None

        results = {}
        for method_key in _DEFECT_METHOD_KEYS:
            r = {}
            png = os.path.join(defect_dir, f"{method_key}.png")
            if os.path.exists(png):
                try:
                    r["defect_visualization"] = np.array(Image.open(png).convert("RGB"))
                except Exception as e:
                    logger.warning(f"[XAI] Defect cache load failed for {method_key}: {e}")
            txt = os.path.join(defect_dir, f"{method_key}.txt")
            if os.path.exists(txt):
                with open(txt) as f:
                    r["report"] = f.read()
            if r:
                results[method_key] = r

        if results:
            logger.info(f"[XAI] Defect cache hit: {defect_dir} ({len(results)} methods)")
        return results if results else None


def _load_cache_dir(cache_path: str):
    if not os.path.isdir(cache_path):
        return None

    results = {}
    for method_key in _METHOD_KEYS:
        r = {}
        png = os.path.join(cache_path, f"{method_key}.png")
        if os.path.exists(png):
            try:
                r["visualization"] = np.array(Image.open(png).convert("RGB"))
            except Exception as e:
                logger.warning(f"[XAI] Cache load failed for {method_key}: {e}")
        txt = os.path.join(cache_path, f"{method_key}.txt")
        if os.path.exists(txt):
            with open(txt) as f:
                r["report"] = f.read()
        if r:
            results[method_key] = r

    rpt = os.path.join(cache_path, "full_report.md")
    if os.path.exists(rpt):
        with open(rpt) as f:
            results["report"] = {"report": f.read()}

    if results:
        logger.info(f"[XAI] Cache hit: {cache_path} ({len(results) - ('report' in results)} methods)")
    return results if results else None
