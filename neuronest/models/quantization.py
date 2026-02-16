import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


def quantize_model_int8(model: nn.Module, model_name: str = "model") -> nn.Module:
    """Dynamic INT8 quantization on nn.Linear layers. Falls back to FP32 on failure."""
    linear_count = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    if linear_count == 0:
        logger.info(f"[Quantize] {model_name}: no nn.Linear layers, skipping")
        return model

    logger.info(f"[Quantize] {model_name}: quantizing {linear_count} nn.Linear → INT8")
    try:
        quantized = torch.quantization.quantize_dynamic(
            model, {nn.Linear}, dtype=torch.qint8
        )
        quantized_count = sum(
            1 for m in quantized.modules()
            if type(m).__name__ == "DynamicQuantizedLinear"
        )
        logger.info(
            f"[Quantize] {model_name}: {quantized_count}/{linear_count} layers quantized"
        )
        return quantized
    except Exception as e:
        logger.warning(f"[Quantize] {model_name}: failed ({e}), using FP32")
        return model
