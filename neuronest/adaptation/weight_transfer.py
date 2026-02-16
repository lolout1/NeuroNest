"""Transfer adapted MAE encoder weights to the EoMT segmentation backbone.

ViT-MAE-Large and DINOv2-Large share the same ViT-Large architecture
(24 layers, 1024 dim, 16 heads). After domain-adapting the MAE encoder
on indoor images, we transfer the fine-tuned layer weights into the EoMT
backbone to improve segmentation on care environments.

Only layers that were actually fine-tuned (last N encoder layers + LayerNorm)
are transferred. The rest of the EoMT backbone is left untouched.
"""

import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import torch

logger = logging.getLogger(__name__)

# ViT-MAE and DINOv2 share the same internal layer structure.
# The key difference is the prefix path through the model hierarchy.
# MAE encoder state_dict (from model.vit.state_dict()):
#   embeddings.*, encoder.layer.{i}.*, layernorm.*
# EoMT backbone (varies by model wrapper):
#   model.backbone.embeddings.*, model.backbone.encoder.layer.{i}.*, etc.

# Layer-level sub-keys that exist in each encoder layer
_LAYER_SUBKEYS = [
    "attention.attention.query.weight",
    "attention.attention.query.bias",
    "attention.attention.key.weight",
    "attention.attention.key.bias",
    "attention.attention.value.weight",
    "attention.attention.value.bias",
    "attention.output.dense.weight",
    "attention.output.dense.bias",
    "intermediate.dense.weight",
    "intermediate.dense.bias",
    "output.dense.weight",
    "output.dense.bias",
    "layernorm_before.weight",
    "layernorm_before.bias",
    "layernorm_after.weight",
    "layernorm_after.bias",
]


def _find_backbone_prefix(eomt_state: Dict[str, torch.Tensor]) -> Optional[str]:
    """Discover the backbone key prefix in the EoMT model state dict.

    Looks for common ViT backbone keys (encoder.layer.0) under various
    possible prefixes.
    """
    target_suffix = "encoder.layer.0.attention.attention.query.weight"

    for key in eomt_state:
        if key.endswith(target_suffix):
            prefix = key[: -len(target_suffix)]
            logger.info(f"Found EoMT backbone prefix: '{prefix}'")
            return prefix

    # Fallback: try common prefixes
    common_prefixes = [
        "model.backbone.",
        "backbone.",
        "model.pixel_level_module.encoder.model.",
        "pixel_level_module.encoder.model.",
        "model.",
        "",
    ]
    for prefix in common_prefixes:
        test_key = f"{prefix}{target_suffix}"
        if test_key in eomt_state:
            logger.info(f"Found EoMT backbone prefix (fallback): '{prefix}'")
            return prefix

    return None


def _count_encoder_layers(state_dict: Dict[str, torch.Tensor], prefix: str = "") -> int:
    """Count the number of encoder layers in a state dict."""
    pattern = re.compile(re.escape(prefix) + r"encoder\.layer\.(\d+)\.")
    layer_ids = set()
    for key in state_dict:
        m = pattern.match(key)
        if m:
            layer_ids.add(int(m.group(1)))
    return len(layer_ids)


def transfer_weights_to_eomt(
    mae_encoder_path: Union[str, Path],
    eomt_model_id: str = "tue-mps/ade20k_semantic_eomt_large_512",
    output_dir: Union[str, Path] = "models/domain_adapted_eomt",
    layers_to_transfer: Optional[List[int]] = None,
    transfer_layernorm: bool = True,
    transfer_embeddings: bool = False,
    dry_run: bool = False,
) -> Dict:
    """Transfer adapted MAE encoder weights into EoMT backbone.

    Args:
        mae_encoder_path: Path to best_encoder.pt from MAE training.
        eomt_model_id: HuggingFace model ID for EoMT.
        output_dir: Directory to save the adapted EoMT model.
        layers_to_transfer: Specific layer indices to transfer (default: last 4).
            If None, auto-detects from the MAE training metadata.
        transfer_layernorm: Also transfer the final LayerNorm weights.
        transfer_embeddings: Also transfer patch embedding weights (usually False).
        dry_run: If True, only report what would be transferred without saving.

    Returns:
        Dict with transfer statistics and output paths.
    """
    mae_encoder_path = Path(mae_encoder_path)
    output_dir = Path(output_dir)

    if not mae_encoder_path.exists():
        raise FileNotFoundError(f"MAE encoder not found: {mae_encoder_path}")

    # --- Load MAE encoder state dict ---
    logger.info(f"Loading MAE encoder from {mae_encoder_path}")
    mae_state = torch.load(mae_encoder_path, map_location="cpu", weights_only=True)

    mae_layers = _count_encoder_layers(mae_state)
    logger.info(f"MAE encoder has {mae_layers} layers")

    # --- Determine which layers to transfer ---
    if layers_to_transfer is None:
        # Try to read from training metadata
        metadata_path = mae_encoder_path.parent / "training_metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                metadata = json.load(f)
            unfreeze_n = metadata.get("config", {}).get("unfreeze_last_n", 4)
            layers_to_transfer = list(range(mae_layers - unfreeze_n, mae_layers))
            logger.info(
                f"Auto-detected from metadata: transferring layers "
                f"{layers_to_transfer} (last {unfreeze_n})"
            )
        else:
            # Default: last 4 layers
            layers_to_transfer = list(range(mae_layers - 4, mae_layers))
            logger.info(f"No metadata found, defaulting to last 4 layers: {layers_to_transfer}")

    # --- Load EoMT model ---
    logger.info(f"Loading EoMT model: {eomt_model_id}")
    from transformers import AutoModelForUniversalSegmentation

    eomt_model = AutoModelForUniversalSegmentation.from_pretrained(eomt_model_id)
    eomt_state = eomt_model.state_dict()

    # --- Find backbone prefix ---
    backbone_prefix = _find_backbone_prefix(eomt_state)
    if backbone_prefix is None:
        raise RuntimeError(
            "Could not find ViT backbone in EoMT model. "
            "The model architecture may have changed. "
            f"Sample keys: {list(eomt_state.keys())[:10]}"
        )

    eomt_layers = _count_encoder_layers(eomt_state, backbone_prefix)
    logger.info(f"EoMT backbone has {eomt_layers} layers (prefix: '{backbone_prefix}')")

    if mae_layers != eomt_layers:
        logger.warning(
            f"Layer count mismatch: MAE has {mae_layers}, EoMT has {eomt_layers}. "
            "Only matching layers will be transferred."
        )

    # --- Build key mapping ---
    transferred = []
    skipped_shape = []
    skipped_missing = []

    keys_to_transfer = {}

    # Encoder layers
    for layer_idx in layers_to_transfer:
        if layer_idx >= min(mae_layers, eomt_layers):
            logger.warning(f"Layer {layer_idx} out of range, skipping")
            continue

        for subkey in _LAYER_SUBKEYS:
            mae_key = f"encoder.layer.{layer_idx}.{subkey}"
            eomt_key = f"{backbone_prefix}encoder.layer.{layer_idx}.{subkey}"

            if mae_key in mae_state and eomt_key in eomt_state:
                if mae_state[mae_key].shape == eomt_state[eomt_key].shape:
                    keys_to_transfer[eomt_key] = mae_state[mae_key]
                    transferred.append((mae_key, eomt_key))
                else:
                    skipped_shape.append(
                        (mae_key, mae_state[mae_key].shape, eomt_state[eomt_key].shape)
                    )
            else:
                skipped_missing.append((mae_key, eomt_key))

    # Final LayerNorm
    if transfer_layernorm:
        for suffix in ["layernorm.weight", "layernorm.bias"]:
            mae_key = suffix
            eomt_key = f"{backbone_prefix}{suffix}"
            if mae_key in mae_state and eomt_key in eomt_state:
                if mae_state[mae_key].shape == eomt_state[eomt_key].shape:
                    keys_to_transfer[eomt_key] = mae_state[mae_key]
                    transferred.append((mae_key, eomt_key))

    # Patch embeddings (usually skip — different training objectives)
    if transfer_embeddings:
        for suffix in [
            "embeddings.patch_embeddings.projection.weight",
            "embeddings.patch_embeddings.projection.bias",
            "embeddings.position_embeddings",
            "embeddings.cls_token",
        ]:
            mae_key = suffix
            eomt_key = f"{backbone_prefix}{suffix}"
            if mae_key in mae_state and eomt_key in eomt_state:
                if mae_state[mae_key].shape == eomt_state[eomt_key].shape:
                    keys_to_transfer[eomt_key] = mae_state[mae_key]
                    transferred.append((mae_key, eomt_key))

    # --- Report ---
    stats = {
        "transferred_count": len(transferred),
        "skipped_shape_mismatch": len(skipped_shape),
        "skipped_missing": len(skipped_missing),
        "layers_transferred": layers_to_transfer,
        "backbone_prefix": backbone_prefix,
        "mae_total_layers": mae_layers,
        "eomt_total_layers": eomt_layers,
    }

    logger.info(
        f"Transfer plan: {len(transferred)} weights to transfer, "
        f"{len(skipped_shape)} shape mismatches, "
        f"{len(skipped_missing)} missing keys"
    )

    if skipped_shape:
        for mae_key, mae_shape, eomt_shape in skipped_shape[:5]:
            logger.warning(f"Shape mismatch: {mae_key} {mae_shape} vs {eomt_shape}")

    if dry_run:
        logger.info("Dry run — no weights transferred")
        stats["dry_run"] = True
        stats["transferred_keys"] = [t[0] for t in transferred]
        return stats

    # --- Apply transfer ---
    eomt_state.update(keys_to_transfer)
    eomt_model.load_state_dict(eomt_state)

    logger.info(f"Transferred {len(transferred)} weight tensors to EoMT backbone")

    # --- Save ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full adapted model (HuggingFace format for easy loading)
    eomt_model.save_pretrained(output_dir / "eomt_adapted")
    logger.info(f"Saved adapted EoMT model to {output_dir / 'eomt_adapted'}")

    # Also save just the modified state dict for lightweight deployment
    torch.save(keys_to_transfer, output_dir / "transferred_weights.pt")

    # Save transfer report
    report = {
        **stats,
        "mae_encoder_path": str(mae_encoder_path),
        "eomt_model_id": eomt_model_id,
        "output_dir": str(output_dir),
        "transferred_keys": [t[0] for t in transferred],
    }
    with open(output_dir / "transfer_report.json", "w") as f:
        json.dump(report, f, indent=2)

    stats["output_dir"] = str(output_dir)
    stats["model_path"] = str(output_dir / "eomt_adapted")
    return stats


def load_adapted_eomt(
    adapted_dir: Union[str, Path],
    device: Optional[str] = None,
) -> "AutoModelForUniversalSegmentation":
    """Load the domain-adapted EoMT model.

    Args:
        adapted_dir: Directory containing the adapted model (from transfer_weights_to_eomt).
        device: Target device. Auto-detects if None.

    Returns:
        The adapted EoMT model ready for inference.
    """
    from transformers import AutoModelForUniversalSegmentation

    adapted_dir = Path(adapted_dir)
    model_path = adapted_dir / "eomt_adapted"

    if not model_path.exists():
        raise FileNotFoundError(
            f"Adapted model not found at {model_path}. "
            "Run transfer_weights_to_eomt first."
        )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForUniversalSegmentation.from_pretrained(model_path)
    model.to(device)
    model.eval()

    logger.info(f"Loaded adapted EoMT from {model_path} on {device}")
    return model
