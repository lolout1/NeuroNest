#!/usr/bin/env python3
"""Train self-supervised domain adaptation (MAE) on indoor images.

Fine-tunes ViT-MAE-Large on indoor room images so the backbone learns
care-environment visual patterns. Optionally transfers adapted weights
into the EoMT segmentation backbone.

Usage examples:

  # ADE20K indoor images only (downloads ~2GB first time)
  python scripts/train_domain_adaptation.py --output models/domain_adapted

  # Local blackspot dataset only
  python scripts/train_domain_adaptation.py \\
      --no-ade20k \\
      --local-dirs data/coco/train data/coco/valid \\
      --output models/domain_adapted

  # ADE20K + local images (recommended)
  python scripts/train_domain_adaptation.py \\
      --local-dirs data/coco/train data/coco/valid \\
      --output models/domain_adapted

  # After training, transfer to EoMT
  python scripts/train_domain_adaptation.py \\
      --transfer-to-eomt \\
      --mae-checkpoint models/domain_adapted/best_encoder.pt \\
      --output models/domain_adapted_eomt

Designed for RTX 2080 (8GB VRAM) with FP16 mixed precision.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("domain_adaptation")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Self-supervised domain adaptation for NeuroNest",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Data sources
    data = parser.add_argument_group("Data")
    data.add_argument(
        "--local-dirs", nargs="+", type=str, default=None,
        help="Local directories containing room images",
    )
    data.add_argument(
        "--no-ade20k", action="store_true",
        help="Skip ADE20K dataset (use only local images)",
    )
    data.add_argument(
        "--max-ade20k", type=int, default=None,
        help="Maximum ADE20K images to use (default: all indoor)",
    )
    data.add_argument(
        "--ade20k-split", type=str, default="train",
        choices=["train", "validation"],
        help="ADE20K split to use (default: train)",
    )

    # Training
    train = parser.add_argument_group("Training")
    train.add_argument(
        "--epochs", type=int, default=20,
        help="Number of training epochs (default: 20)",
    )
    train.add_argument(
        "--batch-size", type=int, default=4,
        help="Batch size (default: 4, fits RTX 2080 8GB)",
    )
    train.add_argument(
        "--lr", type=float, default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    train.add_argument(
        "--unfreeze-layers", type=int, default=4,
        help="Number of last encoder layers to unfreeze (default: 4)",
    )
    train.add_argument(
        "--mask-ratio", type=float, default=0.75,
        help="Fraction of patches to mask (default: 0.75)",
    )
    train.add_argument(
        "--no-fp16", action="store_true",
        help="Disable FP16 mixed precision",
    )
    train.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    # Output
    output = parser.add_argument_group("Output")
    output.add_argument(
        "--output", type=str, default="models/domain_adapted",
        help="Output directory for checkpoints and logs",
    )
    output.add_argument(
        "--device", type=str, default=None,
        help="Device (default: auto-detect cuda/cpu)",
    )

    # Weight transfer
    transfer = parser.add_argument_group("Weight Transfer")
    transfer.add_argument(
        "--transfer-to-eomt", action="store_true",
        help="After training (or standalone), transfer weights to EoMT",
    )
    transfer.add_argument(
        "--mae-checkpoint", type=str, default=None,
        help="Path to MAE encoder checkpoint for standalone transfer "
             "(skip training, just transfer)",
    )
    transfer.add_argument(
        "--eomt-model-id", type=str,
        default="tue-mps/ade20k_semantic_eomt_large_512",
        help="HuggingFace model ID for EoMT",
    )
    transfer.add_argument(
        "--dry-run", action="store_true",
        help="Report what would be transferred without saving",
    )

    return parser.parse_args()


def validate_args(args):
    """Validate argument combinations."""
    if args.mae_checkpoint and not args.transfer_to_eomt:
        logger.error("--mae-checkpoint requires --transfer-to-eomt")
        sys.exit(1)

    if args.no_ade20k and not args.local_dirs and not args.mae_checkpoint:
        logger.error(
            "No data sources specified. Use --local-dirs and/or remove --no-ade20k"
        )
        sys.exit(1)

    if args.local_dirs:
        for d in args.local_dirs:
            if not Path(d).exists():
                logger.warning(f"Directory not found: {d}")


def run_training(args):
    """Run MAE domain adaptation training."""
    import torch

    # Lazy imports (heavy dependencies)
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neuronest.adaptation.dataset import IndoorImageDataset
    from neuronest.adaptation.mae import MAEConfig, MAEDomainAdapter

    # --- Build dataset ---
    logger.info("Building indoor image dataset...")
    t0 = time.time()

    dataset = IndoorImageDataset(
        ade20k=not args.no_ade20k,
        local_dirs=args.local_dirs,
        image_size=224,
        max_ade20k=args.max_ade20k,
        split=args.ade20k_split,
    )

    logger.info(
        f"Dataset ready: {len(dataset)} images in {time.time() - t0:.1f}s"
    )
    logger.info(f"Source distribution: {dataset.source_distribution}")

    if len(dataset) < 20:
        logger.error(
            f"Only {len(dataset)} images found. Need at least 20. "
            "Add more images via --local-dirs or enable ADE20K."
        )
        sys.exit(1)

    # --- Configure training ---
    config = MAEConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        unfreeze_last_n=args.unfreeze_layers,
        mask_ratio=args.mask_ratio,
        fp16=not args.no_fp16,
        seed=args.seed,
    )

    logger.info(f"Training config: {config.to_dict()}")

    # --- Train ---
    adapter = MAEDomainAdapter(
        output_dir=args.output,
        device=args.device,
    )

    logger.info("Starting MAE domain adaptation training...")
    history = adapter.train(dataset, config, verbose=True)

    # --- Summary ---
    metadata = history.get("metadata", {})
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info(f"  Best val loss:  {metadata.get('best_val_loss', 'N/A'):.4f}")
    logger.info(f"  Best epoch:     {metadata.get('best_epoch', 'N/A') + 1}")
    logger.info(f"  Total epochs:   {metadata.get('total_epochs', 'N/A')}")
    logger.info(f"  Dataset size:   {metadata.get('dataset_size', 'N/A')}")
    logger.info(f"  Trainable:      {metadata.get('trainable_params', 0) / 1e6:.1f}M params")
    logger.info(f"  Output:         {args.output}")
    logger.info("=" * 60)

    return Path(args.output) / "best_encoder.pt"


def run_transfer(args, mae_checkpoint_path=None):
    """Transfer MAE weights to EoMT backbone."""
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from neuronest.adaptation.weight_transfer import transfer_weights_to_eomt

    checkpoint = mae_checkpoint_path or args.mae_checkpoint
    if checkpoint is None:
        checkpoint = Path(args.output) / "best_encoder.pt"

    checkpoint = Path(checkpoint)
    if not checkpoint.exists():
        logger.error(f"MAE encoder checkpoint not found: {checkpoint}")
        sys.exit(1)

    transfer_output = Path(args.output) / "eomt_transfer"

    logger.info(f"Transferring weights from {checkpoint} to EoMT...")
    stats = transfer_weights_to_eomt(
        mae_encoder_path=checkpoint,
        eomt_model_id=args.eomt_model_id,
        output_dir=transfer_output,
        dry_run=args.dry_run,
    )

    logger.info("=" * 60)
    logger.info("WEIGHT TRANSFER COMPLETE" if not args.dry_run else "DRY RUN COMPLETE")
    logger.info(f"  Transferred:    {stats['transferred_count']} weight tensors")
    logger.info(f"  Layers:         {stats['layers_transferred']}")
    logger.info(f"  Shape mismatch: {stats['skipped_shape_mismatch']}")
    logger.info(f"  Missing keys:   {stats['skipped_missing']}")
    if not args.dry_run:
        logger.info(f"  Model saved:    {stats.get('model_path', 'N/A')}")
    logger.info("=" * 60)

    return stats


def main():
    args = parse_args()
    validate_args(args)

    # Print GPU info
    try:
        import torch
        if torch.cuda.is_available():
            gpu = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu} ({mem:.1f} GB)")
        else:
            logger.info("No GPU detected — training on CPU (will be slow)")
    except ImportError:
        logger.error("PyTorch not installed. Install with: pip install torch")
        sys.exit(1)

    # Standalone transfer (skip training)
    if args.mae_checkpoint and args.transfer_to_eomt:
        run_transfer(args)
        return

    # Training
    mae_path = run_training(args)

    # Optional: transfer to EoMT after training
    if args.transfer_to_eomt:
        run_transfer(args, mae_checkpoint_path=mae_path)
    else:
        logger.info(
            "To transfer weights to EoMT, run:\n"
            f"  python scripts/train_domain_adaptation.py "
            f"--transfer-to-eomt --mae-checkpoint {mae_path} "
            f"--output {args.output}"
        )


if __name__ == "__main__":
    main()
