#!/usr/bin/env python3
"""Train the room anomaly detector on a directory of room images.

Usage:
    # Basic (auto-detect GPU):
    python scripts/train_anomaly.py --data data/rooms/

    # With all options:
    python scripts/train_anomaly.py \
        --data data/rooms/ \
        --output models/anomaly \
        --cache .cache/anomaly_features \
        --batch-size 8 \
        --epochs 200 \
        --bottleneck 64 \
        --sigma 2.0 \
        --device cuda

    # Quick test on sample images:
    python scripts/train_anomaly.py --data samples/ --output models/anomaly_test
"""

import argparse
import glob
import logging
import os
import sys
import time

# Add repo root to path so neuronest package is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from neuronest.anomaly.extractor import FeatureExtractor
from neuronest.anomaly.detector import AnomalyDetector, TrainConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_anomaly")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}


def find_images(data_dir: str) -> list:
    """Recursively find all image files in a directory."""
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(glob.glob(os.path.join(data_dir, f"**/*{ext}"), recursive=True))
        images.extend(glob.glob(os.path.join(data_dir, f"**/*{ext.upper()}"), recursive=True))
    # Deduplicate (case-insensitive extensions might overlap)
    images = list(set(images))
    images.sort()
    return images


def main():
    parser = argparse.ArgumentParser(
        description="Train NeuroNest room anomaly detector",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train on a folder of room images (GPU):
  python scripts/train_anomaly.py --data data/rooms/ --device cuda

  # Train on samples as a quick test:
  python scripts/train_anomaly.py --data samples/ --epochs 100 --sigma 2.5

  # Use pre-extracted features (skip extraction):
  python scripts/train_anomaly.py --features features.npy --output models/anomaly
        """,
    )

    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--data", type=str,
        help="Directory containing room images (searched recursively)",
    )
    input_group.add_argument(
        "--features", type=str,
        help="Pre-extracted features .npy file (N, 1024) — skips extraction",
    )

    # Output
    parser.add_argument(
        "--output", type=str, default="models/anomaly",
        help="Output directory for model checkpoint and stats (default: models/anomaly)",
    )
    parser.add_argument(
        "--cache", type=str, default=".cache/anomaly_features",
        help="Cache directory for extracted features (default: .cache/anomaly_features)",
    )

    # Extraction
    parser.add_argument(
        "--batch-size", type=int, default=8,
        help="Batch size for feature extraction (default: 8, fits RTX 2080 8GB)",
    )

    # Training
    parser.add_argument("--epochs", type=int, default=200, help="Max training epochs (default: 200)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
    parser.add_argument("--bottleneck", type=int, default=64, help="Autoencoder bottleneck dim (default: 64)")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate (default: 0.1)")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience (default: 30)")
    parser.add_argument("--sigma", type=float, default=2.0, help="Anomaly threshold = mean + sigma*std (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Device
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu', or auto-detect (default: auto)",
    )

    # Evaluation
    parser.add_argument(
        "--eval-dir", type=str, default=None,
        help="Optional: directory of images to evaluate after training",
    )

    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("NeuroNest Anomaly Detector Training")
    logger.info("=" * 60)

    # ---- Step 1: Get features ----

    if args.features:
        logger.info(f"Loading pre-extracted features from {args.features}")
        features = np.load(args.features)
        if features.ndim != 2 or features.shape[1] != 1024:
            logger.error(f"Invalid features shape: {features.shape}, expected (N, 1024)")
            sys.exit(1)
        valid_paths = []
        logger.info(f"Loaded {features.shape[0]} feature vectors")
    else:
        # Find images
        images = find_images(args.data)
        if not images:
            logger.error(f"No images found in {args.data}")
            logger.error(f"Supported formats: {', '.join(sorted(IMAGE_EXTENSIONS))}")
            sys.exit(1)
        logger.info(f"Found {len(images)} images in {args.data}")

        # Extract features
        extractor = FeatureExtractor(
            device=args.device,
            cache_dir=args.cache,
        )

        t0 = time.time()
        features, valid_paths, skipped = extractor.extract_batch(
            images, batch_size=args.batch_size,
        )
        elapsed = time.time() - t0

        logger.info(
            f"Feature extraction: {len(valid_paths)} images in {elapsed:.1f}s "
            f"({elapsed / max(len(valid_paths), 1):.2f}s/image)"
        )
        if skipped:
            logger.warning(f"Skipped {len(skipped)} images (corrupt/invalid)")
            for p in skipped[:5]:
                logger.warning(f"  - {p}")
            if len(skipped) > 5:
                logger.warning(f"  ... and {len(skipped) - 5} more")

        # Save extracted features for reuse
        feat_path = os.path.join(args.output, "features.npy")
        os.makedirs(args.output, exist_ok=True)
        np.save(feat_path, features)
        logger.info(f"Saved features to {feat_path}")

        # Free extractor GPU memory before training
        extractor.unload()

    if features.shape[0] < 10:
        logger.error(
            f"Only {features.shape[0]} valid images. Need at least 10. "
            "Add more room images to your data directory."
        )
        sys.exit(1)

    # ---- Step 2: Train autoencoder ----

    config = TrainConfig(
        bottleneck=args.bottleneck,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        batch_size=32,
        patience=args.patience,
        threshold_sigma=args.sigma,
        seed=args.seed,
    )

    detector = AnomalyDetector(model_dir=args.output, device=args.device)

    t0 = time.time()
    history = detector.train(features, config=config, verbose=True)
    elapsed = time.time() - t0

    stats = history["final_stats"]
    logger.info(f"Training completed in {elapsed:.1f}s")
    logger.info(f"  Epochs:     {stats['epochs_trained']}")
    logger.info(f"  Best val:   {stats['best_val_loss']:.6f}")
    logger.info(f"  Threshold:  {stats['threshold']:.6f}")
    logger.info(f"  Mean error: {stats['mean_error']:.6f}")
    logger.info(f"  Std error:  {stats['std_error']:.6f}")

    # ---- Step 3: Score training set ----

    results = detector.predict_batch(features)
    n_anomalous = sum(1 for r in results if r.is_anomalous)
    confidence_dist = {}
    for r in results:
        confidence_dist[r.confidence] = confidence_dist.get(r.confidence, 0) + 1

    logger.info(f"\nTraining set analysis ({len(results)} images):")
    logger.info(f"  Anomalous:  {n_anomalous} ({100*n_anomalous/len(results):.1f}%)")
    for conf, count in sorted(confidence_dist.items()):
        logger.info(f"  {conf}: {count}")

    if valid_paths and n_anomalous > 0:
        logger.info("\n  Top anomalous images:")
        scored = [(r.anomaly_score, r.z_score, p) for r, p in zip(results, valid_paths)]
        scored.sort(reverse=True)
        for score, z, path in scored[:5]:
            logger.info(f"    z={z:+.2f}  score={score:.6f}  {path}")

    # ---- Step 4: Evaluate on separate set (optional) ----

    if args.eval_dir:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Evaluating on {args.eval_dir}")
        eval_images = find_images(args.eval_dir)
        if not eval_images:
            logger.warning(f"No images found in {args.eval_dir}")
        else:
            extractor = FeatureExtractor(device=args.device, cache_dir=args.cache)
            eval_feats, eval_paths, eval_skipped = extractor.extract_batch(
                eval_images, batch_size=args.batch_size,
            )
            extractor.unload()

            if eval_feats.shape[0] > 0:
                eval_results = detector.predict_batch(eval_feats)
                n_anom = sum(1 for r in eval_results if r.is_anomalous)
                logger.info(f"  {len(eval_results)} images evaluated")
                logger.info(f"  Anomalous: {n_anom} ({100*n_anom/len(eval_results):.1f}%)")

                for r, p in zip(eval_results, eval_paths):
                    flag = " *** ANOMALOUS ***" if r.is_anomalous else ""
                    logger.info(
                        f"  [{r.confidence:>18s}] z={r.z_score:+6.2f} "
                        f"score={r.anomaly_score:.6f}  {os.path.basename(p)}{flag}"
                    )

    # ---- Summary ----

    logger.info(f"\n{'=' * 60}")
    logger.info("Files saved:")
    logger.info(f"  Model:    {args.output}/autoencoder.pt")
    logger.info(f"  Stats:    {args.output}/stats.json")
    logger.info(f"  Errors:   {args.output}/train_errors.npy")
    logger.info(f"  Features: {args.output}/features.npy")
    logger.info(f"\nTo use in inference:")
    logger.info(f"  from neuronest.anomaly import FeatureExtractor, AnomalyDetector")
    logger.info(f"  extractor = FeatureExtractor(device='cuda')")
    logger.info(f"  detector = AnomalyDetector(model_dir='{args.output}')")
    logger.info(f"  detector.load()")
    logger.info(f"  feat = extractor.extract_single('room.jpg')")
    logger.info(f"  result = detector.predict(feat)")
    logger.info(f"  print(result.confidence, result.z_score)")


if __name__ == "__main__":
    main()
