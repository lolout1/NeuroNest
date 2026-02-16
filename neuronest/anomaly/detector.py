"""Autoencoder-based anomaly detector for room images.

Trains on DINOv2-Large CLS features from "normal" rooms. At inference, high
reconstruction error signals an anomalous/hazardous room layout. The architecture
is deliberately compact (1024→256→64→256→1024) since the input features are
already highly compressed by DINOv2.

Training is fast (~2-5 minutes on GPU, ~10-15 minutes on CPU for 500 images)
because we operate on pre-extracted 1024-dim features, not raw pixels.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .extractor import FEATURE_DIM

logger = logging.getLogger(__name__)

# --- Model Architecture ---


class _Encoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, bottleneck),
        )

    def forward(self, x):
        return self.net(x)


class _Decoder(nn.Module):
    def __init__(self, bottleneck: int, output_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(bottleneck, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class _RoomAutoencoder(nn.Module):
    """Symmetric autoencoder: 1024 → 256 → 128 → 64 → 128 → 256 → 1024.

    Uses LayerNorm (not BatchNorm) so it works with batch_size=1 at inference.
    GELU activation for smooth gradients. Dropout for regularization.
    """

    def __init__(
        self,
        input_dim: int = FEATURE_DIM,
        bottleneck: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = _Encoder(input_dim, bottleneck, dropout)
        self.decoder = _Decoder(bottleneck, input_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


# --- Data Result ---


@dataclass
class AnomalyResult:
    """Result from anomaly detection on a single image."""
    anomaly_score: float         # Mean squared reconstruction error
    is_anomalous: bool           # Score exceeds threshold
    z_score: float               # Standard deviations above training mean
    percentile: float            # Approximate percentile rank (0-100)
    confidence: str              # "normal", "borderline", "anomalous", "highly_anomalous"

    def to_dict(self) -> Dict:
        return {
            "anomaly_score": round(self.anomaly_score, 6),
            "is_anomalous": self.is_anomalous,
            "z_score": round(self.z_score, 2),
            "percentile": round(self.percentile, 1),
            "confidence": self.confidence,
        }


# --- Training Configuration ---


@dataclass
class TrainConfig:
    """Training hyperparameters."""
    bottleneck: int = 64
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 200
    batch_size: int = 32
    patience: int = 30           # Early stopping patience
    val_split: float = 0.15      # Validation fraction
    threshold_sigma: float = 2.0 # Anomaly threshold = mean + sigma * std
    seed: int = 42

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# --- Detector ---


class AnomalyDetector:
    """Room anomaly detector: trains on normal room features, detects outliers.

    Args:
        model_dir: Directory for saving/loading model checkpoints and stats.
        device: Torch device (auto-detect if None).
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models/anomaly",
        device: Optional[str] = None,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model: Optional[_RoomAutoencoder] = None
        self._stats: Optional[Dict] = None  # threshold, mean, std, percentiles

    # --- Training ---

    def train(
        self,
        features: np.ndarray,
        config: Optional[TrainConfig] = None,
        verbose: bool = True,
    ) -> Dict:
        """Train the autoencoder on room feature vectors.

        Args:
            features: (N, 1024) float32 array of DINOv2 CLS features.
            config: Training hyperparameters (uses defaults if None).
            verbose: Print training progress.

        Returns:
            Dict with training history and final metrics.

        Raises:
            ValueError: If features are invalid (wrong shape, too few, non-finite).
        """
        config = config or TrainConfig()

        # --- Validate input ---
        if features.ndim != 2 or features.shape[1] != FEATURE_DIM:
            raise ValueError(
                f"Expected (N, {FEATURE_DIM}) features, got {features.shape}"
            )
        if not np.isfinite(features).all():
            nan_count = (~np.isfinite(features)).any(axis=1).sum()
            logger.warning(f"Removing {nan_count} samples with non-finite values")
            mask = np.isfinite(features).all(axis=1)
            features = features[mask]

        n_samples = features.shape[0]
        if n_samples < 10:
            raise ValueError(
                f"Need at least 10 samples for training, got {n_samples}. "
                "Collect more room images."
            )

        # --- Split train/val ---
        rng = np.random.RandomState(config.seed)
        indices = rng.permutation(n_samples)
        n_val = max(2, int(n_samples * config.val_split))
        val_idx = indices[:n_val]
        train_idx = indices[n_val:]

        train_feats = torch.tensor(features[train_idx], dtype=torch.float32)
        val_feats = torch.tensor(features[val_idx], dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(train_feats),
            batch_size=min(config.batch_size, len(train_feats)),
            shuffle=True,
            drop_last=False,
        )

        if verbose:
            logger.info(
                f"Training anomaly detector: {len(train_feats)} train, "
                f"{len(val_feats)} val, {config.epochs} max epochs, "
                f"device={self.device}"
            )

        # --- Initialize model ---
        model = _RoomAutoencoder(
            input_dim=FEATURE_DIM,
            bottleneck=config.bottleneck,
            dropout=config.dropout,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.epochs, eta_min=config.lr * 0.01,
        )
        criterion = nn.MSELoss()

        # --- Training loop with early stopping ---
        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "lr": []}

        for epoch in range(config.epochs):
            # Train
            model.train()
            train_loss_sum = 0.0
            train_batches = 0
            for (batch,) in train_loader:
                batch = batch.to(self.device)
                recon = model(batch)
                loss = criterion(recon, batch)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_loss_sum += loss.item()
                train_batches += 1

            scheduler.step()
            avg_train = train_loss_sum / train_batches

            # Validate
            model.eval()
            with torch.inference_mode():
                val_batch = val_feats.to(self.device)
                val_recon = model(val_batch)
                avg_val = criterion(val_recon, val_batch).item()

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["lr"].append(scheduler.get_last_lr()[0])

            # Early stopping
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch + 1) % 20 == 0:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs}: "
                    f"train={avg_train:.6f}, val={avg_val:.6f}, "
                    f"best_val={best_val_loss:.6f}, patience={patience_counter}/{config.patience}"
                )

            if patience_counter >= config.patience:
                if verbose:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                break

        # --- Load best model ---
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()

        # --- Compute threshold statistics on FULL training set ---
        all_feats = torch.tensor(features, dtype=torch.float32).to(self.device)
        with torch.inference_mode():
            recon = model(all_feats)
            errors = ((all_feats - recon) ** 2).mean(dim=1).cpu().numpy()

        mean_error = float(errors.mean())
        std_error = float(errors.std())
        threshold = mean_error + config.threshold_sigma * std_error

        # Percentile lookup table (for fast percentile estimation at inference)
        percentile_values = np.percentile(errors, np.arange(0, 101, 1)).tolist()

        stats = {
            "threshold": threshold,
            "mean_error": mean_error,
            "std_error": std_error,
            "sigma": config.threshold_sigma,
            "n_samples": n_samples,
            "n_train": len(train_feats),
            "n_val": len(val_feats),
            "best_val_loss": best_val_loss,
            "epochs_trained": epoch + 1,
            "percentile_values": percentile_values,
            "config": config.to_dict(),
        }

        # --- Save ---
        self.model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_dir / "autoencoder.pt")
        with open(self.model_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=2)
        np.save(self.model_dir / "train_errors.npy", errors)

        self._model = model
        self._stats = stats

        if verbose:
            logger.info(
                f"Training complete. Threshold={threshold:.6f} "
                f"(mean={mean_error:.6f} + {config.threshold_sigma}*{std_error:.6f}). "
                f"Saved to {self.model_dir}"
            )

        history["final_stats"] = stats
        return history

    # --- Loading ---

    def load(self, model_dir: Optional[Union[str, Path]] = None) -> bool:
        """Load a trained model from disk.

        Returns True if loaded successfully, False otherwise.
        """
        model_dir = Path(model_dir) if model_dir else self.model_dir
        weights_path = model_dir / "autoencoder.pt"
        stats_path = model_dir / "stats.json"

        if not weights_path.exists() or not stats_path.exists():
            logger.warning(f"No trained model found at {model_dir}")
            return False

        try:
            with open(stats_path) as f:
                self._stats = json.load(f)

            config = self._stats.get("config", {})
            model = _RoomAutoencoder(
                input_dim=FEATURE_DIM,
                bottleneck=config.get("bottleneck", 64),
                dropout=config.get("dropout", 0.1),
            )
            model.load_state_dict(
                torch.load(weights_path, map_location="cpu", weights_only=True)
            )
            model.to(self.device)
            model.eval()
            self._model = model
            self.model_dir = model_dir

            logger.info(
                f"Loaded anomaly detector from {model_dir}: "
                f"threshold={self._stats['threshold']:.6f}, "
                f"trained on {self._stats['n_samples']} samples"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    # --- Inference ---

    @torch.inference_mode()
    def predict(self, feature: np.ndarray) -> AnomalyResult:
        """Score a single feature vector for anomaly.

        Args:
            feature: (1024,) float32 numpy array (L2-normalized DINOv2 CLS).

        Returns:
            AnomalyResult with score, z-score, percentile, and classification.

        Raises:
            RuntimeError: If no model is loaded.
        """
        if self._model is None or self._stats is None:
            raise RuntimeError(
                "No model loaded. Call train() or load() first."
            )

        if feature.shape != (FEATURE_DIM,):
            raise ValueError(f"Expected ({FEATURE_DIM},) feature, got {feature.shape}")

        feat_t = torch.tensor(feature, dtype=torch.float32).unsqueeze(0).to(self.device)
        recon = self._model(feat_t)
        error = float(((feat_t - recon) ** 2).mean().cpu())

        threshold = self._stats["threshold"]
        mean_err = self._stats["mean_error"]
        std_err = self._stats["std_error"]

        z_score = (error - mean_err) / std_err if std_err > 1e-12 else 0.0

        # Percentile from precomputed lookup
        percentiles = self._stats.get("percentile_values", [])
        if percentiles:
            percentile = float(np.searchsorted(percentiles, error))
        else:
            percentile = min(100.0, max(0.0, 50.0 + z_score * 15.87))

        # Confidence classification
        if z_score <= 1.0:
            confidence = "normal"
        elif z_score <= 2.0:
            confidence = "borderline"
        elif z_score <= 3.0:
            confidence = "anomalous"
        else:
            confidence = "highly_anomalous"

        return AnomalyResult(
            anomaly_score=error,
            is_anomalous=error > threshold,
            z_score=z_score,
            percentile=percentile,
            confidence=confidence,
        )

    @torch.inference_mode()
    def predict_batch(self, features: np.ndarray) -> List[AnomalyResult]:
        """Score multiple feature vectors.

        Args:
            features: (N, 1024) float32 array.

        Returns:
            List of AnomalyResult, one per input.
        """
        if self._model is None or self._stats is None:
            raise RuntimeError("No model loaded. Call train() or load() first.")

        feats_t = torch.tensor(features, dtype=torch.float32).to(self.device)
        recon = self._model(feats_t)
        errors = ((feats_t - recon) ** 2).mean(dim=1).cpu().numpy()

        threshold = self._stats["threshold"]
        mean_err = self._stats["mean_error"]
        std_err = self._stats["std_error"]
        percentiles = self._stats.get("percentile_values", [])

        results = []
        for error in errors:
            error = float(error)
            z = (error - mean_err) / std_err if std_err > 1e-12 else 0.0
            pct = float(np.searchsorted(percentiles, error)) if percentiles else 50.0 + z * 15.87
            pct = min(100.0, max(0.0, pct))

            if z <= 1.0:
                conf = "normal"
            elif z <= 2.0:
                conf = "borderline"
            elif z <= 3.0:
                conf = "anomalous"
            else:
                conf = "highly_anomalous"

            results.append(AnomalyResult(
                anomaly_score=error,
                is_anomalous=error > threshold,
                z_score=z,
                percentile=pct,
                confidence=conf,
            ))
        return results

    @property
    def is_loaded(self) -> bool:
        return self._model is not None and self._stats is not None

    @property
    def stats(self) -> Optional[Dict]:
        return self._stats

    def unload(self):
        """Free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        self._stats = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
