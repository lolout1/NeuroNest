"""Masked Autoencoder domain adaptation for ViT-Large.

Fine-tunes a ViT-MAE-Large model on indoor room images using masked image
modeling. Only the last N encoder layers and the full decoder are unfrozen,
keeping the bulk of pretrained knowledge intact while adapting to care
environment visual patterns.

Designed for RTX 2080 (8GB VRAM): FP16 mixed precision, batch_size=4,
224x224 images. MAE masks 75% of patches, so effective compute is ~25%
of a full forward pass.
"""

import gc
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

logger = logging.getLogger(__name__)

MAE_MODEL_ID = "facebook/vit-mae-large"


@dataclass
class MAEConfig:
    """Training configuration for MAE domain adaptation."""
    # Architecture
    unfreeze_last_n: int = 4        # Unfreeze last N encoder layers (+ decoder always)
    mask_ratio: float = 0.75        # Fraction of patches to mask

    # Training
    epochs: int = 20
    batch_size: int = 4             # Fits RTX 2080 8GB with FP16
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 2
    min_lr: float = 1e-6

    # Data
    val_split: float = 0.05         # Small val set (MAE loss is noisy)
    num_workers: int = 2
    image_size: int = 224

    # Mixed precision
    fp16: bool = True               # FP16 for GPU, ignored on CPU

    # Saving
    save_every: int = 5             # Checkpoint every N epochs
    seed: int = 42

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


class MAEDomainAdapter:
    """Fine-tunes ViT-MAE-Large on indoor images for domain adaptation.

    Args:
        output_dir: Directory for checkpoints, logs, and adapted weights.
        device: Torch device (auto-detect if None).
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "models/domain_adapted",
        device: Optional[str] = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = torch.device(
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self._model = None

    def train(
        self,
        dataset: Dataset,
        config: Optional[MAEConfig] = None,
        verbose: bool = True,
    ) -> Dict:
        """Train MAE on indoor image dataset.

        Args:
            dataset: Dataset returning (3, H, W) tensors in [0, 1].
            config: Training hyperparameters.
            verbose: Print progress.

        Returns:
            Training history dict with losses and metadata.
        """
        config = config or MAEConfig()
        torch.manual_seed(config.seed)

        if len(dataset) < 20:
            raise ValueError(
                f"Need at least 20 images, got {len(dataset)}. "
                "Add more images to the dataset."
            )

        # --- Load model ---
        from transformers import ViTMAEForPreTraining, ViTMAEConfig as HFMAEConfig

        if verbose:
            logger.info(f"Loading {MAE_MODEL_ID} on {self.device}...")

        model = ViTMAEForPreTraining.from_pretrained(MAE_MODEL_ID)

        # --- Freeze / unfreeze layers ---
        total_layers = len(model.vit.encoder.layer)
        unfreeze_from = total_layers - config.unfreeze_last_n

        # Freeze everything first
        for param in model.parameters():
            param.requires_grad = False

        # Unfreeze last N encoder layers
        for i in range(unfreeze_from, total_layers):
            for param in model.vit.encoder.layer[i].parameters():
                param.requires_grad = True

        # Unfreeze full decoder (always)
        for param in model.decoder.parameters():
            param.requires_grad = True

        # Unfreeze encoder LayerNorm (adapts to new distribution)
        if hasattr(model.vit, "layernorm"):
            for param in model.vit.layernorm.parameters():
                param.requires_grad = True

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        if verbose:
            logger.info(
                f"Parameters: {trainable / 1e6:.1f}M trainable / "
                f"{total / 1e6:.1f}M total "
                f"(layers {unfreeze_from}-{total_layers - 1} + decoder unfrozen)"
            )

        model.to(self.device)

        # --- Data split ---
        n_val = max(2, int(len(dataset) * config.val_split))
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(
            dataset, [n_train, n_val],
            generator=torch.Generator().manual_seed(config.seed),
        )

        train_loader = DataLoader(
            train_ds, batch_size=config.batch_size,
            shuffle=True, num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=config.batch_size,
            shuffle=False, num_workers=config.num_workers,
            pin_memory=(self.device.type == "cuda"),
        )

        if verbose:
            logger.info(f"Data: {n_train} train, {n_val} val")

        # --- Optimizer + scheduler ---
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
        )

        total_steps = len(train_loader) * config.epochs
        warmup_steps = len(train_loader) * config.warmup_epochs

        def _lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
            return max(config.min_lr / config.lr, 0.5 * (1 + np.cos(np.pi * progress)))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

        # --- Mixed precision ---
        use_amp = config.fp16 and self.device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        # --- Pixel normalization (ImageNet stats for ViT) ---
        imagenet_mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        imagenet_std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)

        # --- Training loop ---
        history = {"train_loss": [], "val_loss": [], "lr": [], "epoch_time": []}
        best_val_loss = float("inf")
        best_epoch = 0

        if verbose:
            logger.info(
                f"Training: {config.epochs} epochs, batch_size={config.batch_size}, "
                f"lr={config.lr}, fp16={use_amp}"
            )

        for epoch in range(config.epochs):
            t0 = time.time()

            # Train
            model.train()
            train_loss_sum = 0.0
            train_steps = 0

            for batch in train_loader:
                pixel_values = batch.to(self.device, non_blocking=True)
                # Normalize to ImageNet stats
                pixel_values = (pixel_values - imagenet_mean) / imagenet_std

                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(pixel_values=pixel_values)
                    loss = outputs.loss

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                train_loss_sum += loss.item()
                train_steps += 1

            avg_train = train_loss_sum / max(train_steps, 1)

            # Validate
            model.eval()
            val_loss_sum = 0.0
            val_steps = 0
            with torch.inference_mode():
                for batch in val_loader:
                    pixel_values = batch.to(self.device, non_blocking=True)
                    pixel_values = (pixel_values - imagenet_mean) / imagenet_std
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        outputs = model(pixel_values=pixel_values)
                    val_loss_sum += outputs.loss.item()
                    val_steps += 1

            avg_val = val_loss_sum / max(val_steps, 1)
            elapsed = time.time() - t0
            current_lr = scheduler.get_last_lr()[0]

            history["train_loss"].append(avg_train)
            history["val_loss"].append(avg_val)
            history["lr"].append(current_lr)
            history["epoch_time"].append(elapsed)

            # Save best
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_epoch = epoch
                self._save_checkpoint(model, epoch, avg_val, config, is_best=True)

            # Periodic save
            if (epoch + 1) % config.save_every == 0:
                self._save_checkpoint(model, epoch, avg_val, config)

            if verbose:
                logger.info(
                    f"Epoch {epoch + 1}/{config.epochs}: "
                    f"train={avg_train:.4f}, val={avg_val:.4f}, "
                    f"lr={current_lr:.2e}, time={elapsed:.1f}s"
                    + (" *best*" if epoch == best_epoch else "")
                )

            # Clear cache periodically
            if self.device.type == "cuda" and (epoch + 1) % 5 == 0:
                torch.cuda.empty_cache()

        # --- Save final + metadata ---
        self._save_checkpoint(model, config.epochs - 1, avg_val, config, is_final=True)

        metadata = {
            "model_id": MAE_MODEL_ID,
            "device": str(self.device),
            "dataset_size": len(dataset),
            "best_val_loss": best_val_loss,
            "best_epoch": best_epoch,
            "total_epochs": config.epochs,
            "trainable_params": trainable,
            "total_params": total,
            "config": config.to_dict(),
        }
        with open(self.output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        if verbose:
            logger.info(
                f"Training complete. Best val loss: {best_val_loss:.4f} "
                f"at epoch {best_epoch + 1}. Saved to {self.output_dir}"
            )

        self._model = model
        history["metadata"] = metadata
        return history

    def _save_checkpoint(
        self, model, epoch, val_loss, config, is_best=False, is_final=False,
    ):
        """Save encoder weights (not decoder — we only need the backbone)."""
        # Save only the ViT encoder state dict (what transfers to EoMT)
        encoder_state = {
            k: v.cpu() for k, v in model.vit.state_dict().items()
        }

        if is_best:
            torch.save(encoder_state, self.output_dir / "best_encoder.pt")
        if is_final:
            torch.save(encoder_state, self.output_dir / "final_encoder.pt")

        # Also save full model for potential continued training
        if is_best or is_final:
            info = {
                "epoch": epoch,
                "val_loss": val_loss,
                "config": config.to_dict(),
            }
            tag = "best" if is_best else "final"
            with open(self.output_dir / f"{tag}_info.json", "w") as f:
                json.dump(info, f, indent=2)

    def get_adapted_encoder(self) -> dict:
        """Return the adapted encoder state dict for weight transfer."""
        best_path = self.output_dir / "best_encoder.pt"
        if best_path.exists():
            return torch.load(best_path, map_location="cpu", weights_only=True)
        if self._model is not None:
            return {k: v.cpu() for k, v in self._model.vit.state_dict().items()}
        raise RuntimeError(
            f"No adapted encoder found. Train first or check {self.output_dir}"
        )
