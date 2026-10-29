"""
ParticleVision — Training Loop
================================
Staged training with:
  - Per-epoch train/val loss and metrics
  - Early stopping on validation mIoU
  - Checkpoint saving (best + last)
  - Cross-stage metric comparison
"""

import gc
import os
import time
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .config import CFG
from .metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Detect and return the best available device (MPS > CUDA > CPU)."""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ Using Apple MPS (Metal Performance Shaders)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"✅ Using CUDA GPU")
    else:
        device = torch.device("cpu")
        print(f"⚠️  Using CPU (no GPU detected)")
    return device


# ---------------------------------------------------------------------------
# Single epoch: train / validate
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: SegmentationMetrics,
) -> Tuple[float, Dict]:
    """
    Train for one epoch.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict with per-class IoU etc.
    """
    model.train()
    metrics.reset()
    running_loss = 0.0
    n_batches = 0
    is_mps = device.type == 'mps'

    for images, masks in tqdm(loader, desc="  Train", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

        # Accumulate metrics
        with torch.no_grad():
            metrics.update_torch(logits, masks)

        # Explicitly release GPU tensors to prevent MPS memory buildup
        del images, masks, logits, loss

        # Periodic MPS cache clearing to prevent intra-epoch accumulation
        if is_mps and n_batches % 20 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    avg_loss = running_loss / max(n_batches, 1)

    # Full cleanup between epochs
    if is_mps:
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    return avg_loss, metrics.summary()


def train_one_epoch_grad_accum(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: SegmentationMetrics,
    accum_steps: int = 4,
) -> Tuple[float, Dict]:
    """
    Train for one epoch with gradient accumulation.

    Parameters
    ----------
    accum_steps : int
        Number of mini-batches to accumulate before an optimizer step.
        Effective batch size = loader.batch_size * accum_steps.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict with per-class IoU etc.
    """
    model.train()
    metrics.reset()
    running_loss = 0.0
    n_batches = 0
    is_mps = device.type == 'mps'

    optimizer.zero_grad()

    for batch_idx, (images, masks) in enumerate(tqdm(loader, desc="  Train", leave=False)):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        # Scale loss by accumulation steps so the total gradient
        # magnitude is the same as a single large batch
        loss_scaled = loss / accum_steps
        loss_scaled.backward()

        running_loss += loss.item()
        n_batches += 1

        # Accumulate metrics
        with torch.no_grad():
            metrics.update_torch(logits, masks)

        # Explicitly release GPU tensors
        del images, masks, logits, loss, loss_scaled

        # Step optimizer every accum_steps batches (or at the end)
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            optimizer.step()
            optimizer.zero_grad()

        # Periodic MPS cache clearing
        if is_mps and n_batches % 20 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    avg_loss = running_loss / max(n_batches, 1)

    # Full cleanup between epochs
    if is_mps:
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    return avg_loss, metrics.summary()



@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    metrics: SegmentationMetrics,
) -> Tuple[float, Dict]:
    """
    Validate for one epoch.

    Returns
    -------
    avg_loss : float
    metrics_dict : dict
    """
    model.eval()
    metrics.reset()
    running_loss = 0.0
    n_batches = 0
    is_mps = device.type == 'mps'

    for images, masks in tqdm(loader, desc="  Val  ", leave=False):
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images)
        loss = criterion(logits, masks)

        running_loss += loss.item()
        n_batches += 1
        metrics.update_torch(logits, masks)

        # Explicitly release GPU tensors
        del images, masks, logits, loss

        # Periodic MPS cache clearing
        if is_mps and n_batches % 20 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    avg_loss = running_loss / max(n_batches, 1)

    # Full cleanup between epochs
    if is_mps:
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    return avg_loss, metrics.summary()


# ---------------------------------------------------------------------------
# Full training stage
# ---------------------------------------------------------------------------

def run_training_stage(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    stage_name: str,
    num_epochs: int,
    learning_rate: float = None,
    checkpoint_dir: str = None,
    patience: int = None,
    optimizer: torch.optim.Optimizer = None,
    accum_steps: int = 1,
    early_stop_metric: str = 'miou',
    save_every_epoch: bool = False,
) -> Dict:
    """
    Run one training stage.

    Parameters
    ----------
    optimizer : torch.optim.Optimizer, optional
        If provided, use this optimizer (e.g. with differential LR groups).
        If None, a default AdamW is created with the given learning_rate.
    accum_steps : int
        Gradient accumulation steps (default 1 = no accumulation).
        Effective batch size = loader.batch_size * accum_steps.
    early_stop_metric : str
        Metric to use for early stopping. 'miou' (default) or 'fines_iou'.
    save_every_epoch : bool
        If True, additionally save a lightweight (model-weights-only)
        checkpoint after every epoch as "{stage_name}_e{epoch:02d}.pth" —
        independent of the best/last checkpoints below. Default False
        preserves prior behavior for all existing callers.

    Returns
    -------
    history : dict with keys 'train_loss', 'val_loss', 'train_metrics',
              'val_metrics', 'best_epoch', 'best_val_miou'.
    """
    if learning_rate is None:
        learning_rate = CFG.learning_rate
    if checkpoint_dir is None:
        checkpoint_dir = CFG.checkpoint_dir
    if patience is None:
        patience = CFG.early_stopping_patience

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Optimizer & scheduler
    if optimizer is None:
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate,
            weight_decay=CFG.weight_decay,
        )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=CFG.scheduler_T_0, T_mult=CFG.scheduler_T_mult
    )

    # Metrics accumulators
    train_metrics_acc = SegmentationMetrics()
    val_metrics_acc = SegmentationMetrics()

    # History
    history = {
        "stage": stage_name,
        "train_loss": [],
        "val_loss": [],
        "train_metrics": [],
        "val_metrics": [],
        "best_epoch": 0,
        "best_val_miou": 0.0,
        "lr_history": [],
    }

    best_val_miou = 0.0
    best_model_state = None
    epochs_no_improve = 0

    print(f"\n{'='*65}")
    print(f"  Stage: {stage_name}")
    print(f"  Epochs: {num_epochs}, LR: {learning_rate}, Patience: {patience}")
    print(f"  Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    print(f"{'='*65}")

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()

        # ── Memory monitoring ──
        if device.type == 'mps':
            mem_alloc = torch.mps.current_allocated_memory() / (1024**2)
            mem_driver = torch.mps.driver_allocated_memory() / (1024**2)
            print(f"  [MPS Mem] Allocated: {mem_alloc:.0f}MB, "
                  f"Driver: {mem_driver:.0f}MB")

        # Train
        if accum_steps > 1:
            train_loss, train_m = train_one_epoch_grad_accum(
                model, train_loader, criterion, optimizer, device,
                train_metrics_acc, accum_steps=accum_steps
            )
        else:
            train_loss, train_m = train_one_epoch(
                model, train_loader, criterion, optimizer, device, train_metrics_acc
            )

        # Force cleanup between train and validate
        if device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()

        # Validate
        val_loss, val_m = validate(
            model, val_loader, criterion, device, val_metrics_acc
        )

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        # Record history
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_metrics"].append(train_m)
        history["val_metrics"].append(val_m)
        history["lr_history"].append(current_lr)

        elapsed = time.time() - t0
        val_miou = val_m["mean_iou"]
        fines_iou = val_m.get("Fines_iou", 0.0)

        # Print epoch summary
        print(
            f"  Epoch {epoch:3d}/{num_epochs} │ "
            f"Loss: {train_loss:.4f}/{val_loss:.4f} │ "
            f"mIoU: {val_miou:.4f} │ "
            f"Fines IoU: {fines_iou:.4f} │ "
            f"LR: {current_lr:.2e} │ "
            f"{elapsed:.1f}s"
        )

        # Early stopping check — use the configured metric
        if early_stop_metric == 'fines_iou':
            tracked_value = fines_iou
        else:
            tracked_value = val_miou

        if tracked_value > best_val_miou:
            best_val_miou = tracked_value
            history["best_epoch"] = epoch
            history["best_val_miou"] = best_val_miou
            # Move state dict to CPU to prevent MPS memory accumulation
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_no_improve = 0

            # Save best checkpoint — move optimizer state to CPU too
            ckpt_path = os.path.join(checkpoint_dir, f"{stage_name}_best.pth")
            opt_state_cpu = {
                k: {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
                     for kk, vv in v.items()} if isinstance(v, dict) else v
                for k, v in optimizer.state_dict()['state'].items()
            }
            torch.save({
                "epoch": epoch,
                "model_state_dict": best_model_state,
                "optimizer_state_dict": {
                    'state': opt_state_cpu,
                    'param_groups': optimizer.state_dict()['param_groups'],
                },
                "val_miou": best_val_miou,
                "val_metrics": val_m,
                "stage": stage_name,
            }, ckpt_path)
            del opt_state_cpu
        else:
            epochs_no_improve += 1

        if save_every_epoch:
            epoch_ckpt_path = os.path.join(
                checkpoint_dir, f"{stage_name}_e{epoch:02d}.pth"
            )
            torch.save(
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                epoch_ckpt_path,
            )

        # ── End-of-EVERY-epoch cleanup (not just best) ──
        if device.type == 'mps':
            torch.mps.synchronize()
            torch.mps.empty_cache()
        gc.collect()

        if epochs_no_improve >= patience:
            print(f"  ⏹ Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs)")
            break

    # Save last checkpoint (all tensors on CPU)
    last_path = os.path.join(checkpoint_dir, f"{stage_name}_last.pth")
    last_model_state = {
        k: v.cpu().clone() for k, v in model.state_dict().items()
    }
    last_opt_state = {
        k: {kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
             for kk, vv in v.items()} if isinstance(v, dict) else v
        for k, v in optimizer.state_dict()['state'].items()
    }
    torch.save({
        "epoch": epoch,
        "model_state_dict": last_model_state,
        "optimizer_state_dict": {
            'state': last_opt_state,
            'param_groups': optimizer.state_dict()['param_groups'],
        },
        "val_miou": val_miou,
        "val_metrics": val_m,
        "stage": stage_name,
    }, last_path)
    del last_model_state, last_opt_state
    if device.type == 'mps':
        torch.mps.synchronize()
        torch.mps.empty_cache()
    gc.collect()

    # Restore best weights
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"  ✅ Restored best model from epoch {history['best_epoch']} "
              f"(val mIoU = {best_val_miou:.4f})")

    # Save history
    history_path = os.path.join(checkpoint_dir, f"{stage_name}_history.json")
    # Convert numpy types to native Python for JSON serialization
    def to_serializable(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, default=to_serializable)

    return history


# ---------------------------------------------------------------------------
# Cross-stage comparison
# ---------------------------------------------------------------------------

def compare_stages(histories: List[Dict]):
    """Print a comparison table across training stages."""
    print(f"\n{'='*75}")
    print(f"  Cross-Stage Comparison")
    print(f"{'='*75}")
    header = (
        f"  {'Stage':<18} {'Best Ep':>7} {'Val mIoU':>9} "
        f"{'Grains':>8} {'Void':>8} {'Water':>8} {'Fines':>8}"
    )
    print(header)
    print(f"  {'─'*69}")

    for h in histories:
        best_ep = h["best_epoch"]
        best_m = h["val_metrics"][best_ep - 1]  # 0-indexed
        print(
            f"  {h['stage']:<18} {best_ep:>7} {best_m['mean_iou']:>9.4f} "
            f"{best_m['Grains_iou']:>8.4f} {best_m['Void_iou']:>8.4f} "
            f"{best_m['Water_iou']:>8.4f} {best_m['Fines_iou']:>8.4f}"
        )
    print(f"{'='*75}\n")
