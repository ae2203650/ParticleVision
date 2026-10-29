"""
Train the microplastic detector (FinesDetector) of the corrected_v1 models, exactly as in notebooks/39 cells 13-16
(training/notebook_39_training_record.ipynb).

    python training/train_detector.py --raw-dir <.../InstSeg/Raw> --seg-dir <.../InstSeg/Segmented> --out <new folder>

  Data: the 79 training / 16 validation images, 512 x 512 patches with 64 px overlap, the same augmentation as the
  U-Net. Target: a Gaussian heatmap (sigma 2 px, cut at 8 px) at the centroid of every labelled microplastic blob of
  at least 3 px. Model: FinesDetector(base_channels=32), trained from scratch. Loss: CenterNet focal loss
  (alpha 2, beta 4). AdamW lr 3e-4 / weight decay 1e-4, CosineAnnealingWarmRestarts (T_0 30, T_mult 1,
  eta_min 1e-6), batch 8, 30 epochs, early stopping after 10 epochs without a lower validation loss. The best epoch
  (lowest validation loss) is <out>/fines_detector_corrected_best.pth.

No random seed is set (as in the original run). --smoke runs 2 images, 2 batches and 1 epoch, only to show the
script works.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (also puts the repository root on sys.path)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from scipy.ndimage import find_objects, label  # noqa: E402

from src.dataset import SandPatchDataset  # noqa: E402

D = common.DETECTOR


# ── From notebooks/39 cell 13 (unchanged) ─────────────────────────────────────────────────────────────────────────
def generate_centroid_heatmap(mask, sigma=2.0, min_blob_px=3):
    H, W = mask.shape
    heatmap = np.zeros((H, W), dtype=np.float32)
    fines_mask = (mask == 3).astype(np.uint8)
    labeled, n_blobs = label(fines_mask)
    if n_blobs == 0:
        return heatmap[np.newaxis, ...]
    slices = find_objects(labeled)
    radius = int(np.ceil(4.0 * sigma))
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        blob_mask = labeled[sl] == (i + 1)
        blob_size = int(blob_mask.sum())
        if blob_size < min_blob_px:
            continue
        y_idx, x_idx = np.nonzero(blob_mask)
        if len(y_idx) == 0:
            continue
        cy = float(y_idx.mean() + sl[0].start)
        cx = float(x_idx.mean() + sl[1].start)
        y0 = max(0, int(cy) - radius)
        y1 = min(H, int(cy) + radius + 1)
        x0 = max(0, int(cx) - radius)
        x1 = min(W, int(cx) + radius + 1)
        yy = np.arange(y0, y1, dtype=np.float32) - cy
        xx = np.arange(x0, x1, dtype=np.float32) - cx
        gaussian = np.exp(-(yy[:, None] ** 2 + xx[None, :] ** 2) / (2.0 * sigma * sigma))
        patch = heatmap[y0:y1, x0:x1]
        np.maximum(patch, gaussian, out=patch)
    return heatmap[np.newaxis, ...]


class FinesHeatmapDataset(SandPatchDataset):
    def __init__(self, image_indices, transform=None, patch_size=None, overlap=None, sigma=2.0, min_blob_px=3):
        super().__init__(image_indices, transform, patch_size, overlap)
        self.sigma = sigma
        self.min_blob_px = min_blob_px

    def __getitem__(self, index):
        img = self.img_patches[index]
        mask = self.mask_patches[index]
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_aug = augmented["mask"]
        mask_np = mask_aug.numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
        heatmap = generate_centroid_heatmap(mask_np, self.sigma, self.min_blob_px)
        heatmap_t = torch.from_numpy(heatmap).float()
        return img_t, heatmap_t


class CenterNetFocalLoss(nn.Module):
    def __init__(self, alpha=2.0, beta=4.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, pred_logits, target):
        pred = torch.sigmoid(pred_logits)
        pred = pred.clamp(1e-6, 1 - 1e-6)
        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()
        pos_loss = -((1 - pred) ** self.alpha) * torch.log(pred) * pos_mask
        neg_weight = (1 - target) ** self.beta
        neg_loss = -(pred**self.alpha) * torch.log(1 - pred) * neg_weight * neg_mask
        num_pos = pos_mask.sum().clamp(min=1)
        return (pos_loss.sum() + neg_loss.sum()) / num_pos


# ── Training loop, as notebooks/39 cells 14-16 ────────────────────────────────────────────────────────────────────
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    common.add_common_args(ap)
    a = ap.parse_args(argv)

    common.apply_settings(a.raw_dir, a.seg_dir)
    out = common.prepare_out_dir(a.out)
    split_path = common.load_split(a.split, a.smoke, out)

    from torch.utils.data import DataLoader

    from src.dataset import get_train_augmentation, get_val_augmentation
    from src.model import FinesDetector

    device = common.pick_device(a.device)
    print(f"Device: {device}")
    with open(split_path, encoding="utf-8") as f:
        split = json.load(f)
    kw = dict(patch_size=512, overlap=64, sigma=D["sigma"], min_blob_px=D["min_blob_px"])
    train_ds = FinesHeatmapDataset(split["train"], transform=get_train_augmentation(), **kw)
    val_ds = FinesHeatmapDataset(split["val"], transform=get_val_augmentation(), **kw)
    if a.smoke:
        train_ds, val_ds = common.limit(train_ds, 2, D["batch_size"]), common.limit(val_ds, 2, D["batch_size"])
    train_loader = DataLoader(train_ds, batch_size=D["batch_size"], shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=D["batch_size"], shuffle=False, num_workers=0, pin_memory=False)
    print(f"Train: {len(train_ds)} patches ({len(train_loader)} batches)")
    print(f"Val:   {len(val_ds)} patches ({len(val_loader)} batches)")

    model = FinesDetector(base_channels=D["base_channels"]).to(device)
    criterion = CenterNetFocalLoss(alpha=D["focal_alpha"], beta=D["focal_beta"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=D["lr"], weight_decay=D["weight_decay"])
    epochs = D["epochs"] if not a.smoke else 1
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=D["epochs"], T_mult=1, eta_min=D["eta_min"]
    )

    def run_epoch(loader, train):
        model.train(train)
        total = 0.0
        if device.type == "mps":
            torch.mps.empty_cache()
        with torch.set_grad_enabled(train):
            for i, (imgs, heatmaps) in enumerate(loader):
                imgs, heatmaps = imgs.to(device), heatmaps.to(device)
                loss = criterion(model(imgs), heatmaps)
                if train:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                total += loss.item()
                if device.type == "mps" and i % 100 == 0:
                    torch.mps.empty_cache()
        return total / len(loader)

    history = {"train_loss": [], "val_loss": []}
    best, no_improve = float("inf"), 0
    for epoch in range(epochs):
        t0 = time.time()
        tr, vl = run_epoch(train_loader, True), run_epoch(val_loader, False)
        scheduler.step()
        history["train_loss"].append(tr)
        history["val_loss"].append(vl)
        lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch + 1:2d}/{epochs} | Loss: {tr:.4f}/{vl:.4f} | LR: {lr:.2e} | {time.time() - t0:.1f}s")
        torch.save(model.state_dict(), os.path.join(out, f"fines_detector_corrected_e{epoch + 1:02d}.pth"))
        if vl < best:
            best, no_improve = vl, 0
            torch.save(model.state_dict(), os.path.join(out, "fines_detector_corrected_best.pth"))
            print(f"  [+] New best (val_loss: {best:.4f})")
        else:
            no_improve += 1
        if no_improve >= D["patience"]:
            print(f"\nEarly stopping after {epoch + 1} epochs.")
            break
    with open(os.path.join(out, "fines_detector_corrected_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"\nBest val loss: {best:.4f}. Final model: {os.path.join(out, 'fines_detector_corrected_best.pth')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
