"""
ParticleVision — Inference Pipeline
=====================================
End-to-end inference on a single full-resolution image:
  1. Load raw image
  2. Dynamically tile
  3. Batch-predict all tiles on GPU
  4. Stitch predictions (logit averaging in overlap zones)
  5. Map class indices → exact RGB colours
  6. Save output .tif + quantitative report
"""

import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from typing import Dict, Optional, Tuple

from .config import CFG
from .tiling import (
    compute_tile_positions,
    stitch_predictions,
    class_map_to_rgb,
)


# ---------------------------------------------------------------------------
# Normalisation (same as albumentations Normalize, but manual for inference)
# ---------------------------------------------------------------------------

def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalise a uint8 (H, W, 3) image to float32 with ImageNet stats."""
    img = img.astype(np.float32) / 255.0
    mean = np.array(CFG.img_mean, dtype=np.float32)
    std = np.array(CFG.img_std, dtype=np.float32)
    img = (img - mean) / std
    return img


# ---------------------------------------------------------------------------
# Test-Time Augmentation (TTA) helpers
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def _tta_transforms():
    """
    Return a list of (forward_fn, inverse_fn) pairs for TTA.
    Each operates on a (C, H, W) numpy array of logits.
    D4 dihedral group: identity, hflip, vflip, rot90.
    """
    return [
        # Identity
        (lambda x: x, lambda x: x),
        # Horizontal flip (flip W axis = axis 2 for CHW, axis 1 for input HWC)
        (lambda x: np.flip(x, axis=2).copy(), lambda x: np.flip(x, axis=2).copy()),
        # Vertical flip (flip H axis = axis 1 for CHW)
        (lambda x: np.flip(x, axis=1).copy(), lambda x: np.flip(x, axis=1).copy()),
        # 90° rotation (rot90 on HW axes)
        (lambda x: np.rot90(x, k=1, axes=(1, 2)).copy(),
         lambda x: np.rot90(x, k=-1, axes=(1, 2)).copy()),
    ]


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def _predict_patch_tta(
    model: nn.Module,
    patch_tensor: torch.Tensor,
    device: torch.device,
) -> np.ndarray:
    """
    Predict a single patch with TTA — average logits over D4 transforms.

    Parameters
    ----------
    patch_tensor : (3, H, W) float32 tensor (already normalised)
    Returns (C, H, W) averaged logit array.
    """
    transforms = _tta_transforms()
    patch_np = patch_tensor.numpy()  # (3, H, W)
    accumulated = None

    for fwd, inv in transforms:
        aug_np = fwd(patch_np)
        aug_t = torch.from_numpy(aug_np).unsqueeze(0).to(device)
        logits = model(aug_t)  # (1, C, H, W)
        logits_np = logits[0].cpu().numpy()  # (C, H, W)
        # Inverse-transform the logits back
        logits_inv = inv(logits_np)
        if accumulated is None:
            accumulated = logits_inv
        else:
            accumulated += logits_inv

    return accumulated / len(transforms)


# ---------------------------------------------------------------------------
# Full inference pipeline
# ---------------------------------------------------------------------------

def predict_full_image(
    model: nn.Module,
    image_path: str,
    device: torch.device,
    batch_size: int = None,
    patch_size: int = None,
    overlap: int = None,
    use_tta: bool = False,
    use_cleanup: bool = False,
    cleanup_min_blob: int = 15,
    cleanup_morph: bool = False,
    use_lab: bool = False,
    model_b2: nn.Module = None,
    fusion_scale: float = 8.0,
    fusion_margin: float = 2.0,
    return_logits: bool = False,
) -> tuple:
    """
    Run inference on a single full-resolution image.

    Parameters
    ----------
    model : trained nn.Module (already on device, in eval mode)
    image_path : path to raw .jpg image
    device : torch.device
    batch_size : int (default: CFG.inference_batch_size)
    patch_size, overlap : int
    use_tta : bool — if True, use test-time augmentation
    use_cleanup : bool — if True, apply post-prediction fines cleanup
    cleanup_min_blob : int — remove fines blobs smaller than this
    cleanup_morph : bool — if True, apply morphological opening to remove halos
    model_b2 : nn.Module — optional Branch 2 model for dual-branch fines detection
    fusion_scale : float — scaling factor for Branch 2 probability
    fusion_margin : float — logit margin for gated fusion

    Returns
    -------
    rgb_output : (H, W, 3) uint8 array — the segmented image in spec colours
    class_map  : (H, W) int64 array — per-pixel class indices
    elapsed    : float — wall-clock seconds for the full pipeline
    """
    if batch_size is None:
        batch_size = CFG.inference_batch_size
    if patch_size is None:
        patch_size = CFG.patch_size
    if overlap is None:
        overlap = CFG.overlap

    t0 = time.time()

    # 1. Load image
    raw = np.array(Image.open(image_path))
    H, W = raw.shape[:2]

    # 2. Compute tile grid
    positions, padded_H, padded_W = compute_tile_positions(
        H, W, patch_size, overlap
    )

    # 3. Pad image
    pad_h = padded_H - H
    pad_w = padded_W - W
    raw_padded = np.pad(
        raw, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
    )

    # 4. Extract & normalise patches
    patches = []
    for y, x in positions:
        patch = raw_padded[y : y + patch_size, x : x + patch_size]
        patch_norm = normalize_image(patch)
        
        # Not used by the ParticleVision pipeline (kept from earlier experiments).
        if use_lab:
            import cv2
            patch_lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
            patch_lab_float = patch_lab.astype(np.float32) / 255.0
            patch_6c = np.concatenate([patch_norm, patch_lab_float], axis=-1)
            patch_tensor = torch.from_numpy(patch_6c.transpose(2, 0, 1)).float()
        else:
            patch_tensor = torch.from_numpy(patch_norm.transpose(2, 0, 1)).float()
            
        patches.append(patch_tensor)

    # 5. Batch predict (Branch 1)
    model.eval()
    all_logits = []
    with torch.no_grad():
        # Not used by the ParticleVision pipeline (kept from earlier experiments).
        if use_tta:
            for patch_t in patches:
                logits = _predict_patch_tta(model, patch_t, device)
                all_logits.append(logits)
        else:
            for i in range(0, len(patches), batch_size):
                batch = torch.stack(patches[i : i + batch_size]).to(device)
                logits = model(batch)  # (B, C, H, W)
                all_logits.extend([l.cpu().numpy() for l in logits])

    # 6. Optional Branch 2 prediction
    b2_heatmap = None
    if model_b2 is not None:
        model_b2.eval()
        heatmap_full = np.zeros((padded_H, padded_W), dtype=np.float32)
        weight_full = np.zeros((padded_H, padded_W), dtype=np.float32)
        
        ramp = np.linspace(0, 1, overlap).astype(np.float32)
        window = np.ones((patch_size, patch_size), dtype=np.float32)
        window[:overlap, :] *= ramp[:, None]
        window[-overlap:, :] *= ramp[::-1, None]
        window[:, :overlap] *= ramp[None, :]
        window[:, -overlap:] *= ramp[None, ::-1]
        
        with torch.no_grad():
            for (y, x), patch_t in zip(positions, patches):
                # We need RGB for Branch 2, not LAB. For now, assume patches are standard RGB (3 channels)
                # or take the first 3 channels if it's 6-channel.
                b2_patch = patch_t[:3].unsqueeze(0).to(device)
                logits = model_b2(b2_patch)
                prob = torch.sigmoid(logits).cpu().numpy()[0, 0]
                heatmap_full[y:y+patch_size, x:x+patch_size] += prob * window
                weight_full[y:y+patch_size, x:x+patch_size] += window
                
        b2_heatmap = heatmap_full[:H, :W] / np.clip(weight_full[:H, :W], 1e-6, None)

    # 7. Stitch and Fuse
    if model_b2 is not None:
        stitched_logits = stitch_predictions(
            all_logits, positions, H, W, padded_H, padded_W,
            patch_size=patch_size, num_classes=CFG.num_classes,
            return_logits=True
        )
        # Gated fusion
        sorted_logits = np.sort(stitched_logits, axis=0)
        ambiguity = sorted_logits[-1] - sorted_logits[-2]
        mask = (ambiguity <= fusion_margin)
        
        stitched_logits[3] += fusion_scale * b2_heatmap * mask
        class_map = np.argmax(stitched_logits, axis=0).astype(np.int64)
    else:
        stitched_logits = stitch_predictions(
            all_logits, positions, H, W, padded_H, padded_W,
            patch_size=patch_size, num_classes=CFG.num_classes,
            return_logits=True
        )
        class_map = np.argmax(stitched_logits, axis=0).astype(np.int64)

    # 8. Optional post-prediction cleanup
    # Not used by the ParticleVision pipeline (kept from earlier experiments).
    if use_cleanup:
        class_map = cleanup_fines_predictions(
            class_map,
            min_blob_size=cleanup_min_blob,
            use_morphological_opening=cleanup_morph,
        )

    # 9. Map to RGB
    rgb_output = class_map_to_rgb(class_map)

    elapsed = time.time() - t0
    if return_logits:
        return rgb_output, class_map, elapsed, stitched_logits
    return rgb_output, class_map, elapsed


# ---------------------------------------------------------------------------
# Post-prediction cleanup for fines
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def cleanup_fines_predictions(
    class_map: np.ndarray,
    min_blob_size: int = 3,
    use_morphological_opening: bool = True,
) -> np.ndarray:
    """
    Remove spurious fines predictions from a class map.

    1. Remove isolated fines blobs smaller than `min_blob_size` pixels.
       These are below the physical particle size (1px ≈ 1µm, real fines ≈ 4µm).
    2. Optionally apply morphological opening (erosion→dilation) to shave off
       1px "halos" around correctly-detected particles.

    Parameters
    ----------
    class_map : (H, W) int64 array of class indices
    min_blob_size : blobs smaller than this are removed (default 3)
    use_morphological_opening : if True, apply opening to shrink halos

    Returns
    -------
    Cleaned class_map (modified in-place).
    """
    from scipy.ndimage import label, find_objects, binary_opening

    fines_mask = (class_map == 3).astype(np.uint8)
    original_count = int(np.sum(fines_mask))

    if original_count == 0:
        return class_map

    cleaned = fines_mask.copy()

    # Step 1: Remove small isolated blobs
    labeled, n_blobs = label(cleaned)
    slices = find_objects(labeled)
    removed_blobs = 0
    for i, sl in enumerate(slices):
        if sl is None:
            continue
        blob_mask = (labeled[sl] == (i + 1))
        size = int(np.sum(blob_mask))
        if size < min_blob_size:
            cleaned[sl][blob_mask] = 0
            removed_blobs += 1

    # Step 2: Morphological opening (erosion then dilation) to remove halos
    if use_morphological_opening:
        # Use a small cross-shaped structuring element
        struct = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.uint8)
        cleaned = binary_opening(cleaned, structure=struct).astype(np.uint8)

    # Apply cleanup back to class_map
    # Where fines were removed, assign the next most likely class
    # Simple approach: replace removed fines with surrounding majority class
    removed_pixels = (fines_mask == 1) & (cleaned == 0)
    if np.sum(removed_pixels) > 0:
        # For simplicity, assign removed fines to Void (most common neighbor)
        # A more sophisticated approach would use the logits, but this is fast
        from scipy.ndimage import maximum_filter
        # Find nearest non-fines class for each removed pixel
        non_fines_map = class_map.copy()
        non_fines_map[class_map == 3] = -1
        # Fill with nearest valid value using dilation
        for _ in range(5):
            filled = maximum_filter(non_fines_map, size=3)
            non_fines_map[non_fines_map == -1] = filled[non_fines_map == -1]
        # Where still unfilled, default to Void
        non_fines_map[non_fines_map == -1] = 1

        class_map[removed_pixels] = non_fines_map[removed_pixels]

    class_map[cleaned == 1] = 3  # Ensure remaining fines are preserved

    return class_map


# ---------------------------------------------------------------------------
# Quantitative report
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def compute_report(class_map: np.ndarray) -> Dict:
    """
    Compute pixel counts, percentages, and derived metrics from a class map.

    Returns
    -------
    report : dict with keys for each class and derived metrics.
    """
    H, W = class_map.shape
    total = H * W

    report = {"image_size": f"{W}×{H}", "total_pixels": int(total)}

    counts = {}
    for idx, name in enumerate(CFG.class_names):
        count = int((class_map == idx).sum())
        pct = 100.0 * count / total
        counts[name] = count
        report[f"{name}_pixels"] = count
        report[f"{name}_pct"] = round(pct, 4)

    # Derived metrics
    pore_pixels = counts["Void"] + counts["Water"]

    report["porosity_pct"] = round(
        100.0 * pore_pixels / total, 4
    ) if total > 0 else 0.0

    report["water_saturation_pct"] = round(
        100.0 * counts["Water"] / max(pore_pixels, 1), 4
    )

    report["fines_pore_concentration_pct"] = round(
        100.0 * counts["Fines"] / max(pore_pixels, 1), 4
    )

    report["fines_total_concentration_pct"] = round(
        100.0 * counts["Fines"] / max(total, 1), 4
    )

    return report


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def print_report(report: Dict, image_name: str = ""):
    """Pretty-print the quantitative report."""
    print(f"\n╔{'═'*54}╗")
    print(f"║  Segmentation Report: {image_name:<30} ║")
    print(f"╠{'═'*54}╣")
    print(f"║ {'Class':<12} │ {'Pixels':>12} │ {'Area %':>8} ║")
    print(f"║{'─'*14}┼{'─'*14}┼{'─'*10}║")
    for name in CFG.class_names:
        px = report[f"{name}_pixels"]
        pct = report[f"{name}_pct"]
        print(f"║ {name:<12} │ {px:>12,d} │ {pct:>7.3f}% ║")
    print(f"╠{'═'*54}╣")
    print(f"║  Derived Metrics                                     ║")
    print(f"║{'─'*54}║")
    print(f"║  Porosity (void+water / total)      │ {report['porosity_pct']:>7.3f}%      ║")
    print(f"║  Water Saturation (water / pore)    │ {report['water_saturation_pct']:>7.3f}%      ║")
    print(f"║  Fines / Pore Concentration         │ {report['fines_pore_concentration_pct']:>7.4f}%      ║")
    print(f"║  Fines / Total                      │ {report['fines_total_concentration_pct']:>7.4f}%      ║")
    print(f"╚{'═'*54}╝")


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def save_output(
    rgb_output: np.ndarray,
    report: Dict,
    image_name: str,
    output_dir: str = None,
):
    """
    Save the segmented image as .tif and the report as .json + .csv.

    Parameters
    ----------
    rgb_output : (H, W, 3) uint8 array
    report : dict from compute_report()
    image_name : e.g. 'Image_001'
    output_dir : directory to save into (default: CFG.output_dir)
    """
    if output_dir is None:
        output_dir = CFG.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Save TIF
    tif_path = os.path.join(output_dir, f"{image_name}_segmented.tif")
    Image.fromarray(rgb_output).save(tif_path)

    # Save JSON
    json_path = os.path.join(output_dir, f"{image_name}_report.json")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # Save CSV
    csv_path = os.path.join(output_dir, f"{image_name}_report.csv")
    with open(csv_path, "w") as f:
        f.write("metric,value\n")
        for k, v in report.items():
            f.write(f"{k},{v}\n")

    print(f"  Saved: {tif_path}")
    print(f"  Saved: {json_path}")
    print(f"  Saved: {csv_path}")

    return tif_path, json_path, csv_path
