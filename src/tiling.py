"""
ParticleVision — Dynamic Tiling & Stitching
=============================================
Handles splitting large images into overlapping patches for training/inference,
and stitching predicted patches back into a full-resolution output image.
All grid calculations are dynamic — computed per-image from its actual dimensions.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple

from .config import CFG


# ---------------------------------------------------------------------------
# Tile grid computation
# ---------------------------------------------------------------------------

def compute_tile_positions(
    H: int, W: int,
    patch_size: int = None,
    overlap: int = None,
) -> Tuple[List[Tuple[int, int]], int, int]:
    """
    Compute (y, x) top-left corner positions for an overlapping tile grid.

    Parameters
    ----------
    H, W : int
        Original image height and width.
    patch_size : int
        Tile side length (default: CFG.patch_size = 512).
    overlap : int
        Overlap in pixels between adjacent tiles (default: CFG.overlap = 64).

    Returns
    -------
    positions : list of (y, x) tuples
        Top-left corners in the *padded* coordinate system.
    padded_H, padded_W : int
        Image dimensions after reflection padding.
    """
    if patch_size is None:
        patch_size = CFG.patch_size
    if overlap is None:
        overlap = CFG.overlap

    stride = patch_size - overlap  # 448

    n_rows = int(np.ceil((H - overlap) / stride))
    n_cols = int(np.ceil((W - overlap) / stride))

    padded_H = overlap + n_rows * stride
    padded_W = overlap + n_cols * stride

    positions = []
    for r in range(n_rows):
        for c in range(n_cols):
            positions.append((r * stride, c * stride))

    return positions, padded_H, padded_W


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def extract_patches(
    image: np.ndarray,
    mask: np.ndarray = None,
    patch_size: int = None,
    overlap: int = None,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[Tuple[int, int]], int, int]:
    """
    Split an image (and optionally its mask) into overlapping patches.

    Parameters
    ----------
    image : (H, W, C) uint8 array
    mask  : (H, W, C) uint8 array or None
    patch_size, overlap : int

    Returns
    -------
    img_patches  : list of (patch_size, patch_size, C) arrays
    mask_patches : list of (patch_size, patch_size, C) arrays (empty if mask is None)
    positions    : list of (y, x) tuples
    padded_H, padded_W : int
    """
    if patch_size is None:
        patch_size = CFG.patch_size
    if overlap is None:
        overlap = CFG.overlap

    H, W = image.shape[:2]
    positions, padded_H, padded_W = compute_tile_positions(
        H, W, patch_size, overlap
    )

    # Pad image with reflection
    pad_h = padded_H - H
    pad_w = padded_W - W
    image_padded = np.pad(
        image, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
    )

    mask_padded = None
    if mask is not None:
        # Mask must use reflect too (nearest would be for resizing, not padding)
        if mask.ndim == 3:
            mask_padded = np.pad(
                mask, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect"
            )
        else:
            # 2D class-index mask (H, W)
            mask_padded = np.pad(
                mask, ((0, pad_h), (0, pad_w)), mode="reflect"
            )

    img_patches = []
    mask_patches = []
    for y, x in positions:
        img_patches.append(image_padded[y : y + patch_size, x : x + patch_size])
        if mask_padded is not None:
            mask_patches.append(
                mask_padded[y : y + patch_size, x : x + patch_size]
            )

    return img_patches, mask_patches, positions, padded_H, padded_W


# ---------------------------------------------------------------------------
# Stitching predictions back together
# ---------------------------------------------------------------------------

def stitch_predictions(
    logits_list: List[np.ndarray],
    positions: List[Tuple[int, int]],
    original_H: int,
    original_W: int,
    padded_H: int,
    padded_W: int,
    patch_size: int = None,
    num_classes: int = None,
    return_logits: bool = False,
) -> np.ndarray:
    """
    Reconstruct a full-resolution prediction by averaging overlapping logits.

    Parameters
    ----------
    logits_list : list of (num_classes, patch_size, patch_size) float arrays
        Raw logits (pre-softmax) for each patch.
    positions : list of (y, x) tuples
        Matching positions from compute_tile_positions.
    original_H, original_W : int
        The original (unpadded) image dimensions.
    padded_H, padded_W : int
        Padded dimensions from compute_tile_positions.
    patch_size : int
    num_classes : int
    return_logits : bool
        If True, return the averaged raw logits (num_classes, original_H, original_W).
        If False, return the argmax class_map (original_H, original_W).

    Returns
    -------
    np.ndarray
    """
    if patch_size is None:
        patch_size = CFG.patch_size
    if num_classes is None:
        num_classes = CFG.num_classes

    # Accumulators
    logit_sum = np.zeros((num_classes, padded_H, padded_W), dtype=np.float32)
    count = np.zeros((1, padded_H, padded_W), dtype=np.float32)

    for logits, (y, x) in zip(logits_list, positions):
        logit_sum[:, y : y + patch_size, x : x + patch_size] += logits
        count[:, y : y + patch_size, x : x + patch_size] += 1.0

    # Average where count > 0
    avg_logits = logit_sum / np.maximum(count, 1.0)

    # Crop to original size and argmax
    avg_logits = avg_logits[:, :original_H, :original_W]
    if return_logits:
        return avg_logits
    
    class_map = np.argmax(avg_logits, axis=0).astype(np.int64)
    return class_map


def stitch_predictions_torch(
    logits_list: List[torch.Tensor],
    positions: List[Tuple[int, int]],
    original_H: int,
    original_W: int,
    padded_H: int,
    padded_W: int,
    patch_size: int = None,
    num_classes: int = None,
    return_logits: bool = False,
) -> torch.Tensor:
    """
    Same as stitch_predictions but operates on PyTorch tensors (stays on CPU
    to avoid OOM on the full image).

    Returns
    -------
    torch.Tensor
    """
    if patch_size is None:
        patch_size = CFG.patch_size
    if num_classes is None:
        num_classes = CFG.num_classes

    logit_sum = torch.zeros(
        (num_classes, padded_H, padded_W), dtype=torch.float32
    )
    count = torch.zeros((1, padded_H, padded_W), dtype=torch.float32)

    for logits, (y, x) in zip(logits_list, positions):
        if logits.device.type != "cpu":
            logits = logits.cpu()
        logit_sum[:, y : y + patch_size, x : x + patch_size] += logits
        count[:, y : y + patch_size, x : x + patch_size] += 1.0

    avg_logits = logit_sum / count.clamp(min=1.0)
    avg_logits = avg_logits[:, :original_H, :original_W]
    if return_logits:
        return avg_logits
        
    class_map = avg_logits.argmax(dim=0)
    return class_map


# ---------------------------------------------------------------------------
# Index → RGB colour map conversion
# ---------------------------------------------------------------------------

def class_map_to_rgb(class_map: np.ndarray) -> np.ndarray:
    """
    Convert a (H, W) class-index array to an (H, W, 3) RGB uint8 image
    using the exact colour codes from the spec.
    """
    lut = CFG.index_to_color  # (num_classes, 3)
    return lut[class_map]
