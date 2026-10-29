"""
ParticleVision — Dataset & Augmentation
========================================
PyTorch Dataset that:
  1. Loads raw / segmented image pairs
  2. Handles dimension mismatches (e.g. Image_009)
  3. Splits images into overlapping patches (via tiling.py)
  4. Converts RGB masks to class-index tensors
  5. Applies data augmentation (training only)
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from typing import List, Optional, Tuple

import albumentations as A
from albumentations.pytorch import ToTensorV2

from .config import CFG
from .tiling import extract_patches


# ---------------------------------------------------------------------------
# RGB mask → class index conversion
# ---------------------------------------------------------------------------

def rgb_to_class_index(mask_rgb: np.ndarray) -> np.ndarray:
    """
    Convert an (H, W, 3) RGB mask to an (H, W) int64 class-index array.
    Uses exact colour matching — any unrecognised pixel maps to class 0 (Grains).
    """
    H, W, _ = mask_rgb.shape
    class_map = np.zeros((H, W), dtype=np.int64)
    for rgb, idx in CFG.color_to_index.items():
        match = np.all(mask_rgb == np.array(rgb, dtype=np.uint8), axis=-1)
        class_map[match] = idx
    return class_map


def clean_fines_noise(
    class_map: np.ndarray,
    max_blob_size: int = None,
) -> np.ndarray:
    """
    Remove tiny fines blobs (≤ max_blob_size pixels) from a class-index mask.

    Each removed pixel is replaced with the locally dominant non-fines class
    (computed via uniform filtering — fully vectorized, no Python loops).

    Parameters
    ----------
    class_map : (H, W) int64 array of class indices
    max_blob_size : blobs with total pixel count ≤ this are removed.
                    Default: CFG.clean_noise_max_size (2).

    Returns
    -------
    cleaned : (H, W) int64 array with small fines blobs removed.
    """
    from scipy.ndimage import label, uniform_filter

    if max_blob_size is None:
        max_blob_size = CFG.clean_noise_max_size

    FINES_CLASS = 3
    NUM_CLASSES = CFG.num_classes
    cleaned = class_map.copy()
    fines_mask = (class_map == FINES_CLASS).astype(np.uint8)

    labeled, n_blobs = label(fines_mask)
    if n_blobs == 0:
        return cleaned

    # Compute size of each blob (vectorized)
    blob_sizes = np.bincount(labeled.ravel())

    # Build mask of all small blobs at once
    small_blob_ids = np.where(
        (blob_sizes <= max_blob_size) & (np.arange(len(blob_sizes)) > 0)
    )[0]
    if len(small_blob_ids) == 0:
        return cleaned

    # Boolean mask of pixels to remove
    remove_mask = np.isin(labeled, small_blob_ids)

    # Compute the dominant non-fines class in a 7x7 window (fully vectorized).
    # For each non-fines class, compute a smoothed density map, then argmax.
    class_counts = np.zeros((NUM_CLASSES, *class_map.shape), dtype=np.float32)
    for c in range(NUM_CLASSES):
        if c == FINES_CLASS:
            continue  # skip fines — we don't want to replace with fines
        class_counts[c] = uniform_filter(
            (class_map == c).astype(np.float32), size=7, mode='reflect'
        )

    # Dominant non-fines class at every pixel
    replacement_map = np.argmax(class_counts, axis=0).astype(np.int64)

    # Apply replacement only where remove_mask is True
    cleaned[remove_mask] = replacement_map[remove_mask]

    return cleaned


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_augmentation() -> A.Compose:
    """Training augmentation applied to both image and mask."""
    return A.Compose([
        A.HorizontalFlip(p=CFG.aug_hflip_p),
        A.VerticalFlip(p=CFG.aug_vflip_p),
        A.RandomRotate90(p=CFG.aug_rot90_p),
        # Image-only colour jitter (mask is not affected)
        A.RandomBrightnessContrast(
            brightness_limit=CFG.aug_brightness_limit,
            contrast_limit=CFG.aug_contrast_limit,
            p=CFG.aug_brightness_contrast_p,
        ),
        A.HueSaturationValue(
            hue_shift_limit=CFG.aug_hue_shift_limit,
            sat_shift_limit=CFG.aug_sat_shift_limit,
            val_shift_limit=CFG.aug_val_shift_limit,
            p=CFG.aug_hue_sat_p,
        ),
        # Normalise to ImageNet stats and convert to tensor
        A.Normalize(mean=CFG.img_mean, std=CFG.img_std),
        ToTensorV2(),
    ])


def get_val_augmentation() -> A.Compose:
    """Validation / test: normalise only."""
    return A.Compose([
        A.Normalize(mean=CFG.img_mean, std=CFG.img_std),
        ToTensorV2(),
    ])


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def get_train_augmentation_lab() -> A.Compose:
    """Training augmentation without normalization (for LAB conversion)."""
    return A.Compose([
        A.HorizontalFlip(p=CFG.aug_hflip_p),
        A.VerticalFlip(p=CFG.aug_vflip_p),
        A.RandomRotate90(p=CFG.aug_rot90_p),
        A.RandomBrightnessContrast(
            brightness_limit=CFG.aug_brightness_limit,
            contrast_limit=CFG.aug_contrast_limit,
            p=CFG.aug_brightness_contrast_p,
        ),
        A.HueSaturationValue(
            hue_shift_limit=CFG.aug_hue_shift_limit,
            sat_shift_limit=CFG.aug_sat_shift_limit,
            val_shift_limit=CFG.aug_val_shift_limit,
            p=CFG.aug_hue_sat_p,
        ),
    ])

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def get_val_augmentation_lab() -> A.Compose:
    """Validation augmentation without normalization."""
    return A.Compose([])


# ---------------------------------------------------------------------------
# Main Dataset
# ---------------------------------------------------------------------------

class SandPatchDataset(Dataset):
    """
    Yields (image_tensor, mask_tensor) pairs of shape:
        image: (3, patch_size, patch_size) float32
        mask:  (patch_size, patch_size) int64

    All patches are pre-extracted in __init__ and stored in RAM.
    For 84 images × ~63 patches × 512×512 × 3 bytes ≈ 5 GB — within 24 GB.
    """

    def __init__(
        self,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
        patch_size: int = None,
        overlap: int = None,
        use_lab: bool = False,
    ):
        """
        Parameters
        ----------
        image_indices : list of int
            Which image numbers to include (e.g. [1, 2, 3, ...]).
        transform : albumentations.Compose or None
            Augmentation pipeline. None → val/test pipeline.
        patch_size, overlap : int
        use_lab : bool
        """
        super().__init__()
        self.use_lab = use_lab
        if self.use_lab and transform is None:
            self.transform = get_val_augmentation_lab()
        else:
            self.transform = transform if transform is not None else get_val_augmentation()
        self.patch_size = patch_size or CFG.patch_size
        self.overlap = overlap or CFG.overlap

        # Pre-extract all patches
        self.img_patches: List[np.ndarray] = []
        self.mask_patches: List[np.ndarray] = []  # stored as class indices
        self.source_image: List[int] = []  # track which image each patch came from

        for idx in image_indices:
            raw, mask_rgb = self._load_pair(idx)

            # Convert full-image RGB mask to class indices
            mask_cls = rgb_to_class_index(mask_rgb)

            # Optionally clean 1-2px fines noise from GT before patching
            if CFG.clean_noise_labels:
                mask_cls = clean_fines_noise(mask_cls)

            img_ps, mask_ps, _, _, _ = extract_patches(
                raw, mask_cls, self.patch_size, self.overlap
            )
            for ip, mp in zip(img_ps, mask_ps):
                self.img_patches.append(ip)
                self.mask_patches.append(mp)  # already class indices
                self.source_image.append(idx)

    @staticmethod
    def _load_pair(idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Load and align a raw/segmented pair, handling known mismatches."""
        raw = np.array(Image.open(CFG.raw_path(idx)))
        mask = np.array(Image.open(CFG.seg_path(idx)))

        # Handle dimension mismatches
        if idx in CFG.mismatched_images:
            action = CFG.mismatched_images[idx]
            if action.startswith("crop_raw_bottom_"):
                n = int(action.split("_")[-1])
                raw = raw[: raw.shape[0] - n, :]

        # Final safety: crop to minimum shared size
        min_h = min(raw.shape[0], mask.shape[0])
        min_w = min(raw.shape[1], mask.shape[1])
        raw = raw[:min_h, :min_w]
        mask = mask[:min_h, :min_w]

        return raw, mask

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.img_patches[index]   # (H, W, 3) uint8
        mask = self.mask_patches[index]  # (H, W) int64

        # Apply augmentation
        augmented = self.transform(image=img, mask=mask)
        
        if self.use_lab:
            import cv2
            img_aug = augmented["image"]
            mask_aug = augmented["mask"]
            
            # Ensure img_aug is a contiguous (H, W, 3) uint8 numpy array
            if isinstance(img_aug, torch.Tensor):
                img_aug = img_aug.numpy()
            if img_aug.ndim == 3 and img_aug.shape[0] == 3:
                # CHW → HWC
                img_aug = img_aug.transpose(1, 2, 0)
            img_aug = np.ascontiguousarray(img_aug)
            
            # Convert to LAB and normalize to [0,1]
            img_lab = cv2.cvtColor(img_aug, cv2.COLOR_RGB2LAB)
            img_lab_float = img_lab.astype(np.float32) / 255.0
            
            # Normalize RGB
            mean = np.array(CFG.img_mean, dtype=np.float32)
            std = np.array(CFG.img_std, dtype=np.float32)
            img_rgb_float = (img_aug.astype(np.float32) / 255.0 - mean) / std
            
            # Concat to 6 channels (R, G, B, L, A, B)
            img_6c = np.concatenate([img_rgb_float, img_lab_float], axis=-1)
            img_t = torch.from_numpy(img_6c.transpose(2, 0, 1))
            
            mask_t = torch.from_numpy(mask_aug).long() if isinstance(mask_aug, np.ndarray) else mask_aug.long()
        else:
            img_t = augmented["image"]   # (3, H, W) float32 tensor
            mask_t = augmented["mask"]   # (H, W) — still numpy → convert to tensor
            if isinstance(mask_t, np.ndarray):
                mask_t = torch.from_numpy(mask_t).long()
            else:
                mask_t = mask_t.long()

        return img_t, mask_t


# ---------------------------------------------------------------------------
# Helper: create datasets from split file
# ---------------------------------------------------------------------------

def create_datasets(
    split_path: str = None,
    patch_size: int = None,
    overlap: int = None,
    train_fraction: float = 1.0,
    seed: int = None,
    use_lab: bool = False,
) -> Tuple[SandPatchDataset, SandPatchDataset, SandPatchDataset]:
    """
    Load data_split.json and create train / val / test datasets.

    Parameters
    ----------
    split_path : path to data_split.json (default: PROJECT_ROOT/data_split.json)
    train_fraction : float in (0, 1] — fraction of train images to use
                     (for staged training).
    seed : random seed for subsampling train images.

    Returns
    -------
    train_ds, val_ds, test_ds : SandPatchDataset instances.
    """
    if split_path is None:
        split_path = os.path.join(CFG.project_root, "data_split.json")
    if seed is None:
        seed = CFG.seed

    with open(split_path) as f:
        split = json.load(f)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    # Subsample train set for staged training
    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n = max(1, int(len(train_indices) * train_fraction))
        train_indices = sorted(rng.choice(train_indices, size=n, replace=False).tolist())

    print(f"Creating datasets: train={len(train_indices)} imgs, "
          f"val={len(val_indices)} imgs, test={len(test_indices)} imgs")

    train_ds = SandPatchDataset(
        train_indices,
        transform=get_train_augmentation_lab() if use_lab else get_train_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        use_lab=use_lab,
    )
    val_ds = SandPatchDataset(
        val_indices,
        transform=get_val_augmentation_lab() if use_lab else get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        use_lab=use_lab,
    )
    test_ds = SandPatchDataset(
        test_indices,
        transform=get_val_augmentation_lab() if use_lab else get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        use_lab=use_lab,
    )

    print(f"  Train patches: {len(train_ds)}")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Test patches:  {len(test_ds)}")

    return train_ds, val_ds, test_ds


# ---------------------------------------------------------------------------
# Fines-weighted sampling
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def compute_fines_weights(dataset: SandPatchDataset, boost: float = 10.0) -> np.ndarray:
    """
    Compute per-patch sampling weights for WeightedRandomSampler.

    Patches containing fines pixels get `boost` times higher weight.
    This ensures the model sees fines in most batches.

    Parameters
    ----------
    dataset : SandPatchDataset
    boost : float — weight multiplier for fines-containing patches

    Returns
    -------
    weights : (N,) array of sampling weights
    """
    weights = np.ones(len(dataset), dtype=np.float64)
    n_with_fines = 0

    for i, mask in enumerate(dataset.mask_patches):
        fines_pixels = int(np.sum(mask == 3))
        if fines_pixels > 0:
            # Scale weight by how many fines pixels (more fines = even more important)
            weights[i] = boost
            n_with_fines += 1

    pct = n_with_fines / len(dataset) * 100
    print(f"  Fines sampling: {n_with_fines}/{len(dataset)} patches contain fines ({pct:.1f}%)")
    print(f"  Boost factor: {boost}x → effective fines patch proportion: "
          f"~{pct * boost / (pct * boost + (100 - pct)):.1f}%")

    return weights


# ---------------------------------------------------------------------------
# Copy-Paste Augmentation for Fines
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class CopyPasteFines:
    """
    Augmentation that pastes real fines blobs from a pool into void/water
    regions of the current patch.

    This multiplies the effective number of fines examples without
    requiring new manual labelling.
    """

    def __init__(
        self,
        fines_pool: List[Tuple[np.ndarray, np.ndarray]],
        paste_prob: float = 0.5,
        max_pastes: int = 5,
        boundary_bias: float = 0.7,
        rng: np.random.RandomState = None,
    ):
        """
        Parameters
        ----------
        fines_pool : list of (blob_img, blob_mask) tuples
            Each blob_img is (h, w, 3) uint8 and blob_mask is (h, w) bool.
        paste_prob : probability of applying copy-paste per sample.
        max_pastes : maximum number of blobs to paste per sample.
        boundary_bias : fraction of pastes that target near-grain-boundary
                        locations (default 0.7 = 70% near boundary, 30% random).
        """
        self.fines_pool = fines_pool
        self.paste_prob = paste_prob
        self.max_pastes = max_pastes
        self.boundary_bias = boundary_bias
        self.rng = rng or np.random.RandomState(CFG.seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply copy-paste fines augmentation.

        Parameters
        ----------
        img : (H, W, 3) uint8
        mask : (H, W) int64 class-index map

        Returns
        -------
        img, mask (potentially modified)
        """
        if not self.fines_pool or self.rng.random() > self.paste_prob:
            return img, mask

        img = img.copy()
        mask = mask.copy()
        H, W = mask.shape

        # Pre-compute grain boundary zone for boundary-biased placement
        from scipy.ndimage import binary_dilation
        grain_mask = (mask == 0).astype(np.uint8)
        grain_dilated = binary_dilation(grain_mask, iterations=5).astype(np.uint8)
        # Boundary zone = near grains but not inside grains, and in void/water
        pasteable_base = np.isin(mask, [1, 2])
        boundary_zone = pasteable_base & (grain_dilated == 1)

        n_pastes = self.rng.randint(1, self.max_pastes + 1)

        for _ in range(n_pastes):
            idx = self.rng.randint(len(self.fines_pool))
            blob_img, blob_mask = self.fines_pool[idx]
            bh, bw = blob_mask.shape

            if bh >= H or bw >= W:
                continue

            # Decide: boundary-biased or random placement
            use_boundary = self.rng.random() < self.boundary_bias

            placed = False
            for _ in range(15):
                y = self.rng.randint(0, H - bh)
                x = self.rng.randint(0, W - bw)

                target_region = mask[y:y+bh, x:x+bw]
                # Only paste into void (1) or water (2) regions
                pasteable = np.isin(target_region, [1, 2])
                valid_mask = blob_mask & pasteable

                if np.sum(valid_mask) < 2:
                    continue

                # If boundary-biased, check that paste location is near a grain
                if use_boundary:
                    boundary_overlap = boundary_zone[y:y+bh, x:x+bw]
                    if np.sum(blob_mask & boundary_overlap) == 0:
                        continue  # Not near a grain boundary, try again

                # Paste the blob
                img[y:y+bh, x:x+bw][valid_mask] = blob_img[valid_mask]
                mask[y:y+bh, x:x+bw][valid_mask] = 3  # Fines class
                placed = True
                break

        return img, mask


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def build_fines_pool(
    image_indices: List[int],
    min_blob_size: int = 3,
    max_blob_size: int = 30,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract real fines blobs from training images to build the copy-paste pool.

    Parameters
    ----------
    image_indices : list of image indices to extract from
    min_blob_size : minimum blob size in pixels (exclude noise)
    max_blob_size : maximum blob bounding box dimension

    Returns
    -------
    pool : list of (blob_img_crop, blob_mask_crop) tuples
    """
    from scipy.ndimage import label, find_objects

    pool = []

    for idx in image_indices:
        raw, mask_rgb = SandPatchDataset._load_pair(idx)
        mask_cls = rgb_to_class_index(mask_rgb)
        fines = (mask_cls == 3).astype(np.uint8)
        labeled, n_blobs = label(fines)
        slices = find_objects(labeled)

        for i, sl in enumerate(slices):
            if sl is None:
                continue
            blob_mask = (labeled[sl] == (i + 1))
            size = int(np.sum(blob_mask))

            if size < min_blob_size:
                continue

            bh = sl[0].stop - sl[0].start
            bw = sl[1].stop - sl[1].start
            if bh > max_blob_size or bw > max_blob_size:
                continue

            blob_img = raw[sl].copy()
            pool.append((blob_img, blob_mask.copy()))

    print(f"  Fines pool: extracted {len(pool)} blobs from {len(image_indices)} images "
          f"(size range: {min_blob_size}-{max_blob_size}px)")

    return pool


# ---------------------------------------------------------------------------
# V2 dataset factory with copy-paste and oversampling
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class SandPatchDatasetV2(SandPatchDataset):
    """
    Extended dataset with copy-paste fines augmentation applied
    before the albumentations pipeline.
    """

    def __init__(
        self,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
        patch_size: int = None,
        overlap: int = None,
        copy_paste: Optional[CopyPasteFines] = None,
    ):
        super().__init__(image_indices, transform, patch_size, overlap)
        self.copy_paste = copy_paste

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.img_patches[index]   # (H, W, 3) uint8
        mask = self.mask_patches[index]  # (H, W) int64

        # Apply copy-paste augmentation BEFORE albumentations
        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)

        # Apply albumentations augmentation
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"]

        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t).long()
        else:
            mask_t = mask_t.long()

        return img_t, mask_t


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def create_datasets_v2(
    split_path: str = None,
    patch_size: int = None,
    overlap: int = None,
    train_fraction: float = 1.0,
    seed: int = None,
    copy_paste_prob: float = 0.5,
    max_pastes: int = 5,
    fines_boost: float = 10.0,
) -> Tuple[SandPatchDatasetV2, SandPatchDataset, SandPatchDataset, np.ndarray]:
    """
    Create V2 datasets with copy-paste augmentation and sampling weights.

    Returns
    -------
    train_ds, val_ds, test_ds, sample_weights
    """
    if split_path is None:
        split_path = os.path.join(CFG.project_root, "data_split.json")
    if seed is None:
        seed = CFG.seed

    with open(split_path) as f:
        split = json.load(f)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n = max(1, int(len(train_indices) * train_fraction))
        train_indices = sorted(rng.choice(train_indices, size=n, replace=False).tolist())

    print(f"Creating V2 datasets: train={len(train_indices)} imgs, "
          f"val={len(val_indices)} imgs, test={len(test_indices)} imgs")

    # Build fines pool from training images
    print("Building fines copy-paste pool...")
    fines_pool = build_fines_pool(train_indices, min_blob_size=3, max_blob_size=30)
    copy_paste = CopyPasteFines(
        fines_pool, paste_prob=copy_paste_prob, max_pastes=max_pastes,
        rng=np.random.RandomState(seed)
    )

    train_ds = SandPatchDatasetV2(
        train_indices,
        transform=get_train_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        copy_paste=copy_paste,
    )
    val_ds = SandPatchDataset(
        val_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )
    test_ds = SandPatchDataset(
        test_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )

    # Compute sampling weights
    sample_weights = compute_fines_weights(train_ds, boost=fines_boost)

    print(f"  Train patches: {len(train_ds)}")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Test patches:  {len(test_ds)}")

    return train_ds, val_ds, test_ds, sample_weights


# ---------------------------------------------------------------------------
# V3 dataset factory with Mosaic Augmentation
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class MosaicFines:
    def __init__(self, dataset: SandPatchDataset, p: float = 0.5, rng: np.random.RandomState = None):
        self.dataset = dataset
        self.p = p
        self.rng = rng if rng is not None else np.random.RandomState(CFG.seed)
        
        self.fines_indices = []
        for i, mask in enumerate(self.dataset.mask_patches):
            if np.any(mask == 3):
                self.fines_indices.append(i)
                
    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.p or len(self.fines_indices) < 3:
            return img, mask
            
        # Select 3 random patches with fines
        other_idx = self.rng.choice(self.fines_indices, size=3, replace=False)
        
        # We need 4 patches total: current (img, mask) + 3 others
        patches = [(img, mask)]
        for idx in other_idx:
            patches.append((self.dataset.img_patches[idx], self.dataset.mask_patches[idx]))
            
        # For each 512x512 patch, take a random 256x256 crop
        crops_img = []
        crops_mask = []
        for p_img, p_mask in patches:
            h, w = p_img.shape[:2]
            y = self.rng.randint(0, h - 256 + 1)
            x = self.rng.randint(0, w - 256 + 1)
            crops_img.append(p_img[y:y+256, x:x+256])
            crops_mask.append(p_mask[y:y+256, x:x+256])
            
        # Stitch into 2x2 grid (512x512)
        top_img = np.concatenate([crops_img[0], crops_img[1]], axis=1)
        bottom_img = np.concatenate([crops_img[2], crops_img[3]], axis=1)
        mosaic_img = np.concatenate([top_img, bottom_img], axis=0)
        
        top_mask = np.concatenate([crops_mask[0], crops_mask[1]], axis=1)
        bottom_mask = np.concatenate([crops_mask[2], crops_mask[3]], axis=1)
        mosaic_mask = np.concatenate([top_mask, bottom_mask], axis=0)
        
        return mosaic_img, mosaic_mask


# ---------------------------------------------------------------------------
# Fines-Safe Elastic Deformation (an earlier experiment)
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
class FinesSafeElasticTransform:
    """
    Applies a mild elastic deformation to both image and mask.

    Safety constraint: after deformation, every fines blob that was ≥3px in
    the original mask must still be ≥3px in the transformed mask. If any such
    blob is destroyed or merged away, the transform is rejected and the
    original image/mask are returned unchanged.

    Parameters
    ----------
    alpha : float — magnitude of displacement field (default 50)
    sigma : float — smoothness of displacement field in pixels (default 8)
    p : float — probability of applying the transform per sample
    min_safe_blob : int — minimum blob size to protect (default 3)
    """

    def __init__(
        self,
        alpha: float = 50.0,
        sigma: float = 8.0,
        p: float = 0.5,
        min_safe_blob: int = 3,
        rng: np.random.RandomState = None,
    ):
        self.alpha = alpha
        self.sigma = sigma
        self.p = p
        self.min_safe_blob = min_safe_blob
        self.rng = rng or np.random.RandomState(CFG.seed)

    def __call__(self, img: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.rng.random() > self.p:
            return img, mask

        import cv2
        from scipy.ndimage import label, gaussian_filter

        H, W = mask.shape

        # Generate smooth random displacement field
        dx = self.rng.uniform(-1, 1, (H, W)).astype(np.float32)
        dy = self.rng.uniform(-1, 1, (H, W)).astype(np.float32)
        dx = gaussian_filter(dx, sigma=self.sigma) * self.alpha
        dy = gaussian_filter(dy, sigma=self.sigma) * self.alpha

        # Build remap grid
        x, y = np.meshgrid(np.arange(W, dtype=np.float32),
                           np.arange(H, dtype=np.float32))
        map_x = (x + dx).astype(np.float32)
        map_y = (y + dy).astype(np.float32)

        # Warp image (bilinear) and mask (nearest-neighbor)
        warped_img = cv2.remap(img, map_x, map_y,
                               interpolation=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REFLECT_101)
        warped_mask = cv2.remap(mask.astype(np.float32), map_x, map_y,
                                interpolation=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_REFLECT_101).astype(np.int64)

        # --- Safety check: verify no fines blob ≥ min_safe_blob was destroyed ---
        orig_fines = (mask == 3).astype(np.uint8)
        warped_fines = (warped_mask == 3).astype(np.uint8)

        # If there were no fines, no risk — accept the warp
        if orig_fines.sum() == 0:
            return warped_img, warped_mask

        orig_labeled, n_orig = label(orig_fines)

        for i in range(1, n_orig + 1):
            blob = (orig_labeled == i)
            blob_size = int(blob.sum())
            if blob_size < self.min_safe_blob:
                continue  # Don't protect noise-sized blobs

            # Check: does this blob still have ≥ min_safe_blob pixels in the
            # warped mask's fines channel, approximately where it was?
            # We check the bounding box region (with a small margin)
            ys, xs = np.where(blob)
            y0, y1 = max(0, ys.min() - 3), min(H, ys.max() + 4)
            x0, x1 = max(0, xs.min() - 3), min(W, xs.max() + 4)

            warped_region = warped_fines[y0:y1, x0:x1]
            if warped_region.sum() < self.min_safe_blob:
                # This blob was destroyed — reject the entire transform
                return img, mask

        return warped_img, warped_mask


# Not used by the ParticleVision pipeline (kept from earlier experiments).
class SandPatchDatasetV3(SandPatchDatasetV2):
    """
    V3 dataset supporting mosaic and/or elastic deformation augmentation
    applied before the albumentations pipeline.
    """
    def __init__(
        self,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
        patch_size: int = None,
        overlap: int = None,
        copy_paste: Optional[CopyPasteFines] = None,
        mosaic_prob: float = 0.0,
        elastic_prob: float = 0.0,
        rng: np.random.RandomState = None,
    ):
        super().__init__(image_indices, transform, patch_size, overlap, copy_paste)
        self.mosaic = MosaicFines(self, p=mosaic_prob, rng=rng) if mosaic_prob > 0 else None
        self.elastic = FinesSafeElasticTransform(p=elastic_prob, rng=rng) if elastic_prob > 0 else None
        
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.img_patches[index]   # (H, W, 3) uint8
        mask = self.mask_patches[index]  # (H, W) int64

        # Apply mosaic first (if enabled)
        if self.mosaic is not None:
            img, mask = self.mosaic(img, mask)

        # Apply fines-safe elastic deformation (if enabled)
        if self.elastic is not None:
            img, mask = self.elastic(img, mask)

        # Apply copy-paste augmentation BEFORE albumentations
        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)

        # Apply albumentations augmentation
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"]

        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t).long()
        else:
            mask_t = mask_t.long()

        return img_t, mask_t

# Keep old class name as alias for backwards compatibility
SandPatchDatasetMosaic = SandPatchDatasetV3

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def create_datasets_v3(
    split_path: str = None,
    patch_size: int = None,
    overlap: int = None,
    train_fraction: float = 1.0,
    seed: int = None,
    copy_paste_prob: float = 0.5,
    max_pastes: int = 5,
    fines_boost: float = 10.0,
    mosaic_prob: float = 0.0,
    elastic_prob: float = 0.0,
) -> Tuple[SandPatchDatasetV3, SandPatchDataset, SandPatchDataset, np.ndarray]:
    if split_path is None:
        split_path = os.path.join(CFG.project_root, "data_split.json")
    if seed is None:
        seed = CFG.seed

    with open(split_path) as f:
        split = json.load(f)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n = max(1, int(len(train_indices) * train_fraction))
        train_indices = sorted(rng.choice(train_indices, size=n, replace=False).tolist())

    augs_desc = []
    if mosaic_prob > 0: augs_desc.append(f"Mosaic@{mosaic_prob:.0%}")
    if elastic_prob > 0: augs_desc.append(f"Elastic@{elastic_prob:.0%}")
    augs_str = " + ".join(augs_desc) if augs_desc else "CopyPaste only"

    print(f"Creating V3 datasets ({augs_str}): train={len(train_indices)} imgs, "
          f"val={len(val_indices)} imgs, test={len(test_indices)} imgs")

    print("Building fines copy-paste pool...")
    fines_pool = build_fines_pool(train_indices, min_blob_size=3, max_blob_size=30)
    
    rng = np.random.RandomState(seed)
    copy_paste = CopyPasteFines(
        fines_pool, paste_prob=copy_paste_prob, max_pastes=max_pastes,
        rng=rng
    )

    train_ds = SandPatchDatasetV3(
        train_indices,
        transform=get_train_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        copy_paste=copy_paste,
        mosaic_prob=mosaic_prob,
        elastic_prob=elastic_prob,
        rng=rng,
    )
    val_ds = SandPatchDataset(
        val_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )
    test_ds = SandPatchDataset(
        test_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )

    sample_weights = compute_fines_weights(train_ds, boost=fines_boost)

    print(f"  Train patches: {len(train_ds)}")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Test patches:  {len(test_ds)}")

    return train_ds, val_ds, test_ds, sample_weights


# ---------------------------------------------------------------------------
# Fines Context Crops (an earlier experiment)
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def extract_fines_context_patches(
    image_indices: List[int],
    patch_size: int = None,
    jitter: int = 20,
    min_blob_size: int = 3,
    max_crops_per_image: int = 60,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[int]]:
    """
    Extract patches deliberately centered on fines blob locations.

    Instead of a uniform grid, this finds every fines blob ≥ min_blob_size,
    randomly samples up to max_crops_per_image of them, computes their
    centroids, adds a random jitter of ±jitter pixels, and extracts a
    patch_size × patch_size crop centered there.

    Parameters
    ----------
    image_indices : list of image indices
    patch_size : patch size (default CFG.patch_size)
    jitter : max random offset in pixels (default ±20)
    min_blob_size : minimum fines blob size to target (default 3px)
    max_crops_per_image : limit to prevent OOM (default 60, matches grid patch count)

    Returns
    -------
    img_patches : list of (patch_size, patch_size, 3) uint8 arrays
    mask_patches : list of (patch_size, patch_size) int64 arrays
    source_images : list of source image indices
    """
    from scipy.ndimage import label, center_of_mass

    if patch_size is None:
        patch_size = CFG.patch_size

    rng = np.random.RandomState(CFG.seed)
    half = patch_size // 2

    img_patches = []
    mask_patches = []
    source_images = []

    for idx in image_indices:
        raw, mask_rgb = SandPatchDataset._load_pair(idx)
        mask_cls = rgb_to_class_index(mask_rgb)
        H, W = mask_cls.shape

        # Pad so we can always extract a full patch even at edges
        pad = half + jitter
        raw_padded = np.pad(raw, ((pad, pad), (pad, pad), (0, 0)), mode="reflect")
        mask_padded = np.pad(mask_cls, ((pad, pad), (pad, pad)), mode="reflect")

        # Find fines blobs
        fines = (mask_cls == 3).astype(np.uint8)
        labeled, n_blobs = label(fines)

        if n_blobs == 0:
            continue

        centroids = center_of_mass(fines, labeled, range(1, n_blobs + 1))
        
        # Filter centroids to only those representing valid sized blobs
        valid_indices = []
        for i in range(n_blobs):
            blob = (labeled == (i + 1))
            if int(blob.sum()) >= min_blob_size:
                valid_indices.append(i)
                
        # Limit the number of crops per image to prevent OOM
        if len(valid_indices) > max_crops_per_image:
            valid_indices = rng.choice(valid_indices, size=max_crops_per_image, replace=False).tolist()

        for i in valid_indices:
            cy, cx = centroids[i]

            # Apply jitter
            jy = rng.randint(-jitter, jitter + 1)
            jx = rng.randint(-jitter, jitter + 1)

            # Center in padded coordinates
            cy_pad = int(cy) + pad + jy
            cx_pad = int(cx) + pad + jx

            y0 = cy_pad - half
            x0 = cx_pad - half

            img_patch = raw_padded[y0:y0 + patch_size, x0:x0 + patch_size]
            mask_patch = mask_padded[y0:y0 + patch_size, x0:x0 + patch_size]

            if img_patch.shape[0] != patch_size or img_patch.shape[1] != patch_size:
                continue  # Edge case safety

            img_patches.append(img_patch.copy())
            mask_patches.append(mask_patch.copy())
            source_images.append(idx)

    print(f"  Fines context crops: {len(img_patches)} patches from {len(image_indices)} images "
          f"(centered on blobs ≥{min_blob_size}px, jitter=±{jitter}px)")

    return img_patches, mask_patches, source_images


# Not used by the ParticleVision pipeline (kept from earlier experiments).
class FinesContextCropDataset(Dataset):
    """
    Dataset that uses fines-centered crops instead of uniform grid patches.

    Combines:
    - Fines context crops (centered on blob locations) as the primary patches
    - Standard grid patches (for non-fines context) mixed in
    - Copy-paste augmentation
    - Albumentations pipeline
    """

    def __init__(
        self,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
        patch_size: int = None,
        overlap: int = None,
        copy_paste: Optional[CopyPasteFines] = None,
        context_crop_ratio: float = 0.5,
        jitter: int = 20,
    ):
        """
        Parameters
        ----------
        context_crop_ratio : float
            Fraction of patches that should be fines-centered crops (rest are
            standard grid patches). Default 0.5 = 50/50 mix.
        jitter : int
            Random offset for fines context crops (default ±20px).
        """
        super().__init__()
        self.transform = transform if transform is not None else get_val_augmentation()
        self.patch_size = patch_size or CFG.patch_size
        self.overlap = overlap or CFG.overlap
        self.copy_paste = copy_paste

        # Extract standard grid patches
        grid_img_patches = []
        grid_mask_patches = []
        grid_sources = []

        for idx in image_indices:
            raw, mask = SandPatchDataset._load_pair(idx)
            img_ps, mask_ps, _, _, _ = extract_patches(
                raw, mask, self.patch_size, self.overlap
            )
            for ip, mp in zip(img_ps, mask_ps):
                grid_img_patches.append(ip)
                grid_mask_patches.append(rgb_to_class_index(mp))
                grid_sources.append(idx)

        # Extract fines-centered context crops
        ctx_img, ctx_mask, ctx_sources = extract_fines_context_patches(
            image_indices,
            patch_size=self.patch_size,
            jitter=jitter,
            max_crops_per_image=60, # Keep roughly 50/50 mix with grid patches
        )

        # Combine: context crops + grid patches
        self.img_patches = ctx_img + grid_img_patches
        self.mask_patches = ctx_mask + grid_mask_patches
        self.source_image = ctx_sources + grid_sources
        self.n_context = len(ctx_img)
        self.n_grid = len(grid_img_patches)

        print(f"  FinesContextCropDataset: {self.n_context} context crops + "
              f"{self.n_grid} grid patches = {len(self)} total")

    def __len__(self):
        return len(self.img_patches)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.img_patches[index]
        mask = self.mask_patches[index]

        if self.copy_paste is not None:
            img, mask = self.copy_paste(img, mask)

        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_t = augmented["mask"]

        if isinstance(mask_t, np.ndarray):
            mask_t = torch.from_numpy(mask_t).long()
        else:
            mask_t = mask_t.long()

        return img_t, mask_t


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def create_datasets_v3_context_crops(
    split_path: str = None,
    patch_size: int = None,
    overlap: int = None,
    train_fraction: float = 1.0,
    seed: int = None,
    copy_paste_prob: float = 0.5,
    max_pastes: int = 5,
    fines_boost: float = 10.0,
    jitter: int = 20,
) -> Tuple[FinesContextCropDataset, SandPatchDataset, SandPatchDataset, np.ndarray]:
    if split_path is None:
        split_path = os.path.join(CFG.project_root, "data_split.json")
    if seed is None:
        seed = CFG.seed

    with open(split_path) as f:
        split = json.load(f)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n = max(1, int(len(train_indices) * train_fraction))
        train_indices = sorted(rng.choice(train_indices, size=n, replace=False).tolist())

    print(f"Creating V3 datasets (Context Crops): train={len(train_indices)} imgs, "
          f"val={len(val_indices)} imgs, test={len(test_indices)} imgs")

    print("Building fines copy-paste pool...")
    fines_pool = build_fines_pool(train_indices, min_blob_size=3, max_blob_size=30)

    rng = np.random.RandomState(seed)
    copy_paste = CopyPasteFines(
        fines_pool, paste_prob=copy_paste_prob, max_pastes=max_pastes,
        rng=rng
    )

    train_ds = FinesContextCropDataset(
        train_indices,
        transform=get_train_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        copy_paste=copy_paste,
        jitter=jitter,
    )
    val_ds = SandPatchDataset(
        val_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )
    test_ds = SandPatchDataset(
        test_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
    )

    sample_weights = compute_fines_weights(train_ds, boost=fines_boost)

    print(f"  Train patches: {len(train_ds)}")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Test patches:  {len(test_ds)}")

    return train_ds, val_ds, test_ds, sample_weights


# ---------------------------------------------------------------------------
# Dual-Head Dataset with Heatmap Targets
# ---------------------------------------------------------------------------

# Not used by the ParticleVision pipeline (kept from earlier experiments).
def generate_fines_heatmap(mask: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """
    Generate a 2D Gaussian heatmap for fines centroids.
    Uses a local window (±4σ) around each centroid for efficiency.

    mask: (H, W) class index mask where Fines == 3
    Returns: (1, H, W) float32 heatmap in [0, 1]
    """
    from scipy.ndimage import label, find_objects
    H, W = mask.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    fines_mask = (mask == 3).astype(np.uint8)
    labeled, n_blobs = label(fines_mask)
    if n_blobs == 0:
        return heatmap[np.newaxis, ...]

    slices = find_objects(labeled)
    radius = int(np.ceil(4.0 * sigma))  # ±4σ captures >99.97% of the Gaussian

    for i, sl in enumerate(slices):
        if sl is None:
            continue
        blob_mask = (labeled[sl] == (i + 1))

        # Find centroid of the blob
        y_indices, x_indices = np.nonzero(blob_mask)
        if len(y_indices) == 0:
            continue

        # absolute coords within the patch
        cy = float(y_indices.mean() + sl[0].start)
        cx = float(x_indices.mean() + sl[1].start)

        # Local window bounds (clipped to image)
        y0 = max(0, int(cy) - radius)
        y1 = min(H, int(cy) + radius + 1)
        x0 = max(0, int(cx) - radius)
        x1 = min(W, int(cx) + radius + 1)

        # Local coordinate grids (float32)
        yy = np.arange(y0, y1, dtype=np.float32) - cy
        xx = np.arange(x0, x1, dtype=np.float32) - cx
        yy2 = yy[:, np.newaxis] ** 2
        xx2 = xx[np.newaxis, :] ** 2
        gaussian = np.exp(-(yy2 + xx2) / (2.0 * sigma * sigma))

        # Max-merge into the heatmap
        patch = heatmap[y0:y1, x0:x1]
        np.maximum(patch, gaussian, out=patch)

    return heatmap[np.newaxis, ...]


# Not used by the ParticleVision pipeline (kept from earlier experiments).
class SandPatchDatasetDualHead(SandPatchDataset):
    """
    Extended dataset that returns (image, seg_mask, heatmap_target).
    """
    def __init__(
        self,
        image_indices: List[int],
        transform: Optional[A.Compose] = None,
        patch_size: int = None,
        overlap: int = None,
        sigma: float = 2.0,
    ):
        super().__init__(image_indices, transform, patch_size, overlap)
        self.sigma = sigma

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img = self.img_patches[index]   # (H, W, 3) uint8
        mask = self.mask_patches[index]  # (H, W) int64

        # Apply augmentation
        augmented = self.transform(image=img, mask=mask)
        img_t = augmented["image"]
        mask_aug = augmented["mask"]

        # Generate heatmap based on augmented mask
        # Since mask_aug can be a tensor if transform includes ToTensorV2, ensure we use numpy
        mask_np = mask_aug.numpy() if isinstance(mask_aug, torch.Tensor) else mask_aug
        
        heatmap = generate_fines_heatmap(mask_np, sigma=self.sigma)
        heatmap_t = torch.from_numpy(heatmap).float()

        if isinstance(mask_aug, np.ndarray):
            mask_t = torch.from_numpy(mask_aug).long()
        else:
            mask_t = mask_aug.long()

        return img_t, mask_t, heatmap_t


# Not used by the ParticleVision pipeline (kept from earlier experiments).
def create_datasets_dual_head(
    split_path: str = None,
    patch_size: int = None,
    overlap: int = None,
    train_fraction: float = 1.0,
    seed: int = None,
    sigma: float = 2.0,
) -> Tuple[SandPatchDatasetDualHead, SandPatchDatasetDualHead, SandPatchDatasetDualHead]:
    """
    Create dual-head datasets.
    """
    if split_path is None:
        split_path = os.path.join(CFG.project_root, "data_split.json")
    if seed is None:
        seed = CFG.seed

    with open(split_path) as f:
        split = json.load(f)

    train_indices = split["train"]
    val_indices = split["val"]
    test_indices = split["test"]

    if train_fraction < 1.0:
        rng = np.random.RandomState(seed)
        n = max(1, int(len(train_indices) * train_fraction))
        train_indices = sorted(rng.choice(train_indices, size=n, replace=False).tolist())

    print(f"Creating Dual-Head datasets: train={len(train_indices)} imgs, "
          f"val={len(val_indices)} imgs, test={len(test_indices)} imgs")

    train_ds = SandPatchDatasetDualHead(
        train_indices,
        transform=get_train_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        sigma=sigma,
    )
    val_ds = SandPatchDatasetDualHead(
        val_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        sigma=sigma,
    )
    test_ds = SandPatchDatasetDualHead(
        test_indices,
        transform=get_val_augmentation(),
        patch_size=patch_size,
        overlap=overlap,
        sigma=sigma,
    )

    print(f"  Train patches: {len(train_ds)}")
    print(f"  Val patches:   {len(val_ds)}")
    print(f"  Test patches:  {len(test_ds)}")

    return train_ds, val_ds, test_ds
