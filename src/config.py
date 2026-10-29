"""
ParticleVision — Central Configuration
=======================================
All tunable hyperparameters and paths in one place.
Edit values here; never hardcode them in notebooks or modules.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Resolve project root relative to this file (src/config.py → project root)
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(_THIS_DIR)

# The app version - the ONE place it is defined. Shown in the app sidebar and written into every run
# (run_log.txt, summary.csv, run_settings.json). 1.1.0 = hollow-particle post-processing runs automatically.
PARTICLEVISION_VERSION = "1.1.0"


@dataclass
class Config:
    """Master configuration for the ParticleVision segmentation pipeline."""

    # ── Paths ──────────────────────────────────────────────────────────────
    project_root: str = PROJECT_ROOT
    raw_dir: str = os.path.join(PROJECT_ROOT, "InstSeg", "Raw")
    seg_dir: str = os.path.join(PROJECT_ROOT, "InstSeg", "Segmented")
    checkpoint_dir: str = os.path.join(PROJECT_ROOT, "checkpoints")
    output_dir: str = os.path.join(PROJECT_ROOT, "outputs")

    # ── Dataset ────────────────────────────────────────────────────────────
    # Values used for the corrected_v1 models. Read only by the training / data code (src/dataset.py, the training
    # scripts), never by the app, inference or post-processing. data_split.json is the authority for which image is
    # in which set; usable_image_indices() below gives the same 111 images.
    num_images: int = 120  # Image_001 ... Image_120 (raw photo + label each)
    excluded_images: List[int] = field(
        default_factory=lambda: [
            115,  # label file unusable (1-bit black/white instead of 4 classes)
            20,  # duplicate photo of 019
            29,  # duplicate of 028 (photo and label)
            79,  # shares its label with 078; no correct label exists for this photo
            90,  # duplicate of 089 (photo and label)
            99,  # duplicate of 098 (photo and label)
            100,  # duplicate of 098 (photo and label)
            109,  # shares its label with 110; the label belongs to 110
            119,  # label registration offset, no reliable correction found
        ]
    )  # 120 - 9 = 111 images. Labels 009 and 019 were shifted into alignment by whole pixels (see training/README.md).
    # Image 009: raw height (3265) > seg height (3259). Crop raw to match.
    mismatched_images: Dict[int, str] = field(default_factory=lambda: {9: "crop_raw_bottom_6"})

    # ── Class definitions ──────────────────────────────────────────────────
    num_classes: int = 4
    class_names: List[str] = field(default_factory=lambda: ["Grains", "Void", "Water", "Fines"])
    # Exact RGB → class index mapping (order matters: index 0‒3)
    class_colors: Dict[str, Tuple[int, int, int]] = field(
        default_factory=lambda: {
            "Grains": (0, 0, 0),  # black
            "Void": (144, 238, 144),  # light green
            "Water": (100, 149, 237),  # cornflower blue
            "Fines": (255, 0, 0),  # pure red
        }
    )

    @property
    def color_to_index(self) -> Dict[Tuple[int, int, int], int]:
        """Map RGB tuple → class index."""
        return {rgb: i for i, rgb in enumerate(self.class_colors.values())}

    @property
    def index_to_color(self) -> np.ndarray:
        """(num_classes, 3) array for fast index→RGB lookup."""
        return np.array(list(self.class_colors.values()), dtype=np.uint8)

    # ── Tiling ─────────────────────────────────────────────────────────────
    patch_size: int = 512
    overlap: int = 64  # pixels of overlap between adjacent tiles
    padding_mode: str = "reflect"  # np.pad / torch.nn.functional.pad mode

    @property
    def stride(self) -> int:
        return self.patch_size - self.overlap  # 448

    # ── Augmentation ───────────────────────────────────────────────────────
    aug_hflip_p: float = 0.5
    aug_vflip_p: float = 0.5
    aug_rot90_p: float = 0.5
    aug_brightness_contrast_p: float = 0.3
    aug_brightness_limit: float = 0.10  # ±10 %
    aug_contrast_limit: float = 0.10
    aug_hue_sat_p: float = 0.2
    aug_hue_shift_limit: int = 5  # OpenCV hue units (0-179) on uint8 patches: ±5 = ±10°
    aug_sat_shift_limit: int = 10  # on the 0-255 scale
    aug_val_shift_limit: int = 10  # on the 0-255 scale

    # ── Train / Val / Test split ───────────────────────────────────────────
    # data_split.json is the authority: 79 / 16 / 16 images. It was made by a random whole-image split
    # (scikit-learn train_test_split, seed 42) of the 119 images left after excluding 115 (84 / 17 / 18), then the
    # 8 further exclusions above were removed from it. The ratios below describe the result; no code reads them.
    split_seed: int = 42
    train_ratio: float = 0.712  # 79 / 111
    val_ratio: float = 0.144  # 16 / 111
    test_ratio: float = 0.144  # 16 / 111

    # ── Label cleaning ────────────────────────────────────────────────────
    clean_noise_labels: bool = False  # remove 1-2px fines blobs from GT masks
    clean_noise_max_size: int = 2  # blobs with ≤ this many pixels are removed

    # ── Model ──────────────────────────────────────────────────────────────
    encoder_name: str = "efficientnet-b3"
    encoder_weights: str = "imagenet"
    decoder_channels: List[int] = field(default_factory=lambda: [256, 128, 64, 32, 16])
    # ImageNet normalisation stats
    img_mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    img_std: Tuple[float, float, float] = (0.229, 0.224, 0.225)

    # ── Loss ───────────────────────────────────────────────────────────────
    focal_alpha: List[float] = field(default_factory=lambda: [1.0, 1.8, 2.9, 50.0])
    focal_gamma: float = 2.0
    dice_smooth: float = 1.0
    dice_fines_weight: float = 3.0  # extra Dice weight multiplier for Fines
    loss_focal_weight: float = 0.5
    loss_dice_weight: float = 0.5

    # ── Training ───────────────────────────────────────────────────────────
    batch_size: int = 8
    num_workers: int = 0  # MPS works best with 0 workers on macOS
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    # CosineAnnealingWarmRestarts scheduler
    scheduler_T_0: int = 10
    scheduler_T_mult: int = 2

    # Staged training: (fraction_of_train_data, epochs)
    training_stages: List[Tuple[float, int]] = field(
        default_factory=lambda: [
            (0.20, 30),  # Stage 1: int(79 x 0.20) = 15 images, up to 30 epochs
            (0.60, 20),  # Stage 2: int(79 x 0.60) = 47 images, up to 20 epochs
            (1.00, 20),  # Stage 3: all 79 training images, up to 20 epochs
        ]
    )
    early_stopping_patience: int = 7

    # ── Inference ──────────────────────────────────────────────────────────
    inference_batch_size: int = 16

    # ── Post-processing of the final map: fill hollow microplastic particles (src/postprocess.py) ──
    # The test metrics reported in the paper were measured on the unfilled model output; hole filling is applied
    # after the model, only to the saved maps.
    #
    # Former three-step method (hole filling, ring completion, circle fit; postprocess_microplastic_steps).
    # Not used by the ParticleVision pipeline (kept from earlier experiments).
    postprocess_fill_holes: bool = True
    postprocess_closing_radius: int = 0  # no closing (r = 1 merged scattered particles)
    postprocess_max_hole_area: int = 39  # px; larger enclosed regions are left as they are
    # (former step 2: per-object ring completion, src/postprocess.complete_microplastic_rings)
    postprocess_ring_method: str = "object_closing"  # "none" switches the ring step off
    postprocess_ring_closing_radius: int = 2
    postprocess_ring_max_object_size: int = 15
    # (former step 3: least-squares circle fit, src/postprocess.fill_fitted_circles)
    postprocess_circle_fit: bool = True
    postprocess_circle_r_min: float = 2.8  # px
    postprocess_circle_r_max: float = 4.2  # px
    postprocess_circle_max_rms: float = 1.0  # px
    postprocess_circle_min_arc: float = 270.0  # degrees
    # ── Closing (src/postprocess.close_microplastic): used for 5 um particles and when no particle size is found in
    # the experiment folder name ("...-D05 (40)"); radius per size below. Closely spaced particles may join.
    postprocess_method: str = (
        "closing"  # Not used by the ParticleVision pipeline (kept from earlier experiments).
    )
    postprocess_closing_radius_by_size: Dict[int, int] = field(
        default_factory=lambda: {5: 1, 7: 2, 10: 3}
    )
    postprocess_closing_default_radius: int = 2
    postprocess_closing_use_size: bool = True  # False = always the default radius
    postprocess_closing_max_hole_area: int = (
        39  # px; enclosed Air / Water regions up to this size are filled
    )
    postprocess_closing_max_added_region: int = (
        50  # px; a larger added patch (a cluster turning into a blob) is not added
    )
    # ── Method per particle size: 7 and 10 um particles are redrawn as filled circles (src/postprocess.
    # reconstruct_circles); 5 um particles and images with no size in the folder name get the closing above
    # (5 um: r 1; no size: r 2).
    postprocess_method_by_size: Dict[int, str] = field(
        default_factory=lambda: {5: "closing", 7: "circles", 10: "circles"}
    )
    postprocess_circle_radii_by_size: Dict[int, List[float]] = field(
        default_factory=lambda: {7: [2.0, 2.3, 2.6], 10: [3.0, 3.3, 3.6]}
    )  # px, ring centre line
    postprocess_circle_min_support: float = 0.6  # red / non-grain pixels in the ring band
    postprocess_circle_band: float = 0.75  # px; ring band = |distance - r| <= band
    postprocess_circle_min_visible: float = 0.5  # at most half of the ring band may be grain
    postprocess_circle_max_core_fill: float = (
        0.5  # the core (distance <= r - 1) must be less than half red
    )
    postprocess_circle_nms_factor: float = (
        1.5  # centres at least 1.5 x (median radius + disc margin) apart
    )
    postprocess_circle_disc_margin: float = 0.5  # disc drawn with radius r + 0.5 px
    postprocess_circle_max_overlaps: int = (
        2  # a circle whose disc overlaps >= 2 other circles is not drawn
    )
    # ── In the segmentation pipeline (version 1.1.0): applied per image after stitching, fusion and argmax, before
    # the result is saved (app/lab_structure.run_lab_batch). The app's Run page (Advanced) can change these per run.
    postprocess_in_pipeline: bool = True  # "Fill hollow microplastic particles" (default ON)
    postprocess_save_raw: bool = (
        False  # also save Fused_St_Filt_Comp_Seg_Raw.tif (the unfilled map)
    )

    # ── Reproducibility ────────────────────────────────────────────────────
    seed: int = 42

    # ── Helpers ────────────────────────────────────────────────────────────
    def ensure_dirs(self):
        """Create output directories if they don't exist."""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)

    def usable_image_indices(self) -> List[int]:
        """Return sorted list of usable image indices (1‒120, minus excluded)."""
        return sorted(set(range(1, self.num_images + 1)) - set(self.excluded_images))

    def raw_path(self, idx: int) -> str:
        return os.path.join(self.raw_dir, f"Image_{idx:03d}.jpg")

    def seg_path(self, idx: int) -> str:
        return os.path.join(self.seg_dir, f"Image_{idx:03d}.tif")


# ---------------------------------------------------------------------------
# Convenience: module-level default instance
# ---------------------------------------------------------------------------
CFG = Config()
