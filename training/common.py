"""Shared settings and helpers of the corrected_v1 training scripts (train_unet.py, train_detector.py).

The SETTINGS below are the values used to train the corrected_v1 models (notebooks/39, run on 29-30 Aug 2026),
taken from the notebook's code and its printed outputs - not from today's defaults. They are written into
src.config.CFG before any training code runs, and checked, so a later change of the defaults can't change them.
"""

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.config import CFG  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SPLIT_FILE = os.path.join(HERE, "data_split.json")

# ── corrected_v1 settings (source in brackets) ────────────────────────────────────────────────────────────────────
SETTINGS = {
    # data and patches [NB39 cells 6-10 output: 15 / 47 / 79 train images; 6,146 train, 1,260 val patches]
    "patch_size": 512,
    "overlap": 64,
    "clean_noise_labels": False,
    "seed": 42,  # [create_datasets(..., seed=CFG.seed): the Stage 1 / 2 image subsets]
    # augmentation [src/dataset.get_train_augmentation, values of src/config.py as of the training]
    "aug_hflip_p": 0.5,
    "aug_vflip_p": 0.5,
    "aug_rot90_p": 0.5,
    "aug_brightness_contrast_p": 0.3,
    "aug_brightness_limit": 0.10,
    "aug_contrast_limit": 0.10,
    "aug_hue_sat_p": 0.2,
    "aug_hue_shift_limit": 5,
    "aug_sat_shift_limit": 10,
    "aug_val_shift_limit": 10,
    # U-Net [NB39 cell 5 output; src/model.build_unet; checkpoint optimizer state]
    "encoder_name": "efficientnet-b3",
    "encoder_weights": "imagenet",
    "decoder_channels": [256, 128, 64, 32, 16],
    "loss_focal_weight": 0.5,
    "loss_dice_weight": 0.5,
    "focal_alpha": [1.0, 1.8, 2.9, 50.0],
    "focal_gamma": 2.0,
    "dice_fines_weight": 3.0,
    "batch_size": 8,
    "num_workers": 0,
    "learning_rate": 1e-4,
    "weight_decay": 1e-4,
    "scheduler_T_0": 10,
    "scheduler_T_mult": 2,
    "early_stopping_patience": 7,
    "training_stages": [(0.20, 30), (0.60, 20), (1.00, 20)],  # [NB39 cells 6, 8, 10 output]
}
# FinesDetector [NB39 cells 13-16]
DETECTOR = {
    "base_channels": 32,
    "sigma": 2.0,
    "min_blob_px": 3,
    "batch_size": 8,
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "epochs": 30,
    "patience": 10,
    "eta_min": 1e-6,
    "focal_alpha": 2.0,
    "focal_beta": 4.0,
}


def apply_settings(raw_dir: str, seg_dir: str):
    """Write SETTINGS and the data folders into CFG, then check them."""
    for k, v in SETTINGS.items():
        setattr(CFG, k, v)
    CFG.raw_dir, CFG.seg_dir = os.path.abspath(raw_dir), os.path.abspath(seg_dir)
    for k, v in SETTINGS.items():
        assert getattr(CFG, k) == v, k
    for d in (CFG.raw_dir, CFG.seg_dir):
        if not os.path.isdir(d):
            sys.exit(f"Folder not found: {d}")


def check_images(indices):
    missing = [i for i in indices if not (os.path.isfile(CFG.raw_path(i)) and os.path.isfile(CFG.seg_path(i)))]
    if missing:
        sys.exit(
            f"Missing image/label pairs for {missing[:10]}{' ...' if len(missing) > 10 else ''}: expected "
            f"{CFG.raw_path(missing[0])} and {CFG.seg_path(missing[0])}"
        )


def load_split(path: str, smoke: bool, out_dir: str) -> str:
    """The split file to use. --smoke: a reduced copy (2 train, 2 val, 2 test images) written into out_dir."""
    with open(path, encoding="utf-8") as f:
        split = json.load(f)
    if smoke:
        split = {k: split[k][:2] for k in ("train", "val", "test")}
        path = os.path.join(out_dir, "smoke_split.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=2)
    check_images(split["train"] + split["val"] + split["test"])
    return path


def prepare_out_dir(out_dir: str) -> str:
    """A NEW (or empty) folder for the checkpoints - never the shipped model folder."""
    out_dir = os.path.abspath(out_dir)
    shipped = os.path.abspath(os.path.join(ROOT, "checkpoints"))
    if os.path.commonpath([out_dir, shipped]) == shipped:
        sys.exit(f"Choose an output folder outside {shipped} (the shipped model files are never overwritten).")
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        sys.exit(f"{out_dir} is not empty - choose a new folder (nothing is overwritten).")
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def pick_device(name):
    import torch

    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():  # the order used in NB39 cell 5
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def limit(dataset, n_batches: int, batch_size: int):
    """--smoke: only the first n_batches batches of a dataset."""
    from torch.utils.data import Subset

    return Subset(dataset, list(range(min(len(dataset), n_batches * batch_size))))


def add_common_args(ap):
    ap.add_argument("--raw-dir", required=True, help="folder with the raw photos Image_001.jpg ... Image_120.jpg")
    ap.add_argument("--seg-dir", required=True, help="folder with the labels Image_001.tif ... Image_120.tif")
    ap.add_argument("--split", default=SPLIT_FILE, help="split file (default: training/data_split.json)")
    ap.add_argument("--out", required=True, help="NEW folder for the checkpoints and histories")
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (default: mps, then cuda, then cpu)")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="smoke test only: 2 train / 2 val images, 2 batches, 1 epoch per stage (checks the script runs end to end)",
    )
