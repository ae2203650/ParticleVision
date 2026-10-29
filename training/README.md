# Training the ParticleVision models

How the two `corrected_v1` models were trained, so the training described in the paper can be checked and repeated.
The app doesn't need anything in this folder; see the [main README](../README.md) to use ParticleVision.

**Images and labels are not included; contact the authors.**

| File | What it is |
|---|---|
| `train_unet.py` | trains the U-Net (4-class segmentation), 3 stages |
| `train_detector.py` | trains the microplastic detector (FinesDetector) |
| `common.py` | the corrected_v1 settings (from notebook 39) and shared helpers |
| `data_split.json` | the exact split file used (MD5 `cd46bc7dabff8c3aa71bbcf7d9d88964`), unchanged |
| `notebook_39_training_record.ipynb` | the original training record, cleaned of images and paths |

---

## 1. The data expected

Two folders with the 120 image / label pairs, named by number:

```
<Raw>/Image_001.jpg … Image_120.jpg          the microscope photos (RGB)
<Segmented>/Image_001.tif … Image_120.tif    the hand labels (RGB, 4 class colours)
```

Label colours: grains `(0, 0, 0)`, air `(144, 238, 144)`, water `(100, 149, 237)`, microplastic `(255, 0, 0)`.

**111 images were used** (120 minus 9), split by whole image as in `data_split.json`: **79 training, 16 validation,
16 test**. The 9 exclusions (also listed in `src/config.py`):

| Image | Why |
|---|---|
| 115 | label file unusable (1-bit black / white, not 4 classes) |
| 020 | duplicate photo of 019 |
| 029 | duplicate of 028 (photo and label) |
| 090 | duplicate of 089 (photo and label) |
| 099, 100 | duplicates of 098 (photo and label) |
| 079 | shares its label with 078; no correct label exists for this photo |
| 109 | shares its label with 110; the label belongs to 110 |
| 119 | label registration offset; no reliable correction was found |

**Alignment fixes (in the label files used):** the labels of Image_009 and Image_019 were shifted by whole pixels to
match their photos. Image_009's label content moved 13 px down and 5 px right; Image_019's moved 1 px up and 2 px
left. The strip exposed at the image edge was filled by copying the nearest edge row / column. Image_009's photo has
6 more rows than its label; the bottom 6 rows of the photo are cropped when it is loaded (`src/config.py`,
`mismatched_images`).

**How the split was made:** a random whole-image split (scikit-learn `train_test_split`, seed 42) of the 119 images
left after excluding 115 gave 84 / 17 / 18; the other 8 exclusions were then removed from it. `data_split.json` is
the authority. Stages 1 and 2 of the U-Net use 15 and 47 of the 79 training images, drawn with NumPy
`RandomState(42)` (`src/dataset.create_datasets`).

## 2. Install

**Training used Python 3.9** (`requirements-dev.txt`); **the app uses Python 3.12** (`requirements.txt`). Install the
training packages into a Python 3.9 environment:

```
python3.9 -m pip install -r requirements-dev.txt
```

## 3. Commands

Run from the repository root. Each script writes into a **new** folder (it refuses an existing, non-empty folder,
and any folder under `checkpoints/`, so the shipped model files are never overwritten).

```
python training/train_unet.py     --raw-dir <Raw> --seg-dir <Segmented> --out training/runs/unet
python training/train_detector.py --raw-dir <Raw> --seg-dir <Segmented> --out training/runs/detector
```

| Model | Result file | Selected by |
|---|---|---|
| U-Net | `<out>/stage3_unet_corrected_best.pth` | highest validation mIoU (patch level) |
| Detector | `<out>/fines_detector_corrected_best.pth` | lowest validation loss |

To use your own models in the app, copy them to `checkpoints/corrected_v1/` under the same names. The app's
self-test then reports that their MD5 checksums differ from the published models.

**Smoke test** (does the script run end to end? 2 images, 2 batches, 1 epoch per stage, on the CPU; a few minutes):

```
python training/train_unet.py     --raw-dir <Raw> --seg-dir <Segmented> --out /tmp/smoke_unet --smoke --device cpu
python training/train_detector.py --raw-dir <Raw> --seg-dir <Segmented> --out /tmp/smoke_det  --smoke --device cpu
```

## 4. The settings used for corrected_v1

From notebook 39's code and printed outputs (`notebook_39_training_record.ipynb`), set explicitly in `common.py`.

**U-Net:** EfficientNet-B3 encoder (ImageNet weights), decoder 256 / 128 / 64 / 32 / 16, 4 classes. Loss
0.5 × focal (class weights 1.0 / 1.8 / 2.9 / 50, γ 2) + 0.5 × Dice (microplastic weight 3). AdamW, learning rate
1e-4, weight decay 1e-4; CosineAnnealingWarmRestarts (T_0 10, T_mult 2), restarted each stage; batch 8;
512 × 512 patches with 64 px overlap.

| Stage | Training images | Patches | Max epochs | Epochs run | Best epoch (val mIoU) |
|---|---|---|---|---|---|
| 1 | 15 | 1,198 | 30 | 21 (early stop, patience 7) | 14 (0.8468) |
| 2 | 47 | 3,666 | 20 | 14 (early stop) | 7 (0.8530) |
| 3 | 79 | 6,146 | 20 | 20 | 19 (0.8634) |

Each stage starts from the previous stage's best weights. Validation: 16 images, 1,260 patches.

**Detector:** 10 dilated convolutions (base 32 channels), from scratch. Target: a Gaussian (σ 2 px) at the centroid
of every labelled microplastic blob of ≥ 3 px. CenterNet focal loss (α 2, β 4). AdamW, learning rate 3e-4, weight
decay 1e-4; CosineAnnealingWarmRestarts (T_0 30, T_mult 1, minimum 1e-6); batch 8; 30 epochs (early-stopping
patience 10, not triggered). Best epoch 29 (validation loss 1.4717).

**Augmentation** (both models; `src/dataset.get_train_augmentation`, applied on the fly to each patch):

| Transform | Probability | Limits |
|---|---|---|
| HorizontalFlip | 0.5 | — |
| VerticalFlip | 0.5 | — |
| RandomRotate90 | 0.5 | a random multiple of 90° |
| RandomBrightnessContrast | 0.3 | contrast factor 1 ± 0.10; brightness ± 0.10 × 255 |
| HueSaturationValue | 0.2 | hue ± 5 OpenCV units (± 10°); saturation and value ± 10 (0–255 scale) |
| Normalize | always | ImageNet mean / std |

Validation patches are only normalised. Mask: flips and rotations only (no resampling).

## 5. Hardware and time

What the training record (`notebook_39_training_record.ipynb`) shows. Cell numbers count from 0 as in the original
notebook; the cleaned copy has one extra note cell at the top, so there they are one higher.

| | Record | Where |
|---|---|---|
| Device | PyTorch **MPS** backend (Metal GPU), so the training ran **on a Mac** | cell 5 prints `Device: mps`; cells 7, 9, 11 print `[MPS Mem]` lines |
| Python | **3.9.6**, packages in a per-user `Library/Python/3.9` folder on macOS | notebook metadata `language_info.version`; cell 1 warning paths of the original record (removed from this cleaned copy) |
| Machine name, chip, OS version, PyTorch version | **not recorded** | — |

| Model | Time (sum of the per-epoch times printed in the record, training plus validation) | Where |
|---|---|---|
| U-Net | about **10.9 h**: Stage 1 1.62 h (21 epochs), Stage 2 2.78 h (14), Stage 3 6.50 h (20) | cells 7, 9, 11 |
| Detector | about **13.0 h** (30 epochs) | cell 16 |

## 6. Reproducibility

The split and the stage subsets were seeded (42), but **no PyTorch or NumPy seed was set** for the weight
initialisation, the data order or the augmentation, as in the original run. A re-run therefore gives similar but
not bit-identical weights. The published model files are the ones trained in notebook 39; their MD5 checksums are
in the main README.
