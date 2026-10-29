"""
Train the U-Net (4-class segmentation) of the corrected_v1 models: three progressive stages, exactly as in
notebooks/39 cells 5-11 (training/notebook_39_training_record.ipynb).

    python training/train_unet.py --raw-dir <.../InstSeg/Raw> --seg-dir <.../InstSeg/Segmented> --out <new folder>

  Stage 1: 15 of the 79 training images (20 %, NumPy seed 42), up to 30 epochs
  Stage 2: 47 images (60 %), up to 20 epochs, starting from Stage 1's best weights
  Stage 3: all 79 images, up to 20 epochs, starting from Stage 2's best weights
  Each stage: ComboLoss (0.5 focal + 0.5 Dice), AdamW lr 1e-4 / weight decay 1e-4, CosineAnnealingWarmRestarts
  (T_0 10, T_mult 2), batch 8, early stopping after 7 epochs without a better validation mIoU; the best epoch (by
  validation mIoU) is kept. The final model is <out>/stage3_unet_corrected_best.pth.

The settings are in training/common.py (taken from notebook 39, not from today's defaults). No random seed is set
for PyTorch or the data order, as in the original run, so a re-run gives similar but not bit-identical weights.
--smoke runs 2 images, 2 batches and 1 epoch per stage on the given device, only to show the script works.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402  (also puts the repository root on sys.path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1].strip())
    common.add_common_args(ap)
    a = ap.parse_args(argv)

    common.apply_settings(a.raw_dir, a.seg_dir)
    out = common.prepare_out_dir(a.out)
    split_path = common.load_split(a.split, a.smoke, out)

    import torch
    from torch.utils.data import DataLoader

    from src.config import CFG
    from src.dataset import create_datasets
    from src.losses import ComboLoss
    from src.model import build_unet
    from src.training import run_training_stage

    device = common.pick_device(a.device)
    print(f"Device: {device}")
    model = build_unet().to(device)
    criterion = ComboLoss().to(device)
    print(
        f"Loss: ComboLoss (Focal weight={CFG.loss_focal_weight}, Dice weight={CFG.loss_dice_weight}); "
        f"focal alpha {CFG.focal_alpha}, gamma {CFG.focal_gamma}; Dice fines weight {CFG.dice_fines_weight}"
    )
    stages = CFG.training_stages if not a.smoke else [(f, 1) for f, _ in CFG.training_stages]
    val_loader = None
    t_all = time.time()
    for n, (fraction, epochs) in enumerate(stages, start=1):
        print(f"\nStage {n}: {fraction * 100:.0f}% of training data, {epochs} epochs")
        train_ds, val_ds, _ = create_datasets(split_path=split_path, train_fraction=fraction, seed=CFG.seed)
        if a.smoke:
            train_ds, val_ds = common.limit(train_ds, 2, CFG.batch_size), common.limit(val_ds, 2, CFG.batch_size)
        train_loader = DataLoader(
            train_ds, batch_size=CFG.batch_size, shuffle=True, num_workers=CFG.num_workers, pin_memory=False
        )
        if val_loader is None:  # NB39 builds the validation loader once, in Stage 1
            val_loader = DataLoader(
                val_ds, batch_size=CFG.batch_size, shuffle=False, num_workers=CFG.num_workers, pin_memory=False
            )
        print(f"Train: {len(train_ds)} patches -> {len(train_loader)} batches")
        run_training_stage(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            device=device,
            stage_name=f"stage{n}_unet_corrected",
            num_epochs=epochs,
            checkpoint_dir=out,
            save_every_epoch=True,
        )
        del train_ds, train_loader
        if device.type == "mps":
            torch.mps.empty_cache()
    print(
        f"\nU-Net done in {(time.time() - t_all) / 3600:.2f} h. Final model: "
        f"{os.path.join(out, 'stage3_unet_corrected_best.pth')}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
