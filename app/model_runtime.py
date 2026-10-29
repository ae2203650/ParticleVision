"""
Where the confirmed models run — device choice, offline model build,
checkpoint loading and out-of-memory fallback.

Nothing here changes the models or the inference maths: architecture, weights,
tiling (512 / 64 overlap), logit averaging, gated fusion (scale 8.0, margin
2.0), TTA off and float32 all stay exactly as confirmed. This module only
decides which device does the work, and retries on a smaller batch / the CPU
when a GPU runs out of memory.
"""

import copy
import gc
import os
import time

import segmentation_models_pytorch as smp
import torch

import src.inference as inference  # called via the module so tests can swap it
from src.config import CFG
from src.model import FinesDetector

# ── Confirmed configuration — do not change without re-verifying ──────────
BRANCH1_CKPT = os.path.join(CFG.checkpoint_dir, "corrected_v1", "stage3_unet_corrected_best.pth")
BRANCH2_CKPT = os.path.join(CFG.checkpoint_dir, "corrected_v1", "fines_detector_corrected_best.pth")
FUSION_SCALE = 8.0
FUSION_MARGIN = 2.0
DEFAULT_BATCH = CFG.inference_batch_size  # 16 tiles per forward pass
OOM_BATCH_LADDER = (DEFAULT_BATCH, 8, 4, 1)

# Seconds per image before a run has measured its own speed.
EXPECTED_SEC_PER_IMAGE = {"cuda": 9.4, "mps": 9.4, "cpu": 72.0}
EXPECTED_SEC_NOTE = {
    "cuda": "typical for a GPU",
    "mps": "measured on an Apple M4 GPU",
    "cpu": "measured on an Apple M4 CPU — a laptop CPU may be slower",
}


# ── Device ─────────────────────────────────────────────────────────────────
def select_device() -> torch.device:
    """CUDA (NVIDIA) → MPS (Apple) → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_device(device: torch.device) -> None:
    """Keep true float32 on NVIDIA: PyTorch otherwise uses TF32 (reduced
    precision) convolutions/matmuls on RTX 30/40-series GPUs."""
    if device.type == "cuda":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False


def describe_device(device: torch.device) -> str:
    if device.type == "cuda":
        return f"{torch.cuda.get_device_name(device.index or 0)} (CUDA)"
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return "CPU"


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


# ── Model build + checkpoint load ──────────────────────────────────────────
def build_unet_offline() -> torch.nn.Module:
    """Same network as src.model.build_unet(), built WITHOUT downloading
    ImageNet weights (encoder_weights=None). Every weight is then overwritten
    by our checkpoint, so the loaded model is identical (checked by comparing every
    weight with a model built the online way). src.model.build_unet() can't
    be used for this: it treats encoder_weights=None as "use the config default"
    (ImageNet)."""
    return smp.Unet(
        encoder_name=CFG.encoder_name,
        encoder_weights=None,
        in_channels=3,
        classes=CFG.num_classes,
        decoder_channels=CFG.decoder_channels,
        activation=None,
    )


def load_models(device: torch.device):
    """Checkpoints are read to CPU first (works on every machine, and keeps the
    optimizer state stored inside the U-Net checkpoint off the GPU), then the
    models are moved to `device`."""
    m1 = build_unet_offline()
    s1 = torch.load(BRANCH1_CKPT, map_location="cpu", weights_only=False)
    if isinstance(s1, dict) and "model_state_dict" in s1:
        s1 = s1["model_state_dict"]
    m1.load_state_dict(s1)  # strict: any missing/unexpected key raises
    m2 = FinesDetector(32)
    m2.load_state_dict(torch.load(BRANCH2_CKPT, map_location="cpu", weights_only=False))
    return m1.to(device).eval(), m2.to(device).eval()


# ── Inference with out-of-memory fallback ──────────────────────────────────
def is_oom_error(e: BaseException) -> bool:
    oom_cls = getattr(torch.cuda, "OutOfMemoryError", None)
    return (oom_cls is not None and isinstance(e, oom_cls)) or "out of memory" in str(e).lower()


class SegmentationRunner:
    """Runs the confirmed pipeline on one image at a time.

    Normal path: batch 16 on the selected device (identical to the confirmed
    setup). Only if the GPU runs out of memory for an image: retry that image
    with batch 8 → 4 → 1, then on the CPU. Every new image starts again at
    batch 16 on the GPU, so one bad moment never slows down the rest of the
    run. An image processed with a fallback gets a note for the UI and
    run_log.txt.
    """

    def __init__(self, device: torch.device = None):
        self.device = device or select_device()
        configure_device(self.device)
        self.label = describe_device(self.device)
        self.m1, self.m2 = load_models(self.device)
        self._cpu_models = None

    def _run(self, path, m1, m2, device, batch):
        return inference.predict_full_image(
            model=m1,
            image_path=path,
            device=device,
            model_b2=m2,
            batch_size=batch,
            fusion_scale=FUSION_SCALE,
            fusion_margin=FUSION_MARGIN,
        )

    def predict(self, path):
        """Returns (rgb, class_map, seconds, note). note == "" on the normal path."""
        t0 = time.time()
        if self.device.type == "cpu":
            rgb, cm, _ = self._run(path, self.m1, self.m2, self.device, DEFAULT_BATCH)
            return rgb, cm, time.time() - t0, ""
        for b in OOM_BATCH_LADDER:  # always starts at batch 16
            try:
                rgb, cm, _ = self._run(path, self.m1, self.m2, self.device, b)
                note = (
                    ""
                    if b == DEFAULT_BATCH
                    else f"used batch {b} instead of {DEFAULT_BATCH} — the GPU ran out of memory"
                )
                return rgb, cm, time.time() - t0, note
            except Exception as e:  # noqa: BLE001 — only OOM is retried
                if not is_oom_error(e):
                    raise
                empty_cache(self.device)
        # every GPU batch size ran out of memory → this image on the CPU
        if self._cpu_models is None:
            self._cpu_models = (
                copy.deepcopy(self.m1).to("cpu").eval(),
                copy.deepcopy(self.m2).to("cpu").eval(),
            )
        m1, m2 = self._cpu_models
        rgb, cm, _ = self._run(path, m1, m2, torch.device("cpu"), DEFAULT_BATCH)
        return rgb, cm, time.time() - t0, "ran on the CPU — the GPU ran out of memory"

    def between_images(self):
        empty_cache(self.device)
