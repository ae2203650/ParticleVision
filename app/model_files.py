"""The model files ParticleVision needs: where they go, their sizes and MD5 checksums, and where to get them.

The public repository doesn't contain them (GitHub's 100 MB file limit); the lab package does. Used by
check_install.py, the app (a clear message instead of a traceback when one is missing) and tools/download_models.py.
"""

import hashlib
import os
import urllib.parse
from typing import List, Optional, Tuple

# The Zenodo record that holds the model files (published together with the paper).
MODEL_DOI = "10.5281/zenodo.23691786"
DOI_URL = f"https://doi.org/{MODEL_DOI}"
DOWNLOAD_URL = "https://zenodo.org/records/23691786/files"  # each file is <DOWNLOAD_URL>/<file name>?download=1

MODEL_FILES = [
    {
        "path": "checkpoints/corrected_v1/stage3_unet_corrected_best.pth",
        "what": "U-Net (4-class segmentation)",
        "size": 154051847,
        "md5": "ade212f6d13f393cb2a27792eb4797b6",
        "required": True,
    },
    {
        "path": "checkpoints/corrected_v1/fines_detector_corrected_best.pth",
        "what": "microplastic detector",
        "size": 951887,
        "md5": "6338e290cfdb999e509cf7f04587bbcd",
        "required": True,
    },
    {
        "path": "accel/unet_fp32.onnx",
        "what": "U-Net, ONNX (optional Windows faster mode only)",
        "size": 50304436,
        "md5": "1b412c00373613cea25ed2f98706db4f",
        "required": False,
    },
    {
        "path": "accel/fines_fp32.onnx",
        "what": "microplastic detector, ONNX (optional Windows faster mode only)",
        "size": 930523,
        "md5": "7f67f77e7dadf82f1d1fe7ef42132680",
        "required": False,
    },
]


def file_url(base_url: str, name: str) -> str:
    """The download address of one model file: <base URL>/<file name>?download=1 (the Zenodo file link)."""
    return f"{base_url.rstrip('/')}/{urllib.parse.quote(name)}?download=1"


def md5_of(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def problems(root: str, required_only: bool = True, verify_md5: bool = True) -> List[Tuple[dict, str]]:
    """[(model file entry, "missing" | "damaged (…)")] for the files that are not in place. With verify_md5=False
    only presence and size are checked (fast; what the app does at every start)."""
    out = []
    for entry in MODEL_FILES:
        if required_only and not entry["required"]:
            continue
        path = os.path.join(root, *entry["path"].split("/"))
        if not os.path.isfile(path):
            out.append((entry, "missing"))
        elif os.path.getsize(path) != entry["size"]:
            out.append((entry, f"damaged (size {os.path.getsize(path):,} bytes, expected {entry['size']:,})"))
        elif verify_md5 and md5_of(path) != entry["md5"]:
            out.append((entry, "damaged (MD5 checksum does not match)"))
    return out


def problem_message(root: str, found: List[Tuple[dict, str]], markdown: bool = False) -> str:
    """Plain-language message: which file is missing or damaged, where to download it, where to put it."""
    tick = "`" if markdown else ""
    lines = ["The model files are not in place:"]
    for entry, why in found:
        lines.append(f"  - {tick}{entry['path']}{tick} ({entry['what']}): {why}")
    folders = sorted({os.path.dirname(os.path.join(root, *e["path"].split("/"))) for e, _ in found})
    lines += [
        "",
        f"Download them from Zenodo: {DOI_URL} (see README.md → 'Download the model files').",
        "Or download them and put them in place in one step (the MD5 checksums are checked):",
        f"  {tick}python tools/download_models.py --base-url {DOWNLOAD_URL}{tick}",
        "If you downloaded them yourself, put each file in this folder, with exactly this name:",
        *[f"  - {tick}{f}{tick}" for f in folders],
        "or, with the downloaded files in one folder, run:",
        f"  {tick}python tools/download_models.py --from-dir <that folder>{tick}",
        "It puts them in place and checks their MD5 checksums.",
    ]
    return "\n".join(lines)


def required_problems_message(root: str, verify_md5: bool = True) -> Optional[str]:
    """The message for the required model files, or None when they are all in place."""
    found = problems(root, required_only=True, verify_md5=verify_md5)
    return problem_message(root, found) if found else None
