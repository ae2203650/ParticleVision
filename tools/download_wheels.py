"""
Offline install: download every package file ParticleVision needs into
wheels/<target>/, on a computer WITH internet. Copy the whole ParticleVision
folder (including wheels/) to the computer without internet and run the setup
script there — it installs from wheels/ automatically, with no internet at all.

    python tools/download_wheels.py                  # for THIS computer (auto-detects the GPU)
    python tools/download_wheels.py --target win-cpu     # Windows, no NVIDIA GPU
    python tools/download_wheels.py --target win-cu126   # Windows, NVIDIA GPU (most)
    python tools/download_wheels.py --target win-cu128   # Windows, NVIDIA RTX 50-series
    python tools/download_wheels.py --target mac         # Mac (Apple Silicon)

Run it with Python 3.12. Sizes: win-cpu ≈ 0.7 GB (measured), win-cu126 / win-cu128 several GB.
"""

import argparse
import os
import platform
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import setup_env  # noqa: E402  (shares the PyTorch version pins and GPU detection)

PLATFORM_TAGS = {
    "win": ["win_amd64"],
    "mac": ["macosx_11_0_arm64", "macosx_14_0_arm64"],
    "linux": ["manylinux_2_28_x86_64", "manylinux2014_x86_64", "manylinux_2_17_x86_64"],
}
# pip evaluates "; platform_system == ..." markers for the computer it RUNS on, so
# packages needed only on the target OS are added explicitly when cross-downloading.
TARGET_ONLY = {
    "win": ["colorama==0.4.6", "watchdog==6.0.0"],
    "linux": ["watchdog==6.0.0"],
    "mac": [],
}


def this_machine_target():
    system = platform.system()
    gpus = [] if system == "Darwin" else setup_env.detect_gpus()[0]
    build, _ = setup_env.choose_torch_build(system, gpus)
    return setup_env.wheels_target(build, system)


def pip_download(args, dest, os_key, cross):
    # --no-deps: requirements.txt already lists the COMPLETE pinned set, so pip must not
    # add anything (it would pick the newest versions for unpinned sub-dependencies).
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--dest",
        dest,
        "--only-binary=:all:",
        "--no-deps",
        *args,
    ]
    if cross:
        cmd += [
            "--python-version",
            "3.12",
            "--implementation",
            "cp",
            "--abi",
            "cp312",
            "--abi",
            "abi3",
            "--abi",
            "none",
        ]
        for tag in PLATFORM_TAGS[os_key]:
            cmd += ["--platform", tag]
        cmd += ["--platform", "any"]
    print("> " + " ".join(cmd), flush=True)
    return subprocess.call(cmd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--target",
        default="auto",
        help="auto, win-cpu, win-cu126, win-cu128, mac, linux-cpu, linux-cu126, linux-cu128",
    )
    a = ap.parse_args()
    target = this_machine_target() if a.target == "auto" else a.target
    os_key = target.split("-")[0]
    build = "mac" if target == "mac" else target.split("-", 1)[1]
    here_key = {"Windows": "win", "Darwin": "mac", "Linux": "linux"}[platform.system()]
    cross = os_key != here_key or sys.version_info[:2] != (3, 12)
    dest = os.path.join(ROOT, "wheels", target)
    os.makedirs(dest, exist_ok=True)
    print(f"Downloading packages for '{target}' into {dest}\n", flush=True)

    torch_pkgs = [
        f"torch=={setup_env.TORCH_VERSION}",
        f"torchvision=={setup_env.TORCHVISION_VERSION}",
    ]
    idx = [] if build == "mac" else ["--index-url", setup_env.TORCH_INDEX.format(build)]
    if pip_download(torch_pkgs + idx, dest, os_key, cross) != 0:
        return 1
    # everything else from PyPI, exactly as pinned (torch lines removed so the
    # CPU build from PyPI never ends up next to the CUDA build)
    with open(os.path.join(ROOT, "requirements.txt"), encoding="utf-8") as f:
        lines = [l for l in f if not l.strip().lower().startswith(("torch==", "torchvision=="))]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as tmp:
        tmp.writelines(lines)
    try:
        if (
            pip_download(
                ["-r", tmp.name] + (TARGET_ONLY[os_key] if cross else []), dest, os_key, cross
            )
            != 0
        ):
            return 1
    finally:
        os.unlink(tmp.name)
    # verify: plan an install from this folder alone (no internet) for the target
    check = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--dry-run",
        "--ignore-installed",
        "--no-index",
        "--find-links",
        dest,
        "--only-binary=:all:",
        "--target",
        tempfile.mkdtemp(),
        *torch_pkgs,
        "-r",
        os.path.join(ROOT, "requirements.txt"),
    ]
    if cross:
        check += ["--python-version", "3.12", "--implementation", "cp"]
        for tag in PLATFORM_TAGS[os_key]:
            check += ["--platform", tag]
    print("\nChecking the folder is complete (offline install plan) ...", flush=True)
    if subprocess.call(check) != 0:
        print("The downloaded set is INCOMPLETE — see the message above.")
        return 1
    size = sum(os.path.getsize(os.path.join(dest, f)) for f in os.listdir(dest))
    print(f"\nDone: {len(os.listdir(dest))} files, {size / 1024**3:.2f} GB in wheels/{target}")
    print(
        "Copy the whole ParticleVision folder (with wheels/) to the offline computer and run setup there."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
