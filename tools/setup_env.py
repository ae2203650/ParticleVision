"""
ParticleVision setup — builds a fresh .pv_env and installs everything.

Run by setup_windows.bat / setup_mac.sh with Python 3.12. Steps:
  1. check Python (3.12, 64-bit) and, on Windows/Linux, Tcl/Tk (folder picker)
  2. delete the old .pv_env (only that folder — never .venv) and create a new one
  3. pick the PyTorch build: NVIDIA GPU → CUDA build, otherwise CPU (Mac: MPS build)
  4. install PyTorch, then requirements.txt — online, or offline from wheels/<target>/
  5. run check_install.py (the self-test) inside the new environment
Everything shown on screen is also written to setup_log.txt.

    python tools/setup_env.py            (normally started by the setup script)
    PV_TORCH=cpu python tools/setup_env.py   force the CPU build (cpu / cu126 / cu128)
"""

import os
import platform
import re
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_DIR = os.path.join(ROOT, ".pv_env")
LOG_PATH = os.path.join(ROOT, "setup_log.txt")
REQUIRED_PYTHON = (3, 12)
TORCH_VERSION, TORCHVISION_VERSION = "2.8.0", "0.23.0"
TORCH_INDEX = "https://download.pytorch.org/whl/{}"
MIN_WINDOWS_DRIVER = 528.33  # oldest NVIDIA driver that runs CUDA 12.x builds on Windows
MIN_LINUX_DRIVER = 525.60

_log = None


def say(msg=""):
    print(msg, flush=True)
    if _log:
        _log.write(msg + "\n")
        _log.flush()


def run(cmd, **kw):
    """Run a command, showing and logging its output line by line. Returns exit code."""
    say("> " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        **kw,
    )
    for line in p.stdout:
        say("  " + line.rstrip())
    return p.wait()


def fail(msg):
    say("")
    say("=" * 70)
    say("  SETUP FAILED")
    say("=" * 70)
    for line in msg.strip().splitlines():
        say("  " + line)
    say("")
    say(f"  The full log is here: {LOG_PATH}")
    say("  If you ask for help, send that file.")
    sys.exit(1)


def env_python():
    if os.name == "nt":
        return os.path.join(ENV_DIR, "Scripts", "python.exe")
    return os.path.join(ENV_DIR, "bin", "python")


# ─────────────────────────────────────────────────────────────────────────
# PyTorch build choice (pure function — unit-tested with fake nvidia-smi output)
# ─────────────────────────────────────────────────────────────────────────
def parse_nvidia_smi(text):
    """'NVIDIA GeForce RTX 4060, 551.86, 8.9' lines → list of (name, driver, cc)."""
    gpus = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name, driver = parts[0], parts[1]
        cc = parts[2] if len(parts) > 2 else ""
        try:
            driver_f = float(re.match(r"\d+(\.\d+)?", driver).group(0))
        except (AttributeError, ValueError):
            driver_f = 0.0
        try:
            cc_f = float(cc)
        except ValueError:
            cc_f = None
        gpus.append((name, driver_f, cc_f))
    return gpus


def choose_torch_build(system, gpus, override=None):
    """→ (build, explanation). build is 'mac', 'cpu', 'cu126' or 'cu128'."""
    if override:
        return override, f"forced by PV_TORCH={override}"
    if system == "Darwin":
        return "mac", "macOS: standard PyTorch build (uses the Apple GPU through MPS)"
    if not gpus:
        return "cpu", "no NVIDIA GPU found → CPU build (works everywhere, slower)"
    name, driver, cc = gpus[0]
    min_driver = MIN_WINDOWS_DRIVER if system == "Windows" else MIN_LINUX_DRIVER
    if driver < min_driver:
        return "cpu", (
            f"NVIDIA GPU found ({name}) but its driver {driver} is too old (need {min_driver} or newer) "
            f"→ CPU build for now. Update the NVIDIA driver and run setup again to use the GPU."
        )
    if cc is not None and cc < 5.0:
        return "cpu", f"NVIDIA GPU found ({name}) but it is too old for PyTorch 2.8 → CPU build"
    if cc is not None and cc >= 10.0:
        return (
            "cu128",
            f"NVIDIA GPU found ({name}, driver {driver}) → CUDA 12.8 build (newest GPUs)",
        )
    return "cu126", f"NVIDIA GPU found ({name}, driver {driver}) → CUDA 12.6 build"


def detect_gpus():
    exe = shutil.which("nvidia-smi")
    if exe is None and os.name == "nt":
        cand = r"C:\Windows\System32\nvidia-smi.exe"
        exe = cand if os.path.exists(cand) else None
    if exe is None:
        return [], "nvidia-smi not found"
    for q in ("name,driver_version,compute_cap", "name,driver_version"):
        try:
            r = subprocess.run(
                [exe, f"--query-gpu={q}", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except Exception as e:  # noqa: BLE001
            return [], f"nvidia-smi failed: {e}"
        if r.returncode == 0 and r.stdout.strip():
            return parse_nvidia_smi(r.stdout), r.stdout.strip()
    return [], "nvidia-smi found but reported no GPU"


def wheels_target(build, system):
    return {"Windows": "win", "Darwin": "mac", "Linux": "linux"}[system] + (
        "" if build == "mac" else f"-{build}"
    )


# ─────────────────────────────────────────────────────────────────────────
def main():
    global _log
    _log = open(LOG_PATH, "w", encoding="utf-8")
    system = platform.system()
    say("=" * 70)
    say("  ParticleVision setup")
    say("=" * 70)
    say(f"Date:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"Folder:   {ROOT}")
    say(f"System:   {platform.platform()} ({platform.machine()})")
    say(f"Python:   {sys.version.split()[0]}  ({sys.executable})")
    say("")

    # 1. Python / Tcl-Tk -------------------------------------------------------
    say("[1/5] Checking Python ...")
    if sys.version_info[:2] != REQUIRED_PYTHON:
        fail(
            f"This is Python {sys.version.split()[0]}, but ParticleVision needs Python 3.12.\n"
            f"Install Python 3.12 from https://www.python.org/downloads/ and run setup again."
        )
    if sys.prefix != sys.base_prefix:
        fail(
            f"This Python is inside a virtual environment ({sys.prefix}).\n"
            "Run setup with the normal Python 3.12 installation (double-click the setup file)."
        )
    if sys.maxsize <= 2**32:
        fail("This Python is 32-bit. Install the 64-bit Python 3.12 from python.org.")
    if system == "Windows" and platform.machine().upper() not in ("AMD64", "X86_64"):
        fail(
            "This Python is built for ARM. Install the 64-bit (x64) Python 3.12 from python.org\n"
            "— Windows on ARM runs it automatically. See docs/TROUBLESHOOTING.md, 'Windows on ARM'."
        )
    if system != "Darwin":
        try:
            import tkinter  # noqa: F401
        except Exception:  # noqa: BLE001
            fail(
                "Tcl/Tk is missing from this Python — the folder picker needs it.\n"
                "Run the Python 3.12 installer again, choose 'Modify', tick 'tcl/tk and IDLE', finish,\n"
                "then run setup again."
            )
    say("      OK")

    # 2. fresh environment ------------------------------------------------------
    say("[2/5] Creating a fresh environment in .pv_env ...")
    if os.path.isdir(ENV_DIR):
        say("      removing the previous .pv_env")
        shutil.rmtree(ENV_DIR)
    if run([sys.executable, "-m", "venv", ENV_DIR]) != 0 or not os.path.exists(env_python()):
        fail(
            "Could not create the environment (.pv_env).\n"
            "Move the ParticleVision folder to a simple location such as C:\\ParticleVision and try again."
        )
    say("      OK")

    # 3. PyTorch build ----------------------------------------------------------
    say("[3/5] Choosing the PyTorch build ...")
    gpus, raw = ([], "not checked on macOS") if system == "Darwin" else detect_gpus()
    say(f"      nvidia-smi: {raw}")
    build, why = choose_torch_build(system, gpus, os.environ.get("PV_TORCH"))
    say(f"      {why}")

    # 4. install ------------------------------------------------------------------
    target = wheels_target(build, system)
    wheel_dir = os.path.join(ROOT, "wheels", target)
    py = env_python()
    torch_pkgs = [f"torch=={TORCH_VERSION}", f"torchvision=={TORCHVISION_VERSION}"]
    req = os.path.join(ROOT, "requirements.txt")
    if os.path.isdir(wheel_dir):
        say(f"[4/5] Installing OFFLINE from wheels/{target} (no internet needed) ...")
        code = run(
            [
                py,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                wheel_dir,
                *torch_pkgs,
                "-r",
                req,
            ]
        )
    else:
        say("[4/5] Installing packages from the internet (this can take 5–20 minutes) ...")
        idx = [] if build == "mac" else ["--index-url", TORCH_INDEX.format(build)]
        code = run([py, "-m", "pip", "install", *torch_pkgs, *idx])
        if code == 0:
            code = run([py, "-m", "pip", "install", "-r", req])
    if code != 0:
        fail(
            "Installing the packages failed (see the lines above).\n"
            "Most common causes: no internet, or a university network that blocks downloads.\n"
            "See docs/TROUBLESHOOTING.md → 'pip install fails' (it also explains the offline install)."
        )
    say("      OK")

    # 5. self-test --------------------------------------------------------------
    say(
        "[5/5] Running the self-test (segments one sample image — up to a few minutes on a CPU) ..."
    )
    code = run([py, os.path.join(ROOT, "check_install.py")], cwd=ROOT)
    say("")
    if code != 0:
        fail(
            "The self-test did not pass (see the lines above).\n"
            "Send selftest/check_install_log.txt and setup_log.txt."
        )
    say("=" * 70)
    say("  SUCCESS — ParticleVision is installed and the self-test passed.")
    say("=" * 70)
    say(
        "  Start the app with: "
        + (
            "run_app.bat (double-click)"
            if system == "Windows"
            else "run_app.command (double-click) or ./run_app.sh"
        )
    )
    say(
        "  Optional, more thorough: "
        + ("run_full_check.bat" if system == "Windows" else "run_full_check.command")
        + " checks 3 more images (Demo_Lab)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
