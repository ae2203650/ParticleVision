"""
Install the OPTIONAL faster mode into the existing .pv_env, prove nothing else
changed, then run the benchmark (tools/benchmark_accel.py).

    .pv_env\\Scripts\\python.exe tools\\setup_accel.py          (setup_accel_windows.bat)
    ... --no-benchmark        install + checks only
    <any Python 3.12> tools/setup_accel.py --download
        on a computer WITH internet: saves the Windows wheels to wheels/win-accel/ for an
        offline install (copy the whole folder, then run setup_accel_windows.bat there)

Installs requirements-accel-windows.txt with `pip install --no-deps` (online, or from
wheels/win-accel/ if that folder exists), then compares `pip freeze` before and after:
every package that was there must still have exactly the same version, and only the
listed packages may be added. Writes setup_accel_log.txt.
"""

import argparse
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REQ = os.path.join(ROOT, "requirements-accel-windows.txt")
WHEELS = os.path.join(ROOT, "wheels", "win-accel")
LOG = os.path.join(ROOT, "setup_accel_log.txt")

_log = None


def say(msg=""):
    print(msg, flush=True)
    if _log:
        _log.write(msg + "\n")
        _log.flush()


def norm(name):
    return re.sub(r"[-_.]+", "-", name.lower())


def accel_pins(windows=True):
    """{name: version} from requirements-accel-windows.txt (Windows markers applied)."""
    pins = {}
    for line in open(REQ, encoding="utf-8"):
        line = line.split("#")[0].strip()
        if not line:
            continue
        req, _, marker = line.partition(";")
        if "platform_system" in marker and ('"Windows"' in marker) != windows:
            continue
        name, ver = req.strip().split("==")
        pins[norm(name)] = ver.strip()
    return pins


def freeze(py):
    out = subprocess.run(
        [py, "-m", "pip", "freeze", "--all"], capture_output=True, text=True
    ).stdout
    pkgs = {}
    for line in out.splitlines():
        if "==" in line:
            n, v = line.split("==", 1)
            pkgs[norm(n)] = v.strip()
    return pkgs


def compare_freeze(before, after, allowed_new):
    """(ok, problems): nothing that existed changed or disappeared; only `allowed_new` added."""
    problems = []
    for n, v in before.items():
        if n not in after:
            problems.append(f"{n} {v} was removed")
        elif after[n] != v:
            problems.append(f"{n} changed {v} → {after[n]}")
    for n in sorted(set(after) - set(before)):
        if n not in allowed_new:
            problems.append(f"unexpected new package {n} {after[n]}")
        elif after[n] != allowed_new[n]:
            problems.append(f"{n} installed as {after[n]}, expected {allowed_new[n]}")
    return not problems, problems


def download():
    os.makedirs(WHEELS, exist_ok=True)
    pins = accel_pins(windows=True)
    cmd = [
        sys.executable,
        "-m",
        "pip",
        "download",
        "--no-deps",
        "--only-binary=:all:",
        "--platform",
        "win_amd64",
        "--python-version",
        "3.12",
        "--implementation",
        "cp",
        "--abi",
        "cp312",
        "-d",
        WHEELS,
        *[f"{n}=={v}" for n, v in pins.items()],
    ]
    code = subprocess.call(cmd)
    print(
        f"\n{'Done' if code == 0 else 'FAILED'}: {len(os.listdir(WHEELS))} file(s) in wheels/win-accel"
    )
    return code


def main():
    global _log
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--no-benchmark", action="store_true")
    a = ap.parse_args()
    if a.download:
        return download()

    _log = open(LOG, "w", encoding="utf-8")
    say("=" * 70)
    say("  ParticleVision — faster mode setup (optional)")
    say("=" * 70)
    say(f"Date:   {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"Python: {sys.version.split()[0]} ({sys.executable})")
    if os.path.basename(sys.prefix) != ".pv_env":
        say(
            "SETUP FAILED — run this with ParticleVision's own Python (.pv_env). "
            "Use setup_accel_windows.bat, after setup_windows.bat has succeeded."
        )
        return 1
    py = sys.executable
    windows = os.name == "nt"
    pins = accel_pins(windows=windows)

    say("\n[1/3] Recording the installed packages ...")
    before = freeze(py)
    say(f"      {len(before)} packages")
    allowed = {n: v for n, v in pins.items() if n not in before}
    say(f"[2/3] Installing (--no-deps): {', '.join(f'{n}=={v}' for n, v in pins.items())}")
    src = ["--no-index", "--find-links", WHEELS] if os.path.isdir(WHEELS) else []
    if src:
        say("      from wheels/win-accel (offline)")
    code = subprocess.call(
        [py, "-m", "pip", "install", "--no-deps", *src, *[f"{n}=={v}" for n, v in pins.items()]]
    )
    if code != 0:
        say(
            "SETUP FAILED — the packages could not be installed (see above). Nothing else was changed."
        )
        return 1

    say("[3/3] Checking that nothing else changed ...")
    after = freeze(py)
    ok, problems = compare_freeze(before, after, allowed)
    for p in problems:
        say(f"      PROBLEM: {p}")
    if not ok:
        say(
            "SETUP FAILED — the normal installation was changed. Run setup_windows.bat again to restore it."
        )
        return 1
    say(
        f"      OK — {len(before)} packages unchanged, added: "
        + (", ".join(f"{n} {after[n]}" for n in sorted(allowed)) or "nothing (already installed)")
    )
    chk = subprocess.run([py, "-m", "pip", "check"], capture_output=True, text=True)
    say(f"      pip check: {(chk.stdout + chk.stderr).strip()}")
    try:
        import onnxruntime as ort

        say(f"      ONNX Runtime {ort.__version__}: {', '.join(ort.get_available_providers())}")
    except Exception as e:  # noqa: BLE001
        say(f"      ONNX Runtime could not be loaded: {type(e).__name__}: {e}")
    _log.close()
    _log = None

    if a.no_benchmark:
        return 0
    print("\nStarting the benchmark (this takes several minutes) ...\n", flush=True)
    return subprocess.call([py, os.path.join(ROOT, "tools", "benchmark_accel.py")], cwd=ROOT)


if __name__ == "__main__":
    sys.exit(main())
