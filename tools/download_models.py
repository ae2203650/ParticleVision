"""
Put the model files in place and check their MD5 checksums (optional helper).

The public repository does not contain the model files (they are larger than GitHub allows). They are on Zenodo
(https://doi.org/10.5281/zenodo.23691786; README.md → "Download the model files"). Either download them and run

    python tools/download_models.py --from-dir <folder with the downloaded files>

or let this script download them (each file from <URL>/<file name>?download=1):

    python tools/download_models.py --base-url https://zenodo.org/records/23691786/files

Add --with-accel to also place the two ONNX files of the optional Windows faster mode. A file that is already in
place with the right checksum is left alone; a file in place with a wrong checksum is moved aside (renamed
*.bad_<date>), never deleted. Nothing is placed unless its MD5 matches.
"""

import argparse
import os
import shutil
import sys
import tempfile
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app"))

from model_files import DOI_URL, DOWNLOAD_URL, MODEL_FILES, file_url, md5_of  # noqa: E402


def find_in_dir(folder: str, name: str):
    """The first file called `name` anywhere under `folder`, or None."""
    for d, _, files in os.walk(folder):
        if name in files:
            return os.path.join(d, name)
    return None


def fetch(url: str, dst: str):
    req = urllib.request.Request(url, headers={"User-Agent": "ParticleVision-download_models"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:  # noqa: S310 (URL given by the user)
        shutil.copyfileobj(r, f, length=1 << 20)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--from-dir", help="folder that contains the downloaded model files (searched recursively)")
    src.add_argument(
        "--base-url",
        help=f"URL of the folder holding the files ({DOWNLOAD_URL}); each file is fetched as "
        "<URL>/<file name>?download=1",
    )
    ap.add_argument("--with-accel", action="store_true", help="also the ONNX files of the optional Windows faster mode")
    a = ap.parse_args(argv)

    if a.base_url and a.base_url.startswith("<"):
        sys.exit(f"Give the real download address: {DOWNLOAD_URL}")
    ok = True
    for entry in MODEL_FILES:
        if not entry["required"] and not a.with_accel:
            continue
        dst = os.path.join(ROOT, *entry["path"].split("/"))
        name = os.path.basename(dst)
        if os.path.isfile(dst) and md5_of(dst) == entry["md5"]:
            print(f"[OK]    {entry['path']} is already in place (MD5 {entry['md5']})")
            continue
        with tempfile.TemporaryDirectory() as tmp:
            got = os.path.join(tmp, name)
            try:
                if a.from_dir:
                    found = find_in_dir(a.from_dir, name)
                    if not found:
                        print(f"[MISS]  {name} is not in {a.from_dir}")
                        ok = False
                        continue
                    shutil.copyfile(found, got)
                else:
                    url = file_url(a.base_url, name)
                    print(f"        downloading {name} ({entry['size'] / 1e6:.1f} MB) from {url}", flush=True)
                    fetch(url, got)
            except OSError as e:
                print(f"[FAIL]  {name}: {type(e).__name__}: {e}")
                if "CERTIFICATE_VERIFY_FAILED" in str(e):
                    print(
                        "        On a Mac, run 'Install Certificates.command' in Applications → Python 3.12 once, "
                        f"or download the files from {DOI_URL} and use --from-dir."
                    )
                ok = False
                continue
            m = md5_of(got)
            if m != entry["md5"]:
                print(
                    f"[FAIL]  {name}: MD5 {m}, expected {entry['md5']} - the download is incomplete or a different file"
                )
                ok = False
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                aside = f"{dst}.bad_{time.strftime('%Y%m%d_%H%M%S')}"
                os.rename(dst, aside)
                print(f"        the wrong file that was there is kept as {os.path.basename(aside)}")
            shutil.move(got, dst)
            print(f"[OK]    {entry['path']} placed (MD5 {m})")
    print("\nAll model files are in place." if ok else "\nSome model files are still missing - see the lines above.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
