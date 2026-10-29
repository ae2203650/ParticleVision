"""
ParticleVision self-test — proves this computer segments exactly like the
verified Mac.

  • prints the system, Python, PyTorch and the device that will be used
  • checks the two model files are the confirmed ones (MD5)
  • quick test (default, used by setup): segments selftest/sample_image.jpg and
    compares the model's map (BEFORE post-processing) with selftest/reference_output.tif (made on the Mac)
  • full check (--full, optional): runs the 3 images of the Demo_Lab folder
    through the real lab pipeline (scan → segment → post-process → write .tif → summary.csv)
    and compares every written raw map (Fused_St_Filt_Comp_Seg_Raw.tif) with selftest/full_check/
    (made on the Mac), and every final result with the post-processing of that raw map
  • post-processing check (both modes): the hole filling (src/postprocess.py) of the raw reference map,
    as a D05 and as a D10 image, must equal selftest/postprocess_check/ exactly

The model checks always compare the RAW maps, so a backend difference is never hidden or created by the
post-processing. PASS = for every image, at least 99.99 % of all pixels identical AND the Fines
pixel count within 0.1 % of the reference ("Fines" = the microplastic class; the app shows it as Microplastic). (CUDA, MPS and CPU may differ on a
handful of near-tie pixels; anything more is a real problem.)

    python check_install.py                 quick test, automatically chosen device
    python check_install.py --full          full check (3 images)
    python check_install.py --device cpu    force the CPU

Quick test → selftest/last_result.json (the app's sidebar shows "Self-test
passed on this machine") and selftest/check_install_log.txt.
Full check → selftest/last_full_result.json and selftest/full_check_log.txt.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
SELFTEST = os.path.join(ROOT, "selftest")
FULL_REF_DIR = os.path.join(SELFTEST, "full_check")
PP_CHECK_DIR = os.path.join(SELFTEST, "postprocess_check")
DEMO_LAB = os.path.join(ROOT, "Demo_Lab")
sys.path[:0] = [ROOT, os.path.join(ROOT, "app")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # segmentation must never need the internet

PASS_OVERALL = 99.99  # % pixels identical
PASS_FINES_COUNT = 0.1  # % max Fines pixel-count difference

_lines = []


def say(msg=""):
    print(msg, flush=True)
    _lines.append(msg)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def machine_fingerprint(device_label):
    """Identifies 'this machine + this environment' so a result copied from
    another computer is never shown as passed here."""
    import torch

    return {
        "host": platform.node(),
        "system": platform.system(),
        "python_executable": sys.executable,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "device": device_label,
    }


def compare(rgb, ref):
    """Agreement numbers between a result and its Mac reference (both H×W×3 uint8)."""
    import numpy as np
    from src.config import CFG

    fines = np.array(CFG.class_colors["Fines"], dtype=np.uint8)
    same = np.all(rgb == ref, axis=-1)
    f_new, f_ref = np.all(rgb == fines, axis=-1), np.all(ref == fines, axis=-1)
    union = f_new | f_ref
    r = {
        "overall_agreement_pct": float(same.mean() * 100),
        "fines_agreement_pct": float((f_new == f_ref)[union].mean() * 100)
        if union.any()
        else 100.0,
        "fines_count_diff_pct": (int(f_new.sum()) - int(f_ref.sum()))
        / max(int(f_ref.sum()), 1)
        * 100,
        "identical": bool(same.all()),
    }
    r["passed"] = (
        r["overall_agreement_pct"] >= PASS_OVERALL
        and abs(r["fines_count_diff_pct"]) <= PASS_FINES_COUNT
    )
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (default: automatic)")
    ap.add_argument(
        "--full",
        action="store_true",
        help="full check: the 3 Demo_Lab images through the lab pipeline",
    )
    a = ap.parse_args()

    import platform_utils as pu

    pu.make_stdio_safe()
    checks = []

    def check(ok, msg):
        checks.append(ok)
        say(("  [PASS] " if ok else "  [FAIL] ") + msg)

    say("=" * 70)
    say("  ParticleVision " + ("FULL CHECK (3 Demo_Lab images)" if a.full else "self-test"))
    say("=" * 70)
    say(f"Date:     {time.strftime('%Y-%m-%d %H:%M:%S')}")
    say(f"Folder:   {ROOT}")
    say(f"System:   {platform.platform()} ({platform.machine()})")
    say(f"Python:   {sys.version.split()[0]}  ({sys.executable})")
    # The model files first: without them nothing else can pass, so stop with a clear message
    from model_files import required_problems_message

    msg = required_problems_message(ROOT)
    if msg:
        say("")
        for line in msg.splitlines():
            say("  " + line if line else "")
        return finish(False, None, {"model_files": "missing or damaged"}, a.full)
    try:
        import torch
        import numpy as np  # noqa: F401
        import scipy.ndimage  # noqa: F401  (src/postprocess.py)
        from PIL import Image  # noqa: F401
        import model_runtime as mr
        from src.config import CFG  # noqa: F401
    except Exception as e:  # noqa: BLE001
        say(f"  [FAIL] a required package is missing: {type(e).__name__}: {e}")
        return finish(False, None, {}, a.full)
    from src.config import PARTICLEVISION_VERSION

    say(f"PyTorch:  {torch.__version__}  (CUDA build: {torch.version.cuda or 'no'})")
    say(f"ParticleVision version: {PARTICLEVISION_VERSION}")

    device = torch.device(a.device) if a.device else mr.select_device()
    say(f"Device:   {mr.describe_device(device)}")
    if device.type == "cpu" and torch.version.cuda:
        say("          (this is a CUDA build of PyTorch but no usable NVIDIA GPU was found —")
        say("           see docs/TROUBLESHOOTING.md → 'NVIDIA card present but Running on: CPU')")
    say("")

    info = json.load(open(os.path.join(SELFTEST, "reference_info.json"), encoding="utf-8"))
    for path, want in (
        (mr.BRANCH1_CKPT, info["checkpoint_md5"]["stage3_unet_corrected_best.pth"]),
        (mr.BRANCH2_CKPT, info["checkpoint_md5"]["fines_detector_corrected_best.pth"]),
    ):
        ok = os.path.isfile(path) and md5(path) == want
        check(ok, f"model file {os.path.basename(path)} is the confirmed version (MD5 {want[:8]}…)")
    try:  # everything the app itself imports
        import streamlit, pandas, matplotlib  # noqa: F401, E401

        matplotlib.use("Agg")
        import matplotlib.pyplot  # noqa: F401
        import lab_structure  # noqa: F401

        check(
            True,
            f"app packages load (streamlit {streamlit.__version__}, pandas {pandas.__version__}, "
            f"matplotlib {matplotlib.__version__})",
        )
    except Exception as e:  # noqa: BLE001
        check(
            False,
            f"an app package cannot be loaded — the app would not start: {type(e).__name__}: {e}",
        )
    try:
        import scipy

        check(
            True,
            f"scipy {scipy.__version__} installed (needed by the post-processing, src/postprocess.py)",
        )
    except Exception as e:  # noqa: BLE001
        check(False, f"scipy is missing — run the setup again: {type(e).__name__}: {e}")
    if pu.os_name() != "Darwin":
        try:
            import tkinter  # noqa: F401

            check(True, "Tcl/Tk available (needed by the folder picker)")
        except Exception:  # noqa: BLE001
            check(False, "Tcl/Tk missing — the folder picker will not work (see INSTALL guide)")
    if not all(checks):
        return finish(False, None, {}, a.full)
    pp_ok, pp_numbers = postprocess_check()
    if not pp_ok:
        return finish(False, None, {"postprocess_check": pp_numbers}, a.full)

    try:
        t0 = time.time()
        runner = mr.SegmentationRunner(device)
        say(f"  [PASS] both models loaded on {runner.label} ({time.time() - t0:.1f} s)")
    except Exception as e:  # noqa: BLE001
        say(f"  [FAIL] models could not be loaded: {type(e).__name__}: {e}")
        return finish(False, None, {}, a.full)

    say("")
    return full_check(runner, pp_numbers) if a.full else quick_test(runner, pp_numbers)


def class_map_of(rgb):
    """H×W×3 class colours → H×W class indices (0-3); 255 where a colour is not a class colour."""
    import numpy as np
    from src.config import CFG

    code = (rgb[..., 0].astype(np.int32) << 16) | (rgb[..., 1].astype(np.int32) << 8) | rgb[..., 2]
    cm = np.full(code.shape, 255, np.uint8)
    for k, (r, g, b) in enumerate(CFG.index_to_color.tolist()):
        cm[code == ((r << 16) | (g << 8) | b)] = k
    return cm


def postprocess_check():
    """The post-processing on its own: the raw reference map processed as a D05 and as a D10 image must equal the
    stored expected outputs exactly (selftest/postprocess_check/, made on the Mac)."""
    import numpy as np
    from PIL import Image
    from src.postprocess import postprocess_microplastic

    say("Post-processing check (fill hollow microplastic particles) ...")
    exp = json.load(open(os.path.join(PP_CHECK_DIR, "expected.json"), encoding="utf-8"))
    raw = class_map_of(
        np.array(Image.open(os.path.join(SELFTEST, "reference_output.tif")).convert("RGB"))
    )
    ok, numbers = True, {}
    for case, c in exp["cases"].items():
        t0 = time.time()
        out, info = postprocess_microplastic(
            raw, particle_size=c["particle_size"], return_info=True
        )
        want = np.array(Image.open(os.path.join(PP_CHECK_DIR, c["expected"])))
        same = out.shape == want.shape and bool(np.array_equal(out.astype(np.uint8), want))
        ok &= same
        numbers[case] = {
            "identical": same,
            "method": info["method"],
            "pixels_changed": info["pixels_changed"],
            "circles_drawn": info["circles_drawn"],
            "seconds": round(time.time() - t0, 2),
        }
        say(
            f"  [{'PASS' if same else 'FAIL'}] {case} image ({info['method']}): {info['pixels_changed']} px filled"
            + (f", {info['circles_drawn']} circles" if info["method"] == "circles" else "")
            + f" — {'identical to' if same else 'DIFFERENT from'} the Mac result"
        )
    return ok, numbers


def quick_test(runner, pp_numbers):
    import numpy as np
    from PIL import Image

    say("Segmenting the sample image (about 10 s on a GPU, 1–3 minutes on a CPU) ...")
    rgb, cm, secs, note = runner.predict(
        os.path.join(SELFTEST, "sample_image.jpg")
    )  # the model's map, unfilled
    ref = np.array(Image.open(os.path.join(SELFTEST, "reference_output.tif")).convert("RGB"))
    if rgb.shape != ref.shape:
        say(f"  [FAIL] output size {rgb.shape} differs from the reference {ref.shape}")
        return finish(False, runner, {}, False)
    r = compare(rgb, ref)
    say(
        f"  pixel agreement, all pixels        : {r['overall_agreement_pct']:.4f} %   (need ≥ {PASS_OVERALL} %)"
    )
    say(f"  pixel agreement, microplastic      : {r['fines_agreement_pct']:.4f} %")
    say(
        f"  microplastic pixel count difference: {r['fines_count_diff_pct']:+.4f} %   (need within ±{PASS_FINES_COUNT} %)"
    )
    say(f"  identical to the Mac reference: {'yes, every pixel' if r['identical'] else 'no'}")
    say(f"  time for this image: {secs:.1f} s" + (f"   [{note}]" if note else ""))
    ok = r.pop("passed")
    return finish(
        ok,
        runner,
        {**r, "seconds_per_image": secs, "note": note, "postprocess_check": pp_numbers},
        False,
    )


def full_check(runner, pp_numbers):
    import numpy as np
    from PIL import Image
    import lab_structure as ls
    from src.postprocess import PipelinePostprocess

    refs = json.load(open(os.path.join(FULL_REF_DIR, "references.json"), encoding="utf-8"))[
        "images"
    ]
    if not os.path.isdir(DEMO_LAB):
        say(
            f"  [FAIL] the Demo_Lab folder is missing ({DEMO_LAB}) — copy it again from the package"
        )
        return finish(False, runner, {}, True)
    scan = ls.scan_lab_folder(DEMO_LAB)
    say(f"Scanned Demo_Lab: {len(scan.images)} images found (expected {len(refs)})")
    if len(scan.images) != len(refs):
        say(
            "  [FAIL] Demo_Lab does not contain the expected 3 images — copy it again from the package"
        )
        return finish(False, runner, {}, True)

    say(
        f"Segmenting {len(refs)} images through the lab pipeline "
        "(about 30 s on a GPU, 3–9 minutes on a CPU) ..."
    )
    tmp = tempfile.mkdtemp(prefix="pv_fullcheck_")
    out_root = os.path.join(tmp, "Demo_Lab_segmented")
    try:

        def on_done(i, img, rec):
            extra = f"  [{rec['note']}]" if rec.get("note") else ""
            say(
                f"  {i}/{len(scan.images)}  {img.label}: {rec['status']}  ({rec['seconds']:.1f} s){extra}"
                + (f"  — {rec['error']}" if rec["status"] != "ok" else "")
            )

        # the real pipeline with the hole filling ON (Auto size: Demo_Lab is a D07 experiment → circles), plus the
        # raw copy - the raw maps are what the Mac references hold
        pp = PipelinePostprocess(enabled=True, particle_size=None, save_raw=True)
        res = ls.run_lab_batch(
            scan,
            out_root,
            runner.predict,
            on_done=on_done,
            between_images=runner.between_images,
            postprocess=pp,
            backend=runner.label,
        )
        rows = ls.read_summary(out_root)
        say("")
        per_image, all_ok = {}, res.failed == 0 and res.done == len(refs)
        src_of = {
            os.path.relpath(ls.output_path_for(out_root, im), out_root).replace(
                os.sep, "/"
            ): im.src_path
            for im in scan.images
        }
        for rel, ref_info in refs.items():
            final = os.path.join(out_root, *rel.split("/"))
            out = os.path.join(
                os.path.dirname(final), ls.RAW_OUTPUT_FILENAME
            )  # the model's map, before filling
            name = rel.split("/", 2)[2]  # e.g. "IN/PV 03/Fused_St_Filt_Comp_Seg_Mod.tif"
            if not (os.path.isfile(out) and os.path.isfile(final)):
                say(f"  [FAIL] {name}: no output was written")
                all_ok = False
                continue
            rgb = np.array(Image.open(out).convert("RGB"))
            ref = np.array(
                Image.open(os.path.join(FULL_REF_DIR, ref_info["reference"])).convert("RGB")
            )
            if rgb.shape != ref.shape:
                say(f"  [FAIL] {name}: size {rgb.shape} differs from the reference {ref.shape}")
                all_ok = False
                continue
            r = compare(rgb, ref)
            all_ok &= r["passed"]
            per_image[name] = r
            say(
                f"  [{'PASS' if r['passed'] else 'FAIL'}] {name:40s} raw map: agreement {r['overall_agreement_pct']:.4f} %  "
                f"microplastic count {r['fines_count_diff_pct']:+.4f} %  "
                f"{'identical' if r['identical'] else 'not identical'}"
            )
            filled, _ = pp.apply(class_map_of(rgb).astype(np.int64), src_of[rel])
            fin_same = bool(
                np.array_equal(class_map_of(np.array(Image.open(final).convert("RGB"))), filled)
            )
            r["final_equals_postprocessed_raw"] = fin_same
            all_ok &= fin_same
            say(
                f"  [{'PASS' if fin_same else 'FAIL'}] {name:40s} final result = the post-processing of its raw map"
            )
        written = sorted(
            os.path.relpath(os.path.join(d, f), out_root).replace(os.sep, "/")
            for d, _, fs in os.walk(out_root)
            for f in fs
            if f.lower().endswith((".tif", ".tiff"))
        )
        want = sorted(
            list(refs) + [r.rsplit("/", 1)[0] + "/" + ls.RAW_OUTPUT_FILENAME for r in refs]
        )
        layout_ok = written == want
        say(
            f"  [{'PASS' if layout_ok else 'FAIL'}] results saved as Before/{ls.OUTPUT_FILENAME} and "
            f"IN|WT/<PV folder>/{ls.OUTPUT_FILENAME} (+ the raw copy {ls.RAW_OUTPUT_FILENAME}) — "
            f"{len(written)} file(s), no other names"
        )
        all_ok &= layout_ok
        ok_rows = sum(1 for row in rows if row.get("status") == "ok")
        rows_ok = ok_rows == len(refs)
        say(
            f"  [{'PASS' if rows_ok else 'FAIL'}] summary.csv has {ok_rows} of {len(refs)} rows marked ok"
        )
        from src.config import PARTICLEVISION_VERSION

        rec_ok = all(
            row.get("app_version") == PARTICLEVISION_VERSION
            and row.get("postprocess") == "on"
            and row.get("pp_method") == "circles"
            and row.get("particle_size_detected") == "7"
            and row.get("backend")
            for row in rows
        )
        say(
            f"  [{'PASS' if rec_ok else 'FAIL'}] summary.csv records version {PARTICLEVISION_VERSION}, the backend "
            f"and the post-processing (D07 → circles) for every image"
        )
        all_ok &= rec_ok
        all_ok &= rows_ok
        secs = res.seconds
        say(
            f"  time: {sum(secs):.1f} s total, {sum(secs) / max(len(secs), 1):.1f} s per image "
            f"(of which post-processing {sum(res.pp_seconds) / max(len(res.pp_seconds), 1):.2f} s)"
        )
    finally:
        shutil.rmtree(tmp, ignore_errors=True)  # our own temporary output folder only
    return finish(
        all_ok,
        runner,
        {
            "images": per_image,
            "identical_images": sum(r["identical"] for r in per_image.values()),
            "seconds_per_image": sum(secs) / max(len(secs), 1),
            "postprocess_seconds_per_image": sum(res.pp_seconds) / max(len(res.pp_seconds), 1),
            "postprocess_check": pp_numbers,
        },
        True,
    )


def finish(ok, runner, numbers, full):
    say("")
    say("=" * 70)
    say(("  FULL CHECK " if full else "  SELF-TEST ") + ("PASSED" if ok else "FAILED"))
    say("=" * 70)
    log_name = "full_check_log.txt" if full else "check_install_log.txt"
    if not ok:
        say(f"  Send selftest/{log_name} to the person who gave you ParticleVision.")
    os.makedirs(SELFTEST, exist_ok=True)
    result = {"passed": bool(ok), "time": time.strftime("%Y-%m-%d %H:%M:%S"), **numbers}
    if runner is not None:
        result["machine"] = machine_fingerprint(runner.label)
    with open(
        os.path.join(SELFTEST, "last_full_result.json" if full else "last_result.json"),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(result, f, indent=2)
    with open(os.path.join(SELFTEST, log_name), "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
