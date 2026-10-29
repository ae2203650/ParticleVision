"""
Faster-mode benchmark: which way of running ParticleVision is fastest on THIS computer
while still giving the same results as the verified Mac?

Runs the self-test image and the 3 Demo_Lab images through
  (a) the normal path (PyTorch, exactly as the app runs today), and
  (b) every available faster backend:
        dml:N    ONNX Runtime + DirectML on graphics adapter N (AMD / Intel / NVIDIA)
        ort-cpu  ONNX Runtime on the processor
compares every result with the Mac reference (same rule as the self-test:
≥ 99.99 % of pixels identical AND Fines pixel count within ±0.1 %, for EVERY image),
and writes accel_result.json.

The app uses a faster backend only if, in THIS run, it PASSED the comparison AND was
faster than the normal path measured on the same images; otherwise it keeps the normal
path, and this benchmark prints why. All speeds are printed side by side.

    .pv_env\\Scripts\\python.exe tools\\benchmark_accel.py            (benchmark_accel.bat)
    ... --backends dml:1,ort-cpu       only these (the normal path is always measured too)
    ... --quick                        only the self-test image (for a quick look; never enables)
"""

import argparse
import json
import os
import platform
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "app")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")

RESULT = os.environ.get("PV_ACCEL_RESULT") or os.path.join(ROOT, "accel_result.json")
LOG = os.path.join(os.path.dirname(RESULT), "accel_benchmark_log.txt")

_lines = []


def say(msg=""):
    print(msg, flush=True)
    _lines.append(msg)


def test_images():
    """[(label, input image, Mac reference)] — the self-test image + the 3 Demo_Lab images."""
    items = [
        (
            "self-test image",
            os.path.join(ROOT, "selftest", "sample_image.jpg"),
            os.path.join(ROOT, "selftest", "reference_output.tif"),
        )
    ]
    refs = json.load(
        open(os.path.join(ROOT, "selftest", "full_check", "references.json"), encoding="utf-8")
    )
    for rel, info in refs["images"].items():
        folder = os.path.join(ROOT, "Demo_Lab", *os.path.dirname(rel).split("/"))
        src = os.path.join(folder, "Fused_St_Filt_Comp.jpg")
        label = "Demo_Lab " + "/".join(os.path.dirname(rel).split("/")[2:])
        items.append((label, src, os.path.join(ROOT, "selftest", "full_check", info["reference"])))
    return items


def choose_backend(backends):
    """(chosen backend or None, reason, {backend: verdict}).

    Chosen = the fastest backend that PASSED the comparison with the Mac AND was faster
    than the normal path measured in the same run. No normal-path measurement → nothing."""
    normal = backends.get("normal") or {}
    n = normal.get("sec_per_image") if normal.get("passed") else None
    verdicts, ok = {}, []
    for name, b in backends.items():
        if name == "normal":
            continue
        if b.get("error"):
            verdicts[name] = f"not used — error: {b['error'][:80]}"
        elif not b.get("passed"):
            verdicts[name] = "not used — results differ from the Mac (accuracy check failed)"
        elif not n:
            verdicts[name] = "not used — the normal path was not measured in this run"
        elif b["sec_per_image"] >= n:
            verdicts[name] = (
                f"not used — slower than the normal path ({b['sec_per_image']:.1f} vs {n:.1f} s/image)"
            )
        else:
            verdicts[name] = f"passes, {n / b['sec_per_image']:.1f}× faster than the normal path"
            ok.append((b["sec_per_image"], name))
    if not ok:
        if not n:
            return None, "the normal path was not measured successfully in this run", verdicts
        return (
            None,
            "no faster backend both passed the accuracy check and beat the normal path",
            verdicts,
        )
    sec, name = min(ok)
    verdicts[name] += "  ← USED"
    return name, "", verdicts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backends", default=None, help="comma list, e.g. dml:1,ort-cpu (normal is always added)"
    )
    ap.add_argument("--quick", action="store_true", help="only the self-test image")
    a = ap.parse_args()

    import platform_utils as pu

    pu.make_stdio_safe()
    import numpy as np
    import torch
    from PIL import Image
    import accel_runtime as ar
    import model_runtime as mr
    from check_install import compare, PASS_OVERALL, PASS_FINES_COUNT

    # ── header: everything needed in ONE screenshot ──────────────────────
    gpus, adapters, rv = ar.windows_gpus(), ar.dxgi_adapters(), ar.runtime_versions()
    say("=" * 78)
    say("  ParticleVision — faster-mode benchmark")
    say("=" * 78)
    say(f"Date:       {time.strftime('%Y-%m-%d %H:%M:%S')}      Computer: {platform.node()}")
    say(f"Processor:  {ar.cpu_model()}")
    say(f"System:     {ar.os_version()}")
    if gpus:
        for g in gpus:
            say(f"Graphics:   {g['name']}  — driver {g['driver_version']} ({g['driver_date']})")
    else:
        say("Graphics:   (not listed on this system)")
    for d in adapters:
        say(
            f"  DirectML adapter {d['index']}: {d['name']}  ({d['dedicated_mb']} MB)"
            + ("  [software — skipped]" if d["software"] else "")
        )
    say(
        f"Python {sys.version.split()[0]} · PyTorch {torch.__version__} · ONNX Runtime {rv['onnxruntime']} "
        f"{rv.get('onnxruntime_providers', '')}"
    )
    ok, why = ar.onnx_files_ok()
    say(f"ONNX model files: {'OK (MD5 as exported)' if ok else 'PROBLEM — ' + why}")
    say("")

    images = test_images()[:1] if a.quick else test_images()
    for _, src, ref in images:
        if not (os.path.isfile(src) and os.path.isfile(ref)):
            say(f"Missing test file: {src if not os.path.isfile(src) else ref}")
            return 1

    if a.backends:
        candidates = ["normal"] + [
            b.strip() for b in a.backends.split(",") if b.strip() and b.strip() != "normal"
        ]
    else:
        candidates = ["normal"]
        if rv["onnxruntime"] and "DmlExecutionProvider" in (rv.get("onnxruntime_providers") or []):
            candidates += [f"dml:{d['index']}" for d in adapters if not d["software"]]
        if rv["onnxruntime"]:
            candidates.append("ort-cpu")
    if not ok:
        candidates = [c for c in candidates if c == "normal"]

    say(f"Testing {len(images)} image(s) with: {', '.join(candidates)}")
    say(
        f"Pass rule (every image): ≥ {PASS_OVERALL} % of pixels identical to the Mac and microplastic count within "
        f"±{PASS_FINES_COUNT} %"
    )
    say("")

    results = {}
    for backend in candidates:
        entry = {"passed": False, "images": []}
        results[backend] = entry
        try:
            t0 = time.time()
            if backend == "normal":
                runner = mr.SegmentationRunner()
                run = lambda p: runner.predict(p)[:2]  # noqa: E731
                entry["label"] = f"{runner.label} (normal path)"
            else:
                runner = ar.AccelRunner(backend, label=ar.backend_label(backend, adapters))
                run = lambda p: runner.predict_fast(p)[:2]  # noqa: E731
                entry["label"] = runner.label
                w = torch.zeros(1, 3, 512, 512)  # warm-up (DirectML compiles once)
                runner.m1(w), runner.m2(w)
            entry["load_seconds"] = round(time.time() - t0, 1)
            say(f"── {backend}: {entry['label']}  (ready in {entry['load_seconds']} s)")
            for label, src, ref in images:
                t = time.time()
                rgb, _ = run(src)
                secs = time.time() - t
                refimg = np.array(Image.open(ref).convert("RGB"))
                if rgb.shape == refimg.shape:
                    r = compare(rgb, refimg)
                else:  # wrong size = fail
                    r = {
                        "passed": False,
                        "overall_agreement_pct": 0.0,
                        "fines_agreement_pct": 0.0,
                        "fines_count_diff_pct": 0.0,
                        "identical": False,
                    }
                r["seconds"] = round(secs, 2)
                r["image"] = label
                entry["images"].append(r)
                say(
                    f"   [{'PASS' if r['passed'] else 'FAIL'}] {label:26s} {secs:6.1f} s   agreement "
                    f"{r['overall_agreement_pct']:.4f} %   microplastic {r['fines_count_diff_pct']:+.4f} %"
                    + ("   identical" if r["identical"] else "")
                )
            entry["passed"] = all(r["passed"] for r in entry["images"])
            entry["sec_per_image"] = round(
                sum(r["seconds"] for r in entry["images"]) / len(entry["images"]), 2
            )
        except Exception as e:  # noqa: BLE001
            entry["error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:200] if str(e) else ''}"
            say(f"   [FAIL] {backend}: {entry['error']}")
        say("")

    chosen, why, verdicts = choose_backend(results)
    for name, v in verdicts.items():
        results[name]["verdict"] = v
    say("=" * 78)
    say(f"  SPEED ON THIS COMPUTER (same run, same {len(images)} image(s))")
    say(f"  {'':40s} {'s/image':>8s}  {'accuracy':8s}  decision")
    for name, b in results.items():
        spi = f"{b['sec_per_image']:.1f}" if b.get("sec_per_image") else "—"
        label = (
            "Normal path — " + b.get("label", "PyTorch").replace(" (normal path)", "") + ", PyTorch"
            if name == "normal"
            else b.get("label", name)
        )
        verdict = "the reference speed" if name == "normal" else verdicts.get(name, "")
        say(f"  {label[:40]:40s} {spi:>8s}  {'PASS' if b['passed'] else 'FAIL':8s}  {verdict}")
    say("=" * 78)
    if chosen:
        say(f"  The app will use: {results[chosen]['label']}  ({verdicts[chosen].split('  ←')[0]})")
    else:
        say(f"  The app will use: the normal path — {why}")
    say("=" * 78)

    out = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "chosen": chosen,
        "reason": why,
        "quick": a.quick,
        "backends": results,
        "machine": ar.fingerprint(chosen or "ort-cpu", gpus, adapters),
        "hardware": {
            "cpu": ar.cpu_model(),
            "os": ar.os_version(),
            "gpus": gpus,
            "directml_adapters": adapters,
        },
    }
    if a.quick and chosen:
        out["chosen"], out["reason"] = (
            None,
            "quick check only (1 image) — run the full benchmark to enable",
        )
        say("  (quick check: accel_result.json does not enable the faster mode)")
    with open(RESULT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(LOG, "w", encoding="utf-8") as f:
        f.write("\n".join(_lines) + "\n")
    say(f"  Saved: {RESULT}")
    say(f"         {LOG}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
