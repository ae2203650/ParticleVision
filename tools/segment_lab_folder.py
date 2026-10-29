"""
Segment a lab folder from the command line - the same pipeline as the app's Run page.

Thin front end for app/lab_structure.run_lab_batch (scan → segment → post-process → write .tif →
summary.csv / run_log.txt / run_settings.json), with the same model runner the app uses (the faster mode only if
benchmark_accel verified it on this computer, otherwise the normal path: CUDA → Apple GPU → CPU).

    python tools/segment_lab_folder.py --input <lab folder> [--output <new folder>]
                                       [--no-postprocess] [--particle-size auto|5|7|10] [--save-raw]
                                       [--continue <output folder of an interrupted run>] [--normal]

  * The lab folder is only read. The output (default: <lab folder>_segmented_<YYYYMMDD_HHMMSS> next to it) mirrors
    its G / experiment / Before | IN | WT / PV folders; every result is Fused_St_Filt_Comp_Seg_Mod.tif.
  * Hollow microplastic particles are filled automatically (src/postprocess.py; method per particle size, read from
    each experiment folder name, or --particle-size for the whole run). --no-postprocess saves the model's map as it is;
    --save-raw also saves the unfilled map as Fused_St_Filt_Comp_Seg_Raw.tif.
  * --continue <folder> resumes an interrupted run - only with the same settings (run_settings.json).
  * Ctrl+C stops after the current image; the run can then be continued.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "app")]
os.environ.setdefault("HF_HUB_OFFLINE", "1")  # segmentation must never need the internet

import lab_structure as ls  # noqa: E402
import platform_utils as pu  # noqa: E402

from src.config import PARTICLEVISION_VERSION  # noqa: E402
from src.postprocess import PipelinePostprocess, describe_info  # noqa: E402


def main(argv=None):
    pu.make_stdio_safe()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument(
        "--input", required=True, help="the lab folder (the one containing G1, G2, ...); only read"
    )
    ap.add_argument(
        "--output",
        help="new output folder (default: <lab folder>_segmented_<timestamp> next to it)",
    )
    ap.add_argument(
        "--continue",
        dest="cont",
        metavar="FOLDER",
        help="continue an interrupted run in this output folder",
    )
    ap.add_argument(
        "--no-postprocess", action="store_true", help="do not fill hollow microplastic particles"
    )
    ap.add_argument(
        "--particle-size",
        default="auto",
        choices=["auto", "5", "7", "10"],
        help="auto = from each experiment folder name (default); 5 / 7 / 10 um for the whole run",
    )
    ap.add_argument(
        "--save-raw",
        action="store_true",
        help="also save the unfilled map (Fused_St_Filt_Comp_Seg_Raw.tif)",
    )
    ap.add_argument(
        "--normal", action="store_true", help="always the normal path (never the faster mode)"
    )
    ap.add_argument("--max-images", type=int, help=argparse.SUPPRESS)  # tests: stop after N images
    a = ap.parse_args(argv)

    if not os.path.isdir(a.input):
        sys.exit(f"Lab folder not found: {a.input}")
    pp = PipelinePostprocess(
        enabled=not a.no_postprocess,
        particle_size=None if a.particle_size == "auto" else int(a.particle_size),
        save_raw=a.save_raw,
    )
    scan = ls.scan_lab_folder(os.path.abspath(a.input))
    plan = None
    if a.cont:
        try:
            plan = ls.plan_resume(scan, a.cont)
            ls.check_resume_settings(a.cont, pp)
        except (ls.OldLayoutError, ls.SettingsMismatchError) as ex:
            sys.exit(f"Not started — {ex}")
    out = a.cont or a.output or ls.make_output_root(a.input)
    print(f"ParticleVision {PARTICLEVISION_VERSION}")
    print(
        f"Lab folder: {os.path.abspath(a.input)}  ({len(scan.images)} image(s), "
        f"{len(scan.skipped_folders)} folder(s) skipped — see run_log.txt)"
    )
    if plan is not None:
        print(f"Continuing: {len(plan.done)} already done, {len(plan.todo)} to process")
    print(f"Output:     {os.path.abspath(out)}")
    print(f"Post-processing: {pp.describe()}")
    problems = ls.path_length_problems(scan, out, pu.max_path_length())
    if problems:
        sys.exit("Not started — " + " ".join(problems))

    import accel_runtime as ar  # loads PyTorch - after the cheap checks
    import model_runtime as mr

    runner, note = ar.select_runner(force_normal=a.normal)
    if runner is None:
        runner = mr.SegmentationRunner()
    print(f"Backend:    {runner.label}" + (f"  ({note})" if note else ""))

    t0 = time.time()
    total = len(plan.todo) if plan is not None else len(scan.images)

    def on_done(i, img, rec):
        if rec["status"] == "ok":
            info = describe_info(rec["postprocess"]) + (
                f"  [{rec['note']}]" if rec.get("note") else ""
            )
        else:
            info = rec.get("error", "")
        el = time.time() - t0
        print(
            f"[{i}/{total}] {rec['status']:6s} {img.label}  {info}   ({rec['seconds']:.1f} s, "
            f"~{el / i * (total - i):.0f} s left)",
            flush=True,
        )
        if a.max_images and i >= a.max_images:
            raise KeyboardInterrupt

    try:
        with pu.KeepAwake():
            res = ls.run_lab_batch(
                scan,
                out,
                runner.predict,
                on_done=on_done,
                between_images=runner.between_images,
                resume_plan=plan,
                postprocess=pp,
                backend=runner.label,
            )
    except KeyboardInterrupt:
        print(f'\nStopped. The finished images are complete; continue with: --continue "{out}"')
        return 130
    except (ls.SettingsMismatchError, ls.OutputExistsError, ValueError) as ex:
        sys.exit(f"Not started — {ex}")
    pps = res.pp_seconds
    print(
        f"\n{res.done} of {res.total} processed, {res.failed} failed"
        + (f", {res.skipped} already done before" if plan is not None else "")
        + (f"; post-processing {sum(pps) / len(pps):.2f} s per image" if pps else "")
    )
    print(f"Reports: {os.path.join(out, 'summary.csv')}, run_log.txt, {ls.RUN_SETTINGS_NAME}")
    return 0 if res.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
