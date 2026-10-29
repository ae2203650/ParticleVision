"""
Fill hollow microplastic particles in a folder of EXISTING segmentations (no segmentation is run).

Thin command-line front end for app/holefill_folder.py (the same engine as the app page "Fix existing
segmentations"); the post-processing is src/postprocess.postprocess_microplastic (method per particle size from the folder
name: 7 / 10 um circle reconstruction, 5 um and no size closing) with the defaults of src/config.py.

    python tools/fix_segmented_folder.py --input <segmented folder> [--output <new folder>] [--dry-run]
                                         [--continue <existing output folder>] [--skip-list <file>]
                                         [--include-manual-labels]

  * The input folder is only read. The output (default: <Desktop>/<input name>_holefilled_<YYYYMMDD_HHMMSS>)
    mirrors its sub-folders and file names exactly.
  * Manual labels (PackBits TIFFs) and files listed in --skip-list (one relative path per line) are copied
    unchanged unless --include-manual-labels is given.
  * --continue <folder> resumes an interrupted run in that folder (same parameters required).
  * Ctrl+C stops after the current file; the folder can then be continued.
Reports in the output folder: holefill_summary.csv, holefill_log.txt, holefill_params.json, holefill_input_md5.csv.
"""

import argparse
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "app")]

import holefill_folder as hf  # noqa: E402
import platform_utils as pu  # noqa: E402


def main(argv=None):
    pu.make_stdio_safe()
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--input", required=True, help="folder with the segmented files (only read)")
    ap.add_argument(
        "--output",
        help="new output folder (default: <Desktop>/<input name>_holefilled_<timestamp>)",
    )
    ap.add_argument(
        "--continue",
        dest="cont",
        metavar="FOLDER",
        help="continue an interrupted run in this output folder",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="report what would change; write nothing"
    )
    ap.add_argument(
        "--skip-list", help="text file: relative paths to copy unchanged (one per line)"
    )
    ap.add_argument(
        "--include-manual-labels",
        action="store_true",
        help="also fill holes in PackBits (manual-label) files",
    )
    ap.add_argument("--max-files", type=int, help=argparse.SUPPRESS)  # tests: stop after N files
    a = ap.parse_args(argv)

    if not os.path.isdir(a.input):
        sys.exit(f"Input folder not found: {a.input}")
    skip = []
    if a.skip_list:
        with open(a.skip_list, encoding="utf-8") as f:
            skip = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    params = hf.Params(include_manual_labels=a.include_manual_labels, skip_list=skip)
    out = a.cont or a.output or hf.default_output_root(a.input)
    entries = hf.scan_folder(a.input)
    manual = (
        sum(hf.is_manual_label(e.src_path) for e in entries) if not a.include_manual_labels else 0
    )
    print(
        f"Input:  {os.path.abspath(a.input)}\n        {len(entries)} segmented file(s), "
        f"{manual} manual label(s) (PackBits) will be copied unchanged"
        + (f", {len(skip)} on the skip list" if skip else "")
    )
    print(
        f"Output: {os.path.abspath(out)}"
        + ("  (DRY RUN — nothing is written)" if a.dry_run else "")
        + ("  (continuing)" if a.cont else "")
    )
    print(
        f"Settings: method per particle size: {params.method_text()}; circles: support >= "
        f"{params.circle_min_support:.0%}, not drawn if overlapping >= {params.circle_max_overlaps} other circles; "
        f"closing: holes up to {params.max_hole_area} px, added patches up to {params.max_added_region} px"
    )
    done = hf.already_postprocessed(a.input)
    if done:
        print(
            f"Note: {done} result(s) in this folder were already post-processed by ParticleVision 1.1.0 or later "
            f"(summary.csv, postprocess = on) - fixing them again is not needed."
        )
    limit = pu.max_path_length()
    if limit and not a.dry_run and hf.longest_output_path(entries, out) > limit:
        sys.exit(
            f"Some output paths would be longer than {limit} characters on this computer — choose a shorter --output."
        )

    t0 = time.time()
    stop = {"flag": False}

    def on_done(i, n, e, row):
        el = time.time() - t0
        eta = el / i * (n - i)
        info = (
            f"{hf.describe_info(row)}, {row.get('px_changed', 0)} px"
            if row["status"] == "fixed"
            else row.get("error", "")
        )
        print(
            f"[{i}/{n}] {row['status']:24s} {e.rel_path}  {info}   ({el:.0f} s, ~{eta:.0f} s left)",
            flush=True,
        )

    try:
        res = hf.run_folder(
            a.input,
            out,
            params,
            dry_run=a.dry_run,
            resume=bool(a.cont),
            on_done=on_done,
            should_stop=lambda: stop["flag"],
            max_files=a.max_files,
        )
    except KeyboardInterrupt:
        print("\nStopped (Ctrl+C). The finished files are complete; continue with --continue", out)
        return 130
    except (hf.ParamsMismatch, FileExistsError, ValueError) as ex:
        sys.exit(f"Not started — {ex}")
    print(
        "\n"
        + ", ".join(f"{k}: {v}" for k, v in res.counts.items())
        + f"; pixels changed: {res.px_changed}"
        + (f"; {res.already_done} already done before" if res.already_done else "")
    )
    if res.stopped:
        print(f'STOPPED before the end — continue with: --continue "{out}"')
    elif not a.dry_run:
        print(
            "Input folder unchanged: "
            + ("YES" if res.input_unchanged else f"NO — {res.input_changed_files[:5]}")
        )
        print(f"Reports: {os.path.join(out, hf.SUMMARY_NAME)}, {hf.LOG_NAME}, {hf.README_NAME}")
    return 0 if (res.counts["error"] == 0 and res.input_unchanged is not False) else 1


if __name__ == "__main__":
    sys.exit(main())
