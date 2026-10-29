"""
Fill hollow microplastic particles in a folder of EXISTING segmentations (no segmentation is run).

The one engine behind both front ends:
  * tools/fix_segmented_folder.py   (command line)
  * the app page "Fix existing segmentations" (app/segmenter_app.py)
The post-processing itself is src/postprocess.postprocess_microplastic, chosen per particle size (never
re-implemented here): 7 / 10 um -> circle reconstruction; 5 um and no size found -> closing of the microplastic class.

Rules
  * The input folder is only READ. Results go to a NEW folder (default: on the Desktop) that mirrors the input's
    sub-folders and file names exactly (G# / experiment / Before | IN | WT / PV ## / Fused_St_Filt_Comp_Seg_Mod.tif),
    so the folders can be swapped. README_HOLEFILL.txt in the output explains what was changed.
  * Manual labels (PackBits-compressed TIFFs; the app writes uncompressed ones) and files on an explicit skip list
    are COPIED unchanged unless include_manual_labels is set. Files with any colour other than the 4 classes are
    copied unchanged too. Files without holes are copied unchanged (byte-identical).
  * Fixed files are written with lab_structure.save_segmentation_tiff (the app's own writer), atomically
    (temporary file + rename), then read back and checked: same size, = the computed map exactly, no
    microplastic pixel removed.
  * Resume: holefill_summary.csv records the MD5 of every output; a file counts as done only if its output exists
    and still has that MD5. The parameters of a resumed folder must be identical (holefill_params.json).
  * The MD5 of every input file is recorded when the file is read and checked again at the end.
"""

import csv
import hashlib
import io
import json
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, List, Optional

import numpy as np
from PIL import Image
from scipy.ndimage import label

import lab_structure as ls
from src.config import CFG
from src.postprocess import describe_info, postprocess_microplastic, postprocess_settings

TARGET_NAME = ls.OUTPUT_FILENAME  # Fused_St_Filt_Comp_Seg_Mod.tif
OLD_SUFFIX = "_Segmented.tif"  # older app versions
SUMMARY_NAME, LOG_NAME, PARAMS_NAME, INPUT_MD5_NAME, README_NAME = (
    "holefill_summary.csv",
    "holefill_log.txt",
    "holefill_params.json",
    "holefill_input_md5.csv",
    "README_HOLEFILL.txt",
)
STATUSES = ("fixed", "unchanged", "skipped_manual_label", "skipped_unknown_colours", "error")
EXTRA_COLUMNS = [
    "rel_path",
    "particle_size_detected",
    "method",
    "radius_used",
    "circles_drawn",
    "circles_skipped_overlap",
    "closing_px_added",
    "hole_px_added",
    "added_patches_dropped",
    "px_changed",
    "grain_to_mp",
    "air_to_mp",
    "water_to_mp",
    "mp_px_before",
    "mp_px_after",
    "mp_objects_3px_before",
    "mp_objects_3px_after",
    "input_md5",
    "output_md5",
]
COLUMNS = ls.SUMMARY_COLUMNS + EXTRA_COLUMNS  # the app's summary.csv columns first, then the extras
_COLOUR_CODES = [(int(r) << 16) | (int(g) << 8) | int(b) for r, g, b in CFG.index_to_color.tolist()]


@dataclass
class Entry:
    rel_path: str
    src_path: str
    g: str = ""
    experiment: str = ""
    phase: str = ""
    pv: str = ""


@dataclass
class Params:
    radius_by_size: dict = field(
        default_factory=lambda: dict(CFG.postprocess_closing_radius_by_size)
    )
    default_radius: int = CFG.postprocess_closing_default_radius
    use_size: bool = CFG.postprocess_closing_use_size
    max_hole_area: int = CFG.postprocess_closing_max_hole_area
    max_added_region: int = CFG.postprocess_closing_max_added_region
    method_by_size: dict = field(default_factory=lambda: dict(CFG.postprocess_method_by_size))
    circle_radii_by_size: dict = field(
        default_factory=lambda: {
            k: list(v) for k, v in CFG.postprocess_circle_radii_by_size.items()
        }
    )
    circle_min_support: float = CFG.postprocess_circle_min_support
    circle_max_overlaps: int = CFG.postprocess_circle_max_overlaps
    include_manual_labels: bool = False
    skip_list: List[str] = field(default_factory=list)  # relative paths (/ or \) copied unchanged

    def pp_kwargs(self) -> dict:
        """The arguments of src.postprocess.postprocess_microplastic for these parameters (+ source_path per file)."""
        return dict(
            radius_by_size=self.radius_by_size,
            default_radius=self.default_radius,
            use_size=self.use_size,
            max_hole_area=self.max_hole_area,
            max_added_region=self.max_added_region,
            method_by_size=self.method_by_size,
            circle_radii_by_size=self.circle_radii_by_size,
            circle_min_support=self.circle_min_support,
            circle_max_overlaps=self.circle_max_overlaps,
        )

    def method_text(self) -> str:
        """e.g. 'D05 closing r 1; D07 circles r 2.0/2.3/2.6 px; D10 circles r 3.0/3.3/3.6 px; no size found: closing r 2'."""
        if not self.use_size:
            return f"closing r {self.default_radius} for all sizes"
        parts = []
        for k in sorted({int(k) for k in self.method_by_size}):
            m = self.method_by_size.get(k, self.method_by_size.get(str(k)))
            if m == "circles":
                radii = self.circle_radii_by_size.get(k, self.circle_radii_by_size.get(str(k)))
                parts.append(f"D{k:02d} circles r {'/'.join(f'{float(x):.1f}' for x in radii)} px")
            else:
                parts.append(
                    f"D{k:02d} closing r {self.radius_by_size.get(k, self.radius_by_size.get(str(k), self.default_radius))}"
                )
        return "; ".join(parts) + f"; no size found: closing r {self.default_radius}"

    def as_dict(self) -> dict:
        d = postprocess_settings()
        d.update(
            radius_by_size={
                str(int(k)): int(v)
                for k, v in sorted(self.radius_by_size.items(), key=lambda kv: int(kv[0]))
            },
            default_radius=int(self.default_radius),
            use_size=bool(self.use_size),
            max_hole_area=int(self.max_hole_area),
            max_added_region=int(self.max_added_region),
            method_by_size={
                str(int(k)): v
                for k, v in sorted(self.method_by_size.items(), key=lambda kv: int(kv[0]))
            },
            circle_radii_by_size={
                str(int(k)): [float(x) for x in v]
                for k, v in sorted(self.circle_radii_by_size.items(), key=lambda kv: int(kv[0]))
            },
            circle_min_support=float(self.circle_min_support),
            circle_max_overlaps=int(self.circle_max_overlaps),
            include_manual_labels=self.include_manual_labels,
            skip_list=sorted(_norm(p) for p in self.skip_list),
            holefill_folder_py_md5=_md5_file(os.path.abspath(__file__)),
        )
        return d


@dataclass
class FolderResult:
    output_root: str
    total: int = 0
    already_done: int = 0
    counts: dict = field(default_factory=lambda: {s: 0 for s in STATUSES})
    px_changed: int = 0
    seconds: float = 0.0
    input_unchanged: Optional[bool] = None
    input_changed_files: List[str] = field(default_factory=list)
    stopped: bool = False
    records: List[dict] = field(default_factory=list)


def _norm(rel: str) -> str:
    return rel.replace("\\", "/").strip("/")


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _is_target(name: str) -> bool:
    return name == TARGET_NAME or (name.endswith(OLD_SUFFIX) and not name.startswith("."))


def scan_folder(input_root: str) -> List[Entry]:
    """Every Fused_St_Filt_Comp_Seg_Mod.tif (or older *_Segmented.tif) below input_root, in natural order."""
    input_root = os.path.abspath(input_root)
    out = []
    for dirpath, dirnames, filenames in os.walk(input_root):
        dirnames[:] = sorted((d for d in dirnames if not d.startswith(".")), key=ls.natural_key)
        for name in sorted(filenames, key=ls.natural_key):
            if _is_target(name):
                src = os.path.join(dirpath, name)
                rel = _norm(os.path.relpath(src, input_root))
                parts = rel.split("/")
                e = Entry(rel_path=rel, src_path=src)
                if len(parts) >= 3 and re.fullmatch(r"G\d+", parts[0]):
                    e.g, e.experiment = parts[0], parts[1]
                    if parts[2] in ("Before", "IN", "WT"):
                        e.phase = parts[2]
                        if e.phase != "Before":
                            e.pv = (
                                parts[3]
                                if len(parts) >= 5
                                else name[: -len(OLD_SUFFIX)]
                                if name.endswith(OLD_SUFFIX)
                                else ""
                            )
                out.append(e)
    return out


def already_postprocessed(input_root: str) -> int:
    """Rows of the input's summary.csv (a ParticleVision 1.1.0+ output folder) whose results were already
    post-processed during segmentation (column postprocess = on). 0 for older folders / no summary.csv."""
    path = os.path.join(input_root or "", "summary.csv")
    if not os.path.isfile(path):
        return 0
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return sum(
                1
                for r in csv.DictReader(f)
                if r.get("postprocess") == "on" and r.get("status") == "ok"
            )
    except (OSError, csv.Error, UnicodeDecodeError):
        return 0


def is_manual_label(path: str) -> bool:
    """The lab's manual labels are PackBits-compressed TIFFs; the app writes uncompressed ones."""
    try:
        with Image.open(path) as im:
            return im.info.get("compression") == "packbits"
    except Exception:  # noqa: BLE001
        return False


def default_output_root(input_root: str, now: Optional[datetime] = None) -> str:
    """<Desktop>/<input name>_holefilled_<YYYYMMDD_HHMMSS> (the home folder if there is no Desktop).
    Not next to the input: the lab's data folder is left untouched."""
    input_root = os.path.abspath(input_root)
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    base = desktop if os.path.isdir(desktop) else os.path.expanduser("~")
    return os.path.join(base, f"{os.path.basename(input_root)}_holefilled_{ts}")


def decode(data: bytes):
    """RGB TIFF bytes -> (class map uint8 or None, number of pixels with an unknown colour)."""
    a = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    code = (a[..., 0].astype(np.uint32) << 16) | (a[..., 1].astype(np.uint32) << 8) | a[..., 2]
    cm = np.full(code.shape, 255, np.uint8)
    for k, c in enumerate(_COLOUR_CODES):
        cm[code == c] = k
    unknown = int((cm == 255).sum())
    return (cm if unknown == 0 else None), unknown


def _objects_3px(mp: np.ndarray) -> int:
    lab, n = label(mp)
    return int((np.bincount(lab.ravel(), minlength=n + 1)[1:] >= 3).sum())


def _area_columns(cm: Optional[np.ndarray]) -> dict:
    row = {}
    counts = np.bincount(cm.ravel(), minlength=4) if cm is not None else None
    for k, (_, disp) in enumerate(ls.CLASS_DISPLAY):
        if counts is None:
            row[f"{disp.lower()}_px"] = row[f"{disp.lower()}_pct"] = ""
        else:
            row[f"{disp.lower()}_px"] = int(counts[k])
            row[f"{disp.lower()}_pct"] = f"{counts[k] / cm.size * 100:.4f}"
    return row


def _copy_unchanged(src: str, dst: str):
    tmp = dst + ls.PARTIAL_SUFFIX
    shutil.copyfile(src, tmp)  # same bytes; normal (writable) permissions, not the input's
    st = os.stat(src)
    os.utime(tmp, (st.st_atime, st.st_mtime))  # keep the file date (manual labels: 2025-09-04)
    os.replace(tmp, dst)


def process_one(
    e: Entry,
    output_root: Optional[str],
    params: Params,
    dry_run: bool = False,
    skip_set: Optional[set] = None,
) -> dict:
    """Process one file. Writes nothing when dry_run or output_root is None. Returns the summary row."""
    t0 = time.time()
    out_path = os.path.join(output_root, *e.rel_path.split("/")) if output_root else ""
    row = {
        "G": e.g,
        "experiment": e.experiment,
        "phase": e.phase,
        "pv_step": e.pv,
        "input_path": e.src_path,
        "output_path": "" if dry_run else out_path,
        "rel_path": e.rel_path,
        "error": "",
    }
    try:
        with open(e.src_path, "rb") as f:
            data = f.read()
        row["input_md5"] = hashlib.md5(data).hexdigest()
        manual = (_norm(e.rel_path) in (skip_set or set())) or (
            not params.include_manual_labels
            and Image.open(io.BytesIO(data)).info.get("compression") == "packbits"
        )
        cm, unknown = decode(data)
        if manual or cm is None:
            row["status"] = "skipped_manual_label" if manual else "skipped_unknown_colours"
            if cm is None:
                row["error"] = f"{unknown} pixel(s) with a colour other than the 4 classes"
            row.update(_area_columns(cm))
            if not dry_run and output_root:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                _copy_unchanged(e.src_path, out_path)
                row["output_md5"] = row["input_md5"]
        else:
            after, info = postprocess_microplastic(
                cm, source_path=e.src_path, return_info=True, **params.pp_kwargs()
            )
            changed = cm != after
            ch = np.bincount(cm[changed], minlength=4)
            row.update(
                particle_size_detected=info["particle_size_detected"] or "",
                method=info["method"],
                radius_used=info["radius_used"],
                circles_drawn=info["circles_drawn"],
                circles_skipped_overlap=info["circles_skipped_overlap"],
                closing_px_added=info["closing_pixels"],
                hole_px_added=info["hole_pixels"],
                added_patches_dropped=info["added_regions_dropped"],
                px_changed=int(changed.sum()),
                grain_to_mp=int(ch[0]),
                air_to_mp=int(ch[1]),
                water_to_mp=int(ch[2]),
                mp_px_before=int((cm == 3).sum()),
                mp_px_after=int((after == 3).sum()),
                mp_objects_3px_before=_objects_3px(cm == 3),
                mp_objects_3px_after=_objects_3px(after == 3),
                **_area_columns(after),
            )
            row["status"] = "fixed" if row["px_changed"] else "unchanged"
            if not dry_run and output_root:
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                if row["status"] == "unchanged":
                    _copy_unchanged(e.src_path, out_path)
                    row["output_md5"] = row["input_md5"]
                else:
                    tmp = out_path + ls.PARTIAL_SUFFIX
                    ls.save_segmentation_tiff(CFG.index_to_color[after], tmp)
                    with open(tmp, "rb") as f:
                        written = f.read()
                    back, unk = decode(written)  # read back and verify before publishing the file
                    if back is None or back.shape != cm.shape or not np.array_equal(back, after):
                        raise RuntimeError("written file does not read back as the computed map")
                    if np.any((cm == 3) & (back != 3)):
                        raise RuntimeError("a microplastic pixel would have been removed")
                    os.replace(tmp, out_path)
                    row["output_md5"] = hashlib.md5(written).hexdigest()
    except Exception as ex:  # noqa: BLE001
        row["status"], row["error"] = "error", f"{type(ex).__name__}: {ex}"
    row["processing_time_s"] = f"{time.time() - t0:.2f}"
    return row


# ── summary / log files ────────────────────────────────────────────────────────────────────
def read_summary(output_root: str) -> List[dict]:
    p = os.path.join(output_root, SUMMARY_NAME)
    if not os.path.isfile(p):
        return []
    with open(p, newline="", encoding=ls.SUMMARY_ENCODING) as f:
        return list(csv.DictReader(f))


class _Summary:
    def __init__(self, path: str, keep_rows: List[dict]):
        self._f = open(path, "w", newline="", encoding=ls.SUMMARY_ENCODING)
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS, extrasaction="ignore")
        self._w.writeheader()
        for r in keep_rows:
            self._w.writerow(r)
        self._sync()

    def _sync(self):
        self._f.flush()
        os.fsync(self._f.fileno())

    def write(self, row: dict):
        self._w.writerow(row)
        self._sync()

    def close(self):
        self._f.close()


class _Log:
    def __init__(self, path: str):
        self.path = path

    def write(self, line: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# ── resume ────────────────────────────────────────────────────────────────────────────────
def done_rows(output_root: str, entries: List[Entry]) -> dict:
    """rel_path -> summary row, for files whose output exists and still has the recorded MD5."""
    rows = {r["rel_path"]: r for r in read_summary(output_root) if r.get("status") in STATUSES[:4]}
    done = {}
    for e in entries:
        r = rows.get(e.rel_path)
        out = os.path.join(output_root, *e.rel_path.split("/"))
        if r and r.get("output_md5") and os.path.isfile(out) and _md5_file(out) == r["output_md5"]:
            done[e.rel_path] = r
    return done


class ParamsMismatch(ValueError):
    """A folder can only be continued with exactly the parameters (and code) it was started with."""


def check_resume_params(output_root: str, params: Params):
    p = os.path.join(output_root, PARAMS_NAME)
    if not os.path.isfile(p):
        return
    old = json.load(open(p, encoding="utf-8"))
    new = params.as_dict()
    diff = [k for k in new if old.get(k) != new[k]]
    if diff:
        raise ParamsMismatch(
            "This output folder was made with different settings or a different version "
            f"of the code ({', '.join(diff)}). Start a new run instead."
        )


def longest_output_path(entries: List[Entry], output_root: str) -> int:
    return max(
        (
            len(os.path.join(output_root, *e.rel_path.split("/"))) + len(ls.PARTIAL_SUFFIX)
            for e in entries
        ),
        default=0,
    )


def write_readme(
    output_root: str, input_root: str, params: Params, res: Optional["FolderResult"] = None
):
    """README_HOLEFILL.txt in the output folder: what was changed, the parameters, what was copied unchanged,
    and that the retention maps must be regenerated. Rewritten at the end of every session with the totals."""
    pp = params.as_dict()
    c = res.counts if res is not None else {}
    state = (
        "IN PROGRESS (or stopped) — continue the run in the app: Fix Segmentations → Continue an interrupted run"
        if res is None or res.stopped
        else "FINISHED"
    )
    txt = f"""ParticleVision — hollow microplastic particles filled
=====================================================

Status:  {state}
Written: {datetime.now():%Y-%m-%d %H:%M:%S}
Input:   {input_root}   (only read; nothing in it was changed)
Output:  {output_root}

WHAT WAS CHANGED
To show microplastic particles as FULL particles (no Air or Water inside a ring), every app-made segmentation was
post-processed with a method chosen by the particle size in the experiment folder name (e.g. "...-SCA-D10 (40)"):
  {params.method_text()}
In both methods only Air / Water pixels become microplastic - never grain - and no microplastic pixel is removed.
Every image keeps the same size and file format as the app's output. holefill_summary.csv gives, per file, the size
found and the method used (columns particle_size_detected, method, radius_used, circles_drawn, circles_skipped_overlap).

CIRCLES (7 and 10 um): each particle is redrawn as its own filled circle
  1. ring-shaped particles are found with a circle detector for the radius window of the size (ring centre line,
     7 um: 2.0 / 2.3 / 2.6 px, 10 um: 3.0 / 3.3 / 3.6 px; measured from the rings in the lab data);
  2. a circle is accepted only if at least {pp["circle_min_support"]:.0%} of its ring band is red (grain pixels not counted), at most
     half of the ring lies in grain, and its core is less than {pp["circle_max_core_fill"]:.0%} red (solid specks are left alone);
     centres are at least 1.5 x the particle radius apart, so two circles never sit on one particle;
  3. each accepted circle is drawn as a filled disc (radius r + {pp["circle_disc_margin"]} px); nothing between neighbouring discs is filled;
  4. safeguard: a circle whose disc overlaps {pp["circle_max_overlaps"]} or more other circles (a dense cluster) is NOT drawn; such places
     stay exactly as the model drew them.

CLOSING (5 um, and files with no size in the folder name)
  closing of the microplastic mask with a disk of radius r (5 um: r 1; no size: r 2); enclosed Air / Water regions up
  to {pp["max_hole_area"]} px are filled; an added patch larger than {pp["max_added_region"]} px (separate particles turning into one blob)
  is not added.

MEASURED on the 93 app-made IN / WT PV 15 files of the lab data (Oct 21):
  5 um (closing r 1):  microplastic area +0.8 %, objects -0.9 %
  7 um (circles):      microplastic area +12.1 %, objects -2.5 %; changed objects median 23 px (one 7 um particle: 19.6 px)
  10 um (circles):     microplastic area +29.4 %, objects -10.4 %; changed objects median 46 px (one 10 um particle: 40.1 px)
  0 grain pixels changed.

LIMITS
  - About 7-10 % of the clearly ring-shaped particles are not found (gaps too large, or most of the ring in grain);
    they stay as the model drew them.
  - Dense clusters where red and Air / Water are mixed are left as they were (overlap safeguard), so they still look
    like irregular red patches and some rings inside them stay hollow.
  - There are no hand labels for 7 / 10 um particles: the result was judged visually and by particle size, not
    against labels. On the labelled 5 um test images (closing r 1) the microplastic IoU changes by -0.001
    (0.462 -> 0.461); the official results of the paper are reported WITHOUT this post-processing.

PARAMETERS
  method per size       {params.method_text()}
  circles               support >= {pp["circle_min_support"]:.0%}, ring band +/- {pp["circle_band"]} px, at most half of the ring in grain,
                        core < {pp["circle_max_core_fill"]:.0%} red, centre spacing {pp["circle_nms_factor"]} x (radius + {pp["circle_disc_margin"]}), skip if >= {pp["circle_max_overlaps"]} overlaps
  closing               maximum hole area = {pp["max_hole_area"]} px, maximum added patch = {pp["max_added_region"]} px
  manual labels filled  = {"YES (opt-in)" if pp["include_manual_labels"] else "no"}
  code                  = src/postprocess.py (MD5 {pp["postprocess_py_md5"]})

MANUAL LABELS
The lab's manual segmentations (PackBits-compressed TIFFs: the legacy experiments (1)-(11), IN and WT) were
{"FILLED as well (opt-in)" if pp["include_manual_labels"] else "COPIED UNCHANGED (byte-identical)"}, so this folder is complete.
{f"Copied unchanged in this folder: {c.get('skipped_manual_label', 0)} file(s)." if c else ""}

FILES
Same sub-folders and file names as the input (Fused_St_Filt_Comp_Seg_Mod.tif); every file is in the same place as
its original. Per-file status in holefill_summary.csv:
  fixed                   microplastic particles closed / filled
  unchanged               no hollow particles; byte-identical copy
  skipped_manual_label    manual label; byte-identical copy
  skipped_unknown_colours a colour other than the 4 classes; byte-identical copy
  error                   could not be processed (see holefill_log.txt)
{"Totals: " + ", ".join(f"{k} {v}" for k, v in c.items()) if c else ""}
{"Input folder unchanged (every input file re-checked by MD5): " + ("YES" if res.input_unchanged else "NO — see holefill_log.txt") if res is not None and res.input_unchanged is not None else ""}

IMPORTANT
The retention maps (..._RetentionMap_*.tif, UnifiedScale_RetentionGrids.xlsx) were made from the ORIGINAL
segmentations. They are NOT in this folder and must be REGENERATED from this folder.
"""
    with open(os.path.join(output_root, README_NAME), "w", encoding="utf-8") as f:
        f.write(txt)


# ── the run ───────────────────────────────────────────────────────────────────────────────
def run_folder(
    input_root: str,
    output_root: str,
    params: Params,
    dry_run: bool = False,
    resume: bool = False,
    on_start: Optional[Callable[[int, int, Entry], None]] = None,
    on_done: Optional[Callable[[int, int, Entry, dict], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
    max_files: Optional[int] = None,
) -> FolderResult:
    """Fix every segmentation below input_root into output_root (see the module docstring).
    resume=True continues an existing output folder. max_files (tests) stops after that many files, like a Stop."""
    input_root, output_root = os.path.abspath(input_root), os.path.abspath(output_root)
    ls._assert_outside_input(output_root, input_root)
    entries = scan_folder(input_root)
    res = FolderResult(output_root=output_root, total=len(entries))
    skip_set = {_norm(p) for p in params.skip_list}
    t_start = time.time()
    if dry_run:
        for i, e in enumerate(entries, 1):
            if on_start:
                on_start(i, len(entries), e)
            row = process_one(e, None, params, dry_run=True, skip_set=skip_set)
            res.counts[row["status"]] += 1
            res.px_changed += int(row.get("px_changed") or 0)
            res.records.append(row)
            if on_done:
                on_done(i, len(entries), e, row)
        res.seconds = time.time() - t_start
        return res

    exists = os.path.isdir(output_root)
    if exists and not resume and os.listdir(output_root):
        raise FileExistsError(
            f"{output_root} already exists and is not empty — choose 'Continue' or a new folder."
        )
    if resume:
        check_resume_params(output_root, params)
    os.makedirs(output_root, exist_ok=True)
    done = done_rows(output_root, entries) if resume else {}
    res.already_done = len(done)
    keep = [done[e.rel_path] for e in entries if e.rel_path in done]
    md5_path = os.path.join(output_root, INPUT_MD5_NAME)
    first_md5 = {}
    if os.path.isfile(md5_path):
        with open(md5_path, newline="", encoding="utf-8") as f:
            first_md5 = {r["rel_path"]: r["md5"] for r in csv.DictReader(f)}
    json.dump(
        params.as_dict(),
        open(os.path.join(output_root, PARAMS_NAME), "w", encoding="utf-8"),
        indent=1,
    )
    write_readme(output_root, input_root, params)
    log = _Log(os.path.join(output_root, LOG_NAME))
    pp = params.as_dict()
    log.write(
        ("\n== Continued" if resume else "ParticleVision hole filling")
        + f" — {datetime.now():%Y-%m-%d %H:%M:%S} =="
    )
    log.write(f"Input:  {input_root}\nOutput: {output_root}")
    log.write(
        f"Parameters: method per particle size: {params.method_text()}. Circles: ring support >= "
        f"{pp['circle_min_support']:.0%}, at most half of the ring in grain, core < {pp['circle_max_core_fill']:.0%} red, "
        f"a disc overlapping >= {pp['circle_max_overlaps']} other discs not drawn. Closing: enclosed Air/Water up to "
        f"{pp['max_hole_area']} px filled, added patches larger than {pp['max_added_region']} px not added. "
        f"include_manual_labels={pp['include_manual_labels']}, skip list: {len(pp['skip_list'])} file(s)"
    )
    log.write(
        f"Code: src/postprocess.py MD5 {pp['postprocess_py_md5']}; app/holefill_folder.py MD5 {pp['holefill_folder_py_md5']}"
    )
    log.write(
        f"Files found: {len(entries)}; already done (output present with its recorded MD5): {len(done)}"
    )
    summary = _Summary(os.path.join(output_root, SUMMARY_NAME), keep)
    md5_rows = dict(first_md5)
    n_run = 0
    try:
        todo = [e for e in entries if e.rel_path not in done]
        for i, e in enumerate(todo, 1):
            if (should_stop and should_stop()) or (max_files is not None and n_run >= max_files):
                res.stopped = True
                break
            if on_start:
                on_start(i, len(todo), e)
            out_path = os.path.join(output_root, *e.rel_path.split("/"))
            if os.path.lexists(
                out_path
            ):  # an output without a valid record: keep it aside, never overwrite
                aside = f"{out_path}.replaced_{datetime.now():%Y%m%d_%H%M%S}"
                os.rename(out_path, aside)
                log.write(
                    f"  MOVED   {e.rel_path}  —  earlier output without a valid record kept as {os.path.basename(aside)}"
                )
            row = process_one(e, output_root, params, skip_set=skip_set)
            summary.write(row)
            n_run += 1
            res.counts[row["status"]] += 1
            res.px_changed += int(row.get("px_changed") or 0)
            res.records.append(row)
            if (
                e.rel_path in first_md5
                and row.get("input_md5")
                and first_md5[e.rel_path] != row["input_md5"]
            ):
                res.input_changed_files.append(e.rel_path)
            md5_rows.setdefault(e.rel_path, row.get("input_md5", ""))
            extra = (
                f"  ({describe_info(row)}, {row.get('px_changed', 0)} px)"
                if row["status"] == "fixed"
                else f"  —  {row['error']}"
                if row.get("error")
                else ""
            )
            log.write(f"  {row['status'].upper():24s}{e.rel_path}{extra}")
            if on_done:
                on_done(i, len(todo), e, row)
    except BaseException:
        res.stopped = True  # Ctrl+C, the app's Stop (Streamlit interrupts the script), errors:
        raise  # every finished file is complete; the folder can be continued
    finally:
        summary.close()
        with open(md5_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["rel_path", "md5"])
            for k, v in md5_rows.items():
                w.writerow([k, v])
        for e in entries:  # files done in earlier sessions count too
            if e.rel_path in done:
                res.counts[done[e.rel_path]["status"]] += 1
        res.seconds = time.time() - t_start
        if not res.stopped:
            # input check: every input file again, against the MD5 recorded when it was first read
            changed = [
                e.rel_path
                for e in entries
                if md5_rows.get(e.rel_path) and _md5_file(e.src_path) != md5_rows[e.rel_path]
            ]
            res.input_changed_files = sorted(set(res.input_changed_files) | set(changed))
            res.input_unchanged = not res.input_changed_files
        log.write(
            f"{'Stopped' if res.stopped else 'Finished'} {datetime.now():%Y-%m-%d %H:%M:%S} "
            f"({res.seconds:.0f} s this session): "
            + ", ".join(f"{k} {v}" for k, v in res.counts.items())
            + f"; pixels changed this session {res.px_changed}"
        )
        write_readme(output_root, input_root, params, res)
        if res.input_unchanged is not None:
            log.write(
                "Input folder unchanged: "
                + (
                    "YES — every input file has the MD5 recorded when it was read"
                    if res.input_unchanged
                    else f"NO — {len(res.input_changed_files)} file(s) changed: "
                    + ", ".join(res.input_changed_files[:10])
                )
            )
    return res


# ── preview (app page) ────────────────────────────────────────────────────────────────────
def preview(e: Entry, params: Params):
    """(map before, map after, info) for one file, or (None, None, reason) if it can't be filled. Writes nothing."""
    with open(e.src_path, "rb") as f:
        data = f.read()
    if (
        not params.include_manual_labels
        and Image.open(io.BytesIO(data)).info.get("compression") == "packbits"
    ):
        return None, None, "manual label (PackBits) — copied unchanged"
    cm, unknown = decode(data)
    if cm is None:
        return (
            None,
            None,
            f"{unknown} pixel(s) with a colour other than the 4 classes — copied unchanged",
        )
    after, info = postprocess_microplastic(
        cm, source_path=e.src_path, return_info=True, **params.pp_kwargs()
    )
    return cm, after, info


def most_changed_window(before: np.ndarray, after: np.ndarray, size: int = 120):
    """Top-left (y, x) of the size x size window with the most changed pixels (for the preview zoom)."""
    from scipy.ndimage import uniform_filter

    h, w = before.shape
    size = min(size, h, w)
    density = uniform_filter((before != after).astype(np.float32), size=size, mode="constant")
    cy, cx = np.unravel_index(int(np.argmax(density)), density.shape)
    y0 = int(np.clip(cy - size // 2, 0, h - size))
    x0 = int(np.clip(cx - size // 2, 0, w - size))
    return y0, x0, size
