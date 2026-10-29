"""
Lab folder structure: scanning (input) and mirrored output writing.

Pure file-system logic — no Streamlit, no model code. The app, the command-line
tool (tools/segment_lab_folder.py) and the tests all call these same functions,
so what the tests verify is exactly what the app does.

Expected input layout (everything found by pattern, nothing hardcoded):

    <lab folder>/
      G1/ G2/ ...                                   level 1: G + digits
        G1-C0.05-F1.0-P9.0-S30000-SCP-D07 (18)/      level 2: experiment
          Before/  IN/  WT/                         level 3: phase
            Fused_St_Filt_Comp.jpg                  (Before: image directly inside)
            PV 03/ PV 06/ ...                       level 4 (IN/WT only)
              Fused_St_Filt_Comp.jpg

Output layout (a new sibling folder <lab folder>_segmented_<timestamp>/), mirroring the input:

    G1/<experiment>/Before/Fused_St_Filt_Comp_Seg_Mod.tif
    G1/<experiment>/IN/PV 03/Fused_St_Filt_Comp_Seg_Mod.tif     (PV folder named exactly as in the input)
    G1/<experiment>/WT/PV 03/Fused_St_Filt_Comp_Seg_Mod.tif
    (+ Fused_St_Filt_Comp_Seg_Raw.tif next to each one if "Also save the raw segmentation" is on)

Since version 1.1.0 every result is post-processed (hollow microplastic particles filled, method per particle
size; src/postprocess.PipelinePostprocess) in run_lab_batch, before it is saved - the one call site for every
backend. run_settings.json in the output folder records the settings; a run is only continued with the same ones.

Nothing under the input folder is ever written, moved or modified, and an
existing file is never overwritten.
"""

import csv
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

# ─────────────────────────────────────────────────────────────────────────
# Matching rules
# ─────────────────────────────────────────────────────────────────────────
TARGET_STEM = "Fused_St_Filt_Comp"  # exact, case-sensitive
TARGET_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")  # any letter case
# Every output (Before, IN, WT) — the same name the lab uses for its manual segmentations.
# Inputs with this name are never processed: only files named exactly TARGET_STEM are.
OUTPUT_FILENAME = "Fused_St_Filt_Comp_Seg_Mod.tif"
# Optional (Run page → Advanced → "Also save the raw (unfilled) segmentation"): the map before post-processing.
RAW_OUTPUT_FILENAME = "Fused_St_Filt_Comp_Seg_Raw.tif"
RUN_SETTINGS_NAME = (
    "run_settings.json"  # post-processing settings + versions of a run (Continue check)
)
# Older versions wrote Before/Fused_St_Filt_Comp_Segmented.tif and IN|WT/<PV>_Segmented.tif
# (no PV folders). Such output folders can still be opened on the Charts page (via
# summary.csv) but cannot be continued.
OLD_OUTPUT_SUFFIX = "_Segmented.tif"
PHASES = ("Before", "IN", "WT")  # physical sequence order
PV_PHASES = ("IN", "WT")
# Windows creates these in picture folders; treated like hidden files.
SYSTEM_FILES = {"thumbs.db", "desktop.ini"}

_NUM = r"\d+(?:\.\d+)?"
RE_G = re.compile(r"G(\d+)")
# G#-C#-F#-P#-S#-SC?-D## with an optional " (n)" suffix
RE_EXPERIMENT = re.compile(rf"(G\d+)-C{_NUM}-F{_NUM}-P{_NUM}-S\d+-SC[A-Z]-D\d+(?: \(\d+\))?")
RE_PV = re.compile(r"PV ?(\d+)")

EXPERIMENT_PATTERN_TEXT = "G#-C#-F#-P#-S#-SC?-D## with optional ' (n)'"


def natural_key(s: str):
    """Sort 'G2' before 'G10' and 'PV 03' before 'PV 12'."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]


# ─────────────────────────────────────────────────────────────────────────
# Scan result types
# ─────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class LabImage:
    g: str  # "G1"
    experiment: str  # "G1-C0.05-F1.0-P9.0-S30000-SCP-D07 (18)"
    phase: str  # "Before" | "IN" | "WT"
    pv: Optional[str]  # "PV 03" (folder name as-is), None for Before
    src_path: str  # absolute path of the image to segment
    rel_dirs: Tuple[str, ...]  # folder names from the lab root to the image's folder

    @property
    def pv_number(self) -> Optional[int]:
        return int(RE_PV.fullmatch(self.pv).group(1)) if self.pv else None

    @property
    def label(self) -> str:
        return " › ".join(self.rel_dirs)

    @property
    def sequence_key(self):
        """Physical order within an experiment: Before → IN PV↑ → WT PV↑."""
        return (PHASES.index(self.phase), self.pv_number or 0, natural_key(self.pv or ""))

    @property
    def sort_key(self):
        return (int(self.g[1:]), natural_key(self.experiment)) + self.sequence_key


@dataclass(frozen=True)
class Skip:
    rel_path: str  # relative to the lab root
    kind: str  # "folder" | "file"
    reason: str


@dataclass
class ScanResult:
    root: str
    images: List[LabImage] = field(default_factory=list)
    skipped: List[Skip] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def skipped_folders(self) -> List[Skip]:
        return [s for s in self.skipped if s.kind == "folder"]

    @property
    def skipped_files(self) -> List[Skip]:
        return [s for s in self.skipped if s.kind == "file"]

    def experiment_rows(self) -> List[dict]:
        """One row per experiment: G, experiment, Before/IN/WT counts, total."""
        rows = {}
        for img in self.images:
            key = (img.g, img.experiment)
            r = rows.setdefault(
                key,
                {
                    "G": img.g,
                    "Experiment": img.experiment,
                    "Before": 0,
                    "IN": 0,
                    "WT": 0,
                    "Total": 0,
                },
            )
            r[img.phase] += 1
            r["Total"] += 1
        return [rows[k] for k in sorted(rows, key=lambda k: (int(k[0][1:]), natural_key(k[1])))]


# ─────────────────────────────────────────────────────────────────────────
# Scanner
# ─────────────────────────────────────────────────────────────────────────
def _entries(path: str, scan: ScanResult, rel: str):
    """Naturally-sorted directory entries; hidden entries are logged and dropped."""
    try:
        with os.scandir(path) as it:
            items = sorted(it, key=lambda e: natural_key(e.name))
    except OSError as e:
        scan.skipped.append(Skip(rel or ".", "folder", f"could not read folder ({e.strerror})"))
        return []
    visible = []
    for e in items:
        r = os.path.join(rel, e.name) if rel else e.name
        if e.name.startswith(".") or e.name.lower() in SYSTEM_FILES:
            scan.skipped.append(
                Skip(
                    r,
                    "folder" if e.is_dir(follow_symlinks=False) else "file",
                    "hidden / system file",
                )
            )
        elif e.is_symlink():
            scan.skipped.append(Skip(r, "file", "symbolic link (not followed)"))
        else:
            visible.append(e)
    return visible


def _is_target(name: str) -> bool:
    stem, ext = os.path.splitext(name)
    return stem == TARGET_STEM and ext.lower() in TARGET_EXTS


def _file_skip_reason(name: str) -> str:
    stem, ext = os.path.splitext(name)
    if stem == TARGET_STEM:
        return f"unsupported extension '{ext}'"
    if stem.startswith(TARGET_STEM):
        return f"name is not exactly {TARGET_STEM} (exact match required)"
    return "not the target image"


def _find_target(folder: str, rel: str, scan: ScanResult) -> Optional[str]:
    """Return the single target image in `folder`, or None (reason logged)."""
    candidates = []
    for e in _entries(folder, scan, rel):
        r = os.path.join(rel, e.name)
        if e.is_dir():
            scan.skipped.append(Skip(r, "folder", "subfolder here is not scanned"))
        elif _is_target(e.name):
            candidates.append(e)
        else:
            scan.skipped.append(Skip(r, "file", _file_skip_reason(e.name)))

    if len(candidates) == 1:
        return candidates[0].path
    if not candidates:
        scan.skipped.append(
            Skip(rel, "folder", f"no target image ({TARGET_STEM} with .jpg/.jpeg/.png/.tif/.tiff)")
        )
    else:
        names = ", ".join(c.name for c in candidates)
        scan.skipped.append(
            Skip(
                rel,
                "folder",
                f"{len(candidates)} candidate images ({names}) — ambiguous, not processed",
            )
        )
    return None


def scan_lab_folder(root: str) -> ScanResult:
    root = os.path.abspath(root)
    scan = ScanResult(root=root)

    for g_entry in _entries(root, scan, ""):
        g = g_entry.name
        if not g_entry.is_dir():
            scan.skipped.append(Skip(g, "file", "file at top level (only G folders are processed)"))
            continue
        if not RE_G.fullmatch(g):
            scan.skipped.append(
                Skip(g, "folder", "not a G folder (expected 'G' followed by digits)")
            )
            continue

        n_before = len(scan.images)
        for x_entry in _entries(g_entry.path, scan, g):
            exp = x_entry.name
            rel_x = os.path.join(g, exp)
            if not x_entry.is_dir():
                scan.skipped.append(
                    Skip(
                        rel_x,
                        "file",
                        "file inside G folder (only experiment folders are processed)",
                    )
                )
                continue
            m = RE_EXPERIMENT.fullmatch(exp)
            if not m:
                scan.skipped.append(
                    Skip(
                        rel_x,
                        "folder",
                        f"name does not follow the experiment structure ({EXPERIMENT_PATTERN_TEXT})",
                    )
                )
                continue
            if m.group(1) != g:
                scan.skipped.append(
                    Skip(
                        rel_x,
                        "folder",
                        f"G prefix '{m.group(1)}' does not match parent folder '{g}'",
                    )
                )
                continue
            _scan_experiment(x_entry.path, g, exp, scan)

        if len(scan.images) == n_before:
            scan.warnings.append(f"{g}: no processable images found")

    scan.images.sort(key=lambda im: im.sort_key)
    return scan


def _scan_experiment(path: str, g: str, exp: str, scan: ScanResult):
    rel_x = os.path.join(g, exp)
    seen_phases = set()
    for p_entry in _entries(path, scan, rel_x):
        phase = p_entry.name
        rel_p = os.path.join(rel_x, phase)
        if not p_entry.is_dir():
            scan.skipped.append(Skip(rel_p, "file", "file directly inside experiment folder"))
            continue
        if phase not in PHASES:
            scan.skipped.append(
                Skip(rel_p, "folder", "not a phase folder (expected exactly Before, IN or WT)")
            )
            continue
        seen_phases.add(phase)

        if phase == "Before":
            src = _find_target(p_entry.path, rel_p, scan)
            if src:
                scan.images.append(LabImage(g, exp, phase, None, src, (g, exp, phase)))
            continue

        pv_nums = {}
        for pv_entry in _entries(p_entry.path, scan, rel_p):
            pv = pv_entry.name
            rel_pv = os.path.join(rel_p, pv)
            if not pv_entry.is_dir():
                reason = (
                    "image directly inside IN/WT, not inside a PV folder"
                    if _is_target(pv)
                    else "file directly inside IN/WT"
                )
                scan.skipped.append(Skip(rel_pv, "file", reason))
                continue
            m = RE_PV.fullmatch(pv)
            if not m:
                scan.skipped.append(
                    Skip(
                        rel_pv, "folder", "not a PV folder (expected 'PV', optional space, digits)"
                    )
                )
                continue
            src = _find_target(pv_entry.path, rel_pv, scan)
            if src:
                num = int(m.group(1))
                if num in pv_nums:
                    scan.warnings.append(
                        f"{rel_p}: '{pv_nums[num]}' and '{pv}' are the same PV step ({num}); both processed"
                    )
                pv_nums.setdefault(num, pv)
                scan.images.append(LabImage(g, exp, phase, pv, src, (g, exp, phase, pv)))

    missing = [p for p in PHASES if p not in seen_phases]
    if missing:
        scan.warnings.append(f"{rel_x}: missing phase folder(s): {', '.join(missing)}")


# ─────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────
CLASS_DISPLAY = [("Grains", "Grains"), ("Void", "Air"), ("Water", "Water"), ("Fines", "Fines")]

SUMMARY_COLUMNS = (
    ["G", "experiment", "phase", "pv_step", "input_path", "output_path"]
    + [f"{d.lower()}_{s}" for _, d in CLASS_DISPLAY for s in ("px", "pct")]
    + ["processing_time_s", "status", "error"]
)
# Added in 1.1.0 AFTER the original columns (so the lab's scripts still find theirs in the same order): which
# app version and backend made each result, and what the post-processing did. The *_px / *_pct columns above are
# counted on the final (post-processed) map; mp_px_raw / mp_px_final = microplastic pixels before / after.
RUN_COLUMNS = [
    "app_version",
    "backend",
    "postprocess",
    "particle_size_detected",
    "pp_method",
    "mp_px_raw",
    "mp_px_final",
    "circles_drawn",
    "circles_skipped_overlap",
]


def make_output_root(input_dir: str, now: Optional[datetime] = None) -> str:
    """<parent>/<input name>_segmented_<timestamp> — a sibling, never inside the input."""
    input_dir = os.path.abspath(input_dir)
    ts = (now or datetime.now()).strftime("%Y%m%d_%H%M%S")
    return os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_segmented_{ts}")


def output_path_for(output_root: str, img: LabImage) -> str:
    """Before: <G>/<exp>/Before/Fused_St_Filt_Comp_Seg_Mod.tif
    IN/WT:  <G>/<exp>/<IN|WT>/<PV>/Fused_St_Filt_Comp_Seg_Mod.tif — <PV> is the input's
            PV folder name exactly (e.g. "PV 03", same spacing and zero-padding)."""
    return os.path.join(output_root, *img.rel_dirs, OUTPUT_FILENAME)


def raw_output_path_for(output_root: str, img: LabImage) -> str:
    """Where the optional unfilled map goes: next to the normal result, named Fused_St_Filt_Comp_Seg_Raw.tif."""
    return os.path.join(output_root, *img.rel_dirs, RAW_OUTPUT_FILENAME)


def save_segmentation_tiff(rgb, path: str):
    """THE way a segmentation result is written (RGB, uncompressed TIFF, the 4 class colours as given).
    Used by run_lab_batch and by the hole-filling tool (app/holefill_folder.py), so both write the same format."""
    from PIL import Image

    Image.fromarray(rgb).save(path, format="TIFF")


class OutputExistsError(RuntimeError):
    """A file the run would write already exists — the run stops instead of overwriting."""


class OldLayoutError(ValueError):
    """The chosen output folder was made by an older version (old names / layout)."""


OLD_LAYOUT_MESSAGE = (
    "This output folder was made by an older version of ParticleVision (results named "
    "…_Segmented.tif, IN/WT results not in PV folders). It can't be continued with this version — "
    'choose "Start a new run" instead. The old folder is left as it is, and the Charts page can '
    "still open it."
)


def uses_old_layout(output_root: str) -> bool:
    """True if `output_root` holds results in the old naming/layout (summary.csv paths or files)."""
    path = os.path.join(output_root, "summary.csv")
    if os.path.isfile(path):
        for r in read_summary(output_root):
            out = r.get("output_path") or ""
            if (
                r.get("status") == "ok"
                and out
                and os.path.basename(out.replace("\\", "/")) != OUTPUT_FILENAME
            ):
                return True
    for _, _, files in os.walk(output_root):
        if any(f.endswith((OLD_OUTPUT_SUFFIX, OLD_OUTPUT_SUFFIX + PARTIAL_SUFFIX)) for f in files):
            return True
    return False


def _check_new_run_targets(scan: ScanResult, output_root: str, save_raw: bool = False):
    """Before a new run writes anything: every target path is distinct and none exists yet."""
    seen = {}
    targets = [(img, output_path_for(output_root, img)) for img in scan.images]
    if save_raw:
        targets += [(img, raw_output_path_for(output_root, img)) for img in scan.images]
    for img, out in targets:
        key = os.path.normcase(out)
        if key in seen:
            raise OutputExistsError(
                f"Two images would be written to the same file ({out}): "
                f"{seen[key]} and {img.label}. Nothing was written."
            )
        seen[key] = img.label
        if os.path.lexists(out):
            raise OutputExistsError(
                f"{out} already exists — stopped so it is not overwritten. Nothing was written."
            )


def _assert_outside_input(output_root: str, input_root: str):
    # normcase: Windows paths are case-insensitive (C:\\PV == c:\\pv)
    o = os.path.normcase(os.path.realpath(output_root))
    i = os.path.normcase(os.path.realpath(input_root))
    if o == i or o.startswith(i + os.sep):
        raise ValueError(
            f"Output folder {o} is inside the input folder {i}; refusing to write there."
        )


SUMMARY_ENCODING = "utf-8-sig"  # UTF-8 with BOM, so Excel on Windows shows accents correctly


class SummaryCSV:
    """summary.csv, one row per image, flushed to disk after every row.
    `keep_rows` (resume) are written first, unchanged."""

    def __init__(self, path: str, keep_rows: Optional[List[dict]] = None):
        self.path = path
        self._f = open(path, "w", newline="", encoding=SUMMARY_ENCODING)
        self._w = csv.DictWriter(
            self._f, fieldnames=SUMMARY_COLUMNS + RUN_COLUMNS, extrasaction="ignore"
        )
        self._w.writeheader()
        for row in keep_rows or []:
            self._w.writerow(row)
        self._sync()

    def _sync(self):
        self._f.flush()
        os.fsync(self._f.fileno())

    def write(
        self,
        img: LabImage,
        out_path: str,
        class_counts: Optional[dict],
        total_px: Optional[int],
        seconds: float,
        status: str,
        error: str = "",
        run: Optional[dict] = None,
    ):
        row = {
            "G": img.g,
            "experiment": img.experiment,
            "phase": img.phase,
            "pv_step": img.pv or "",
            "input_path": img.src_path,
            "output_path": out_path if status == "ok" else "",
            "processing_time_s": f"{seconds:.2f}",
            "status": status,
            "error": error,
        }
        for cname, disp in CLASS_DISPLAY:
            if class_counts is not None and total_px:
                row[f"{disp.lower()}_px"] = class_counts[cname]
                row[f"{disp.lower()}_pct"] = f"{class_counts[cname] / total_px * 100:.4f}"
            else:
                row[f"{disp.lower()}_px"] = row[f"{disp.lower()}_pct"] = ""
        row.update({k: v for k, v in (run or {}).items() if k in RUN_COLUMNS})
        self._w.writerow(row)
        self._sync()

    def close(self):
        self._f.close()


class RunLog:
    """run_log.txt: scan results + skips first, then failures as they happen."""

    def __init__(
        self, path: str, scan: ScanResult, output_root: str, header: Optional[List[str]] = None
    ):
        self.path = path
        with open(path, "w", encoding="utf-8") as f:
            f.write(f"ParticleVision lab-folder run — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            for line in header or []:
                f.write(line + "\n")
            f.write(f"Input:  {scan.root}\nOutput: {output_root}\n")
            f.write(f"Images selected: {len(scan.images)}\n")
            f.write(
                f"Output files: <G>/<experiment>/Before/{OUTPUT_FILENAME} and "
                f"<G>/<experiment>/<IN|WT>/<PV folder>/{OUTPUT_FILENAME}\n"
            )
            f.write(
                f"Note: {OUTPUT_FILENAME} is also the name of the lab's manual segmentations — "
                f"do not copy these files into the original lab folders.\n\n"
            )
            f.write(f"== Warnings ({len(scan.warnings)}) ==\n")
            for w in scan.warnings:
                f.write(f"  {w}\n")
            for title, items in (
                ("Skipped folders", scan.skipped_folders),
                ("Skipped files", scan.skipped_files),
            ):
                f.write(f"\n== {title} ({len(items)}) ==\n")
                for s in items:
                    f.write(f"  {s.rel_path}  —  {s.reason}\n")
            f.write("\n== Processing ==\n")

    def append(self, line: str):
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    @classmethod
    def resumed(
        cls, path: str, scan: ScanResult, plan: "ResumePlan", header: Optional[List[str]] = None
    ) -> "RunLog":
        """Continue an existing run_log.txt with a clearly marked resume section."""
        log = cls.__new__(cls)
        log.path = path
        lines = (
            [f"\n== Resumed {datetime.now():%Y-%m-%d %H:%M:%S} =="]
            + list(header or [])
            + [
                f"Input:  {scan.root}",
                f"Images selected: {len(scan.images)}",
                f"Already done (skipped): {len(plan.done)}",
                f"To process now: {len(plan.todo)}",
            ]
        )
        for label, why in plan.redo_reasons.items():
            lines.append(f"  redo    {label}  —  {why}")
        if plan.foreign_rows:
            lines.append(
                f"  kept {plan.foreign_rows} summary.csv row(s) for images no longer found in the lab folder"
            )
        lines.append("\n== Processing (resumed) ==")
        log.append("\n".join(lines))
        return log


class SettingsMismatchError(ValueError):
    """Continue refused: the output folder was made with other post-processing settings or another version."""


def read_run_settings(output_root: str) -> Optional[dict]:
    """run_settings.json of an output folder (version 1.1.0+), or None."""
    import json

    path = os.path.join(output_root, RUN_SETTINGS_NAME)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _size_text(v) -> str:
    return "Auto (from the folder name)" if v in ("auto", None) else f"{v} µm"


def settings_differences(old: dict, new: dict) -> Tuple[List[str], bool]:
    """(plain-language differences, fixable by the user on the Run page)."""
    diffs, fixable = [], True
    if old.get("app_version") != new.get("app_version"):
        diffs.append(
            f"made with ParticleVision {old.get('app_version')}, this is {new.get('app_version')}"
        )
        fixable = False
    if old.get("postprocess") != new.get("postprocess"):
        diffs.append(
            f'"Fill hollow microplastic particles" was {old.get("postprocess")} for this run, '
            f"now {new.get('postprocess')}"
        )
    if old.get("particle_size_setting") != new.get("particle_size_setting"):
        diffs.append(
            f'"Particle size" was {_size_text(old.get("particle_size_setting"))}, '
            f"now {_size_text(new.get('particle_size_setting'))}"
        )
    if bool(old.get("save_raw")) != bool(new.get("save_raw")):
        diffs.append(
            f'"Also save the raw segmentation" was {"on" if old.get("save_raw") else "off"}, '
            f"now {'on' if new.get('save_raw') else 'off'}"
        )
    o, n = old.get("postprocess_settings") or {}, new.get("postprocess_settings") or {}
    keys = sorted(k for k in set(o) | set(n) if o.get(k) != n.get(k))
    if keys:
        fixable = False
        diffs.append(
            "the post-processing code (src/postprocess.py) is a different version"
            if keys == ["postprocess_py_md5"]
            else f"the post-processing parameters differ ({', '.join(k for k in keys if k != 'postprocess_py_md5')})"
        )
    return diffs, fixable


def check_resume_settings(output_root: str, postprocess) -> None:
    """A run may only be continued with the SAME post-processing settings and code version (run_settings.json)."""
    old = read_run_settings(output_root)
    if old is None:
        raise SettingsMismatchError(
            "This output folder was made by a ParticleVision version before 1.1.0 (its results are not "
            "post-processed and no settings were recorded), so it can't be continued with this version — "
            'choose "Start a new run". The old folder is left as it is.'
        )
    diffs, fixable = settings_differences(old, postprocess.record())
    if diffs:
        raise SettingsMismatchError(
            "This run can't be continued with the current settings: "
            + "; ".join(diffs)
            + ". "
            + (
                "Set them back as they were (Run page → Advanced), or start a new run."
                if fixable
                else "Start a new run instead (it creates a new output folder)."
            )
        )


@dataclass
class BatchResult:
    output_root: str
    total: int  # images processed in THIS run (excludes skipped)
    done: int = 0
    failed: int = 0
    skipped: int = 0  # resume: already done before this run
    failures: List[Tuple[str, str]] = field(default_factory=list)  # (label, error)
    seconds: List[float] = field(default_factory=list)
    pp_seconds: List[float] = field(
        default_factory=list
    )  # time of the post-processing step per image


def run_header(postprocess, backend: str) -> List[str]:
    """The lines at the top of run_log.txt: app version, backend, post-processing and code MD5."""
    rec = postprocess.record()
    return [
        f"ParticleVision version: {rec['app_version']}",
        f"Backend: {backend or 'not recorded'}  (an image that needed a fallback says so on its own line)",
        f"Post-processing (fill hollow microplastic particles): {postprocess.describe()}",
        f"Post-processing code: src/postprocess.py MD5 {rec['postprocess_settings']['postprocess_py_md5']}",
    ]


def run_lab_batch(
    scan: ScanResult,
    output_root: str,
    predict_fn: Callable[[str], Tuple],  # path -> (rgb_out, class_map, elapsed[, note])
    on_start: Optional[Callable[[int, LabImage], None]] = None,
    on_done: Optional[Callable[[int, LabImage, dict], None]] = None,
    between_images: Optional[Callable[[], None]] = None,
    resume_plan: Optional["ResumePlan"] = None,
    postprocess=None,  # src.postprocess.PipelinePostprocess; None = the defaults (on, Auto size)
    backend: str = "",  # the runner's label, e.g. "Apple GPU (MPS)" - recorded per image
) -> BatchResult:
    """Segment every scanned image, writing each output as soon as it's finished.

    Every image's map from `predict_fn` (the model, any backend) is post-processed here by
    postprocess.apply() - THE pipeline call site - before it is saved and its class areas are counted.
    `on_done` receives a record dict with the same values written to summary.csv.
    With `resume_plan` (from plan_resume) the existing output folder is continued:
    images already done are skipped, the rest are processed into the same folder - only if the
    folder's run_settings.json matches `postprocess` (check_resume_settings).
    """
    import json

    import numpy as np

    from src.config import CFG, PARTICLEVISION_VERSION
    from src.postprocess import PipelinePostprocess, describe_info

    pp = postprocess if postprocess is not None else PipelinePostprocess()
    header = run_header(pp, backend)
    _assert_outside_input(output_root, scan.root)
    if resume_plan is None:
        if os.path.exists(output_root):
            raise OutputExistsError(
                f"The output folder {output_root} already exists — stopped so nothing "
                f"in it is overwritten."
            )
        _check_new_run_targets(scan, output_root, save_raw=pp.save_raw)
        os.makedirs(output_root, exist_ok=False)
        with open(os.path.join(output_root, RUN_SETTINGS_NAME), "w", encoding="utf-8") as f:
            json.dump(
                {
                    **pp.record(),
                    "backend": backend,
                    "created": f"{datetime.now():%Y-%m-%d %H:%M:%S}",
                },
                f,
                indent=2,
            )
        summary = SummaryCSV(os.path.join(output_root, "summary.csv"))
        log = RunLog(os.path.join(output_root, "run_log.txt"), scan, output_root, header)
        todo = scan.images
        res = BatchResult(output_root=output_root, total=len(todo))
    else:
        if not os.path.isfile(os.path.join(output_root, "summary.csv")):
            raise ValueError(
                f"{output_root} has no summary.csv — not a previous run's output folder."
            )
        check_resume_settings(output_root, pp)  # before anything is written
        summary = SummaryCSV(
            os.path.join(output_root, "summary.csv"), keep_rows=resume_plan.kept_rows
        )
        log = RunLog.resumed(os.path.join(output_root, "run_log.txt"), scan, resume_plan, header)
        todo = resume_plan.todo
        res = BatchResult(output_root=output_root, total=len(todo), skipped=len(resume_plan.done))

    try:
        for i, img in enumerate(todo, start=1):
            if on_start:
                on_start(i, img)
            out_path = output_path_for(output_root, img)
            raw_path = raw_output_path_for(output_root, img)
            for p in [out_path] + ([raw_path] if pp.save_raw else []):
                if os.path.lexists(p):
                    if resume_plan is None:
                        raise OutputExistsError(
                            f"{p} already exists — stopped so it is not overwritten."
                        )
                    # resume: an existing output that plan_resume judged unusable is moved aside, not overwritten
                    aside = f"{p}.replaced_{datetime.now():%Y%m%d_%H%M%S}"
                    os.rename(p, aside)
                    log.append(
                        f"  MOVED   {img.label}  —  unusable earlier output kept as {os.path.basename(aside)}"
                    )
            t0 = time.time()
            run = {
                "app_version": PARTICLEVISION_VERSION,
                "backend": backend,
                "postprocess": "on" if pp.enabled else "off",
            }
            try:
                out = predict_fn(img.src_path)
                rgb_raw, raw_map = out[0], out[1]
                note = out[3] if len(out) > 3 else ""
                if note:
                    run["backend"] = (
                        f"{backend} — fallback: {note}" if backend else f"fallback: {note}"
                    )
                # ── THE post-processing call site of the pipeline (every backend passes here) ──
                t_pp = time.time()
                class_map, info = pp.apply(raw_map, img.src_path)
                pp_secs = time.time() - t_pp
                rgb_out = rgb_raw if not pp.enabled else CFG.index_to_color[class_map]
                run.update(
                    particle_size_detected=info["particle_size_detected"] or "",
                    pp_method=info["method"],
                    mp_px_raw=info["mp_px_raw"],
                    mp_px_final=info["mp_px_final"],
                    circles_drawn=info.get("circles_drawn", 0),
                    circles_skipped_overlap=info.get("circles_skipped_overlap", 0),
                )
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                writes = ([(rgb_raw, raw_path)] if pp.save_raw else []) + [(rgb_out, out_path)]
                for (
                    rgb,
                    path,
                ) in (
                    writes
                ):  # the raw copy first: a result counts as done once its final file exists
                    tmp = path + PARTIAL_SUFFIX
                    save_segmentation_tiff(rgb, tmp)
                    if os.path.lexists(path):  # appeared while this image was running
                        raise OutputExistsError(
                            f"{path} appeared during the run — stopped so it is not "
                            f"overwritten (the new result is kept as {os.path.basename(tmp)})."
                        )
                    os.replace(tmp, path)  # never leaves a half-written .tif
                counts = np.bincount(class_map.ravel(), minlength=4)
                class_counts = {c: int(counts[k]) for k, (c, _) in enumerate(CLASS_DISPLAY)}
                total_px = int(class_map.size)
                secs = time.time() - t0
                summary.write(img, out_path, class_counts, total_px, secs, "ok", run=run)
                log.append(
                    f"  OK      {img.label}  ({secs:.1f}s, post-processing {pp_secs:.2f}s)  —  "
                    f"{describe_info(info)}" + (f"  —  {note}" if note else "")
                )
                res.done += 1
                res.seconds.append(secs)
                res.pp_seconds.append(pp_secs)
                record = {
                    "status": "ok",
                    "out_path": out_path,
                    "class_counts": class_counts,
                    "total_px": total_px,
                    "seconds": secs,
                    "note": note,
                    "postprocess": info,
                    "pp_seconds": pp_secs,
                    "raw_path": raw_path if pp.save_raw else None,
                }
            except OutputExistsError:
                raise  # never overwrite: stop the whole run
            except Exception as e:
                secs = time.time() - t0
                err = f"{type(e).__name__}: {e}"
                summary.write(img, out_path, None, None, secs, "failed", err, run=run)
                log.append(f"  FAILED  {img.label}  —  {err}")
                res.failed += 1
                res.failures.append((img.label, err))
                record = {"status": "failed", "error": err, "seconds": secs}
            if on_done:
                on_done(i, img, record)
            if between_images:
                between_images()
    except OutputExistsError as e:
        log.append(f"  STOPPED — {e}")
        raise
    finally:
        summary.close()
        log.append(
            f"\nFinished {datetime.now():%Y-%m-%d %H:%M:%S}: "
            f"{res.done} of {res.total} processed, {res.failed} failed"
            + (f", {res.skipped} skipped (already done before resuming)." if resume_plan else ".")
        )
    return res


# ─────────────────────────────────────────────────────────────────────────
# Reading results back (Charts page) — summary.csv is the single source
# ─────────────────────────────────────────────────────────────────────────
def read_summary(output_root: str) -> List[dict]:
    """All rows of <output_root>/summary.csv, exactly as written."""
    # utf-8-sig also reads files written before the BOM was added
    with open(os.path.join(output_root, "summary.csv"), newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def row_sequence_key(row: dict):
    """Physical order within an experiment: Before → IN PV↑ → WT PV↑."""
    pv = row.get("pv_step") or ""
    m = RE_PV.fullmatch(pv)
    return (
        PHASES.index(row["phase"]) if row["phase"] in PHASES else len(PHASES),
        int(m.group(1)) if m else 0,
        natural_key(pv),
    )


def row_step_label(row: dict) -> str:
    return row["phase"] if not row.get("pv_step") else f"{row['phase']} {row['pv_step']}"


def experiment_sequence(rows: List[dict], g: str, experiment: str) -> List[dict]:
    """Successful rows of one experiment, in physical sequence order."""
    sel = [r for r in rows if r["G"] == g and r["experiment"] == experiment and r["status"] == "ok"]
    return sorted(sel, key=row_sequence_key)


# ─────────────────────────────────────────────────────────────────────────
# Pre-run checks: path length and disk space
# ─────────────────────────────────────────────────────────────────────────
PARTIAL_SUFFIX = ".partial"  # temporary name while a .tif is being written


def longest_paths(scan: ScanResult, output_root: str) -> Tuple[str, str]:
    """(longest input path, longest output path incl. the temporary suffix)."""
    if not scan.images:
        return "", ""
    inp = max((im.src_path for im in scan.images), key=len)
    out = max((output_path_for(output_root, im) + PARTIAL_SUFFIX for im in scan.images), key=len)
    return inp, out


def path_length_problems(scan: ScanResult, output_root: str, limit: Optional[int]) -> List[str]:
    """Plain-language problems if any input/output path exceeds `limit`
    (None = no limit on this computer). Names the exact folder to move."""
    if limit is None or not scan.images:
        return []
    inp, out = longest_paths(scan, output_root)
    problems = []
    if len(out) > limit:
        problems.append(
            f"The longest output file would need {len(out)} characters, over this computer's limit of "
            f"{limit}: {out}. The output is created next to the lab folder '{scan.root}', so move that "
            f"lab folder to a short location such as C:\\PV\\ and select it again."
        )
    if len(inp) > limit:
        problems.append(
            f"An input image path is {len(inp)} characters, over the limit of {limit}: {inp}. "
            f"Move the lab folder '{scan.root}' to a short location such as C:\\PV\\."
        )
    return problems


def estimate_output_bytes(images: List[LabImage]) -> int:
    """Size of the uncompressed RGB .tif files the run will write (reads image
    headers only)."""
    from PIL import Image

    total = 0
    for im in images:
        try:
            with Image.open(im.src_path) as src:
                w, h = src.size
        except Exception:  # noqa: BLE001
            w, h = 4550, 3300  # typical lab image
        total += w * h * 3 + 16 * 1024
    return total


# ─────────────────────────────────────────────────────────────────────────
# Resume an interrupted run
# ─────────────────────────────────────────────────────────────────────────
@dataclass
class ResumePlan:
    output_root: str
    done: List[LabImage] = field(default_factory=list)  # valid output + ok row → skipped
    todo: List[LabImage] = field(default_factory=list)  # processed (again) now
    redo_reasons: Dict[str, str] = field(
        default_factory=dict
    )  # label → why an existing output is redone
    kept_rows: List[dict] = field(default_factory=list)  # summary.csv rows carried over
    foreign_rows: int = 0  # rows for images not in this scan (kept)


def _row_key(g, experiment, phase, pv) -> tuple:
    return (g, experiment, phase, pv or "")


def output_is_valid(out_path: str, src_path: str, allowed_colors) -> Tuple[bool, str]:
    """A finished output: opens, same size as its input, only the class colours."""
    from PIL import Image

    if not os.path.isfile(out_path):
        return False, "no output yet"
    try:
        with Image.open(out_path) as o, Image.open(src_path) as s:
            if o.size != s.size:
                return False, f"output size {o.size} differs from input {s.size}"
            if o.mode != "RGB":
                return False, f"output is {o.mode}, not RGB"
            colors = o.getcolors(maxcolors=len(allowed_colors))
    except Exception as e:  # noqa: BLE001
        return False, f"output can't be opened ({type(e).__name__})"
    if colors is None or not {c for _, c in colors} <= set(allowed_colors):
        return False, "output contains colours other than the 4 classes"
    return True, ""


def plan_resume(scan: ScanResult, output_root: str, allowed_colors=None) -> ResumePlan:
    """Decide which images of `scan` are already done in `output_root`.

    Done = output .tif valid (output_is_valid) AND summary.csv has its 'ok' row.
    Everything else is processed again; an existing but invalid output is
    replaced, and the reason is recorded (never silently)."""
    if uses_old_layout(output_root):
        raise OldLayoutError(OLD_LAYOUT_MESSAGE)
    if allowed_colors is None:
        from src.config import CFG

        allowed_colors = [tuple(c) for c in CFG.index_to_color.tolist()]
    rows = (
        read_summary(output_root)
        if os.path.isfile(os.path.join(output_root, "summary.csv"))
        else []
    )
    settings = read_run_settings(output_root) or {}
    need_raw = bool(
        settings.get("save_raw")
    )  # then an image is only done if its raw copy is there too
    ok_rows = {
        _row_key(r["G"], r["experiment"], r["phase"], r["pv_step"]): r
        for r in rows
        if r.get("status") == "ok"
    }
    plan = ResumePlan(output_root=output_root)
    scanned = set()
    for img in scan.images:
        key = _row_key(img.g, img.experiment, img.phase, img.pv)
        scanned.add(key)
        out = output_path_for(output_root, img)
        valid, why = output_is_valid(out, img.src_path, allowed_colors)
        if valid and need_raw:
            valid, why = output_is_valid(
                raw_output_path_for(output_root, img), img.src_path, allowed_colors
            )
            why = f"raw copy: {why}" if why else why
        row = ok_rows.get(key)
        if valid and row is not None:
            plan.done.append(img)
            plan.kept_rows.append(row)
            continue
        plan.todo.append(img)
        if os.path.exists(out) and not valid:
            plan.redo_reasons[img.label] = (
                f"existing output unusable ({why}) — redone; the old file is kept aside"
            )
        elif valid and row is None:
            plan.redo_reasons[img.label] = (
                "output existed but had no 'ok' row in summary.csv — redone; "
                "the old file is kept aside"
            )
    foreign = [
        r
        for r in rows
        if _row_key(r["G"], r["experiment"], r["phase"], r["pv_step"]) not in scanned
    ]
    plan.kept_rows.extend(foreign)
    plan.foreign_rows = len(foreign)
    return plan
