"""
ParticleVision Segmenter — local desktop UI
=============================================
Runs the confirmed-best pipeline (Branch 1 U-Net + Branch 2 FinesDetector,
Gated Continuous-Prior Fusion, fusion_scale=8.0 / fusion_margin=2.0, no TTA)
on a lab folder (G* › experiment › Before / IN › PV / WT › PV) and writes a
mirrored output folder of segmented .tif files plus summary.csv and run_log.txt.
Folder rules and output writing live in app/lab_structure.py.

Uses the corrected_v1 models (checkpoints/corrected_v1/): Fines IoU 0.4620 /
blob detection 84.1% on the 16 official test images, measured on the unfilled
model output (the hole filling is applied afterwards, to the saved maps).

This file only orchestrates the UI. All modeling logic (tiling, inference,
fusion, color mapping) is imported unmodified from src/.

Launch with: ./run_app.sh   (or: python3 -m streamlit run app/segmenter_app.py)
"""

import html
import os
import sys
import time
from datetime import datetime

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
APP_DIR = os.path.dirname(os.path.abspath(__file__))
if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

import accel_runtime as ar
import holefill_folder as hf
import lab_structure as ls
import model_runtime as mr
import platform_utils as pu

pu.make_stdio_safe()  # printing can never crash the app (Windows consoles / log files)

import platform

import torch
import model_files
from src.config import CFG, PARTICLEVISION_VERSION
from src.postprocess import PARTICLE_SIZES, PipelinePostprocess, describe_info

# ─────────────────────────────────────────────────────────────────────────
# Confirmed-best configuration — defined in model_runtime.py (checkpoints,
# fusion scale 8.0 / margin 2.0, batch 16, no TTA). Do not change without
# re-verifying against the official test-set result.
# ─────────────────────────────────────────────────────────────────────────
BRANCH1_CKPT = mr.BRANCH1_CKPT
BRANCH2_CKPT = mr.BRANCH2_CKPT
FUSION_SCALE = mr.FUSION_SCALE
FUSION_MARGIN = mr.FUSION_MARGIN
CONFIRMED_IOU = 0.4620
CONFIRMED_DETECTION = 84.1
BYTES_PER_GB = 1024**3


def fmt_bytes(b: float) -> str:
    return f"{b / BYTES_PER_GB:.1f} GB" if b >= 0.1 * BYTES_PER_GB else f"{b / 1024**2:.0f} MB"


RUNTIME_ENV = {
    "python": sys.version.split()[0],
    "torch": torch.__version__.split("+")[0],
    "executable": sys.executable,
}
SELFTEST_RESULT = os.path.join(PROJECT_ROOT, "selftest", "last_result.json")
FULLCHECK_RESULT = os.path.join(PROJECT_ROOT, "selftest", "last_full_result.json")


def _result_here(path: str, device_label: str):
    """(result dict or None, made_here). made_here only if the result was produced
    on THIS machine with THIS Python, PyTorch and device."""
    import json

    try:
        with open(path, encoding="utf-8") as f:
            r = json.load(f)
    except (OSError, ValueError):
        return None, False
    m = r.get("machine") or {}
    here = (
        m.get("host") == platform.node()
        and m.get("python_executable") == sys.executable
        and m.get("torch") == torch.__version__
        and m.get("device") == device_label
    )
    return r, here


def selftest_status(device_label: str):
    """(passed_here, text). Green only if check_install.py passed on THIS machine,
    with THIS Python, PyTorch and device — a result copied from another computer
    (e.g. inside a copied project folder) does not count."""
    r, here = _result_here(SELFTEST_RESULT, device_label)
    if r is None:
        return False, "Self-test not run on this machine yet — run check_install.py"
    if not r.get("passed"):
        return False, "Self-test FAILED on this machine — see selftest/check_install_log.txt"
    if not here:
        return False, "Self-test not run on this machine yet — run check_install.py"
    text = f"Self-test passed on this machine ({r.get('time', '')[:10]})"
    full, full_here = _result_here(FULLCHECK_RESULT, device_label)
    if full is not None and full_here:
        text += (
            f" · Full check (3 images) passed ({full.get('time', '')[:10]})"
            if full.get("passed")
            else " · Full check FAILED — see selftest/full_check_log.txt"
        )
    return True, text


DISPLAY_NAMES = {
    "Grains": "Grains",
    "Void": "Air",
    "Water": "Water",
    "Fines": "Microplastic",
}  # what users see
CLASS_HEX = {name: "#%02x%02x%02x" % rgb for name, rgb in CFG.class_colors.items()}
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")

st.set_page_config(page_title="ParticleVision Segmenter", page_icon="🔬", layout="wide")

# Model files missing (e.g. a copy of the public repository): say which, where to get them and where to put them,
# instead of a traceback when the models are loaded. Presence and size only - check_install.py checks the MD5s.
_model_problems = model_files.problems(PROJECT_ROOT, required_only=True, verify_md5=False)
if _model_problems:
    st.error("ParticleVision can't start yet.\n\n" + model_files.problem_message(PROJECT_ROOT, _model_problems, markdown=True))
    st.stop()

# ─────────────────────────────────────────────────────────────────────────
# Styling — light theme, card layout, sidebar nav (base theme set in
# .streamlit/config.toml; this fills in card/log/donut/sidebar specifics)
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* NOTE: do NOT hide header[data-testid="stHeader"] entirely — that also hides
       the sidebar re-open arrow once the sidebar is collapsed, trapping the user
       with no way to bring it back. Only hide the hamburger menu / footer, and
       make the header itself blend into the page instead of disappearing. */
    #MainMenu, footer { visibility: hidden; }
    header[data-testid="stHeader"] { background: transparent; }
    div.block-container { padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1180px; }

    section[data-testid="stSidebar"] {
        width: 240px !important; background: #f7f8fc;
        border-right: 1px solid #e8eaf1;
    }
    section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
    section[data-testid="stSidebar"] div.stButton > button {
        width: 100%; text-align: left; justify-content: flex-start;
        background: transparent; border: none; color: #4b5568;
        font-weight: 500; padding: 0.5rem 0.7rem; border-radius: 8px;
        box-shadow: none;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background: #eef1fb; color: #1f2937;
    }
    section[data-testid="stSidebar"] div.stButton > button[kind="primary"] {
        background: #e8effe; color: #2563eb; font-weight: 700;
        border: none; box-shadow: none;
    }

    .pv-sidebar-title { display:flex; align-items:center; gap:10px; margin: 0 0 26px 4px; }
    .pv-sidebar-logo {
        width: 38px; height: 38px; border-radius: 10px; background: #2563eb;
        display:flex; align-items:center; justify-content:center; flex-shrink:0;
    }
    .pv-sidebar-name { line-height:1.15; }
    .pv-sidebar-name b { font-size: 1.02rem; color:#111827; display:block; }
    .pv-sidebar-name span { font-size: 0.78rem; color:#8a92a3; }

    .pv-status-chip {
        position: fixed; bottom: 22px; width: 205px;
        background: #ffffff; border: 1px solid #e8eaf1; border-radius: 10px;
        padding: 10px 12px; font-size: 0.8rem; color:#4b5568;
    }
    .pv-status-chip .dot { width:8px; height:8px; border-radius:50%; background:#22c55e; display:inline-block; margin-right:6px; }
    .pv-status-chip b { display:block; color:#111827; font-size: 0.83rem; }
    .pv-status-chip .pv-env { margin-top:6px; padding-top:6px; border-top:1px solid #eef0f4;
        font-size:0.74rem; color:#6b7280; line-height:1.35; }
    .pv-status-warn { border-color:#f59e0b; background:#fffbeb; }
    .pv-status-warn .dot { background:#f59e0b; }
    .pv-status-warn .pv-env { color:#b45309; font-weight:600; }

    .pv-card {
        background:#ffffff; border:1px solid #eaecf2; border-radius:14px;
        padding: 22px 24px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    }
    .pv-card h4 { font-size: 1.02rem; font-weight:700; color:#111827; margin:0 0 16px 0; }

    /* Native st.container(key=...) cards — used where widgets/plots need to be
       genuinely nested inside the card (a raw open/close <div> split across
       separate st.markdown() calls does NOT nest later widgets; each call is
       parsed independently, which silently produced empty boxes on Charts). */
    div[class*="st-key-pv_"] {
        background:#ffffff; border:1px solid #eaecf2; border-radius:14px;
        padding: 22px 24px; margin-bottom: 20px; box-shadow: 0 1px 2px rgba(16,24,40,0.03);
    }
    .pv-label { font-weight:600; font-size:0.92rem; color:#1f2937; margin-bottom:2px; }
    .pv-sublabel { font-size:0.8rem; color:#8a92a3; margin-bottom:8px; }
    .pv-check-ok { color:#2563eb; font-size:0.85rem; font-weight:600; margin-top:6px; }
    .pv-check-warn { color:#d97706; font-size:0.85rem; font-weight:600; margin-top:6px; }
    .pv-check-bad { color:#b91c1c; font-size:0.85rem; font-weight:600; margin-top:6px;
        word-break:break-word; }
    .pv-check-info { color:#4b5563; font-size:0.84rem; margin-top:6px; }

    .pv-banner-wrap { border-radius:14px; overflow:hidden; border:1px solid #eaecf2; }
    .st-key-banner_wrap img { border-radius:14px; border:1px solid #eaecf2; }

    .pv-info-banner {
        display:flex; justify-content:space-between; align-items:center;
        background:#f5f8ff; border:1px solid #dbe6fe; border-radius:12px;
        padding:14px 18px; margin-bottom:8px;
    }
    .pv-info-banner .txt b { color:#111827; font-size:0.9rem; }
    .pv-info-banner .txt p { color:#5b6472; font-size:0.83rem; margin:2px 0 0 0; }
    .pv-mps-chip {
        background:#eaf2ff; color:#2563eb; border-radius:10px; padding:8px 14px;
        font-size:0.78rem; text-align:center; white-space:nowrap;
    }
    .pv-mps-chip b { display:block; font-size:0.83rem; }

    .pv-metric-grid { display:flex; gap:14px; margin-top:18px; flex-wrap:wrap; }
    .pv-metric-tile { flex:1; min-width:130px; background:#f8f9fc; border-radius:10px; padding:12px 14px; }
    .pv-metric-tile .lbl { font-size:0.76rem; color:#8a92a3; margin-bottom:4px; }
    .pv-metric-tile .val { font-size:1.25rem; font-weight:700; color:#111827; }

    .pv-log-row { display:flex; justify-content:space-between; padding:5px 0; border-bottom:1px solid #f1f2f6; font-size:0.83rem; }
    .pv-log-row:last-child { border-bottom:none; }
    .pv-log-ts { color:#9aa2b1; font-family:'SF Mono',Menlo,monospace; width:74px; flex-shrink:0; }
    .pv-log-msg { color:#374151; flex:1; }
    .pv-log-status-ok { color:#16a34a; font-weight:600; white-space:nowrap; }
    .pv-log-status-warn { color:#dc2626; font-weight:600; white-space:nowrap; }
    .pv-log-status-info { color:#6b7280; font-size:0.85em; }
    .pv-log-status-run { color:#2563eb; font-weight:600; white-space:nowrap; }
    .pv-log-box { max-height: 300px; overflow-y:auto; }

    .pv-legend-row { display:flex; align-items:center; gap:9px; font-size:0.87rem; color:#374151; margin:9px 0; }
    .pv-legend-row .dot { width:10px; height:10px; border-radius:50%; flex-shrink:0; }
    .pv-legend-row b { margin-left:auto; }

    .pv-empty { text-align:center; color:#8a92a3; padding: 34px 10px; }

    div.stButton > button[kind="primary"] {
        border-radius: 10px; font-weight:700; padding: 0.7em 1em;
    }
    /* Buttons (esp. "Browse" next to a narrow text field) must never wrap their
       label onto a second line — that's what produced the cut-in-half look. */
    div.stButton > button, div.stButton > button p {
        white-space: nowrap !important;
    }
    .pv-output-preview { font-size:0.82rem; color:#5b6472; margin-top:10px; }
    .pv-picked { display:flex; flex-direction:column; gap:2px; background:#f8f9fc;
        border:1px solid #e5e7eb; border-radius:10px; padding:7px 12px; min-height:42px; }
    .pv-picked b { font-size:0.95rem; color:#111827; }
    .pv-picked span { font-family:'SF Mono',Menlo,monospace; font-size:0.76rem; color:#6b7280;
        word-break:break-all; user-select:text; }
    .pv-picked-empty { color:#8a92a3; justify-content:center; }
    .pv-scan-head { font-weight:700; color:#111827; font-size:0.95rem; margin:18px 0 2px 0; }
    .pv-g-line { font-size:0.84rem; color:#374151; margin:10px 0 6px 0; }
    .pv-output-preview code { background:#f1f3f8; padding:1px 6px; border-radius:5px; color:#374151; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────
# Cached model / device loading — happens once per server process
# ─────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading models...")
def _normal_pipeline() -> "mr.SegmentationRunner":
    runner = mr.SegmentationRunner()
    print(
        f"[ParticleVision] Python {RUNTIME_ENV['python']} ({RUNTIME_ENV['executable']}), "
        f"torch {RUNTIME_ENV['torch']} — running on: {runner.label} — "
        f"{selftest_status(runner.label)[1]}",
        flush=True,
    )
    return runner


@st.cache_resource(show_spinner="Loading the faster mode...")
def _accel_pipeline(result_stamp: float):
    """(AccelRunner or None, note) — a backend only if benchmark_accel verified it on this machine.
    Keyed on accel_result.json's modification time, so a new benchmark result is picked up."""
    runner, note = ar.select_runner()
    print(f"[ParticleVision] faster mode: {runner.label if runner else note}", flush=True)
    return runner, note


def _accel_stamp() -> float:
    try:
        return os.path.getmtime(ar.RESULT_FILE)
    except OSError:
        return 0.0


def faster_mode_wanted() -> bool:
    """The sidebar switch (on by default) and PV_FORCE_NORMAL=1 turn the faster mode off."""
    return bool(st.session_state.get("use_faster_mode", True)) and not ar.force_normal_env()


def load_pipeline():
    """The runner for this session. Without accel_result.json (every normal install) this is
    always the normal path, model_runtime.SegmentationRunner, exactly as before."""
    if os.path.isfile(ar.RESULT_FILE) and faster_mode_wanted():
        runner, _ = _accel_pipeline(_accel_stamp())
        if runner is not None:
            return runner
    return _normal_pipeline()


def normal_device_label() -> str:
    """The device of the normal path — what the self-test result is tied to."""
    return mr.describe_device(mr.select_device())


def pick_folder(
    initial_dir: str, prompt: str = "Select the lab folder (the one containing G1, G2, …)"
) -> str:
    """Native folder picker for this OS (see platform_utils). "" = cancelled.
    If no dialog could be shown at all, the reason is kept for the page to show."""
    st.session_state.picker_error = ""
    try:
        return pu.pick_folder(initial_dir, prompt)
    except pu.PickerError as e:
        st.session_state.picker_error = str(e)
        return ""


def _open_folder(path: str):
    """on_click callback: open a folder in Explorer / Finder / file manager."""
    st.session_state.open_error = pu.open_folder(path) or ""


def stats_key(runner) -> str:
    """Which kind of run a measured speed belongs to ("mps", "cpu", "accel:dml:1", …)."""
    return getattr(runner, "stats_key", runner.device.type)


def expected_seconds(runner) -> tuple:
    """(seconds per image, where the number comes from) for the pre-run estimate."""
    stats = st.session_state.last_run_stats
    if stats and stats.get("avg") and stats.get("device") == stats_key(runner):
        return stats["avg"], "measured in your last run"
    if getattr(runner, "expected_sec", None):
        return runner.expected_sec, "measured by benchmark_accel on this computer"
    t = runner.device.type
    return mr.EXPECTED_SEC_PER_IMAGE[t], mr.EXPECTED_SEC_NOTE[t]


def _scan_into_state(path: str):
    path = os.path.normpath(path)
    st.session_state.input_dir = path
    scan = ls.scan_lab_folder(path)
    st.session_state.scan = scan
    st.session_state.scan_bytes = ls.estimate_output_bytes(scan.images)
    st.session_state.scan_confirmed = False
    st.session_state.run_mode = "new"
    st.session_state.resume_dir = None
    st.session_state.resume_plan = None


def _browse_and_scan():
    """
    on_click callback for Browse (callbacks run before the rerun, so the new
    folder and its scan are in session_state when the page draws). Cancel = no change.
    """
    start = st.session_state.input_dir
    if start and os.path.isdir(start):
        start = os.path.dirname(start)  # open next to the previous choice
    picked = pick_folder(start or os.path.expanduser("~"))
    if picked:
        _scan_into_state(picked)


def _rescan():
    if st.session_state.input_dir and os.path.isdir(st.session_state.input_dir):
        _scan_into_state(st.session_state.input_dir)


def resolve_output_root(lab_root: str):
    """(output folder, fell_back_to_desktop). Normally a sibling of the lab folder;
    the Desktop only if that location can't be written (e.g. a read-only drive).
    Writability is tested by creating a file — os.access is unreliable on Windows."""
    out = ls.make_output_root(lab_root)
    if pu.dir_is_writable(os.path.dirname(out)):
        return out, False
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    base = desktop if os.path.isdir(desktop) else os.path.expanduser("~")
    return os.path.join(base, os.path.basename(out)), True


def fmt_duration(s: float) -> str:
    s = max(0, int(round(s)))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h} h {m:02d} min"
    return f"{m} min {sec:02d} s" if m else f"{sec} s"


def fmt_mmss(s: float) -> str:
    s = max(0, int(round(s)))
    m, sec = divmod(s, 60)
    return f"{m:02d}:{sec:02d}"


def summary_donut_svg(completed, processing, pending, failed, total, size=140) -> str:
    r, cx, cy = 52, size / 2, size / 2
    circumference = 2 * 3.14159265 * r
    segments = [
        (completed, "#22c55e"),
        (processing, "#2563eb"),
        (failed, "#ef4444"),
        (pending, "#d9dce3"),
    ]
    circles = []
    offset = 0.0
    for value, color in segments:
        frac = (value / total) if total else 0
        length = frac * circumference
        if length > 0:
            circles.append(
                f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="{color}" stroke-width="14" '
                f'stroke-dasharray="{length:.2f} {circumference:.2f}" '
                f'stroke-dashoffset="{-offset:.2f}" transform="rotate(-90 {cx} {cy})"/>'
            )
        offset += length
    circles_html = "".join(circles)
    return f"""
    <svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">
      <circle cx="{cx}" cy="{cy}" r="{r}" fill="none" stroke="#eef0f5" stroke-width="14"/>
      {circles_html}
      <text x="{cx}" y="{cy - 2}" text-anchor="middle" font-size="28" font-weight="800"
            fill="#111827" font-family="sans-serif">{completed}</text>
      <text x="{cx}" y="{cy + 20}" text-anchor="middle" font-size="12" fill="#8a92a3"
            font-family="sans-serif">of {total}</text>
    </svg>
    """


def icon_svg(name: str, size: int = 14, color: str = "currentColor") -> str:
    """Small inline stroke-style line icon (Feather/Lucide-like), used inside raw
    HTML strings — deliberately not emoji, to match the reference UI's icon style.
    (st.button has a native `icon=":material/x:"` param used separately for buttons;
    this helper covers everywhere else: checkmarks, metric tiles, log status, etc.)"""
    paths = {
        "check": '<circle cx="12" cy="12" r="9"/><path d="M8.5 12.5l2.3 2.3L16 9.5"/>',
        "warn": (
            '<path d="M10.3 3.9 1.9 18a2 2 0 0 0 1.7 3h16.8a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/><circle cx="12" cy="16.5" r="0.6" fill="{c}" stroke="none"/>'
        ),
        "info": (
            '<circle cx="12" cy="12" r="9"/><line x1="12" y1="11" x2="12" y2="16"/>'
            '<circle cx="12" cy="7.7" r="0.6" fill="{c}" stroke="none"/>'
        ),
        "clock": '<circle cx="12" cy="12" r="9"/><polyline points="12 7 12 12 15.5 14"/>',
        "hourglass": '<path d="M6 3h12M6 21h12M7 3c0 6 5 6.5 5 9s-5 3-5 9M17 3c0 6-5 6.5-5 9s5 3 5 9"/>',
        "zap": '<polygon points="13 2 4 14 12 14 11 22 20 10 12 10 13 2"/>',
        "x-circle": (
            '<circle cx="12" cy="12" r="9"/><line x1="15" y1="9" x2="9" y2="15"/>'
            '<line x1="9" y1="9" x2="15" y2="15"/>'
        ),
        "arrow-right": '<line x1="4" y1="12" x2="19" y2="12"/><polyline points="13 6 19 12 13 18"/>',
    }
    body = paths[name].replace("{c}", color)
    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="2" stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-2px;display:inline-block;">{body}</svg>'
    )


def render_log_html(entries: list, max_rows: int = None) -> str:
    rows = entries if max_rows is None else entries[-max_rows:]
    html = '<div class="pv-log-box">'
    for ts, msg, status_html in rows:
        html += (
            f'<div class="pv-log-row"><span class="pv-log-ts">{ts}</span>'
            f'<span class="pv-log-msg">{msg}</span>{status_html}</div>'
        )
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────────────────────────
defaults = {
    "page": "dashboard",
    "input_dir": "",
    "scan": None,
    "last_output_dir": None,
    "last_batch": None,
    "log_entries": [],
    "results": [],
    "current_thumb": None,
    "last_run_stats": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f"""
        <div class="pv-sidebar-title">
          <div class="pv-sidebar-logo">
            <svg width="20" height="20" viewBox="0 0 20 20">
              <circle cx="6.5" cy="7" r="3.4" fill="none" stroke="white" stroke-width="1.4"/>
              <circle cx="13.5" cy="6.2" r="2" fill="none" stroke="white" stroke-width="1.4"/>
              <circle cx="12.5" cy="13.2" r="2.6" fill="none" stroke="white" stroke-width="1.4"/>
              <circle cx="5.5" cy="14" r="1.3" fill="white"/>
            </svg>
          </div>
          <div class="pv-sidebar-name"><b>ParticleVision</b><span>Segmenter · version {PARTICLEVISION_VERSION}</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("dashboard", "Dashboard", ":material/home:"),
        ("run", "Run Pipeline", ":material/play_arrow:"),
        ("results", "Results", ":material/folder:"),
        ("charts", "Charts", ":material/bar_chart:"),
        ("logs", "Logs", ":material/list_alt:"),
        ("fixholes", "Fix Segmentations", ":material/healing:"),
    ]
    for key, label, mat_icon in nav_items:
        is_active = st.session_state.page == key
        if st.button(
            label, key=f"nav_{key}", icon=mat_icon, type="primary" if is_active else "secondary"
        ):
            st.session_state.page = key
            st.rerun()

    try:
        raw_label = load_pipeline().label
        device_label = html.escape(raw_label)
        test_ok, env_note = selftest_status(normal_device_label())
    except Exception as e:  # noqa: BLE001
        device_label = f"Model failed to load: {html.escape(type(e).__name__)}"
        test_ok, env_note = False, "See the terminal window for the error"
    env_line = f"Python {RUNTIME_ENV['python']} · torch {RUNTIME_ENV['torch']}"
    st.markdown(
        f'<div class="pv-status-chip{"" if test_ok else " pv-status-warn"}" '
        f'title="{html.escape(RUNTIME_ENV["executable"])}">'
        f'<span class="dot"></span>Running on<b>{device_label}</b>'
        f'<div class="pv-env">{env_line}<br>{html.escape(env_note)}</div></div>',
        unsafe_allow_html=True,
    )
    if os.path.isfile(ar.RESULT_FILE):  # only after benchmark_accel has run
        st.toggle(
            "Faster mode",
            key="use_faster_mode",
            value=True,
            help="Off = always the normal path. The faster mode is only used if benchmark_accel "
            "verified it on this computer.",
        )
        note = ""
        if ar.force_normal_env():
            note = "Faster mode off (PV_FORCE_NORMAL=1) — normal path"
        elif faster_mode_wanted():
            note = _accel_pipeline(_accel_stamp())[1]
        if note:
            st.caption(note)


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Dashboard
# ─────────────────────────────────────────────────────────────────────────
def page_dashboard():
    st.markdown("## Dashboard")
    st.caption("Overview of your ParticleVision Segmenter setup")

    stats = st.session_state.last_run_stats
    if stats is None:
        st.markdown(
            '<div class="pv-card"><div class="pv-empty">No pipeline runs yet.<br>'
            "Head to <b>Run Pipeline</b> to segment your first batch of images.</div></div>",
            unsafe_allow_html=True,
        )
        if st.button("Go to Run Pipeline", icon=":material/play_arrow:", type="primary"):
            st.session_state.page = "run"
            st.rerun()
    else:
        c1, c2, c3, c4 = st.columns(4)
        for col, lbl, val in zip(
            [c1, c2, c3, c4],
            ["Last batch", "Succeeded", "Failed", "Avg time / image"],
            [f"{stats['total']} images", stats["done"], stats["failed"], f"{stats['avg']:.1f}s"],
        ):
            with col:
                st.markdown(
                    f'<div class="pv-card" style="text-align:center;">'
                    f'<div class="pv-sublabel">{lbl}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:#111827;">{val}</div></div>',
                    unsafe_allow_html=True,
                )
        st.markdown("<br>", unsafe_allow_html=True)
        cc1, cc2 = st.columns(2)
        with cc1:
            if st.button("Run another batch", icon=":material/play_arrow:", type="primary"):
                st.session_state.page = "run"
                st.rerun()
        with cc2:
            if st.button("View last results", icon=":material/folder:"):
                st.session_state.page = "results"
                st.rerun()

    st.markdown(
        f'<div class="pv-card" style="margin-top:8px;">'
        f'<div class="pv-sublabel">Confirmed deployed configuration</div>'
        f'<div style="font-size:0.9rem;color:#374151;">Branch 1 (U-Net + EfficientNet-B3) + Branch 2 '
        f"(dedicated microplastic detector), Gated Continuous-Prior Fusion — Microplastic IoU "
        f"<b>{CONFIRMED_IOU}</b>, blob detection <b>{CONFIRMED_DETECTION}%</b> on the official test set.</div>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Results
# ─────────────────────────────────────────────────────────────────────────
def page_results():
    st.markdown("## Results")
    st.caption("Outputs from your most recent pipeline run")

    if not st.session_state.results:
        st.markdown(
            '<div class="pv-card"><div class="pv-empty">No results yet. Run the pipeline first.</div></div>',
            unsafe_allow_html=True,
        )
        return

    if st.session_state.last_output_dir:
        st.button(
            "Open output folder",
            icon=":material/folder_open:",
            key="open_results_dir",
            on_click=_open_folder,
            args=(st.session_state.last_output_dir,),
        )
        if st.session_state.get("open_error"):
            st.warning(st.session_state.open_error)

    rows = []
    for r in st.session_state.results:
        rows.append(
            {
                "File": r["name"],
                "Status": "Done" if r["status"] == "ok" else "Failed",
                "Time (s)": f"{r['elapsed']:.1f}" if r.get("elapsed") is not None else "—",
                "Output file": r.get("out_name", "—")
                if r["status"] == "ok"
                else r.get("error", ""),
            }
        )
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Charts
# ─────────────────────────────────────────────────────────────────────────
def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#d1d5db")
    ax.spines["bottom"].set_color("#d1d5db")
    ax.tick_params(colors="#374151", labelsize=9)
    ax.yaxis.label.set_color("#374151")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)


def _charts_browse():
    """on_click: pick a previous run's output folder (must contain summary.csv)."""
    start = st.session_state.get("charts_dir") or st.session_state.last_output_dir or ""
    picked = pick_folder(
        os.path.dirname(start) if start else os.path.expanduser("~"),
        "Select a results folder (…_segmented_<timestamp>)",
    )
    if not picked:
        st.session_state.charts_msg = st.session_state.get("picker_error", "")
        return
    picked = os.path.normpath(picked)
    if os.path.isfile(os.path.join(picked, "summary.csv")):
        st.session_state.charts_dir = picked
        st.session_state.charts_msg = ""
    else:
        st.session_state.charts_msg = (
            f"'{os.path.basename(picked)}' has no summary.csv — pick an "
            f"output folder named …_segmented_<timestamp>."
        )


def page_charts():
    st.markdown("## Charts")
    st.caption("Built only from summary.csv — the same file as the run report")

    # Which run: the one picked here, else the most recent run in this session.
    src = st.session_state.get("charts_dir") or st.session_state.last_output_dir
    has_csv = bool(src) and os.path.isfile(os.path.join(src, "summary.csv"))

    with st.container(key="pv_charts_src_card"):
        st.markdown(
            '<div class="pv-label">Results folder</div>'
            '<div class="pv-sublabel">Latest run by default — or open any earlier '
            "…_segmented_&lt;timestamp&gt; folder</div>",
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns([6, 1.1])
        with c1:
            if has_csv:
                st.markdown(
                    f'<div class="pv-picked"><b>{html.escape(os.path.basename(src))}</b>'
                    f"<span>{html.escape(os.path.join(src, 'summary.csv'))}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pv-picked pv-picked-empty">No results yet — run the '
                    "pipeline, or open a results folder</div>",
                    unsafe_allow_html=True,
                )
        with c2:
            st.button(
                "Open",
                key="charts_browse",
                icon=":material/folder_open:",
                width="stretch",
                on_click=_charts_browse,
            )
        if st.session_state.get("charts_msg"):
            st.markdown(
                f'<div class="pv-check-warn">{icon_svg("warn", 13)} '
                f"{html.escape(st.session_state.charts_msg)}</div>",
                unsafe_allow_html=True,
            )
    if not has_csv:
        return

    rows = ls.read_summary(src)
    ok = [r for r in rows if r["status"] == "ok"]
    n_failed = len(rows) - len(ok)
    if not ok:
        st.markdown(
            '<div class="pv-card"><div class="pv-empty">This run has no successfully '
            "processed images to chart.</div></div>",
            unsafe_allow_html=True,
        )
        return

    # ── Filters: G → experiment → phase → PV ─────────────────────────────
    with st.container(key="pv_filters_card"):
        st.markdown("#### Filters")
        if n_failed:
            st.caption(
                f"{n_failed} failed image{'s' if n_failed != 1 else ''} in this run "
                f"{'are' if n_failed != 1 else 'is'} not charted (see run_log.txt)."
            )
        f1, f2, f3, f4 = st.columns([0.7, 2.6, 0.9, 1])
        gs = sorted({r["G"] for r in ok}, key=ls.natural_key)
        with f1:
            g = st.selectbox("G folder", gs, key="cf_g")
        exps = sorted({r["experiment"] for r in ok if r["G"] == g}, key=ls.natural_key)
        with f2:
            exp = st.selectbox("Experiment", exps, key="cf_exp")
        seq = ls.experiment_sequence(ok, g, exp)
        phases = [p for p in ls.PHASES if any(r["phase"] == p for r in seq)]
        with f3:
            phase = st.selectbox("Phase", phases, key="cf_phase")
        pvs = [r["pv_step"] for r in seq if r["phase"] == phase and r["pv_step"]]
        with f4:
            pv = st.selectbox("PV step", pvs if pvs else ["—"], key="cf_pv", disabled=not pvs)
        unit = st.radio(
            "Units", ["Pixel count", "% of image area"], horizontal=True, key="charts_unit"
        )
    use_pct = unit == "% of image area"
    suffix = "pct" if use_pct else "px"
    y_label = "% of image area" if use_pct else "Pixel count"

    row = next(
        r for r in seq if r["phase"] == phase and (r["pv_step"] or "—") == (pv if pvs else "—")
    )

    # ── Per-image class breakdown ────────────────────────────────────────
    with st.container(key="pv_chart_a_card"):
        st.markdown("#### Per-Image Class Breakdown")
        st.caption(" › ".join([g, exp, phase] + ([pv] if pvs else [])))
        fig, ax = plt.subplots(figsize=(6, 3.6))
        disp = [d for _, d in ls.CLASS_DISPLAY]
        ax.bar(
            disp,
            [float(row[f"{d.lower()}_{suffix}"]) for d in disp],
            color=[CLASS_HEX[c] for c, _ in ls.CLASS_DISPLAY],
            edgecolor="#374151",
            linewidth=0.6,
            width=0.6,
        )
        ax.set_ylabel(y_label)
        _style_axes(ax)
        fig.tight_layout()
        st.pyplot(fig, clear_figure=True, width="content")
        st.markdown(
            '<div class="pv-sublabel" style="margin-top:10px;">Exact values for this image '
            "(from summary.csv)</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Class": d,
                        "Pixel count": f"{int(row[f'{d.lower()}_px']):,}",
                        "% of image": f"{float(row[f'{d.lower()}_pct']):.3f}%",
                    }
                    for d in disp
                ]
            ),
            width="stretch",
            hide_index=True,
        )

    # ── Fines across the experiment's physical sequence ──────────────────
    with st.container(key="pv_chart_b_card"):
        st.markdown("#### Microplastic Across the Experiment Sequence")
        st.caption(f"{exp} — Before → IN PV↑ → WT PV↑")
        labels = [ls.row_step_label(r) for r in seq]
        vals = [float(r[f"fines_{suffix}"]) for r in seq]
        n = len(seq)
        fig2, ax2 = plt.subplots(figsize=(min(max(6.0, 0.55 * n + 1.5), 22.0), 3.6), dpi=130)
        ax2.plot(range(n), vals, marker="o", color=CLASS_HEX["Fines"], linewidth=1.8, markersize=5)
        for i in range(1, n):  # faint divider where the phase changes
            if seq[i]["phase"] != seq[i - 1]["phase"]:
                ax2.axvline(i - 0.5, color="#9ca3af", linewidth=0.8, linestyle="--", alpha=0.7)
        sel_i = seq.index(row)
        ax2.plot(
            [sel_i],
            [vals[sel_i]],
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor="#111827",
            markeredgewidth=1.4,
        )
        ax2.set_xticks(range(n))
        ax2.set_xticklabels(
            labels,
            rotation=45 if n <= 15 else 60,
            ha="right",
            fontsize=9 if n <= 12 else (8 if n <= 20 else 7),
        )
        ax2.set_ylabel(f"Microplastic — {y_label}")
        _style_axes(ax2)
        fig2.subplots_adjust(bottom=0.34, left=0.08, right=0.98, top=0.95)
        st.pyplot(fig2, clear_figure=True, width="content")
        st.markdown(
            '<div class="pv-sublabel" style="margin-top:10px;">Exact microplastic values '
            "(from summary.csv)</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Step": ls.row_step_label(r),
                        "Microplastic pixel count": f"{int(r['fines_px']):,}",
                        "Microplastic % of image": f"{float(r['fines_pct']):.4f}%",
                    }
                    for r in seq
                ]
            ),
            width="stretch",
            hide_index=True,
        )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Logs
# ─────────────────────────────────────────────────────────────────────────
def page_logs():
    st.markdown("## Logs")
    st.caption("Full run log")
    with st.container(key="pv_logs_card"):
        if not st.session_state.log_entries:
            st.markdown('<div class="pv-empty">No log entries yet.</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                render_log_html(st.session_state.log_entries, max_rows=None).replace(
                    "max-height: 300px", "max-height: 520px"
                ),
                unsafe_allow_html=True,
            )


def pending_count(scan) -> int:
    """Images the next run will process: all (new run) or the not-yet-done ones (resume)."""
    if scan is None:
        return 0
    if st.session_state.get("run_mode") == "resume":
        plan = st.session_state.get("resume_plan")
        return len(plan.todo) if plan is not None else 0
    return len(scan.images)


def _browse_resume():
    """on_click: pick the output folder of an interrupted run and check what's done."""
    scan = st.session_state.scan
    st.session_state.resume_msg = ""
    picked = pick_folder(
        os.path.dirname(scan.root),
        "Select the output folder of the interrupted run (…_segmented_<timestamp>)",
    )
    if not picked:
        st.session_state.resume_msg = st.session_state.get("picker_error", "")
        return
    picked = os.path.normpath(picked)
    if not os.path.isfile(os.path.join(picked, "summary.csv")):
        st.session_state.resume_msg = (
            f"'{os.path.basename(picked)}' has no summary.csv — choose the "
            f"output folder of the interrupted run."
        )
        return
    try:
        plan = ls.plan_resume(scan, picked)
    except ls.OldLayoutError as e:
        st.session_state.resume_dir = None
        st.session_state.resume_plan = None
        st.session_state.resume_msg = str(e)
        return
    st.session_state.resume_dir = picked
    st.session_state.resume_plan = plan
    st.session_state.scan_confirmed = False


def _run_log_input(output_root: str) -> str:
    """The 'Input:' lab folder recorded in an output folder's run_log.txt."""
    try:
        with open(
            os.path.join(output_root, "run_log.txt"), encoding="utf-8", errors="replace"
        ) as f:
            for line in f:
                if line.startswith("Input:"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return ""


def render_resume_choice(scan):
    c1, c2 = st.columns([6, 1.3])
    rdir = st.session_state.get("resume_dir")
    with c1:
        if rdir:
            st.markdown(
                f'<div class="pv-picked"><b>{html.escape(os.path.basename(rdir))}</b>'
                f"<span>{html.escape(rdir)}</span></div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pv-picked pv-picked-empty">Choose the output folder of the '
                "interrupted run</div>",
                unsafe_allow_html=True,
            )
    with c2:
        st.button(
            "Choose",
            key="browse_resume",
            icon=":material/folder_open:",
            width="stretch",
            on_click=_browse_resume,
        )
    if st.session_state.get("resume_msg"):
        st.markdown(
            f'<div class="pv-check-warn">{icon_svg("warn", 13)} '
            f"{html.escape(st.session_state.resume_msg)}</div>",
            unsafe_allow_html=True,
        )
    plan = st.session_state.get("resume_plan")
    if not rdir or plan is None:
        return
    lab_name = os.path.basename(os.path.normpath(scan.root))
    if not os.path.basename(rdir).startswith(lab_name + "_segmented_"):
        st.markdown(
            f"<div class=\"pv-check-warn\">{icon_svg('warn', 13)} This folder's name doesn't start "
            f'with "{html.escape(lab_name)}_segmented_" — make sure it belongs to this lab folder.</div>',
            unsafe_allow_html=True,
        )
    logged = _run_log_input(rdir)
    if logged and os.path.normcase(os.path.normpath(logged)) != os.path.normcase(scan.root):
        st.markdown(
            f'<div class="pv-check-warn">{icon_svg("warn", 13)} Its run_log.txt says it was made '
            f"from a different lab folder: {html.escape(logged)}</div>",
            unsafe_allow_html=True,
        )
    st.markdown(
        f'<div class="pv-check-info"><b>{len(plan.done)}</b> image(s) already done — they will be '
        f"<b>skipped</b>. <b>{len(plan.todo)}</b> image(s) will be processed into this same folder; "
        f"summary.csv and run_log.txt are continued, and run_log.txt records that the run was resumed.</div>",
        unsafe_allow_html=True,
    )
    if plan.redo_reasons:
        with st.expander(f"{len(plan.redo_reasons)} existing output(s) will be redone — why"):
            st.dataframe(
                pd.DataFrame([{"Image": k, "Reason": v} for k, v in plan.redo_reasons.items()]),
                width="stretch",
                hide_index=True,
            )
    if not plan.todo:
        st.markdown(
            f'<div class="pv-check-info">{icon_svg("check", 13)} Every image in this lab folder is '
            f"already done in that output folder — nothing left to process.</div>",
            unsafe_allow_html=True,
        )


def render_preflight(scan, runner, sec):
    """Checks before a run can start. Sets st.session_state.preflight_block when a
    run must not start (it would fail part-way)."""
    block = False
    resume = st.session_state.get("run_mode") == "resume"
    out_root = st.session_state.get("resume_dir") if resume else resolve_output_root(scan.root)[0]
    todo_n = pending_count(scan)
    lines = []

    if out_root:
        for msg in ls.path_length_problems(scan, out_root, pu.max_path_length()):
            block = True
            lines.append(
                f'<div class="pv-check-bad">{icon_svg("x-circle", 13)} Paths too long for Windows. '
                f"{html.escape(msg)}</div>"
            )

        n = max(len(scan.images), 1)
        need = int(st.session_state.get("scan_bytes", 0) * todo_n / n)
        free = pu.free_bytes(out_root)
        if free is not None and todo_n:
            if free < need * 1.05:
                lines.append(
                    f'<div class="pv-check-bad">{icon_svg("warn", 13)} Not enough free disk space: '
                    f"this run writes about {fmt_bytes(need)} (≈44 MB per image) but only "
                    f"{fmt_bytes(free)} is free where the output goes. Free up space or move "
                    f"the lab folder to a bigger drive, otherwise images will fail once the disk is full.</div>"
                )
            else:
                lines.append(
                    f'<div class="pv-check-info">Disk space: this run writes about '
                    f"{fmt_bytes(need)}; {fmt_bytes(free)} free.</div>"
                )

    if todo_n and getattr(runner, "on_cpu", runner.device.type == "cpu"):
        lines.append(
            f'<div class="pv-check-warn">{icon_svg("warn", 13)} No GPU found — running on the CPU. '
            f"Expect about {sec:.0f} s per image, roughly {fmt_duration(todo_n * sec)} for "
            f"{todo_n} images. The exact speed is measured after the first image.</div>"
        )
    if todo_n and pu.on_battery():
        lines.append(
            f'<div class="pv-check-warn">{icon_svg("warn", 13)} This computer is running on battery. '
            f"Plug it in before a long run. The app keeps the computer awake during a run, but "
            f"closing the lid will still put a laptop to sleep.</div>"
        )

    if resume and out_root and st.session_state.get("resume_plan") is not None:
        try:
            ls.check_resume_settings(out_root, _pp_settings())
        except ls.SettingsMismatchError as e:
            block = True
            lines.append(
                f'<div class="pv-check-bad">{icon_svg("x-circle", 13)} {html.escape(str(e))}</div>'
            )

    if lines:
        st.markdown("".join(lines), unsafe_allow_html=True)
    st.session_state.preflight_block = block


def render_scan_preview(scan):
    """Scan summary shown right after a folder is chosen, before anything runs."""
    n = len(scan.images)
    rows = scan.experiment_rows()
    runner = load_pipeline()
    sec, basis = expected_seconds(runner)
    todo_n = pending_count(scan)
    basis = f"{html.escape(runner.label)} · ~{sec:.0f} s / image ({basis})"

    st.markdown(
        f'<div class="pv-scan-head">{icon_svg("check" if n else "warn", 14)} Scan result</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""<div class="pv-metric-grid">
          <div class="pv-metric-tile"><div class="lbl">Images to segment</div><div class="val">{n}</div></div>
          <div class="pv-metric-tile"><div class="lbl">G folders</div><div class="val">{len({r["G"] for r in rows})}</div></div>
          <div class="pv-metric-tile"><div class="lbl">Experiments</div><div class="val">{len(rows)}</div></div>
          <div class="pv-metric-tile"><div class="lbl">{icon_svg("hourglass", 12)} Estimated run time</div>
               <div class="val">{fmt_duration(todo_n * sec)}</div><div class="lbl">{basis}</div></div>
        </div>""",
        unsafe_allow_html=True,
    )

    if n == 0:
        st.markdown(
            f'<div class="pv-check-warn">{icon_svg("warn", 13)} No images matching the lab '
            f"structure were found. Check that you picked the top-level folder (the one "
            f"containing G1, G2, …) — see the skipped list below for why each folder was ignored.</div>",
            unsafe_allow_html=True,
        )
    else:
        per_g = {}
        for r in rows:
            per_g[r["G"]] = per_g.get(r["G"], 0) + r["Total"]
        st.markdown(
            '<div class="pv-g-line">'
            + " &nbsp;·&nbsp; ".join(
                f"<b>{html.escape(g)}</b>: {c} image{'s' if c != 1 else ''}"
                for g, c in per_g.items()
            )
            + "</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            pd.DataFrame(rows),
            width="stretch",
            hide_index=True,
            height=min(38 + 35 * len(rows), 360),
        )

    if scan.warnings:
        with st.expander(f"Warnings ({len(scan.warnings)})", expanded=False):
            for w in scan.warnings:
                st.markdown(f"- {html.escape(w)}")

    folders = scan.skipped_folders
    with st.expander(f"Skipped folders ({len(folders)})", expanded=0 < len(folders) <= 12):
        if folders:
            st.dataframe(
                pd.DataFrame([{"Folder": s.rel_path, "Reason": s.reason} for s in folders]),
                width="stretch",
                hide_index=True,
            )
        else:
            st.caption("None")
    files = scan.skipped_files
    with st.expander(f"Ignored files ({len(files)}) — everything that is not the target image"):
        if files:
            st.dataframe(
                pd.DataFrame([{"File": s.rel_path, "Reason": s.reason} for s in files]),
                width="stretch",
                hide_index=True,
                height=min(38 + 35 * len(files), 360),
            )
        else:
            st.caption("None")

    if n:
        mode = st.radio(
            "Run mode",
            ["new", "resume"],
            key="run_mode",
            horizontal=True,
            format_func=lambda m: {
                "new": "Start a new run",
                "resume": "Continue a previous (interrupted) run",
            }[m],
        )
        if mode == "resume":
            render_resume_choice(scan)
        else:
            out_name = os.path.basename(os.path.normpath(scan.root)) + "_segmented_<timestamp>"
            _, fallback = resolve_output_root(scan.root)
            where = "on your <b>Desktop</b>" if fallback else "next to the lab folder"
            st.markdown(
                f'<div class="pv-output-preview">Output goes to a new folder <code>{html.escape(out_name)}</code> '
                f"{where}, mirroring its G › experiment › phase › PV folders: every result is saved as "
                f"<code>{ls.OUTPUT_FILENAME}</code> in <code>Before</code> or in its PV folder "
                f"(e.g. <code>IN/PV 03/{ls.OUTPUT_FILENAME}</code>). Plus <code>summary.csv</code> and "
                f"<code>run_log.txt</code>. Nothing inside the lab folder is changed, and no existing file is "
                f"overwritten.</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<div class="pv-check-warn">{icon_svg("warn", 13)} The results have the same name as the '
                f"lab's manual segmentations (<code>{ls.OUTPUT_FILENAME[:-4]}</code>). Don't copy them into "
                f"the original lab folders, or you may overwrite manual segmentations.</div>",
                unsafe_allow_html=True,
            )
            if fallback:
                st.markdown(
                    f'<div class="pv-check-warn">{icon_svg("warn", 13)} The folder that contains the lab '
                    f"folder can't be written to (for example a read-only drive), so the output will be "
                    f"saved on your Desktop instead.</div>",
                    unsafe_allow_html=True,
                )
        render_preflight(scan, runner, sec)
    else:
        st.session_state.preflight_block = True
    c1, c2 = st.columns([4, 1.1])
    with c1:
        todo_label = pending_count(scan)
        st.checkbox(
            f"I've checked this scan — process {'these' if todo_label == n else 'the remaining'} "
            f"{todo_label} images",
            key="scan_confirmed",
            disabled=(todo_label == 0),
        )
    with c2:
        st.button(
            "Rescan", key="rescan_btn", icon=":material/refresh:", width="stretch", on_click=_rescan
        )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Run Pipeline (the main working screen)
# ─────────────────────────────────────────────────────────────────────────
PP_SIZE_LABELS = {"auto": "Auto (from the folder name)", 5: "5 µm", 7: "7 µm", 10: "10 µm"}


def _pp_settings() -> PipelinePostprocess:
    """The post-processing settings of the next run, from the Run page → Advanced (defaults: src/config.py)."""
    size = st.session_state.get("pp_size", "auto")
    return PipelinePostprocess(
        enabled=st.session_state.get("pp_enabled", CFG.postprocess_in_pipeline),
        particle_size=None if size == "auto" else int(size),
        save_raw=st.session_state.get("pp_save_raw", CFG.postprocess_save_raw),
    )


def render_advanced():
    """Run page → Advanced: the automatic hole filling (always rendered, so the choices are kept)."""
    with st.expander("Advanced", expanded=False):
        st.checkbox(
            "Fill hollow microplastic particles",
            key="pp_enabled",
            value=CFG.postprocess_in_pipeline,
            help="On (recommended): after segmentation, ring-shaped microplastic particles are filled - "
            "7 and 10 µm particles are redrawn as filled circles, 5 µm particles are closed. "
            "Off: the model's map is saved exactly as it is.",
        )
        st.selectbox(
            "Particle size",
            ["auto", *PARTICLE_SIZES],
            key="pp_size",
            format_func=PP_SIZE_LABELS.get,
            help="Auto reads D05 / D07 / D10 from each experiment folder name (no size found: closing "
            "r = 2). A chosen size is used for every image of the run.",
        )
        st.checkbox(
            "Also save the raw (unfilled) segmentation",
            key="pp_save_raw",
            value=CFG.postprocess_save_raw,
            help="For research: also saves Fused_St_Filt_Comp_Seg_Raw.tif (the map before filling) next "
            "to each result. The normal result keeps its name, Fused_St_Filt_Comp_Seg_Mod.tif.",
        )
        st.caption(
            "Settings of the next run: " + _pp_settings().describe() + ". They are recorded in "
            "run_log.txt, summary.csv and run_settings.json; an interrupted run can only be continued "
            "with the same settings."
        )


def page_run():
    hcol1, hcol2 = st.columns([2.2, 1])
    with hcol1:
        st.markdown("# Run Segmentation Pipeline")
        st.caption("Process raw images and generate segmented outputs")
    with hcol2:
        banner_path = os.path.join(ASSETS_DIR, "banner.png")
        if os.path.exists(banner_path):
            with st.container(key="banner_wrap"):
                st.image(banner_path, width="content")

    # ── 1. Select Lab Folder (Browse only — no typed paths) ────────────
    scan = st.session_state.scan
    with st.container(key="pv_input_card"):
        st.markdown("#### 1. Select Lab Folder")
        st.markdown(
            '<div class="pv-label">Lab Folder</div>'
            '<div class="pv-sublabel">The top-level folder that contains G1, G2, … '
            "— choose it with Browse</div>",
            unsafe_allow_html=True,
        )
        ic1, ic2 = st.columns([6, 1.1])
        with ic1:
            if st.session_state.input_dir:
                path = st.session_state.input_dir
                name = os.path.basename(os.path.normpath(path)) or path
                st.markdown(
                    f'<div class="pv-picked"><b>{html.escape(name)}</b>'
                    f"<span>{html.escape(path)}</span></div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<div class="pv-picked pv-picked-empty">No folder selected yet</div>',
                    unsafe_allow_html=True,
                )
        with ic2:
            st.button(
                "Browse",
                key="browse_input",
                icon=":material/folder_open:",
                width="stretch",
                on_click=_browse_and_scan,
            )
        if st.session_state.get("picker_error"):
            st.markdown(
                f'<div class="pv-check-bad">{icon_svg("x-circle", 13)} '
                f"{html.escape(st.session_state.picker_error)}</div>",
                unsafe_allow_html=True,
            )
        if scan is not None:
            render_scan_preview(scan)

    render_advanced()
    n_images = pending_count(scan)
    ready = (
        n_images > 0
        and st.session_state.get("scan_confirmed", False)
        and not st.session_state.get("preflight_block", False)
    )

    # ── Start button — directly under folder selection, not buried below ──
    run_clicked = st.button(
        "Start Processing",
        icon=":material/play_arrow:",
        type="primary",
        width="stretch",
        disabled=not ready,
        key="start_processing_btn",
    )
    resume = st.session_state.get("run_mode") == "resume"
    plan = st.session_state.get("resume_plan")
    if scan is None:
        st.caption("Select a lab folder to scan it")
    elif st.session_state.get("preflight_block"):
        st.caption("Fix the problem shown above before starting")
    elif resume and plan is None:
        st.caption("Choose the output folder of the interrupted run to continue it")
    elif not n_images:
        st.caption("Nothing to process")
    elif not ready:
        st.caption("Confirm the scan above to enable processing")
    elif resume:
        st.caption(
            f"This will process the {n_images} remaining images "
            f"({len(plan.done)} already done will be skipped)"
        )
    else:
        st.caption(f"This will process the {n_images} images found by the scan")
    st.write("")

    # ── Info banner ──────────────────────────────────────────────────
    st.markdown(
        f"""
        <div class="pv-info-banner">
          <div class="txt">
            <b>{icon_svg("info", 15)} About the Pipeline</b>
            <p>Each image is processed with the confirmed configuration — dynamic per-image tiling,
            dual-branch inference, and gated logit fusion (no test-time augmentation), the exact
            setup verified at Microplastic IoU {CONFIRMED_IOU} / {CONFIRMED_DETECTION}% detection (measured
            without post-processing). Then hollow microplastic particles are filled, with a method per
            particle size (see Advanced).</p>
          </div>
          <div class="pv-mps-chip"><span class="dot" style="background:#2563eb;width:8px;height:8px;
               border-radius:50%;display:inline-block;margin-right:5px;"></span>Running on
               <b>{html.escape(load_pipeline().label)}</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── 2. Processing Progress ─────────────────────────────────────────
    progress_card = st.container(key="pv_progress_card")
    progress_card.markdown("#### 2. Processing Progress")
    progress_slot = progress_card.empty()

    def render_progress(idx, total, name, thumb, frac, elapsed_s, avg_s, eta_s, speed):
        with progress_slot.container():
            pcol1, pcol2 = st.columns([1, 3])
            with pcol1:
                if thumb is not None:
                    st.image(thumb, width="content")
                else:
                    st.markdown(
                        '<div class="pv-empty" style="padding:40px 4px;">No image yet</div>',
                        unsafe_allow_html=True,
                    )
            with pcol2:
                top1, top2 = st.columns([3, 1])
                with top1:
                    if total and idx:
                        st.markdown(f"**Processing Image {idx} of {total}**")
                        st.caption(name)
                    else:
                        st.markdown("**Waiting to start**")
                        st.caption(
                            "Select the lab folder, confirm the scan, then click Start Processing"
                        )
                with top2:
                    st.markdown(
                        f'<div style="text-align:right;font-weight:700;color:#111827;">{frac * 100:.0f}%</div>',
                        unsafe_allow_html=True,
                    )
                st.progress(frac)
                st.markdown(
                    f"""
                    <div class="pv-metric-grid">
                      <div class="pv-metric-tile"><div class="lbl">{icon_svg("clock", 12)} Elapsed Time</div><div class="val">{elapsed_s}</div></div>
                      <div class="pv-metric-tile"><div class="lbl">{icon_svg("clock", 12)} Avg. Time / Image</div><div class="val">{avg_s}</div></div>
                      <div class="pv-metric-tile"><div class="lbl">{icon_svg("hourglass", 12)} Est. Time Left</div><div class="val">{eta_s}</div></div>
                      <div class="pv-metric-tile"><div class="lbl">{icon_svg("zap", 12)} Speed</div><div class="val">{speed}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    render_progress(0, n_images, "", None, 0.0, "00:00", "—", "—", "—")

    # ── 3. Live Log + 4. Summary ─────────────────────────────────────
    lcol, scol = st.columns([1.5, 1])
    with lcol, st.container(key="pv_log_card"):
        st.markdown("#### 3. Live Log")
        log_slot = st.empty()
        log_slot.markdown(
            render_log_html(st.session_state.log_entries, max_rows=14), unsafe_allow_html=True
        )
    with scol, st.container(key="pv_summary_card"):
        st.markdown("#### 4. Summary")
        summary_slot = st.empty()

        def render_summary(done, processing, pending, failed, total):
            with summary_slot.container():
                d1, d2 = st.columns([1, 1.1])
                with d1:
                    st.markdown(
                        summary_donut_svg(done, processing, pending, failed, max(total, 1)),
                        unsafe_allow_html=True,
                    )
                with d2:
                    for label, val, color in [
                        ("Completed", done, "#22c55e"),
                        ("Processing", processing, "#2563eb"),
                        ("Pending", pending, "#d9dce3"),
                        ("Failed", failed, "#ef4444"),
                    ]:
                        st.markdown(
                            f'<div class="pv-legend-row"><span class="dot" style="background:{color};"></span>'
                            f"{label}<b>{val}</b></div>",
                            unsafe_allow_html=True,
                        )

        render_summary(0, 0, n_images, 0, n_images)
        # A plain st.button re-created inside render_summary() would collide (duplicate
        # key) since render_summary() is called repeatedly inside the processing loop.
        # It's rendered once here instead, outside the live-updating placeholder.
        if st.button(
            "View Results", key="view_results_btn", icon=":material/folder:", width="stretch"
        ):
            st.session_state.page = "results"
            st.rerun()

    # (Start Processing button already rendered above, right under folder selection.)
    if run_clicked and ready:
        _run_batch(scan, render_progress, render_summary, log_slot)


def _run_batch(scan, render_progress, render_summary, log_slot):
    """UI wrapper around ls.run_lab_batch — all file writing happens there."""
    runner = load_pipeline()
    pp = _pp_settings()
    resume = st.session_state.get("run_mode") == "resume"
    fell_back = False
    if resume:
        output_root = st.session_state.get("resume_dir")
        # Re-check at start: files may have changed since the folder was chosen.
        try:
            plan = ls.plan_resume(scan, output_root)
        except ls.OldLayoutError as e:
            st.error(f"Not started — {e}")
            return
        st.session_state.resume_plan = plan
    else:
        output_root, fell_back = resolve_output_root(scan.root)
        plan = None
    problems = ls.path_length_problems(scan, output_root, pu.max_path_length())
    if problems:
        st.error("Not started — " + " ".join(problems))
        return
    st.session_state.last_output_dir = output_root

    st.session_state.log_entries = []
    st.session_state.results = []
    total = len(plan.todo) if plan is not None else len(scan.images)
    state = {"done": 0, "failed": 0, "times": [], "thumb": None, "notes": 0}
    batch_start = time.time()

    def add_log(msg, status_html=""):
        st.session_state.log_entries.append((datetime.now().strftime("%H:%M:%S"), msg, status_html))

    def refresh_log():
        log_slot.markdown(
            render_log_html(st.session_state.log_entries, max_rows=14), unsafe_allow_html=True
        )

    def show_progress(i, img, frac, processing):
        times = state["times"]
        avg = sum(times) / len(times) if times else 0
        remaining = total - (i - 1 if processing else i)
        eta = avg * remaining if avg else 0
        render_progress(
            i,
            total,
            img.label,
            state["thumb"],
            frac,
            fmt_mmss(time.time() - batch_start),
            fmt_mmss(avg) if avg else "00:00",
            fmt_duration(eta) if avg else "—",
            f"{60 / avg:.1f} img/min" if avg else "—",
        )
        render_summary(
            state["done"],
            1 if processing else 0,
            total - state["done"] - state["failed"] - (1 if processing else 0),
            state["failed"],
            total,
        )

    def on_start(i, img):
        try:
            thumb = Image.open(img.src_path).copy()
            thumb.thumbnail((300, 300))
        except Exception:
            thumb = None
        state["thumb"] = thumb
        add_log(img.label, '<span class="pv-log-status-run">In progress...</span>')
        refresh_log()
        show_progress(i, img, (i - 1) / total, processing=True)

    def on_done(i, img, rec):
        ts = st.session_state.log_entries[-1][0]
        if rec["status"] == "ok":
            state["done"] += 1
            state["times"].append(rec["seconds"])
            st.session_state.results.append(
                {
                    "name": img.label,
                    "status": "ok",
                    "elapsed": rec["seconds"],
                    "out_name": os.path.relpath(rec["out_path"], output_root),
                    "class_counts": rec["class_counts"],
                    "total_px": rec["total_px"],
                }
            )
            status = (
                f'<span class="pv-log-status-ok">{icon_svg("check", 13)} Done ({rec["seconds"]:.1f}s)</span> '
                f'<span class="pv-log-status-info">{html.escape(describe_info(rec["postprocess"]))}</span>'
            )
            if rec.get("note"):
                state["notes"] += 1
                status += f' <span class="pv-log-status-warn">{html.escape(rec["note"])}</span>'
        else:
            state["failed"] += 1
            st.session_state.results.append(
                {"name": img.label, "status": "failed", "elapsed": None, "error": rec["error"]}
            )
            status = f'<span class="pv-log-status-warn">{icon_svg("x-circle", 13)} Failed — {html.escape(rec["error"][:80])}</span>'
        st.session_state.log_entries[-1] = (ts, img.label, status)
        refresh_log()
        show_progress(i, img, i / total, processing=False)

    if plan is not None:
        add_log(f"Resuming: {len(plan.done)} images already done (skipped), {total} to process")
    else:
        add_log(
            f"Scan: {total} images selected, {len(scan.skipped_folders)} folders skipped (see run_log.txt)"
        )
    add_log(f"ParticleVision {PARTICLEVISION_VERSION} — running on: {runner.label}")
    add_log(f"Post-processing: {pp.describe()}")
    add_log(
        f"Output: {output_root}"
        + ("  (on the Desktop — the lab folder's location is not writable)" if fell_back else "")
    )
    refresh_log()

    try:
        with pu.KeepAwake() as awake:
            add_log(
                "Keeping the computer awake during the run"
                if awake.method != "none"
                else "Could not stop the computer from sleeping — keep it plugged in and awake"
            )
            refresh_log()
            res = ls.run_lab_batch(
                scan,
                output_root,
                runner.predict,
                on_start=on_start,
                on_done=on_done,
                between_images=runner.between_images,
                resume_plan=plan,
                postprocess=pp,
                backend=runner.label,
            )
    except ls.SettingsMismatchError as e:
        st.error(f"Not started — {e}")
        return
    except ls.OutputExistsError as e:
        st.error(
            f"Stopped — {e} Nothing was overwritten. Start a new run (it creates a new output folder)."
        )
        return
    except Exception as e:
        st.error(f"Could not start the batch: {type(e).__name__}: {e}")
        return

    st.session_state.last_batch = res
    st.session_state.last_run_stats = {
        "total": res.total,
        "done": res.done,
        "failed": res.failed,
        "device": stats_key(runner),
        "avg": (sum(res.seconds) / len(res.seconds)) if res.seconds else 0,
    }
    if plan is not None:
        st.session_state.resume_plan = ls.plan_resume(scan, output_root)

    msg = f"{res.done} of {res.total} processed, {res.failed} failed"
    if plan is not None:
        msg += f", {res.skipped} skipped (already done before resuming)"
    msg += f". Saved to: {output_root}"
    if fell_back:
        msg += (
            "  —  NOTE: saved on your Desktop, not next to the lab folder, because that "
            "location could not be written to."
        )
    if state["notes"]:
        msg += (
            f"  —  {state['notes']} image(s) needed the out-of-memory fallback (smaller batch or CPU); "
            f"see run_log.txt."
        )
    if res.failed == 0 and not fell_back and not state["notes"]:
        st.success(msg)
    else:
        st.warning(msg)
    if res.failed:
        with st.expander(f"Failed images ({res.failed})", expanded=True):
            st.dataframe(
                pd.DataFrame([{"Image": l, "Error": e} for l, e in res.failures]),
                width="stretch",
                hide_index=True,
            )
    st.caption("summary.csv and run_log.txt are in the output folder root.")
    st.button(
        "Open output folder",
        icon=":material/folder_open:",
        key="reveal_after_run",
        on_click=_open_folder,
        args=(output_root,),
    )


# ─────────────────────────────────────────────────────────────────────────
# PAGE: Fix existing segmentations (hole filling; no segmentation is run)
# All work is done by app/holefill_folder.py — the same engine as tools/fix_segmented_folder.py.
# ─────────────────────────────────────────────────────────────────────────
def _hf_scan(path: str):
    path = (
        os.path.normpath(os.path.expanduser(path.strip().strip('"').strip("'")))
        if path.strip()
        else ""
    )
    st.session_state.hf_input = path
    st.session_state.hf_scan = None
    st.session_state.hf_error = ""
    st.session_state.hf_preview = None
    st.session_state.hf_result = None
    if not path:
        return
    if not os.path.isdir(path):
        st.session_state.hf_error = f"Folder not found: {path}"
        return
    entries = hf.scan_folder(path)
    manual = {e.rel_path for e in entries if hf.is_manual_label(e.src_path)}
    st.session_state.hf_scan = {
        "entries": entries,
        "manual": manual,
        "bytes": sum(os.path.getsize(e.src_path) for e in entries),
    }
    st.session_state.hf_output = hf.default_output_root(path)


def _hf_path_entered():
    _hf_scan(st.session_state.hf_path_box)


def _hf_browse():
    start = st.session_state.get("hf_input") or os.path.expanduser("~")
    picked = pick_folder(
        os.path.dirname(start) if os.path.isdir(start) else start,
        "Select the folder with the segmented files (the one containing G1, G2, …)",
    )
    if picked:
        st.session_state.hf_path_box = os.path.normpath(picked)
        _hf_scan(picked)


def _hf_browse_output():
    picked = pick_folder(
        os.path.dirname(st.session_state.hf_output), "Choose where to create the output folder"
    )
    if picked:
        st.session_state.hf_output = os.path.join(
            os.path.normpath(picked), os.path.basename(st.session_state.hf_output)
        )


def _hf_browse_continue():
    picked = pick_folder(
        os.path.dirname(st.session_state.get("hf_input") or os.path.expanduser("~")),
        "Select the interrupted output folder (…_holefilled_…)",
    )
    if picked:
        st.session_state.hf_continue_dir = os.path.normpath(picked)


def _hf_warnings(scan, out_dir: str) -> list:
    """(level, message) checks shown before a run; level 'block' disables the Run button."""
    msgs = []
    for what, p in (("input", st.session_state.hf_input), ("output", out_dir)):
        if "onedrive" in (p or "").lower():
            msgs.append(
                (
                    "warn",
                    f"The {what} folder is inside OneDrive. OneDrive may still be downloading or syncing "
                    "files; set the folder to 'Always keep on this device', or copy it outside OneDrive.",
                )
            )
    limit = pu.max_path_length()
    if limit and hf.longest_output_path(scan["entries"], out_dir) > limit:
        msgs.append(
            (
                "block",
                f"Some output paths would be longer than {limit} characters — choose a shorter output location.",
            )
        )
    parent = os.path.dirname(out_dir)
    free = pu.free_bytes(parent if os.path.isdir(parent) else os.path.dirname(parent))
    if free is not None and free < scan["bytes"] * 1.05:
        msgs.append(
            (
                "block",
                f"Not enough free disk space at the output location: about {fmt_bytes(scan['bytes'])} "
                f"needed, {fmt_bytes(free)} free.",
            )
        )
    try:
        hf.ls._assert_outside_input(out_dir, st.session_state.hf_input)
    except ValueError:
        msgs.append(("block", "The output folder can't be inside the input folder."))
    done = hf.already_postprocessed(st.session_state.hf_input)
    if done:
        msgs.append(
            (
                "warn",
                f"This folder was segmented by ParticleVision 1.1.0 or later: {done} result(s) are already "
                "post-processed (summary.csv, column postprocess = on). Fixing them again is not needed.",
            )
        )
    return msgs


def _hf_params() -> "hf.Params":
    return hf.Params(include_manual_labels=bool(st.session_state.get("hf_include_manual", False)))


def page_fixholes():
    st.markdown("# Fix existing segmentations")
    st.caption(
        "Fills hollow microplastic particles (red rings with Air, Water or grain inside) in folders that were "
        "segmented earlier. No segmentation is run. The selected folder is only read; the results go to a new "
        "folder with the same sub-folders and file names."
    )
    for k, v in (
        ("hf_input", ""),
        ("hf_scan", None),
        ("hf_error", ""),
        ("hf_output", ""),
        ("hf_mode", "new"),
        ("hf_continue_dir", ""),
        ("hf_preview", None),
        ("hf_result", None),
        ("hf_running", False),
        ("hf_interrupted", False),
    ):
        if k not in st.session_state:
            st.session_state[k] = v
    if st.session_state.hf_running:  # the page was left or stopped while a run was going
        st.session_state.hf_running = False
        st.session_state.hf_mode = "continue"
        st.session_state.hf_continue_dir = st.session_state.get("hf_active_output", "")
        st.session_state.hf_interrupted = True
    if st.session_state.hf_interrupted:  # shown until a run is started
        st.warning(
            "The last run was stopped before the end. Every finished file is complete. "
            f"Select the same input folder, then **Continue run** to finish it: "
            f"`{st.session_state.get('hf_active_output', '')}`"
        )

    # ── 1. input folder ──────────────────────────────────────────────────────
    with st.container(key="pv_hf_input"):
        st.markdown("#### 1. Folder with segmented files")
        c1, c2 = st.columns([6, 1.1])
        with c1:
            st.text_input(
                "Folder",
                key="hf_path_box",
                on_change=_hf_path_entered,
                label_visibility="collapsed",
                placeholder="Paste the folder path and press Enter, or use Browse",
            )
        with c2:
            st.button(
                "Browse",
                key="hf_browse",
                icon=":material/folder_open:",
                width="stretch",
                on_click=_hf_browse,
            )
        if st.session_state.get("picker_error"):
            st.error(st.session_state.picker_error)
        if st.session_state.hf_error:
            st.error(st.session_state.hf_error)
        scan = st.session_state.hf_scan
        if scan is not None:
            n, m = len(scan["entries"]), len(scan["manual"])
            if n == 0:
                st.warning("No Fused_St_Filt_Comp_Seg_Mod.tif files found in this folder.")
            else:
                st.markdown(
                    f"**{n}** segmented files found · **{m}** manual labels (PackBits TIFFs) will be copied "
                    f"unchanged · **{n - m}** app-made files will be checked for hollow particles · "
                    f"{fmt_bytes(scan['bytes'])}"
                )
    if scan is None or not scan["entries"]:
        return

    # ── 2. output ────────────────────────────────────────────────────────────
    st.markdown("#### 2. Output")
    st.radio(
        "Run",
        ["new", "continue"],
        key="hf_mode",
        horizontal=True,
        label_visibility="collapsed",
        format_func=lambda x: {"new": "Start a new run", "continue": "Continue an interrupted run"}[
            x
        ],
    )
    resume = st.session_state.hf_mode == "continue"
    if resume:
        c1, c2 = st.columns([6, 1.1])
        with c1:
            d = st.session_state.hf_continue_dir
            st.markdown(
                f"Interrupted output folder: `{d}`"
                if d
                else "Choose the interrupted output folder (…_holefilled_…)"
            )
        with c2:
            st.button(
                "Browse",
                key="hf_browse_continue",
                icon=":material/folder_open:",
                width="stretch",
                on_click=_hf_browse_continue,
            )
        out_dir = st.session_state.hf_continue_dir
        ok_dir = bool(out_dir) and os.path.isfile(os.path.join(out_dir, hf.PARAMS_NAME))
        if out_dir and not ok_dir:
            st.error(
                "This folder was not made by 'Fix existing segmentations' (no holefill_params.json)."
            )
        elif ok_dir:
            st.caption(
                f"{len(hf.read_summary(out_dir))} file(s) recorded in holefill_summary.csv; each finished file is "
                "checked against its recorded MD5 when the run continues."
            )
    else:
        c1, c2 = st.columns([6, 1.1])
        with c1:
            st.markdown(f"New output folder: `{st.session_state.hf_output}`")
        with c2:
            st.button(
                "Change",
                key="hf_browse_output",
                icon=":material/drive_folder_upload:",
                width="stretch",
                on_click=_hf_browse_output,
                help="Choose another location for the new output folder",
            )
        out_dir = st.session_state.hf_output
        ok_dir = True
    st.checkbox(
        "Also fill holes in the manual labels (not recommended: they are the lab's ground truth)",
        key="hf_include_manual",
        value=False,
        disabled=resume,
    )
    p = _hf_params()
    st.caption(
        f"Settings (src/config.py), method by particle size from the folder name: {p.method_text()}. "
        f"Circles (7 / 10 µm): each ring-shaped particle is redrawn as its own filled circle if ≥ "
        f"{p.circle_min_support:.0%} of its ring is red; a circle overlapping ≥ {p.circle_max_overlaps} others "
        f"(dense cluster) is not drawn. Closing (5 µm, no size): enclosed Air / Water up to {p.max_hole_area} px is "
        f"filled; an added patch larger than {p.max_added_region} px is not added. Only Air / Water pixels become "
        f"microplastic (never grain, no microplastic pixel removed)."
    )
    warnings = _hf_warnings(scan, out_dir) if out_dir else []
    for level, msg in warnings:
        (st.error if level == "block" else st.warning)(msg)

    # ── 3. preview ───────────────────────────────────────────────────────────
    st.markdown("#### 3. Preview one file (optional)")
    choices = [
        e.rel_path
        for e in scan["entries"]
        if p.include_manual_labels or e.rel_path not in scan["manual"]
    ]
    if choices:
        c1, c2 = st.columns([6, 1.1])
        with c1:
            st.selectbox("File", choices, key="hf_preview_file", label_visibility="collapsed")
        with c2:
            go = st.button(
                "Preview", key="hf_preview_btn", icon=":material/visibility:", width="stretch"
            )
        if go:
            e = next(x for x in scan["entries"] if x.rel_path == st.session_state.hf_preview_file)
            before, after, info = hf.preview(e, p)
            st.session_state.hf_preview = (e.rel_path, before, after, info)
        pv = st.session_state.hf_preview
        if pv is not None:
            rel, before, after, info = pv
            if before is None:
                st.info(f"{rel}: {info}")
            elif info["pixels_changed"] == 0:
                st.info(f"{rel}: no hollow particles to fill — the file would be copied unchanged.")
            else:
                y0, x0, s = hf.most_changed_window(before, after)
                crop = lambda m: Image.fromarray(
                    CFG.index_to_color[m[y0 : y0 + s, x0 : x0 + s]]
                ).resize((360, 360), Image.NEAREST)
                c1, c2 = st.columns(2)
                c1.image(crop(before), caption="before", width="stretch")
                c2.image(
                    crop(after),
                    caption=f"after — {hf.describe_info(info)}, {info['pixels_changed']} px added",
                    width="stretch",
                )
                st.caption(
                    f"Zoom: {s} × {s} px at y = {y0}, x = {x0} (the area with the most changed pixels) in {rel}"
                )

    # ── 4. run ───────────────────────────────────────────────────────────────
    st.markdown("#### 4. Run")
    blocked = any(level == "block" for level, _ in warnings) or not ok_dir or not out_dir
    run = st.button(
        "Continue run" if resume else "Start",
        key="hf_run_btn",
        type="primary",
        width="stretch",
        icon=":material/play_arrow:",
        disabled=blocked,
    )
    if run:
        _hf_run(scan, out_dir, p, resume)
    res = st.session_state.hf_result
    if res is not None and not run:
        _hf_show_result(res)


def _hf_show_result(res):
    msg = (
        f"{'Stopped' if res.stopped else 'Finished'}: "
        + ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in res.counts.items())
        + f" · pixels changed this session {res.px_changed:,}"
    )
    if res.already_done:
        msg += f" · {res.already_done} already done before"
    ok = not res.stopped and res.counts["error"] == 0 and res.input_unchanged
    (st.success if ok else st.warning)(msg)
    if res.input_unchanged is not None:
        st.markdown(
            "Input folder unchanged: **"
            + (
                "yes"
                if res.input_unchanged
                else f"NO — {len(res.input_changed_files)} file(s) changed"
            )
            + "** (every input file re-checked by MD5)"
        )
    st.caption(
        f"Output: {res.output_root} — report: {hf.SUMMARY_NAME}, {hf.LOG_NAME}, {hf.PARAMS_NAME}"
    )
    st.button(
        "Open output folder",
        icon=":material/folder_open:",
        key="hf_reveal",
        on_click=_open_folder,
        args=(res.output_root,),
    )


def _hf_run(scan, out_dir, params, resume):
    st.session_state.hf_running = True  # stays True if the run is interrupted (Stop / page change)
    st.session_state.hf_interrupted = False
    st.session_state.hf_active_output = out_dir
    st.button(
        "Stop",
        key="hf_stop",
        icon=":material/stop:",
        help="Stops after the current file. Every finished file is complete; continue later.",
    )
    bar = st.progress(0.0)
    line = st.empty()
    t0 = time.time()

    def on_start(i, n, e):
        line.markdown(f"File {i} of {n} · `{e.rel_path}`")

    def on_done(i, n, e, row):
        el = time.time() - t0
        bar.progress(i / n)
        left = (
            fmt_duration(el / i * (n - i))
            if i < n
            else "checking the input folder (re-reads every input file)…"
        )
        line.markdown(
            f"File {i} of {n} done · {row['status'].replace('_', ' ')} · `{e.rel_path}` · time left: {left}"
        )

    try:
        with pu.KeepAwake():
            res = hf.run_folder(
                st.session_state.hf_input,
                out_dir,
                params,
                resume=resume,
                on_start=on_start,
                on_done=on_done,
            )
    except (hf.ParamsMismatch, FileExistsError, ValueError) as ex:
        st.session_state.hf_running = False
        st.error(f"Not started — {ex}")
        return
    st.session_state.hf_running = False
    st.session_state.hf_result = res
    bar.progress(1.0)
    line.empty()
    _hf_show_result(res)


# ─────────────────────────────────────────────────────────────────────────
# Router
# ─────────────────────────────────────────────────────────────────────────
page = st.session_state.page
if page == "dashboard":
    page_dashboard()
elif page == "run":
    page_run()
elif page == "results":
    page_results()
elif page == "charts":
    page_charts()
elif page == "logs":
    page_logs()
elif page == "fixholes":
    page_fixholes()
