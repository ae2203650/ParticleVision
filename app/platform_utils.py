"""
Operating-system specific helpers for the app (no Streamlit, no model code):
folder picker, opening a folder, keeping the computer awake, battery status,
Windows path-length limit, free disk space, and crash-proof printing.

Every function asks `os_name()` which system it is on, so tests can pretend to
be Windows / Linux / macOS by patching that one function.
"""

import ctypes
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from typing import Optional


def os_name() -> str:
    """'Windows', 'Darwin' (macOS) or 'Linux'."""
    return platform.system()


def make_stdio_safe() -> None:
    """Never crash on printing: Windows consoles/log files can't encode every
    character (e.g. ✅). Unencodable characters are replaced instead."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except Exception:  # noqa: BLE001
            pass


# ─────────────────────────────────────────────────────────────────────────
# Folder picker
# ─────────────────────────────────────────────────────────────────────────
class PickerError(Exception):
    """The picker could not be shown at all (message is shown to the user)."""


def _applescript_str(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _pick_folder_macos(initial_dir: str, prompt: str) -> str:
    """Native macOS picker via AppleScript (osascript) in a separate process.
    Shown by System Events after `activate` so it opens in front of the
    browser; if that isn't allowed, falls back to a plain dialog."""
    prompt_s = _applescript_str(prompt)
    loc_line, loc_clause = "", ""
    if initial_dir and os.path.isdir(initial_dir):
        loc_line = f"set defLoc to (POSIX file {_applescript_str(initial_dir)}) as alias"
        loc_clause = " default location defLoc"
    front = "\n".join(
        filter(
            None,
            [
                loc_line,
                'tell application "System Events"',
                "    activate",
                f"    set chosen to choose folder with prompt {prompt_s}{loc_clause}",
                "end tell",
                "return POSIX path of chosen",
            ],
        )
    )
    plain = "\n".join(
        filter(
            None,
            [
                loc_line,
                f"return POSIX path of (choose folder with prompt {prompt_s}{loc_clause})",
            ],
        )
    )
    for script in (front, plain):
        try:
            r = subprocess.run(
                ["osascript", "-e", script], capture_output=True, text=True, timeout=900
            )
        except Exception:  # noqa: BLE001
            return ""
        if r.returncode == 0:
            return r.stdout.strip()
        if "-128" in r.stderr:  # user pressed Cancel
            return ""
    return ""


# Runs in its own Python process: tkinter must own the main thread, and
# Streamlit runs our code on a background thread.
TK_PICKER_SCRIPT = r"""
import sys
try:
    import tkinter as tk
    from tkinter import filedialog
except Exception as e:
    sys.stderr.write("NO_TKINTER: %s" % e)
    sys.exit(3)
try:
    root = tk.Tk()
except Exception as e:
    sys.stderr.write("NO_DISPLAY: %s" % e)
    sys.exit(4)
root.withdraw()
root.attributes("-topmost", True)      # open above the browser window
root.update()
path = filedialog.askdirectory(parent=root, initialdir=sys.argv[1] or None,
                               title=sys.argv[2], mustexist=True)
root.destroy()
sys.stdout.write(path or "")
"""


def _pick_folder_tk(initial_dir: str, prompt: str) -> str:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    kwargs = {}
    if os_name() == "Windows":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)
    try:
        r = subprocess.run(
            [sys.executable, "-c", TK_PICKER_SCRIPT, initial_dir or "", prompt],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=900,
            **kwargs,
        )
    except subprocess.TimeoutExpired:
        return ""
    except Exception as e:  # noqa: BLE001
        raise PickerError(f"The folder picker could not start ({e}).")
    if r.returncode == 3 or "NO_TKINTER" in r.stderr:
        if os_name() == "Windows":
            raise PickerError(
                "The folder picker needs Tcl/Tk, which is missing from this Python. "
                "Re-run the Python installer, choose Modify, and tick 'tcl/tk and IDLE'."
            )
        raise PickerError(
            "The folder picker needs Tk. Install it (e.g. 'sudo apt install python3-tk')."
        )
    if r.returncode == 4 or "NO_DISPLAY" in r.stderr:
        raise PickerError("The folder picker could not open a window on this computer.")
    if r.returncode != 0:
        raise PickerError(f"The folder picker failed: {r.stderr.strip()[:200]}")
    picked = r.stdout.strip()
    return os.path.normpath(picked) if picked else ""  # Tk returns C:/… on Windows


def pick_folder(
    initial_dir: str, prompt: str = "Select the lab folder (the one containing G1, G2, …)"
) -> str:
    """Native folder dialog. Returns the chosen folder, or "" if cancelled.
    Raises PickerError only when no dialog could be shown."""
    if os_name() == "Darwin":
        return _pick_folder_macos(initial_dir, prompt)
    return _pick_folder_tk(initial_dir, prompt)


# ─────────────────────────────────────────────────────────────────────────
# Open a folder in the file manager
# ─────────────────────────────────────────────────────────────────────────
def open_folder(path: str) -> Optional[str]:
    """Open `path` in Explorer / Finder / the Linux file manager.
    Returns an error message, or None on success."""
    try:
        system = os_name()
        if system == "Windows":
            os.startfile(path)  # type: ignore[attr-defined]
        elif system == "Darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
        return None
    except Exception as e:  # noqa: BLE001
        return f"Could not open the folder ({e}). It is here: {path}"


# ─────────────────────────────────────────────────────────────────────────
# Keep the computer awake during a run
# ─────────────────────────────────────────────────────────────────────────
ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


class KeepAwake:
    """Context manager: stops the computer from going to sleep while a run is
    in progress (the screen may still turn off). Closing a laptop lid will
    still put it to sleep on most machines. `method` says what was used."""

    def __init__(self):
        self.method = "none"
        self._proc = None

    def __enter__(self):
        system = os_name()
        try:
            if system == "Windows":
                # Must be called from the thread that runs the batch (it is).
                if ctypes.windll.kernel32.SetThreadExecutionState(
                    ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                ):
                    self.method = "SetThreadExecutionState"
            elif system == "Darwin":
                self._proc = subprocess.Popen(
                    ["caffeinate", "-i", "-w", str(os.getpid())],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                self.method = "caffeinate"
        except Exception:  # noqa: BLE001 — best effort
            self.method = "none"
        return self

    def __exit__(self, *exc):
        try:
            if self.method == "SetThreadExecutionState":
                ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
            elif self._proc is not None:
                self._proc.terminate()
        except Exception:  # noqa: BLE001
            pass
        return False


# ─────────────────────────────────────────────────────────────────────────
# Battery
# ─────────────────────────────────────────────────────────────────────────
class _SYSTEM_POWER_STATUS(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_ubyte),
        ("BatteryFlag", ctypes.c_ubyte),
        ("BatteryLifePercent", ctypes.c_ubyte),
        ("SystemStatusFlag", ctypes.c_ubyte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def on_battery() -> Optional[bool]:
    """True = running on battery, False = plugged in / no battery, None = unknown."""
    try:
        system = os_name()
        if system == "Windows":
            st = _SYSTEM_POWER_STATUS()
            if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
                return None
            if st.BatteryFlag == 128:  # no system battery (desktop PC)
                return False
            return {0: True, 1: False}.get(st.ACLineStatus)
        if system == "Darwin":
            out = subprocess.run(
                ["pmset", "-g", "batt"], capture_output=True, text=True, timeout=5
            ).stdout
            if "Battery Power" in out:
                return True
            if "AC Power" in out:
                return False
            return None
        base = "/sys/class/power_supply"
        for name in os.listdir(base):
            p = os.path.join(base, name, "online")
            if name.upper().startswith(("AC", "ADP")) and os.path.exists(p):
                return open(p).read().strip() == "0"
    except Exception:  # noqa: BLE001
        return None
    return None


# ─────────────────────────────────────────────────────────────────────────
# Paths and disk space
# ─────────────────────────────────────────────────────────────────────────
WINDOWS_MAX_PATH = 259  # MAX_PATH 260 includes the terminating NUL


def windows_long_paths_enabled() -> bool:
    try:
        import winreg  # Windows only

        with winreg.OpenKey(
            winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\FileSystem"
        ) as k:
            return winreg.QueryValueEx(k, "LongPathsEnabled")[0] == 1
    except Exception:  # noqa: BLE001
        return False


def max_path_length() -> Optional[int]:
    """Longest full path this computer can handle, or None if there is no
    practical limit (macOS, Linux, Windows with long paths enabled)."""
    if os_name() == "Windows" and not windows_long_paths_enabled():
        return WINDOWS_MAX_PATH
    return None


def dir_is_writable(path: str) -> bool:
    """Actually try to create a file (os.access is unreliable on Windows)."""
    try:
        with tempfile.TemporaryFile(dir=path):
            return True
    except OSError:
        return False


def free_bytes(path: str) -> Optional[int]:
    """Free space on the drive holding `path` (or its nearest existing parent)."""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            return None
        p = parent
    try:
        return shutil.disk_usage(p).free
    except OSError:
        return None
