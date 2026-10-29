# Troubleshooting

Find the message you see, then follow the steps. The five most common problems are also in the
[README](../README.md) → *Troubleshooting*. If nothing helps, send these files (in the ParticleVision folder) to
the person who gave you ParticleVision, with a screenshot of the error:
**`setup_log.txt`** and **`selftest/check_install_log.txt`**.

| Section | Problem |
|---|---|
| [1](#1-python-is-not-recognized--python-312-was-not-found) | "'python' is not recognized" / "Python 3.12 was not found" |
| [2](#2-windows-smartscreen-or-defender-blocks-the-bat-file) | Windows SmartScreen or Defender blocks the .bat file |
| [3](#3-dll-load-failed--winerror-126--c10dll) | "DLL load failed" / "WinError 126" / "c10.dll" |
| [4](#4-tcltk-is-missing--the-browse-button-shows-a-red-message) | "Tcl/Tk is missing" / Browse shows a red message |
| [5](#5-the-folder-window-does-not-appear-after-clicking-browse) | The folder window does not appear after Browse |
| [6](#6-nvidia-card-present-but-running-on-cpu) | NVIDIA card present but "Running on: CPU" |
| [7](#7-out-of-memory--used-batch-8-instead-of-16) | "Out of memory" / "used batch 8 instead of 16" |
| [8](#8-paths-too-long-for-windows) | "Paths too long for Windows" |
| [9](#9-very-slow-or-files-locked--access-denied-during-a-run) | Very slow, or files "locked" / "Access denied" |
| [10](#10-the-laptop-went-to-sleep-during-a-run) | The laptop went to sleep during a run |
| [11](#11-port-already-in-use--the-browser-shows-the-wrong-page) | "Port already in use" / wrong page |
| [12](#12-the-self-test-or-the-full-check-failed) | The self-test or the full check failed |
| [13](#13-cant-be-continued-with-the-current-settings) | "can't be continued with the current settings" |
| [14](#14-scipy-is-missing-or-the-post-processing-check-failed) | "scipy is missing" / post-processing check FAILED |
| [15](#15-i-want-the-result-without-the-hole-filling) | I want the result without the hole filling |
| [16](#16-pip-install-fails-proxy--firewall--offline-install) | "pip install" fails (proxy, firewall, offline install) |
| [17](#17-windows-on-arm) | Windows on ARM |
| [18](#18-the-copied-folder-has-a-venv-folder) | The copied folder has a `.venv` folder |
| [19](#19-the-model-files-are-not-in-place) | "The model files are not in place" |

---

## 1. "'python' is not recognized" / "Python 3.12 was not found"

**Cause:** Python 3.12 is not installed, or "Add python.exe to PATH" was not ticked.

1. Run the Python 3.12 installer again ([INSTALL_WINDOWS.md](INSTALL_WINDOWS.md), step 3): **Modify** → **Next**
   → tick **Add Python to environment variables** → **Install**.
2. Run `setup_windows.bat` again.
3. Check: in *Command Prompt*, `py -3.12 --version` should print `Python 3.12.10`.

## 2. Windows SmartScreen or Defender blocks the .bat file

"Windows protected your PC" → **More info** → **Run anyway**. If your antivirus deletes or quarantines the file,
restore it and allow it — the scripts only install Python packages into the ParticleVision folder.

## 3. "DLL load failed" / "WinError 126" / "c10.dll"

**Cause:** the Microsoft Visual C++ runtime is missing (seen when importing torch).

1. Install <https://aka.ms/vs/17/release/vc_redist.x64.exe>.
2. Restart the computer and run `setup_windows.bat` again.

## 4. "Tcl/Tk is missing" / the Browse button shows a red message

**Cause:** the Python install is missing tcl/tk, which the folder window needs.

1. Run the Python 3.12 installer again → **Modify** → tick **tcl/tk and IDLE** → finish.
2. Run `setup_windows.bat` again.

## 5. The folder window does not appear after clicking Browse

It sometimes opens **behind** the browser: look at the **taskbar** for a new window and click it. On a Mac,
click **OK** if asked to allow control of "System Events".

## 6. NVIDIA card present but "Running on: CPU"

1. Update the NVIDIA driver (<https://www.nvidia.com/Download/index.aspx>) and restart the computer.
2. Run `setup_windows.bat` again — it checks the card again and installs the matching version.

`setup_log.txt` (step 3) says which card and driver it found. A driver older than 528 is too old: setup then
installs the CPU version and tells you to update.

## 7. "Out of memory" / "used batch 8 instead of 16"

Not an error: graphics cards with 4–6 GB of memory can run short. ParticleVision then retries that image in
smaller pieces, and if needed on the processor. **The results are the same**; it is only slower. The note appears
in the live log and in `run_log.txt`. Closing other programs (games, video editors, many browser tabs) helps.

## 8. "Paths too long for Windows"

Windows cannot use paths longer than 260 characters; the message names the lab folder to move. Move it to a short
place such as **`C:\PV\`** and click **Browse** again. (The results folder is created next to the lab folder, so
both must be short.)

## 9. Very slow, or files "locked" / "Access denied" during a run

Probably OneDrive, Dropbox or an antivirus is scanning each new file. Keep the ParticleVision folder and the lab
folders **outside OneDrive** (e.g. `C:\ParticleVision` and `C:\PV\`). Without a graphics card, 1–3 minutes per
image is normal.

## 10. The laptop went to sleep during a run

The app keeps the computer awake during a run, but **closing the lid still sleeps the laptop**, and a flat
battery stops everything. Plug it in and leave the lid open. To finish the run: [USER_GUIDE.md](USER_GUIDE.md) →
*If a run was interrupted* — finished images are not redone.

## 11. "Port already in use" / the browser shows the wrong page

ParticleVision automatically uses the next free port (8502, 8503, …) and opens the right address; the window
shows it (*ParticleVision is running:  http://localhost:8502*). If an old ParticleVision window is still open,
close it first.

## 12. The self-test or the full check failed

The last lines say why. Send `selftest/check_install_log.txt` and `setup_log.txt` (full check:
`selftest/full_check_log.txt`).

| Message | Fix |
|---|---|
| *the Demo_Lab folder is missing* | Copy the `Demo_Lab` folder again from the original package. |
| *model file … is the confirmed version* **FAIL** | A model file is missing or damaged: copy the `checkpoints` folder again from the original package. |
| *pixel agreement* below 99.99 % | This computer produces different results. Don't use it for real work; send the log files. |

## 13. "can't be continued with the current settings"

A run is only continued with the **same** settings that started it (Run Pipeline → **Advanced**: hole filling
on/off, particle size, raw copy) and the same ParticleVision version — otherwise one results folder would mix
differently processed images. The message says what differs: set it back and choose the folder again, or start a
new run. Folders made before 1.1.0 can't be continued; start a new run.

## 14. "scipy is missing" or the post-processing check FAILED

The hole filling needs the scipy package (new in 1.1.0). Run the setup again (`setup_windows.bat` /
`setup_mac.command`); it rebuilds the environment with scipy. If the post-processing check still fails, send
`selftest/check_install_log.txt`.

## 15. I want the result without the hole filling

Run Pipeline → **Advanced** → untick **Fill hollow microplastic particles**, or tick **Also save the raw
(unfilled) segmentation** to get both. Command line: `tools/segment_lab_folder.py --no-postprocess` or
`--save-raw` ([USER_GUIDE.md](USER_GUIDE.md) → *Command line*).

## 16. "pip install" fails (proxy / firewall / offline install)

- **Easiest:** run the setup once on a network without restrictions (e.g. home or a phone hotspot). After that
  ParticleVision needs **no internet at all**.
- **Proxy:** ask IT for its address. Then, in *Command Prompt* in the ParticleVision folder (with their address):
  ```
  set HTTPS_PROXY=http://proxy.example.edu:8080
  setup_windows.bat
  ```
- **Offline install** (a computer that can never reach the internet):
  1. On another computer **with** internet and Python 3.12, in the ParticleVision folder:
     ```
     python tools\download_wheels.py --target win-cpu
     ```
     (`--target win-cu126` for an NVIDIA card). This fills `wheels\win-cpu` (about 0.7 GB) or `wheels\win-cu126`.
  2. Copy the whole ParticleVision folder, including `wheels`, to the offline computer.
  3. Run `setup_windows.bat` — it installs from `wheels` without internet.

## 17. Windows on ARM

Check *Settings → System → About → System type*. If it says **ARM-based processor**: PyTorch does not fully
support Windows on ARM yet. What should work: install the normal **Windows installer (64-bit)** Python 3.12.10 —
Windows runs it through its built-in x64 emulation — then run `setup_windows.bat` as usual. It uses the
processor only and is slow. **This has not been tested yet**; if the self-test fails, send the logs.

## 18. The copied folder has a `.venv` folder

That is fine. `.venv` belongs to a development Mac and is never used; ParticleVision always builds and uses its own
`.pv_env` folder. You do not need to delete `.venv`.

## 19. "The model files are not in place"

`check_install.py` and the app say this when a model file is missing or damaged (for example in a copy of the
public repository, which doesn't include them). The message names the file and the folder it belongs in.

1. Download the files from Zenodo, <https://doi.org/10.5281/zenodo.23691786> (see the [README](../README.md) →
   *Download the model files*).
2. Put them in the folders shown there, or run this (on a Mac: `python3`), which places them and checks their MD5
   checksums:
   ```
   python tools/download_models.py --from-dir <download folder>
   ```
   Or download and place them in one step:
   ```
   python tools/download_models.py --base-url https://zenodo.org/records/23691786/files
   ```
3. Run `check_install.py` again.
