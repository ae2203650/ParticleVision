# Installing ParticleVision on Windows

Step-by-step version of the Windows quick start in the [README](../README.md). You do this **once**. It takes
about 20–40 minutes, most of it waiting, and needs an internet connection (no internet: see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) → *"pip install" fails*, offline install).

---

## 1. Check your computer type

1. Press the **Windows key**, type **About your PC**, press Enter.
2. Look at **System type**:

| System type | What to do |
|---|---|
| "64-bit operating system, x64-based processor" | Continue with step 2. |
| "ARM-based processor" | Read [TROUBLESHOOTING.md](TROUBLESHOOTING.md) → *Windows on ARM* first. |

## 2. Put the ParticleVision folder in a short place

Windows cannot handle very long folder paths, so the folder must be near the top of the drive.

1. Right-click the `.zip` file → **Extract All…** → type `C:\` → **Extract**.
2. You now have the folder **`C:\ParticleVision`** (with `README.md` inside).

Do **not** put it on the Desktop, in Downloads, or in OneDrive: those paths are long, and OneDrive can lock or
slow down files while a run is writing them.

**Updating from an earlier version (to 1.1.0)?** Extract the new version **next to** the old one (for example
`C:\ParticleVision_1.1.0`) — don't copy it over the old one — and run the setup in the new folder (step 5). If
you use the faster mode, run `setup_accel_windows.bat` again afterwards.

## 3. Install Python 3.12

1. Open <https://www.python.org/downloads/release/python-31210/>.
2. Scroll to *Files* and click **Windows installer (64-bit)**. Open the downloaded file.
3. On the **first screen**, tick **"Add python.exe to PATH"** at the bottom. This is important.
4. Click **Install Now**. (With *Customize installation*, keep **"tcl/tk and IDLE"** and **"py launcher"**
   ticked — the folder picker needs tcl/tk.)
5. When it says *Setup was successful*, click **Close**.

## 4. Update the NVIDIA driver (only with an NVIDIA graphics card)

A graphics card makes ParticleVision about **8× faster**. To check: press **Ctrl + Shift + Esc** (Task Manager)
→ **Performance** → look for **GPU … NVIDIA**.

| You see | What to do |
|---|---|
| NVIDIA listed | Install the newest driver from <https://www.nvidia.com/Download/index.aspx> (or the *NVIDIA App*), then restart the computer. |
| No NVIDIA (Intel / AMD only) | Skip this step. ParticleVision uses the processor: same results, slower. For a faster option see [FASTER_MODE_WINDOWS.md](FASTER_MODE_WINDOWS.md). |

## 5. Run the setup

1. Open `C:\ParticleVision` and **double-click `setup_windows.bat`**.
2. If a blue box says **"Windows protected your PC"**: click **More info** → **Run anyway**.
3. A black window shows the progress. **Leave it open.** It downloads about 1–3 GB.

Success looks like this:

```
[1/5] Checking Python ...
      OK
[2/5] Creating a fresh environment in .pv_env ...
      OK
[3/5] Choosing the PyTorch build ...
      NVIDIA GPU found (NVIDIA GeForce RTX 4060 Laptop GPU, driver 551.86) → CUDA 12.6 build
[4/5] Installing packages from the internet (this can take 5–20 minutes) ...
      OK
[5/5] Running the self-test (segments one sample image — up to a few minutes on a CPU) ...
...
  pixel agreement, all pixels        : 100.0000 %   (need ≥ 99.99 %)
  microplastic pixel count difference: +0.0000 %   (need within ±0.1 %)
...
  SELF-TEST PASSED
======================================================================
  SUCCESS — ParticleVision is installed and the self-test passed.
======================================================================
```

- Without an NVIDIA card, step 3 says *"no NVIDIA GPU found → CPU build"* — that is fine, and the agreement can
  be very slightly below 100 %. Anything that says **SELF-TEST PASSED** is correct.
- **Take a screenshot of the end of that window** and send it to the person who gave you ParticleVision.
  Then press any key to close the window.
- If it says **SETUP FAILED**, read the message — it says what to do — and see
  [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The full log is `C:\ParticleVision\setup_log.txt`.

## 6. Start ParticleVision

1. **Double-click `run_app.bat`.** A black window opens and, after up to a minute the first time, your web
   browser opens ParticleVision by itself. The window shows the address:
   ```
     ParticleVision is running:  http://localhost:8501
   ```
2. **Keep the black window open** while you use the app. Closing it stops ParticleVision.
3. Nothing is sent to the internet: the app runs only on your computer.

Then follow the [README](../README.md) → *Segment your images*, or the [USER_GUIDE.md](USER_GUIDE.md).

## 7. Full check (optional, recommended on a new computer)

The setup's self-test checks one image. The full check runs **3 more images** (the `Demo_Lab` folder, in the real
lab folder layout) through every step of a real run and compares each result with the Mac's.

1. Double-click **`run_full_check.bat`**. It takes about 30 seconds with an NVIDIA card, 3–9 minutes without one.
2. Success ends with:
   ```
     [PASS] Before/Fused_St_Filt_Comp_Seg_Mod.tif    raw map: agreement 100.0000 %  microplastic count +0.0000 %  identical
     [PASS] Before/Fused_St_Filt_Comp_Seg_Mod.tif    final result = the post-processing of its raw map
     ... (the same two lines for IN/PV 03 and WT/PV 03)
     [PASS] summary.csv has 3 of 3 rows marked ok
     [PASS] summary.csv records version 1.1.0, the backend and the post-processing (D07 → circles) for every image
   ...
     FULL CHECK PASSED
   ```

The result is saved in `C:\ParticleVision\selftest\full_check_log.txt`. The box at the bottom left of the app
then also says *Full check (3 images) passed*.

## 8. Where to keep the lab folders

Keep the lab data in a short location too, for example **`C:\PV\`** (e.g. `C:\PV\Lab 2026\`). The results are
written into a new folder next to the lab folder, and Windows refuses paths longer than 260 characters. If a lab
folder is too deep, ParticleVision tells you before starting — it never fails halfway through for this reason.
