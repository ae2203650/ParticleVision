# Installing ParticleVision on a Mac

Step-by-step version of the Mac quick start in the [README](../README.md). You do this **once**. It takes about
15–30 minutes, most of it waiting, and needs an internet connection (no internet: see
[TROUBLESHOOTING.md](TROUBLESHOOTING.md) → *"pip install" fails*, offline install).

Apple Silicon Macs (M1, M2, M3, M4 …) use their built-in graphics (MPS) and take about 10 seconds per image.
Older Intel Macs work too, but use the processor and are slower.

---

## 1. Put the ParticleVision folder in your home folder

1. Double-click the `.zip` file.
2. Drag the **ParticleVision** folder into your home folder (in Finder: **Go → Home**).
   Avoid iCloud-synced folders such as Desktop or Documents if iCloud Drive is on.

**Updating from an earlier version (to 1.1.0)?** Put the new folder **next to** the old one — don't copy it over
the old one — and run the setup in the new folder (step 3).

## 2. Install Python 3.12

1. Open <https://www.python.org/downloads/release/python-31210/>.
2. Click **macOS 64-bit universal2 installer**, open the downloaded file and click through the installer.

## 3. Run the setup

1. In the ParticleVision folder, **double-click `setup_mac.command`**.
2. If macOS says it *cannot be opened because it is from an unidentified developer*: **right-click** the file →
   **Open** → **Open** (only the first time).
3. A Terminal window shows the progress. Leave it open. Success ends with:
   ```
     SELF-TEST PASSED
   ======================================================================
     SUCCESS — ParticleVision is installed and the self-test passed.
   ======================================================================
   ```
4. Take a screenshot of that window and send it to the person who gave you ParticleVision.

If it says **SETUP FAILED**, read the message and [TROUBLESHOOTING.md](TROUBLESHOOTING.md). The log is
`setup_log.txt` in the ParticleVision folder.

## 4. Start ParticleVision

1. **Double-click `run_app.command`** (right-click → **Open** the first time). Your browser opens ParticleVision;
   the Terminal window shows the address, e.g. `http://localhost:8501`.
2. Keep the Terminal window open while you use the app; closing it stops the app.
3. The first time you click **Browse**, macOS may ask whether Terminal may control "System Events" — click
   **OK**. That lets the folder window open in front of the browser.

Then follow the [README](../README.md) → *Segment your images*, or the [USER_GUIDE.md](USER_GUIDE.md).

## 5. Full check (optional)

Double-click **`run_full_check.command`**. It segments the 3 images in `Demo_Lab` through the same steps as a real
run and compares each result with the reference (about 30 seconds). Success ends with **FULL CHECK PASSED**; the
log is `selftest/full_check_log.txt`.
