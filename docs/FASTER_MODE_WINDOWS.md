# Faster mode on Windows (optional)

Details for the [README](../README.md) → *Faster mode on Windows without NVIDIA*.

**When to use it:** only on a Windows PC **without an NVIDIA card** (for example AMD Radeon or Intel graphics),
where ParticleVision otherwise runs on the processor at about 30–70 seconds per image. With an NVIDIA card you
don't need it: the normal mode is already fast.

It never replaces the normal mode blindly. ParticleVision uses the faster mode only if, **on this PC**:
1. it gives the same results as the verified Mac on 4 test images (the same rule as the self-test: at least
   99.99 % of pixels identical, microplastic count within ±0.1 %), **and**
2. it is faster than the normal mode, measured in the same test.

Otherwise ParticleVision keeps the normal mode, and the test says why.

---

## 1. Before you start

- ParticleVision must already be installed and working (`setup_windows.bat` ended with **SUCCESS**).
- Keep the laptop **plugged in**. The test takes about 10–20 minutes.

## 2. Install and test

In `C:\ParticleVision`:

1. Double-click **`setup_accel_windows.bat`**. It adds 2 packages (`onnxruntime-directml`, `flatbuffers`), checks
   that nothing else changed, and then runs the test automatically.
2. Later, to run the test again (for example after a graphics-driver update), double-click
   **`benchmark_accel.bat`**.

## 3. What PASS looks like

The top of the window lists the processor, the Windows version and each graphics card with its driver version.
The end looks like this (example numbers):

```
  SPEED ON THIS COMPUTER (same run, same 4 image(s))
                                            s/image  accuracy  decision
  Normal path — CPU, PyTorch                   34.8  PASS      the reference speed
  AMD Radeon Pro WX 7130 (DirectML)             9.6  PASS      passes, 3.6× faster than the normal path  ← USED
  Intel(R) UHD Graphics 630 (DirectML)         52.1  PASS      not used — slower than the normal path (52.1 vs 34.8 s/image)
  CPU (ONNX Runtime CPU)                       29.9  PASS      passes, 1.2× faster than the normal path
==============================================================================
  The app will use: AMD Radeon Pro WX 7130 (DirectML)  (passes, 3.6× faster than the normal path)
```

- Start ParticleVision with `run_app.bat` as usual. The box at the bottom left then says, for example,
  **"Running on AMD Radeon Pro WX 7130 (DirectML), verified"**, and has a **Faster mode** switch (off = the
  normal mode).
- If the last line says **"The app will use: the normal path — …"**, nothing changes: ParticleVision works
  exactly as before, and the line says why.
- If the faster mode ever fails on one image during a run, that image is redone in the normal mode
  automatically, and `run_log.txt` records it.

## 4. What to send

Always send these, whether it passed or failed:
1. **A screenshot of the whole benchmark window** (the top shows the processor, Windows version and graphics
   drivers; the end shows the speeds and the decision).
2. From `C:\ParticleVision`: **`accel_benchmark_log.txt`**, **`accel_result.json`** and **`setup_accel_log.txt`**.

## 5. Turning it off

- Use the **Faster mode** switch in the app, or
- delete `C:\ParticleVision\accel_result.json` (the faster mode is then never used).

After a graphics-driver update, the faster mode switches itself off until `benchmark_accel.bat` has run again.
