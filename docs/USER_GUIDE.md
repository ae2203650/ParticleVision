# Using ParticleVision

The details behind the [README](../README.md) (sections 4–7): every page of the app, the result files and the
hole filling. Installing: [INSTALL_WINDOWS.md](INSTALL_WINDOWS.md) / [INSTALL_MAC.md](INSTALL_MAC.md).

| Colour | Class |
|---|---|
| black | Grains |
| light green | Air |
| blue | Water |
| red | Microplastic (tiny particles, a few pixels across; called `fines` in the `summary.csv` column names) |

ParticleVision works on a whole **lab folder** at once and never changes anything inside it.

---

## 1. The app

Start it with `run_app.command` (Mac) or `run_app.bat` (Windows); keep the Terminal / black window open.

- The top left shows the version (**Segmenter · version 1.1.0**).
- The box at the bottom left shows **Running on** — what does the work, e.g. *NVIDIA GeForce RTX 4060 (CUDA)*,
  *Apple GPU (MPS)* or *CPU* — and **Self-test passed on this machine** (green). If it is amber, run the setup
  again. After the optional full check it also says **Full check (3 images) passed**.
- The pages on the left: **Dashboard**, **Run Pipeline**, **Results**, **Charts**, **Logs**, **Fix Segmentations**.

## 2. Choose and scan the lab folder

1. Click **Run Pipeline**.
2. Click **Browse**. A folder window opens (on Windows it may appear behind the browser — check the taskbar).
3. Select the **top-level lab folder** — the one that contains `G1`, `G2`, … — and confirm.

The app scans the folder straight away and shows:
- how many images it found, per G folder, experiment, and Before / IN / WT;
- the **estimated run time** on this computer;
- **Skipped folders** and **Ignored files**, each with the reason (e.g. *Aborted Runs*, a badly named folder,
  the lab's own `…_Seg_Mod` files);
- checks: enough free disk space (each result is about 44 MB), paths short enough, running on battery.

**Only want to check a folder?** Stop here — scanning never processes or changes anything.

What it picks: in each experiment's `Before` folder and in each `IN\PV ##` / `WT\PV ##` folder, the one image
named exactly **`Fused_St_Filt_Comp`** (.jpg/.jpeg/.png/.tif/.tiff). Everything else is ignored and listed.

## 3. Run

1. Tick **"I've checked this scan — process these … images"**.
2. (Optional) Open **Advanced** to change the hole filling for this run (section 6). The defaults are right for
   normal lab runs.
3. Click **Start Processing**.

You see the current image (e.g. *G1 › G1-C0.05-F1.0-P9.0-S30000-SCP-D07 (18) › IN › PV 03*), how many are done,
the time per image and the time left. **Leave the computer on and plugged in**: the app keeps it awake during a
run, but **closing a laptop lid still puts it to sleep**.

| Computer | Time per image | 400 images |
|---|---|---|
| NVIDIA card or Apple Silicon Mac | about 10 s | about 1 hour |
| Processor only | about 1–3 min | 8–20 hours (start in the evening) |

If one image fails, the run continues; the end message says *"X of Y processed, Z failed"* and lists the failures.

## 4. If a run was interrupted

(Computer went to sleep, the window was closed, power cut …)

1. Browse to the **same lab folder** again.
2. Under **Run mode**, choose **Continue a previous (interrupted) run**.
3. Click **Choose** and select the output folder of that run (`…_segmented_<date>_<time>`).
4. The app shows how many images are already done (they are **skipped**) and how many are left. A result only
   counts as done if it opens, has the right size and only the four class colours; anything else is redone,
   and the reason is shown.
5. Tick the box and click **Start Processing**. Results go into the same folder.

A run can only be continued **with the same settings and the same version** that started it (they are saved in
`run_settings.json`). If something under **Advanced** differs, the app shows *"can't be continued with the
current settings"* and says what — set it back, or start a new run. Results folders made before 1.1.0 (no
`run_settings.json`) or with the old names (`…_Segmented.tif`) can't be continued — start a new run. The
**Charts** page can still open them.

## 5. The results

The output folder and its layout are shown in the [README](../README.md) → *Segment your images*. Click **Open
output folder** at the end of a run to open it. ParticleVision never writes into the lab folder and never
overwrites a file. (Files named `Fused_St_Filt_Comp_Seg_Mod` inside the lab folder are the lab's manual
segmentations — they are never processed; only `Fused_St_Filt_Comp` is.)

| File | Contents |
|---|---|
| `Fused_St_Filt_Comp_Seg_Mod.tif` | The coloured result, same size as the photo, with hollow microplastic particles filled (section 6). Same name as the lab's manual segmentations — **don't copy it into the original lab folders.** |
| `Fused_St_Filt_Comp_Seg_Raw.tif` | Only with *Also save the raw (unfilled) segmentation*: the model's result before the filling. |
| `summary.csv` | One row per image (opens in Excel), columns below. |
| `run_log.txt` | At the top: version, backend, hole-filling settings, code version. Then every skipped folder or file with the reason, every failure, per image the particle size and the method used, notes such as "used a smaller batch because the graphics card ran out of memory", and whether the run was resumed. |
| `run_settings.json` | The settings of the run (used to check a Continue). |

**`summary.csv` columns** (the original ones first, in the same order as before, so existing scripts keep
working):

| Columns | Meaning |
|---|---|
| `G`, `experiment`, `phase`, `pv_step`, `input_path`, `output_path` | Where the image comes from and where its result is |
| `grains_px` … `fines_pct` | Pixel count and % of the image for Grains / Air / Water / Microplastic, of the final (filled) result |
| `processing_time_s`, `status`, `error` | Seconds, `ok` / `failed`, the error message |
| `app_version`, `backend` | Version, and what did the work (with any fallback) |
| `postprocess`, `particle_size_detected`, `pp_method` | Hole filling on / off, the size found, the method (circles / closing / off) |
| `mp_px_raw`, `mp_px_final` | Microplastic pixels before / after the filling |
| `circles_drawn`, `circles_skipped_overlap` | Circles drawn, and circles not drawn because of a dense cluster |

## 6. Hollow microplastic particles (filled automatically)

The model sometimes draws a microplastic particle as a red **ring** (red outline, Air or Water inside), mostly for
the larger 7 and 10 µm particles. Every result is corrected right after segmentation, before it is saved:

| Particle size | What is done |
|---|---|
| 10 µm | each ring-shaped particle is redrawn as its own **filled circle** (radius about 3.5 px) |
| 7 µm | the same, with the 7 µm size (radius about 2.8 px) |
| 5 µm | small gaps are closed (closing, radius 1 px) |
| size unknown | closing, radius 2 px |

- Only Air or Water pixels become microplastic — grains are never changed, no microplastic is removed.
- A circle is only drawn where the red pixels really form most of a ring; dense clusters are left as drawn.
- **The particle size** comes from the experiment folder name — `…-D05 (n)`, `…-D07 (n)`, `…-D10 (n)` — unless
  you choose one size for the run.

**Settings** (Run Pipeline → **Advanced**, before **Start Processing**):

| Setting | Default | Effect |
|---|---|---|
| Fill hollow microplastic particles | on | Off saves the model's result exactly as it is. |
| Particle size | Auto (from the folder name) | 5 / 7 / 10 µm for every image of the run. |
| Also save the raw (unfilled) segmentation | off | Also saves `Fused_St_Filt_Comp_Seg_Raw.tif` next to each result. |

## 7. Charts

Click **Charts**. It shows the latest run (or click **Open** to pick an earlier results folder). Use the filters
**G folder → Experiment → Phase → PV step**:
- **Per-Image Class Breakdown** — bars for the four classes of the chosen image, with the exact numbers.
- **Microplastic Across the Experiment Sequence** — the microplastic amount through the experiment:
  Before → IN PV 03 … → WT PV 03 …, with the exact numbers.

Switch between pixel count and % of image area at the top (**Units**). The charts read `summary.csv`, so they
always match the report.

## 8. Fix Segmentations (folders segmented with an older version)

Results folders made **before version 1.1.0** have hollow particles. This page fills them without segmenting again.

1. Click **Fix Segmentations** on the left.
2. **1. Folder with segmented files:** paste the results folder (`…_segmented_<date>_<time>`) and press Enter, or
   click **Browse**. The page shows how many result files it found.
3. **2. Output:** **Start a new run** writes a fixed copy to a **new** folder — by default on the Desktop,
   `<name>_holefilled_<date>_<time>` — with the same sub-folders and file names (**Change** picks another place).
   The chosen folder is not changed. The lab's manual segmentations are copied unchanged (tick *Also fill holes
   in the manual labels* only if you really want them changed — not recommended).
4. **3. Preview one file (optional):** choose a file and click **Preview** (before | after, zoomed on the most
   changed area).
5. **4. Run:** click **Start**. **Stop** ends after the current file; **Open output folder** opens the result.

`README_HOLEFILL.txt` in the new folder explains what was changed; `holefill_summary.csv` lists every file with
the size found and the method used. An interrupted fix: **2. Output → Continue an interrupted run**, **Browse**
to its folder, then **Continue run**. Folders segmented with 1.1.0 or later are already filled — the page says so.

## 9. Command line (optional)

The same work from a terminal, with the Python of the installation:

| What | Windows | Mac |
|---|---|---|
| Segment a lab folder | `.pv_env\Scripts\python tools\segment_lab_folder.py --input <lab folder>` | `.pv_env/bin/python tools/segment_lab_folder.py --input <lab folder>` |
| Fix an older results folder | `.pv_env\Scripts\python tools\fix_segmented_folder.py --input <results folder>` | `.pv_env/bin/python tools/fix_segmented_folder.py --input <results folder>` |

| `segment_lab_folder.py` option | Effect |
|---|---|
| `--output <folder>` | another output folder (default: next to the lab folder) |
| `--no-postprocess` | don't fill hollow particles |
| `--particle-size 5\|7\|10` | one size for the whole run (default: from the folder name) |
| `--save-raw` | also save `Fused_St_Filt_Comp_Seg_Raw.tif` |
| `--continue <results folder>` | continue an interrupted run (same settings required) |
| `--normal` | never use the Windows faster mode |

`fix_segmented_folder.py` options: `--output`, `--continue`, `--dry-run`, `--skip-list <file>`,
`--include-manual-labels`. Add `--help` to either tool for the full list.
