"""
OPTIONAL faster mode — run the two networks through ONNX Runtime (DirectML on the
graphics card, or ONNX Runtime CPU) instead of PyTorch. Off unless benchmark_accel
proved it on this exact machine: same results as the Mac, and faster than the normal
path measured in the same benchmark run.

What is NOT different from the normal path:
  • the networks: the same weights, exported once to float32 ONNX (accel/*.onnx,
    MD5s in accel/manifest.json); no quantization, no half precision;
  • everything around them: the unchanged src/inference.predict_full_image does the
    tiling (512 / 64), stitching, gated fusion (margin 2.0, scale 8.0) and colours —
    only the two network calls go to the ONNX runtime (OnnxNet below).

When it is used:
  • accel_result.json (written by tools/benchmark_accel.py) says a backend PASSED the
    comparison with the Mac references on THIS machine — same computer, Python, runtime
    versions, ONNX files and, for a GPU, the same graphics card and driver;
  • and the user has not switched it off (sidebar switch, or PV_FORCE_NORMAL=1).
Otherwise the app uses the normal path (model_runtime.SegmentationRunner), unchanged.

During a run, any error on the fast path (including out-of-memory) → that image is
redone on the normal path, with a note in the log and run_log.txt.
"""

import hashlib
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np
import torch

import model_runtime as mr
import src.inference as inference  # called via the module, exactly like model_runtime

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCEL_DIR = os.path.join(ROOT, "accel")
MANIFEST = os.path.join(ACCEL_DIR, "manifest.json")
# PV_ACCEL_RESULT: tests point this elsewhere so they never write into the project folder
RESULT_FILE = os.environ.get("PV_ACCEL_RESULT") or os.path.join(ROOT, "accel_result.json")
MODEL_FILES = {"unet": "unet_fp32.onnx", "fines": "fines_fp32.onnx"}

BACKEND_NAMES = {"dml": "DirectML", "ort-cpu": "ONNX Runtime CPU"}


def valid_backend(name):
    """'ort-cpu' or 'dml:<adapter number>'."""
    return name == "ort-cpu" or (
        isinstance(name, str) and name.startswith("dml:") and name[4:].isdigit()
    )


# ─────────────────────────────────────────────────────────────────────────
# Machine / hardware description (for the benchmark header and the fingerprint)
# ─────────────────────────────────────────────────────────────────────────
def _run(cmd, timeout=30):
    kw = {"creationflags": 0x08000000} if os.name == "nt" else {}  # CREATE_NO_WINDOW
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, **kw)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception:  # noqa: BLE001
        return ""


def _powershell_json(query):
    out = _run(
        [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            f"{query} | ConvertTo-Json -Compress",
        ]
    )
    try:
        data = json.loads(out) if out else []
    except ValueError:
        return []
    return data if isinstance(data, list) else [data]


def cpu_model():
    if os.name == "nt":
        rows = _powershell_json("Get-CimInstance Win32_Processor | Select-Object Name")
        if rows and rows[0].get("Name"):
            return " ".join(rows[0]["Name"].split())
    if sys.platform == "darwin":
        return _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    return platform.processor() or platform.machine()


def os_version():
    if os.name == "nt":
        rows = _powershell_json(
            "Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber"
        )
        if rows:
            r = rows[0]
            return f"{r.get('Caption', 'Windows')} (version {r.get('Version', '?')}, build {r.get('BuildNumber', '?')})"
    if sys.platform == "darwin":
        return f"macOS {_run(['sw_vers', '-productVersion'])}"
    return platform.platform()


def windows_gpus():
    """[{name, driver_version, driver_date}] for every display adapter (Windows only)."""
    if os.name != "nt":
        return []
    rows = _powershell_json(
        "Get-CimInstance Win32_VideoController | Select-Object Name,DriverVersion,DriverDate"
    )
    return [
        {
            "name": (r.get("Name") or "").strip(),
            "driver_version": r.get("DriverVersion") or "",
            "driver_date": str(r.get("DriverDate") or "")[:10],
        }
        for r in rows
        if r.get("Name")
    ]


def dxgi_adapters():
    """Graphics adapters in the order DirectML numbers them (DXGI EnumAdapters1):
    [{index, name, vendor_id, dedicated_mb, software}]. Windows only; [] elsewhere."""
    if os.name != "nt":
        return []
    try:
        import ctypes
        from ctypes import wintypes

        class GUID(ctypes.Structure):
            _fields_ = [
                ("Data1", wintypes.DWORD),
                ("Data2", wintypes.WORD),
                ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8),
            ]

        class LUID(ctypes.Structure):
            _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]

        class DXGI_ADAPTER_DESC1(ctypes.Structure):
            _fields_ = [
                ("Description", wintypes.WCHAR * 128),
                ("VendorId", wintypes.UINT),
                ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT),
                ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t),
                ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t),
                ("AdapterLuid", LUID),
                ("Flags", wintypes.UINT),
            ]

        # IID_IDXGIFactory1 = {770aae78-f26f-4dba-a829-253c83d1b387}
        iid = GUID(
            0x770AAE78,
            0xF26F,
            0x4DBA,
            (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87),
        )
        factory = ctypes.c_void_p()
        if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(iid), ctypes.byref(factory)) != 0:
            return []

        def method(obj, index, restype, *argtypes):
            vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
            return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(vtbl[index])

        adapters = []
        i = 0
        while i < 16:
            adapter = ctypes.c_void_p()
            # IDXGIFactory1::EnumAdapters1 = vtable slot 12
            if (
                method(factory, 12, ctypes.c_long, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(
                    factory, i, ctypes.byref(adapter)
                )
                != 0
            ):
                break
            desc = DXGI_ADAPTER_DESC1()
            # IDXGIAdapter1::GetDesc1 = vtable slot 10
            method(adapter, 10, ctypes.c_long, ctypes.POINTER(DXGI_ADAPTER_DESC1))(
                adapter, ctypes.byref(desc)
            )
            adapters.append(
                {
                    "index": i,
                    "name": desc.Description.strip(),
                    "vendor_id": desc.VendorId,
                    "dedicated_mb": int(desc.DedicatedVideoMemory // (1024 * 1024)),
                    "software": bool(desc.Flags & 2),
                }
            )
            method(adapter, 2, ctypes.c_ulong)(adapter)  # Release
            i += 1
        method(factory, 2, ctypes.c_ulong)(factory)
        return adapters
    except Exception:  # noqa: BLE001
        return []


def driver_for(adapter_name, gpus=None):
    gpus = windows_gpus() if gpus is None else gpus
    for g in gpus:
        if g["name"].lower() == adapter_name.lower():
            return g["driver_version"]
    return ""


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def runtime_versions():
    v = {}
    try:
        import onnxruntime as ort

        v["onnxruntime"] = ort.__version__
        v["onnxruntime_providers"] = ort.get_available_providers()
    except Exception:  # noqa: BLE001
        v["onnxruntime"] = None
    return v


def onnx_files_ok():
    """(ok, reason): both .onnx files present with the MD5s recorded at export."""
    try:
        man = json.load(open(MANIFEST, encoding="utf-8"))
    except (OSError, ValueError):
        return False, "accel/manifest.json is missing"
    for key, fn in MODEL_FILES.items():
        p = os.path.join(ACCEL_DIR, fn)
        if not os.path.isfile(p):
            return False, f"accel/{fn} is missing"
        if md5(p) != man["models"][key]["md5"]:
            return False, f"accel/{fn} is not the exported file (MD5 differs)"
    return True, ""


def fingerprint(backend, gpus=None, adapters=None):
    """What must be identical for a benchmark result to count on this machine."""
    fp = {
        "host": platform.node(),
        "system": platform.system(),
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cpu": cpu_model(),
        "os": os_version(),
        "onnx_md5": {
            k: md5(os.path.join(ACCEL_DIR, f))
            for k, f in MODEL_FILES.items()
            if os.path.isfile(os.path.join(ACCEL_DIR, f))
        },
    }
    fp["onnxruntime"] = runtime_versions()["onnxruntime"]
    if backend.startswith("dml:"):
        idx = int(backend.split(":")[1])
        adapters = dxgi_adapters() if adapters is None else adapters
        name = next((a["name"] for a in adapters if a["index"] == idx), f"adapter {idx}")
        fp["gpu"] = name
        fp["gpu_driver"] = driver_for(name, gpus)
    return fp


# ─────────────────────────────────────────────────────────────────────────
# The networks, through ONNX Runtime (DirectML or CPU)
# ─────────────────────────────────────────────────────────────────────────
class OnnxNet:
    """Stands in for the PyTorch network inside src/inference.predict_full_image:
    called with a float32 (N, 3, 512, 512) tensor, returns the logits as a CPU tensor."""

    def __init__(self, run_fn, name):
        self._run = run_fn
        self.name = name

    def eval(self):
        return self

    def __call__(self, x):
        arr = np.ascontiguousarray(x.detach().cpu().numpy(), dtype=np.float32)
        return torch.from_numpy(self._run(arr))


def _ort_net(path, backend):
    import onnxruntime as ort

    so = ort.SessionOptions()
    if backend.startswith("dml:"):
        so.enable_mem_pattern = False  # required by DirectML
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL  # required by DirectML
        providers = [("DmlExecutionProvider", {"device_id": int(backend.split(":")[1])})]
        want = "DmlExecutionProvider"
    else:
        so.enable_cpu_mem_arena = False  # give memory back between calls (no effect on results)
        providers = ["CPUExecutionProvider"]
        want = "CPUExecutionProvider"
    sess = ort.InferenceSession(path, sess_options=so, providers=providers)
    if sess.get_providers()[0] != want:
        raise RuntimeError(f"{want} is not available (got {sess.get_providers()})")
    out = sess.get_outputs()[0].name
    inp = sess.get_inputs()[0].name
    return OnnxNet(lambda a: sess.run([out], {inp: a})[0], os.path.basename(path))


def make_nets(backend):
    if not valid_backend(backend):
        raise ValueError(f"unknown backend {backend!r}")
    paths = {k: os.path.join(ACCEL_DIR, f) for k, f in MODEL_FILES.items()}
    return _ort_net(paths["unet"], backend), _ort_net(paths["fines"], backend)


def backend_label(backend, adapters=None):
    if backend.startswith("dml:"):
        idx = int(backend.split(":")[1])
        adapters = dxgi_adapters() if adapters is None else adapters
        name = next((a["name"] for a in adapters if a["index"] == idx), f"graphics adapter {idx}")
        return f"{name} (DirectML)"
    return f"CPU ({BACKEND_NAMES[backend]})"


class AccelRunner:
    """Same interface as model_runtime.SegmentationRunner (predict / between_images /
    label / device), but the two networks run through ONNX Runtime (DirectML or CPU)."""

    is_accel = True

    def __init__(self, backend, label=None, expected_sec=None):
        self.backend = backend
        self.device = torch.device("cpu")  # tensors handed to the ONNX networks live on the CPU
        self.on_cpu = not backend.startswith("dml:")
        self.label = label or backend_label(backend)
        self.stats_key = f"accel:{backend}"
        self.expected_sec = expected_sec
        self.m1, self.m2 = make_nets(backend)
        self._normal = None
        self.fallbacks = 0

    def normal_runner(self):
        if self._normal is None:
            self._normal = mr.SegmentationRunner()
        return self._normal

    def predict_fast(self, path):
        """The fast path only — no fallback (used by the benchmark)."""
        return inference.predict_full_image(
            model=self.m1,
            image_path=path,
            device=torch.device("cpu"),
            model_b2=self.m2,
            batch_size=mr.DEFAULT_BATCH,
            fusion_scale=mr.FUSION_SCALE,
            fusion_margin=mr.FUSION_MARGIN,
        )

    def predict(self, path):
        """Returns (rgb, class_map, seconds, note), like SegmentationRunner.predict."""
        t0 = time.time()
        try:
            rgb, cm, _ = self.predict_fast(path)
            return rgb, cm, time.time() - t0, ""
        except Exception as e:  # noqa: BLE001 — incl. out-of-memory
            self.fallbacks += 1
            rgb, cm, _, note = self.normal_runner().predict(path)
            why = f"{type(e).__name__}: {str(e).splitlines()[0][:150] if str(e) else ''}"
            return (
                rgb,
                cm,
                time.time() - t0,
                (
                    f"faster mode failed on this image ({why}) — redone on the "
                    f"normal path" + (f"; {note}" if note else "")
                ),
            )

    def between_images(self):
        if self._normal is not None:
            self._normal.between_images()


# ─────────────────────────────────────────────────────────────────────────
# Should the app use it?
# ─────────────────────────────────────────────────────────────────────────
def force_normal_env():
    return os.environ.get("PV_FORCE_NORMAL", "").strip() in ("1", "true", "yes")


def load_result(path=RESULT_FILE):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def check_result(result, current_fp=None):
    """(backend or None, reason). backend only if the benchmark passed on THIS machine."""
    if not result:
        return None, ""
    backend = result.get("chosen")
    if not backend:
        return None, "the faster-mode benchmark found no backend that passed and was faster"
    if not valid_backend(backend):
        return None, f"unknown backend {backend!r} in accel_result.json"
    ok, why = onnx_files_ok()
    if not ok:
        return None, why
    recorded = result.get("machine") or {}
    now = current_fp if current_fp is not None else fingerprint(backend)
    names = {
        "host": "computer",
        "python_executable": "Python",
        "torch": "PyTorch",
        "onnx_md5": "ONNX files",
        "onnxruntime": "ONNX Runtime version",
        "gpu": "graphics card",
        "gpu_driver": "graphics driver",
        "cpu": "processor",
        "system": "operating system",
        "os": "Windows version",
    }
    for key, label in names.items():
        if key in recorded or key in now:
            if recorded.get(key) != now.get(key):
                return None, (
                    f"the {label} changed since benchmark_accel ran "
                    f"({recorded.get(key)!s:.60} → {now.get(key)!s:.60}) — run benchmark_accel again"
                )
    b = (result.get("backends") or {}).get(backend) or {}
    if not b.get("passed"):
        return None, "the chosen backend did not pass the benchmark"
    normal = (result.get("backends") or {}).get("normal") or {}
    if not (
        normal.get("passed")
        and normal.get("sec_per_image")
        and b.get("sec_per_image")
        and b["sec_per_image"] < normal["sec_per_image"]
    ):
        return None, "the benchmark did not show it faster than the normal path on this computer"
    return backend, ""


def select_runner(force_normal=False, result_path=RESULT_FILE):
    """(AccelRunner or None, status). None → the app uses the normal path.
    With no accel_result.json the answer is (None, "") and nothing else is imported."""
    if force_normal or force_normal_env():
        return None, "Faster mode switched off — normal path"
    result = load_result(result_path)
    if result is None:
        return None, ""
    backend, why = check_result(result)
    if backend is None:
        return None, f"Faster mode not used: {why}"
    try:
        b = result["backends"][backend]
        runner = AccelRunner(
            backend,
            label=f"{b.get('label') or backend_label(backend)}, verified",
            expected_sec=b.get("sec_per_image"),
        )
    except Exception as e:  # noqa: BLE001
        return None, f"Faster mode not used: could not start {backend} ({type(e).__name__}: {e})"
    return runner, ""
