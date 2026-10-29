"""
Start the ParticleVision app and open it in the browser.

Used by run_app.bat (Windows) and run_app.sh (when .pv_env exists). It:
  • picks a free port (8501, 8502, … if 8501 is busy),
  • starts Streamlit listening on this computer only (127.0.0.1 — no Windows
    Firewall question, not reachable from the network),
  • waits until the app answers, then opens the default browser,
  • keeps running until this window is closed (or Ctrl+C); however it is
    stopped, it stops the app with it (no app left running in the background).
"""

import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def free_port(start=8501, tries=50):
    for port in range(start, start + tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                continue
    raise SystemExit(
        "No free port found between 8501 and 8550 — close other programs and try again."
    )


def main():
    port = free_port()
    url = f"http://localhost:{port}"
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1", HF_HUB_OFFLINE="1")
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        os.path.join(ROOT, "app", "segmenter_app.py"),
        "--server.port",
        str(port),
        "--server.address",
        "127.0.0.1",
        "--server.headless",
        "true",
        "--browser.gatherUsageStats",
        "false",
    ]
    print("Starting ParticleVision ... (the first start can take up to a minute)", flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env=env)
    try:
        return run(proc, port, url)
    finally:
        stop(proc)


def _on_signal(signum, frame):
    raise KeyboardInterrupt


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


def run(proc, port, url):
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):  # window closed / stopped by the system
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), _on_signal)
    health = f"http://127.0.0.1:{port}/_stcore/health"
    for _ in range(240):  # up to 2 minutes
        if proc.poll() is not None:
            print("\nThe app stopped while starting — see the messages above.", flush=True)
            return proc.returncode or 1
        try:
            with urllib.request.urlopen(health, timeout=1) as r:
                if r.status == 200:
                    break
        except Exception:  # noqa: BLE001 — not up yet
            time.sleep(0.5)
    else:
        print(f"\nThe app did not answer in time. Try opening {url} in your browser.", flush=True)
    webbrowser.open(url)
    print("\n" + "=" * 70)
    print(f"  ParticleVision is running:  {url}")
    print("  Your browser should open by itself. If it doesn't, open that address.")
    print("  Keep this window open while you use the app. Close it to stop the app.")
    print("=" * 70 + "\n", flush=True)
    try:
        return proc.wait()
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
