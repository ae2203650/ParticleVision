#!/bin/bash
# Launches the ParticleVision Segmenter local UI.
#
# 1) If this folder has .pv_env (built by setup_mac.sh — the handoff / new-machine
#    setup), the app runs from it and opens the browser.
# 2) Otherwise it runs on the VERIFIED environment — Python 3.9.6 + torch 2.8.0 at
#    /usr/bin/python3 — the one that produced Fines IoU 0.4620 / 84.1% detection
#    and ~9 s/image. Deliberately NOT plain `python3`: inside a terminal where the
#    project's .venv (Python 3.12, editor checks only) is active, `python3` would
#    resolve to that instead.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

if [ -x ".pv_env/bin/python" ]; then
    exec .pv_env/bin/python tools/launch_app.py
fi

VERIFIED_PY=/usr/bin/python3

if ! "$VERIFIED_PY" -c "import streamlit, torch" 2>/dev/null; then
    echo "ERROR: $VERIFIED_PY is missing streamlit or torch — not the verified environment." >&2
    echo "       On a new computer, run ./setup_mac.sh first." >&2
    exit 1
fi
"$VERIFIED_PY" -c "import sys, torch, streamlit; print(f'ParticleVision Segmenter — Python {sys.version.split()[0]} ({sys.executable}), torch {torch.__version__}, streamlit {streamlit.__version__}')"
exec "$VERIFIED_PY" -m streamlit run app/segmenter_app.py
