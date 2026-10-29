#!/bin/bash
# ParticleVision setup for macOS — builds .pv_env with Python 3.12 and runs the self-test.
# Run it from Terminal (./setup_mac.sh) or double-click setup_mac.command.
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8

PY=""
# python.org install first (what docs/INSTALL_MAC.md asks for), then Homebrew, then PATH.
# An interpreter inside a virtual environment (e.g. an activated .venv) is skipped.
for c in /Library/Frameworks/Python.framework/Versions/3.12/bin/python3.12 \
         /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3.12; do
    if command -v "$c" >/dev/null 2>&1 && \
       "$c" -c 'import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) and sys.prefix == sys.base_prefix else 1)' 2>/dev/null; then
        PY="$c"; break
    fi
done
if [ -z "$PY" ]; then
    echo "======================================================================"
    echo "  SETUP FAILED - Python 3.12 was not found on this Mac."
    echo "======================================================================"
    echo "  Install Python 3.12 from https://www.python.org/downloads/macos/"
    echo "  (the 'macOS 64-bit universal2 installer'), then run setup again."
    echo "  Step-by-step instructions: README.md, details in docs/INSTALL_MAC.md"
    exit 1
fi
"$PY" tools/setup_env.py
