#!/bin/bash
# Double-click in Finder: full check — segments the 3 Demo_Lab images and
# compares them with the Mac results (about 30 s on Apple Silicon).
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
if [ -x .pv_env/bin/python ]; then
    .pv_env/bin/python check_install.py --full
    echo
    echo "The result is saved in selftest/full_check_log.txt"
else
    echo "ParticleVision is not installed on this Mac yet."
    echo "Double-click setup_mac.command first. Instructions: README.md, details in docs/INSTALL_MAC.md"
fi
echo
read -r -p "Press Enter to close this window ... " _
