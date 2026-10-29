@echo off
setlocal
title ParticleVision - full check
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".pv_env\Scripts\python.exe" goto nosetup
echo.
echo  Full check: segments the 3 Demo_Lab images and compares them with the Mac results.
echo  About 30 seconds with an NVIDIA graphics card, 3-9 minutes without one.
echo.
".pv_env\Scripts\python.exe" "check_install.py" --full
echo.
echo  The result is saved in selftest\full_check_log.txt
pause
exit /b 0

:nosetup
echo.
echo  ParticleVision is not installed on this computer yet.
echo  Double-click setup_windows.bat first. Instructions: README.md, details in docs\INSTALL_WINDOWS.md
echo.
pause
exit /b 1
