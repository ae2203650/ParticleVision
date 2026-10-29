@echo off
setlocal
title ParticleVision
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".pv_env\Scripts\python.exe" goto nosetup
".pv_env\Scripts\python.exe" "tools\launch_app.py"
echo.
echo  ParticleVision has stopped.
pause
exit /b 0

:nosetup
echo.
echo  ParticleVision is not installed on this computer yet.
echo  Double-click setup_windows.bat first. Instructions: README.md, details in docs\INSTALL_WINDOWS.md
echo.
pause
exit /b 1
