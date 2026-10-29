@echo off
setlocal
title ParticleVision - faster mode setup (optional)
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".pv_env\Scripts\python.exe" goto nosetup
echo.
echo  Optional faster mode for PCs without an NVIDIA card.
echo  Step 1 installs 4 extra packages into ParticleVision (nothing else changes).
echo  Step 2 runs the benchmark automatically (several minutes).
echo.
".pv_env\Scripts\python.exe" "tools\setup_accel.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
pause
exit /b %RESULT%

:nosetup
echo.
echo  ParticleVision is not installed on this computer yet.
echo  Run setup_windows.bat first, then this file.
echo.
pause
exit /b 1
