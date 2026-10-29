@echo off
setlocal
title ParticleVision - faster mode benchmark
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

if not exist ".pv_env\Scripts\python.exe" goto nosetup
echo.
echo  Faster-mode benchmark: 4 test images through the normal path and every
echo  available faster backend, each compared with the Mac results.
echo  This takes several minutes. Keep the laptop plugged in.
echo.
".pv_env\Scripts\python.exe" "tools\benchmark_accel.py" %*
set "RESULT=%ERRORLEVEL%"
echo.
echo  Take a screenshot of this window and send it, with accel_benchmark_log.txt
echo  and accel_result.json from the ParticleVision folder.
echo.
pause
exit /b %RESULT%

:nosetup
echo.
echo  ParticleVision is not installed on this computer yet.
echo  Run setup_windows.bat first.
echo.
pause
exit /b 1
