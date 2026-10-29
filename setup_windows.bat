@echo off
setlocal
title ParticleVision setup
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1

echo.
echo  ParticleVision setup
echo  This installs everything ParticleVision needs. It takes 5 to 20 minutes.
echo  Please keep this window open until it says SUCCESS or FAILED.
echo.

set "PY="
py -3.12 -c "import sys" >nul 2>&1 && set "PY=py -3.12"
if defined PY goto found
python -c "import sys; sys.exit(0 if sys.version_info[:2]==(3,12) and sys.prefix==sys.base_prefix else 1)" >nul 2>&1 && set "PY=python"
if defined PY goto found

echo  ======================================================================
echo    SETUP FAILED - Python 3.12 was not found on this computer.
echo  ======================================================================
echo    Install Python 3.12 from https://www.python.org/downloads/
echo    On the first installer screen, tick "Add python.exe to PATH".
echo    Then double-click setup_windows.bat again.
echo    Step-by-step instructions: README.md, details in docs\INSTALL_WINDOWS.md
echo.
pause
exit /b 1

:found
%PY% "%~dp0tools\setup_env.py"
set "RESULT=%ERRORLEVEL%"
echo.
if "%RESULT%"=="0" goto ok
echo  Setup did not finish. Read the messages above.
echo  The full log is setup_log.txt in the ParticleVision folder.
echo.
pause
exit /b %RESULT%

:ok
echo  You can close this window now and start the app with run_app.bat
echo.
pause
exit /b 0
