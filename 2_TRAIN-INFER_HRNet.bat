@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ---- Paths ----
set "HERE=%~dp0"
for %%I in ("%HERE%\..\..") do set "ROOT=%%~fI"
set "TOOL_DIR=%ROOT%\tools\2_Landmarking_v1.0"
set "PHOTOS_DIR=%ROOT%\photos"
set "LOG_DIR=%TOOL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

rem ---- Localities base picker (Windows dialog) ----
powershell -NoProfile -ExecutionPolicy Bypass -File "%TOOL_DIR%\scripts\choose_localities.ps1" -Silent
if errorlevel 1 (
  echo [WARN] Localities base picker failed or was cancelled.>>"%LOG_DIR%\trainer_last.log"
  echo [WARN] Localities base picker failed or was cancelled.
)

rem ---- Resolve PHOTOS_DIR from cfg\last_base.txt, if present ----
set "CFG_DIR=%TOOL_DIR%\cfg"
set "LAST_BASE=%CFG_DIR%\last_base.txt"
if exist "%LAST_BASE%" (
  set /p PHOTOS_DIR=<"%LAST_BASE%"
)
if not defined PHOTOS_DIR (
  set "PHOTOS_DIR=%ROOT%\photos"
)

rem ---- Python resolver ----
set "PY=%TOOL_DIR%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
  where.exe py >nul 2>&1 && (set "PY=py -3")
)
if /I "%PY%"=="py -3" (
  py -3 -c "import sys" >nul 2>&1 || set "PY="
)
if not defined PY (
  where.exe python >nul 2>&1 && (set "PY=python")
)
if not defined PY (
  echo [ERR] Landmarking environment not found.>>"%LOG_DIR%\trainer_last.log"
  echo [ERR] Landmarking environment not found.
  echo Run 0_INSTALL_ENV.ps1.
  pause
  exit /b 1
)

title == GM Landmarking: HRNet Trainer v1.0 ==
echo == GM Landmarking: HRNet Trainer v1.0 ==

if not exist "%PHOTOS_DIR%" (
  echo [ERR] Photos dir not found: %PHOTOS_DIR%
  pause
  exit /b 1
)

rem ---- Launch trainer menu (Python) ----
set "TR_LOG=%LOG_DIR%\trainer_menu_last.log"
%PY% "%TOOL_DIR%\scripts\trainer_menu.py" --root "%ROOT%" 1> "%TR_LOG%" 2>&1
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" (
  echo [ERR] Trainer menu exited with code %RC%. See "%TR_LOG%" below:
  type "%TR_LOG%" | more
  pause
)

exit /b 0
