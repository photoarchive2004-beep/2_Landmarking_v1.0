@echo off
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul

rem ---- Paths (relative) ----
set "HERE=%~dp0"
for %%I in ("%HERE%\..\..") do set "ROOT=%%~fI"
set "TOOL_DIR=%ROOT%\tools\2_Landmarking_v1.0"
set "PHOTOS_DIR=%ROOT%\photos"
set "LOG_DIR=%TOOL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

rem ---- Python resolver (venv -> py -3 -> python) ----
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
  echo [ERR] Landmarking environment not found.>>"%LOG_DIR%\annotator_last.log"
  echo [ERR] Landmarking environment not found.
  echo [HINT] Run 0_INSTALL_ENV.ps1 first.
  pause
  exit /b 1
)

title == GM Landmarking: Points Annotator v1.0 ==
echo == GM Landmarking: Points Annotator v1.0 ==

if not exist "%PHOTOS_DIR%" (
  echo [ERR] Photos dir not found: %PHOTOS_DIR%
  pause
  exit /b 1
)

rem ---- Print menu (non-interactive) ----
echo.
%PY% "%TOOL_DIR%\scripts\menu_list.py" --print --root "%ROOT%"

rem ---- Ask number in BAT ----
set "SEL_LOC="
set /p CH=Enter number (or Q to quit): 
if /I "!CH!"=="Q" goto :EOF

rem ---- Validate selection via Python -> temp file (avoid FOR /F issues) ----
set "TMP_SEL=%TEMP%\gm_sel_%RANDOM%.txt"
%PY% "%TOOL_DIR%\scripts\menu_list.py" --pick !CH! --root "%ROOT%" 1> "!TMP_SEL!" 2> "%LOG_DIR%\menu_pick_last.err"
if exist "!TMP_SEL!" (
  set /p SEL_LOC=<"!TMP_SEL!"
  del /q "!TMP_SEL!" 2>nul
)

if not defined SEL_LOC (
  echo [ERR] Invalid selection.
  pause
  exit /b 2
)

echo [INFO] Selected locality: !SEL_LOC!
set "PNG_DIR=%PHOTOS_DIR%\!SEL_LOC!\png"
if not exist "!PNG_DIR!" (
  echo [ERR] Locality path not found: !PNG_DIR!
  pause
  exit /b 2
)

rem TODO: engine select + GUI launch next
echo.
echo [INFO] Placeholder: next step is GUI launch for "!SEL_LOC!" (custom engine).
echo [INFO] Press any key to exit...
pause >nul
exit /b 0