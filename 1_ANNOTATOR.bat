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

rem ---- Menu of localities (done/total, pct, Set Scale!) ----
echo.
for /f "usebackq delims=" %%L in (`"%PY%" "%TOOL_DIR%\scripts\menu_list.py"`) do set "SEL_LOC=%%L"

if not defined SEL_LOC (
  echo [INFO] No locality selected. Exiting...
  goto :EOF
)

echo [INFO] Selected locality: !SEL_LOC!
set "PNG_DIR=%PHOTOS_DIR%\!SEL_LOC!\png"

if not exist "!PNG_DIR!" (
  echo [ERR] Locality path not found: !PNG_DIR!
  pause
  exit /b 2
)

rem TODO: engine select (custom/std) and GUI launch will be added next
echo.
echo [INFO] Placeholder: next step is GUI launch for "!SEL_LOC!" (custom engine).
echo [INFO] Press any key to exit...
pause >nul
exit /b 0