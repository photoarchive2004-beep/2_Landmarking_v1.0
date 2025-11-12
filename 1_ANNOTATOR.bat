@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
for %%I in ("%HERE%\..\..") do set "ROOT=%%~fI"
set "TOOL_DIR=%ROOT%\tools\2_Landmarking_v1.0"
set "PHOTOS_DIR=%ROOT%\photos"
set "LOG_DIR=%TOOL_DIR%\logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%" >nul 2>&1

set "PY=%TOOL_DIR%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" set "PY=py -3"
where %PY% >nul 2>nul
if errorlevel 1 (
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

rem TODO: replace stub with real GUI launch per NP (engine select, TPS-Utils CSV, QC, scale)
echo [INFO] ROOT      = %ROOT%
echo [INFO] TOOL_DIR  = %TOOL_DIR%
echo [INFO] PHOTOS    = %PHOTOS_DIR%
echo [INFO] LM_number = 
type "%TOOL_DIR%\LM_number.txt" 2>nul || echo [WARN] LM_number.txt missing

echo.
echo [INFO] Stub annotator is running. Press any key to exit...
pause >nul
exit /b 0
