@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
for %%A in ("%HERE%..") do set "TOOLS=%%~fA"
for %%A in ("%TOOLS%\..") do set "ROOT=%%~fA"
set "CFG=%HERE%cfg"
set "LAST_BASE_FILE=%CFG%\last_base.txt"
if not exist "%CFG%" mkdir "%CFG%"
if not exist "%HERE%logs" mkdir "%HERE%logs"

title GM Points Annotator v1.0 (folder picker, safe)

rem ---- Python ----
set "PY=%HERE%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
  where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
)

rem ==== args: --base / --select (опционально) ====
set "ARG_BASE="
set "ARG_SEL="
:parse
if "%~1"=="" goto after_args
if /i "%~1"=="--base"   (shift & if not "%~1"=="" (set "ARG_BASE=%~1") & shift & goto parse)
echo %~1 | findstr /i /r "^--base="   >nul && (for /f "tokens=2 delims==" %%Z in ("%~1") do set "ARG_BASE=%%~Z") & shift & goto parse
if /i "%~1"=="--select" (shift & if not "%~1"=="" (set "ARG_SEL=%~1") & shift & goto parse)
echo %~1 | findstr /i /r "^--select=" >nul && (for /f "tokens=2 delims==" %%Z in ("%~1") do set "ARG_SEL=%%~Z") & shift & goto parse
shift
goto parse
:after_args

set "BASE_DEFAULT=%ROOT%\photos"
if not defined ARG_BASE (
  if exist "%LAST_BASE_FILE%" (
    set /p ARG_BASE=<"%LAST_BASE_FILE%"
  ) else (
    set "ARG_BASE=%BASE_DEFAULT%"
  )
)

rem ==== если аргументы не заданы — показать диалог и сохранить выбор в файл ====
if "%~1"=="" if "%~2"=="" if "%~3"=="" (
  powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%scripts\PickFolder.ps1" -Title "Choose localities root (folder with <locality>\png)" -InitialPath "%ARG_BASE%" -OutFile "%LAST_BASE_FILE%"
  if errorlevel 1 (
    echo [INFO] Canceled by user.
    goto end
  )
  set "ARG_BASE="
  set /p ARG_BASE=<"%LAST_BASE_FILE%"
)

if not defined ARG_BASE (
  echo [ERR] Base path empty.
  goto end
)

echo.
echo == GM Landmarking: Points Annotator v1.0 ==
echo [BASE] %ARG_BASE%
if not exist "%ARG_BASE%" (
  echo [ERR] Folder not found: "%ARG_BASE%"
  goto end
)

rem ---- формируем список локальностей надёжно (без FOR /F) ----
set "LISTTMP=%TEMP%\locs_%RANDOM%.txt"
del /q "%LISTTMP%" 2>nul
set /a IDX=0

for /d %%D in ("%ARG_BASE%\*") do (
  if exist "%%~fD\png\*.png" (
    set "LOC=%%~nxD"
    set "PNGDIR=%%~fD\png"
    for /f %%c in ('dir /b /a-d "%%~fD\png\*.png" 2^>nul ^| find /c /v ""') do set "NUMPNG=%%c"
    if not defined NUMPNG set "NUMPNG=0"
    for /f %%c in ('dir /b /a-d "%%~fD\png\*.csv" 2^>nul ^| find /c /v ""') do set "NUMCSV=%%c"
    if not defined NUMCSV set "NUMCSV=0"
    set /a PCT=0
    if not "!NUMPNG!"=="0" set /a PCT=(100*!NUMCSV!)/!NUMPNG!

    set "FIRSTPNG="
    for /f "usebackq delims=" %%F in (`dir /b /a-d /on "%%~fD\png\*.png" 2^>nul`) do if not defined FIRSTPNG set "FIRSTPNG=%%~nF"

    set "SCALE="
    if defined FIRSTPNG (
      if not exist "%%~fD\png\!FIRSTPNG!.scale.csv" set "SCALE=Set Scale!"
    )

    set /a IDX+=1
    >>"%LISTTMP%" echo !IDX!^|!LOC!^|!NUMPNG!^|!NUMCSV!^|!PCT!^|!SCALE!^|!PNGDIR!
    set "NUMPNG=" & set "NUMCSV=" & set "PCT=" & set "FIRSTPNG=" & set "SCALE="
  )
)

if not exist "%LISTTMP%" (
  echo [ERR] No localities with \png found in "%ARG_BASE%".
  goto end
)

set "PNG=" & set "LOCNAME="

if defined ARG_SEL (
  for /f "usebackq tokens=1-7 delims=|" %%A in ("%LISTTMP%") do (
    if "%%A"=="%ARG_SEL%" (set "PNG=%%G" & set "LOCNAME=%%B")
  )
  if not defined PNG echo [ERR] Invalid --select %ARG_SEL%.
)

if not defined PNG (
  echo.
  echo Localities under: "%ARG_BASE%"
  echo.
  for /f "usebackq tokens=1-7 delims=|" %%A in ("%LISTTMP%") do (
    set "idx=%%A" & set "loc=%%B" & set "nimg=%%C" & set "nann=%%D" & set "pct=%%E" & set "sc=%%F"
    setlocal EnableDelayedExpansion
    if defined sc (echo  !idx!) !loc! [!nimg!/!nann!] !pct!%%  !sc!) else (echo  !idx!) !loc! [!nimg!/!nann!] !pct!%%)
    endlocal
  )
  echo.
  set "SEL="
  set /p SEL=Enter number (Q=quit): 
  if /i "%SEL%"=="Q" goto end
  for /f "usebackq tokens=1-7 delims=|" %%A in ("%LISTTMP%") do (
    if "%%A"=="%SEL%" (set "PNG=%%G" & set "LOCNAME=%%B")
  )
)

del /q "%LISTTMP%" 2>nul

if "%PNG%"=="" (
  echo [ERR] Invalid selection.
  goto end
)

echo [RUN] locality: %LOCNAME%
echo [DIR] images:  "%PNG%"
echo [ROOT] project: "%ROOT%"
echo.

"%PY%" "%HERE%\annot_gui_custom.py" --root "%ROOT%" --images "%PNG%"
set "EXITCODE=%ERRORLEVEL%"

> "%HERE%\logs\annotator_run_last.txt" echo RUN_TS=%DATE% %TIME%
>>"%HERE%\logs\annotator_run_last.txt" echo BASE=%ARG_BASE%
>>"%HERE%\logs\annotator_run_last.txt" echo LOC=%LOCNAME%
>>"%HERE%\logs\annotator_run_last.txt" echo PNG=%PNG%
>>"%HERE%\logs\annotator_run_last.txt" echo EXIT=%EXITCODE%

:end
exit /b 0