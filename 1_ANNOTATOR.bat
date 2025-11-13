@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
for %%A in ("%HERE%..") do set "TOOLS=%%~fA"
for %%A in ("%TOOLS%\..") do set "ROOT=%%~fA"
set "CFG=%HERE%cfg"
set "LAST_BASE_FILE=%CFG%\last_base.txt"
if not exist "%CFG%" mkdir "%CFG%"

title GM Points Annotator v1.0 (folder picker)

rem ---- Python ----
set "PY=%HERE%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
  where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
)

rem ==== args (сохраняем совместимость): --base, --select ====
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

rem ==== базовый путь: либо из аргумента, либо из памяти, либо ROOT\photos ====
set "BASE_DEFAULT=%ROOT%\photos"
set "BASE_MEM="
if exist "%LAST_BASE_FILE%" (
  set /p BASE_MEM=<"%LAST_BASE_FILE%"
)
if not defined ARG_BASE (
  if defined BASE_MEM (set "ARG_BASE=%BASE_MEM%") else (set "ARG_BASE=%BASE_DEFAULT%")
)

rem ==== если базовый путь не задан явно — предлагаем диалог выбора папки ====
if "%~1"=="" if "%~2"=="" if "%~3"=="" (
  rem только если не переданы аргументы
  for /f "usebackq delims=" %%P in (`powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%scripts\PickFolder.ps1" -Title "Choose localities root (folder with <locality>\png)" -InitialPath "%ARG_BASE%"`) do (
    set "ARG_BASE=%%P"
  )
)

if not defined ARG_BASE (
  echo [INFO] Canceled by user.
  goto end
)

echo.
echo == GM Landmarking: Points Annotator v1.0 ==
echo [BASE] %ARG_BASE%

if not exist "%ARG_BASE%" (
  echo [ERR] Folder not found: "%ARG_BASE%"
  goto end
)

rem ---- сохраняем выбранный путь как последний ----
> "%LAST_BASE_FILE%" echo %ARG_BASE%

rem ---- собираем локальности (подпапки, где есть \png и *.png) ----
setlocal DisableDelayedExpansion
set "LISTTMP=%TEMP%\locs_%RANDOM%.txt"
del /q "%LISTTMP%" 2>nul
set /a IDX=0

for /f "delims=" %%D in ('dir /b /ad "%ARG_BASE%"') do (
  set "LOC=%%D"
  setlocal EnableDelayedExpansion
  set "PNGDIR=%ARG_BASE%\!LOC!\png"
  if exist "!PNGDIR!" (
    for /f %%c in ('dir /b /a-d "!PNGDIR!\*.png" 2^>nul ^| find /c /v ""') do set "NUMPNG=%%c"
    if "!NUMPNG!"=="" set "NUMPNG=0"
    for /f %%c in ('dir /b /a-d "!PNGDIR!\*.csv" 2^>nul ^| find /c /v ""') do set "NUMCSV=%%c"
    if "!NUMCSV!"=="" set "NUMCSV=0"
    if !NUMPNG! GTR 0 (set /a PCT=(100*!NUMCSV!)/!NUMPNG!) else (set "PCT=0")

    set "FIRSTPNG="
    for /f "usebackq delims=" %%F in (`dir /b /a-d /on "!PNGDIR!\*.png" 2^>nul`) do if not defined FIRSTPNG set "FIRSTPNG=%%~nF"

    set "SCALEFLAG="
    if defined FIRSTPNG (
      if exist "!PNGDIR!\!FIRSTPNG!.scale.csv" (
        rem ok
      ) else (
        set "SCALEFLAG=  Set Scale!"
      )
    )

    set /a IDX+=1
    >>"%LISTTMP%" echo !IDX!^|!LOC!^|!NUMPNG!^|!NUMCSV!^|!PCT!!SCALEFLAG!^|!PNGDIR!
  )
  endlocal
)

if not exist "%LISTTMP%" (
  echo [ERR] No localities (with \png) found in "%ARG_BASE%".
  goto end
)

set "PNG="
set "LOCNAME="

if defined ARG_SEL (
  for /f "usebackq tokens=1-6 delims=|" %%A in ("%LISTTMP%") do (
    if "%%A"=="%ARG_SEL%" (set "PNG=%%F" & set "LOCNAME=%%B")
  )
  if not defined PNG (
    echo [ERR] Invalid --select %ARG_SEL%. Showing menu...
  )
)

if not defined PNG (
  echo.
  echo Localities under: "%ARG_BASE%"
  echo.
  for /f "usebackq tokens=1-6 delims=|" %%A in ("%LISTTMP%") do (
    set "idx=%%A" & set "loc=%%B" & set "nimg=%%C" & set "nann=%%D" & set "pct=%%E"
    setlocal EnableDelayedExpansion
    echo  !idx!) !loc! [!nimg!/!nann!] !pct!%%!SCALEFLAG!
    endlocal
  )
  echo.
  set "SEL="
  set /p SEL=Enter number (Q=quit): 
  if /i "%SEL%"=="Q" goto end
  for /f "usebackq tokens=1-6 delims=|" %%A in ("%LISTTMP%") do (
    if "%%A"=="%SEL%" (set "PNG=%%F" & set "LOCNAME=%%B")
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

rem ---- короткий лог запуска ----
if not exist "%HERE%\logs" mkdir "%HERE%\logs"
> "%HERE%\logs\annotator_run_last.txt" echo RUN_TS=%DATE% %TIME%
>>"%HERE%\logs\annotator_run_last.txt" echo BASE=%ARG_BASE%
>>"%HERE%\logs\annotator_run_last.txt" echo LOC=%LOCNAME%
>>"%HERE%\logs\annotator_run_last.txt" echo PNG=%PNG%
>>"%HERE%\logs\annotator_run_last.txt" echo EXIT=%EXITCODE%

:end
exit /b 0