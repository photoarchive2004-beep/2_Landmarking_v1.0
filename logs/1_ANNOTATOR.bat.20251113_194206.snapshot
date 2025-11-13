@echo off
chcp 65001 >nul
setlocal EnableExtensions EnableDelayedExpansion

set "HERE=%~dp0"
for %%A in ("%HERE%..") do set "TOOLS=%%~fA"
for %%A in ("%TOOLS%\..") do set "ROOT=%%~fA"
title GM Points Annotator v1.0 (choose localities root)

rem ---- Python ----
set "PY=%HERE%\.venv_lm\Scripts\python.exe"
if not exist "%PY%" (
  where py >nul 2>nul && (set "PY=py -3") || (set "PY=python")
)

echo.
echo == GM Landmarking: Points Annotator v1.0 ==
echo.

rem ---- спросить корень локальностей ----
set "BASE_DEFAULT=%ROOT%\photos"
set "BASE_INPUT="
set /p BASE_INPUT=Enter localities root folder [default: %BASE_DEFAULT%]: 
if "%BASE_INPUT%"=="" (set "BASE=%BASE_DEFAULT%") else (set "BASE=%BASE_INPUT%")

if not exist "%BASE%" (
  echo [ERR] Folder not found: "%BASE%"
  pause
  exit /b 1
)

rem ---- собрать меню локальностей: подпапки, где есть \png с .png ----
setlocal DisableDelayedExpansion
set "LISTTMP=%TEMP%\locs_%RANDOM%.txt"
del /q "%LISTTMP%" 2>nul
set /a IDX=0

for /f "delims=" %%D in ('dir /b /ad "%BASE%"') do (
  set "LOC=%%D"
  setlocal EnableDelayedExpansion
  set "PNGDIR=%BASE%\!LOC!\png"
  if exist "!PNGDIR!" (
    for /f %%c in ('dir /b /a-d "!PNGDIR!\*.png" 2^>nul ^| find /c /v ""') do set "NUMPNG=%%c"
    if "!NUMPNG!"=="" set "NUMPNG=0"
    for /f %%c in ('dir /b /a-d "!PNGDIR!\*.csv" 2^>nul ^| find /c /v ""') do set "NUMCSV=%%c"
    if "!NUMCSV!"=="" set "NUMCSV=0"
    if !NUMPNG! GTR 0 (set /a PCT=(100*!NUMCSV!)/!NUMPNG!) else (set "PCT=0")

    rem первый PNG по алфавиту
    set "FIRSTPNG="
    for /f "usebackq delims=" %%F in (`dir /b /a-d /on "!PNGDIR!\*.png" 2^>nul`) do if not defined FIRSTPNG set "FIRSTPNG=%%~nF"

    rem индикатор масштаба: принимаем оба варианта имён (basename.scale.csv ИЛИ basename.png.scale.csv)
    set "SCALEFLAG="
    if defined FIRSTPNG (
      if exist "!PNGDIR!\!FIRSTPNG!.scale.csv" (
        rem ок
      ) else if exist "!PNGDIR!\!FIRSTPNG!.png.scale.csv" (
        rem ок (совместимость с текущим GUI)
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
  echo [ERR] No localities (with \png) found in "%BASE%".
  pause
  exit /b 2
)

echo.
echo Localities under: "%BASE%"
echo.

for /f "usebackq tokens=1-6 delims=|" %%A in ("%LISTTMP%") do (
  set "idx=%%A" & set "loc=%%B" & set "nimg=%%C" & set "nann=%%D" & set "pct=%%E" & set "pngdir=%%F"
  setlocal EnableDelayedExpansion
  echo  !idx!) !loc! [!nimg!/!nann!] !pct!%%!SCALEFLAG!
  endlocal
)

echo.
set "SEL="
set /p SEL=Enter number (Q=quit): 
if /i "%SEL%"=="Q" exit /b 0

for /f "usebackq tokens=1-6 delims=|" %%A in ("%LISTTMP%") do (
  if "%%A"=="%SEL%" (
    set "PNG=%%F"
    set "LOCNAME=%%B"
  )
)
del /q "%LISTTMP%" 2>nul

if "%PNG%"=="" (
  echo [ERR] Invalid selection.
  pause
  exit /b 3
)

echo.
echo [RUN] locality: %LOCNAME%
echo [DIR] images:  "%PNG%"
echo [ROOT] project: "%ROOT%"
echo.

"%PY%" "%HERE%\annot_gui_custom.py" --root "%ROOT%" --images "%PNG%"
if errorlevel 1 (
  echo [ERR] GUI exited with code %ERRORLEVEL%
  pause
)

exit /b 0
