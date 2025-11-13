@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "HERE=%~dp0"
rem 1) Выбор/запоминание базы локальностей + джанкшн ROOT\photos
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%scripts\choose_localities.ps1"
if errorlevel 1 (
  echo [ERR] Folder picking failed or was cancelled.
  goto end
)
rem 2) Запуск твоего рабочего аннотатора (GUI не трогаем)
call "%HERE%1_ANNOTATOR.bat"
:end
exit /b %ERRORLEVEL%