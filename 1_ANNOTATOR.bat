@echo off
chcp 65001 >nul
setlocal EnableExtensions
set "HERE=%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%HERE%scripts\start_annotator.ps1"
exit /b %ERRORLEVEL%