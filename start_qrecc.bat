@echo off
setlocal
cd /d "%~dp0"
title SimpleRAG Research - QReCC
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0launch.ps1" -Dataset qrecc
if errorlevel 1 pause
endlocal
