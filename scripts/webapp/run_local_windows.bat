@echo off
setlocal
powershell -ExecutionPolicy Bypass -File "%~dp0deploy_windows.ps1" %*
