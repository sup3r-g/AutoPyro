@echo off
setlocal
if not exist "%~dp0runtime\python.exe" (
    echo Bundled Python is missing. Extract the complete ZIP and try again. 1>&2
    exit /b 2
)
"%~dp0runtime\python.exe" -I -X utf8 "%~dp0install_and_check.py"
exit /b %errorlevel%
