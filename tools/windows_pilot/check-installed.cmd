@echo off
setlocal
"%~dp0runtime\python.exe" -I -X utf8 "%~dp0install_and_check.py" check
set "result=%errorlevel%"
if "%result%"=="0" (echo PASS) else (echo FAILED: %result%)
echo %~dp0checks
pause
exit /b %result%
