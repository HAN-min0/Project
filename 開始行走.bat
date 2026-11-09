@echo off
call "%~dp0run_python.bat" "%~dp0eval.py" %*
if errorlevel 1 (
    echo Demo failed. Read the error above and README.md.
    pause
)
