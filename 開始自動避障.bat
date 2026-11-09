@echo off
call "%~dp0run_python.bat" "%~dp0obstacle_demo.py" %*
if errorlevel 1 (
    echo The simulation stopped with an error or collision. Read the result above.
    pause
)
