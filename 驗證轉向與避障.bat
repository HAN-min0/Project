@echo off
call "%~dp0run_python.bat" -m unittest -v test_heading test_obstacles
pause
