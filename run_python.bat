@echo off
setlocal
pushd "%~dp0"
if exist "%~dp0.venv\Scripts\python.exe" goto local_python
where python >nul 2>nul
if not errorlevel 1 goto system_python
where py >nul 2>nul
if not errorlevel 1 goto launcher_python
echo Python was not found. Install Python 3.11, then try again.
set "ROBOT_EXIT=1"
goto finish
:local_python
"%~dp0.venv\Scripts\python.exe" %*
set "ROBOT_EXIT=%ERRORLEVEL%"
goto finish
:system_python
python %*
set "ROBOT_EXIT=%ERRORLEVEL%"
goto finish
:launcher_python
py -3 %*
set "ROBOT_EXIT=%ERRORLEVEL%"
:finish
popd
exit /b %ROBOT_EXIT%
