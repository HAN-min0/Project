@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" goto install
call "%~dp0run_python.bat" -m venv .venv
if errorlevel 1 goto failed
:install
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto failed
echo Installation completed. Double-click the walking launcher.
goto finish
:failed
echo Installation failed. Read the error above.
:finish
popd
pause
