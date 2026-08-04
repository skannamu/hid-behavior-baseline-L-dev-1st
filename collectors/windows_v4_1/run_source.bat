@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Run build_exe.bat first to create the virtual environment.
  exit /b 1
)

call ".venv\Scripts\activate.bat"
python hid_behavior_collector_v4.py
endlocal
