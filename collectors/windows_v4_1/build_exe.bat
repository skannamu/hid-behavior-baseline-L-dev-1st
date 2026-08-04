@echo off
setlocal
cd /d "%~dp0"

echo ==============================================
echo HID Behavior Collector v4.1 build
echo ==============================================

where py >nul 2>nul
if errorlevel 1 (
  echo [FAIL] Python launcher "py" was not found.
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  py -3 -m venv .venv
  if errorlevel 1 exit /b 1
)

call ".venv\Scripts\activate.bat"
python -m pip install --upgrade pip
if errorlevel 1 exit /b 1

python -m pip install -r requirements-build.txt
if errorlevel 1 exit /b 1

python -m unittest tests.test_feature_schema_v2 -v
if errorlevel 1 (
  echo [FAIL] Feature core tests failed.
  exit /b 1
)

python -m py_compile hid_behavior_collector_v4.py
if errorlevel 1 (
  echo [FAIL] Collector source syntax validation failed.
  exit /b 1
)

python -m PyInstaller --noconfirm --clean HID_Behavior_Collector_v4_1.spec
if errorlevel 1 (
  echo [FAIL] PyInstaller build failed.
  exit /b 1
)

if not exist "dist\HID_Behavior_Collector_v4_1.exe" (
  echo [FAIL] Expected EXE was not created.
  exit /b 1
)

copy /Y "specs\feature_schema_v2\FEATURE_SCHEMA_V2_SPEC.md" "dist\" >nul
copy /Y "specs\feature_schema_v2\feature_schema_v2.yaml" "dist\" >nul

echo.
echo [OK] Build complete:
echo %CD%\dist\HID_Behavior_Collector_v4_1.exe
echo.
endlocal
