# -*- mode: python ; coding: utf-8 -*-

from PyInstaller.utils.hooks import collect_submodules

hiddenimports = (
    collect_submodules("win32com")
    + ["win32gui", "win32api", "win32con", "pythoncom", "pywintypes"]
)

a = Analysis(
    ["hid_behavior_collector_v4.py"],
    pathex=["."],
    binaries=[],
    datas=[
        (
            "specs/feature_schema_v2/feature_schema_v2.yaml",
            "specs/feature_schema_v2",
        ),
        (
            "specs/feature_schema_v2/FEATURE_SCHEMA_V2_SPEC.md",
            "specs/feature_schema_v2",
        ),
    ],
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="HID_Behavior_Collector_v4_1",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
)
