@echo off
REM ------------------------------------------------------------------
REM  GNR-602 Building Extraction - environment setup (Windows)
REM ------------------------------------------------------------------
REM  Usage:  setup.bat
REM ------------------------------------------------------------------

setlocal enabledelayedexpansion

set VENV_DIR=venv

REM ---- 1. Python check ---------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python not found on PATH. Install Python 3.8+ first.
    exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PY_VER=%%v
echo [1/4] Found Python %PY_VER%

REM ---- 2. Virtual environment --------------------------------------
if exist %VENV_DIR% (
    echo [2/4] Reusing existing venv at .\%VENV_DIR%
) else (
    echo [2/4] Creating venv at .\%VENV_DIR% ...
    python -m venv %VENV_DIR%
    if errorlevel 1 (
        echo ERROR: failed to create venv.
        exit /b 1
    )
)

call %VENV_DIR%\Scripts\activate.bat

REM ---- 3. Install dependencies -------------------------------------
echo [3/4] Upgrading pip ...
python -m pip install --upgrade pip >nul

echo [3/4] Installing requirements (this can take a minute) ...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    exit /b 1
)

REM ---- 4. Done -----------------------------------------------------
echo [4/4] Setup complete.
echo.
echo Next steps:
echo     %VENV_DIR%\Scripts\activate          ^&^& REM activate the env in new shells
echo     python app.py                        ^&^& REM start web UI on http://localhost:5000
echo.
echo     REM or run from the command line directly:
echo     python building_extraction.py path\to\image.png
echo     python sentinel_pipeline.py sat_image

endlocal
