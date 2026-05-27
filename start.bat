@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0"

echo ============================================
echo   Noion - AI Ear-copy Service
echo ============================================
echo.

REM --- Python check ---
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

REM --- Install Python deps ---
echo [1/4] Installing Python packages...
python -m pip install -q -r requirements.txt
python -m pip install -q -r requirements-web.txt
python -m pip install -q fastapi "uvicorn[standard]" python-multipart aiofiles

REM --- Node.js check (optional) ---
set HAS_NODE=0
where node >nul 2>nul
if not errorlevel 1 set HAS_NODE=1

REM --- Frontend build ---
if "%HAS_NODE%"=="1" (
    if not exist "frontend\dist\index.html" (
        echo [2/4] Building frontend...
        pushd frontend
        if not exist "node_modules" call npm install
        call npm run build
        popd
    ) else (
        echo [2/4] Frontend already built - skip
    )
) else (
    echo [2/4] Node.js not found - skipping frontend build
    echo       API will be available at http://localhost:8000/docs
)

REM --- Open browser after delay ---
echo [3/4] Will open browser in 3 seconds...
start "" /b cmd /c "timeout /t 3 /nobreak >nul && start http://localhost:8000"

REM --- Start server ---
echo [4/4] Starting server on http://localhost:8000
echo.
echo Press Ctrl+C to stop.
echo ============================================
echo.

python run.py

echo.
echo Server stopped.
pause
