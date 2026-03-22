@echo off
title FX Options Workstation
echo.
echo  ================================================================
echo   FX OPTIONS WORKSTATION - Setup ^& Launch
echo  ================================================================
echo.

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo  [!] Python not found. Installing Python 3.12...
    echo.
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Could not install Python automatically.
        echo  Please install Python manually from: https://python.org/downloads
        echo  IMPORTANT: Check "Add Python to PATH" during install!
        echo.
        pause
        exit /b 1
    )
    echo.
    echo  [OK] Python installed. You may need to restart this script.
    echo.
)

:: Navigate to project directory
cd /d "%~dp0"
echo  [1/3] Installing dependencies...
echo.
pip install dash plotly numpy scipy pandas dash-bootstrap-components 2>&1 | findstr /i "Successfully already"
echo.
echo  [2/3] Dependencies ready.
echo.
echo  [3/3] Launching dashboard...
echo.
echo  ================================================================
echo   Opening http://localhost:8050 in your browser...
echo  ================================================================
echo.

:: Open browser after a short delay
start "" "http://localhost:8050"

:: Run the app
python app.py

:: If it exits, pause so user can see errors
echo.
echo  Dashboard stopped. Press any key to exit.
pause
