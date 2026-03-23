@echo off
title FX Options Workstation
echo.
echo  ================================================================
echo   FX OPTIONS WORKSTATION - Setup ^& Launch
echo  ================================================================
echo.

:: Navigate to project directory first
cd /d "%~dp0"

:: Find Python - try multiple common locations
set "PYTHON="

:: Try 'python' on PATH
python --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=python"
    goto :found_python
)

:: Try 'py' launcher
py -3 --version >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py -3"
    goto :found_python
)

:: Try common install locations
for %%P in (
    "%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    "%LOCALAPPDATA%\Programs\Python\Python39\python.exe"
    "C:\Python312\python.exe"
    "C:\Python311\python.exe"
) do (
    if exist %%P (
        set "PYTHON=%%~P"
        goto :found_python
    )
)

:: Python not found anywhere
echo.
echo  [ERROR] Python not found.
echo.
echo  Please install Python from: https://python.org/downloads
echo  IMPORTANT: Check "Add Python to PATH" during install!
echo.
pause
exit /b 1

:found_python
echo  [OK] Found Python: %PYTHON%
echo.

:: Install/verify dependencies
echo  [1/3] Checking dependencies...
echo.
%PYTHON% -c "import dash, plotly, numpy, scipy, pandas, dash_bootstrap_components; print('  All dependencies present.')" 2>nul
if %errorlevel% neq 0 (
    echo  Installing missing packages...
    %PYTHON% -m pip install --quiet dash plotly numpy scipy pandas dash-bootstrap-components
    if %errorlevel% neq 0 (
        echo.
        echo  [ERROR] Package installation failed.
        echo  Try running manually: pip install dash plotly numpy scipy pandas dash-bootstrap-components
        echo.
        pause
        exit /b 1
    )
    echo  [OK] Packages installed.
)
echo.
echo  [2/3] Dependencies ready.
echo.
echo  [3/3] Launching dashboard...
echo.
echo  ================================================================
echo   Opening http://localhost:8765 in your browser...
echo   Press Ctrl+C in this window to stop the server.
echo  ================================================================
echo.

:: Open browser after a short delay (2 seconds for server to start)
start "" cmd /c "timeout /t 8 /nobreak >nul & start http://localhost:8765"

:: Run the app
%PYTHON% app.py

:: If it exits, pause so user can see errors
echo.
echo  Dashboard stopped. Press any key to exit.
pause
