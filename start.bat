@echo off
title Sentinel — Technomak AI Console
cd /d "%~dp0"
\

echo.
echo  Sentinel ^| Technomak AI Video Analytics
echo  =========================================
echo.

REM Activate the virtual environment
call venv\Scripts\activate.bat

REM Start the backend server in a new window
start "Sentinel API Server" cmd /k "cd /d "%~dp0" && call venv\Scripts\activate.bat && uvicorn app.main:app --reload"

REM Give the server 4 seconds to start up
echo  Starting backend server...
timeout /t 4 /nobreak > nul

REM Open the browser
echo  Opening console in browser...
start "" "http://127.0.0.1:8000/console"

echo.
echo  Console:  http://127.0.0.1:8000/console
echo  API docs: http://127.0.0.1:8000/docs
echo.
echo  To stop: close the "Sentinel API Server" window.
echo.
pause
