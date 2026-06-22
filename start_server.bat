@echo off
echo =========================================
echo   Sentinel AI Camera Server
echo =========================================
echo.
echo Starting backend API on http://127.0.0.1:8000
echo Open technomak-video-analytics-console.html in your browser.
echo.
cd /d "%~dp0"
call venv\Scripts\activate
uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
pause
