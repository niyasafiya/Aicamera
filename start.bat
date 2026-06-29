@echo off
title Sentinel Backend
echo Starting Sentinel backend on http://127.0.0.1:8001 ...
echo Open your browser at: http://127.0.0.1:8001
echo.
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001 --reload
pause
