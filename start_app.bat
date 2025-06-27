@echo off
chcp 65001 >nul
echo AI Image Generator - Auto Start Script
echo =====================================
echo.

echo Step 1: Cleaning ports...
python clean_ports.py

echo.
echo Step 2: Starting main program...
echo.

python app_new.py

echo.
echo Program exited
pause
