@echo off
echo Tree Species Classification Application
echo ======================================

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python is not installed or not in PATH. Please install Python 3.7 or higher.
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo Failed to create virtual environment. Please install venv package.
        pause
        exit /b 1
    )
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies if requirements.txt exists
if exist requirements.txt (
    echo Installing dependencies...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo Failed to install dependencies.
        pause
        exit /b 1
    )
)

REM Check if model exists, download if not
if not exist model\efficientnetb0_tree_classifier (
    echo Downloading pre-trained model...
    python download_model.py
    if %errorlevel% neq 0 (
        echo Failed to download model.
        pause
        exit /b 1
    )
)

REM Run the application
echo Starting the application...
python run.py --run

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

pause