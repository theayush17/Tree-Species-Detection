#!/bin/bash

echo "Tree Species Classification Application"
echo "======================================"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3.7 or higher."
    exit 1
fi

# Check if virtual environment exists, create if not
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment. Please install venv package."
        exit 1
    fi
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install dependencies if requirements.txt exists
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies..."
    pip install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
fi

# Check if model exists, download if not
if [ ! -d "model/efficientnetb0_tree_classifier" ]; then
    echo "Downloading pre-trained model..."
    python download_model.py
    if [ $? -ne 0 ]; then
        echo "Failed to download model."
        exit 1
    fi
fi

# Run the application
echo "Starting the application..."
python run.py --run

# Deactivate virtual environment
deactivate