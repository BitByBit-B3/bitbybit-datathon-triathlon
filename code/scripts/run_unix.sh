#!/bin/bash

# BitByBit Datathon - Unix/Linux Runner Script

set -e  # Exit on any error

echo "=== BitByBit Datathon 2025 - Pipeline Runner ==="
echo "Platform: Unix/Linux/macOS"
echo

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
echo "Python version: $python_version"

if [[ $(echo "$python_version >= 3.11" | bc -l) -ne 1 ]]; then
    echo "Error: Python 3.11+ required, found $python_version"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip and install requirements
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Check data files exist
echo "Checking data files..."
required_files=(
    "data/raw/bookings_train.csv"
    "data/raw/tasks.csv" 
    "data/raw/staffing_train.csv"
    "data/raw/task1_test_inputs.csv"
    "data/raw/task2_test_inputs.csv"
)

for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "Warning: $file not found"
    fi
done

# Run the main pipeline
echo
echo "=== Running submission pipeline ==="
python scripts/make_submission.py

# Check outputs
echo
echo "=== Checking outputs ==="
if [ -f "../task1_predictions.csv" ]; then
    lines=$(wc -l < "../task1_predictions.csv")
    echo "✓ task1_predictions.csv: $lines lines"
else
    echo "✗ task1_predictions.csv not found"
fi

if [ -f "../task2_predictions.csv" ]; then
    lines=$(wc -l < "../task2_predictions.csv")
    echo "✓ task2_predictions.csv: $lines lines" 
else
    echo "✗ task2_predictions.csv not found"
fi

echo
echo "=== Pipeline complete! ==="
echo "Submission files are ready at the project root."