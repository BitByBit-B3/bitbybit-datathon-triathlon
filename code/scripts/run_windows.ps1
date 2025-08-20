# BitByBit Datathon - Windows PowerShell Runner Script

Write-Host "=== BitByBit Datathon 2025 - Pipeline Runner ===" -ForegroundColor Cyan
Write-Host "Platform: Windows (PowerShell)"
Write-Host ""

# Check Python version
try {
    $pythonVersion = (python --version 2>&1).Split()[1]
    Write-Host "Python version: $pythonVersion"
    
    $versionParts = $pythonVersion.Split('.')
    $major = [int]$versionParts[0]
    $minor = [int]$versionParts[1]
    
    if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
        Write-Host "Error: Python 3.11+ required, found $pythonVersion" -ForegroundColor Red
        exit 1
    }
}
catch {
    Write-Host "Error: Python not found or not accessible" -ForegroundColor Red
    exit 1
}

# Create virtual environment if it doesn't exist
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment..."
    python -m venv .venv
}

# Activate virtual environment
Write-Host "Activating virtual environment..."
& ".venv\Scripts\Activate.ps1"

# Upgrade pip and install requirements
Write-Host "Installing dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Check data files exist
Write-Host "Checking data files..."
$requiredFiles = @(
    "data\raw\bookings_train.csv",
    "data\raw\tasks.csv",
    "data\raw\staffing_train.csv", 
    "data\raw\task1_test_inputs.csv",
    "data\raw\task2_test_inputs.csv"
)

foreach ($file in $requiredFiles) {
    if (-not (Test-Path $file)) {
        Write-Host "Warning: $file not found" -ForegroundColor Yellow
    }
}

# Run the main pipeline
Write-Host ""
Write-Host "=== Running submission pipeline ===" -ForegroundColor Green
python scripts\make_submission.py

# Check outputs
Write-Host ""
Write-Host "=== Checking outputs ===" -ForegroundColor Green
if (Test-Path "..\task1_predictions.csv") {
    $lines = (Get-Content "..\task1_predictions.csv").Count
    Write-Host "✓ task1_predictions.csv: $lines lines" -ForegroundColor Green
}
else {
    Write-Host "✗ task1_predictions.csv not found" -ForegroundColor Red
}

if (Test-Path "..\task2_predictions.csv") {
    $lines = (Get-Content "..\task2_predictions.csv").Count
    Write-Host "✓ task2_predictions.csv: $lines lines" -ForegroundColor Green
}
else {
    Write-Host "✗ task2_predictions.csv not found" -ForegroundColor Red
}

Write-Host ""
Write-Host "=== Pipeline complete! ===" -ForegroundColor Cyan
Write-Host "Submission files are ready at the project root."