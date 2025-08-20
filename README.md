# BitByBit Datathon 2025 - Tech Triathlon

## Overview

This repository contains our submission for the Tech-Triathlon 2025 Datathon. The solution predicts:

1. **Task 1**: Service processing time (minutes) after staff start
2. **Task 2**: Number of employees needed per section per day

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### Running the Pipeline

#### Windows
```powershell
cd code
.\scripts\run_windows.ps1
```

#### Unix/Linux/macOS
```bash
cd code
chmod +x scripts/run_unix.sh
./scripts/run_unix.sh
```

### Manual Setup
```bash
cd code
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python scripts/make_submission.py
```

## Data Setup

Place the following CSV files in `code/data/raw/`:
- `bookings_train.csv`
- `tasks.csv`
- `staffing_train.csv`
- `task1_test_inputs.csv`
- `task2_test_inputs.csv`

## Outputs

After running the pipeline, find these files at the root:
- `task1_predictions.csv` - Processing time predictions
- `task2_predictions.csv` - Staffing level predictions

## Project Structure

```
BitByBit_Datathon/
├── task1_predictions.csv          # Generated predictions (root)
├── task2_predictions.csv          # Generated predictions (root)
├── demo_video_link.txt            # Video demonstration link
├── README.md                      # This file
├── code/
│   ├── requirements.txt           # Python dependencies
│   ├── scripts/                   # Orchestration scripts
│   ├── src/                       # Core implementation
│   ├── data/                      # Data directory (raw CSVs here)
│   ├── artifacts/                 # Saved models/baselines
│   └── notebooks/                 # EDA notebook
├── docs/                          # Documentation
└── reports/                       # Report templates
```

## Algorithm Summary

### Task 1: Processing Time Prediction
- **Target**: `check_out_time - check_in_time` in minutes
- **Method**: Median-based baseline by (task_id, hour, weekday)
- **Fallback**: Task-level median → Global median

### Task 2: Staffing Prediction  
- **Target**: `employees_on_duty` (or derived from total_task_time_minutes)
- **Method**: Median-based baseline by section_id
- **Fallback**: Global median

Both models use robust statistical baselines with intelligent fallbacks to ensure reliable predictions even with sparse data.

## Validation

The pipeline validates all outputs to ensure:
- Correct CSV headers and format
- Integer predictions ≥ 0
- No missing values
- Row counts match test inputs

## Team: BitByBit

See `docs/` for detailed documentation and `reports/` for analysis templates.