# Code Documentation

## Quick Start

1. **Data Setup**: Place training and test CSVs in `data/raw/`
2. **Run Pipeline**: Execute `python scripts/make_submission.py`
3. **Check Outputs**: Find `task1_predictions.csv` and `task2_predictions.csv` at project root

## Architecture

### Core Modules

- **`io_safe.py`**: Robust CSV reading with error handling
- **`time_utils.py`**: DateTime parsing and duration calculations  
- **`preprocessing.py`**: Data cleaning and merging
- **`baselines.py`**: Statistical baseline models
- **`train_*.py`**: Model training logic
- **`predict_*.py`**: Prediction generation

### Pipeline Flow

1. **Training Phase**:
   - Load and clean training data
   - Build targets from check_in/check_out times
   - Train baseline models (median-based)
   - Save artifacts for prediction

2. **Prediction Phase**:
   - Load test inputs and trained artifacts
   - Generate predictions using baselines
   - Validate and save final CSVs

### Key Features

- **Robust Error Handling**: Graceful handling of missing/malformed data
- **Intelligent Fallbacks**: Multi-level fallback strategies for predictions
- **Validation**: Comprehensive output validation
- **Logging**: Detailed logging throughout pipeline

## Customization

Set environment variables to customize behavior:
- `USE_ADVANCED_MODELS=1`: Enable optional linear regression models
- `LOG_LEVEL=DEBUG`: Increase logging verbosity

## Testing

Run the EDA notebook to explore data:
```bash
jupyter notebook notebooks/01_quick_eda.ipynb
```

## Artifacts

The `artifacts/` directory contains:
- Baseline lookup tables
- Global fallback values
- Model metadata