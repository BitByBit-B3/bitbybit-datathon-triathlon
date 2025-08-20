from pathlib import Path

# Get project directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
CODE_ROOT = PROJECT_ROOT / "code"
SRC_ROOT = CODE_ROOT / "src"

# Data directories
DATA_ROOT = CODE_ROOT / "data"
RAW_DATA_DIR = DATA_ROOT / "raw"
PROCESSED_DATA_DIR = DATA_ROOT / "processed"

# Output directories
ARTIFACTS_DIR = CODE_ROOT / "artifacts"
NOTEBOOKS_DIR = CODE_ROOT / "notebooks"

# Ensure directories exist
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Data file paths
BOOKINGS_TRAIN_PATH = RAW_DATA_DIR / "bookings_train.csv"
TASKS_PATH = RAW_DATA_DIR / "tasks.csv"
STAFFING_TRAIN_PATH = RAW_DATA_DIR / "staffing_train.csv"
TASK1_TEST_PATH = RAW_DATA_DIR / "task1_test_inputs.csv"
TASK2_TEST_PATH = RAW_DATA_DIR / "task2_test_inputs.csv"

# Output file paths
TASK1_PREDICTIONS_PATH = CODE_ROOT / "task1_predictions.csv"
TASK2_PREDICTIONS_PATH = CODE_ROOT / "task2_predictions.csv"

# Artifact file paths
TASK1_ARTIFACTS_PATH = ARTIFACTS_DIR / "task1_baseline.pkl"
TASK2_ARTIFACTS_PATH = ARTIFACTS_DIR / "task2_baseline.pkl"
