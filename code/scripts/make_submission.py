import logging
import os
import sys
from pathlib import Path

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from io_safe import read_csv_safe
from predict_task1 import main as predict_task1
from predict_task2 import main as predict_task2
from train_task1 import main as train_task1
from train_task2 import main as train_task2

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def validate_predictions(task_name: str, predictions_path: str, test_inputs_path: str) -> None:
    """Validate prediction file format and content."""
    logger.info(f"Validating {task_name} predictions...")

    # Check file exists
    if not Path(predictions_path).exists():
        msg = f"Predictions file not found: {predictions_path}"
        raise FileNotFoundError(msg)

    # Load predictions
    pred_df = read_csv_safe(predictions_path)
    test_df = read_csv_safe(test_inputs_path)

    # Validate columns
    expected_cols = ["row_id", "prediction"]
    if list(pred_df.columns) != expected_cols:
        msg = f"{task_name}: Expected columns {expected_cols}, got {list(pred_df.columns)}"
        raise ValueError(msg)

    # Validate row count
    if len(pred_df) != len(test_df):
        msg = f"{task_name}: Prediction count {len(pred_df)} != test count {len(test_df)}"
        raise ValueError(msg)

    # Validate no NaNs
    if pred_df.isnull().any().any():
        msg = f"{task_name}: Contains NaN values"
        raise ValueError(msg)

    # Validate prediction types and values
    if pred_df["prediction"].dtype.kind not in "iu":
        logger.warning(f"{task_name}: Converting predictions to integer")
        pred_df["prediction"] = pred_df["prediction"].astype(int)

    if (pred_df["prediction"] < 0).any():
        msg = f"{task_name}: Contains negative predictions"
        raise ValueError(msg)

    # Save corrected predictions if needed
    pred_df.to_csv(predictions_path, index=False)

    logger.info(f"{task_name}: Validation passed - {len(pred_df)} predictions")


def main() -> None:
    """Main orchestration function."""
    logger.info("Starting BitByBit Datathon submission pipeline...")

    # Get paths
    script_dir = Path(__file__).parent
    code_dir = script_dir.parent
    project_dir = code_dir.parent

    os.chdir(code_dir)

    try:
        # Training phase
        logger.info("=== TRAINING PHASE ===")
        logger.info("Training Task 1 model...")
        train_task1()

        logger.info("Training Task 2 model...")
        train_task2()

        # Prediction phase
        logger.info("=== PREDICTION PHASE ===")
        logger.info("Generating Task 1 predictions...")
        predict_task1()

        logger.info("Generating Task 2 predictions...")
        predict_task2()

        # Move predictions to root
        logger.info("=== MOVING PREDICTIONS TO ROOT ===")
        task1_source = code_dir / "task1_predictions.csv"
        task2_source = code_dir / "task2_predictions.csv"
        task1_target = project_dir / "task1_predictions.csv"
        task2_target = project_dir / "task2_predictions.csv"

        if task1_source.exists():
            task1_source.rename(task1_target)
            logger.info(f"Moved task1_predictions.csv to {task1_target}")

        if task2_source.exists():
            task2_source.rename(task2_target)
            logger.info(f"Moved task2_predictions.csv to {task2_target}")

        # Validation
        logger.info("=== VALIDATION PHASE ===")
        validate_predictions(
            "Task1",
            str(task1_target),
            str(code_dir / "data" / "raw" / "task1_test_inputs.csv"),
        )
        validate_predictions(
            "Task2",
            str(task2_target),
            str(code_dir / "data" / "raw" / "task2_test_inputs.csv"),
        )

        logger.info("=== SUBMISSION COMPLETE ===")
        logger.info(f"✓ task1_predictions.csv: {task1_target}")
        logger.info(f"✓ task2_predictions.csv: {task2_target}")
        logger.info("Pipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise


if __name__ == "__main__":
    main()
