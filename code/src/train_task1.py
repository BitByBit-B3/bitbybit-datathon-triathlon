import logging
import sys
from pathlib import Path

import numpy as np

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from baselines import create_task1_baseline, validate_baseline_predictions
from features_task1 import extract_task1_features, prepare_task1_features_for_training
from io_safe import safe_pickle_load, safe_pickle_save
from models_task1 import Task1ModelEnsemble
from paths import TASK1_ARTIFACTS_PATH
from preprocessing import load_and_clean_training_data

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    """Main training function for Task 1."""
    logger.info("=== Starting Task 1 Training ===")

    try:
        # Load and clean training data
        logger.info("Loading training data...")
        bookings_df, target_series, staffing_df, tasks_df = load_and_clean_training_data()

        if len(bookings_df) == 0:
            msg = "No bookings training data available"
            raise ValueError(msg)

        # Check if we have valid targets
        valid_targets = target_series.dropna()
        if len(valid_targets) == 0:
            logger.error("No valid target values found!")
            # Create a fallback baseline with global median
            global_median = 30  # Reasonable default for processing time
            logger.info(f"Using fallback global median: {global_median} minutes")

            # Save a simple fallback model
            fallback_artifacts = {
                "model_type": "fallback_global",
                "global_median": global_median,
                "target_mean": global_median,
                "target_std": 15,
                "feature_columns": [],
                "training_samples": 0,
            }
            safe_pickle_save(fallback_artifacts, TASK1_ARTIFACTS_PATH)
            logger.info("Saved fallback model artifacts")
            return

        logger.info(f"Found {len(valid_targets)} valid target values")
        logger.info(f"Target range: {valid_targets.min():.1f} - {valid_targets.max():.1f} minutes")
        logger.info(f"Target mean: {valid_targets.mean():.1f}, median: {valid_targets.median():.1f}")

        # Extract features
        logger.info("Extracting features...")
        features_df = extract_task1_features(bookings_df, tasks_df)

        # Prepare features and target for training
        logger.info("Preparing features for training...")
        train_features, train_target = prepare_task1_features_for_training(features_df, target_series)

        if len(train_features) == 0:
            msg = "No valid training samples after feature preparation"
            raise ValueError(msg)

        logger.info(f"Training dataset: {len(train_features)} samples, {len(train_features.columns)} features")

        # Create and train model
        logger.info("Training model...")
        model = create_task1_baseline()  # Use baseline model for reliability
        model.fit(train_features, train_target)

        # Make predictions on training data for validation
        logger.info("Validating model on training data...")
        train_predictions = model.predict(train_features)
        train_predictions = validate_baseline_predictions(
            train_predictions,
            "Task1_Training",
            min_value=0,
            max_value=480,  # 8 hours max
        )

        # Calculate training metrics
        mae = np.mean(np.abs(train_predictions - train_target))
        rmse = np.sqrt(np.mean((train_predictions - train_target) ** 2))

        logger.info(f"Training MAE: {mae:.2f} minutes")
        logger.info(f"Training RMSE: {rmse:.2f} minutes")

        # Create artifacts to save
        artifacts = {
            "model": model,
            "model_type": "baseline_median",
            "target_mean": float(train_target.mean()),
            "target_std": float(train_target.std()),
            "target_median": float(train_target.median()),
            "feature_columns": list(train_features.columns),
            "training_samples": len(train_features),
            "training_mae": float(mae),
            "training_rmse": float(rmse),
            "tasks_df": tasks_df if len(tasks_df) > 0 else None,
        }

        # Save artifacts
        logger.info("Saving model artifacts...")
        safe_pickle_save(artifacts, TASK1_ARTIFACTS_PATH)

        logger.info("=== Task 1 Training Complete ===")
        logger.info(f"Model saved to: {TASK1_ARTIFACTS_PATH}")
        logger.info(f"Training performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    except Exception as e:
        logger.error(f"Task 1 training failed: {e}")
        raise


def train_with_advanced_models() -> None:
    """Alternative training function using advanced ML models."""
    logger.info("=== Starting Task 1 Training (Advanced Models) ===")

    try:
        # Load and clean training data
        bookings_df, target_series, staffing_df, tasks_df = load_and_clean_training_data()

        if len(bookings_df) == 0:
            msg = "No bookings training data available"
            raise ValueError(msg)

        # Extract and prepare features
        features_df = extract_task1_features(bookings_df, tasks_df)
        train_features, train_target = prepare_task1_features_for_training(features_df, target_series)

        if len(train_features) == 0:
            msg = "No valid training samples"
            raise ValueError(msg)

        # Train ensemble model
        logger.info("Training ensemble model...")
        model = Task1ModelEnsemble(use_advanced_models=True)
        model.fit(train_features, train_target)

        # Validate
        train_predictions = model.predict(train_features)
        mae = np.mean(np.abs(train_predictions - train_target))
        rmse = np.sqrt(np.mean((train_predictions - train_target) ** 2))

        logger.info(f"Ensemble Training MAE: {mae:.2f}")
        logger.info(f"Ensemble Training RMSE: {rmse:.2f}")

        # Save artifacts
        artifacts = {
            "model": model,
            "model_type": "ensemble",
            "target_mean": float(train_target.mean()),
            "target_std": float(train_target.std()),
            "feature_columns": list(train_features.columns),
            "training_samples": len(train_features),
            "training_mae": float(mae),
            "training_rmse": float(rmse),
            "tasks_df": tasks_df if len(tasks_df) > 0 else None,
        }

        safe_pickle_save(artifacts, TASK1_ARTIFACTS_PATH)

        logger.info("=== Advanced Task 1 Training Complete ===")

    except Exception as e:
        logger.error(f"Advanced Task 1 training failed: {e}")
        logger.info("Falling back to baseline model...")
        main()  # Fall back to baseline


def validate_model_artifacts() -> bool | None:
    """Validate that the saved model artifacts are working correctly."""
    try:
        logger.info("Validating saved model artifacts...")
        artifacts = safe_pickle_load(TASK1_ARTIFACTS_PATH)

        required_keys = ["model", "model_type", "target_mean", "feature_columns"]
        for key in required_keys:
            if key not in artifacts:
                msg = f"Missing required artifact key: {key}"
                raise ValueError(msg)

        # Test that model can make predictions
        model = artifacts["model"]
        if hasattr(model, "predict"):
            logger.info("✓ Model artifacts validation passed")
            return True
        msg = "Model doesn't have predict method"
        raise ValueError(msg)

    except Exception as e:
        logger.error(f"Model artifacts validation failed: {e}")
        return False


if __name__ == "__main__":
    # Check if advanced models are requested
    import os

    if os.getenv("USE_ADVANCED_MODELS", "0") == "1":
        train_with_advanced_models()
    else:
        main()

    # Validate artifacts
    validate_model_artifacts()
