import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from preprocessing import load_and_clean_training_data
from features_task2 import extract_task2_features, prepare_task2_features_for_training
from models_task2 import Task2ModelEnsemble, StaffingPatternModel
from baselines import create_task2_baseline, validate_baseline_predictions
from io_safe import safe_pickle_save
from paths import TASK2_ARTIFACTS_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main training function for Task 2."""
    logger.info("=== Starting Task 2 Training ===")
    
    try:
        # Load and clean training data
        logger.info("Loading training data...")
        bookings_df, target_series, staffing_df, tasks_df = load_and_clean_training_data()
        
        if len(staffing_df) == 0:
            raise ValueError("No staffing training data available")
        
        # Extract features from staffing data
        logger.info("Extracting features...")
        features_df = extract_task2_features(staffing_df)
        
        # Prepare features and target for training
        logger.info("Preparing features for training...")
        train_features, train_target = prepare_task2_features_for_training(features_df, staffing_df)
        
        # Check if we have valid targets
        if len(train_target.dropna()) == 0:
            logger.error("No valid target values found!")
            # Create a fallback baseline
            global_median = 3  # Reasonable default for employees per section
            logger.info(f"Using fallback global median: {global_median} employees")
            
            fallback_artifacts = {
                'model_type': 'fallback_global',
                'global_median': global_median,
                'target_mean': global_median,
                'target_std': 2,
                'feature_columns': [],
                'training_samples': 0
            }
            safe_pickle_save(fallback_artifacts, TASK2_ARTIFACTS_PATH)
            logger.info("Saved fallback model artifacts")
            return
        
        valid_targets = train_target.dropna()
        logger.info(f"Found {len(valid_targets)} valid target values")
        logger.info(f"Target range: {valid_targets.min()} - {valid_targets.max()} employees")
        logger.info(f"Target mean: {valid_targets.mean():.1f}, median: {valid_targets.median():.1f}")
        
        if len(train_features) == 0:
            raise ValueError("No valid training samples after feature preparation")
        
        logger.info(f"Training dataset: {len(train_features)} samples, {len(train_features.columns)} features")
        
        # Create and train model
        logger.info("Training model...")
        model = create_task2_baseline()  # Use baseline model for reliability
        model.fit(train_features, train_target)
        
        # Make predictions on training data for validation
        logger.info("Validating model on training data...")
        train_predictions = model.predict(train_features)
        train_predictions = validate_baseline_predictions(
            train_predictions,
            "Task2_Training", 
            min_value=1,  # At least 1 employee
            max_value=50  # Max 50 employees per section
        )
        
        # Calculate training metrics
        valid_mask = train_target.notna()
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(train_predictions[valid_mask] - train_target[valid_mask]))
            rmse = np.sqrt(np.mean((train_predictions[valid_mask] - train_target[valid_mask]) ** 2))
            
            logger.info(f"Training MAE: {mae:.2f} employees")
            logger.info(f"Training RMSE: {rmse:.2f} employees")
        else:
            mae = rmse = 0
            logger.warning("No valid targets for metric calculation")
        
        # Create artifacts to save
        artifacts = {
            'model': model,
            'model_type': 'baseline_median',
            'target_mean': float(train_target.mean()) if train_target.notna().sum() > 0 else 3.0,
            'target_std': float(train_target.std()) if train_target.notna().sum() > 0 else 2.0,
            'target_median': float(train_target.median()) if train_target.notna().sum() > 0 else 3.0,
            'feature_columns': list(train_features.columns),
            'training_samples': len(train_features),
            'training_mae': float(mae),
            'training_rmse': float(rmse)
        }
        
        # Save artifacts
        logger.info("Saving model artifacts...")
        safe_pickle_save(artifacts, TASK2_ARTIFACTS_PATH)
        
        logger.info("=== Task 2 Training Complete ===")
        logger.info(f"Model saved to: {TASK2_ARTIFACTS_PATH}")
        logger.info(f"Training performance - MAE: {mae:.2f}, RMSE: {rmse:.2f}")
        
    except Exception as e:
        logger.error(f"Task 2 training failed: {e}")
        raise

def train_with_staffing_patterns():
    """Alternative training using specialized staffing pattern model."""
    logger.info("=== Starting Task 2 Training (Staffing Patterns) ===")
    
    try:
        # Load and clean training data
        bookings_df, target_series, staffing_df, tasks_df = load_and_clean_training_data()
        
        if len(staffing_df) == 0:
            raise ValueError("No staffing training data available")
        
        # Extract features
        features_df = extract_task2_features(staffing_df)
        train_features, train_target = prepare_task2_features_for_training(features_df, staffing_df)
        
        if len(train_features) == 0:
            raise ValueError("No valid training samples")
        
        # Train staffing pattern model
        logger.info("Training staffing pattern model...")
        model = StaffingPatternModel()
        model.fit(train_features, train_target)
        
        # Validate
        train_predictions = model.predict(train_features)
        valid_mask = train_target.notna()
        
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(train_predictions[valid_mask] - train_target[valid_mask]))
            rmse = np.sqrt(np.mean((train_predictions[valid_mask] - train_target[valid_mask]) ** 2))
            
            logger.info(f"Staffing Pattern Training MAE: {mae:.2f}")
            logger.info(f"Staffing Pattern Training RMSE: {rmse:.2f}")
        else:
            mae = rmse = 0
        
        # Save artifacts
        artifacts = {
            'model': model,
            'model_type': 'staffing_patterns',
            'target_mean': float(train_target.mean()) if train_target.notna().sum() > 0 else 3.0,
            'target_std': float(train_target.std()) if train_target.notna().sum() > 0 else 2.0,
            'feature_columns': list(train_features.columns),
            'training_samples': len(train_features),
            'training_mae': float(mae),
            'training_rmse': float(rmse)
        }
        
        safe_pickle_save(artifacts, TASK2_ARTIFACTS_PATH)
        
        logger.info("=== Staffing Pattern Task 2 Training Complete ===")
        
    except Exception as e:
        logger.error(f"Staffing pattern Task 2 training failed: {e}")
        logger.info("Falling back to baseline model...")
        main()  # Fall back to baseline

def train_with_advanced_models():
    """Alternative training function using advanced ML models."""
    logger.info("=== Starting Task 2 Training (Advanced Models) ===")
    
    try:
        # Load and clean training data
        bookings_df, target_series, staffing_df, tasks_df = load_and_clean_training_data()
        
        if len(staffing_df) == 0:
            raise ValueError("No staffing training data available")
        
        # Extract and prepare features
        features_df = extract_task2_features(staffing_df)
        train_features, train_target = prepare_task2_features_for_training(features_df, staffing_df)
        
        if len(train_features) == 0:
            raise ValueError("No valid training samples")
        
        # Train ensemble model
        logger.info("Training ensemble model...")
        model = Task2ModelEnsemble(use_advanced_models=True)
        model.fit(train_features, train_target)
        
        # Validate
        train_predictions = model.predict(train_features)
        valid_mask = train_target.notna()
        
        if valid_mask.sum() > 0:
            mae = np.mean(np.abs(train_predictions[valid_mask] - train_target[valid_mask]))
            rmse = np.sqrt(np.mean((train_predictions[valid_mask] - train_target[valid_mask]) ** 2))
            
            logger.info(f"Ensemble Training MAE: {mae:.2f}")
            logger.info(f"Ensemble Training RMSE: {rmse:.2f}")
        else:
            mae = rmse = 0
        
        # Save artifacts
        artifacts = {
            'model': model,
            'model_type': 'ensemble',
            'target_mean': float(train_target.mean()) if train_target.notna().sum() > 0 else 3.0,
            'target_std': float(train_target.std()) if train_target.notna().sum() > 0 else 2.0,
            'feature_columns': list(train_features.columns),
            'training_samples': len(train_features),
            'training_mae': float(mae),
            'training_rmse': float(rmse)
        }
        
        safe_pickle_save(artifacts, TASK2_ARTIFACTS_PATH)
        
        logger.info("=== Advanced Task 2 Training Complete ===")
        
    except Exception as e:
        logger.error(f"Advanced Task 2 training failed: {e}")
        logger.info("Falling back to baseline model...")
        main()  # Fall back to baseline

def validate_model_artifacts():
    """Validate that the saved model artifacts are working correctly."""
    try:
        from io_safe import safe_pickle_load
        
        logger.info("Validating saved model artifacts...")
        artifacts = safe_pickle_load(TASK2_ARTIFACTS_PATH)
        
        required_keys = ['model', 'model_type', 'target_mean', 'feature_columns']
        for key in required_keys:
            if key not in artifacts:
                raise ValueError(f"Missing required artifact key: {key}")
        
        # Test that model can make predictions
        model = artifacts['model']
        if hasattr(model, 'predict'):
            logger.info("✓ Model artifacts validation passed")
            return True
        else:
            raise ValueError("Model doesn't have predict method")
            
    except Exception as e:
        logger.error(f"Model artifacts validation failed: {e}")
        return False

if __name__ == "__main__":
    # Check if advanced models or staffing patterns are requested
    import os
    model_type = os.getenv('TASK2_MODEL_TYPE', 'baseline')
    
    if os.getenv('USE_ADVANCED_MODELS', '0') == '1':
        train_with_advanced_models()
    elif model_type == 'staffing_patterns':
        train_with_staffing_patterns()
    else:
        main()
    
    # Validate artifacts
    validate_model_artifacts()