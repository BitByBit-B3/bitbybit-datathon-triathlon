import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from features_task1 import extract_test_features_task1, clean_features_for_modeling
from baselines import validate_baseline_predictions
from io_safe import read_csv_safe, write_csv_safe, safe_pickle_load
from paths import TASK1_TEST_PATH, TASK1_PREDICTIONS_PATH, TASK1_ARTIFACTS_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main prediction function for Task 1."""
    logger.info("=== Starting Task 1 Prediction ===")
    
    try:
        # Load test inputs
        logger.info("Loading test inputs...")
        if not TASK1_TEST_PATH.exists():
            raise FileNotFoundError(f"Test inputs not found: {TASK1_TEST_PATH}")
        
        test_inputs_df = read_csv_safe(TASK1_TEST_PATH)
        logger.info(f"Loaded {len(test_inputs_df)} test samples")
        logger.info(f"Test input columns: {list(test_inputs_df.columns)}")
        
        # Validate test input format
        required_cols = ['row_id', 'date', 'time', 'task_id']
        for col in required_cols:
            if col not in test_inputs_df.columns:
                logger.error(f"Missing required column in test inputs: {col}")
                raise ValueError(f"Test inputs missing column: {col}")
        
        # Load trained model artifacts
        logger.info("Loading trained model...")
        if not TASK1_ARTIFACTS_PATH.exists():
            raise FileNotFoundError(f"Model artifacts not found: {TASK1_ARTIFACTS_PATH}")
        
        artifacts = safe_pickle_load(TASK1_ARTIFACTS_PATH)
        model = artifacts['model']
        model_type = artifacts.get('model_type', 'unknown')
        tasks_df = artifacts.get('tasks_df', None)
        
        logger.info(f"Loaded {model_type} model")
        logger.info(f"Training samples: {artifacts.get('training_samples', 'unknown')}")
        
        # Handle fallback models
        if model_type == 'fallback_global':
            logger.info("Using fallback global model")
            predictions = np.full(len(test_inputs_df), artifacts['global_median'])
        else:
            # Extract features from test inputs
            logger.info("Extracting test features...")
            test_features_df = extract_test_features_task1(test_inputs_df, tasks_df)
            
            # Clean features for modeling
            test_features_clean = clean_features_for_modeling(test_features_df)
            
            # Ensure we have all the features the model expects
            expected_features = artifacts.get('feature_columns', [])
            if expected_features:
                for col in expected_features:
                    if col not in test_features_clean.columns:
                        logger.warning(f"Missing expected feature: {col}, filling with default")
                        # Fill with appropriate default based on column name
                        if col.startswith('appt_') or col.startswith('checkin_'):
                            if 'hour' in col:
                                default_val = 12  # Noon
                            elif 'weekday' in col:
                                default_val = 1  # Tuesday
                            elif 'month' in col:
                                default_val = 6  # June
                            else:
                                default_val = 0
                        else:
                            default_val = 0
                        test_features_clean[col] = default_val
                
                # Reorder columns to match training
                test_features_clean = test_features_clean[expected_features]
            
            logger.info(f"Test features shape: {test_features_clean.shape}")
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = model.predict(test_features_clean)
        
        # Validate predictions
        predictions = validate_baseline_predictions(
            predictions,
            "Task1_Test",
            min_value=0,
            max_value=480  # 8 hours max
        )
        
        # Create output DataFrame
        output_df = pd.DataFrame({
            'row_id': test_inputs_df['row_id'],
            'prediction': predictions
        })
        
        # Ensure predictions are integers
        output_df['prediction'] = output_df['prediction'].astype(int)
        
        # Validate output format
        if len(output_df) != len(test_inputs_df):
            raise ValueError(f"Output length mismatch: {len(output_df)} != {len(test_inputs_df)}")
        
        if output_df.isnull().any().any():
            logger.error("Output contains null values!")
            output_df = output_df.fillna(30)  # Default to 30 minutes
        
        # Save predictions
        logger.info("Saving predictions...")
        write_csv_safe(output_df, TASK1_PREDICTIONS_PATH)
        
        # Log prediction statistics
        pred_stats = output_df['prediction'].describe()
        logger.info(f"Prediction statistics:")
        logger.info(f"  Count: {pred_stats['count']}")
        logger.info(f"  Mean: {pred_stats['mean']:.1f}")
        logger.info(f"  Std: {pred_stats['std']:.1f}")
        logger.info(f"  Min: {pred_stats['min']}")
        logger.info(f"  25%: {pred_stats['25%']:.1f}")
        logger.info(f"  50%: {pred_stats['50%']:.1f}")
        logger.info(f"  75%: {pred_stats['75%']:.1f}")
        logger.info(f"  Max: {pred_stats['max']}")
        
        logger.info("=== Task 1 Prediction Complete ===")
        logger.info(f"Predictions saved to: {TASK1_PREDICTIONS_PATH}")
        logger.info(f"Generated {len(output_df)} predictions")
        
    except Exception as e:
        logger.error(f"Task 1 prediction failed: {e}")
        
        # Create fallback predictions in case of error
        try:
            logger.info("Creating fallback predictions...")
            test_inputs_df = read_csv_safe(TASK1_TEST_PATH)
            
            # Use a reasonable default (30 minutes processing time)
            fallback_predictions = np.full(len(test_inputs_df), 30)
            
            output_df = pd.DataFrame({
                'row_id': test_inputs_df['row_id'],
                'prediction': fallback_predictions
            })
            
            write_csv_safe(output_df, TASK1_PREDICTIONS_PATH)
            logger.info(f"Saved fallback predictions to: {TASK1_PREDICTIONS_PATH}")
            
        except Exception as fallback_error:
            logger.error(f"Even fallback prediction failed: {fallback_error}")
            raise
        
        raise

def validate_predictions():
    """Validate the generated predictions file."""
    try:
        logger.info("Validating predictions file...")
        
        # Check if file exists
        if not TASK1_PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"Predictions file not found: {TASK1_PREDICTIONS_PATH}")
        
        # Load predictions
        pred_df = read_csv_safe(TASK1_PREDICTIONS_PATH)
        
        # Check format
        if list(pred_df.columns) != ['row_id', 'prediction']:
            raise ValueError(f"Invalid columns: {list(pred_df.columns)}")
        
        # Check for missing values
        if pred_df.isnull().any().any():
            raise ValueError("Predictions contain null values")
        
        # Check data types
        if not pred_df['prediction'].dtype.kind in 'iu':
            raise ValueError(f"Predictions not integer type: {pred_df['prediction'].dtype}")
        
        # Check value ranges
        if (pred_df['prediction'] < 0).any():
            raise ValueError("Predictions contain negative values")
        
        if (pred_df['prediction'] > 480).any():
            logger.warning("Some predictions exceed 8 hours")
        
        logger.info("✓ Predictions validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Predictions validation failed: {e}")
        return False

if __name__ == "__main__":
    main()
    validate_predictions()