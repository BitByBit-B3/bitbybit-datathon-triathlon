import pandas as pd
import numpy as np
import logging
import sys
from pathlib import Path

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from features_task2 import extract_test_features_task2, clean_features_for_modeling
from baselines import validate_baseline_predictions
from io_safe import read_csv_safe, write_csv_safe, safe_pickle_load
from paths import TASK2_TEST_PATH, TASK2_PREDICTIONS_PATH, TASK2_ARTIFACTS_PATH

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main prediction function for Task 2."""
    logger.info("=== Starting Task 2 Prediction ===")
    
    try:
        # Load test inputs
        logger.info("Loading test inputs...")
        if not TASK2_TEST_PATH.exists():
            raise FileNotFoundError(f"Test inputs not found: {TASK2_TEST_PATH}")
        
        test_inputs_df = read_csv_safe(TASK2_TEST_PATH)
        logger.info(f"Loaded {len(test_inputs_df)} test samples")
        logger.info(f"Test input columns: {list(test_inputs_df.columns)}")
        
        # Validate test input format
        required_cols = ['row_id', 'date', 'section_id']
        for col in required_cols:
            if col not in test_inputs_df.columns:
                logger.error(f"Missing required column in test inputs: {col}")
                raise ValueError(f"Test inputs missing column: {col}")
        
        # Load trained model artifacts
        logger.info("Loading trained model...")
        if not TASK2_ARTIFACTS_PATH.exists():
            raise FileNotFoundError(f"Model artifacts not found: {TASK2_ARTIFACTS_PATH}")
        
        artifacts = safe_pickle_load(TASK2_ARTIFACTS_PATH)
        model = artifacts['model']
        model_type = artifacts.get('model_type', 'unknown')
        
        logger.info(f"Loaded {model_type} model")
        logger.info(f"Training samples: {artifacts.get('training_samples', 'unknown')}")
        
        # Handle fallback models
        if model_type == 'fallback_global':
            logger.info("Using fallback global model")
            predictions = np.full(len(test_inputs_df), artifacts['global_median'])
        else:
            # Extract features from test inputs
            logger.info("Extracting test features...")
            test_features_df = extract_test_features_task2(test_inputs_df)
            
            # Clean features for modeling
            test_features_clean = clean_features_for_modeling(test_features_df)
            
            # Ensure we have all the features the model expects
            expected_features = artifacts.get('feature_columns', [])
            if expected_features:
                for col in expected_features:
                    if col not in test_features_clean.columns:
                        logger.warning(f"Missing expected feature: {col}, filling with default")
                        # Fill with appropriate default based on column name
                        if 'weekday' in col:
                            default_val = 1  # Tuesday
                        elif 'month' in col:
                            default_val = 6  # June
                        elif 'is_weekend' in col:
                            default_val = 0  # Weekday
                        elif 'quarter' in col:
                            default_val = 2  # Q2
                        else:
                            default_val = 0
                        test_features_clean[col] = default_val
                
                # Reorder columns to match training
                available_features = [col for col in expected_features if col in test_features_clean.columns]
                if available_features:
                    test_features_clean = test_features_clean[available_features]
            
            logger.info(f"Test features shape: {test_features_clean.shape}")
            
            # Make predictions
            logger.info("Making predictions...")
            predictions = model.predict(test_features_clean)
        
        # Validate predictions
        predictions = validate_baseline_predictions(
            predictions,
            "Task2_Test",
            min_value=1,   # At least 1 employee
            max_value=50   # Max 50 employees per section
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
            output_df = output_df.fillna(3)  # Default to 3 employees
        
        # Save predictions
        logger.info("Saving predictions...")
        write_csv_safe(output_df, TASK2_PREDICTIONS_PATH)
        
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
        
        # Log section-wise predictions if available
        if 'section_id' in test_inputs_df.columns:
            section_summary = test_inputs_df.copy()
            section_summary['prediction'] = output_df['prediction']
            section_stats = section_summary.groupby('section_id')['prediction'].agg(['count', 'mean', 'min', 'max'])
            
            logger.info("Section-wise prediction summary:")
            for section_id, stats in section_stats.iterrows():
                logger.info(f"  Section {section_id}: {stats['count']} predictions, "
                           f"avg={stats['mean']:.1f}, range={stats['min']}-{stats['max']}")
        
        logger.info("=== Task 2 Prediction Complete ===")
        logger.info(f"Predictions saved to: {TASK2_PREDICTIONS_PATH}")
        logger.info(f"Generated {len(output_df)} predictions")
        
    except Exception as e:
        logger.error(f"Task 2 prediction failed: {e}")
        
        # Create fallback predictions in case of error
        try:
            logger.info("Creating fallback predictions...")
            test_inputs_df = read_csv_safe(TASK2_TEST_PATH)
            
            # Use reasonable defaults based on section if available
            fallback_predictions = []
            
            if 'section_id' in test_inputs_df.columns:
                # Simple heuristic: different sections might need different staffing
                section_defaults = {}
                for _, row in test_inputs_df.iterrows():
                    section_id = row['section_id']
                    if section_id not in section_defaults:
                        # Vary defaults slightly by section to be more realistic
                        section_defaults[section_id] = max(1, 3 + (hash(str(section_id)) % 5))
                    fallback_predictions.append(section_defaults[section_id])
            else:
                # Use fixed default if no section info
                fallback_predictions = [3] * len(test_inputs_df)
            
            output_df = pd.DataFrame({
                'row_id': test_inputs_df['row_id'],
                'prediction': fallback_predictions
            })
            
            write_csv_safe(output_df, TASK2_PREDICTIONS_PATH)
            logger.info(f"Saved fallback predictions to: {TASK2_PREDICTIONS_PATH}")
            
        except Exception as fallback_error:
            logger.error(f"Even fallback prediction failed: {fallback_error}")
            raise
        
        raise

def validate_predictions():
    """Validate the generated predictions file."""
    try:
        logger.info("Validating predictions file...")
        
        # Check if file exists
        if not TASK2_PREDICTIONS_PATH.exists():
            raise FileNotFoundError(f"Predictions file not found: {TASK2_PREDICTIONS_PATH}")
        
        # Load predictions
        pred_df = read_csv_safe(TASK2_PREDICTIONS_PATH)
        
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
        if (pred_df['prediction'] < 1).any():
            raise ValueError("Predictions contain values less than 1 employee")
        
        if (pred_df['prediction'] > 100).any():
            logger.warning("Some predictions exceed 100 employees (may be unrealistic)")
        
        logger.info("✓ Predictions validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Predictions validation failed: {e}")
        return False

def analyze_predictions():
    """Analyze the generated predictions for insights."""
    try:
        logger.info("Analyzing predictions...")
        
        # Load predictions and test inputs
        pred_df = read_csv_safe(TASK2_PREDICTIONS_PATH)
        test_inputs_df = read_csv_safe(TASK2_TEST_PATH)
        
        # Combine for analysis
        analysis_df = test_inputs_df.merge(pred_df, on='row_id')
        
        # Overall statistics
        logger.info(f"Total predictions: {len(pred_df)}")
        logger.info(f"Unique sections: {analysis_df['section_id'].nunique()}")
        
        # Section-wise analysis
        section_analysis = analysis_df.groupby('section_id')['prediction'].agg(['count', 'mean', 'std', 'min', 'max']).round(2)
        logger.info("Top 5 sections by average staffing:")
        top_sections = section_analysis.sort_values('mean', ascending=False).head()
        for section_id, stats in top_sections.iterrows():
            logger.info(f"  Section {section_id}: avg={stats['mean']}, std={stats['std']}, range={stats['min']}-{stats['max']}")
        
        # Date-wise patterns (if date parsing works)
        try:
            analysis_df['date_parsed'] = pd.to_datetime(analysis_df['date'])
            analysis_df['weekday'] = analysis_df['date_parsed'].dt.weekday
            weekday_analysis = analysis_df.groupby('weekday')['prediction'].mean().round(2)
            logger.info("Average staffing by weekday (0=Monday):")
            for weekday, avg_staff in weekday_analysis.items():
                logger.info(f"  Weekday {weekday}: {avg_staff}")
        except Exception as e:
            logger.debug(f"Could not analyze date patterns: {e}")
        
        return True
        
    except Exception as e:
        logger.warning(f"Prediction analysis failed: {e}")
        return False

if __name__ == "__main__":
    main()
    validate_predictions()
    analyze_predictions()