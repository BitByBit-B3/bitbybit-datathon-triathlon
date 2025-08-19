import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

from time_utils import extract_time_features, parse_date_safe, parse_time_safe

logger = logging.getLogger(__name__)

def extract_task1_features(bookings_df: pd.DataFrame, tasks_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract features for Task 1 (processing time prediction).
    
    Args:
        bookings_df: Cleaned bookings DataFrame
        tasks_df: Optional tasks DataFrame for additional features
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting Task 1 features")
    
    features_df = pd.DataFrame(index=bookings_df.index)
    
    # Basic ID features
    if 'booking_id' in bookings_df.columns:
        features_df['booking_id'] = bookings_df['booking_id']
    
    if 'task_id' in bookings_df.columns:
        features_df['task_id'] = bookings_df['task_id']
    
    # Task-related features from tasks table
    if tasks_df is not None and 'task_id' in bookings_df.columns:
        task_features = extract_task_features(bookings_df['task_id'], tasks_df)
        features_df = pd.concat([features_df, task_features], axis=1)
    
    # Time-based features from appointment_time
    if 'appointment_time' in bookings_df.columns:
        logger.info("Extracting appointment time features")
        appt_time_features = extract_time_features(bookings_df['appointment_time'])
        
        # Rename columns to be specific to appointment time
        appt_time_features.columns = [f'appt_{col}' for col in appt_time_features.columns]
        features_df = pd.concat([features_df, appt_time_features], axis=1)
    
    # Time-based features from check_in_time
    if 'check_in_time' in bookings_df.columns:
        logger.info("Extracting check-in time features")
        checkin_time_features = extract_time_features(bookings_df['check_in_time'])
        
        # Rename columns to be specific to check-in time
        checkin_time_features.columns = [f'checkin_{col}' for col in checkin_time_features.columns]
        features_df = pd.concat([features_df, checkin_time_features], axis=1)
    
    # Booking behavior features
    if 'booking_created_time' in bookings_df.columns and 'appointment_time' in bookings_df.columns:
        logger.info("Extracting booking advance time feature")
        advance_time = bookings_df['appointment_time'] - bookings_df['booking_created_time']
        features_df['booking_advance_hours'] = advance_time.dt.total_seconds() / 3600.0
    
    # Queue-related features
    if 'queue_number' in bookings_df.columns:
        features_df['queue_number'] = bookings_df['queue_number']
        features_df['has_queue_number'] = bookings_df['queue_number'].notna()
    
    # Document upload feature
    if 'document_uploaded' in bookings_df.columns:
        features_df['document_uploaded'] = bookings_df['document_uploaded'].fillna(False).astype(int)
    
    # Satisfaction rating (if available, though it might be post-service)
    if 'satisfaction_rating' in bookings_df.columns:
        features_df['satisfaction_rating'] = bookings_df['satisfaction_rating']
        features_df['has_satisfaction_rating'] = bookings_df['satisfaction_rating'].notna()
    
    logger.info(f"Extracted {len(features_df.columns)} Task 1 features")
    
    return features_df

def extract_task_features(task_ids: pd.Series, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features related to tasks.
    
    Args:
        task_ids: Series of task IDs
        tasks_df: Tasks DataFrame
        
    Returns:
        DataFrame with task-related features
    """
    logger.info("Extracting task-related features")
    
    features_df = pd.DataFrame(index=task_ids.index)
    
    # Merge with tasks data
    task_info = task_ids.to_frame('task_id').merge(tasks_df, on='task_id', how='left')
    
    # Section ID (categorical feature)
    if 'section_id' in task_info.columns:
        features_df['section_id'] = task_info['section_id']
    
    # Task name (we'll keep it as categorical for now)
    if 'task_name' in task_info.columns:
        features_df['task_name'] = task_info['task_name']
        features_df['has_task_name'] = task_info['task_name'].notna()
    
    # Section name (we'll keep it as categorical for now)
    if 'section_name' in task_info.columns:
        features_df['section_name'] = task_info['section_name']
        features_df['has_section_name'] = task_info['section_name'].notna()
    
    logger.info(f"Extracted {len(features_df.columns)} task-related features")
    
    return features_df

def extract_test_features_task1(test_inputs_df: pd.DataFrame, 
                               tasks_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """
    Extract features from Task 1 test inputs.
    
    Args:
        test_inputs_df: Test inputs with columns: row_id, date, time, task_id
        tasks_df: Optional tasks DataFrame
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting Task 1 test features")
    
    features_df = pd.DataFrame(index=test_inputs_df.index)
    
    # Preserve row_id for final output
    features_df['row_id'] = test_inputs_df['row_id']
    features_df['task_id'] = test_inputs_df['task_id']
    
    # Parse date and time to create datetime features
    if 'date' in test_inputs_df.columns and 'time' in test_inputs_df.columns:
        logger.info("Parsing test input date and time")
        
        # Combine date and time strings
        datetime_strings = test_inputs_df['date'].astype(str) + ' ' + test_inputs_df['time'].astype(str)
        
        # Parse to datetime
        test_datetimes = pd.to_datetime(datetime_strings, errors='coerce')
        
        # Extract time features using the same logic as training
        time_features = extract_time_features(test_datetimes)
        
        # Rename to match appointment time features from training
        time_features.columns = [f'appt_{col}' for col in time_features.columns]
        features_df = pd.concat([features_df, time_features], axis=1)
        
        # Also create check-in time features (assuming they might be similar)
        time_features_checkin = extract_time_features(test_datetimes)
        time_features_checkin.columns = [f'checkin_{col}' for col in time_features_checkin.columns]
        features_df = pd.concat([features_df, time_features_checkin], axis=1)
    
    # Task-related features
    if tasks_df is not None and 'task_id' in test_inputs_df.columns:
        task_features = extract_task_features(test_inputs_df['task_id'], tasks_df)
        features_df = pd.concat([features_df, task_features], axis=1)
    
    # Fill missing values for features that might not be present in test
    default_features = {
        'queue_number': np.nan,
        'has_queue_number': False,
        'document_uploaded': 0,
        'booking_advance_hours': np.nan,
        'satisfaction_rating': np.nan,
        'has_satisfaction_rating': False
    }
    
    for feature, default_val in default_features.items():
        if feature not in features_df.columns:
            features_df[feature] = default_val
    
    logger.info(f"Extracted {len(features_df.columns)} Task 1 test features")
    
    return features_df

def prepare_task1_features_for_training(features_df: pd.DataFrame, 
                                       target_series: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Task 1 features for model training.
    
    Args:
        features_df: Raw features DataFrame
        target_series: Target values
        
    Returns:
        Tuple of (prepared_features, valid_target)
    """
    logger.info("Preparing Task 1 features for training")
    
    # Align features and target
    common_idx = features_df.index.intersection(target_series.index)
    features_aligned = features_df.loc[common_idx].copy()
    target_aligned = target_series.loc[common_idx].copy()
    
    # Remove rows with missing target
    valid_mask = target_aligned.notna()
    features_valid = features_aligned[valid_mask].copy()
    target_valid = target_aligned[valid_mask].copy()
    
    logger.info(f"Valid training samples: {len(features_valid)} / {len(features_df)}")
    
    # Basic feature cleaning
    features_clean = clean_features_for_modeling(features_valid)
    
    return features_clean, target_valid

def clean_features_for_modeling(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean features for modeling by handling missing values and categorical variables.
    
    Args:
        features_df: Raw features DataFrame
        
    Returns:
        Cleaned features DataFrame
    """
    logger.info("Cleaning features for modeling")
    
    cleaned_df = features_df.copy()
    
    # Handle categorical variables - convert to category codes
    categorical_cols = cleaned_df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in ['row_id', 'booking_id']:  # Skip ID columns
            continue
            
        # Convert to categorical and then to codes
        cleaned_df[col] = pd.Categorical(cleaned_df[col]).codes
        
        # Replace -1 (missing) with a large number to distinguish from valid categories
        cleaned_df[col] = cleaned_df[col].replace(-1, 999999)
    
    # Fill missing values for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].isna().any():
            # Use median for numeric features
            median_val = cleaned_df[col].median()
            cleaned_df[col] = cleaned_df[col].fillna(median_val)
    
    logger.info(f"Cleaned {len(cleaned_df.columns)} features for modeling")
    
    return cleaned_df