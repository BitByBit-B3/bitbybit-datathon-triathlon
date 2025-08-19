import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict, Any

from io_safe import read_csv_safe, validate_columns
from time_utils import parse_datetime_series, calculate_duration_minutes, clean_duration_values
from paths import BOOKINGS_TRAIN_PATH, TASKS_PATH, STAFFING_TRAIN_PATH

logger = logging.getLogger(__name__)

def clean_bookings(bookings_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Clean bookings training data and extract target variable.
    
    Args:
        bookings_df: Raw bookings DataFrame
        
    Returns:
        Tuple of (cleaned_bookings, target_series)
    """
    logger.info("Cleaning bookings data")
    df = bookings_df.copy()
    
    # Log initial state
    logger.info(f"Initial bookings data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Parse datetime columns
    datetime_cols = ['booking_created_time', 'appointment_time', 'check_in_time', 'check_out_time']
    for col in datetime_cols:
        if col in df.columns:
            logger.info(f"Parsing datetime column: {col}")
            df[col] = parse_datetime_series(df[col])
    
    # Calculate target: processing time in minutes
    target = None
    if 'check_in_time' in df.columns and 'check_out_time' in df.columns:
        logger.info("Calculating target from check_in_time and check_out_time")
        target = calculate_duration_minutes(df['check_in_time'], df['check_out_time'])
        
        # Clean duration values (remove negative and extremely long durations)
        target = clean_duration_values(target, min_minutes=0, max_minutes=8*60)
        
        # Log target statistics
        valid_target = target.dropna()
        if len(valid_target) > 0:
            logger.info(f"Target statistics: {len(valid_target)} valid values")
            logger.info(f"Min: {valid_target.min():.1f}, Max: {valid_target.max():.1f}")
            logger.info(f"Mean: {valid_target.mean():.1f}, Median: {valid_target.median():.1f}")
        else:
            logger.warning("No valid target values found!")
    else:
        logger.warning("Cannot calculate target: missing check_in_time or check_out_time")
        # Create a placeholder target series
        target = pd.Series(np.nan, index=df.index)
    
    # Clean numeric columns
    numeric_cols = ['queue_number', 'satisfaction_rating']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Clean boolean columns
    boolean_cols = ['document_uploaded']
    for col in boolean_cols:
        if col in df.columns:
            # Convert various representations to boolean
            df[col] = df[col].map({
                True: True, False: False, 1: True, 0: False,
                '1': True, '0': False, 'true': True, 'false': False,
                'True': True, 'False': False, 'yes': True, 'no': False,
                'Yes': True, 'No': False, 'Y': True, 'N': False
            })
    
    logger.info(f"Cleaned bookings data: {len(df)} rows")
    
    return df, target

def clean_staffing(staffing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean staffing training data.
    
    Args:
        staffing_df: Raw staffing DataFrame
        
    Returns:
        Cleaned staffing DataFrame
    """
    logger.info("Cleaning staffing data")
    df = staffing_df.copy()
    
    # Log initial state
    logger.info(f"Initial staffing data: {len(df)} rows, {len(df.columns)} columns")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Parse date column
    if 'date' in df.columns:
        logger.info("Parsing date column")
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Clean numeric columns
    numeric_cols = ['employees_on_duty', 'total_task_time_minutes']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Log statistics
            valid_values = df[col].dropna()
            if len(valid_values) > 0:
                logger.info(f"{col} stats: min={valid_values.min()}, max={valid_values.max()}, mean={valid_values.mean():.1f}")
    
    # Calculate fallback for employees_on_duty if missing
    if 'employees_on_duty' in df.columns and 'total_task_time_minutes' in df.columns:
        # Calculate median minutes per employee from valid data
        valid_mask = df['employees_on_duty'].notna() & df['total_task_time_minutes'].notna()
        valid_data = df[valid_mask]
        
        if len(valid_data) > 0:
            minutes_per_employee = valid_data['total_task_time_minutes'] / valid_data['employees_on_duty']
            median_minutes_per_employee = minutes_per_employee.median()
            
            logger.info(f"Median minutes per employee: {median_minutes_per_employee:.1f}")
            
            # Fill missing employees_on_duty using this ratio
            missing_mask = df['employees_on_duty'].isna() & df['total_task_time_minutes'].notna()
            if missing_mask.sum() > 0:
                estimated_employees = df.loc[missing_mask, 'total_task_time_minutes'] / median_minutes_per_employee
                df.loc[missing_mask, 'employees_on_duty'] = estimated_employees.round().astype(int)
                logger.info(f"Filled {missing_mask.sum()} missing employees_on_duty values using total_task_time_minutes")
    
    logger.info(f"Cleaned staffing data: {len(df)} rows")
    
    return df

def merge_with_tasks(df: pd.DataFrame, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge a DataFrame with task information.
    
    Args:
        df: DataFrame with task_id column
        tasks_df: Tasks DataFrame
        
    Returns:
        Merged DataFrame with task information
    """
    logger.info("Merging with task information")
    
    # Validate inputs
    if 'task_id' not in df.columns:
        logger.warning("No task_id column found in DataFrame")
        return df
    
    if tasks_df is None or len(tasks_df) == 0:
        logger.warning("No task data provided")
        return df
    
    if 'task_id' not in tasks_df.columns:
        logger.warning("No task_id column found in tasks DataFrame")
        return df
    
    initial_rows = len(df)
    
    # Perform left merge to preserve all original rows
    merged_df = df.merge(tasks_df, on='task_id', how='left')
    
    # Log merge results
    merged_rows = len(merged_df)
    if merged_rows != initial_rows:
        logger.warning(f"Row count changed during merge: {initial_rows} → {merged_rows}")
    
    # Check for unmatched task_ids
    unmatched_mask = merged_df['task_name'].isna() if 'task_name' in tasks_df.columns else merged_df['section_id'].isna()
    unmatched_count = unmatched_mask.sum()
    
    if unmatched_count > 0:
        logger.warning(f"{unmatched_count} rows had unmatched task_ids")
        unmatched_task_ids = df.loc[unmatched_mask, 'task_id'].unique()
        logger.debug(f"Unmatched task_ids: {unmatched_task_ids[:10]}...")  # Log first 10
    
    logger.info(f"Merge complete: {len(merged_df)} rows, {len(merged_df.columns)} columns")
    
    return merged_df

def load_and_clean_training_data() -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Load and clean all training data files.
    
    Returns:
        Tuple of (bookings_df, target_series, staffing_df, tasks_df)
    """
    logger.info("Loading and cleaning all training data")
    
    # Load raw data
    try:
        bookings_raw = read_csv_safe(BOOKINGS_TRAIN_PATH)
        logger.info("✓ Loaded bookings_train.csv")
    except Exception as e:
        logger.error(f"Failed to load bookings_train.csv: {e}")
        bookings_raw = pd.DataFrame()
    
    try:
        tasks_raw = read_csv_safe(TASKS_PATH)
        logger.info("✓ Loaded tasks.csv")
    except Exception as e:
        logger.error(f"Failed to load tasks.csv: {e}")
        tasks_raw = pd.DataFrame()
    
    try:
        staffing_raw = read_csv_safe(STAFFING_TRAIN_PATH)
        logger.info("✓ Loaded staffing_train.csv")
    except Exception as e:
        logger.error(f"Failed to load staffing_train.csv: {e}")
        staffing_raw = pd.DataFrame()
    
    # Clean data
    bookings_df, target_series = clean_bookings(bookings_raw)
    staffing_df = clean_staffing(staffing_raw)
    tasks_df = tasks_raw  # Tasks data typically doesn't need much cleaning
    
    # Merge bookings with tasks if both are available
    if len(bookings_df) > 0 and len(tasks_df) > 0:
        bookings_df = merge_with_tasks(bookings_df, tasks_df)
    
    logger.info("✓ All training data loaded and cleaned")
    
    return bookings_df, target_series, staffing_df, tasks_df

def extract_basic_features(df: pd.DataFrame, feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Extract basic features from a cleaned DataFrame.
    
    Args:
        df: Cleaned DataFrame
        feature_cols: Specific columns to extract features from
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting basic features")
    
    features_df = pd.DataFrame(index=df.index)
    
    if feature_cols is None:
        feature_cols = df.columns.tolist()
    
    for col in feature_cols:
        if col not in df.columns:
            continue
            
        # Numeric features (keep as-is)
        if pd.api.types.is_numeric_dtype(df[col]):
            features_df[col] = df[col]
        
        # Categorical features (keep as-is for now)
        elif pd.api.types.is_object_dtype(df[col]):
            features_df[col] = df[col]
        
        # Boolean features (convert to int)
        elif pd.api.types.is_bool_dtype(df[col]):
            features_df[f"{col}_int"] = df[col].astype(int)
        
        # Datetime features (extract basic components)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            features_df[f"{col}_hour"] = df[col].dt.hour
            features_df[f"{col}_weekday"] = df[col].dt.weekday
            features_df[f"{col}_month"] = df[col].dt.month
    
    logger.info(f"Extracted {len(features_df.columns)} basic features")
    
    return features_df