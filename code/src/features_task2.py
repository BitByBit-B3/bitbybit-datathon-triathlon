import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple

from time_utils import parse_date_safe

logger = logging.getLogger(__name__)

def extract_task2_features(staffing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features for Task 2 (staffing prediction).
    
    Args:
        staffing_df: Cleaned staffing DataFrame
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting Task 2 features")
    
    features_df = pd.DataFrame(index=staffing_df.index)
    
    # Basic features
    if 'section_id' in staffing_df.columns:
        features_df['section_id'] = staffing_df['section_id']
    
    # Date-based features
    if 'date' in staffing_df.columns:
        logger.info("Extracting date-based features")
        date_col = staffing_df['date']
        
        # Extract date components
        features_df['weekday'] = date_col.dt.weekday  # 0=Monday, 6=Sunday
        features_df['month'] = date_col.dt.month
        features_df['day'] = date_col.dt.day
        features_df['year'] = date_col.dt.year
        
        # Weekend indicator
        features_df['is_weekend'] = date_col.dt.weekday >= 5
        
        # Month categories (seasonal patterns)
        features_df['quarter'] = date_col.dt.quarter
        features_df['is_month_end'] = date_col.dt.is_month_end
        features_df['is_month_start'] = date_col.dt.is_month_start
    
    # Task time features
    if 'total_task_time_minutes' in staffing_df.columns:
        features_df['total_task_time_minutes'] = staffing_df['total_task_time_minutes']
        features_df['has_task_time'] = staffing_df['total_task_time_minutes'].notna()
        
        # Task time categories
        task_time = staffing_df['total_task_time_minutes'].fillna(0)
        features_df['task_time_category'] = pd.cut(
            task_time,
            bins=[0, 120, 240, 480, 960, float('inf')],
            labels=['very_low', 'low', 'medium', 'high', 'very_high'],
            include_lowest=True
        )
    
    # Historical staffing level (if available)
    if 'employees_on_duty' in staffing_df.columns:
        features_df['historical_employees'] = staffing_df['employees_on_duty']
    
    logger.info(f"Extracted {len(features_df.columns)} Task 2 features")
    
    return features_df

def extract_test_features_task2(test_inputs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from Task 2 test inputs.
    
    Args:
        test_inputs_df: Test inputs with columns: row_id, date, section_id
        
    Returns:
        DataFrame with extracted features
    """
    logger.info("Extracting Task 2 test features")
    
    features_df = pd.DataFrame(index=test_inputs_df.index)
    
    # Preserve row_id for final output
    features_df['row_id'] = test_inputs_df['row_id']
    features_df['section_id'] = test_inputs_df['section_id']
    
    # Parse and extract date features
    if 'date' in test_inputs_df.columns:
        logger.info("Parsing test input dates")
        
        # Parse dates
        test_dates = pd.to_datetime(test_inputs_df['date'], errors='coerce')
        
        # Extract same date features as training
        features_df['weekday'] = test_dates.dt.weekday
        features_df['month'] = test_dates.dt.month
        features_df['day'] = test_dates.dt.day
        features_df['year'] = test_dates.dt.year
        features_df['is_weekend'] = test_dates.dt.weekday >= 5
        features_df['quarter'] = test_dates.dt.quarter
        features_df['is_month_end'] = test_dates.dt.is_month_end
        features_df['is_month_start'] = test_dates.dt.is_month_start
    
    # Default values for features not available in test
    default_features = {
        'total_task_time_minutes': np.nan,
        'has_task_time': False,
        'task_time_category': 'medium',  # Default category
        'historical_employees': np.nan
    }
    
    for feature, default_val in default_features.items():
        features_df[feature] = default_val
    
    logger.info(f"Extracted {len(features_df.columns)} Task 2 test features")
    
    return features_df

def prepare_task2_features_for_training(features_df: pd.DataFrame, 
                                       staffing_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare Task 2 features for model training and extract target.
    
    Args:
        features_df: Raw features DataFrame
        staffing_df: Staffing DataFrame with target information
        
    Returns:
        Tuple of (prepared_features, target_series)
    """
    logger.info("Preparing Task 2 features for training")
    
    # Extract target variable
    target_series = extract_task2_target(staffing_df)
    
    # Align features and target
    common_idx = features_df.index.intersection(target_series.index)
    features_aligned = features_df.loc[common_idx].copy()
    target_aligned = target_series.loc[common_idx].copy()
    
    # Remove rows with missing target
    valid_mask = target_aligned.notna()
    features_valid = features_aligned[valid_mask].copy()
    target_valid = target_aligned[valid_mask].copy()
    
    logger.info(f"Valid training samples: {len(features_valid)} / {len(features_df)}")
    
    # Clean features for modeling
    features_clean = clean_features_for_modeling(features_valid)
    
    return features_clean, target_valid

def extract_task2_target(staffing_df: pd.DataFrame) -> pd.Series:
    """
    Extract target variable for Task 2 (number of employees needed).
    
    Args:
        staffing_df: Staffing DataFrame
        
    Returns:
        Target series (employees_on_duty)
    """
    logger.info("Extracting Task 2 target variable")
    
    if 'employees_on_duty' in staffing_df.columns:
        target = staffing_df['employees_on_duty'].copy()
        
        # Clean target values
        target = target.replace([np.inf, -np.inf], np.nan)
        target = target[target >= 0]  # Remove negative values
        
        # Log target statistics
        valid_target = target.dropna()
        if len(valid_target) > 0:
            logger.info(f"Target statistics: {len(valid_target)} valid values")
            logger.info(f"Min: {valid_target.min()}, Max: {valid_target.max()}")
            logger.info(f"Mean: {valid_target.mean():.1f}, Median: {valid_target.median():.1f}")
        else:
            logger.warning("No valid target values found!")
        
        return target
    
    elif 'total_task_time_minutes' in staffing_df.columns:
        logger.warning("employees_on_duty not found, deriving from total_task_time_minutes")
        
        # Use a simple heuristic: assume 8 hours per employee per day
        minutes_per_employee_day = 8 * 60  # 480 minutes
        
        target = staffing_df['total_task_time_minutes'] / minutes_per_employee_day
        target = target.round().astype(int)
        
        # Ensure minimum of 1 employee if there's any work
        target = target.clip(lower=1)
        
        logger.info(f"Derived {target.notna().sum()} employee estimates from task time")
        
        return target
    
    else:
        logger.error("Cannot extract target: no employees_on_duty or total_task_time_minutes")
        return pd.Series(np.nan, index=staffing_df.index)

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
        if col in ['row_id']:  # Skip ID columns
            continue
            
        # Convert to categorical and then to codes
        cleaned_df[col] = pd.Categorical(cleaned_df[col]).codes
        
        # Replace -1 (missing) with a large number to distinguish from valid categories
        cleaned_df[col] = cleaned_df[col].replace(-1, 999999)
    
    # Fill missing values for numeric columns
    numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if cleaned_df[col].isna().any():
            # Use median for numeric features, 0 for count-like features
            if col in ['employees_on_duty', 'total_task_time_minutes']:
                fill_val = 0
            else:
                fill_val = cleaned_df[col].median()
            
            cleaned_df[col] = cleaned_df[col].fillna(fill_val)
    
    # Handle boolean columns
    bool_cols = cleaned_df.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        cleaned_df[col] = cleaned_df[col].astype(int)
    
    logger.info(f"Cleaned {len(cleaned_df.columns)} features for modeling")
    
    return cleaned_df

def create_section_aggregates(staffing_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create aggregate features by section to capture section-level patterns.
    
    Args:
        staffing_df: Cleaned staffing DataFrame
        
    Returns:
        DataFrame with section-level aggregate features
    """
    logger.info("Creating section-level aggregate features")
    
    if 'section_id' not in staffing_df.columns:
        logger.warning("No section_id column found")
        return pd.DataFrame()
    
    aggregates_list = []
    
    # Group by section
    for section_id, group in staffing_df.groupby('section_id'):
        section_stats = {'section_id': section_id}
        
        # Employee statistics
        if 'employees_on_duty' in group.columns:
            employees = group['employees_on_duty'].dropna()
            if len(employees) > 0:
                section_stats.update({
                    'section_avg_employees': employees.mean(),
                    'section_median_employees': employees.median(),
                    'section_max_employees': employees.max(),
                    'section_min_employees': employees.min(),
                    'section_std_employees': employees.std()
                })
        
        # Task time statistics
        if 'total_task_time_minutes' in group.columns:
            task_time = group['total_task_time_minutes'].dropna()
            if len(task_time) > 0:
                section_stats.update({
                    'section_avg_task_time': task_time.mean(),
                    'section_median_task_time': task_time.median(),
                    'section_total_observations': len(group)
                })
        
        aggregates_list.append(section_stats)
    
    if aggregates_list:
        aggregates_df = pd.DataFrame(aggregates_list)
        logger.info(f"Created aggregates for {len(aggregates_df)} sections")
        return aggregates_df
    else:
        logger.warning("No section aggregates created")
        return pd.DataFrame()