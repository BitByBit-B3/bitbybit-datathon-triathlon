import pandas as pd
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List, Union
from collections import defaultdict

logger = logging.getLogger(__name__)

class MedianBaseline:
    """
    Median-based baseline predictor with multi-level fallbacks.
    
    This model predicts using median values at different levels of granularity,
    falling back to coarser levels when specific combinations aren't available.
    """
    
    def __init__(self, fallback_levels: List[List[str]], target_col: str = 'target'):
        """
        Initialize the MedianBaseline.
        
        Args:
            fallback_levels: List of feature column lists, ordered from most specific to most general
            target_col: Name of the target column
        """
        self.fallback_levels = fallback_levels
        self.target_col = target_col
        self.median_maps = {}
        self.global_median = None
        self.is_fitted = False
        
        logger.info(f"Initialized MedianBaseline with {len(fallback_levels)} fallback levels")
        for i, level in enumerate(fallback_levels):
            logger.info(f"  Level {i}: {level}")
    
    def fit(self, features_df: pd.DataFrame, target_series: pd.Series):
        """
        Fit the baseline model by computing medians at different levels.
        
        Args:
            features_df: Features DataFrame
            target_series: Target values
        """
        logger.info("Fitting MedianBaseline")
        
        # Combine features and target
        data_df = features_df.copy()
        data_df[self.target_col] = target_series
        
        # Remove rows with missing target
        valid_data = data_df.dropna(subset=[self.target_col])
        logger.info(f"Training on {len(valid_data)} valid samples")
        
        if len(valid_data) == 0:
            logger.warning("No valid training data!")
            self.global_median = 0
            self.is_fitted = True
            return self
        
        # Compute global median as ultimate fallback
        self.global_median = valid_data[self.target_col].median()
        logger.info(f"Global median: {self.global_median:.2f}")
        
        # Compute medians for each fallback level
        self.median_maps = {}
        
        for level_idx, feature_cols in enumerate(self.fallback_levels):
            logger.info(f"Computing medians for level {level_idx}: {feature_cols}")
            
            # Check which columns are available
            available_cols = [col for col in feature_cols if col in valid_data.columns]
            
            if not available_cols:
                logger.warning(f"No available columns for level {level_idx}")
                self.median_maps[level_idx] = {}
                continue
            
            # Group by available columns and compute median
            if len(available_cols) == 1:
                grouped = valid_data.groupby(available_cols[0])[self.target_col].median()
            else:
                grouped = valid_data.groupby(available_cols)[self.target_col].median()
            
            # Convert to dictionary for fast lookup
            median_dict = {}
            if isinstance(grouped.index, pd.MultiIndex):
                for key, value in grouped.items():
                    median_dict[key] = value
            else:
                for key, value in grouped.items():
                    median_dict[(key,)] = value  # Wrap single keys in tuple for consistency
            
            self.median_maps[level_idx] = median_dict
            logger.info(f"Level {level_idx}: computed {len(median_dict)} median values")
        
        self.is_fitted = True
        logger.info("MedianBaseline fitting complete")
        
        return self
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the fitted baseline model.
        
        Args:
            features_df: Features DataFrame
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Making predictions for {len(features_df)} samples")
        
        predictions = []
        level_usage_count = defaultdict(int)
        
        for idx, row in features_df.iterrows():
            prediction = self._predict_single(row, level_usage_count)
            predictions.append(prediction)
        
        # Log fallback level usage
        total_predictions = len(predictions)
        for level_idx, count in level_usage_count.items():
            percentage = (count / total_predictions) * 100
            if level_idx == 'global':
                logger.info(f"Used global fallback: {count} ({percentage:.1f}%)")
            else:
                logger.info(f"Used level {level_idx}: {count} ({percentage:.1f}%)")
        
        predictions = np.array(predictions)
        logger.info(f"Prediction stats: min={predictions.min():.2f}, max={predictions.max():.2f}, mean={predictions.mean():.2f}")
        
        return predictions
    
    def _predict_single(self, row: pd.Series, level_usage_count: Dict) -> float:
        """
        Make a prediction for a single row using fallback strategy.
        
        Args:
            row: Single row of features
            level_usage_count: Dictionary to track fallback level usage
            
        Returns:
            Predicted value
        """
        # Try each fallback level in order
        for level_idx, feature_cols in enumerate(self.fallback_levels):
            # Check which columns are available
            available_cols = [col for col in feature_cols if col in row.index]
            
            if not available_cols:
                continue
            
            # Build lookup key
            key_values = []
            has_missing = False
            
            for col in available_cols:
                val = row[col]
                if pd.isna(val):
                    has_missing = True
                    break
                key_values.append(val)
            
            if has_missing:
                continue
            
            # Look up median value
            if len(key_values) == 1:
                lookup_key = (key_values[0],)
            else:
                lookup_key = tuple(key_values)
            
            median_map = self.median_maps.get(level_idx, {})
            
            if lookup_key in median_map:
                level_usage_count[level_idx] += 1
                return median_map[lookup_key]
        
        # Use global median as final fallback
        level_usage_count['global'] += 1
        return self.global_median if self.global_median is not None else 0

def create_task1_baseline() -> MedianBaseline:
    """
    Create a baseline model for Task 1 (processing time prediction).
    
    Returns:
        Configured MedianBaseline for Task 1
    """
    # Define fallback levels from most specific to most general
    fallback_levels = [
        # Level 0: Task + Time + Weekday (most specific)
        ['task_id', 'appt_hour', 'appt_weekday'],
        
        # Level 1: Task + Time (ignore weekday)
        ['task_id', 'appt_hour'],
        
        # Level 2: Task + Weekday (ignore specific hour)
        ['task_id', 'appt_weekday'],
        
        # Level 3: Task only
        ['task_id'],
        
        # Level 4: Hour + Weekday (ignore task)
        ['appt_hour', 'appt_weekday'],
        
        # Level 5: Hour only
        ['appt_hour'],
        
        # Level 6: Weekday only
        ['appt_weekday']
    ]
    
    return MedianBaseline(fallback_levels, target_col='processing_time_minutes')

def create_task2_baseline() -> MedianBaseline:
    """
    Create a baseline model for Task 2 (staffing prediction).
    
    Returns:
        Configured MedianBaseline for Task 2
    """
    # Define fallback levels from most specific to most general
    fallback_levels = [
        # Level 0: Section + Weekday (most specific for staffing)
        ['section_id', 'weekday'],
        
        # Level 1: Section only
        ['section_id'],
        
        # Level 2: Weekday only (general staffing patterns)
        ['weekday'],
        
        # Level 3: Weekend vs weekday
        ['is_weekend']
    ]
    
    return MedianBaseline(fallback_levels, target_col='employees_on_duty')

class SimpleStatsBaseline:
    """
    Simple statistical baseline using mean/median by key columns.
    """
    
    def __init__(self, key_cols: List[str], stat_type: str = 'median'):
        """
        Initialize SimpleStatsBaseline.
        
        Args:
            key_cols: Columns to group by
            stat_type: 'median' or 'mean'
        """
        self.key_cols = key_cols
        self.stat_type = stat_type
        self.stats_map = {}
        self.global_stat = None
        self.is_fitted = False
    
    def fit(self, features_df: pd.DataFrame, target_series: pd.Series):
        """Fit the baseline model."""
        logger.info(f"Fitting SimpleStatsBaseline with {self.stat_type} by {self.key_cols}")
        
        # Combine data
        data_df = features_df.copy()
        data_df['target'] = target_series
        
        # Remove missing targets
        valid_data = data_df.dropna(subset=['target'])
        
        if len(valid_data) == 0:
            logger.warning("No valid training data!")
            self.global_stat = 0
            self.is_fitted = True
            return self
        
        # Compute global statistic
        if self.stat_type == 'median':
            self.global_stat = valid_data['target'].median()
        else:
            self.global_stat = valid_data['target'].mean()
        
        # Group by key columns
        available_cols = [col for col in self.key_cols if col in valid_data.columns]
        
        if available_cols:
            if self.stat_type == 'median':
                grouped = valid_data.groupby(available_cols)['target'].median()
            else:
                grouped = valid_data.groupby(available_cols)['target'].mean()
            
            self.stats_map = grouped.to_dict()
            logger.info(f"Computed statistics for {len(self.stats_map)} groups")
        else:
            logger.warning("No available key columns")
        
        self.is_fitted = True
        return self
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in features_df.iterrows():
            # Build key
            key_values = []
            has_missing = False
            
            for col in self.key_cols:
                if col in row.index:
                    val = row[col]
                    if pd.isna(val):
                        has_missing = True
                        break
                    key_values.append(val)
                else:
                    has_missing = True
                    break
            
            if has_missing or len(key_values) != len(self.key_cols):
                prediction = self.global_stat
            else:
                if len(key_values) == 1:
                    lookup_key = key_values[0]
                else:
                    lookup_key = tuple(key_values)
                
                prediction = self.stats_map.get(lookup_key, self.global_stat)
            
            predictions.append(prediction)
        
        return np.array(predictions)

def validate_baseline_predictions(predictions: np.ndarray, 
                                task_name: str,
                                min_value: float = 0,
                                max_value: Optional[float] = None) -> np.ndarray:
    """
    Validate and clean baseline predictions.
    
    Args:
        predictions: Raw predictions
        task_name: Name of the task for logging
        min_value: Minimum allowed value
        max_value: Maximum allowed value (optional)
        
    Returns:
        Cleaned predictions
    """
    logger.info(f"Validating {task_name} baseline predictions")
    
    original_predictions = predictions.copy()
    
    # Replace NaN/inf values
    nan_mask = ~np.isfinite(predictions)
    if nan_mask.sum() > 0:
        logger.warning(f"Found {nan_mask.sum()} non-finite predictions, replacing with {min_value}")
        predictions[nan_mask] = min_value
    
    # Apply bounds
    predictions = np.clip(predictions, min_value, max_value)
    
    # Round to integers for final output
    predictions = np.round(predictions).astype(int)
    
    # Log statistics
    logger.info(f"Final predictions: min={predictions.min()}, max={predictions.max()}, mean={predictions.mean():.1f}")
    
    # Check for major changes
    if not np.array_equal(original_predictions, predictions):
        changed_count = (original_predictions != predictions).sum()
        logger.info(f"Modified {changed_count} predictions during validation")
    
    return predictions