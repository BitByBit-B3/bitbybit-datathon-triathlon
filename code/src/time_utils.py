import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Union, Optional, Tuple
import dateutil.parser
import pytz

logger = logging.getLogger(__name__)

def parse_datetime_safe(dt_str: str, default_tz: str = 'UTC') -> Optional[datetime]:
    """
    Safely parse datetime strings with various formats.
    
    Args:
        dt_str: String representation of datetime
        default_tz: Default timezone if none specified
        
    Returns:
        Parsed datetime object or None if parsing fails
    """
    if pd.isna(dt_str) or not dt_str:
        return None
    
    try:
        # Try parsing with dateutil (handles many formats)
        dt = dateutil.parser.parse(str(dt_str))
        
        # If no timezone info, assume UTC
        if dt.tzinfo is None:
            dt = pytz.timezone(default_tz).localize(dt)
        
        return dt
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse datetime '{dt_str}': {e}")
        return None

def parse_datetime_series(series: pd.Series, default_tz: str = 'UTC') -> pd.Series:
    """
    Parse a pandas Series of datetime strings.
    
    Args:
        series: Series containing datetime strings
        default_tz: Default timezone if none specified
        
    Returns:
        Series with parsed datetime objects
    """
    logger.info(f"Parsing datetime series with {len(series)} values")
    
    # First try pandas built-in parsing (fastest)
    try:
        parsed = pd.to_datetime(series, errors='coerce', utc=True)
        valid_count = parsed.notna().sum()
        logger.info(f"Pandas parsing succeeded for {valid_count}/{len(series)} values")
        
        if valid_count > len(series) * 0.8:  # If 80%+ successful, use it
            return parsed
    except:
        pass
    
    # Fallback to manual parsing
    logger.info("Using manual datetime parsing")
    parsed_values = []
    
    for val in series:
        parsed_dt = parse_datetime_safe(val, default_tz)
        parsed_values.append(parsed_dt)
    
    result = pd.Series(parsed_values, index=series.index)
    valid_count = result.notna().sum()
    logger.info(f"Manual parsing succeeded for {valid_count}/{len(series)} values")
    
    return result

def calculate_duration_minutes(start_series: pd.Series, end_series: pd.Series) -> pd.Series:
    """
    Calculate duration in minutes between two datetime series.
    
    Args:
        start_series: Series of start times
        end_series: Series of end times
        
    Returns:
        Series with duration in minutes (float)
    """
    logger.info("Calculating duration in minutes")
    
    # Ensure both series are datetime
    if not pd.api.types.is_datetime64_any_dtype(start_series):
        start_series = parse_datetime_series(start_series)
    
    if not pd.api.types.is_datetime64_any_dtype(end_series):
        end_series = parse_datetime_series(end_series)
    
    # Calculate duration
    duration_timedelta = end_series - start_series
    duration_minutes = duration_timedelta.dt.total_seconds() / 60.0
    
    # Log statistics
    valid_mask = duration_minutes.notna()
    if valid_mask.sum() > 0:
        valid_durations = duration_minutes[valid_mask]
        logger.info(f"Duration stats: min={valid_durations.min():.1f}, "
                   f"max={valid_durations.max():.1f}, "
                   f"mean={valid_durations.mean():.1f}, "
                   f"median={valid_durations.median():.1f}")
        
        # Warn about negative durations
        negative_count = (valid_durations < 0).sum()
        if negative_count > 0:
            logger.warning(f"Found {negative_count} negative durations")
        
        # Warn about very long durations (>8 hours = 480 minutes)
        long_count = (valid_durations > 480).sum()
        if long_count > 0:
            logger.warning(f"Found {long_count} durations > 8 hours")
    
    return duration_minutes

def extract_time_features(dt_series: pd.Series) -> pd.DataFrame:
    """
    Extract time-based features from a datetime series.
    
    Args:
        dt_series: Series of datetime objects
        
    Returns:
        DataFrame with extracted features
    """
    if not pd.api.types.is_datetime64_any_dtype(dt_series):
        dt_series = parse_datetime_series(dt_series)
    
    features_df = pd.DataFrame(index=dt_series.index)
    
    # Extract basic time components
    features_df['hour'] = dt_series.dt.hour
    features_df['weekday'] = dt_series.dt.weekday  # 0=Monday, 6=Sunday
    features_df['month'] = dt_series.dt.month
    features_df['day'] = dt_series.dt.day
    features_df['year'] = dt_series.dt.year
    
    # Time of day categories
    features_df['time_of_day'] = pd.cut(
        dt_series.dt.hour,
        bins=[0, 6, 12, 18, 24],
        labels=['night', 'morning', 'afternoon', 'evening'],
        include_lowest=True
    )
    
    # Weekend indicator
    features_df['is_weekend'] = dt_series.dt.weekday >= 5
    
    # Business hour indicator (9 AM - 5 PM on weekdays)
    is_weekday = dt_series.dt.weekday < 5
    is_business_hour = (dt_series.dt.hour >= 9) & (dt_series.dt.hour < 17)
    features_df['is_business_hour'] = is_weekday & is_business_hour
    
    logger.info(f"Extracted {len(features_df.columns)} time features")
    
    return features_df

def parse_date_safe(date_str: str) -> Optional[datetime]:
    """
    Safely parse date strings (YYYY-MM-DD format primarily).
    
    Args:
        date_str: String representation of date
        
    Returns:
        Parsed date object or None if parsing fails
    """
    if pd.isna(date_str) or not date_str:
        return None
    
    try:
        # Try common date formats first
        for fmt in ['%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y', '%Y/%m/%d']:
            try:
                return datetime.strptime(str(date_str), fmt)
            except ValueError:
                continue
        
        # Fallback to dateutil
        return dateutil.parser.parse(str(date_str)).replace(hour=0, minute=0, second=0)
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse date '{date_str}': {e}")
        return None

def parse_time_safe(time_str: str) -> Optional[Tuple[int, int]]:
    """
    Safely parse time strings (HH:MM format).
    
    Args:
        time_str: String representation of time
        
    Returns:
        Tuple of (hour, minute) or None if parsing fails
    """
    if pd.isna(time_str) or not time_str:
        return None
    
    try:
        time_str = str(time_str).strip()
        
        # Handle HH:MM format
        if ':' in time_str:
            parts = time_str.split(':')
            if len(parts) == 2:
                hour = int(parts[0])
                minute = int(parts[1])
                
                if 0 <= hour <= 23 and 0 <= minute <= 59:
                    return (hour, minute)
        
        # Handle HHMM format (no colon)
        elif len(time_str) == 4 and time_str.isdigit():
            hour = int(time_str[:2])
            minute = int(time_str[2:])
            
            if 0 <= hour <= 23 and 0 <= minute <= 59:
                return (hour, minute)
        
        return None
        
    except (ValueError, TypeError) as e:
        logger.debug(f"Failed to parse time '{time_str}': {e}")
        return None

def clean_duration_values(duration_series: pd.Series, 
                         min_minutes: float = 0, 
                         max_minutes: float = 480) -> pd.Series:
    """
    Clean duration values by removing outliers and invalid values.
    
    Args:
        duration_series: Series of duration values in minutes
        min_minutes: Minimum valid duration
        max_minutes: Maximum valid duration (default 8 hours)
        
    Returns:
        Cleaned series with outliers set to NaN
    """
    original_count = len(duration_series)
    valid_original = duration_series.notna().sum()
    
    # Remove negative durations
    cleaned = duration_series.copy()
    cleaned[cleaned < min_minutes] = np.nan
    
    # Remove extremely long durations
    cleaned[cleaned > max_minutes] = np.nan
    
    valid_final = cleaned.notna().sum()
    removed_count = valid_original - valid_final
    
    if removed_count > 0:
        logger.info(f"Removed {removed_count} invalid durations "
                   f"(outside range {min_minutes}-{max_minutes} minutes)")
    
    logger.info(f"Duration cleaning: {original_count} → {valid_final} valid values")
    
    return cleaned