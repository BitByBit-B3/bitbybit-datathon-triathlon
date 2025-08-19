import pandas as pd
import logging
from pathlib import Path
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)

def read_csv_safe(file_path: Path, **kwargs) -> pd.DataFrame:
    """
    Safely read a CSV file with robust error handling.
    
    Args:
        file_path: Path to the CSV file
        **kwargs: Additional arguments passed to pd.read_csv
        
    Returns:
        DataFrame with the CSV data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file cannot be parsed
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    logger.info(f"Reading CSV: {file_path}")
    
    try:
        # Default arguments for robust CSV reading
        default_args = {
            'encoding': 'utf-8',
            'engine': 'python',
            'on_bad_lines': 'skip'
        }
        default_args.update(kwargs)
        
        df = pd.read_csv(file_path, **default_args)
        logger.info(f"Successfully loaded {len(df)} rows, {len(df.columns)} columns")
        
        if df.empty:
            logger.warning(f"Warning: File {file_path} is empty")
        
        return df
        
    except UnicodeDecodeError:
        logger.warning(f"UTF-8 failed for {file_path}, trying latin-1")
        try:
            default_args['encoding'] = 'latin-1'
            df = pd.read_csv(file_path, **default_args)
            logger.info(f"Successfully loaded with latin-1: {len(df)} rows, {len(df.columns)} columns")
            return df
        except Exception as e:
            logger.error(f"Failed to read {file_path} with latin-1: {e}")
            raise ValueError(f"Cannot parse CSV file: {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to read {file_path}: {e}")
        raise ValueError(f"Cannot parse CSV file: {file_path}")

def write_csv_safe(df: pd.DataFrame, file_path: Path, **kwargs) -> None:
    """
    Safely write a DataFrame to CSV with validation.
    
    Args:
        df: DataFrame to write
        file_path: Output file path
        **kwargs: Additional arguments passed to df.to_csv
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing CSV: {file_path} ({len(df)} rows, {len(df.columns)} columns)")
    
    try:
        # Default arguments for CSV writing
        default_args = {
            'index': False,
            'encoding': 'utf-8'
        }
        default_args.update(kwargs)
        
        df.to_csv(file_path, **default_args)
        logger.info(f"Successfully wrote CSV to {file_path}")
        
    except Exception as e:
        logger.error(f"Failed to write CSV to {file_path}: {e}")
        raise

def validate_columns(df: pd.DataFrame, required_cols: list, file_name: str = "DataFrame") -> None:
    """
    Validate that a DataFrame contains required columns.
    
    Args:
        df: DataFrame to validate
        required_cols: List of required column names
        file_name: Name for logging purposes
        
    Raises:
        ValueError: If required columns are missing
    """
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        available_cols = list(df.columns)
        logger.error(f"{file_name} missing columns: {missing_cols}")
        logger.error(f"Available columns: {available_cols}")
        raise ValueError(f"{file_name} missing required columns: {missing_cols}")
    
    logger.info(f"{file_name} has all required columns: {required_cols}")

def safe_pickle_save(obj: Any, file_path: Path) -> None:
    """
    Safely save an object using pickle.
    
    Args:
        obj: Object to save
        file_path: Output file path
    """
    import pickle
    
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving pickle: {file_path}")
    
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved pickle to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
        raise

def safe_pickle_load(file_path: Path) -> Any:
    """
    Safely load an object using pickle.
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        The loaded object
    """
    import pickle
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"Pickle file not found: {file_path}")
    
    logger.info(f"Loading pickle: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            obj = pickle.load(f)
        logger.info(f"Successfully loaded pickle from {file_path}")
        return obj
    except Exception as e:
        logger.error(f"Failed to load pickle from {file_path}: {e}")
        raise