import pandas as pd
import numpy as np
import logging
from typing import Optional, Tuple, Dict, Any
import os

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, using baseline models only")

from baselines import MedianBaseline, create_task2_baseline

logger = logging.getLogger(__name__)

class Task2ModelEnsemble:
    """
    Ensemble model for Task 2 combining baseline and optional ML models.
    """
    
    def __init__(self, use_advanced_models: bool = None):
        """
        Initialize Task2ModelEnsemble.
        
        Args:
            use_advanced_models: Whether to use ML models. If None, check environment.
        """
        if use_advanced_models is None:
            use_advanced_models = os.getenv('USE_ADVANCED_MODELS', '0') == '1'
        
        self.use_advanced_models = use_advanced_models and SKLEARN_AVAILABLE
        self.baseline_model = create_task2_baseline()
        self.ml_model = None
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False
        
        if self.use_advanced_models:
            logger.info("Task2ModelEnsemble: Using advanced ML models")
        else:
            logger.info("Task2ModelEnsemble: Using baseline model only")
    
    def fit(self, features_df: pd.DataFrame, target_series: pd.Series) -> 'Task2ModelEnsemble':
        """
        Fit the ensemble model.
        
        Args:
            features_df: Training features
            target_series: Training targets
            
        Returns:
            Self for method chaining
        """
        logger.info("Fitting Task2ModelEnsemble")
        
        # Always fit baseline model
        self.baseline_model.fit(features_df, target_series)
        
        # Optionally fit ML model
        if self.use_advanced_models:
            try:
                self._fit_ml_model(features_df, target_series)
            except Exception as e:
                logger.warning(f"ML model fitting failed, using baseline only: {e}")
                self.use_advanced_models = False
        
        self.is_fitted = True
        logger.info("Task2ModelEnsemble fitting complete")
        
        return self
    
    def _fit_ml_model(self, features_df: pd.DataFrame, target_series: pd.Series):
        """Fit the ML component of the ensemble."""
        logger.info("Fitting ML model for Task 2")
        
        # Prepare features for ML
        ml_features = self._prepare_ml_features(features_df)
        
        # Remove rows with missing targets
        valid_mask = target_series.notna()
        X_train = ml_features[valid_mask]
        y_train = target_series[valid_mask]
        
        if len(X_train) < 10:
            logger.warning("Too few valid samples for ML model")
            return
        
        # Store feature columns for later use
        self.feature_columns = X_train.columns.tolist()
        
        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Fit linear regression (simple and robust)
        self.ml_model = LinearRegression()
        self.ml_model.fit(X_scaled, y_train)
        
        # Log performance
        y_pred = self.ml_model.predict(X_scaled)
        mae = mean_absolute_error(y_train, y_pred)
        rmse = np.sqrt(mean_squared_error(y_train, y_pred))
        
        logger.info(f"ML model training MAE: {mae:.2f}")
        logger.info(f"ML model training RMSE: {rmse:.2f}")
        
        logger.info("ML model fitting complete")
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions using the ensemble.
        
        Args:
            features_df: Features for prediction
            
        Returns:
            Array of predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        logger.info(f"Making Task 2 predictions for {len(features_df)} samples")
        
        # Always get baseline predictions
        baseline_preds = self.baseline_model.predict(features_df)
        
        if not self.use_advanced_models or self.ml_model is None:
            return self._post_process_predictions(baseline_preds)
        
        # Get ML predictions
        try:
            ml_preds = self._predict_ml(features_df)
            
            # Ensemble: weighted average (favor baseline for robustness)
            ensemble_preds = 0.7 * baseline_preds + 0.3 * ml_preds
            
            logger.info("Using ensemble predictions (70% baseline, 30% ML)")
            
            return self._post_process_predictions(ensemble_preds)
            
        except Exception as e:
            logger.warning(f"ML prediction failed, using baseline only: {e}")
            return self._post_process_predictions(baseline_preds)
    
    def _predict_ml(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using the ML model."""
        if self.ml_model is None or self.scaler is None:
            raise ValueError("ML model not fitted")
        
        # Prepare features
        ml_features = self._prepare_ml_features(features_df)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in ml_features.columns:
                ml_features[col] = 0  # Default value for missing features
        
        ml_features = ml_features[self.feature_columns]  # Reorder columns
        
        # Scale and predict
        X_scaled = self.scaler.transform(ml_features)
        predictions = self.ml_model.predict(X_scaled)
        
        return predictions
    
    def _prepare_ml_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features specifically for ML model."""
        # Select numeric features only for simplicity
        numeric_features = features_df.select_dtypes(include=[np.number]).copy()
        
        # Fill missing values
        numeric_features = numeric_features.fillna(numeric_features.median())
        
        # Handle infinite values
        numeric_features = numeric_features.replace([np.inf, -np.inf], np.nan)
        numeric_features = numeric_features.fillna(0)
        
        return numeric_features
    
    def _post_process_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Post-process predictions to ensure valid values."""
        # Ensure positive values (can't have negative employees)
        predictions = np.maximum(predictions, 1)  # At least 1 employee
        
        # Cap at reasonable maximum (50 employees per section)
        predictions = np.minimum(predictions, 50)
        
        # Round to integers
        predictions = np.round(predictions).astype(int)
        
        return predictions

def create_simple_task2_model(model_type: str = 'baseline') -> Any:
    """
    Create a simple model for Task 2.
    
    Args:
        model_type: Type of model ('baseline', 'linear', 'ensemble')
        
    Returns:
        Configured model
    """
    if model_type == 'baseline':
        return create_task2_baseline()
    elif model_type == 'ensemble':
        return Task2ModelEnsemble(use_advanced_models=True)
    elif model_type == 'linear' and SKLEARN_AVAILABLE:
        # Simple wrapper around sklearn LinearRegression
        return SklearnWrapper(LinearRegression())
    else:
        logger.warning(f"Model type '{model_type}' not available, using baseline")
        return create_task2_baseline()

class SklearnWrapper:
    """Simple wrapper to make sklearn models compatible with our interface."""
    
    def __init__(self, model):
        self.model = model
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_fitted = False
    
    def fit(self, features_df: pd.DataFrame, target_series: pd.Series):
        # Prepare features
        X = self._prepare_features(features_df)
        y = target_series.reindex(X.index).dropna()
        X = X.reindex(y.index)
        
        if len(X) == 0:
            raise ValueError("No valid training data")
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale and fit
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        return self
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X = self._prepare_features(features_df)
        
        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Post-process
        predictions = np.maximum(predictions, 1)  # At least 1 employee
        predictions = np.minimum(predictions, 50)  # Max 50 employees
        predictions = np.round(predictions).astype(int)
        
        return predictions
    
    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).copy()
        numeric_features = numeric_features.fillna(numeric_features.median())
        numeric_features = numeric_features.replace([np.inf, -np.inf], 0)
        return numeric_features

class StaffingPatternModel:
    """
    Specialized model for staffing patterns that considers typical business patterns.
    """
    
    def __init__(self):
        self.section_patterns = {}
        self.weekday_patterns = {}
        self.global_average = 1
        self.is_fitted = False
    
    def fit(self, features_df: pd.DataFrame, target_series: pd.Series):
        """Fit the staffing pattern model."""
        logger.info("Fitting StaffingPatternModel")
        
        # Combine features and target
        data_df = features_df.copy()
        data_df['employees'] = target_series
        valid_data = data_df.dropna(subset=['employees'])
        
        if len(valid_data) == 0:
            logger.warning("No valid training data for staffing patterns")
            self.global_average = 1
            self.is_fitted = True
            return self
        
        # Global average
        self.global_average = max(1, int(valid_data['employees'].mean()))
        
        # Section patterns
        if 'section_id' in valid_data.columns:
            section_stats = valid_data.groupby('section_id')['employees'].agg(['mean', 'median', 'count'])
            
            for section_id, stats in section_stats.iterrows():
                # Use median for robustness, but ensure at least 1 employee
                self.section_patterns[section_id] = {
                    'median': max(1, int(stats['median'])),
                    'mean': max(1, int(stats['mean'])),
                    'count': stats['count']
                }
        
        # Weekday patterns
        if 'weekday' in valid_data.columns:
            weekday_stats = valid_data.groupby('weekday')['employees'].agg(['mean', 'median'])
            
            for weekday, stats in weekday_stats.iterrows():
                self.weekday_patterns[weekday] = {
                    'median': max(1, int(stats['median'])),
                    'mean': max(1, int(stats['mean']))
                }
        
        logger.info(f"Learned patterns for {len(self.section_patterns)} sections, {len(self.weekday_patterns)} weekdays")
        self.is_fitted = True
        return self
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Make predictions using staffing patterns."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = []
        
        for _, row in features_df.iterrows():
            prediction = self.global_average
            
            # Try section-specific pattern first
            if 'section_id' in row.index and not pd.isna(row['section_id']):
                section_id = row['section_id']
                if section_id in self.section_patterns:
                    section_pattern = self.section_patterns[section_id]
                    # Use median if we have enough data points, otherwise use mean
                    if section_pattern['count'] >= 3:
                        prediction = section_pattern['median']
                    else:
                        prediction = section_pattern['mean']
                    
                    # Adjust for weekday if available
                    if 'weekday' in row.index and not pd.isna(row['weekday']):
                        weekday = int(row['weekday'])
                        if weekday in self.weekday_patterns:
                            weekday_pattern = self.weekday_patterns[weekday]
                            # Blend section and weekday patterns
                            prediction = int(0.7 * prediction + 0.3 * weekday_pattern['median'])
            
            # Fallback to weekday pattern if no section info
            elif 'weekday' in row.index and not pd.isna(row['weekday']):
                weekday = int(row['weekday'])
                if weekday in self.weekday_patterns:
                    prediction = self.weekday_patterns[weekday]['median']
            
            # Ensure at least 1 employee
            prediction = max(1, prediction)
            predictions.append(prediction)
        
        return np.array(predictions)