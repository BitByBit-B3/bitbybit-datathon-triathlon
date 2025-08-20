import logging
import os
from typing import Any

import numpy as np
import pandas as pd

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.preprocessing import StandardScaler

    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("scikit-learn not available, using baseline models only")

from baselines import create_task1_baseline

logger = logging.getLogger(__name__)


class Task1ModelEnsemble:
    """Ensemble model for Task 1 combining baseline and optional ML models."""

    def __init__(self, use_advanced_models: bool | None = None) -> None:
        """
        Initialize Task1ModelEnsemble.

        Args:
            use_advanced_models: Whether to use ML models. If None, check environment.
        """
        if use_advanced_models is None:
            use_advanced_models = os.getenv("USE_ADVANCED_MODELS", "0") == "1"

        self.use_advanced_models = use_advanced_models and SKLEARN_AVAILABLE
        self.baseline_model = create_task1_baseline()
        self.ml_model = None
        self.scaler = None
        self.feature_columns = None
        self.is_fitted = False

        if self.use_advanced_models:
            logger.info("Task1ModelEnsemble: Using advanced ML models")
        else:
            logger.info("Task1ModelEnsemble: Using baseline model only")

    def fit(self, features_df: pd.DataFrame, target_series: pd.Series) -> "Task1ModelEnsemble":
        """
        Fit the ensemble model.

        Args:
            features_df: Training features
            target_series: Training targets

        Returns:
            Self for method chaining
        """
        logger.info("Fitting Task1ModelEnsemble")

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
        logger.info("Task1ModelEnsemble fitting complete")

        return self

    def _fit_ml_model(self, features_df: pd.DataFrame, target_series: pd.Series) -> None:
        """Fit the ML component of the ensemble."""
        logger.info("Fitting ML model for Task 1")

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
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        logger.info(f"Making Task 1 predictions for {len(features_df)} samples")

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
            msg = "ML model not fitted"
            raise ValueError(msg)

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
        # Ensure positive values (processing time can't be negative)
        predictions = np.maximum(predictions, 0)

        # Cap at reasonable maximum (8 hours = 480 minutes)
        predictions = np.minimum(predictions, 480)

        # Round to integers
        predictions = np.round(predictions).astype(int)

        return predictions


def create_simple_task1_model(model_type: str = "baseline") -> Any:
    """
    Create a simple model for Task 1.

    Args:
        model_type: Type of model ('baseline', 'linear', 'ensemble')

    Returns:
        Configured model
    """
    if model_type == "baseline":
        return create_task1_baseline()
    if model_type == "ensemble":
        return Task1ModelEnsemble(use_advanced_models=True)
    if model_type == "linear" and SKLEARN_AVAILABLE:
        # Simple wrapper around sklearn LinearRegression
        return SklearnWrapper(LinearRegression())
    logger.warning(f"Model type '{model_type}' not available, using baseline")
    return create_task1_baseline()


class SklearnWrapper:
    """Simple wrapper to make sklearn models compatible with our interface."""

    def __init__(self, model) -> None:
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
            msg = "No valid training data"
            raise ValueError(msg)

        # Store feature columns
        self.feature_columns = X.columns.tolist()

        # Scale and fit
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        if not self.is_fitted:
            msg = "Model must be fitted before prediction"
            raise ValueError(msg)

        X = self._prepare_features(features_df)

        # Ensure same columns as training
        for col in self.feature_columns:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_columns]

        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)

        # Post-process
        predictions = np.maximum(predictions, 0)
        predictions = np.minimum(predictions, 480)
        predictions = np.round(predictions).astype(int)

        return predictions

    def _prepare_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        # Select numeric features
        numeric_features = features_df.select_dtypes(include=[np.number]).copy()
        numeric_features = numeric_features.fillna(numeric_features.median())
        numeric_features = numeric_features.replace([np.inf, -np.inf], 0)
        return numeric_features
