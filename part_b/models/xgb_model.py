"""
XGBoost regression model for Traffic-Based Route Guidance System (TBRGS).

This module provides functions to:
  - train_xgb: train an XGBoost regressor on flattened sequence data
  - predict_xgb: make predictions with a trained XGBoost model
"""

import os
import numpy as np
import xgboost as xgb


def train_xgb(
    X_train: np.ndarray,
    y_train: np.ndarray,
    params: dict = None,
    models_dir: str = "models"
) -> tuple:
    """
    Train an XGBoost regressor on time-series sequence data.

    This function:
      1. Flattens X_train from (n_samples, timesteps, 1) to (n_samples, timesteps)
      2. Trains XGBRegressor with given parameters
      3. Saves model to JSON

    Args:
        X_train (np.ndarray): Training data of shape (n_samples, timesteps, 1).
        y_train (np.ndarray): Training labels of shape (n_samples,).
        params (dict, optional): XGBRegressor parameters. Defaults to None.
        models_dir (str, optional): Directory to save model. Defaults to "models".

    Returns:
        tuple:
            str: Path to the saved model JSON file.
            xgb.XGBRegressor: Trained XGBoost model.
    """
    # Ensure the model directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Flatten sequence dimension for XGBoost
    X_flat = X_train.reshape(len(X_train), -1)

    # Default parameters if none provided
    if params is None:
        params = {"n_estimators": 100, "learning_rate": 0.1}

    # Initialize and train the model
    model = xgb.XGBRegressor(**params)
    model.fit(X_flat, y_train)

    # Save trained model to JSON
    model_path = os.path.join(models_dir, "xgb_model.json")
    model.save_model(model_path)
    return model_path, model


def predict_xgb(model: xgb.XGBRegressor, X: np.ndarray) -> np.ndarray:
    """
    Generate predictions using a trained XGBoost model.

    Args:
        model (xgb.XGBRegressor): Trained XGBoost regressor.
        X (np.ndarray): Input data of shape (n_samples, timesteps, 1).

    Returns:
        np.ndarray: Predicted values of shape (n_samples,).
    """
    # Flatten input for prediction
    X_flat = X.reshape(len(X), -1)
    return model.predict(X_flat)
