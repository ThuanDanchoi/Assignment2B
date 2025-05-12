"""
Model evaluation script for Traffic-Based Route Guidance System (TBRGS).

This module:
  - Loads processed test data
  - Loads trained models (LSTM, GRU, XGBoost)
  - Computes Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE) for each model
  - Prints a summary table of results
"""

import os
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model
import xgboost as xgb

# Define base directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_test_data():
    """
    Load test data arrays from disk.

    Returns:
        tuple:
            X_test (numpy.ndarray): Test input sequences
            y_test (numpy.ndarray): True labels for test data
    """
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_test, y_test


if __name__ == "__main__":
    # Load test dataset
    X_test, y_test = load_test_data()
    results = {}

    # Evaluate LSTM model
    lstm_path = os.path.join(MODELS_DIR, "lstm_model.h5")
    lstm = load_model(lstm_path, compile=False)
    y_pred = lstm.predict(X_test)
    # Compute metrics
    mse = mean_squared_error(y_test, y_pred)
    results["LSTM"] = (
        mean_absolute_error(y_test, y_pred),  # MAE
        np.sqrt(mse)                           # RMSE
    )

    # Evaluate GRU model
    gru_path = os.path.join(MODELS_DIR, "gru_model.h5")
    gru = load_model(gru_path, compile=False)
    y_pred = gru.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results["GRU"] = (
        mean_absolute_error(y_test, y_pred),
        np.sqrt(mse)
    )

    # Evaluate XGBoost model
    xgb_path = os.path.join(MODELS_DIR, "xgb_model.json")
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(xgb_path)
    # Reshape X_test for XGBoost (flatten time-window dimension)
    X_flat = X_test.reshape(len(X_test), -1)
    y_pred = xgb_model.predict(X_flat)
    mse = mean_squared_error(y_test, y_pred)
    results["XGB"] = (
        mean_absolute_error(y_test, y_pred),
        np.sqrt(mse)
    )

    # Print results table
    print("Model |   MAE   |  RMSE")
    print("-------------------------")
    for model_name, (mae, rmse) in results.items():
        print(f"{model_name:5s} | {mae:7.3f} | {rmse:7.3f}")
