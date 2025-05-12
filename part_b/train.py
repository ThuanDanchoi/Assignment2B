"""
Training and evaluation script for Traffic-Based Route Guidance System (TBRGS).

This module:
  1. Loads processed data and splits off a validation set
  2. Trains LSTM, GRU, and XGBoost models
  3. Evaluates each model on the test set (MAE & RMSE)
  4. Saves a CSV summary of metrics
  5. Generates bar charts comparing MAE and RMSE across models
"""

import os
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

from models.lstm_model import train_lstm
from models.gru_model import train_gru
from models.xgb_model import train_xgb

# Define base directories
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")


def load_data():
    """
    Load processed training and test datasets from disk.

    Returns:
        tuple:
            X_train (numpy.ndarray): Training input sequences
            X_test (numpy.ndarray): Test input sequences
            y_train (numpy.ndarray): Training labels
            y_test (numpy.ndarray): Test labels
    """
    X_train = np.load(os.path.join(DATA_DIR, "X_train.npy"))
    y_train = np.load(os.path.join(DATA_DIR, "y_train.npy"))
    X_test = np.load(os.path.join(DATA_DIR, "X_test.npy"))
    y_test = np.load(os.path.join(DATA_DIR, "y_test.npy"))
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    # 1) Load data and split off a validation set
    X_train, X_test, y_train, y_test = load_data()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train,
        test_size=0.1,
        random_state=42
    )

    # 2) Train LSTM model
    print("Training LSTM…")
    lstm_path, _ = train_lstm(
        X_tr, y_tr,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        models_dir=MODELS_DIR
    )

    # 3) Train GRU model
    print("Training GRU…")
    gru_path, _ = train_gru(
        X_tr, y_tr,
        X_val, y_val,
        epochs=20,
        batch_size=32,
        models_dir=MODELS_DIR
    )

    # 4) Train XGBoost model
    print("Training XGBoost…")
    xgb_path, xgb_model = train_xgb(
        X_tr, y_tr,
        params={"n_estimators": 100, "learning_rate": 0.1},
        models_dir=MODELS_DIR
    )

    # 5) Evaluate all models on the test set
    print("Evaluating on test set…")
    results = {}

    # Evaluate LSTM
    lstm = load_model(lstm_path, compile=False)
    y_pred = lstm.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results["LSTM"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mse))
    }

    # Evaluate GRU
    gru = load_model(gru_path, compile=False)
    y_pred = gru.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    results["GRU"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mse))
    }

    # Evaluate XGBoost (flatten input sequence dimension)
    X_flat = X_test.reshape(len(X_test), -1)
    y_pred = xgb_model.predict(X_flat)
    mse = mean_squared_error(y_test, y_pred)
    results["XGB"] = {
        "MAE": float(mean_absolute_error(y_test, y_pred)),
        "RMSE": float(np.sqrt(mse))
    }

    # 6) Save metrics summary to CSV
    metrics_path = os.path.join(BASE_DIR, "metrics_summary.csv")
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "MAE", "RMSE"]);
        for model_name, metrics in results.items():
            writer.writerow([model_name, metrics["MAE"], metrics["RMSE"]])

    # 7) Plot and save comparison charts
    models = list(results.keys())
    maes = [results[m]["MAE"] for m in models]
    rmses = [results[m]["RMSE"] for m in models]
    indices = range(len(models))

    # MAE comparison chart
    plt.figure()
    plt.bar(indices, maes)
    plt.xticks(indices, models)
    plt.ylabel("MAE")
    plt.title("MAE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "mae_comparison.png"))

    # RMSE comparison chart
    plt.figure()
    plt.bar(indices, rmses)
    plt.xticks(indices, models)
    plt.ylabel("RMSE")
    plt.title("RMSE Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "rmse_comparison.png"))

    print("Done! Metrics summary at:", metrics_path)
    print("Charts generated:", "mae_comparison.png", ", rmse_comparison.png")
