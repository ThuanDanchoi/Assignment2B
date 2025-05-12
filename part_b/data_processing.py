"""
Data processing module for Traffic-Based Route Guidance System (TBRGS).

This module provides functions to:
  - Load raw SCATS traffic count data in wide format
  - Transform data into long format with proper timestamps
  - Generate sequence features and labels for modeling
  - Split data into training and testing sets
  - Apply Min-Max scaling to sequence data
"""

import os
import re
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Base directories and file paths
BASE_DIR = os.path.dirname(__file__)
RAW_CSV = os.path.join(BASE_DIR, "data", "raw", "Scats Data October 2006.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")


def load_raw():
    """
    Load raw SCATS CSV and convert to long-format DataFrame.

    Steps:
        1. Read wide-format CSV, skipping the metadata row.
        2. Rename essential columns for consistency.
        3. Identify and select columns matching pattern V00…V95.
        4. Melt into a long-format DataFrame.
        5. Parse 'date' and 'time_code' into a single timestamp.

    Returns:
        pandas.DataFrame: Columns ['site_id', 'timestamp', 'lat', 'lon', 'flow']
    """
    # Read the CSV, skipping the first metadata row
    df = pd.read_csv(RAW_CSV, skiprows=1)
    df = df.rename(columns={
        "SCATS Number": "site_id",
        "NB_LATITUDE":  "lat",
        "NB_LONGITUDE": "lon",
        "Date":         "date"
    })

    # Filter columns V00…V95 for melting
    time_cols = [c for c in df.columns if re.match(r'^V\d+$', c)]
    if not time_cols:
        raise ValueError("No columns matching pattern V00…V95 found in CSV!")

    # Melt data into long format
    df_long = (
        df.melt(
            id_vars=["site_id", "lat", "lon", "date"],
            value_vars=time_cols,
            var_name="time_code",
            value_name="flow"
        )
        .dropna(subset=["flow"])
    )

    # Convert 'date' to datetime (day first) and extract numeric time index
    df_long["date"] = pd.to_datetime(df_long["date"], dayfirst=True)
    df_long["time_index"] = (
        df_long["time_code"]
        .str.extract(r'(\d+)', expand=False)
        .astype(int)
    )

    # Create a timestamp by adding 15-minute intervals
    df_long["timestamp"] = (
        df_long["date"]
        + pd.to_timedelta(df_long["time_index"] * 15, unit="m")
    )

    # Select and return relevant columns
    return df_long[["site_id", "timestamp", "lat", "lon", "flow"]]


def feature_engineer(df_long, window_size=4):
    """
    Generate sequences and labels from the long-format DataFrame.

    Args:
        df_long (pandas.DataFrame): DataFrame containing 'site_id', 'timestamp', 'flow'.
        window_size (int): Number of past flow values to use for each sample.

    Returns:
        tuple:
            X (numpy.ndarray): Array of shape (n_samples, window_size, 1) for input sequences.
            y (numpy.ndarray): Array of shape (n_samples,) for labels (next flow value).
    """
    sequences, labels = [], []

    # Iterate over each site and sort by timestamp
    for site, group in df_long.groupby("site_id"):
        flows = group.sort_values("timestamp")["flow"].values
        # Build sliding windows of size 'window_size'
        for i in range(len(flows) - window_size):
            sequences.append(flows[i : i + window_size])
            labels.append(flows[i + window_size])

    # Convert lists to numpy arrays and add feature dimension
    X = np.array(sequences)[..., np.newaxis]
    y = np.array(labels)
    return X, y


def split_and_scale(X, y, train_frac=0.7):
    """
    Split sequences into training and testing sets, then apply Min-Max scaling.

    Args:
        X (numpy.ndarray): Input sequences of shape (n_samples, window_size, 1).
        y (numpy.ndarray): Labels of shape (n_samples,).
        train_frac (float): Fraction of data to allocate to training.

    Returns:
        tuple:
            X_train_s (numpy.ndarray): Scaled training sequences.
            X_test_s (numpy.ndarray): Scaled testing sequences.
            y_train (numpy.ndarray): Training labels.
            y_test (numpy.ndarray): Testing labels.
    """
    # Determine split index
    n_train = int(len(X) * train_frac)
    X_train, X_test = X[:n_train], X[n_train:]
    y_train, y_test = y[:n_train], y[n_train:]

    # Initialize Min-Max scaler for feature dimension
    scaler = MinMaxScaler()
    ns, window, features = X_train.shape

    # Flatten training data for fitting scaler
    flat_train = X_train.reshape(-1, features)
    X_train_s = scaler.fit_transform(flat_train).reshape(ns, window, features)

    # Scale testing data
    flat_test = X_test.reshape(-1, features)
    X_test_s = scaler.transform(flat_test).reshape(len(X_test), window, features)

    return X_train_s, X_test_s, y_train, y_test


def ensure_dirs():
    """
    Create the processed data directory if it does not exist.
    """
    os.makedirs(PROCESSED_DIR, exist_ok=True)


if __name__ == "__main__":
    # Ensure output directory exists
    ensure_dirs()
    print(">> Loading raw data …")
    df_long = load_raw()
    print(f"   Rows after melt: {len(df_long)}")

    print(">> Feature engineering …")
    X, y = feature_engineer(df_long, window_size=4)
    print(f"   Samples: {len(X)}, window_size: {X.shape[1]}")

    print(">> Splitting & scaling …")
    X_train, X_test, y_train, y_test = split_and_scale(X, y, train_frac=0.7)

    # Save processed arrays to disk
    np.save(os.path.join(PROCESSED_DIR, "X_train.npy"), X_train)
    np.save(os.path.join(PROCESSED_DIR, "y_train.npy"), y_train)
    np.save(os.path.join(PROCESSED_DIR, "X_test.npy"),  X_test)
    np.save(os.path.join(PROCESSED_DIR, "y_test.npy"),  y_test)
    print(">> Saved to:", PROCESSED_DIR)
