"""
LSTM-based regression model for Traffic-Based Route Guidance System (TBRGS).

This module provides functions to:
  - build_lstm: construct and compile an LSTM neural network for flow prediction
  - train_lstm: train the LSTM model on training data and save the best checkpoint
"""

import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import ModelCheckpoint


def build_lstm(input_shape):
    """
    Build and compile an LSTM model for sequence-to-one regression.

    Args:
        input_shape (tuple): Shape of input sequences (window_size, features).

    Returns:
        tensorflow.keras.Model: Compiled LSTM regression model.
    """
    model = Sequential([
        # Single LSTM layer with 64 units
        LSTM(64, input_shape=input_shape),
        # Output layer for single value prediction
        Dense(1)
    ])
    # Compile with MSE loss and MAE metric for regression
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


def train_lstm(
    X_train,
    y_train,
    X_val,
    y_val,
    epochs=20,
    batch_size=32,
    models_dir="models"
):
    """
    Train the LSTM model and save the best model checkpoint based on validation loss.

    Args:
        X_train (numpy.ndarray): Training input sequences.
        y_train (numpy.ndarray): Training labels.
        X_val (numpy.ndarray): Validation input sequences.
        y_val (numpy.ndarray): Validation labels.
        epochs (int, optional): Number of training epochs. Defaults to 20.
        batch_size (int, optional): Batch size for training. Defaults to 32.
        models_dir (str, optional): Directory to save model checkpoints. Defaults to "models".

    Returns:
        tuple:
            str: File path to the saved best LSTM model (.h5).
            tensorflow.keras.callbacks.History: Training history object.
    """
    # Ensure the models directory exists
    os.makedirs(models_dir, exist_ok=True)

    # Build the LSTM model
    model = build_lstm(input_shape=X_train.shape[1:])

    # Define checkpoint to save best weights by validation loss
    ckpt_path = os.path.join(models_dir, "lstm_model.h5")
    checkpoint = ModelCheckpoint(
        filepath=ckpt_path,
        save_best_only=True,
        monitor="val_loss",
        verbose=1
    )

    # Train the model
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint],
        verbose=2
    )

    return ckpt_path, history