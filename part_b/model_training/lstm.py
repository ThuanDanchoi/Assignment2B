"""
LSTM model training script

This script trains an LSTM model to predict traffic volume based on historical data
using a sliding window approach. The trained model is saved to disk along with test data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load data
df = pd.read_csv("../data/processed/cleaned_scats_data.csv", parse_dates=["Datetime"])
df = df.sort_values(by=["SiteID", "Datetime"]).reset_index(drop=True)

# Sliding window
LOOKBACK = 4
X_all, y_all = [], []
for _, group in df.groupby("SiteID"):
    volume = group["Volume"].values
    for i in range(len(volume) - LOOKBACK):
        X_all.append(volume[i:i+LOOKBACK])
        y_all.append(volume[i+LOOKBACK])

X_all = np.array(X_all)
y_all = np.array(y_all)

# Normalize
scaler = MinMaxScaler()
X_all_scaled = scaler.fit_transform(X_all)

# Reshape for LSTM: (samples, timesteps, features)
X_all_scaled = X_all_scaled.reshape((X_all_scaled.shape[0], X_all_scaled.shape[1], 1))

# Split
X_train, X_test, y_train, y_test = train_test_split(X_all_scaled, y_all, test_size=0.2, shuffle=False)

np.save("../models/lstm/X_test.npy", X_test)
np.save("../models/lstm/y_test.npy", y_test)

# Model
model = Sequential()
model.add(LSTM(64, input_shape=(LOOKBACK, 1)))
model.add(Dropout(0.2))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# Train
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test))

# Predict
y_pred = model.predict(X_test).flatten()

# Evaluate
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R2 Score: {r2:.4f}")

model.save("../models/lstm/lstm.h5")
