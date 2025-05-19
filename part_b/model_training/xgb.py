"""
XGBoost model training script

This script trains an XGBoost regressor to predict traffic volume using
a sliding window approach. The trained model and test data are saved for future evaluation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

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

# Split
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, shuffle=False)

np.save("../models/xgb/X_test.npy", X_test)
np.save("../models/xgb/y_test.npy", y_test)

# Model
model = XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test MSE: {mse:.2f}")
print(f"Test MAE: {mae:.2f}")
print(f"Test R2 Score: {r2:.4f}")

joblib.dump(model, "../models/xgb/xgb.joblib")