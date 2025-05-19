"""
Model evaluation and comparison script

This script loads pre-trained LSTM, GRU, and XGBoost models,
runs prediction on their respective test sets, evaluates performance
(MSE, MAE, R²), and visualizes prediction results for comparison.
"""

import numpy as np
import joblib
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.losses import MeanSquaredError
import matplotlib.pyplot as plt

# ====== STEP 1: Load test data ======
X_test_lstm = np.load("../models/lstm/X_test.npy")
y_test_lstm = np.load("../models/lstm/y_test.npy")

X_test_gru = np.load("../models/gru/X_test.npy")
y_test_gru = np.load("../models/gru/y_test.npy")

X_test_xgb = np.load("../models/xgb/X_test.npy")
y_test_xgb = np.load("../models/xgb/y_test.npy")

# Reshape LSTM/GRU input
X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
X_test_gru = X_test_gru.reshape((X_test_gru.shape[0], X_test_gru.shape[1], 1))

# ====== STEP 2: Load trained models ======
lstm_model = load_model("../models/lstm/lstm.h5", compile=False)
gru_model = load_model("../models/gru/gru.h5", compile=False)
xgb_model = joblib.load("../models/xgb/xgb.joblib")

# ====== STEP 3: Predict ======
y_pred_lstm = lstm_model.predict(X_test_lstm).flatten()
y_pred_gru = gru_model.predict(X_test_gru).flatten()
y_pred_xgb = xgb_model.predict(X_test_xgb)

# ====== STEP 4: Evaluation Function ======
def evaluate_model(name, y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{name} Evaluation:")
    print(f"  MSE : {mse:.2f}")
    print(f"  MAE : {mae:.2f}")
    print(f"  R2  : {r2:.4f}")
    print("-" * 40)
    return {"Model": name, "MSE": mse, "MAE": mae, "R2": r2}

# ====== STEP 5: Evaluate All Models ======
results = []
results.append(evaluate_model("LSTM", y_test_lstm, y_pred_lstm))
results.append(evaluate_model("GRU", y_test_gru, y_pred_gru))
results.append(evaluate_model("XGBoost", y_test_xgb, y_pred_xgb))

# Save to CSV
pd.DataFrame(results).to_csv("../models/model_comparison_results.csv", index=False)

# ====== STEP 6: Plot for Visual Comparison ======
plt.figure(figsize=(10, 5))
plt.plot(y_test_lstm[:100], label="Actual (LSTM)", linewidth=2)
plt.plot(y_pred_lstm[:100], label="LSTM")

plt.plot(y_test_gru[:100], label="Actual (GRU)", linewidth=2, linestyle='dotted')
plt.plot(y_pred_gru[:100], label="GRU")

plt.plot(y_test_xgb[:100], label="Actual (XGB)", linewidth=2, linestyle='dashdot')
plt.plot(y_pred_xgb[:100], label="XGBoost")

plt.title("Model Comparison (First 100 Samples)")
plt.xlabel("Sample Index")
plt.ylabel("Traffic Volume")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
