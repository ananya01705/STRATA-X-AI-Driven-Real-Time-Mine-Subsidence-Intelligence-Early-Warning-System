import json
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ==========================================
# SETTINGS
# ==========================================

WINDOW_SIZE = 30
TTF_BIAS = 0.3879


# ==========================================
# LOAD MODEL + SCALERS
# ==========================================

model = load_model("ttf_lstm_model.keras")

feature_scaler = joblib.load(
    "lstm_feature_scaler.pkl"
)

target_scaler = joblib.load(
    "lstm_target_scaler.pkl"
)

print("Model and scalers loaded successfully!")


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv(
    "all_creep_scenarios.csv"
)


# ==========================================
# LOAD SCENARIO SPLIT
# ==========================================

with open("scenario_split.json", "r") as f:
    split = json.load(f)


test_scenarios = split["test"]

print("\nTest scenarios:")
print(test_scenarios)


# ==========================================
# CREATE TEST SEQUENCES
# ==========================================

X_test = []
y_test = []

for scenario_id in test_scenarios:

    scenario = data[
        data["scenario_id"] == scenario_id
    ].copy()

    scenario = scenario.reset_index(drop=True)

    for i in range(
        len(scenario) - WINDOW_SIZE + 1
    ):

        window = scenario.iloc[
            i:i + WINDOW_SIZE
        ].copy()

        features = window[
            [
                "velocity",
                "deformation",
                "inverse_velocity"
            ]
        ].values

        target = scenario[
            "ttf"
        ].iloc[
            i + WINDOW_SIZE - 1
        ]

        X_test.append(features)
        y_test.append(target)


X_test = np.array(X_test)
y_test = np.array(y_test)


print("\nTest data shape:")
print("X:", X_test.shape)
print("y:", y_test.shape)


# ==========================================
# SCALE FEATURES
# ==========================================

num_features = X_test.shape[2]

X_test_scaled = feature_scaler.transform(
    X_test.reshape(-1, num_features)
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)


# ==========================================
# RAW LSTM PREDICTIONS
# ==========================================

predictions_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

raw_predictions = (
    target_scaler
    .inverse_transform(predictions_scaled)
    .flatten()
)


# ==========================================
# BIAS CORRECTION
# ==========================================

corrected_predictions = (
    raw_predictions - TTF_BIAS
)

corrected_predictions = np.maximum(
    corrected_predictions,
    0
)


# ==========================================
# RAW MODEL METRICS
# ==========================================

raw_mae = mean_absolute_error(
    y_test,
    raw_predictions
)

raw_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        raw_predictions
    )
)

raw_r2 = r2_score(
    y_test,
    raw_predictions
)


# ==========================================
# CORRECTED MODEL METRICS
# ==========================================

corrected_mae = mean_absolute_error(
    y_test,
    corrected_predictions
)

corrected_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        corrected_predictions
    )
)

corrected_r2 = r2_score(
    y_test,
    corrected_predictions
)


# ==========================================
# RESULTS
# ==========================================

print("\n")
print("=" * 45)
print("RAW vs CORRECTED LSTM")
print("=" * 45)

print("\nRAW LSTM")
print("-" * 20)

print(
    f"MAE:  {raw_mae:.4f} hours"
)

print(
    f"RMSE: {raw_rmse:.4f} hours"
)

print(
    f"R²:   {raw_r2:.4f}"
)


print("\nBIAS-CORRECTED LSTM")
print("-" * 20)

print(
    f"MAE:  {corrected_mae:.4f} hours"
)

print(
    f"RMSE: {corrected_rmse:.4f} hours"
)

print(
    f"R²:   {corrected_r2:.4f}"
)


# ==========================================
# IMPROVEMENT
# ==========================================

mae_change = (
    raw_mae - corrected_mae
)

rmse_change = (
    raw_rmse - corrected_rmse
)

r2_change = (
    corrected_r2 - raw_r2
)


print("\n")
print("=" * 45)
print("IMPROVEMENT")
print("=" * 45)

print(
    f"MAE improvement:  {mae_change:+.4f} hours"
)

print(
    f"RMSE improvement: {rmse_change:+.4f} hours"
)

print(
    f"R² change:        {r2_change:+.4f}"
)


# ==========================================
# FINAL DECISION
# ==========================================

print("\n")
print("=" * 45)

if corrected_mae < raw_mae:

    print(
        "✅ Bias correction improved MAE."
    )

else:

    print(
        "❌ Bias correction did NOT improve MAE."
    )


if corrected_rmse < raw_rmse:

    print(
        "✅ Bias correction improved RMSE."
    )

else:

    print(
        "❌ Bias correction did NOT improve RMSE."
    )

print("=" * 45)