import json
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model


# ==========================================
# SETTINGS
# ==========================================

WINDOW_SIZE = 30
SCENARIO_ID = 52


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

scenario = data[
    data["scenario_id"] == SCENARIO_ID
].copy()

scenario = scenario.reset_index(drop=True)


# ==========================================
# CREATE SEQUENCES
# ==========================================

X = []
actual_ttf = []
time_values = []

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

    X.append(features)

    target_index = (
        i + WINDOW_SIZE - 1
    )

    actual_ttf.append(
        scenario["ttf"].iloc[target_index]
    )

    time_values.append(
        scenario["time_hours"].iloc[target_index]
    )


X = np.array(X)

actual_ttf = np.array(
    actual_ttf
)

time_values = np.array(
    time_values
)


# ==========================================
# SCALE FEATURES
# ==========================================

num_features = X.shape[2]

X_scaled = feature_scaler.transform(
    X.reshape(-1, num_features)
)

X_scaled = X_scaled.reshape(
    X.shape
)


# ==========================================
# PREDICT
# ==========================================

predictions_scaled = model.predict(
    X_scaled,
    verbose=0
)

predicted_ttf = (
    target_scaler
    .inverse_transform(
        predictions_scaled
    )
    .flatten()
)

# ==========================================
# PRINT SELECTED PREDICTIONS
# ==========================================

print("\n")
print("=" * 60)
print("SELECTED ACTUAL vs PREDICTED TTF")
print("=" * 60)

indices = np.linspace(
    0,
    len(actual_ttf) - 1,
    12,
    dtype=int
)

for index in indices:

    actual = actual_ttf[index]
    predicted = predicted_ttf[index]

    error = predicted - actual

    print(
        f"Time: {time_values[index]:6.3f} h | "
        f"Actual: {actual:7.3f} h | "
        f"Predicted: {predicted:7.3f} h | "
        f"Error: {error:+7.3f} h"
    )
# ==========================================
# PRINT BASIC INFO
# ==========================================

print("\n")
print("=" * 50)
print("ACTUAL vs PREDICTED TTF")
print("=" * 50)

print(
    f"Scenario: {SCENARIO_ID}"
)

print(
    f"Number of predictions: {len(predicted_ttf)}"
)


# ==========================================
# PLOT
# ==========================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    time_values,
    actual_ttf,
    label="Actual TTF"
)

plt.plot(
    time_values,
    predicted_ttf,
    label="Predicted TTF"
)

plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Time to Failure (hours)"
)

plt.title(
    f"Actual vs Predicted TTF - Scenario {SCENARIO_ID}"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.show()