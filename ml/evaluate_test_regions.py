import json
import numpy as np
import pandas as pd
import joblib

from tensorflow.keras.models import load_model
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


# ==========================================
# SETTINGS
# ==========================================

WINDOW_SIZE = 30


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
# LOAD TEST SCENARIOS
# ==========================================

with open("scenario_split.json", "r") as f:
    split = json.load(f)

test_scenarios = split["test"]


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


# ==========================================
# SCALE
# ==========================================

num_features = X_test.shape[2]

X_test_scaled = feature_scaler.transform(
    X_test.reshape(-1, num_features)
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)


# ==========================================
# PREDICT
# ==========================================

predictions_scaled = model.predict(
    X_test_scaled,
    verbose=0
)

predictions = (
    target_scaler
    .inverse_transform(
        predictions_scaled
    )
    .flatten()
)


# ==========================================
# DEFINE REGIONS
# ==========================================

regions = {

    "Early (>8 h)": y_test > 8,

    "Middle (4–8 h)": (
        (y_test > 4) &
        (y_test <= 8)
    ),

    "Late (2–4 h)": (
        (y_test > 2) &
        (y_test <= 4)
    ),

    "Near failure (≤2 h)": (
        y_test <= 2
    )
}


# ==========================================
# EVALUATE EACH REGION
# ==========================================

print("\n")
print("=" * 50)
print("TEST SET ERROR BY TTF REGION")
print("=" * 50)


for region_name, mask in regions.items():

    actual = y_test[mask]
    predicted = predictions[mask]

    if len(actual) == 0:
        continue

    mae = mean_absolute_error(
        actual,
        predicted
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mean_error = np.mean(
        predicted - actual
    )

    print("\n" + region_name)
    print("-" * 30)

    print(
        f"Samples: {len(actual)}"
    )

    print(
        f"MAE: {mae:.4f} hours"
    )

    print(
        f"RMSE: {rmse:.4f} hours"
    )

    print(
        f"Mean error: {mean_error:+.4f} hours"
    )


print("\n")
print("=" * 50)
print("DONE")
print("=" * 50)