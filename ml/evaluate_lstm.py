import pandas as pd
import numpy as np
import json
import joblib
import matplotlib.pyplot as plt

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


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv(
    "all_creep_scenarios.csv"
)

print("Dataset loaded")


# ==========================================
# LOAD SCENARIO SPLIT
# ==========================================

with open(
    "scenario_split.json",
    "r"
) as f:

    scenario_split = json.load(f)


test_scenarios = scenario_split["test"]


print("\nTest scenarios")
print("======================")
print(test_scenarios)


# ==========================================
# LOAD MODEL
# ==========================================

model = load_model(
    "ttf_lstm_model.keras"
)

print("\nLSTM model loaded")


# ==========================================
# LOAD SCALERS
# ==========================================

feature_scaler = joblib.load(
    "lstm_feature_scaler.pkl"
)

target_scaler = joblib.load(
    "lstm_target_scaler.pkl"
)

print("Scalers loaded")


# ==========================================
# CREATE TEST SEQUENCES
# ==========================================

X_test = []
y_test = []

scenario_labels = []


for scenario_id in test_scenarios:

    scenario = data[
        data["scenario_id"] == scenario_id
    ].copy()

    scenario = scenario.reset_index(
        drop=True
    )


    for i in range(
        len(scenario) - WINDOW_SIZE
    ):

        window = scenario.iloc[
            i:i + WINDOW_SIZE
        ]


        sequence = window[
            [
                "velocity",
                "deformation"
            ]
        ].values


        target = scenario[
            "ttf"
        ].iloc[
            i + WINDOW_SIZE - 1
        ]


        X_test.append(sequence)

        y_test.append(target)

        scenario_labels.append(
            scenario_id
        )


X_test = np.array(
    X_test,
    dtype=np.float32
)

y_test = np.array(
    y_test,
    dtype=np.float32
)


# ==========================================
# SCALE FEATURES
# ==========================================

num_features = X_test.shape[2]

X_test_2d = X_test.reshape(
    -1,
    num_features
)

X_test_scaled = feature_scaler.transform(
    X_test_2d
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)


# ==========================================
# PREDICT
# ==========================================

print("\nGenerating predictions...")

predictions_scaled = model.predict(
    X_test_scaled,
    verbose=0
)


predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()


# ==========================================
# OVERALL METRICS
# ==========================================

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print("\n==============================")
print("LSTM TEST EVALUATION")
print("==============================")

print(
    f"MAE  : {mae:.3f} hours"
)

print(
    f"RMSE : {rmse:.3f} hours"
)

print(
    f"R²   : {r2:.3f}"
)


# ==========================================
# SAMPLE PREDICTIONS
# ==========================================

print("\nSample predictions")
print("------------------------------")

for i in range(
    min(10, len(y_test))
):

    print(
        f"Scenario {scenario_labels[i]} | "
        f"Actual: {y_test[i]:.2f} h | "
        f"Predicted: {predictions[i]:.2f} h"
    )
    # ==========================================
# VISUALIZATION
# ==========================================

# ------------------------------------------
# GRAPH 1: ACTUAL VS PREDICTED
# ------------------------------------------

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)

# Perfect prediction line

min_value = min(
    y_test.min(),
    predictions.min()
)

max_value = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [min_value, max_value],
    [min_value, max_value],
    linestyle="--"
)

plt.xlabel("Actual TTF (hours)")
plt.ylabel("Predicted TTF (hours)")

plt.title(
    "Actual vs Predicted Time-to-Failure"
)

plt.grid(True)

plt.show()


# ------------------------------------------
# GRAPH 2: ACTUAL VS PREDICTED SEQUENCE
# ------------------------------------------

plt.figure(figsize=(10, 6))

sample_size = min(
    500,
    len(y_test)
)

plt.plot(
    y_test[:sample_size],
    label="Actual TTF"
)

plt.plot(
    predictions[:sample_size],
    label="Predicted TTF"
)

plt.xlabel("Test sample")
plt.ylabel("TTF (hours)")

plt.title(
    "Actual vs Predicted TTF Over Test Samples"
)

plt.legend()

plt.grid(True)

plt.show()


# ------------------------------------------
# GRAPH 3: PREDICTION ERROR
# ------------------------------------------

errors = predictions - y_test

plt.figure(figsize=(8, 6))

plt.hist(
    errors,
    bins=40
)

plt.axvline(
    0,
    linestyle="--"
)

plt.xlabel(
    "Prediction Error (hours)"
)

plt.ylabel(
    "Number of Samples"
)

plt.title(
    "LSTM Prediction Error Distribution"
)

plt.grid(True)

plt.show()

# ==========================================
# ERROR ANALYSIS BY TTF REGION
# ==========================================

print("\n==============================")
print("ERROR ANALYSIS BY TTF REGION")
print("==============================")


def evaluate_region(name, mask):

    actual = y_test[mask]
    predicted = predictions[mask]

    if len(actual) == 0:
        print(f"\n{name}: No samples")
        return

    region_mae = mean_absolute_error(
        actual,
        predicted
    )

    region_rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    region_r2 = r2_score(
        actual,
        predicted
    ) if len(actual) > 1 else np.nan

    mean_error = np.mean(
        predicted - actual
    )

    print(f"\n{name}")

    print(
        f"Samples: {len(actual)}"
    )

    print(
        f"MAE  : {region_mae:.3f} hours"
    )

    print(
        f"RMSE : {region_rmse:.3f} hours"
    )

    print(
        f"R²   : {region_r2:.3f}"
    )

    print(
        f"Mean error: {mean_error:.3f} hours"
    )


# ------------------------------------------
# DEFINE REGIONS
# ------------------------------------------

early_mask = y_test > 8

middle_mask = (
    (y_test > 4) &
    (y_test <= 8)
)

late_mask = (
    (y_test > 2) &
    (y_test <= 4)
)

near_failure_mask = y_test <= 2


# ------------------------------------------
# EVALUATE EACH REGION
# ------------------------------------------

evaluate_region(
    "EARLY (> 8 hours)",
    early_mask
)

evaluate_region(
    "MIDDLE (4–8 hours)",
    middle_mask
)

evaluate_region(
    "LATE (2–4 hours)",
    late_mask
)

evaluate_region(
    "NEAR FAILURE (<= 2 hours)",
    near_failure_mask
)