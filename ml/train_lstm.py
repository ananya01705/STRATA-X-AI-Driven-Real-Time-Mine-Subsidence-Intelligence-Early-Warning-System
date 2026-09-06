import pandas as pd
import numpy as np
import json

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    LSTM,
    Dense,
    Dropout
)

from tensorflow.keras.callbacks import EarlyStopping
import joblib


# ==========================================
# SETTINGS
# ==========================================

WINDOW_SIZE = 30

RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv(
    "all_creep_scenarios.csv"
)

print("Dataset loaded")

print(
    "Number of scenarios:",
    data["scenario_id"].nunique()
)


# ==========================================
# CREATE SEQUENCES
# ==========================================

def create_sequences(scenario_ids):

    X = []
    y = []

    for scenario_id in scenario_ids:

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


            # ------------------------------
            # Input sequence
            # ------------------------------

            sequence = window[
    [
        "velocity",
        "deformation",
        "inverse_velocity"
    ]
].values


            # ------------------------------
            # Target
            # ------------------------------

            target = scenario[
                "ttf"
            ].iloc[
                i + WINDOW_SIZE - 1
            ]


            X.append(sequence)
            y.append(target)


    return (
        np.array(X, dtype=np.float32),
        np.array(y, dtype=np.float32)
    )


# ==========================================
# SCENARIO-LEVEL SPLIT
# ==========================================

scenario_ids = sorted(
    data["scenario_id"].unique()
)

np.random.shuffle(
    scenario_ids
)


total = len(
    scenario_ids
)


train_end = int(
    total * 0.70
)

val_end = int(
    total * 0.85
)


train_scenarios = scenario_ids[
    :train_end
]

val_scenarios = scenario_ids[
    train_end:val_end
]

test_scenarios = scenario_ids[
    val_end:
]


print("\nScenario split")
print("======================")

print(
    "Training scenarios:",
    len(train_scenarios)
)

print(
    "Validation scenarios:",
    len(val_scenarios)
)

print(
    "Test scenarios:",
    len(test_scenarios)
)

# ==========================================
# SAVE SCENARIO SPLIT
# ==========================================

scenario_split = {
    "train": [int(x) for x in train_scenarios],
    "validation": [int(x) for x in val_scenarios],
    "test": [int(x) for x in test_scenarios]
}

with open(
    "scenario_split.json",
    "w"
) as f:

    json.dump(
        scenario_split,
        f,
        indent=4
    )

print(
    "\nScenario split saved as:"
)

print(
    "scenario_split.json"
)
# ==========================================
# CREATE SEQUENCES
# ==========================================

X_train, y_train = create_sequences(
    train_scenarios
)

X_val, y_val = create_sequences(
    val_scenarios
)

X_test, y_test = create_sequences(
    test_scenarios
)


print("\nSequence shapes")
print("======================")

print(
    "X_train:",
    X_train.shape
)

print(
    "X_val:",
    X_val.shape
)

print(
    "X_test:",
    X_test.shape
)


# ==========================================
# SCALE INPUT FEATURES
# ==========================================

# We need to temporarily flatten the
# training sequences.

num_features = X_train.shape[2]


X_train_2d = X_train.reshape(
    -1,
    num_features
)


X_val_2d = X_val.reshape(
    -1,
    num_features
)


X_test_2d = X_test.reshape(
    -1,
    num_features
)


# IMPORTANT:
# Fit ONLY on training data

feature_scaler = StandardScaler()

feature_scaler.fit(
    X_train_2d
)


# Transform all datasets

X_train_scaled = feature_scaler.transform(
    X_train_2d
)

X_val_scaled = feature_scaler.transform(
    X_val_2d
)

X_test_scaled = feature_scaler.transform(
    X_test_2d
)


# Restore 3D shape

X_train_scaled = X_train_scaled.reshape(
    X_train.shape
)

X_val_scaled = X_val_scaled.reshape(
    X_val.shape
)

X_test_scaled = X_test_scaled.reshape(
    X_test.shape
)


# ==========================================
# SCALE TARGET
# ==========================================

target_scaler = StandardScaler()

y_train_scaled = target_scaler.fit_transform(
    y_train.reshape(-1, 1)
).flatten()


y_val_scaled = target_scaler.transform(
    y_val.reshape(-1, 1)
).flatten()


y_test_scaled = target_scaler.transform(
    y_test.reshape(-1, 1)
).flatten()


print("\nScaling complete")


# ==========================================
# BUILD LSTM
# ==========================================

model = Sequential([

    LSTM(
        64,
        input_shape=(
            WINDOW_SIZE,
            num_features
        )
    ),

    Dropout(0.2),

    Dense(32, activation="relu"),

    Dense(1)
])


# ==========================================
# COMPILE
# ==========================================

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"]
)


print("\nModel architecture")

model.summary()


# ==========================================
# EARLY STOPPING
# ==========================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True
)


# ==========================================
# TRAIN
# ==========================================

print("\nTraining LSTM...")

history = model.fit(

    X_train_scaled,

    y_train_scaled,

    validation_data=(
        X_val_scaled,
        y_val_scaled
    ),

    epochs=40,

    batch_size=256,

    callbacks=[
        early_stopping
    ],

    verbose=1
)


print("\nTraining complete!")



# ==========================================
# PREDICTIONS
# ==========================================

predictions_scaled = model.predict(
    X_test_scaled
)


# Convert predictions back to hours

predictions = target_scaler.inverse_transform(
    predictions_scaled
).flatten()


# ==========================================
# EVALUATION
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
# ==========================================
# VALIDATION BIAS / CALIBRATION
# ==========================================

val_predictions_scaled = model.predict(
    X_val_scaled,
    verbose=0
)

val_predictions = target_scaler.inverse_transform(
    val_predictions_scaled
).flatten()

# Calculate validation errors
val_errors = val_predictions - y_val

# Mean prediction bias
bias = np.mean(val_errors)

print("\n======================================")
print("VALIDATION BIAS ANALYSIS")
print("======================================")

print(f"Mean error / bias: {bias:.4f} hours")

print(f"Validation MAE before: "
      f"{mean_absolute_error(y_val, val_predictions):.4f} hours")

print(f"Validation RMSE before: "
      f"{np.sqrt(mean_squared_error(y_val, val_predictions)):.4f} hours")


# Apply bias correction
val_predictions_corrected = (
    val_predictions - bias
)

print(f"\nValidation MAE after: "
      f"{mean_absolute_error(y_val, val_predictions_corrected):.4f} hours")

print(f"Validation RMSE after: "
      f"{np.sqrt(mean_squared_error(y_val, val_predictions_corrected)):.4f} hours")

print("\n==============================")
print("FINAL LSTM TEST PERFORMANCE")
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
        f"Actual: {y_test[i]:.2f} h"
        f" | Predicted: {predictions[i]:.2f} h"
    )


# ==========================================
# SAVE MODEL AND SCALERS
# ==========================================

model.save(
    "ttf_lstm_model.keras"
)

joblib.dump(
    feature_scaler,
    "lstm_feature_scaler.pkl"
)

joblib.dump(
    target_scaler,
    "lstm_target_scaler.pkl"
)


print("\nFiles saved:")
print("------------------------------")

print("ttf_lstm_model.keras")
print("lstm_feature_scaler.pkl")
print("lstm_target_scaler.pkl")