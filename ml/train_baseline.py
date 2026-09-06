import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

import joblib


# -----------------------------
# Load datasets
# -----------------------------

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


print("Original shape:")
print("X_train:", X_train.shape)


# -----------------------------
# Flatten the windows
# -----------------------------

# Random Forest expects
# [samples, features]

X_train_flat = X_train.reshape(
    X_train.shape[0],
    -1
)

X_val_flat = X_val.reshape(
    X_val.shape[0],
    -1
)

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


print("\nFlattened shape:")
print("X_train:", X_train_flat.shape)


# -----------------------------
# Create model
# -----------------------------

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


# -----------------------------
# Train
# -----------------------------

print("\nTraining model...")

model.fit(
    X_train_flat,
    y_train
)

print("Training complete!")


# -----------------------------
# Validation prediction
# -----------------------------

val_predictions = model.predict(
    X_val_flat
)


val_mae = mean_absolute_error(
    y_val,
    val_predictions
)

val_rmse = np.sqrt(
    mean_squared_error(
        y_val,
        val_predictions
    )
)

val_r2 = r2_score(
    y_val,
    val_predictions
)


print("\nValidation Results")
print("------------------")

print("MAE :", val_mae)
print("RMSE:", val_rmse)
print("R²  :", val_r2)


# -----------------------------
# Test prediction
# -----------------------------

test_predictions = model.predict(
    X_test_flat
)


test_mae = mean_absolute_error(
    y_test,
    test_predictions
)

test_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        test_predictions
    )
)

test_r2 = r2_score(
    y_test,
    test_predictions
)


print("\nTest Results")
print("------------")

print("MAE :", test_mae)
print("RMSE:", test_rmse)
print("R²  :", test_r2)


# -----------------------------
# Show sample predictions
# -----------------------------

print("\nSample Predictions")
print("------------------")

for i in range(10):

    print(
        f"Actual: {y_test[i]:.2f} h"
        f" | Predicted: {test_predictions[i]:.2f} h"
    )


# -----------------------------
# Save model
# -----------------------------

joblib.dump(
    model,
    "ttf_random_forest.pkl"
)

print(
    "\nModel saved as "
    "ttf_random_forest.pkl"
)