import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error


# --------------------------------
# Load test data
# --------------------------------

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


# --------------------------------
# Load trained model
# --------------------------------

model = joblib.load(
    "ttf_random_forest.pkl"
)


# --------------------------------
# Flatten test windows
# --------------------------------

X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


# --------------------------------
# Make predictions
# --------------------------------

predictions = model.predict(
    X_test_flat
)


# --------------------------------
# Calculate MAE
# --------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

print(
    f"Test MAE: {mae:.3f} hours"
)


# --------------------------------
# Plot actual vs predicted
# --------------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    y_test[:500],
    label="Actual TTF"
)

plt.plot(
    predictions[:500],
    label="Predicted TTF"
)

plt.xlabel("Test Sample")
plt.ylabel("Time-to-Failure (hours)")

plt.title(
    "Actual vs Predicted Time-to-Failure"
)

plt.legend()

plt.tight_layout()

plt.show()