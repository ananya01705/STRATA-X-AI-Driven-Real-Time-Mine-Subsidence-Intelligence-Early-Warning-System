import numpy as np
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import mean_absolute_error


# Load data
X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")


# Load model
model = joblib.load(
    "ttf_random_forest.pkl"
)


# Flatten
X_test_flat = X_test.reshape(
    X_test.shape[0],
    -1
)


# Predict
predictions = model.predict(
    X_test_flat
)


# MAE
mae = mean_absolute_error(
    y_test,
    predictions
)


# --------------------------------
# Scatter plot
# --------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    y_test,
    predictions,
    alpha=0.4
)


# Perfect prediction line
minimum = min(
    y_test.min(),
    predictions.min()
)

maximum = max(
    y_test.max(),
    predictions.max()
)

plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--",
    label="Perfect Prediction"
)


plt.xlabel(
    "Actual TTF (hours)"
)

plt.ylabel(
    "Predicted TTF (hours)"
)

plt.title(
    f"TTF Prediction\nMAE = {mae:.2f} hours"
)

plt.legend()

plt.tight_layout()

plt.show()