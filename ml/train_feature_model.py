import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# =====================================
# 1. Load engineered features
# =====================================

X = np.load("X_features.npy")
y = np.load("y_features.npy")

print("Dataset loaded")
print("X shape:", X.shape)
print("y shape:", y.shape)


# =====================================
# 2. Split data
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print("\nTraining samples:", len(X_train))
print("Testing samples :", len(X_test))


# =====================================
# 3. Create Random Forest
# =====================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


# =====================================
# 4. Train
# =====================================

print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print("Training complete!")


# =====================================
# 5. Predictions
# =====================================

predictions = model.predict(
    X_test
)


# =====================================
# 6. Evaluation
# =====================================

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
print("MODEL PERFORMANCE")
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


# =====================================
# 7. Show sample predictions
# =====================================

print("\nSample predictions")
print("------------------------------")

for i in range(
    min(10, len(y_test))
):

    print(
        f"Actual: {y_test[i]:.2f} h"
        f" | Predicted: {predictions[i]:.2f} h"
    )


# =====================================
# 8. Save model
# =====================================

joblib.dump(
    model,
    "ttf_feature_model.pkl"
)

print(
    "\nModel saved as:"
)

print(
    "ttf_feature_model.pkl"
)

feature_names = [
    "mean_velocity",
    "max_velocity",
    "velocity_std",
    "velocity_slope",
    "mean_inverse_velocity",
    "inverse_velocity_slope",
    "mean_deformation",
    "deformation_change"
]


print("\nFeature importance")
print("------------------------------")

for name, importance in zip(
    feature_names,
    model.feature_importances_
):

    print(
        f"{name:30s} {importance:.4f}"
    )