import pandas as pd
import joblib

# Load trained model
model = joblib.load("subsidence_model.pkl")

# Test data
test_data = pd.DataFrame({
    "tilt_x": [0.5, 0.6, 0.4, 3.5, 5.2],
    "tilt_y": [0.4, 0.5, 0.3, 3.1, 4.8],
    "vibration": [0.10, 0.12, 0.08, 0.40, 0.70]
})

features = [
    "tilt_x",
    "tilt_y",
    "vibration"
]

X_test = test_data[features]

predictions = model.predict(X_test)

for i, prediction in enumerate(predictions):

    if prediction == 1:
        status = "NORMAL 🟢"
    else:
        status = "ANOMALY 🔴"

    print(
        f"Reading {i + 1}: "
        f"tilt_x={test_data.iloc[i]['tilt_x']}, "
        f"tilt_y={test_data.iloc[i]['tilt_y']} "
        f"→ {status}"
    )