import pandas as pd
from sklearn.ensemble import IsolationForest
import joblib

# Load normal sensor data
data = pd.read_csv("normal_data.csv")

# Features used by the model
features = [
    "tilt_x",
    "tilt_y",
    "vibration"
]

X = data[features]

# Create model
model = IsolationForest(
    n_estimators=200,
    contamination=0.02,
    random_state=42
)

# Train
model.fit(X)

# Save trained model
joblib.dump(model, "subsidence_model.pkl")

print("Model trained successfully!")