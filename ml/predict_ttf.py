import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from tensorflow.keras.models import load_model

try:
    from risk_engine import classify_risk
except ImportError:
    from ml.risk_engine import classify_risk


# ==========================================
# LOAD MODEL AND SCALERS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

model = load_model(BASE_DIR / "ttf_lstm_model.keras")
feature_scaler = joblib.load(BASE_DIR / "lstm_feature_scaler.pkl")
target_scaler = joblib.load(BASE_DIR / "lstm_target_scaler.pkl")

WINDOW_SIZE = 30


# ==========================================
# PREDICTION FUNCTION
# ==========================================

def predict_ttf(sensor_data):
    if len(sensor_data) < WINDOW_SIZE:
        raise ValueError("At least 30 readings are required.")

    # Take the latest 30 readings
    recent_data = sensor_data[-WINDOW_SIZE:].copy()

    # Calculate inverse velocity with zero-division safeguard
    velocity_safe = np.maximum(recent_data["velocity"].values.astype(float), 1e-4)
    recent_data["inverse_velocity"] = 1.0 / velocity_safe

    # Select features in EXACT training order
    features = recent_data[
        [
            "velocity",
            "deformation",
            "inverse_velocity"
        ]
    ].values

    # Scale using the SAME scaler from training
    features_scaled = feature_scaler.transform(features)

    # Add batch dimension
    X = np.expand_dims(features_scaled, axis=0)

    # Predict
    prediction_scaled = model.predict(X, verbose=0)

    # Convert prediction back to hours
    prediction = target_scaler.inverse_transform(prediction_scaled)

    # Raw LSTM prediction
    prediction = float(prediction[0][0])

    return prediction


# ==========================================
# TEST PREDICTIONS AT DIFFERENT POINTS
# ==========================================

if __name__ == "__main__":
    csv_path = BASE_DIR / "all_creep_scenarios.csv"
    if not csv_path.exists():
        print(f"Dataset not found at {csv_path}")
    else:
        data = pd.read_csv(csv_path)

        test_scenario = data[data["scenario_id"] == 1].copy()
        test_scenario = test_scenario.reset_index(drop=True)

        print("\n======================================")
        print("TTF PROGRESSION TEST - SCENARIO 1")
        print("======================================")

        # Points where we want to test
        test_points = [
            100,
            300,
            500,
            700,
            900,
            1100,
            len(test_scenario)
        ]

        for point in test_points:
            if point < WINDOW_SIZE:
                continue

            current_data = test_scenario.iloc[:point].copy()
            prediction = predict_ttf(current_data)
            risk = classify_risk(prediction)
            actual_ttf = current_data["ttf"].iloc[-1]

            print(f"\nReading: {point}")
            print(f"Actual TTF:     {actual_ttf:.3f} h")
            print(f"Predicted TTF:  {prediction:.3f} h")
            print(f"Prototype Risk: {risk}")
            print(f"Error:          {prediction - actual_ttf:+.3f} h")