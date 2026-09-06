import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestRegressor

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

print(
    "Number of scenarios:",
    data["scenario_id"].nunique()
)


# ==========================================
# INVERSE VELOCITY
# ==========================================

data["inverse_velocity"] = (
    1 / data["velocity"]
)


# ==========================================
# FEATURE EXTRACTION
# ==========================================

def extract_features(window):

    time = window["time_hours"].values

    velocity = window["velocity"].values

    inverse_velocity = (
        window["inverse_velocity"].values
    )

    deformation = (
        window["deformation"].values
    )


    mean_velocity = np.mean(
        velocity
    )

    max_velocity = np.max(
        velocity
    )

    velocity_std = np.std(
        velocity
    )

    velocity_slope = np.polyfit(
        time,
        velocity,
        1
    )[0]


    mean_inverse_velocity = np.mean(
        inverse_velocity
    )

    inverse_velocity_slope = np.polyfit(
        time,
        inverse_velocity,
        1
    )[0]


    mean_deformation = np.mean(
        deformation
    )

    deformation_change = (
        deformation[-1]
        - deformation[0]
    )


    return [
        mean_velocity,
        max_velocity,
        velocity_std,
        velocity_slope,
        mean_inverse_velocity,
        inverse_velocity_slope,
        mean_deformation,
        deformation_change
    ]


# ==========================================
# CREATE WINDOWS FOR GIVEN SCENARIOS
# ==========================================

def create_dataset(
    scenario_ids
):

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


            features = extract_features(
                window
            )


            target = scenario[
                "ttf"
            ].iloc[
                i + WINDOW_SIZE - 1
            ]


            X.append(features)

            y.append(target)


    return (
        np.array(X),
        np.array(y)
    )


# ==========================================
# SPLIT SCENARIOS
# ==========================================

scenario_ids = sorted(
    data["scenario_id"].unique()
)

np.random.seed(42)

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
print("====================")

print(
    "Training:",
    train_scenarios
)

print(
    "Validation:",
    val_scenarios
)

print(
    "Testing:",
    test_scenarios
)


# ==========================================
# CREATE WINDOWS AFTER SPLITTING
# ==========================================

X_train, y_train = create_dataset(
    train_scenarios
)

X_val, y_val = create_dataset(
    val_scenarios
)

X_test, y_test = create_dataset(
    test_scenarios
)


print("\nDataset sizes")
print("====================")

print(
    "Training:",
    X_train.shape
)

print(
    "Validation:",
    X_val.shape
)

print(
    "Testing:",
    X_test.shape
)


# ==========================================
# TRAIN MODEL
# ==========================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)


print("\nTraining model...")

model.fit(
    X_train,
    y_train
)

print(
    "Training complete!"
)


# ==========================================
# VALIDATION
# ==========================================

val_predictions = model.predict(
    X_val
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


print("\nVALIDATION PERFORMANCE")
print("============================")

print(
    f"MAE  : {val_mae:.3f} hours"
)

print(
    f"RMSE : {val_rmse:.3f} hours"
)

print(
    f"R²   : {val_r2:.3f}"
)


# ==========================================
# FINAL TEST
# ==========================================

test_predictions = model.predict(
    X_test
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


print("\nFINAL TEST PERFORMANCE")
print("============================")

print(
    f"MAE  : {test_mae:.3f} hours"
)

print(
    f"RMSE : {test_rmse:.3f} hours"
)

print(
    f"R²   : {test_r2:.3f}"
)