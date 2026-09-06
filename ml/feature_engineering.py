import pandas as pd
import numpy as np


WINDOW_SIZE = 30


# -----------------------------------
# Load dataset
# -----------------------------------

data = pd.read_csv(
    "all_creep_scenarios.csv"
)


# -----------------------------------
# Calculate inverse velocity
# -----------------------------------

data["inverse_velocity"] = (
    1 / data["velocity"]
)


# -----------------------------------
# Feature extraction function
# -----------------------------------

def extract_features(window):

    time = window["time_hours"].values

    velocity = window["velocity"].values

    inverse_velocity = (
        window["inverse_velocity"].values
    )

    displacement = (
        window["deformation"].values
    )


    # -------------------------------
    # Basic velocity features
    # -------------------------------

    mean_velocity = np.mean(
        velocity
    )

    max_velocity = np.max(
        velocity
    )

    velocity_std = np.std(
        velocity
    )


    # -------------------------------
    # Velocity trend
    # -------------------------------

    velocity_slope = np.polyfit(
        time,
        velocity,
        1
    )[0]


    # -------------------------------
    # Inverse velocity features
    # -------------------------------

    mean_inverse_velocity = np.mean(
        inverse_velocity
    )

    inverse_velocity_slope = np.polyfit(
        time,
        inverse_velocity,
        1
    )[0]


    # -------------------------------
    # Displacement features
    # -------------------------------

    mean_displacement = np.mean(
        displacement
    )

    displacement_change = (
        displacement[-1]
        - displacement[0]
    )


    return [
        mean_velocity,
        max_velocity,
        velocity_std,
        velocity_slope,
        mean_inverse_velocity,
        inverse_velocity_slope,
        mean_displacement,
        displacement_change
    ]


# -----------------------------------
# Create feature dataset
# -----------------------------------

X = []
y = []


for scenario_id, scenario in data.groupby(
    "scenario_id"
):

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


# -----------------------------------
# Convert to NumPy
# -----------------------------------

X = np.array(X)

y = np.array(y)


# -----------------------------------
# Feature names
# -----------------------------------

feature_names = [
    "mean_velocity",
    "max_velocity",
    "velocity_std",
    "velocity_slope",
    "mean_inverse_velocity",
    "inverse_velocity_slope",
    "mean_displacement",
    "displacement_change"
]


# -----------------------------------
# Display information
# -----------------------------------

print(
    "Feature matrix shape:",
    X.shape
)

print(
    "Target shape:",
    y.shape
)

print("\nFeatures:")

for name in feature_names:

    print("-", name)


# -----------------------------------
# Save
# -----------------------------------

np.save(
    "X_features.npy",
    X
)

np.save(
    "y_features.npy",
    y
)

print(
    "\nFeature dataset saved!"
)