import pandas as pd
import numpy as np


WINDOW_SIZE = 30


# ==========================================
# LOAD DATA
# ==========================================

data = pd.read_csv(
    "all_creep_scenarios.csv"
)

print("Dataset loaded")

print(
    "Scenarios:",
    data["scenario_id"].nunique()
)


# ==========================================
# CREATE SEQUENCES
# ==========================================

X = []
y = []


scenario_ids = sorted(
    data["scenario_id"].unique()
)


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


        # Two time-series features
        sequence = window[
    [
        "velocity",
        "deformation",
        "inverse_velocity"
    ]
].values


        # TTF at end of window
        target = scenario[
            "ttf"
        ].iloc[
            i + WINDOW_SIZE - 1
        ]


        X.append(sequence)

        y.append(target)


# ==========================================
# CONVERT TO NUMPY
# ==========================================

X = np.array(X)

y = np.array(y)


print("\nLSTM dataset created")

print(
    "X shape:",
    X.shape
)

print(
    "y shape:",
    y.shape
)


# ==========================================
# SAVE
# ==========================================

np.save(
    "X_lstm.npy",
    X
)

np.save(
    "y_lstm.npy",
    y
)


print(
    "\nSaved:"
)

print(
    "X_lstm.npy"
)

print(
    "y_lstm.npy"
)