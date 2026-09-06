import pandas as pd
import numpy as np

# -----------------------------
# Load dataset
# -----------------------------

data = pd.read_csv("all_creep_scenarios.csv")

WINDOW_SIZE = 30

# -----------------------------
# Split scenarios
# -----------------------------

scenario_ids = data["scenario_id"].unique()

np.random.seed(42)

np.random.shuffle(scenario_ids)

n = len(scenario_ids)

train_end = int(0.70 * n)
val_end = int(0.85 * n)

train_ids = scenario_ids[:train_end]
val_ids = scenario_ids[train_end:val_end]
test_ids = scenario_ids[val_end:]

print("Training scenarios:", len(train_ids))
print("Validation scenarios:", len(val_ids))
print("Testing scenarios:", len(test_ids))


# -----------------------------
# Function to create windows
# -----------------------------

def create_windows(data, scenario_ids):

    X = []
    y = []

    selected_data = data[
        data["scenario_id"].isin(scenario_ids)
    ]

    for scenario_id, scenario in selected_data.groupby(
        "scenario_id"
    ):

        scenario = scenario.reset_index(drop=True)

        velocity = scenario["velocity"].values
        inverse_velocity = scenario["inverse_velocity"].values
        ttf = scenario["ttf"].values

        for i in range(
            len(scenario) - WINDOW_SIZE
        ):

            velocity_window = velocity[
                i:i + WINDOW_SIZE
            ]

            inverse_window = inverse_velocity[
                i:i + WINDOW_SIZE
            ]

            window = np.column_stack([
                velocity_window,
                inverse_window
            ])

            target = ttf[
                i + WINDOW_SIZE - 1
            ]

            X.append(window)
            y.append(target)

    return np.array(X), np.array(y)


# -----------------------------
# Create datasets
# -----------------------------

X_train, y_train = create_windows(
    data,
    train_ids
)

X_val, y_val = create_windows(
    data,
    val_ids
)

X_test, y_test = create_windows(
    data,
    test_ids
)


# -----------------------------
# Print shapes
# -----------------------------

print("\nDataset shapes:")

print("X_train:", X_train.shape)
print("y_train:", y_train.shape)

print("X_val:", X_val.shape)
print("y_val:", y_val.shape)

print("X_test:", X_test.shape)
print("y_test:", y_test.shape)


# -----------------------------
# Save datasets
# -----------------------------

np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)

np.save("X_val.npy", X_val)
np.save("y_val.npy", y_val)

np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("\nML datasets saved successfully!")