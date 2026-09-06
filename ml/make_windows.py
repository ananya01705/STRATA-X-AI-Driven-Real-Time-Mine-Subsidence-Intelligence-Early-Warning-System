import pandas as pd
import numpy as np

# Load our dataset
data = pd.read_csv(
    "all_creep_scenarios.csv"
)

# Number of readings the model sees
WINDOW_SIZE = 30

X = []
y = []


# Process each scenario separately
for scenario_id, scenario in data.groupby("scenario_id"):

    # Reset row numbering
    scenario = scenario.reset_index(drop=True)

    velocity = scenario["velocity"].values
    inverse_velocity = scenario["inverse_velocity"].values
    ttf = scenario["ttf"].values

    # Sliding window
    for i in range(
        len(scenario) - WINDOW_SIZE
    ):

        # Sensor history
        velocity_window = velocity[
            i:i + WINDOW_SIZE
        ]

        inverse_window = inverse_velocity[
            i:i + WINDOW_SIZE
        ]

        # Combine features
        window = np.column_stack([
            velocity_window,
            inverse_window
        ])

        # Target = TTF at end of window
        target = ttf[
            i + WINDOW_SIZE - 1
        ]

        X.append(window)
        y.append(target)


# Convert to NumPy arrays
X = np.array(X)
y = np.array(y)


print("X shape:")
print(X.shape)

print("\ny shape:")
print(y.shape)

print("\nExample input:")
print(X[0])

print("\nExample target:")
print(y[0])