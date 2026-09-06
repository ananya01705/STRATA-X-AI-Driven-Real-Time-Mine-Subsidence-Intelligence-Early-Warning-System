import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# --------------------------------
# Load our synthetic data
# --------------------------------

data = pd.read_csv(
    "all_creep_scenarios.csv"
)


# --------------------------------
# Select ONE scenario
# --------------------------------

scenario_id = 0

scenario = data[
    data["scenario_id"] == scenario_id
].copy()


# --------------------------------
# Calculate inverse velocity
# --------------------------------

scenario["inverse_velocity"] = (
    1 / scenario["velocity"]
)


# --------------------------------
# Select tertiary creep region
# --------------------------------

# We only want the late-stage
# accelerating region.

failure_time = scenario["time_hours"].max()

tertiary_start = failure_time * 0.70

tertiary = scenario[
    scenario["time_hours"] >= tertiary_start
].copy()


# --------------------------------
# Remove extreme final points
# --------------------------------

# The last few synthetic points can
# become extremely large because of
# our mathematical simulation.

tertiary = tertiary[
    tertiary["inverse_velocity"] > 0
]


# --------------------------------
# Fit a straight line
# --------------------------------

x = tertiary["time_hours"].values

y = tertiary["inverse_velocity"].values


slope, intercept = np.polyfit(
    x,
    y,
    1
)


# --------------------------------
# Calculate estimated failure time
# --------------------------------

estimated_failure_time = (
    -intercept / slope
)


print(
    "Estimated failure time:",
    estimated_failure_time,
    "hours"
)

print(
    "Actual failure time:",
    failure_time,
    "hours"
)


# --------------------------------
# Plot
# --------------------------------

plt.figure(figsize=(10, 5))

plt.scatter(
    x,
    y,
    alpha=0.4,
    label="Inverse velocity"
)


# Fitted line
fit_y = (
    slope * x
    + intercept
)

plt.plot(
    x,
    fit_y,
    linestyle="--",
    label="Fitted trend"
)


# Zero line
plt.axhline(
    0,
    linestyle=":"
)


# Estimated failure
plt.axvline(
    estimated_failure_time,
    linestyle="--",
    label="Estimated failure"
)


plt.xlabel(
    "Time (hours)"
)

plt.ylabel(
    "Inverse velocity (1/v)"
)

plt.title(
    "Fukuzono Inverse-Velocity Analysis"
)

plt.legend()

plt.tight_layout()

plt.show()