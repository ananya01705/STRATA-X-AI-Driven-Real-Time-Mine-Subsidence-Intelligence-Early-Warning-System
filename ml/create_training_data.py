import numpy as np
import pandas as pd

np.random.seed(42)


def generate_scenario(failure_time, dt=0.01):

    primary_end = failure_time * 0.30
    secondary_end = failure_time * 0.70

    time = np.arange(0, failure_time, dt)

    velocity = np.zeros(len(time))

    # --------------------------------
    # Slight scenario variation
    # --------------------------------

    v_initial = np.random.uniform(0.12, 0.20)
    v_secondary = np.random.uniform(0.035, 0.065)

    tau = np.random.uniform(0.6, 1.2)

    # --------------------------------
    # PRIMARY CREEP
    # --------------------------------

    primary = time < primary_end

    t_primary = time[primary]

    velocity[primary] = (
        v_secondary
        + (v_initial - v_secondary)
        * np.exp(-t_primary / tau)
    )

    # --------------------------------
    # SECONDARY CREEP
    # --------------------------------

    secondary = (
        (time >= primary_end)
        &
        (time < secondary_end)
    )

    velocity[secondary] = v_secondary

    # --------------------------------
    # TERTIARY CREEP
    # --------------------------------

    tertiary = time >= secondary_end

    t_tertiary = time[tertiary]

    remaining = failure_time - t_tertiary

    velocity[tertiary] = (
        v_secondary
        * (failure_time - secondary_end)
        / remaining
    )

    # --------------------------------
    # Sensor noise
    # --------------------------------

    noise = np.random.normal(
        0,
        0.002,
        len(velocity)
    )

    velocity = velocity + noise

    velocity = np.maximum(
        velocity,
        0.001
    )

    # --------------------------------
    # Deformation
    # --------------------------------

    deformation = np.cumsum(
        velocity * dt
    )

    # --------------------------------
    # Inverse velocity
    # --------------------------------

    inverse_velocity = 1 / velocity

    # --------------------------------
    # TTF
    # --------------------------------

    ttf = failure_time - time

    return pd.DataFrame({
        "time_hours": time,
        "velocity": velocity,
        "deformation": deformation,
        "inverse_velocity": inverse_velocity,
        "ttf": ttf
    })


# ====================================
# Generate multiple scenarios
# ====================================

all_data = []

number_of_scenarios = 100

for scenario_id in range(number_of_scenarios):

    failure_time = np.random.uniform(
        8,
        12
    )

    scenario = generate_scenario(
        failure_time
    )

    scenario["scenario_id"] = scenario_id

    all_data.append(scenario)


# ====================================
# Combine everything
# ====================================

dataset = pd.concat(
    all_data,
    ignore_index=True
)


# ====================================
# Save
# ====================================

dataset.to_csv(
    "all_creep_scenarios.csv",
    index=False
)


print("Dataset created successfully!")

print("\nNumber of scenarios:")
print(dataset["scenario_id"].nunique())

print("\nDataset shape:")
print(dataset.shape)

print("\nFirst rows:")
print(dataset.head())

print("\nColumns:")
print(dataset.columns.tolist())