import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Reproducible randomness
np.random.seed(42)

# -----------------------------
# Simulation settings
# -----------------------------

failure_time = 10.0       # hours

primary_end = 3.0
secondary_end = 7.0

dt = 0.01

time = np.arange(0, failure_time, dt)

# Array for velocity
velocity = np.zeros_like(time)


# -----------------------------
# PRIMARY CREEP
# -----------------------------

primary_mask = time < primary_end

t_primary = time[primary_mask]

# Starts relatively fast and gradually slows down
v_initial = 0.15
v_secondary = 0.05
tau = 0.8

velocity[primary_mask] = (
    v_secondary
    + (v_initial - v_secondary)
    * np.exp(-t_primary / tau)
)


# -----------------------------
# SECONDARY CREEP
# -----------------------------

secondary_mask = (
    (time >= primary_end)
    & (time < secondary_end)
)

velocity[secondary_mask] = v_secondary


# -----------------------------
# TERTIARY CREEP
# -----------------------------

tertiary_mask = time >= secondary_end

t_tertiary = time[tertiary_mask]

# Velocity increases toward failure
remaining = failure_time - t_tertiary

velocity[tertiary_mask] = (
    v_secondary
    * (failure_time - secondary_end)
    / remaining
)


# -----------------------------
# Add small sensor noise
# -----------------------------

noise = np.random.normal(
    0,
    0.002,
    len(velocity)
)

velocity_noisy = velocity + noise

# Prevent negative velocity
velocity_noisy = np.maximum(
    velocity_noisy,
    0.001
)


# -----------------------------
# Calculate deformation
# -----------------------------

deformation = np.cumsum(
    velocity_noisy * dt
)


# -----------------------------
# Create DataFrame
# -----------------------------

data = pd.DataFrame({
    "time_hours": time,
    "velocity": velocity_noisy,
    "deformation": deformation
})


# Save dataset

data.to_csv(
    "rock_creep_dataset.csv",
    index=False
)

print("Dataset created!")
print(data.head())

# -----------------------------
# Plot velocity
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    velocity_noisy
)

plt.axvline(
    primary_end,
    linestyle="--",
    label="Primary → Secondary"
)

plt.axvline(
    secondary_end,
    linestyle="--",
    label="Secondary → Tertiary"
)

plt.axvline(
    failure_time,
    linestyle="--",
    label="Failure"
)

plt.xlabel("Time (hours)")
plt.ylabel("Deformation velocity")

plt.title(
    "Synthetic Rock Creep Velocity"
)

plt.legend()

plt.show()


# -----------------------------
# Plot deformation
# -----------------------------

plt.figure(figsize=(10, 5))

plt.plot(
    time,
    deformation
)

plt.xlabel("Time (hours)")
plt.ylabel("Cumulative deformation")

plt.title(
    "Synthetic Rock Creep Deformation"
)

plt.show()