import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Make results reproducible
np.random.seed(42)

# Number of time points
n = 1000

# Time
time = np.linspace(0, 10, n)

# Choose an approximate failure time
failure_time = 10

# Distance from failure
remaining_time = failure_time - time

# Avoid division by zero
remaining_time = np.maximum(remaining_time, 0.05)

# -------------------------------
# Synthetic deformation signal
# -------------------------------

# Base deformation
deformation = (
    0.05 * time
    + 0.002 / remaining_time
)

# Add small measurement noise
noise = np.random.normal(0, 0.002, n)

deformation = deformation + noise

# Put into a DataFrame
data = pd.DataFrame({
    "time": time,
    "deformation": deformation
})

# Save dataset
data.to_csv("synthetic_creep_data.csv", index=False)

print("Dataset created!")
print(data.head())

# -------------------------------
# Plot
# -------------------------------

plt.plot(time, deformation)

plt.xlabel("Time")
plt.ylabel("Deformation")

plt.title("Synthetic Ground Deformation")

plt.show()