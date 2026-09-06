import numpy as np
import pandas as pd

np.random.seed(42)

# Number of normal readings
n = 2000

# Generate normal sensor behavior
tilt_x = np.random.normal(0.5, 0.15, n)
tilt_y = np.random.normal(0.4, 0.15, n)
vibration = np.random.normal(0.10, 0.03, n)

normal_data = pd.DataFrame({
    "tilt_x": tilt_x,
    "tilt_y": tilt_y,
    "vibration": vibration
})

# Save normal data
normal_data.to_csv("normal_data.csv", index=False)

print("Normal dataset created!")
print(normal_data.head())