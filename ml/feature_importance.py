import numpy as np
import matplotlib.pyplot as plt
import joblib


model = joblib.load(
    "ttf_random_forest.pkl"
)


importance = model.feature_importances_


# 30 velocity + 30 inverse velocity
velocity_importance = importance[:30]

inverse_importance = importance[30:]


total_velocity = velocity_importance.sum()
total_inverse = inverse_importance.sum()


print(
    "Total velocity importance:",
    total_velocity
)

print(
    "Total inverse velocity importance:",
    total_inverse
)


plt.bar(
    ["Velocity", "Inverse Velocity"],
    [total_velocity, total_inverse]
)

plt.ylabel(
    "Total Feature Importance"
)

plt.title(
    "Feature Importance for TTF Prediction"
)

plt.tight_layout()

plt.show()