import pandas as pd

data = pd.read_csv("synthetic_creep_data.csv")

print("First 5 rows:")
print(data.head())

print("\nDataset shape:")
print(data.shape)

print("\nColumn names:")
print(data.columns)

print("\nStatistics:")
print(data.describe())

print("\nMissing values:")
print(data.isnull().sum())