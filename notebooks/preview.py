import pandas as pd

# Load original Kaggle dataset
df = pd.read_csv("data/houses.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe())

print("\nColumn names:")
print(df.columns)

print("\nMissing values:")
print(df.isnull().sum())