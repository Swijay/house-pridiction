import pandas as pd

# Load original Kaggle dataset
df = pd.read_csv("data/houses.csv")

print("Before cleaning:")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

# Remove duplicate rows
df = df.drop_duplicates()

# Drop rows where target column is missing
df = df.dropna(subset=["price"])

# Fill numeric missing values with median
numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

# Fill categorical missing values with mode
categorical_cols = df.select_dtypes(include=["object"]).columns

for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# Save cleaned dataset
df.to_csv("data/houses_clean.csv", index=False)

print("\nAfter cleaning:")
print("Shape:", df.shape)
print("\nMissing values:")
print(df.isnull().sum())

print("\nCleaned file created successfully: data/houses_clean.csv")