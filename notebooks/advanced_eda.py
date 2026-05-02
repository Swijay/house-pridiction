import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it does not exist
os.makedirs("images", exist_ok=True)

# Load dataset
df = pd.read_csv("data/houses.csv")

print("Dataset loaded successfully")
print("Shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns)

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

# Log transform price
df["log_price"] = np.log1p(df["price"])

# Plot original price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("Original Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/original_price_distribution.png")
plt.show()

# Plot log-transformed price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["log_price"], kde=True)
plt.title("Log Transformed Price Distribution")
plt.xlabel("Log Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/log_price_distribution.png")
plt.show()

# Area vs price before clipping
plt.figure(figsize=(8, 5))
sns.scatterplot(x="area", y="price", data=df)
plt.title("Area vs Price Before Clipping")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("images/area_vs_price_before_clipping.png")
plt.show()

# Clip extreme area values using 99th percentile
area_upper_limit = df["area"].quantile(0.99)
df["area_clipped"] = df["area"].clip(upper=area_upper_limit)

print("\nArea 99th percentile limit:", area_upper_limit)

# Area vs price after clipping
plt.figure(figsize=(8, 5))
sns.scatterplot(x="area_clipped", y="price", data=df)
plt.title("Area vs Price After Clipping")
plt.xlabel("Clipped Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("images/area_vs_price_after_clipping.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap After Log Transform and Area Clipping")
plt.tight_layout()
plt.savefig("images/advanced_correlation_heatmap.png")
plt.show()

print("\nAdvanced EDA completed successfully.")
print("Graphs saved inside images/ folder.")