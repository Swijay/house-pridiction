import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create images folder if it does not exist
os.makedirs("images", exist_ok=True)

# Load cleaned dataset
df = pd.read_csv("data/houses_clean.csv")

print("Dataset shape:", df.shape)
print(df.head())

# 1. Price distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["price"], kde=True)
plt.title("House Price Distribution")
plt.xlabel("Price")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("images/price_distribution.png")
plt.show()

# 2. Area vs Price
plt.figure(figsize=(8, 5))
sns.scatterplot(x="area", y="price", data=df)
plt.title("Area vs Price")
plt.xlabel("Area")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("images/area_vs_price.png")
plt.show()

# 3. Bedrooms vs Price
plt.figure(figsize=(8, 5))
sns.boxplot(x="bedrooms", y="price", data=df)
plt.title("Bedrooms vs Price")
plt.xlabel("Bedrooms")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("images/bedrooms_vs_price.png")
plt.show()

# 4. Furnishing Status vs Price
plt.figure(figsize=(8, 5))
sns.boxplot(x="furnishingstatus", y="price", data=df)
plt.title("Furnishing Status vs Price")
plt.xlabel("Furnishing Status")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("images/furnishing_vs_price.png")
plt.show()

# 5. Correlation heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("images/correlation_heatmap.png")
plt.show()

print("EDA completed successfully.")
print("Graphs saved inside images/ folder.")