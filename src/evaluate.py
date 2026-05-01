import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import add_features


# Create folders
os.makedirs("images", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load cleaned dataset
df = pd.read_csv("data/houses_clean.csv")

# Add engineered features
df = add_features(df)

# Separate input and target
X = df.drop("price", axis=1)
y = df["price"]

# Same split used during training
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Load saved best model
model = joblib.load("models/house_price_model.pkl")

# Make predictions
predictions = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)


print("Final Model Evaluation")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R2   : {r2:.4f}")


# Save evaluation result
with open("outputs/final_evaluation.txt", "w") as file:
    file.write("FINAL MODEL EVALUATION\n")
    file.write("=" * 30 + "\n")
    file.write(f"MAE  : {mae:.2f}\n")
    file.write(f"RMSE : {rmse:.2f}\n")
    file.write(f"R2   : {r2:.4f}\n")


# Actual vs Predicted graph
plt.figure(figsize=(8, 6))
plt.scatter(y_test, predictions)
plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted House Price")
plt.tight_layout()
plt.savefig("images/actual_vs_predicted.png")
plt.show()


# Residual plot
residuals = y_test - predictions

plt.figure(figsize=(8, 6))
plt.scatter(predictions, residuals)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted House Price")
plt.ylabel("Residuals")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("images/residual_plot.png")
plt.show()


print("Evaluation completed successfully.")
print("Evaluation saved at: outputs/final_evaluation.txt")
print("Graphs saved inside images/ folder.")