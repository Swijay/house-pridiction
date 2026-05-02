import os
import sys
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import add_features


DATA_PATH = "data/cleaned_data.csv"
MODEL_PATH = "models/model.pkl"
EVALUATION_PATH = "outputs/final_evaluation.txt"


def evaluate_model():
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("images", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Cleaned data not found. Run src/train.py first.")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run src/train.py first.")

    df = pd.read_csv(DATA_PATH)

    df = add_features(df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    model = joblib.load(MODEL_PATH)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print("Final Model Evaluation")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    with open(EVALUATION_PATH, "w") as file:
        file.write("FINAL MODEL EVALUATION\n")
        file.write("=" * 30 + "\n")
        file.write(f"MAE  : {mae:.2f}\n")
        file.write(f"RMSE : {rmse:.2f}\n")
        file.write(f"R2   : {r2:.4f}\n")

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions)
    plt.xlabel("Actual House Price")
    plt.ylabel("Predicted House Price")
    plt.title("Actual vs Predicted House Price")
    plt.tight_layout()
    plt.savefig("images/actual_vs_predicted.png")
    plt.show()

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

    print("\nEvaluation completed successfully.")
    print(f"Evaluation saved at: {EVALUATION_PATH}")


if __name__ == "__main__":
    evaluate_model()