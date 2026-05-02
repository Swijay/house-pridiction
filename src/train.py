import os
import sys
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Allow imports from src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import add_features
from preprocess import preprocessor


DATA_PATH = "data/housing.csv"
CLEANED_DATA_PATH = "data/cleaned_data.csv"
MODEL_PATH = "models/model.pkl"
RESULTS_PATH = "outputs/model_results.txt"


def train_models():
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print("Shape:", df.shape)

    df = df.drop_duplicates()

    df = df.dropna(subset=["price"])

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    for col in categorical_cols:
        df[col] = df[col].fillna(df[col].mode()[0])

    df.to_csv(CLEANED_DATA_PATH, index=False)
    print(f"Cleaned data saved at: {CLEANED_DATA_PATH}")

    df = add_features(df)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree Regressor": DecisionTreeRegressor(random_state=42),
        "Random Forest Regressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42,
        ),
    }

    best_model = None
    best_model_name = ""
    best_rmse = float("inf")

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        print(f"{model_name} Results:")
        print(f"MAE  : {mae:.2f}")
        print(f"RMSE : {rmse:.2f}")
        print(f"R2   : {r2:.4f}")

        result_text = (
            f"{model_name}\n"
            f"MAE  : {mae:.2f}\n"
            f"RMSE : {rmse:.2f}\n"
            f"R2   : {r2:.4f}\n"
            f"{'-' * 40}\n"
        )

        results.append(result_text)

        if rmse < best_rmse:
            best_rmse = rmse
            best_model = pipeline
            best_model_name = model_name

    joblib.dump(best_model, MODEL_PATH)

    with open(RESULTS_PATH, "w") as file:
        file.write("HOUSE PRICE PREDICTION MODEL RESULTS\n")
        file.write("=" * 45 + "\n\n")

        for result in results:
            file.write(result)

        file.write(f"\nBest Model: {best_model_name}\n")
        file.write(f"Best RMSE : {best_rmse:.2f}\n")

    print("\nTraining completed successfully.")
    print(f"Best Model: {best_model_name}")
    print(f"Model saved at: {MODEL_PATH}")
    print(f"Results saved at: {RESULTS_PATH}")


if __name__ == "__main__":
    train_models()