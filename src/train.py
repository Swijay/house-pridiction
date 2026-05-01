import pandas as pd
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from features import add_features
from pipeline import preprocessor


# Create required folders if they do not exist
os.makedirs("models", exist_ok=True)
os.makedirs("outputs", exist_ok=True)


# Load cleaned dataset
df = pd.read_csv("data/houses_clean.csv")

# Add engineered features
df = add_features(df)

# Separate input features and target
X = df.drop("price", axis=1)
y = df["price"]

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# Define regression models
models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree Regressor": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest Regressor": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}


best_model = None
best_model_name = ""
best_rmse = float("inf")

results = []


# Train and evaluate each model
for model_name, model in models.items():

    print(f"\nTraining {model_name}...")

    # Complete ML pipeline: preprocessing + model
    ml_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train model
    ml_pipeline.fit(X_train, y_train)

    # Predict test data
    predictions = ml_pipeline.predict(X_test)

    # Evaluation metrics
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    print(f"{model_name} Results:")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    # Store result as text
    result_text = (
        f"{model_name}\n"
        f"MAE  : {mae:.2f}\n"
        f"RMSE : {rmse:.2f}\n"
        f"R2   : {r2:.4f}\n"
        f"{'-' * 40}\n"
    )

    results.append(result_text)

    # Select best model based on lowest RMSE
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = ml_pipeline
        best_model_name = model_name


# Save best model
joblib.dump(best_model, "models/house_price_model.pkl")

# Save model comparison results
with open("outputs/model_results.txt", "w") as file:
    file.write("HOUSE PRICE PREDICTION MODEL RESULTS\n")
    file.write("=" * 45 + "\n\n")

    for result in results:
        file.write(result)

    file.write(f"\nBest Model: {best_model_name}\n")
    file.write(f"Best RMSE : {best_rmse:.2f}\n")


print("\nTraining completed successfully.")
print(f"Best Model: {best_model_name}")
print("Model saved at: models/house_price_model.pkl")
print("Results saved at: outputs/model_results.txt")