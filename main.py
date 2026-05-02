import os
import sys
import joblib
import pandas as pd

sys.path.append("src")

from feature_engineering import add_features


MODEL_PATH = "models/model.pkl"


def predict_sample_house():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model not found. Run python src/train.py first.")

    model = joblib.load(MODEL_PATH)

    sample_house = pd.DataFrame(
        [
            {
                "area": 7500,
                "bedrooms": 4,
                "bathrooms": 3,
                "stories": 2,
                "mainroad": "yes",
                "guestroom": "yes",
                "basement": "no",
                "hotwaterheating": "no",
                "airconditioning": "yes",
                "parking": 2,
                "prefarea": "yes",
                "furnishingstatus": "semi-furnished",
            }
        ]
    )

    sample_house = add_features(sample_house)

    predicted_price = model.predict(sample_house)[0]

    print("House Price Prediction")
    print("-" * 30)
    print("Input House Details:")
    print(sample_house)

    print("\nPredicted House Price:")
    print(round(predicted_price, 2))


if __name__ == "__main__":
    predict_sample_house()