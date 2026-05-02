import os
import sys
import joblib
import pandas as pd

sys.path.append("src")

from feature_engineering import add_features


MODEL_PATH = "models/model.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Model file not found. Please train the model first.")

    model = joblib.load(MODEL_PATH)

    return model


def predict_price(input_data: dict) -> float:
    model = load_model()

    df = pd.DataFrame([input_data])

    df = add_features(df)

    prediction = model.predict(df)[0]

    return round(float(prediction), 2)