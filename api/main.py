from fastapi import FastAPI
from api.schemas import HouseInput, PredictionOutput
from api.utils import predict_price


app = FastAPI(
    title="House Price Prediction API",
    description="FastAPI backend for predicting house prices using a trained regression model.",
    version="1.0.0",
)


@app.get("/")
def home():
    return {
        "message": "House Price Prediction API is running successfully."
    }


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: HouseInput):
    prediction = predict_price(input_data.dict())

    return {
        "predicted_price": prediction
    }