# 🏡 House Price Prediction System

<p align="center">
  <strong><span style="color:#2d6cdf">Predict housing prices with a full ML pipeline, API, and dashboard.</span></strong>
</p>

---

## ✨ Project Overview

This repository contains a complete end-to-end machine learning solution for predicting house prices in India. It includes:

- ✅ Data cleaning and preprocessing
- ✅ Feature engineering
- ✅ Model training and evaluation
- ✅ FastAPI backend for predictions
- ✅ Streamlit dashboard for interactive input
- ✅ Sample prediction script
- ✅ Clean, modular project structure

---

## 🎯 What It Solves

Estimating house prices manually is difficult because property values depend on many variables such as:

- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `parking`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `prefarea`
- `furnishingstatus`

This project uses regression models to learn from historical housing data and predict future prices with a reusable production-ready pipeline.

---

## 🧠 Project Architecture

1. **Data ingestion and cleaning**
   - Raw housing data: `data/housing.csv`
   - Cleaned output: `data/cleaned_data.csv`
2. **Feature engineering**
   - Created derived features such as `total_rooms`, `area_per_bedroom`, `area_per_bathroom`, and `has_parking`
3. **Preprocessing pipeline**
   - Numeric scaling with `StandardScaler`
   - Categorical encoding with `OneHotEncoder`
4. **Model training**
   - Linear Regression
   - Decision Tree Regressor
   - Random Forest Regressor
   - Best model saved to `models/model.pkl`
5. **API + dashboard**
   - FastAPI endpoint in `api/main.py`
   - Streamlit UI in `dashboard/dashboard.py`

---

## 📁 Folder Structure

- `api/` — FastAPI application and request schemas
- `dashboard/` — Streamlit dashboard interface
- `data/` — raw and cleaned housing datasets
- `models/` — saved trained model file
- `outputs/` — evaluation metrics and model results
- `src/` — core ML pipeline, preprocessing, feature engineering, and training
- `main.py` — sample prediction script

---

## 🚀 How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python src/train.py
```

This script will:
- load `data/housing.csv`
- clean and impute missing values
- perform feature engineering
- train multiple regression models
- save the best model to `models/model.pkl`
- write evaluation metrics to `outputs/model_results.txt`

### 3. Start the API

```bash
uvicorn api.main:app --reload
```

Then open:

- `http://127.0.0.1:8000/docs` for Swagger UI

### 4. Run the dashboard

```bash
streamlit run dashboard/dashboard.py
```

The dashboard sends requests to the API and displays price predictions instantly.

---

## 🧪 Prediction Options

### Sample prediction script

Use this command to test a default house example:

```bash
python main.py
```

### API request

Send a POST request to `/predict` with JSON body:

```json
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
  "furnishingstatus": "semi-furnished"
}
```

Response:

```json
{
  "predicted_price": 1234567.89
}
```

---

## 🧩 Important Files

- `src/train.py` — trains models and selects the best one
- `src/pipeline.py` — sample inference script using saved model
- `src/preprocess.py` — preprocessing pipeline for numeric and categorical features
- `src/feature_engineering.py` — derived feature generation
- `api/main.py` — FastAPI server endpoints
- `api/schemas.py` — request and response models
- `api/utils.py` — load model and predict function
- `dashboard/dashboard.py` — Streamlit UI for prediction input

---

## 📊 Evaluation & Outputs

After training, the model evaluation summary is saved at:

- `outputs/model_results.txt`

It contains metrics for each model including:
- `MAE`
- `RMSE`
- `R2`

The best-performing model is automatically persisted to `models/model.pkl`.

---

## 🧾 Dataset Columns

The dataset includes these columns:

- `price` (target)
- `area`
- `bedrooms`
- `bathrooms`
- `stories`
- `mainroad`
- `guestroom`
- `basement`
- `hotwaterheating`
- `airconditioning`
- `parking`
- `prefarea`
- `furnishingstatus`

---

## 📌 Notes

- Ensure the FastAPI server is running before using the Streamlit dashboard.
- The project uses a production-style preprocessing pipeline so the same data transformations apply during training and prediction.
- You can extend this project by adding more models, improving feature engineering, or deploying as a cloud service.

---

## 🙌 Credits

Built with Python, scikit-learn, FastAPI, Streamlit, and pandas for a modern house price prediction experience.
