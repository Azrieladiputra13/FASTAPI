from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Inisialisasi FastAPI
app = FastAPI(title="GDP Prediction Based on Inflation, Unemployment, and GDP per Capita")

# Load model dan scaler
rf_model = joblib.load('model.pkl')  # Path ke model yang sudah di-upload
scaler = joblib.load('scaler.pkl')  # Path ke scaler yang sudah di-upload

# Skema input untuk prediksi
class EconomicData(BaseModel):
    InflationRate: float
    UnemploymentRate: float
    GDPPerCapita: float

# Fungsi preprocessing data
def preprocess_input(data: EconomicData):
    # Membuat DataFrame dari input
    df = pd.DataFrame([{
        "Inflation Rate (%)": data.InflationRate,
        "Unemployment Rate (%)": data.UnemploymentRate,
        "GDP per Capita (USD)": data.GDPPerCapita
    }])

    # Menstandarisasi data menggunakan scaler yang sudah disimpan
    df_scaled = scaler.transform(df)
    return df_scaled

@app.get("/")
def read_root():
    return {"message": "GDP Prediction API is running"}

# Endpoint prediksi untuk estimasi GDP
@app.post("/predict")
def predict_gdp(data: EconomicData):
    # Preprocessing input data
    processed_data = preprocess_input(data)
    
    # Prediksi dengan model Random Forest
    prediction = rf_model.predict(processed_data)[0]
    
    # Mengembalikan hasil prediksi
    return {
        "Inflation Rate": data.InflationRate,
        "Unemployment Rate": data.UnemploymentRate,
        "GDP per Capita": data.GDPPerCapita,
        "Predicted GDP (in billion USD)": prediction
    }
