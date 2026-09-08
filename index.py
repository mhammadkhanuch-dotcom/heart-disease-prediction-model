from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Heart Disease Prediction API")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_disease_model.pkl"
model = joblib.load(MODEL_PATH)


class PatientData(BaseModel):
    age: int = Field(..., ge=18, le=100)
    sex: int = Field(..., ge=0, le=1)
    cp: int = Field(..., ge=0, le=3)
    trestbps: int = Field(..., ge=70, le=250)
    chol: int = Field(..., ge=100, le=700)
    fbs: int = Field(..., ge=0, le=1)
    restecg: int = Field(..., ge=0, le=2)
    thalach: int = Field(..., ge=60, le=250)
    exang: int = Field(..., ge=0, le=1)
    oldpeak: float = Field(..., ge=0, le=10)
    slope: int = Field(..., ge=0, le=2)
    ca: int = Field(..., ge=0, le=4)
    thal: int = Field(..., ge=0, le=3)


@app.get("/api")
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/predict")
def predict(data: PatientData):
    patient = pd.DataFrame([data.model_dump()])

    prediction = int(model.predict(patient)[0])
    probability = float(model.predict_proba(patient)[0][1]) * 100

    result = (
        "Higher likelihood of heart disease"
        if prediction == 1
        else "Lower likelihood of heart disease"
    )

    return {
        "prediction": prediction,
        "result": result,
        "probability": round(probability, 2)
    }
