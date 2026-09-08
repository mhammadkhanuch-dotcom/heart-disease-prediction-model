from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "heart_disease_model.pkl"
)


class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int


@app.get("/api")
def home():
    return {
        "message": "Heart Disease Prediction API is running"
    }


@app.post("/api/predict")
def predict(data: PatientData):

    patient = pd.DataFrame([
        data.model_dump()
    ])

    prediction = int(
        model.predict(patient)[0]
    )

    probability = float(
        model.predict_proba(patient)[0][1]
    ) * 100

    if prediction == 1:
        result = "Higher likelihood of heart disease"
    else:
        result = "Lower likelihood of heart disease"

    return {
        "prediction": prediction,
        "result": result,
        "probability": round(probability, 2)
    }
