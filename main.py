from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Mount static files (HTML, CSS, JS) from the current directory
# This will serve 'index.html' when accessing the root '/'
app.mount("/", StaticFiles(directory=".", html=True), name="static")

model_path = "heart_disease_model.pkl"

# Ensure the model is loaded only once
model = joblib.load(model_path)


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


@app.get("/api") # Changed the root API endpoint to '/api' to avoid conflict with static files
def home():
    return {"message": "Heart Disease Prediction API is running"}


@app.post("/api/predict") # Changed the predict API endpoint
def predict(data: PatientData):

    patient = pd.DataFrame([data.model_dump()])

    prediction = model.predict(patient)[0]
    probability = model.predict_proba(patient)[0][1]

    if prediction == 1:
        result = "Heart Disease Detected"
    else:
        result = "No Heart Disease Detected"

    return {
        "prediction": int(prediction),
        "result": result,
        "probability": round(
            float(probability) * 100,
            2
        )
    }
