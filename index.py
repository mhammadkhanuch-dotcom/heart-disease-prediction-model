@app.get("/api")
def home():
    return {"message": "Heart Disease Prediction API is running"}


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
