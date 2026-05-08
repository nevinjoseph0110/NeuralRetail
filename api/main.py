from fastapi import FastAPI
from pydantic import BaseModel

import pandas as pd
import pickle

app = FastAPI()

# Load saved model
with open("models/churn_model.pkl", "rb") as file:
    model = pickle.load(file)

# Input structure
class CustomerData(BaseModel):
    Recency: int
    Frequency: int
    Monetary: float

# Home route
@app.get("/")
def home():
    return {"message": "NeuralRetail API Running"}

# Prediction route
@app.post("/predict_churn")
def predict(data: CustomerData):

    sample = pd.DataFrame([{
        "Recency": data.Recency,
        "Frequency": data.Frequency,
        "Monetary": data.Monetary
    }])

    prediction = model.predict(sample)[0]

    result = "High Risk" if prediction == 1 else "Low Risk"

    return {
        "prediction": result
    }
