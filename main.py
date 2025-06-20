from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] to allow all origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
# Charger ton modèle ML sauvegardé
model = joblib.load('insurance_rf_model.pkl')


# Définir le format des données d'entrée
class InsuranceData(BaseModel):
    age: int
    sex: int
    bmi: float
    children: int
    smoker: int
    region_northwest: int
    region_southeast: int
    region_southwest: int


# Définir l'API route POST pour faire des prédictions
@app.post("/predict")
def predict(data: InsuranceData):
    features = np.array([[data.age, data.sex, data.bmi, data.children,
                          data.smoker, data.region_northwest,
                          data.region_southeast, data.region_southwest]])

    prediction = model.predict(features)
    return {"charges": prediction[0]}
