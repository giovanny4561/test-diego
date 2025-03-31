from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
from typing import List

# Modelo de entrada
class Entrada(BaseModel):
    edad: int
    ingresos: float

app = FastAPI()

# Carga el modelo
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.post("/predict")
def predict(entradas: List[Entrada]):
    X = np.array([[e.edad, e.ingresos] for e in entradas])
    pred = model.predict(X).tolist()
    return {"predicciones": pred}