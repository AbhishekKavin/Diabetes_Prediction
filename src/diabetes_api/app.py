from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
import mlflow
import os

from diabetes_api.schemas import PatientFeatures, PredictionResponse

ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    model_uri = os.getenv("MODEL_URI", "models:/diabetes_progression_model@champion")

    if model_uri.startswith("models:/"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    ml_models["champion"] = mlflow.sklearn.load_model(model_uri)
    yield
    ml_models.clear()

app = FastAPI(title = "Diabetes Progression API", lifespan = lifespan)

@app.get("/health")
async def health_check():
    if "champion" not in ml_models or ml_models["champion"] is None:
        raise HTTPException(status_code = 503, detail = "model artifact unavailable")
    return {"status":"up", "model_loaded": True}

@app.post("/predict", response_model = PredictionResponse)
async def predict(patient: PatientFeatures):
    if "champion" not in ml_models:
        raise HTTPException(status_code = 503, detail = "Model artifact unavailable.")

    X_new = patient.to_numpy()
    score = float(ml_models["champion"].predict(X_new)[0])

    return PredictionResponse(disease_progression_score = score)