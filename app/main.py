# app/main.py
import os
import logging
from typing import Optional

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# -------- Config --------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/spam_pipeline.joblib")
MODEL_VERSION = "v1.0"

# -------- Logging --------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("spam-api")

# -------- FastAPI --------
app = FastAPI(title="Spam Detection API", version=MODEL_VERSION)

# -------- Request Schema --------
class PredictRequest(BaseModel):
    text: str = Field(..., example="Congratulations! You won a prize!")

    @validator("text")
    def check_empty(cls, v):
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v

# -------- Response Schema --------
class PredictResponse(BaseModel):
    prediction: int
    probability: float
    model_version: str

# -------- Load Model --------
def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        logger.error(f"Model not found at {MODEL_PATH}")
        raise FileNotFoundError("Model file missing.")
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)

try:
    PIPELINE = load_pipeline()
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.exception("Failed to load model.")
    PIPELINE = None

# -------- Routes --------
@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": PIPELINE is not None}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    prob = float(PIPELINE.predict_proba([req.text])[0][1])
    pred = int(prob >= 0.5)

    return PredictResponse(
        prediction=pred,
        probability=round(prob, 4),
        model_version=MODEL_VERSION
    )
