from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import pandas as pd
import uvicorn

app = FastAPI(title="Penguin ML Inference API", version="1.0")

# Load model once at startup
try:
    model = joblib.load("penguin_classifier.pkl")
except Exception as e:
    model = None
    load_error = str(e)


class PredictIn(BaseModel):
    bill_length_mm: float = Field(..., gt=0)
    bill_depth_mm: float = Field(..., gt=0)
    flipper_length_mm: float = Field(..., gt=0)
    body_mass_g: float = Field(..., gt=0)


class PredictOut(BaseModel):
    predicted_species: str


@app.get("/health")
def health():
    if model is None:
        return {"status": "error", "detail": load_error}
    return {"status": "ok", "model_loaded": True}


@app.post("/predict", response_model=PredictOut)
def predict(x: PredictIn):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    X = pd.DataFrame([x.model_dump()])

    try:
        pred = model.predict(X)[0]
        return {"predicted_species": str(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


if __name__ == "__main__":
    uvicorn.run(
        "penguin_classifier:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
    )