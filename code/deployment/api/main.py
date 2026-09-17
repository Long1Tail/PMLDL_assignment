"""
FastAPI Model Serving API for Wine Quality Prediction.

Endpoints:
- GET /: Welcome and service overview
- GET /health: Health check and model readiness
- GET /metadata: Model parameters and evaluation metrics
- POST /predict: Run inference on wine features
- POST /reload: Reload model from disk (after automated pipeline retraining)
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Setup path for features module
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent.parent

for candidate_dir in [CURRENT_DIR, CURRENT_DIR.parent.parent / "models", PROJECT_ROOT]:
    if str(candidate_dir) not in sys.path:
        sys.path.insert(0, str(candidate_dir))

try:
    from features import RAW_FEATURE_COLUMNS, engineer_features
except ImportError:
    try:
        from code.models.features import RAW_FEATURE_COLUMNS, engineer_features
    except ImportError:
        # Fallback if imported from local directory
        sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        from features import RAW_FEATURE_COLUMNS, engineer_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s - %(message)s")
logger = logging.getLogger("WineModelAPI")

app = FastAPI(
    title="Wine Quality Prediction API",
    description="Production-ready FastAPI service delivering real-time predictions for wine quality.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration from environment variables
MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
METADATA_PATH = Path(os.getenv("METADATA_PATH", "models/metadata.json"))

# Global model and metadata state
_model = None
_metadata = {}
_model_loaded_at: Optional[str] = None


def load_model_from_disk():
    """Load model pipeline and metadata from disk."""
    global _model, _metadata, _model_loaded_at
    resolved_model_path = MODEL_PATH.resolve()

    if not resolved_model_path.exists():
        # Check relative fallbacks
        fallbacks = [
            Path("/app/models/model.joblib"),
            PROJECT_ROOT / "models" / "model.joblib",
            CURRENT_DIR / "model.joblib"
        ]
        for fb in fallbacks:
            if fb.exists():
                resolved_model_path = fb
                break

    if not resolved_model_path.exists():
        logger.warning("Model file not found at %s or fallback paths", resolved_model_path)
        _model = None
        return False

    try:
        _model = joblib.load(resolved_model_path)
        _model_loaded_at = datetime.utcnow().isoformat() + "Z"
        logger.info("Successfully loaded model from %s", resolved_model_path)
    except Exception as e:
        logger.error("Failed loading model from %s: %s", resolved_model_path, e)
        _model = None
        return False

    # Load metadata if present
    resolved_meta_path = METADATA_PATH.resolve()
    if not resolved_meta_path.exists():
        meta_fallbacks = [
            Path("/app/models/metadata.json"),
            PROJECT_ROOT / "models" / "metadata.json",
            CURRENT_DIR / "metadata.json"
        ]
        for fb in meta_fallbacks:
            if fb.exists():
                resolved_meta_path = fb
                break

    if resolved_meta_path.exists():
        try:
            with open(resolved_meta_path, "r", encoding="utf-8") as f:
                _metadata = json.load(f)
            logger.info("Loaded model metadata from %s", resolved_meta_path)
        except Exception as e:
            logger.warning("Failed loading metadata: %s", e)
            _metadata = {}
    else:
        _metadata = {}

    return True


# Load model at startup
@app.on_event("startup")
def startup_event():
    load_model_from_disk()


class WineFeatures(BaseModel):
    fixed_acidity: float = Field(7.4, description="Fixed acidity (g/dm³)", ge=0.0, le=25.0)
    volatile_acidity: float = Field(0.70, description="Volatile acidity (g/dm³)", ge=0.0, le=5.0)
    citric_acid: float = Field(0.00, description="Citric acid (g/dm³)", ge=0.0, le=3.0)
    residual_sugar: float = Field(1.9, description="Residual sugar (g/dm³)", ge=0.0, le=50.0)
    chlorides: float = Field(0.076, description="Chlorides (g/dm³)", ge=0.0, le=2.0)
    free_sulfur_dioxide: float = Field(11.0, description="Free sulfur dioxide (mg/dm³)", ge=0.0, le=200.0)
    total_sulfur_dioxide: float = Field(34.0, description="Total sulfur dioxide (mg/dm³)", ge=0.0, le=500.0)
    density: float = Field(0.9978, description="Density (g/cm³)", ge=0.9, le=1.1)
    ph: float = Field(3.51, description="pH value", ge=1.0, le=5.0)
    sulphates: float = Field(0.56, description="Sulphates (g/dm³)", ge=0.0, le=5.0)
    alcohol: float = Field(9.4, description="Alcohol (% vol)", ge=5.0, le=20.0)

    class Config:
        json_schema_extra = {
            "example": {
                "fixed_acidity": 7.4,
                "volatile_acidity": 0.70,
                "citric_acid": 0.00,
                "residual_sugar": 1.9,
                "chlorides": 0.076,
                "free_sulfur_dioxide": 11.0,
                "total_sulfur_dioxide": 34.0,
                "density": 0.9978,
                "ph": 3.51,
                "sulphates": 0.56,
                "alcohol": 9.4
            }
        }


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description="Binary classification (1 for Good, 0 for Standard)")
    prediction_label: str = Field(..., description="Human-readable label")
    probability_good: float = Field(..., description="Confidence probability for Good quality")
    probability_standard: float = Field(..., description="Confidence probability for Standard quality")
    status: str = "success"
    model_loaded_at: Optional[str] = None


@app.get("/", tags=["General"])
def root():
    return {
        "service": "Wine Quality Prediction API",
        "status": "online",
        "docs_url": "/docs",
        "model_loaded": _model is not None
    }


@app.get("/health", tags=["Monitoring"])
def health_check():
    """Health check verifying model readiness."""
    if _model is None:
        # Try reloading in case model was just generated
        load_model_from_disk()

    is_ready = _model is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": is_ready,
        "model_loaded_at": _model_loaded_at,
        "run_id": _metadata.get("run_id", "unknown"),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.get("/metadata", tags=["Model Info"])
def get_metadata():
    """Get model training metadata and evaluation metrics."""
    if not _metadata:
        load_model_from_disk()
    return _metadata or {"message": "No metadata available yet"}


@app.post("/reload", tags=["Management"])
def reload_model():
    """Trigger dynamic model reload from disk."""
    success = load_model_from_disk()
    if not success:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Failed to load model from disk"
        )
    return {
        "message": "Model reloaded successfully",
        "loaded_at": _model_loaded_at,
        "run_id": _metadata.get("run_id", "unknown")
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(features: WineFeatures):
    """Run real-time inference on input features."""
    global _model
    if _model is None:
        if not load_model_from_disk():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not loaded. Please ensure Stage 2 model training has completed."
            )

    try:
        # Convert Pydantic model to DataFrame matching RAW_FEATURE_COLUMNS
        input_dict = features.model_dump()
        df_raw = pd.DataFrame([input_dict])

        # Apply feature engineering
        df_fe = engineer_features(df_raw)

        # Run inference through sklearn pipeline
        pred = int(_model.predict(df_fe)[0])
        probs = _model.predict_proba(df_fe)[0]
        prob_standard = float(probs[0])
        prob_good = float(probs[1])

        label = "Good Quality (Score >= 6)" if pred == 1 else "Standard Quality (Score < 6)"

        return PredictionResponse(
            prediction=pred,
            prediction_label=label,
            probability_good=round(prob_good, 4),
            probability_standard=round(prob_standard, 4),
            status="success",
            model_loaded_at=_model_loaded_at
        )
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error executing prediction: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
