"""
FastAPI Server for Gold Price Forecasting
Provides REST endpoints for predictions, health checks, and model management
"""

import os
import sys
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import yaml
import uvicorn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetch import fetch_all_data, DatabaseManager, GoldDataFetcher
from src.models import ModelFactory, MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader"""

    @staticmethod
    def load() -> Dict[str, Any]:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


# Initialize FastAPI
app = FastAPI(
    title="Gold Price Forecasting API",
    description="Real-time gold price prediction using Prophet, ARIMA, and LSTM models",
    version="1.0.0",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Configuration
config = Config.load()
db_manager = DatabaseManager(config)

# Models storage
models = {}
models_loaded = False


# Pydantic models
class PredictionRequest(BaseModel):
    model_type: str = Field(
        default="all", description="Model type: prophet, arima, lstm, or all"
    )
    horizon: int = Field(default=1, ge=1, le=30, description="Forecast horizon in days")
    use_live_data: bool = Field(
        default=True, description="Use live data for prediction"
    )


class PredictionResponse(BaseModel):
    model_type: str
    predictions: List[float]
    lower_bound: Optional[List[float]] = None
    upper_bound: Optional[List[float]] = None
    horizon: int
    timestamp: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    available_models: List[str]
    database_connected: bool
    latest_price: Optional[float] = None


class MetricsResponse(BaseModel):
    model_type: str
    mae: float
    rmse: float
    mape: float


class RetrainRequest(BaseModel):
    model_types: Optional[List[str]] = None


def load_models() -> Dict[str, Any]:
    """Load trained models from disk"""
    global models, models_loaded

    if models_loaded:
        return models

    models_dir = Path(__file__).parent.parent / "models"

    for model_type in ["prophet", "arima", "lstm"]:
        model_path = models_dir / f"{model_type}_model.pkl"

        if model_path.exists():
            try:
                with open(model_path, "rb") as f:
                    models[model_type] = pickle.load(f)
                logger.info(f"Loaded {model_type} model")
            except Exception as e:
                logger.error(f"Failed to load {model_type}: {e}")
        else:
            logger.warning(f"Model file not found: {model_path}")

    models_loaded = True
    return models


def get_latest_data(days: int = 60) -> pd.DataFrame:
    """Get latest gold price data"""
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=days + 30)).strftime("%Y-%m-%d")

    # Try database first
    try:
        db_manager.connect()
        df = db_manager.load_gold_prices(start_date, end_date)
        if not df.empty:
            return df
    except Exception as e:
        logger.warning(f"Database fetch failed: {e}")

    # Fallback to API
    try:
        fetcher = GoldDataFetcher(config)
        df = fetcher.fetch_from_yfinance(start_date, end_date)
        return df
    except Exception as e:
        logger.error(f"Data fetch failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch data")


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("Starting Gold Price Forecasting API...")

    # Load models
    load_models()

    # Try to connect to database
    try:
        db_manager.connect()
        db_manager.create_tables()
    except Exception as e:
        logger.warning(f"Database connection failed: {e}")

    logger.info("API ready")


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "Gold Price Forecasting API",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    # Check database
    db_connected = False
    latest_price = None

    try:
        db_manager.connect()
        df = db_manager.load_gold_prices(
            (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d"),
            datetime.now().strftime("%Y-%m-%d"),
        )
        if not df.empty:
            db_connected = True
            latest_price = float(df["gold_price"].iloc[-1])
    except Exception as e:
        logger.warning(f"Health check DB error: {e}")

    return HealthResponse(
        status="healthy" if models_loaded else "degraded",
        models_loaded=models_loaded,
        available_models=list(models.keys()),
        database_connected=db_connected,
        latest_price=latest_price,
    )


@app.post("/predict", response_model=List[PredictionResponse], tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Generate price predictions

    Args:
        request: Prediction request with model_type, horizon, etc.

    Returns:
        List of predictions from requested models
    """
    # Load models if not loaded
    load_models()

    # Get latest data
    data = get_latest_data()

    if data.empty or "gold_price" not in data.columns:
        raise HTTPException(status_code=400, detail="No data available")

    responses = []

    # Determine which models to use
    if request.model_type == "all":
        model_types = list(models.keys())
    else:
        model_types = [request.model_type]

    for model_type in model_types:
        if model_type not in models:
            continue

        try:
            model = models[model_type]

            # Handle both old format (direct model) and new format (dict with 'model' key)
            if isinstance(model, dict) and "model" in model:
                actual_model = model["model"]
            else:
                actual_model = model

            if model_type == "lstm":
                result = actual_model.predict_with_confidence(data, request.horizon)
            elif hasattr(actual_model, "predict_with_confidence"):
                # ARIMAModel wrapper
                result = actual_model.predict_with_confidence(request.horizon)
            else:
                # Raw pmdarima model
                forecasts, conf_int = actual_model.predict(
                    n_periods=request.horizon, return_conf_int=True
                )
                result = {
                    "predictions": forecasts.tolist()
                    if hasattr(forecasts, "tolist")
                    else list(forecasts),
                    "lower_bound": conf_int[:, 0].tolist()
                    if hasattr(conf_int, "tolist")
                    else list(conf_int[:, 0]),
                    "upper_bound": conf_int[:, 1].tolist()
                    if hasattr(conf_int, "tolist")
                    else list(conf_int[:, 1]),
                }

            responses.append(
                PredictionResponse(
                    model_type=model_type,
                    predictions=result.get("predictions", []),
                    lower_bound=result.get("lower_bound"),
                    upper_bound=result.get("upper_bound"),
                    horizon=request.horizon,
                    timestamp=datetime.now().isoformat(),
                )
            )

        except Exception as e:
            logger.error(f"Prediction error for {model_type}: {e}")
            raise HTTPException(
                status_code=500, detail=f"Prediction failed for {model_type}: {str(e)}"
            )

    if not responses:
        raise HTTPException(status_code=404, detail="No models available")

    return responses


@app.get("/predict/latest", response_model=Dict[str, Any], tags=["Predictions"])
async def predict_latest(model_type: str = "prophet", horizon: int = 1):
    """
    Quick prediction endpoint

    Args:
        model_type: Model to use
        horizon: Forecast horizon

    Returns:
        Predictions
    """
    load_models()

    if model_type not in models:
        raise HTTPException(status_code=404, detail=f"Model {model_type} not found")

    data = get_latest_data()

    try:
        model = models[model_type]

        if model_type == "lstm":
            predictions = model.predict_multiple(data, horizon)
        else:
            predictions = model.predict(horizon)

        return {
            "model_type": model_type,
            "predictions": predictions.tolist()
            if hasattr(predictions, "tolist")
            else list(predictions),
            "horizon": horizon,
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics", response_model=List[MetricsResponse], tags=["Metrics"])
async def get_metrics():
    """
    Get model performance metrics
    """
    results_path = Path(__file__).parent.parent / "training_results.json"

    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No training results found")

    with open(results_path, "r") as f:
        results = json.load(f)

    metrics = results.get("metrics", {})

    return [
        MetricsResponse(
            model_type=model_type,
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
        )
        for model_type, metrics in metrics.items()
    ]


@app.post("/retrain", tags=["Model Management"])
async def retrain_models(request: RetrainRequest, background_tasks: BackgroundTasks):
    """
    Trigger model retraining

    Args:
        request: Retrain request with model types

    Returns:
        Status message
    """
    from src.train import run_full_pipeline

    def train_task():
        try:
            run_full_pipeline(
                model_types=request.model_types or ["prophet", "arima", "lstm"]
            )
            logger.info("Retraining complete")
        except Exception as e:
            logger.error(f"Retraining failed: {e}")

    background_tasks.add_task(train_task)

    return {"message": "Retraining started in background"}


@app.get("/data/latest", tags=["Data"])
async def get_latest_prices(days: int = 30):
    """Get latest price data"""
    data = get_latest_data(days + 30)

    if data.empty:
        raise HTTPException(status_code=404, detail="No data available")

    data = data.tail(days)

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in data.index],
        "prices": data["gold_price"].tolist(),
        "volumes": data.get("volume", [None] * len(data)).tolist(),
    }


import json


if __name__ == "__main__":
    import sys
    import os

    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    server_config = config.get("server", {})
    uvicorn.run(
        "src.api:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        workers=1,  # Use single worker for development
        reload=False,
    )
