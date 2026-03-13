"""
FastAPI Server for Gold Price Forecasting
Provides REST endpoints for predictions, health checks, and model management
"""

import os
import sys
import logging
import pickle
import json
import time
import threading
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict, Any, Literal

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Query
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
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
            config = yaml.safe_load(f) or {}

        api_config = config.setdefault("api", {})
        database_config = config.setdefault("database", {})
        server_config = config.setdefault("server", {})

        metalprice_key = os.environ.get("METALPRICEAPI_KEY")
        if metalprice_key:
            api_config["metalpriceapi_key"] = metalprice_key

        database_env_map = {
            "DATABASE_HOST": "host",
            "DATABASE_PORT": "port",
            "DATABASE_NAME": "name",
            "DATABASE_USER": "user",
            "DATABASE_PASSWORD": "password",
        }
        for env_name, config_key in database_env_map.items():
            env_value = os.environ.get(env_name)
            if env_value is not None and env_value != "":
                database_config[config_key] = (
                    int(env_value) if config_key == "port" else env_value
                )

        host_override = os.environ.get("SERVER_HOST")
        if host_override:
            server_config["host"] = host_override

        port_override = os.environ.get("SERVER_PORT")
        if port_override:
            server_config["port"] = int(port_override)

        return config


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

# Security and rate-limit settings
API_TOKEN = os.environ.get("API_TOKEN", "")
RATE_LIMIT_REQUESTS = int(os.environ.get("API_RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.environ.get("API_RATE_LIMIT_WINDOW_SECONDS", "60"))
AUTH_EXEMPT_PATHS = {"/", "/health", "/docs", "/openapi.json", "/redoc"}

security = HTTPBearer(auto_error=False)
_rate_limit_lock = threading.Lock()
_rate_limit_store: Dict[str, deque] = {}


def _get_client_id(request: Request) -> str:
    """Get a stable client identifier for rate limiting."""
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()

    if request.client and request.client.host:
        return request.client.host

    return "unknown"


def enforce_rate_limit(
    client_id: str,
    max_requests: Optional[int] = None,
    window_seconds: Optional[int] = None,
) -> None:
    """Simple in-memory rate limiter."""
    req_limit = max_requests if max_requests is not None else RATE_LIMIT_REQUESTS
    time_window = (
        window_seconds if window_seconds is not None else RATE_LIMIT_WINDOW_SECONDS
    )
    if req_limit <= 0 or time_window <= 0:
        return

    current_time = time.time()

    with _rate_limit_lock:
        timestamps = _rate_limit_store.setdefault(client_id, deque())
        while timestamps and timestamps[0] <= current_time - time_window:
            timestamps.popleft()

        if len(timestamps) >= req_limit:
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please retry shortly.",
            )

        timestamps.append(current_time)


def require_api_token(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> None:
    """Require bearer token when API_TOKEN is configured."""
    if request.url.path in AUTH_EXEMPT_PATHS:
        return

    if not API_TOKEN:
        return

    if not credentials or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Missing bearer token")

    if credentials.credentials != API_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid bearer token")


# Pydantic models
class PredictionRequest(BaseModel):
    model_type: Literal["prophet", "arima", "lstm", "all"] = Field(
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


class FutureForecastResponse(BaseModel):
    model_type: str
    horizon: int
    dates: List[str]
    predictions: List[float]
    source: str = "trained_results"


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return consistent validation error payloads."""
    return JSONResponse(
        status_code=422,
        content={
            "error": "validation_error",
            "path": request.url.path,
            "details": exc.errors(),
        },
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Return a controlled payload for unexpected failures."""
    logger.exception("Unhandled server error on %s: %s", request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "path": request.url.path,
        },
    )


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


def load_training_results_file() -> Dict[str, Any]:
    """Load persisted training results JSON from disk."""
    results_path = Path(__file__).parent.parent / "training_results.json"
    if not results_path.exists():
        raise HTTPException(status_code=404, detail="No training results found")

    with open(results_path, "r") as f:
        return json.load(f)


def normalize_future_forecasts(
    forecasts_payload: Any,
    model_type: Optional[str] = None,
    horizon: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Normalize persisted future forecast payload into API response records."""
    normalized: List[Dict[str, Any]] = []

    if isinstance(forecasts_payload, dict):
        for name, values in forecasts_payload.items():
            if not isinstance(values, dict):
                continue
            if model_type and name != model_type:
                continue

            dates = values.get("dates", [])
            predictions = values.get("predictions", [])
            if not isinstance(dates, list) or not isinstance(predictions, list):
                continue

            if horizon is not None:
                dates = dates[:horizon]
                predictions = predictions[:horizon]

            if len(dates) == 0 or len(predictions) == 0:
                continue

            clipped_horizon = min(len(dates), len(predictions))
            try:
                normalized.append(
                    {
                        "model_type": name,
                        "horizon": clipped_horizon,
                        "dates": [str(item) for item in dates[:clipped_horizon]],
                        "predictions": [
                            float(item) for item in predictions[:clipped_horizon]
                        ],
                        "source": "trained_results",
                    }
                )
            except (TypeError, ValueError):
                continue

    return normalized


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
async def predict(
    request: PredictionRequest,
    raw_request: Request,
    _auth: None = Depends(require_api_token),
):
    """
    Generate price predictions

    Args:
        request: Prediction request with model_type, horizon, etc.

    Returns:
        List of predictions from requested models
    """
    enforce_rate_limit(_get_client_id(raw_request))

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
async def predict_latest(
    raw_request: Request,
    model_type: Literal["prophet", "arima", "lstm"] = "prophet",
    horizon: int = 1,
    _auth: None = Depends(require_api_token),
):
    """
    Quick prediction endpoint

    Args:
        model_type: Model to use
        horizon: Forecast horizon

    Returns:
        Predictions
    """
    enforce_rate_limit(_get_client_id(raw_request))

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
    results = load_training_results_file()

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


@app.get(
    "/forecasts/trained",
    response_model=List[FutureForecastResponse],
    tags=["Predictions"],
)
async def get_trained_forecasts(
    raw_request: Request,
    model_type: Literal["all", "prophet", "arima", "lstm"] = "all",
    horizon: Optional[int] = Query(default=None, ge=1, le=365),
    _auth: None = Depends(require_api_token),
):
    """Get latest persisted future forecasts generated during model training."""
    enforce_rate_limit(_get_client_id(raw_request))

    results = load_training_results_file()
    forecasts_payload = results.get("future_forecasts", {})

    selected_model = None if model_type == "all" else model_type
    forecasts = normalize_future_forecasts(
        forecasts_payload, model_type=selected_model, horizon=horizon
    )

    if not forecasts:
        raise HTTPException(
            status_code=404,
            detail="No trained future forecasts available. Run training first.",
        )

    return [FutureForecastResponse(**item) for item in forecasts]


@app.post("/retrain", tags=["Model Management"])
async def retrain_models(
    request: RetrainRequest,
    background_tasks: BackgroundTasks,
    raw_request: Request,
    _auth: None = Depends(require_api_token),
):
    """
    Trigger model retraining

    Args:
        request: Retrain request with model types

    Returns:
        Status message
    """
    enforce_rate_limit(_get_client_id(raw_request), max_requests=5, window_seconds=60)

    from src.train import run_full_pipeline

    if request.model_types:
        invalid_models = {
            model_name
            for model_name in request.model_types
            if model_name not in {"prophet", "arima", "lstm"}
        }
        if invalid_models:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid model types: {sorted(invalid_models)}",
            )

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
