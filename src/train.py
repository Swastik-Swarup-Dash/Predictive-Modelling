"""
Training Pipeline for Gold Price Forecasting Models
Orchestrates data fetching, model training, evaluation, and model persistence
"""

import os
import sys
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import pandas as pd
import numpy as np
import yaml

try:
    import mlflow  # type: ignore[import-not-found]
except Exception:
    mlflow = None

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_fetch import fetch_all_data, DataPreprocessor, DatabaseManager
from src.models import (
    ModelFactory,
    train_and_evaluate_models,
    MetricsCalculator,
    ProphetModel,
    ARIMAModel,
    LSTMModel,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader"""

    @staticmethod
    def load() -> Dict[str, Any]:
        """Load configuration from config.yaml with environment overrides"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f) or {}

        api_config = config.setdefault("api", {})
        database_config = config.setdefault("database", {})

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

        return config


class ModelTrainer:
    """
    Main training pipeline orchestrator
    Handles data preparation, model training, evaluation, and persistence
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.load()
        self.mlflow_uri = os.environ.get("MLFLOW_TRACKING_URI", "mlruns")
        self.models_dir = Path(__file__).parent.parent / "models"
        self.models_dir.mkdir(exist_ok=True)

        # Initialize components
        self.preprocessor = DataPreprocessor()
        self.db_manager = DatabaseManager(self.config)

        # Results storage
        self.trained_models = {}
        self.metrics = {}

    def setup_mlflow(self):
        """Setup MLflow tracking"""
        if mlflow is None:
            logger.warning("MLflow not installed; skipping MLflow setup")
            return

        mlflow.set_tracking_uri(self.mlflow_uri)
        mlflow.set_experiment("gold-price-forecasting")

        logger.info(f"MLflow tracking: {self.mlflow_uri}")

    def prepare_data(
        self,
        start_date: str = "2004-01-01",
        end_date: Optional[str] = None,
        csv_path: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch and prepare training data

        Args:
            start_date: Start date for data
            end_date: End date for data
            csv_path: Path to historical CSV

        Returns:
            Prepared DataFrame
        """
        logger.info("Preparing training data...")

        # Fetch data
        df = fetch_all_data(start_date, end_date, csv_path)
        self._validate_training_data(df, stage="raw")

        if not isinstance(df.index, pd.DatetimeIndex):
            if "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                df = df.dropna(subset=["date"]).set_index("date")
            else:
                raise ValueError("Input data must have a DatetimeIndex or a 'date' column")

        # Add time features
        df = self.preprocessor.add_time_features(df)

        # Drop columns that are entirely NaN
        df = df.dropna(axis=1, how="all")

        # Remove rows with NaN in core columns only (not feature-engineered columns)
        df = df.dropna(subset=["gold_price"])

        # Handle remaining missing values in feature columns
        df = self.preprocessor.handle_missing_values(df)

        self._validate_training_data(df, stage="processed")

        logger.info(f"Data prepared: {df.shape[0]} samples, {df.shape[1]} features")

        return df

    def _validate_training_data(self, df: pd.DataFrame, stage: str) -> None:
        """Validate data quality before model training."""
        if df is None or df.empty:
            raise ValueError(f"No data available at '{stage}' stage")

        if "gold_price" not in df.columns:
            raise ValueError("Required column 'gold_price' not found in dataset")

        train_config = self.config.get("training", {})
        min_samples = int(train_config.get("min_samples", 365))
        if len(df) < min_samples:
            raise ValueError(
                f"Insufficient samples for training: {len(df)} < {min_samples}"
            )

        if df["gold_price"].isna().all():
            raise ValueError("Column 'gold_price' contains only missing values")

        missing_ratio = float(df["gold_price"].isna().mean())
        max_missing_ratio = float(train_config.get("max_missing_ratio", 0.05))
        if missing_ratio > max_missing_ratio:
            raise ValueError(
                f"Too many missing values in 'gold_price': {missing_ratio:.2%} > {max_missing_ratio:.2%}"
            )

    def split_data(
        self, df: pd.DataFrame, include_validation: bool = False
    ) -> Tuple[pd.DataFrame, ...]:
        """Split data into train/test or train/validation/test sets."""
        self._validate_training_data(df, stage="split")

        train_config = self.config.get("training", {})
        test_size = float(train_config.get("test_size", 0.2))
        validation_size = float(train_config.get("validation_size", 0.1))

        if not 0 < test_size < 1:
            raise ValueError("training.test_size must be between 0 and 1")
        if not 0 <= validation_size < 1:
            raise ValueError("training.validation_size must be in [0, 1)")

        if include_validation and (test_size + validation_size >= 1):
            raise ValueError(
                "training.test_size + training.validation_size must be < 1"
            )

        n_rows = len(df)
        test_rows = max(1, int(round(n_rows * test_size)))
        split_idx = n_rows - test_rows

        train_df = df.iloc[:split_idx]
        test_df = df.iloc[split_idx:]

        if include_validation:
            val_rows = max(1, int(round(n_rows * validation_size)))
            train_end = n_rows - (test_rows + val_rows)
            train_end = max(1, train_end)
            val_end = train_end + val_rows

            train_df = df.iloc[:train_end]
            val_df = df.iloc[train_end:val_end]
            test_df = df.iloc[val_end:]

            if train_df.empty or val_df.empty or test_df.empty:
                raise ValueError(
                    "Data split produced an empty partition. Adjust test/validation sizes."
                )

            logger.info(
                f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}"
            )
            return train_df, val_df, test_df

        if train_df.empty or test_df.empty:
            raise ValueError("Data split produced an empty train or test partition")

        logger.info(f"Train: {len(train_df)}, Test: {len(test_df)}")

        return train_df, test_df

    def train_model(
        self,
        model_type: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        full_history_df: Optional[pd.DataFrame] = None,
        forecast_horizon: Optional[int] = None,
        use_mlflow: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a single model

        Args:
            model_type: Type of model to train
            train_df: Training DataFrame
            test_df: Test DataFrame
            use_mlflow: Whether to use MLflow tracking

        Returns:
            Dictionary with model, metrics, and predictions
        """
        logger.info(f"\n=== Training {model_type} ===")

        # Skip mlflow for now to avoid segfault
        use_mlflow = False
        forecast_horizon = forecast_horizon or int(
            self.config.get("forecast", {}).get("horizon", 30)
        )

        try:
            # Create model
            model = ModelFactory.create(model_type, self.config)

            # Fit model
            model.fit(train_df)

            # Generate predictions
            if model_type == "lstm":
                predictions = model.predict_multiple(train_df, len(test_df))
            else:
                predictions = model.predict(len(test_df))

            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # Ensure same length
            predictions = predictions[: len(test_df)]
            actual = test_df["gold_price"].values[: len(predictions)]

            # Calculate metrics
            metrics = MetricsCalculator.calculate_all(actual, predictions)

            logger.info(
                f"{model_type} Metrics: "
                f"MAE={metrics['mae']:.2f}, "
                f"RMSE={metrics['rmse']:.2f}, "
                f"MAPE={metrics['mape']:.2f}%"
            )

            # Log to MLflow
            if use_mlflow:
                mlflow.log_params(
                    {
                        "model_type": model_type,
                        "train_samples": len(train_df),
                        "test_samples": len(test_df),
                    }
                )
                mlflow.log_metrics(
                    {
                        "mae": metrics["mae"],
                        "rmse": metrics["rmse"],
                        "mape": metrics["mape"],
                    }
                )

                # Log model
                model_path = self.models_dir / f"{model_type}_model"
                # Note: For LSTM, you'd need to save differently
                mlflow.keras.log_model(model.model, f"{model_type}_model")

                mlflow.end_run()

            history_df = full_history_df if full_history_df is not None else pd.concat(
                [train_df, test_df]
            )
            history_df = history_df.sort_index()

            future_forecast = self._generate_future_forecast(
                model_type=model_type,
                base_model=model,
                history_df=history_df,
                horizon=forecast_horizon,
            )

            model_to_save = model

            if model_type == "lstm" and not history_df.empty:
                logger.info(
                    "Retraining LSTM on full historical values for future forecasting..."
                )
                model_to_save = ModelFactory.create(model_type, self.config)
                model_to_save.fit(history_df)

                future_forecast = self._generate_future_forecast(
                    model_type=model_type,
                    base_model=model_to_save,
                    history_df=history_df,
                    horizon=forecast_horizon,
                )

            return {
                "model": model_to_save,
                "metrics": metrics,
                "predictions": predictions.tolist(),
                "future_forecast": future_forecast,
            }

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")
            if use_mlflow:
                mlflow.end_run(status="FAILED")
            raise

    @staticmethod
    def _build_future_dates(last_date: pd.Timestamp, horizon: int) -> List[str]:
        """Build future daily date strings from the last observed date."""
        start_date = pd.to_datetime(last_date) + pd.Timedelta(days=1)
        return pd.date_range(start=start_date, periods=horizon, freq="D").strftime(
            "%Y-%m-%d"
        ).tolist()

    def _generate_future_forecast(
        self,
        model_type: str,
        base_model: Any,
        history_df: pd.DataFrame,
        horizon: int,
    ) -> Dict[str, Any]:
        """Generate future forecast payload for a trained model."""
        if history_df.empty:
            return {
                "model_type": model_type,
                "horizon": horizon,
                "dates": [],
                "predictions": [],
            }

        last_date = pd.to_datetime(history_df.index.max())
        future_dates = self._build_future_dates(last_date, horizon)

        if model_type == "lstm":
            future_predictions = base_model.predict_multiple(history_df, horizon)
        else:
            future_predictions = base_model.predict(horizon)

        if not isinstance(future_predictions, np.ndarray):
            future_predictions = np.array(future_predictions)

        future_predictions = future_predictions[:horizon]

        return {
            "model_type": model_type,
            "horizon": horizon,
            "dates": future_dates,
            "predictions": future_predictions.tolist(),
        }

    def train_all_models(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        model_types: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Train all models and compare performance

        Args:
            train_df: Training DataFrame
            test_df: Test DataFrame
            model_types: List of model types to train

        Returns:
            Dictionary with all results
        """
        if model_types is None:
            model_types = ["prophet", "arima", "lstm"]

        results = {
            "models": {},
            "metrics": {},
            "predictions": {},
            "future_forecasts": {},
            "best_model": None,
            "best_mape": float("inf"),
        }

        forecast_horizon = int(self.config.get("forecast", {}).get("horizon", 30))
        full_history_df = pd.concat([train_df, test_df]).sort_index()

        for model_type in model_types:
            result = None
            try:
                result = self.train_model(
                    model_type,
                    train_df,
                    test_df,
                    full_history_df=full_history_df,
                    forecast_horizon=forecast_horizon,
                )

                results["models"][model_type] = result["model"]
                results["metrics"][model_type] = result["metrics"]
                results["predictions"][model_type] = result["predictions"]
                results["future_forecasts"][model_type] = result.get(
                    "future_forecast", {}
                )

                # Track best model
                if result["metrics"]["mape"] < results["best_mape"]:
                    results["best_mape"] = result["metrics"]["mape"]
                    results["best_model"] = model_type

            except (KeyboardInterrupt, SystemExit):
                logger.warning(f"Training interrupted for {model_type}")
                raise
            except Exception as e:
                logger.error(f"Failed to train {model_type}: {e}")
                result = None

            # Save model after successful training
            if result is not None:
                try:
                    import pickle

                    model_path = self.models_dir / f"{model_type}_model.pkl"
                    model_to_save = results["models"].get(model_type)
                    if model_to_save:
                        with open(model_path, "wb") as f:
                            pickle.dump(model_to_save, f)
                        logger.info(f"Saved {model_type} model to {model_path}")
                except Exception as e:
                    logger.warning(f"Could not save {model_type} model: {e}")

        # Print summary
        logger.info("\n=== Model Comparison ===")
        for model_type, metrics in results["metrics"].items():
            logger.info(
                f"{model_type}: MAE={metrics['mae']:.2f}, "
                f"RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%"
            )

        logger.info(
            f"\nBest Model: {results['best_model']} (MAPE: {results['best_mape']:.2f}%)"
        )

        return results

    def cross_validate_model(
        self, model_type: str, df: pd.DataFrame, n_splits: int = 5
    ) -> Dict[str, float]:
        """
        Perform time series cross-validation

        Args:
            model_type: Type of model
            df: Full DataFrame
            n_splits: Number of CV splits

        Returns:
            Dictionary with CV metrics
        """
        logger.info(f"Cross-validating {model_type}...")

        tscv = TimeSeriesSplit(n_splits=n_splits)

        cv_metrics = {"mae": [], "rmse": [], "mape": []}

        for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            try:
                result = self.train_model(
                    model_type, train_df, test_df, use_mlflow=False
                )

                cv_metrics["mae"].append(result["metrics"]["mae"])
                cv_metrics["rmse"].append(result["metrics"]["rmse"])
                cv_metrics["mape"].append(result["metrics"]["mape"])

            except Exception as e:
                logger.warning(f"Fold {fold} failed: {e}")

        # Average metrics
        avg_metrics = {
            "mae": np.mean(cv_metrics["mae"]),
            "rmse": np.mean(cv_metrics["rmse"]),
            "mape": np.mean(cv_metrics["mape"]),
            "std_mae": np.std(cv_metrics["mae"]),
            "std_rmse": np.std(cv_metrics["rmse"]),
            "std_mape": np.std(cv_metrics["mape"]),
        }

        logger.info(
            f"{model_type} CV: MAE={avg_metrics['mae']:.2f}±{avg_metrics['std_mae']:.2f}, "
            f"RMSE={avg_metrics['rmse']:.2f}±{avg_metrics['std_rmse']:.2f}, "
            f"MAPE={avg_metrics['mape']:.2f}±{avg_metrics['std_mape']:.2f}%"
        )

        return avg_metrics

    def save_model(self, model: Any, model_type: str):
        """Save trained model to disk"""
        import pickle

        model_path = self.models_dir / f"{model_type}_model.pkl"

        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info(f"Model saved to {model_path}")

        return str(model_path)

    def load_model(self, model_type: str) -> Any:
        """Load trained model from disk"""
        import pickle

        model_path = self.models_dir / f"{model_type}_model.pkl"

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        with open(model_path, "rb") as f:
            model = pickle.load(f)

        logger.info(f"Model loaded from {model_path}")

        return model

    def save_results(
        self, results: Dict[str, Any], filename: str = "training_results.json"
    ):
        """Save training results to JSON"""
        # Convert non-serializable items
        serializable_results = {
            "timestamp": datetime.now().isoformat(),
            "metrics": results.get("metrics", {}),
            "future_forecasts": results.get("future_forecasts", {}),
            "best_model": results.get("best_model"),
            "best_mape": results.get("best_mape"),
        }

        results_path = Path(__file__).parent.parent / filename

        with open(results_path, "w") as f:
            json.dump(serializable_results, f, indent=2)

        logger.info(f"Results saved to {results_path}")


class TimeSeriesSplit:
    """Time series cross-validation splitter"""

    def __init__(self, n_splits: int = 5, test_size: int = 30):
        self.n_splits = n_splits
        self.test_size = test_size

    def split(self, df: pd.DataFrame):
        """Generate train/test splits"""
        n = len(df)

        for i in range(self.n_splits):
            # Calculate test indices
            test_end = n - (self.n_splits - 1 - i) * self.test_size
            test_start = test_end - self.test_size

            if test_start < self.test_size:
                break

            train_indices = range(0, test_start)
            test_indices = range(test_start, test_end)

            yield list(train_indices), list(test_indices)


def run_full_pipeline(
    config_path: Optional[str] = None,
    start_date: str = "2004-01-01",
    end_date: Optional[str] = None,
    csv_path: Optional[str] = None,
    model_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Run the full training pipeline

    Args:
        config_path: Path to config file
        start_date: Start date for data
        end_date: End date for data
        csv_path: Path to historical CSV
        model_types: Models to train

    Returns:
        Training results
    """
    # Load config
    if config_path:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    else:
        config = Config.load()

    # Initialize trainer
    trainer = ModelTrainer(config)
    trainer.setup_mlflow()

    # Prepare data
    df = trainer.prepare_data(start_date, end_date, csv_path)

    # Split data
    train_df, test_df = trainer.split_data(df)

    # Train models
    results = trainer.train_all_models(train_df, test_df, model_types)

    # Save results
    trainer.save_results(results)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train Gold Price Forecasting Models")
    parser.add_argument("--start-date", default="2004-01-01", help="Start date")
    parser.add_argument("--end-date", help="End date")
    parser.add_argument("--csv-path", help="Path to historical CSV")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["prophet", "arima", "lstm"],
        help="Models to train",
    )

    args = parser.parse_args()

    results = run_full_pipeline(
        start_date=args.start_date,
        end_date=args.end_date,
        csv_path=args.csv_path,
        model_types=args.models,
    )

    print("\n=== Training Complete ===")
    print(f"Best Model: {results['best_model']}")
    print(f"Best MAPE: {results['best_mape']:.2f}%")
