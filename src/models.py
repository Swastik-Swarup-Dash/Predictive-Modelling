"""
Time Series Models for Gold Price Forecasting
Implements Prophet, ARIMA/SARIMA, and LSTM/Bi-LSTM models
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
)
from sklearn.model_selection import TimeSeriesSplit

import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate evaluation metrics for time series forecasting"""

    @staticmethod
    def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return mean_absolute_percentage_error(y_true, y_pred) * 100

    @staticmethod
    def calculate_all(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate all metrics"""
        return {
            "mae": MetricsCalculator.calculate_mae(y_true, y_pred),
            "rmse": MetricsCalculator.calculate_rmse(y_true, y_pred),
            "mape": MetricsCalculator.calculate_mape(y_true, y_pred),
        }


class ProphetModel:
    """
    Facebook Prophet model for time series forecasting
    Automatically handles trends and seasonality
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        from prophet import Prophet

        self.config = config or {}
        prophet_config = self.config.get("models", {}).get("prophet", {})

        self.model = Prophet(
            changepoint_prior_scale=prophet_config.get("changepoint_prior_scale", 0.05),
            seasonality_prior_scale=prophet_config.get("seasonality_prior_scale", 10),
            daily_seasonality=prophet_config.get("daily_seasonality", False),
            weekly_seasonality=prophet_config.get("weekly_seasonality", True),
            yearly_seasonality=prophet_config.get("yearly_seasonality", True),
            interval_width=0.95,
        )

        self.is_fitted = False
        self.model_name = "Prophet"

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data in Prophet format (ds, y)"""
        # Get the date from index or column
        if isinstance(df.index, pd.DatetimeIndex):
            prophet_df = pd.DataFrame({"ds": df.index, "y": df["gold_price"].values})
        else:
            prophet_df = df.reset_index()
            # Try to find date column
            date_col = None
            for col in ["date", "Date", "datetime", "ds"]:
                if col in prophet_df.columns:
                    date_col = col
                    break
            if date_col:
                prophet_df = prophet_df.rename(
                    columns={date_col: "ds", "gold_price": "y"}
                )
            else:
                prophet_df = prophet_df.rename(
                    columns={prophet_df.columns[0]: "ds", "gold_price": "y"}
                )

        # Ensure ds is datetime
        prophet_df["ds"] = pd.to_datetime(prophet_df["ds"])
        prophet_df["y"] = pd.to_numeric(prophet_df["y"], errors="coerce")
        prophet_df = prophet_df.dropna()

        return prophet_df[["ds", "y"]]

    def fit(self, df: pd.DataFrame) -> "ProphetModel":
        """Fit Prophet model"""
        logger.info("Training Prophet model...")

        train_df = self.prepare_data(df)
        self.model.fit(train_df)

        self.is_fitted = True
        logger.info("Prophet model trained successfully")

        return self

    def predict(self, periods: int = 30) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)

        # Return only the predictions (yhat values), excluding historical data
        # Filter to only future dates
        last_train_date = forecast[forecast["yhat"].notna()]["ds"].max()
        future_forecast = forecast[forecast["ds"] > last_train_date]

        return future_forecast["yhat"].values

    def predict_with_confidence(self, periods: int = 30) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals"""
        future = self.model.make_future_dataframe(periods=periods)
        forecast = self.model.predict(future)

        # Get future predictions only
        last_train_date = forecast[forecast["yhat"].notna()]["ds"].max()
        future_forecast = forecast[forecast["ds"] > last_train_date]

        result = {
            "dates": future_forecast["ds"].dt.strftime("%Y-%m-%d").tolist(),
            "predictions": future_forecast["yhat"].tolist(),
            "lower_bound": future_forecast["yhat_lower"].tolist(),
            "upper_bound": future_forecast["yhat_upper"].tolist(),
        }

        return result

    def get_component_plot(self) -> Any:
        """Get Prophet component plots"""
        return self.model.plot_components

    def cross_validate(
        self,
        df: pd.DataFrame,
        horizon: int = 30,
        cutoffs: Optional[List[datetime]] = None,
    ) -> Dict[str, float]:
        """Time series cross-validation"""
        from prophet.diagnostics import cross_validation, performance_metrics

        if cutoffs is None:
            initial = f"{len(df) - horizon * 3} days"
            period = f"{horizon} days"
            horizon = f"{horizon} days"

        df_cv = cross_validation(
            self.model, cutoffs=cutoffs, initial=initial, period=period, horizon=horizon
        )

        metrics = performance_metrics(df_cv)

        return {
            "mae": metrics["mae"].mean(),
            "rmse": metrics["rmse"].mean(),
            "mape": metrics["mape"].mean() * 100,
        }


class ARIMAModel:
    """
    ARIMA/SARIMA model for time series forecasting
    Uses auto_arima for automatic parameter selection
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        arima_config = self.config.get("models", {}).get("arima", {})

        self.model = None
        self.best_model = None
        self.is_fitted = False
        self.model_name = "ARIMA"

        self.seasonal = arima_config.get("seasonal", True)
        self.m = arima_config.get("m", 7)  # Weekly seasonality
        self.max_p = arima_config.get("max_p", 5)
        self.max_q = arima_config.get("max_q", 5)
        self.max_d = arima_config.get("max_d", 2)

    def fit(self, df: pd.DataFrame) -> "ARIMAModel":
        """Fit ARIMA model with auto parameter selection"""
        from pmdarima import auto_arima

        logger.info("Training ARIMA model...")

        series = df["gold_price"].values

        self.model = auto_arima(
            series,
            seasonal=self.seasonal,
            m=self.m,
            max_p=self.max_p,
            max_q=self.max_q,
            max_d=self.max_d,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            trace=True,
            n_fits=50,
        )

        self.best_model = self.model
        self.is_fitted = True

        logger.info(
            f"ARIMA model fitted: {self.model.order}, seasonal: {self.model.seasonal_order}"
        )

        return self

    def predict(self, periods: int = 30) -> np.ndarray:
        """Generate forecasts"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts = self.model.predict(n_periods=periods)
        return forecasts

    def predict_with_confidence(self, periods: int = 30) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        forecasts, conf_int = self.model.predict(
            n_periods=periods, return_conf_int=True
        )

        return {
            "predictions": forecasts.tolist()
            if hasattr(forecasts, "tolist")
            else list(forecasts),
            "lower_bound": conf_int[:, 0].tolist(),
            "upper_bound": conf_int[:, 1].tolist(),
        }

    def get_summary(self) -> str:
        """Get model summary"""
        if self.model:
            return self.model.summary().as_text()
        return "Model not fitted"


class LSTMModel:
    """
    LSTM/Bi-LSTM model for time series forecasting
    Uses 60-day sequences to predict next day close price
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        lstm_config = self.config.get("models", {}).get("lstm", {})

        self.sequence_length = lstm_config.get("sequence_length", 60)
        self.layers = lstm_config.get("layers", 2)
        self.units = lstm_config.get("units", 50)
        self.dropout = lstm_config.get("dropout", 0.2)
        self.epochs = lstm_config.get("epochs", 50)
        self.batch_size = lstm_config.get("batch_size", 32)
        self.learning_rate = lstm_config.get("learning_rate", 0.001)

        self.model = None
        self.scaler = MinMaxScaler()
        self.is_fitted = False
        self.model_name = "LSTM"

        self._build_model()

    def _build_model(self):
        """Build LSTM model architecture"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Bidirectional, Dense, Dropout

        self.model = Sequential()

        # First LSTM layer
        if self.layers == 1:
            self.model.add(
                LSTM(
                    self.units,
                    return_sequences=False,
                    input_shape=(self.sequence_length, 1),
                )
            )
        else:
            self.model.add(
                LSTM(
                    self.units,
                    return_sequences=True,
                    input_shape=(self.sequence_length, 1),
                )
            )
            self.model.add(Dropout(self.dropout))

            # Middle LSTM layers
            for _ in range(self.layers - 2):
                self.model.add(LSTM(self.units, return_sequences=True))
                self.model.add(Dropout(self.dropout))

            # Last LSTM layer
            self.model.add(LSTM(self.units, return_sequences=False))
            self.model.add(Dropout(self.dropout))

        # Output layer
        self.model.add(Dense(1))

        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        logger.info("LSTM model architecture built")

    def _create_sequences(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X, y = [], []

        for i in range(len(data) - self.sequence_length):
            X.append(data[i : i + self.sequence_length])
            y.append(data[i + self.sequence_length])

        return np.array(X), np.array(y)

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for LSTM"""
        # Use only gold price
        prices = df["gold_price"].values.reshape(-1, 1)

        # Scale
        scaled_prices = self.scaler.fit_transform(prices)

        # Create sequences
        X, y = self._create_sequences(scaled_prices)

        return X, y

    def fit(self, df: pd.DataFrame, validation_split: float = 0.2) -> "LSTMModel":
        """Fit LSTM model"""
        import os

        # Limit TensorFlow memory
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

        import tensorflow as tf

        tf.config.set_soft_device_placement(True)

        # Limit GPU memory if available
        gpus = tf.config.list_physical_devices("GPU")
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        else:
            # Limit CPU memory
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
        import tensorflow as tf

        logger.info("Training LSTM model...")

        # Prepare data
        X, y = self.prepare_data(df)

        logger.info(f"Training data shape: X={X.shape}, y={y.shape}")

        # Callbacks
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-6
        )

        # Train
        self.model.fit(
            X,
            y,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=validation_split,
            callbacks=[early_stop, reduce_lr],
            verbose=1,
        )

        self.is_fitted = True
        logger.info("LSTM model trained successfully")

        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Generate predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        # Prepare last sequence
        prices = df["gold_price"].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)

        # Get last sequence_length prices
        last_sequence = scaled_prices[-self.sequence_length :].reshape(
            1, self.sequence_length, 1
        )

        # Predict
        prediction = self.model.predict(last_sequence, verbose=0)

        # Inverse transform
        prediction = self.scaler.inverse_transform(prediction)

        return prediction.flatten()

    def predict_multiple(self, df: pd.DataFrame, periods: int = 30) -> np.ndarray:
        """Generate multiple forecasts iteratively"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []

        # Get last sequence
        prices = df["gold_price"].values.reshape(-1, 1)
        scaled_prices = self.scaler.transform(prices)

        current_sequence = scaled_prices[-self.sequence_length :].copy()

        for _ in range(periods):
            # Reshape for prediction
            input_seq = current_sequence.reshape(1, self.sequence_length, 1)

            # Predict next value
            pred = self.model.predict(input_seq, verbose=0)
            predictions.append(pred[0, 0])

            # Update sequence
            current_sequence = np.roll(current_sequence, -1)
            current_sequence[-1] = pred[0, 0]

        # Inverse transform all predictions
        predictions = np.array(predictions).reshape(-1, 1)
        predictions = self.scaler.inverse_transform(predictions)

        return predictions.flatten()

    def predict_with_confidence(
        self, df: pd.DataFrame, periods: int = 30
    ) -> Dict[str, Any]:
        """Generate forecasts with confidence intervals using Monte Carlo dropout"""
        # Get predictions with some variance
        predictions = []

        for _ in range(10):  # Monte Carlo simulations
            pred = self.predict_multiple(df, periods)
            predictions.append(pred)

        predictions = np.array(predictions)

        return {
            "predictions": predictions.mean(axis=0).tolist(),
            "lower_bound": (
                predictions.mean(axis=0) - predictions.std(axis=0)
            ).tolist(),
            "upper_bound": (
                predictions.mean(axis=0) + predictions.std(axis=0)
            ).tolist(),
        }


class BiLSTMModel(LSTMModel):
    """
    Bidirectional LSTM model
    Inherits from LSTMModel with bidirectional layers
    """

    def _build_model(self):
        """Build Bi-LSTM model architecture"""
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout

        self.model = Sequential()

        # First Bi-LSTM layer
        if self.layers == 1:
            self.model.add(
                Bidirectional(
                    LSTM(self.units, return_sequences=False),
                    input_shape=(self.sequence_length, 1),
                )
            )
        else:
            self.model.add(
                Bidirectional(
                    LSTM(self.units, return_sequences=True),
                    input_shape=(self.sequence_length, 1),
                )
            )
            self.model.add(Dropout(self.dropout))

            # Middle Bi-LSTM layers
            for _ in range(self.layers - 2):
                self.model.add(Bidirectional(LSTM(self.units, return_sequences=True)))
                self.model.add(Dropout(self.dropout))

            # Last Bi-LSTM layer
            self.model.add(Bidirectional(LSTM(self.units, return_sequences=False)))
            self.model.add(Dropout(self.dropout))

        # Output layer
        self.model.add(Dense(1))

        # Compile
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"],
        )

        self.model_name = "Bi-LSTM"
        logger.info("Bi-LSTM model architecture built")


class ModelFactory:
    """Factory class to create model instances"""

    MODELS = {
        "prophet": ProphetModel,
        "arima": ARIMAModel,
        "lstm": LSTMModel,
        "bi-lstm": BiLSTMModel,
    }

    @classmethod
    def create(cls, model_type: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """Create a model instance"""
        if model_type.lower() not in cls.MODELS:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Available: {list(cls.MODELS.keys())}"
            )

        return cls.MODELS[model_type.lower()](config)

    @classmethod
    def create_all(cls, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Create all model instances"""
        return {name: cls.create(name, config) for name in cls.MODELS}


class EnsembleModel:
    """
    Ensemble model combining multiple base models
    Uses weighted averaging based on validation performance
    """

    def __init__(
        self, models: Dict[str, Any], weights: Optional[Dict[str, float]] = None
    ):
        self.models = models
        self.weights = weights
        self.is_fitted = False
        self.model_name = "Ensemble"

    def fit(self, df: pd.DataFrame, validation_size: float = 0.2) -> "EnsembleModel":
        """Fit all base models and calculate weights"""
        # Split for weight calculation
        train_size = int(len(df) * (1 - validation_size))
        train_df = df.iloc[:train_size]
        val_df = df.iloc[train_size:]

        # Fit all models
        val_predictions = {}

        for name, model in self.models.items():
            model.fit(train_df)
            pred = model.predict(len(val_df))

            if isinstance(pred, np.ndarray):
                val_predictions[name] = pred[: len(val_df)]
            else:
                val_predictions[name] = np.array(pred)[: len(val_df)]

        # Calculate weights based on MAPE
        if self.weights is None:
            errors = {}
            for name, pred in val_predictions.items():
                actual = val_df["gold_price"].values[: len(pred)]
                errors[name] = MetricsCalculator.calculate_mape(actual, pred)

            # Inverse error weighting
            inv_errors = {k: 1 / v for k, v in errors.items()}
            total = sum(inv_errors.values())
            self.weights = {k: v / total for k, v in inv_errors.items()}

        self.is_fitted = True
        logger.info(f"Ensemble weights: {self.weights}")

        return self

    def predict(self, periods: int = 30) -> np.ndarray:
        """Generate ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")

        predictions = []

        for name, model in self.models.items():
            pred = model.predict(periods)
            if isinstance(pred, np.ndarray):
                predictions.append(pred * self.weights[name])
            else:
                predictions.append(np.array(pred) * self.weights[name])

        return sum(predictions)


def train_and_evaluate_models(
    df: pd.DataFrame, config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Train all models and evaluate their performance

    Args:
        df: Input DataFrame with gold prices
        config: Configuration dictionary

    Returns:
        Dictionary with trained models and metrics
    """
    # Split data
    train_size = int(len(df) * 0.8)
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]

    results = {"models": {}, "metrics": {}, "predictions": {}}

    # Train each model
    for model_type in ["prophet", "arima", "lstm"]:
        try:
            logger.info(f"\nTraining {model_type}...")

            model = ModelFactory.create(model_type, config)
            model.fit(train_df)

            # Predict on test set
            if model_type == "lstm":
                # LSTM needs iterative prediction
                predictions = model.predict_multiple(train_df, len(test_df))
            else:
                predictions = model.predict(len(test_df))

            # Handle different return types
            if not isinstance(predictions, np.ndarray):
                predictions = np.array(predictions)

            # Ensure same length
            predictions = predictions[: len(test_df)]

            # Calculate metrics
            actual = test_df["gold_price"].values[: len(predictions)]
            metrics = MetricsCalculator.calculate_all(actual, predictions)

            results["models"][model_type] = model
            results["metrics"][model_type] = metrics
            results["predictions"][model_type] = predictions.tolist()

            logger.info(
                f"{model_type} - MAE: {metrics['mae']:.2f}, "
                f"RMSE: {metrics['rmse']:.2f}, MAPE: {metrics['mape']:.2f}%"
            )

        except Exception as e:
            logger.error(f"Error training {model_type}: {e}")

    return results


if __name__ == "__main__":
    # Example usage
    import yaml

    # Load config
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Generate sample data for testing
    dates = pd.date_range(start="2020-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)
    prices = 1800 + np.cumsum(np.random.randn(len(dates)) * 10)
    df = pd.DataFrame({"gold_price": prices}, index=dates)

    # Train and evaluate
    results = train_and_evaluate_models(df, config)

    print("\n=== Model Results ===")
    for model_name, metrics in results["metrics"].items():
        print(
            f"{model_name}: MAE={metrics['mae']:.2f}, RMSE={metrics['rmse']:.2f}, MAPE={metrics['mape']:.2f}%"
        )
