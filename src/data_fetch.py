"""
Data Fetching Module for Gold Price Forecasting
Fetches historical and real-time gold price data from multiple sources
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from sqlalchemy import create_engine, text
import yaml

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader"""

    @staticmethod
    def load() -> Dict[str, Any]:
        """Load configuration from config.yaml"""
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


class GoldDataFetcher:
    """
    Fetches gold price data from multiple sources:
    - MetalpriceAPI (real-time)
    - yfinance (backup/redundancy)
    - CSV historical data
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.load()
        self.api_key = self.config.get("api", {}).get("metalpriceapi_key", "")
        self.db_config = self.config.get("database", {})

    def fetch_from_metalpriceapi(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """
        Fetch gold prices from MetalpriceAPI

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with gold prices or None on failure
        """
        if not self.api_key or self.api_key == "YOUR_METALPRICEAPI_KEY":
            logger.warning("MetalpriceAPI key not configured. Using yfinance instead.")
            return None

        try:
            url = "https://api.metalpriceapi.com/v1/timeframe"
            headers = {"Authorization": f"API_KEY {self.api_key}"}
            params = {
                "api_key": self.api_key,
                "start_date": start_date,
                "end_date": end_date,
                "unit": "toz",  # troy ounce
                "currency": "USD",
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("success"):
                records = data.get("prices", [])
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                df = df.rename(columns={"price": "gold_price"})
                df = df.set_index("date").sort_index()
                logger.info(f"Fetched {len(df)} records from MetalpriceAPI")
                return df

        except requests.exceptions.RequestException as e:
            logger.error(f"MetalpriceAPI request failed: {e}")
        except Exception as e:
            logger.error(f"Error fetching from MetalpriceAPI: {e}")

        return None

    def fetch_from_yfinance(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch gold prices from yfinance (GC=F for Gold Futures)

        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format

        Returns:
            DataFrame with gold prices
        """
        try:
            ticker = yf.Ticker("GC=F")
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                logger.warning("yfinance returned empty data")
                return pd.DataFrame()

            df = df.reset_index()
            df["Date"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
            df = df.rename(
                columns={"Date": "date", "Close": "gold_price", "Volume": "volume"}
            )
            df = df[["date", "gold_price", "volume"]].set_index("date")

            logger.info(f"Fetched {len(df)} records from yfinance")
            return df

        except Exception as e:
            logger.error(f"Error fetching from yfinance: {e}")
            return pd.DataFrame()

    def fetch_additional_indicators(
        self, start_date: str, end_date: str
    ) -> pd.DataFrame:
        """
        Fetch additional economic indicators:
        - US Dollar Index (DXY)
        - Oil prices (CL=F)
        - Interest rates (^IRX)

        Args:
            start_date: Start date
            end_date: End date

        Returns:
            DataFrame with additional indicators
        """
        indicators = pd.DataFrame()

        try:
            # US Dollar Index
            dxy = yf.Ticker("DXY")
            dxy_data = dxy.history(start=start_date, end=end_date)
            if not dxy_data.empty:
                indicators["dxy"] = dxy_data["Close"]

        except Exception as e:
            logger.warning(f"Could not fetch DXY: {e}")

        try:
            # Crude Oil
            oil = yf.Ticker("CL=F")
            oil_data = oil.history(start=start_date, end=end_date)
            if not oil_data.empty:
                indicators["oil_price"] = oil_data["Close"]

        except Exception as e:
            logger.warning(f"Could not fetch oil prices: {e}")

        try:
            # 13-week T-Bill Rate (interest rate proxy)
            ir = yf.Ticker("^IRX")
            ir_data = ir.history(start=start_date, end=end_date)
            if not ir_data.empty:
                indicators["interest_rate"] = ir_data["Close"]

        except Exception as e:
            logger.warning(f"Could not fetch interest rates: {e}")

        if not indicators.empty:
            indicators = indicators.reset_index()
            indicators["Date"] = pd.to_datetime(indicators["Date"]).dt.tz_localize(None)
            indicators = indicators.set_index("Date")
            logger.info(f"Fetched additional indicators: {list(indicators.columns)}")

        return indicators

    def load_historical_csv(self, filepath: str) -> pd.DataFrame:
        """
        Load historical gold price data from CSV

        Args:
            filepath: Path to CSV file

        Returns:
            DataFrame with gold prices
        """
        try:
            df = pd.read_csv(filepath, parse_dates=["Date"])
            df = df.set_index("Date").sort_index()
            logger.info(f"Loaded {len(df)} historical records from {filepath}")
            return df
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            return pd.DataFrame()

    def merge_data_sources(
        self, primary: pd.DataFrame, secondary: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Merge primary and secondary data sources, filling gaps

        Args:
            primary: Primary data source
            secondary: Secondary data source for filling gaps

        Returns:
            Merged DataFrame
        """
        if primary.empty:
            return secondary
        if secondary.empty:
            return primary

        combined = pd.concat([primary, secondary])
        combined = combined[~combined.index.duplicated(keep="first")]
        combined = combined.sort_index()

        return combined


class DataPreprocessor:
    """
    Preprocesses gold price data for modeling
    """

    def __init__(self):
        pass

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values using forward fill then backward fill

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with no missing values
        """
        df = df.copy()
        df = df.ffill().bfill()

        if df.isnull().any().any():
            logger.warning(
                f"Still have missing values after filling: {df.isnull().sum()}"
            )

        return df

    def check_stationarity(
        self, series: pd.Series, alpha: float = 0.05
    ) -> Tuple[bool, float]:
        """
        Check stationarity using Augmented Dickey-Fuller test

        Args:
            series: Time series data
            alpha: Significance level

        Returns:
            Tuple of (is_stationary, p_value)
        """
        from statsmodels.tsa.stattools import adfuller

        result = adfuller(series.dropna(), autolag="AIC")
        p_value = result[1]
        is_stationary = p_value < alpha

        logger.info(f"ADF Test: p-value={p_value:.4f}, stationary={is_stationary}")

        return is_stationary, p_value

    def difference_series(self, series: pd.Series, order: int = 1) -> pd.Series:
        """
        Difference the series to make it stationary

        Args:
            series: Time series
            order: Difference order

        Returns:
            Differenced series
        """
        diff_series = series.copy()
        for _ in range(order):
            diff_series = diff_series.diff()

        return diff_series.dropna()

    def add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add time-based features for modeling

        Args:
            df: Input DataFrame with date index

        Returns:
            DataFrame with added features
        """
        df = df.copy()

        # Basic time features
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year

        # Cyclical features
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # Lag features
        for lag in [1, 2, 3, 7, 14, 30]:
            df[f"lag_{lag}"] = df["gold_price"].shift(lag)

        # Rolling statistics
        for window in [7, 14, 30]:
            df[f"rolling_mean_{window}"] = (
                df["gold_price"].rolling(window=window).mean()
            )
            df[f"rolling_std_{window}"] = df["gold_price"].rolling(window=window).std()

        # Returns
        df["returns"] = df["gold_price"].pct_change()
        df["log_returns"] = np.log(df["gold_price"] / df["gold_price"].shift(1))

        return df

    def normalize_features(
        self, df: pd.DataFrame, scaler=None
    ) -> Tuple[pd.DataFrame, Any]:
        """
        Normalize features using MinMaxScaler

        Args:
            df: Input DataFrame
            scaler: Pre-fitted scaler (optional)

        Returns:
            Tuple of (normalized DataFrame, scaler)
        """
        from sklearn.preprocessing import MinMaxScaler

        if scaler is None:
            scaler = MinMaxScaler()
            scaled_values = scaler.fit_transform(df.values)
        else:
            scaled_values = scaler.transform(df.values)

        df_scaled = pd.DataFrame(scaled_values, index=df.index, columns=df.columns)

        return df_scaled, scaler


class DatabaseManager:
    """
    Manages PostgreSQL database operations
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.load()
        self.db_config = self.config.get("database", {})
        self.engine = None

    def get_connection_string(self) -> str:
        """Get SQLAlchemy connection string"""
        return (
            f"postgresql://{self.db_config.get('user', 'postgres')}:"
            f"{self.db_config.get('password', 'postgres')}@"
            f"{self.db_config.get('host', 'localhost')}:"
            f"{self.db_config.get('port', 5432)}/"
            f"{self.db_config.get('name', 'gold_forecast')}"
        )

    def connect(self):
        """Establish database connection"""
        try:
            self.engine = create_engine(self.get_connection_string())
            logger.info("Database connection established")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def create_tables(self):
        """Create necessary database tables"""
        if not self.engine:
            self.connect()

        create_gold_prices_table = """
        CREATE TABLE IF NOT EXISTS gold_prices (
            id SERIAL PRIMARY KEY,
            date DATE UNIQUE NOT NULL,
            gold_price DECIMAL(12, 2),
            volume BIGINT,
            dxy DECIMAL(10, 4),
            oil_price DECIMAL(10, 4),
            interest_rate DECIMAL(8, 4),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_predictions_table = """
        CREATE TABLE IF NOT EXISTS predictions (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            model_name VARCHAR(50) NOT NULL,
            predicted_price DECIMAL(12, 2),
            actual_price DECIMAL(12, 2),
            horizon INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        create_model_versions_table = """
        CREATE TABLE IF NOT EXISTS model_versions (
            id SERIAL PRIMARY KEY,
            model_name VARCHAR(50) NOT NULL,
            version VARCHAR(20) NOT NULL,
            metrics JSONB,
            trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT TRUE
        );
        """

        with self.engine.connect() as conn:
            conn.execute(text(create_gold_prices_table))
            conn.execute(text(create_predictions_table))
            conn.execute(text(create_model_versions_table))
            conn.commit()

        logger.info("Database tables created/verified")

    def save_gold_prices(self, df: pd.DataFrame):
        """Save gold prices to database"""
        if not self.engine:
            self.connect()

        df = df.reset_index()
        df = df.rename(columns={"index": "date"})

        df.to_sql("gold_prices", self.engine, if_exists="append", index=False)
        logger.info(f"Saved {len(df)} records to gold_prices table")

    def load_gold_prices(
        self, start_date: Optional[str] = None, end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """Load gold prices from database"""
        if not self.engine:
            self.connect()

        query = "SELECT * FROM gold_prices"
        conditions = []

        if start_date:
            conditions.append(f"date >= '{start_date}'")
        if end_date:
            conditions.append(f"date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY date"

        df = pd.read_sql(query, self.engine)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            df = df.set_index("date")

        return df

    def save_predictions(self, predictions: List[Dict[str, Any]]):
        """Save model predictions to database"""
        if not self.engine:
            self.connect()

        df = pd.DataFrame(predictions)
        df.to_sql("predictions", self.engine, if_exists="append", index=False)
        logger.info(f"Saved {len(predictions)} predictions to database")


def fetch_all_data(
    start_date: str = "2004-01-01",
    end_date: Optional[str] = None,
    csv_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Main function to fetch all gold price data

    Args:
        start_date: Start date for fetching data
        end_date: End date for fetching data (default: today)
        csv_path: Path to historical CSV file (optional)

    Returns:
        Combined DataFrame with all data
    """
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    fetcher = GoldDataFetcher()
    preprocessor = DataPreprocessor()

    df = pd.DataFrame()

    # Try loading historical CSV first
    if csv_path and Path(csv_path).exists():
        df = fetcher.load_historical_csv(csv_path)

    # Fetch from MetalpriceAPI
    api_data = fetcher.fetch_from_metalpriceapi(start_date, end_date)
    if api_data is not None:
        df = fetcher.merge_data_sources(df, api_data)

    # Fetch from yfinance as primary/backup
    yf_data = fetcher.fetch_from_yfinance(start_date, end_date)
    df = fetcher.merge_data_sources(df, yf_data)

    # Fetch additional indicators
    indicators = fetcher.fetch_additional_indicators(start_date, end_date)
    if not indicators.empty:
        df = df.join(indicators, how="left")

    # Preprocess
    df = preprocessor.handle_missing_values(df)

    logger.info(f"Final dataset shape: {df.shape}")
    logger.info(f"Date range: {df.index.min()} to {df.index.max()}")

    return df


if __name__ == "__main__":
    # Example usage
    data = fetch_all_data()
    print(data.head())
    print(f"\nDataset shape: {data.shape}")
    print(f"Date range: {data.index.min()} to {data.index.max()}")
