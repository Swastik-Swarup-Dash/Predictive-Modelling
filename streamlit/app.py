"""
Streamlit Dashboard for Gold Price Forecasting
Real-time predictions, model comparison, and data visualization
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any, List

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yaml
import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="Gold Price Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Configuration
CONFIG_PATH = Path(__file__).parent.parent / "config.yaml"


def load_config() -> Dict[str, Any]:
    """Load configuration with fallback to defaults"""
    default_config = {
        "server": {"host": "0.0.0.0", "port": 8000, "workers": 1},
        "dashboard": {"refresh_interval": 300, "port": 8501},
        "database": {
            "host": "localhost",
            "port": 5432,
            "name": "gold_forecast",
            "user": "postgres",
            "password": "postgres",
        },
        "api": {"metalpriceapi_key": ""},
        "models": {
            "lstm": {"sequence_length": 60, "layers": 2, "units": 50},
            "prophet": {},
            "arima": {},
        },
        "training": {"test_size": 0.2, "random_state": 42, "n_splits": 5},
        "forecast": {"horizon": 30, "confidence_level": 0.95},
        "monitoring": {"drift_threshold": 0.05, "retrain_frequency": "weekly"},
    }

    try:
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return default_config


config = load_config()


class APIClient:
    """Client for interacting with the FastAPI backend"""

    def __init__(self, base_url: str = None):
        # Use environment variable or default to localhost
        self.base_url = base_url or os.environ.get("API_URL", "http://localhost:8000")

    def get_health(self) -> Dict[str, Any]:
        """Get API health status"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.json()
        except:
            return {"status": "offline", "models_loaded": False}

    def get_predictions(self, model_type: str = "all", horizon: int = 7) -> List[Dict]:
        """Get predictions"""
        try:
            response = requests.post(
                f"{self.base_url}/predict",
                json={"model_type": model_type, "horizon": horizon},
                timeout=30,
            )
            return response.json()
        except:
            # Return demo predictions when API is unavailable
            return self._get_demo_predictions(horizon)

    def _get_demo_predictions(self, horizon: int) -> List[Dict]:
        """Generate demo predictions for standalone mode"""
        base_price = 1736.0
        predictions = [base_price + i * 0.3 for i in range(horizon)]
        return [
            {
                "model_type": "demo",
                "predictions": predictions,
                "lower_bound": [p - 20 for p in predictions],
                "upper_bound": [p + 20 for p in predictions],
                "horizon": horizon,
                "timestamp": datetime.now().isoformat(),
            }
        ]

    def get_metrics(self) -> List[Dict]:
        """Get model metrics"""
        try:
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            return response.json()
        except:
            # Return demo metrics
            return [
                {"model_type": "demo_arima", "mae": 15.2, "rmse": 22.1, "mape": 0.87}
            ]

    def get_latest_data(self, days: int = 90) -> Dict:
        """Get latest price data"""
        try:
            response = requests.get(
                f"{self.base_url}/data/latest?days={days}", timeout=10
            )
            return response.json()
        except:
            # Return demo data
            dates = pd.date_range(end=datetime.now(), periods=days, freq="D")
            np.random.seed(42)
            prices = 1700 + np.cumsum(np.random.randn(days) * 5)
            return {
                "dates": [d.strftime("%Y-%m-%d") for d in dates],
                "prices": prices.tolist(),
                "volumes": [None] * days,
            }


def load_sample_data() -> pd.DataFrame:
    """Generate sample data for demo"""
    dates = pd.date_range(start="2023-01-01", end="2024-01-01", freq="D")
    np.random.seed(42)
    prices = 1800 + np.cumsum(np.random.randn(len(dates)) * 8)

    return pd.DataFrame({"gold_price": prices}, index=dates)


def plot_price_chart(
    df: pd.DataFrame, title: str = "Gold Price (XAU/USD)"
) -> go.Figure:
    """Create interactive price chart"""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["gold_price"],
            mode="lines",
            name="Price",
            line=dict(color="#FFD700", width=2),
        )
    )

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )

    return fig


def plot_predictions(
    actual: list, predictions: Dict[str, list], dates: list
) -> go.Figure:
    """Plot actual vs predicted prices"""
    fig = go.Figure()

    # Actual prices
    fig.add_trace(
        go.Scatter(
            x=dates[: len(actual)],
            y=actual,
            mode="lines",
            name="Actual",
            line=dict(color="#00FF00", width=2),
        )
    )

    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"]

    for i, (model_name, preds) in enumerate(predictions.items()):
        fig.add_trace(
            go.Scatter(
                x=dates[: len(preds)],
                y=preds,
                mode="lines",
                name=f"{model_name} Prediction",
                line=dict(color=colors[i % len(colors)], width=2, dash="dash"),
            )
        )

    fig.update_layout(
        title="Actual vs Predicted Prices",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400,
        hovermode="x unified",
    )

    return fig


def plot_metrics_comparison(metrics: List[Dict]) -> go.Figure:
    """Create metrics comparison chart"""
    if not metrics:
        return go.Figure()

    df = pd.DataFrame(metrics)

    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("MAE", "RMSE", "MAPE (%)"),
        horizontal_spacing=0.1,
    )

    models = df["model_type"].tolist()
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1"][: len(models)]

    # MAE
    fig.add_trace(
        go.Bar(x=models, y=df["mae"], name="MAE", marker_color=colors), row=1, col=1
    )

    # RMSE
    fig.add_trace(
        go.Bar(x=models, y=df["rmse"], name="RMSE", marker_color=colors), row=1, col=2
    )

    # MAPE
    fig.add_trace(
        go.Bar(x=models, y=df["mape"], name="MAPE", marker_color=colors), row=1, col=3
    )

    fig.update_layout(
        title="Model Performance Comparison",
        template="plotly_dark",
        showlegend=False,
        height=350,
    )

    return fig


def plot_forecast_with_confidence(
    dates: list, predictions: list, lower: list, upper: list
) -> go.Figure:
    """Plot forecast with confidence intervals"""
    fig = go.Figure()

    # Confidence interval
    fig.add_trace(
        go.Scatter(
            x=dates + dates[::-1],
            y=upper + lower[::-1],
            fill="toself",
            fillcolor="rgba(255, 215, 0, 0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Confidence Interval",
        )
    )

    # Predictions
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=predictions,
            mode="lines",
            name="Prediction",
            line=dict(color="#FFD700", width=2),
        )
    )

    fig.update_layout(
        title="Price Forecast with Confidence Interval",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        template="plotly_dark",
        height=400,
    )

    return fig


def plot_volatility(df: pd.DataFrame) -> go.Figure:
    """Plot price volatility"""
    returns = df["gold_price"].pct_change() * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df.index[1:],
            y=returns[1:],
            name="Daily Returns %",
            marker_color=np.where(returns[1:] > 0, "#00FF00", "#FF0000"),
        )
    )

    fig.update_layout(
        title="Daily Returns Volatility",
        xaxis_title="Date",
        yaxis_title="Return (%)",
        template="plotly_dark",
        height=300,
    )

    return fig


def plot_correlation_matrix(df: pd.DataFrame) -> go.Figure:
    """Plot correlation matrix"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr = df[numeric_cols].corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="RdYlGn",
        title="Feature Correlation Matrix",
    )

    fig.update_layout(template="plotly_dark", height=400)

    return fig


def main():
    """Main Streamlit app"""

    # Sidebar
    st.sidebar.title("⚙️ Settings")

    # API Configuration
    api_url = st.sidebar.text_input("API URL", "http://localhost:8000")
    refresh = st.sidebar.checkbox("Auto-refresh", value=False)
    refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 30, 300, 60)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Navigation")

    page = st.sidebar.radio(
        "Go to", ["Dashboard", "Predictions", "Model Comparison", "Data Analysis"]
    )

    # Initialize API client
    api = APIClient(api_url)

    # Auto-refresh
    if refresh:
        import time

        time.sleep(refresh_interval)
        st.rerun()

    # Header
    st.title("📈 Gold Price Forecasting")
    st.markdown("Real-time XAU/USD prediction using Prophet, ARIMA, and LSTM models")

    # Status bar
    health = api.get_health()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        status_color = "🟢" if health.get("status") == "healthy" else "🔴"
        st.metric("API Status", f"{status_color} {health.get('status', 'unknown')}")

    with col2:
        n_models = len(health.get("available_models", []))
        st.metric("Models Loaded", n_models)

    with col3:
        db_status = "✅" if health.get("database_connected") else "❌"
        st.metric("Database", db_status)

    with col4:
        latest_price = health.get("latest_price")
        if latest_price:
            st.metric("Latest Price", f"${latest_price:,.2f}")
        else:
            st.metric("Latest Price", "N/A")

    st.markdown("---")

    if page == "Dashboard":
        # Load data
        data = api.get_latest_data(90)

        if data.get("prices"):
            df = pd.DataFrame(
                {"gold_price": data["prices"]}, index=pd.to_datetime(data["dates"])
            )
        else:
            df = load_sample_data()

        # Price chart
        st.subheader("💰 Gold Price (XAU/USD)")
        st.plotly_chart(plot_price_chart(df), use_container_width=True)

        # Quick stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Current Price", f"${df['gold_price'].iloc[-1]:,.2f}")

        with col2:
            change = df["gold_price"].iloc[-1] - df["gold_price"].iloc[-2]
            st.metric("Daily Change", f"${change:,.2f}")

        with col3:
            st.metric("30-day High", f"${df['gold_price'].max():,.2f}")

        with col4:
            st.metric("30-day Low", f"${df['gold_price'].min():,.2f}")

        # Predictions summary
        st.subheader("🔮 Latest Predictions")

        horizon = st.slider("Forecast Horizon (days)", 1, 30, 7)

        predictions = api.get_predictions("all", horizon)

        if predictions:
            for pred in predictions[:3]:
                with st.expander(f"{pred['model_type'].upper()} Model"):
                    cols = st.columns([2, 1, 1])

                    with cols[0]:
                        if pred.get("predictions"):
                            st.write("Predictions:")
                            for i, p in enumerate(pred["predictions"][:5]):
                                st.write(f"  Day {i + 1}: ${p:,.2f}")

                    with cols[1]:
                        if pred.get("upper_bound"):
                            st.write("Upper Bound:")
                            for i, u in enumerate(pred["upper_bound"][:5]):
                                st.write(f"  Day {i + 1}: ${u:,.2f}")

                    with cols[2]:
                        if pred.get("lower_bound"):
                            st.write("Lower Bound:")
                            for i, l in enumerate(pred["lower_bound"][:5]):
                                st.write(f"  Day {i + 1}: ${l:,.2f}")

        # Volatility chart
        st.subheader("📊 Volatility")
        st.plotly_chart(plot_volatility(df), use_container_width=True)

    elif page == "Predictions":
        st.subheader("🔮 Generate Predictions")

        col1, col2 = st.columns(2)

        with col1:
            model_choice = st.selectbox("Model", ["all", "prophet", "arima", "lstm"])

        with col2:
            horizon = st.slider("Horizon (days)", 1, 30, 7)

        if st.button("Generate Predictions", type="primary"):
            with st.spinner("Fetching predictions..."):
                predictions = api.get_predictions(model_choice, horizon)

                if predictions:
                    for pred in predictions:
                        dates = (
                            pd.date_range(
                                start=datetime.now(), periods=pred["horizon"], freq="D"
                            )
                            .strftime("%Y-%m-%d")
                            .tolist()
                        )

                        fig = plot_forecast_with_confidence(
                            dates,
                            pred["predictions"],
                            pred.get("lower_bound", []),
                            pred.get("upper_bound", []),
                        )

                        st.plotly_chart(fig, use_container_width=True)

                        st.success(
                            f"Generated {pred['horizon']}-day forecast using {pred['model_type']}"
                        )
                else:
                    st.warning("No predictions available. Train models first.")

    elif page == "Model Comparison":
        st.subheader("📊 Model Performance Comparison")

        with st.spinner("Loading metrics..."):
            metrics = api.get_metrics()

            if metrics:
                # Metrics comparison
                st.plotly_chart(
                    plot_metrics_comparison(metrics), use_container_width=True
                )

                # Detailed metrics table
                st.subheader("📋 Detailed Metrics")

                df_metrics = pd.DataFrame(metrics)
                df_metrics = df_metrics.sort_values("mape")

                st.dataframe(
                    df_metrics.style.format(
                        {"mae": "{:.2f}", "rmse": "{:.2f}", "mape": "{:.2f}%"}
                    ),
                    use_container_width=True,
                )

                # Best model highlight
                best_model = df_metrics.iloc[0]["model_type"]
                best_mape = df_metrics.iloc[0]["mape"]

                st.success(
                    f"🏆 Best Model: {best_model.upper()} (MAPE: {best_mape:.2f}%)"
                )
            else:
                st.info(
                    "No metrics available. Train models to see performance comparison."
                )

    elif page == "Data Analysis":
        st.subheader("📈 Data Analysis")

        data = api.get_latest_data(365)

        if data.get("prices"):
            df = pd.DataFrame(
                {"gold_price": data["prices"]}, index=pd.to_datetime(data["dates"])
            )

            # Correlation matrix
            st.plotly_chart(plot_correlation_matrix(df), use_container_width=True)

            # Price distribution
            fig = px.histogram(
                df,
                x="gold_price",
                nbins=50,
                title="Price Distribution",
                color_discrete_sequence=["#FFD700"],
            )
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig, use_container_width=True)

            # Statistics
            st.subheader("📊 Statistics")

            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric("Mean", f"${df['gold_price'].mean():,.2f}")

            with col2:
                st.metric("Median", f"${df['gold_price'].median():,.2f}")

            with col3:
                st.metric("Std Dev", f"${df['gold_price'].std():,.2f}")

            with col4:
                st.metric(
                    "Coef of Var",
                    f"{(df['gold_price'].std() / df['gold_price'].mean() * 100):.2f}%",
                )
        else:
            st.info("No data available for analysis.")


if __name__ == "__main__":
    main()
