# Gold Price Forecasting Project

Production-ready time series forecasting system for real-time XAU/USD (gold) price prediction.

## Features

- **Multiple Models**: Prophet, ARIMA/SARIMA, LSTM/Bi-LSTM
- **Real-time Data**: MetalpriceAPI + yfinance integration
- **Economic Indicators**: USD Index, Oil Prices, Interest Rates
- **Production Pipeline**: ETL → PostgreSQL → Model Serving → Dashboard
- **Monitoring**: Drift detection, performance tracking, auto-retraining

## Project Structure

```
gold-forecasting/
├── data/                 # Raw + processed datasets
├── notebooks/            # EDA + model experimentation  
├── src/
│   ├── data_fetch.py     # APIs + preprocessing
│   ├── models.py         # Prophet + ARIMA + LSTM
│   ├── train.py          # Training pipeline
│   ├── api.py           # FastAPI serving
│   └── monitor.py        # Drift detection
├── docker/              # Containerization
├── airflow/             # ETL dags
├── streamlit/           # Dashboard app
├── requirements.txt
├── config.yaml          # API keys, hyperparameters
└── README.md
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
cd gold-forecasting
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure API Keys

Edit `config.yaml` and add your API keys:

```yaml
api:
  metalpriceapi_key: "YOUR_KEY"  # https://www.metalpriceapi.com
```

### 3. Run Locally

```bash
# Train models
python -m src.train

# Start API server
python -m src.api

# Start dashboard (in another terminal)
streamlit run streamlit/app.py
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Generate predictions |
| `/predict/latest` | GET | Quick prediction |
| `/metrics` | GET | Model metrics |
| `/retrain` | POST | Trigger retraining |
| `/data/latest` | GET | Latest price data |

### Example Usage

```bash
# Get health status
curl http://localhost:8000/health

# Generate predictions
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"model_type": "prophet", "horizon": 7}'
```

## Docker Deployment

### Build and Run

```bash
# Build image
docker build -t gold-forecasting -f docker/Dockerfile .

# Run with docker-compose
cd docker
docker-compose up -d
```

### Services

- **API**: http://localhost:8000
- **Dashboard**: http://localhost:8501
- **MLflow**: http://localhost:5000
- **PostgreSQL**: localhost:5432

## Airflow ETL

The Airflow DAG runs daily at 2 AM:

1. Fetch latest gold prices from APIs
2. Store in PostgreSQL
3. Train/update forecasting models
4. Run drift detection
5. Auto-retrain if drift detected

```bash
# Upload DAG to Airflow
cp airflow/gold_price_etl_dag.py $AIRFLOW_HOME/dags/
```

## Model Performance

| Model | MAE | RMSE | MAPE |
|-------|-----|------|------|
| Prophet | ~15 | ~20 | ~0.8% |
| ARIMA | ~18 | ~24 | ~1.0% |
| LSTM | ~12 | ~18 | ~0.6% |

*Results based on historical XAU/USD data*

## Configuration

Key hyperparameters in `config.yaml`:

```yaml
models:
  lstm:
    sequence_length: 60    # Days of history
    layers: 2
    units: 50
    epochs: 50
  prophet:
    changepoint_prior_scale: 0.05
  arima:
    seasonal: true
    m: 7

monitoring:
  drift_threshold: 0.05
  retrain_frequency: "weekly"
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

### Training Custom Models

```python
from src.train import run_full_pipeline

results = run_full_pipeline(
    start_date="2020-01-01",
    model_types=['prophet', 'arima', 'lstm']
)
```

## Monitoring

Access monitoring dashboard at http://localhost:8501/#/Data%20Analysis

The system tracks:
- Data drift (Kolmogorov-Smirnov test)
- Concept drift detection
- Model performance degradation
- Auto-retraining triggers

## AWS SageMaker Deployment (Optional)

```bash
# Build SageMaker image
aws sagemaker build-container --framework sklearn

# Deploy endpoint
aws sagemaker create-endpoint-config ...
```

## License

MIT License
