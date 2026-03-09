"""
Airflow DAG for Daily Gold Price ETL Pipeline
Fetches data, trains models, and runs monitoring
"""

from datetime import datetime, timedelta
from pathlib import Path

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.utils.task_group import TaskGroup
from airflow.models import Variable

default_args = {
    "owner": "data-science",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def fetch_gold_prices(**context):
    """Fetch latest gold prices from APIs"""
    import sys

    sys.path.insert(0, "/opt/airflow/plugins")

    from src.data_fetch import fetch_all_data, DatabaseManager

    config = {
        "api": {"metalpriceapi_key": Variable.get("metalpriceapi_key", default_var="")},
        "database": {
            "host": Variable.get("db_host", default_var="postgres"),
            "port": 5432,
            "name": "gold_forecast",
            "user": "postgres",
            "password": Variable.get("db_password", default_var="postgres"),
        },
    }

    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")

    df = fetch_all_data(start_date, end_date)

    db_manager = DatabaseManager(config)
    db_manager.connect()
    db_manager.create_tables()
    db_manager.save_gold_prices(df)

    return f"Fetched {len(df)} records"


def train_models(**context):
    """Train forecasting models"""
    import sys

    sys.path.insert(0, "/opt/airflow/plugins")

    from src.train import run_full_pipeline

    results = run_full_pipeline(model_types=["prophet", "arima", "lstm"])

    return f"Trained models. Best: {results.get('best_model')}"


def run_monitoring(**context):
    """Run drift detection and monitoring"""
    import sys

    sys.path.insert(0, "/opt/airflow/plugins")

    from src.monitor import run_scheduled_monitoring

    results = run_scheduled_monitoring()

    actions = results.get("actions", [])
    if any("retrain" in action for action in actions):
        return "retrain_required"

    return "no_action_required"


def check_drift(**context):
    """Check if drift was detected"""
    ti = context["ti"]
    result = ti.xcom_pull(task_ids="run_monitoring")

    if result == "retrain_required":
        return "retrain_models"

    return "end"


def retrain_models(**context):
    """Retrain models after drift detection"""
    import sys

    sys.path.insert(0, "/opt/airflow/plugins")

    from src.train import run_full_pipeline

    run_full_pipeline(model_types=["prophet", "arima", "lstm"])

    return "Models retrained successfully"


with DAG(
    "gold_price_etl",
    default_args=default_args,
    description="Daily ETL pipeline for gold price forecasting",
    schedule_interval="0 2 * * *",  # Run at 2 AM daily
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=["gold", "forecasting", "ml"],
) as dag:
    start = EmptyOperator(task_id="start")

    with TaskGroup("data_ingestion"):
        create_tables = PostgresOperator(
            task_id="create_tables",
            postgres_conn_id="postgres_default",
            sql="""
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
                
                CREATE TABLE IF NOT EXISTS predictions (
                    id SERIAL PRIMARY KEY,
                    date DATE NOT NULL,
                    model_name VARCHAR(50) NOT NULL,
                    predicted_price DECIMAL(12, 2),
                    actual_price DECIMAL(12, 2),
                    horizon INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS model_versions (
                    id SERIAL PRIMARY KEY,
                    model_name VARCHAR(50) NOT NULL,
                    version VARCHAR(20) NOT NULL,
                    metrics JSONB,
                    trained_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    is_active BOOLEAN DEFAULT TRUE
                );
            """,
        )

        fetch_data = PythonOperator(
            task_id="fetch_gold_prices",
            python_callable=fetch_gold_prices,
            provide_context=True,
        )

        create_tables >> fetch_data

    with TaskGroup("model_training"):
        train = PythonOperator(
            task_id="train_models", python_callable=train_models, provide_context=True
        )

    with TaskGroup("monitoring"):
        monitor = PythonOperator(
            task_id="run_monitoring",
            python_callable=run_monitoring,
            provide_context=True,
        )

        check = BranchPythonOperator(
            task_id="check_drift", python_callable=check_drift, provide_context=True
        )

        retrain = PythonOperator(
            task_id="retrain_models",
            python_callable=retrain_models,
            provide_context=True,
        )

        end_monitoring = EmptyOperator(task_id="end_monitoring")

        monitor >> check
        check >> retrain >> end_monitoring
        check >> end_monitoring

    end = EmptyOperator(task_id="end", trigger_rule="none_failed_or_skipped")

    start >> create_tables >> train >> monitor >> end
