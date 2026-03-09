"""
Monitoring Module for Drift Detection and Model Performance Tracking
Detects data drift, concept drift, and triggers retraining
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
from scipy import stats
import requests
from sqlalchemy import create_engine, text

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Config:
    """Configuration loader"""

    @staticmethod
    def load() -> Dict[str, Any]:
        config_path = Path(__file__).parent.parent / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)


class DriftDetector:
    """
    Detects data drift and concept drift using statistical tests
    """

    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reference_data = None
        self.baseline_stats = {}

    def set_baseline(self, df: pd.DataFrame, column: str = "gold_price"):
        """
        Set baseline data for comparison

        Args:
            df: Reference DataFrame
            column: Column to analyze
        """
        self.reference_data = df[column].values

        # Calculate baseline statistics
        self.baseline_stats = {
            "mean": np.mean(self.reference_data),
            "std": np.std(self.reference_data),
            "median": np.median(self.reference_data),
            "min": np.min(self.reference_data),
            "max": np.max(self.reference_data),
            "q25": np.percentile(self.reference_data, 25),
            "q75": np.percentile(self.reference_data, 75),
        }

        logger.info(f"Baseline set: mean={self.baseline_stats['mean']:.2f}")

    def detect_distribution_shift(
        self, current_data: np.ndarray, method: str = "ks"
    ) -> Dict[str, Any]:
        """
        Detect distribution shift using Kolmogorov-Smirnov test

        Args:
            current_data: Current data to compare
            method: Test method ('ks', 'mann-whitney')

        Returns:
            Dictionary with test results
        """
        if self.reference_data is None:
            raise ValueError("Baseline not set. Call set_baseline first.")

        if method == "ks":
            statistic, p_value = stats.ks_2samp(self.reference_data, current_data)
            test_name = "Kolmogorov-Smirnov"
        elif method == "mann-whitney":
            statistic, p_value = stats.mannwhitneyu(
                self.reference_data, current_data, alternative="two-sided"
            )
            test_name = "Mann-Whitney U"
        else:
            raise ValueError(f"Unknown method: {method}")

        drift_detected = p_value < self.threshold

        result = {
            "test": test_name,
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": drift_detected,
            "threshold": self.threshold,
        }

        logger.info(f"Drift detection: {result}")

        return result

    def detect_mean_shift(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect mean shift using t-test"""
        if self.reference_data is None:
            raise ValueError("Baseline not set")

        t_stat, p_value = stats.ttest_ind(self.reference_data, current_data)

        current_mean = np.mean(current_data)
        baseline_mean = self.baseline_stats["mean"]
        mean_shift = current_mean - baseline_mean
        mean_shift_pct = (mean_shift / baseline_mean) * 100

        return {
            "test": "T-test for mean shift",
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "drift_detected": p_value < self.threshold,
            "baseline_mean": float(baseline_mean),
            "current_mean": float(current_mean),
            "mean_shift": float(mean_shift),
            "mean_shift_pct": float(mean_shift_pct),
        }

    def detect_volatility_change(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Detect change in volatility using Levene's test"""
        if self.reference_data is None:
            raise ValueError("Baseline not set")

        statistic, p_value = stats.levene(self.reference_data, current_data)

        baseline_std = self.baseline_stats["std"]
        current_std = np.std(current_data)

        return {
            "test": "Levene's test for variance",
            "statistic": float(statistic),
            "p_value": float(p_value),
            "drift_detected": p_value < self.threshold,
            "baseline_std": float(baseline_std),
            "current_std": float(current_std),
            "std_ratio": float(current_std / baseline_std) if baseline_std > 0 else 0,
        }

    def run_full_drift_analysis(self, current_data: np.ndarray) -> Dict[str, Any]:
        """Run complete drift analysis"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "reference_size": len(self.reference_data),
            "current_size": len(current_data),
            "baseline_stats": self.baseline_stats,
            "current_stats": {
                "mean": float(np.mean(current_data)),
                "std": float(np.std(current_data)),
                "median": float(np.median(current_data)),
            },
            "tests": {},
        }

        # Run all tests
        results["tests"]["distribution_shift"] = self.detect_distribution_shift(
            current_data
        )
        results["tests"]["mean_shift"] = self.detect_mean_shift(current_data)
        results["tests"]["volatility_change"] = self.detect_volatility_change(
            current_data
        )

        # Overall drift detection
        any_drift = any(test["drift_detected"] for test in results["tests"].values())

        results["overall_drift_detected"] = any_drift

        return results


class PerformanceMonitor:
    """
    Monitors model performance over time
    Tracks prediction accuracy and detects degradation
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.load()
        self.db_manager = None
        self.performance_history = []

    def connect_database(self):
        """Connect to database for metrics storage"""
        db_config = self.config.get("database", {})

        conn_string = (
            f"postgresql://{db_config.get('user', 'postgres')}:"
            f"{db_config.get('password', 'postgres')}@"
            f"{db_config.get('host', 'localhost')}:"
            f"{db_config.get('port', 5432)}/"
            f"{db_config.get('name', 'gold_forecast')}"
        )

        self.engine = create_engine(conn_string)

    def log_prediction(
        self, model_name: str, predicted: float, actual: float, horizon: int = 1
    ):
        """Log a prediction for later analysis"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "predicted": predicted,
            "actual": actual,
            "horizon": horizon,
            "error": abs(predicted - actual),
            "pct_error": abs(predicted - actual) / actual * 100 if actual > 0 else 0,
        }

        self.performance_history.append(entry)

        # Save to database
        try:
            if not self.engine:
                self.connect_database()

            df = pd.DataFrame([entry])
            df.to_sql("predictions", self.engine, if_exists="append", index=False)
        except Exception as e:
            logger.warning(f"Failed to log prediction: {e}")

    def calculate_performance_metrics(
        self, model_name: str, window_days: int = 7
    ) -> Dict[str, Any]:
        """Calculate performance metrics over a time window"""
        if not self.performance_history:
            return {}

        # Filter by model and time window
        cutoff = datetime.now() - timedelta(days=window_days)

        recent_predictions = [
            p
            for p in self.performance_history
            if p["model_name"] == model_name
            and datetime.fromisoformat(p["timestamp"]) > cutoff
        ]

        if not recent_predictions:
            return {}

        errors = [p["error"] for p in recent_predictions]
        pct_errors = [p["pct_error"] for p in recent_predictions]

        return {
            "model_name": model_name,
            "window_days": window_days,
            "n_predictions": len(recent_predictions),
            "mae": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
            "mape": float(np.mean(pct_errors)),
            "max_error": float(np.max(errors)),
            "std_error": float(np.std(errors)),
        }

    def check_performance_degradation(
        self, model_name: str, baseline_mape: float, degradation_threshold: float = 0.2
    ) -> Dict[str, Any]:
        """
        Check if model performance has degraded

        Args:
            model_name: Name of model
            baseline_mape: Baseline MAPE to compare against
            degradation_threshold: Threshold for degradation (e.g., 0.2 = 20% worse)

        Returns:
            Dictionary with degradation analysis
        """
        current_metrics = self.calculate_performance_metrics(model_name)

        if not current_metrics:
            return {"status": "insufficient_data"}

        current_mape = current_metrics["mape"]
        degradation = (
            (current_mape - baseline_mape) / baseline_mape if baseline_mape > 0 else 0
        )

        return {
            "model_name": model_name,
            "baseline_mape": baseline_mape,
            "current_mape": current_mape,
            "degradation_pct": float(degradation * 100),
            "degradation_detected": degradation > degradation_threshold,
            "recommendation": "retrain"
            if degradation > degradation_threshold
            else "ok",
        }


class MonitoringService:
    """
    Main monitoring service that coordinates drift detection and performance monitoring
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or Config.load()
        self.drift_detector = DriftDetector(
            threshold=self.config.get("monitoring", {}).get("drift_threshold", 0.05)
        )
        self.performance_monitor = PerformanceMonitor(config)
        self.retrain_triggered = False

    def run_monitoring_cycle(self, current_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete monitoring cycle

        Args:
            current_data: Current DataFrame

        Returns:
            Monitoring results
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "drift_analysis": {},
            "performance_analysis": {},
            "actions": [],
        }

        # Check for data drift
        if self.drift_detector.reference_data is None:
            self.drift_detector.set_baseline(current_data)
            results["actions"].append("baseline_established")
        else:
            drift_results = self.drift_detector.run_full_drift_analysis(
                current_data["gold_price"].values
            )
            results["drift_analysis"] = drift_results

            if drift_results["overall_drift_detected"]:
                results["actions"].append("retrain_required_drift")

        # Check performance
        for model_name in ["prophet", "arima", "lstm"]:
            perf = self.performance_monitor.calculate_performance_metrics(model_name)
            if perf:
                results["performance_analysis"][model_name] = perf

                # Check for degradation
                baseline_mape = self._get_baseline_mape(model_name)
                if baseline_mape:
                    degradation = (
                        self.performance_monitor.check_performance_degradation(
                            model_name, baseline_mape
                        )
                    )
                    if degradation.get("degradation_detected"):
                        results["actions"].append(
                            f"retrain_required_performance_{model_name}"
                        )

        # Decide on retraining
        retrain_frequency = self.config.get("monitoring", {}).get(
            "retrain_frequency", "weekly"
        )

        if any("retrain_required" in action for action in results["actions"]):
            if not self.retrain_triggered:
                results["actions"].append("trigger_retraining")
                self.retrain_triggered = True
                logger.warning("Retraining triggered due to drift or degradation")

        return results

    def _get_baseline_mape(self, model_name: str) -> Optional[float]:
        """Get baseline MAPE for a model"""
        results_path = Path(__file__).parent.parent / "training_results.json"

        if not results_path.exists():
            return None

        with open(results_path, "r") as f:
            results = json.load(f)

        metrics = results.get("metrics", {}).get(model_name, {})

        return metrics.get("mape")

    def generate_monitoring_report(
        self, current_data: pd.DataFrame, output_path: Optional[str] = None
    ) -> str:
        """Generate monitoring report"""
        results = self.run_monitoring_cycle(current_data)

        report = f"""
Gold Price Forecasting - Monitoring Report
==========================================
Generated: {results["timestamp"]}

Drift Analysis
--------------
"""

        if results["drift_analysis"]:
            drift = results["drift_analysis"]
            report += f"Distribution Shift: {drift.get('tests', {}).get('distribution_shift', {}).get('drift_detected', False)}\n"
            report += f"Mean Shift: {drift.get('tests', {}).get('mean_shift', {}).get('mean_shift_pct', 0):.2f}%\n"
            report += f"Volatility Change: {drift.get('tests', {}).get('volatility_change', {}).get('std_ratio', 1):.2f}x\n"

        report += "\nPerformance Analysis\n--------------------\n"

        for model_name, perf in results.get("performance_analysis", {}).items():
            report += f"{model_name}: MAPE={perf['mape']:.2f}%, MAE={perf['mae']:.2f}\n"

        report += "\nActions\n-------\n"

        for action in results.get("actions", []):
            report += f"- {action}\n"

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            logger.info(f"Report saved to {output_path}")

        return report


def send_slack_alert(message: str, webhook_url: Optional[str] = None):
    """Send alert to Slack"""
    if not webhook_url:
        webhook_url = os.environ.get("SLACK_WEBHOOK_URL")

    if not webhook_url:
        logger.warning("Slack webhook not configured")
        return

    try:
        requests.post(webhook_url, json={"text": message})
        logger.info("Slack alert sent")
    except Exception as e:
        logger.error(f"Failed to send Slack alert: {e}")


def run_scheduled_monitoring():
    """Run scheduled monitoring check"""
    from src.data_fetch import fetch_all_data

    # Fetch latest data
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")

    try:
        data = fetch_all_data(start_date, end_date)

        # Run monitoring
        monitor = MonitoringService()
        results = monitor.run_monitoring_cycle(data)

        # Generate report
        report = monitor.generate_monitoring_report(data)

        # Alert if issues detected
        if any("retrain" in action for action in results.get("actions", [])):
            send_slack_alert(f"🚨 Model Retraining Required\n\n{report}")

        # Save results
        output_path = Path(__file__).parent.parent / "monitoring_results.json"
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)

        logger.info("Monitoring cycle complete")

        return results

    except Exception as e:
        logger.error(f"Monitoring failed: {e}")
        send_slack_alert(f"❌ Monitoring Failed: {str(e)}")
        raise


if __name__ == "__main__":
    run_scheduled_monitoring()
