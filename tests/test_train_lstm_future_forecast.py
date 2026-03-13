import unittest
from typing import Any

import numpy as np
import pandas as pd

from src.train import ModelTrainer


class _DummyModel:
    def __init__(self, values):
        self._values = np.array(values)

    def predict(self, horizon: int):
        return self._values[:horizon]

    def predict_multiple(self, df: pd.DataFrame, horizon: int):
        return self._values[:horizon]


class TrainLSTMFutureForecastTests(unittest.TestCase):
    def _build_trainer(self) -> ModelTrainer:
        config = {
            "training": {"min_samples": 10, "test_size": 0.2},
            "forecast": {"horizon": 5},
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "gold_forecast",
                "user": "postgres",
                "password": "postgres",
            },
        }
        return ModelTrainer(config=config)

    def test_build_future_dates(self):
        last_date = pd.Timestamp("2026-03-10")
        dates = ModelTrainer._build_future_dates(last_date, 3)

        self.assertEqual(dates, ["2026-03-11", "2026-03-12", "2026-03-13"])

    def test_generate_future_forecast_for_lstm(self):
        trainer = self._build_trainer()
        history_index = pd.date_range("2026-01-01", periods=20, freq="D")
        history_df = pd.DataFrame({"gold_price": np.linspace(1800, 1820, 20)}, index=history_index)

        model = _DummyModel([1830.0, 1831.5, 1833.2, 1835.1])
        payload = trainer._generate_future_forecast("lstm", model, history_df, 3)

        self.assertEqual(payload["model_type"], "lstm")
        self.assertEqual(payload["horizon"], 3)
        self.assertEqual(len(payload["dates"]), 3)
        self.assertEqual(payload["predictions"], [1830.0, 1831.5, 1833.2])

    def test_generate_future_forecast_handles_empty_history(self):
        trainer = self._build_trainer()
        empty_df = pd.DataFrame(columns=["gold_price"])
        model = _DummyModel([1, 2, 3])

        payload = trainer._generate_future_forecast("lstm", model, empty_df, 2)

        self.assertEqual(payload["dates"], [])
        self.assertEqual(payload["predictions"], [])


if __name__ == "__main__":
    unittest.main()
