import unittest

import numpy as np
import pandas as pd

from src.train import ModelTrainer


class TrainPipelineValidationTests(unittest.TestCase):
    def _build_trainer(self, min_samples: int = 10) -> ModelTrainer:
        config = {
            "training": {
                "test_size": 0.2,
                "validation_size": 0.1,
                "min_samples": min_samples,
                "max_missing_ratio": 0.2,
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "name": "gold_forecast",
                "user": "postgres",
                "password": "postgres",
            },
        }
        return ModelTrainer(config=config)

    def _sample_df(self, rows: int = 50) -> pd.DataFrame:
        dates = pd.date_range("2024-01-01", periods=rows, freq="D")
        prices = 1800 + np.arange(rows)
        return pd.DataFrame({"gold_price": prices}, index=dates)

    def test_split_data_train_test(self):
        trainer = self._build_trainer(min_samples=10)
        df = self._sample_df(50)

        train_df, test_df = trainer.split_data(df)

        self.assertEqual(len(train_df), 40)
        self.assertEqual(len(test_df), 10)
        self.assertGreater(train_df.index.max(), train_df.index.min())

    def test_split_data_train_val_test(self):
        trainer = self._build_trainer(min_samples=10)
        df = self._sample_df(50)

        train_df, val_df, test_df = trainer.split_data(df, include_validation=True)

        self.assertEqual(len(train_df), 35)
        self.assertEqual(len(val_df), 5)
        self.assertEqual(len(test_df), 10)

    def test_split_data_requires_gold_price_column(self):
        trainer = self._build_trainer(min_samples=10)
        dates = pd.date_range("2024-01-01", periods=20, freq="D")
        df = pd.DataFrame({"close": np.arange(20)}, index=dates)

        with self.assertRaises(ValueError):
            trainer.split_data(df)

    def test_split_data_enforces_minimum_samples(self):
        trainer = self._build_trainer(min_samples=30)
        df = self._sample_df(20)

        with self.assertRaises(ValueError):
            trainer.split_data(df)


if __name__ == "__main__":
    unittest.main()
