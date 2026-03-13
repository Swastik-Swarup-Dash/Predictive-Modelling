import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import src.api as api


STREAMLIT_MODULE_PATH = Path(__file__).resolve().parent.parent / "streamlit" / "app.py"
spec = importlib.util.spec_from_file_location("gold_streamlit_app", STREAMLIT_MODULE_PATH)
streamlit_module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(streamlit_module)


class FutureForecastApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(api.app)

    def test_normalize_future_forecasts(self):
        payload = {
            "lstm": {
                "horizon": 4,
                "dates": ["2026-03-15", "2026-03-16", "2026-03-17", "2026-03-18"],
                "predictions": [1801.2, 1803.1, 1802.7, 1804.5],
            }
        }

        result = api.normalize_future_forecasts(payload, model_type="lstm", horizon=2)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["model_type"], "lstm")
        self.assertEqual(result[0]["horizon"], 2)
        self.assertEqual(result[0]["dates"], ["2026-03-15", "2026-03-16"])

    def test_trained_forecasts_endpoint(self):
        mocked_results = {
            "future_forecasts": {
                "lstm": {
                    "horizon": 3,
                    "dates": ["2026-03-15", "2026-03-16", "2026-03-17"],
                    "predictions": [1801.1, 1802.2, 1803.3],
                }
            }
        }

        with patch("src.api.load_training_results_file", return_value=mocked_results):
            response = self.client.get("/forecasts/trained?model_type=lstm&horizon=2")

        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(len(body), 1)
        self.assertEqual(body[0]["model_type"], "lstm")
        self.assertEqual(body[0]["horizon"], 2)
        self.assertEqual(len(body[0]["predictions"]), 2)

    def test_trained_forecasts_endpoint_404(self):
        mocked_results = {"future_forecasts": {}}

        with patch("src.api.load_training_results_file", return_value=mocked_results):
            response = self.client.get("/forecasts/trained")

        self.assertEqual(response.status_code, 404)


class StreamlitFutureForecastPayloadTests(unittest.TestCase):
    def test_normalize_trained_forecasts_payload(self):
        payload = {
            "lstm": {
                "horizon": 3,
                "dates": ["2026-03-15", "2026-03-16", "2026-03-17"],
                "predictions": [1801.0, 1802.0, 1803.0],
            }
        }

        normalized = streamlit_module.normalize_trained_forecasts_payload(payload)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["model_type"], "lstm")
        self.assertEqual(normalized[0]["horizon"], 3)
        self.assertEqual(len(normalized[0]["dates"]), 3)


if __name__ == "__main__":
    unittest.main()
