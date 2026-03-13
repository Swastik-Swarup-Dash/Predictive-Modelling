import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = (
    Path(__file__).resolve().parent.parent / "streamlit" / "app.py"
)


spec = importlib.util.spec_from_file_location("gold_streamlit_app", MODULE_PATH)
module = importlib.util.module_from_spec(spec)
assert spec is not None and spec.loader is not None
spec.loader.exec_module(module)


class StreamlitMetricsNormalizationTests(unittest.TestCase):
    def test_scalar_metrics_dict_is_normalized(self):
        payload = {
            "model_type": "arima",
            "mae": 10.5,
            "rmse": 12.1,
            "mape": 1.8,
        }

        normalized = module.normalize_metrics_payload(payload)

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0]["model_type"], "arima")
        self.assertAlmostEqual(normalized[0]["mape"], 1.8)

    def test_nested_metrics_dict_is_normalized(self):
        payload = {
            "metrics": {
                "prophet": {"mae": 11, "rmse": 14, "mape": 2.1},
                "arima": {"mae": 9, "rmse": 13, "mape": 1.7},
            }
        }

        normalized = module.normalize_metrics_payload(payload)

        self.assertEqual(len(normalized), 2)
        model_names = sorted([row["model_type"] for row in normalized])
        self.assertEqual(model_names, ["arima", "prophet"])

    def test_invalid_error_payload_returns_empty(self):
        payload = {"detail": "No training results found"}

        normalized = module.normalize_metrics_payload(payload)

        self.assertEqual(normalized, [])


if __name__ == "__main__":
    unittest.main()
