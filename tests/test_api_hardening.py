import unittest
from unittest.mock import patch

from fastapi import HTTPException
from starlette.requests import Request

import src.api as api


class ApiHardeningTests(unittest.TestCase):
    def _build_request(self, path: str = "/predict", host: str = "127.0.0.1") -> Request:
        scope = {
            "type": "http",
            "method": "GET",
            "path": path,
            "headers": [],
            "client": (host, 12345),
            "scheme": "http",
            "server": ("testserver", 80),
        }
        return Request(scope)

    def test_get_client_id_prefers_x_forwarded_for(self):
        request = self._build_request()
        request.scope["headers"] = [(b"x-forwarded-for", b"1.2.3.4, 5.6.7.8")]

        client_id = api._get_client_id(request)

        self.assertEqual(client_id, "1.2.3.4")

    def test_rate_limit_raises_after_threshold(self):
        api._rate_limit_store.clear()

        with patch("src.api.time.time", side_effect=[1000.0, 1000.1, 1000.2]):
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)
            with self.assertRaises(HTTPException) as context:
                api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)

        self.assertEqual(context.exception.status_code, 429)

    def test_rate_limit_window_resets(self):
        api._rate_limit_store.clear()

        with patch("src.api.time.time", side_effect=[1000.0, 1000.1, 1061.0, 1061.1]):
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)
            api.enforce_rate_limit("test-client", max_requests=2, window_seconds=60)

    def test_require_api_token_allows_when_token_not_set(self):
        request = self._build_request(path="/predict")

        with patch("src.api.API_TOKEN", ""):
            api.require_api_token(request=request, credentials=None)

    def test_require_api_token_rejects_invalid_token(self):
        request = self._build_request(path="/predict")

        with patch("src.api.API_TOKEN", "secret-token"):
            with self.assertRaises(HTTPException) as context:
                api.require_api_token(request=request, credentials=None)

        self.assertEqual(context.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
