from unittest.mock import patch
from pathlib import Path

import pytest

from zrive.module6.src.app import format_response, log_metrics_to_txt


# ---------- FIXTURES ----------

@pytest.fixture(scope="class")
def mock_request_results():
    def _build_response_data(
        status_code: int,
        prediction: float,
        user_id: str = "user123",
        info: str = "Test info",
    ):
        return status_code, user_id, prediction, info

    return _build_response_data


@pytest.fixture(scope="class")
def mock_metrics():
    def _build_mock_metrics(
        status_code: int,
        user_id: str = "user123",
        prediction: float = 0.5,
        latency: float = 7.21,
    ):
        return status_code, user_id, prediction, latency

    return _build_mock_metrics


# ---------- TESTS FORMAT_RESPONSE ----------

class TestFormatResponse:
    def test_format_response_200(self, mock_request_results):
        status_code, user_id, prediction, info = mock_request_results(
            status_code=200,
            prediction=0.85,
        )

        response = format_response(
            status_code=status_code,
            user_id=user_id,
            prediction=prediction,
            info=info,
        )

        assert response["status_code"] == 200
        assert response["user_id"] == user_id
        assert response["prediction"] == prediction
        assert response["info"] == f"Success: {info}"

    def test_format_response_400(self, mock_request_results):
        status_code, user_id, prediction, info = mock_request_results(
            status_code=400,
            prediction=None,
        )

        response = format_response(
            status_code=status_code,
            user_id=user_id,
            prediction=prediction,
            info=info,
        )

        assert response["status_code"] == 400
        assert response["user_id"] == user_id
        assert response["prediction"] is None
        assert response["info"] == f"Bad Request: {info}"

    def test_format_response_503(self, mock_request_results):
        status_code, user_id, prediction, info = mock_request_results(
            status_code=503,
            prediction=None,
        )

        response = format_response(
            status_code=status_code,
            user_id=user_id,
            prediction=prediction,
            info=info,
        )

        assert response["status_code"] == 503
        assert response["user_id"] == user_id
        assert response["prediction"] is None
        assert response["info"] == f"Service Unavailable: {info}"

    def test_format_response_other(self, mock_request_results):
        status_code, user_id, prediction, info = mock_request_results(
            status_code=500,
            prediction=None,
        )

        response = format_response(
            status_code=status_code,
            user_id=user_id,
            prediction=prediction,
            info=info,
        )

        assert response["status_code"] == 500
        assert response["user_id"] == user_id
        assert response["prediction"] is None
        assert response["info"] == f"Error {status_code}: {info}"


# ---------- TESTS LOG_METRICS_TO_TXT ----------

class TestLogMetricsToTxt:
    def test_log_file_exists(self, tmp_path: Path, mock_metrics):
        status_code, user_id, prediction, latency = mock_metrics(status_code=200)
        log_file_path = tmp_path / "test_metrics_log.txt"

        log_metrics_to_txt(
            user_id=user_id,
            prediction=prediction,
            latency=latency,
            status_code=status_code,
            tmp_path=log_file_path,
        )

        assert log_file_path.exists()
        assert log_file_path.stat().st_size > 0

    def test_case_200_no_inc_counter(self, tmp_path: Path, mock_metrics):
        status_code, user_id, prediction, latency = mock_metrics(status_code=200)
        log_file_path = tmp_path / "test_metrics_log.txt"

        with patch(
            "zrive.module6.src.app.PREDICTION_ERRORS_COUNTER"
        ) as mock_counter:
            log_metrics_to_txt(
                user_id=user_id,
                prediction=prediction,
                latency=latency,
                status_code=status_code,
                tmp_path=log_file_path,
            )

        mock_counter.inc.assert_not_called()

    def test_case_not_200_inc_counter(self, tmp_path: Path, mock_metrics):
        status_code, user_id, prediction, latency = mock_metrics(status_code=400)
        log_file_path = tmp_path / "test_metrics_log.txt"

        with patch(
            "zrive.module6.src.app.PREDICTION_ERRORS_COUNTER"
        ) as mock_counter:
            log_metrics_to_txt(
                user_id=user_id,
                prediction=prediction,
                latency=latency,
                status_code=status_code,
                tmp_path=log_file_path,
            )

        mock_counter.inc.assert_called_once()

    def test_case_not_200_prediction_set_to_minus_one(
        self, tmp_path: Path, mock_metrics
    ):
        status_code, user_id, prediction, latency = mock_metrics(status_code=500)
        log_file_path = tmp_path / "test_metrics_log.txt"

        log_metrics_to_txt(
            user_id=user_id,
            prediction=prediction,
            latency=latency,
            status_code=status_code,
            tmp_path=log_file_path,
        )

        with open(log_file_path, "r") as f:
            lines = f.readlines()
            last_line = lines[-1]
            logged_prediction = float(last_line.split(",")[2])

        assert logged_prediction == -1.0
