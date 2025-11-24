from datetime import datetime
import uvicorn
from fastapi import FastAPI
import logging
from pydantic import BaseModel
import time
import numpy as np
from pathlib import Path
from prometheus_client import Counter, Gauge, Histogram, make_asgi_app
from starlette.middleware.base import BaseHTTPMiddleware
from prometheus_fastapi_instrumentator import Instrumentator

from zrive.module6.src.basket_model.basket_model import BasketModel
from zrive.module6.src.basket_model.feature_store import FeatureStore


logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s - %(asctime)s - %(name)s - %(message)s")


METRICS_FILE = Path("metrics_log.txt")


PREDICTION_REQUESTS_COUNTER = Counter(
    "prediction_requests_total",
    "Total number of prediction requests received"
)

PREDICTIONS_MADE_COUNTER = Counter(
    "predictions_made_total",
    "Total number of predictions made by the model"
)

PREDICTION_LATENCY_HISTOGRAM = Histogram(
    "prediction_latency_seconds",
    "Latency in seconds for prediction requests"
)

LATEST_PREDICTION_VALUE = Gauge(
    "latest_prediction_value",
    "Latest prediction value made by the model"
)

PREDICTION_ERRORS_COUNTER = Counter(
    "prediction_errors_total",
    "Total number of prediction errors"
)

LAST_LATENCY_GAUGE = Gauge(
    "last_prediction_latency_seconds",
    "Latency in seconds for the last prediction request"
)


app = FastAPI()
metrics_app = make_asgi_app()


class RequestSchema(BaseModel):
    user_id: str


class MetricsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        start_time = time.time()
        response = await call_next(request)
        latency = time.time() - start_time

        path = request.url.path
        if path != "/metrics":
            PREDICTION_REQUESTS_COUNTER.inc()
            PREDICTION_LATENCY_HISTOGRAM.observe(latency)
            LAST_LATENCY_GAUGE.set(latency)

        return response


app.add_middleware(MetricsMiddleware)


def format_response(**kwargs) -> dict:
    status_code = kwargs.get("status_code")
    prediction = kwargs.get("prediction")
    user_id = kwargs.get("user_id")
    info = kwargs.get("info", "")

    match status_code:
        case 200:
            response = {
                "status_code": status_code,
                "user_id": user_id,
                "prediction": prediction,
                "info": f"Success: {info}"
            }
        case 400:
            response = {
                "status_code": status_code,
                "user_id": user_id,
                "prediction": None,
                "info": f"Bad Request: {info}"
            }
        case 503:
            response = {
                "status_code": status_code,
                "user_id": user_id,
                "prediction": None,
                "info": f"Service Unavailable: {info}"
            }
        case _:
            response = {
                "status_code": status_code,
                "user_id": user_id,
                "prediction": None,
                "info": f"Error {status_code}: {info}"
            }
    return response


def log_metrics_to_txt(user_id: str,
                       prediction: float,
                       latency: float,
                       status_code: int,
                       tmp_path: Path = METRICS_FILE) -> None:
    requests_count = PREDICTION_REQUESTS_COUNTER._value.get()
    predictions_count = PREDICTIONS_MADE_COUNTER._value.get()

    if status_code != 200:
        PREDICTION_ERRORS_COUNTER.inc()
        prediction = -1.0

    header = "timestamp,user_id,prediction,latency_seconds,total_requests," \
        "total_predictions,status_code\n"
    file_exists = tmp_path.exists()

    with open(tmp_path, "a") as f:
        if not file_exists:
            f.write(header)
        log_line = (
            f"{datetime.now().isoformat()},{user_id},{prediction},"
            f"{latency},{int(requests_count)},{int(predictions_count)},"
            f"{int(status_code)}\n"
        )
        f.write(log_line)


def alternative_heuristic(user_id: str, feature_store: FeatureStore) -> float:
    prior_basket_value = feature_store.feature_store['prior_basket_value'].loc[user_id]
    if not isinstance(prior_basket_value, (float, np.float64)) and len(prior_basket_value) > 1:
        prior_basket_value = prior_basket_value.iloc[-2]
    return float(prior_basket_value)


@app.get("/")
def read_root():
    logging.info("Root endpoint called")
    return {"message": "Hello, World!"}


@app.get("/status")
def read_status():
    logging.info("Status endpoint called")
    return {"status_code": 200}


@app.post("/predict")
def predict(request: RequestSchema):
    tic = time.time()
    logging.info(f"Received prediction request for user_id: {request.user_id}")

    # Load model from model registry
    logging.info("Loading model...")
    model = BasketModel()

    # Load the features from feature store
    logging.info("Loading features...")
    feature_store = FeatureStore()
    logging.info(f"Fetching features for user_id: {request.user_id}")
    features = feature_store.get_features(str(request.user_id))

    # Make prediction
    try:
        prediction = model.predict(features)
    except Exception as e:
        logging.error(f"Prediction failed: {e}")
        logging.info("Using alternative heuristic for prediction...")
        prediction = alternative_heuristic(request.user_id, feature_store)
    status_code = 200
    PREDICTIONS_MADE_COUNTER.inc()

    toc = time.time()
    latency = toc - tic

    # Log performance metrics
    log_metrics_to_txt(request.user_id, prediction, latency, status_code)

    return format_response(
        status_code=status_code,
        user_id=request.user_id,
        prediction=prediction,
        info="Prediction successful."
    )


Instrumentator().instrument(app).expose(app, endpoint="/metrics")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
