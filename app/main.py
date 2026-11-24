"""Machine Guard prediction API.

Start it from the repo root:

    API_KEY=<your key> uvicorn app.main:app --host 0.0.0.0 --port 8000

Then open /docs in the browser to try it.

Endpoints:
    GET  /health   no key needed: is the service up, and which model is loaded?
    POST /predict  needs the X-API-Key header: failure probability for one machine run
"""
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.logging_utils import log_event, save_prediction, setup_logging
from app.schemas import Health, Prediction, SensorReading
from app.security import require_api_key
from machine_guard.features import RAW_FEATURES

THRESHOLD = 0.5
APP_VERSION = os.environ.get("APP_VERSION", "dev")  # baked into the image: docker build --build-arg APP_VERSION=v2

setup_logging()


@lru_cache(maxsize=1)
def load_model():
    """Load the model and its metadata once, the first time they are needed.

    MODEL_PATH is read here, not at import time, so tests can point it at another file.
    model_meta.json sits next to the model file.
    """
    model_path = Path(os.environ.get("MODEL_PATH", "models/model.joblib"))
    model = joblib.load(model_path)  # FileNotFoundError here means the model file is missing
    meta_path = model_path.parent / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.setdefault("model_version", "unknown")
    return model, meta


@asynccontextmanager
async def lifespan(app: FastAPI):
    _, meta = load_model()  # load at startup, so a missing model fails loudly and early
    log_event("startup", model_version=meta["model_version"],
              app_version=APP_VERSION,
              api_key_set=bool(os.environ.get("API_KEY")))
    yield


app = FastAPI(
    title="Machine Guard",
    description="Predicts whether a machine run will fail, from six sensor readings.",
    version="1.0",
    lifespan=lifespan,
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Write one JSON log line for every request, including rejected ones (401, 422)."""
    request.state.request_id = uuid.uuid4().hex
    start = time.perf_counter()
    response = await call_next(request)
    fields = {
        "request_id": request.state.request_id,
        "method": request.method,
        "path": request.url.path,
        "status": response.status_code,
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
        "app_version": APP_VERSION,
    }
    fields.update(getattr(request.state, "log_fields", {}))  # prediction details, if any
    log_event("request", **fields)
    response.headers["X-Request-ID"] = request.state.request_id
    response.headers["X-App-Version"] = APP_VERSION  # lets you see which image answered
    return response


@app.get("/health", response_model=Health)
def health():
    try:
        _, meta = load_model()
    except Exception as err:  # the service is up, but cannot predict
        return JSONResponse(status_code=503, content={"status": "error", "detail": str(err)})
    # TODO 1: return a dict with "status" set to "ok" and "model_version" taken from meta.
    # Example of the answer: {"status": "ok", "model_version": "machine-guard-model-3"}
    return {"status": "TODO", "model_version": "TODO"}


@app.post("/predict", response_model=Prediction, dependencies=[Depends(require_api_key)])
def predict(reading: SensorReading, request: Request):
    start = time.perf_counter()
    model, meta = load_model()

    # The pipeline expects a table with the six raw columns, in training order.
    row = pd.DataFrame([reading.model_dump()], columns=RAW_FEATURES)

    # TODO 2: ask the model for the failure probability and turn it into a 0/1 prediction.
    #   probability = float(model.predict_proba(row)[0, 1])   # column 1 = "fail"
    #   prediction = 1 if the probability is at or above THRESHOLD, else 0 (use int(...))
    # Then DELETE the raise line below: it is only there until you have done this TODO.
    raise NotImplementedError("TODO 2 in app/main.py: compute probability and prediction")

    latency_ms = round((time.perf_counter() - start) * 1000, 2)

    result = {
        "failure_probability": round(probability, 4),
        "prediction": prediction,
        "threshold": THRESHOLD,
        "model_version": meta["model_version"],
    }
    # TODO 3: save this prediction in the SQLite log. Select the 9 lines below (save_prediction
    # down to "})") and press Ctrl+/ to uncomment them all at once.
    # save_prediction({
    #     "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
    #     "request_id": request.state.request_id,
    #     **reading.model_dump(),
    #     "failure_probability": result["failure_probability"],
    #     "prediction": prediction,
    #     "model_version": meta["model_version"],
    #     "latency_ms": latency_ms,
    # })
    # Picked up by the middleware, so the request's log line also says what was predicted.
    request.state.log_fields = {"model_version": meta["model_version"],
                                "failure_probability": result["failure_probability"],
                                "prediction": prediction}
    return result
