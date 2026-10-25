"""FastAPI service for the production logS predictor (``solpred.predict.SolubilityPredictor``).

Endpoints
---------
``GET  /health``   liveness + whether the model is loaded
``GET  /model``    the model card (``models/final/metadata.json``)
``POST /predict``  ``{"smiles": ["CCO", ...], "alpha": 0.1}`` -> one result per SMILES, same order. At most 1000
                   SMILES per request; malformed requests (missing/empty list, >1000 items, alpha outside
                   [0.01, 0.5]) are rejected with HTTP 422. Chemically invalid SMILES do *not* fail the request:
                   they come back with ``valid=false`` and an ``error`` / ``warnings`` entry.

Run (from the project root)::

    .venv/Scripts/python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000

The model directory can be changed with the ``SOLPRED_MODEL_DIR`` environment variable.
"""
from __future__ import annotations

import math
import os
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StringConstraints

from solpred.predict import DEFAULT_MODEL_DIR, SolubilityPredictor

MAX_BATCH = 1000
MAX_SMILES_LEN = 2000
_STATE: dict[str, Any] = {"predictor": None, "error": None}


def model_dir() -> str:
    return os.environ.get("SOLPRED_MODEL_DIR", DEFAULT_MODEL_DIR)


def get_predictor() -> SolubilityPredictor:
    """Load the predictor once (lazily, so the app also works without the lifespan hook)."""
    if _STATE["predictor"] is None:
        try:
            _STATE["predictor"] = SolubilityPredictor.load(model_dir())
            _STATE["error"] = None
        except Exception as exc:  # noqa: BLE001 - reported through /health and 503s
            _STATE["error"] = f"{type(exc).__name__}: {exc}"
            raise HTTPException(status_code=503, detail=f"model not available: {_STATE['error']}") from exc
    return _STATE["predictor"]


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        get_predictor()
    except HTTPException:
        pass  # /health reports the problem; /predict answers 503
    yield


app = FastAPI(title="solpred - aqueous solubility (logS) predictor", version="1.0.0", lifespan=lifespan,
              description="LightGBM bagged ensemble trained on curated AqSolDB with scaffold-calibrated conformal "
                          "intervals and an ECFP4 applicability domain.")


class PredictRequest(BaseModel):
    smiles: list[Annotated[str, StringConstraints(max_length=MAX_SMILES_LEN)]] = Field(
        ..., min_length=1, max_length=MAX_BATCH, description=f"1-{MAX_BATCH} SMILES strings")
    alpha: float = Field(0.1, ge=0.01, le=0.5, description="miscoverage; interval level = 1 - alpha")


class Prediction(BaseModel):
    input_smiles: str
    standardized_smiles: Optional[str] = None
    valid: bool
    error: Optional[str] = None
    logS_pred: Optional[float] = None
    logS_std: Optional[float] = None
    pi_low: Optional[float] = None
    pi_high: Optional[float] = None
    pi_level: float
    solubility_class: Optional[str] = None
    solubility_mg_per_L: Optional[float] = None
    ad_maxsim: Optional[float] = None
    ad_in_domain: Optional[bool] = None
    nearest_train_id: Optional[str] = None
    nearest_train_smiles: Optional[str] = None
    nearest_train_logS: Optional[float] = None
    in_training_set: Optional[bool] = None
    warnings: str = ""


class PredictResponse(BaseModel):
    config_hash: Optional[str] = None
    alpha: float
    n: int
    n_valid: int
    predictions: list[Prediction]


def _clean(obj):
    """JSON-safe copy: NaN/inf/NA -> None (FastAPI refuses NaN)."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if obj is pd.NA:
        return None
    return obj


def _records(df: pd.DataFrame) -> list[dict]:
    df = df.astype(object).where(pd.notna(df), None)
    return [_clean(r) for r in df.to_dict(orient="records")]


@app.get("/health")
def health() -> dict:
    loaded = _STATE["predictor"] is not None
    return {"status": "ok" if loaded else "degraded", "model_loaded": loaded, "model_dir": model_dir(),
            "error": _STATE["error"]}


@app.get("/model")
def model_card() -> dict:
    return _clean(get_predictor().metadata)


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    pred = get_predictor()
    df = pred.predict(req.smiles, alpha=req.alpha)
    rows = _records(df)
    return PredictResponse(config_hash=pred.metadata.get("config_hash"), alpha=req.alpha, n=len(rows),
                           n_valid=int(df["valid"].sum()), predictions=[Prediction(**r) for r in rows])
