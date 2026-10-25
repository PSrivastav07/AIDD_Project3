"""Tests for the FastAPI service (``app/api.py``) and a smoke run of the Streamlit dashboard.

Need the trained model in ``models/final``; the API client (and therefore the model) is created once per module.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from solpred.config import ROOT

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_DIR = Path(os.environ.get("SOLPRED_MODEL_DIR", ROOT / "models" / "final"))  # override for ad-hoc runs
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.joblib").exists(),
                                reason="production model not trained (run scripts/11_train_final_model.py)")


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_model_card(client):
    r = client.get("/model")
    assert r.status_code == 200
    meta = r.json()
    assert meta["dataset"]["n_train"] > 9000
    assert {"MAE", "RMSE", "R2"} <= set(meta["cv"]["metrics"]["overall"])
    assert meta["config_hash"]


def test_predict_valid_and_invalid(client):
    r = client.post("/predict", json={"smiles": ["CCO", "not_a_smiles", "c1ccccc1"], "alpha": 0.1})
    assert r.status_code == 200
    body = json.loads(r.text)  # strict JSON: no NaN tokens
    assert body["n"] == 3 and body["n_valid"] == 2
    p = body["predictions"]
    assert [x["input_smiles"] for x in p] == ["CCO", "not_a_smiles", "c1ccccc1"]
    assert p[0]["valid"] and p[0]["pi_low"] <= p[0]["logS_pred"] <= p[0]["pi_high"]
    assert p[0]["pi_level"] == pytest.approx(0.9)
    assert not p[1]["valid"] and p[1]["logS_pred"] is None and "invalid SMILES" in p[1]["warnings"]
    assert p[0]["logS_pred"] > p[2]["logS_pred"]


def test_default_alpha(client):
    r = client.post("/predict", json={"smiles": ["CCO"]})
    assert r.status_code == 200 and r.json()["alpha"] == pytest.approx(0.1)


@pytest.mark.parametrize("payload", [
    {"smiles": []},                                   # empty list
    {"smiles": ["CCO"] * 1001},                       # too many
    {"smiles": ["CCO"], "alpha": 0.9},                # alpha out of range
    {"smiles": ["CCO"], "alpha": "abc"},              # wrong type
    {"alpha": 0.1},                                   # missing field
    {"smiles": "CCO"},                                # not a list
    {"smiles": [1, 2]},                               # non-string items
])
def test_validation_errors(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_max_batch_accepted(client):
    r = client.post("/predict", json={"smiles": ["CCO", "CCN"] * 500})
    assert r.status_code == 200 and r.json()["n"] == 1000


def test_dashboard_runs():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app" / "dashboard.py"), default_timeout=180).run()
    assert not at.exception
    at.text_area[0].input("CCO ethanol\nc1ccccc1 benzene\nxyz( broken").run()
    at.button[0].click().run()
    assert not at.exception
    df = at.dataframe[0].value
    assert len(df) == 3 and df["name"].tolist() == ["ethanol", "benzene", "broken"]
    assert {m.label: m.value for m in at.metric}["predicted"] == "2"
