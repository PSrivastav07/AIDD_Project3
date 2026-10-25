"""Tests for the production predictor (``solpred.predict``).

They need the trained artefact in ``models/final`` (``scripts/11_train_final_model.py``); the model is loaded
once per module and every prediction batch is computed once.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from solpred.config import ROOT
from solpred.metrics import solubility_class
from solpred.predict import OUTPUT_COLUMNS, SolubilityPredictor, main

MODEL_DIR = Path(os.environ.get("SOLPRED_MODEL_DIR", ROOT / "models" / "final"))  # override for ad-hoc runs
pytestmark = pytest.mark.skipif(not (MODEL_DIR / "model.joblib").exists(),
                                reason="production model not trained (run scripts/11_train_final_model.py)")

SMILES = {
    "ethanol": "CCO",
    "benzene": "c1ccccc1",
    "naphthalene": "c1ccc2ccccc2c1",
    "anthracene": "c1ccc2cc3ccccc3cc2c1",
    "garbage": "not_a_smiles",
    "empty": "",
    "unclosed_ring": "C1CC",
    "cisplatin": "N[Pt](N)(Cl)Cl",
    "sodium_benzoate": "[Na+].O=C([O-])c1ccccc1",
    "ethanol_noncanonical": "OCC",
    "cubane": "C12C3C4C1C5C2C3C45",
}
NAMES = list(SMILES)


@pytest.fixture(scope="module")
def predictor() -> SolubilityPredictor:
    return SolubilityPredictor.load(MODEL_DIR)


@pytest.fixture(scope="module")
def batch(predictor) -> pd.DataFrame:
    return predictor.predict(list(SMILES.values()), alpha=0.1).set_index(pd.Index(NAMES))


def test_schema_and_order(batch):
    assert list(batch.columns) == OUTPUT_COLUMNS
    assert batch["input_smiles"].tolist() == list(SMILES.values())
    assert (batch["pi_level"] == 0.9).all()


def test_invalid_smiles_do_not_crash_batch(batch):
    for name in ("garbage", "empty", "unclosed_ring"):
        row = batch.loc[name]
        assert not row["valid"]
        assert np.isnan(row["logS_pred"]) and np.isnan(row["pi_low"])
        assert "invalid SMILES" in row["warnings"]
        assert isinstance(row["error"], str) and row["error"]
    rej = batch.loc["cisplatin"]
    assert not rej["valid"] and "rejected by curation rules" in rej["warnings"]
    ok = batch.drop(index=["garbage", "empty", "unclosed_ring", "cisplatin"])
    assert ok["valid"].all() and ok["logS_pred"].notna().all()


def test_interval_contains_prediction(batch):
    ok = batch[batch["valid"]]
    assert (ok["pi_low"] <= ok["logS_pred"]).all() and (ok["logS_pred"] <= ok["pi_high"]).all()
    assert (ok["logS_std"] >= 0).all()
    assert (ok["pi_high"] - ok["pi_low"] > 0).all()


def test_lower_alpha_gives_wider_interval(predictor, batch):
    wide = predictor.predict([SMILES["naphthalene"]], alpha=0.05).iloc[0]
    narrow = batch.loc["naphthalene"]
    assert wide["pi_high"] - wide["pi_low"] > narrow["pi_high"] - narrow["pi_low"]
    assert wide["logS_pred"] == pytest.approx(narrow["logS_pred"])


def test_training_compound_has_maxsim_one(predictor, batch):
    i = 17
    smi, cid = predictor.ad.train_smiles_[i], predictor.ad.train_ids_[i]
    row = predictor.predict([smi]).iloc[0]
    assert row["ad_maxsim"] == pytest.approx(1.0)
    assert bool(row["ad_in_domain"]) and bool(row["in_training_set"])
    assert row["nearest_train_id"] == cid
    assert "in training set" in row["warnings"]
    # a non-canonical spelling standardises to the same structure
    assert batch.loc["ethanol_noncanonical", "ad_maxsim"] == pytest.approx(1.0)
    assert batch.loc["ethanol_noncanonical", "logS_pred"] == pytest.approx(batch.loc["ethanol", "logS_pred"])


def test_solubility_ordering(batch):
    p = batch["logS_pred"]
    assert p["ethanol"] > p["benzene"] > p["naphthalene"] > p["anthracene"]


def test_flags_and_derived_columns(batch):
    assert "multicomponent" in batch.loc["sodium_benzoate", "warnings"]
    ok = batch[batch["valid"]]
    assert ok["solubility_class"].tolist() == solubility_class(ok["logS_pred"]).tolist()
    eth = batch.loc["ethanol"]
    assert eth["solubility_mg_per_L"] == pytest.approx(10 ** eth["logS_pred"] * 46.069 * 1000, rel=1e-3)
    flagged_ood = ok[ok["ad_in_domain"] == False]  # noqa: E712 - nullable boolean column
    assert all("outside applicability domain" in w for w in flagged_ood["warnings"])


def test_edge_inputs(predictor):
    empty = predictor.predict([])
    assert list(empty.columns) == OUTPUT_COLUMNS and len(empty) == 0
    with pytest.raises(ValueError):
        predictor.predict(["CCO"], alpha=1.5)
    none_row = predictor.predict([None]).iloc[0]
    assert not none_row["valid"]


def test_cli_roundtrip(predictor, tmp_path, monkeypatch):
    monkeypatch.setattr(SolubilityPredictor, "load", classmethod(lambda cls, *a, **k: predictor))
    src = tmp_path / "in.csv"
    pd.DataFrame({"name": ["ethanol", "bad"], "smiles": ["CCO", "xyz("]}).to_csv(src, index=False)
    out = tmp_path / "out.csv"
    assert main(["--input", str(src), "--output", str(out), "--alpha", "0.2"]) == 0
    res = pd.read_csv(out)
    assert res.columns[0] == "name" and list(res.columns[1:]) == OUTPUT_COLUMNS
    assert res["valid"].tolist() == [True, False]
    assert res["pi_level"].iloc[0] == pytest.approx(0.8)
