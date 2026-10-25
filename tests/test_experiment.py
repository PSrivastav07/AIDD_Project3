"""Tests for ``solpred.experiment`` (train / evaluate / persist) and the model registry it drives.

``run_split`` is exercised on a tiny synthetic linear problem by monkeypatching the feature/label
loaders inside ``solpred.experiment``, so the tests are fast and independent of the real data; one
smoke test runs the real pipeline on a 400-compound slice of a stored split (skipped without caches).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import solpred.experiment as ex
from conftest import FEATURES, N_THREADS, SPLITS, require
from solpred.metrics import regression_metrics

N, D = 160, 6


def _synthetic(seed: int, shift: float = 0.0):
    rng = np.random.RandomState(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)
    w = np.array([1.0, -0.5, 0.25, 0.0, 0.8, -1.2])
    y = X @ w - 3.0 + shift + rng.normal(0, 0.1, N)
    ids = np.array([f"S-{i}" for i in range(N)])
    return X, ids, [f"f{j}" for j in range(D)], pd.Series(y, index=ids)


@pytest.fixture
def synth(monkeypatch):
    """Two synthetic datasets ('synth', 'synth2': same ids, labels shifted by +10) behind the loaders."""
    data = {"synth": _synthetic(0), "synth2": _synthetic(0, shift=10.0)}
    monkeypatch.setattr(ex, "load_or_compute", lambda dataset, kind, cfg=None: data[dataset][:3])
    monkeypatch.setattr(ex, "labels", lambda dataset: data[dataset][3])
    return data


def _split(name_meta: dict | None = None) -> dict:
    ids = [f"S-{i}" for i in range(N)]
    return {"train": ids[:100], "val": ids[100:130], "test": ids[130:],
            "meta": name_meta or {"split_type": "random", "seed": 3}}


def test_run_split_linear_recovers_signal(synth):
    sp = _split()
    row, preds, model = ex.run_split(sp, "any", "esol_linear", seed=3, n_jobs=N_THREADS, train_dataset="synth")
    assert row["MAE"] < 0.2 and row["R2"] > 0.98
    assert (row["n_train"], row["n_val"], row["n_test"]) == (100, 30, 30)
    assert row["split_type"] == "random" and row["split_name"] == "seed3" and row["seed"] == 3
    assert row["train_dataset"] == row["test_dataset"] == "synth"
    assert list(preds.columns) == ["compound_id", "subset", "y_true", "y_pred", "y_std"]
    assert preds["subset"].tolist() == ["val"] * 30 + ["test"] * 30
    assert preds["compound_id"].tolist() == sp["val"] + sp["test"]
    y = synth["synth"][3]
    np.testing.assert_allclose(preds["y_true"], y.loc[sp["val"] + sp["test"]].to_numpy())
    assert preds["y_std"].isna().all()
    te = preds[preds["subset"] == "test"]
    assert row["MAE"] == pytest.approx(regression_metrics(te["y_true"], te["y_pred"])["MAE"])
    va = preds[preds["subset"] == "val"]
    assert row["val_MAE"] == pytest.approx(regression_metrics(va["y_true"], va["y_pred"])["MAE"])


def test_mean_model_predicts_train_mean(synth):
    sp = _split()
    row, preds, _ = ex.run_split(sp, "any", "mean", train_dataset="synth")
    np.testing.assert_allclose(preds["y_pred"], synth["synth"][3].loc[sp["train"]].mean())


def test_rf_returns_tree_spread(synth):
    row, preds, model = ex.run_split(_split(), "any", "rf", seed=0, n_jobs=N_THREADS,
                                     params={"n_estimators": 25}, train_dataset="synth")
    assert np.isfinite(preds["y_std"]).all() and (preds["y_std"] >= 0).all()
    assert row["MAE"] < 1.0


def test_lgbm_early_stops_on_val(synth):
    row, preds, model = ex.run_split(_split(), "any", "lgbm", seed=0, n_jobs=N_THREADS,
                                     params={"n_estimators": 300, "learning_rate": 0.1, "num_leaves": 7,
                                             "min_child_samples": 5}, train_dataset="synth")
    assert 0 < model.best_iteration_ <= 300
    assert np.isfinite(preds["y_pred"]).all()


def test_run_split_is_deterministic(synth):
    a = ex.run_split(_split(), "any", "rf", seed=1, n_jobs=N_THREADS, params={"n_estimators": 15},
                     train_dataset="synth")[1]
    b = ex.run_split(_split(), "any", "rf", seed=1, n_jobs=N_THREADS, params={"n_estimators": 15},
                     train_dataset="synth")[1]
    pd.testing.assert_frame_equal(a, b)


def test_external_split_uses_test_dataset_labels(synth):
    sp = {**_split({"split_type": "external", "name": "synth_to_synth2"}),
          "train_dataset": "synth", "test_dataset": "synth2"}
    row, preds, _ = ex.run_split(sp, "any", "esol_linear")
    assert row["train_dataset"] == "synth" and row["test_dataset"] == "synth2"
    assert row["split_name"] == "synth_to_synth2"
    te = preds[preds["subset"] == "test"]
    np.testing.assert_allclose(te["y_true"], synth["synth2"][3].loc[sp["test"]].to_numpy())
    assert row["MAE"] == pytest.approx(10.0, abs=0.3)               # label shift is fully visible


def test_loso_split_name(synth):
    row, _, _ = ex.run_split(_split({"split_type": "loso", "seed": 0, "held_out_source": "B"}), "any", "mean",
                             train_dataset="synth")
    assert row["split_name"] == "B"


def test_save_run_and_summarize(synth, tmp_path):
    rows = []
    for seed in (0, 1):
        row, preds, _ = ex.run_split(_split({"split_type": "random", "seed": seed}), "any", "esol_linear",
                                     seed=seed, train_dataset="synth")
        assert not ex.run_exists(tmp_path, "random", f"seed{seed}", "esol_linear", "any")
        ex.save_run(row, preds, tmp_path, "unit")
        assert ex.run_exists(tmp_path, "random", f"seed{seed}", "esol_linear", "any")
        rows.append(row)
    m = pd.read_csv(tmp_path / "metrics_runs.csv")
    assert list(m.columns) == ex.METRIC_COLUMNS and len(m) == 2      # header written once
    assert (m["experiment"] == "unit").all()
    saved = pd.read_csv(ex.prediction_path(tmp_path, "random", "seed1", "esol_linear", "any"))
    assert len(saved) == 60
    s = ex.summarize(m)
    assert len(s) == 1 and s.loc[0, "n_runs"] == 2
    assert s.loc[0, "MAE_mean"] == pytest.approx(np.mean([r["MAE"] for r in rows]))
    assert s.loc[0, "MAE_std"] == pytest.approx(np.std([r["MAE"] for r in rows], ddof=1))


def test_run_split_real_smoke():
    """End-to-end on real cached esol4 features: a 300/50/50 slice of random seed 0."""
    require(FEATURES / "aqsoldb__esol4.npz", SPLITS / "aqsoldb" / "random" / "seed0.json")
    from solpred.splits import load_split

    full = load_split("random", 0)
    sp = {"train": full["train"][:300], "val": full["val"][:50], "test": full["test"][:50], "meta": full["meta"]}
    row, preds, _ = ex.run_split(sp, "esol4", "esol_linear", seed=0, n_jobs=1)
    assert row["n_test"] == 50 and 0.3 < row["MAE"] < 2.0
    lab = ex.labels("aqsoldb")
    np.testing.assert_allclose(preds["y_true"], lab.loc[sp["val"] + sp["test"]].to_numpy())
