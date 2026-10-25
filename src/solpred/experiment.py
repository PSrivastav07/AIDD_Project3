"""Shared train / evaluate / persist helpers used by every experiment script.

Prediction files (one per run) - CSV with columns:
    compound_id, subset ("val" | "test"), y_true, y_pred, y_std (NaN when the model has no spread)
Metric rows (appended to ``<out_dir>/metrics_runs.csv``) - one row per (run, subset="test"):
    experiment, train_dataset, test_dataset, split_type, split_name, seed, featurizer, model,
    n_train, n_val, n_test, MAE, RMSE, R2, Pearson, Spearman, val_MAE, fit_seconds
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from solpred.features import load_dataset, load_or_compute, subset
from solpred.metrics import regression_metrics
from solpred.models import fit_model, get_model, predict_with_std

METRIC_COLUMNS = ["experiment", "train_dataset", "test_dataset", "split_type", "split_name", "seed",
                  "featurizer", "model", "n_train", "n_val", "n_test", "MAE", "RMSE", "R2", "Pearson",
                  "Spearman", "val_MAE", "fit_seconds"]

_LABEL_CACHE: dict = {}


def labels(dataset: str) -> pd.Series:
    if dataset not in _LABEL_CACHE:
        df = load_dataset(dataset)
        _LABEL_CACHE[dataset] = df.set_index("compound_id")["logS"].astype(float)
    return _LABEL_CACHE[dataset]


def xy(dataset: str, featurizer: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X, all_ids, _ = load_or_compute(dataset, featurizer)
    return subset(X, all_ids, ids), labels(dataset).loc[ids].to_numpy()


def run_split(split: dict, featurizer: str, model_name: str, seed: int = 0, params: dict | None = None,
              n_jobs: int = 4, train_dataset: str = "aqsoldb", test_dataset: str | None = None):
    """Fit on ``split['train']`` (val for early stopping) and predict val + test.

    Returns ``(metrics_dict, predictions_df, fitted_model)``.
    """
    test_dataset = test_dataset or split.get("test_dataset", train_dataset)
    train_dataset = split.get("train_dataset", train_dataset)
    Xtr, ytr = xy(train_dataset, featurizer, split["train"])
    Xva, yva = xy(train_dataset, featurizer, split["val"])
    Xte, yte = xy(test_dataset, featurizer, split["test"])
    model = get_model(model_name, seed=seed, n_jobs=n_jobs, params=params)
    t0 = time.time()
    fit_model(model, Xtr, ytr, Xva, yva)
    fit_s = time.time() - t0
    pv, sv = predict_with_std(model, Xva)
    pt, st = predict_with_std(model, Xte)
    preds = pd.concat([
        pd.DataFrame({"compound_id": split["val"], "subset": "val", "y_true": yva, "y_pred": pv,
                      "y_std": sv if sv is not None else np.nan}),
        pd.DataFrame({"compound_id": split["test"], "subset": "test", "y_true": yte, "y_pred": pt,
                      "y_std": st if st is not None else np.nan}),
    ], ignore_index=True)
    m = regression_metrics(yte, pt)
    meta = split.get("meta", {})
    row = {"train_dataset": train_dataset, "test_dataset": test_dataset,
           "split_type": meta.get("split_type"), "seed": meta.get("seed", seed),
           "split_name": meta.get("name") or meta.get("held_out_source") or f"seed{meta.get('seed', seed)}",
           "featurizer": featurizer, "model": model_name, "n_train": len(ytr), "n_val": len(yva),
           "n_test": len(yte), **{k: m[k] for k in ("MAE", "RMSE", "R2", "Pearson", "Spearman")},
           "val_MAE": regression_metrics(yva, pv)["MAE"], "fit_seconds": round(fit_s, 2)}
    return row, preds, model


def prediction_path(out_dir: Path, split_type: str, split_name: str, model_name: str, featurizer: str) -> Path:
    return Path(out_dir) / "predictions" / split_type / split_name / f"{model_name}__{featurizer}.csv"


def save_run(row: dict, preds: pd.DataFrame, out_dir: Path, experiment: str) -> None:
    out_dir = Path(out_dir)
    p = prediction_path(out_dir, row["split_type"], row["split_name"], row["model"], row["featurizer"])
    p.parent.mkdir(parents=True, exist_ok=True)
    preds.to_csv(p, index=False)
    row = {"experiment": experiment, **row}
    mfile = out_dir / "metrics_runs.csv"
    df = pd.DataFrame([row])[METRIC_COLUMNS]
    df.to_csv(mfile, mode="a", header=not mfile.exists(), index=False)


def run_exists(out_dir: Path, split_type: str, split_name: str, model_name: str, featurizer: str) -> bool:
    return prediction_path(out_dir, split_type, split_name, model_name, featurizer).exists()


def summarize(metrics: pd.DataFrame, by=("split_type", "featurizer", "model")) -> pd.DataFrame:
    """Mean and std over seeds for every metric."""
    cols = ["MAE", "RMSE", "R2", "Pearson", "Spearman"]
    g = metrics.groupby(list(by))[cols]
    out = g.mean().add_suffix("_mean").join(g.std().add_suffix("_std")).join(g.size().rename("n_runs"))
    return out.reset_index()
