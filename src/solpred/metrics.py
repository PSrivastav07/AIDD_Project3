"""Regression metrics: MAE (primary), RMSE, R2, Pearson r, Spearman rho, with bootstrap CIs."""
from __future__ import annotations

import numpy as np
from scipy import stats

METRICS = ("MAE", "RMSE", "R2", "Pearson", "Spearman")


def regression_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    err = p - y
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    const = np.allclose(p, p[0]) or np.allclose(y, y[0])
    return {
        "n": int(len(y)),
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "Pearson": float("nan") if const else float(stats.pearsonr(y, p)[0]),
        "Spearman": float("nan") if const else float(stats.spearmanr(y, p)[0]),
    }


def bootstrap_ci(y_true, y_pred, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0) -> dict:
    """Percentile bootstrap CI for every metric: ``{metric: (low, high)}``."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    rng = np.random.default_rng(seed)
    boots = {m: [] for m in METRICS}
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        r = regression_metrics(y[idx], p[idx])
        for m in METRICS:
            boots[m].append(r[m])
    return {m: (float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2))))
            for m, v in boots.items()}


def paired_bootstrap_mae_diff(y_true, pred_a, pred_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """MAE(a) - MAE(b) with a paired bootstrap CI and the fraction of resamples where a is worse."""
    y = np.asarray(y_true, dtype=float)
    ea = np.abs(np.asarray(pred_a, dtype=float) - y)
    eb = np.abs(np.asarray(pred_b, dtype=float) - y)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        diffs[i] = ea[idx].mean() - eb[idx].mean()
    return {"mae_diff": float(ea.mean() - eb.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5)), "p_a_not_better": float((diffs >= 0).mean())}


def solubility_class(logS) -> np.ndarray:
    """AqSolDB solubility classes: >=0 highly soluble, [-2,0) soluble, [-4,-2) slightly soluble, <-4 insoluble."""
    s = np.asarray(logS, dtype=float)
    return np.select([s >= 0, s >= -2, s >= -4], ["highly_soluble", "soluble", "slightly_soluble"], "insoluble")
