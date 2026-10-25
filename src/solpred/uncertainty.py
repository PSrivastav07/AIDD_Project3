"""Split-conformal prediction intervals for regression.

Standard conformal uses the absolute residual |y - y_hat| as non-conformity score and yields
constant-width intervals. Normalised conformal divides the residual by a per-compound difficulty
estimate (e.g. ensemble std or a nearest-neighbour distance), giving adaptive widths. Under
exchangeability of calibration and test compounds, the interval covers the truth with probability
>= 1 - alpha.

Stable API (used by the production predictor)::

    cp = ConformalRegressor(normalized=True).fit(y_cal, pred_cal, scale=std_cal)
    low, high = cp.interval(pred_test, scale=std_test, alpha=0.1)
"""
from __future__ import annotations

import numpy as np


class ConformalRegressor:
    def __init__(self, normalized: bool = True, eps: float = 1e-3):
        self.normalized = normalized
        self.eps = eps

    def _scale(self, scale, n: int) -> np.ndarray:
        if not self.normalized:
            return np.ones(n)
        if scale is None:
            raise ValueError("normalised conformal needs a per-sample scale")
        return np.asarray(scale, dtype=float) + self.eps

    def fit(self, y_true, y_pred, scale=None) -> "ConformalRegressor":
        y = np.asarray(y_true, dtype=float)
        p = np.asarray(y_pred, dtype=float)
        self.scores_ = np.abs(y - p) / self._scale(scale, len(y))
        self.n_cal_ = len(y)
        return self

    def quantile(self, alpha: float) -> float:
        """Finite-sample corrected (1 - alpha) quantile of the calibration scores."""
        n = self.n_cal_
        k = int(np.ceil((n + 1) * (1 - alpha)))  # rank of the k-th smallest score
        if k > n:  # too few calibration points for this alpha: the interval is unbounded
            return float("inf")
        return float(np.sort(self.scores_)[k - 1])

    def interval(self, y_pred, scale=None, alpha: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
        p = np.asarray(y_pred, dtype=float)
        half = self.quantile(alpha) * self._scale(scale, len(p))
        return p - half, p + half


def coverage(y_true, low, high) -> float:
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= low) & (y <= high)))


# =========================================================================== extensions (uncertainty study)
# Everything below is additive: ConformalRegressor / coverage above are unchanged.
from scipy import stats as _stats  # noqa: E402

DEFAULT_ALPHAS = (0.2, 0.1, 0.05)


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float)


def gaussian_nll(y_true, y_pred, sigma, floor: float = 1e-3) -> float:
    """Mean Gaussian negative log-likelihood of ``y_true`` under N(y_pred, sigma^2)."""
    r = _arr(y_true) - _arr(y_pred)
    s = np.maximum(_arr(sigma), floor)
    return float(np.mean(0.5 * np.log(2 * np.pi * s ** 2) + r ** 2 / (2 * s ** 2)))


def fit_sigma_scale(y_true, y_pred, sigma, floor: float = 1e-3) -> float:
    """Maximum-likelihood factor ``c`` for N(y_pred, (c * sigma)^2): ``c^2 = mean(r^2 / sigma^2)``.

    Fit it on a validation/calibration subset, then apply ``c * sigma`` to the test subset.
    """
    r = _arr(y_true) - _arr(y_pred)
    s = np.maximum(_arr(sigma), floor)
    return float(np.sqrt(np.mean((r / s) ** 2)))


def coverage_curve(y_true, y_pred, sigma, levels=None) -> tuple[np.ndarray, np.ndarray]:
    """Expected vs observed coverage of central Gaussian intervals ``y_pred +- z_{(1+p)/2} sigma``."""
    levels = np.linspace(0.01, 0.99, 99) if levels is None else _arr(levels)
    z = np.abs(_arr(y_true) - _arr(y_pred)) / np.maximum(_arr(sigma), 1e-12)
    q = _stats.norm.ppf(0.5 + levels / 2)
    observed = (z[None, :] <= q[:, None]).mean(axis=1)
    return levels, observed


def miscalibration_area(y_true, y_pred, sigma, levels=None) -> float:
    """Mean |observed - expected| coverage over the levels (approximates the area between the curves)."""
    lv, obs = coverage_curve(y_true, y_pred, sigma, levels)
    return float(np.mean(np.abs(obs - lv)))


def sparsification_curve(abs_err, sigma, fractions=None) -> dict:
    """MAE of the retained compounds after removing the fraction ``f`` with the largest ``sigma``.

    Returns ``{"fractions", "sigma", "oracle", "random"}``; the oracle removes by the true |error|, the
    random reference is the full-set MAE (expected value of random removal).
    """
    e = _arr(abs_err)
    fractions = np.round(np.arange(0.0, 0.905, 0.01), 4) if fractions is None else _arr(fractions)
    n = len(e)
    by_sigma = e[np.argsort(-_arr(sigma), kind="stable")]
    by_oracle = np.sort(e)[::-1]
    cs_sig = np.cumsum(by_sigma[::-1])[::-1]  # cs[i] = sum of elements i..n-1
    cs_orc = np.cumsum(by_oracle[::-1])[::-1]
    k = np.minimum(np.round(fractions * n).astype(int), n - 1)
    keep = n - k
    return {"fractions": fractions, "sigma": cs_sig[k] / keep, "oracle": cs_orc[k] / keep,
            "random": np.full(len(fractions), e.mean())}


def sparsification_metrics(abs_err, sigma, fractions=None) -> dict:
    """AUSE (mean excess MAE of the sigma curve over the oracle curve across the removal range, log units)
    and the sparsification gain ``(A_random - A_sigma) / (A_random - A_oracle)`` (1 = oracle, 0 = random)."""
    c = sparsification_curve(abs_err, sigma, fractions)
    f = c["fractions"]
    span = f[-1] - f[0]
    area = {k: float(np.trapezoid(c[k], f) / span) for k in ("sigma", "oracle", "random")}
    denom = area["random"] - area["oracle"]
    return {"AUSE": area["sigma"] - area["oracle"],
            "sparsification_gain": (area["random"] - area["sigma"]) / denom if denom > 0 else float("nan"),
            "MAE_retained_50pct": float(np.interp(0.5, f, c["sigma"]))}


def error_flag_auroc(abs_err, sigma, threshold: float = 1.0) -> float:
    """AUROC of ``sigma`` as a score for flagging compounds with |error| > ``threshold``."""
    from sklearn.metrics import roc_auc_score

    flag = _arr(abs_err) > threshold
    if flag.all() or not flag.any():
        return float("nan")
    return float(roc_auc_score(flag, _arr(sigma)))


def uncertainty_quality(y_true, y_pred, sigma, y_val=None, p_val=None, sigma_val=None,
                        error_threshold: float = 1.0, floor: float = 1e-3) -> dict:
    """Ranking (Spearman, AUROC, sparsification) and calibration (after a val-fitted scale) metrics."""
    y, p, s = _arr(y_true), _arr(y_pred), _arr(sigma)
    e = np.abs(y - p)
    out = {"n": int(len(y)), "MAE": float(e.mean()),
           "spearman_sigma_abs_err": float(_stats.spearmanr(s, e)[0]) if np.ptp(s) > 0 else float("nan"),
           "auroc_err_gt_thr": error_flag_auroc(e, s, error_threshold),
           **sparsification_metrics(e, s)}
    if y_val is not None:
        c = fit_sigma_scale(y_val, p_val, sigma_val, floor)
        sc = c * np.maximum(s, floor)
        out.update({"scale_factor": c, "NLL": gaussian_nll(y, p, sc, floor),
                    "miscalibration_area": miscalibration_area(y, p, sc),
                    "coverage_68": float(np.mean(e <= sc)), "coverage_95": float(np.mean(e <= 1.959964 * sc))})
    return out


def conformal_evaluate(y_cal, p_cal, y_test, p_test, scale_cal=None, scale_test=None,
                       alphas=DEFAULT_ALPHAS) -> list[dict]:
    """Split-conformal intervals (standard if ``scale_cal`` is None, else normalised) at several alphas:
    empirical coverage and mean / median interval width on the test compounds."""
    normalized = scale_cal is not None
    cp = ConformalRegressor(normalized=normalized).fit(y_cal, p_cal, scale=scale_cal)
    rows = []
    for a in alphas:
        lo, hi = cp.interval(p_test, scale=scale_test, alpha=a)
        w = hi - lo
        rows.append({"alpha": a, "nominal": 1 - a, "coverage": coverage(y_test, lo, hi),
                     "mean_width": float(np.mean(w)), "median_width": float(np.median(w)),
                     "q_score": cp.quantile(a), "n_cal": cp.n_cal_, "n_test": int(len(_arr(p_test)))})
    return rows
