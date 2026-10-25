"""Tests for ``solpred.metrics`` (against scikit-learn / SciPy references) and ``solpred.uncertainty``.

Conformal tests use synthetic Gaussian residuals, for which split-conformal intervals must reach the
nominal marginal coverage (up to Monte-Carlo error) and normalised intervals must adapt their width to
the supplied per-sample scale.
"""
from __future__ import annotations

import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from solpred.metrics import (METRICS, bootstrap_ci, paired_bootstrap_mae_diff, regression_metrics,
                             solubility_class)
from solpred.uncertainty import ConformalRegressor, coverage


@pytest.fixture
def yp():
    rng = np.random.default_rng(0)
    y = rng.normal(-3, 2, 300)
    return y, y + rng.normal(0, 0.7, 300)


# --------------------------------------------------------------------------- point metrics
def test_regression_metrics_match_references(yp):
    y, p = yp
    m = regression_metrics(y, p)
    assert m["n"] == 300
    assert m["MAE"] == pytest.approx(mean_absolute_error(y, p))
    assert m["RMSE"] == pytest.approx(np.sqrt(mean_squared_error(y, p)))
    assert m["R2"] == pytest.approx(r2_score(y, p))
    assert m["Pearson"] == pytest.approx(stats.pearsonr(y, p)[0])
    assert m["Spearman"] == pytest.approx(stats.spearmanr(y, p)[0])
    assert set(METRICS) <= set(m)


def test_regression_metrics_accepts_lists_and_is_symmetric_where_expected(yp):
    y, p = yp
    a, b = regression_metrics(list(y), list(p)), regression_metrics(p, y)
    assert a["MAE"] == pytest.approx(b["MAE"]) and a["Pearson"] == pytest.approx(b["Pearson"])
    assert a["R2"] != pytest.approx(b["R2"])                       # R2 is not symmetric


def test_perfect_and_constant_predictions():
    y = np.array([-1.0, -2.0, -3.0, 0.5])
    perfect = regression_metrics(y, y)
    assert perfect["MAE"] == 0 and perfect["R2"] == 1 and perfect["Pearson"] == pytest.approx(1)
    const = regression_metrics(y, np.full(4, y.mean()))
    assert const["R2"] == pytest.approx(0.0)
    assert np.isnan(const["Pearson"]) and np.isnan(const["Spearman"])
    flat = regression_metrics(np.ones(3), np.array([1.0, 2.0, 3.0]))
    assert np.isnan(flat["R2"])                                    # undefined for constant targets


# --------------------------------------------------------------------------- bootstrap
def test_bootstrap_ci_contains_estimate_and_is_seeded(yp):
    y, p = yp
    ci = bootstrap_ci(y, p, n_boot=200, seed=1)
    m = regression_metrics(y, p)
    for k in METRICS:
        lo, hi = ci[k]
        assert lo <= m[k] <= hi
    assert ci == bootstrap_ci(y, p, n_boot=200, seed=1)
    assert ci != bootstrap_ci(y, p, n_boot=200, seed=2)


def test_paired_bootstrap(yp):
    y, p = yp
    same = paired_bootstrap_mae_diff(y, p, p, n_boot=200)
    assert same["mae_diff"] == 0 and same["ci_low"] == 0 and same["ci_high"] == 0
    worse = p + 0.5
    r = paired_bootstrap_mae_diff(y, worse, p, n_boot=500)
    assert r["mae_diff"] == pytest.approx(mean_absolute_error(y, worse) - mean_absolute_error(y, p))
    assert r["ci_low"] > 0 and r["p_a_not_better"] == 1.0
    r2 = paired_bootstrap_mae_diff(y, p, worse, n_boot=500)
    assert r2["ci_high"] < 0 and r2["p_a_not_better"] == 0.0


def test_solubility_class_boundaries():
    got = solubility_class([0.0, 0.5, -0.01, -2.0, -2.01, -4.0, -4.01, -12])
    assert got.tolist() == ["highly_soluble", "highly_soluble", "soluble", "soluble", "slightly_soluble",
                            "slightly_soluble", "insoluble", "insoluble"]


# --------------------------------------------------------------------------- conformal prediction
def _repeated_coverage(n_cal: int, alpha: float, reps: int = 300, n_test: int = 2000, seed: int = 0) -> float:
    """Mean coverage of standard split-conformal intervals over ``reps`` calibration draws (Gaussian residuals)."""
    rng = np.random.default_rng(seed)
    covs = []
    for _ in range(reps):
        p_cal, p_test = rng.normal(size=n_cal), rng.normal(size=n_test)
        cp = ConformalRegressor(normalized=False).fit(p_cal + rng.normal(size=n_cal), p_cal)
        lo, hi = cp.interval(p_test, alpha=alpha)
        covs.append(coverage(p_test + rng.normal(size=n_test), lo, hi))
    return float(np.mean(covs))


@pytest.mark.parametrize("alpha", [0.1, 0.2])
def test_standard_conformal_is_valid(alpha):
    """Marginal coverage >= 1 - alpha (Monte-Carlo s.e. of the mean ~0.002 with 300 x 2000 draws)."""
    assert _repeated_coverage(50, alpha) >= 1 - alpha - 0.006


def test_standard_conformal_width_gaussian():
    rng = np.random.default_rng(42)
    p = rng.normal(size=4000)
    cp = ConformalRegressor(normalized=False).fit(p + rng.normal(size=4000), p)
    for alpha in (0.1, 0.32):
        lo, hi = cp.interval(np.zeros(3), alpha=alpha)
        assert (hi - lo)[0] / 2 == pytest.approx(stats.norm.ppf(1 - alpha / 2), rel=0.06)


def test_known_conformal_exact_finite_sample_coverage():
    n, alpha = 50, 0.1
    k = int(np.ceil((n + 1) * (1 - alpha)))
    assert _repeated_coverage(n, alpha) == pytest.approx(k / (n + 1), abs=0.006)   # 46/51 = 0.902


def test_normalised_conformal_adapts_to_scale():
    rng = np.random.default_rng(7)
    n_cal, n_test = 3000, 20000
    s_cal, s_test = rng.uniform(0.2, 2.0, n_cal), rng.uniform(0.2, 2.0, n_test)
    p_cal, p_test = rng.normal(size=n_cal), rng.normal(size=n_test)
    y_cal, y_test = p_cal + s_cal * rng.normal(size=n_cal), p_test + s_test * rng.normal(size=n_test)
    cp = ConformalRegressor(normalized=True).fit(y_cal, p_cal, scale=s_cal)
    lo, hi = cp.interval(p_test, scale=s_test, alpha=0.1)
    assert coverage(y_test, lo, hi) == pytest.approx(0.9, abs=0.015)
    width = hi - lo
    assert np.corrcoef(width, s_test)[0, 1] > 0.99
    small, large = s_test < 0.6, s_test > 1.6                       # conditional coverage also ~ nominal
    assert coverage(y_test[small], lo[small], hi[small]) == pytest.approx(0.9, abs=0.03)
    assert coverage(y_test[large], lo[large], hi[large]) == pytest.approx(0.9, abs=0.03)


def test_conformal_finite_sample_quantile():
    cp = ConformalRegressor(normalized=False).fit(np.arange(1, 10, dtype=float), np.zeros(9))
    assert cp.quantile(0.1) == 9.0            # ceil(10*0.9)/9 > 1 -> clipped to the max score
    assert cp.quantile(0.5) == 5.0           # ceil((9+1)*0.5) = 5 -> 5th smallest score
    with pytest.raises(ValueError):
        ConformalRegressor(normalized=True).fit([1.0], [0.0])


def test_known_conformal_quantile_order_statistic():
    cp = ConformalRegressor(normalized=False).fit(np.arange(1, 10, dtype=float), np.zeros(9))
    assert cp.quantile(0.5) == 5.0            # ceil((9+1)*0.5) = 5 -> 5th smallest score


def test_coverage_function():
    assert coverage([1, 2, 3, 4], [0, 2.5, 3, 5], [2, 3, 3, 6]) == 0.5
