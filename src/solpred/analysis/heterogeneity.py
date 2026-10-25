"""Measurement heterogeneity, data reliability and dataset shift in aqueous-solubility data.

Background
----------
AqSolDB merges nine source datasets (ID letters A-I). For a compound that occurs in several sources it keeps
ONE value and stores the number of distinct reports (``n_occurrences``) and their spread (``aqsol_sd``),
from which the reliability groups G1-G5 are defined. A forensic check against the ESOL values that AqSolDB
partly inherited shows that ``aqsol_sd`` is the *population* standard deviation (ddof = 0): for two reports
|x1 - x2| / aqsol_sd = 2.000 for most two-report compounds (it would be sqrt(2) for ddof = 1, and 1 if the
kept value were the mean). See :func:`sd_convention_check`.

Methods
-------
1. **Intrinsic measurement noise.** Let s_i be the ddof=0 SD over the n_i reports of compound i.

   * naive RMS SD           sqrt(mean_i s_i^2)                       (the figure quoted in the data card)
   * unbiased pooled SD     sqrt(sum_i n_i s_i^2 / sum_i (n_i - 1))   (per-report noise sigma, ddof-corrected)
   * robust SD (pairs)      median_i |x1 - x2| / (sqrt(2) * 0.6745),  |x1 - x2| = 2 s_i for n_i = 2
     (insensitive to the gross errors - unit/transcription mistakes - that dominate the RMS)

   If a label is one report with additive Gaussian noise of SD sigma, a *perfect* model of the true value
   still has RMSE = sigma and MAE = sigma * sqrt(2/pi) against the labels (the "error floor"). A
   distribution-free MAE floor from pairs is E|x1 - x2| / sqrt(2) (exact for Gaussian noise).
   Assumptions: noise independent of the compound, reports are independent measurements, and the noise of
   multi-report compounds is representative of single-report (G1) compounds. The last two are doubtful:
   multi-report compounds are better-studied chemicals, and some "reports" are copies of the same
   measurement passed between compilations (near-zero SD), which biases sigma downwards.
   Multi-report compounds are measurably unrepresentative of the single-report (G1) majority: smaller
   (mean MW 233 vs 273 g/mol) and far more often ESOL members (32% vs 6%). The ddof-corrected pooled SD
   (0.645) is dominated by a few gross G2/G4 discrepancies (unit/transcription errors), so it is not a
   usable floor. Recommended headline floors: the G3+G5 pooled SD (~0.25, a lower bound because the groups
   are truncated at SD <= 0.5) and the distribution-free pair MAE floor (~0.36); the RMS SD (0.478) is
   reported as the overall inter-report spread, not as a noise floor.
2. **Error vs reliability.** Reference (Phase-1) test-set predictions are reduced to one absolute error per
   compound (mean over the seeds in which it was a test compound), so bootstrap CIs resample independent
   compounds. Groups are compared with Kruskal-Wallis tests, bootstrap differences in MAE, and an OLS
   model of |error| that adjusts for nearest-neighbour similarity to the training set and logS.
3. **Training-data quality ablation.** LightGBM is refit with (b) inconsistent multi-report compounds
   (G2 + G4) removed from TRAIN, (c) our de-duplication conflicts removed, (d) inverse-variance weights
   w_i = 1 / (sd_i^2 + s0^2), where sd_i = aqsol_sd for multi-report compounds and sd_i = s0 (the pooled
   noise) for single-report compounds, i.e. single reports get w = 1/(2 s0^2) and consistent repeats are
   up-weighted, (e) a size-matched random removal (control for b) and (f) the IVW weights randomly permuted
   over the training compounds (control for d). s0 = 0.478 is a single dataset-level constant: the naive
   RMS of ``aqsol_sd`` over ALL multi-report AqSolDB compounds (SD metadata shipped with AqSolDB, computed
   once, identical for every split/seed). It uses no test-set labels - only the per-compound report spread,
   which is metadata - although test compounds' SDs enter this global average (one scalar; a per-split
   recomputation was not run to avoid refitting). Validation and test sets are never changed. Effects are paired MAE differences vs the full-data model (a) with paired bootstrap CIs;
   seeds are pooled with a bootstrap stratified by seed.
4. **Cross-source transfer.** Leave-one-source-out (LOSO) errors are compared with (i) the same compounds'
   errors in random-split test sets and (ii) a size-matched in-distribution control (K-fold over the held-out
   source, training set subsampled to the LOSO training size) so that the source-shift penalty is not
   confounded with training-set size. A source-to-source matrix trains on equal-size samples of one source.
5. **External transfer** between AqSolDB and ESOL, including Delaney's published equation
   logS = 0.16 - 0.63 cLogP - 0.0062 MW + 0.066 RB - 0.74 AP (here with Crippen cLogP).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from solpred.metrics import regression_metrics

GROUP_ORDER = ["G1", "G2", "G3", "G4", "G5"]
HIGH_RELIABILITY = ("G3", "G5")   # several reports that agree (SD <= 0.5)
LOW_RELIABILITY = ("G2", "G4")    # several reports that disagree (SD > 0.5)
REL3_ORDER = ["single (G1)", "consistent (G3+G5)", "inconsistent (G2+G4)"]
SD_BIN_EDGES = [0.0, 0.1, 0.25, 0.5, 1.0, np.inf]
SD_BIN_LABELS = ["single report", "(0, 0.1]", "(0.1, 0.25]", "(0.25, 0.5]", "(0.5, 1]", "> 1"]
OCC_BIN_LABELS = ["1", "2", "3", "4", ">=5"]
SIM_BIN_EDGES = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0001]
DELANEY_COEF = {"intercept": 0.16, "clogp": -0.63, "mw": -0.0062, "rotatable_bonds": 0.066,
                "aromatic_proportion": -0.74}
Z_MAD = 0.6744897501960817  # median of |N(0, 1)|


# =========================================================================== generic statistics
def bootstrap_stat(fn, *arrays, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float]:
    """Percentile bootstrap CI of ``fn(*arrays)``; rows of all arrays are resampled jointly."""
    arrs = [np.asarray(a) for a in arrays]
    n = len(arrs[0])
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[b] = fn(*(a[idx] for a in arrs))
    return float(np.nanpercentile(vals, 100 * alpha / 2)), float(np.nanpercentile(vals, 100 * (1 - alpha / 2)))


def metrics_with_bias(y_true, y_pred) -> dict:
    """The five standard metrics plus the mean signed error ``bias = mean(y_pred - y_true)``.

    A positive bias means the model over-predicts solubility on that set.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    return {**regression_metrics(y, p), "bias": float(np.mean(p - y))}


def mae_ci(ae, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """MAE (mean of per-compound absolute errors) with a percentile bootstrap CI."""
    ae = np.asarray(ae, dtype=float)
    lo, hi = bootstrap_stat(np.mean, ae, n_boot=n_boot, seed=seed)
    return float(ae.mean()), lo, hi


def mae_diff_unpaired(ae_a, ae_b, n_boot: int = 2000, seed: int = 0) -> dict:
    """MAE(a) - MAE(b) for two independent groups of compounds, bootstrap CI and two-sided p-value."""
    a = np.asarray(ae_a, dtype=float)
    b = np.asarray(ae_b, dtype=float)
    rng = np.random.default_rng(seed)
    d = np.array([a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
                  for _ in range(n_boot)])
    p = 2 * min((d <= 0).mean(), (d >= 0).mean())
    return {"mae_diff": float(a.mean() - b.mean()), "ci_low": float(np.percentile(d, 2.5)),
            "ci_high": float(np.percentile(d, 97.5)), "p_boot": float(min(1.0, p))}


def pooled_paired_delta(blocks: list[tuple[np.ndarray, np.ndarray, np.ndarray]], n_boot: int = 2000,
                        seed: int = 0) -> dict:
    """Mean over blocks (seeds) of MAE(b) - MAE(a) on paired predictions, bootstrap stratified by block.

    ``blocks`` = [(y, pred_a, pred_b), ...]; each block is resampled independently (compounds within a seed),
    and the per-block differences are averaged, so every seed has equal weight.
    """
    return pooled_paired_delta_ae([(np.abs(pa - y), np.abs(pb - y)) for y, pa, pb in blocks], n_boot=n_boot,
                                  seed=seed)


def pooled_paired_delta_ae(errs: list[tuple[np.ndarray, np.ndarray]], n_boot: int = 2000, seed: int = 0) -> dict:
    """As :func:`pooled_paired_delta` but from paired per-compound absolute errors ``[(ae_a, ae_b), ...]``.

    Because MAE is linear in the per-compound errors, ``ae_b`` may be the mean |error| over several replicate
    models (e.g. random-removal controls); the delta is then the replicate-averaged MAE difference and the
    bootstrap still resamples each compound once (replicates share the resampled compounds).
    """
    errs = [(np.asarray(ea, dtype=float), np.asarray(eb, dtype=float)) for ea, eb in errs]
    point = float(np.mean([eb.mean() - ea.mean() for ea, eb in errs]))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        ds = []
        for ea, eb in errs:
            idx = rng.integers(0, len(ea), len(ea))
            ds.append(eb[idx].mean() - ea[idx].mean())
        boots[i] = np.mean(ds)
    return {"delta_mae": point, "ci_low": float(np.percentile(boots, 2.5)),
            "ci_high": float(np.percentile(boots, 97.5)),
            "p_not_better": float((boots >= 0).mean()), "p_not_worse": float((boots <= 0).mean()),
            "n_blocks": len(errs), "n_compounds": int(sum(len(e[0]) for e in errs))}


# =========================================================================== 1. intrinsic noise
def multi_report(df: pd.DataFrame) -> pd.DataFrame:
    """Compounds with >= 2 distinct AqSolDB reports (reliability groups G2-G5)."""
    return df[df["n_occurrences"] >= 2]


def sd_convention_check(overlap: pd.DataFrame) -> dict:
    """Infer the ddof of ``aqsol_sd`` from two-report compounds whose ESOL value differs from AqSolDB's.

    If AqSolDB kept one report x1 and ESOL holds the other report x2, |x1 - x2| / SD equals 2 for ddof = 0,
    sqrt(2) for ddof = 1 and 1 if AqSolDB had kept the mean of the two.
    """
    d = overlap[(overlap["n_occurrences"] == 2) & (overlap["diff_aqsoldb_minus_esol"].abs() > 0.01)]
    r = d["diff_aqsoldb_minus_esol"].abs() / d["aqsol_sd"]
    return {"n_two_report_nonidentical": int(len(r)), "ratio_median": float(r.median()),
            "frac_ratio_within_2pct_of_2": float(((r - 2).abs() < 0.04).mean()),
            "frac_ratio_within_2pct_of_sqrt2": float(((r - np.sqrt(2)).abs() < 0.03).mean()),
            "frac_ratio_within_2pct_of_1": float(((r - 1).abs() < 0.02).mean()),
            "conclusion": "aqsol_sd is the population SD (ddof=0)" if ((r - 2).abs() < 0.04).mean() > 0.5
            else "inconclusive"}


def pooled_noise(sd, n) -> dict:
    """Naive RMS of the (ddof=0) SDs and the unbiased pooled per-report SD."""
    sd = np.asarray(sd, dtype=float)
    n = np.asarray(n, dtype=float)
    return {"rms_sd_naive": float(np.sqrt(np.mean(sd ** 2))),
            "pooled_sd_unbiased": float(np.sqrt(np.sum(n * sd ** 2) / np.sum(n - 1)))}


def robust_sigma_pairs(sd_pairs) -> float:
    """Per-report sigma from the median absolute pair difference |x1 - x2| = 2 SD (Gaussian core)."""
    return float(np.median(2 * np.asarray(sd_pairs, dtype=float)) / (np.sqrt(2) * Z_MAD))


def nonparametric_mae_floor_pairs(sd_pairs) -> float:
    """E|x1 - x2| / sqrt(2): MAE of one report against the true value (exact for Gaussian noise)."""
    return float(np.mean(2 * np.asarray(sd_pairs, dtype=float)) / np.sqrt(2))


def gaussian_floors(sigma: float) -> dict:
    """Error floors of a perfect predictor when labels carry N(0, sigma^2) noise."""
    return {"sigma": float(sigma), "RMSE_floor": float(sigma), "MAE_floor": float(sigma * np.sqrt(2 / np.pi))}


def noise_estimates(df: pd.DataFrame, n_boot: int = 2000, seed: int = 0) -> pd.DataFrame:
    """All noise estimators with bootstrap CIs (compounds resampled) and their implied error floors."""
    m = multi_report(df)
    sd, n = m["aqsol_sd"].to_numpy(float), m["n_occurrences"].to_numpy(float)
    pairs = m.loc[m["n_occurrences"] == 2, "aqsol_sd"].to_numpy(float)
    consistent = m[m["reliability_group"].isin(HIGH_RELIABILITY)]
    rows = []

    def add(name, value, ci, n_cmp, note):
        rows.append({"estimator": name, "sigma": value, "ci_low": ci[0], "ci_high": ci[1], "n_compounds": n_cmp,
                     "RMSE_floor": value, "MAE_floor_gaussian": value * np.sqrt(2 / np.pi), "note": note})

    roles = {"RMS SD (ddof=0, naive)": "overall inter-report spread (not a floor; = IVW s0)",
             "pooled SD (ddof-corrected)": "not recommended: dominated by G2/G4 outliers",
             "pooled SD, consistent reports only (G3+G5)": "RECOMMENDED headline floor (sigma; lower bound)",
             "robust SD from pairs (median |x1-x2|)": "sensitivity (Gaussian core only)",
             "distribution-free MAE floor from pairs (E|x1-x2|/sqrt2)": "RECOMMENDED headline MAE floor"}
    add("RMS SD (ddof=0, naive)", pooled_noise(sd, n)["rms_sd_naive"],
        bootstrap_stat(lambda s: np.sqrt(np.mean(s ** 2)), sd, n_boot=n_boot, seed=seed), len(sd),
        "data-card figure; under-states per-report noise (ddof=0 SDs)")
    add("pooled SD (ddof-corrected)", pooled_noise(sd, n)["pooled_sd_unbiased"],
        bootstrap_stat(lambda s, k: np.sqrt(np.sum(k * s ** 2) / np.sum(k - 1)), sd, n, n_boot=n_boot, seed=seed),
        len(sd), "unbiased per-report SD; dominated by gross discrepancies (G2/G4)")
    cs, cn = consistent["aqsol_sd"].to_numpy(float), consistent["n_occurrences"].to_numpy(float)
    add("pooled SD, consistent reports only (G3+G5)", pooled_noise(cs, cn)["pooled_sd_unbiased"],
        bootstrap_stat(lambda s, k: np.sqrt(np.sum(k * s ** 2) / np.sum(k - 1)), cs, cn, n_boot=n_boot, seed=seed),
        len(cs), "truncated at SD<=0.5, so a lower bound on the noise")
    add("robust SD from pairs (median |x1-x2|)", robust_sigma_pairs(pairs),
        bootstrap_stat(robust_sigma_pairs, pairs, n_boot=n_boot, seed=seed), len(pairs),
        "Gaussian core of the noise, ignores gross errors")
    out = pd.DataFrame(rows)
    np_floor = nonparametric_mae_floor_pairs(pairs)
    np_ci = bootstrap_stat(nonparametric_mae_floor_pairs, pairs, n_boot=n_boot, seed=seed)
    out = pd.concat([out, pd.DataFrame([{
        "estimator": "distribution-free MAE floor from pairs (E|x1-x2|/sqrt2)", "sigma": np.nan,
        "ci_low": np_ci[0], "ci_high": np_ci[1], "n_compounds": len(pairs), "RMSE_floor": np.nan,
        "MAE_floor_gaussian": np_floor, "note": "MAE of one report vs truth; ci refers to this MAE floor"}])],
        ignore_index=True)
    out["role"] = out["estimator"].map(roles)
    out["caveat"] = ("multi-report compounds are not representative of the single-report G1 majority "
                     "(smaller, mean MW 233 vs 273; 32% vs 6% in ESOL); floors may not transfer to G1")
    return out


def sd_group_table(df: pd.DataFrame) -> pd.DataFrame:
    """Distribution of aqsol_sd (and pair difference) per reliability group of multi-report compounds."""
    rows = []
    m = multi_report(df)
    for g, d in [("G2-G5 (all multi-report)", m)] + [(g, m[m["reliability_group"] == g]) for g in GROUP_ORDER[1:]]:
        s = d["aqsol_sd"]
        rows.append({"group": g, "n": len(d), "mean_n_reports": d["n_occurrences"].mean(), "sd_mean": s.mean(),
                     "sd_median": s.median(), "sd_q25": s.quantile(0.25), "sd_q75": s.quantile(0.75),
                     "sd_q95": s.quantile(0.95), "sd_max": s.max(), "rms_sd": float(np.sqrt((s ** 2).mean())),
                     "frac_sd_lt_0.01": (s < 0.01).mean(), "frac_sd_gt_1": (s > 1).mean()})
    return pd.DataFrame(rows)


def sd_correlates(df: pd.DataFrame, clogp: pd.Series, n_bins: int = 5) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spearman correlation of aqsol_sd with logS, MW and cLogP, and quantile-binned SD statistics.

    Returns ``(correlations, binned)`` computed on multi-report compounds.
    """
    m = multi_report(df).copy()
    m["clogp"] = clogp.reindex(m["compound_id"]).to_numpy()
    corr, binned = [], []
    for var in ("logS", "mol_wt", "clogp"):
        rho, p = stats.spearmanr(m[var], m["aqsol_sd"])
        rho_gross, p_gross = stats.mannwhitneyu(m.loc[m["aqsol_sd"] > 0.5, var], m.loc[m["aqsol_sd"] <= 0.5, var])
        corr.append({"variable": var, "n": len(m), "spearman_rho": rho, "p_value": p,
                     "median_if_sd_gt_0.5": m.loc[m["aqsol_sd"] > 0.5, var].median(),
                     "median_if_sd_le_0.5": m.loc[m["aqsol_sd"] <= 0.5, var].median(),
                     "mannwhitney_p": p_gross})
        q = pd.qcut(m[var], n_bins, duplicates="drop")
        for iv, d in m.groupby(q, observed=True):
            binned.append({"variable": var, "bin": str(iv), "bin_center": float(d[var].median()), "n": len(d),
                           "sd_median": d["aqsol_sd"].median(), "rms_sd": float(np.sqrt((d["aqsol_sd"] ** 2).mean())),
                           "frac_sd_gt_0.5": (d["aqsol_sd"] > 0.5).mean()})
    return pd.DataFrame(corr), pd.DataFrame(binned)


def selection_bias_table(df: pd.DataFrame, clogp: pd.Series) -> pd.DataFrame:
    """Compare single-report (G1) and multi-report compounds: are the noise-estimating compounds typical?"""
    d = df.copy()
    d["clogp"] = clogp.reindex(d["compound_id"]).to_numpy()
    d["multi"] = d["n_occurrences"] >= 2
    rows = []
    for var in ("logS", "mol_wt", "clogp", "n_heavy_atoms"):
        a, b = d.loc[~d["multi"], var], d.loc[d["multi"], var]
        rows.append({"variable": var, "median_single_G1": a.median(), "median_multi_G2_G5": b.median(),
                     "mean_single_G1": a.mean(), "mean_multi_G2_G5": b.mean(),
                     "ks_stat": stats.ks_2samp(a, b).statistic, "ks_p": stats.ks_2samp(a, b).pvalue})
    for var in ("is_multicomponent", "in_esol"):
        a, b = d.loc[~d["multi"], var].astype(bool), d.loc[d["multi"], var].astype(bool)
        rows.append({"variable": f"frac {var}", "median_single_G1": np.nan, "median_multi_G2_G5": np.nan,
                     "mean_single_G1": a.mean(), "mean_multi_G2_G5": b.mean(), "ks_stat": np.nan, "ks_p": np.nan})
    return pd.DataFrame(rows)


def dedup_summary(df: pd.DataFrame) -> dict:
    """Our own stereo/representation de-duplication: spread of merged groups (merged_logS_sd)."""
    g = df[df["n_merged"] > 1]
    s = g["merged_logS_sd"]
    return {"n_groups": int(len(g)), "n_rows_merged": int(g["n_merged"].sum()),
            "merged_sd_median": float(s.median()), "merged_sd_rms": float(np.sqrt((s ** 2).mean())),
            "merged_sd_q90": float(s.quantile(0.9)), "n_conflict_sd_gt_0.5": int(g["dedup_conflict"].sum()),
            "frac_sd_lt_0.01": float((s < 0.01).mean()),
            "conflict_reliability_groups": g.loc[g["dedup_conflict"], "reliability_group"].value_counts().to_dict()}


def bland_altman(a, b, n_boot: int = 2000, seed: int = 0) -> dict:
    """Bland-Altman agreement of two measurement series (difference = a - b)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    lo, hi = bootstrap_stat(np.mean, d, n_boot=n_boot, seed=seed)
    return {"n": int(len(d)), "bias": bias, "bias_ci_low": lo, "bias_ci_high": hi, "sd_diff": sd,
            "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd,
            "median_diff": float(np.median(d)), "q025_diff": float(np.quantile(d, 0.025)),
            "q975_diff": float(np.quantile(d, 0.975)), "MAE": float(np.abs(d).mean()),
            "RMSE": float(np.sqrt((d ** 2).mean())), "frac_identical_lt_0.01": float((np.abs(d) < 0.01).mean()),
            "frac_absdiff_gt_0.5": float((np.abs(d) > 0.5).mean()), "frac_absdiff_gt_1": float((np.abs(d) > 1).mean())}


def overlap_breakdown(overlap: pd.DataFrame, col: str) -> pd.DataFrame:
    """Agreement statistics of the ESOL / AqSolDB overlap broken down by ``col`` (source letter or group)."""
    rows = []
    d_all = overlap["diff_aqsoldb_minus_esol"]
    for key, d in overlap.groupby(col):
        x = d["diff_aqsoldb_minus_esol"]
        rows.append({col: key, "n": len(d), "frac_of_overlap": len(d) / len(overlap),
                     "frac_identical_lt_0.01": (x.abs() < 0.01).mean(), "MAE": x.abs().mean(),
                     "bias": x.mean(), "frac_absdiff_gt_0.5": (x.abs() > 0.5).mean()})
    rows.append({col: "all", "n": len(overlap), "frac_of_overlap": 1.0,
                 "frac_identical_lt_0.01": (d_all.abs() < 0.01).mean(), "MAE": d_all.abs().mean(),
                 "bias": d_all.mean(), "frac_absdiff_gt_0.5": (d_all.abs() > 0.5).mean()})
    return pd.DataFrame(rows)


# =========================================================================== 2. error vs reliability
def reliability3(groups: pd.Series) -> pd.Series:
    """Collapse G1-G5 into single / consistent / inconsistent reports."""
    return groups.map({"G1": REL3_ORDER[0], "G3": REL3_ORDER[1], "G5": REL3_ORDER[1],
                       "G2": REL3_ORDER[2], "G4": REL3_ORDER[2]})


def sd_bin(sd: pd.Series, n_occ: pd.Series) -> pd.Series:
    """aqsol_sd bins; single-report compounds form their own bin (their SD of 0 is not a measurement)."""
    cut = pd.cut(sd, SD_BIN_EDGES, labels=SD_BIN_LABELS[1:], include_lowest=False).astype(object)
    cut[n_occ <= 1] = SD_BIN_LABELS[0]
    cut[(n_occ > 1) & (sd <= 0)] = SD_BIN_LABELS[1]
    return cut


def occ_bin(n_occ: pd.Series) -> pd.Series:
    return n_occ.clip(upper=5).astype(int).astype(str).replace({"5": ">=5"})


def load_reference_predictions(pred_root: Path, split_type: str, model_file: str, seeds, subset: str = "test",
                               ) -> pd.DataFrame:
    """Concatenate reference prediction files ``<pred_root>/<split_type>/seed<k>/<model_file>.csv``."""
    frames = []
    for s in seeds:
        f = Path(pred_root) / split_type / f"seed{s}" / f"{model_file}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"compound_id": str})
        d = d[d["subset"] == subset].copy()
        d["seed"] = s
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def compound_errors(preds: pd.DataFrame) -> pd.DataFrame:
    """One row per compound: mean |error| and mean signed error over the seeds where it was a test compound."""
    p = preds.assign(ae=(preds["y_pred"] - preds["y_true"]).abs(), se=preds["y_pred"] - preds["y_true"])
    agg = {"ae": "mean", "se": "mean", "y_true": "first", "y_pred": "mean", "seed": "count"}
    if "maxsim" in p:
        agg["maxsim"] = "mean"
    return p.groupby("compound_id").agg(agg).rename(columns={"seed": "n_seeds"}).reset_index()


def annotate(err: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Attach reliability information (group, rel3, sd bin, occurrence bin, source letter)."""
    cols = ["compound_id", "reliability_group", "n_occurrences", "aqsol_sd", "source", "dedup_conflict", "logS"]
    out = err.merge(meta[cols], on="compound_id", how="left")
    out["rel3"] = reliability3(out["reliability_group"])
    out["sd_bin"] = sd_bin(out["aqsol_sd"], out["n_occurrences"])
    out["occ_bin"] = occ_bin(out["n_occurrences"])
    return out


def group_mae_table(err: pd.DataFrame, col: str, order: list | None = None, n_boot: int = 2000,
                    seed: int = 0, min_n: int = 1) -> pd.DataFrame:
    """MAE with bootstrap CI, RMSE, bias and median |error| per level of ``col``."""
    rows = []
    levels = order or sorted(err[col].dropna().unique())
    for lev in levels:
        d = err[err[col] == lev]
        if len(d) < min_n:
            continue
        mae, lo, hi = mae_ci(d["ae"], n_boot=n_boot, seed=seed)
        rows.append({col: lev, "n": len(d), "MAE": mae, "MAE_ci_low": lo, "MAE_ci_high": hi,
                     "median_AE": d["ae"].median(), "bias": d["se"].mean(),
                     "mean_label_sd": d["aqsol_sd"].mean()})
    return pd.DataFrame(rows)


def kruskal_test(err: pd.DataFrame, col: str, min_n: int = 10) -> dict:
    """Kruskal-Wallis H test of compound-level |error| across levels of ``col`` (levels with >= min_n)."""
    samples = [d["ae"].to_numpy() for _, d in err.groupby(col) if len(d) >= min_n]
    if len(samples) < 2:
        return {"H": np.nan, "p": np.nan, "k": len(samples)}
    h, p = stats.kruskal(*samples)
    return {"H": float(h), "p": float(p), "k": len(samples)}


def adjusted_reliability_effects(err: pd.DataFrame) -> pd.DataFrame:
    """OLS of compound-level |error| on reliability class, adjusting for similarity to train and logS.

    |e| ~ C(rel3, ref=consistent) + maxsim + logS + logS^2, heteroscedasticity-robust (HC3) SEs.
    """
    import statsmodels.formula.api as smf

    d = err.dropna(subset=["maxsim", "rel3"]).copy()
    d["rel"] = d["rel3"].map({REL3_ORDER[0]: "single", REL3_ORDER[1]: "consistent", REL3_ORDER[2]: "inconsistent"})
    fit = smf.ols("ae ~ C(rel, Treatment('consistent')) + maxsim + logS + I(logS**2)", data=d).fit(cov_type="HC3")
    ci = fit.conf_int()
    rows = []
    for term in fit.params.index:
        rows.append({"term": term.replace("C(rel, Treatment('consistent'))", "rel"), "coef": fit.params[term],
                     "ci_low": ci.loc[term, 0], "ci_high": ci.loc[term, 1], "p": fit.pvalues[term]})
    out = pd.DataFrame(rows)
    out["n"] = int(fit.nobs)
    out["r2"] = fit.rsquared
    return out


# =========================================================================== 3. training-data quality
def ivw_weights(aqsol_sd, n_occurrences, s0: float) -> np.ndarray:
    """Inverse-variance weights w = 1/(sd^2 + s0^2), sd = s0 for single reports; normalised to mean 1."""
    sd = np.asarray(aqsol_sd, dtype=float).copy()
    n = np.asarray(n_occurrences, dtype=float)
    sd[n <= 1] = s0
    w = 1.0 / (sd ** 2 + s0 ** 2)
    return w / w.mean()


ABLATION_VARIANTS = {
    "a_all": "all training data",
    "b_drop_G2G4": "drop inconsistent multi-report (G2+G4) from train",
    "c_drop_dedup_conflict": "drop de-duplication conflicts from train",
    "d_ivw": "inverse-variance sample weights",
    "e_random_drop": "size-matched random removal (control for b)",
    "f_ivw_permuted": "IVW weights randomly permuted over train (control for d)",
}


def ablation_training_set(train_ids: list[str], meta: pd.DataFrame, variant: str, s0: float,
                          rng_seed: int = 0) -> tuple[list[str], np.ndarray | None, int]:
    """Training ids, optional sample weights and number of dropped compounds for an ablation variant.

    ``meta`` must be indexed by compound_id. ``e_random_drop*`` removes as many random training compounds
    as ``b_drop_G2G4`` would remove; ``f_ivw_permuted*`` keeps the weight distribution of ``d_ivw`` but
    shuffles it across training compounds, so it controls for the effect of re-weighting per se.
    """
    m = meta.loc[train_ids]
    low = m["reliability_group"].isin(LOW_RELIABILITY).to_numpy()
    if variant == "a_all":
        return list(train_ids), None, 0
    if variant == "b_drop_G2G4":
        keep = ~low
    elif variant == "c_drop_dedup_conflict":
        keep = ~m["dedup_conflict"].astype(bool).to_numpy()
    elif variant == "d_ivw":
        return list(train_ids), ivw_weights(m["aqsol_sd"], m["n_occurrences"], s0), 0
    elif variant.startswith("f_ivw_permuted"):
        w = ivw_weights(m["aqsol_sd"], m["n_occurrences"], s0)
        return list(train_ids), np.random.default_rng(rng_seed).permutation(w), 0
    elif variant.startswith("e_random_drop"):
        rng = np.random.default_rng(rng_seed)
        drop = rng.choice(len(train_ids), size=int(low.sum()), replace=False)
        keep = np.ones(len(train_ids), dtype=bool)
        keep[drop] = False
    else:
        raise ValueError(f"unknown ablation variant {variant!r}")
    ids = [c for c, k in zip(train_ids, keep) if k]
    return ids, None, int((~keep).sum())


def fit_lgbm(X_train, y_train, X_val, y_val, seed: int = 0, n_jobs: int = 2, sample_weight=None,
             params: dict | None = None):
    """LightGBM (registry defaults) with early stopping on the *unweighted* validation MAE.

    Mirrors :func:`solpred.models.fit_model` but passes ``sample_weight`` to the training set; with
    ``sample_weight=None`` it reproduces the shared code path exactly.
    """
    import lightgbm as lgb

    from solpred.models import get_model

    model = get_model("lgbm", seed=seed, n_jobs=n_jobs, params=params)
    cb = [lgb.early_stopping(100, verbose=False)]
    model.fit(X_train, y_train, sample_weight=sample_weight, eval_X=(X_val,), eval_y=(y_val,),
              eval_metric="l1", callbacks=cb)
    return model


# =========================================================================== 4. source transfer
def subsample(ids: list[str], n: int, seed: int) -> list[str]:
    """Random subset of size ``min(n, len(ids))`` (order preserved from the permutation)."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ids))[: min(n, len(ids))]
    return [ids[i] for i in idx]


def loso_control_split(meta: pd.DataFrame, source: str, n_train: int, n_folds: int, fold: int,
                       seed: int = 0, val_frac: float = 0.1) -> dict:
    """Size-matched *in-distribution* control for a LOSO split.

    The held-out source is cut into ``n_folds`` folds; fold ``fold`` is the test set and the remaining
    compounds (all sources, including the other folds of ``source``) form a pool from which a random
    validation set (``val_frac``) and a training set of exactly ``n_train`` compounds (the LOSO training
    size) are drawn.
    """
    rng = np.random.default_rng(seed)
    src_ids = meta.loc[meta["source"] == source, "compound_id"].to_numpy(str)
    folds = np.array_split(rng.permutation(src_ids), n_folds)
    test = list(folds[fold])
    test_set = set(test)
    pool = [c for c in meta["compound_id"].astype(str) if c not in test_set]
    perm = list(np.asarray(pool)[np.random.default_rng(seed + 1 + fold).permutation(len(pool))])
    n_val = int(round(val_frac * len(pool)))
    val, rest = perm[:n_val], perm[n_val:]
    return {"train": rest[:n_train], "val": val, "test": test,
            "meta": {"split_type": "loso_control", "held_out_source": source, "fold": fold}}


def similarity_adjusted_effect(ae, treat, sim, groups) -> dict:
    """Effect of a binary condition on compound-level |error| at equal nearest-neighbour similarity.

    OLS |e| ~ treat + maxsim + maxsim^2 with SEs clustered by ``groups`` (the compound, which appears under
    both conditions), returning the ``treat`` coefficient with its 95% CI and p-value.
    """
    import statsmodels.formula.api as smf

    d = pd.DataFrame({"ae": np.asarray(ae, float), "treat": np.asarray(treat, float),
                      "maxsim": np.asarray(sim, float), "g": pd.factorize(np.asarray(groups))[0]})
    fit = smf.ols("ae ~ treat + maxsim + I(maxsim**2)", data=d).fit(cov_type="cluster", cov_kwds={"groups": d["g"]})
    ci = fit.conf_int().loc["treat"]
    return {"coef": float(fit.params["treat"]), "ci_low": float(ci[0]), "ci_high": float(ci[1]),
            "p": float(fit.pvalues["treat"]), "n_rows": int(fit.nobs)}


def similarity_adjusted_penalty(ae_shift, ae_ctrl, sim_shift, sim_ctrl) -> dict:
    """Source-shift penalty net of nearest-neighbour similarity to the training set.

    Each compound is predicted twice (LOSO = shifted, control = in-distribution); the two rows per compound are
    stacked and :func:`similarity_adjusted_effect` is applied. The coefficient is the extra error of the unseen
    source at equal similarity (a protocol/source effect), whereas the raw paired difference also contains the
    loss of close training analogues.
    """
    n = len(ae_shift)
    r = similarity_adjusted_effect(np.r_[ae_shift, ae_ctrl], np.r_[np.ones(n), np.zeros(n)],
                                   np.r_[sim_shift, sim_ctrl], np.r_[np.arange(n), np.arange(n)])
    return {"adj_penalty": r["coef"], "adj_penalty_ci_low": r["ci_low"], "adj_penalty_ci_high": r["ci_high"],
            "adj_penalty_p": r["p"]}


def source_transfer_design(meta: pd.DataFrame, sources: list[str], test_frac: float = 0.2,
                           seed: int = 0) -> dict:
    """Fixed per-source held-out test sets and training pools for the source-to-source matrix."""
    out = {}
    rng = np.random.default_rng(seed)
    for s in sources:
        ids = meta.loc[meta["source"] == s, "compound_id"].to_numpy(str)
        perm = rng.permutation(ids)
        n_test = int(round(test_frac * len(ids)))
        out[s] = {"test": list(perm[:n_test]), "pool": list(perm[n_test:])}
    return out


# =========================================================================== 5. external transfer
def delaney_equation(X_esol4: np.ndarray) -> np.ndarray:
    """Delaney (2004) published ESOL equation applied to [cLogP, MW, RB, AP] (here Crippen cLogP)."""
    X = np.asarray(X_esol4, dtype=float)
    c = DELANEY_COEF
    return (c["intercept"] + c["clogp"] * X[:, 0] + c["mw"] * X[:, 1] + c["rotatable_bonds"] * X[:, 2]
            + c["aromatic_proportion"] * X[:, 3])


def similarity_bin_table(sim, ae, edges=SIM_BIN_EDGES, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """MAE (bootstrap CI) per bin of nearest-neighbour Tanimoto similarity to the training set."""
    sim = np.asarray(sim, dtype=float)
    ae = np.asarray(ae, dtype=float)
    labels = [f"[{lo:.1f}, {min(hi, 1.0):.1f}{']' if hi > 1 else ')'}" for lo, hi in zip(edges[:-1], edges[1:])]
    b = np.digitize(sim, edges[1:-1])
    rows = []
    for i, lab in enumerate(labels):
        m = b == i
        if m.sum() == 0:
            rows.append({"sim_bin": lab, "n": 0, "MAE": np.nan, "MAE_ci_low": np.nan, "MAE_ci_high": np.nan})
            continue
        mae, lo, hi = mae_ci(ae[m], n_boot=n_boot, seed=seed)
        rows.append({"sim_bin": lab, "n": int(m.sum()), "MAE": mae, "MAE_ci_low": lo, "MAE_ci_high": hi})
    return pd.DataFrame(rows)


def similarity_standardised_mae(sim, ae, ref_sim, edges=SIM_BIN_EDGES) -> float:
    """MAE re-weighted to the nearest-neighbour-similarity distribution of a reference test set.

    sum_b f_ref(b) * MAE(b), over the similarity bins populated in both sets (f_ref renormalised), i.e. the MAE
    the model would have if its test compounds were as similar to its training set as the reference ones.
    """
    sim, ae, ref_sim = (np.asarray(a, dtype=float) for a in (sim, ae, ref_sim))
    b, rb = np.digitize(sim, edges[1:-1]), np.digitize(ref_sim, edges[1:-1])
    num = den = 0.0
    for k in range(len(edges) - 1):
        if (b == k).any() and (rb == k).any():
            f = (rb == k).mean()
            num += f * ae[b == k].mean()
            den += f
    return float(num / den) if den > 0 else float("nan")
