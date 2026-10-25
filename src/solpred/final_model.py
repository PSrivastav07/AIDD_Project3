"""Production logS model: bagged ensemble + scaffold-grouped conformal calibration + applicability domain.

Method
------
1. **Data / features.** All curated AqSolDB compounds (``data/processed/aqsoldb_clean.csv``) are featurised
   with the cached featuriser (default ``rdkit2d+morgan_counts``).
2. **Bagged ensemble.** ``n_members`` estimators from :func:`solpred.models.get_model`, each fitted on a
   bootstrap resample of the training pool with its own seed (bagging). LightGBM members are early-stopped
   (patience 100, L1) on the *out-of-bag* compounds of their own resample - an inner validation set of
   ~36.8 % of the pool that costs no data and never touches calibration or test compounds; the registry
   ``n_estimators`` is only the upper bound. The point prediction is the member mean and the per-compound
   uncertainty scale is the member standard deviation (ddof=1).
3. **Conformal calibration on novel chemotypes.** 5-fold ``GroupKFold`` over Bemis-Murcko scaffolds. For each
   fold an ensemble is trained *exactly as in step 2* on the other four folds and predicts the held-out fold,
   giving out-of-fold (OOF) mean + std for every compound. These feed
   ``solpred.uncertainty.ConformalRegressor(normalized=True, eps=beta)``, i.e. score = |y - mean| / (std + beta).
   The offset ``beta`` regularises the difficulty estimate: with beta = 0 the compounds with the smallest ensemble
   std are under-covered and those with the largest over-covered (the std is only weakly rank-correlated with the
   error). beta is chosen from a small grid on the calibration (OOF) data alone, by leave-one-fold-out conformal
   coverage: the value whose coverage is most uniform across ensemble-std quintiles (near-ties -> narrower
   intervals). ``recalibrate()`` refits only this step from ``oof_predictions.csv``.
   Generic scaffolds are handled explicitly: the empty scaffold (all 2,439 acyclic compounds) and any scaffold
   with more than ``max_group_size`` members (plain benzene) are *not* chemotypes - a production query with
   such a scaffold always has many training analogues - so their compounds become singleton groups.
   Without this, one fold would contain every acyclic compound and be predicted by a model that has never
   seen one, a situation the production model never faces. All other scaffolds stay intact (held out en bloc).
4. **Applicability domain.** :class:`solpred.applicability.ApplicabilityDomain` (ECFP4 max-Tanimoto) on all
   training SMILES with ids and labels.

Every fit is cached under ``<out>/_work/<config-hash>/`` so an interrupted run resumes where it stopped.
"""
from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import GroupKFold

from solpred.applicability import ApplicabilityDomain
from solpred.config import ROOT, load_config, load_json, path, save_json
from solpred.features import load_dataset, load_or_compute
from solpred.metrics import regression_metrics
from solpred.models import DEFAULT_PARAMS, fit_model, get_model
from solpred.splits import max_similarity, morgan_bits
from solpred.uncertainty import ConformalRegressor, coverage

BUNDLE_FORMAT_VERSION = 1
ALPHAS_REPORTED = (0.1, 0.05)
SCALE_OFFSET_GRID = (0.0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.25, 0.3, 0.4, 0.5)
MIN_EPS = 1e-3
GENERIC_PREFIX = "__generic__:"


@dataclass
class TrainConfig:
    """Everything that determines the trained artefact (hashed to key the resumable cache)."""

    model: str = "lgbm"
    features: str = "rdkit2d+morgan_counts"
    n_members: int = 5
    params: dict = field(default_factory=dict)
    n_folds: int = 5
    seed: int = 0
    early_stopping: str = "oob"     # "oob" (LightGBM only) or "none" (use params' n_estimators as is)
    bootstrap: bool = True
    max_group_size: int = 500       # scaffold groups larger than this are split into singletons
    dataset: str = "aqsoldb"

    def key(self) -> str:
        blob = json.dumps(asdict(self), sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:10]


# --------------------------------------------------------------------------- grouping / folds
def scaffold_groups(scaffolds: Sequence[str], ids: Sequence[str], max_group_size: int) -> tuple[np.ndarray, np.ndarray]:
    """Group label per compound and a boolean mask of compounds whose scaffold was treated as generic."""
    sc = pd.Series(list(scaffolds), dtype=str).fillna("")
    sizes = sc.map(sc.value_counts())
    generic = (sc == "").to_numpy() | (sizes > max_group_size).to_numpy()
    groups = np.where(generic, GENERIC_PREFIX + pd.Series(list(ids), dtype=str), sc.to_numpy())
    return groups.astype(str), generic


def fold_assignment(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold index per compound from sklearn ``GroupKFold`` (size-balanced, groups never split).

    Group labels are replaced by a seeded random permutation of integer codes first, so ties between
    equal-sized groups (e.g. all singletons) are broken randomly rather than alphabetically.
    """
    uniq, inv = np.unique(groups, return_inverse=True)
    codes = np.random.default_rng(seed).permutation(len(uniq))[inv]
    folds = np.full(len(groups), -1, dtype=int)
    for k, (_, test_idx) in enumerate(GroupKFold(n_splits=n_folds).split(np.zeros(len(groups)), groups=codes)):
        folds[test_idx] = k
    return folds


def member_seed(base: int, fold: int, member: int) -> int:
    """Deterministic seed: fold = -1 for the production ensemble, 0..K-1 for CV ensembles."""
    return base * 100_000 + (fold + 1) * 1_000 + member


# --------------------------------------------------------------------------- ensemble
def fit_member(X: np.ndarray, y: np.ndarray, cfg: TrainConfig, seed: int, n_jobs: int = 2):
    """Fit one bagged member; returns ``(model, info)``."""
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, n) if cfg.bootstrap else np.arange(n)
    oob = np.setdiff1d(np.arange(n), idx)
    model = get_model(cfg.model, seed=seed % (2 ** 31), n_jobs=n_jobs, params=cfg.params)
    use_es = cfg.model == "lgbm" and cfg.early_stopping == "oob" and len(oob) > 0
    t0 = time.time()
    if use_es:
        fit_model(model, X[idx], y[idx], X[oob], y[oob])
    else:
        fit_model(model, X[idx], y[idx])
    info = {"seed": int(seed), "n_fit": int(n), "n_oob": int(len(oob)), "fit_seconds": round(time.time() - t0, 1)}
    if cfg.model == "lgbm":
        best = getattr(model, "best_iteration_", None)
        info["n_iter"] = int(best) if best else int(model.n_estimators)
    if len(oob):
        info["oob_MAE"] = float(np.mean(np.abs(model.predict(X[oob]) - y[oob])))
    return model, info


def ensemble_predict(members: list, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Member mean and member standard deviation (ddof=1; zeros for a single member)."""
    P = np.stack([np.asarray(m.predict(X), dtype=float) for m in members])
    std = P.std(axis=0, ddof=1) if len(members) > 1 else np.zeros(P.shape[1])
    return P.mean(axis=0), std


# --------------------------------------------------------------------------- cross-validation
def cross_validate(X: np.ndarray, y: np.ndarray, folds: np.ndarray, cfg: TrainConfig, work: Path,
                   n_jobs: int = 2, log: Callable = print) -> tuple[np.ndarray, list[dict]]:
    """OOF member predictions, shape (n_members, n); each fold's ensemble is cached to ``work``."""
    work.mkdir(parents=True, exist_ok=True)
    P = np.full((cfg.n_members, len(y)), np.nan)
    infos = []
    for k in range(cfg.n_folds):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        for m in range(cfg.n_members):
            f = work / f"cv_fold{k}_member{m}.npz"
            if f.exists():
                z = np.load(f, allow_pickle=False)
                assert np.array_equal(z["idx"], te), f"stale cache {f}"
                P[m, te] = z["pred"]
                info = json.loads(str(z["info"]))
            else:
                model, info = fit_member(X[tr], y[tr], cfg, member_seed(cfg.seed, k, m), n_jobs)
                info.update(fold=k, member=m)
                P[m, te] = model.predict(X[te])
                np.savez(f, idx=te, pred=P[m, te], info=json.dumps(info))
                log(f"  cv fold {k} member {m}: n_iter={info.get('n_iter', '-')} "
                    f"oob_MAE={info.get('oob_MAE', float('nan')):.3f} fold_MAE={np.mean(np.abs(P[m, te] - y[te])):.3f} "
                    f"({info['fit_seconds']:.0f}s)")
            infos.append(info)
    return P, infos


def oof_max_similarity(smiles: Sequence[str], folds: np.ndarray) -> np.ndarray:
    """ECFP4 max Tanimoto of each compound to the training part of its CV fold (same FP as the AD)."""
    fps = morgan_bits(list(smiles))
    out = np.full(len(fps), np.nan)
    for k in np.unique(folds):
        te = np.where(folds == k)[0]
        tr = np.where(folds != k)[0]
        out[te] = max_similarity([fps[i] for i in te], [fps[i] for i in tr])
    return out


# --------------------------------------------------------------------------- calibration diagnostics
def _interval_stats(cp: ConformalRegressor, y, pred, scale, alpha: float) -> dict:
    lo, hi = cp.interval(pred, scale=scale, alpha=alpha)
    w = hi - lo
    return {"coverage": coverage(y, lo, hi), "mean_width": float(np.mean(w)), "median_width": float(np.median(w))}


def lofo_coverage(y: np.ndarray, p: np.ndarray, s: np.ndarray, folds: np.ndarray, alpha: float,
                  eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Leave-one-fold-out normalised conformal: per-compound coverage indicator and interval width."""
    cov = np.zeros(len(y), dtype=bool)
    width = np.zeros(len(y))
    for k in np.unique(folds):
        te = folds == k
        cp = ConformalRegressor(normalized=True, eps=eps).fit(y[~te], p[~te], scale=s[~te])
        lo, hi = cp.interval(p[te], scale=s[te], alpha=alpha)
        cov[te] = (y[te] >= lo) & (y[te] <= hi)
        width[te] = hi - lo
    return cov, width


def select_scale_offset(y: np.ndarray, p: np.ndarray, s: np.ndarray, folds: np.ndarray, alpha: float = 0.1,
                        grid: Sequence[float] = SCALE_OFFSET_GRID, n_bins: int = 5) -> tuple[float, list[dict]]:
    """Choose beta in score = |y - p| / (s + beta) on calibration data only (module docstring, step 3).

    Criterion: leave-one-fold-out coverage per ensemble-std quantile bin; minimise the largest deviation from
    ``1 - alpha``, rounded to 0.005 so that near-ties are decided by the smaller mean interval width.
    """
    bins = pd.qcut(s, n_bins, labels=False, duplicates="drop")
    rows = []
    for b in grid:
        cov, width = lofo_coverage(y, p, s, folds, alpha, eps=float(b) + MIN_EPS)
        per_bin = [float(cov[bins == i].mean()) for i in np.unique(bins)]
        rows.append({"beta": float(b), "coverage_lofo": float(cov.mean()), "mean_width": float(width.mean()),
                     "coverage_by_std_bin": per_bin,
                     "max_bin_deviation": float(np.max(np.abs(np.array(per_bin) - (1 - alpha))))})
    best = min(rows, key=lambda r: (round(r["max_bin_deviation"] / 0.005), r["mean_width"]))
    return best["beta"], rows


def calibration_diagnostics(oof: pd.DataFrame, sim_threshold: float, alphas: Sequence[float] = ALPHAS_REPORTED,
                            eps: float = MIN_EPS) -> dict:
    """Conformal coverage/width on the OOF predictions (in-sample, leave-one-fold-out, by subgroup).

    ``eps`` is the scale offset of the normalised calibrator (beta + MIN_EPS).
    """
    y, p, s = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_std"].to_numpy()
    cp_n = ConformalRegressor(normalized=True, eps=eps).fit(y, p, scale=s)
    cp_a = ConformalRegressor(normalized=False).fit(y, p)
    err = np.abs(y - p)
    out: dict = {"n_cal": int(len(y)),
                 "spearman_std_vs_abs_error": float(stats.spearmanr(s, err)[0]),
                 "std_mean": float(np.mean(s)), "std_median": float(np.median(s))}
    in_ad = oof["ad_maxsim"].to_numpy() >= sim_threshold
    generic = oof["generic_scaffold"].to_numpy(bool)
    for a in alphas:
        tag = f"alpha_{a:g}"
        res = {"nominal": 1 - a, "q_normalized": cp_n.quantile(a), "q_absolute": cp_a.quantile(a),
               "normalized_in_sample": _interval_stats(cp_n, y, p, s, a),
               "absolute_in_sample": _interval_stats(cp_a, y, p, None, a)}
        # leave-one-fold-out: calibrate on 4 folds, evaluate on the 5th (scaffold-novel exchangeability check)
        lofo = []
        for k in sorted(oof["fold"].unique()):
            te = (oof["fold"] == k).to_numpy()
            cp_k = ConformalRegressor(normalized=True, eps=eps).fit(y[~te], p[~te], scale=s[~te])
            lofo.append(_interval_stats(cp_k, y[te], p[te], s[te], a)["coverage"])
        res["normalized_leave_one_fold_out"] = {"per_fold": lofo, "mean": float(np.mean(lofo)),
                                                "min": float(np.min(lofo)), "max": float(np.max(lofo))}
        lo, hi = cp_n.interval(p, scale=s, alpha=a)
        cov = (y >= lo) & (y <= hi)
        width = hi - lo
        std_q = pd.qcut(s, 5, labels=False, duplicates="drop")
        groups = ([("in_domain", in_ad), ("out_of_domain", ~in_ad), ("generic_scaffold", generic),
                   ("specific_scaffold", ~generic)]
                  + [(f"std_quintile_{int(i) + 1}", std_q == i) for i in np.unique(std_q)])
        res["by_subgroup"] = {
            name: {"n": int(mask.sum()), "coverage": float(cov[mask].mean()) if mask.any() else float("nan"),
                   "mean_width": float(width[mask].mean()) if mask.any() else float("nan")}
            for name, mask in groups}
        out[tag] = res
    return out


def subgroup_metrics(oof: pd.DataFrame, sim_threshold: float) -> dict:
    """OOF regression metrics overall, per fold, inside / outside the AD and by scaffold type."""
    y, p = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy()
    in_ad = oof["ad_maxsim"].to_numpy() >= sim_threshold
    generic = oof["generic_scaffold"].to_numpy(bool)
    res = {"overall": regression_metrics(y, p),
           "per_fold": {int(k): regression_metrics(g["y_true"], g["y_pred"]) for k, g in oof.groupby("fold")},
           "in_domain": regression_metrics(y[in_ad], p[in_ad]),
           "out_of_domain": regression_metrics(y[~in_ad], p[~in_ad]) if (~in_ad).sum() > 2 else None,
           "generic_scaffold": regression_metrics(y[generic], p[generic]),
           "specific_scaffold": regression_metrics(y[~generic], p[~generic]),
           "frac_in_domain": float(in_ad.mean()),
           "ad_maxsim_median": float(np.median(oof["ad_maxsim"]))}
    bins = [0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    cut = pd.cut(oof["ad_maxsim"], bins, right=False)
    res["by_maxsim_bin"] = [{"bin": str(b), "n": int(len(g)), "MAE": float(np.mean(np.abs(g["y_true"] - g["y_pred"]))),
                             "mean_std": float(g["y_std"].mean())}
                            for b, g in oof.groupby(cut, observed=True)]
    return res


def calibrate(oof: pd.DataFrame, sim_threshold: float, normalized: bool) -> tuple[ConformalRegressor, dict]:
    """Fit the production conformal calibrator on OOF predictions; returns it with its metadata block.

    Also (re)writes the ``pi90_low`` / ``pi90_high`` columns of ``oof`` in place.
    """
    y, p, s = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_std"].to_numpy()
    meta: dict = {"normalized": normalized}
    if normalized:
        beta, grid = select_scale_offset(y, p, s, oof["fold"].to_numpy())
        conformal = ConformalRegressor(normalized=True, eps=beta + MIN_EPS).fit(y, p, scale=s)
        meta.update(method="split-conformal on scaffold-grouped OOF predictions, normalised by (ensemble std + beta)",
                    scale_offset_beta=beta, eps=conformal.eps,
                    scale_offset_selection={"criterion": "leave-one-fold-out 90% coverage most uniform across "
                                                         "ensemble-std quintiles (calibration data only)",
                                            "grid": grid})
        meta.update(calibration_diagnostics(oof, sim_threshold, eps=conformal.eps))
    else:
        conformal = ConformalRegressor(normalized=False).fit(y, p)
        meta.update(method="split-conformal on scaffold-grouped OOF predictions, absolute residuals",
                    eps=conformal.eps)
    lo, hi = conformal.interval(p, scale=s if normalized else None, alpha=0.1)
    oof["pi90_low"], oof["pi90_high"] = lo, hi
    return conformal, meta


def recalibrate(out_dir: str | Path, project_cfg: dict | None = None, log: Callable = print) -> dict:
    """Refit only the conformal calibrator of an existing model directory from its ``oof_predictions.csv``.

    No model is retrained; ``model.joblib``, ``metadata.json`` and the OOF interval columns are updated.
    """
    pcfg = project_cfg or load_config()
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    bundle = load_bundle(out_dir)
    meta = load_json(out_dir / "metadata.json")
    oof = pd.read_csv(out_dir / "oof_predictions.csv", keep_default_na=False, na_values=[""])
    thr = float(pcfg["applicability_domain"]["similarity_in_domain"])
    conformal, cmeta = calibrate(oof, thr, normalized=bool(meta["conformal"]["normalized"]))
    bundle["conformal"] = conformal
    meta["conformal"] = cmeta
    meta["recalibrated_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    joblib.dump(bundle, out_dir / "model.joblib", compress=3)
    save_json(meta, out_dir / "metadata.json")
    log(f"[recalibrate] beta={cmeta.get('scale_offset_beta')} q(alpha=0.1)={conformal.quantile(0.1):.3f}")
    return meta


# --------------------------------------------------------------------------- persistence helpers
def package_versions() -> dict:
    import lightgbm
    import rdkit
    import sklearn

    import solpred

    return {"python": platform.python_version(), "solpred": solpred.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "lightgbm": lightgbm.__version__,
            "rdkit": rdkit.__version__, "joblib": joblib.__version__}


def dataset_version(cfg: dict, dataset: str) -> dict:
    v = load_json(path("processed", cfg) / "DATASET_VERSION.json")
    return {"dataset": dataset, "dataset_version": v.get("dataset_version"), "rdkit_curation": v.get("rdkit"),
            "file": f"{dataset}_clean.csv", "sha256": v.get("files", {}).get(f"{dataset}_clean.csv")}


def load_bundle(model_dir: str | Path) -> dict:
    """Load ``model.joblib`` (dict with ensemble, conformal, AD, ...) from a model directory."""
    return joblib.load(Path(model_dir) / "model.joblib")


def retrain_command(cfg: TrainConfig, out_dir: Path) -> str:
    rel = out_dir.relative_to(ROOT) if out_dir.is_relative_to(ROOT) else out_dir
    params = json.dumps(cfg.params, separators=(",", ":"))
    return (f".venv/Scripts/python.exe scripts/11_train_final_model.py --model {cfg.model} --features {cfg.features} "
            f"--n-members {cfg.n_members} --params-json '{params}' --out {rel.as_posix()}")


# --------------------------------------------------------------------------- main entry point
def train_final_model(cfg: TrainConfig, out_dir: str | Path, n_jobs: int = 2, project_cfg: dict | None = None,
                      cv_only: bool = False, log: Callable = print) -> dict:
    """Train, calibrate and persist the production model; returns the metadata dict."""
    pcfg = project_cfg or load_config()
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    work = out_dir / "_work" / cfg.key()
    work.mkdir(parents=True, exist_ok=True)
    save_json(asdict(cfg), work / "config.json")
    ad_cfg = pcfg["applicability_domain"]
    t_start = time.time()

    df = load_dataset(cfg.dataset, pcfg)
    X, ids, feat_names = load_or_compute(cfg.dataset, cfg.features, pcfg)
    df = df.set_index("compound_id").loc[list(ids)].reset_index()
    y = df["logS"].to_numpy(dtype=float)
    groups, generic = scaffold_groups(df["scaffold"], df["compound_id"], cfg.max_group_size)
    folds = fold_assignment(groups, cfg.n_folds, cfg.seed)
    n_specific = int((~generic).sum())
    grouping = {"scheme": "GroupKFold over Bemis-Murcko scaffolds (generic scaffolds -> singleton groups)",
                "n_folds": cfg.n_folds, "fold_sizes": np.bincount(folds).tolist(),
                "n_groups": int(len(np.unique(groups))),
                "n_generic_compounds": int(generic.sum()),
                "n_generic_acyclic": int((df["scaffold"] == "").sum()),
                "generic_scaffolds": sorted({s for s, g in zip(df["scaffold"], generic) if g and s}),
                "n_specific_compounds": n_specific,
                "n_specific_scaffolds": int(df.loc[~generic, "scaffold"].nunique()),
                "max_group_size": cfg.max_group_size}
    log(f"[final] {len(y)} compounds, {X.shape[1]} features ({cfg.features}); folds {grouping['fold_sizes']}; "
        f"{grouping['n_generic_compounds']} generic-scaffold compounds as singletons")

    # ---- 1. scaffold-grouped CV -> OOF ensemble predictions
    P, cv_infos = cross_validate(X, y, folds, cfg, work, n_jobs=n_jobs, log=log)
    oof_mean = P.mean(axis=0)
    oof_std = P.std(axis=0, ddof=1) if cfg.n_members > 1 else np.zeros(len(y))
    maxsim_file = work / "oof_maxsim.npy"
    if maxsim_file.exists():
        oof_sim = np.load(maxsim_file)
    else:
        oof_sim = oof_max_similarity(df["smiles"].tolist(), folds)
        np.save(maxsim_file, oof_sim)
    normalized = cfg.n_members > 1
    oof = pd.DataFrame({"compound_id": df["compound_id"], "fold": folds, "group": groups, "generic_scaffold": generic,
                        "scaffold": df["scaffold"], "source": df["source"], "y_true": y, "y_pred": oof_mean,
                        "y_std": oof_std, "ad_maxsim": oof_sim})
    thr = float(ad_cfg["similarity_in_domain"])
    conformal, conformal_meta = calibrate(oof, thr, normalized)
    for m in range(cfg.n_members):
        oof[f"member{m}"] = P[m]
    oof.to_csv(out_dir / "oof_predictions.csv", index=False)
    cv_metrics = subgroup_metrics(oof, thr)
    cv_metrics["member_mean_MAE"] = float(np.mean([np.mean(np.abs(P[m] - y)) for m in range(cfg.n_members)]))
    log(f"[final] OOF MAE={cv_metrics['overall']['MAE']:.3f} RMSE={cv_metrics['overall']['RMSE']:.3f} "
        f"R2={cv_metrics['overall']['R2']:.3f} (single-member mean MAE {cv_metrics['member_mean_MAE']:.3f})")

    meta = {
        "model_name": "solpred-final", "format_version": BUNDLE_FORMAT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "config_hash": cfg.key(),
        "dataset": {**dataset_version(pcfg, cfg.dataset), "n_train": int(len(y))},
        "features": {"kind": cfg.features, "n_features": int(X.shape[1])},
        "model": {"name": cfg.model, "params": {**DEFAULT_PARAMS.get(cfg.model, {}), **cfg.params},
                  "n_members": cfg.n_members, "bagging": "bootstrap resample per member" if cfg.bootstrap else "none",
                  "early_stopping": ("LightGBM: patience 100 (L1) on each member's out-of-bag compounds"
                                     if cfg.model == "lgbm" and cfg.early_stopping == "oob" else "none"),
                  "uncertainty": "std of member predictions (ddof=1)"},
        "cv": {**grouping, "metrics": cv_metrics, "member_fits": cv_infos},
        "conformal": conformal_meta,
        "applicability_domain": {"fingerprint": "Morgan r=2, 2048 bits (ECFP4)", "metric": "max Tanimoto to training set",
                                 "k": int(ad_cfg["knn_k"]), "sim_threshold": thr},
        "training_logS_range": [float(y.min()), float(y.max())],
        "package_versions": package_versions(),
        "retrain_command": retrain_command(cfg, out_dir),
    }
    if cv_only:
        meta["cv_only"] = True
        save_json(meta, out_dir / "metadata_cv_only.json")
        return meta

    # ---- 2. production ensemble on all compounds
    members, final_infos = [], []
    for m in range(cfg.n_members):
        f = work / f"final_member{m}.joblib"
        if f.exists():
            model, info = joblib.load(f)
        else:
            model, info = fit_member(X, y, cfg, member_seed(cfg.seed, -1, m), n_jobs)
            info.update(member=m)
            joblib.dump((model, info), f, compress=3)
            log(f"  final member {m}: n_iter={info.get('n_iter', '-')} oob_MAE={info.get('oob_MAE', float('nan')):.3f} "
                f"({info['fit_seconds']:.0f}s)")
        members.append(model)
        final_infos.append(info)
    meta["model"]["member_fits"] = final_infos

    # ---- 3. applicability domain + bundle
    ad = ApplicabilityDomain(k=int(ad_cfg["knn_k"]), sim_threshold=thr).fit(
        df["smiles"].tolist(), ids=df["compound_id"].tolist(), labels=y)
    bundle = {
        "format_version": BUNDLE_FORMAT_VERSION, "members": members, "model_name": cfg.model,
        "features": cfg.features, "feature_names": list(feat_names), "conformal": conformal, "ad": ad,
        "training_logS_range": (float(y.min()), float(y.max())),
        "train_index": dict(zip(df["smiles_2d"], zip(df["compound_id"], y.astype(float)))),
        "curation": {"organic_elements": list(pcfg["curation"]["organic_elements"]),
                     "counterion_elements": list(pcfg["curation"]["counterion_elements"])},
        "features_cfg": dict(pcfg["features"]),  # Morgan radius/nbits used at training time
        "config": asdict(cfg),
    }
    meta["training_seconds_this_run"] = round(time.time() - t_start, 1)
    joblib.dump(bundle, out_dir / "model.joblib", compress=3)
    save_json(meta, out_dir / "metadata.json")
    for m in range(cfg.n_members):  # the bundle now holds the members; keep only the small CV caches
        (work / f"final_member{m}.joblib").unlink(missing_ok=True)
    log(f"[final] saved {out_dir / 'model.joblib'} ({(out_dir / 'model.joblib').stat().st_size / 1e6:.1f} MB)")
    return meta
