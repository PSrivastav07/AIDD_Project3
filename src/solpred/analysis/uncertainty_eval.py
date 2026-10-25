"""Uncertainty quantification, conformal prediction and applicability-domain (AD) evaluation.

Experimental design (one *run* = one split: random / scaffold / cluster x seeds 0-4, plus the two
external transfers AqSolDB->ESOL and ESOL->AqSolDB)

1. **Calibration hold-out.** A random 10 % of the split's training pool is held out as ``cal_random``;
   these compounds are exchangeable with the training compounds but never used for fitting or early
   stopping. The models are fitted on the remaining 90 % (``train_fit``).
2. **Uncertainty sources.**
   (a) a 5-member LightGBM ensemble on ``rdkit2d+morgan_counts`` (registry defaults except
       ``colsample_bytree=0.5``; members differ only in their random seed, i.e. in the row bagging
       (``subsample=0.8``) and column subsampling); prediction = member mean, sigma = member std;
   (b) the across-tree std of the Phase-1 random forest on ``rdkit2d`` (re-used from
       ``results/baseline``; that RF was fitted on the full training pool);
   (c) a model-agnostic distance: ``knn_dist = 1 - mean Tanimoto similarity of the 5 nearest training
       neighbours`` (ECFP4, 2048 bits) with respect to the compounds the model was fitted on.
3. **Quality of sigma** (on test; any rescaling is fitted on ``val``): Spearman(sigma, |error|),
   AUROC of sigma for flagging |error| > 1 log unit, sparsification curves and the area under the
   sparsification error (AUSE), Gaussian NLL / miscalibration area after a single variance scale factor
   fitted on val by maximum likelihood.
4. **Conformal prediction.** (A) *matched calibration*: calibrate on the split's ``val`` subset (for
   scaffold / cluster splits these are held-out scaffold / cluster groups, i.e. as novel as the test
   set); (B) *calibration under shift*: calibrate on ``cal_random`` (exchangeable with train) and
   evaluate on the random, cluster and scaffold test sets and on the external test sets. Because the
   *same fitted model* is used in A and B, the only difference is which compounds the conformal
   quantile is computed on.
5. **Applicability domain.** Max / mean top-5 Tanimoto similarity to ``train_fit`` and Mahalanobis
   distance in the top-20 PCA space of train-standardised RDKit descriptors; MAE per AD bin with
   bootstrap CIs; an in-domain similarity threshold chosen on the *val* subsets (Youden J for
   flagging |error| > 1).

Pipeline stages (``scripts/08_uncertainty_ad.py``): ``train_ensembles`` (expensive, resumable per member) ->
``collect_all`` (merge members + RF + AD scores per run) -> ``analyze_all`` (tables ``results/tables/uncertainty_*``
and ``ad_summary``, figures ``results/figures/uncertainty/``, per-run CSVs ``results/uncertainty/analysis/`` and
``results/uncertainty/summary.json``). ``set_output_base`` redirects all outputs (smoke tests).
"""
from __future__ import annotations

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from solpred.config import load_json, path, save_json
from solpred.experiment import labels, xy
from solpred.models import fit_model, get_model
from solpred.splits import load_split

FEATURIZER = "rdkit2d+morgan_counts"
ENS_NAME = "lgbm_ens5"
N_MEMBERS = 5
ENS_PARAMS = {"subsample": 0.8, "colsample_bytree": 0.5}
CAL_FRACTION = 0.10
KNN_K = 5
SPLIT_TYPES = ("random", "scaffold", "cluster")
EXTERNAL = ("aqsoldb_to_esol", "esol_to_aqsoldb")
PALETTE = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52",
           "aqsoldb_to_esol": "#C44E52", "esol_to_aqsoldb": "#8C1D2B"}
LABELS = {"random": "random", "scaffold": "scaffold", "cluster": "cluster",
          "aqsoldb_to_esol": "external AqSolDB->ESOL", "esol_to_aqsoldb": "external ESOL->AqSolDB"}


# --------------------------------------------------------------------------- run bookkeeping
def run_list(split_types=SPLIT_TYPES, seeds=range(5), external: bool = True) -> list[tuple[str, str, int]]:
    """``(split_type, split_name, seed)`` for every run; externals first (they are cheap/informative)."""
    runs = [("external", name, 0) for name in EXTERNAL] if external else []
    for seed in seeds:
        runs += [(st, f"seed{seed}", int(seed)) for st in split_types]
    return runs


def get_split(split_type: str, split_name: str) -> dict:
    if split_type == "external":
        return load_split("external", name=split_name)
    return load_split(split_type, int(split_name.replace("seed", "")))


def datasets_of(split: dict) -> tuple[str, str]:
    tr = split.get("train_dataset", "aqsoldb")
    return tr, split.get("test_dataset", tr)


_OUT_DIRS: dict[str, Path | None] = {"results": None, "figures": None, "tables": None}


def set_output_base(base: str | Path | None) -> None:
    """Redirect every output of this module below ``base`` (``<base>/uncertainty``, ``<base>/figures/uncertainty``,
    ``<base>/tables``); used for smoke tests. ``None`` restores the project defaults. Inputs (splits, features,
    ``results/baseline``) are always read from the project."""
    if base is None:
        _OUT_DIRS.update(results=None, figures=None, tables=None)
        return
    b = Path(base)
    _OUT_DIRS.update(results=b / "uncertainty", figures=b / "figures" / "uncertainty", tables=b / "tables")


def out_root() -> Path:
    p = _OUT_DIRS["results"] or path("results") / "uncertainty"
    p.mkdir(parents=True, exist_ok=True)
    return p


def tables_dir() -> Path:
    p = _OUT_DIRS["tables"] or path("tables")
    p.mkdir(parents=True, exist_ok=True)
    return p


def design_path(split_type: str, split_name: str) -> Path:
    return out_root() / "designs" / split_type / f"{split_name}.json"


def member_path(split_type: str, split_name: str, m: int) -> Path:
    return out_root() / "members" / split_type / split_name / f"member{m}.csv"


def prediction_file(split_type: str, split_name: str) -> Path:
    return out_root() / "predictions" / split_type / split_name / f"{ENS_NAME}__{FEATURIZER}.csv"


# --------------------------------------------------------------------------- stage 1: ensembles
def make_design(split: dict, split_type: str, split_name: str, seed: int) -> dict:
    """Random calibration hold-out from the training pool (stored so every later stage re-uses it)."""
    f = design_path(split_type, split_name)
    if f.exists():
        return load_json(f)
    pool = list(map(str, split["train"]))
    rng = np.random.default_rng(10_000 + seed)
    n_cal = int(round(CAL_FRACTION * len(pool)))
    cal_idx = set(rng.choice(len(pool), size=n_cal, replace=False).tolist())
    design = {"split_type": split_type, "split_name": split_name, "seed": seed,
              "train_dataset": datasets_of(split)[0], "test_dataset": datasets_of(split)[1],
              "cal_fraction": CAL_FRACTION, "rng_seed": 10_000 + seed,
              "train_fit": [c for i, c in enumerate(pool) if i not in cal_idx],
              "cal_random": [c for i, c in enumerate(pool) if i in cal_idx],
              "val": list(map(str, split["val"])), "test": list(map(str, split["test"]))}
    save_json(design, f)
    return design


def fit_member(design: dict, m: int, n_jobs: int = 2, extra_params: dict | None = None) -> pd.DataFrame:
    """Fit ensemble member ``m`` on ``train_fit`` (early stopping on val); predict cal_random/val/test.

    ``extra_params`` (e.g. ``{"n_estimators": 150}``) is only meant for smoke tests."""
    tr_ds, te_ds = design["train_dataset"], design["test_dataset"]
    Xtr, ytr = xy(tr_ds, FEATURIZER, design["train_fit"])
    Xva, yva = xy(tr_ds, FEATURIZER, design["val"])
    seed = 100 * design["seed"] + m
    model = get_model("lgbm", seed=seed, n_jobs=n_jobs, params={**ENS_PARAMS, **(extra_params or {})})
    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit_model(model, Xtr, ytr, Xva, yva)
    fit_s = time.time() - t0
    parts = []
    for subset, ds in (("cal_random", tr_ds), ("val", tr_ds), ("test", te_ds)):
        X, y = xy(ds, FEATURIZER, design[subset])
        parts.append(pd.DataFrame({"compound_id": design[subset], "subset": subset, "y_true": y,
                                   "y_pred": model.predict(X)}))
    out = pd.concat(parts, ignore_index=True)
    out.attrs = {"member": m, "model_seed": seed, "best_iteration": int(getattr(model, "best_iteration_", 0) or 0),
                 "fit_seconds": round(fit_s, 1)}
    return out


def train_status(runs: list[tuple[str, str, int]]) -> tuple[int, int]:
    """``(members on disk, members required)`` for the given runs."""
    have = sum(member_path(st, sn, m).exists() for st, sn, _ in runs for m in range(N_MEMBERS))
    return have, len(runs) * N_MEMBERS


def train_ensembles(runs: list[tuple[str, str, int]], n_jobs: int = 2, members=range(N_MEMBERS),
                    log=print, extra_params: dict | None = None) -> None:
    """Resumable: a member is skipped when its prediction file exists (files are written atomically)."""
    for split_type, split_name, seed in runs:
        if all(member_path(split_type, split_name, m).exists() for m in members):
            continue
        split = get_split(split_type, split_name)
        design = make_design(split, split_type, split_name, seed)
        for m in members:
            f = member_path(split_type, split_name, m)
            if f.exists():
                continue
            df = fit_member(design, m, n_jobs=n_jobs, extra_params=extra_params)
            f.parent.mkdir(parents=True, exist_ok=True)
            save_json(df.attrs, f.with_suffix(".json"))
            tmp = f.with_suffix(".csv.tmp")
            df.to_csv(tmp, index=False)
            tmp.replace(f)  # the .csv only appears once complete -> safe to resume after an interruption
            te = df[df.subset == "test"]
            log(f"{split_type:9s} {split_name:16s} member{m} best_iter={df.attrs['best_iteration']:5d} "
                f"test MAE={np.mean(np.abs(te.y_true - te.y_pred)):.3f} ({df.attrs['fit_seconds']:.0f}s)")


# --------------------------------------------------------------------------- stage 2: collect
KNN_BETA = 0.1  # offset of the kNN-distance scale (avoids zero-width intervals; fixed a priori)
_FP_CACHE: dict = {}
_SMILES_CACHE: dict = {}


def _fingerprints(dataset: str) -> dict:
    """ECFP4 (r=2, 2048 bits) bit vectors keyed by compound id (cached per dataset)."""
    if dataset not in _FP_CACHE:
        from solpred.features import load_dataset
        from solpred.splits import morgan_bits

        df = load_dataset(dataset)
        _FP_CACHE[dataset] = dict(zip(df["compound_id"], morgan_bits(df["smiles"].tolist())))
    return _FP_CACHE[dataset]


def _similarity_block(query_fps: list, ref_fps: list, ref_mask: np.ndarray, k: int = KNN_K) -> np.ndarray:
    """Per query: (maxsim, mean top-k sim, nn index) against ``ref_fps[ref_mask]``."""
    from rdkit import DataStructs

    idx = np.where(ref_mask)[0]
    out = np.empty((len(query_fps), 3))
    for i, fp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))[idx]
        j = int(np.argmax(sims))
        out[i] = (sims[j], np.partition(sims, -k)[-k:].mean(), idx[j])
    return out


def similarity_scores(design: dict) -> pd.DataFrame:
    """Tanimoto AD / kNN-distance columns for every cal_random / val / test compound.

    ``ad_*`` and ``knn_dist`` are computed against ``train_fit`` (what the ensemble was fitted on); the
    ``*_pool`` columns use the full training pool (``train_fit`` + ``cal_random``, what the Phase-1 RF was
    fitted on) and are only defined for val / test.
    """
    tr_fp, te_fp = _fingerprints(design["train_dataset"]), _fingerprints(design["test_dataset"])
    pool_ids = design["train_fit"] + design["cal_random"]
    pool_fps = [tr_fp[i] for i in pool_ids]
    fit_mask = np.r_[np.ones(len(design["train_fit"]), bool), np.zeros(len(design["cal_random"]), bool)]
    parts = []
    for subset in ("cal_random", "val", "test"):
        fps = [(te_fp if subset == "test" else tr_fp)[i] for i in design[subset]]
        a = _similarity_block(fps, pool_fps, fit_mask)
        d = pd.DataFrame({"compound_id": design[subset], "subset": subset, "ad_maxsim": a[:, 0],
                          f"ad_meansim{KNN_K}": a[:, 1], "nn_train_id": [pool_ids[int(j)] for j in a[:, 2]]})
        if subset == "cal_random":
            d["ad_maxsim_pool"] = np.nan
            d[f"ad_meansim{KNN_K}_pool"] = np.nan
        else:
            b = _similarity_block(fps, pool_fps, np.ones(len(pool_ids), bool))
            d["ad_maxsim_pool"], d[f"ad_meansim{KNN_K}_pool"] = b[:, 0], b[:, 1]
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    out["knn_dist"] = 1.0 - out[f"ad_meansim{KNN_K}"]
    out["knn_dist_pool"] = 1.0 - out[f"ad_meansim{KNN_K}_pool"]
    return out


def descriptor_scores(design: dict, n_components: int = 20) -> tuple[pd.DataFrame, dict]:
    """Mahalanobis / leverage AD in the top-PCA space of train-standardised rdkit2d descriptors."""
    from solpred.applicability import DescriptorAD

    Xtr, _ = xy(design["train_dataset"], "rdkit2d", design["train_fit"])
    dad = DescriptorAD(n_components=n_components).fit(Xtr)
    parts = []
    for subset in ("cal_random", "val", "test"):
        ds = design["test_dataset"] if subset == "test" else design["train_dataset"]
        X, _ = xy(ds, "rdkit2d", design[subset])
        parts.append(dad.score(X).assign(compound_id=design[subset], subset=subset))
    info = {"n_descriptors_used": int(len(dad.keep_)), "n_components": int(len(dad.eigenvalues_)),
            "explained_variance": float(dad.explained_variance_ratio_.sum()), "h_star": float(dad.h_star_),
            "train_frac_above_h_star": float(np.mean(1 / dad.n_train_ + dad.train_dist_ ** 2 / (dad.n_train_ - 1)
                                                     > dad.h_star_))}
    return pd.concat(parts, ignore_index=True), info


def _baseline_preds(split_type: str, split_name: str, model: str, feat: str, prefix: str) -> pd.DataFrame | None:
    f = path("results") / "baseline" / "predictions" / split_type / split_name / f"{model}__{feat}.csv"
    if not f.exists():
        return None
    b = pd.read_csv(f, dtype={"compound_id": str})
    b = b.rename(columns={"y_pred": f"{prefix}_pred", "y_std": f"{prefix}_std"})
    return b[["compound_id", "subset", f"{prefix}_pred", f"{prefix}_std"]]


def collect_run(split_type: str, split_name: str, force: bool = False) -> Path | None:
    """Merge members, baselines and AD scores into one prediction file per run (None if incomplete)."""
    f = prediction_file(split_type, split_name)
    meta_f = f.with_name("run_meta.json")
    if f.exists() and not force:
        has_rf = "rf_pred" in pd.read_csv(f, nrows=1).columns
        if has_rf or split_type == "external":
            return f
    mfiles = [member_path(split_type, split_name, m) for m in range(N_MEMBERS)]
    if not all(m.exists() for m in mfiles):
        return None
    design = load_json(design_path(split_type, split_name))
    members = [pd.read_csv(m, dtype={"compound_id": str}) for m in mfiles]
    base = members[0][["compound_id", "subset", "y_true"]].copy()
    for m in members[1:]:
        assert (m["compound_id"].to_numpy() == base["compound_id"].to_numpy()).all()
    P = np.column_stack([m["y_pred"].to_numpy() for m in members])
    df = base.assign(y_pred=P.mean(1), y_std=P.std(1))
    for j in range(N_MEMBERS):
        df[f"pred_m{j}"] = P[:, j]
    if split_type != "external":
        for model, feat, prefix in (("rf", "rdkit2d", "rf"), ("lgbm", FEATURIZER, "lgbm1")):
            b = _baseline_preds(split_type, split_name, model, feat, prefix)
            if b is not None:
                df = df.merge(b, on=["compound_id", "subset"], how="left")
        df = df.drop(columns=["lgbm1_std"], errors="ignore")
    df = df.merge(similarity_scores(design), on=["compound_id", "subset"], how="left")
    dsc, dinfo = descriptor_scores(design)
    df = df.merge(dsc, on=["compound_id", "subset"], how="left")
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(f)
    mmeta = [load_json(m.with_suffix(".json")) for m in mfiles]
    save_json({"split_type": split_type, "split_name": split_name, "seed": design["seed"],
               "train_dataset": design["train_dataset"], "test_dataset": design["test_dataset"],
               "n_train_fit": len(design["train_fit"]), "n_cal_random": len(design["cal_random"]),
               "n_val": len(design["val"]), "n_test": len(design["test"]), "members": mmeta,
               "descriptor_ad": dinfo, "has_rf": "rf_pred" in df.columns}, meta_f)
    return f


def collect_all(runs: list[tuple[str, str, int]], force: bool = False, log=print) -> None:
    for split_type, split_name, _ in runs:
        t0 = time.time()
        f = collect_run(split_type, split_name, force=force)
        log(f"collect {split_type:9s} {split_name:16s} -> {'incomplete (members missing)' if f is None else f.name}"
            f" ({time.time() - t0:.0f}s)")


# --------------------------------------------------------------------------- stage 3: analysis helpers
GROUP_ORDER = ["random", "scaffold", "cluster", "aqsoldb_to_esol", "esol_to_aqsoldb"]
# uncertainty sources: name -> (prediction column, sigma column, label)
SIGMA_SOURCES = {
    "ens_std": ("y_pred", "y_std", "LGBM ensemble std"),
    "knn_dist": ("y_pred", "u_knn", "kNN distance (1 - mean top-5 sim)"),
    "one_minus_maxsim": ("y_pred", "u_maxsim", "1 - max Tanimoto sim"),
    "mahalanobis": ("y_pred", "ad_mahalanobis", "Mahalanobis (PCA-20 rdkit2d)"),
    "constant": ("y_pred", "u_const", "constant (homoscedastic)"),
    "rf_std": ("rf_pred", "rf_std", "RF across-tree std"),
    "rf_knn_dist": ("rf_pred", "u_knn_pool", "kNN distance (for RF)"),
}
SOURCE_COLORS = {"ens_std": "#2a78d6", "knn_dist": "#eb6834", "one_minus_maxsim": "#eda100",
                 "mahalanobis": "#e87ba4", "constant": "#9a9a9a", "rf_std": "#1baf7a", "rf_knn_dist": "#7fd3b4"}
# conformal methods: name -> (prediction column, scale column or None, label)
CONFORMAL_METHODS = {
    "standard": ("y_pred", None, "standard |r|"),
    "norm_ens_std": ("y_pred", "y_std", "normalised: ensemble std"),
    "norm_knn": ("y_pred", "u_knn", "normalised: kNN distance"),
    "rf_standard": ("rf_pred", None, "RF standard |r|"),
    "rf_norm_std": ("rf_pred", "rf_std", "RF normalised: tree std"),
}
METHOD_COLORS = {"standard": "#4a4a4a", "norm_ens_std": "#2a78d6", "norm_knn": "#eb6834",
                 "rf_standard": "#9a9a9a", "rf_norm_std": "#1baf7a"}
ALPHAS = (0.2, 0.1, 0.05)
MAXSIM_BINS = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0001]
MEANSIM_BINS = [0.0, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 1.0001]
MAHAL_PCT_BINS = [0.0, 50.0, 75.0, 90.0, 95.0, 99.0, 100.0001]


def group_of(split_type: str, split_name: str) -> str:
    return split_name if split_type == "external" else split_type


def figdir() -> Path:
    p = _OUT_DIRS["figures"] or path("figures") / "uncertainty"
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_runs() -> pd.DataFrame:
    """All merged prediction files (long format) with derived sigma columns."""
    parts = []
    for f in sorted((out_root() / "predictions").glob(f"*/*/{ENS_NAME}__{FEATURIZER}.csv")):
        st, sn = f.parent.parent.name, f.parent.name
        d = pd.read_csv(f, dtype={"compound_id": str, "nn_train_id": str})
        d.insert(0, "split_name", sn)
        d.insert(0, "split_type", st)
        d.insert(0, "group", group_of(st, sn))
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df["run"] = df["split_type"] + "/" + df["split_name"]
    df["abs_err"] = (df["y_true"] - df["y_pred"]).abs()
    df["u_knn"] = df["knn_dist"] + KNN_BETA
    df["u_knn_pool"] = df["knn_dist_pool"] + KNN_BETA
    df["u_maxsim"] = 1.0 - df["ad_maxsim"] + KNN_BETA
    df["u_const"] = 1.0
    if "rf_pred" not in df.columns:
        df["rf_pred"], df["rf_std"] = np.nan, np.nan
    df["rf_abs_err"] = (df["y_true"] - df["rf_pred"]).abs()
    return df


def aggregate(df: pd.DataFrame, by: list[str], cols: list[str]) -> pd.DataFrame:
    """Mean / sd over runs (seeds) for each metric column."""
    g = df.groupby(by, sort=False)
    out = g[cols].mean().add_suffix("_mean").join(g[cols].std().add_suffix("_sd")).join(g.size().rename("n_runs"))
    return out.reset_index()


def fmt_pm(df: pd.DataFrame, cols: list[str], digits: int = 3) -> pd.DataFrame:
    """Replace ``c_mean`` / ``c_sd`` pairs by a ``"mean ± sd"`` string column (Markdown tables)."""
    out = df.copy()
    for c in cols:
        m, s = out.pop(f"{c}_mean"), out.pop(f"{c}_sd")
        txt = []
        for a, b in zip(m, s):
            if not np.isfinite(a):
                txt.append("")
            elif not np.isfinite(b):
                txt.append(f"{a:.{digits}f}")
            else:
                txt.append(f"{a:.{digits}f} ± {b:.{digits}f}")
        out[c] = txt
    return out


def write_table(df: pd.DataFrame, name: str, md: pd.DataFrame | None = None) -> None:
    t = tables_dir()
    df.round(4).to_csv(t / f"{name}.csv", index=False)
    (t / f"{name}.md").write_text((md if md is not None else df.round(3)).to_markdown(index=False), encoding="utf-8")


def boot_mae_ci(abs_err, n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Percentile bootstrap 95 % CI of the mean absolute error."""
    e = np.asarray(abs_err, dtype=float)
    if len(e) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = e[rng.integers(0, len(e), (n_boot, len(e)))].mean(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10, "legend.fontsize": 8})
    return plt


# --------------------------------------------------------------------------- accuracy of the models
def accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per run: ensemble test MAE/RMSE/R2 plus Phase-1 RF and single-LGBM MAE for context."""
    from solpred.metrics import regression_metrics

    rows = []
    for (grp, run), d in df[df.subset == "test"].groupby(["group", "run"], sort=False):
        m = regression_metrics(d.y_true, d.y_pred)
        r = {"group": grp, "run": run, "ens_MAE": m["MAE"], "ens_RMSE": m["RMSE"], "ens_R2": m["R2"]}
        for pref in ("rf", "lgbm1"):
            col = f"{pref}_pred"
            r[f"{pref}_MAE"] = float(np.mean(np.abs(d.y_true - d[col]))) if col in d and d[col].notna().all() \
                else float("nan")
        r["ens_val_MAE"] = float(df[(df.run == run) & (df.subset == "val")].abs_err.mean())
        r["ens_cal_random_MAE"] = float(df[(df.run == run) & (df.subset == "cal_random")].abs_err.mean())
        r["test_maxsim_median"] = float(d.ad_maxsim.median())
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- uncertainty quality
def quality_runs(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Per run x source: ranking + calibration metrics on test (scale fitted on val);
    also long-format sparsification and expected-vs-observed coverage curves."""
    from solpred.uncertainty import coverage_curve, fit_sigma_scale, sparsification_curve, uncertainty_quality

    rows, spars, covs = [], [], []
    for (grp, run), d in df.groupby(["group", "run"], sort=False):
        te, va = d[d.subset == "test"], d[d.subset == "val"]
        for src, (pcol, scol, _) in SIGMA_SOURCES.items():
            if te[pcol].isna().any() or te[scol].isna().any():
                continue
            q = uncertainty_quality(te.y_true, te[pcol], te[scol], va.y_true, va[pcol], va[scol])
            if src == "constant":
                for k in ("spearman_sigma_abs_err", "auroc_err_gt_thr", "AUSE", "sparsification_gain",
                          "MAE_retained_50pct"):
                    q[k] = float("nan")
            rows.append({"group": grp, "run": run, "source": src, **q})
            e = (te.y_true - te[pcol]).abs().to_numpy()
            if src != "constant":
                c = sparsification_curve(e, te[scol].to_numpy())
                spars.append(pd.DataFrame({"group": grp, "run": run, "source": src, "fraction": c["fractions"],
                                           "mae_sigma": c["sigma"], "mae_oracle": c["oracle"],
                                           "mae_random": c["random"]}))
            scale = fit_sigma_scale(va.y_true, va[pcol], va[scol])
            lv, obs = coverage_curve(te.y_true, te[pcol], scale * np.maximum(te[scol].to_numpy(), 1e-3))
            covs.append(pd.DataFrame({"group": grp, "run": run, "source": src, "expected": lv, "observed": obs}))
    return pd.DataFrame(rows), pd.concat(spars, ignore_index=True), pd.concat(covs, ignore_index=True)


def error_calibration_bins(df: pd.DataFrame, n_bins: int = 8) -> pd.DataFrame:
    """RMSE vs root-mean-variance (RMV) in equal-count bins of the val-calibrated sigma (pooled over seeds)."""
    from solpred.uncertainty import fit_sigma_scale

    rows = []
    for src in ("ens_std", "knn_dist", "rf_std"):
        pcol, scol, _ = SIGMA_SOURCES[src]
        parts = []
        for (grp, run), d in df.groupby(["group", "run"], sort=False):
            te, va = d[d.subset == "test"], d[d.subset == "val"]
            if te[pcol].isna().any():
                continue
            c = fit_sigma_scale(va.y_true, va[pcol], va[scol])
            parts.append(pd.DataFrame({"group": grp, "sig": c * np.maximum(te[scol].to_numpy(), 1e-3),
                                       "r": (te.y_true - te[pcol]).to_numpy()}))
        if not parts:
            continue
        p = pd.concat(parts, ignore_index=True)
        for grp, g in p.groupby("group", sort=False):
            g = g.assign(b=pd.qcut(g.sig.rank(method="first"), n_bins, labels=False))
            for b, h in g.groupby("b"):
                rows.append({"source": src, "group": grp, "bin": int(b), "RMV": float(np.sqrt(np.mean(h.sig ** 2))),
                             "RMSE": float(np.sqrt(np.mean(h.r ** 2))), "n": len(h)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- conformal
def conformal_runs(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage / width for every run x method x alpha under two calibration sets:
    ``val`` (matched: same split protocol as test) and ``cal_random`` (exchangeable with train)."""
    from solpred.uncertainty import conformal_evaluate

    rows = []
    for (grp, run), d in df.groupby(["group", "run"], sort=False):
        te = d[d.subset == "test"]
        for cal_name in ("val", "cal_random"):
            if cal_name == "val" and grp in EXTERNAL:
                continue  # no matched calibration set exists for an external test set
            ca = d[d.subset == cal_name]
            for meth, (pcol, scol, _) in CONFORMAL_METHODS.items():
                if ca[pcol].isna().any() or te[pcol].isna().any():
                    continue  # RF has no cal_random predictions (it was fitted on the full pool)
                res = conformal_evaluate(ca.y_true, ca[pcol], te.y_true, te[pcol],
                                         None if scol is None else ca[scol], None if scol is None else te[scol],
                                         alphas=ALPHAS)
                rows += [{"group": grp, "run": run, "calibration": cal_name, "method": meth, **r} for r in res]
    return pd.DataFrame(rows)


def conditional_coverage(df: pd.DataFrame, alpha: float = 0.1, calibration: str = "cal_random") -> pd.DataFrame:
    """Test coverage per max-similarity bin (pooled over seeds) for the ensemble-based conformal methods."""
    from solpred.uncertainty import ConformalRegressor

    parts = []
    for (grp, run), d in df.groupby(["group", "run"], sort=False):
        if calibration == "val" and grp in EXTERNAL:
            continue
        te, ca = d[d.subset == "test"], d[d.subset == calibration]
        for meth in ("standard", "norm_ens_std", "norm_knn"):
            pcol, scol, _ = CONFORMAL_METHODS[meth]
            cp = ConformalRegressor(normalized=scol is not None).fit(ca.y_true, ca[pcol],
                                                                      None if scol is None else ca[scol])
            lo, hi = cp.interval(te[pcol], None if scol is None else te[scol], alpha=alpha)
            y = te.y_true.to_numpy()
            parts.append(pd.DataFrame({"group": grp, "run": run, "method": meth, "maxsim": te.ad_maxsim.to_numpy(),
                                       "covered": (y >= lo) & (y <= hi), "width": hi - lo}))
    p = pd.concat(parts, ignore_index=True)
    p["bin"] = pd.cut(p.maxsim, MAXSIM_BINS, right=False)
    out = p.groupby(["group", "method", "bin"], observed=True).agg(
        coverage=("covered", "mean"), mean_width=("width", "mean"), n=("covered", "size")).reset_index()
    out["bin_mid"] = [iv.left + (min(iv.right, 1.0) - iv.left) / 2 for iv in out["bin"]]
    out["bin"] = out["bin"].astype(str)
    out.insert(0, "calibration", calibration)
    out.insert(1, "alpha", alpha)
    return out


# --------------------------------------------------------------------------- applicability domain
def choose_similarity_threshold(df: pd.DataFrame, grid=None, err_thr: float = 1.0) -> tuple[float, pd.DataFrame]:
    """In-domain max-similarity threshold chosen on the pooled *val* subsets of the random / scaffold /
    cluster runs: maximise Youden's J of the rule ``maxsim < t`` for flagging |error| > ``err_thr``."""
    grid = np.round(np.arange(0.20, 0.701, 0.01), 2) if grid is None else grid
    v = df[(df.subset == "val") & df.group.isin(SPLIT_TYPES)]
    bad = v.abs_err.to_numpy() > err_thr
    rows = []
    for t in grid:
        flag = v.ad_maxsim.to_numpy() < t
        tpr, fpr = flag[bad].mean(), flag[~bad].mean()
        rows.append({"threshold": t, "youden_J": tpr - fpr, "tpr": tpr, "fpr": fpr, "frac_flagged": flag.mean(),
                     "MAE_in": v.abs_err[~flag].mean(), "MAE_out": v.abs_err[flag].mean() if flag.any() else np.nan,
                     "n_val": len(v), "n_val_err_gt1": int(bad.sum())})
    tab = pd.DataFrame(rows)
    return float(tab.loc[tab.youden_J.idxmax(), "threshold"]), tab


def ad_runs(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Per run: fraction flagged and MAE in / out of domain for each AD rule (test subset, ensemble errors)."""
    from solpred.uncertainty import error_flag_auroc

    rows = []
    for (grp, run), d in df[df.subset == "test"].groupby(["group", "run"], sort=False):
        e = d.abs_err.to_numpy()
        r = {"group": grp, "run": run, "n_test": len(d), "MAE_all": e.mean(),
             "auroc_1-maxsim": error_flag_auroc(e, -d.ad_maxsim), "auroc_knn_dist": error_flag_auroc(e, d.knn_dist),
             "auroc_mahalanobis": error_flag_auroc(e, d.ad_mahalanobis)}
        rules = {f"sim<{t:.2f}": d.ad_maxsim.to_numpy() < t for t in sorted(set(thresholds.values()))}
        rules["mahal_h*"] = ~d.ad_desc_in_domain.astype(bool).to_numpy()
        rules["sim_or_mahal"] = rules[f"sim<{thresholds['selected']:.2f}"] | rules["mahal_h*"]
        for name, flag in rules.items():
            r[f"{name}|frac_flagged"] = flag.mean()
            r[f"{name}|MAE_in"] = e[~flag].mean() if (~flag).any() else np.nan
            r[f"{name}|MAE_out"] = e[flag].mean() if flag.any() else np.nan
            r[f"{name}|frac_err_gt1_in"] = (e[~flag] > 1).mean() if (~flag).any() else np.nan
            r[f"{name}|frac_err_gt1_out"] = (e[flag] > 1).mean() if flag.any() else np.nan
        rows.append(r)
    return pd.DataFrame(rows)


def ad_binned(df: pd.DataFrame, n_boot: int = 1000) -> pd.DataFrame:
    """MAE (ensemble) per AD-score bin, pooled over seeds, with bootstrap 95 % CIs."""
    te = df[df.subset == "test"]
    rows = []
    for score, bins in (("ad_maxsim", MAXSIM_BINS), (f"ad_meansim{KNN_K}", MEANSIM_BINS),
                        ("ad_mahal_pct", MAHAL_PCT_BINS)):
        b = pd.cut(te[score], bins, right=False)
        for (grp, iv), g in te.groupby([te.group, b], observed=True, sort=False):
            lo, hi = boot_mae_ci(g.abs_err, n_boot)
            rows.append({"score": score, "group": grp, "bin": str(iv), "bin_left": iv.left,
                         "bin_mid": iv.left + (min(iv.right, bins[-1] - 1e-4) - iv.left) / 2, "n": len(g),
                         "MAE": g.abs_err.mean(), "ci_low": lo, "ci_high": hi,
                         "frac_err_gt1": float((g.abs_err > 1).mean())})
    return pd.DataFrame(rows).sort_values(["score", "group", "bin_left"])


def conformal_shift_table(conf: pd.DataFrame) -> pd.DataFrame:
    """Per run x method x alpha: coverage / width with matched (``val``) vs shifted (``cal_random``) calibration."""
    c = conf[conf.method.isin(["standard", "norm_ens_std", "norm_knn"])]
    piv = c.pivot_table(index=["group", "run", "method", "alpha"], columns="calibration",
                        values=["coverage", "mean_width"])
    piv.columns = [f"{v}_{cal}" for v, cal in piv.columns]
    piv = piv.reindex(columns=[f"{v}_{cal}" for v in ("coverage", "mean_width") for cal in ("val", "cal_random")])
    out = piv.reset_index()
    out = out[out["coverage_cal_random"].notna()].copy()
    out["nominal"] = 1 - out["alpha"]
    out["gap_val"] = out["coverage_val"] - out["nominal"]
    out["gap_cal_random"] = out["coverage_cal_random"] - out["nominal"]
    out["shift_drop"] = out["coverage_cal_random"] - out["coverage_val"]
    m_order = {m: i for i, m in enumerate(CONFORMAL_METHODS)}
    out = out.assign(_m=out.method.map(m_order)).sort_values(["_m", "alpha"], ascending=[True, False])
    return _order(out.drop(columns="_m"))


def ad_summary_table(adr: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    """Long format (group x AD rule): fraction flagged and MAE / P(|err|>1) in vs out of domain, mean +- sd
    over seeds; ``score_auroc`` is the AUROC of the continuous score behind the rule for |err| > 1."""
    cfg, sel = thresholds["config"], thresholds["selected"]
    rules = {f"sim<{cfg:.2f}": (f"maxsim < {cfg:.2f} (config)", "auroc_1-maxsim"),
             f"sim<{sel:.2f}": (f"maxsim < {sel:.2f} (selected on val)", "auroc_1-maxsim"),
             "mahal_h*": ("leverage > h* (descriptor PCA-20)", "auroc_mahalanobis"),
             "sim_or_mahal": (f"maxsim < {sel:.2f} OR leverage > h*", None)}
    stats = ["frac_flagged", "MAE_in", "MAE_out", "frac_err_gt1_in", "frac_err_gt1_out"]
    parts = []
    for rule, (label, score_col) in rules.items():
        cols = {f"{rule}|{s}": s for s in stats}
        d = adr[["group", "run", "n_test", "MAE_all"] + list(cols)].rename(columns=cols)
        d["score_auroc"] = adr[score_col] if score_col else np.nan
        parts.append(d.assign(rule=label))
    long = pd.concat(parts, ignore_index=True)
    return _order(aggregate(long, ["group", "rule"], ["n_test", "MAE_all", "score_auroc"] + stats))


# --------------------------------------------------------------------------- stage 3: figures
def _order(df: pd.DataFrame, col: str = "group") -> pd.DataFrame:
    """Stable sort of rows by ``GROUP_ORDER``."""
    key = {g: i for i, g in enumerate(GROUP_ORDER)}
    return df.assign(_o=df[col].map(key)).sort_values("_o", kind="stable").drop(columns="_o").reset_index(drop=True)


def _save(fig, name: str) -> Path:
    import matplotlib.pyplot as plt

    f = figdir() / f"{name}.png"
    fig.savefig(f, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return f


def _legend_below(fig, axes, ncol: int = 4) -> None:
    """One figure-level legend (deduplicated) below the panels."""
    seen: dict = {}
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(lab, h)
    fig.legend(list(seen.values()), list(seen.keys()), loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 0.0))


def _style(ax) -> None:
    ax.grid(alpha=0.25, lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def _short(g: str) -> str:
    return LABELS[g].replace("external ", "")


def fig_sparsification(spars: pd.DataFrame, groups: list[str]) -> Path:
    """MAE of the retained test compounds vs fraction removed (mean over seeds), one panel per split."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(groups), figsize=(3.2 * len(groups), 3.2), squeeze=False)
    for ax, g in zip(axes[0], groups):
        m = spars[spars.group == g].groupby(["source", "fraction"], sort=False).mean(numeric_only=True).reset_index()
        e = m[m.source == "ens_std"]
        ax.plot(e.fraction, e.mae_oracle, color="black", ls="--", lw=1.5, label="oracle (ensemble |error|)")
        ax.plot(e.fraction, e.mae_random, color="#9a9a9a", ls=":", lw=1.5, label="random removal (ensemble)")
        for s in ("ens_std", "knn_dist", "one_minus_maxsim", "mahalanobis", "rf_std"):
            x = m[m.source == s]
            if len(x):
                lab = SIGMA_SOURCES[s][2] + (" [RF errors]" if s == "rf_std" else "")
                ax.plot(x.fraction, x.mae_sigma, color=SOURCE_COLORS[s], lw=2, ls="-." if s == "rf_std" else "-",
                        label=lab)
        ax.set_title(LABELS[g])
        ax.set_xlabel("fraction of test set removed")
        _style(ax)
    axes[0, 0].set_ylabel("MAE of retained compounds (log mol/L)")
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout()
    return _save(fig, "sparsification_curves")


def fig_coverage_curves(covs: pd.DataFrame, groups: list[str]) -> Path:
    """Observed vs expected coverage of Gaussian intervals with val-rescaled sigma (mean over seeds)."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.3), squeeze=False)
    for ax, g in zip(axes[0], groups):
        m = covs[covs.group == g].groupby(["source", "expected"], sort=False)["observed"].mean().reset_index()
        ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="perfect calibration")
        for s in ("constant", "ens_std", "knn_dist", "rf_std"):
            x = m[m.source == s]
            if len(x):
                ax.plot(x.expected, x.observed, color=SOURCE_COLORS[s], lw=2, ls="-." if s == "rf_std" else "-",
                        label=SIGMA_SOURCES[s][2] + (" [RF]" if s == "rf_std" else ""))
        ax.set_title(LABELS[g])
        ax.set_xlabel("expected coverage (val-scaled σ)")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        _style(ax)
    axes[0, 0].set_ylabel("observed test coverage")
    _legend_below(fig, axes, ncol=5)
    fig.tight_layout()
    return _save(fig, "coverage_curves")


def fig_error_calibration(eb: pd.DataFrame, groups: list[str]) -> Path:
    """RMSE vs root-mean-variance in equal-count bins of the val-calibrated sigma (pooled over seeds)."""
    plt = _mpl()
    fig, axes = plt.subplots(1, len(groups), figsize=(3.0 * len(groups), 3.3), squeeze=False)
    hi = float(np.nanmax(eb[["RMV", "RMSE"]].to_numpy())) * 1.05 if len(eb) else 1.0
    for ax, g in zip(axes[0], groups):
        ax.plot([0, hi], [0, hi], color="black", lw=1, ls="--", label="RMSE = RMV")
        for s in ("ens_std", "knn_dist", "rf_std"):
            x = eb[(eb.group == g) & (eb.source == s)]
            if len(x):
                ax.plot(x.RMV, x.RMSE, color=SOURCE_COLORS[s], lw=2, marker="o", ms=4,
                        ls="-." if s == "rf_std" else "-", label=SIGMA_SOURCES[s][2] + (" [RF]" if s == "rf_std" else ""))
        ax.set_title(LABELS[g])
        ax.set_xlabel("RMV of calibrated σ (log mol/L)")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_aspect("equal")
        _style(ax)
    axes[0, 0].set_ylabel("observed RMSE (log mol/L)")
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout()
    return _save(fig, "error_calibration_rmse_vs_rmv")


def fig_uq_metrics(q: pd.DataFrame, groups: list[str]) -> Path:
    """Spearman(sigma,|err|), AUROC(|err|>1) and val-calibrated test NLL per split and source (mean +- sd)."""
    plt = _mpl()
    panels = [("spearman_sigma_abs_err", "Spearman ρ(σ, |error|)"), ("auroc_err_gt_thr", "AUROC for |error| > 1"),
              ("NLL", "test Gaussian NLL (val-scaled σ)")]
    srcs = ["ens_std", "knn_dist", "one_minus_maxsim", "mahalanobis", "rf_std", "constant"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.6), squeeze=False)
    x0 = np.arange(len(groups))
    for ax, (metric, ylab) in zip(axes[0], panels):
        use = [s for s in srcs if (metric == "NLL" or s != "constant") and s in set(q.source)]
        w = 0.84 / len(use)
        for j, s in enumerate(use):
            d = q[q.source == s].groupby("group")[metric]
            mu, sd = d.mean().reindex(groups), d.std().reindex(groups).fillna(0.0)
            ax.bar(x0 - 0.42 + w * (j + 0.5), mu, w * 0.9, yerr=sd, color=SOURCE_COLORS[s], capsize=1.5,
                   error_kw={"lw": 0.8}, label=SIGMA_SOURCES[s][2] + (" [RF]" if s == "rf_std" else ""))
        if metric == "auroc_err_gt_thr":
            ax.axhline(0.5, color="black", lw=0.8, ls="--")
            ax.set_ylim(0.4, 1.0)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_xticks(x0, [_short(g) for g in groups], rotation=20, ha="right")
        ax.set_ylabel(ylab)
        _style(ax)
    _legend_below(fig, axes, ncol=6)
    fig.tight_layout()
    return _save(fig, "uq_ranking_metrics")


def fig_conformal_shift(shift: pd.DataFrame, groups: list[str], alpha: float = 0.1) -> Path:
    """Coverage and width of conformal intervals: matched (val) vs random-hold-out (cal_random) calibration."""
    plt = _mpl()
    s = shift[np.isclose(shift.alpha, alpha)]
    meths = ["standard", "norm_ens_std", "norm_knn"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), squeeze=False)
    x0 = np.arange(len(groups))
    w = 0.84 / (2 * len(meths))
    for ax, (v, ylab) in zip(axes[0], [("coverage", f"test coverage (nominal {1 - alpha:.0%})"),
                                        ("mean_width", "mean interval width (log mol/L)")]):
        k = 0
        for meth in meths:
            for cal, hatch, lab in (("val", None, "calibrated on val (matched)"),
                                    ("cal_random", "///", "calibrated on random hold-out")):
                d = s[s.method == meth].groupby("group")[f"{v}_{cal}"]
                mu, sd = d.mean().reindex(groups), d.std().reindex(groups).fillna(0.0)
                ax.bar(x0 - 0.42 + w * (k + 0.5), mu, w * 0.9, yerr=sd, color=METHOD_COLORS[meth],
                       alpha=1.0 if cal == "val" else 0.55, hatch=hatch, edgecolor="white", lw=0, capsize=1.5,
                       error_kw={"lw": 0.8}, label=f"{CONFORMAL_METHODS[meth][2]}, {lab}")
                k += 1
        if v == "coverage":
            ax.axhline(1 - alpha, color="black", lw=1, ls="--")
            lo = float(np.nanmin(s[["coverage_val", "coverage_cal_random"]].to_numpy()))
            ax.set_ylim(max(0.0, min(lo, 1 - alpha) - 0.08), 1.0)
        ax.set_xticks(x0, [_short(g) for g in groups], rotation=20, ha="right")
        ax.set_ylabel(ylab)
        _style(ax)
    _legend_below(fig, axes, ncol=3)
    fig.tight_layout()
    return _save(fig, f"conformal_matched_vs_shift_alpha{alpha:g}")


def fig_conditional_coverage(cc: pd.DataFrame, groups: list[str], alpha: float = 0.1, min_n: int = 20) -> Path:
    """Conformal test coverage per max-similarity bin; rows = calibration set, columns = split."""
    plt = _mpl()
    cals = [c for c in ("cal_random", "val") if c in set(cc.calibration)]
    fig, axes = plt.subplots(len(cals), len(groups), figsize=(3.0 * len(groups), 3.0 * len(cals)),
                             squeeze=False, sharey=True)
    for i, cal in enumerate(cals):
        for ax, g in zip(axes[i], groups):
            d = cc[(cc.calibration == cal) & (cc.group == g) & (cc.n >= min_n)]
            if d.empty:
                ax.set_axis_off()
                continue
            for meth in ("standard", "norm_ens_std", "norm_knn"):
                x = d[d.method == meth]
                ax.plot(x.bin_mid, x.coverage, color=METHOD_COLORS[meth], lw=2, marker="o", ms=4,
                        label=CONFORMAL_METHODS[meth][2])
            ax.axhline(1 - alpha, color="black", lw=1, ls="--")
            ax.set_title(f"{LABELS[g]}\ncalibrated on {'random hold-out' if cal == 'cal_random' else 'val (matched)'}",
                         fontsize=9)
            ax.set_xlabel("max Tanimoto similarity to train")
            ax.set_xlim(0.1, 1.0)
            _style(ax)
        axes[i, 0].set_ylabel(f"test coverage (nominal {1 - alpha:.0%})")
    _legend_below(fig, axes, ncol=3)
    fig.tight_layout()
    return _save(fig, f"conformal_conditional_coverage_alpha{alpha:g}")


def fig_ad_bins(adb: pd.DataFrame, groups: list[str], thresholds: dict, min_n: int = 15) -> Path:
    """Test MAE (ensemble) per AD-score bin with bootstrap 95 % CIs, one line per split (pooled over seeds)."""
    plt = _mpl()
    panels = [("ad_maxsim", "max Tanimoto similarity to train"),
              (f"ad_meansim{KNN_K}", f"mean top-{KNN_K} Tanimoto similarity to train"),
              ("ad_mahal_pct", "Mahalanobis distance percentile\n(training distribution, PCA-20)")]
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8), squeeze=False, sharey=True)
    markers = {"random": "o", "scaffold": "s", "cluster": "^", "aqsoldb_to_esol": "D", "esol_to_aqsoldb": "v"}
    for ax, (score, xlab) in zip(axes[0], panels):
        for g in groups:
            d = adb[(adb.score == score) & (adb.group == g) & (adb.n >= min_n)].sort_values("bin_left")
            if d.empty:
                continue
            ax.errorbar(d.bin_mid, d.MAE, yerr=[d.MAE - d.ci_low, d.ci_high - d.MAE], color=PALETTE[g], lw=2,
                        marker=markers[g], ms=5, capsize=2, label=LABELS[g])
        if score == "ad_maxsim":
            ax.axvline(thresholds["config"], color="#555555", ls=":", lw=1.2,
                       label=f"config threshold {thresholds['config']:.2f}")
            ax.axvline(thresholds["selected"], color="black", ls="--", lw=1.2,
                       label=f"val-selected threshold {thresholds['selected']:.2f}")
        ax.set_xlabel(xlab)
        _style(ax)
    axes[0, 0].set_ylabel("test MAE (log mol/L)")
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout()
    return _save(fig, "ad_mae_vs_score_bins")


def fig_threshold_scan(scan: pd.DataFrame, thresholds: dict) -> Path:
    """Selection of the in-domain similarity threshold on the pooled val subsets."""
    plt = _mpl()
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), squeeze=False)
    a, b = axes[0]
    a.plot(scan.threshold, scan.youden_J, color="black", lw=2, label="Youden J")
    a.plot(scan.threshold, scan.tpr, color="#2a78d6", lw=2, label="TPR (|error| > 1 flagged)")
    a.plot(scan.threshold, scan.fpr, color="#eb6834", lw=2, label="FPR (|error| <= 1 flagged)")
    a.plot(scan.threshold, scan.frac_flagged, color="#9a9a9a", lw=2, ls=":", label="fraction flagged")
    a.set_ylabel("rate (pooled val subsets)")
    b.plot(scan.threshold, scan.MAE_in, color="#1baf7a", lw=2, label="MAE in domain")
    b.plot(scan.threshold, scan.MAE_out, color="#e87ba4", lw=2, label="MAE out of domain")
    b.set_ylabel("val MAE (log mol/L)")
    for ax in (a, b):
        ax.axvline(thresholds["config"], color="#555555", ls=":", lw=1.2, label=f"config {thresholds['config']:.2f}")
        ax.axvline(thresholds["selected"], color="black", ls="--", lw=1.2,
                   label=f"selected {thresholds['selected']:.2f}")
        ax.set_xlabel("max-similarity threshold t (flag if maxsim < t)")
        _style(ax)
    _legend_below(fig, axes, ncol=4)
    fig.tight_layout()
    return _save(fig, "ad_threshold_selection")


# --------------------------------------------------------------------------- stage 3: driver + summary
QUALITY_METRICS = ["MAE", "spearman_sigma_abs_err", "auroc_err_gt_thr", "AUSE", "sparsification_gain",
                   "MAE_retained_50pct", "scale_factor", "NLL", "miscalibration_area", "coverage_68", "coverage_95"]
ACC_COLS = ["ens_MAE", "ens_RMSE", "ens_R2", "lgbm1_MAE", "rf_MAE", "ens_val_MAE", "ens_cal_random_MAE",
            "test_maxsim_median"]
SUMMARY_SOURCES = ("ens_std", "knn_dist", "one_minus_maxsim", "mahalanobis", "rf_std", "constant")
SUMMARY_Q = ("spearman_sigma_abs_err", "auroc_err_gt_thr", "AUSE", "sparsification_gain", "NLL",
             "miscalibration_area", "coverage_68", "coverage_95", "scale_factor")


def _ms(x) -> dict:
    """``{"mean", "sd", "n_runs"}`` of a series (NaNs dropped; sd is None for a single run)."""
    s = pd.Series(x, dtype=float).dropna()
    if s.empty:
        return {"mean": None, "sd": None, "n_runs": 0}
    return {"mean": round(float(s.mean()), 4), "sd": round(float(s.std()), 4) if len(s) > 1 else None,
            "n_runs": int(len(s))}


def _rel(p: Path) -> str:
    root = path("results").parent.resolve()
    try:
        return Path(p).resolve().relative_to(root).as_posix()
    except ValueError:
        return Path(p).as_posix()


def _scan_row(scan: pd.DataFrame, t: float) -> dict:
    hit = scan[np.isclose(scan.threshold, t)]
    return {} if hit.empty else {k: round(float(v), 4) for k, v in hit.iloc[0].to_dict().items()}


def build_summary(groups: list[str], acc: pd.DataFrame, q: pd.DataFrame, shift: pd.DataFrame,
                  scan: pd.DataFrame, thresholds: dict, adr: pd.DataFrame, run_meta: list[dict],
                  files: dict) -> dict:
    """Machine-readable headline numbers (mean / sd over the seeds of a split type; externals are single runs)."""
    cfg, sel = thresholds["config"], thresholds["selected"]
    rules = {f"sim<{cfg:.2f}": "maxsim_lt_config", f"sim<{sel:.2f}": "maxsim_lt_selected",
             "mahal_h*": "leverage_gt_hstar", "sim_or_mahal": "maxsim_lt_selected_or_leverage"}
    ad_stats = ("frac_flagged", "MAE_in", "MAE_out", "frac_err_gt1_in", "frac_err_gt1_out")
    conf_stats = ("coverage_val", "coverage_cal_random", "shift_drop", "mean_width_val", "mean_width_cal_random")
    s01 = shift[np.isclose(shift.alpha, 0.1)]
    return {
        "description": "LightGBM 5-member ensemble (rdkit2d+morgan_counts) uncertainty, split-conformal intervals "
                       "and applicability domain on AqSolDB random/scaffold/cluster splits (seeds 0-4) and the two "
                       "external transfers. All metrics are on test subsets; the sigma scale, conformal quantiles "
                       "and the AD threshold were fitted on val (or on the random calibration hold-out).",
        "design": {"ensemble": ENS_NAME, "featurizer": FEATURIZER, "n_members": N_MEMBERS,
                   "lgbm_param_overrides": ENS_PARAMS, "cal_fraction_of_train_pool": CAL_FRACTION,
                   "knn_k": KNN_K, "knn_beta": KNN_BETA, "alphas": list(ALPHAS),
                   "n_runs": {g: int(acc[acc.group == g].run.nunique()) for g in groups}},
        "accuracy_test": {g: {c: _ms(acc.loc[acc.group == g, c]) for c in ("ens_MAE", "lgbm1_MAE", "rf_MAE", "ens_R2")}
                          for g in groups},
        "uncertainty_quality_test": {
            g: {s: {m: _ms(q.loc[(q.group == g) & (q.source == s), m]) for m in SUMMARY_Q}
                for s in SUMMARY_SOURCES if ((q.group == g) & (q.source == s)).any()} for g in groups},
        "conformal_alpha_0.1": {
            g: {meth: {k: _ms(s01.loc[(s01.group == g) & (s01.method == meth), k]) for k in conf_stats}
                for meth in ("standard", "norm_ens_std", "norm_knn")} for g in groups},
        "applicability_domain": {
            "threshold_config": cfg, "threshold_selected": sel,
            "selection_rule": "max Youden J of 'maxsim < t' for flagging |error| > 1 on the pooled val subsets of "
                              "the random/scaffold/cluster runs (grid 0.20-0.70, step 0.01)",
            "val_scan_at_selected": _scan_row(scan, sel), "val_scan_at_config": _scan_row(scan, cfg),
            "test": {g: {name: {k: _ms(adr.loc[adr.group == g, f"{rule}|{k}"]) for k in ad_stats}
                         for rule, name in rules.items()} for g in groups},
            "descriptor_ad": {k: _ms([m["descriptor_ad"][k] for m in run_meta])
                              for k in ("explained_variance", "h_star", "train_frac_above_h_star")}},
        "files": files,
    }


def analyze_all(log=print, n_boot: int = 1000) -> dict:
    """Stage 3: every table, figure and ``summary.json`` from the merged prediction files (cheap, rerunnable)."""
    from solpred.config import load_config

    df = load_runs()
    groups = [g for g in GROUP_ORDER if g in set(df.group)]
    adir = out_root() / "analysis"
    adir.mkdir(parents=True, exist_ok=True)
    log(f"analyze: {df.run.nunique()} runs ({', '.join(groups)}), {len(df)} prediction rows")

    acc = accuracy_table(df)
    acc.to_csv(adir / "accuracy_runs.csv", index=False)
    acc_agg = _order(aggregate(acc, ["group"], ACC_COLS))
    write_table(acc_agg, "uncertainty_accuracy", fmt_pm(acc_agg, ACC_COLS))

    q, spars, covs = quality_runs(df)
    q.to_csv(adir / "quality_runs.csv", index=False)
    spars.to_csv(adir / "sparsification_curves.csv", index=False)
    covs.to_csv(adir / "coverage_curves.csv", index=False)
    q_agg = _order(aggregate(q, ["group", "source"], QUALITY_METRICS))
    write_table(q_agg, "uncertainty_quality", fmt_pm(q_agg, QUALITY_METRICS))
    eb = error_calibration_bins(df)
    eb.to_csv(adir / "error_calibration_bins.csv", index=False)
    log("analyze: uncertainty quality done")

    conf = conformal_runs(df)
    conf.to_csv(adir / "conformal_runs.csv", index=False)
    c_cols = ["coverage", "mean_width", "median_width", "q_score"]
    c_agg = _order(aggregate(conf, ["group", "calibration", "method", "alpha"], c_cols))
    write_table(c_agg, "uncertainty_conformal", fmt_pm(c_agg, c_cols))
    shift = conformal_shift_table(conf)
    shift.to_csv(adir / "conformal_shift_runs.csv", index=False)
    s_cols = ["coverage_val", "coverage_cal_random", "shift_drop", "mean_width_val", "mean_width_cal_random"]
    s_agg = _order(aggregate(shift, ["group", "method", "alpha"], s_cols))
    write_table(s_agg, "uncertainty_conformal_shift", fmt_pm(s_agg, s_cols))
    cc = pd.concat([conditional_coverage(df, 0.1, "cal_random"), conditional_coverage(df, 0.1, "val")],
                   ignore_index=True)
    write_table(_order(cc), "uncertainty_conditional_coverage")
    log("analyze: conformal done")

    thresholds = {"config": float(load_config()["applicability_domain"]["similarity_in_domain"])}
    thresholds["selected"], scan = choose_similarity_threshold(df)
    write_table(scan, "uncertainty_ad_threshold")
    adr = ad_runs(df, thresholds)
    adr.to_csv(adir / "ad_runs.csv", index=False)
    adb = _order(ad_binned(df, n_boot))
    write_table(adb, "uncertainty_ad_bins")
    ad_sum = ad_summary_table(adr, thresholds)
    a_cols = ["n_test", "MAE_all", "score_auroc", "frac_flagged", "MAE_in", "MAE_out", "frac_err_gt1_in",
              "frac_err_gt1_out"]
    ad_md = fmt_pm(ad_sum, a_cols)
    ad_md["n_test"] = ad_sum["n_test_mean"].round().astype(int)
    write_table(ad_sum, "ad_summary", ad_md)
    log(f"analyze: AD done (selected threshold {thresholds['selected']:.2f}, config {thresholds['config']:.2f})")

    figs = [fig_sparsification(spars, groups), fig_coverage_curves(covs, groups), fig_error_calibration(eb, groups),
            fig_uq_metrics(q, groups), fig_conformal_shift(shift, groups, 0.1),
            fig_conditional_coverage(cc, groups, 0.1), fig_ad_bins(adb, groups, thresholds),
            fig_threshold_scan(scan, thresholds)]
    log(f"analyze: {len(figs)} figures -> {figdir()}")

    run_meta = [load_json(p) for p in sorted((out_root() / "predictions").glob("*/*/run_meta.json"))]
    tnames = ("uncertainty_accuracy", "uncertainty_quality", "uncertainty_conformal", "uncertainty_conformal_shift",
              "uncertainty_conditional_coverage", "uncertainty_ad_threshold", "uncertainty_ad_bins", "ad_summary")
    files = {"figures": [_rel(f) for f in figs], "tables": [_rel(tables_dir() / f"{n}.csv") for n in tnames],
             "per_run_analysis": _rel(adir), "merged_predictions": _rel(out_root() / "predictions")}
    summary = build_summary(groups, acc, q, shift, scan, thresholds, adr, run_meta, files)
    save_json(summary, out_root() / "summary.json")
    log(f"analyze: summary -> {out_root() / 'summary.json'}")
    return summary
