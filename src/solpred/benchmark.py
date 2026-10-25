"""Classical-model benchmark: validation-only tuning, a featuriser x learner grid and its statistics.

Protocol
--------
1. **Hyper-parameter tuning (validation data only).** For each tuned (learner, featuriser) pair a
   fixed list of configurations is drawn by random search from a hand-chosen space
   (:func:`sample_configs`). Every configuration is fitted on the *train* part of the random seed-0
   and the scaffold seed-0 split and scored on the matching *val* part; the objective is the mean of
   the two val MAEs, so the chosen hyper-parameters have to work both for interpolation (random)
   and for unseen scaffolds. Test subsets are never loaded while tuning. LightGBM early-stops on the
   same val subset (as everywhere in the project), so its val MAE is slightly optimistic; all
   LightGBM trials share that bias, and selection only happens within a learner. Featurisers that
   are not tuned inherit the parameters of the tuned featuriser of similar size/type
   (:data:`PARAM_TRANSFER`); ``ridge`` (RidgeCV, internal GCV over alpha) and ``mlp`` use the
   registry defaults.
2. **Grid.** Baselines (training mean and the ESOL equation refit on ``esol4``; Tanimoto k-NN on
   ``morgan_bits``) plus ``{ridge, rf, lgbm, mlp} x {rdkit2d, morgan_counts, maccs,
   rdkit2d+morgan_counts}`` on random / scaffold / cluster x seeds 0-4, through
   :func:`solpred.experiment.run_split` / :func:`~solpred.experiment.save_run` (resumable).
   ``ridge`` is the registry pipeline with ``RidgeCV(gcv_mode="svd")``: the same exact leave-one-out
   GCV criterion and alpha grid, but one SVD instead of one O(p^3 + n p^2) solve per alpha (the
   default ``auto`` mode resolves to the covariance route, ~190 s per fit on 2,265 features).
3. **Statistics.** Test metrics are recomputed from the saved prediction files (the files are the
   source of truth; ``metrics_runs.csv`` only contributes timings) and aggregated as mean +- sd
   over the five seeds. The generalisation gap is MAE(scaffold or cluster) - MAE(random) of the
   same configuration (different test sets, so the sd of the gap assumes independence).
   Configurations are compared within a split type by (a) a paired bootstrap of the MAE difference
   on the test predictions pooled over the five seeds (pairs = same seed and compound; this
   captures compound sampling noise but not split-to-split variation) and (b) a Wilcoxon
   signed-rank test over the five per-seed MAEs (exact; with n = 5 the smallest attainable p is
   0.0625 two-sided / 0.031 one-sided, i.e. the test has very low power).

Outputs (driven by ``scripts/06_benchmark.py tune | grid | analyze``)
---------------------------------------------------------------------
``results/benchmark/``: ``tuning_trials.csv`` (one row per trial x tuning split, with wall and CPU
seconds), ``tuning_objective.md``, ``predictions/`` + ``metrics_runs.csv`` (shared format),
``grid_timing.csv`` (wall / CPU seconds per run), ``compute_log.csv`` (per invocation),
``dropped_configs.json`` (configs dropped for budget, with reason), ``runs_recomputed.csv`` and
``summary.json``. ``configs/models_tuned.yaml``: selected params. ``results/tables/benchmark_*``:
summary (long + wide), generalisation gap, significance, featurisation ranking, test-set logS
spread. ``results/figures/benchmark/``: grouped MAE bars, featuriser x learner heatmaps, parity
hexbins of the best config of each split (seed 0), per-seed strip plot, tuning scatter.
"""
from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin

import solpred.experiment as experiment
from solpred.config import save_json
from solpred.experiment import prediction_path, run_exists, run_split, save_run, xy
from solpred.features import load_or_compute
from solpred.metrics import METRICS, paired_bootstrap_mae_diff, regression_metrics
from solpred.models import DEFAULT_PARAMS, fit_model
from solpred.models import get_model as registry_get_model
from solpred.splits import load_split

EXPERIMENT = "benchmark"
SPLIT_TYPES = ("random", "scaffold", "cluster")
SEEDS = (0, 1, 2, 3, 4)
FEATURIZERS = ("rdkit2d", "morgan_counts", "maccs", "rdkit2d+morgan_counts")
LEARNERS = ("ridge", "rf", "lgbm", "mlp")
BASELINES = (("mean", "esol4"), ("esol_linear", "esol4"), ("knn", "morgan_bits"))
SPLIT_COLORS = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52"}

# ------------------------------------------------------------------------------------ tuning setup
TUNING_SPLITS = (("random", 0), ("scaffold", 0))
# tuned (learner, featuriser) pairs, in execution order (interleaved so that a time-out still
# leaves every learner with at least one finished search)
TUNED_PAIRS = (("knn", "morgan_bits"), ("lgbm", "rdkit2d+morgan_counts"), ("rf", "rdkit2d"),
               ("lgbm", "rdkit2d"), ("rf", "rdkit2d+morgan_counts"), ("lgbm", "morgan_counts"))
# (learner, featuriser) without its own search -> (learner, featuriser) whose tuned params it uses
PARAM_TRANSFER = {("lgbm", "maccs"): ("lgbm", "rdkit2d"),
                  ("rf", "maccs"): ("rf", "rdkit2d"),
                  ("rf", "morgan_counts"): ("rf", "rdkit2d+morgan_counts")}
KNN_K = (3, 5, 10, 20)
# max_features candidates per featuriser: fractions of the 217 descriptors, much smaller fractions
# of the 2,265-column concatenation (sparse count fingerprints; cost grows linearly with it, and
# 0.1 already tries 227 columns per split, more than the whole descriptor block)
RF_MAX_FEATURES = {"rdkit2d": ["sqrt", 0.1, 0.2, 0.33, 0.5],
                   "rdkit2d+morgan_counts": ["sqrt", 0.02, 0.03, 0.05, 0.1]}
# The number of trees is not tuned: more trees only reduce the variance of a random forest
# (Probst & Boulesteix 2018, JMLR 18:181), so it is fixed at a cost-bounded value.
RF_N_ESTIMATORS = 300
RF_MIN_SAMPLES_LEAF = (1, 2, 3, 5)


def config_label(model: str, featurizer: str) -> str:
    return f"{model}__{featurizer}"


def grid_configs(learners: Iterable[str] = LEARNERS, featurizers: Iterable[str] = FEATURIZERS,
                 baselines: bool = True) -> list[tuple[str, str]]:
    """All (model, featuriser) cells of the benchmark grid, baselines first."""
    cells = list(BASELINES) if baselines else []
    return cells + [(m, f) for m in learners for f in featurizers]


def sample_configs(model: str, featurizer: str, n: int, seed: int = 2024) -> list[dict]:
    """Deterministic random-search configurations for one (learner, featuriser) pair.

    ``lgbm``: trial 0 is the registry default (the Phase-1 reference), then ``n - 1`` random draws
    that are identical for every featuriser (common random numbers). ``rf``: stratified random
    search with ``n_estimators`` fixed (:data:`RF_N_ESTIMATORS`): ``max_features`` cycles through a
    random permutation of the featuriser's candidates (every candidate is tried once ``n >= 5``),
    ``min_samples_leaf`` is drawn at random; configs are distinct and the max_features *index*
    sequence is shared across featurisers. ``knn``: the k grid (``n`` ignored). Draws are
    prefix-stable: the first ``m`` configs for ``n > m`` equal the configs for ``n = m``, so a
    search can be extended by re-running with a larger ``n``.
    """
    rng = np.random.default_rng(seed)
    if model == "knn":
        return [{"n_neighbors": k} for k in KNN_K]
    if model == "lgbm":
        out = [{}]
        for _ in range(n - 1):
            out.append({"learning_rate": round(float(10 ** rng.uniform(np.log10(0.03), np.log10(0.15))), 4),
                        "num_leaves": int(rng.choice([15, 31, 63, 127])),
                        "min_child_samples": int(rng.choice([5, 10, 20, 40])),
                        "colsample_bytree": float(rng.choice([0.1, 0.2, 0.3, 0.5])),
                        "subsample": float(rng.choice([0.6, 0.8, 1.0])),
                        "subsample_freq": 1,
                        "reg_lambda": round(float(10 ** rng.uniform(-2, 1)), 4),
                        "n_estimators": 3000})
        return out
    if model == "rf":
        mf = RF_MAX_FEATURES[featurizer]
        n = min(n, len(mf) * len(RF_MIN_SAMPLES_LEAF))
        out, seen, i = [], set(), 0
        perm = rng.permutation(len(mf))
        while len(out) < n:
            if i and i % len(mf) == 0:
                perm = rng.permutation(len(mf))
            c = {"n_estimators": RF_N_ESTIMATORS,
                 "min_samples_leaf": int(rng.choice(RF_MIN_SAMPLES_LEAF)),
                 "max_features": mf[int(perm[i % len(mf)])]}
            i += 1
            if (key := json.dumps(c, sort_keys=True)) not in seen:
                seen.add(key)
                out.append(c)
        return out
    raise ValueError(f"no search space for {model!r}")


# ------------------------------------------------------------------------------------ model / IO patches
class TanimotoKNNRegressor(RegressorMixin, BaseEstimator):
    """k-NN regression with Jaccard/Tanimoto distance on binarised fingerprints via dense products.

    Same estimator as the registry's ``knn`` (``KNeighborsRegressor(metric="jaccard",
    algorithm="brute")`` after ``X > 0``): distance = (|a OR b| - |a AND b|) / |a OR b| in float64,
    ``weights="distance"`` with sklearn's rule for zero distances (exact matches get all the weight).
    Only the neighbour order among exactly tied distances may differ. sklearn's generic jaccard
    kernel needs ~80 s per 917 x 7337 query block on the shared CPU; this needs < 1 s.
    """

    def __init__(self, n_neighbors: int = 5, weights: str = "distance", chunk: int = 2048):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.chunk = chunk

    def fit(self, X, y):
        self.X_ = (np.asarray(X) > 0).astype(np.float32)
        self.count_ = self.X_.sum(1).astype(np.float64)
        self.y_ = np.asarray(y, dtype=float)
        return self

    def kneighbors(self, X) -> tuple[np.ndarray, np.ndarray]:
        Q = (np.asarray(X) > 0).astype(np.float32)
        k = self.n_neighbors
        dists, idxs = [], []
        for s in range(0, len(Q), self.chunk):
            q = Q[s:s + self.chunk]
            inter = (q @ self.X_.T).astype(np.float64)  # exact integer counts (<= 2048)
            union = q.sum(1).astype(np.float64)[:, None] + self.count_[None, :] - inter
            with np.errstate(invalid="ignore", divide="ignore"):
                d = np.where(union > 0, (union - inter) / union, 0.0)
            part = np.argpartition(d, k - 1, axis=1)[:, :k]
            pd_ = np.take_along_axis(d, part, axis=1)
            order = np.argsort(pd_, axis=1, kind="stable")
            idxs.append(np.take_along_axis(part, order, axis=1))
            dists.append(np.take_along_axis(pd_, order, axis=1))
        return np.vstack(dists), np.vstack(idxs)

    def predict(self, X) -> np.ndarray:
        d, idx = self.kneighbors(X)
        yn = self.y_[idx]
        if self.weights != "distance":
            return yn.mean(1)
        with np.errstate(divide="ignore"):
            w = 1.0 / d
        inf = np.isinf(w)
        rows = inf.any(1)
        w[rows] = inf[rows]
        return (w * yn).sum(1) / w.sum(1)


def benchmark_model(name: str, seed: int = 0, n_jobs: int = 4, params: dict | None = None):
    """Registry estimator with two exact, faster equivalents: ``ridge`` uses the SVD GCV route of
    RidgeCV and ``knn`` uses :class:`TanimotoKNNRegressor`."""
    if name == "knn":
        p = {**DEFAULT_PARAMS["knn"], **(params or {})}
        return TanimotoKNNRegressor(n_neighbors=int(p["n_neighbors"]), weights=p.get("weights", "distance"))
    model = registry_get_model(name, seed=seed, n_jobs=n_jobs, params=params)
    if name == "ridge":
        model.steps[-1][1].set_params(gcv_mode="svd")
    return model


_FEATURE_CACHE: dict[tuple[str, str], tuple] = {}


def _cached_load_or_compute(dataset: str, kind: str, cfg: dict | None = None):
    key = (dataset, kind)
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = load_or_compute(dataset, kind, cfg)
    return _FEATURE_CACHE[key]


@contextlib.contextmanager
def benchmark_runtime() -> Iterator[None]:
    """Temporarily route :mod:`solpred.experiment` through :func:`benchmark_model` and an in-memory
    feature cache (``run_split`` otherwise re-reads the compressed feature files three times per run).
    """
    old_model, old_load = experiment.get_model, experiment.load_or_compute
    experiment.get_model, experiment.load_or_compute = benchmark_model, _cached_load_or_compute
    try:
        yield
    finally:
        experiment.get_model, experiment.load_or_compute = old_model, old_load


# ------------------------------------------------------------------------------------ tuning
TRIAL_COLUMNS = ["model", "featurizer", "trial", "params", "split_type", "seed", "n_train", "n_val",
                 "val_MAE", "val_RMSE", "val_R2", "best_iteration", "fit_seconds", "cpu_seconds"]


def evaluate_on_val(split: dict, featurizer: str, model_name: str, seed: int, params: dict,
                    n_jobs: int) -> dict:
    """Fit on ``split['train']`` and score on ``split['val']`` only (the test ids are not touched).

    ``fit_seconds`` is wall time of fit + val prediction; ``cpu_seconds`` the process CPU time over
    the same interval (all threads), a load-independent cost measure used to size the full run.
    """
    with benchmark_runtime():
        Xtr, ytr = xy("aqsoldb", featurizer, split["train"])
        Xva, yva = xy("aqsoldb", featurizer, split["val"])
    model = benchmark_model(model_name, seed=seed, n_jobs=n_jobs, params=params)
    t0, c0 = time.time(), time.process_time()
    fit_model(model, Xtr, ytr, Xva, yva)
    pred = model.predict(Xva)
    fit_s, cpu_s = time.time() - t0, time.process_time() - c0
    m = regression_metrics(yva, pred)
    best_it = getattr(model, "best_iteration_", None)
    return {"n_train": len(ytr), "n_val": len(yva), "val_MAE": m["MAE"], "val_RMSE": m["RMSE"],
            "val_R2": m["R2"], "best_iteration": best_it if best_it else np.nan,
            "fit_seconds": round(fit_s, 2), "cpu_seconds": round(cpu_s, 2)}


def run_tuning(trials_csv: Path, models: Iterable[str], n_trials: dict[str, int], n_jobs: int,
               deadline: float | None = None, log: Callable[[str], None] = print,
               pairs: Iterable[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Random search; every (trial, tuning split) is appended to ``trials_csv`` as soon as it is done.

    Resumable: rows already present (same model, featuriser, params, split) are skipped. Returns all
    trials. Stops early (without error) once ``time.time() > deadline``. ``pairs`` restricts the
    searched (learner, featuriser) pairs (default :data:`TUNED_PAIRS`).
    """
    trials_csv = Path(trials_csv)
    done = pd.read_csv(trials_csv) if trials_csv.exists() else pd.DataFrame(columns=TRIAL_COLUMNS)
    seen = set(zip(done["model"], done["featurizer"], done["params"], done["split_type"], done["seed"].astype(int)))
    splits = {(t, s): load_split(t, s) for t, s in TUNING_SPLITS}
    for model, feat in (pair for pair in (pairs or TUNED_PAIRS) if pair[0] in set(models)):
        for trial, params in enumerate(sample_configs(model, feat, n_trials.get(model, 8))):
            pj = json.dumps(params, sort_keys=True)
            for (stype, seed), split in splits.items():
                if (model, feat, pj, stype, seed) in seen:
                    continue
                if deadline is not None and time.time() > deadline:
                    log("tuning time budget reached - stopping (re-run to resume)")
                    return pd.read_csv(trials_csv) if trials_csv.exists() else done
                res = evaluate_on_val(split, feat, model, seed, params, n_jobs)
                row = {"model": model, "featurizer": feat, "trial": trial, "params": pj,
                       "split_type": stype, "seed": seed, **res}
                pd.DataFrame([row])[TRIAL_COLUMNS].to_csv(trials_csv, mode="a", index=False,
                                                          header=not trials_csv.exists())
                log(f"tune {model:5s} {feat:22s} trial {trial:2d} {stype:8s} val MAE={res['val_MAE']:.3f} "
                    f"it={res['best_iteration']} ({res['fit_seconds']:.0f}s) {pj}")
    return pd.read_csv(trials_csv)


def tuning_objective(trials: pd.DataFrame) -> pd.DataFrame:
    """Mean val MAE over the tuning splits per configuration (only configs scored on all of them)."""
    g = trials.groupby(["model", "featurizer", "trial", "params"])
    out = g.agg(objective=("val_MAE", "mean"), n_splits=("val_MAE", "size"),
                fit_seconds=("fit_seconds", "sum")).reset_index()
    for stype, _ in TUNING_SPLITS:
        sub = trials[trials["split_type"] == stype].set_index(["model", "featurizer", "trial"])["val_MAE"]
        out[f"val_MAE_{stype}"] = [sub.get((m, f, t), np.nan) for m, f, t in
                                   zip(out["model"], out["featurizer"], out["trial"])]
    out = out[out["n_splits"] == len(TUNING_SPLITS)]
    out["rank"] = out.groupby(["model", "featurizer"])["objective"].rank(method="first")
    return out.sort_values(["model", "featurizer", "objective"]).reset_index(drop=True)


def select_params(trials: pd.DataFrame) -> dict:
    """``{model: {featuriser: {...}}}`` with the best configuration of every tuned pair plus the
    transferred pairs; params are fully resolved (registry defaults merged in)."""
    obj = tuning_objective(trials)
    out: dict = {}
    for (model, feat), grp in obj.groupby(["model", "featurizer"]):
        best = grp.sort_values("objective").iloc[0]
        params = {**DEFAULT_PARAMS.get(model, {}), **json.loads(best["params"])}
        out.setdefault(model, {})[feat] = {
            "params": _plain(params), "source": "tuned", "trial": int(best["trial"]),
            "n_trials": int(len(grp)), "objective_val_MAE": round(float(best["objective"]), 4),
            **{f"val_MAE_{s}": round(float(best[f"val_MAE_{s}"]), 4) for s, _ in TUNING_SPLITS},
            "default_objective_val_MAE": _default_objective(grp)}
    for (model, feat), (src_model, src_feat) in PARAM_TRANSFER.items():
        if src_feat in out.get(src_model, {}):
            out[model][feat] = {"params": out[src_model][src_feat]["params"],
                                "source": f"transferred from {src_model}/{src_feat}"}
    return out


def _default_objective(grp: pd.DataFrame) -> float | None:
    d = grp[grp["params"] == "{}"]
    return round(float(d["objective"].iloc[0]), 4) if len(d) else None


def _plain(d: dict) -> dict:
    """JSON/YAML-friendly copy (tuples -> lists, numpy scalars -> python)."""
    return json.loads(json.dumps(d, default=lambda o: o.item() if hasattr(o, "item") else list(o)))


def save_tuned_yaml(selected: dict, file: Path, header: str = "") -> None:
    import yaml

    doc = {"objective": "mean val MAE over " + " and ".join(f"{t}/seed{s}" for t, s in TUNING_SPLITS),
           "models": selected}
    with open(file, "w", encoding="utf-8") as fh:
        if header:
            fh.write("".join(f"# {line}\n" for line in header.splitlines()))
        yaml.safe_dump(doc, fh, sort_keys=False)


def load_tuned_params(file: Path) -> dict[tuple[str, str], dict]:
    """``{(model, featuriser): params}`` from ``configs/models_tuned.yaml`` (empty if missing)."""
    import yaml

    file = Path(file)
    if not file.exists():
        return {}
    with open(file, encoding="utf-8") as fh:
        doc = yaml.safe_load(fh) or {}
    out = {}
    for model, feats in (doc.get("models") or {}).items():
        for feat, entry in feats.items():
            p = dict(entry["params"])
            if "hidden_layer_sizes" in p:
                p["hidden_layer_sizes"] = tuple(p["hidden_layer_sizes"])
            out[(model, feat)] = p
    return out


def params_for(model: str, featurizer: str, tuned: dict[tuple[str, str], dict]) -> dict | None:
    """Tuned params for a grid cell; the k-NN baseline takes the k chosen on ``morgan_bits``."""
    return tuned.get((model, featurizer))


# ------------------------------------------------------------------------------------ grid
def run_grid(out_dir: Path, configs: list[tuple[str, str]], split_types: Iterable[str], seeds: Iterable[int],
             tuned: dict[tuple[str, str], dict], n_jobs: int, deadline: float | None = None,
             log: Callable[[str], None] = print) -> int:
    """Run every missing (config, split type, seed); returns the number of runs executed.

    Besides the shared outputs of :func:`~solpred.experiment.save_run`, the wall and process-CPU
    seconds of every run (fit + prediction) and the thread count go to ``grid_timing.csv``.
    """
    n_done = 0
    split_cache: dict = {}
    timing_csv = Path(out_dir) / "grid_timing.csv"
    for model, feat in configs:
        params = params_for(model, feat, tuned)
        for stype in split_types:
            for seed in seeds:
                if run_exists(out_dir, stype, f"seed{seed}", model, feat):
                    continue
                if deadline is not None and time.time() > deadline:
                    log("grid time budget reached - stopping (re-run to resume)")
                    return n_done
                split = split_cache.setdefault((stype, seed), load_split(stype, seed))
                t0, c0 = time.time(), time.process_time()
                with benchmark_runtime():
                    row, preds, _ = run_split(split, feat, model, seed=seed, params=params, n_jobs=n_jobs)
                wall, cpu = time.time() - t0, time.process_time() - c0
                save_run(row, preds, out_dir, EXPERIMENT)
                pd.DataFrame([{"split_type": stype, "seed": seed, "model": model, "featurizer": feat,
                               "fit_seconds": row["fit_seconds"], "run_wall_seconds": round(wall, 2),
                               "run_cpu_seconds": round(cpu, 2), "n_jobs": n_jobs}]).to_csv(
                    timing_csv, mode="a", header=not timing_csv.exists(), index=False)
                n_done += 1
                log(f"{stype:8s} seed{seed} {model:11s} {feat:22s} MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} "
                    f"R2={row['R2']:.3f} val_MAE={row['val_MAE']:.3f} ({wall:.0f}s wall, {cpu:.0f}s cpu)")
    return n_done


# ------------------------------------------------------------------------------------ loading results
def load_runs(out_dir: Path, configs: list[tuple[str, str]], split_types: Iterable[str] = SPLIT_TYPES,
              seeds: Iterable[int] = SEEDS) -> pd.DataFrame:
    """Per-run test metrics recomputed from the prediction files (+ fit time / val MAE from the log)."""
    rows = []
    for model, feat in configs:
        for stype in split_types:
            for seed in seeds:
                p = prediction_path(out_dir, stype, f"seed{seed}", model, feat)
                if not p.exists():
                    continue
                df = pd.read_csv(p)
                te, va = df[df["subset"] == "test"], df[df["subset"] == "val"]
                m = regression_metrics(te["y_true"], te["y_pred"])
                rows.append({"split_type": stype, "seed": seed, "model": model, "featurizer": feat,
                             "config": config_label(model, feat), "n_test": m["n"],
                             **{k: m[k] for k in METRICS},
                             "val_MAE": regression_metrics(va["y_true"], va["y_pred"])["MAE"],
                             "test_logS_mean": float(te["y_true"].mean()),
                             "test_logS_std": float(te["y_true"].std(ddof=1))})
    runs = pd.DataFrame(rows)
    mfile = Path(out_dir) / "metrics_runs.csv"
    if len(runs) and mfile.exists():
        log = pd.read_csv(mfile).drop_duplicates(["split_type", "seed", "model", "featurizer"], keep="last")
        runs = runs.merge(log[["split_type", "seed", "model", "featurizer", "fit_seconds"]],
                          on=["split_type", "seed", "model", "featurizer"], how="left")
    return runs


def pooled_test_predictions(out_dir: Path, split_type: str, model: str, featurizer: str,
                            seeds: Iterable[int] = SEEDS) -> pd.DataFrame:
    """Test predictions of one config concatenated over seeds (columns seed, compound_id, y_true, y_pred)."""
    parts = []
    for seed in seeds:
        df = pd.read_csv(prediction_path(out_dir, split_type, f"seed{seed}", model, featurizer))
        parts.append(df[df["subset"] == "test"].assign(seed=seed)[["seed", "compound_id", "y_true", "y_pred"]])
    return pd.concat(parts, ignore_index=True)


# ------------------------------------------------------------------------------------ summaries
def summarize_runs(runs: pd.DataFrame, n_expected: int = len(SEEDS)) -> pd.DataFrame:
    """Mean and sd over seeds per (split type, config) for every metric; ``complete`` = all seeds."""
    keys = ["split_type", "model", "featurizer", "config"]
    g = runs.groupby(keys)
    out = g[list(METRICS)].mean().add_suffix("_mean").join(g[list(METRICS)].std(ddof=1).add_suffix("_std"))
    out = out.join(g.size().rename("n_seeds")).join(g["val_MAE"].mean().rename("val_MAE_mean"))
    if "fit_seconds" in runs:
        out = out.join(g["fit_seconds"].mean().rename("fit_seconds_mean"))
    out = out.reset_index()
    out["complete"] = out["n_seeds"] == n_expected
    order = {s: i for i, s in enumerate(SPLIT_TYPES)}
    return out.sort_values(["split_type", "MAE_mean"], key=lambda c: c.map(order) if c.name == "split_type" else c)\
              .reset_index(drop=True)


def fmt_pm(mean: float, sd: float, nd: int = 3) -> str:
    return "n/a" if pd.isna(mean) else f"{mean:.{nd}f} ± {0.0 if pd.isna(sd) else sd:.{nd}f}"


def wide_table(summary: pd.DataFrame, metrics: Iterable[str] = ("MAE", "RMSE", "R2"),
               formatted: bool = True) -> pd.DataFrame:
    """One row per config; random / scaffold / cluster side by side for each metric."""
    rows = []
    for cfg, grp in summary.groupby("config", sort=False):
        r = {"model": grp["model"].iloc[0], "featurizer": grp["featurizer"].iloc[0]}
        by = grp.set_index("split_type")
        for m in metrics:
            for s in SPLIT_TYPES:
                if s in by.index:
                    mu, sd = by.loc[s, f"{m}_mean"], by.loc[s, f"{m}_std"]
                    r[f"{m} {s}"] = fmt_pm(mu, sd) if formatted else mu
                    if not formatted:
                        r[f"{m} {s} sd"] = sd
                else:
                    r[f"{m} {s}"] = "n/a" if formatted else np.nan
        r["_sort"] = by["MAE_mean"].mean()
        rows.append(r)
    return pd.DataFrame(rows).sort_values("_sort").drop(columns="_sort").reset_index(drop=True)


def generalization_gap(summary: pd.DataFrame) -> pd.DataFrame:
    """MAE (and R2) of scaffold / cluster relative to random for every config."""
    rows = []
    for cfg, grp in summary.groupby("config", sort=False):
        by = grp.set_index("split_type")
        if not set(SPLIT_TYPES) <= set(by.index):
            continue
        r = {"config": cfg, "model": grp["model"].iloc[0], "featurizer": grp["featurizer"].iloc[0]}
        for s in SPLIT_TYPES:
            r[f"MAE_{s}"] = by.loc[s, "MAE_mean"]
        for s in ("scaffold", "cluster"):
            r[f"gap_{s}_minus_random"] = by.loc[s, "MAE_mean"] - by.loc["random", "MAE_mean"]
            r[f"gap_{s}_sd"] = float(np.hypot(by.loc[s, "MAE_std"], by.loc["random", "MAE_std"]))
            r[f"ratio_{s}_over_random"] = by.loc[s, "MAE_mean"] / by.loc["random", "MAE_mean"]
            r[f"R2_drop_{s}"] = by.loc["random", "R2_mean"] - by.loc[s, "R2_mean"]
        rows.append(r)
    return pd.DataFrame(rows).sort_values("MAE_random").reset_index(drop=True)


def target_stats(runs: pd.DataFrame) -> pd.DataFrame:
    """Test-set logS mean / sd per split (identical for every config, so one config is enough)."""
    one = runs.drop_duplicates(["split_type", "seed"])[["split_type", "seed", "n_test", "test_logS_mean",
                                                        "test_logS_std"]]
    return one.sort_values(["split_type", "seed"], key=lambda c: c.map({s: i for i, s in enumerate(SPLIT_TYPES)})
                           if c.name == "split_type" else c).reset_index(drop=True)


def featurisation_ranking(summary: pd.DataFrame, learners: Iterable[str] = LEARNERS) -> dict:
    """Per learner and split type: featurisers ordered by mean MAE; plus the mean rank over splits."""
    out: dict = {}
    for model in learners:
        sub = summary[(summary["model"] == model) & summary["complete"]]
        if sub.empty:
            continue
        per_split, ranks = {}, {}
        for s in SPLIT_TYPES:
            ss = sub[sub["split_type"] == s].sort_values("MAE_mean")
            per_split[s] = [{"featurizer": f, "MAE_mean": round(m, 4), "MAE_std": round(sd, 4)}
                            for f, m, sd in zip(ss["featurizer"], ss["MAE_mean"], ss["MAE_std"])]
            for rank, f in enumerate(ss["featurizer"], 1):
                ranks.setdefault(f, []).append(rank)
        mean_rank = {f: float(np.mean(r)) for f, r in ranks.items() if len(r) == len(SPLIT_TYPES)}
        out[model] = {"by_split": per_split,
                      "mean_rank_over_splits": dict(sorted(mean_rank.items(), key=lambda kv: kv[1]))}
    return out


# ------------------------------------------------------------------------------------ significance
def compare_configs(out_dir: Path, split_type: str, a: tuple[str, str], b: tuple[str, str],
                    runs_a: pd.DataFrame, runs_b: pd.DataFrame, n_boot: int = 2000,
                    out_dir_b: Path | None = None) -> dict:
    """Paired bootstrap (pooled test predictions) + Wilcoxon over per-seed MAE for config a vs b.

    Uses the seeds present in both ``runs_a`` and ``runs_b``."""
    sa = runs_a.set_index("seed")["MAE"]
    sb = runs_b.set_index("seed")["MAE"]
    seeds = sorted(set(sa.index) & set(sb.index))
    pa = pooled_test_predictions(out_dir, split_type, *a, seeds=seeds)
    pb = pooled_test_predictions(out_dir_b or out_dir, split_type, *b, seeds=seeds)
    m = pa.merge(pb, on=["seed", "compound_id"], suffixes=("_a", "_b"), validate="one_to_one")
    assert np.allclose(m["y_true_a"], m["y_true_b"]), "labels differ between the two configs"
    boot = paired_bootstrap_mae_diff(m["y_true_a"], m["y_pred_a"], m["y_pred_b"], n_boot=n_boot, seed=0)
    d = sa.loc[seeds].to_numpy() - sb.loc[seeds].to_numpy()
    if len(seeds) >= 2 and np.any(d != 0):
        w2 = stats.wilcoxon(d, alternative="two-sided")
        w1 = stats.wilcoxon(d, alternative="less")
        w_stat, p2, p1 = float(w2.statistic), float(w2.pvalue), float(w1.pvalue)
    else:
        w_stat = p2 = p1 = float("nan")
    return {"split_type": split_type, "config_a": config_label(*a), "config_b": config_label(*b),
            "MAE_a_mean_over_seeds": float(sa.loc[seeds].mean()), "MAE_b_mean_over_seeds": float(sb.loc[seeds].mean()),
            "n_pooled": int(len(m)), "pooled_MAE_a": float(np.mean(np.abs(m["y_pred_a"] - m["y_true_a"]))),
            "pooled_MAE_b": float(np.mean(np.abs(m["y_pred_b"] - m["y_true_b"]))),
            "mae_diff_a_minus_b": boot["mae_diff"], "ci95_low": boot["ci_low"], "ci95_high": boot["ci_high"],
            "p_boot_a_not_better": boot["p_a_not_better"], "n_seeds": len(seeds),
            "seeds_a_better": int(np.sum(d < 0)), "wilcoxon_W": w_stat, "wilcoxon_p_two_sided": p2,
            "wilcoxon_p_one_sided_a_better": p1}


def significance_table(out_dir: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                       baseline_dir: Path | None = None, n_boot: int = 2000,
                       seeds: Iterable[int] = SEEDS) -> pd.DataFrame:
    """Per split type: best config vs esol_linear, vs best single-featurisation alternative, vs
    runner-up; plus tuned-vs-default for the two Phase-1 configs when their predictions exist."""
    rows = []
    for s in SPLIT_TYPES:
        sub = summary[(summary["split_type"] == s) & summary["complete"]].sort_values("MAE_mean")
        if len(sub) < 2:
            continue
        cfgs = list(zip(sub["model"], sub["featurizer"]))
        best, done = cfgs[0], set()
        for name, other in comparison_partners(cfgs):
            if other is None or other == best or other not in cfgs or other in done:
                continue
            done.add(other)
            r = compare_configs(out_dir, s, best, other, _runs_of(runs, s, best), _runs_of(runs, s, other), n_boot)
            rows.append({"comparison": name, **r})
        if baseline_dir is not None:
            rows += _tuned_vs_default(out_dir, baseline_dir, s, runs, n_boot, tuple(seeds))
    return pd.DataFrame(rows)


def comparison_partners(cfgs: list[tuple[str, str]]) -> list[tuple[str, tuple[str, str] | None]]:
    """Configs the best one (``cfgs[0]``, list sorted by mean MAE) is tested against, in order:
    the ESOL refit, the Tanimoto k-NN similarity baseline, the same learner with its best other
    featurisation (isolates the featurisation effect), the best config built on another
    featurisation (single-block if the best one is a concatenation), and the runner-up.
    Duplicates are dropped by the caller (first label wins)."""
    best = cfgs[0]
    rest = cfgs[1:]
    same_learner = next((c for c in rest if c[0] == best[0] and c[1] != best[1]), None)
    if "+" in best[1]:
        alt = next((c for c in rest if c[1] in FEATURIZERS and "+" not in c[1]), None)
        alt_name = "vs best single-featurisation config (any learner)"
    else:
        alt = next((c for c in rest if c[1] in FEATURIZERS and c[1] != best[1]), None)
        alt_name = "vs best config with another featurisation (any learner)"
    return [("vs esol_linear (ESOL refit)", ("esol_linear", "esol4")),
            ("vs Tanimoto k-NN (similarity baseline)", ("knn", "morgan_bits")),
            ("vs same learner, best other featurisation", same_learner),
            (alt_name, alt),
            ("vs runner-up", rest[0] if rest else None)]


def _runs_of(runs: pd.DataFrame, split_type: str, cfg: tuple[str, str]) -> pd.DataFrame:
    return runs[(runs["split_type"] == split_type) & (runs["model"] == cfg[0]) & (runs["featurizer"] == cfg[1])]


def _tuned_vs_default(out_dir: Path, baseline_dir: Path, split_type: str, runs: pd.DataFrame,
                      n_boot: int, seeds: tuple[int, ...] = SEEDS) -> list[dict]:
    """Tuned benchmark params vs the Phase-1 default-param predictions in ``results/baseline``."""
    rows = []
    for cfg in (("lgbm", "rdkit2d+morgan_counts"), ("rf", "rdkit2d")):
        if not all(prediction_path(baseline_dir, split_type, f"seed{k}", *cfg).exists() for k in seeds):
            continue
        mine = _runs_of(runs, split_type, cfg)
        if len(mine) < len(seeds):
            continue
        base = load_runs(baseline_dir, [cfg], [split_type], seeds)
        r = compare_configs(out_dir, split_type, cfg, cfg, mine, base, n_boot, out_dir_b=baseline_dir)
        r["config_b"] += " (Phase-1 default params)"
        rows.append({"comparison": "tuned vs default params", **r})
    return rows


# ------------------------------------------------------------------------------------ markdown helper
def to_markdown(df: pd.DataFrame, file: Path, title: str | None = None, notes: str | None = None) -> None:
    text = (f"# {title}\n\n" if title else "") + (f"{notes}\n\n" if notes else "")
    text += df.to_markdown(index=False, floatfmt=".3f") + "\n"
    Path(file).write_text(text, encoding="utf-8")


# ------------------------------------------------------------------------------------ figures
def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e3e3e3", "grid.linewidth": 0.6, "axes.axisbelow": True})
    return plt


def _config_order(summary: pd.DataFrame, by: str = "random") -> list[str]:
    s = summary[summary["split_type"] == by].sort_values("MAE_mean")
    rest = [c for c in summary["config"].unique() if c not in set(s["config"])]
    return list(s["config"]) + rest


def _pretty(cfg: str) -> str:
    model, feat = cfg.split("__")
    return f"{model} | {feat}"


def plot_grouped_bars(summary: pd.DataFrame, file: Path, n_seeds: int = len(SEEDS)) -> None:
    """MAE mean +- sd for every config, three bars (random / scaffold / cluster) per config."""
    plt = _plt()
    order = _config_order(summary)
    x = np.arange(len(order))
    w = 0.27
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(order)), 4.8))
    for i, s in enumerate(SPLIT_TYPES):
        sub = summary[summary["split_type"] == s].set_index("config").reindex(order)
        ax.bar(x + (i - 1) * w, sub["MAE_mean"], w * 0.92, yerr=sub["MAE_std"], color=SPLIT_COLORS[s],
               label=s, error_kw={"elinewidth": 0.8, "capsize": 1.5, "ecolor": "#333333"})
    ax.set_xticks(x, [_pretty(c) for c in order], rotation=50, ha="right", fontsize=7.5)
    ax.set_ylabel(f"test MAE (log mol/L), mean ± sd over {n_seeds} seeds")
    ax.set_title("Classical models on AqSolDB: test MAE by split type (sorted by random-split MAE)")
    ax.legend(title="split", ncols=3, loc="upper left", frameon=False)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_heatmaps(summary: pd.DataFrame, file: Path, learners=LEARNERS, featurizers=FEATURIZERS,
                  n_seeds: int = len(SEEDS)) -> None:
    """Featuriser x learner mean-MAE heatmap per split type (shared colour scale; grey = not run)."""
    plt = _plt()
    sub = summary[summary["model"].isin(learners) & summary["featurizer"].isin(featurizers) & summary["complete"]]
    vmin, vmax = sub["MAE_mean"].min(), sub["MAE_mean"].max()
    fig, axes = plt.subplots(1, len(SPLIT_TYPES), figsize=(13, 4.2), sharey=True)
    im = None
    for ax, s in zip(axes, SPLIT_TYPES):
        mat = np.full((len(featurizers), len(learners)), np.nan)
        sd = np.full_like(mat, np.nan)
        ss = sub[sub["split_type"] == s].set_index(["featurizer", "model"])
        for i, f in enumerate(featurizers):
            for j, m in enumerate(learners):
                if (f, m) in ss.index:
                    mat[i, j], sd[i, j] = ss.loc[(f, m), "MAE_mean"], ss.loc[(f, m), "MAE_std"]
        ax.imshow(np.where(np.isnan(mat), 1.0, np.nan), cmap="Greys", vmin=0, vmax=8, aspect="auto")
        im = ax.imshow(np.ma.masked_invalid(mat), cmap="Purples", vmin=vmin, vmax=vmax, aspect="auto")
        best = int(np.nanargmin(mat)) if np.isfinite(mat).any() else None
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isnan(mat[i, j]):
                    ax.text(j, i, "not run", ha="center", va="center", fontsize=7.5, color="#555555")
                    continue
                dark = (mat[i, j] - vmin) / max(vmax - vmin, 1e-9) > 0.55
                is_best = best is not None and np.ravel_multi_index((i, j), mat.shape) == best
                ax.text(j, i, f"{mat[i, j]:.3f}\n±{sd[i, j]:.3f}", ha="center", va="center", fontsize=8,
                        color="white" if dark else "#1a1a1a", fontweight="bold" if is_best else "normal")
        ax.set_xticks(range(len(learners)), learners)
        ax.set_yticks(range(len(featurizers)), featurizers)
        ax.set_title(f"{s} split", color=SPLIT_COLORS[s], fontweight="bold")
        ax.set_xlabel("learner")
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
    axes[0].set_ylabel("featurisation")
    cb = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cb.set_label(f"test MAE (log mol/L), mean over {n_seeds} seeds")
    fig.suptitle("Featurisation x learner: test MAE ± sd (bold = best cell of the panel)", y=0.99)
    fig.savefig(file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_parity(out_dir: Path, cfgs: dict[str, tuple[str, str]], file: Path, seed: int = 0) -> dict:
    """Hexbin parity plots, one panel per split type, each showing that split's ``cfgs[split]``
    (the best configuration of the split) on its seed-``seed`` test set."""
    plt = _plt()
    from matplotlib.colors import LinearSegmentedColormap

    splits = [s for s in SPLIT_TYPES if s in cfgs]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.6 * len(splits), 4.4), sharex=True, sharey=True,
                             squeeze=False, layout="constrained")
    axes = axes[0]
    lo, hi = -13.5, 2.8
    annot = {}
    for ax, s in zip(axes, splits):
        cfg = cfgs[s]
        df = pd.read_csv(prediction_path(out_dir, s, f"seed{seed}", *cfg))
        te = df[df["subset"] == "test"]
        m = regression_metrics(te["y_true"], te["y_pred"])
        cmap = LinearSegmentedColormap.from_list(s, ["#f2f2f2", SPLIT_COLORS[s], "#1b1b1b"])
        hb = ax.hexbin(te["y_true"], te["y_pred"], gridsize=38, extent=(lo, hi, lo, hi), mincnt=1, cmap=cmap,
                       bins="log", linewidths=0.2)
        ax.plot([lo, hi], [lo, hi], color="#333333", lw=1)
        for d in (-1, 1):
            ax.plot([lo, hi], [lo + d, hi + d], color="#777777", lw=0.8, ls="--")
        sd_y = float(te["y_true"].std(ddof=1))
        ax.text(0.03, 0.97, f"n = {m['n']}\nMAE = {m['MAE']:.3f}\nRMSE = {m['RMSE']:.3f}\nR² = {m['R2']:.3f}\n"
                f"Spearman = {m['Spearman']:.3f}\nsd(test logS) = {sd_y:.2f}", transform=ax.transAxes,
                va="top", fontsize=8, bbox={"boxstyle": "round", "fc": "white", "ec": "#cccccc"})
        ax.set_title(f"{s} split, seed {seed}\n{_pretty(config_label(*cfg))}", color=SPLIT_COLORS[s],
                     fontweight="bold")
        ax.set_xlabel("measured logS (log mol/L)")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        fig.colorbar(hb, ax=ax, shrink=0.75, pad=0.02).set_label("compounds per hexagon")
        annot[s] = {"config": config_label(*cfg), **{k: m[k] for k in METRICS}, "n": m["n"],
                    "test_logS_std": sd_y}
    axes[0].set_ylabel("predicted logS (log mol/L)")
    fig.suptitle("Parity of the best configuration of each split type (test set; dashed: ±1 log unit)")
    fig.savefig(file, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return annot


def plot_seed_strip(runs: pd.DataFrame, summary: pd.DataFrame, file: Path) -> None:
    """Per-seed test MAE of every config (dots) with the seed mean (dash) for each split type."""
    plt = _plt()
    order = _config_order(summary)
    pos = {c: i for i, c in enumerate(order)}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(max(9, 0.62 * len(order)), 4.8))
    for k, s in enumerate(SPLIT_TYPES):
        sub = runs[(runs["split_type"] == s) & runs["config"].isin(pos)]
        xs = sub["config"].map(pos).to_numpy() + (k - 1) * 0.27 + rng.uniform(-0.06, 0.06, len(sub))
        ax.scatter(xs, sub["MAE"], s=11, color=SPLIT_COLORS[s], alpha=0.85, lw=0, label=s)
        for c, v in sub.groupby("config")["MAE"].mean().items():
            x0 = pos[c] + (k - 1) * 0.27
            ax.plot([x0 - 0.11, x0 + 0.11], [v, v], color=SPLIT_COLORS[s], lw=1.6)
    ax.set_xticks(range(len(order)), [_pretty(c) for c in order], rotation=50, ha="right", fontsize=7.5)
    ax.set_ylabel("test MAE (log mol/L)")
    ax.set_title("Per-seed test MAE (dots = split seeds, dash = mean): the spread is split-to-split variability")
    ax.legend(title="split", ncols=3, loc="upper left", frameon=False, markerscale=1.5)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(file, dpi=200)
    plt.close(fig)


def plot_tuning(trials: pd.DataFrame, file: Path) -> None:
    """Val MAE on random seed-0 vs scaffold seed-0 for every tuning trial (star = selected)."""
    plt = _plt()
    obj = tuning_objective(trials)
    pairs = [p for p in TUNED_PAIRS if ((obj["model"] == p[0]) & (obj["featurizer"] == p[1])).any()]
    fig, axes = plt.subplots(1, len(pairs), figsize=(2.9 * len(pairs), 3.3), squeeze=False)
    for ax, (model, feat) in zip(axes[0], pairs):
        g = obj[(obj["model"] == model) & (obj["featurizer"] == feat)]
        ax.scatter(g["val_MAE_random"], g["val_MAE_scaffold"], s=18, color="#777777", lw=0, label="trial")
        d = g[g["params"] == "{}"]
        if len(d):
            ax.scatter(d["val_MAE_random"], d["val_MAE_scaffold"], s=46, facecolor="none", edgecolor="#C44E52",
                       lw=1.2, label="registry default")
        b = g.sort_values("objective").iloc[0]
        ax.scatter([b["val_MAE_random"]], [b["val_MAE_scaffold"]], marker="*", s=130, color="#1b1b1b",
                   label="selected")
        ax.set_title(f"{model} | {feat}", fontsize=8)
        ax.set_xlabel("val MAE, random seed 0 (log mol/L)", fontsize=7.5)
        ax.tick_params(labelsize=7)
    axes[0][0].set_ylabel("val MAE, scaffold seed 0 (log mol/L)", fontsize=7.5)
    axes[0][-1].legend(fontsize=7, frameon=False, loc="lower right")
    fig.suptitle("Random-search trials (objective = mean of the two val MAEs; test data never used)", y=1.02)
    fig.tight_layout()
    fig.savefig(file, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------------------------ analysis driver
def _records(df: pd.DataFrame, nd: int = 4) -> list[dict]:
    return json.loads(df.round(nd).to_json(orient="records"))


def _summary_markdown(summary: pd.DataFrame, wide: pd.DataFrame, file: Path, n_seeds: int = len(SEEDS)) -> None:
    long = summary[summary["complete"]].copy()
    for m in METRICS:
        long[m] = [fmt_pm(a, b) for a, b in zip(long[f"{m}_mean"], long[f"{m}_std"])]
    parts = [f"# Classical-model benchmark (AqSolDB, {n_seeds} split seeds)\n",
             f"Test-set metrics, mean ± sd over {n_seeds} split seeds; logS in log10(mol/L). "
             "Rows sorted by MAE averaged over the three split types. Numeric version: benchmark_summary.csv "
             "(long) and benchmark_summary_wide.csv. Test-set logS spread per split: "
             "benchmark_test_target_stats.md (R2 depends on it).\n",
             "## Random / scaffold / cluster side by side\n", wide.to_markdown(index=False), "\n"]
    for s in SPLIT_TYPES:
        t = long[long["split_type"] == s][["model", "featurizer", *METRICS, "n_seeds"]]
        parts += [f"\n## {s} split (all metrics, sorted by MAE)\n", t.to_markdown(index=False), "\n"]
    Path(file).write_text("\n".join(parts), encoding="utf-8")


def _target_aggregate(tstats: pd.DataFrame) -> pd.DataFrame:
    return tstats.groupby("split_type", sort=False).agg(
        n_test=("n_test", "mean"), logS_mean=("test_logS_mean", "mean"), logS_mean_sd=("test_logS_mean", "std"),
        logS_std_mean=("test_logS_std", "mean"), logS_std_sd=("test_logS_std", "std"),
        logS_std_min=("test_logS_std", "min"), logS_std_max=("test_logS_std", "max")).reset_index()


def _dropped(runs: pd.DataFrame, planned: list[tuple[str, str]], drops: dict, n_seeds: int = len(SEEDS)) -> dict:
    """Planned configs with fewer than all (split type, seed) runs, with the recorded reason."""
    n_exp = len(SPLIT_TYPES) * n_seeds
    counts = runs.groupby("config").size()
    out = {}
    for m, f in planned:
        lab = config_label(m, f)
        if counts.get(lab, 0) < n_exp:
            out[lab] = {"runs_done": int(counts.get(lab, 0)), "runs_planned": n_exp,
                        "reason": drops.get(lab, "not completed within the compute budget")}
    return out


def _compute_time(out_dir: Path, runs: pd.DataFrame) -> dict:
    """Wall-clock per stage (compute_log.csv), tuning / grid fit seconds and the thread counts used."""
    compute: dict = {}
    clog = out_dir / "compute_log.csv"
    if clog.exists():
        c = pd.read_csv(clog)
        compute["wall_seconds_by_stage"] = {k: round(float(v), 1)
                                            for k, v in c.groupby("stage")["wall_seconds"].sum().items()}
        if "n_jobs" in c:
            compute["n_jobs_used"] = sorted(int(v) for v in c["n_jobs"].dropna().unique())
    trials_csv = out_dir / "tuning_trials.csv"
    if trials_csv.exists():
        tr = pd.read_csv(trials_csv)
        compute["tuning_fit_seconds"] = round(float(tr["fit_seconds"].sum()), 1)
        compute["tuning_n_fits"] = int(len(tr))
        compute["tuning_fit_seconds_by_pair"] = {f"{m}__{f}": round(float(v), 1) for (m, f), v in
                                                 tr.groupby(["model", "featurizer"])["fit_seconds"].sum().items()}
    if "fit_seconds" in runs:
        compute["grid_fit_seconds"] = round(float(runs["fit_seconds"].sum()), 1)
        compute["grid_fit_seconds_by_model"] = {k: round(float(v), 1)
                                                for k, v in runs.groupby("model")["fit_seconds"].sum().items()}
        compute["grid_n_runs"] = int(len(runs))
    timing = out_dir / "grid_timing.csv"
    if timing.exists():
        t = pd.read_csv(timing).drop_duplicates(["split_type", "seed", "model", "featurizer"], keep="last")
        compute["grid_mean_run_wall_seconds_by_config"] = {
            f"{m}__{f}": round(float(v), 1)
            for (m, f), v in t.groupby(["model", "featurizer"])["run_wall_seconds"].mean().items()}
    compute["note"] = "wall times depend on concurrent CPU load; n_jobs threads per fit, BLAS limited to n_jobs"
    return compute


def _best_per_split(complete: pd.DataFrame) -> dict:
    """Best (lowest mean test MAE) complete config of every split type with its metrics and runner-up."""
    best = {}
    for s in SPLIT_TYPES:
        ss = complete[complete["split_type"] == s].sort_values("MAE_mean")
        if ss.empty:
            continue
        b = ss.iloc[0]
        best[s] = {"config": b["config"],
                   **{m: {"mean": round(float(b[f"{m}_mean"]), 4), "sd": round(float(b[f"{m}_std"]), 4)}
                      for m in METRICS},
                   "val_MAE_mean": round(float(b["val_MAE_mean"]), 4),
                   "runner_up": ss.iloc[1]["config"] if len(ss) > 1 else None,
                   "runner_up_MAE_mean": round(float(ss.iloc[1]["MAE_mean"]), 4) if len(ss) > 1 else None}
    return best


def _write_tables(out_dir: Path, tables_dir: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                  baseline_dir: Path | None, n_boot: int, seeds: tuple[int, ...]) -> dict:
    """All CSV + Markdown tables; returns the frames needed for summary.json."""
    complete = summary[summary["complete"]]
    n = len(seeds)
    summary.to_csv(tables_dir / "benchmark_summary.csv", index=False)
    wide_table(complete, METRICS, formatted=False).to_csv(tables_dir / "benchmark_summary_wide.csv", index=False)
    _summary_markdown(summary, wide_table(complete, ("MAE", "RMSE", "R2")), tables_dir / "benchmark_summary.md", n)
    gap = generalization_gap(complete)
    gap.to_csv(tables_dir / "benchmark_generalization_gap.csv", index=False)
    to_markdown(gap.drop(columns=["config"]), tables_dir / "benchmark_generalization_gap.md",
                "Generalisation gap: scaffold / cluster vs random (test MAE, log mol/L)",
                f"gap = MAE(split) - MAE(random) of the same configuration (means over {n} seeds); gap_sd assumes "
                "independent splits; ratio = MAE(split) / MAE(random); R2_drop = R2(random) - R2(split).")
    tstats = target_stats(runs)
    tagg = _target_aggregate(tstats)
    tstats.to_csv(tables_dir / "benchmark_test_target_stats.csv", index=False)
    to_markdown(tagg, tables_dir / "benchmark_test_target_stats.md", "Test-set logS distribution per split type",
                f"Mean over the {n} seeds of each test set's logS mean and sd (ddof=1). R2 = 1 - MSE / var(test "
                "logS): the same RMSE yields a higher R2 on a more spread-out test set. Per seed: "
                "benchmark_test_target_stats.csv.")
    sig = significance_table(out_dir, runs, complete, baseline_dir, n_boot, seeds)
    sig.to_csv(tables_dir / "benchmark_significance.csv", index=False)
    to_markdown(sig, tables_dir / "benchmark_significance.md", "Pairwise comparisons of test MAE",
                f"mae_diff = MAE(a) - MAE(b) on test predictions pooled over {n} seeds (negative: a better); the 95% "
                f"CI and p_boot (fraction of {n_boot} paired bootstrap resamples with diff >= 0) resample compounds "
                f"only, not splits. Wilcoxon signed-rank test on the {n} per-seed MAEs (exact): with n = 5 the "
                "smallest attainable p is 0.0625 two-sided / 0.031 one-sided, so it cannot reach p < 0.05 "
                "two-sided (very low power).")
    ranking = featurisation_ranking(complete)
    rank_df = pd.DataFrame([{"model": m, "split_type": s, "rank": i + 1, **e}
                            for m, d in ranking.items() for s, lst in d["by_split"].items()
                            for i, e in enumerate(lst)])
    rank_df.to_csv(tables_dir / "benchmark_featurisation_ranking.csv", index=False)
    to_markdown(rank_df, tables_dir / "benchmark_featurisation_ranking.md",
                f"Featurisation ranking per learner (test MAE, mean and sd over {n} seeds)")
    return {"gap": gap, "tagg": tagg, "sig": sig, "ranking": ranking}


def _round_any(v):
    return v if isinstance(v, str) else round(float(v), 4)


def analyze(out_dir: Path, tables_dir: Path, fig_dir: Path, baseline_dir: Path | None, tuned_yaml: Path,
            drops: dict, n_boot: int = 2000, log: Callable[[str], None] = print,
            seeds: Iterable[int] = SEEDS) -> dict:
    """Write all benchmark tables, figures and ``summary.json``; returns the summary dict.

    A configuration counts as complete for a split type when all ``seeds`` were run; only complete
    configurations enter the tables, comparisons and figures (the others are listed in summary.json
    under ``dropped_or_incomplete_configs`` with the reason recorded by ``grid --drop``).
    """
    import yaml

    out_dir, tables_dir, fig_dir = Path(out_dir), Path(tables_dir), Path(fig_dir)
    seeds = tuple(seeds)
    fig_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    planned = grid_configs()
    runs = load_runs(out_dir, planned, SPLIT_TYPES, seeds)
    if runs.empty:
        raise RuntimeError(f"no benchmark predictions under {out_dir}")
    runs.to_csv(out_dir / "runs_recomputed.csv", index=False)
    summary = summarize_runs(runs, len(seeds))
    complete = summary[summary["complete"]]
    t = _write_tables(out_dir, tables_dir, runs, summary, baseline_dir, n_boot, seeds)

    best = _best_per_split(complete)
    full = complete.groupby("config").filter(lambda g: len(g) == len(SPLIT_TYPES))
    avg = full.groupby("config")["MAE_mean"].mean().sort_values()

    # -- figures
    plot_grouped_bars(complete, fig_dir / "mae_grouped_bars.png", len(seeds))
    plot_heatmaps(summary, fig_dir / "mae_heatmap_featurizer_x_model.png", n_seeds=len(seeds))
    parity = plot_parity(out_dir, {s: tuple(b["config"].split("__")) for s, b in best.items()},
                         fig_dir / "parity_best_config_per_split_seed0.png", seed=seeds[0])
    plot_seed_strip(runs, complete, fig_dir / "mae_per_seed_strip.png")
    trials_csv = out_dir / "tuning_trials.csv"
    if trials_csv.exists():
        plot_tuning(pd.read_csv(trials_csv), fig_dir / "tuning_val_mae.png")

    tuned_doc = {}
    if Path(tuned_yaml).exists():
        with open(tuned_yaml, encoding="utf-8") as fh:
            tuned_doc = yaml.safe_load(fh) or {}
    gap_rec = {r["config"]: {k: round(float(r[k]), 4) for k in r.index if k not in ("config", "model", "featurizer")}
               for _, r in t["gap"].iterrows()}
    result = {
        "experiment": EXPERIMENT,
        "protocol": {"split_types": list(SPLIT_TYPES), "seeds": list(seeds), "aggregation": "mean ± sd over seeds",
                     "tuning_objective": tuned_doc.get("objective"), "tuned_params_file": Path(tuned_yaml).name,
                     "ridge_note": "RidgeCV(gcv_mode='svd'): same GCV criterion and alpha grid as the registry",
                     "knn_note": "Tanimoto k-NN on binarised morgan_bits (exact dense re-implementation)"},
        "best_config_per_split": best,
        "parity_seed0_best_per_split": {s: {k: _round_any(v) for k, v in d.items()} for s, d in parity.items()},
        "best_config_overall": ({"config": avg.index[0], "mean_MAE_over_splits": round(float(avg.iloc[0]), 4),
                                 "ranking_mean_MAE_over_splits": {k: round(float(v), 4) for k, v in avg.items()}}
                                if len(avg) else None),
        "featurisation_ranking": t["ranking"],
        "generalization_gap": gap_rec,
        "test_target_stats": _records(t["tagg"]),
        "significance": _records(t["sig"]),
        "tuned_params": tuned_doc.get("models", {}),
        "dropped_or_incomplete_configs": _dropped(runs, planned, drops, len(seeds)),
        "compute": _compute_time(out_dir, runs),
        "n_configs_complete_all_splits": int(len(avg)),
    }
    save_json(result, out_dir / "summary.json")
    log(f"analysis written: {tables_dir}, {fig_dir}, {out_dir / 'summary.json'}")
    return result
