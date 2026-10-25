"""Step 11 - train the production logS predictor (bagged ensemble + scaffold-grouped conformal + AD).

Trains on ALL curated AqSolDB compounds (see ``solpred.final_model`` for the method) and writes
``models/final/``: ``model.joblib`` (ensemble, feature kind, conformal calibrator, applicability domain,
training logS range), ``metadata.json`` (dataset hash, params, OOF CV metrics, conformal coverage,
package versions), ``oof_predictions.csv``, ``summary.json`` and ``figures/*.png``.

Examples
--------
Default (LightGBM on rdkit2d+morgan_counts, registry params, 5 members, n_jobs=3)::

    .venv/Scripts/python.exe scripts/11_train_final_model.py

Retrain with parameters recommended by the benchmark (inline JSON, a flat JSON file, or a JSON file
with ``{"model": ..., "features": ..., "params": {...}}``)::

    .venv/Scripts/python.exe scripts/11_train_final_model.py --params-json results/benchmark/best_params.json --force

or from the tuned-parameter YAML written by the tuning step (one entry per key; an entry is either a flat
parameter mapping or ``{model:, features:, params: {...}}``; a key of the form ``<model>__<features>`` also
supplies model and features)::

    .venv/Scripts/python.exe scripts/11_train_final_model.py --params-yaml configs/models_tuned.yaml         --params-key lgbm__rdkit2d+morgan_counts --n-jobs 3 --force

The run is resumable: every CV/final fit is cached under ``<out>/_work/<config-hash>/``; re-running with the
same configuration continues where it stopped (``model.joblib`` with the same config hash => nothing to do).
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from solpred.config import ROOT, load_config, load_json, save_json  # noqa: E402
from solpred.final_model import MIN_EPS, TrainConfig, lofo_coverage, recalibrate, train_final_model  # noqa: E402
from solpred.uncertainty import ConformalRegressor, coverage  # noqa: E402

SCAFFOLD_COLOR = "#DD8452"   # scaffold-grouped CV
RANDOM_COLOR = "#4C72B0"
CLUSTER_COLOR = "#55A868"
EXTERNAL_COLOR = "#C44E52"
# bookkeeping fields a tuning YAML may store next to the hyper-parameters (never passed to the learner)
NON_PARAM_KEY = re.compile(r"^(best|val|cv|test|score|mae|rmse|r2|n_trials|trials?|metric|source|notes?|split)", re.I)


def load_params(arg: str | None) -> dict:
    """``--params-json``: a path to a JSON file or an inline JSON object."""
    if not arg:
        return {}
    p = Path(arg)
    if not p.is_absolute():
        p = ROOT / p
    text = p.read_text(encoding="utf-8") if p.exists() else arg
    obj = json.loads(text)
    if not isinstance(obj, dict):
        raise ValueError("--params-json must be a JSON object")
    return obj


def _find_entry(tree: dict, key: str) -> dict | None:
    """``tree[key]``, also looking one level down (e.g. ``{"models": {key: ...}}``) and in dotted paths."""
    if key in tree and isinstance(tree[key], dict):
        return tree[key]
    node = tree
    for part in key.split("."):
        node = node.get(part) if isinstance(node, dict) else None
    if isinstance(node, dict):
        return node
    for sub in tree.values():
        if isinstance(sub, dict) and isinstance(sub.get(key), dict):
            return sub[key]
    return None


def load_params_yaml(yaml_path: str, key: str | None) -> dict:
    """``--params-yaml/--params-key``: one entry of a YAML file of tuned params, normalised to the JSON form.

    Returns ``{"model": ..., "features": ..., "params": {...}}`` (model/features only when known). A key of the
    form ``<model>__<features>`` fills model and features when the entry itself does not name them.
    """
    import yaml

    p = Path(yaml_path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        raise SystemExit(f"--params-yaml {p} does not exist")
    tree = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(tree, dict):
        raise SystemExit(f"{p} must contain a YAML mapping")
    if key is None:
        if len(tree) != 1:
            raise SystemExit(f"{p} has several entries; choose one with --params-key (keys: {sorted(tree)})")
        key = next(iter(tree))
    entry = _find_entry(tree, key)
    if entry is None:
        raise SystemExit(f"--params-key {key!r} not found in {p} (top-level keys: {sorted(tree)})")
    params = entry.get("params", entry) if isinstance(entry.get("params"), dict) else entry
    out = {"params": {k: v for k, v in params.items()
                      if k not in ("model", "features", "featurizer") and not isinstance(v, (dict, list))
                      and not NON_PARAM_KEY.match(str(k))}}
    model = entry.get("model")
    features = entry.get("features") or entry.get("featurizer")
    if "__" in key:
        km, kf = key.split("__", 1)
        model, features = model or km, features or kf
    if model:
        out["model"] = model
    if features:
        out["features"] = features
    return out


def resolve_config(args) -> TrainConfig:
    if args.params_json and args.params_yaml:
        raise SystemExit("use either --params-json or --params-yaml, not both")
    raw = load_params_yaml(args.params_yaml, args.params_key) if args.params_yaml else load_params(args.params_json)
    params = raw.get("params", raw) if isinstance(raw.get("params"), dict) else raw
    params = {k: v for k, v in params.items() if k not in ("model", "features", "featurizer")}
    model = args.model or raw.get("model") or "lgbm"
    features = args.features or raw.get("features") or raw.get("featurizer") or "rdkit2d+morgan_counts"
    return TrainConfig(model=model, features=features, n_members=args.n_members, params=params,
                       n_folds=args.n_folds, seed=args.seed, early_stopping=args.early_stopping,
                       max_group_size=args.max_group_size)


# --------------------------------------------------------------------------- figures + summary
def fig_parity(oof: pd.DataFrame, thr: float, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 5))
    ins = oof["ad_maxsim"] >= thr
    ax.scatter(oof.loc[ins, "y_true"], oof.loc[ins, "y_pred"], s=5, alpha=0.35, c=SCAFFOLD_COLOR,
               label=f"in domain (max sim >= {thr:g}), n={ins.sum()}")
    ax.scatter(oof.loc[~ins, "y_true"], oof.loc[~ins, "y_pred"], s=7, alpha=0.6, c=EXTERNAL_COLOR, marker="x",
               label=f"out of domain, n={(~ins).sum()}")
    lim = [min(oof["y_true"].min(), oof["y_pred"].min()) - 0.3, max(oof["y_true"].max(), oof["y_pred"].max()) + 0.3]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("measured logS (log mol/L)")
    ax.set_ylabel("OOF predicted logS (log mol/L)")
    ax.set_title("Scaffold-grouped 5-fold CV of the production ensemble")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def fig_calibration(oof: pd.DataFrame, eps: float, out: Path) -> dict:
    """Coverage vs nominal level, mean width, and 90 % coverage by ensemble-std quintile.

    Compares the production calibrator (normalised by std + beta, ``eps`` = beta + 1e-3), plain std normalisation
    (beta = 0) and absolute-residual conformal; leave-one-fold-out (LOFO) curves use calibration folds only.
    """
    y, p, s = oof["y_true"].to_numpy(), oof["y_pred"].to_numpy(), oof["y_std"].to_numpy()
    folds = oof["fold"].to_numpy()
    alphas = np.array([0.5, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05, 0.02, 0.01])
    cp_n = ConformalRegressor(True, eps=eps).fit(y, p, scale=s)
    cp_a = ConformalRegressor(False).fit(y, p)
    rows = []
    for a in alphas:
        lo, hi = cp_n.interval(p, scale=s, alpha=a)
        la, ha = cp_a.interval(p, alpha=a)
        lofo = []
        for k in np.unique(folds):
            te = folds == k
            ck = ConformalRegressor(True, eps=eps).fit(y[~te], p[~te], scale=s[~te])
            l2, h2 = ck.interval(p[te], scale=s[te], alpha=a)
            lofo.append(coverage(y[te], l2, h2))
        cov0, w0 = lofo_coverage(y, p, s, folds, a, eps=MIN_EPS)
        rows.append({"nominal": 1 - a, "cov_normalized": coverage(y, lo, hi), "width_normalized": float(np.mean(hi - lo)),
                     "cov_absolute": coverage(y, la, ha), "width_absolute": float(np.mean(ha - la)),
                     "cov_lofo_mean": float(np.mean(lofo)), "cov_lofo_min": float(np.min(lofo)),
                     "cov_lofo_max": float(np.max(lofo)), "cov_beta0_lofo": float(cov0.mean()),
                     "width_beta0_lofo": float(w0.mean())})
    t = pd.DataFrame(rows)
    beta = eps - MIN_EPS
    q = pd.qcut(s, 5, labels=False, duplicates="drop")
    by_q = []
    for name, e in [(f"std + {beta:g} (production)", eps), ("std only (beta = 0)", MIN_EPS), ("absolute residual", 1e6)]:
        cov, _ = lofo_coverage(y, p, s, folds, 0.1, eps=e)
        by_q.append({"calibrator": name, "coverage_by_std_quintile": [float(cov[q == i].mean()) for i in np.unique(q)]})
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    ax = axes[0]
    ax.plot([0.45, 1], [0.45, 1], "k--", lw=1, label="ideal")
    ax.plot(t["nominal"], t["cov_normalized"], "o-", c=SCAFFOLD_COLOR, label=f"std + {beta:g} (in-sample OOF)")
    ax.errorbar(t["nominal"], t["cov_lofo_mean"], yerr=[t["cov_lofo_mean"] - t["cov_lofo_min"],
                                                         t["cov_lofo_max"] - t["cov_lofo_mean"]],
                fmt="s:", c="0.35", ms=4, capsize=3, label=f"std + {beta:g}, leave-one-fold-out (min-max)")
    ax.plot(t["nominal"], t["cov_absolute"], "^-", c=RANDOM_COLOR, ms=4, label="absolute residual")
    ax.set_xlabel("nominal coverage (1 - alpha)")
    ax.set_ylabel("empirical coverage (fraction of compounds)")
    ax.set_title("Marginal calibration (scaffold-grouped OOF)")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[1]
    ax.plot(t["nominal"], t["width_normalized"], "o-", c=SCAFFOLD_COLOR, label=f"normalised by std + {beta:g}")
    ax.plot(t["nominal"], t["width_beta0_lofo"], "d:", c=CLUSTER_COLOR, label="normalised by std (beta = 0), LOFO")
    ax.plot(t["nominal"], t["width_absolute"], "^-", c=RANDOM_COLOR, label="absolute residual (constant)")
    ax.set_xlabel("nominal coverage (1 - alpha)")
    ax.set_ylabel("mean interval width (log mol/L)")
    ax.set_title("Interval efficiency")
    ax.legend(fontsize=8, frameon=False)
    ax = axes[2]
    xs = np.arange(1, len(by_q[0]["coverage_by_std_quintile"]) + 1)
    for (r, c, m) in zip(by_q, [SCAFFOLD_COLOR, CLUSTER_COLOR, RANDOM_COLOR], ["o-", "d:", "^-"]):
        ax.plot(xs, r["coverage_by_std_quintile"], m, c=c, label=r["calibrator"])
    ax.axhline(0.9, c="k", ls="--", lw=1)
    ax.set_xticks(xs)
    ax.set_xlabel("ensemble-std quintile (1 = most confident)")
    ax.set_ylabel("LOFO coverage of 90% interval")
    ax.set_title("Conditional coverage by predicted uncertainty")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return {"table": t.round(4).to_dict(orient="records"), "coverage90_by_std_quintile_lofo": by_q}


def fig_similarity(oof: pd.DataFrame, thr: float, out: Path) -> list[dict]:
    """Absolute error, ensemble std, 90% interval width and coverage vs OOF max similarity to training."""
    bins = [0, 0.3, 0.35, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001]
    oof = oof.assign(abs_err=(oof["y_true"] - oof["y_pred"]).abs(), width90=oof["pi90_high"] - oof["pi90_low"],
                     covered90=(oof["y_true"] >= oof["pi90_low"]) & (oof["y_true"] <= oof["pi90_high"]))
    g = oof.groupby(pd.cut(oof["ad_maxsim"], bins, right=False), observed=True)
    t = pd.DataFrame({"n": g.size(), "MAE": g["abs_err"].mean(), "mean_std": g["y_std"].mean(),
                      "mean_width90": g["width90"].mean(), "coverage90": g["covered90"].mean()}).reset_index()
    t["bin"] = t["ad_maxsim"].astype(str)
    centers = [(iv.left + min(iv.right, 1.0)) / 2 for iv in t["ad_maxsim"]]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4.2))
    ax[0].plot(centers, t["MAE"], "o-", c=SCAFFOLD_COLOR, label="MAE of OOF prediction")
    ax[0].plot(centers, t["mean_width90"] / 2, "s--", c="0.3", label="mean half-width of 90% interval")
    ax[0].plot(centers, t["mean_std"], "^:", c=RANDOM_COLOR, label="mean ensemble std")
    ax[0].axvline(thr, c=EXTERNAL_COLOR, lw=1, ls="--", label=f"AD threshold {thr:g}")
    ax[0].set_xlabel("max Tanimoto (ECFP4) to training folds")
    ax[0].set_ylabel("log mol/L")
    ax[0].set_title("Error and uncertainty vs similarity")
    ax[0].legend(fontsize=8, frameon=False)
    ax[1].bar(centers, t["coverage90"], width=0.04, color=SCAFFOLD_COLOR)
    ax[1].axhline(0.9, c="k", ls="--", lw=1)
    ax[1].axvline(thr, c=EXTERNAL_COLOR, lw=1, ls="--")
    for c, cov, n in zip(centers, t["coverage90"], t["n"]):
        ax[1].text(c, min(cov + 0.01, 1.02), f"n={n}", ha="center", fontsize=7)
    ax[1].set_ylim(0.5, 1.06)
    ax[1].set_xlabel("max Tanimoto (ECFP4) to training folds")
    ax[1].set_ylabel("empirical coverage of 90% interval")
    ax[1].set_title("Conditional coverage by similarity bin")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return t.drop(columns="ad_maxsim").round(4).to_dict(orient="records")


def _rel(p: Path) -> str:
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else p.as_posix()


def write_report_artifacts(out_dir: Path) -> dict:
    meta = load_json(out_dir / "metadata.json")
    oof = pd.read_csv(out_dir / "oof_predictions.csv", keep_default_na=False, na_values=[""])
    thr = meta["applicability_domain"]["sim_threshold"]
    figs = out_dir / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    fig_parity(oof, thr, figs / "oof_parity.png")
    calib = fig_calibration(oof, float(meta["conformal"]["eps"]), figs / "conformal_calibration.png")
    sim = fig_similarity(oof, thr, figs / "error_vs_similarity.png")
    cvm, cf = meta["cv"]["metrics"], meta["conformal"]
    summary = {
        "source": f"{_rel(out_dir)}/metadata.json + oof_predictions.csv (scripts/11_train_final_model.py)",
        "n_train": meta["dataset"]["n_train"], "features": meta["features"], "model": meta["model"]["name"],
        "n_members": meta["model"]["n_members"],
        "member_n_iter": [f.get("n_iter") for f in meta["model"].get("member_fits", [])],
        "cv_grouping": {k: meta["cv"][k] for k in ("scheme", "fold_sizes", "n_groups", "n_generic_compounds",
                                                     "n_specific_compounds", "n_specific_scaffolds")},
        "oof_metrics": cvm["overall"], "oof_member_mean_MAE": cvm["member_mean_MAE"],
        "oof_per_fold_MAE": {k: v["MAE"] for k, v in cvm["per_fold"].items()},
        "oof_in_domain": cvm["in_domain"], "oof_out_of_domain": cvm["out_of_domain"],
        "oof_generic_scaffold": cvm["generic_scaffold"], "oof_specific_scaffold": cvm["specific_scaffold"],
        "oof_frac_in_domain": cvm["frac_in_domain"],
        "conformal": {a: {"q_normalized": cf[a]["q_normalized"], "q_absolute": cf[a]["q_absolute"],
                          "coverage_in_sample": cf[a]["normalized_in_sample"]["coverage"],
                          "mean_width_normalized": cf[a]["normalized_in_sample"]["mean_width"],
                          "mean_width_absolute": cf[a]["absolute_in_sample"]["mean_width"],
                          "coverage_lofo_mean": cf[a]["normalized_leave_one_fold_out"]["mean"],
                          "coverage_lofo_min": cf[a]["normalized_leave_one_fold_out"]["min"],
                          "coverage_lofo_max": cf[a]["normalized_leave_one_fold_out"]["max"],
                          "by_subgroup": cf[a]["by_subgroup"]}
                      for a in ("alpha_0.1", "alpha_0.05")},
        "spearman_std_vs_abs_error": cf["spearman_std_vs_abs_error"],
        "conformal_scale_offset_beta": cf.get("scale_offset_beta"),
        "scale_offset_selection": [{k: r[k] for k in ("beta", "coverage_lofo", "mean_width", "max_bin_deviation")}
                                   for r in cf.get("scale_offset_selection", {}).get("grid", [])],
        "coverage90_by_std_quintile_lofo": calib["coverage90_by_std_quintile_lofo"],
        "calibration_curve": calib["table"], "by_similarity_bin": sim,
        "training_logS_range": meta["training_logS_range"], "dataset_sha256": meta["dataset"]["sha256"],
        "figures": sorted(_rel(p) for p in figs.glob("*.png")),
    }
    save_json(summary, out_dir / "summary.json")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=["lgbm", "rf"], default=None, help="learner (default lgbm)")
    ap.add_argument("--features", default=None, help="feature kind (default rdkit2d+morgan_counts)")
    ap.add_argument("--n-members", type=int, default=5, help="bagged ensemble size")
    ap.add_argument("--params-json", default=None, help="path or inline JSON with model params (overrides registry)")
    ap.add_argument("--params-yaml", default=None,
                    help="YAML file of tuned params (e.g. configs/models_tuned.yaml); pick the entry with --params-key")
    ap.add_argument("--params-key", default=None,
                    help="entry of --params-yaml, e.g. lgbm__rdkit2d+morgan_counts (implies model and features)")
    ap.add_argument("--n-folds", type=int, default=5, help="scaffold-grouped CV folds for conformal calibration")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--early-stopping", choices=["oob", "none"], default="oob",
                    help="LightGBM: early-stop each member on its out-of-bag compounds (oob) or use n_estimators")
    ap.add_argument("--max-group-size", type=int, default=500,
                    help="scaffold groups larger than this (and the acyclic group) become singleton groups")
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--out", default="models/final")
    ap.add_argument("--cv-only", action="store_true", help="only run the calibration CV (sensitivity analyses)")
    ap.add_argument("--recalibrate", action="store_true",
                    help="refit only the conformal calibrator from oof_predictions.csv (no retraining), then redraw")
    ap.add_argument("--figures-only", action="store_true", help="re-draw figures/summary from an existing run")
    ap.add_argument("--force", action="store_true", help="retrain even if model.joblib has the same config hash")
    ap.add_argument("--config", default=None, help="project YAML config")
    args = ap.parse_args()

    out_dir = Path(args.out) if Path(args.out).is_absolute() else ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.recalibrate:
        recalibrate(out_dir, project_cfg=load_config(args.config))
    if args.figures_only or args.recalibrate:
        s = write_report_artifacts(out_dir)
        print(json.dumps(s["oof_metrics"], indent=1))
        return
    cfg = resolve_config(args)
    meta_file = out_dir / "metadata.json"
    if (not args.force and not args.cv_only and meta_file.exists() and (out_dir / "model.joblib").exists()
            and load_json(meta_file).get("config_hash") == cfg.key()):
        print(f"model with config hash {cfg.key()} already exists in {out_dir}; use --force to retrain")
        return
    print(f"config: {cfg} (hash {cfg.key()})", flush=True)
    meta = train_final_model(cfg, out_dir, n_jobs=args.n_jobs, project_cfg=load_config(args.config),
                             cv_only=args.cv_only, log=lambda s: print(s, flush=True))
    if not args.cv_only:
        s = write_report_artifacts(out_dir)
        print(json.dumps({"oof": s["oof_metrics"], "conformal_0.1": s["conformal"]["alpha_0.1"]["coverage_lofo_mean"]},
                         indent=1))
    else:
        print(json.dumps(meta["cv"]["metrics"]["overall"], indent=1))


if __name__ == "__main__":
    main()
