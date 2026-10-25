"""Step 9 - measurement heterogeneity, data reliability and dataset shift (cross-source / external transfer).

Model unless stated: LightGBM on rdkit2d+morgan_counts with the registry default parameters, early-stopped
on the validation subset (never on test). Methods are documented in ``solpred.analysis.heterogeneity``.

Stages (``--stages``; fitting stages are resumable - runs whose prediction file exists are skipped):
  noise        intrinsic measurement variability, de-dup conflicts, ESOL vs AqSolDB agreement (no fitting)
  reliability  test error vs data reliability from the results/baseline reference predictions (no fitting)
  ablation     training-data quality ablation on random + scaffold splits, seeds 0-4 (fits)
  loso         leave-one-source-out + size-matched in-distribution control (fits)
  transfer     source-to-source matrix (A, B, C, F) with size-matched training sets (fits)
  external     AqSolDB <-> ESOL: LightGBM, RF, ESOL-linear refit, Delaney's equation (fits)
  report       tables, figures and summary.json from everything that exists

Outputs: results/heterogeneity/ (predictions, per-stage tables, summary.json),
results/figures/heterogeneity/*.png, results/tables/heterogeneity_*.{csv,md}.

Examples::

    python scripts/09_heterogeneity_transfer.py --stages ablation --split-types random --n-jobs 1
    python scripts/09_heterogeneity_transfer.py --stages loso transfer external --n-jobs 2
    python scripts/09_heterogeneity_transfer.py --stages noise reliability report
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from solpred.analysis import heterogeneity as H
from solpred.config import load_config, path, save_json
from solpred.experiment import run_split
from solpred.features import load_dataset, load_or_compute
from solpred.metrics import paired_bootstrap_mae_diff
from solpred.splits import load_split

FEAT = "rdkit2d+morgan_counts"
LGBM_FILE = f"lgbm__{FEAT}"
RF_FILE = "rf__rdkit2d"
LOSO_SOURCES = ["A", "B", "C", "E", "F"]
MATRIX_SOURCES = ["A", "B", "C", "F"]
EXTERNAL = ["aqsoldb_to_esol", "esol_to_aqsoldb"]
SIZEMATCHED = "aqsoldb_to_esol_sizematched"   # AqSolDB training set subsampled to the ESOL size
EXT_MODELS = [("lgbm", FEAT), ("rf", "rdkit2d"), ("esol_linear", "esol4")]


# --------------------------------------------------------------------------- helpers
class Paths:
    """All output locations of this step."""

    def __init__(self, cfg: dict):
        res = path("results", cfg)
        self.out = res / "heterogeneity"
        self.pred = self.out / "predictions"
        self.baseline = res / "baseline" / "predictions"
        self.fig = path("figures", cfg) / "heterogeneity"
        self.tab = path("tables", cfg)
        for p in (self.out, self.pred, self.fig, self.tab):
            p.mkdir(parents=True, exist_ok=True)


class Matrix:
    """Row lookup for a cached feature matrix."""

    def __init__(self, dataset: str, kind: str):
        self.X, ids, _ = load_or_compute(dataset, kind)
        self.pos = {c: i for i, c in enumerate(ids.astype(str))}

    def rows(self, wanted: list[str]) -> np.ndarray:
        return self.X[[self.pos[w] for w in wanted]]


def save_predictions(file: Path, parts: dict[str, tuple[list[str], np.ndarray, np.ndarray]], info: dict) -> None:
    """Write ``compound_id, subset, y_true, y_pred`` for every subset plus a JSON run record next to it."""
    frames = [pd.DataFrame({"compound_id": ids, "subset": name, "y_true": y, "y_pred": p})
              for name, (ids, y, p) in parts.items()]
    file.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(frames, ignore_index=True).to_csv(file, index=False)
    save_json(info, file.with_suffix(".json"))


def pooled_noise_s0(meta: pd.DataFrame) -> float:
    """s0 for inverse-variance weights = naive RMS of aqsol_sd over multi-report compounds (0.478)."""
    m = H.multi_report(meta)
    return H.pooled_noise(m["aqsol_sd"], m["n_occurrences"])["rms_sd_naive"]


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


# --------------------------------------------------------------------------- fitting stages
def stage_ablation(args, P: Paths, meta: pd.DataFrame) -> None:
    """Training-data quality ablation (variants a-e) on random and scaffold splits."""
    mx = Matrix("aqsoldb", FEAT)
    y = meta.set_index("compound_id")["logS"]
    midx = meta.set_index("compound_id")
    s0 = pooled_noise_s0(meta)
    for stype in args.split_types:
        for seed in args.seeds:
            sp = load_split(stype, seed)
            for variant in args.variants:
                out = P.pred / "ablation" / stype / f"seed{seed}" / f"{variant}.csv"
                if out.exists():
                    continue
                base = P.baseline / stype / f"seed{seed}" / f"{LGBM_FILE}.csv"
                if variant == "a_all" and not args.refit_reference and base.exists():
                    # identical code path (verified: max |diff| ~1e-15 on random seed0) -> reuse
                    d = pd.read_csv(base, dtype={"compound_id": str})[["compound_id", "subset", "y_true", "y_pred"]]
                    out.parent.mkdir(parents=True, exist_ok=True)
                    d.to_csv(out, index=False)
                    te = d[d["subset"] == "test"]
                    save_json({"split_type": stype, "seed": seed, "variant": variant, "n_train": len(sp["train"]),
                               "n_dropped": 0, "source": str(base), "val_MAE": float(
                                   (d.loc[d.subset == "val", "y_pred"] - d.loc[d.subset == "val", "y_true"]).abs().mean()),
                               "test_MAE": float((te["y_pred"] - te["y_true"]).abs().mean())}, out.with_suffix(".json"))
                    log(f"ablation {stype} seed{seed} {variant}: reused baseline prediction")
                    continue
                rep = int(variant.rsplit("rep", 1)[1]) if "_rep" in variant else 0
                tr_ids, w, n_drop = H.ablation_training_set(sp["train"], midx, variant, s0,
                                                            rng_seed=10_000 + 100 * seed + rep)
                t0 = time.time()
                model = H.fit_lgbm(mx.rows(tr_ids), y.loc[tr_ids].to_numpy(), mx.rows(sp["val"]),
                                   y.loc[sp["val"]].to_numpy(), seed=seed, n_jobs=args.n_jobs, sample_weight=w)
                fit_s = time.time() - t0
                pv, pt = model.predict(mx.rows(sp["val"])), model.predict(mx.rows(sp["test"]))
                yv, yt = y.loc[sp["val"]].to_numpy(), y.loc[sp["test"]].to_numpy()
                info = {"split_type": stype, "seed": seed, "variant": variant, "n_train": len(tr_ids),
                        "n_dropped": n_drop, "s0": s0, "best_iteration": int(model.best_iteration_ or 0),
                        "val_MAE": float(np.abs(pv - yv).mean()), "test_MAE": float(np.abs(pt - yt).mean()),
                        "fit_seconds": round(fit_s, 1),
                        "weights_min_max": [float(w.min()), float(w.max())] if w is not None else None}
                save_predictions(out, {"val": (sp["val"], yv, pv), "test": (sp["test"], yt, pt)}, info)
                log(f"ablation {stype} seed{seed} {variant}: n_train={len(tr_ids)} test MAE={info['test_MAE']:.3f} "
                    f"({fit_s:.0f}s)")


def stage_loso(args, P: Paths, meta: pd.DataFrame) -> None:
    """LOSO fits and the size-matched in-distribution control (K folds over each held-out source)."""
    mx = Matrix("aqsoldb", FEAT)
    y = meta.set_index("compound_id")["logS"]
    for src in args.sources:
        sp = load_split("loso", source=src)
        jobs = [(P.pred / "loso" / f"source_{src}.csv", sp, "loso")]
        for k in range(args.control_folds):
            csp = H.loso_control_split(meta, src, n_train=len(sp["train"]), n_folds=args.control_folds, fold=k,
                                       seed=0)
            jobs.append((P.pred / "loso_control" / f"source_{src}" / f"fold{k}.csv", csp, f"control_fold{k}"))
        for out, s, kind in jobs:
            if out.exists():
                continue
            t0 = time.time()
            model = H.fit_lgbm(mx.rows(s["train"]), y.loc[s["train"]].to_numpy(), mx.rows(s["val"]),
                               y.loc[s["val"]].to_numpy(), seed=0, n_jobs=args.n_jobs)
            pv, pt = model.predict(mx.rows(s["val"])), model.predict(mx.rows(s["test"]))
            yv, yt = y.loc[s["val"]].to_numpy(), y.loc[s["test"]].to_numpy()
            info = {"source": src, "kind": kind, "n_train": len(s["train"]), "n_val": len(s["val"]),
                    "n_test": len(s["test"]), "best_iteration": int(model.best_iteration_ or 0),
                    "val_MAE": float(np.abs(pv - yv).mean()), "test_MAE": float(np.abs(pt - yt).mean()),
                    "fit_seconds": round(time.time() - t0, 1)}
            save_predictions(out, {"val": (s["val"], yv, pv), "test": (s["test"], yt, pt)}, info)
            log(f"loso {src} {kind}: n_train={info['n_train']} test MAE={info['test_MAE']:.3f} "
                f"({info['fit_seconds']:.0f}s)")


def stage_transfer(args, P: Paths, meta: pd.DataFrame) -> None:
    """Source-to-source matrix: train on n_train compounds of one source (or a mix), test on every source."""
    mx = Matrix("aqsoldb", FEAT)
    y = meta.set_index("compound_id")["logS"]
    design = H.source_transfer_design(meta, MATRIX_SOURCES, test_frac=0.2, seed=0)
    save_json({s: {"n_test": len(d["test"]), "n_pool": len(d["pool"]), "test": d["test"]} for s, d in design.items()},
              P.out / "transfer_design.json")
    test_ids = [c for s in MATRIX_SOURCES for c in design[s]["test"]]
    for rep in range(args.transfer_reps):
        for k, src in enumerate(MATRIX_SOURCES + ["mixed"]):
            out = P.pred / "transfer" / f"rep{rep}" / f"train_{src}.csv"
            if out.exists():
                continue
            pool = design[src]["pool"] if src != "mixed" else [c for s in MATRIX_SOURCES for c in design[s]["pool"]]
            draw = H.subsample(pool, args.transfer_n_train + args.transfer_n_val, seed=1000 * rep + k)
            val, tr = draw[: args.transfer_n_val], draw[args.transfer_n_val:]
            t0 = time.time()
            model = H.fit_lgbm(mx.rows(tr), y.loc[tr].to_numpy(), mx.rows(val), y.loc[val].to_numpy(), seed=rep,
                               n_jobs=args.n_jobs)
            pv, pt = model.predict(mx.rows(val)), model.predict(mx.rows(test_ids))
            info = {"train_source": src, "rep": rep, "n_train": len(tr), "n_val": len(val),
                    "best_iteration": int(model.best_iteration_ or 0), "fit_seconds": round(time.time() - t0, 1),
                    "train_ids": tr}
            save_predictions(out, {"val": (val, y.loc[val].to_numpy(), pv),
                                   "test": (test_ids, y.loc[test_ids].to_numpy(), pt)}, info)
            log(f"transfer rep{rep} train_{src}: ({info['fit_seconds']:.0f}s)")


def sizematched_external_split(seed: int) -> dict:
    """AqSolDB -> ESOL with the AqSolDB training + validation set subsampled to the ESOL sizes (1004 + 111).

    Isolates the training-set-size part of the AqSolDB/ESOL transfer asymmetry; the ESOL test set and the
    removal of the 1,097 overlapping compounds from training are unchanged.
    """
    fwd, rev = load_split("external", name="aqsoldb_to_esol"), load_split("external", name="esol_to_aqsoldb")
    n_tr, n_va = len(rev["train"]), len(rev["val"])
    draw = H.subsample(list(fwd["train"]) + list(fwd["val"]), n_tr + n_va, seed=500 + seed)
    return {"train": draw[n_va:], "val": draw[:n_va], "test": fwd["test"], "train_dataset": "aqsoldb",
            "test_dataset": "esol", "meta": {"split_type": "external", "name": SIZEMATCHED, "seed": seed,
                                             "n_train": n_tr, "n_val": n_va}}


def stage_external(args, P: Paths, meta: pd.DataFrame) -> None:
    """AqSolDB <-> ESOL transfer (LightGBM, RF with model seeds, ESOL-linear refit) + size-matched AqSolDB -> ESOL."""
    for name in EXTERNAL + [SIZEMATCHED]:
        for model, feat in EXT_MODELS:
            seeds = [0] if (model == "esol_linear" and name != SIZEMATCHED) else args.external_seeds
            for seed in seeds:
                out = P.pred / "external" / name / f"seed{seed}" / f"{model}__{feat}.csv"
                if out.exists():
                    continue
                sp = sizematched_external_split(seed) if name == SIZEMATCHED else load_split("external", name=name)
                row, preds, _ = run_split(sp, feat, model, seed=seed, n_jobs=args.n_jobs)
                out.parent.mkdir(parents=True, exist_ok=True)
                preds.to_csv(out, index=False)
                if name == SIZEMATCHED:
                    row["train_ids"] = sp["train"]
                save_json(row, out.with_suffix(".json"))
                log(f"external {name} {model} seed{seed}: MAE={row['MAE']:.3f} R2={row['R2']:.3f} "
                    f"({row['fit_seconds']:.0f}s)")


# --------------------------------------------------------------------------- tables / plotting helpers
SPLIT_COLORS = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52"}
GROUP_COLORS = {"G1": "#9C9C9C", "G2": "#E15759", "G3": "#59A14F", "G4": "#B07AA1", "G5": "#4E9F9B"}
LOGS_UNIT = "log mol/L"


def write_table(df: pd.DataFrame, P: Paths, name: str) -> None:
    """CSV + Markdown (3 decimals) under results/tables/heterogeneity_<name>.*"""
    df.to_csv(P.tab / f"heterogeneity_{name}.csv", index=False)
    (P.tab / f"heterogeneity_{name}.md").write_text(df.round(3).to_markdown(index=False), encoding="utf-8")


def plt_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def savefig(fig, P: Paths, name: str, rect: tuple | None = None) -> None:
    """tight_layout (optionally within ``rect``), save as PNG (dpi 200) under results/figures/heterogeneity."""
    fig.tight_layout(rect=rect) if rect else fig.tight_layout()
    fig.savefig(P.fig / f"{name}.png", dpi=200)
    import matplotlib.pyplot as plt

    plt.close(fig)


def esol4_clogp(dataset: str = "aqsoldb") -> pd.Series:
    X, ids, names = load_or_compute(dataset, "esol4")
    return pd.Series(X[:, names.index("clogp")], index=ids.astype(str), name="clogp")


def fingerprints(dataset: str) -> dict:
    from solpred.splits import morgan_bits

    df = load_dataset(dataset)
    return dict(zip(df["compound_id"].astype(str), morgan_bits(df["smiles"].tolist())))


def maxsim(query: list[str], ref: list[str], fq: dict, fr: dict) -> np.ndarray:
    from solpred.splits import max_similarity

    return max_similarity([fq[i] for i in query], [fr[i] for i in ref])


# --------------------------------------------------------------------------- 1. noise
def stage_noise(args, P: Paths, meta: pd.DataFrame) -> None:
    """Intrinsic measurement variability, our de-dup conflicts and ESOL/AqSolDB agreement."""
    out = P.out / "noise"
    out.mkdir(exist_ok=True)
    clogp = esol4_clogp()
    overlap = pd.read_csv(path("processed") / "esol_aqsoldb_overlap.csv", keep_default_na=False, na_values=[""])

    sd_tab = H.sd_group_table(meta)
    est = H.noise_estimates(meta, n_boot=args.n_boot)
    corr, binned = H.sd_correlates(meta, clogp)
    sel = H.selection_bias_table(meta, clogp)
    conv = H.sd_convention_check(overlap)
    dedup = H.dedup_summary(meta)
    ba = H.bland_altman(overlap["logS_aqsoldb"], overlap["logS_esol"], n_boot=args.n_boot)
    by_src = H.overlap_breakdown(overlap, "source")
    by_grp = H.overlap_breakdown(overlap, "reliability_group")
    for name, df in [("sd_by_group", sd_tab), ("noise_estimates", est), ("sd_correlates", corr),
                     ("sd_binned", binned), ("selection_bias", sel), ("overlap_by_source", by_src),
                     ("overlap_by_group", by_grp)]:
        df.to_csv(out / f"{name}.csv", index=False)
    write_table(sd_tab, P, "sd_by_group")
    sb = sel.set_index("variable")
    caveat = (f"multi-report compounds are not representative of the single-report G1 majority "
              f"(mean MW {sb.loc['mol_wt', 'mean_multi_G2_G5']:.0f} vs {sb.loc['mol_wt', 'mean_single_G1']:.0f} g/mol; "
              f"{100 * sb.loc['frac in_esol', 'mean_multi_G2_G5']:.0f}% vs "
              f"{100 * sb.loc['frac in_esol', 'mean_single_G1']:.0f}% in ESOL), so the floors may not transfer to G1; "
              "the ddof-corrected pooled SD is dominated by G2/G4 outliers")
    est["caveat"] = caveat
    write_table(est, P, "noise_floors")
    write_table(corr, P, "sd_correlates")
    write_table(sel, P, "selection_bias")
    write_table(by_src, P, "esol_overlap_by_source")
    write_table(by_grp, P, "esol_overlap_by_group")
    dedup_groups = meta[meta["n_merged"] > 1][["compound_id", "name", "n_merged", "merged_logS_sd",
                                               "merged_logS_range", "dedup_conflict", "reliability_group"]]
    dedup_groups.sort_values("merged_logS_sd", ascending=False).to_csv(out / "dedup_groups.csv", index=False)

    m = H.multi_report(meta)
    summary = {
        "n_multi_report": int(len(m)), "frac_single_report_G1": float((meta["n_occurrences"] == 1).mean()),
        "sd_convention": conv,
        "noise_estimates": est.set_index("estimator")[["sigma", "ci_low", "ci_high", "RMSE_floor",
                                                       "MAE_floor_gaussian", "n_compounds", "role"]].to_dict("index"),
        "noise_floor_caveat": caveat,
        "recommended_floors": recommended_floors(est),
        "sd_median_multi": float(m["aqsol_sd"].median()), "frac_multi_sd_lt_0.01": float((m["aqsol_sd"] < 0.01).mean()),
        "frac_multi_sd_gt_0.5": float((m["aqsol_sd"] > 0.5).mean()),
        "sd_by_group": sd_tab.set_index("group")[["n", "sd_median", "rms_sd", "sd_q95"]].to_dict("index"),
        "sd_correlates_spearman": corr.set_index("variable")[["spearman_rho", "p_value"]].to_dict("index"),
        "selection_bias": sel.set_index("variable")[["mean_single_G1", "mean_multi_G2_G5", "median_single_G1",
                                                     "median_multi_G2_G5"]].to_dict("index"),
        "dedup": dedup, "esol_aqsoldb_bland_altman": ba,
        "overlap_by_source": by_src.set_index("source")[["n", "frac_identical_lt_0.01", "MAE"]].to_dict("index"),
    }
    save_json(summary, P.out / "summary_noise.json")
    plot_noise(P, meta, clogp, est, overlap, ba, by_src)
    log(f"noise: estimators\n{est[['estimator', 'sigma', 'ci_low', 'ci_high', 'MAE_floor_gaussian']].round(3)}")


def recommended_floors(est: pd.DataFrame) -> dict:
    """Headline floors: G3+G5 pooled sigma and the distribution-free pair MAE floor; RMS SD = overall spread."""
    e = est.set_index("estimator")
    cons = e.loc["pooled SD, consistent reports only (G3+G5)"]
    npf = e.loc["distribution-free MAE floor from pairs (E|x1-x2|/sqrt2)"]
    rms = e.loc["RMS SD (ddof=0, naive)"]
    pooled = e.loc["pooled SD (ddof-corrected)"]
    return {
        "headline_sigma_G3G5_pooled": [float(cons["sigma"]), float(cons["ci_low"]), float(cons["ci_high"])],
        "headline_sigma_G3G5_note": "lower bound (G3/G5 truncated at SD <= 0.5); Gaussian MAE floor "
                                    f"{cons['MAE_floor_gaussian']:.3f}",
        "headline_MAE_floor_pairs_distribution_free": [float(npf["MAE_floor_gaussian"]), float(npf["ci_low"]),
                                                       float(npf["ci_high"])],
        "overall_inter_report_spread_rms_sd": [float(rms["sigma"]), float(rms["ci_low"]), float(rms["ci_high"])],
        "not_recommended_pooled_sd_ddof_corrected": float(pooled["sigma"]),
        "not_recommended_reason": "dominated by a few gross G2/G4 discrepancies (unit/transcription errors)",
    }


def plot_noise(P: Paths, meta, clogp, est, overlap, ba, by_src) -> None:
    plt = plt_setup()
    m = H.multi_report(meta)
    # (1) SD distributions by reliability group + ECDF of pair differences
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    bins = np.logspace(np.log10(0.004), np.log10(4.5), 40)
    for g in ["G3", "G5", "G2", "G4"]:
        s = m.loc[m["reliability_group"] == g, "aqsol_sd"]
        ax[0].hist(s, bins=bins, histtype="step", lw=1.6, color=GROUP_COLORS[g],
                   label=f"{g} (n={len(s)}, median {s.median():.2f})")
    ax[0].axvline(0.5, color="k", ls=":", lw=1)
    ax[0].text(0.52, ax[0].get_ylim()[1] * 0.55, "G2/G4\nthreshold", fontsize=7)
    ax[0].set_xscale("log")
    n_off = int((m["aqsol_sd"] < bins[0]).sum())
    ax[0].set_xlabel(f"inter-report SD in AqSolDB, aqsol_sd ({LOGS_UNIT}); {n_off} compounds with SD < {bins[0]} not shown")
    ax[0].set_ylabel("compounds")
    ax[0].set_title("(a) Spread of repeated reports by reliability group")
    ax[0].legend(frameon=False, fontsize=7, loc="upper right")
    pairs = np.sort(2 * m.loc[m["n_occurrences"] == 2, "aqsol_sd"].to_numpy())
    ecdf = np.arange(1, len(pairs) + 1) / len(pairs)
    ax[1].plot(pairs, ecdf, color="k", lw=1.5, label=f"observed |x1 - x2| (n={len(pairs)} pairs)")
    from scipy import stats as st

    e = est.set_index("estimator")
    for key, col, ls in [("robust SD from pairs (median |x1-x2|)", "#4C72B0", "--"),
                         ("pooled SD (ddof-corrected)", "#C44E52", ":")]:
        sig = e.loc[key, "sigma"]
        xs = np.logspace(-2.2, np.log10(pairs.max()), 400)
        ax[1].plot(xs, 2 * st.norm.cdf(xs / (np.sqrt(2) * sig)) - 1, color=col, ls=ls, lw=1.3,
                   label=f"Gaussian, sigma={sig:.2f} ({key.split(' (')[0]})")
    ax[1].set_xscale("log")
    ax[1].set_xlabel(f"absolute difference between two reports ({LOGS_UNIT})")
    ax[1].set_ylabel("cumulative fraction")
    ax[1].set_title("(b) Two-report compounds: heavy-tailed disagreement")
    ax[1].legend(frameon=False, loc="lower right")
    savefig(fig, P, "sd_by_reliability_group")

    # (2) SD vs logS / MW / cLogP
    mm = m.assign(clogp=clogp.reindex(m["compound_id"]).to_numpy())
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for a, (var, lab) in zip(ax, [("logS", f"logS ({LOGS_UNIT})"), ("mol_wt", "molecular weight (g/mol)"),
                                  ("clogp", "Crippen cLogP (unitless)")]):
        x = mm[var].to_numpy()
        yv = np.clip(mm["aqsol_sd"].to_numpy(), 0.004, None)
        a.scatter(x, yv, s=4, alpha=0.25, color="#4C72B0", lw=0)
        q = pd.qcut(mm[var], 8, duplicates="drop")
        g = mm.groupby(q, observed=True)
        a.plot(g[var].median(), g["aqsol_sd"].median(), "o-", color="k", ms=3, lw=1.2, label="bin median")
        a.plot(g[var].median(), g["aqsol_sd"].apply(lambda s: np.sqrt((s ** 2).mean())), "s--", color="#C44E52",
               ms=3, lw=1.2, label="bin RMS")
        rho = st.spearmanr(mm[var], mm["aqsol_sd"])
        a.set_title(f"Spearman rho = {rho[0]:.2f} (p = {rho[1]:.1e})")
        a.set_yscale("log")
        a.set_xlim(*np.nanpercentile(x, [0.5, 99.5]))
        a.set_xlabel(f"{lab}; axis = 0.5-99.5th pct.")
    ax[0].set_ylabel(f"aqsol_sd ({LOGS_UNIT}), multi-report compounds")
    ax[0].legend(frameon=False)
    savefig(fig, P, "sd_vs_properties")

    # (3) Bland-Altman ESOL vs AqSolDB + source composition of the overlap
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 3.9), gridspec_kw={"width_ratios": [1.6, 1]})
    mean = (overlap["logS_aqsoldb"] + overlap["logS_esol"]) / 2
    diff = overlap["diff_aqsoldb_minus_esol"]
    for g in H.GROUP_ORDER:
        sel = overlap["reliability_group"] == g
        ax[0].scatter(mean[sel], diff[sel], s=7, alpha=0.6, color=GROUP_COLORS[g], lw=0, label=f"{g} (n={sel.sum()})")
    for v, ls, lab in [(ba["bias"], "-", f"bias {ba['bias']:+.3f}"), (ba["loa_low"], "--", f"LoA {ba['loa_low']:.2f}"),
                       (ba["loa_high"], "--", f"LoA {ba['loa_high']:+.2f}")]:
        ax[0].axhline(v, color="k", ls=ls, lw=0.9)
        ax[0].text(mean.max(), v, lab, ha="right", va="bottom", fontsize=7)
    ax[0].set_xlabel(f"mean of AqSolDB and ESOL logS ({LOGS_UNIT})")
    ax[0].set_ylabel(f"AqSolDB - ESOL logS ({LOGS_UNIT})")
    ax[0].set_title(f"(a) Bland-Altman, n={ba['n']}; {100 * ba['frac_identical_lt_0.01']:.1f}% identical (|d|<0.01)")
    ax[0].legend(frameon=False, ncol=2, markerscale=2, loc="lower left")
    b = by_src[by_src["source"] != "all"].sort_values("n", ascending=False)
    ident = b["n"] * b["frac_identical_lt_0.01"]
    ax[1].bar(b["source"], ident, color="#9C9C9C", label="identical value (|d| < 0.01)")
    ax[1].bar(b["source"], b["n"] - ident, bottom=ident, color="#C44E52", label="different value")
    ax[1].set_xlabel("AqSolDB source letter of the kept value")
    ax[1].set_ylabel("overlapping compounds")
    ax[1].set_title("(b) Where the 1,097 shared compounds come from")
    ax[1].legend(frameon=False)
    savefig(fig, P, "esol_aqsoldb_bland_altman")


# --------------------------------------------------------------------------- 2. error vs reliability
def test_maxsim_table(P: Paths, meta: pd.DataFrame) -> pd.DataFrame:
    """Max Tanimoto (ECFP4) of every test compound to its training set, for all reference splits (cached)."""
    f = P.out / "reliability" / "test_maxsim.csv"
    if f.exists():
        return pd.read_csv(f, dtype={"compound_id": str})
    fp = fingerprints("aqsoldb")
    rows = []
    for stype in ("random", "scaffold", "cluster"):
        for seed in range(5):
            sp = load_split(stype, seed)
            rows.append(pd.DataFrame({"split_type": stype, "seed": seed, "compound_id": sp["test"],
                                      "maxsim": maxsim(sp["test"], sp["train"], fp, fp)}))
    out = pd.concat(rows, ignore_index=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(f, index=False)
    return out


def stage_reliability(args, P: Paths, meta: pd.DataFrame) -> None:
    """Reference-model test error by reliability group, n_occurrences, source letter and SD bin."""
    out = P.out / "reliability"
    out.mkdir(exist_ok=True)
    sims = test_maxsim_table(P, meta)
    long, tests, contrasts, adjusted, overall = [], [], [], [], []
    groupings = {"reliability_group": H.GROUP_ORDER, "rel3": H.REL3_ORDER, "occ_bin": H.OCC_BIN_LABELS,
                 "source": sorted(meta["source"].unique()), "sd_bin": H.SD_BIN_LABELS}
    for model in (LGBM_FILE, RF_FILE):
        for stype in ("random", "scaffold", "cluster"):
            preds = H.load_reference_predictions(P.baseline, stype, model, range(5))
            n_seeds = preds["seed"].nunique() if len(preds) else 0
            if n_seeds < 5:
                log(f"reliability: WARNING {model} {stype} has {n_seeds}/5 seeds")
            if not len(preds):
                continue
            preds = preds.merge(sims[sims["split_type"] == stype], on=["seed", "compound_id"], how="left")
            err = H.annotate(H.compound_errors(preds), meta)
            err.to_csv(out / f"compound_errors__{stype}__{model}.csv", index=False)
            mae, lo, hi = H.mae_ci(err["ae"], n_boot=args.n_boot)
            overall.append({"model": model, "split_type": stype, "n_seeds": n_seeds, "n_compounds": len(err),
                            "MAE_compound_level": mae, "ci_low": lo, "ci_high": hi})
            for col, order in groupings.items():
                t = H.group_mae_table(err, col, order, n_boot=args.n_boot).rename(columns={col: "level"})
                long.append(t.assign(model=model, split_type=stype, grouping=col))
                tests.append({"model": model, "split_type": stype, "grouping": col, **H.kruskal_test(err, col)})
            hi_rel = err.loc[err["rel3"] == H.REL3_ORDER[1], "ae"]
            for lab, other in [("inconsistent (G2+G4) - consistent (G3+G5)", H.REL3_ORDER[2]),
                               ("single (G1) - consistent (G3+G5)", H.REL3_ORDER[0])]:
                d = H.mae_diff_unpaired(err.loc[err["rel3"] == other, "ae"], hi_rel, n_boot=args.n_boot)
                contrasts.append({"model": model, "split_type": stype, "contrast": lab, **d})
            adj = H.adjusted_reliability_effects(err)
            adjusted.append(adj.assign(model=model, split_type=stype))
    long = pd.concat(long, ignore_index=True)
    cols = ["model", "split_type", "grouping", "level", "n", "MAE", "MAE_ci_low", "MAE_ci_high", "median_AE", "bias",
            "mean_label_sd"]
    long = long[cols]
    tests, contrasts = pd.DataFrame(tests), pd.DataFrame(contrasts)
    adjusted, overall = pd.concat(adjusted, ignore_index=True), pd.DataFrame(overall)
    for name, df in [("mae_by_group", long), ("kruskal", tests), ("contrasts", contrasts),
                     ("adjusted_ols", adjusted), ("overall", overall)]:
        df.to_csv(out / f"{name}.csv", index=False)
    write_table(long[long["grouping"].isin(["reliability_group", "rel3"])], P, "mae_by_reliability")
    write_table(long[long["grouping"].isin(["occ_bin", "sd_bin", "source"])], P, "mae_by_occurrence_sd_source")
    write_table(pd.concat([contrasts, tests.rename(columns={"grouping": "contrast"})], ignore_index=True),
                P, "reliability_tests")
    write_table(adjusted[adjusted["term"].str.startswith("rel") | (adjusted["term"] == "maxsim")], P,
                "reliability_adjusted_ols")
    lg = long[(long["model"] == LGBM_FILE) & (long["grouping"] == "rel3")]
    summary = {
        "unit": "compound-level MAE (mean |error| over the seeds in which the compound was a test compound)",
        "overall": overall.to_dict("records"),
        "lgbm_mae_by_rel3": {f"{r.split_type}|{r.level}": {"n": r.n, "MAE": r.MAE, "ci": [r.MAE_ci_low, r.MAE_ci_high]}
                             for r in lg.itertuples()},
        "contrasts": contrasts.to_dict("records"), "kruskal": tests.to_dict("records"),
        "adjusted_ols_rel_terms": adjusted[adjusted["term"].str.startswith("rel")].to_dict("records"),
    }
    save_json(summary, P.out / "summary_reliability.json")
    plot_reliability(P, long)
    log("reliability: done\n" + contrasts.round(3).to_string())


def plot_reliability(P: Paths, long: pd.DataFrame) -> None:
    plt = plt_setup()
    floors = {}
    f = P.out / "noise" / "noise_estimates.csv"
    if f.exists():
        e = pd.read_csv(f).set_index("estimator")
        floors = {"Gaussian MAE floor, RMS SD": e.loc["RMS SD (ddof=0, naive)", "MAE_floor_gaussian"],
                  "Gaussian MAE floor, robust SD": e.loc["robust SD from pairs (median |x1-x2|)", "MAE_floor_gaussian"]}
    splits = ["random", "scaffold", "cluster"]
    for grouping, order, fname, xlabel in [
            ("reliability_group", H.GROUP_ORDER, "mae_by_reliability_group", "AqSolDB reliability group"),
            ("sd_bin", H.SD_BIN_LABELS, "mae_by_sd_bin", f"inter-report SD bin ({LOGS_UNIT})"),
            ("source", sorted(long.loc[long["grouping"] == "source", "level"].unique()), "mae_by_source",
             "AqSolDB source letter")]:
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
        for ax, model, title in zip(axes, (LGBM_FILE, RF_FILE), ("LightGBM (rdkit2d + Morgan counts)", "RF (rdkit2d)")):
            w = 0.8 / len(splits)
            for k, st in enumerate(splits):
                d = long[(long["model"] == model) & (long["split_type"] == st) & (long["grouping"] == grouping)]
                d = d.set_index("level").reindex(order)
                x = np.arange(len(order)) + (k - 1) * w
                ax.bar(x, d["MAE"], w, color=SPLIT_COLORS[st], label=st,
                       yerr=[d["MAE"] - d["MAE_ci_low"], d["MAE_ci_high"] - d["MAE"]], capsize=2,
                       error_kw={"lw": 0.8})
                for xi, n in zip(x, d["n"]):
                    if pd.notna(n):
                        ax.text(xi, 0.02, f"n={int(n)}", ha="center", va="bottom", fontsize=5.5, rotation=90,
                                color="white")
            for (lab, v), ls in zip(floors.items(), ("--", ":")):
                ax.axhline(v, color="k", ls=ls, lw=0.9, label=f"{lab} ({v:.2f})")
            ax.set_xticks(np.arange(len(order)))
            ax.set_xticklabels(order, rotation=20 if grouping == "sd_bin" else 0)
            ax.set_xlabel(xlabel)
            ax.set_title(title)
        axes[0].set_ylabel(f"test MAE ({LOGS_UNIT}), 95% bootstrap CI\n(compound-level, seeds 0-4 pooled)")
        top = max(a.get_ylim()[1] for a in axes)
        for a in axes:
            a.set_ylim(0, top * 1.22)
        axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="upper left")
        savefig(fig, P, fname)


# --------------------------------------------------------------------------- 3. ablation report
def load_ablation(P: Paths) -> pd.DataFrame:
    rows = []
    for f in sorted((P.pred / "ablation").glob("*/seed*/*.csv")):
        d = pd.read_csv(f, dtype={"compound_id": str})
        d = d[d["subset"] == "test"].assign(split_type=f.parent.parent.name, seed=int(f.parent.name[4:]),
                                            variant=f.stem)
        rows.append(d)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


ABLATION_SUBSETS = {"full test": None, "high-reliability test (G3+G5)": H.HIGH_RELIABILITY,
                    "single-report test (G1)": ("G1",), "inconsistent test (G2+G4)": H.LOW_RELIABILITY}


ABLATION_ORDER = ["b_drop_G2G4", "e_random_drop", "c_drop_dedup_conflict", "d_ivw", "f_ivw_permuted"]
ABLATION_CONTRASTS = [("b_drop_G2G4", "e_random_drop"), ("d_ivw", "f_ivw_permuted")]


def ablation_family(variant: str) -> str:
    """Variant name without the control-replicate suffix (``e_random_drop_rep1`` -> ``e_random_drop``)."""
    return variant.split("_rep")[0]


def ablation_errors(pr: pd.DataFrame) -> pd.DataFrame:
    """Per (split, seed, variant family, compound) |error|, averaged over control replicates of the family."""
    pr = pr.assign(ae=(pr["y_pred"] - pr["y_true"]).abs(), family=pr["variant"].map(ablation_family))
    return pr.groupby(["split_type", "seed", "family", "compound_id"], as_index=False).agg(
        ae=("ae", "mean"), n_reps=("variant", "nunique"), reliability_group=("reliability_group", "first"))


def ablation_blocks(err: pd.DataFrame, stype: str, fam_a: str, fam_b: str, groups) -> dict[int, tuple]:
    """Paired per-seed (ae_a, ae_b) arrays on the test compounds of ``groups`` (None = all)."""
    blocks = {}
    for seed in range(5):
        a = err[(err["split_type"] == stype) & (err["seed"] == seed) & (err["family"] == fam_a)]
        b = err[(err["split_type"] == stype) & (err["seed"] == seed) & (err["family"] == fam_b)]
        if a.empty or b.empty:
            continue
        a = a.set_index("compound_id")
        b = b.set_index("compound_id").reindex(a.index)
        if groups is not None:
            keep = a["reliability_group"].isin(groups).to_numpy()
            a, b = a[keep], b[keep]
        blocks[seed] = (a["ae"].to_numpy(), b["ae"].to_numpy())
    return blocks


def report_ablation(args, P: Paths, meta: pd.DataFrame) -> dict:
    """Paired delta-MAE of every ablation variant vs (a), per seed and pooled over seeds (seed-stratified bootstrap).

    Control replicates (``e_random_drop``, ``e_random_drop_rep1`` ...) are averaged at the level of
    per-compound absolute errors, i.e. the reported effect is the replicate-mean MAE difference.
    """
    pr = load_ablation(P)
    if pr.empty:
        return {}
    grp = meta.set_index("compound_id")["reliability_group"]
    pr["reliability_group"] = grp.reindex(pr["compound_id"]).to_numpy()
    err = ablation_errors(pr)
    reps = err.groupby(["split_type", "family"])["n_reps"].min()
    families = [v for v in sorted(err["family"].unique()) if v != "a_all"]
    per_seed, pooled, be = [], [], []
    for stype in sorted(err["split_type"].unique()):
        for fam in families:
            for sname, groups in ABLATION_SUBSETS.items():
                blocks = ablation_blocks(err, stype, "a_all", fam, groups)
                if not blocks:
                    continue
                for seed, (ea, eb) in blocks.items():
                    d = H.pooled_paired_delta_ae([(ea, eb)], n_boot=args.n_boot, seed=seed)
                    per_seed.append({"split_type": stype, "variant": fam, "subset": sname, "seed": seed, "n": len(ea),
                                     "MAE_a_all": float(ea.mean()), "MAE_variant": float(eb.mean()),
                                     "delta_MAE": d["delta_mae"], "ci_low": d["ci_low"], "ci_high": d["ci_high"]})
                r = H.pooled_paired_delta_ae(list(blocks.values()), n_boot=args.n_boot)
                pooled.append({"split_type": stype, "variant": fam, "description": H.ABLATION_VARIANTS.get(fam, fam),
                               "n_control_reps": int(reps.get((stype, fam), 1)), "subset": sname,
                               "n_seeds": r["n_blocks"], "n_test_per_seed": r["n_compounds"] / r["n_blocks"],
                               "MAE_a_all": float(np.mean([ea.mean() for ea, _ in blocks.values()])),
                               "MAE_variant": float(np.mean([eb.mean() for _, eb in blocks.values()])),
                               "delta_MAE": r["delta_mae"], "ci_low": r["ci_low"], "ci_high": r["ci_high"],
                               "p_not_better": r["p_not_better"], "p_not_worse": r["p_not_worse"]})
        # treatment - matched control: (b) - (e) targeted vs random removal, (d) - (f) IVW vs permuted weights
        for treat, ctrl in ABLATION_CONTRASTS:
            for sname, groups in ABLATION_SUBSETS.items():
                blocks = ablation_blocks(err, stype, ctrl, treat, groups)
                if blocks:
                    r = H.pooled_paired_delta_ae(list(blocks.values()), n_boot=args.n_boot)
                    be.append({"split_type": stype, "contrast": f"{treat} - {ctrl}",
                               "n_control_reps": int(reps.get((stype, ctrl), 1)), "subset": sname,
                               "n_seeds": r["n_blocks"], "delta_MAE": r["delta_mae"], "ci_low": r["ci_low"],
                               "ci_high": r["ci_high"], "p_not_better": r["p_not_better"]})
    runs = pd.DataFrame([json.loads(f.read_text()) for f in sorted((P.pred / "ablation").glob("*/seed*/*.json"))])
    runs["family"] = runs["variant"].map(ablation_family)
    per_seed, pooled, be = pd.DataFrame(per_seed), pd.DataFrame(pooled), pd.DataFrame(be)
    out = P.out / "ablation"
    out.mkdir(exist_ok=True)
    per_seed.to_csv(out / "delta_per_seed.csv", index=False)
    pooled.to_csv(out / "delta_pooled.csv", index=False)
    runs.to_csv(out / "runs.csv", index=False)
    be.to_csv(out / "treatment_vs_control.csv", index=False)
    write_table(pooled, P, "ablation")
    write_table(per_seed[per_seed["subset"].isin(list(ABLATION_SUBSETS)[:2])], P, "ablation_per_seed")
    if not be.empty:
        write_table(be, P, "ablation_vs_controls")
    plot_ablation(P, pooled, be)
    sizes = runs.groupby(["split_type", "family"]).agg(n_runs=("variant", "size"), n_train=("n_train", "mean"),
                                                        n_dropped=("n_dropped", "mean")).reset_index()
    return {"unit": "delta MAE = MAE(variant) - MAE(a_all) on the same test compounds; mean over seeds 0-4, "
                    "95% CI from a seed-stratified paired bootstrap; negative = variant better",
            "s0_ivw": float(runs["s0"].dropna().iloc[0]) if "s0" in runs and runs["s0"].notna().any() else None,
            "s0_ivw_definition": "dataset-level constant: naive RMS of AqSolDB aqsol_sd (report-spread metadata) "
                                 "over all multi-report compounds, computed once and identical for every split/seed; "
                                 "no test-compound logS labels are used (their SD metadata enters this one scalar)",
            "control_replicates": {f"{k[0]}|{k[1]}": int(v) for k, v in reps.items() if v > 1},
            "pooled": pooled.to_dict("records"), "treatment_minus_control": be.to_dict("records"),
            "training_sizes": {f"{r.split_type}|{r.family}": {"n_runs": int(r.n_runs), "n_train": float(r.n_train),
                                                             "n_dropped": float(r.n_dropped)}
                               for r in sizes.itertuples()}}


def plot_ablation(P: Paths, pooled: pd.DataFrame, contrasts: pd.DataFrame) -> None:
    """Forest plot: delta MAE vs (a) per variant (left two panels) and treatment - control contrasts (right)."""
    plt = plt_setup()
    subsets = ["full test", "high-reliability test (G3+G5)"]
    variants = [v for v in ABLATION_ORDER if v in set(pooled["variant"])]
    reps = pooled.groupby("variant")["n_control_reps"].max()
    labels = [f"({v[0]}) " + H.ABLATION_VARIANTS.get(v, v) + (f", {int(reps[v])} reps" if reps.get(v, 1) > 1 else "")
              for v in variants]
    fig, axes = plt.subplots(1, 3, figsize=(16, 0.55 * len(variants) + 2.8),
                             gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, sname in zip(axes[:2], subsets):
        for k, st in enumerate(["random", "scaffold"]):
            d = pooled[(pooled["split_type"] == st) & (pooled["subset"] == sname)].set_index("variant").reindex(variants)
            yy = np.arange(len(variants)) + (-0.15 if k == 0 else 0.15)
            ax.errorbar(d["delta_MAE"], yy, xerr=[d["delta_MAE"] - d["ci_low"], d["ci_high"] - d["delta_MAE"]],
                        fmt="o", color=SPLIT_COLORS[st], capsize=3, ms=5, label=f"{st} split")
        ax.axvline(0, color="k", lw=0.8)
        ax.set_yticks(np.arange(len(variants)))
        ax.set_yticklabels(labels if ax is axes[0] else [])
        ax.set_ylim(len(variants) - 0.5, -0.5)
        ax.set_xlabel(f"MAE(variant) - MAE(a: all data) ({LOGS_UNIT})\nnegative = better; mean of 5 seeds, 95% CI")
        n = pooled.loc[pooled["subset"] == sname, "n_test_per_seed"]
        ax.set_title(f"vs (a), {sname}\n(~{n.min():.0f}-{n.max():.0f} test compounds per seed)")
    handles, labs = axes[0].get_legend_handles_labels()
    ax = axes[2]
    if not contrasts.empty:
        c = contrasts[contrasts["subset"].isin(subsets)]
        order = [f"{t} - {k}" for t, k in ABLATION_CONTRASTS]
        keys = [(t, sub) for t in order for sub in subsets if ((c["contrast"] == t) & (c["subset"] == sub)).any()]
        creps = c.groupby("contrast")["n_control_reps"].max()
        for k, st in enumerate(["random", "scaffold"]):
            d = c[c["split_type"] == st].set_index(["contrast", "subset"]).reindex(keys)
            yy = np.arange(len(keys)) + (-0.15 if k == 0 else 0.15)
            ax.errorbar(d["delta_MAE"], yy, xerr=[d["delta_MAE"] - d["ci_low"], d["ci_high"] - d["delta_MAE"]],
                        fmt="s", color=SPLIT_COLORS[st], capsize=3, ms=5)
        ax.set_yticks(np.arange(len(keys)))
        ax.set_yticklabels([f"({t[0]}) - ({t.split(' - ')[1][0]})"
                            + (f" [{int(creps[t])} ctrl reps]" if creps.get(t, 1) > 1 else "")
                            + f"\n{sub.replace(' test', '')}" for t, sub in keys], fontsize=7)
        ax.set_ylim(len(keys) - 0.5, -0.5)
        ax.axvline(0, color="k", lw=0.8)
        ax.set_xlabel(f"MAE(treatment) - MAE(matched control)\n({LOGS_UNIT}); 95% CI")
        ax.set_title("treatment vs matched control\n(b)-(e) targeted vs random removal; (d)-(f) IVW vs permuted")
    fig.legend(handles, labs, frameon=False, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(P.fig / "ablation_forest.png", dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


# --------------------------------------------------------------------------- 4. LOSO + matrix report
def random_split_metrics(rand: pd.DataFrame, ids: set[str]) -> dict:
    """MAE/RMSE/R2/bias of the random-split reference model on the compounds in ``ids``.

    Metrics are computed per seed on the held-out-source compounds that fall in that seed's random test set,
    then averaged over seeds (so R2 refers to the same compound population as the LOSO R2).
    """
    per = []
    for _, d in rand[rand["compound_id"].isin(ids)].groupby("seed"):
        per.append({"n": len(d), **H.metrics_with_bias(d["y_true"], d["y_pred"])})
    if not per:
        return {}
    per = pd.DataFrame(per)
    return {"random_n_per_seed": float(per["n"].mean()), "random_MAE": float(per["MAE"].mean()),
            "random_RMSE": float(per["RMSE"].mean()), "random_R2": float(per["R2"].mean()),
            "random_bias": float(per["bias"].mean())}


def control_similarities(meta: pd.DataFrame, src: str, n_train: int, n_folds: int, fp: dict) -> pd.Series:
    """NN Tanimoto of each control test compound to its control training set (splits regenerated, seed 0)."""
    parts = []
    for k in range(n_folds):
        csp = H.loso_control_split(meta, src, n_train=n_train, n_folds=n_folds, fold=k, seed=0)
        parts.append(pd.Series(maxsim(csp["test"], csp["train"], fp, fp), index=csp["test"]))
    return pd.concat(parts)


LOSO_PENALTY_NOTES = {
    "source_shift_penalty (= penalty_vs_control)": "HEADLINE: MAE(LOSO) - MAE(size-matched in-distribution control) "
        "on the same held-out-source compounds; training-set size matched to the LOSO training set",
    "penalty_vs_random (NOT size-matched)": "MAE(LOSO) - MAE(random-split reference) on the subset of held-out "
        "compounds that fell in random-split test sets; the random-split model had a LARGER training set "
        "(and saw the held-out source), so this mixes source shift with training-set size - not size-matched",
}


def report_loso(args, P: Paths, meta: pd.DataFrame) -> dict:
    rand = H.load_reference_predictions(P.baseline, "random", LGBM_FILE, range(5))
    rand_err = H.compound_errors(rand).set_index("compound_id") if len(rand) else pd.DataFrame()
    fp = fingerprints("aqsoldb")
    rows, comp = [], []
    for src in LOSO_SOURCES:
        f = P.pred / "loso" / f"source_{src}.csv"
        if not f.exists():
            continue
        d = pd.read_csv(f, dtype={"compound_id": str})
        te = d[d["subset"] == "test"].set_index("compound_id")
        sp = load_split("loso", source=src)
        sims = pd.Series(maxsim(list(te.index), sp["train"], fp, fp), index=te.index)
        m = H.metrics_with_bias(te["y_true"], te["y_pred"])
        bias_ci = H.bootstrap_stat(np.mean, (te["y_pred"] - te["y_true"]).to_numpy(), n_boot=args.n_boot)
        mae_ci = H.bootstrap_stat(np.mean, (te["y_pred"] - te["y_true"]).abs().to_numpy(), n_boot=args.n_boot)
        row = {"source": src, "n_test": len(te), "n_train": len(sp["train"]), **{k: m[k] for k in
               ("MAE", "RMSE", "R2", "Pearson", "Spearman", "bias")}, "MAE_ci_low": mae_ci[0], "MAE_ci_high": mae_ci[1],
               "bias_ci_low": bias_ci[0], "bias_ci_high": bias_ci[1], "test_maxsim_median": float(sims.median()),
               "test_logS_mean": float(te["y_true"].mean()),
               "train_logS_mean": float(meta.set_index("compound_id").loc[sp["train"], "logS"].mean())}
        # size-matched in-distribution control (all folds)
        cf = sorted((P.pred / "loso_control" / f"source_{src}").glob("fold*.csv"))
        if cf:
            c = pd.concat([pd.read_csv(x, dtype={"compound_id": str}) for x in cf])
            c = c[c["subset"] == "test"].set_index("compound_id")
            common = te.index.intersection(c.index)
            mc = H.metrics_with_bias(c.loc[common, "y_true"], c.loc[common, "y_pred"])
            dd = paired_bootstrap_mae_diff(te.loc[common, "y_true"], te.loc[common, "y_pred"],
                                           c.loc[common, "y_pred"], n_boot=args.n_boot)
            csim = control_similarities(meta, src, len(sp["train"]), len(cf), fp).reindex(common)
            assert csim.notna().all(), f"regenerated control split for {src} does not match saved predictions"
            adj = H.similarity_adjusted_penalty(
                (te.loc[common, "y_pred"] - te.loc[common, "y_true"]).abs().to_numpy(),
                (c.loc[common, "y_pred"] - c.loc[common, "y_true"]).abs().to_numpy(),
                sims.loc[common].to_numpy(), csim.to_numpy())
            row.update({"control_n": len(common), "control_folds": len(cf), "control_MAE": mc["MAE"],
                        "control_RMSE": mc["RMSE"], "control_R2": mc["R2"], "control_bias": mc["bias"],
                        "control_test_maxsim_median": float(csim.median()),
                        "penalty_vs_control": dd["mae_diff"], "penalty_vs_control_ci_low": dd["ci_low"],
                        "penalty_vs_control_ci_high": dd["ci_high"], **adj})
            comp.append(pd.DataFrame({"source": src, "compound_id": common, "y_true": te.loc[common, "y_true"].values,
                                      "pred_loso": te.loc[common, "y_pred"].values,
                                      "pred_control": c.loc[common, "y_pred"].values,
                                      "maxsim_loso": sims.loc[common].values}))
        # same compounds in random-split test sets (baseline LightGBM, averaged over seeds)
        if len(rand_err):
            common = te.index.intersection(rand_err.index)
            ae_l = (te.loc[common, "y_pred"] - te.loc[common, "y_true"]).abs().to_numpy()
            ae_r = rand_err.loc[common, "ae"].to_numpy()
            rng = np.random.default_rng(0)
            boots = [np.mean((ae_l - ae_r)[rng.integers(0, len(common), len(common))]) for _ in range(args.n_boot)]
            row.update({"random_matched_n": len(common), "random_matched_MAE_random": float(ae_r.mean()),
                        "random_matched_MAE_loso": float(ae_l.mean()), "penalty_vs_random": float((ae_l - ae_r).mean()),
                        "penalty_vs_random_ci_low": float(np.percentile(boots, 2.5)),
                        "penalty_vs_random_ci_high": float(np.percentile(boots, 97.5)),
                        "random_matched_bias_random": float(rand_err.loc[common, "se"].mean())})
            row.update(random_split_metrics(rand, set(te.index)))
        rows.append(row)
    if not rows:
        return {}
    loso = pd.DataFrame(rows)
    out = P.out / "loso"
    out.mkdir(exist_ok=True)
    loso.to_csv(out / "loso_metrics.csv", index=False)
    if comp:
        pd.concat(comp).to_csv(out / "loso_vs_control_predictions.csv", index=False)
    tab = loso.rename(columns={**{k: k.replace("penalty_vs_control", "source_shift_penalty")
                                  for k in loso.columns if k.startswith("penalty_vs_control")},
                               **{k: k.replace("penalty_vs_random", "penalty_vs_random_NOT_size_matched")
                                  for k in loso.columns if k.startswith("penalty_vs_random")}})
    write_table(tab, P, "loso")
    plot_loso(P, loso)
    return {"loso": loso.to_dict("records"), "penalty_definitions": LOSO_PENALTY_NOTES}


def plot_loso(P: Paths, loso: pd.DataFrame) -> None:
    plt = plt_setup()
    has_adj = "adj_penalty" in loso
    fig, ax = plt.subplots(1, 3 if has_adj else 2, figsize=(15.5 if has_adj else 10.5, 3.9))
    x = np.arange(len(loso))
    w = 0.38
    has_c = "control_MAE" in loso
    if has_c:
        ax[0].bar(x - w / 2, loso["control_MAE"], w, color=SPLIT_COLORS["random"],
                  label="in-distribution control (same compounds, size-matched train)")
    ax[0].bar(x + w / 2 if has_c else x, loso["MAE"], w, color=SPLIT_COLORS["external"],
              yerr=[loso["MAE"] - loso["MAE_ci_low"], loso["MAE_ci_high"] - loso["MAE"]], capsize=2,
              label="leave-one-source-out (source unseen)")
    if "random_matched_MAE_random" in loso:
        ax[0].scatter(x - w / 2 if has_c else x, loso["random_matched_MAE_random"], marker="D", color="k", s=18,
                      zorder=3, label="random-split test (matched subset, 5-seed mean)")
    ax[0].set_xticks(x)
    ax[0].set_xticklabels([f"{s}\n(n={n})" for s, n in zip(loso["source"], loso["n_test"])])
    ax[0].set_xlabel("held-out AqSolDB source")
    ax[0].set_ylabel(f"MAE ({LOGS_UNIT})")
    ax[0].set_title("(a) Error on the held-out source")
    ax[0].legend(frameon=False, fontsize=7, loc="upper left")
    ax[0].set_ylim(0, max(loso["MAE_ci_high"].max(), 0.1) * 1.35)
    if has_c:
        ax[1].bar(x - w / 2, loso["control_bias"], w, color=SPLIT_COLORS["random"], label="in-distribution control")
    ax[1].bar(x + w / 2 if has_c else x, loso["bias"], w, color=SPLIT_COLORS["external"],
              yerr=[loso["bias"] - loso["bias_ci_low"], loso["bias_ci_high"] - loso["bias"]], capsize=2,
              label="leave-one-source-out")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(loso["source"])
    ax[1].set_xlabel("held-out AqSolDB source")
    ax[1].set_ylabel(f"mean signed error, predicted - measured ({LOGS_UNIT})")
    ax[1].set_title("(b) Systematic bias (positive = over-predicted solubility)")
    ax[1].legend(frameon=False, fontsize=7)
    if has_adj:
        for off, col, key, lab in [(-w / 2, "#9C9C9C", "penalty_vs_control", "raw: MAE(LOSO) - MAE(control)"),
                                   (w / 2, SPLIT_COLORS["external"], "adj_penalty",
                                    "at equal NN similarity (OLS, compound-clustered CI)")]:
            ax[2].bar(x + off, loso[key], w, color=col, label=lab,
                      yerr=[loso[key] - loso[f"{key}_ci_low"], loso[f"{key}_ci_high"] - loso[key]], capsize=2)
        ax[2].axhline(0, color="k", lw=0.8)
        ax[2].set_xticks(x)
        ax[2].set_xticklabels([f"{s}\nsim {a:.2f} vs {b:.2f}" for s, a, b in
                               zip(loso["source"], loso["test_maxsim_median"], loso["control_test_maxsim_median"])])
        ax[2].set_xlabel("held-out source (median NN similarity: LOSO vs control)")
        ax[2].set_ylabel(f"source-shift penalty in MAE ({LOGS_UNIT})")
        ax[2].set_title("(c) Penalty of an unseen source, 95% CI")
        ax[2].legend(frameon=False, fontsize=7)
    savefig(fig, P, "loso_mae_bias")


def report_transfer(args, P: Paths, meta: pd.DataFrame) -> dict:
    files = sorted((P.pred / "transfer").glob("rep*/train_*.csv"))
    if not files:
        return {}
    design = json.loads((P.out / "transfer_design.json").read_text())
    test_src = {c: s for s, d in design.items() for c in d["test"]}
    fp = fingerprints("aqsoldb")
    rows, cmp_rows = [], []
    for f in files:
        info = json.loads(f.with_suffix(".json").read_text())
        d = pd.read_csv(f, dtype={"compound_id": str})
        d = d[d["subset"] == "test"].copy()
        d["test_source"] = d["compound_id"].map(test_src)
        sims = pd.Series(maxsim(d["compound_id"].tolist(), info["train_ids"], fp, fp), index=d.index)
        cmp_rows.append(d.assign(ae=(d["y_pred"] - d["y_true"]).abs(), maxsim=sims, rep=info["rep"],
                                 train_source=info["train_source"]))
        for t, g in d.groupby("test_source"):
            m = H.metrics_with_bias(g["y_true"], g["y_pred"])
            rows.append({"rep": info["rep"], "train_source": info["train_source"], "test_source": t, "n_test": len(g),
                         "n_train": info["n_train"], **{k: m[k] for k in ("MAE", "RMSE", "R2", "bias")},
                         "test_maxsim_median": float(sims.loc[g.index].median())})
    runs = pd.DataFrame(rows)
    agg = runs.groupby(["train_source", "test_source"]).agg(
        n_reps=("rep", "nunique"), n_test=("n_test", "first"), n_train=("n_train", "first"), MAE=("MAE", "mean"),
        MAE_sd=("MAE", "std"), RMSE=("RMSE", "mean"), R2=("R2", "mean"), bias=("bias", "mean"),
        test_maxsim_median=("test_maxsim_median", "mean")).reset_index()
    out = P.out / "transfer"
    out.mkdir(exist_ok=True)
    runs.to_csv(out / "transfer_runs.csv", index=False)
    agg.to_csv(out / "transfer_matrix_long.csv", index=False)
    write_table(agg, P, "transfer_matrix")
    single = agg[agg["train_source"] != "mixed"]
    diag = single[single["train_source"] == single["test_source"]]
    off = single[single["train_source"] != single["test_source"]]
    # per-target: in-source vs mean cross-source MAE
    per_t = []
    for t in H_sorted(single["test_source"]):
        dt = single[single["test_source"] == t]
        per_t.append({"test_source": t, "in_source_MAE": float(dt.loc[dt["train_source"] == t, "MAE"].iloc[0]),
                      "cross_source_MAE_mean": float(dt.loc[dt["train_source"] != t, "MAE"].mean()),
                      "mixed_MAE": float(agg.loc[(agg["train_source"] == "mixed") & (agg["test_source"] == t), "MAE"].iloc[0])
                      if ((agg["train_source"] == "mixed") & (agg["test_source"] == t)).any() else np.nan})
    per_t = pd.DataFrame(per_t)
    # cross-source penalty at equal NN similarity (single-source training sets only)
    cm = pd.concat(cmp_rows, ignore_index=True)
    cm = cm[cm["train_source"] != "mixed"]
    adj = []
    for t in per_t["test_source"]:
        g = cm[cm["test_source"] == t]
        r = H.similarity_adjusted_effect(g["ae"], (g["train_source"] != t).astype(float), g["maxsim"],
                                         g["compound_id"])
        adj.append({"cross_source_penalty_raw": float(per_t.loc[per_t["test_source"] == t, "cross_source_MAE_mean"].iloc[0]
                                                      - per_t.loc[per_t["test_source"] == t, "in_source_MAE"].iloc[0]),
                    "cross_source_penalty_adj": r["coef"], "adj_ci_low": r["ci_low"], "adj_ci_high": r["ci_high"],
                    "adj_p": r["p"],
                    "in_source_maxsim_median": float(g.loc[g["train_source"] == t, "maxsim"].median()),
                    "cross_source_maxsim_median": float(g.loc[g["train_source"] != t, "maxsim"].median())})
    per_t = pd.concat([per_t, pd.DataFrame(adj)], axis=1)
    per_t.to_csv(out / "in_vs_cross_source.csv", index=False)
    write_table(per_t, P, "transfer_in_vs_cross_source")
    plot_transfer(P, agg)
    return {"matrix": agg.to_dict("records"), "diag_MAE_mean": float(diag["MAE"].mean()),
            "offdiag_MAE_mean": float(off["MAE"].mean()), "per_target": per_t.to_dict("records")}


def H_sorted(s: pd.Series) -> list:
    return sorted(s.unique())


def plot_transfer(P: Paths, agg: pd.DataFrame) -> None:
    plt = plt_setup()
    tr = MATRIX_SOURCES + (["mixed"] if (agg["train_source"] == "mixed").any() else [])
    mat = agg.pivot(index="train_source", columns="test_source", values="MAE").reindex(index=tr, columns=MATRIX_SOURCES)
    bias = agg.pivot(index="train_source", columns="test_source", values="bias").reindex(index=tr, columns=MATRIX_SOURCES)
    sim = agg.pivot(index="train_source", columns="test_source", values="test_maxsim_median").reindex(
        index=tr, columns=MATRIX_SOURCES)
    fig, ax = plt.subplots(figsize=(6.2, 4.6))
    im = ax.imshow(mat.to_numpy(), cmap="viridis_r", aspect="auto")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat.iat[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.2f}\nbias {bias.iat[i, j]:+.2f}\nsim {sim.iat[i, j]:.2f}", ha="center",
                        va="center", fontsize=7, color="white" if v > np.nanmin(mat.to_numpy()) + 0.5 * (
                            np.nanmax(mat.to_numpy()) - np.nanmin(mat.to_numpy())) else "black")
    ax.set_xticks(range(len(MATRIX_SOURCES)))
    ax.set_xticklabels(MATRIX_SOURCES)
    ax.set_yticks(range(len(tr)))
    n_train = int(agg["n_train"].iloc[0])
    ax.set_yticklabels([f"{t} (n={n_train})" for t in tr])
    ax.set_xlabel("test source (held-out 20%)")
    ax.set_ylabel("training source")
    ax.set_title("Source-to-source transfer, LightGBM\n(MAE, mean signed error, median NN similarity)")
    fig.colorbar(im, ax=ax, label=f"MAE ({LOGS_UNIT})")
    savefig(fig, P, "source_transfer_matrix")


# --------------------------------------------------------------------------- 5. external report
EXT_LABELS = {"lgbm": "LightGBM (rdkit2d+Morgan)", "rf": "RF (rdkit2d)", "esol_linear": "ESOL-linear refit (esol4)"}
EXT_TITLES = {"aqsoldb_to_esol": "train AqSolDB -> test ESOL", "esol_to_aqsoldb": "train ESOL -> test AqSolDB",
              SIZEMATCHED: "train AqSolDB (size-matched, n=1004) -> test ESOL"}


def load_external_runs(P: Paths, name: str, model: str, feat: str, sp: dict, fq: dict, fr: dict
                       ) -> tuple[list[np.ndarray], list[np.ndarray], list[list[str]]]:
    """Per-seed test predictions (ordered as ``sp['test']``), NN similarity to that seed's training set, train ids."""
    preds, sims, trains, cache = [], [], [], {}
    for f in sorted((P.pred / "external" / name).glob(f"seed*/{model}__{feat}.csv")):
        d = pd.read_csv(f, dtype={"compound_id": str})
        preds.append(d[d["subset"] == "test"].set_index("compound_id").loc[sp["test"], "y_pred"].to_numpy())
        tr = json.loads(f.with_suffix(".json").read_text()).get("train_ids", sp["train"])
        key = hash(tuple(tr))
        if key not in cache:
            cache[key] = maxsim(sp["test"], tr, fq, fr)
        sims.append(cache[key])
        trains.append(tr)
    return preds, sims, trains


def domain_diagnostics(name: str, train_ids: list[str], test_ids: list[str], trdf: pd.DataFrame,
                       tdf: pd.DataFrame, sims: np.ndarray, ae_lgbm: np.ndarray | None) -> tuple[dict, np.ndarray]:
    """Applicability-domain descriptors of an external test set w.r.t. its training set (asymmetry diagnostics).

    Returns the diagnostics and the boolean 'inside descriptor domain' mask (no new element, single component,
    no charged atoms, MW <= training max).
    """
    tr_el = set(";".join(trdf.loc[train_ids, "elements"]).split(";"))
    te = tdf.loc[test_ids]
    new_el = te["elements"].apply(lambda e: bool(set(e.split(";")) - tr_el)).to_numpy()
    mw_max, ytr = trdf.loc[train_ids, "mol_wt"].max(), trdf.loc[train_ids, "logS"]
    multi = te["is_multicomponent"].astype(bool).to_numpy()
    charged = (te["formal_charge_atoms"] > 0).to_numpy()
    out_y = ((te["logS"] < ytr.min()) | (te["logS"] > ytr.max())).to_numpy()
    inside = ~(new_el | multi | charged | (te["mol_wt"] > mw_max).to_numpy())
    dom = {"direction": name, "n_train": len(train_ids), "n_test": len(te),
           "train_logS_mean": ytr.mean(), "train_logS_sd": ytr.std(), "train_logS_min": ytr.min(),
           "train_logS_max": ytr.max(), "test_logS_mean": te["logS"].mean(), "test_logS_sd": te["logS"].std(),
           "test_maxsim_median": float(np.median(sims)), "frac_test_maxsim_lt_0.4": float((sims < 0.4).mean()),
           "frac_test_maxsim_ge_0.6": float((sims >= 0.6).mean()), "frac_test_multicomponent": multi.mean(),
           "frac_test_charged": charged.mean(), "frac_test_new_element": new_el.mean(),
           "frac_test_mw_gt_train_max": float((te["mol_wt"] > mw_max).mean()),
           "frac_test_logS_outside_train_range": out_y.mean(), "frac_test_in_descriptor_domain": inside.mean()}
    if ae_lgbm is not None:
        dom.update({"lgbm_MAE_all": ae_lgbm.mean(), "lgbm_MAE_in_descriptor_domain": ae_lgbm[inside].mean(),
                    "lgbm_MAE_outside_descriptor_domain": ae_lgbm[~inside].mean() if (~inside).any() else np.nan,
                    "lgbm_MAE_in_domain_and_maxsim_ge_0.5": ae_lgbm[inside & (sims >= 0.5)].mean()})
    return dom, inside


def report_external(args, P: Paths, meta: pd.DataFrame) -> dict:
    """External AqSolDB <-> ESOL transfer: metrics, MAE vs similarity, domain diagnostics, asymmetry decomposition.

    Per-model metrics are means over model seeds (per-seed metrics); compound-level analyses (similarity bins,
    domain, reliability) use the per-compound |error| averaged over seeds, never the seed-ensemble prediction.
    """
    es = load_dataset("esol")
    fps = {"aqsoldb": fingerprints("aqsoldb"), "esol": fingerprints("esol")}
    rows, simrows, domain, parity, rel_rows, cmp = [], [], [], {}, [], {}
    for name in EXTERNAL + [SIZEMATCHED]:
        sp = load_split("external", name="aqsoldb_to_esol" if name == SIZEMATCHED else name)
        trd, ted = sp["train_dataset"], sp["test_dataset"]
        tdf = (meta if ted == "aqsoldb" else es).set_index("compound_id")
        trdf = (meta if trd == "aqsoldb" else es).set_index("compound_id")
        y = tdf.loc[sp["test"], "logS"].to_numpy()
        preds, aes, sims_by_model, train0 = {}, {}, {}, None
        for model, feat in EXT_MODELS:
            per_seed, sims, trains = load_external_runs(P, name, model, feat, sp, fps[ted], fps[trd])
            if not per_seed:
                continue
            label = EXT_LABELS[model]
            ms = pd.DataFrame([H.metrics_with_bias(y, p) for p in per_seed])
            rows.append({"direction": name, "model": label, "n_train": len(trains[0]), "n_test": len(y),
                         "n_seeds": len(per_seed), **{k: ms[k].mean() for k in ("MAE", "RMSE", "R2", "Pearson",
                                                                                  "Spearman", "bias")},
                         **{f"{k}_sd": ms[k].std() if len(per_seed) > 1 else 0.0 for k in ("MAE", "RMSE", "R2")}})
            preds[label] = (np.mean(per_seed, axis=0), ms.mean().to_dict())
            aes[label] = np.mean([np.abs(p - y) for p in per_seed], axis=0)
            sims_by_model[label] = np.mean(sims, axis=0)
            simrows.append(H.similarity_bin_table(sims_by_model[label], aes[label]).assign(direction=name, model=label))
            if model == "lgbm":
                train0 = trains[0]
            if ted == "aqsoldb" and model in ("lgbm", "rf"):
                rel = H.reliability3(tdf.loc[sp["test"], "reliability_group"]).to_numpy()
                for g in H.REL3_ORDER:
                    mae, lo, hi = H.mae_ci(aes[label][rel == g], n_boot=args.n_boot)
                    rel_rows.append({"direction": name, "model": label, "rel3": g, "n": int((rel == g).sum()),
                                     "MAE": mae, "ci_low": lo, "ci_high": hi})
        if not preds:
            continue
        sim0 = sims_by_model.get(EXT_LABELS["lgbm"], next(iter(sims_by_model.values())))
        if name in EXTERNAL:
            # Delaney: published equation (approx. in-sample for ESOL, out of sample for AqSolDB) + published preds
            X4, ids4, _ = load_or_compute(ted, "esol4")
            pos = {c: i for i, c in enumerate(ids4.astype(str))}
            ref = [("Delaney published equation (Crippen cLogP)", H.delaney_equation(X4[[pos[c] for c in sp["test"]]]))]
            if ted == "esol":
                ref.append(("Delaney published ESOL predictions (in-sample)",
                            tdf.loc[sp["test"], "esol_pred"].to_numpy(float)))
            for label, p in ref:
                m = H.metrics_with_bias(y, p)
                rows.append({"direction": name, "model": label, "n_train": 0 if "equation" in label else np.nan,
                             "n_test": len(y), "n_seeds": 1, **{k: m[k] for k in ("MAE", "RMSE", "R2", "Pearson",
                                                                                 "Spearman", "bias")},
                             "MAE_sd": 0.0, "RMSE_sd": 0.0, "R2_sd": 0.0})
                preds[label] = (p, m)
                simrows.append(H.similarity_bin_table(sim0, np.abs(p - y)).assign(direction=name, model=label))
            parity[name] = (y, preds)
        dom, inside = domain_diagnostics(name, train0 or sp["train"], sp["test"], trdf, tdf, sim0,
                                         aes.get(EXT_LABELS["lgbm"]))
        domain.append(dom)
        rel3 = H.reliability3(tdf.loc[sp["test"], "reliability_group"]).to_numpy() if ted == "aqsoldb" else None
        cmp[name] = {"ae": aes.get(EXT_LABELS["lgbm"]), "sim": sim0, "inside": inside, "rel3": rel3}
    if not rows:
        return {}
    ext, sim_tab, dom = pd.DataFrame(rows), pd.concat(simrows, ignore_index=True), pd.DataFrame(domain)
    asym = asymmetry_table(cmp, args.n_boot)
    out = P.out / "external"
    out.mkdir(exist_ok=True)
    ext.to_csv(out / "external_metrics.csv", index=False)
    sim_tab.to_csv(out / "external_mae_by_similarity.csv", index=False)
    dom.to_csv(out / "external_domain.csv", index=False)
    asym.to_csv(out / "asymmetry_decomposition.csv", index=False)
    write_table(ext, P, "external")
    write_table(sim_tab[["direction", "model", "sim_bin", "n", "MAE", "MAE_ci_low", "MAE_ci_high"]], P,
                "external_similarity")
    write_table(dom.set_index("direction").T.reset_index().rename(columns={"index": "quantity"}), P,
                "external_domain")
    write_table(asym, P, "external_asymmetry")
    rel = pd.DataFrame(rel_rows)
    if not rel.empty:
        rel.to_csv(out / "esol_to_aqsoldb_by_reliability.csv", index=False)
        write_table(rel, P, "external_by_reliability")
    plot_external(P, parity, sim_tab, asym)
    return {"unit": "metrics = mean over model seeds; compound-level analyses use seed-averaged |error|",
            "metrics": ext.to_dict("records"), "domain": dom.to_dict("records"),
            "asymmetry_decomposition_lgbm": asym.to_dict("records"),
            "esol_to_aqsoldb_by_reliability": rel_rows}


def asymmetry_table(cmp: dict, n_boot: int) -> pd.DataFrame:
    """LightGBM MAE of the two transfer directions under progressively matched conditions (asymmetry decomposition)."""
    rows = []

    def add(scenario: str, ae: np.ndarray, note: str = "") -> None:
        mae, lo, hi = H.mae_ci(ae, n_boot=n_boot)
        rows.append({"scenario": scenario, "n_test": len(ae), "MAE": mae, "ci_low": lo, "ci_high": hi, "note": note})

    f, r, s = cmp.get("aqsoldb_to_esol"), cmp.get("esol_to_aqsoldb"), cmp.get(SIZEMATCHED)
    if f is None or r is None or f["ae"] is None or r["ae"] is None:
        return pd.DataFrame(rows)
    add("AqSolDB -> ESOL, full training set (n_train=7267)", f["ae"])
    if s is not None and s["ae"] is not None:
        add("AqSolDB -> ESOL, training set size-matched to ESOL (n_train=1004, 5 draws)", s["ae"])
    add("ESOL -> AqSolDB, all test compounds (n_train=1004)", r["ae"])
    add("ESOL -> AqSolDB, inside descriptor domain", r["ae"][r["inside"]],
        "single-component, uncharged, no new element, MW <= train max")
    add("ESOL -> AqSolDB, consistent-label test compounds (G3+G5)", r["ae"][r["rel3"] == H.REL3_ORDER[1]])
    both = r["inside"] & (r["rel3"] == H.REL3_ORDER[1])
    add("ESOL -> AqSolDB, inside domain AND consistent labels", r["ae"][both])
    ref = s if (s is not None and s["ae"] is not None) else f
    for lab, m in [("all test compounds", np.ones(len(r["ae"]), bool)), ("inside domain AND consistent labels", both)]:
        v = H.similarity_standardised_mae(r["sim"][m], r["ae"][m], ref["sim"])
        rows.append({"scenario": f"ESOL -> AqSolDB, {lab}, similarity-standardised", "n_test": int(m.sum()),
                     "MAE": v, "ci_low": np.nan, "ci_high": np.nan,
                     "note": "bin-MAEs re-weighted to the NN-similarity distribution of the size-matched "
                             "AqSolDB -> ESOL test set"})
    return pd.DataFrame(rows)


def plot_external(P: Paths, parity: dict, sim_tab: pd.DataFrame, asym: pd.DataFrame) -> None:
    """Parity grid, MAE vs NN similarity for all directions, and the asymmetry decomposition."""
    plt = plt_setup()
    cols = [EXT_LABELS["lgbm"], EXT_LABELS["rf"], EXT_LABELS["esol_linear"], "Delaney"]
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.8))
    for i, name in enumerate(EXTERNAL):
        if name not in parity:
            continue
        y, preds = parity[name]
        for j, col in enumerate(cols):
            ax = axes[i, j]
            key = col if col in preds else next((k for k in preds if k.startswith(col) and "in-sample" in k),
                                                next((k for k in preds if k.startswith(col)), None))
            if key is None:
                ax.axis("off")
                continue
            p, m = preds[key]
            ax.scatter(y, p, s=4 if len(y) > 2000 else 7, alpha=0.35, color=SPLIT_COLORS["external"], lw=0)
            lo, hi = min(y.min(), p.min()) - 0.5, max(y.max(), p.max()) + 0.5
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.text(0.03, 0.97, f"MAE {m['MAE']:.2f}\nRMSE {m['RMSE']:.2f}\nR2 {m['R2']:.2f}\nbias {m['bias']:+.2f}",
                    transform=ax.transAxes, va="top", fontsize=7)
            ax.set_title(f"{EXT_TITLES[name]}\n{key}", fontsize=8)
            ax.set_xlabel(f"measured logS ({LOGS_UNIT})")
            ax.set_ylabel(f"predicted logS ({LOGS_UNIT})")
    fig.text(0.5, 0.005, "points: seed-averaged prediction; metrics: mean over model seeds", ha="center", fontsize=7)
    savefig(fig, P, "external_parity")

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.9), sharey=True)
    styles = {"aqsoldb_to_esol": ("#C44E52", "-", "o"), SIZEMATCHED: ("#E8A0A2", ":", "^"),
              "esol_to_aqsoldb": ("#7A1F2B", "--", "s")}
    bins = list(dict.fromkeys(sim_tab["sim_bin"]))
    x = np.arange(len(bins))
    for ax, model in zip(axes, [EXT_LABELS["lgbm"], EXT_LABELS["rf"]]):
        for k, name in enumerate(["aqsoldb_to_esol", SIZEMATCHED, "esol_to_aqsoldb"]):
            d = sim_tab[(sim_tab["direction"] == name) & (sim_tab["model"] == model)].set_index("sim_bin").reindex(bins)
            d = d[d["n"] >= 10]
            if d.empty:
                continue
            c, ls, mk = styles[name]
            xi = np.array([bins.index(b) for b in d.index]) + (k - 1) * 0.06
            ax.errorbar(xi, d["MAE"], yerr=[d["MAE"] - d["MAE_ci_low"], d["MAE_ci_high"] - d["MAE"]], color=c, ls=ls,
                        marker=mk, capsize=2, ms=4, label=EXT_TITLES[name])
            for xv, n, v in zip(xi, d["n"], d["MAE"]):
                ax.text(xv, v, f" {int(n)}", fontsize=6, color=c, va="bottom")
        ax.set_xticks(x)
        ax.set_xticklabels(bins, rotation=20)
        ax.set_xlabel("max Tanimoto similarity (ECFP4) of test compound to training set")
        ax.set_title(model)
    axes[0].set_ylabel(f"MAE ({LOGS_UNIT}), 95% CI; labels = n (bins n>=10)")
    axes[0].legend(frameon=False, fontsize=7)
    savefig(fig, P, "external_mae_vs_similarity")

    if asym.empty:
        return
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(asym) + 1.2))
    yy = np.arange(len(asym))
    col = ["#E8A0A2" if s.startswith("AqSolDB") else SPLIT_COLORS["external"] for s in asym["scenario"]]
    ax.barh(yy, asym["MAE"], color=col, xerr=[(asym["MAE"] - asym["ci_low"]).fillna(0),
                                              (asym["ci_high"] - asym["MAE"]).fillna(0)], capsize=2)
    for yv, v, m, n in zip(yy, asym["ci_high"].fillna(asym["MAE"]), asym["MAE"], asym["n_test"]):
        ax.text(v + 0.02, yv, f"{m:.2f} (n={n})", va="center", fontsize=7)
    ax.set_yticks(yy)
    ax.set_yticklabels(asym["scenario"], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(f"LightGBM test MAE ({LOGS_UNIT}), 95% bootstrap CI")
    ax.set_title("Why ESOL -> AqSolDB transfers worse than AqSolDB -> ESOL")
    ax.set_xlim(0, asym["MAE"].max() * 1.3)
    savefig(fig, P, "external_asymmetry")



# --------------------------------------------------------------------------- report driver
def _r(x, nd: int = 4):
    """Round floats for the headline block (keeps ints / None / NaN-as-None)."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return round(float(x), nd) if isinstance(x, (float, np.floating)) else x


def _ci(row, key: str = "MAE", lo: str = "ci_low", hi: str = "ci_high") -> list:
    return [_r(row[key]), _r(row[lo]), _r(row[hi])]


def headline_numbers(P: Paths) -> dict:
    """Compact headline numbers (value, or [value, 95% CI low, high]) read back from the stage outputs."""
    out = P.out
    h: dict = {}
    f = out / "summary_noise.json"
    if f.exists():
        s = json.loads(f.read_text())
        est = s["noise_estimates"]
        rms, pooled = est["RMS SD (ddof=0, naive)"], est["pooled SD (ddof-corrected)"]
        rob = est["robust SD from pairs (median |x1-x2|)"]
        cons = est["pooled SD, consistent reports only (G3+G5)"]
        npf = est["distribution-free MAE floor from pairs (E|x1-x2|/sqrt2)"]
        h["noise"] = {
            "n_multi_report": s["n_multi_report"], "frac_single_report_G1": _r(s["frac_single_report_G1"]),
            "rms_sd_ddof0_ci": [_r(rms["sigma"]), _r(rms["ci_low"]), _r(rms["ci_high"])],
            "pooled_sd_ddof_corrected_ci": [_r(pooled["sigma"]), _r(pooled["ci_low"]), _r(pooled["ci_high"])],
            "robust_sd_pairs_ci": [_r(rob["sigma"]), _r(rob["ci_low"]), _r(rob["ci_high"])],
            "pooled_sd_consistent_G3G5": _r(cons["sigma"]),
            "recommended_floors": s.get("recommended_floors"),
            "noise_floor_caveat": s.get("noise_floor_caveat"),
            "gaussian_floor_from_rms_sd": {"RMSE": _r(rms["RMSE_floor"]), "MAE": _r(rms["MAE_floor_gaussian"])},
            "gaussian_floor_from_robust_sd": {"RMSE": _r(rob["RMSE_floor"]), "MAE": _r(rob["MAE_floor_gaussian"])},
            "distribution_free_MAE_floor_pairs_ci": [_r(npf["MAE_floor_gaussian"]), _r(npf["ci_low"]),
                                                     _r(npf["ci_high"])],
            "sd_median_by_group": {g: _r(v["sd_median"]) for g, v in s["sd_by_group"].items()},
            "sd_spearman": {k: _r(v["spearman_rho"]) for k, v in s["sd_correlates_spearman"].items()},
            "sd_convention": s["sd_convention"]["conclusion"],
            "dedup_groups": s["dedup"]["n_groups"], "dedup_conflicts_sd_gt_0.5": s["dedup"]["n_conflict_sd_gt_0.5"],
            "esol_overlap_bland_altman": {k: _r(s["esol_aqsoldb_bland_altman"][k]) for k in
                                          ("n", "bias", "loa_low", "loa_high", "MAE", "frac_identical_lt_0.01",
                                           "frac_absdiff_gt_0.5")},
        }
    f = out / "reliability" / "mae_by_group.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[(d["model"] == LGBM_FILE) & (d["grouping"] == "rel3")]
        h["error_vs_reliability_lgbm"] = {f"{r.split_type}|{r.level}": [_r(r.MAE), _r(r.MAE_ci_low),
                                                                        _r(r.MAE_ci_high), int(r.n)]
                                          for r in d.itertuples()}
        c = pd.read_csv(out / "reliability" / "contrasts.csv")
        c = c[c["model"] == LGBM_FILE]
        h["reliability_contrasts_lgbm"] = {f"{r.split_type}|{r.contrast}": [_r(r.mae_diff), _r(r.ci_low),
                                                                            _r(r.ci_high)] for r in c.itertuples()}
    f = out / "ablation" / "delta_pooled.csv"
    if f.exists():
        d = pd.read_csv(f)
        d = d[d["subset"].isin(["full test", "high-reliability test (G3+G5)"])]
        h["ablation_delta_MAE"] = {f"{r.split_type}|{r.variant}|{r.subset}": [_r(r.delta_MAE), _r(r.ci_low),
                                                                               _r(r.ci_high)] for r in d.itertuples()}
        h["ablation_MAE_a_all"] = {f"{r.split_type}|{r.subset}": _r(r.MAE_a_all)
                                   for r in d.drop_duplicates(["split_type", "subset"]).itertuples()}
        c = pd.read_csv(out / "ablation" / "treatment_vs_control.csv")
        c = c[c["subset"].isin(["full test", "high-reliability test (G3+G5)"])]
        h["ablation_treatment_minus_control"] = {f"{r.split_type}|{r.contrast}|{r.subset}":
                                                 [_r(r.delta_MAE), _r(r.ci_low), _r(r.ci_high)] for r in c.itertuples()}
    f = out / "loso" / "loso_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        h["loso"] = {r["source"]: {"n_test": int(r["n_test"]), "MAE": _ci(r, "MAE", "MAE_ci_low", "MAE_ci_high"),
                                   "RMSE": _r(r["RMSE"]), "R2": _r(r["R2"]), "bias": _r(r["bias"]),
                                   "control_MAE": _r(r.get("control_MAE")),
                                   "random_split_MAE_same_compounds": _r(r.get("random_MAE")),
                                   "random_split_R2_same_compounds": _r(r.get("random_R2")),
                                   "source_shift_penalty_vs_size_matched_control": _ci(
                                       r, "penalty_vs_control", "penalty_vs_control_ci_low",
                                       "penalty_vs_control_ci_high"),
                                   "penalty_vs_random_split_NOT_size_matched_larger_train_subset": _ci(
                                       r, "penalty_vs_random", "penalty_vs_random_ci_low",
                                       "penalty_vs_random_ci_high"),
                                   "penalty_at_equal_similarity": _ci(r, "adj_penalty", "adj_penalty_ci_low",
                                                                      "adj_penalty_ci_high")}
                     for _, r in d.iterrows()}
        h["loso_penalty_definitions"] = LOSO_PENALTY_NOTES
    f = out / "transfer" / "in_vs_cross_source.csv"
    if f.exists():
        d = pd.read_csv(f)
        h["source_transfer"] = {r["test_source"]: {"in_source_MAE": _r(r["in_source_MAE"]),
                                                   "cross_source_MAE_mean": _r(r["cross_source_MAE_mean"]),
                                                   "mixed_MAE": _r(r["mixed_MAE"]),
                                                   "cross_penalty_at_equal_similarity": _ci(
                                                       r, "cross_source_penalty_adj", "adj_ci_low", "adj_ci_high")}
                                for _, r in d.iterrows()}
    f = out / "external" / "external_metrics.csv"
    if f.exists():
        d = pd.read_csv(f)
        h["external"] = {f"{r.direction}|{r.model}": {"MAE": _r(r.MAE), "RMSE": _r(r.RMSE), "R2": _r(r.R2),
                                                      "Pearson": _r(r.Pearson), "Spearman": _r(r.Spearman),
                                                      "bias": _r(r.bias), "n_seeds": int(r.n_seeds)}
                         for r in d.itertuples()}
        a = pd.read_csv(out / "external" / "asymmetry_decomposition.csv")
        h["external_asymmetry_lgbm_MAE"] = {r.scenario: _r(r.MAE) for r in a.itertuples()}
    return h


def stage_report(args, P: Paths, meta: pd.DataFrame) -> None:
    """Build every table/figure from available predictions and merge stage summaries into summary.json."""
    parts = {"ablation": report_ablation, "loso": report_loso, "transfer": report_transfer, "external": report_external}
    for name, fn in parts.items():
        res = fn(args, P, meta)
        if res:
            save_json(res, P.out / f"summary_{name}.json")
            log(f"report: {name} done")
        else:
            log(f"report: {name} skipped (no predictions yet)")
    summary = {"component": "heterogeneity", "model": f"LightGBM {FEAT} (registry defaults) unless stated",
               "script": "scripts/09_heterogeneity_transfer.py", "module": "src/solpred/analysis/heterogeneity.py",
               "n_boot": args.n_boot, "headline": headline_numbers(P), "details": {}}
    for name in ("noise", "reliability", "ablation", "loso", "transfer", "external"):
        f = P.out / f"summary_{name}.json"
        if f.exists():
            summary["details"][name] = json.loads(f.read_text())
    save_json(summary, P.out / "summary.json")
    log("report: summary.json written")


# --------------------------------------------------------------------------- driver
STAGES =["noise", "reliability", "ablation", "loso", "transfer", "external", "report"]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--n-jobs", type=int, default=2, help="threads per model fit")
    ap.add_argument("--split-types", nargs="+", default=["random", "scaffold"], help="ablation split types")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4], help="ablation split seeds")
    ap.add_argument("--variants", nargs="+", default=list(H.ABLATION_VARIANTS),
                    help="ablation variants (e_random_drop_rep1 etc. add control replicates)")
    ap.add_argument("--refit-reference", action="store_true",
                    help="refit variant a_all instead of reusing the bit-identical baseline prediction")
    ap.add_argument("--sources", nargs="+", default=LOSO_SOURCES, help="LOSO held-out sources")
    ap.add_argument("--control-folds", type=int, default=3, help="folds of the size-matched LOSO control")
    ap.add_argument("--transfer-reps", type=int, default=3)
    ap.add_argument("--transfer-n-train", type=int, default=500)
    ap.add_argument("--transfer-n-val", type=int, default=100)
    ap.add_argument("--external-seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    cfg = load_config(args.config)
    P = Paths(cfg)
    meta = load_dataset("aqsoldb", cfg)
    for st in args.stages:
        log(f"== stage {st}")
        globals()[f"stage_{st}"](args, P, meta)
