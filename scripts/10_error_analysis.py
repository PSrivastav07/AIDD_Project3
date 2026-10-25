"""Step 10 - similarity-aware error analysis, chemical-space map and explainability.

Analyses the stored test-set predictions of ONE model (``--pred-root`` / ``--model-tag``; default:
the Phase-1 reference LightGBM on RDKit2D + Morgan counts in results/baseline) and writes

* ``sim``      MAE vs. test->train max Tanimoto similarity (ECFP4) per split type, pooled over seeds,
               cluster-bootstrap CIs, plus the fraction of test compounds per similarity bin;
* ``label``    MAE by true-logS bin, label distribution of the test sets, direct standardisation of the
               random-split errors to the scaffold/cluster strata and an OLS adjusted split penalty;
* ``chem``     MAE / mean signed error by physicochemical bins and ~20 SMARTS functional-group classes;
* ``worst``    top-30 absolute errors of the random+scaffold seed-0 test sets with rule-based hypotheses
               and a structure grid;
* ``cliffs``   AqSolDB pairs with Tanimoto >= 0.9 and |dlogS| >= 2, typed as fingerprint-identical /
               homologues (ECFP4 bit-vector saturation) / same-size pairs; only the same-size pairs are
               candidate label noise or true activity cliffs;
* ``scaffold`` scaffold-level errors in the scaffold splits (scaffolds with >= 5 pooled test rows);
* ``umap``     UMAP (Jaccard on ECFP4 bits) of AqSolDB with ESOL overlaid, coloured by logS, split
               assignment and error;
* ``shap``     TreeSHAP for LightGBM on rdkit2d and on rdkit2d+morgan_counts (random seed 0).

Each step merges its headline numbers into ``results/error_analysis[/<tag>]/summary.json`` so the
steps can be run in chunks (``--steps sim label``). Model-independent intermediates (nearest-neighbour
similarities, functional-group matrix, similar pairs, UMAP coordinates, SHAP values) are cached in
``results/error_analysis/cache``.

Reference run (Phase-1 LightGBM; writes the un-tagged outputs)::

    .venv/Scripts/python.exe scripts/10_error_analysis.py --n-jobs 3

Re-run on another model (e.g. the final best benchmark / GNN model) without overwriting the reference
outputs: ``--tag`` writes results to ``results/error_analysis/<tag>/``, figures to
``results/figures/error_analysis/<tag>/`` and tables to ``results/tables/error_<tag>_*``::

    .venv/Scripts/python.exe scripts/10_error_analysis.py --pred-root results/gnn \
        --model-tag <model>__<featurizer> --tag <name> --secondary-model-tag "" \
        --compare-model-tags --steps sim label chem worst scaffold umap --n-jobs 3

(``cliffs`` and ``shap`` do not depend on the analysed model, apart from the cliff/worst-30 overlap count;
``umap`` re-uses the cached embedding and only re-colours the error panel with the new model.)
"""
from __future__ import annotations

import argparse
import os
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("NUMBA_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "3")

from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem  # noqa: E402
from rdkit.Chem import Draw  # noqa: E402
from scipy import stats  # noqa: E402

from solpred.analysis import chemspace as cs  # noqa: E402
from solpred.analysis import errors as ea  # noqa: E402
from solpred.config import ROOT, load_json, path, save_json  # noqa: E402
from solpred.features import load_dataset, load_or_compute, subset  # noqa: E402
from solpred.splits import load_split, morgan_bits  # noqa: E402

warnings.filterwarnings("ignore", message="missing predictions")
warnings.filterwarnings("ignore", category=stats.ConstantInputWarning)
SPLIT_COLORS = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52"}
DPI = 200
LOGS_UNIT = "log mol/L"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 11, "legend.fontsize": 8.5, "figure.dpi": 100})
META_COLS = ["compound_id", "name", "smiles", "scaffold", "source", "reliability_group", "n_occurrences",
             "aqsol_sd", "dedup_conflict", "is_multicomponent", "formal_charge_atoms", "parent_smiles"]


# =========================================================================== context / io
class Ctx:
    """Paths, CLI options and lazily loaded shared tables for one analysed model."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        base = path("results") / "error_analysis"
        self.res = base / args.out_subdir if args.out_subdir else base
        self.cache = base / "cache"
        self.fig = path("figures") / "error_analysis" / args.out_subdir if args.out_subdir \
            else path("figures") / "error_analysis"
        self.tab = path("tables")
        self.prefix = f"error_{args.out_subdir}_" if args.out_subdir else "error_"
        for d in (self.res, self.cache, self.fig):
            d.mkdir(parents=True, exist_ok=True)
        self.pred_root = Path(args.pred_root) if Path(args.pred_root).is_absolute() else ROOT / args.pred_root
        self._errors: dict[str, pd.DataFrame] = {}
        self._aq: pd.DataFrame | None = None
        self._fg: pd.DataFrame | None = None

    # ---------------------------------------------------------------- shared tables
    @property
    def aq(self) -> pd.DataFrame:
        if self._aq is None:
            self._aq = load_dataset("aqsoldb")
        return self._aq

    def nn_table(self) -> pd.DataFrame:
        return ea.nn_similarity_table(self.args.split_types, self.args.seeds, self.cache / "nn_similarity.csv")

    def errors(self, model_tag: str | None = None) -> pd.DataFrame:
        """Pooled test errors of a model with NN similarity, bins and compound metadata."""
        tag = model_tag or self.args.model_tag
        if tag not in self._errors:
            e = ea.load_predictions(self.pred_root, tag, self.args.split_types, self.args.seeds)
            nn = self.nn_table()
            e = e.merge(nn, on=["split_type", "split_name", "compound_id"], how="left")
            e = e.merge(self.aq[META_COLS], on="compound_id", how="left")
            ea.add_bins(e, "nn_sim", ea.SIM_EDGES, "sim_bin", labels=ea.sim_labels())
            ea.add_bins(e, "nn_sim", ea.SIM_EDGES_COARSE, "sim_coarse",
                        labels=["<0.4", "0.4-0.5", "0.5-0.6", "0.6-0.7", ">=0.7"])
            ea.add_bins(e, "y_true", ea.LOGS_EDGES, "logS_bin",
                        labels=["< -6", "-6 to -4", "-4 to -2", "-2 to 0", ">= 0"])
            self._errors[tag] = e
        return self._errors[tag]

    def functional_groups(self) -> pd.DataFrame:
        f = self.cache / "functional_groups_aqsoldb.csv"
        if self._fg is None:
            if f.exists():
                self._fg = pd.read_csv(f, dtype={"compound_id": str})
            else:
                fg = ea.functional_group_matrix(self.aq["smiles"].tolist())
                fg.insert(0, "compound_id", self.aq["compound_id"].to_numpy())
                fg.to_csv(f, index=False)
                self._fg = fg
        return self._fg

    # ---------------------------------------------------------------- outputs
    def table(self, df: pd.DataFrame, name: str) -> None:
        df.to_csv(self.tab / f"{self.prefix}{name}.csv", index=False)
        md = df.copy()
        for c in md.select_dtypes(include=["object", "string"]).columns:
            md[c] = md[c].astype(str).str.replace("|", "\\|", regex=False)
        (self.tab / f"{self.prefix}{name}.md").write_text(md.to_markdown(index=False, floatfmt=".3f"),
                                                           encoding="utf-8")

    def savefig(self, fig, name: str) -> None:
        fig.savefig(self.fig / f"{name}.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  figure -> {self.fig / name}.png", flush=True)

    def update_summary(self, key: str, value) -> None:
        f = self.res / "summary.json"
        s = load_json(f) if f.exists() else {}
        s["model"] = {"pred_root": self.pred_root.relative_to(ROOT).as_posix()
                      if self.pred_root.is_relative_to(ROOT) else self.pred_root.as_posix(), "model_tag": self.args.model_tag,
                      "split_types": self.args.split_types, "seeds": self.args.seeds,
                      "signed_error": ea.SIGNED_ERROR_NOTE,
                      "ci": "95% percentile cluster bootstrap over unique compounds "
                            f"({self.args.n_boot} resamples), seeds pooled"}
        s[key] = value
        save_json(_clean(s), f)

    def model_label(self, tag: str | None = None) -> str:
        tag = tag or self.args.model_tag
        m, _, f = tag.partition("__")
        return {"lgbm": "LightGBM", "rf": "Random forest", "esol_linear": "ESOL refit", "mean": "Mean"}.get(m, m) \
            + (f" ({f})" if f else "")


def _clean(o):
    """Recursively convert numpy / NaN values for JSON."""
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 4)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _ci(row) -> dict:
    return {"MAE": row["MAE"], "lo": row["MAE_lo"], "hi": row["MAE_hi"], "n": row["n"],
            "n_compounds": row["n_compounds"], "mean_signed_error": row["mean_signed_error"]}


def mol_grid(smiles: list[str], legends: list[str], ncols: int, title: str, cell: float = 2.3):
    """Matplotlib grid of RDKit depictions with readable multi-line captions."""
    n = len(smiles)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(cell * ncols, cell * 1.18 * nrows))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    for k, (s, leg) in enumerate(zip(smiles, legends)):
        ax = np.atleast_1d(axes).ravel()[k]
        m = Chem.MolFromSmiles(s) if s else None
        if m is not None:
            ax.imshow(Draw.MolToImage(m, size=(360, 300)))
        ax.set_title(leg, fontsize=6.3, pad=2, loc="center")
    fig.suptitle(title, fontsize=11, y=1.0)
    fig.tight_layout()
    return fig


# =========================================================================== step: similarity
def present_splits(e: pd.DataFrame) -> list[str]:
    return [s for s in ea.SPLIT_ORDER if s in set(e["split_type"])]


def step_sim(ctx: Ctx) -> None:
    """Figure 1: MAE vs nearest-training-neighbour similarity, per split type."""
    e = ctx.errors()
    e.to_csv(ctx.res / "test_errors_with_similarity.csv", index=False)
    nb = ctx.args.n_boot
    tab = ea.error_summary(e, ["split_type", "sim_bin"], n_boot=nb)
    tot = e.groupby("split_type").size()
    tab["frac_of_split"] = tab["n"] / tab["split_type"].map(tot)
    ctx.table(tab, "similarity_bins")
    overall = ea.error_summary(e, ["split_type"], n_boot=nb)
    per_seed = e.groupby(["split_type", "split_name"])["abs_error"].mean().groupby("split_type").agg(["mean", "std",
                                                                                                         "size"])
    sec = None
    if ctx.args.secondary_model_tag:
        try:
            es = ctx.errors(ctx.args.secondary_model_tag)
            sec = ea.error_summary(es, ["split_type", "sim_bin"], n_boot=200)
        except FileNotFoundError:
            print("  secondary model predictions not found - skipped", flush=True)
    _plot_similarity(ctx, tab, overall, sec)

    # headline numbers: high vs low similarity
    out = {}
    for st in present_splits(e):
        d = e[e["split_type"] == st]
        parts = {}
        for key, mask in {"sim_ge_0.7": d["nn_sim"] >= 0.7, "sim_0.4_0.7": (d["nn_sim"] >= 0.4) & (d["nn_sim"] < 0.7),
                          "sim_lt_0.4": d["nn_sim"] < 0.4}.items():
            g = d[mask]
            m, lo, hi = ea.grouped_bootstrap_mean(g["abs_error"], g["compound_id"], nb)
            parts[key] = {"MAE": m, "lo": lo, "hi": hi, "n": len(g), "frac": float(mask.mean())}
        o = overall[overall["split_type"] == st].iloc[0]
        rho = stats.spearmanr(d["nn_sim"], d["abs_error"])
        out[st] = {"pooled": _ci(o), "per_seed_MAE_mean": per_seed.loc[st, "mean"],
                   "per_seed_MAE_std": per_seed.loc[st, "std"], "n_seeds": per_seed.loc[st, "size"],
                   "median_nn_sim": float(d["nn_sim"].median()), **parts,
                   "spearman_nn_sim_vs_abs_error": float(rho.statistic), "spearman_p": float(rho.pvalue)}
    out["notes"] = [
        "NN similarity = max ECFP4 Tanimoto to the TRAINING compounds of the same split (validation excluded; "
        "the model is not fit on validation compounds)",
        "the top bin (>= 0.9) of the random split is affected by binary-ECFP4 saturation: many neighbours are "
        "fingerprint-identical (Tanimoto 1.0) homologues (e.g. cyclosiloxanes, chain-length series) whose "
        "solubility differs legitimately, which explains the error uptick in that bin",
    ]
    ctx.update_summary("similarity", out)
    if ctx.args.compare_model_tags:
        _plot_model_comparison(ctx)


def _plot_similarity(ctx: Ctx, tab: pd.DataFrame, overall: pd.DataFrame, sec: pd.DataFrame | None) -> None:
    labels = ea.sim_labels()
    x = np.arange(len(labels))
    splits = [s for s in ea.SPLIT_ORDER if s in set(tab["split_type"])]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.6, 7.2), sharex=True,
                                   gridspec_kw={"height_ratios": [2.3, 1], "hspace": 0.08})
    width = 0.8 / len(splits)
    for k, st in enumerate(splits):
        d = tab[tab["split_type"] == st].set_index("sim_bin").reindex(labels)
        off = (k - (len(splits) - 1) / 2) * 0.09
        ok = d["n_compounds"].fillna(0) >= 20
        o = overall[overall["split_type"] == st].iloc[0]
        lab = f"{st} (MAE {o['MAE']:.2f}, n={int(o['n'])})"
        ax1.plot(x + off, d["MAE"], "-", color=SPLIT_COLORS[st], lw=1.6, zorder=2)
        yerr = np.vstack([d["MAE"] - d["MAE_lo"], d["MAE_hi"] - d["MAE"]])
        ax1.errorbar((x + off)[ok], d["MAE"][ok], yerr=yerr[:, ok], fmt="o", color=SPLIT_COLORS[st],
                     ms=5.5, capsize=2.5, lw=1.2, label=lab, zorder=3)
        if (~ok & d["MAE"].notna()).any():
            ax1.plot((x + off)[~ok], d["MAE"][~ok], "o", mfc="white", color=SPLIT_COLORS[st], ms=5.5, zorder=3)
        if sec is not None:
            s2 = sec[sec["split_type"] == st].set_index("sim_bin").reindex(labels)
            ok2 = s2["n_compounds"].fillna(0) >= 20
            ax1.plot((x + off)[ok2], s2["MAE"][ok2], "--", color=SPLIT_COLORS[st], lw=1.1, alpha=0.65)
        ax2.bar(x + (k - (len(splits) - 1) / 2) * width, d["frac_of_split"].fillna(0), width=width,
                color=SPLIT_COLORS[st], label=st)
    if sec is not None:
        ax1.plot([], [], "--", color="0.4", label=f"{ctx.model_label(ctx.args.secondary_model_tag)} (dashed)")
    ax1.set_ylabel(f"MAE ({LOGS_UNIT})")
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="lower left", frameon=False, title=ctx.model_label(), title_fontsize=8.5)
    ax1.set_title("Test error vs. similarity to the training set (seeds 0-4 pooled, 95% bootstrap CI)")
    ax1.text(0.99, 0.02, "open markers: < 20 compounds in bin\n>= 0.9 bin: ECFP4 saturation (homologues with "
                         "Tanimoto 1.0)", transform=ax1.transAxes, fontsize=7.5, color="0.35", ha="right")
    ax2.set_ylabel("fraction of test set")
    ax2.set_xticks(x, labels)
    ax2.set_xlabel("max Tanimoto similarity of test compound to training set (ECFP4, 2048 bits)")
    ctx.savefig(fig, "mae_vs_similarity")


def _plot_model_comparison(ctx: Ctx) -> None:
    """Supplement: MAE vs similarity for several models of the same prediction root."""
    labels = ea.sim_labels()
    x = np.arange(len(labels))
    tags = [t for t in ctx.args.compare_model_tags]
    splits = [s for s in ("random", "scaffold", "cluster") if s in ctx.args.split_types]
    fig, axes = plt.subplots(1, len(splits), figsize=(4.2 * len(splits), 3.8), sharey=True)
    styles = ["-o", "--s", ":^", "-.D", "-v"]
    rows = []
    for t_i, tag in enumerate(tags):
        try:
            e = ctx.errors(tag)
        except FileNotFoundError:
            continue
        tab = ea.error_summary(e, ["split_type", "sim_bin"], n_boot=200)
        tab.insert(0, "model_tag", tag)
        rows.append(tab)
        for ax, st in zip(np.atleast_1d(axes), splits):
            d = tab[tab["split_type"] == st].set_index("sim_bin").reindex(labels)
            ok = d["n_compounds"].fillna(0) >= 20
            ax.plot(x[ok], d["MAE"][ok], styles[t_i % len(styles)], color=plt.cm.Dark2(t_i), ms=4,
                    lw=1.3, label=ctx.model_label(tag))
    for ax, st in zip(np.atleast_1d(axes), splits):
        ax.set_title(st, color=SPLIT_COLORS[st])
        ax.set_xticks(x, labels, rotation=45, ha="right", fontsize=8)
        ax.set_xlabel("max Tanimoto to training set (ECFP4)")
    np.atleast_1d(axes)[0].set_ylabel(f"MAE ({LOGS_UNIT})")
    np.atleast_1d(axes)[0].legend(frameon=False, fontsize=7.5)
    fig.tight_layout()
    ctx.savefig(fig, "mae_vs_similarity_by_model")
    if rows:
        ctx.table(pd.concat(rows, ignore_index=True), "similarity_bins_by_model")


# =========================================================================== step: label shift
def step_label(ctx: Ctx) -> None:
    """Figure 2: label shift vs structural novelty."""
    e = ctx.errors()
    e = e[e["split_type"].isin(["random", "scaffold", "cluster"])]
    nb = ctx.args.n_boot
    by_label = ea.error_summary(e, ["split_type", "logS_bin"], n_boot=nb)
    tot = e.groupby("split_type").size()
    by_label["frac_of_split"] = by_label["n"] / by_label["split_type"].map(tot)
    ctx.table(by_label, "logS_bins")
    cells = ea.error_summary(e, ["split_type", "sim_coarse", "logS_bin"], n_boot=200)
    cells = cells[cells["n_compounds"] >= 10]
    wide = cells.pivot_table(index=["sim_coarse", "logS_bin"], columns="split_type", values="MAE", observed=True)
    wide.columns = [f"MAE_{c}" for c in wide.columns]
    ncell = cells.pivot_table(index=["sim_coarse", "logS_bin"], columns="split_type", values="n", observed=True)
    ncell.columns = [f"n_{c}" for c in ncell.columns]
    ctx.table(wide.join(ncell).reset_index(), "matched_strata")

    src = e[e["split_type"] == "random"]
    if src.empty or e["split_type"].nunique() < 2:  # nothing to standardise against
        print("  random split plus another split type needed for standardisation - skipped", flush=True)
        _plot_label(ctx, by_label, pd.DataFrame())
        ctx.update_summary("label_shift", {"MAE_by_logS_bin": {f"{r.split_type} | {r.logS_bin}": {"MAE": r.MAE,
                                                                                                  "n": r.n}
                                                               for r in by_label.itertuples()}})
        return
    std_rows = []
    for tgt in ("scaffold", "cluster"):
        t = e[e["split_type"] == tgt]
        if t.empty:
            continue
        obs = ea.grouped_bootstrap_mean(t["abs_error"], t["compound_id"], nb)
        raw = ea.grouped_bootstrap_mean(src["abs_error"], src["compound_id"], nb)
        std_rows.append({"target": tgt, "estimate": "observed", "MAE": obs[0], "lo": obs[1], "hi": obs[2]})
        std_rows.append({"target": tgt, "estimate": "random, raw", "MAE": raw[0], "lo": raw[1], "hi": raw[2]})
        for lab, strata, fb in (("random std. to similarity", ["sim_coarse"], None),
                                ("random std. to logS", ["logS_bin"], None),
                                ("random std. to similarity x logS", ["sim_coarse", "logS_bin"], "sim_coarse")):
            r = ea.standardized_mae(src, t, strata, fallback=fb, n_boot=300)
            j = ea.standardization_joint(src, t, strata, fallback=fb, n_boot=nb)
            assert abs(j["MAE_standardized"] - r["MAE_standardized"]) < 1e-9
            std_rows.append({"target": tgt, "estimate": lab, "MAE": r["MAE_standardized"], "lo": r["lo"],
                             "hi": r["hi"], "weight_covered_by_cells": r["weight_covered_by_cells"],
                             **{k: v for k, v in j.items() if k != "MAE_standardized"}})
    std = pd.DataFrame(std_rows)
    std.loc[std["estimate"].isin(["observed", "random, raw"]), "frac_of_penalty_explained"] = np.nan

    ols = pd.concat([ea.adjusted_split_penalty(e, cov) for cov in ([], ["sim_coarse"], ["logS_bin"],
                                                                    ["sim_coarse", "logS_bin"])],
                    ignore_index=True)
    ctx.table(ols, "split_penalty_adjusted")
    # additive OLS residual as an alternative estimator of the fraction accounted for
    o_rows = []
    for tgt in std["target"].unique():
        o0 = ols[(ols["split_type"] == tgt) & (ols["adjusted_for"] == "none")]
        o2 = ols[(ols["split_type"] == tgt) & (ols["adjusted_for"] == "sim_coarse+logS_bin")]
        if o0.empty or o2.empty:
            continue
        p0, a = float(o0["penalty_vs_random"].iloc[0]), o2.iloc[0]
        o_rows.append({"target": tgt, "estimate": "OLS additive residual (sim + logS)",
                       "ols_raw_penalty": p0, "ols_residual_penalty": float(a["penalty_vs_random"]),
                       "ols_residual_lo": float(a["lo"]), "ols_residual_hi": float(a["hi"]),
                       "frac_of_penalty_explained": 1 - float(a["penalty_vs_random"]) / p0,
                       "frac_joint_ci_lo": 1 - float(a["hi"]) / p0, "frac_joint_ci_hi": 1 - float(a["lo"]) / p0})
    std = pd.concat([std, pd.DataFrame(o_rows)], ignore_index=True)
    std["note"] = ""
    std.loc[std["estimate"].str.startswith("random std."), "note"] = (
        "frac CI: joint cluster bootstrap (random AND target compounds resampled, weights recomputed); "
        "assumes within-stratum exchangeability of random and target compounds")
    std.loc[std["estimate"].str.startswith("OLS"), "note"] = (
        "1 - residual/raw OLS split penalty; CI from the residual's clustered CI only (raw penalty fixed)")
    ctx.table(std, "standardization")
    matched = matched_similarity_gap(e)
    ctx.table(matched, "matched_similarity_gap")
    _plot_label(ctx, by_label, std)

    test_mean = e.groupby("split_type")["y_true"].agg(["mean", "std"]).to_dict(orient="index")
    ctx.update_summary("label_shift", {
        "test_logS_mean_sd": test_mean,
        "frac_test_logS_lt_-4": e.assign(x=e["y_true"] < -4).groupby("split_type")["x"].mean().to_dict(),
        "MAE_by_logS_bin": {f"{r.split_type} | {r.logS_bin}": {"MAE": r.MAE, "n": r.n}
                            for r in by_label.itertuples()},
        "standardization": std.to_dict(orient="records"),
        "ols_split_penalty": ols.to_dict(orient="records"),
        "penalty_decomposition": penalty_decomposition(std, matched),
    })


def matched_similarity_gap(e: pd.DataFrame) -> pd.DataFrame:
    """MAE of scaffold/cluster vs random test compounds inside the same fine similarity bin (>= 0.5)."""
    rows = []
    for b in ea.sim_labels():
        if float(b.split("-")[0]) < 0.5:
            continue
        r = e[(e["split_type"] == "random") & (e["sim_bin"] == b)]
        for tgt in ("scaffold", "cluster"):
            t = e[(e["split_type"] == tgt) & (e["sim_bin"] == b)]
            if len(r) < 20 or len(t) < 20:
                continue
            rows.append({"sim_bin": b, "target": tgt, "MAE_target": t["abs_error"].mean(), "n_target": len(t),
                         "MAE_random": r["abs_error"].mean(), "n_random": len(r),
                         "gap": t["abs_error"].mean() - r["abs_error"].mean()})
    return pd.DataFrame(rows)


def penalty_decomposition(std: pd.DataFrame, matched: pd.DataFrame) -> dict:
    """Conservative wording of how much of the split penalty similarity + label shift account for."""
    out = {"assumption": "direct standardisation assumes that within a (similarity x logS) stratum random- and "
                         "target-split test compounds are exchangeable (same expected error); the matched-similarity "
                         "gaps below show this is only approximately true",
           "nn_similarity_reference": "NN similarity is measured to the TRAINING set only (validation excluded; the "
                                      "model is not fit on validation compounds)"}
    for tgt in std["target"].unique():
        d = std[std["target"] == tgt].set_index("estimate")
        if "random std. to similarity x logS" not in d.index:
            continue
        sx = d.loc["random std. to similarity x logS"]
        o = d.loc["OLS additive residual (sim + logS)"] if "OLS additive residual (sim + logS)" in d.index else None
        m = matched[matched["target"] == tgt] if len(matched) else matched
        out[tgt] = {
            "observed_MAE": d.loc["observed", "MAE"], "random_raw_MAE": d.loc["random, raw", "MAE"],
            "standardised_sim_x_logS": {"MAE": sx["MAE"], "frac_of_penalty": sx["frac_of_penalty_explained"],
                                        "frac_joint_95ci": [sx["frac_joint_ci_lo"], sx["frac_joint_ci_hi"]],
                                        "min_source_rows_per_stratum": sx["min_source_rows_per_stratum"],
                                        "frac_target_weight_fallback": sx["frac_target_weight_fallback"],
                                        "frac_target_weight_fallback_boot_mean":
                                            sx["frac_target_weight_fallback_boot_mean"]},
            "ols_additive": None if o is None else {
                "raw_penalty": o["ols_raw_penalty"], "residual_penalty": o["ols_residual_penalty"],
                "residual_95ci": [o["ols_residual_lo"], o["ols_residual_hi"]],
                "frac_of_penalty": o["frac_of_penalty_explained"]},
            "matched_similarity_gap_ge_0.5": {r.sim_bin: {"MAE_target": r.MAE_target, "MAE_random": r.MAE_random,
                                                          "n_target": r.n_target, "n_random": r.n_random}
                                              for r in m.itertuples()},
        }
    out["conclusion"] = ("similarity and label shift together account for most of the scaffold/cluster penalty - "
                         "roughly 2/3 (additive OLS) to all (direct standardisation, whose joint CI is wide) depending "
                         "on the estimator; a residual scaffold-specific penalty remains (OLS residual > 0, and at "
                         "matched NN similarity >= 0.5 scaffold MAE still exceeds random MAE). Not 'fully explained'.")
    return out


def _plot_label(ctx: Ctx, by_label: pd.DataFrame, std: pd.DataFrame) -> None:
    labels = ["< -6", "-6 to -4", "-4 to -2", "-2 to 0", ">= 0"]
    x = np.arange(len(labels))
    splits = [s for s in ("random", "scaffold", "cluster") if s in set(by_label["split_type"])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), gridspec_kw={"width_ratios": [1.1, 1, 1.25]})
    for k, st in enumerate(splits):
        d = by_label[by_label["split_type"] == st].set_index("logS_bin").reindex(labels)
        off = (k - 1) * 0.08
        yerr = np.vstack([d["MAE"] - d["MAE_lo"], d["MAE_hi"] - d["MAE"]])
        axes[0].errorbar(x + off, d["MAE"], yerr=yerr, fmt="-o", color=SPLIT_COLORS[st], ms=5, capsize=2.5,
                         label=st)
        axes[1].bar(x + (k - 1) * 0.27, d["frac_of_split"].fillna(0), width=0.27, color=SPLIT_COLORS[st], label=st)
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel(f"measured logS bin ({LOGS_UNIT})")
    axes[0].set_ylabel(f"MAE ({LOGS_UNIT})")
    axes[0].set_title("(a) error by true solubility")
    axes[0].legend(frameon=False)
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel(f"measured logS bin ({LOGS_UNIT})")
    axes[1].set_ylabel("fraction of test set")
    axes[1].set_title("(b) label distribution of test sets")
    axes[1].legend(frameon=False)
    # (c) decomposition
    ax = axes[2]
    if std.empty:
        ax.axis("off")
        fig.tight_layout()
        ctx.savefig(fig, "label_shift_vs_similarity")
        return
    order = ["observed", "random, raw", "random std. to similarity", "random std. to logS",
             "random std. to similarity x logS"]
    legend_txt = ["observed on target split", "random split, raw", "random, re-weighted to similarity bins",
                  "random, re-weighted to logS bins", "random, re-weighted to similarity x logS"]
    targets = list(std["target"].unique())
    w = 0.8 / len(order)
    for ti, tgt in enumerate(targets):
        d = std[std["target"] == tgt].set_index("estimate").reindex(order)
        for oi, est in enumerate(order):
            xx = ti + (oi - (len(order) - 1) / 2) * w
            col = SPLIT_COLORS[tgt] if est == "observed" else SPLIT_COLORS["random"]
            alpha = 1.0 if est in ("observed", "random, raw") else 0.5
            hatch = None if est in ("observed", "random, raw") else ["", "", "//", "..", "xx"][oi]
            ax.bar(xx, d.loc[est, "MAE"], width=w * 0.95, color=col, alpha=alpha, hatch=hatch, edgecolor="white",
                   label=legend_txt[oi] if ti == 0 and est != "observed" else None)
            ax.errorbar(xx, d.loc[est, "MAE"], yerr=[[d.loc[est, "MAE"] - d.loc[est, "lo"]],
                                                     [d.loc[est, "hi"] - d.loc[est, "MAE"]]],
                        color="0.2", capsize=2, lw=0.9)
            ax.text(xx, d.loc[est, "hi"] + 0.01, f"{d.loc[est, 'MAE']:.2f}", ha="center", fontsize=6.5)
    ax.bar([], [], color="0.5", label="observed on target split (split colour)")
    ax.set_xticks(range(len(targets)), [f"{t} test sets" for t in targets])
    ax.set_ylabel(f"MAE ({LOGS_UNIT})")
    ax.set_ylim(0, std["hi"].max() * 1.45)
    ax.legend(frameon=False, fontsize=7, loc="upper center", ncol=2)
    ax.set_title("(c) random-split error re-weighted to target strata\n(accounts for ~2/3 (OLS) to all (std.) "
                 "of the penalty; residual remains)", fontsize=9)
    fig.tight_layout()
    ctx.savefig(fig, "label_shift_vs_similarity")


# =========================================================================== step: chemistry
def step_chem(ctx: Ctx) -> None:
    e = ctx.errors()
    e = e[e["split_type"].isin(["random", "scaffold", "cluster"])]
    nb = ctx.args.n_boot
    phys = ea.physchem_frame("aqsoldb")
    pl = ea.physchem_long(e, phys)
    ptab = ea.error_summary(pl, ["property", "bin_order", "bin", "split_type"], n_boot=nb)
    ptab = ea.flag_bias(ptab)
    ctx.table(ptab, "physchem_bins")
    _plot_physchem(ctx, ptab)

    fg = ctx.functional_groups()
    fl = ea.functional_group_long(e, fg)
    ftab = ea.error_summary(fl, ["group", "split_type"], n_boot=nb)
    allrow = ea.error_summary(e.assign(group="ALL compounds"), ["group", "split_type"], n_boot=nb)
    ftab = ea.flag_bias(pd.concat([allrow, ftab], ignore_index=True))
    prevalence = fg.drop(columns="compound_id").mean().rename("prevalence_in_aqsoldb")
    ftab = ftab.merge(prevalence, left_on="group", right_index=True, how="left")
    overall = allrow.set_index("split_type")["MAE"]
    ftab["MAE_ratio_to_split"] = ftab["MAE"] / ftab["split_type"].map(overall)
    ftab["hard_class"] = (ftab["MAE_lo"] > ftab["split_type"].map(overall)) & (ftab["n_compounds"] >= 20)
    ctx.table(ftab, "functional_groups")
    _plot_fg(ctx, ftab)

    out = {"bias_rule": "n_compounds >= 20, 95% CI of mean signed error excludes 0, |mean signed error| >= 0.2",
           "hard_class_rule": "lower CI of class MAE above the split's overall MAE (n_compounds >= 20)"}
    for st in ("random", "scaffold", "cluster"):
        d = ftab[(ftab["split_type"] == st) & (ftab["group"] != "ALL compounds") & (ftab["n_compounds"] >= 20)]
        out[st] = {
            "worst_groups_by_MAE": [{"group": r.group, "MAE": r.MAE, "lo": r.MAE_lo, "hi": r.MAE_hi,
                                     "n_compounds": r.n_compounds, "mean_signed_error": r.mean_signed_error}
                                    for r in d.nlargest(6, "MAE").itertuples()],
            "best_groups_by_MAE": [{"group": r.group, "MAE": r.MAE, "n_compounds": r.n_compounds}
                                   for r in d.nsmallest(3, "MAE").itertuples()],
            "biased_groups": [{"group": r.group, "mean_signed_error": r.mean_signed_error, "lo": r.MSE_lo,
                               "hi": r.MSE_hi, "n_compounds": r.n_compounds} for r in d[d["biased"]].itertuples()],
            "hard_groups": list(d.loc[d["hard_class"], "group"]),
            "biased_physchem_bins": [{"property": r.property, "bin": r.bin, "mean_signed_error": r.mean_signed_error,
                                      "MAE": r.MAE, "n_compounds": r.n_compounds}
                                     for r in ptab[(ptab["split_type"] == st) & ptab["biased"]].itertuples()],
        }
    ctx.update_summary("chemistry", out)


def _plot_physchem(ctx: Ctx, ptab: pd.DataFrame) -> None:
    props = list(ea.PHYSCHEM_BINS)
    fig, axes = plt.subplots(2, 4, figsize=(16, 7.4), sharey=True)
    for ax, prop in zip(axes.ravel(), props):
        d = ptab[ptab["property"] == prop]
        bins = d.sort_values("bin_order")["bin"].drop_duplicates().tolist()
        x = np.arange(len(bins))
        for k, st in enumerate(("random", "scaffold", "cluster")):
            s = d[d["split_type"] == st].set_index("bin").reindex(bins)
            ok = s["n_compounds"].fillna(0) >= 10
            off = (k - 1) * 0.1
            yerr = np.vstack([s["MAE"] - s["MAE_lo"], s["MAE_hi"] - s["MAE"]])
            ax.errorbar((x + off)[ok], s["MAE"][ok], yerr=yerr[:, ok], fmt="-o", color=SPLIT_COLORS[st], ms=4,
                        capsize=2, lw=1.2, label=st)
        ntxt = d[d["split_type"] == "random"].set_index("bin").reindex(bins)["n_compounds"].fillna(0).astype(int)
        ax.set_xticks(x, [f"{b}\n(n={n})" for b, n in zip(bins, ntxt)], fontsize=7, rotation=0)
        ax.set_xlabel(ea.PHYSCHEM_BINS[prop][1], fontsize=9)
        ax.grid(axis="y", alpha=0.3)
    for ax in axes[:, 0]:
        ax.set_ylabel(f"MAE ({LOGS_UNIT})")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Test error by physicochemical property (seeds pooled; n = unique compounds in random test sets; "
                 "bins with < 10 compounds hidden)", fontsize=10.5)
    fig.tight_layout()
    ctx.savefig(fig, "error_by_physchem")


def _plot_fg(ctx: Ctx, ftab: pd.DataFrame) -> None:
    r = ftab[ftab["split_type"] == "random"].set_index("group")
    groups = r[r["n_compounds"] >= 10].sort_values("MAE").index.tolist()
    if "ALL compounds" in groups:
        groups.remove("ALL compounds")
    groups = ["ALL compounds"] + groups
    y = np.arange(len(groups))
    fig, axes = plt.subplots(1, 2, figsize=(13, 0.36 * len(groups) + 1.6), sharey=True)
    for k, st in enumerate(("random", "scaffold", "cluster")):
        d = ftab[ftab["split_type"] == st].set_index("group").reindex(groups)
        ok = d["n_compounds"].fillna(0) >= 10
        off = (k - 1) * 0.22
        for ax, col, lo, hi in ((axes[0], "MAE", "MAE_lo", "MAE_hi"),
                                (axes[1], "mean_signed_error", "MSE_lo", "MSE_hi")):
            xerr = np.vstack([d[col] - d[lo], d[hi] - d[col]])
            ax.errorbar(d[col][ok], (y + off)[ok], xerr=xerr[:, ok], fmt="o", color=SPLIT_COLORS[st], ms=4,
                        capsize=1.5, lw=1, label=st)
        biased = d["biased"].fillna(False).astype(bool) & ok
        axes[1].plot(d["mean_signed_error"][biased], (y + off)[biased], "o", mfc="none", mec="k", ms=8, mew=0.8)
    ylabels = [f"{g} (n={int(r.loc[g, 'n_compounds']) if g in r.index else 0})" for g in groups]
    axes[0].set_yticks(y, ylabels, fontsize=8)
    axes[0].set_xlabel(f"MAE ({LOGS_UNIT})")
    axes[0].set_title("(a) MAE by functional-group class")
    axes[0].legend(frameon=False, loc="lower right")
    axes[1].axvline(0, color="0.3", lw=0.8)
    axes[1].set_xlabel(f"mean signed error, pred - true ({LOGS_UNIT})")
    axes[1].set_title("(b) bias (circled: CI excludes 0 and |bias| >= 0.2, n >= 20)")
    for ax in axes:
        ax.grid(axis="x", alpha=0.3)
        ax.axhline(0.5, color="0.6", lw=0.6)
    fig.text(0.01, 0.0, "n = unique compounds in the pooled random-split test sets; classes with < 10 compounds hidden",
             fontsize=7.5, color="0.35")
    fig.tight_layout()
    ctx.savefig(fig, "error_by_functional_group")


# =========================================================================== step: worst predictions
def compound_meta(ctx: Ctx) -> pd.DataFrame:
    phys = ea.physchem_frame("aqsoldb")[["compound_id", "MolWt", "NumRotatableBonds", "NumAromaticRings"]]
    fg = ctx.functional_groups()
    return phys.merge(fg, on="compound_id")


# substrings of the rule texts produced by solpred.analysis.errors.hypothesis -> short tags
HYPOTHESIS_TAGS = {"chain-length homologue": "chain-length homologue NN",
                   "label disagrees with a close analogue": "close-analogue label conflict",
                   "conflicting replicate measurements": "conflicting replicates",
                   "out of domain": "out of domain", "salt (counter-ion": "salt",
                   "neutral multi-component": "multi-component (neutral)",
                   "hydrolytically reactive": "hydrolytically reactive", "ionisable": "ionisable",
                   "surfactant-like": "long alkyl chain", "extreme label": "extreme label",
                   "sparse chemistry": "Si/B", "large / flexible": "large/flexible",
                   "rigid polyaromatic": "polyaromatic too soluble", "no rule fired": "no rule"}


def hypothesis_tags(text: str) -> str:
    """Comma-separated short tags of a hypothesis string (see ``HYPOTHESIS_TAGS``)."""
    return ", ".join(v for k, v in HYPOTHESIS_TAGS.items() if k in str(text))


def _wrap(text: str, width: int) -> str:
    import textwrap

    return "\n".join(textwrap.wrap(text, width)[:2])


def step_worst(ctx: Ctx) -> None:
    e = ctx.errors()
    sel = e[e["split_type"].isin(["random", "scaffold"]) & (e["split_name"] == "seed0")]
    if sel.empty:
        print("  no random/scaffold seed-0 predictions - skipped", flush=True)
        return
    ranked = ea.worst_predictions(sel, compound_meta(ctx), k=len(sel))  # every pool compound, with hypotheses
    worst = ranked.head(30).copy()
    names = ctx.aq.set_index("compound_id")
    worst["nn_name"] = worst["nn_id"].map(names["name"])
    worst["nn_smiles"] = worst["nn_id"].map(names["smiles"])
    worst["hypothesis_tags"] = worst["hypothesis"].map(hypothesis_tags)
    cols = ["rank", "split_type", "compound_id", "name", "smiles", "y_true", "y_pred", "error", "nn_sim", "nn_id",
            "nn_name", "nn_logS", "reliability_group", "source", "n_occurrences", "aqsol_sd", "hypothesis_tags",
            "hypothesis"]
    ctx.table(worst[cols], "worst30")
    legends = [f"{r.rank}. {str(r.name)[:30]} [{r.split_type[:4]}]\ntrue {r.y_true:.2f} / pred {r.y_pred:.2f}"
               f"\nNN sim {r.nn_sim:.2f} (NN logS {r.nn_logS:.2f}), {r.reliability_group}"
               f"\n{_wrap(r.hypothesis_tags, 40)}"
               for r in worst.itertuples()]
    fig = mol_grid(worst["smiles"].tolist(), legends, ncols=6, cell=2.6,
                   title=f"30 largest absolute errors - {ctx.model_label()}, random + scaffold seed-0 test sets "
                         "(bottom line: rule-based hypotheses)")
    ctx.savefig(fig, "worst30_structures")
    counts = {v: int(worst["hypothesis"].str.contains(k, regex=False).sum()) for k, v in HYPOTHESIS_TAGS.items()}
    pool = sel.drop_duplicates("compound_id")
    ctx.update_summary("worst_predictions", {
        "definition": "pooled seed-0 test sets of the random (n={}) and scaffold (n={}) splits; compounds present "
                      "in both are counted once at their larger |error|; ranked by |y_pred - y_true|".format(
                          int((sel["split_type"] == "random").sum()), int((sel["split_type"] == "scaffold").sum())),
        "n_pool_rows": len(sel), "n_pool_compounds": int(pool["compound_id"].nunique()),
        "n_from_split": worst["split_type"].value_counts().to_dict(),
        "abs_error_range": [float(worst["error"].abs().min()), float(worst["error"].abs().max())],
        "n_over_predicted_solubility": int((worst["error"] > 0).sum()),
        "hypothesis_counts": counts,
        "hypothesis_base_rate_in_pool": {v: float(ranked["hypothesis"].str.contains(k, regex=False).mean())
                                         for k, v in HYPOTHESIS_TAGS.items()},
        "binomial_p_all_worst30_G1": float(stats.binomtest(
            int((worst["reliability_group"] == "G1").sum()), len(worst),
            float((ranked["reliability_group"] == "G1").mean()), alternative="greater").pvalue),
        "reliability_groups": worst["reliability_group"].value_counts().to_dict(),
        "pool_reliability_groups_frac": pool["reliability_group"].value_counts(normalize=True).to_dict(),
        "G1_enrichment": {
            "worst30_frac_G1": float((worst["reliability_group"] == "G1").mean()),
            "top100_frac_G1": float((ranked.head(100)["reliability_group"] == "G1").mean()),
            "pool_frac_G1": float((ranked["reliability_group"] == "G1").mean()),
            "binomial_p_top100_G1": float(stats.binomtest(
                int((ranked.head(100)["reliability_group"] == "G1").sum()), 100,
                float((ranked["reliability_group"] == "G1").mean()), alternative="greater").pvalue),
            "interpretation": "the largest errors are enriched for single-measurement (G1) compounds - consistent "
                              "with, but not proof of, label noise; G1 also marks unusual chemotypes",
        },
        "median_nn_sim_worst30": float(worst["nn_sim"].median()),
        "median_nn_sim_pool": float(sel["nn_sim"].median()),
    })


# =========================================================================== step: cliffs
def step_cliffs(ctx: Ctx) -> None:
    aq = ctx.aq.reset_index(drop=True)
    f = ctx.cache / "similar_pairs_ge0.9.csv"
    if f.exists():
        pairs = pd.read_csv(f)
    else:
        print("  all-pairs Tanimoto over AqSolDB ...", flush=True)
        pairs = cs.similar_pairs(morgan_bits(aq["smiles"].tolist()), sim_min=0.9)
        pairs.to_csv(f, index=False)
    ann = cs.annotate_pairs(pairs, aq)
    cliffs = ann[ann["abs_dlogS"] >= 2.0].sort_values("abs_dlogS", ascending=False).copy()
    cliffs["sim_count"] = cs.count_tanimoto(cliffs["smiles_1"].tolist(), cliffs["smiles_2"].tolist())
    cliffs["candidate_label_noise_or_cliff"] = cliffs["pair_type"].str.startswith("same-size")
    cliffs.to_csv(ctx.res / "cliff_pairs.csv", index=False)
    top = cliffs.head(25)[["sim", "compound_id_1", "name_1", "logS_1", "source_1", "reliability_group_1",
                           "compound_id_2", "name_2", "logS_2", "source_2", "reliability_group_2", "abs_dlogS",
                           "d_heavy_atoms", "sim_count", "pair_type", "candidate_label_noise_or_cliff"]]
    ctx.table(top, "cliff_pairs_top25")
    ctx.table(cliffs[cliffs["candidate_label_noise_or_cliff"]][
        ["sim", "sim_count", "compound_id_1", "name_1", "logS_1", "source_1", "reliability_group_1", "compound_id_2",
         "name_2", "logS_2", "source_2", "reliability_group_2", "abs_dlogS", "d_heavy_atoms"]], "cliff_pairs_same_size")
    floor = cs.nn_label_floor(pairs, aq["logS"].to_numpy(dtype=float), len(aq))

    # compare with the model in the >= 0.9 similarity bin of the random split
    e = ctx.errors()
    hi = e[(e["split_type"] == "random") & (e["nn_sim"] >= 0.9)]
    m_hi = ea.grouped_bootstrap_mean(hi["abs_error"], hi["compound_id"], ctx.args.n_boot) if len(hi) else (np.nan,) * 3
    nn_hi = (hi["y_true"] - hi["nn_logS"]).abs().mean() if len(hi) else np.nan
    in_cliff = set(cliffs["compound_id_1"]) | set(cliffs["compound_id_2"])
    wf = ctx.tab / f"{ctx.prefix}worst30.csv"
    n_worst_in_cliff = int(pd.read_csv(wf)["compound_id"].isin(in_cliff).sum()) if wf.exists() else None
    _plot_cliffs(ctx, ann, cliffs)
    same_size = cliffs["pair_type"].str.startswith("same-size")
    homol = cliffs["pair_type"].str.startswith("homologues")
    salt = cliffs["pair_type"].str.startswith("salt")
    ctx.update_summary("cliffs", {
        "definition": "AqSolDB pairs with ECFP4 Tanimoto >= 0.9 (2048 bits, full SMILES) and |dlogS| >= 2 "
                      "('high-similarity large-dlogS pairs'; NOT all activity cliffs / near-duplicates / label noise)",
        "interpretation": "most pairs are a binary-ECFP4 bit-vector saturation artefact: fingerprint-identical "
                          "(Tanimoto 1.0) pairs and homologues differing by >= 2 heavy atoms (e.g. cyclosiloxanes, "
                          "chain-length series) whose solubility legitimately differs. Only the same-size pairs "
                          "(|d heavy atoms| <= 1, different parent) are candidates for label noise or true cliffs.",
        "pair_breakdown": {
            "total": len(cliffs),
            "fingerprint_identical_tanimoto_1": int(cliffs["fp_identical"].sum()),
            "homologues_d_heavy_atoms_ge_2": int(homol.sum()),
            "fingerprint_identical_and_homologue": int((cliffs["fp_identical"] & homol).sum()),
            "salt_vs_parent": int(salt.sum()),
            "same_size_candidates_label_noise_or_true_cliff": int(same_size.sum()),
            "same_size_both_G1": int((same_size & cliffs["both_single_measurement"]).sum()),
            "median_count_tanimoto_all": float(cliffs["sim_count"].median()),
            "median_count_tanimoto_same_size": float(cliffs.loc[same_size, "sim_count"].median())
            if same_size.any() else None,
            "n_count_tanimoto_ge_0.9": int((cliffs["sim_count"] >= 0.9).sum()),
        },
        "n_pairs_sim_ge_0.9": len(ann), "n_compounds_in_sim_ge_0.9_pairs": int(len(set(ann["i"]) | set(ann["j"]))),
        "n_cliff_pairs": len(cliffs), "n_compounds_in_cliff_pairs": len(in_cliff),
        "frac_of_sim_ge_0.9_pairs_that_are_cliffs": len(cliffs) / max(len(ann), 1),
        "n_pairs_dlogS_ge_1": int((ann["abs_dlogS"] >= 1).sum()),
        "cliff_pair_types": cliffs["pair_type"].value_counts().to_dict(),
        "all_sim_ge_0.9_pair_types": ann["pair_type"].value_counts().to_dict(),
        "median_abs_dlogS_by_pair_type": ann.groupby("pair_type")["abs_dlogS"].median().to_dict(),
        "mean_abs_dlogS_by_pair_type": ann.groupby("pair_type")["abs_dlogS"].mean().to_dict(),
        "cliff_frac_by_pair_type": ann.groupby("pair_type")["abs_dlogS"].apply(lambda s: float((s >= 2).mean()))
                                      .to_dict(),
        "cliff_pairs_same_source": int(cliffs["same_source"].sum()),
        "cliff_pairs_both_single_measurement_G1": int(cliffs["both_single_measurement"].sum()),
        "cliff_pairs_fingerprint_identical": int(cliffs["fp_identical"].sum()),
        "median_d_heavy_atoms_cliff_pairs": float(cliffs["d_heavy_atoms"].median()),
        "same_size_cliff_pairs_both_G1": int((cliffs["pair_type"].str.startswith("same-size")
                                              & cliffs["both_single_measurement"]).sum()),
        "median_abs_dlogS_sim_ge_0.9_pairs": float(ann["abs_dlogS"].median()),
        "mean_abs_dlogS_sim_ge_0.9_pairs": float(ann["abs_dlogS"].mean()),
        "nn_label_floor_sim_ge_0.9": floor,
        "model_random_split_sim_ge_0.9": {"MAE": m_hi[0], "lo": m_hi[1], "hi": m_hi[2], "n": len(hi),
                                          "MAE_of_nn_label_predictor_same_rows": nn_hi},
        "n_worst30_compounds_in_cliff_pairs": n_worst_in_cliff,
    })


def _plot_cliffs(ctx: Ctx, ann: pd.DataFrame, cliffs: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bins = np.arange(0, ann["abs_dlogS"].max() + 0.25, 0.25)
    ax.hist(ann["abs_dlogS"], bins=bins, color="#4C72B0", alpha=0.85)
    ax.axvline(2.0, color="#C44E52", ls="--", lw=1.2)
    ax.set_yscale("log")
    ax.set_xlabel(f"|logS difference| within pair ({LOGS_UNIT})")
    ax.set_ylabel("number of pairs (log scale)")
    n_ss = int(cliffs["pair_type"].str.startswith("same-size").sum())
    ax.set_title(f"AqSolDB pairs with Tanimoto >= 0.9 (n={len(ann)}); {len(cliffs)} with |dlogS| >= 2\n"
                 f"({int(cliffs['fp_identical'].sum())} fingerprint-identical, mostly homologues; "
                 f"only {n_ss} same-size pairs)", fontsize=9)
    fig.tight_layout()
    ctx.savefig(fig, "cliff_pairs_histogram")
    top = distinct_pairs(cliffs, 8)
    smi, leg = [], []
    for r in top.itertuples():
        for k in (1, 2):
            smi.append(getattr(r, f"smiles_{k}"))
            leg.append(f"{str(getattr(r, f'name_{k}'))[:30]}\nlogS {getattr(r, f'logS_{k}'):.2f} "
                       f"(src {getattr(r, f'source_{k}')}, {getattr(r, f'reliability_group_{k}')})"
                       + (f"\npair sim {r.sim:.2f}, |dlogS| {r.abs_dlogS:.2f}" if k == 1
                          else f"\n{r.pair_type.split(' (')[0]}, d(heavy atoms) {int(r.d_heavy_atoms)}"))
    fig = mol_grid(smi, leg, ncols=4, title="Largest label differences among ECFP4 Tanimoto >= 0.9 pairs (mostly "
                                             "homologues / fingerprint saturation; pairs side by side)")
    ctx.savefig(fig, "cliff_pairs_structures")


def distinct_pairs(pairs: pd.DataFrame, k: int) -> pd.DataFrame:
    """First ``k`` rows of ``pairs`` (already sorted) in which neither compound appeared before."""
    seen: set[str] = set()
    keep = []
    for idx, a, b in zip(pairs.index, pairs["compound_id_1"], pairs["compound_id_2"]):
        if a in seen or b in seen:
            continue
        seen.update((a, b))
        keep.append(idx)
        if len(keep) == k:
            break
    return pairs.loc[keep]


# =========================================================================== step: scaffold-level
def step_scaffold(ctx: Ctx) -> None:
    e = ctx.errors()
    sc = e[e["split_type"] == "scaffold"].drop(columns=["scaffold", "name"])
    if sc.empty:
        print("  no scaffold-split predictions - skipped", flush=True)
        return
    tab = ea.scaffold_errors(sc, ctx.aq[["compound_id", "scaffold", "name"]], min_compounds=5,
                             n_boot=ctx.args.n_boot)
    tab.to_csv(ctx.res / "scaffold_errors_min5.csv", index=False)
    worst = tab.head(15).copy()
    worst.insert(0, "rank", np.arange(1, len(worst) + 1))
    ctx.table(worst[["rank", "scaffold", "n", "n_compounds", "n_seeds", "MAE", "MAE_lo", "MAE_hi",
                     "mean_signed_error", "mean_true", "sd_true", "mean_pred", "example_names"]], "scaffold_worst15")
    legends = [f"{r.rank}. MAE {r.MAE:.2f} (n={r.n}, {r.n_compounds} cpds)\nbias {r.mean_signed_error:+.2f}, "
               f"mean logS {r.mean_true:.2f}\ne.g. {str(r.example_names).split(';')[0][:28]}"
               for r in worst.itertuples()]
    fig = mol_grid(worst["scaffold"].tolist(), legends, ncols=5,
                   title="Bemis-Murcko scaffolds with the largest error in scaffold-split test sets "
                         "(>= 5 distinct test compounds, seeds pooled)")
    ctx.savefig(fig, "scaffold_worst15_structures")
    total_abs = sc["abs_error"].sum()
    worst_rows = sc.merge(ctx.aq[["compound_id", "scaffold"]], on="compound_id")
    share = worst_rows[worst_rows["scaffold"].isin(worst["scaffold"])]["abs_error"].sum() / total_abs
    ctx.update_summary("scaffold_level", {
        "n_scaffolds_with_ge5_compounds": len(tab), "n_rows_covered": int(tab["n"].sum()),
        "frac_scaffold_test_rows_covered": float(tab["n"].sum() / len(sc)),
        "median_scaffold_MAE": float(tab["MAE"].median()),
        "worst15": [{"scaffold": r.scaffold, "MAE": r.MAE, "n": r.n, "n_compounds": r.n_compounds,
                     "mean_signed_error": r.mean_signed_error, "mean_true": r.mean_true,
                     "example": str(r.example_names).split(";")[0]} for r in worst.itertuples()],
        "worst15_share_of_total_abs_error": float(share),
        "worst15_share_of_rows": float(worst["n"].sum() / len(sc)),
        "spearman_scaffold_MAE_vs_sd_true": float(stats.spearmanr(tab["MAE"], tab["sd_true"]).statistic),
    })


# =========================================================================== step: UMAP
def umap_coordinates(ctx: Ctx) -> pd.DataFrame:
    f = ctx.cache / "umap_coordinates_base.csv"
    if f.exists() and not ctx.args.force:
        return pd.read_csv(f, dtype={"compound_id": str, "aqsoldb_id": str}, keep_default_na=False,
                           na_values=[""])
    aq = ctx.aq
    es = load_dataset("esol")
    in_aq = es["in_aqsoldb"].astype(str).str.lower() == "true"
    es_only = es[~in_aq]
    smiles = aq["smiles"].tolist() + es_only["smiles"].tolist()
    print(f"  UMAP on {len(smiles)} compounds (AqSolDB + {len(es_only)} ESOL-only) ...", flush=True)
    X = cs.fingerprint_matrix(smiles)
    emb = cs.umap_embedding(X, n_neighbors=30, min_dist=0.1, random_state=0, metric="jaccard")
    base = pd.DataFrame({"compound_id": aq["compound_id"].tolist() + es_only["compound_id"].tolist(),
                         "dataset": ["aqsoldb"] * len(aq) + ["esol_only"] * len(es_only),
                         "umap1": emb[:, 0], "umap2": emb[:, 1],
                         "logS": np.concatenate([aq["logS"].to_numpy(float), es_only["logS"].to_numpy(float)])})
    pos = base.set_index("compound_id")
    ov = es[in_aq]
    esol_rows = pd.DataFrame({"compound_id": ov["compound_id"].to_numpy(), "dataset": "esol_overlap",
                              "umap1": pos.loc[ov["aqsoldb_id"], "umap1"].to_numpy(),
                              "umap2": pos.loc[ov["aqsoldb_id"], "umap2"].to_numpy(),
                              "logS": ov["logS"].to_numpy(float), "aqsoldb_id": ov["aqsoldb_id"].to_numpy()})
    base = pd.concat([base, esol_rows], ignore_index=True)
    base.to_csv(f, index=False)
    return base


def step_umap(ctx: Ctx) -> None:
    co = umap_coordinates(ctx)
    for st in ("scaffold", "cluster"):
        sp = load_split(st, 0)
        lab = {c: k for k in ("train", "val", "test") for c in sp[k]}
        co[f"{st}_seed0_set"] = co["compound_id"].map(lab)
    e = ctx.errors()
    sc = e[e["split_type"] == "scaffold"]
    co["scaffold_mean_abs_error"] = co["compound_id"].map(sc.groupby("compound_id")["abs_error"].mean())
    co["scaffold_n_seeds_in_test"] = co["compound_id"].map(sc.groupby("compound_id").size())
    s0 = e[(e["split_type"] == "scaffold") & (e["split_name"] == "seed0")].set_index("compound_id")
    co["scaffold_seed0_nn_sim"] = co["compound_id"].map(s0["nn_sim"])
    co.to_csv(ctx.res / "umap_coordinates.csv", index=False)
    _plot_umap(ctx, co)
    aqm = co["dataset"] == "aqsoldb"
    ctx.update_summary("umap", {
        "params": {"metric": "jaccard", "fingerprint": "ECFP4 bits 2048", "n_neighbors": 30, "min_dist": 0.1,
                   "random_state": 0},
        "n_aqsoldb": int(aqm.sum()), "n_esol_overlap": int((co["dataset"] == "esol_overlap").sum()),
        "n_esol_only": int((co["dataset"] == "esol_only").sum()),
        "n_aqsoldb_ever_in_scaffold_test": int(co.loc[aqm, "scaffold_mean_abs_error"].notna().sum()),
        "acyclic_vs_cyclic": acyclic_regions(ctx, co[aqm]),
        "coordinates_file": (ctx.res / "umap_coordinates.csv").relative_to(ROOT).as_posix(),
    })


def acyclic_regions(ctx: Ctx, co: pd.DataFrame) -> dict:
    """Where do acyclic compounds (empty Murcko scaffold; never in scaffold test sets) sit in the map?

    Compares the lower 40 % of UMAP-2 (the acyclic-rich lobe of the reference embedding) with the rest.
    """
    d = co.merge(ctx.aq[["compound_id", "scaffold"]], on="compound_id", how="left")
    acyc = d["scaffold"].isna() | (d["scaffold"].astype(str) == "")
    lower = d["umap2"] < np.percentile(d["umap2"], 40)
    out = {"n_acyclic": int(acyc.sum()), "frac_acyclic": float(acyc.mean()),
           "region_rule": "lower = UMAP-2 below its 40th percentile"}
    for name, m in (("lower", lower), ("upper", ~lower)):
        out[name] = {"n": int(m.sum()), "frac_acyclic": float(acyc[m].mean()),
                     "mean_logS": float(d.loc[m, "logS"].mean()),
                     "frac_scaffold_seed0_test": float((d.loc[m, "scaffold_seed0_set"] == "test").mean()),
                     "frac_cluster_seed0_test": float((d.loc[m, "cluster_seed0_set"] == "test").mean())}
    return out


def _plot_umap(ctx: Ctx, co: pd.DataFrame) -> None:
    aq = co[co["dataset"] == "aqsoldb"]
    es = co[co["dataset"] != "aqsoldb"]
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    kw = dict(s=2.0, linewidths=0, rasterized=True)
    ax = axes[0, 0]
    ax.scatter(aq["umap1"], aq["umap2"], c="#C8C8C8", label=f"AqSolDB (n={len(aq)})", **kw)
    ax.scatter(es["umap1"], es["umap2"], c=SPLIT_COLORS["external"], s=3.5, linewidths=0, rasterized=True,
               label=f"ESOL (n={len(es)})")
    ax.legend(frameon=False, markerscale=4, loc="best")
    ax.set_title("(a) AqSolDB with ESOL overlaid")
    ax = axes[0, 1]
    sc = ax.scatter(aq["umap1"], aq["umap2"], c=aq["logS"], cmap="viridis", vmin=-9, vmax=1.5, **kw)
    fig.colorbar(sc, ax=ax, label=f"logS ({LOGS_UNIT})", shrink=0.85)
    ax.set_title("(b) measured solubility")
    for ax, st, title in ((axes[0, 2], "scaffold", "(c) scaffold split, seed 0"),
                          (axes[1, 0], "cluster", "(d) cluster split, seed 0")):
        colors = {"train": "#D3D3D3", "val": "#7F7F7F", "test": SPLIT_COLORS[st]}
        for k in ("train", "val", "test"):
            d = aq[aq[f"{st}_seed0_set"] == k]
            ax.scatter(d["umap1"], d["umap2"], c=colors[k], label=f"{k} (n={len(d)})",
                       **{**kw, "s": 2.0 if k == "train" else 3.0})
        ax.legend(frameon=False, markerscale=4)
        ax.set_title(title)
    ax = axes[1, 1]
    have = aq["scaffold_mean_abs_error"].notna()
    ax.scatter(aq.loc[~have, "umap1"], aq.loc[~have, "umap2"], c="#E3E3E3", **kw)
    d = aq[have].sort_values("scaffold_mean_abs_error")
    sc = ax.scatter(d["umap1"], d["umap2"], c=d["scaffold_mean_abs_error"], cmap="magma_r", vmin=0, vmax=2.5,
                    **{**kw, "s": 3.0})
    fig.colorbar(sc, ax=ax, label=f"|error| ({LOGS_UNIT})", shrink=0.85)
    ax.set_title(f"(e) |error| in scaffold test sets (mean over seeds; n={have.sum()})")
    ax = axes[1, 2]
    have = aq["scaffold_seed0_nn_sim"].notna()
    ax.scatter(aq.loc[~have, "umap1"], aq.loc[~have, "umap2"], c="#E3E3E3", **kw)
    d = aq[have]
    sc = ax.scatter(d["umap1"], d["umap2"], c=d["scaffold_seed0_nn_sim"], cmap="cividis", vmin=0.2, vmax=1.0,
                    **{**kw, "s": 3.0})
    fig.colorbar(sc, ax=ax, label="max Tanimoto to train (ECFP4)", shrink=0.85)
    ax.set_title("(f) scaffold seed-0 test: similarity to training set")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
    fig.suptitle("Chemical space of AqSolDB (UMAP, Jaccard distance on ECFP4 bits; n_neighbors=30, min_dist=0.1)",
                 fontsize=12)
    fig.tight_layout()
    ctx.savefig(fig, "umap_chemical_space")


# =========================================================================== step: SHAP
def _fit_lgbm(featurizer: str, n_jobs: int):
    from solpred.experiment import run_split

    split = load_split("random", 0)
    row, preds, model = run_split(split, featurizer, "lgbm", seed=0, n_jobs=n_jobs)
    X, ids, names = load_or_compute("aqsoldb", featurizer)
    Xte = subset(X, ids, list(map(str, split["test"])))
    return row, model, Xte, names


def shap_values_cached(ctx: Ctx, featurizer: str) -> dict:
    f = ctx.cache / f"shap_lgbm__{featurizer}__random_seed0.npz"
    if f.exists() and not ctx.args.force:
        z = np.load(f, allow_pickle=False)
        return {k: z[k] for k in z.files}
    print(f"  fitting LightGBM on {featurizer} (random seed 0, n_jobs={ctx.args.n_jobs}) ...", flush=True)
    row, model, Xte, names = _fit_lgbm(featurizer, ctx.args.n_jobs)
    if featurizer == "rdkit2d":
        import shap

        expl = shap.TreeExplainer(model)
        sv = np.asarray(expl.shap_values(Xte))
        base = float(np.ravel(expl.expected_value)[0])
    else:  # identical TreeSHAP algorithm, LightGBM's native multi-threaded implementation
        contrib = model.predict(Xte, pred_contrib=True)
        sv, base = contrib[:, :-1], float(contrib[0, -1])
    out = {"shap": sv.astype(np.float32), "X": Xte.astype(np.float32), "names": np.asarray(names),
           "base": np.asarray(base), "test_MAE": np.asarray(row["MAE"]), "val_MAE": np.asarray(row["val_MAE"]),
           "n_trees": np.asarray(model.booster_.num_trees())}
    np.savez_compressed(f, **out)
    return out


def lipophilicity_response(ctx: Ctx, r2d: dict, names: list[str], clogp: np.ndarray, win: np.ndarray) -> dict:
    """How strongly does the rdkit2d model's prediction respond to cLogP (cf. GSE slope of -1)?

    Slopes (log units per cLogP unit, OLS on test compounds with 0 <= cLogP <= 6) of: the MolLogP SHAP
    value alone, the summed SHAP of the whole lipophilicity family (correlated descriptors share the
    credit), the full prediction, and the measured logS. The mean MolLogP SHAP outside the training-dense
    range shows the saturation of tree models (no extrapolation beyond the last split threshold).
    """
    sv = r2d["shap"]
    fam = [k for k, n in enumerate(names) if cs.feature_family(n) == "lipophilicity"]
    pred = float(r2d["base"]) + sv.sum(1)
    te_ids = list(map(str, load_split("random", 0)["test"]))
    y = ctx.aq.set_index("compound_id").loc[te_ids, "logS"].to_numpy(float)
    j = names.index("MolLogP")

    def slope(v: np.ndarray) -> float:
        return float(np.polyfit(clogp[win], v[win], 1)[0])

    hi, lo = clogp > 7, clogp < 0
    return {"window": "0 <= cLogP <= 6", "n_in_window": int(win.sum()),
            "slope_MolLogP_shap": slope(sv[:, j]),
            "slope_lipophilicity_family_shap_sum": slope(sv[:, fam].sum(1)),
            "n_lipophilicity_features": len(fam),
            "slope_predicted_logS": slope(pred), "slope_measured_logS": slope(y),
            "MAE_of_base_plus_shap_sum": float(np.abs(pred - y).mean()),  # additivity check vs test_MAE
            "MolLogP_shap_mean_cLogP_gt_7": float(sv[hi, j].mean()) if hi.any() else None,
            "n_cLogP_gt_7": int(hi.sum()),
            "MolLogP_shap_mean_cLogP_lt_0": float(sv[lo, j].mean()) if lo.any() else None,
            "n_cLogP_lt_0": int(lo.sum()),
            "gse_reference": "general solubility equation logS = 0.5 - logP - 0.01 (mp - 25): slope -1"}


def step_shap(ctx: Ctx) -> None:
    import shap

    r2d = shap_values_cached(ctx, "rdkit2d")
    names = [str(n) for n in r2d["names"]]
    imp = cs.shap_importance(r2d["shap"], names)
    X = r2d["X"].astype(float)
    rho = []
    for feat in imp["feature"]:
        j = names.index(feat)
        x, s = X[:, j], r2d["shap"][:, j]
        ok = np.isfinite(x)
        rho.append(stats.spearmanr(x[ok], s[ok]).statistic if ok.sum() > 10 and np.nanstd(x[ok]) > 0 else np.nan)
    imp["spearman_value_vs_shap"] = rho
    ctx.table(imp.head(30), "shap_top30_rdkit2d")
    fam2d = cs.family_importance(imp)

    # beeswarm + bar
    expl = shap.Explanation(values=r2d["shap"], base_values=np.full(len(X), float(r2d["base"])), data=X,
                            feature_names=names)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    shap.plots.beeswarm(expl, max_display=20, show=False, ax=ax, plot_size=None, group_remaining_features=False,
                        color_bar_label="descriptor value (low to high)")
    ax.set_xlabel(f"SHAP value: contribution to predicted logS ({LOGS_UNIT})")
    ax.set_title(f"LightGBM on RDKit 2D descriptors (random seed 0, test MAE {float(r2d['test_MAE']):.3f})")
    ctx.savefig(fig, "shap_beeswarm_rdkit2d")
    fam_colors = {f: plt.cm.tab10(i) for i, f in enumerate([f for f, _ in cs.FAMILY_RULES] + ["other"])}
    top = imp.head(20).iloc[::-1]
    fig, ax = plt.subplots(figsize=(6.8, 6.2))
    ax.barh(top["feature"], top["mean_abs_shap"], color=[fam_colors[f] for f in top["family"]])
    for fam in pd.unique(top["family"][::-1]):
        ax.barh([], [], color=fam_colors[fam], label=fam)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_xlabel(f"mean |SHAP| ({LOGS_UNIT})")
    ax.set_title("Top-20 descriptors by mean |SHAP| (rdkit2d LightGBM)")
    ctx.savefig(fig, "shap_bar_rdkit2d")

    # dependence plot for MolLogP
    j = names.index("MolLogP")
    inter = shap.utils.approximate_interactions(j, r2d["shap"], X)[0]
    xs, ys, cc = X[:, j], r2d["shap"][:, j], X[:, inter]
    win = (xs >= 0) & (xs <= 6)
    slope, icpt = np.polyfit(xs[win], ys[win], 1)
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    vmin, vmax = np.nanpercentile(cc, [5, 95])
    sc = ax.scatter(xs, ys, c=cc, cmap="coolwarm", s=10, vmin=vmin, vmax=vmax, edgecolors="none")
    fig.colorbar(sc, ax=ax, label=names[inter])
    xx = np.linspace(0, 6, 10)
    ax.plot(xx, slope * xx + icpt, "k--", lw=1.1, label=f"linear fit on 0 <= cLogP <= 6: slope {slope:.2f}")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlabel("Crippen cLogP (MolLogP)")
    ax.set_ylabel(f"SHAP value of MolLogP ({LOGS_UNIT})")
    ax.set_title("Dependence of predicted logS on lipophilicity")
    ax.legend(frameon=False)
    ctx.savefig(fig, "shap_dependence_MolLogP")
    lipo = lipophilicity_response(ctx, r2d, names, xs, win)

    # combined model: descriptors vs fingerprint bits
    comb = shap_values_cached(ctx, "rdkit2d+morgan_counts")
    cnames = [str(n) for n in comb["names"]]
    cimp = cs.shap_importance(comb["shap"], cnames)
    is_fp = cimp["feature"].str.match(r"^m[cb]\d+$")
    group = {"RDKit descriptors": float(cimp.loc[~is_fp, "mean_abs_shap"].sum()),
             "Morgan count bits": float(cimp.loc[is_fp, "mean_abs_shap"].sum())}
    tot = sum(group.values())
    famc = cs.family_importance(cimp)
    aq_smiles = ctx.aq["smiles"].tolist()
    topbits = cimp[is_fp].head(10).copy()
    envs = [cs.morgan_bit_environment(int(b[2:]), aq_smiles, max_mols=len(aq_smiles)) for b in topbits["feature"]]
    topbits["example_environment"] = [e_[0] for e_ in envs]
    topbits["n_compounds_with_env"] = [e_[1] for e_ in envs]
    cimp_top = cimp.head(30).merge(topbits[["feature", "example_environment"]], on="feature", how="left")
    ctx.table(cimp_top, "shap_top30_combined")
    fam_tab = fam2d.assign(model="lgbm__rdkit2d").merge(
        famc[["family", "mean_abs_shap", "fraction"]].rename(columns={"mean_abs_shap": "mean_abs_shap_combined",
                                                                      "fraction": "fraction_combined"}),
        on="family", how="outer")
    ctx.table(fam_tab.drop(columns="model"), "shap_family_importance")
    _plot_groups(ctx, fam2d, famc, group)

    base_file = ea.prediction_file(ctx.pred_root, "random", "seed0", "lgbm__rdkit2d+morgan_counts")
    base_mae = None
    if base_file.exists():
        b = pd.read_csv(base_file)
        b = b[b["subset"] == "test"]
        base_mae = float((b["y_pred"] - b["y_true"]).abs().mean())
    ctx.update_summary("shap", {
        "rdkit2d_model": {"test_MAE": float(r2d["test_MAE"]), "val_MAE": float(r2d["val_MAE"]),
                          "n_trees": int(r2d["n_trees"]), "n_test": int(len(X)),
                          "top10": [{"feature": r.feature, "mean_abs_shap": r.mean_abs_shap, "family": r.family,
                                     "spearman_value_vs_shap": r.spearman_value_vs_shap}
                                    for r in imp.head(10).itertuples()],
                          "family_fraction": dict(zip(fam2d["family"], fam2d["fraction"])),
                          "MolLogP_shap_slope_0_to_6": float(slope),
                          "MolLogP_top_interaction": names[inter],
                          "lipophilicity_response": lipo},
        "combined_model": {"test_MAE_refit": float(comb["test_MAE"]), "test_MAE_baseline_file": base_mae,
                           "n_trees": int(comb["n_trees"]),
                           "group_fraction": {k: v / tot for k, v in group.items()},
                           "family_fraction": dict(zip(famc["family"], famc["fraction"])),
                           "top10": [{"feature": r.feature, "mean_abs_shap": r.mean_abs_shap}
                                     for r in cimp.head(10).itertuples()],
                           "top_bits": [{"bit": r.feature, "env": r.example_environment,
                                         "mean_abs_shap": r.mean_abs_shap} for r in topbits.itertuples()]},
    })


def _plot_groups(ctx: Ctx, fam2d: pd.DataFrame, famc: pd.DataFrame, group: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.0), gridspec_kw={"width_ratios": [1.2, 1.2, 0.7]})
    for ax, fam, title in ((axes[0], fam2d, "(a) LightGBM rdkit2d: share of mean |SHAP|"),
                           (axes[1], famc, "(b) LightGBM rdkit2d+Morgan: share of mean |SHAP|")):
        d = fam.iloc[::-1]
        ax.barh(d["family"], d["fraction"], color="#4C72B0")
        for y_, (v, n) in enumerate(zip(d["fraction"], d["n_features"])):
            ax.text(v + 0.005, y_, f"{v:.0%} ({n} feats)", va="center", fontsize=7.5)
        ax.set_xlabel("fraction of total mean |SHAP|")
        ax.set_title(title, fontsize=10)
        ax.set_xlim(0, max(d["fraction"]) * 1.35)
    tot = sum(group.values())
    axes[2].bar([k.replace(" ", "\n", 1) for k in group], [v / tot for v in group.values()],
                color=["#4C72B0", "#8172B3"])
    for i, v in enumerate(group.values()):
        axes[2].text(i, v / tot + 0.01, f"{v / tot:.0%}", ha="center")
    axes[2].set_ylabel("fraction of total mean |SHAP|")
    axes[2].set_title("(c) combined model: descriptors vs bits", fontsize=10)
    axes[2].set_ylim(0, 1.05)
    fig.tight_layout()
    ctx.savefig(fig, "shap_group_importance")


# =========================================================================== main
STEPS = {"sim": step_sim, "label": step_label, "chem": step_chem, "worst": step_worst, "cliffs": step_cliffs,
         "scaffold": step_scaffold, "umap": step_umap, "shap": step_shap}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred-root", default="results/baseline", help="experiment dir containing predictions/")
    ap.add_argument("--model-tag", default="lgbm__rdkit2d+morgan_counts", help="<model>__<featurizer>")
    ap.add_argument("--secondary-model-tag", default="rf__rdkit2d",
                    help="overlaid (dashed) in the similarity figure; '' to disable")
    ap.add_argument("--compare-model-tags", nargs="*",
                    default=["esol_linear__esol4", "rf__rdkit2d", "lgbm__rdkit2d+morgan_counts"],
                    help="models for the supplementary MAE-vs-similarity comparison")
    ap.add_argument("--tag", "--out-subdir", dest="out_subdir", default="",
                    help="output tag: sub-directory of results/figures and table prefix error_<tag>_ "
                         "(empty = reference outputs)")
    ap.add_argument("--split-types", nargs="+", default=["random", "scaffold", "cluster"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--steps", nargs="+", default=list(STEPS), choices=list(STEPS))
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--n-jobs", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="recompute cached UMAP / SHAP")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ctx = Ctx(args)
    for name in args.steps:
        print(f"[{name}]", flush=True)
        STEPS[name](ctx)
    print("done ->", ctx.res / "summary.json")
