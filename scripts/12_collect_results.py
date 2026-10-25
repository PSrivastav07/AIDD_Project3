"""Step 12 - merge the classical benchmark and the GNN runs into the headline results tables.

Reads ``results/benchmark/metrics_runs.csv`` and ``results/gnn/metrics_runs.csv`` (same columns, see
``solpred.experiment``) and writes

* ``results/tables/main_results.{csv,md}``       MAE / RMSE / R2 (mean +- sd over seeds), random | scaffold | cluster
* ``results/tables/main_results_full.{csv,md}``  all five metrics, long format, one row per config x split
* ``results/tables/featurization_comparison.{csv,md}``  best learner per featurisation family and split
* ``results/figures/main/mae_by_family.png``     grouped bars of the best config per family
* ``results/main_summary.json``                  headline numbers cited by the report
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from solpred.config import ROOT, load_config, path, save_json

SPLITS = ["random", "scaffold", "cluster"]
METRICS = ["MAE", "RMSE", "R2", "Pearson", "Spearman"]
COLORS = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868"}
FAMILY = {
    "esol4": "ESOL descriptors (4)",
    "rdkit2d": "RDKit 2D descriptors",
    "maccs": "MACCS keys",
    "morgan_counts": "Morgan fingerprint",
    "morgan_bits": "Morgan fingerprint",
    "rdkit2d+morgan_counts": "Descriptors + Morgan",
    "graph": "Learned graph (D-MPNN)",
    "graph+rdkit2d": "Graph + descriptors (D-MPNN)",
}
FAMILY_ORDER = list(dict.fromkeys(FAMILY.values()))


def load_runs(results: str) -> pd.DataFrame:
    frames = []
    for exp in ("benchmark", "gnn"):
        f = ROOT / results / exp / "metrics_runs.csv"
        if f.exists():
            frames.append(pd.read_csv(f))
    runs = pd.concat(frames, ignore_index=True)
    runs = runs[runs["split_type"].isin(SPLITS)]
    # keep the last record of a (config, split, seed) in case a run was repeated
    runs = runs.drop_duplicates(["model", "featurizer", "split_type", "seed"], keep="last")
    runs["config"] = runs["model"] + "__" + runs["featurizer"]
    runs["family"] = runs["featurizer"].map(FAMILY)
    return runs


def summarise(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby(["config", "model", "featurizer", "family", "split_type"])
    out = g[METRICS].mean().add_suffix("_mean").join(g[METRICS].std().add_suffix("_std"))
    return out.join(g.size().rename("n_seeds")).reset_index()


def pm(m: float, s: float) -> str:
    return f"{m:.3f} ± {s:.3f}" if np.isfinite(s) else f"{m:.3f}"


def wide(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cfg, grp in summary.groupby("config"):
        row = {"model": grp["model"].iat[0], "featurizer": grp["featurizer"].iat[0],
               "n_seeds": int(grp["n_seeds"].min())}
        for sp in SPLITS:
            r = grp[grp["split_type"] == sp]
            for m in ("MAE", "RMSE", "R2"):
                row[f"{sp} {m}"] = pm(r[f"{m}_mean"].iat[0], r[f"{m}_std"].iat[0]) if len(r) else ""
            row[f"_{sp}_mae"] = r["MAE_mean"].iat[0] if len(r) else np.nan
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("_scaffold_mae")
    return df.drop(columns=[c for c in df.columns if c.startswith("_")])


def family_best(summary: pd.DataFrame) -> pd.DataFrame:
    """Best config (lowest mean MAE) of each featurisation family, selected per split."""
    rows = []
    for (fam, sp), grp in summary.groupby(["family", "split_type"]):
        b = grp.sort_values("MAE_mean").iloc[0]
        rows.append({"family": fam, "split_type": sp, "best_config": b["config"], "n_seeds": int(b["n_seeds"]),
                     **{f"{m}_mean": b[f"{m}_mean"] for m in METRICS},
                     **{f"{m}_std": b[f"{m}_std"] for m in METRICS}})
    df = pd.DataFrame(rows)
    df["family"] = pd.Categorical(df["family"], FAMILY_ORDER, ordered=True)
    df["split_type"] = pd.Categorical(df["split_type"], SPLITS, ordered=True)
    return df.sort_values(["family", "split_type"]).reset_index(drop=True)


def plot_family(fb: pd.DataFrame, file) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fams = [f for f in FAMILY_ORDER if f in set(fb["family"].astype(str))]
    x = np.arange(len(fams))
    w = 0.27
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    for i, sp in enumerate(SPLITS):
        d = fb[fb["split_type"] == sp].set_index(fb[fb["split_type"] == sp]["family"].astype(str)).reindex(fams)
        ax.bar(x + (i - 1) * w, d["MAE_mean"], w, yerr=d["MAE_std"], capsize=3, color=COLORS[sp], label=sp)
    ax.set_xticks(x)
    ax.set_xticklabels([f.replace(" (", "\n(") for f in fams], fontsize=8.5)
    ax.set_ylabel("Test MAE (log mol/L), mean ± sd over seeds")
    ax.set_title("Best learner per featurisation family: random vs scaffold vs cluster split")
    ax.legend(title="split")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(file, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    tables = path("tables", cfg)
    figdir = path("figures", cfg) / "main"
    figdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(cfg["paths"]["results"])
    summary = summarise(runs)
    w = wide(summary)
    w.to_csv(tables / "main_results.csv", index=False)
    (tables / "main_results.md").write_text(
        "# Main results: test-set metrics, mean ± sd over seeds (sorted by scaffold MAE)\n\n"
        + w.to_markdown(index=False) + "\n", encoding="utf-8")
    full = summary.copy()
    full["split_type"] = pd.Categorical(full["split_type"], SPLITS, ordered=True)
    full = full.sort_values(["config", "split_type"])
    full.to_csv(tables / "main_results_full.csv", index=False)
    show = full[["config", "split_type", "n_seeds"]].copy()
    for m in METRICS:
        show[m] = [pm(a, b) for a, b in zip(full[f"{m}_mean"], full[f"{m}_std"])]
    (tables / "main_results_full.md").write_text(
        "# All configurations, all metrics (mean ± sd over seeds)\n\n" + show.to_markdown(index=False) + "\n",
        encoding="utf-8")

    fb = family_best(summary)
    fb.to_csv(tables / "featurization_comparison.csv", index=False)
    fshow = fb[["family", "split_type", "best_config", "n_seeds"]].copy()
    for m in METRICS:
        fshow[m] = [pm(a, b) for a, b in zip(fb[f"{m}_mean"], fb[f"{m}_std"])]
    (tables / "featurization_comparison.md").write_text(
        "# Featurisation comparison: best learner per family and split (mean ± sd over seeds)\n\n"
        + fshow.to_markdown(index=False) + "\n", encoding="utf-8")
    plot_family(fb, figdir / "mae_by_family.png")

    best = {}
    for sp in SPLITS:
        s = summary[summary["split_type"] == sp].sort_values("MAE_mean")
        b = s.iloc[0]
        best[sp] = {"config": b["config"], "n_seeds": int(b["n_seeds"]),
                    **{m: {"mean": round(float(b[f"{m}_mean"]), 4), "std": round(float(b[f"{m}_std"]), 4)}
                       for m in METRICS},
                    "top5": s["config"].head(5).tolist()}
    gaps = {}
    for c, grp in summary.groupby("config"):
        mae = grp.set_index("split_type")["MAE_mean"]
        if {"random", "scaffold", "cluster"} <= set(mae.index):
            gaps[c] = {"scaffold_minus_random": round(float(mae["scaffold"] - mae["random"]), 4),
                       "cluster_minus_random": round(float(mae["cluster"] - mae["random"]), 4)}
    save_json({"source_files": ["results/benchmark/metrics_runs.csv", "results/gnn/metrics_runs.csv"],
               "n_configs": int(summary["config"].nunique()), "best_per_split": best,
               "mae_gaps": gaps}, ROOT / cfg["paths"]["results"] / "main_summary.json")
    print(w.to_string(index=False))


if __name__ == "__main__":
    main()
