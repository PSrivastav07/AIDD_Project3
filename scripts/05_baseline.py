"""Step 5 (Phase 1) - reference baselines on every random / scaffold / cluster split (5 seeds).

Models: training-mean, ESOL equation refit (esol4 + OLS), random forest on RDKit descriptors
(with across-tree std) and LightGBM on descriptors + Morgan counts. Outputs go to results/baseline/
and are the *reference predictions* re-used by the uncertainty, heterogeneity and error analyses.
Re-running skips runs whose prediction file already exists (use --force to redo).
"""
import argparse
import shutil

import pandas as pd

from solpred.config import load_config, path
from solpred.experiment import run_exists, run_split, save_run, summarize
from solpred.splits import load_split

CONFIGS = [("mean", "esol4"), ("esol_linear", "esol4"), ("rf", "rdkit2d"), ("lgbm", "rdkit2d+morgan_counts")]

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--split-types", nargs="+", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    out = path("results", cfg) / "baseline"
    if args.force and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    for stype in args.split_types or cfg["splits"]["types"]:
        for seed in cfg["splits"]["seeds"]:
            split = load_split(stype, seed)
            for model, feat in CONFIGS:
                if run_exists(out, stype, f"seed{seed}", model, feat):
                    continue
                row, preds, _ = run_split(split, feat, model, seed=seed, n_jobs=args.n_jobs)
                save_run(row, preds, out, "baseline")
                print(f"{stype:9s} seed{seed} {model:12s} {feat:24s} MAE={row['MAE']:.3f} "
                      f"RMSE={row['RMSE']:.3f} R2={row['R2']:.3f} ({row['fit_seconds']:.0f}s)", flush=True)
    m = pd.read_csv(out / "metrics_runs.csv")
    s = summarize(m)
    s.to_csv(out / "summary.csv", index=False)
    print(s.round(3).to_string())
