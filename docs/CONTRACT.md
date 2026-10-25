# Engineering contract (shared conventions for all project code)

This file fixes the interfaces, file formats and conventions every module and script must follow.

## Environment
* Interpreter: `.venv/Scripts/python.exe` (Windows, Python 3.13). Never use bare `python` (it resolves to
  Inkscape's interpreter). The `solpred` package is installed in editable mode (`pip install -e .`),
  so `import solpred` works from any script.
* Installed: rdkit 2026.09, scikit-learn 1.9, numpy 2.5, pandas 3.0, scipy, matplotlib, seaborn,
  lightgbm, xgboost, torch 2.14 (CPU), umap-learn, shap, fastapi, uvicorn, streamlit, pytest, httpx, statsmodels.
  If you need another package, `pip install` it into `.venv` and add it to `requirements.txt` (pinned).
* CPU only, 12 logical cores shared by several concurrent experiments: use `n_jobs<=3`
  (LightGBM/RF/sklearn) and `torch.set_num_threads(3)` unless told otherwise.
* Long jobs: tool calls time out after 10 min. Make scripts resumable (skip runs whose output file
  exists) and expose CLI filters (e.g. `--split-types scaffold --seeds 0 1`) so they can be run in
  chunks, or launch them in the background and poll the log file.

## Layout
```
configs/default.yaml            central config (paths, curation, splits, features, AD)
data/raw/                       downloaded files + MANIFEST.json (url, sha256, date)
data/processed/                 aqsoldb_clean.csv, esol_clean.csv, esol_aqsoldb_overlap.csv,
                                *_removed.csv, curation_log.json, DATASET_VERSION.json
data/splits/aqsoldb/{random,scaffold,cluster}/seed{0..4}.json
data/splits/aqsoldb/loso/source_{A,B,C,E,F}.json
data/splits/external/{aqsoldb_to_esol,esol_to_aqsoldb}.json
data/splits/split_summary.csv, aqsoldb_butina_clusters.csv
data/features/{dataset}__{kind}.npz   (X, ids, names)
src/solpred/                    package (see modules below)
scripts/NN_*.py                 numbered pipeline steps (thin CLIs over the package)
results/<experiment>/           metrics_runs.csv, predictions/<split_type>/<split_name>/<model>__<featurizer>.csv
results/figures/<experiment>/   PNG figures (dpi 200)
results/tables/                 final summary tables (CSV + Markdown)
models/                         trained production model artefacts
app/                            FastAPI service + Streamlit dashboard
reports/                        final report, literature review
docs/                           methodology, data card, model card, this contract
tests/                          pytest suite
```

## Existing modules (do not modify; report problems instead)
* `solpred.config`: `load_config()`, `path(key)` (absolute dir under project root, created), `set_seed`, `save_json`, `load_json`, `ROOT`.
* `solpred.data.curate` / `solpred.data.download`: produce `data/processed/*`.
* `solpred.splits`: `load_split(split_type, seed=None, source=None, name=None)` returns
  `{"train": [ids], "val": [ids], "test": [ids], "meta": {...}}`; external splits also carry
  `train_dataset`/`test_dataset`. Also `morgan_bits(smiles)`, `max_similarity(query_fps, ref_fps, k=1)`.
* `solpred.features`: `load_dataset(name)` returns the curated DataFrame (`scaffold` "" for acyclic, ids str);
  `load_or_compute(dataset, kind)` returns `(X float32, ids, names)`, cached; `featurize_smiles(smiles, kind)`;
  `subset(X, ids, wanted_ids)`. Kinds: `rdkit2d` (217), `morgan_counts` (2048), `morgan_bits`,
  `maccs` (166), `esol4` (4), and `a+b` concatenations.
* `solpred.metrics`: `regression_metrics(y, p)` returns {n, MAE, RMSE, R2, Pearson, Spearman};
  `bootstrap_ci`, `paired_bootstrap_mae_diff`, `solubility_class`.
* `solpred.models`: `get_model(name, seed, n_jobs, params)` for mean, esol_linear, ridge, rf, lgbm, knn, mlp;
  `fit_model(model, Xtr, ytr, Xva, yva)` (LightGBM early-stops on val); `predict_with_std` (RF tree std).
* `solpred.experiment`: `run_split(split, featurizer, model_name, seed, params, n_jobs)` returns
  `(row, preds, model)`; `save_run(row, preds, out_dir, experiment)`; `run_exists(...)`; `summarize(df)`;
  `xy(dataset, featurizer, ids)`; `labels(dataset)`.

## Data columns
`aqsoldb_clean.csv`: compound_id (AqSolDB ID, e.g. `A-3`), name, smiles (standardised, isomeric canonical),
smiles_2d (dedup key), inchikey, **logS** (log10 mol/L), source (letter A-I = original AqSolDB sub-dataset),
aqsol_sd (SD across occurrences in AqSolDB), n_occurrences, reliability_group (G1 single measurement;
G2 two, SD>0.5; G3 two, SD<=0.5; G4 >=3, SD>0.5; G5 >=3, SD<=0.5), n_merged / merged_ids / merged_logS_sd /
dedup_conflict (our own stereo/representation de-duplication), parent_smiles, scaffold, n_fragments,
is_multicomponent, n_heavy_atoms, formal_charge_atoms, mol_wt, elements, inchikey_consistent, in_esol.
`esol_clean.csv`: compound_id (`ESOL-0000`), name, smiles, smiles_2d, logS, esol_pred (Delaney's published ESOL
prediction), ..., in_aqsoldb, aqsoldb_id, aqsoldb_logS.
Always read with `solpred.features.load_dataset()` (or `keep_default_na=False, na_values=[""]`).

## Reference (Phase-1) predictions available to every analysis
`results/baseline/` holds, for random/scaffold/cluster × seeds 0-4, val+test predictions of:
`mean__esol4`, `esol_linear__esol4`, `rf__rdkit2d` (with `y_std` = across-tree std) and
`lgbm__rdkit2d+morgan_counts` (default params). Path:
`results/baseline/predictions/<split_type>/seed<k>/<model>__<featurizer>.csv` (columns compound_id, subset, y_true, y_pred, y_std).

## Metrics and reporting
* Always report MAE (primary), RMSE, R2, Pearson, Spearman. Aggregate as mean ± std over the 5 seeds.
* Random and scaffold (and cluster) results are always shown side by side.
* Tables: CSV + Markdown (`df.to_markdown(index=False)`; install `tabulate` if missing), 3 decimals.
* Figures: matplotlib, dpi 200, readable fonts, axis labels with units ("logS (log mol/L)"), saved as PNG
  under `results/figures/<experiment>/`. Use a consistent split palette: random `#4C72B0`,
  scaffold `#DD8452`, cluster `#55A868`, external `#C44E52`.
* Every analysis writes a short machine-readable `summary.json` with the headline numbers it found,
  so the report can cite them exactly.
* Seeds: use the split seed as model seed. Never tune on test sets (use the `val` subset).
