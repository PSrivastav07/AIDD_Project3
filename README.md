# solpred: a similarity-aware aqueous-solubility (logS) benchmark and predictor

This is the course project for *AI in Drug Discovery*, Challenge 1 (ADMET / physicochemical): predicting
aqueous solubility from structure. It provides a curated, versioned AqSolDB + ESOL dataset; random,
scaffold and cluster splits evaluated side by side, plus cross-source and cross-dataset transfer; a comparison
of five featurisation families, including a graph neural network; an analysis of uncertainty, applicability
domain, measurement heterogeneity and model errors; and a deployable predictor (CLI, Python API, FastAPI
service and Streamlit dashboard) that returns logS together with a conformal interval and an
applicability-domain flag.

**Available reports:** [Literature review](reports/literature_review.md) and [references](reports/references.md). The final report files were not present in the supplied project; benchmark tables and figures are included under `results/`.

## Key results (test MAE in log mol/L, mean ± SD over 5 seeds; lower is better)

| Model | Features | Random | Scaffold | Cluster |
|---|---|---|---|---|
| LightGBM | RDKit 2D descriptors + Morgan counts | **0.574 ± 0.019** | **0.717 ± 0.027** | **0.670 ± 0.037** |
| LightGBM | RDKit 2D descriptors | 0.580 ± 0.020 | 0.722 ± 0.024 | 0.677 ± 0.044 |
| D-MPNN (graph network) + descriptors | graph + RDKit 2D (3 seeds) | 0.577 ± 0.013 | 0.805 ± 0.024 | 0.707 ± 0.064 |
| D-MPNN (graph network) | molecular graph | 0.653 ± 0.013 | 0.811 ± 0.030 | 0.718 ± 0.034 |
| Random forest | RDKit 2D descriptors | 0.647 ± 0.016 | 0.774 ± 0.040 | 0.739 ± 0.038 |
| LightGBM | Morgan counts only | 0.695 ± 0.012 | 0.871 ± 0.041 | 0.809 ± 0.035 |
| ESOL equation (refit) | cLogP, MW, rotatable bonds, aromatic proportion | 1.187 ± 0.043 | 1.251 ± 0.094 | 1.226 ± 0.091 |

These rows are a selection. The full table, with RMSE, R², Pearson and Spearman for all 19 configurations,
is in `results/tables/main_results.md` and `main_results_full.md`.

Main findings:
- **Novel chemotypes cost about 25 % in MAE.** For the best model, the scaffold split costs +0.14 log units
  and the cluster split +0.10, relative to the random split. Lower similarity to the training set, together
  with the scaffold split's shift toward insoluble compounds, explains most of that penalty. A
  scaffold-specific residual of about 0.05 remains.
- **Physicochemical descriptors carry most of the signal.** Fingerprints add little on top of them. A graph
  network without hyper-parameter tuning matches gradient boosting on random splits only when it is also
  given the descriptors, and it falls behind on novel scaffolds.
- **Measurements differ between sources.** Compounds with several reports have an RMS inter-report SD of 0.48
  log units. Holding out a whole source raises MAE by up to 0.19, compared with a size-matched control on the
  same compounds. Cleaning or re-weighting the training data by reliability changes MAE by less than 0.01.

## Repository layout

```
configs/      default.yaml (paths, curation, splits, AD), models_tuned.yaml (validation-tuned hyper-parameters)
data/raw/     downloaded AqSolDB and ESOL files plus MANIFEST.json (URLs, SHA-256, dates)
data/processed/  curated datasets, removal logs, curation_log.json, DATASET_VERSION.json
data/splits/  random / scaffold / cluster (x5 seeds), leave-one-source-out and external splits (JSON)
src/solpred/  Python package: data curation, splits, features, models (incl. the D-MPNN), metrics,
              uncertainty, applicability domain, analyses, final model, predictor
scripts/      numbered pipeline steps 01-13 (thin CLIs over the package)
results/      benchmark, gnn, uncertainty, heterogeneity, error_analysis, baseline, qa outputs;
              tables/ (CSV + Markdown) and figures/ (PNG)
models/final/ production model (model.joblib), metadata, MODEL_CARD.md, cross-validation figures
app/          FastAPI service (api.py) and Streamlit dashboard (dashboard.py)
examples/     example SMILES, predictions and a triage-ranking demo
reports/      final report (md/html/pdf), literature review, references
docs/         methodology, data card, AqSolDB source map, engineering contract
tests/        pytest suite (curation, splits, features, metrics, experiment, predictor, API)
```

## Installation (Windows, Python 3.13; CPU only)

```bash
py -3.13 -m venv .venv
```

```bash
.venv/Scripts/python.exe -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

```bash
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

```bash
.venv/Scripts/python.exe -m pip install -e .
```

On Linux or macOS use `.venv/bin/python` instead. Exact versions are pinned in `requirements.txt`, and the full
transitive lock is in `requirements-lock.txt`.

## Using the predictor

Predict from the command line:

```bash
.venv/Scripts/solpred-predict.exe --smiles "CC(=O)Oc1ccccc1C(=O)O" "c1ccc2ccccc2c1"
```

Run batch predictions from a CSV file:

```bash
.venv/Scripts/solpred-predict.exe --input examples/example_smiles.csv --smiles-col smiles --output predictions.csv
```

Start the REST API (interactive docs at http://127.0.0.1:8000/docs):

```bash
.venv/Scripts/python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000
```

Start the dashboard:

```bash
.venv/Scripts/python.exe -m streamlit run app/dashboard.py
```

The Python API:

```python
from solpred.predict import SolubilityPredictor
pred = SolubilityPredictor.load("models/final")
df = pred.predict(["CCO", "c1ccc2ccccc2c1"], alpha=0.1)   # logS, std, 90 % interval, class, mg/L, AD flag, warnings
```

Each prediction includes the standardised structure, logS, the ensemble SD, a conformal interval (90 % by
default), the AqSolDB solubility class, solubility in mg/L, the maximum Tanimoto similarity to the training
set, the nearest measured training compound, and warnings: invalid input, out-of-scope chemistry,
multi-component input, out of domain, or outside the training label range. Read `models/final/MODEL_CARD.md`
before use. Interval coverage is close to nominal for compounds like the training set but lower for unusual
chemistry (0.84 at the 90 % level outside the applicability domain).

## Reproducing everything

```bash
powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
```

The steps in order are: download, curate, split, featurise, baselines, benchmark (tune / grid / analyze),
GNN, uncertainty and AD, heterogeneity and transfer, error analysis, final model, collect results, and tests.
Every step is resumable. A full run takes about 6–7 h on a 12-thread laptop CPU, dominated by the GNN and the
model grid. To render the report:

```bash
.venv/Scripts/python.exe scripts/13_render_report.py
```

To run the tests:

```bash
.venv/Scripts/python.exe -m pytest tests -q
```

## Data, licences and responsible use

AqSolDB (Sorkun, Khetan & Er, *Scientific Data* 2019; doi:10.7910/DVN/OVHAW8) is released under CC0. ESOL
(Delaney, *J. Chem. Inf. Comput. Sci.* 2004) is used via the MoleculeNet/DeepChem mirror. Labels combine
measurements from different laboratories, temperatures and solid forms, and 77 % of compounds rest on a single
report. Provenance, curation decisions and limitations are in [docs/data_card.md](docs/data_card.md).

The model is a research tool for early, in-silico triage: deciding which compounds to measure first. It does
not replace experimental solubility measurement. Do not use it for formulation, safety or regulatory decisions.
Out-of-domain flags and interval widths should be read alongside the prediction itself.

## Documentation

- [docs/methodology.md](docs/methodology.md): every method, design decision and leakage safeguard.
- [docs/data_card.md](docs/data_card.md): data provenance, the curation funnel and limitations.
- [models/final/MODEL_CARD.md](models/final/MODEL_CARD.md): the production model's intended use, metrics and
  caveats.
- [reports/literature_review.md](reports/literature_review.md): literature review, with verified references in
  `reports/references.md`.
- [results/qa/foundation_review.md](results/qa/foundation_review.md): leakage and data-quality audit.

