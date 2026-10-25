# solpred serving layer

Two front-ends over the same production model (`models/final`, produced by
`scripts/11_train_final_model.py`, loaded with `solpred.predict.SolubilityPredictor`):

| file | what | run (from the project root) |
|---|---|---|
| `api.py` | FastAPI JSON service | `.venv/Scripts/python.exe -m uvicorn app.api:app --host 127.0.0.1 --port 8000` |
| `dashboard.py` | Streamlit dashboard | `.venv/Scripts/python.exe -m streamlit run app/dashboard.py` |

Both read the model directory from the `SOLPRED_MODEL_DIR` environment variable (default `models/final`).
Interactive API docs are served at `http://127.0.0.1:8000/docs` once uvicorn is running.

## REST API

| method | path | body / response |
|---|---|---|
| GET | `/health` | `{"status": "ok", "model_loaded": true, ...}` (`"degraded"` + `error` if the model failed to load) |
| GET | `/model` | the model card = `models/final/metadata.json` (data hash, params, CV metrics, conformal coverage, versions) |
| POST | `/predict` | request `{"smiles": ["CCO", "c1ccccc1"], "alpha": 0.1}`; response `{"config_hash", "alpha", "n", "n_valid", "predictions": [...]}` |

Request rules: 1-1000 SMILES per call, each at most 2000 characters, `alpha` in [0.01, 0.5] (default 0.1,
i.e. 90 % intervals). Violations return **HTTP 422** with pydantic's error list. Chemically invalid or
out-of-scope SMILES do not fail the request; they come back with `valid: false`, an `error` text and a
`warnings` entry, and their numeric fields are `null`.

Each prediction carries the columns of `SolubilityPredictor.predict`:
`input_smiles, standardized_smiles, valid, error, logS_pred, logS_std, pi_low, pi_high, pi_level,
solubility_class, solubility_mg_per_L, ad_maxsim, ad_in_domain, nearest_train_id, nearest_train_smiles,
nearest_train_logS, in_training_set, warnings`.

```bash
curl -s -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" \
     -d '{"smiles": ["CC(=O)Oc1ccccc1C(=O)O", "not_a_smiles"], "alpha": 0.1}'
```

```python
import httpx
r = httpx.post("http://127.0.0.1:8000/predict", json={"smiles": ["CCO", "c1ccc2ccccc2c1"]})
print(r.json()["predictions"][1]["logS_pred"])
```

## Dashboard

* **Input**: paste SMILES (one per line, optionally followed by a name: `CCO ethanol`) or upload a CSV and pick
  the SMILES / name columns (first 1000 rows are predicted).
* **Sidebar**: model card (training-set size and hash, scaffold-grouped CV MAE/RMSE/R2, conformal coverage, AD
  definition) and the prediction-interval level (50-95 %).
* **Results**: colour-coded table (AD: green = in domain, red = outside; solubility class shading; warnings in
  red), CSV download of all columns, histogram of predicted logS with class boundaries, and RDKit depictions of
  the first 24 valid structures labelled with prediction, interval and an `OOD` tag.

Verified headless with `streamlit.testing.v1.AppTest` (see `tests/test_api.py::test_dashboard_runs`).

## Interpreting the output

* `pi_low`-`pi_high`: normalised split-conformal interval (score = |error| / (ensemble std + beta), beta = 0.175 chosen on the calibration folds) calibrated on
  out-of-fold predictions of a 5-fold scaffold-grouped CV of the same ensemble. The coverage guarantee is
  *marginal*: about `pi_level` of compounds that resemble AqSolDB are covered; it is not a per-compound promise
  and is weaker for compounds flagged `outside applicability domain`.
* `ad_maxsim` / `ad_in_domain`: max ECFP4 Tanimoto similarity to the 9,171 training compounds; below 0.35 the
  compound is flagged out of domain. `nearest_train_*` gives that neighbour and its measured logS - a quick
  sanity check of the prediction.
* `in_training_set`: the structure (non-isomeric) is in the training data; the prediction is then in-sample and
  the measured value (`nearest_train_logS`) should be preferred.
* `multicomponent`: salts/mixtures are predicted as the whole species (AqSolDB measured them that way).
* Inputs outside the curation scope (no carbon, elements other than H B C N O F Si P S Cl Se Br I, metals other
  than Li/Na/K/Mg/Ca monoatomic counter-ions) are rejected rather than extrapolated.

## (Re)training the served model

The served artefact is `models/final/` (`model.joblib` 14.7 MB, `metadata.json`, `summary.json`,
`oof_predictions.csv`, `figures/`, `MODEL_CARD.md`, `train.log`). It is produced by

```bash
# default: LightGBM on rdkit2d+morgan_counts, registry params, 5 bagged members, 5-fold scaffold-grouped conformal
.venv/Scripts/python.exe scripts/11_train_final_model.py --n-jobs 3

# one-liner with tuned hyper-parameters (configs/models_tuned.yaml is written by the tuning step)
.venv/Scripts/python.exe scripts/11_train_final_model.py --params-yaml configs/models_tuned.yaml     --params-key lgbm__rdkit2d+morgan_counts --n-jobs 3 --force
```

`--params-key` selects one entry of the YAML (top level, one level down, or a dotted path `a.b`). An entry may be a
flat parameter mapping or `{model: ..., features: ..., params: {...}}`; a key `<model>__<features>` also sets the
model and feature kind unless the entry names them. Bookkeeping fields (`best_*`, `val_*`, `cv_*`, `n_trials`, ...)
are ignored. `--params-json` accepts the same information as inline JSON or a JSON file. Hyper-parameters must come
from validation-set tuning, never from test sets. A full run took about 65 min (fit time with 2 threads on the shared CPU; 25 CV + 5
final LightGBM fits). Fits are cached under `models/final/_work/<config-hash>/` while training so an interrupted
run resumes; the final members are removed from the cache once `model.joblib` is written, and the remaining
`_work/` directory (CV caches, ~0.9 MB) can be deleted afterwards - it is not part of the deliverable. Re-running
with an unchanged configuration is a no-op (the config hash is compared with `metadata.json`). After retraining,
re-run `solpred-predict` on `examples/example_smiles.csv` and the tests.

## Tests

```bash
.venv/Scripts/python.exe -m pytest tests/test_predict.py tests/test_api.py
```
