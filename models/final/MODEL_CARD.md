# Model card - solpred-final (aqueous solubility, logS)

All numbers below are read from `models/final/metadata.json` and `models/final/summary.json`
(written by `scripts/11_train_final_model.py`); figures are in `models/final/figures/`.

## Model details

| item | value |
|---|---|
| task | regression of aqueous solubility logS (log10 mol/L) from a 2-D structure (SMILES) |
| learner | bagged ensemble of 5 LightGBM regressors (bootstrap resample per member; each member early-stopped, patience 100, on its own out-of-bag compounds; 986-1408 trees) |
| hyper-parameters | registry defaults: learning_rate 0.03, num_leaves 63, min_child_samples 10, subsample 0.8, colsample_bytree 0.3, reg_lambda 1.0, max 2000 trees (not tuned on any test set) |
| features | `rdkit2d+morgan_counts`: 217 RDKit 2-D descriptors + 2048 Morgan (r=2) count bits = 2,265 features |
| uncertainty | ensemble std (member disagreement) -> normalised split-conformal interval |
| applicability domain | max Tanimoto (ECFP4, 2048 bits) to the 9,171 training compounds; in domain if >= 0.35 |
| config hash / artefact | `0d47ae7a78`; `model.joblib` 14.7 MB (joblib, compress 3) |
| software | Python 3.13.7, RDKit 2026.09.1, LightGBM 4.7.0, scikit-learn 1.9.1, numpy 2.5.4, pandas 3.0.6 |
| interfaces | `solpred.predict.SolubilityPredictor`, CLI `solpred-predict`, FastAPI `app/api.py`, Streamlit `app/dashboard.py` |

## Intended use

* Early-stage triage and prioritisation of organic, drug-like or industrial small molecules: ranking a
  virtual library, flagging likely-insoluble compounds before synthesis or assay, and teaching.
* Every prediction must be read together with its interval, AD flag and nearest training neighbour.

**Not intended for**: regulatory or safety decisions, formulation design, pH-dependent / buffered solubility,
polymorph- or temperature-specific solubility, inorganic or organometallic species, peptides/biologics,
polymers, or any decision where a measured value is available or required.

## Training data

* Curated AqSolDB: 9,982 records -> **9,171** unique compounds after standardisation and de-duplication
  (dataset version 1.0.0, `aqsoldb_clean.csv` sha256 `5ad364af...dc84`). Training logS range -13.17 to 2.14.
* Label noise: AqSolDB merges measurements from 9 source datasets; the pooled RMS inter-measurement SD of
  compounds measured more than once is 0.478 log units (n = 2,065), a practical floor for any model's error.
* ESOL (Delaney) is **not** an independent test set for this model: 1,097 of its 1,115 curated compounds (98.4 %)
  are in AqSolDB and therefore in the training data.

## Evaluation (out-of-fold, scaffold-grouped 5-fold CV of the same training procedure)

Each fold's ensemble is trained exactly like the production ensemble on the other four folds. Folds are
grouped by Bemis-Murcko scaffold; the acyclic "scaffold" (2,439 compounds) and plain benzene (the only scaffold
with > 500 members) are generic, so their 4,184 compounds are spread as singletons - production queries with
these scaffolds always have training analogues. The 4,987 other compounds (1,863 scaffolds) are held out with
their whole scaffold.

| subset | n | MAE | RMSE | R2 |
|---|---:|---:|---:|---:|
| all compounds (OOF) | 9,171 | 0.628 | 0.908 | 0.850 |
| specific scaffolds (whole scaffold unseen) | 4,987 | 0.711 | 0.980 | 0.831 |
| generic scaffolds (acyclic / benzene) | 4,184 | 0.529 | 0.814 | 0.842 |
| in AD (max sim >= 0.35) | 8,231 | 0.610 | 0.888 | 0.859 |
| outside AD | 940 | 0.782 | 1.066 | 0.750 |

* Per-fold MAE 0.588-0.660; the 5-member ensemble improves on the average single member (MAE 0.651).
* MAE rises from 0.50-0.60 at max similarity >= 0.6 to 0.85 below 0.3 (`figures/error_vs_similarity.png`).
* For context, the Phase-1 single LightGBM of the benchmark (`results/baseline/summary.csv`) reaches MAE 0.574
  (random split) and 0.717 (scaffold split), mean of 5 seeds; the specific-scaffold OOF MAE (0.711) is the
  number comparable to the scaffold split.

## Uncertainty: conformal prediction intervals

* Score = |y - mean| / (ensemble std + beta), calibrated on the 9,171 OOF predictions. With beta = 0 the
  ensemble std (only weakly related to the error: Spearman 0.27) produced intervals that under-covered the
  most confident fifth of compounds (78.6 %) and over-covered the least confident (96.5 %). The offset
  **beta = 0.175** was selected on the calibration folds only (leave-one-fold-out coverage most uniform across
  std quintiles); it makes the 90 % coverage 0.893-0.906 in every std quintile and shrinks the mean 90 %
  interval from 3.21 to 2.80 log units (`figures/conformal_calibration.png`).
* Marginal coverage: 90 % nominal -> 0.900 in-sample, 0.899 leave-one-fold-out (folds 0.882-0.910), mean
  width 2.80; 95 % nominal -> 0.950 (folds 0.941-0.963), mean width 3.67. An absolute-residual interval of the
  same coverage would be 2.92 (90 %) / 3.91 (95 %) wide.
* Conditional coverage is **not** guaranteed: 90 % intervals cover 0.907 of in-domain but only 0.841 of
  out-of-domain compounds (0.82 below similarity 0.3), and 0.881 of compounds with unseen specific scaffolds.
  beta was chosen on the same OOF data that calibrates the quantile (one scalar; mild optimism possible).
* For compounds that are in the training set the prediction is in-sample (much closer to the label than the
  interval suggests); the output flags them (`in_training_set`) and the measured value should be used.

## Limitations

* Labels are heterogeneous literature values (different methods, temperatures, pH, solid forms); errors
  below ~0.5 log units cannot be expected and the model inherits source biases.
* No pH, counter-ion, polymorph or temperature awareness; salts are predicted as the measured whole species.
* Extreme values are poorly covered: OOF MAE ~1.0 for logS <= -6 and 0.70 for logS > 0 versus 0.52-0.55 in
  the -4 to 0 range (computed from `oof_predictions.csv`).
* The AD is a single similarity threshold; similar fingerprints do not guarantee similar solubility
  (activity cliffs), and dissimilar compounds may still be predicted well.
* Inputs outside the curation scope (no carbon, elements outside H B C N O F Si P S Cl Se Br I, metals other than
  Li/Na/K/Mg/Ca counter-ions) are rejected, not extrapolated.

## Responsible use

* Treat outputs as a prioritisation aid; confirm decisions with measurement. Prefer the interval and
  the tier logic in `examples/triage_demo.py` over the point estimate.
* Report the model version (config hash, dataset sha256) with any prediction you share.
* Do not present predictions as measured data, and do not use the model to justify skipping required safety,
  toxicity or regulatory testing.

## Reproduction

```bash
.venv/Scripts/python.exe scripts/11_train_final_model.py --n-jobs 3            # ~65 min of fitting on a shared CPU
.venv/Scripts/python.exe scripts/11_train_final_model.py --recalibrate          # refit conformal only (seconds)
.venv/Scripts/python.exe scripts/11_train_final_model.py --params-yaml configs/models_tuned.yaml \
    --params-key lgbm__rdkit2d+morgan_counts --n-jobs 3 --force                  # retrain with tuned params
```
