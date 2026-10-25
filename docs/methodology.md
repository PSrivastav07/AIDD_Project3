# Methodology

This document describes how each result in the project is produced. Numbers are in
`reports/final_report.md` and the `results/` tree. Every step corresponds to a numbered script in `scripts/`
(see `run_pipeline.ps1`) and a module in `src/solpred/`.

## 1. Data (`scripts/01–02`, `solpred.data`)

AqSolDB (9,982 records) and ESOL (1,128) are downloaded with SHA-256 provenance. The curation steps are:
standardisation (RDKit `rdMolStandardize.Cleanup`), removal of inorganic and organometallic compounds, and
stereo-agnostic de-duplication with a median label. This yields 9,171 AqSolDB and 1,115 ESOL compounds. ESOL
records that appear in AqSolDB are linked by `smiles_2d` (1,097 compounds). Full details, design decisions and
limitations are in [data_card.md](data_card.md). The target is logS in log10(mol/L). Labels are used as
published; there is no unit or temperature correction.

## 2. Splits (`scripts/03`, `solpred.splits`)

Every split is stored as JSON (`data/splits/`) and checked for disjointness and completeness. Split
diagnostics are in `data/splits/split_summary.csv`. They cover the test→train maximum ECFP4 Tanimoto
similarity, the fraction of test scaffolds seen in training, and the test-set label statistics.

| Split | Construction | Purpose |
|---|---|---|
| random (×5 seeds) | uniform 80/10/10 | in-distribution reference |
| scaffold (×5) | Bemis–Murcko scaffold of the largest organic fragment, with Chemprop's *balanced* assignment: scaffold sets larger than half the val/test size go to train first, the rest are shuffled with the seed and filled greedily into train → val → test | generalisation to unseen chemotypes |
| cluster (×5) | Butina sphere-exclusion clustering on ECFP4 (radius 2, 2048 bits), Tanimoto distance cut-off 0.6, with whole clusters assigned by the same balanced procedure | similarity-based novelty without the acyclic artefact (below) |
| LOSO | leave one AqSolDB source letter out (A, B, C, E, F; ≥ 200 compounds), with validation = random 10 % of the remaining sources | cross-source (inter-laboratory) transfer |
| external | `aqsoldb_to_esol`: train on AqSolDB minus the 1,097 ESOL-overlap compounds, test on all 1,115 ESOL; `esol_to_aqsoldb`: train on ESOL, test on the 8,074 non-overlapping AqSolDB compounds | cross-dataset transfer |

**Known confounder of the scaffold split.** The 2,439 acyclic compounds share the empty scaffold, which forms
one giant group that always lands in train. Scaffold test sets therefore contain no acyclic compounds, and
their labels are shifted toward insolubility (mean logS −3.58 vs −2.91 for random). The cluster split has a
similar similarity shift (median test→train max similarity 0.47–0.50 vs 0.61 for random) but keeps a
representative label distribution. The error analysis separates the two effects (§8).

## 3. Featurisations (`scripts/04`, `solpred.features`)

| Kind | Content |
|---|---|
| `esol4` | Delaney's ESOL descriptors: Crippen cLogP, MW, rotatable bonds, aromatic proportion |
| `rdkit2d` | all 217 RDKit 2D descriptors (`Ipc` log-transformed; non-finite values → NaN, median-imputed inside the model pipelines) |
| `morgan_counts` / `morgan_bits` | Morgan/ECFP4 radius-2, 2048-bit folded count and binary fingerprints |
| `maccs` | 166 MACCS structural keys |
| `rdkit2d+morgan_counts` | concatenation (2,265 columns) |
| `graph` | atom features (element one-hot including Na/K/Li/Mg/Ca, degree, formal charge, chirality, H count, hybridisation, aromaticity, mass) and bond features (type, conjugation, ring, stereo) for the D-MPNN |
| `graph+rdkit2d` | graph plus the 217 descriptors, median-imputed and standardised on the training rows, concatenated to the readout |

## 4. Models

**Baselines.** The baselines are a training-mean predictor, an OLS refit of the ESOL equation on `esol4` (the
classic physically motivated baseline), and Tanimoto k-NN on `morgan_bits`.

**Classical learners** (`solpred.models`, `solpred.benchmark`, `scripts/06`): RidgeCV, random forest,
LightGBM (early stopping on the validation MAE) and an MLP, each crossed with
{rdkit2d, morgan_counts, maccs, rdkit2d+morgan_counts}. The MLP was not run on the two fingerprint-heavy
inputs: the fits took 6–9 minutes each and the smoke test showed clearly worse validation MAE (0.96 / 0.83 vs
0.68).

**Hyper-parameter tuning, using validation data only.** A random search was run for LightGBM (registry
default plus 5 random draws over learning rate, leaves, minimum child samples, column/row subsampling and L2)
on rdkit2d, morgan_counts and rdkit2d+morgan_counts. For the random forest, 6 stratified draws covered
`max_features` and `min_samples_leaf`, with the number of trees fixed at 300. The k for k-NN was chosen from
{3, 5, 10, 20}. The objective is the mean validation MAE over the random-seed-0 and scaffold-seed-0
validation subsets. Pairs without their own search inherit parameters (for example maccs inherits from
rdkit2d). Trials are in `results/benchmark/tuning_trials.csv`; the chosen values are in
`configs/models_tuned.yaml`. Test sets are never used for selection.

**Graph neural network** (`solpred.models.gnn`, `scripts/07`): a Chemprop-style directed message-passing
network (D-MPNN; Yang et al. 2019), implemented directly in PyTorch:
- hidden states on directed bonds, with reverse-edge subtraction;
- depth 3;
- atom-level aggregation, so bond-less counter-ions still contribute;
- mean readout and a two-layer feed-forward head;
- hidden size 200 and dropout 0.1;
- Noam learning-rate schedule (1e-4 → 1e-3 → 1e-4), batch size 64, MSE on standardised targets;
- at most 50 epochs, with early stopping on the validation MAE (patience 10).

The hyper-parameters were fixed a priori to Chemprop defaults, except hidden size 200 instead of 300, a
compute-budget choice documented in `results/gnn/profile.json`. The two variants are `dmpnn` (graph only;
5 seeds per split) and `dmpnn_desc` (graph + RDKit descriptors; seeds 0–2). Predictive spread comes from MC
dropout in the head (10 passes).

## 5. Evaluation

- **Metrics** (`solpred.metrics`): MAE is the primary metric. RMSE, R², Pearson r and Spearman ρ are also
  reported, always on the held-out test subset.
- **Aggregation:** mean ± SD over the 5 seeds. Random, scaffold and cluster results are shown side by side.
  R² depends on the target variance of each test set, which differs between split types, so MAE/RMSE are the
  comparable quantities.
- **Statistics:** percentile bootstrap CIs. For paired model comparisons, a paired bootstrap of the MAE
  difference on the pooled test predictions of all seeds, plus a Wilcoxon signed-rank test over per-seed MAEs.
  The Wilcoxon test is reported with the caveat of low power at n = 5.
- **Headline tables:** `scripts/12_collect_results.py` merges `results/benchmark` and `results/gnn` into
  `results/tables/main_results.*` and `featurization_comparison.*`.

## 6. Uncertainty and applicability domain (`scripts/08`, `solpred.analysis.uncertainty_eval`)

**Training design.** For each split, a random 10 % of the training pool is held out as an exchangeable
calibration set (`cal_random`). A 5-member LightGBM ensemble (members differ in seed, with row bagging 0.8 and
column subsampling 0.5) is trained on the remaining 90 % and early-stopped on the split's validation subset.

**Uncertainty signals.** Three are compared:
- ensemble standard deviation;
- across-tree standard deviation of the Phase-1 random forest;
- k-NN distance: 1 − the mean Tanimoto similarity of the 5 nearest training neighbours.

**Quality of each signal.** Measured as:
- Spearman(σ, |error|);
- AUROC for flagging |error| > 1 log unit;
- sparsification curves and AUSE;
- Gaussian NLL and miscalibration area after a single variance scale factor fitted on the validation subset.

**Conformal prediction** (`solpred.uncertainty.ConformalRegressor`). Split-conformal intervals, both standard
(absolute residual) and normalised (residual / σ), at α = 0.2, 0.1 and 0.05. The quantile is the k-th smallest
calibration score, with k = ⌈(n+1)(1−α)⌉. Two calibration settings are compared:
- *matched*: calibrated on the validation subset of the same split, which is as novel as the test set;
- *under shift*: calibrated on `cal_random` and applied to scaffold, cluster and external test sets.

The same fitted models serve both settings, so the comparison isolates exchangeability. The validation subset
also serves early stopping, which may make matched coverage marginally optimistic; `cal_random` is the
leakage-free reference.

**Applicability domain** (`solpred.applicability`). Compounds are scored on:
- maximum and mean top-5 ECFP4 Tanimoto similarity to the training set (`ApplicabilityDomain`);
- Mahalanobis distance and leverage in the top-20 PCA space of training-standardised RDKit descriptors
  (`DescriptorAD`, warning leverage h* = 3(p+1)/n).

The in-domain similarity threshold is chosen on the pooled validation subsets by Youden's J for flagging
|error| > 1. It is never chosen on test data.

## 7. Measurement heterogeneity and transfer (`scripts/09`, `solpred.analysis.heterogeneity`)

The default learner here is LightGBM on `rdkit2d+morgan_counts`.

**(i) Intrinsic noise.** AqSolDB inter-report SD by reliability group, pooled RMS SD with a bootstrap CI,
implied Gaussian and distribution-free error floors, and a Bland–Altman comparison of ESOL vs AqSolDB on the
overlap. The selection-bias caveat applies: only the 22.5 % of compounds with several reports carry an SD.

**(ii) Test error by data reliability.** Error broken down by reliability group, number of reports, source
letter and SD bin.

**(iii) Training-data quality ablation** (random and scaffold splits × 5 seeds):

| Variant | Training data |
|---|---|
| (a) | all training data |
| (b) | drop the inconsistent G2+G4 compounds |
| (c) | drop de-duplication conflicts |
| (d) | inverse-variance weights 1/(SD² + s0²), with s0 = 0.478, a dataset-level constant from the AqSolDB SD metadata |
| (e) | size-matched random removal (3 replicates), the control for (b) |
| (f) | permuted weights, the control for (d) |

Variants are scored on the full test set and on the high-reliability (G3+G5) test subset, with paired
bootstrap ΔMAE.

**(iv) Leave-one-source-out.** Compared against a size-matched in-distribution control on the *same*
compounds; the difference is the source-shift penalty. A 500-compound source-to-source transfer matrix for
A, B, C and F is also reported.

**(v) External transfer in both directions.** Compares LightGBM, RF, the ESOL-linear refit, and Delaney's
published ESOL predictions. Delaney's predictions are in-sample for ESOL and therefore optimistic.

## 8. Error analysis and explainability (`scripts/10`, `solpred.analysis.errors`, `.chemspace`)

The reference model is LightGBM on `rdkit2d+morgan_counts`, with test errors pooled over seeds.

- **Error vs similarity:** MAE against test→train maximum ECFP4 similarity, measured to the training set
  only. The validation subset is excluded because the model is not fitted on it.
- **Penalty decomposition:** direct standardisation of random-split errors to the scaffold and cluster test
  compositions (similarity × logS strata), complemented by an additive OLS adjustment. Together these
  estimate how much of the split penalty is due to lower similarity, how much to label shift, and how much
  remains.
- **Error by chemistry:** error per physicochemical bin and per SMARTS functional-group class, as MAE and mean
  signed error.
- **Worst cases:** the worst-predicted compounds, ECFP4 near-identical pairs with large label differences
  (separated into fingerprint-identical homologues and same-size pairs), and the worst scaffolds.
- **Chemical-space map:** UMAP on ECFP4 bits (Jaccard distance, 30 neighbours, min_dist 0.1).
- **Explainability:** SHAP TreeExplainer on a descriptor-only LightGBM, computed on the held-out random-seed-0
  test set.

## 9. Production model (`scripts/11`, `solpred.final_model`, `solpred.predict`)

**Model.** A 5-member bagged LightGBM ensemble on `rdkit2d+morgan_counts`, trained on all 9,171 curated
compounds. The number of boosting rounds per member comes from cross-validation.

**Conformal calibration.** 5-fold `GroupKFold` over Bemis–Murcko scaffolds; acyclic compounds and scaffold
groups too large for balanced folds become singleton groups. Calibration uses the out-of-fold residuals with
normalised scores |y − ŷ| / (σ_ens + β). The offset β is chosen by leave-one-fold-out on the out-of-fold data
alone; it stabilises coverage across the confidence quintiles.

**Applicability domain.** Maximum ECFP4 similarity to the training set, plus the nearest training neighbour and
its measured label.

**Output per SMILES:**
- standardised structure;
- predicted logS, σ and a conformal interval;
- solubility class (AqSolDB bands) and mg/L;
- applicability-domain flag;
- warnings: invalid input, rejected chemistry, multi-component, outside the training label range.

The predictor is exposed as a Python API, the `solpred-predict` command-line tool, a FastAPI service
(`app/api.py`) and a Streamlit dashboard (`app/dashboard.py`).

## 10. Leakage safeguards (checked by `tests/` and `results/qa/foundation_review.md`)

- De-duplication on stereo-agnostic SMILES, so no identical 2D input can sit in both train and test.
- Overlap removal for the external transfer.
- Scaffold and cluster groups are never split across subsets.
- Every selection and calibration step uses validation or calibration data only: hyper-parameters, early
  stopping, σ scaling, conformal quantiles, AD thresholds and β.
- Feature scaling and imputation are fitted on training rows only.
- The audit quantifies the remaining near-leakage. Charge-state or tautomer twins make up ≤ 0.4 % of test
  sets with a median label difference of 0.016. Salt / free-parent pairs make up 3.9 %; they are predicted
  *worse* than average, so they add difficulty rather than leakage. The estimated inflation is ≤ 0.01 MAE.

## 11. Reproducibility

- **Determinism:** seeds are fixed (the split seed is also the model seed).
- **Software:** versions are pinned in `requirements.txt` (full lock in `requirements-lock.txt`).
- **Data:** hashes are recorded in `data/raw/MANIFEST.json` and `data/processed/DATASET_VERSION.json`.
- **Restartability:** every script is resumable. `run_pipeline.ps1` reproduces the full study in about 6–7
  hours on a 12-thread laptop CPU without a GPU.
