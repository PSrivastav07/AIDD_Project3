# Foundation review (QA): data curation, splits, features, metrics

Adversarial audit of the shared foundation (`data/processed`, `data/splits`, `data/features`,
`src/solpred/{data,splits,features,metrics,experiment,uncertainty}`). Nothing under `data/` or `src/`
was modified.

* Evidence script: `results/qa/foundation_audit.py` (about 9 min on 3 threads; log in `results/qa/logs/foundation_audit.log`)
* Numbers: `results/qa/audit_numbers.json` (everything below), condensed into `results/qa/summary.json` by `results/qa/build_summary.py`
* Tables: `results/qa/tables/*.csv`. Figures: `results/qa/figures/near_duplicate_leakage.png`, `results/qa/figures/label_plausibility.png`
* Unit tests: `tests/test_{curation,splits,features,metrics,experiment}.py` (+ `tests/conftest.py`)

Severity scale: **High** means a headline result could change. **Medium** means a number could be biased by a few hundredths of log units, or a sub-analysis is affected. **Low** means a cosmetic or data-hygiene issue that is worth mentioning but changes nothing.

---

## 0. Test suite

`.venv/Scripts/python.exe -m pytest tests -q` reported **163 tests: 157 passed, 6 xfailed (strict), 0 failed, 0 errors**,
with 114 s wall time (`results/qa/pytest_results.xml`, `results/qa/logs/pytest_full.log`).

* QA-owned files: 141 tests (135 passed + 6 xfailed), about 37 s.
* Other agents' files: `test_api.py` (13 passed) and `test_predict.py` (9 passed). `test_api.py::test_max_batch_accepted`
  alone takes about 40 s, so it dominates the runtime.
* The run also prints a harmless `StarletteDeprecationWarning`: the starlette `TestClient` warns about `httpx`.

The six `xfail(strict=True)` tests each document a real limitation of read-only shared code. If one of them starts to XPASS, the
limitation has been fixed and the marker should be removed:

| test | module | limitation | severity |
|---|---|---|---|
| `test_known_no_carbanion_from_neutral_metal` | `data/curate.py` | A neutral metal written as `[Na]` beside a molecule with no acidic H makes the cleanup deprotonate a C-H, which gives carbanions (§6) | Low |
| `test_known_charge_states_not_merged` | `data/curate.py` | No neutralisation step, so an acid and its carboxylate get different `smiles_2d` and are never merged (§1) | Low-Medium |
| `test_known_isotopologues_not_merged` | `data/curate.py` | Isotope labels are dropped but an explicit `[H]` remains, so CDCl3 and CHCl3 are not merged (seen in the data: `HEDRZPFGACZZDS` group) | Low |
| `test_known_cache_detects_structure_change` | `features/__init__.py` | The feature cache is validated on compound ids only. If curation changes a structure but keeps its id, stale features come back silently. Today's caches are consistent (`test_stored_cache_matches_fresh_computation_real` passes for rdkit2d and morgan_counts) | Medium (reproducibility hazard) |
| `test_known_conformal_exact_finite_sample_coverage`, `test_known_conformal_quantile_order_statistic` | `uncertainty.py` | `ConformalRegressor.quantile` computes `np.quantile(scores, ceil((n+1)(1-a))/n, method="higher")`. This returns the (k+1)-th order statistic instead of the k-th (for example the 6th of 9 instead of the 5th). Intervals are slightly too wide, so coverage is conservative. Fix: `np.sort(scores)[k-1]` with `k = ceil((n+1)(1-a))` | Low (conservative; with n≈800 calibration points the effect is about 1/800 in quantile level) |

---

## 1. Duplicates across train and test (near-duplicate leakage)

**Method.** For every stored split, each test compound is compared with the **train** subset under three keys of
decreasing strictness:

1. `smiles_2d`, the de-duplication key.
2. `ik14`, the InChIKey connectivity block of the full standardised structure. It catches charge states without a counter-ion, tautomers and isotopologues.
3. `parent_ik14`, the connectivity block of the neutralised largest organic fragment. It catches a salt in test whose free acid or base is in train.

Validation is used only for early stopping, so val is not counted. Per-split rows are in
`tables/leakage_by_split.csv` and matched pairs in `tables/leakage_matched_pairs.csv`.

| split type | same smiles_2d | same ik14 (% of test) | same neutral parent (% of test) | LGBM MAE all | LGBM MAE on parent matches | LGBM MAE without parent matches |
|---|---|---|---|---|---|---|
| random (5 seeds, n_test 917) | 0 | 3.8 cpds (0.41 %) | 35.6 (3.88 %) | 0.574 | 0.746 | 0.567 |
| scaffold (5) | 0 | 0.4 (0.04 %) | 1.2 (0.13 %) | 0.717 | 0.982 (3 seeds with matches) | 0.716 |
| cluster (5) | 0 | 2.0 (0.22 %) | 20.8 (2.27 %) | 0.670 | 0.842 | 0.665 |
| LOSO (A,B,C,E,F) | 0 | 4.6 (0.56 %) | 30.2 (2.12 %) | - | - | - |
| external (2 directions) | 0 | 17.5 (0.91 %) | 41.0 (2.13 %) | - | - | - |

(`audit_numbers.json → leakage`. LGBM = `results/baseline` `lgbm__rdkit2d+morgan_counts` test predictions.)

**Exact duplicates.** There are none in any split, so the dedup key is applied correctly.

**Surviving charge-state and tautomer duplicates (`ik14`).**
* Inside AqSolDB there are 18 groups (36 compounds) that share an InChIKey connectivity block but not `smiles_2d` (`tables/within_aqsoldb_groups_ik14.csv`).
* 10 of the 18 are an acid (or base) next to the same compound drawn as a bare anion, cation or zwitterion, with a near-identical label (|ΔlogS| ≤ 0.1). Another 4 charge-state or isotopologue pairs differ by 0.18–0.80. Examples: `Cc1ccc(C(=O)O)cc1C` −3.160 vs its carboxylate −3.065, and N-acetyltryptophan −2.310/−2.312. These are the same measurement copied across AqSolDB sub-sources in a different protonation state, so they are **genuine leakage**.
* The remaining 4 are tautomers or charge forms with conflicting labels: valsartan 1H/2H-tetrazole 1.46, barbital keto vs enolate 1.35, amphotericin neutral vs zwitterion 1.00, chlordiazepoxide 1.21. These are noise, not leakage.
* Pooled over all splits there are 89 test–train `ik14` pairs, with median |ΔlogS| = 0.016 and 73 % below 0.3.
* **The effect on reported metrics is negligible.** It touches 0.41 % of random test sets (≤ 6 compounds). LGBM does not predict those compounds better than the rest (MAE 0.858 vs 0.574 overall), so they do not make the random-split MAE optimistic.
* **External splits.** ESOL→AqSolDB overlap was removed by `smiles_2d`, but 18 ESOL test compounds (aqsoldb_to_esol) and 17 (esol_to_aqsoldb) still have a charge or tautomer twin in the training set, with median |ΔlogS| 0.014 and 0.008. That is 1.6 % of the 1,115 ESOL test compounds carrying an essentially leaked label. The maximum optimistic bias is about 0.016 × MAE ≈ 0.01 log units.
* **Severity: Low. Fix:** de-duplicate (and remove external overlap) on `ik14` of the `Uncharger`-neutralised structure instead of `smiles_2d`.

**Salt vs free acid/base (parent-only matches).**
* 432 pooled test–train pairs share the neutral parent but not the full structure. Examples: sodium 4-hydroxybenzenesulfonate vs the free sulfonic acid, and calcium 2-methyloctanoate vs nonanoic acid.
* Their labels differ a lot: median |ΔlogS| = 0.72, mean 1.27, and 41 % differ by more than 1 log unit.
* Inside AqSolDB there are 185 such groups (444 compounds). 90 % contain a multicomponent member, and 87 groups span more than 1 log unit (`tables/within_aqsoldb_groups_parent_ik14.csv`).
* Salt formation legitimately changes solubility by orders of magnitude, so these pairs are **not leakage**. They are hard cases, not easy ones: LGBM MAE on them is 0.75 vs 0.57 (random) and 0.84 vs 0.67 (cluster). Removing them would *lower* the random MAE by 0.007, so they make results slightly pessimistic, not inflated.
* Caveat: some parent groups expose **representation errors in the source**. Pentaerythritol tetra-esters are encoded as disconnected "acid + alcohol" mixtures, which makes their largest fragment nonanoic acid (group `FBUKVWPVBMHYJY`, labels −8.77 to −2.41).
* **Severity: Low (no leakage).** Suggested fix: optionally report a "parent-grouped" random split as a robustness check.

## 2. Scaffold and cluster leakage, split shift

`tables/split_shift_and_similarity.csv`. Morgan r=2 / 2048-bit Tanimoto, maximum similarity of each test compound to train.

| split | Murcko scaffold in train | generic (CSK) scaffold in train | acyclic test cpds | train / test mean logS | train-mean predictor MAE | median max-sim | test max-sim = 1 |
|---|---|---|---|---|---|---|---|
| random | 84.8 % | 92.2 % | 26.3 % | −2.887 / −2.914 | 1.871 | 0.611 | 4.8 % |
| scaffold | **0 %** | 75.8 % (65–85 %) | **0 %** | −2.722 / **−3.576** | 2.070 | 0.458 | 0.9 % |
| cluster | 78.7 % | 88.7 % | 28.5 % | −2.889 / −3.096 | 1.971 | 0.483 | 0 % |

* **Scaffold split is correct.** No test Murcko scaffold occurs in train, and recomputing every stored scaffold from `parent_smiles` gives 0 mismatches.
* **Scaffold split is weaker than its name.** 76 % of test compounds share a *generic* carbon skeleton with train, for example a pyridine test compound whose benzene analogue is in train. It measures novelty of exact ring systems, not of chemotypes, so read the scaffold "out-of-distribution" gap (~0.57 → ~0.71 MAE) as a lower bound.
* **Scaffold split is confounded by label shift.**
  * All acyclic compounds (26 % of the data, mostly small and soluble) share the empty scaffold, so they always land in train.
  * Scaffold test sets are 0.85 log units less soluble than train on average (range of seed means −2.91 to −4.08). Even a train-mean predictor is 0.20 worse on them than on random test sets (2.070 vs 1.871).
  * Part of the random→scaffold MAE gap is therefore label/composition shift, not structural novelty.
  * Severity: **Medium** for interpretation. Fix: report the train-mean baseline per split (or MAE relative to it), and/or a scaffold split with acyclics distributed as their own pseudo-scaffold groups.
* **Cluster split.** Its own test sets share Murcko scaffolds with train for 78.7 % of compounds, yet are about as dissimilar as scaffold test sets by Morgan similarity (median max-sim 0.48 vs 0.46) and never contain a fingerprint-identical train neighbour (0 % vs 4.8 % random). This is the expected behaviour of Butina clustering at the configured cutoff, and it makes the cluster split the cleaner "similarity" split. No leakage.
* **Random split.** 4.8 % of test compounds have a fingerprint-identical train compound (stereo/positional isomers or homologues beyond radius 2). This is legitimate for a random split, but kNN/Tanimoto-based applicability-domain analyses should know that such compounds exist.

## 3. Butina clustering correctness

`audit_numbers.json → butina`.
* **Toy chain.** Fingerprints x⊂y, z, w are built with Tanimoto(x,y)=0.6, (y,z)=0.43, (z,w)=0.43, using the configured distance cutoff 0.6 (similarity ≥ 0.4). `solpred.splits.butina_clusters` returns `[0,0,0,1]`, as expected by hand: y has the most neighbours, takes x and z, and w is left alone. Members join only the centroid's cluster (no chaining).
* **Reference.** On the first 1,500 AqSolDB compounds the partition is **identical** to `rdkit.ML.Cluster.Butina.ClusterData`, once RDKit's high-index tie-break is aligned (by reversing the list).
* **Stored file.** A full re-computation over 9,171 compounds (103 s) reproduces `data/splits/aqsoldb_butina_clusters.csv` **exactly**.
* **Shape.** 2,569 clusters, 1,446 singletons (15.8 % of compounds), largest cluster 263, 5 clusters ≥ 100.
* **Verdict: correct.** Also covered by `tests/test_splits.py`.

## 4. Removed compounds (`data/processed/aqsoldb_removed.csv`)

521 rows were removed. A seeded sample of up to 30 per reason, with an automatic chemistry annotation, is in `tables/removed_sample.csv`.

| reason | n | automatic annotation | verdict |
|---|---|---|---|
| no_carbon | 325 | 325/325 truly carbon-free (e.g. NdOH3, RhI3, AlTi3Cl12, Ca(H2PO4)2) | justified |
| disallowed_element | 194 | 153 ionic salts of disallowed ions (Zn 33, Cu 16, Co 16, Cr 12, Fe 11, Ba 10 … dyes and pigments), 32 organometallics with a metal–C bond (Sn 21: tributyltin oxide, dibutyltin diacetate), 9 covalent metal complexes | justified for a small-organic benchmark; organotins are arguably organic, but RDKit descriptors handle Sn poorly |
| parse_error | 2 | B-1866 / B-1579: pyridine N-oxides mis-drawn as `[NH++]([O-])[CH-]` | justified (unparseable as given); recoverable by hand. The annotation "ionic salt" in the table is a fallback label of the annotator, not chemistry |

* No sampled removal lost a compound that should have been kept, and the removal reasons agree with the structures. **Verdict: justified.**
* Kept edge cases (`tables/kept_inorganic_carbon.csv`, 48 compounds with ≤ 1 carbon and no C–H): urea, CS2, CF4, CO, carbonates, cyanate, Li triflate, and **boron carbide** `B12B3B4B1C234` (a ceramic). Most are reasonable small molecules. Carbonates, CO and boron carbide are inorganic and could be dropped.
* **Severity: Low.**

## 5. Suspicious labels

`tables/suspicious_labels.csv` (460 rows, 5.0 % of the data flagged by at least one rule). Figure: `figures/label_plausibility.png`.

* **logS > 1.5 (13 compounds).** Examples: hexane-1,6-diol 1.63, acetamide 1.58, dimethylamine 1.56, HCN 1.57, azanium glycolate 1.70, 4-methyl-2-isobutyl-oxan-4-ol 2.14, bis(THP) sulfate 1.97.
  * A physical bound applies: logS ≤ log10(1000·ρ/MW) for a pure liquid/solid with density ρ.
  * **27** compounds exceed it even at ρ = 2 g/mL, and 41 at ρ = 1.2.
  * These are mostly fully miscible liquids whose "solubility" was recorded as an arbitrary large mass concentration (for example 1000 g/L). They are not wrong measurements but **not solubilities**. The same holds for the Si-PEG compound A-1490 (1.58, bound 0.60).
  * Severity **Low** (0.3 % of data, at the edge of the range). Fix: cap at the bound or flag as "miscible".
* **logS < −10 (19 compounds).** Polychlorinated dioxins, furans, biphenyls and diphenyl ethers (e.g. OCDD −12.06, dechlorane plus −13.17, decachlorodiphenyl ether −12.95).
  * These are chemically plausible (hydrophobic, high-melting) but sit near analytical detection limits.
  * 9 of the 19 have several concordant measurements (n_occurrences 2–3), so they are real.
  * Severity **Low**: keep them, but note that they dominate squared errors (RMSE) in the insoluble tail.
* **Clearly wrong labels found during the audit.**
  * Sodium 4-hydroxybenzenesulfonate (logS −8.36) next to its free acid (+0.76). A sodium aryl-sulfonate cannot be 10⁻⁸ M soluble.
  * Mg dimethanolate −3.99 vs methanol +1.49. The alkoxide hydrolyses, so this is reasonable but the label is not methanol's.
  * Pentaerythritol esters encoded as mixtures (§1).
* **Broader screen.** Single-component compounds with |logS − Delaney ESOL equation| > 3: 276 (3.3 % of 8,472 single-component compounds; residual MAE 0.99, SD 1.34).
* **Severity overall: Medium.** Single gross label errors exist (about 1–2 % of the data by the screens above). They add to the noise floor on every test MAE: the pooled inter-measurement SD is 0.478 (curation log), and label noise in test sets inflates MAE for *all* models equally. They do not bias model comparisons.
* **Fix:** a sensitivity analysis that drops the 460 flagged compounds from the *test* sets only and re-scores existing predictions. This is cheap and needs no retraining.

**Near-duplicates with wildly different labels.**
* 124 compounds have AqSolDB `aqsol_sd > 1` across their source measurements (reliability G2/G4).
* 20 of our own stereo/representation merges span more than 1 log unit (`tables/dedup_merged_groups.csv`). Of the 257 merge groups, 233 merged a defined-stereo record with an undefined one (median range 0.014, which is fine). 24 merged **distinct defined stereoisomers** (median range 0.215; 5 flagged `dedup_conflict`). Diastereomers and racemate/enantiomer pairs can differ genuinely, but the effect is small.
* 87 parent groups span more than 1 log unit (§1), which is legitimate.
* Severity **Low**.

**Repeated labels.** 202 logS values are shared by at least 5 compounds (top: −2.68 ×16, −2.70 ×15). That is consistent with rounding in the original sources, and nothing suggests copy errors at scale.

## 6. The 17 InChIKey-inconsistent records, and charge artefacts

`tables/inchikey_inconsistent.csv`. In all 17 the raw AqSolDB SMILES matches its own InChIKey. The *standardiser* changed the structure. Classified by hand:

* **Benign salt formation (9).** A neutral metal or halogen written covalently (`[Na].acid`, `[K].oxalic acid`, `[Mg]×3.citric acid`, `[Li].triflic acid`, `[Cl].amine`) became an ion pair.
  * A-2987, A-3003, A-3289, A-3648, A-3771, A-5156, A-4061 (guanidinium benzoate) are correct.
  * A-4216 gained an unbalanced Cl⁻ with no counter-cation.
  * A-1819 "octadecanol phosphate K salt" is already wrong in the source (octadecane + H3PO4).
* **Benign tautomerisation (2).** B-1228 and B-4603: hydroxyquinoline zwitterions became neutral quinolinone tautomers. Acceptable, but the hydroxy form would be standard.
* **Wrong structures introduced (6).**
  * Carbanions: A-1921 (Na dodecylbenzenesulfonate became `[CH2-]` dodecyl…), A-3392 (tert-butyl `[CH2-]`), A-4434 (Ca acetylacetonate, one ligand got a terminal `[CH2-]`).
  * Misplaced alkoxides: A-3339 (ascorbate charge moved to the CH2-O), A-4404 (two hydroxyethyl O⁻ instead of sulfonates).
  * A-5194: thiolate vs alkoxide swap. This one is chemically defensible.
  * Across the whole set, the audit finds 6 carbanions and 2 created alkoxides (`tables/charge_artefacts.csv`).
* **Impact.** These are 6 of 9,171 compounds (0.07 %), and only descriptor values of large surfactant/dye salts are affected. **Severity: Low.**
* **Fix.** In `curate.py`, connect neutral metals only when an acidic X–H exists. Otherwise keep the metal neutral, or drop the record.
  `test_known_no_carbanion_from_neutral_metal` tracks this.

**Other charge findings (all Low).**
* 43 HCl salts drawn with a bare `[H+]` fragment, kept verbatim (e.g. A-775 `NNCc1ccccc1.[Cl-].[Cl-].[H+].[H+]`).
* 83 net-charged single species (quaternary ammoniums or anions without counter-ion, most frequent in source F at 2.6 %), 42 single-fragment zwitterions, and 15 charge-unbalanced salts (e.g. `N#CN.[Ca+2]`).
* For these, Crippen logP is unchanged by RDKit's `Uncharger` (mean |Δ| = 0.00, n = 125), because quaternary centres and zwitterions cannot be neutralised. Descriptors are therefore not distorted by an inconsistent neutralisation.

## 7. Descriptor NaNs (rdkit2d)

`tables/rdkit2d_nan_rows.csv`; `audit_numbers.json → descriptor_nans`.

* **Where the NaNs are.** 399 compounds (4.35 %) have at least one NaN, which is 0.16 % of all cells.
  * The NaNs sit almost entirely in the 8 `BCUT2D_*` columns (399 rows each) and the 4 Gasteiger `*PartialCharge` columns (7 rows).
  * **All 390 metal-containing compounds** (Li/Na/K/Mg/Ca salts) have NaN BCUTs, because RDKit has no Crippen/Gasteiger parameters for the metal cations. Only 9 non-metal compounds are affected (6 organoselenium, CO, BF3-amine, PF6⁻).
  * 98 % of NaN rows are multicomponent.
* **Bias risk.** NaN rows are much more soluble (mean logS −1.10 vs −2.97). Missingness is therefore informative: with LightGBM's native NaN handling, "BCUT is missing" acts as an "is a metal salt" flag.
  * That is legitimate (it is derivable from the structure, and test compounds are featurised identically), not leakage.
  * Median-imputing models (RF, ridge, MLP via `SimpleImputer`) lose that signal, and this may contribute to any LGBM-over-RF advantage on salts (not tested here).
  * Two columns are constant (`SMR_VSA8`, `SlogP_VSA9`) and are dropped by the `VarianceThreshold` in linear/MLP pipelines.
* **Severity: Low.** Fix: compute descriptors on the parent fragment and add explicit counter-ion features (`n_fragments`, metal flags), or add a missing-indicator before imputation.

## 8. Other things that could inflate or bias results

* **LOSO labels are not single-source.** AqSolDB's merged labels average measurements from several sources. The LOSO test fractions whose label comes from multiple measurements are: A 28 %, B 16 %, C 19 %, **E 65 %**, F 9 %.
  * A compound assigned to source E is excluded from train, so there is no structural leakage. But its label partly comes from other labs, so LOSO-E measures less inter-laboratory transfer than its name implies.
  * Severity **Medium** for LOSO interpretation (especially E, n_test = 207). Fix: report LOSO on single-measurement (G1) test compounds as well.
* **ESOL is not an independent external set.** 98.4 % of ESOL compounds were also in AqSolDB (curation facts). The external split removes them from training, which is correct (0 exact matches; 18 `ik14` twins remain, §1), but ESOL→AqSolDB performance is *not* evidence of generalisation to new chemistry. It is a dataset-shift test.
* **Val leakage into model selection.** Every split has a val subset used only for LightGBM early stopping (`experiment.run_split`). Test was never used for selection in the code paths audited (`tests/test_splits.py` checks that every stored split is disjoint and complete, and `tests/test_experiment.py::test_lgbm_early_stops_on_val` checks that early stopping happens on val).
* **Feature cache staleness** (§0, `test_known_cache_detects_structure_change`). Today's caches equal fresh computation, but any future re-curation must delete `data/features/*.npz`.
* **Conformal quantile off-by-one** (§0). This is conservative, so reported coverages are, if anything, slightly above nominal for the wrong reason (≈ +1/n_cal).

## Bottom line

* **No finding inflates the headline metrics** by more than about 0.01 log units in MAE.
  * Exact duplicates across train and test: 0.
  * Charge/tautomer twins: ≤ 0.4 % of random test sets, and not predicted better than other compounds.
  * Salt/free-parent pairs: 3.9 % of random test sets. They are predicted *worse* than average, so they add difficulty, not leakage.
* The biggest interpretation risks are:
  1. the scaffold split mixes structural novelty with a 0.85 log-unit label shift (no acyclics in test);
  2. about 1–2 % of labels are not real solubilities (miscible liquids above the physical bound, mis-encoded mixtures, an obviously wrong sulfonate salt), which sets a noise floor on all MAEs;
  3. LOSO-E labels are mostly multi-source averages.
* Curation, Butina clustering, scaffold computation and the feature caches are correct to the extent tested. Six strict xfail tests pin the known limitations.
