# Literature review: predicting aqueous solubility (logS) from structure

*Scope.* This review covers the parts of the literature that shape this project's design: why solubility
matters, how reliable its measurement is, which public datasets exist and how they relate, which model families
have been used and what they achieve, and how the evaluation protocol (splitting, applicability domain,
uncertainty) changes the conclusions. Numbered citations refer to `reports/references.md`. Every bibliographic
entry and every quoted performance figure was checked against the primary source, or is explicitly marked as
reported by a secondary source. The verified letter-to-source mapping for AqSolDB is in `docs/aqsoldb_sources.md`.

---

## 1. Why aqueous solubility matters in drug discovery

**Absorption.** In the Biopharmaceutics Classification System (BCS), drug dissolution and gastrointestinal
permeability are the two fundamental parameters that control the rate and extent of oral absorption. Drugs fall
into four classes defined by high or low solubility and high or low permeability [1]. Low solubility can
therefore limit exposure independently of potency or permeability. Lipinski et al. [2] described turbidimetric
(kinetic) solubility screening in a discovery setting. They observed that leads from high-throughput screening
(HTS) tend to have higher molecular weight and logP, and lower turbidimetric solubility, than leads from the
pre-HTS era. They also noted that exact solubility prediction in development is hampered by polymorphism.

**Development and formulation.** Poor solubility causes problems in both *in vitro* and *in vivo* assays and
adds a substantial burden to development. Discovery and development also need different solubility assays [3].
Barrett et al. [4] argue that "solubility" is not a single measurement and that an artificial boundary at the
discovery–development interface hinders data sharing. A model trained on pooled literature data inherits that
heterogeneity (Section 2).

**Assay artefacts.** Compounds that are insoluble in assay buffer or in DMSO stocks produce underestimated
activity, reduced HTS hit rates, variable data, inaccurate structure–activity relationships, disagreements between
enzyme and cell assays, and inaccurate *in vitro* ADME-Tox results [5]. A solubility model therefore serves two
purposes: it ranks compounds for synthesis, and it flags compounds whose other assay data may be unreliable.

**What accuracy is useful.** Jorgensen and Duffy [6] concluded that methods with less than one log unit of
uncertainty are adequate for prescreening synthetic candidates or designing libraries. They added that further
progress would need a database of highly accurate solubilities for a large, diverse set of drug-like molecules.
AqSolDB uses coarse solubility classes: logS ≥ 0 highly soluble, 0 to −2 soluble, −2 to −4 slightly soluble,
< −4 insoluble [7]. These classes match how predictions are actually used, so we report class-level agreement
alongside MAE.

## 2. Measurement and its variability

**Thermodynamic versus kinetic solubility.** Thermodynamic (equilibrium) solubility is the maximum concentration
of a compound in solution at equilibrium with its most stable crystalline form [8]. Kinetic solubility comes from
high-throughput precipitation assays, such as the turbidimetric assay of [2]. Alsenz and Kansy review those assays
and their trade-offs between speed and data quality [9]. Saal and Petereit [10] measured 465 research compounds in
both kinds of assay. Kinetic assays often gave considerably higher values, because they usually measure the
amorphous rather than the crystalline solid. Thermodynamic data, ideally with a solid-state assessment, were the
better guide for optimisation.

**Intrinsic, apparent and water solubility.** Llompart et al. [8] distinguish three "thermodynamic" quantities.
*Water* solubility is measured in unbuffered water, so the solution self-buffers. *Apparent* (buffer) solubility is
measured at a fixed pH. *Intrinsic* solubility (S0) is that of the neutral species. The three are related through
the Henderson–Hasselbalch equation, under assumptions discussed at length by Avdeef [11]. For ionisable drugs the
quantities can differ by orders of magnitude when the pH is far from the pKa. Large public compilations, including AqSolDB and OChem, are
"undefined mixtures" of intrinsic, apparent and water solubility [8]. The distributed AqSolDB file does not
record temperature, pH, solid form or assay type for individual entries (`docs/data_card.md`).

**Temperature and solid form.** AqSolDB filtered its two largest sources (eChemPortal and the EPI Suite WATERNT
set) to 25 ± 5 °C, and the Raevsky set was reported at 25 ± 5 °C. The Huuskonen data cover 20–25 °C, and the
Delaney and Goodman sets were measured at 25 °C [7]. Llinàs et al.
measured at 25 °C and an ionic strength of 0.15 M [12]. The window of up to 10 °C is therefore a recognised but
unquantified source of spread. Solid form matters more:

* Solubility ratios between polymorphs are typically below 2 (< 0.3 log units), with occasional larger values.
  Anhydrate/hydrate ratios are more spread out [13].
* For amorphous solids, thermodynamics predicts a 10- to 1,600-fold advantage (1–3.2 log units). The measured
  advantage is usually much smaller, partly because equilibrium is hard to reach [14].
* Attia et al. [15] stress that solubility errors are often systematic rather than random. Solids are frequently
  isolated as amorphous material, hydrates, polymorphs or impure co-crystals instead of the most stable pure crystal.

**Inter-laboratory reproducibility.** The table lists the estimates most often cited. They split into two regimes.

| Study | Data | Estimate (log units) |
|---|---|---|
| Palmer & Mitchell 2014 [16] | literature compilations for drug-like molecules vs single-method CheqSol | 0.6–0.7 (literature); typically < 0.05 (CheqSol) |
| Llinàs & Avdeef 2019; Llinàs, Oprisiu & Avdeef 2020 [17, 18] | curated multi-lab, mostly shake-flask, intrinsic solubility | ≈ 0.17 ("tight" set); ≈ 0.62 ("loose", contentious set) |
| Avdeef 2019 [19] | Wiki-pS0 intrinsic solubility: 233 CheqSol values (145 drugs) vs 838 mostly shake-flask values (124 of them) | SDi 0.15 (CheqSol), 0.24 (shake-flask); 0.17 averaged over 870 multi-source molecules; single-lab CheqSol errors (0.05) need × 3 to match inter-lab spread |
| Katritzky et al. 1998; Hughes et al. 2008, as collated by Attia et al. [15, 20, 21] | heterogeneous aqueous data | 0.58 (411 compounds); 0.60 |
| Sorkun et al. 2019, 2021 [7, 22] | AqSolDB sub-datasets | 0.5 adopted as the agreement threshold; per-source SD 0.274 (Huuskonen) to 0.717 (eChemPortal); 0.495 overall |
| This project (`docs/data_card.md`) | curated AqSolDB, 2,065 multi-occurrence compounds | RMS SD 0.478, median 0.124 |

Careful single-laboratory or multi-laboratory measurements of drug intrinsic solubility therefore reproduce to
about 0.15–0.25 log units. Heterogeneous compilations of the AqSolDB type carry about 0.5–0.7 log units of
between-source disagreement. In our data the distribution is heavy-tailed: the median SD (0.124) is a quarter of
the RMS SD (0.478). Most repeated values agree closely, and a minority disagree by more than a log unit.

**Implication for evaluation.** Sorkun et al. [22] separate a model's *actual* error from its *observed* error.
Because the test labels themselves are noisy, the observed error of even a perfect model equals the internal error
of the test set. If label noise were Gaussian with σ ≈ 0.48, the floor would be an RMSE of about 0.48 and an MAE
of about 0.8σ ≈ 0.38 log units. This is a heuristic only, since the noise of the 7,106 single-measurement (G1)
compounds is unobserved.

**Does data quality limit accuracy?** The evidence is mixed:

* Palmer and Mitchell [16] trained the same QSPR methods on low-noise CheqSol data and on noisy literature data.
  The CheqSol models were no more accurate, which suggests that model or descriptor deficiencies, not data noise,
  limited prediction for drug-like molecules.
* In the first Solubility Challenge, entrants predicted 32 intrinsic solubilities from 100 CheqSol training values
  [12, 23]. As summarised by Llompart et al. [8], only about a third of the compounds were predicted correctly and
  the best RMSE was around 0.6 log units.
* The second challenge let participants use their own training sets [17, 18]. According to [8], it showed no
  significant improvement, with a minimum RMSE of about 0.70. Panapitiya et al. [24] report mean competitor RMSEs
  of 1.14 and 1.62 on the tight and loose sets.
* Boobier et al. [25] and Attia et al. [15] argue that well-designed models now approach the noise level of the
  available test data.

The two positions apply to different regimes: precise drug-like intrinsic data and large heterogeneous
compilations.

## 3. Datasets

**Huuskonen (2000).** Huuskonen [26] modelled 1,297 diverse organic compounds with values from the AQUASOL and
PHYSPROP databases [7]. The data were randomly divided into 884 training and 413 test compounds. A 30-12-1 neural
network on 24 atom-type E-state indices and six topological indices achieved test r² = 0.92 and s = 0.60. Multiple
linear regression gave r² = 0.88 and s = 0.71. This set became the standard benchmark of the following decade, and
it is subset E of AqSolDB.

**Delaney's ESOL (2004).** Delaney [27] fitted ESOL by linear regression against nine candidate properties on
2,874 measured solubilities. In order of importance, the four retained terms are calculated logP, molecular weight
(MW), the proportion of heavy atoms in aromatic systems (AP) and the number of rotatable bonds (RB). The published
equation, as reproduced in [28] (the original article is paywalled), is

  logS = 0.16 − 0.63 clogP − 0.0062 MW + 0.066 RB − 0.74 AP

with clogP from the Daylight toolkit. ESOL predicted the three validation sets within a factor of 5–8
(≈ 0.7–0.9 log units) and was competitive with the General Solubility Equation (GSE) for medicinal and
agrochemical-sized molecules [27]. Walters' refit with RDKit descriptors gave broadly similar coefficients, with
the largest change in the rotatable-bond term (0.066 → 0.0034), which he attributes to differing rotatable-bond
definitions [28].

MoleculeNet [29] distributes 1,128 Delaney compounds as the "ESOL" regression task, with a random split and RMSE as
the recommended protocol. Its benchmark gives a best conventional model of RMSE 0.99 (XGBoost) and a best graph
model of 0.58 (MPNN), both on random splits. Delaney's set is also subset G of AqSolDB. After curation, 1,097 of
our 1,115 ESOL compounds (98.4 %) also occur in AqSolDB (`data/processed/curation_log.json`). ESOL is therefore
not an independent external test for an AqSolDB-trained model unless the overlap is removed.

**AqSolDB (2019).** Sorkun, Khetan and Er [7] merged nine public datasets into 19,796 instances covering 9,982
unique compounds, identified by InChI:

* A: eChemPortal, 6,110 entries after pre-processing.
* B and D: two US EPA EPI Suite Data sets (the WATERNT and WSKOWWIN methodology and validation files), 4,651 and
  2,115.
* C: Raevsky et al. [30], 2,603.
* E: Huuskonen, 1,291.
* F and H: two sets from Wang et al. [31], 1,210 and 578; H is their de-duplicated version of the Jain–Yalkowsky
  set [32].
* G: Delaney, 1,144.
* I: the Goodman/Llinàs challenge data [12], 94.

The letter on each AqSolDB ID records the source of the *selected* value. For compounds with three or more
distinct values, the value closest to the mean is kept. For compounds with exactly two, the value closest to an
ALOGPS prediction is kept. Compounds are then assigned to reliability groups G1–G5 using an SD threshold of 0.5.
The file also includes 17 RDKit 2D descriptors. Section 2 of `docs/aqsoldb_sources.md` discusses the consequences
of the selection rule. For example, the labels of 14.6 % of our curated compounds were chosen by proximity to a
model prediction.

The same group later measured per-source SDs [22]. Huuskonen (0.274) was the most reproducible source, and
eChemPortal (0.717) and Wang/Beilstein (0.582) the least. Training AqSolPred (a consensus of a neural network,
random forest and XGBoost) without A and F, then testing on the 1,290 Huuskonen compounds, gave MAE 0.397,
RMSE 0.539 and R² 0.93. Leave-one-out with Huuskonen data included gave MAE 0.348 and RMSE 0.483 [22]. This
quality-oriented selection is a source-level holdout, and it inspired the leave-one-source-out (LOSO) splits used
here.

**TDC Solubility_AqSolDB.** Therapeutics Data Commons [33] redistributes the 9,982 AqSolDB compounds as an ADMET
benchmark: a scaffold split holding out 20 % for test, with MAE as the metric. On the public leaderboard (accessed
2026-10-13, 18 entries), MiniMol leads with MAE 0.741 ± 0.013. It is followed by Chemprop-RDKit (0.761 ± 0.025),
DeepMol AutoML (0.775 ± 0.006) and AttentiveFP (0.776 ± 0.008). RDKit2D + MLP scores 0.827 ± 0.047, plain Chemprop 0.829 ± 0.022, and
Morgan + MLP 1.203 ± 0.019. The top ten entries span less than 0.09 log units of MAE.

**Larger or re-curated compilations.**

* Meng et al. [34] curated seven aqueous datasets and found lower RMSE on every curated thermodynamic set.
* Panapitiya et al. [24] assembled 17,149 molecules.
* Llompart et al. [8] re-curated AqSolDB with explicit rules, including removing entries with inter-laboratory
  SD > 0.5. This lowered RMSE by about 0.10 log units. When they applied published models to an OChem-derived set
  outside their training space, the best model (AqSolPred) reached only R² = 0.56. They concluded that models
  without a defined applicability domain are not ready for public use.

## 4. Methods

**Physically motivated equations.** Yalkowsky and Valvani's General Solubility Equation [35], revised by Jain and
Yalkowsky [32], is

  log Sw = 0.5 − 0.01 (MP − 25) − log Kow

where MP is the melting point in °C [8]. The revised GSE was validated on 580 non-electrolytes [32] and uses no
fitted parameters [36]. On 150 compounds studied by Jorgensen and Duffy it gave an average absolute error of 0.43,
against 0.56 for their five-parameter model [36]. With experimental inputs it typically reaches an RMSE of 0.7–0.8, but it
applies to non-electrolytes and needs a melting point [8]. That is a real limitation for virtual compounds, since
melting-point prediction itself is poor (RMSE 52.8 °C in [20]). ESOL [27] replaces the melting point with
structure-only terms and serves as our interpretable linear baseline.

**Descriptor and fingerprint machine learning.**

* *Representations.* Extended-connectivity fingerprints (ECFP) [37] and physicochemical or topological
  descriptors are the standard inputs. Random forests [38], introduced to QSAR by Svetnik et al. [39], and
  gradient-boosted trees (XGBoost [40], LightGBM [41]) are the usual learners.
* *Support vector machines.* Hughes et al. [20] found support vector machines on 2D/3D MOE descriptors best for
  logS, with external RMSE 0.90. They identified sparse data at the extremes of the range as a major error source.
* *Intrinsic solubility.* Lovrić et al. [42] modelled 829 drug-like compounds. LASSO reached test RMSE 0.70, and a
  random forest with only 17 features reached 0.72. The random forest generalised better on a harder PCA-based split.
* *Descriptors versus learned representations.* Panapitiya et al. [24] compared descriptors, SMILES, graphs and 3D
  coordinates across four network types. Descriptor-based models performed best, and graph networks were close
  behind. Deng et al. [43] trained 62,820 models on MoleculeNet and related data. Representation-learning models
  showed limited advantage over fixed representations in most datasets, and dataset size was essential for them to
  excel.
* *Prospective evidence.* Fang et al. [44] collected 120 prospective internal datasets over 20 months, covering six
  ADME endpoints including solubility. Gradient-boosted trees and deep models consistently beat random forests over
  time, and regular retraining mattered more than hyperparameter tuning.

**Graph neural networks and transformers.**

* *Message passing.* Gilmer et al. [45] framed message-passing neural networks. Yang et al. [46] introduced the
  directed MPNN (D-MPNN; Chemprop [47]) and benchmarked it on 19 public and 16 proprietary datasets. It matched or
  beat fixed-descriptor models, but the authors noted that such models "have yet to reach the level of experimental
  reproducibility".
* *Attention.* AttentiveFP [48] adds graph attention for interpretability. SolTranNet [49] is a molecule-attention
  transformer for solubility. Its final model has only 3,393 parameters because larger models performed worse. Its
  3-fold scaffold-split cross-validation RMSE on AqSolDB is 1.459, rising to 1.711 on a withheld test set. Used as
  a filter for insoluble compounds, it had 94.8 % sensitivity on the second-challenge data.
* *Hybrid physics and machine learning.* Boobier et al. [25] combined a small set of descriptors from computational
  chemistry (solvation energies from DFT and HF/SMD) and the experimental melting point with six learners. For most
  models about 70 % of aqueous predictions fell within ±0.7 log units, "as accurate as the training data".
  Ghanavati et al. [50] build an ensemble of a GCN on molecular graphs, EdgeConv on electrostatic-potential (ESP)
  maps and XGBoost on ESP and Mordred descriptors. Attia et al. [15] (organic solvents, FASTSOLV/Chemprop) report
  extrapolation to unseen solutes that approaches the aleatoric limit of 0.5–1 logS.

**Reported performance depends on the split.**

| Model | Data / test set | Split | Metric | Value | Ref. |
|---|---|---|---|---|---|
| ANN (E-state) | Huuskonen, 413 test | random | s | 0.60 | [26] |
| Revised GSE (no fitting) | 150 compounds | — | AAE | 0.43 | [36] |
| XGBoost / MPNN | MoleculeNet ESOL | random | RMSE | 0.99 / 0.58 | [29] |
| SVM (MOE 2D/3D) | external test | not stated in abstract | RMSE | 0.90 | [20] |
| LASSO / RF | 829 intrinsic | external test | RMSE | 0.70 / 0.72 | [42] |
| AqSolPred consensus | AqSolDB without A, F, E → E (1,290) | source holdout | MAE / RMSE | 0.397 / 0.539 | [22] |
| GNN | 2nd challenge, tight / loose | external | RMSE | 0.91 / 1.17 | [24] |
| SolTranNet | AqSolDB | 3-fold scaffold CV | RMSE | 1.459 | [49] |
| MiniMol / Chemprop-RDKit / AttentiveFP (ranks 1, 2, 4) | TDC AqSolDB | scaffold, 20 % test | MAE | 0.741 / 0.761 / 0.776 | [33] |

Nominal errors range from about 0.4 to 1.5 log units across these rows. The dataset, the split and the metric
explain more of that range than the architecture does. Within a fixed protocol such as the TDC leaderboard,
the ten leading architectures differ by less than 0.1 log units. That gap is smaller than the inter-laboratory noise discussed in
Section 2.

## 5. Splitting, generalisation, applicability domain and uncertainty

**Scaffold and cluster splits.**

* *Scaffold splits.* Bemis and Murcko [51] reduced molecules to ring-and-linker frameworks. In 5,120 drugs they
  found 1,179 graph frameworks, and the 32 most frequent described half of the drugs. MoleculeNet [29] and Chemprop
  [46] popularised scaffold splits as a proxy for prospective (temporal) evaluation. Yang et al. found the scaffold
  split a reasonable approximation of the temporal split. However, on Amgen's solubility data the chronological
  split was significantly harder than both random and scaffold splits, and scaffold was significantly harder than
  random on solubility only [46].
* *Cluster splits.* Butina's sphere-exclusion clustering on fingerprint Tanimoto similarity [52] is the usual way
  to build cluster-level splits.

**Critiques.**

* Sheridan [53] showed that random selection overestimates prospective R². A leave-class-out analogue
  underestimates it, and time-split selection comes closest.
* Wallach and Heifets [54] showed that train–validation redundancy (the AVE bias) strongly predicts benchmark
  performance.
* Guo et al. [55] found that scaffold splits still overestimate performance, because molecules with different
  scaffolds are often highly similar. UMAP-based clustering splits gave much worse performance on 60 NCI-60 datasets.
* Deng et al. [43] note that a scaffold does not necessarily map to a property value.
* Tossou et al. [56] characterise deployment shift by the distribution of distances to the training set. Such
  shifts degraded performance by up to 60 % and uncertainty calibration by up to 40 %. They propose distance-based
  splitting protocols.
* Lo-Hi [57] builds hit-identification and lead-optimisation splits and finds existing benchmarks unrealistic and
  over-optimistic. SIMPD [58] generates simulated time splits of public data that reproduce the property and
  performance differences of real temporal splits more closely than random or neighbour splits.

**Scaffold splits on solubility data.** A problem specific to solubility is worth stating. All acyclic compounds
share the empty scaffold, so a size-ordered scaffold split places them all in training. In our splits the scaffold
test sets contain 0 % acyclic compounds, and the test-set mean logS ranges from −2.9 to −4.1 against about −2.7 in
training (`data/splits/split_summary.csv`). Part of the "scaffold penalty" is therefore label shift towards
insoluble, polycyclic molecules, not structural novelty alone. A similarity-aware benchmark has to separate the two.

**Applicability domain (AD).** Sheridan et al. [59] showed that similarity to the nearest training molecule, and
the number of training neighbours above a similarity cutoff, best discriminate prediction accuracy. The trend does not
depend on the QSAR method or descriptor. It is strongest for narrow training sets and holds to a lesser degree for
diverse ones. Tropsha and co-workers [60, 61] made rigorous external validation, and an explicit AD for predicting
external compounds, central to QSAR best practice. The OECD validation principles require an AD. Sahigara et al.
[62] compare range-based and geometric (bounding box, convex hull), distance-based (including k-nearest-neighbour
and leverage) and probability-density definitions. For solubility,
Llompart et al. [8] identify the missing AD as the main barrier to public use of current models.

**Uncertainty quantification (UQ).**

* *Deep ensembles.* Lakshminarayanan et al. [63] proposed ensembles as a simple and scalable estimator of
  predictive uncertainty. Random-forest tree variance is the corresponding baseline for tree models.
* *Conformal prediction.* Norinder et al. [64] introduced conformal prediction to QSAR as an alternative to AD
  estimation. It gives prediction intervals with a user-chosen error rate (confidence 0.8 means at most 20 % of
  true values fall outside), valid under exchangeability [65].
* *Benchmarks.* Hirschfeld et al. [66] compared UQ methods on five regression datasets. None was unequivocally
  superior, and none ranked errors reliably across datasets.
* *Under shift.* Calibration should therefore be assessed per split type, because coverage guaranteed under random
  splitting need not hold under distribution shift [56].

**Statistical practice.** Ash et al. [67] propose guidelines for statistically rigorous method comparison with
domain-appropriate metrics in small-molecule property modelling, with annotated open-source examples. Their
motivation applies here: ML predictions drive costly decisions such as synthesis and *in vivo* studies. In this
project we therefore aggregate over five seeds and use paired bootstrap tests on shared test sets, so that
differences smaller than the resampling and label noise are not reported as gains.

## 6. Research gap, assumptions and intended contribution

**Gaps in the reviewed literature.**

* Headline solubility results are mostly reported for one protocol: random splits of the 1,128-compound ESOL set
  [29], one scaffold split [33, 49], or a single external set [22]. Random, scaffold, cluster, source and external
  splits are rarely reported side by side on the same curated data with several seeds. As a result, split effects
  are confounded with architecture effects that are themselves smaller than 0.1 log units [33].
* Few studies tie the reported error to an error floor estimated from the training database's own replicate
  measurements. Sorkun et al. [22] and Llompart et al. [8] are partial exceptions. Results are seldom expressed as a
  function of test-to-train similarity, although that is the variable that predicts error [59] and defines shift [56].
* The two most used public sets are nested: ESOL ⊂ AqSolDB for 98.4 % of ESOL compounds. "External" ESOL
  validation of AqSolDB-trained models is optimistic unless the overlap is removed.
* AqSolDB's sources differ in noise by a factor of about 2.6 (SD 0.274 to 0.717 [22]). Its label-selection rule
  partly depends on a model (ALOGPS) [7]. Source effects and label provenance are rarely analysed.
* Applicability-domain and uncertainty estimates for solubility models are seldom validated under the split that
  defines the intended use [8, 56, 66].

**Working assumptions.**

* AqSolDB logS approximates aqueous solubility at about 25 °C. It is a mixture of intrinsic, apparent and water
  solubility with unknown solid forms [7, 8]. We treat the residual as label noise with an effective SD of about
  0.5 log units [7, 22] (0.478 RMS in our curated multi-occurrence subset), and we expect test MAE floors on the
  order of 0.4 log units.
* 2D structure is the input. Stereoisomers are merged, salts are kept as salts, and tautomers are not
  canonicalised (`docs/data_card.md`). Stereo- and form-dependent effects are part of the noise.
* ECFP4 maximum Tanimoto similarity to the training set is a usable proxy for domain shift [56, 59]. We check this
  against scaffold and cluster definitions rather than assume it.

**Intended contribution.**

* A similarity-aware benchmark on curated AqSolDB with random, Bemis–Murcko scaffold and Butina cluster splits
  (5 seeds each), LOSO splits for sources A, B, C, E and F, and AqSolDB → ESOL external validation with the 1,097
  overlapping compounds removed from training. Results are always reported side by side.
* Baselines from the ESOL equation to random-forest, LightGBM and neural models. Errors are stratified by
  test-to-train similarity, reliability group and source, and compared with the replicate-noise floor.
* Separation of structural novelty from label shift in scaffold splits, by comparing scaffold, cluster and
  similarity-matched subsets.
* Evaluation of applicability-domain (similarity and neighbour count) and uncertainty (tree or ensemble variance,
  conformal intervals) estimates under each split type, with coverage reported per split.
* A documented data provenance (`docs/aqsoldb_sources.md`, `docs/data_card.md`) that makes the label-selection
  and source-mixing assumptions explicit.
