# Examples

| file | content |
|---|---|
| `example_smiles.csv` | 20 inputs: 5 reference compounds (ethanol, benzene, naphthalene, anthracene, cubane as an unusual cage), 14 drugs (one sodium salt) and cisplatin as an out-of-scope inorganic complex. Columns `name, smiles, category, expected_formula`. |
| `example_predictions.csv` | output of the CLI below (input columns + all predictor columns) |
| `triage_demo.py` | ranks predictions and assigns a measure / don't-measure tier from the conformal interval |
| `triage_ranking.md` | output of `triage_demo.py` (reproduced below) |

Every SMILES was parsed with RDKit and its `CalcMolFormula` checked against `expected_formula` (all 20 agree;
cisplatin is written in Hill order `Cl2H6N2Pt`, which RDKit prints as `H6Cl2N2Pt` - same composition).

## Reproduce

```bash
.venv/Scripts/solpred-predict.exe --input examples/example_smiles.csv --smiles-col smiles     --output examples/example_predictions.csv --alpha 0.1
.venv/Scripts/python.exe examples/triage_demo.py --input examples/example_predictions.csv     --threshold -4 --output examples/triage_ranking.md
```

Quick inline use: `.venv/Scripts/solpred-predict.exe --smiles CCO c1ccccc1 "not a smiles"` prints a compact table
(the invalid entry gets `valid=False` and an error message; it never aborts the batch).

## Triage-ranking demo (90 % conformal intervals, decision threshold logS = -4, i.e. 100 umol/L)

Tier rule: `soluble` if the whole interval is above -4, `insoluble` if it is entirely below, otherwise
`uncertain: measure`. Flags: `OOD` = max ECFP4 Tanimoto to training < 0.35; `train` = the structure is in
the training set (the prediction is in-sample - use the measured value shown in `nearest_train_logS`);
`salt` = multicomponent input.

|   rank | compound          |   logS pred |   90% PI low |   90% PI high | class            |      mg/L |   AD max sim | tier               | flags      |
|-------:|:------------------|------------:|-------------:|--------------:|:-----------------|----------:|-------------:|:-------------------|:-----------|
|      1 | ethanol           |        1.14 |         0.04 |          2.23 | highly_soluble   | 628000.00 |         1.00 | soluble            | train      |
|      2 | metformin         |       -0.61 |        -2.89 |          1.66 | soluble          |  31400.00 |         0.89 | soluble            |            |
|      3 | caffeine          |       -0.98 |        -1.80 |         -0.16 | soluble          |  20300.00 |         1.00 | soluble            | train      |
|      4 | paracetamol       |       -1.45 |        -2.91 |          0.00 | soluble          |   5330.00 |         1.00 | soluble            | train      |
|      5 | diclofenac sodium |       -1.49 |        -3.52 |          0.55 | soluble          |  10400.00 |         1.00 | soluble            | train salt |
|      6 | aspirin           |       -1.75 |        -2.70 |         -0.79 | soluble          |   3230.00 |         1.00 | soluble            | train      |
|      7 | benzene           |       -2.03 |        -4.13 |          0.07 | slightly_soluble |    730.00 |         1.00 | uncertain: measure | train      |
|      8 | cubane            |       -2.83 |        -5.67 |          0.01 | slightly_soluble |    155.00 |         0.14 | uncertain: measure | OOD        |
|      9 | hydrocortisone    |       -3.33 |        -4.65 |         -2.01 | slightly_soluble |    170.00 |         1.00 | uncertain: measure | train      |
|     10 | carbamazepine     |       -3.50 |        -5.15 |         -1.84 | slightly_soluble |     75.60 |         1.00 | uncertain: measure | train      |
|     11 | ketoconazole      |       -3.59 |        -5.27 |         -1.91 | slightly_soluble |    138.00 |         0.44 | uncertain: measure |            |
|     12 | sildenafil        |       -3.61 |        -6.21 |         -1.01 | slightly_soluble |    116.00 |         0.50 | uncertain: measure |            |
|     13 | ibuprofen         |       -3.72 |        -5.34 |         -2.09 | slightly_soluble |     39.50 |         1.00 | uncertain: measure | train      |
|     14 | phenytoin         |       -3.84 |        -5.15 |         -2.52 | slightly_soluble |     36.80 |         1.00 | uncertain: measure | train      |
|     15 | naproxen          |       -3.97 |        -5.05 |         -2.90 | slightly_soluble |     24.60 |         1.00 | uncertain: measure | train      |
|     16 | griseofulvin      |       -4.11 |        -4.95 |         -3.28 | insoluble        |     27.10 |         1.00 | uncertain: measure | train      |
|     17 | naphthalene       |       -4.22 |        -5.12 |         -3.31 | insoluble        |      7.75 |         1.00 | uncertain: measure | train      |
|     18 | imatinib          |       -4.53 |        -6.18 |         -2.88 | insoluble        |     14.50 |         0.32 | uncertain: measure | OOD        |
|     19 | anthracene        |       -6.02 |        -7.64 |         -4.41 | insoluble        |      0.17 |         1.00 | insoluble          | train      |

Tiers at threshold logS = -4: {'uncertain: measure': 12, 'soluble': 6, 'insoluble': 1}; 1 input(s) not predicted.

How to read it:

* The ranking itself is sensible (ethanol > benzene > naphthalene > anthracene; small polar drugs above
  lipophilic ones), but 90 % intervals are wide (mean width 2.80 log units over all compounds in scaffold-grouped
  CV, `models/final/summary.json`), so the threshold only resolves compounds far from it: here 6 of 19 are
  confidently `soluble`, 1 confidently `insoluble` (anthracene) and 12 need measurement.
* 14 of the 19 predicted compounds are in the training set: for them the model reproduces known data and the
  measured value should be used. The genuinely prospective rows are metformin, cubane, ketoconazole,
  sildenafil and imatinib.
* Cubane (max similarity 0.14) and imatinib (0.32) are flagged out of domain; their nearest training
  neighbours (logS -0.10 and -1.27) are structurally poor analogues and the intervals should be treated as
  optimistic (CV coverage of the nominal 90 % interval is 84 % outside the domain and 82 % below
  similarity 0.3).
* Cisplatin is rejected (`no_carbon`): inorganic complexes are outside the curation scope and are not
  extrapolated.
* Ionisable drugs (metformin, imatinib, diclofenac) are predicted for the species as it was measured in
  AqSolDB; the model has no notion of pH or polymorph, so treat these as rough intrinsic/salt-form estimates.
