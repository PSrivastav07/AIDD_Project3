# Data card: curated AqSolDB and ESOL solubility sets (v1.0.0)

## 1. Provenance

| Dataset | Source | Licence | Raw records | Raw file SHA-256 |
|---|---|---|---|---|
| AqSolDB (Sorkun, Khetan & Er, *Sci. Data* 2019) | Harvard Dataverse, doi:10.7910/DVN/OVHAW8, file `curated-solubility-dataset.tab` (dataset version 1.0) | CC0 1.0 | 9,982 | `7c47e021…8630332` |
| ESOL / Delaney (*J. Chem. Inf. Comput. Sci.* 2004) | DeepChem mirror `delaney-processed.csv` | public research dataset | 1,128 | `8c06a76f…602f189` |

Full URLs, byte sizes, download timestamps and complete hashes are in `data/raw/MANIFEST.json`. The TDC
`Solubility_AqSolDB` task is a redistribution of the same AqSolDB file, so it was not downloaded separately.
AqSolDB merges nine earlier datasets. The first letter of every AqSolDB ID (A–I) identifies the source of the
selected value; the verified mapping from letter to original dataset is in [aqsoldb_sources.md](aqsoldb_sources.md).

**Target.** `logS` = log10 of aqueous solubility in mol/L. AqSolDB reports one value per compound. When a
compound occurs in several source datasets, the value combines those occurrences, and the file gives the
standard deviation (`SD`) and the number of occurrences (`Ocurrences`). The curated file does not record
temperature, pH, solid form or measurement type (kinetic or thermodynamic) for each entry.

## 2. Curation pipeline (`src/solpred/data/curate.py`, `scripts/02_curate_data.py`)

| Step | AqSolDB | ESOL |
|---|---|---|
| Raw records | 9,982 | 1,128 |
| SMILES parse failure (two pyridine N-oxides with invalid `[NH++]` valence) | −2 | 0 |
| No carbon atom (inorganic) | −325 | 0 |
| Element outside the organic set (H B C N O F Si P S Cl Se Br I), or a metal bonded covalently. Monoatomic Li/Na/K/Mg/Ca counter-ions are allowed. Removals are mostly Zn/Sn/Co/Cu/Cr/Ba/Al/Mn compounds and dyes. | −194 | 0 |
| Non-finite label | 0 | 0 |
| De-duplication on non-isomeric canonical SMILES (median label) | −290 rows (257 groups; 29 with SD > 0.5) | −13 rows (13 groups; 1 with SD > 0.5) |
| **Curated compounds** | **9,171** | **1,115** |

Standardisation uses RDKit `rdMolStandardize.Cleanup`: sanitisation, removal of explicit H, metal
disconnection, functional-group normalisation and re-ionisation. Canonical SMILES come from RDKit 2026.09.

Design decisions and their rationale:
* **Salts are kept as salts.** The solubility of a salt is a different property from that of its free acid or
  base, so counter-ions are not stripped. The curated set keeps 699 multi-component records, 390 of which carry a
  Li/Na/K/Mg/Ca counter-ion. Scaffolds are computed on the largest organic fragment.
* **Stereo-agnostic de-duplication.** All featurisations except the graph network are 2D and blind to
  stereochemistry. Keeping stereoisomers and E/Z isomers as separate rows would create identical inputs with
  different labels, and that duplication could leak between train and test. These records are therefore
  merged on `smiles_2d`, with the label set to the median and the spread kept in `merged_logS_sd` and
  `dedup_conflict`. The 29 conflicting groups are mostly E/Z pairs (for example maleate vs fumarate diesters)
  and natural products with inconsistent reports (for example reserpine, range 2.05 log units).
* **Tautomers are not canonicalised.** Tautomer canonicalisation can change the chemistry and is slow. The QA
  audit checks how many InChIKey-connectivity duplicates remain (`results/qa/foundation_review.md`).
* For 17 AqSolDB records, the InChIKey computed from the SMILES has a different connectivity block from the
  InChIKey stored in the file (`inchikey_consistent = False`). These records are kept and flagged.

## 3. Composition of the curated AqSolDB set

* logS: mean −2.89, SD 2.34, range −13.17 to 2.14; median heavy-atom count 15 (1–388); median MW 228 g/mol.
* Source letters: A 3,143 · B 3,405 · C 1,263 · D 157 · E 207 · F 775 · G 126 · H 61 · I 34.
* Reliability groups (AqSolDB definition, confirmed against `Ocurrences`/`SD`): G1 7,106 single measurement;
  G2 218 two measurements with SD > 0.5; G3 1,121 two with SD ≤ 0.5; G4 136 three or more with SD > 0.5;
  G5 590 three or more with SD ≤ 0.5.
* Measurement spread: across the 2,065 compounds with several measurements, the RMS inter-source SD is
  0.478 log units (median 0.124).
* Chemotypes: 1,865 unique Bemis–Murcko scaffolds; 2,439 acyclic compounds (empty scaffold).
* **ESOL overlap:** 1,097 of the 1,115 curated ESOL compounds (98.4 %) are also in AqSolDB. Between the two
  sources, MAE is 0.089 log units, 64.8 % of values are identical and 5.4 % differ by more than 0.5. ESOL is
  therefore **not** an independent external test set unless the overlap is removed from training, which the
  `aqsoldb_to_esol` split does.

## 4. Files (`data/processed/`)

| File | Content |
|---|---|
| `aqsoldb_clean.csv` | curated AqSolDB, one row per compound (schema in `docs/CONTRACT.md`) |
| `esol_clean.csv` | curated ESOL, with Delaney's published ESOL prediction (`esol_pred`) and an AqSolDB cross-reference |
| `esol_aqsoldb_overlap.csv` | per-compound comparison of the two sources |
| `aqsoldb_removed.csv`, `esol_removed.csv` | every removed record and the reason |
| `curation_log.json` | the funnel above, in machine-readable form |
| `DATASET_VERSION.json` | dataset version, RDKit version and SHA-256 of every processed file |

Splits are in `data/splits/`, described in `docs/methodology.md`.

## 5. Known limitations

* **Heterogeneous labels.** Values come from different laboratories, protocols, temperatures and solid forms.
  77 % of compounds rest on a single report (G1), whose noise cannot be estimated from the data alone. The
  noise estimate of about 0.48 log units comes from multi-report compounds, and that subset may not be
  representative.
* **Chemical scope.** The data cover small organic molecules, including pesticides, dyes, industrial chemicals
  and drugs, plus simple alkali and alkaline-earth salts. Organometallics, inorganic compounds, polymers,
  peptides larger than a few residues and mixtures defined by composition are out of scope.
* **No temperature or pH metadata.** Ionisable compounds are represented by a single logS value even though
  their solubility depends strongly on pH.
* **Standardisation edge cases** (found by the QA audit, `results/qa/foundation_review.md`). The pipeline
  has no neutralisation step, so an acid and its carboxylate salt, or the charged and neutral forms of a
  compound, get different `smiles_2d`. This leaves 18 InChIKey-connectivity duplicate groups in AqSolDB, and
  17–18 ESOL compounds escape the overlap removal. A neutral alkali metal written next to a molecule without
  an acidic H can be standardised to a carbanion (≈ 5 records). The audit estimates the resulting leakage at
  ≤ 0.01 MAE.
* **Redundancy with ESOL.** Benchmarks that train on AqSolDB and test on ESOL without removing the overlap
  overstate external performance.

## 6. Responsible use

The data support early, in-silico triage and method benchmarking. Do not use them as the basis for
formulation, safety or regulatory decisions without experimental confirmation. Licences: AqSolDB is CC0.
ESOL is redistributed by MoleculeNet/DeepChem for research use; cite Delaney (2004).
