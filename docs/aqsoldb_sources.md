# AqSolDB source datasets (letters A–I)

AqSolDB [Sorkun, Khetan & Er 2019, *Sci. Data* 6:143, doi:10.1038/s41597-019-0151-1] merges nine public
aqueous-solubility datasets. The first character of every AqSolDB `ID` (for example `A-3`, `E-1021`) names the
dataset that supplied the **selected** value for that compound. The mapping and sizes below come from Table 1 and
the "Methods / Step 1" text of the open-access paper (Europe PMC PMC6687799, read 2026-10-13). Per-subset
inter-measurement SDs come from Table 1 of Sorkun, Koelman & Er 2021 (*iScience* 24:101961,
doi:10.1016/j.isci.2020.101961, PMC7788089). Reference numbers in square brackets point to
`reports/references.md`.

## 1. Letter → original dataset (as published)

The authors named the datasets A to I in descending order of size after pre-processing.

| Letter | Original source (as cited in AqSolDB) | Original size | After pre-processing | Native identifiers / units | Temperature / notes (as stated in [7]) | Inter-measurement SD, logS [22] |
|---|---|---:|---:|---|---|---:|
| A | **eChemPortal** (OECD), query "experimental studies" + "water solubility" | 14,180 | 6,110 | name, CAS; g/L, mg/L, µg/L | filtered to 25 ± 5 °C (8,419 left), 6,183 converted to SMILES, 6,110 after validation | 0.717 |
| B | **EPI Suite Data**, US EPA: WATERNT (Water Solubility Fragment) program validation files (1995) | 5,764 | 4,651 | name, CAS; LogS | liquid and crystalline organics; SMILES for 5,367; filtered to 25 ± 5 °C (5,206); 4,651 after InChI validation | 0.372 |
| C | **Raevsky et al. 2014**, *J. Chem. Inf. Model.* 54:683–691 [30] | 2,603 | 2,603 | name, SMILES; LogS | crystalline un-ionised organic chemicals and drugs, 25 ± 5 °C | 0.380 |
| D | **EPI Suite Data**, US EPA: WSKOWWIN program validation files (1994) | 2,267 | 2,115 | name, CAS; LogS | liquid and crystalline organics | 0.361 |
| E | **Huuskonen 2000**, *J. Chem. Inf. Comput. Sci.* 40:773–777 [26] (extended version from cheminformatics.org) | 1,291 | 1,291 | name, SMILES, CAS; LogS | 20–25 °C; values from AQUASOL (Univ. of Arizona) and SRC PHYSPROP | 0.274 |
| F | **Wang, Hou & Xu 2009**, *J. Chem. Inf. Model.* 49:571–581 [31]; 1,210 compounds extracted from Beilstein | 1,210 | 1,210 | SLN only; LogS | 93 SLN strings could not be parsed by RDKit and were completed with MolView | 0.582 |
| G | **Delaney 2004** (ESOL), *J. Chem. Inf. Comput. Sci.* 44:1000–1005 [27] | 1,144 | 1,144 | name, SMILES; LogS | 25 °C | 0.392 |
| H | **Wang et al. 2009** [31]: their de-duplicated version of the set used by **Jain & Yalkowsky 2001** [32] | 578 | 578 | SLN only; LogS | 322 liquids and 256 solids | 0.383 |
| I | **Goodman group / Llinàs, Glen & Goodman 2008** solubility-challenge data, corrected version [12] | 105 | 94 | name, SMILES, InChI; µM | 25 °C, CheqSol intrinsic solubility; 11 rows without a value removed | 0.338 |
| All | merged AqSolDB | 19,796 instances | **9,982 unique compounds** | | | 0.495 (non-AF subset: 0.356) |

The SD column is the SD of the multiple measurements that involve each subset, from [22] Table 1 (computed on
3,093 / 1,215 / 668 / 179 / 337 / 202 / 170 / 100 / 46 repeated values for A–I). The SDs of A (eChemPortal) and
F (Beilstein via Wang et al.) are the largest; the AqSolPred authors therefore left A and F out of training and kept
E (Huuskonen) as the high-quality test set [22].

## 2. How a single value per compound was selected

AqSolDB identifies compounds by InChI and treats values within 0.01 logS units as identical [7]:

* **G1** (7,746 compounds): one unique value, or several values within 0.01 log units. The value is accepted as is.
* **≥ 3 different values** (819 compounds): the "closest to the mean" value is kept. **G4** if SD > 0.5 (183),
  otherwise **G5** (636).
* **Exactly 2 different values** (1,417 compounds): the value closest to an **ALOGPS 2.1 prediction** is kept
  ("closest to the reference"; ALOGPS reported RMSE 0.49 log units). **G2** if SD > 0.5 (235), otherwise **G3** (1,182).

Two points matter for this project:

1. **The letter shows where the selected value came from, not every dataset that contains the compound.** Dataset A
   contributes 6,110 entries after pre-processing, but only 3,656 of the 9,982 AqSolDB IDs carry the letter A. A
   leave-one-source-out (LOSO) split on the letter therefore holds out the compounds *whose selected value* came
   from that source. Other sources' values for the same compound are not in the distributed file.
2. **Label selection for G2/G3 compounds depends on a model.** For compounds measured exactly twice, the chosen
   label is the one nearer to an ALOGPS neural-network prediction. That covers 1,417 / 9,982 = 14.2 % of raw
   AqSolDB and 1,339 / 9,171 = 14.6 % of our curated set (G2 + G3). This adds a small bias towards labels a
   structure-based model finds "predictable". The SD threshold in the distributed file is strict: SD > 0.5 → G2/G4,
   and SD = 0.5000 exactly → G3. The paper's Methods text says "≥ 0.5" but its Technical Validation says "> 0.5";
   the file follows the second.

## 3. Composition by letter in this project

Computed with the snippet in §4 from `data/raw/aqsoldb_curated-solubility-dataset.csv` (raw, 9,982 rows) and from
`solpred.features.load_dataset("aqsoldb")` (curated, 9,171 rows; see `docs/data_card.md`).

| Letter | Raw IDs | Raw IDs with > 1 occurrence | RMS SD of those, logS | Curated n | Curated logS mean ± SD | Curated logS range | G1 / G2+G3 / G4+G5 (curated) | Acyclic fraction | Multi-component fraction | Also in ESOL |
|---|---:|---:|---:|---:|---|---|---|---:|---:|---:|
| A | 3,656 | 1,032 | 0.629 | 3,143 | −2.96 ± 2.63 | −10.23 to 2.14 | 2,256 / 533 / 354 | 0.360 | 0.205 | 172 |
| B | 3,542 | 545 | 0.412 | 3,405 | −2.63 ± 2.30 | −13.17 to 1.58 | 2,868 / 401 / 136 | 0.293 | 0.016 | 515 |
| C | 1,325 | 252 | 0.399 | 1,263 | −3.44 ± 2.07 | −10.60 to 1.00 | 1,017 / 167 / 79 | 0.040 | 0.000 | 115 |
| D | 163 | 69 | 0.400 | 157 | −2.45 ± 2.02 | −7.46 to 1.56 | 88 / 57 / 12 | 0.357 | 0.006 | 34 |
| E | 249 | 140 | 0.301 | 207 | −2.59 ± 1.86 | −8.71 to 1.15 | 72 / 69 / 66 | 0.304 | 0.000 | 97 |
| F | 798 | 74 | 0.494 | 775 | −2.70 ± 1.44 | −8.56 to 1.13 | 703 / 50 / 22 | 0.124 | 0.000 | 16 |
| G | 139 | 67 | 0.310 | 126 | −3.38 ± 2.11 | −11.60 to 0.64 | 61 / 38 / 27 | 0.230 | 0.000 | 126 |
| H | 64 | 36 | 0.255 | 61 | −5.32 ± 2.97 | −12.95 to −0.24 | 27 / 14 / 20 | 0.246 | 0.000 | 17 |
| I | 46 | 21 | 0.416 | 34 | −3.42 ± 0.98 | −5.79 to −1.72 | 14 / 10 / 10 | 0.000 | 0.000 | 5 |
| **All** | **9,982** | **2,236** | **0.516** (median 0.132) | **9,171** | −2.89 ± 2.34 | −13.17 to 2.14 | 7,106 / 1,339 / 726 | 0.266 | 0.076 | 1,097 |

Notes on reading this table:

* The "RMS SD" column groups compounds by the letter of the *selected* value. The SD itself spans every source
  that measured the compound, so this is not the same quantity as the per-subset SD in §1. Both rank A as the
  noisiest source. Our curated set has a lower RMS SD (0.478 over 2,065 multi-occurrence compounds, `docs/data_card.md`)
  than raw AqSolDB (0.516 over 2,236). The difference comes from the raw records that curation dropped (inorganic,
  organometallic and other excluded entries, see `docs/data_card.md`): the 142 multi-occurrence raw IDs not
  represented in the curated set have an RMS SD of 0.925, against 0.476 for the 2,094 that are retained (computed
  from the raw file and the curated `compound_id`/`merged_ids` columns, 2026-10-13).
* Letter E keeps only 249 of Huuskonen's 1,291 compounds as selected values. Most Huuskonen compounds also occur
  in A or B and took their label (and letter) from there. The held-out set in the `loso/source_E` split is
  therefore small (207 compounds) and close to the training data: 91.8 % of its test scaffolds also occur in
  training, and the median test-to-train maximum Tanimoto similarity is 0.64 (`data/splits/split_summary.csv`).
* Source A has the highest share of salts and multi-component records (20.5 %), so it is chemically the most
  heterogeneous source.
* ESOL (Delaney) entries mostly took their AqSolDB label from B (515) or A (172). Only 126 carry the letter G,
  yet 1,097 of the 1,115 curated ESOL compounds are in AqSolDB (`data/processed/curation_log.json`).

## 4. Reproduction snippet

```python
import numpy as np, pandas as pd
from solpred.features import load_dataset

raw = pd.read_csv("data/raw/aqsoldb_curated-solubility-dataset.csv")
raw["src"] = raw["ID"].str[0]
cur = load_dataset("aqsoldb")
for s in "ABCDEFGHI":
    r, c = raw[raw.src == s], cur[cur.source == s]
    m = r[r.Ocurrences > 1]
    print(s, len(r), len(m), round(float(np.sqrt((m.SD ** 2).mean())), 3), len(c),
          round(c.logS.mean(), 2), round(c.logS.std(), 2),
          c.reliability_group.value_counts().to_dict(),
          round(float((c.scaffold.fillna("") == "").mean()), 3), int(c.in_esol.astype(bool).sum()))
print(raw.Group.value_counts().to_dict())   # {'G1': 7746, 'G3': 1182, 'G5': 636, 'G2': 235, 'G4': 183}
```

## 5. Verification status

* Letters, source names, original and filtered sizes, temperature filters and group counts: checked against the
  full text of Sorkun et al. 2019 (Table 1, Methods, Fig. 4 text). The raw file's group counts (G1 7,746;
  G2 235; G3 1,182; G4 183; G5 636) match the paper exactly.
* EPI Suite file names (WATERNT for B, WSKOWWIN for D) come from the paper's reference list (refs 15 and 17).
  The original URLs (esc.syrres.com) are no longer checked here.
* Per-subset SDs: Sorkun, Koelman & Er 2021, Table 1 (all nine values, the N(SD) counts, "All" 0.495 and
  "Non-AF" 0.356 re-read from the full text, PMC7788089).
* ID-prefix counts in `data/raw/aqsoldb_curated-solubility-dataset.csv` were recomputed on 2026-10-13:
  A 3,656; B 3,542; C 1,325; D 163; E 249; F 798; G 139; H 64; I 46 (sum 9,982). They are much smaller than the
  per-source sizes in §1 (sum 19,796) because each of the 9,982 compounds carries only the letter of its selected
  value; the per-letter counts in §3 are therefore not source sizes.
* The threshold remark in §2 was checked in the file: the three compounds with SD exactly 0.5 are in G3.
