|   rank | compound          |   logS pred |   90% PI low |   90% PI high | class            |      mg/L |   AD max sim | tier               | flags      |
|-------:|:------------------|------------:|-------------:|--------------:|:-----------------|----------:|-------------:|:-------------------|:-----------|
|      1 | ethanol           |        1.14 |         0.04 |          2.23 | highly_soluble   | 628000.00 |         1.00 | soluble            | train      |
|      2 | metformin         |       -0.61 |        -2.89 |          1.66 | soluble          |  31400.00 |         0.89 | soluble            |            |
|      3 | caffeine          |       -0.98 |        -1.80 |         -0.16 | soluble          |  20300.00 |         1.00 | soluble            | train      |
|      4 | paracetamol       |       -1.45 |        -2.91 |          0.00 | soluble          |   5330.00 |         1.00 | soluble            | train      |
|      5 | diclofenac sodium |       -1.49 |        -3.52 |          0.55 | soluble          |  10400.00 |         1.00 | soluble            | train salt |
|      6 | aspirin           |       -1.75 |        -2.70 |         -0.79 | soluble          |   3230.00 |         1.00 | soluble            | train      |
|      7 | benzene           |       -2.03 |        -4.13 |          0.07 | slightly_soluble |    730.00 |         1.00 | uncertain: measure | train      |
|      8 | cubane            |       -2.83 |        -5.66 |          0.01 | slightly_soluble |    155.00 |         0.14 | uncertain: measure | OOD        |
|      9 | hydrocortisone    |       -3.33 |        -4.65 |         -2.01 | slightly_soluble |    170.00 |         1.00 | uncertain: measure | train      |
|     10 | carbamazepine     |       -3.50 |        -5.15 |         -1.84 | slightly_soluble |     75.60 |         1.00 | uncertain: measure | train      |
|     11 | ketoconazole      |       -3.59 |        -5.26 |         -1.91 | slightly_soluble |    138.00 |         0.44 | uncertain: measure |            |
|     12 | sildenafil        |       -3.61 |        -6.21 |         -1.01 | slightly_soluble |    116.00 |         0.50 | uncertain: measure |            |
|     13 | ibuprofen         |       -3.72 |        -5.34 |         -2.09 | slightly_soluble |     39.50 |         1.00 | uncertain: measure | train      |
|     14 | phenytoin         |       -3.84 |        -5.15 |         -2.52 | slightly_soluble |     36.80 |         1.00 | uncertain: measure | train      |
|     15 | naproxen          |       -3.97 |        -5.04 |         -2.90 | slightly_soluble |     24.60 |         1.00 | uncertain: measure | train      |
|     16 | griseofulvin      |       -4.11 |        -4.95 |         -3.28 | insoluble        |     27.10 |         1.00 | uncertain: measure | train      |
|     17 | naphthalene       |       -4.22 |        -5.12 |         -3.31 | insoluble        |      7.75 |         1.00 | uncertain: measure | train      |
|     18 | imatinib          |       -4.53 |        -6.18 |         -2.88 | insoluble        |     14.50 |         0.32 | uncertain: measure | OOD        |
|     19 | anthracene        |       -6.02 |        -7.64 |         -4.41 | insoluble        |      0.17 |         1.00 | insoluble          | train      |

Tiers at threshold logS = -4: {'uncertain: measure': 12, 'soluble': 6, 'insoluble': 1}; 1 input(s) not predicted.
