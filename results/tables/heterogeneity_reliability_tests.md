| model                       | split_type   | contrast                                  |   mae_diff |   ci_low |   ci_high |   p_boot |       H |   p |   k |
|:----------------------------|:-------------|:------------------------------------------|-----------:|---------:|----------:|---------:|--------:|----:|----:|
| lgbm__rdkit2d+morgan_counts | random       | inconsistent (G2+G4) - consistent (G3+G5) |      0.23  |    0.119 |     0.34  |    0     | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | random       | single (G1) - consistent (G3+G5)          |      0.192 |    0.156 |     0.228 |    0     | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | scaffold     | inconsistent (G2+G4) - consistent (G3+G5) |      0.044 |   -0.066 |     0.16  |    0.476 | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | scaffold     | single (G1) - consistent (G3+G5)          |      0.156 |    0.094 |     0.215 |    0     | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | cluster      | inconsistent (G2+G4) - consistent (G3+G5) |      0.255 |    0.151 |     0.358 |    0     | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | cluster      | single (G1) - consistent (G3+G5)          |      0.23  |    0.186 |     0.275 |    0     | nan     | nan | nan |
| rf__rdkit2d                 | random       | inconsistent (G2+G4) - consistent (G3+G5) |      0.201 |    0.094 |     0.317 |    0     | nan     | nan | nan |
| rf__rdkit2d                 | random       | single (G1) - consistent (G3+G5)          |      0.215 |    0.17  |     0.255 |    0     | nan     | nan | nan |
| rf__rdkit2d                 | scaffold     | inconsistent (G2+G4) - consistent (G3+G5) |     -0.011 |   -0.127 |     0.106 |    0.83  | nan     | nan | nan |
| rf__rdkit2d                 | scaffold     | single (G1) - consistent (G3+G5)          |      0.173 |    0.104 |     0.237 |    0     | nan     | nan | nan |
| rf__rdkit2d                 | cluster      | inconsistent (G2+G4) - consistent (G3+G5) |      0.25  |    0.134 |     0.37  |    0     | nan     | nan | nan |
| rf__rdkit2d                 | cluster      | single (G1) - consistent (G3+G5)          |      0.274 |    0.227 |     0.324 |    0     | nan     | nan | nan |
| lgbm__rdkit2d+morgan_counts | random       | reliability_group                         |    nan     |  nan     |   nan     |  nan     |  67.896 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | random       | rel3                                      |    nan     |  nan     |   nan     |  nan     |  67.562 |   0 |   3 |
| lgbm__rdkit2d+morgan_counts | random       | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  49.646 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | random       | source                                    |    nan     |  nan     |   nan     |  nan     |  69.856 |   0 |   9 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin                                    |    nan     |  nan     |   nan     |  nan     |  84.668 |   0 |   6 |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group                         |    nan     |  nan     |   nan     |  nan     |  26.984 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | scaffold     | rel3                                      |    nan     |  nan     |   nan     |  nan     |  22.894 |   0 |   3 |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  26.473 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source                                    |    nan     |  nan     |   nan     |  nan     |  49.915 |   0 |   9 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin                                    |    nan     |  nan     |   nan     |  nan     |  24.685 |   0 |   6 |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group                         |    nan     |  nan     |   nan     |  nan     |  76.496 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | cluster      | rel3                                      |    nan     |  nan     |   nan     |  nan     |  76.252 |   0 |   3 |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  50.923 |   0 |   5 |
| lgbm__rdkit2d+morgan_counts | cluster      | source                                    |    nan     |  nan     |   nan     |  nan     | 101.529 |   0 |   9 |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin                                    |    nan     |  nan     |   nan     |  nan     |  86.649 |   0 |   6 |
| rf__rdkit2d                 | random       | reliability_group                         |    nan     |  nan     |   nan     |  nan     |  85.71  |   0 |   5 |
| rf__rdkit2d                 | random       | rel3                                      |    nan     |  nan     |   nan     |  nan     |  84.076 |   0 |   3 |
| rf__rdkit2d                 | random       | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  72.658 |   0 |   5 |
| rf__rdkit2d                 | random       | source                                    |    nan     |  nan     |   nan     |  nan     |  71.963 |   0 |   9 |
| rf__rdkit2d                 | random       | sd_bin                                    |    nan     |  nan     |   nan     |  nan     |  94.922 |   0 |   6 |
| rf__rdkit2d                 | scaffold     | reliability_group                         |    nan     |  nan     |   nan     |  nan     |  36.333 |   0 |   5 |
| rf__rdkit2d                 | scaffold     | rel3                                      |    nan     |  nan     |   nan     |  nan     |  34.942 |   0 |   3 |
| rf__rdkit2d                 | scaffold     | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  36.961 |   0 |   5 |
| rf__rdkit2d                 | scaffold     | source                                    |    nan     |  nan     |   nan     |  nan     |  71.203 |   0 |   9 |
| rf__rdkit2d                 | scaffold     | sd_bin                                    |    nan     |  nan     |   nan     |  nan     |  40.401 |   0 |   6 |
| rf__rdkit2d                 | cluster      | reliability_group                         |    nan     |  nan     |   nan     |  nan     | 110.184 |   0 |   5 |
| rf__rdkit2d                 | cluster      | rel3                                      |    nan     |  nan     |   nan     |  nan     | 107.81  |   0 |   3 |
| rf__rdkit2d                 | cluster      | occ_bin                                   |    nan     |  nan     |   nan     |  nan     |  88.146 |   0 |   5 |
| rf__rdkit2d                 | cluster      | source                                    |    nan     |  nan     |   nan     |  nan     | 103.796 |   0 |   9 |
| rf__rdkit2d                 | cluster      | sd_bin                                    |    nan     |  nan     |   nan     |  nan     | 121.055 |   0 |   6 |