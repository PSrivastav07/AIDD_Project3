| direction                   | model                                          | sim_bin    |    n |   MAE |   MAE_ci_low |   MAE_ci_high |
|:----------------------------|:-----------------------------------------------|:-----------|-----:|------:|-------------:|--------------:|
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.0, 0.3) |   43 | 0.675 |        0.547 |         0.826 |
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.3, 0.4) |  104 | 0.535 |        0.458 |         0.616 |
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.4, 0.5) |  178 | 0.492 |        0.433 |         0.56  |
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.5, 0.6) |  289 | 0.422 |        0.386 |         0.464 |
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.6, 0.7) |  274 | 0.373 |        0.337 |         0.412 |
| aqsoldb_to_esol             | LightGBM (rdkit2d+Morgan)                      | [0.7, 1.0] |  227 | 0.403 |        0.354 |         0.456 |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.0, 0.3) |   43 | 0.635 |        0.504 |         0.778 |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.3, 0.4) |  104 | 0.528 |        0.433 |         0.621 |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.4, 0.5) |  178 | 0.487 |        0.426 |         0.559 |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.5, 0.6) |  289 | 0.461 |        0.418 |         0.51  |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.6, 0.7) |  274 | 0.427 |        0.383 |         0.469 |
| aqsoldb_to_esol             | RF (rdkit2d)                                   | [0.7, 1.0] |  227 | 0.442 |        0.388 |         0.504 |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.0, 0.3) |   43 | 0.961 |        0.753 |         1.182 |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.3, 0.4) |  104 | 0.948 |        0.811 |         1.093 |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.4, 0.5) |  178 | 0.997 |        0.89  |         1.121 |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.5, 0.6) |  289 | 1.086 |        0.991 |         1.186 |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.6, 0.7) |  274 | 1.036 |        0.95  |         1.13  |
| aqsoldb_to_esol             | ESOL-linear refit (esol4)                      | [0.7, 1.0] |  227 | 1.111 |        0.992 |         1.246 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.0, 0.3) |   43 | 0.741 |        0.583 |         0.918 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.3, 0.4) |  104 | 0.792 |        0.661 |         0.926 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.4, 0.5) |  178 | 0.86  |        0.765 |         0.978 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.5, 0.6) |  289 | 0.86  |        0.78  |         0.945 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.6, 0.7) |  274 | 0.829 |        0.751 |         0.909 |
| aqsoldb_to_esol             | Delaney published equation (Crippen cLogP)     | [0.7, 1.0] |  227 | 0.875 |        0.766 |         0.982 |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.0, 0.3) |   43 | 0.722 |        0.596 |         0.87  |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.3, 0.4) |  104 | 0.657 |        0.564 |         0.759 |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.4, 0.5) |  178 | 0.749 |        0.665 |         0.85  |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.5, 0.6) |  289 | 0.74  |        0.669 |         0.815 |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.6, 0.7) |  274 | 0.673 |        0.609 |         0.739 |
| aqsoldb_to_esol             | Delaney published ESOL predictions (in-sample) | [0.7, 1.0] |  227 | 0.656 |        0.581 |         0.73  |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.0, 0.3) | 2377 | 1.262 |        1.224 |         1.301 |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.3, 0.4) | 2147 | 0.967 |        0.928 |         1.005 |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.4, 0.5) | 1520 | 0.839 |        0.797 |         0.879 |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.5, 0.6) | 1170 | 0.728 |        0.684 |         0.771 |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.6, 0.7) |  539 | 0.61  |        0.562 |         0.659 |
| esol_to_aqsoldb             | LightGBM (rdkit2d+Morgan)                      | [0.7, 1.0] |  321 | 0.759 |        0.676 |         0.846 |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.0, 0.3) | 2377 | 1.265 |        1.229 |         1.304 |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.3, 0.4) | 2147 | 1.001 |        0.961 |         1.038 |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.4, 0.5) | 1520 | 0.883 |        0.842 |         0.924 |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.5, 0.6) | 1170 | 0.777 |        0.736 |         0.82  |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.6, 0.7) |  539 | 0.66  |        0.611 |         0.709 |
| esol_to_aqsoldb             | RF (rdkit2d)                                   | [0.7, 1.0] |  321 | 0.812 |        0.719 |         0.901 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.0, 0.3) | 2377 | 1.582 |        1.492 |         1.701 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.3, 0.4) | 2147 | 1.186 |        1.109 |         1.281 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.4, 0.5) | 1520 | 1.081 |        1.013 |         1.155 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.5, 0.6) | 1170 | 1.006 |        0.933 |         1.083 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.6, 0.7) |  539 | 0.918 |        0.806 |         1.058 |
| esol_to_aqsoldb             | ESOL-linear refit (esol4)                      | [0.7, 1.0] |  321 | 1.222 |        1.052 |         1.409 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.0, 0.3) | 2377 | 1.383 |        1.311 |         1.474 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.3, 0.4) | 2147 | 1.088 |        1.03  |         1.155 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.4, 0.5) | 1520 | 1.017 |        0.968 |         1.075 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.5, 0.6) | 1170 | 0.968 |        0.913 |         1.024 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.6, 0.7) |  539 | 0.963 |        0.885 |         1.057 |
| esol_to_aqsoldb             | Delaney published equation (Crippen cLogP)     | [0.7, 1.0] |  321 | 1.293 |        1.165 |         1.433 |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.0, 0.3) |  145 | 0.686 |        0.609 |         0.765 |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.3, 0.4) |  323 | 0.632 |        0.586 |         0.68  |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.4, 0.5) |  335 | 0.551 |        0.508 |         0.595 |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.5, 0.6) |  209 | 0.503 |        0.452 |         0.558 |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.6, 0.7) |   63 | 0.464 |        0.38  |         0.56  |
| aqsoldb_to_esol_sizematched | LightGBM (rdkit2d+Morgan)                      | [0.7, 1.0] |   40 | 0.722 |        0.532 |         0.922 |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.0, 0.3) |  145 | 0.642 |        0.557 |         0.728 |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.3, 0.4) |  323 | 0.628 |        0.577 |         0.68  |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.4, 0.5) |  335 | 0.567 |        0.523 |         0.618 |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.5, 0.6) |  209 | 0.529 |        0.474 |         0.589 |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.6, 0.7) |   63 | 0.512 |        0.417 |         0.623 |
| aqsoldb_to_esol_sizematched | RF (rdkit2d)                                   | [0.7, 1.0] |   40 | 0.78  |        0.577 |         0.987 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.0, 0.3) |  145 | 0.925 |        0.797 |         1.049 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.3, 0.4) |  323 | 1.004 |        0.92  |         1.091 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.4, 0.5) |  335 | 1.052 |        0.966 |         1.136 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.5, 0.6) |  209 | 0.981 |        0.872 |         1.093 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.6, 0.7) |   63 | 1.099 |        0.912 |         1.289 |
| aqsoldb_to_esol_sizematched | ESOL-linear refit (esol4)                      | [0.7, 1.0] |   40 | 1.59  |        1.193 |         1.969 |