| model                       | split_type   | grouping   | level         |    n |   MAE |   MAE_ci_low |   MAE_ci_high |   median_AE |   bias |   mean_label_sd |
|:----------------------------|:-------------|:-----------|:--------------|-----:|------:|-------------:|--------------:|------------:|-------:|----------------:|
| lgbm__rdkit2d+morgan_counts | random       | occ_bin    | 1             | 2911 | 0.606 |        0.583 |         0.629 |       0.407 | -0.016 |           0     |
| lgbm__rdkit2d+morgan_counts | random       | occ_bin    | 2             |  558 | 0.451 |        0.414 |         0.491 |       0.317 |  0.008 |           0.245 |
| lgbm__rdkit2d+morgan_counts | random       | occ_bin    | 3             |  200 | 0.441 |        0.381 |         0.508 |       0.284 |  0.109 |           0.283 |
| lgbm__rdkit2d+morgan_counts | random       | occ_bin    | 4             |   63 | 0.386 |        0.311 |         0.468 |       0.263 |  0.047 |           0.306 |
| lgbm__rdkit2d+morgan_counts | random       | occ_bin    | >=5           |   44 | 0.598 |        0.415 |         0.806 |       0.375 |  0.252 |           0.542 |
| lgbm__rdkit2d+morgan_counts | random       | source     | A             | 1270 | 0.712 |        0.67  |         0.756 |       0.431 |  0.011 |           0.096 |
| lgbm__rdkit2d+morgan_counts | random       | source     | B             | 1408 | 0.48  |        0.457 |         0.504 |       0.345 | -0.01  |           0.04  |
| lgbm__rdkit2d+morgan_counts | random       | source     | C             |  530 | 0.488 |        0.452 |         0.526 |       0.356 |  0.013 |           0.051 |
| lgbm__rdkit2d+morgan_counts | random       | source     | D             |   68 | 0.414 |        0.333 |         0.498 |       0.357 |  0.029 |           0.047 |
| lgbm__rdkit2d+morgan_counts | random       | source     | E             |   98 | 0.39  |        0.318 |         0.468 |       0.295 |  0.05  |           0.119 |
| lgbm__rdkit2d+morgan_counts | random       | source     | F             |  315 | 0.643 |        0.576 |         0.713 |       0.43  | -0.113 |           0.026 |
| lgbm__rdkit2d+morgan_counts | random       | source     | G             |   56 | 0.41  |        0.329 |         0.492 |       0.355 |  0.056 |           0.109 |
| lgbm__rdkit2d+morgan_counts | random       | source     | H             |   19 | 0.752 |        0.384 |         1.34  |       0.413 |  0.284 |           0.065 |
| lgbm__rdkit2d+morgan_counts | random       | source     | I             |   12 | 0.757 |        0.412 |         1.129 |       0.559 |  0.53  |           0.107 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | single report | 2911 | 0.606 |        0.583 |         0.629 |       0.407 | -0.016 |           0     |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | (0, 0.1]      |  384 | 0.37  |        0.329 |         0.41  |       0.236 |  0.023 |           0.039 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | (0.1, 0.25]   |  197 | 0.404 |        0.356 |         0.455 |       0.313 |  0.106 |           0.163 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | (0.25, 0.5]   |  143 | 0.545 |        0.456 |         0.63  |       0.388 |  0.029 |           0.364 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | (0.5, 1]      |   89 | 0.585 |        0.47  |         0.715 |       0.379 |  0.065 |           0.698 |
| lgbm__rdkit2d+morgan_counts | random       | sd_bin     | > 1           |   52 | 0.747 |        0.565 |         0.948 |       0.43  |  0.009 |           1.449 |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin    | 1             | 2596 | 0.759 |        0.731 |         0.787 |       0.552 |  0.021 |           0     |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin    | 2             |  395 | 0.58  |        0.528 |         0.639 |       0.408 |  0.056 |           0.264 |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin    | 3             |  134 | 0.662 |        0.563 |         0.769 |       0.479 |  0.135 |           0.327 |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin    | 4             |   37 | 0.677 |        0.536 |         0.834 |       0.706 | -0.035 |           0.512 |
| lgbm__rdkit2d+morgan_counts | scaffold     | occ_bin    | >=5           |   28 | 0.717 |        0.439 |         1.051 |       0.355 |  0.321 |           0.453 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | A             |  911 | 0.846 |        0.79  |         0.904 |       0.57  |  0.166 |           0.074 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | B             | 1124 | 0.638 |        0.605 |         0.673 |       0.486 | -0.056 |           0.043 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | C             |  623 | 0.678 |        0.637 |         0.725 |       0.524 |  0.02  |           0.037 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | D             |   56 | 0.715 |        0.589 |         0.859 |       0.541 |  0.056 |           0.118 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | E             |   66 | 0.553 |        0.449 |         0.667 |       0.405 | -0.055 |           0.122 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | F             |  308 | 0.869 |        0.786 |         0.957 |       0.658 | -0.107 |           0.045 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | G             |   55 | 0.527 |        0.429 |         0.626 |       0.48  |  0.115 |           0.082 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | H             |   30 | 1.183 |        0.919 |         1.47  |       1.076 |  0.746 |           0.111 |
| lgbm__rdkit2d+morgan_counts | scaffold     | source     | I             |   17 | 0.757 |        0.514 |         1.029 |       0.52  |  0.407 |           0.275 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | single report | 2596 | 0.759 |        0.731 |         0.787 |       0.552 |  0.021 |           0     |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | (0, 0.1]      |  238 | 0.573 |        0.507 |         0.636 |       0.404 |  0.093 |           0.039 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | (0.1, 0.25]   |  129 | 0.617 |        0.524 |         0.718 |       0.412 |  0.057 |           0.159 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | (0.25, 0.5]   |  107 | 0.651 |        0.529 |         0.789 |       0.478 |  0.217 |           0.375 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | (0.5, 1]      |   79 | 0.606 |        0.493 |         0.735 |       0.455 |  0.019 |           0.691 |
| lgbm__rdkit2d+morgan_counts | scaffold     | sd_bin     | > 1           |   41 | 0.724 |        0.546 |         0.911 |       0.518 | -0.146 |           1.354 |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin    | 1             | 2952 | 0.701 |        0.675 |         0.727 |       0.476 | -0.051 |           0     |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin    | 2             |  547 | 0.508 |        0.464 |         0.554 |       0.352 | -0.05  |           0.265 |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin    | 3             |  208 | 0.541 |        0.472 |         0.619 |       0.393 |  0.054 |           0.306 |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin    | 4             |   68 | 0.507 |        0.396 |         0.628 |       0.377 |  0.002 |           0.346 |
| lgbm__rdkit2d+morgan_counts | cluster      | occ_bin    | >=5           |   48 | 0.567 |        0.41  |         0.757 |       0.383 | -0.158 |           0.403 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | A             | 1208 | 0.805 |        0.756 |         0.852 |       0.525 |  0.041 |           0.1   |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | B             | 1443 | 0.545 |        0.516 |         0.573 |       0.374 | -0.074 |           0.045 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | C             |  565 | 0.61  |        0.566 |         0.657 |       0.461 | -0.086 |           0.043 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | D             |   60 | 0.494 |        0.411 |         0.578 |       0.422 | -0.055 |           0.094 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | E             |   80 | 0.432 |        0.357 |         0.518 |       0.339 | -0.08  |           0.125 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | F             |  346 | 0.816 |        0.745 |         0.888 |       0.621 | -0.218 |           0.034 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | G             |   76 | 0.511 |        0.439 |         0.584 |       0.464 |  0.103 |           0.115 |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | H             |   33 | 0.733 |        0.472 |         1.08  |       0.45  |  0.084 |           0.08  |
| lgbm__rdkit2d+morgan_counts | cluster      | source     | I             |   12 | 0.88  |        0.624 |         1.163 |       0.748 |  0.388 |           0.186 |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | single report | 2952 | 0.701 |        0.675 |         0.727 |       0.476 | -0.051 |           0     |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | (0, 0.1]      |  362 | 0.41  |        0.369 |         0.452 |       0.292 | -0.061 |           0.04  |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | (0.1, 0.25]   |  211 | 0.496 |        0.432 |         0.57  |       0.349 |  0.06  |           0.159 |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | (0.25, 0.5]   |  135 | 0.597 |        0.498 |         0.708 |       0.421 | -0.013 |           0.376 |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | (0.5, 1]      |  107 | 0.657 |        0.551 |         0.774 |       0.513 | -0.022 |           0.681 |
| lgbm__rdkit2d+morgan_counts | cluster      | sd_bin     | > 1           |   56 | 0.858 |        0.653 |         1.084 |       0.613 | -0.176 |           1.422 |
| rf__rdkit2d                 | random       | occ_bin    | 1             | 2911 | 0.683 |        0.658 |         0.708 |       0.477 | -0.029 |           0     |
| rf__rdkit2d                 | random       | occ_bin    | 2             |  558 | 0.494 |        0.452 |         0.536 |       0.348 |  0.034 |           0.245 |
| rf__rdkit2d                 | random       | occ_bin    | 3             |  200 | 0.493 |        0.416 |         0.576 |       0.288 |  0.153 |           0.283 |
| rf__rdkit2d                 | random       | occ_bin    | 4             |   63 | 0.436 |        0.343 |         0.535 |       0.305 |  0.047 |           0.306 |
| rf__rdkit2d                 | random       | occ_bin    | >=5           |   44 | 0.727 |        0.511 |         0.968 |       0.407 |  0.279 |           0.542 |
| rf__rdkit2d                 | random       | source     | A             | 1270 | 0.787 |        0.741 |         0.835 |       0.486 |  0.036 |           0.096 |
| rf__rdkit2d                 | random       | source     | B             | 1408 | 0.558 |        0.532 |         0.585 |       0.409 | -0.035 |           0.04  |
| rf__rdkit2d                 | random       | source     | C             |  530 | 0.534 |        0.494 |         0.576 |       0.402 |  0.005 |           0.051 |
| rf__rdkit2d                 | random       | source     | D             |   68 | 0.458 |        0.373 |         0.547 |       0.346 |  0.029 |           0.047 |
| rf__rdkit2d                 | random       | source     | E             |   98 | 0.411 |        0.348 |         0.478 |       0.336 |  0.055 |           0.119 |
| rf__rdkit2d                 | random       | source     | F             |  315 | 0.755 |        0.682 |         0.832 |       0.573 | -0.15  |           0.026 |
| rf__rdkit2d                 | random       | source     | G             |   56 | 0.41  |        0.322 |         0.509 |       0.347 |  0.156 |           0.109 |
| rf__rdkit2d                 | random       | source     | H             |   19 | 0.793 |        0.421 |         1.367 |       0.491 |  0.356 |           0.065 |
| rf__rdkit2d                 | random       | source     | I             |   12 | 0.718 |        0.491 |         0.994 |       0.562 |  0.409 |           0.107 |
| rf__rdkit2d                 | random       | sd_bin     | single report | 2911 | 0.683 |        0.658 |         0.708 |       0.477 | -0.029 |           0     |
| rf__rdkit2d                 | random       | sd_bin     | (0, 0.1]      |  384 | 0.431 |        0.384 |         0.479 |       0.281 |  0.056 |           0.039 |
| rf__rdkit2d                 | random       | sd_bin     | (0.1, 0.25]   |  197 | 0.452 |        0.396 |         0.512 |       0.342 |  0.11  |           0.163 |
| rf__rdkit2d                 | random       | sd_bin     | (0.25, 0.5]   |  143 | 0.59  |        0.487 |         0.694 |       0.346 |  0.035 |           0.364 |
| rf__rdkit2d                 | random       | sd_bin     | (0.5, 1]      |   89 | 0.596 |        0.479 |         0.728 |       0.429 |  0.107 |           0.698 |
| rf__rdkit2d                 | random       | sd_bin     | > 1           |   52 | 0.796 |        0.614 |         0.998 |       0.6   |  0.14  |           1.449 |
| rf__rdkit2d                 | scaffold     | occ_bin    | 1             | 2596 | 0.833 |        0.804 |         0.863 |       0.633 |  0.039 |           0     |
| rf__rdkit2d                 | scaffold     | occ_bin    | 2             |  395 | 0.627 |        0.57  |         0.69  |       0.454 |  0.078 |           0.264 |
| rf__rdkit2d                 | scaffold     | occ_bin    | 3             |  134 | 0.721 |        0.606 |         0.852 |       0.454 |  0.168 |           0.327 |
| rf__rdkit2d                 | scaffold     | occ_bin    | 4             |   37 | 0.69  |        0.553 |         0.853 |       0.627 | -0.036 |           0.512 |
| rf__rdkit2d                 | scaffold     | occ_bin    | >=5           |   28 | 0.724 |        0.47  |         1.027 |       0.469 |  0.282 |           0.453 |
| rf__rdkit2d                 | scaffold     | source     | A             |  911 | 0.914 |        0.856 |         0.971 |       0.627 |  0.161 |           0.074 |
| rf__rdkit2d                 | scaffold     | source     | B             | 1124 | 0.705 |        0.668 |         0.743 |       0.523 | -0.047 |           0.043 |
| rf__rdkit2d                 | scaffold     | source     | C             |  623 | 0.751 |        0.707 |         0.798 |       0.625 |  0.103 |           0.037 |
| rf__rdkit2d                 | scaffold     | source     | D             |   56 | 0.786 |        0.627 |         0.952 |       0.667 |  0.017 |           0.118 |
| rf__rdkit2d                 | scaffold     | source     | E             |   66 | 0.501 |        0.401 |         0.607 |       0.372 | -0.097 |           0.122 |
| rf__rdkit2d                 | scaffold     | source     | F             |  308 | 0.971 |        0.888 |         1.059 |       0.782 | -0.106 |           0.045 |
| rf__rdkit2d                 | scaffold     | source     | G             |   55 | 0.571 |        0.451 |         0.692 |       0.372 |  0.228 |           0.082 |
| rf__rdkit2d                 | scaffold     | source     | H             |   30 | 1.249 |        0.935 |         1.593 |       1.247 |  0.758 |           0.111 |
| rf__rdkit2d                 | scaffold     | source     | I             |   17 | 0.815 |        0.562 |         1.086 |       0.529 |  0.3   |           0.275 |
| rf__rdkit2d                 | scaffold     | sd_bin     | single report | 2596 | 0.833 |        0.804 |         0.863 |       0.633 |  0.039 |           0     |
| rf__rdkit2d                 | scaffold     | sd_bin     | (0, 0.1]      |  238 | 0.621 |        0.547 |         0.691 |       0.407 |  0.139 |           0.039 |
| rf__rdkit2d                 | scaffold     | sd_bin     | (0.1, 0.25]   |  129 | 0.73  |        0.633 |         0.848 |       0.525 |  0.056 |           0.159 |
| rf__rdkit2d                 | scaffold     | sd_bin     | (0.25, 0.5]   |  107 | 0.659 |        0.524 |         0.819 |       0.413 |  0.192 |           0.375 |
| rf__rdkit2d                 | scaffold     | sd_bin     | (0.5, 1]      |   79 | 0.617 |        0.501 |         0.739 |       0.434 |  0.023 |           0.691 |
| rf__rdkit2d                 | scaffold     | sd_bin     | > 1           |   41 | 0.71  |        0.553 |         0.886 |       0.529 | -0.065 |           1.354 |
| rf__rdkit2d                 | cluster      | occ_bin    | 1             | 2952 | 0.775 |        0.747 |         0.802 |       0.545 | -0.029 |           0     |
| rf__rdkit2d                 | cluster      | occ_bin    | 2             |  547 | 0.549 |        0.501 |         0.598 |       0.379 | -0.006 |           0.265 |
| rf__rdkit2d                 | cluster      | occ_bin    | 3             |  208 | 0.561 |        0.478 |         0.651 |       0.332 |  0.128 |           0.306 |
| rf__rdkit2d                 | cluster      | occ_bin    | 4             |   68 | 0.504 |        0.389 |         0.626 |       0.299 |  0.047 |           0.346 |
| rf__rdkit2d                 | cluster      | occ_bin    | >=5           |   48 | 0.532 |        0.355 |         0.736 |       0.325 | -0.076 |           0.403 |
| rf__rdkit2d                 | cluster      | source     | A             | 1208 | 0.876 |        0.824 |         0.926 |       0.562 |  0.098 |           0.1   |
| rf__rdkit2d                 | cluster      | source     | B             | 1443 | 0.614 |        0.583 |         0.646 |       0.431 | -0.08  |           0.045 |
| rf__rdkit2d                 | cluster      | source     | C             |  565 | 0.649 |        0.601 |         0.698 |       0.481 | -0.034 |           0.043 |
| rf__rdkit2d                 | cluster      | source     | D             |   60 | 0.578 |        0.472 |         0.687 |       0.462 | -0.087 |           0.094 |
| rf__rdkit2d                 | cluster      | source     | E             |   80 | 0.412 |        0.338 |         0.494 |       0.329 | -0.042 |           0.125 |
| rf__rdkit2d                 | cluster      | source     | F             |  346 | 0.902 |        0.825 |         0.975 |       0.726 | -0.193 |           0.034 |
| rf__rdkit2d                 | cluster      | source     | G             |   76 | 0.496 |        0.401 |         0.596 |       0.361 |  0.212 |           0.115 |
| rf__rdkit2d                 | cluster      | source     | H             |   33 | 0.744 |        0.487 |         1.094 |       0.493 |  0.176 |           0.08  |
| rf__rdkit2d                 | cluster      | source     | I             |   12 | 0.856 |        0.565 |         1.2   |       0.639 |  0.513 |           0.186 |
| rf__rdkit2d                 | cluster      | sd_bin     | single report | 2952 | 0.775 |        0.747 |         0.802 |       0.545 | -0.029 |           0     |
| rf__rdkit2d                 | cluster      | sd_bin     | (0, 0.1]      |  362 | 0.436 |        0.39  |         0.483 |       0.296 | -0.043 |           0.04  |
| rf__rdkit2d                 | cluster      | sd_bin     | (0.1, 0.25]   |  211 | 0.554 |        0.483 |         0.633 |       0.403 |  0.122 |           0.159 |
| rf__rdkit2d                 | cluster      | sd_bin     | (0.25, 0.5]   |  135 | 0.591 |        0.486 |         0.718 |       0.337 |  0.038 |           0.376 |
| rf__rdkit2d                 | cluster      | sd_bin     | (0.5, 1]      |  107 | 0.655 |        0.541 |         0.789 |       0.506 |  0.072 |           0.681 |
| rf__rdkit2d                 | cluster      | sd_bin     | > 1           |   56 | 0.931 |        0.72  |         1.171 |       0.697 | -0     |           1.422 |