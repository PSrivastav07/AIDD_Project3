| model                       | split_type   | grouping          | level                |    n |   MAE |   MAE_ci_low |   MAE_ci_high |   median_AE |   bias |   mean_label_sd |
|:----------------------------|:-------------|:------------------|:---------------------|-----:|------:|-------------:|--------------:|------------:|-------:|----------------:|
| lgbm__rdkit2d+morgan_counts | random       | reliability_group | G1                   | 2911 | 0.606 |        0.583 |         0.629 |       0.407 | -0.016 |           0     |
| lgbm__rdkit2d+morgan_counts | random       | reliability_group | G2                   |   85 | 0.615 |        0.495 |         0.743 |       0.377 | -0.052 |           0.925 |
| lgbm__rdkit2d+morgan_counts | random       | reliability_group | G3                   |  473 | 0.421 |        0.383 |         0.462 |       0.293 |  0.018 |           0.123 |
| lgbm__rdkit2d+morgan_counts | random       | reliability_group | G4                   |   56 | 0.689 |        0.514 |         0.881 |       0.435 |  0.19  |           1.051 |
| lgbm__rdkit2d+morgan_counts | random       | reliability_group | G5                   |  251 | 0.4   |        0.356 |         0.448 |       0.282 |  0.101 |           0.163 |
| lgbm__rdkit2d+morgan_counts | random       | rel3              | single (G1)          | 2911 | 0.606 |        0.583 |         0.629 |       0.407 | -0.016 |           0     |
| lgbm__rdkit2d+morgan_counts | random       | rel3              | consistent (G3+G5)   |  724 | 0.414 |        0.384 |         0.446 |       0.288 |  0.047 |           0.137 |
| lgbm__rdkit2d+morgan_counts | random       | rel3              | inconsistent (G2+G4) |  141 | 0.644 |        0.541 |         0.756 |       0.419 |  0.044 |           0.975 |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group | G1                   | 2596 | 0.759 |        0.731 |         0.787 |       0.552 |  0.021 |           0     |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group | G2                   |   72 | 0.654 |        0.524 |         0.791 |       0.438 |  0.019 |           0.886 |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group | G3                   |  323 | 0.564 |        0.507 |         0.626 |       0.407 |  0.065 |           0.126 |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group | G4                   |   48 | 0.636 |        0.492 |         0.796 |       0.5   | -0.122 |           0.964 |
| lgbm__rdkit2d+morgan_counts | scaffold     | reliability_group | G5                   |  151 | 0.685 |        0.591 |         0.785 |       0.483 |  0.21  |           0.193 |
| lgbm__rdkit2d+morgan_counts | scaffold     | rel3              | single (G1)          | 2596 | 0.759 |        0.731 |         0.787 |       0.552 |  0.021 |           0     |
| lgbm__rdkit2d+morgan_counts | scaffold     | rel3              | consistent (G3+G5)   |  474 | 0.602 |        0.551 |         0.657 |       0.428 |  0.111 |           0.147 |
| lgbm__rdkit2d+morgan_counts | scaffold     | rel3              | inconsistent (G2+G4) |  120 | 0.647 |        0.547 |         0.749 |       0.485 | -0.038 |           0.917 |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group | G1                   | 2952 | 0.701 |        0.675 |         0.727 |       0.476 | -0.051 |           0     |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group | G2                   |   96 | 0.722 |        0.594 |         0.868 |       0.535 | -0.098 |           0.916 |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group | G3                   |  451 | 0.462 |        0.42  |         0.51  |       0.315 | -0.039 |           0.126 |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group | G4                   |   67 | 0.732 |        0.583 |         0.912 |       0.573 | -0.041 |           0.964 |
| lgbm__rdkit2d+morgan_counts | cluster      | reliability_group | G5                   |  257 | 0.487 |        0.429 |         0.55  |       0.331 |  0.025 |           0.163 |
| lgbm__rdkit2d+morgan_counts | cluster      | rel3              | single (G1)          | 2952 | 0.701 |        0.675 |         0.727 |       0.476 | -0.051 |           0     |
| lgbm__rdkit2d+morgan_counts | cluster      | rel3              | consistent (G3+G5)   |  708 | 0.471 |        0.436 |         0.506 |       0.325 | -0.016 |           0.14  |
| lgbm__rdkit2d+morgan_counts | cluster      | rel3              | inconsistent (G2+G4) |  163 | 0.726 |        0.628 |         0.829 |       0.543 | -0.075 |           0.936 |
| rf__rdkit2d                 | random       | reliability_group | G1                   | 2911 | 0.683 |        0.658 |         0.708 |       0.477 | -0.029 |           0     |
| rf__rdkit2d                 | random       | reliability_group | G2                   |   85 | 0.604 |        0.507 |         0.715 |       0.491 |  0.035 |           0.925 |
| rf__rdkit2d                 | random       | reliability_group | G3                   |  473 | 0.474 |        0.431 |         0.519 |       0.335 |  0.034 |           0.123 |
| rf__rdkit2d                 | random       | reliability_group | G4                   |   56 | 0.769 |        0.564 |         0.987 |       0.472 |  0.247 |           1.051 |
| rf__rdkit2d                 | random       | reliability_group | G5                   |  251 | 0.458 |        0.4   |         0.523 |       0.269 |  0.128 |           0.163 |
| rf__rdkit2d                 | random       | rel3              | single (G1)          | 2911 | 0.683 |        0.658 |         0.708 |       0.477 | -0.029 |           0     |
| rf__rdkit2d                 | random       | rel3              | consistent (G3+G5)   |  724 | 0.468 |        0.433 |         0.506 |       0.308 |  0.066 |           0.137 |
| rf__rdkit2d                 | random       | rel3              | inconsistent (G2+G4) |  141 | 0.67  |        0.563 |         0.788 |       0.474 |  0.119 |           0.975 |
| rf__rdkit2d                 | scaffold     | reliability_group | G1                   | 2596 | 0.833 |        0.804 |         0.863 |       0.633 |  0.039 |           0     |
| rf__rdkit2d                 | scaffold     | reliability_group | G2                   |   72 | 0.625 |        0.504 |         0.748 |       0.46  |  0.053 |           0.886 |
| rf__rdkit2d                 | scaffold     | reliability_group | G3                   |  323 | 0.628 |        0.561 |         0.697 |       0.454 |  0.084 |           0.126 |
| rf__rdkit2d                 | scaffold     | reliability_group | G4                   |   48 | 0.684 |        0.534 |         0.844 |       0.528 | -0.098 |           0.964 |
| rf__rdkit2d                 | scaffold     | reliability_group | G5                   |  151 | 0.726 |        0.618 |         0.844 |       0.473 |  0.224 |           0.193 |
| rf__rdkit2d                 | scaffold     | rel3              | single (G1)          | 2596 | 0.833 |        0.804 |         0.863 |       0.633 |  0.039 |           0     |
| rf__rdkit2d                 | scaffold     | rel3              | consistent (G3+G5)   |  474 | 0.659 |        0.602 |         0.721 |       0.457 |  0.128 |           0.147 |
| rf__rdkit2d                 | scaffold     | rel3              | inconsistent (G2+G4) |  120 | 0.649 |        0.556 |         0.744 |       0.463 | -0.007 |           0.917 |
| rf__rdkit2d                 | cluster      | reliability_group | G1                   | 2952 | 0.775 |        0.747 |         0.802 |       0.545 | -0.029 |           0     |
| rf__rdkit2d                 | cluster      | reliability_group | G2                   |   96 | 0.742 |        0.627 |         0.873 |       0.605 |  0.007 |           0.916 |
| rf__rdkit2d                 | cluster      | reliability_group | G3                   |  451 | 0.508 |        0.457 |         0.561 |       0.34  | -0.009 |           0.126 |
| rf__rdkit2d                 | cluster      | reliability_group | G4                   |   67 | 0.762 |        0.57  |         0.993 |       0.425 |  0.104 |           0.964 |
| rf__rdkit2d                 | cluster      | reliability_group | G5                   |  257 | 0.488 |        0.425 |         0.552 |       0.302 |  0.075 |           0.163 |
| rf__rdkit2d                 | cluster      | rel3              | single (G1)          | 2952 | 0.775 |        0.747 |         0.802 |       0.545 | -0.029 |           0     |
| rf__rdkit2d                 | cluster      | rel3              | consistent (G3+G5)   |  708 | 0.501 |        0.462 |         0.541 |       0.33  |  0.022 |           0.14  |
| rf__rdkit2d                 | cluster      | rel3              | inconsistent (G2+G4) |  163 | 0.75  |        0.644 |         0.872 |       0.582 |  0.047 |           0.936 |