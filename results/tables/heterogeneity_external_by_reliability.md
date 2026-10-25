| direction       | model                     | rel3                 |    n |   MAE |   ci_low |   ci_high |
|:----------------|:--------------------------|:---------------------|-----:|------:|---------:|----------:|
| esol_to_aqsoldb | LightGBM (rdkit2d+Morgan) | single (G1)          | 6673 | 0.998 |    0.976 |     1.02  |
| esol_to_aqsoldb | LightGBM (rdkit2d+Morgan) | consistent (G3+G5)   | 1120 | 0.756 |    0.711 |     0.801 |
| esol_to_aqsoldb | LightGBM (rdkit2d+Morgan) | inconsistent (G2+G4) |  281 | 0.944 |    0.848 |     1.044 |
| esol_to_aqsoldb | RF (rdkit2d)              | single (G1)          | 6673 | 1.029 |    1.008 |     1.051 |
| esol_to_aqsoldb | RF (rdkit2d)              | consistent (G3+G5)   | 1120 | 0.799 |    0.753 |     0.845 |
| esol_to_aqsoldb | RF (rdkit2d)              | inconsistent (G2+G4) |  281 | 0.926 |    0.831 |     1.028 |