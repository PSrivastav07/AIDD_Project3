# Featurisation comparison: best learner per family and split (mean ± sd over seeds)

| family                       | split_type   | best_config                 |   n_seeds | MAE           | RMSE          | R2            | Pearson       | Spearman      |
|:-----------------------------|:-------------|:----------------------------|----------:|:--------------|:--------------|:--------------|:--------------|:--------------|
| ESOL descriptors (4)         | random       | esol_linear__esol4          |         5 | 1.187 ± 0.043 | 1.556 ± 0.045 | 0.566 ± 0.034 | 0.760 ± 0.027 | 0.786 ± 0.012 |
| ESOL descriptors (4)         | scaffold     | esol_linear__esol4          |         5 | 1.251 ± 0.094 | 1.738 ± 0.218 | 0.498 ± 0.061 | 0.726 ± 0.041 | 0.810 ± 0.055 |
| ESOL descriptors (4)         | cluster      | esol_linear__esol4          |         5 | 1.226 ± 0.091 | 1.586 ± 0.149 | 0.567 ± 0.075 | 0.768 ± 0.056 | 0.801 ± 0.039 |
| RDKit 2D descriptors         | random       | lgbm__rdkit2d               |         5 | 0.580 ± 0.020 | 0.845 ± 0.023 | 0.872 ± 0.011 | 0.934 ± 0.006 | 0.929 ± 0.006 |
| RDKit 2D descriptors         | scaffold     | lgbm__rdkit2d               |         5 | 0.722 ± 0.024 | 0.993 ± 0.039 | 0.833 ± 0.028 | 0.914 ± 0.015 | 0.909 ± 0.015 |
| RDKit 2D descriptors         | cluster      | lgbm__rdkit2d               |         5 | 0.677 ± 0.044 | 0.967 ± 0.068 | 0.839 ± 0.026 | 0.916 ± 0.014 | 0.916 ± 0.012 |
| MACCS keys                   | random       | lgbm__maccs                 |         5 | 0.790 ± 0.012 | 1.098 ± 0.020 | 0.784 ± 0.011 | 0.886 ± 0.006 | 0.883 ± 0.012 |
| MACCS keys                   | scaffold     | lgbm__maccs                 |         5 | 0.973 ± 0.037 | 1.299 ± 0.059 | 0.715 ± 0.052 | 0.850 ± 0.031 | 0.851 ± 0.030 |
| MACCS keys                   | cluster      | lgbm__maccs                 |         5 | 0.938 ± 0.104 | 1.273 ± 0.148 | 0.718 ± 0.071 | 0.849 ± 0.040 | 0.857 ± 0.038 |
| Morgan fingerprint           | random       | lgbm__morgan_counts         |         5 | 0.695 ± 0.012 | 0.968 ± 0.017 | 0.832 ± 0.014 | 0.912 ± 0.008 | 0.900 ± 0.008 |
| Morgan fingerprint           | scaffold     | lgbm__morgan_counts         |         5 | 0.871 ± 0.041 | 1.166 ± 0.049 | 0.767 ± 0.059 | 0.879 ± 0.033 | 0.866 ± 0.034 |
| Morgan fingerprint           | cluster      | lgbm__morgan_counts         |         5 | 0.809 ± 0.035 | 1.119 ± 0.046 | 0.784 ± 0.035 | 0.886 ± 0.020 | 0.884 ± 0.019 |
| Descriptors + Morgan         | random       | lgbm__rdkit2d+morgan_counts |         5 | 0.574 ± 0.019 | 0.840 ± 0.030 | 0.873 ± 0.012 | 0.935 ± 0.006 | 0.930 ± 0.006 |
| Descriptors + Morgan         | scaffold     | lgbm__rdkit2d+morgan_counts |         5 | 0.717 ± 0.027 | 0.987 ± 0.044 | 0.836 ± 0.026 | 0.916 ± 0.014 | 0.910 ± 0.014 |
| Descriptors + Morgan         | cluster      | lgbm__rdkit2d+morgan_counts |         5 | 0.670 ± 0.037 | 0.959 ± 0.065 | 0.841 ± 0.026 | 0.918 ± 0.014 | 0.917 ± 0.012 |
| Learned graph (D-MPNN)       | random       | dmpnn__graph                |         5 | 0.653 ± 0.013 | 0.929 ± 0.034 | 0.845 ± 0.015 | 0.920 ± 0.008 | 0.912 ± 0.010 |
| Learned graph (D-MPNN)       | scaffold     | dmpnn__graph                |         5 | 0.811 ± 0.030 | 1.106 ± 0.006 | 0.792 ± 0.046 | 0.891 ± 0.026 | 0.879 ± 0.025 |
| Learned graph (D-MPNN)       | cluster      | dmpnn__graph                |         5 | 0.718 ± 0.034 | 1.011 ± 0.050 | 0.823 ± 0.031 | 0.908 ± 0.017 | 0.905 ± 0.016 |
| Graph + descriptors (D-MPNN) | random       | dmpnn_desc__graph+rdkit2d   |         3 | 0.577 ± 0.013 | 0.841 ± 0.014 | 0.872 ± 0.013 | 0.935 ± 0.007 | 0.927 ± 0.010 |
| Graph + descriptors (D-MPNN) | scaffold     | dmpnn_desc__graph+rdkit2d   |         3 | 0.805 ± 0.024 | 1.104 ± 0.037 | 0.822 ± 0.021 | 0.913 ± 0.007 | 0.908 ± 0.010 |
| Graph + descriptors (D-MPNN) | cluster      | dmpnn_desc__graph+rdkit2d   |         3 | 0.707 ± 0.064 | 1.000 ± 0.062 | 0.834 ± 0.040 | 0.915 ± 0.021 | 0.914 ± 0.017 |
