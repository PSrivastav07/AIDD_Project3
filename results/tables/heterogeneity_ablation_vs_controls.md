| split_type   | contrast                    |   n_control_reps | subset                        |   n_seeds |   delta_MAE |   ci_low |   ci_high |   p_not_better |
|:-------------|:----------------------------|-----------------:|:------------------------------|----------:|------------:|---------:|----------:|---------------:|
| random       | b_drop_G2G4 - e_random_drop |                3 | full test                     |         5 |      -0.001 |   -0.005 |     0.004 |          0.413 |
| random       | b_drop_G2G4 - e_random_drop |                3 | high-reliability test (G3+G5) |         5 |      -0     |   -0.01  |     0.01  |          0.486 |
| random       | b_drop_G2G4 - e_random_drop |                3 | single-report test (G1)       |         5 |      -0.002 |   -0.007 |     0.004 |          0.243 |
| random       | b_drop_G2G4 - e_random_drop |                3 | inconsistent test (G2+G4)     |         5 |       0.028 |   -0.004 |     0.059 |          0.956 |
| random       | d_ivw - f_ivw_permuted      |                1 | full test                     |         5 |      -0.002 |   -0.007 |     0.003 |          0.21  |
| random       | d_ivw - f_ivw_permuted      |                1 | high-reliability test (G3+G5) |         5 |      -0.003 |   -0.013 |     0.007 |          0.257 |
| random       | d_ivw - f_ivw_permuted      |                1 | single-report test (G1)       |         5 |      -0.001 |   -0.007 |     0.004 |          0.33  |
| random       | d_ivw - f_ivw_permuted      |                1 | inconsistent test (G2+G4)     |         5 |      -0.013 |   -0.036 |     0.013 |          0.156 |
| scaffold     | b_drop_G2G4 - e_random_drop |                3 | full test                     |         5 |       0.007 |    0.002 |     0.012 |          0.997 |
| scaffold     | b_drop_G2G4 - e_random_drop |                3 | high-reliability test (G3+G5) |         5 |       0.002 |   -0.011 |     0.016 |          0.64  |
| scaffold     | b_drop_G2G4 - e_random_drop |                3 | single-report test (G1)       |         5 |       0.007 |    0.001 |     0.013 |          0.994 |
| scaffold     | b_drop_G2G4 - e_random_drop |                3 | inconsistent test (G2+G4)     |         5 |       0.022 |   -0.005 |     0.05  |          0.948 |
| scaffold     | d_ivw - f_ivw_permuted      |                1 | full test                     |         5 |      -0     |   -0.006 |     0.005 |          0.446 |
| scaffold     | d_ivw - f_ivw_permuted      |                1 | high-reliability test (G3+G5) |         5 |      -0.004 |   -0.016 |     0.009 |          0.298 |
| scaffold     | d_ivw - f_ivw_permuted      |                1 | single-report test (G1)       |         5 |      -0.001 |   -0.007 |     0.005 |          0.436 |
| scaffold     | d_ivw - f_ivw_permuted      |                1 | inconsistent test (G2+G4)     |         5 |       0.016 |   -0.014 |     0.048 |          0.862 |