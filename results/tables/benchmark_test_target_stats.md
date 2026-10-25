# Test-set logS distribution per split type

Mean over the 5 seeds of each test set's logS mean and sd (ddof=1). R2 = 1 - MSE / var(test logS): the same RMSE yields a higher R2 on a more spread-out test set. Per seed: benchmark_test_target_stats.csv.

| split_type   |   n_test |   logS_mean |   logS_mean_sd |   logS_std_mean |   logS_std_sd |   logS_std_min |   logS_std_max |
|:-------------|---------:|------------:|---------------:|----------------:|--------------:|---------------:|---------------:|
| random       |  917.000 |      -2.914 |          0.074 |           2.367 |         0.085 |          2.273 |          2.463 |
| scaffold     |  918.000 |      -3.576 |          0.488 |           2.456 |         0.244 |          2.106 |          2.692 |
| cluster      |  918.000 |      -3.096 |          0.282 |           2.426 |         0.201 |          2.205 |          2.732 |
