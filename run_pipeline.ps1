# Reproduce the whole project end to end (Windows PowerShell). Every step is resumable:
# re-running skips work whose outputs already exist. Approximate wall-clock times are for a
# 12-thread laptop CPU (i5-13420H, no GPU).
#   powershell -ExecutionPolicy Bypass -File run_pipeline.ps1
$ErrorActionPreference = "Stop"
$py = ".\.venv\Scripts\python.exe"
function Step($cmd) { Write-Host ">> $cmd" -ForegroundColor Cyan; Invoke-Expression $cmd; if ($LASTEXITCODE -ne 0) { throw "failed: $cmd" } }

# Phase 1 - data, splits, features, baselines                                    (~20 min)
Step "$py scripts/01_download_data.py"
Step "$py scripts/02_curate_data.py"
Step "$py scripts/03_make_splits.py"
Step "$py scripts/04_featurize.py"
Step "$py scripts/05_baseline.py --n-jobs 8"

# Phase 2 - modelling, uncertainty, heterogeneity, error analysis                (~5-6 h)
Step "$py scripts/06_benchmark.py tune --n-jobs 5 --lgbm-trials 10 --rf-trials 6"
Step "$py scripts/06_benchmark.py grid --n-jobs 5 --drop mlp__morgan_counts mlp__rdkit2d+morgan_counts --drop-reason 'compute budget: slow fits and clearly inferior validation MAE in the smoke test'"
Step "$py scripts/06_benchmark.py analyze"
Step "$py scripts/07_train_gnn.py --threads 3 --workers 3"
Step "$py scripts/07_train_gnn.py --analyze"
Step "$py scripts/08_uncertainty_ad.py train --n-jobs 4"
Step "$py scripts/08_uncertainty_ad.py collect"
Step "$py scripts/08_uncertainty_ad.py analyze"
Step "$py scripts/09_heterogeneity_transfer.py --stages noise reliability ablation loso transfer external report --n-jobs 3"
Step "$py scripts/10_error_analysis.py --n-jobs 3"

# Phase 3 - production model, headline tables, tests                             (~30 min)
Step "$py scripts/11_train_final_model.py --params-yaml configs/models_tuned.yaml --params-key lgbm__rdkit2d+morgan_counts --n-jobs 4"
Step "$py scripts/12_collect_results.py"
Step "$py -m pytest tests -q"
