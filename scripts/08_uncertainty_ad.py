"""Step 8 - uncertainty quantification, conformal prediction and applicability domain (AD).

Stages (run in order; each is resumable and can be run on its own):

``train``    5-member LightGBM ensembles (rdkit2d+morgan_counts; subsample 0.8, colsample_bytree 0.5, members differ
             by seed) for random / scaffold / cluster x seeds 0-4 and the two external transfers (85 fits). Each
             model is fitted on the training pool minus a random 10 % calibration hold-out (``cal_random``) and
             early-stopped on the split's val subset. Member predictions go to
             results/uncertainty/members/<split_type>/<split_name>/member<m>.csv; existing members are skipped.
``collect``  merge members + Phase-1 RF (tree std) + kNN distance + AD scores (Tanimoto, Mahalanobis) into
             results/uncertainty/predictions/<split_type>/<split_name>/lgbm_ens5__rdkit2d+morgan_counts.csv
             (single-threaded, ~1 min per split; existing files are skipped unless --force).
``analyze``  uncertainty quality, calibration, conformal (matched and under shift), AD tables / figures and
             results/uncertainty/summary.json (single-threaded, a few minutes).
``status``   print how many ensemble members / merged files exist.

Examples::

    .venv/Scripts/python.exe scripts/08_uncertainty_ad.py train --n-jobs 4
    .venv/Scripts/python.exe scripts/08_uncertainty_ad.py train --split-types scaffold --seeds 0 1 --no-external
    .venv/Scripts/python.exe scripts/08_uncertainty_ad.py collect
    .venv/Scripts/python.exe scripts/08_uncertainty_ad.py analyze
    # smoke test of the whole pipeline on one split into a throw-away directory (capped boosting rounds)
    .venv/Scripts/python.exe scripts/08_uncertainty_ad.py all --split-types scaffold --seeds 0 --no-external \
        --smoke --out-dir <tmpdir>
"""
from __future__ import annotations

import argparse
import os
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "2")

from solpred.analysis import uncertainty_eval as ue  # noqa: E402

STAGES = ("train", "collect", "analyze", "all", "status")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", nargs="?", choices=STAGES, help="pipeline stage (default: all)")
    ap.add_argument("--stage", dest="stage_opt", choices=STAGES, help="same as the positional stage (legacy)")
    ap.add_argument("--split-types", nargs="+", default=list(ue.SPLIT_TYPES), choices=list(ue.SPLIT_TYPES))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--no-external", action="store_true", help="skip the two external transfer splits")
    ap.add_argument("--members", nargs="+", type=int, default=list(range(ue.N_MEMBERS)))
    ap.add_argument("--n-jobs", type=int, default=3, help="LightGBM threads per fit (train stage)")
    ap.add_argument("--force", action="store_true", help="collect: rebuild merged prediction files")
    ap.add_argument("--n-boot", type=int, default=1000, help="analyze: bootstrap resamples for AD-bin CIs")
    ap.add_argument("--out-dir", default=None,
                    help="write ALL outputs below this directory instead of results/ (smoke tests)")
    ap.add_argument("--smoke", action="store_true",
                    help="cap LightGBM at 150 boosting rounds (only allowed together with --out-dir)")
    args = ap.parse_args()
    args.stage = args.stage or args.stage_opt or "all"
    if args.smoke and not args.out_dir:
        ap.error("--smoke must be combined with --out-dir so smoke outputs never mix with real results")
    return args


def main() -> None:
    args = parse_args()
    ue.set_output_base(args.out_dir)
    log = lambda s: print(s, flush=True)  # noqa: E731
    runs = ue.run_list(args.split_types, args.seeds, external=not args.no_external)
    t0 = time.time()
    have, need = ue.train_status(runs)
    log(f"[08] stage={args.stage} runs={len(runs)} members on disk {have}/{need} -> {ue.out_root()}")
    if args.stage == "status":
        n_merged = sum(ue.prediction_file(st, sn).exists() for st, sn, _ in runs)
        log(f"[08] merged prediction files {n_merged}/{len(runs)}; "
            f"summary.json {'present' if (ue.out_root() / 'summary.json').exists() else 'missing'}")
        return
    if args.stage in ("train", "all"):
        extra = {"n_estimators": 150} if args.smoke else None
        ue.train_ensembles(runs, n_jobs=args.n_jobs, members=args.members, log=log, extra_params=extra)
        log(f"[08] train done: members {ue.train_status(runs)[0]}/{need} ({time.time() - t0:.0f}s)")
    if args.stage in ("collect", "all"):
        ue.collect_all(runs, force=args.force, log=log)
    if args.stage in ("analyze", "all"):
        ue.analyze_all(log=log, n_boot=args.n_boot)
    log(f"[08] finished stage={args.stage} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
