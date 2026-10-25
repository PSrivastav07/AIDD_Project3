"""Step 6 - classical-model benchmark with featurisation comparison.

Stages (run them in order; each is resumable and can be re-run to continue after a time-out):

``tune``     random search on VALIDATION data only (random seed-0 + scaffold seed-0 val subsets):
             LightGBM on rdkit2d / morgan_counts / rdkit2d+morgan_counts (registry default + N-1 random
             configs), random forest on rdkit2d / rdkit2d+morgan_counts, k of the Tanimoto k-NN.
             Writes <out>/tuning_trials.csv and configs/models_tuned.yaml (re-written from all trials
             after every invocation, so a larger --lgbm-trials simply extends the search).
``grid``     baselines + {ridge, rf, lgbm, mlp} x {rdkit2d, morgan_counts, maccs, rdkit2d+morgan_counts}
             on random / scaffold / cluster x seeds 0-4 with the tuned params
             (<out>/predictions/..., metrics_runs.csv, grid_timing.csv).
``analyze``  tables (results/tables/benchmark_*), figures (results/figures/benchmark/) and
             <out>/summary.json.

With a non-default ``--out`` (e.g. a smoke test) the tuned YAML, tables and figures go to
``<out>/models_tuned.yaml``, ``<out>/tables`` and ``<out>/figures`` unless given explicitly, so a
smoke run never touches the real outputs.

Examples::

    python scripts/06_benchmark.py tune --n-jobs 4
    python scripts/06_benchmark.py grid --n-jobs 4 --models lgbm --split-types scaffold
    python scripts/06_benchmark.py grid --drop mlp__morgan_counts --drop-reason "compute budget"
    python scripts/06_benchmark.py analyze
    python scripts/06_benchmark.py tune --out results/benchmark/_smoke --lgbm-trials 2 --rf-trials 2
"""
from __future__ import annotations

import os
import sys


def _early_n_jobs(argv: list[str], default: int = 3) -> int:
    """``--n-jobs`` value read before numpy / lightgbm are imported (thread pools size at import)."""
    for i, a in enumerate(argv):
        if a == "--n-jobs" and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith("--n-jobs="):
            return int(a.split("=", 1)[1])
    return default


for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(_early_n_jobs(sys.argv)))

import argparse  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from datetime import datetime  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402
from threadpoolctl import threadpool_limits  # noqa: E402

from solpred import benchmark as bm  # noqa: E402
from solpred.config import ROOT, load_config, load_json, path, save_json  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_TUNED_YAML = ROOT / "configs" / "models_tuned.yaml"


# ------------------------------------------------------------------------------------ paths / logging
def _paths(args, cfg: dict) -> dict[str, Path]:
    """Output locations; a non-default ``--out`` redirects the YAML, tables and figures under it."""
    default_out = path("results", cfg) / bm.EXPERIMENT
    out = Path(args.out).resolve() if args.out else default_out
    is_default = out == default_out.resolve()
    p = {"out": out,
         "yaml": Path(args.tuned_yaml) if args.tuned_yaml else
         (DEFAULT_TUNED_YAML if is_default else out / "models_tuned.yaml"),
         "tables": Path(args.tables_dir) if args.tables_dir else (path("tables", cfg) if is_default else out / "tables"),
         "figures": Path(args.fig_dir) if args.fig_dir else
         (path("figures", cfg) / bm.EXPERIMENT if is_default else out / "figures"),
         "baseline": Path(args.baseline_dir) if args.baseline_dir else path("results", cfg) / "baseline"}
    out.mkdir(parents=True, exist_ok=True)
    return p


def _log_factory(out: Path):
    logfile = out / "run.log"

    def log(msg: str) -> None:
        line = f"[{datetime.now():%H:%M:%S}] {msg}"
        print(line, flush=True)
        with open(logfile, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")
    return log


def _record_stage(out: Path, stage: str, seconds: float, n_units: int, n_jobs: int, note: str = "") -> None:
    """Append wall-clock time of one invocation to compute_log.csv (summed in summary.json)."""
    f = out / "compute_log.csv"
    pd.DataFrame([{"stage": stage, "finished": datetime.now().isoformat(timespec="seconds"),
                   "wall_seconds": round(seconds, 1), "n_units": n_units, "n_jobs": n_jobs, "note": note}]).to_csv(
        f, mode="a", header=not f.exists(), index=False)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


# ------------------------------------------------------------------------------------ stages
def cmd_tune(args, cfg) -> None:
    """Random search on val; then (re)write the tuned-params YAML from all trials so far."""
    p = _paths(args, cfg)
    out, log = p["out"], _log_factory(p["out"])
    trials_csv = out / "tuning_trials.csv"
    n_before = len(pd.read_csv(trials_csv)) if trials_csv.exists() else 0
    pairs = [tuple(x.split("__")) for x in args.pairs] if args.pairs else None
    t0 = time.time()
    deadline = t0 + 60 * args.max_minutes if args.max_minutes else None
    with threadpool_limits(limits=args.n_jobs):
        trials = bm.run_tuning(trials_csv, args.models or ["knn", "lgbm", "rf"],
                               {"lgbm": args.lgbm_trials, "rf": args.rf_trials}, args.n_jobs, deadline, log, pairs)
    _record_stage(out, "tune", time.time() - t0, len(trials) - n_before, args.n_jobs)
    selected = bm.select_params(trials)
    obj = bm.tuning_objective(trials)
    header = ("Hyper-parameters chosen by scripts/06_benchmark.py tune (random search, validation data only).\n"
              f"Objective: mean val MAE over random/seed0 and scaffold/seed0. Trials: {_rel(trials_csv)}.\n"
              f"Generated {datetime.now():%Y-%m-%d %H:%M}. ridge (RidgeCV) and mlp use registry defaults.")
    bm.save_tuned_yaml(selected, p["yaml"], header)
    bm.to_markdown(obj, out / "tuning_objective.md",
                   "Tuning objective per configuration (val MAE, lower is better)")
    for model, feats in selected.items():
        for feat, e in feats.items():
            log(f"selected {model}/{feat}: {e.get('objective_val_MAE', e['source'])} {e['params']}")
    log(f"tuned params -> {_rel(p['yaml'])}")


def cmd_grid(args, cfg) -> None:
    p = _paths(args, cfg)
    out, log = p["out"], _log_factory(p["out"])
    tuned = bm.load_tuned_params(p["yaml"])
    if not tuned:
        log(f"WARNING: {_rel(p['yaml'])} missing - registry defaults are used")
    configs = bm.grid_configs(baselines=not args.no_baselines)
    if args.models:  # learners and/or baseline names (mean, esol_linear, knn)
        configs = [c for c in configs if c[0] in args.models]
    if args.featurizers:
        configs = [c for c in configs if c[1] in args.featurizers]
    if args.configs:
        configs = [c for c in configs if bm.config_label(*c) in set(args.configs)]
    if args.drop:
        _record_drops(out, args.drop, args.drop_reason)
    drops = set(_load_drops(out))
    configs = [c for c in configs if bm.config_label(*c) not in drops]
    log(f"grid: {len(configs)} configs x {args.split_types} x seeds {args.seeds} (n_jobs={args.n_jobs})")
    t0 = time.time()
    deadline = t0 + 60 * args.max_minutes if args.max_minutes else None
    with threadpool_limits(limits=args.n_jobs):
        n = bm.run_grid(out, configs, args.split_types, args.seeds, tuned, args.n_jobs, deadline, log)
    _record_stage(out, "grid", time.time() - t0, n, args.n_jobs)


def _record_drops(out: Path, labels: list[str], reason: str) -> None:
    f = out / "dropped_configs.json"
    d = load_json(f) if f.exists() else {}
    for lab in labels:
        d[lab] = reason or "dropped to stay within the compute budget"
    save_json(d, f)


def _load_drops(out: Path) -> dict:
    f = out / "dropped_configs.json"
    return load_json(f) if f.exists() else {}


def cmd_analyze(args, cfg) -> None:
    p = _paths(args, cfg)
    bm.analyze(p["out"], p["tables"], p["figures"], p["baseline"], p["yaml"], _load_drops(p["out"]),
               n_boot=args.n_boot, log=_log_factory(p["out"]), seeds=args.seeds)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["tune", "grid", "analyze"])
    ap.add_argument("--config", default=None, help="config YAML (default configs/default.yaml)")
    ap.add_argument("--n-jobs", type=int, default=3, help="threads per model fit (and BLAS / OpenMP limit)")
    ap.add_argument("--out", default=None, help="output directory (default results/benchmark)")
    ap.add_argument("--tuned-yaml", default=None, help="tuned-params YAML (default configs/models_tuned.yaml)")
    ap.add_argument("--tables-dir", default=None, help="analyze: tables directory (default results/tables)")
    ap.add_argument("--fig-dir", default=None, help="analyze: figure directory (default results/figures/benchmark)")
    ap.add_argument("--baseline-dir", default=None, help="analyze: Phase-1 predictions (default results/baseline)")
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="stop starting new fits after this many minutes (resume by re-running)")
    ap.add_argument("--models", nargs="+", default=None,
                    help="tune: subset of knn/lgbm/rf; grid: learners and/or baselines (mean, esol_linear, knn)")
    ap.add_argument("--pairs", nargs="+", default=None, help="tune: restrict to model__featurizer pairs")
    ap.add_argument("--featurizers", nargs="+", default=None, help="grid: restrict featurisers")
    ap.add_argument("--configs", nargs="+", default=None, help="grid: restrict to model__featurizer labels")
    ap.add_argument("--split-types", nargs="+", default=list(bm.SPLIT_TYPES), help="grid: split types")
    ap.add_argument("--seeds", nargs="+", type=int, default=list(bm.SEEDS),
                    help="grid: split seeds to run; analyze: seeds a config needs to count as complete")
    ap.add_argument("--no-baselines", action="store_true", help="grid: skip mean / esol_linear / knn")
    ap.add_argument("--drop", nargs="+", default=None, help="grid: config labels (model__featurizer) to drop")
    ap.add_argument("--drop-reason", default="", help="grid: reason stored with --drop (reported in summary.json)")
    ap.add_argument("--lgbm-trials", type=int, default=12, help="tune: LightGBM configs (default + random)")
    ap.add_argument("--rf-trials", type=int, default=6, help="tune: random-forest configs")
    ap.add_argument("--n-boot", type=int, default=2000, help="analyze: paired-bootstrap resamples")
    return ap


if __name__ == "__main__":
    a = build_parser().parse_args()
    conf = load_config(a.config)
    {"tune": cmd_tune, "grid": cmd_grid, "analyze": cmd_analyze}[a.stage](a, conf)
