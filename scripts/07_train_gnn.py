"""Step 7 - Chemprop-style directed message-passing neural networks (D-MPNN) on AqSolDB splits.

Models (experiment "gnn"; implementation in ``solpred.models.gnn`` / ``solpred.features.graph``)
    dmpnn       molecular graph only                                    featurizer "graph"
    dmpnn_desc  graph + RDKit 2D descriptors (217; median-imputed and   featurizer "graph+rdkit2d"
                standardised on the training rows, clipped to +-10) concatenated to the readout

Plan: dmpnn on random / scaffold / cluster x seeds 0-4, dmpnn_desc on seeds 0-2 of each split
(``--desc-seeds``). Hyper-parameters are fixed a priori (Chemprop defaults, see RATIONALE); the
only data-driven choice is early stopping on the validation MAE. The test subset is used only
for the final metrics.

Outputs (results/gnn/, formats identical to ``solpred.experiment``)
    metrics_runs.csv                                         one row per finished run (test subset)
    predictions/<split_type>/seed<k>/<model>__<featurizer>.csv
        compound_id, subset (val|test), y_true, y_pred, y_std; y_pred is the deterministic
        (dropout-off) prediction, y_std the std over ``mc_passes`` Monte-Carlo-dropout passes
    runs/<split_type>/seed<k>/<model>.json                   metrics row + training info + MC-dropout check
    curves/<split_type>/seed<k>/<model>.csv                  per-epoch training curves
    cache/aqsoldb__graph.npz                                 featurised graphs (rebuilt if SMILES change)
    --analyze: training_curves.csv, summary.csv/.md, comparison_vs_lgbm.csv/.md, summary.json
    Figures: results/figures/gnn/. Checkpoint of the scaffold-seed0 dmpnn: models/gnn/.

Modes
    (default)   train the selected runs, skipping finished ones (resumable). ``--workers N`` starts N
                worker processes with ``--threads / N`` torch threads each; workers claim jobs from the
                priority-ordered list (all dmpnn jobs first) through lock files in <out>/locks/, so faster
                cores take more jobs; logs go to <out>/logs/. On this contended 4P+4E-core CPU one thread
                per worker gave the best throughput (see results/gnn/profile.json).
    --profile   time featurisation, training epochs, validation and MC-dropout inference
                (random split, seed 0) and extrapolate the cost of the full plan
    --analyze   rebuild metrics_runs.csv, tables, figures and summary.json from finished runs

Examples
    python scripts/07_train_gnn.py --threads 6 --workers 6              # full plan
    python scripts/07_train_gnn.py --split-types scaffold --seeds 0 --models dmpnn --threads 3
    python scripts/07_train_gnn.py --profile --hidden 200 300 --threads 3
    python scripts/07_train_gnn.py --analyze
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from solpred.config import ROOT, load_json, path, save_json
from solpred.experiment import METRIC_COLUMNS, labels, prediction_path, summarize
from solpred.features import load_or_compute, subset
from solpred.features.graph import ATOM_FDIM, BOND_FDIM, dataset_store
from solpred.metrics import paired_bootstrap_mae_diff, regression_metrics
from solpred.models.gnn import (DescriptorScaler, GNNConfig, GraphSet, GraphTensors, embed, mc_dropout_std,
                                predict_from_embedding, save_checkpoint, set_determinism, train_dmpnn)
from solpred.splits import load_split

warnings.filterwarnings("ignore", category=UserWarning, module="torch")

EXPERIMENT = "gnn"
DATASET = "aqsoldb"
MODELS = {"dmpnn": "graph", "dmpnn_desc": "graph+rdkit2d"}
MODEL_LABELS = {"dmpnn": "D-MPNN (graph)", "dmpnn_desc": "D-MPNN + RDKit 2D"}
SPLIT_TYPES = ("random", "scaffold", "cluster")
PALETTE = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52"}
BASELINE_REF = ("lgbm", "rdkit2d+morgan_counts")
CHECKPOINT_RUN = ("scaffold", 0, "dmpnn")
CACHE_DIR = path("results") / EXPERIMENT / "cache"

DEFAULT_CFG = GNNConfig(hidden=200, depth=3, ffn_hidden=200, ffn_layers=2, dropout=0.1, encoder_dropout=0.0,
                        aggregation="mean", batch_size=64, eval_batch_size=1024, max_epochs=50, patience=10,
                        warmup_epochs=2.0, init_lr=1e-4, max_lr=1e-3, final_lr=1e-4, weight_decay=0.0,
                        grad_clip=10.0, mc_passes=10)
RATIONALE = {
    "architecture": "Chemprop defaults (Yang et al. 2019) except hidden size: depth 3 (2 directed-edge updates + "
                    "atom aggregation), mean readout, 2-layer FFN; hidden = FFN width = 200 instead of 300 because "
                    "an epoch costs 22.8 s vs 42.2 s per CPU thread (profile.json), which halves the plan's wall "
                    "time (1.2 h vs 2.1 h expected at 6 threads) and leaves room for contention; 2 epochs gave "
                    "similar val MAE (1.09 vs 1.04). No hyper-parameter search (compute budget), so no selection "
                    "effects on val or test beyond early stopping",
    "dropout": "p=0.1 in the FFN head only: mild regularisation and the Monte-Carlo-dropout rate; encoder dropout 0 "
               "(Chemprop default) keeps the encoder deterministic, so 10 MC passes cost one encoder pass",
    "optimisation": "Adam, MSE on train-standardised logS, batch 64, Noam schedule (2 warm-up epochs 1e-4 -> 1e-3, "
                    "exponential decay to 1e-4 at epoch 50), gradient-norm clip 10",
    "early_stopping": "patience 10 epochs on validation MAE (log units, dropout off); best-epoch weights restored",
    "budget": "max 50 epochs: dmpnn 3 splits x 5 seeds + dmpnn_desc 3 splits x 3 seeds = 24 runs, estimated "
              "1.2 h (worst case 1.3 h) with 6 single-thread workers (results/gnn/profile.json, plan_estimate)",
}


@dataclass
class Dirs:
    out: Path
    fig: Path
    model: Path


@dataclass
class SplitPart:
    ids: list[str]
    gs: GraphSet


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def build_config(args: argparse.Namespace) -> GNNConfig:
    """DEFAULT_CFG with command-line overrides (ffn_hidden follows hidden)."""
    cfg = GNNConfig.from_dict(DEFAULT_CFG.to_dict())
    for k in ("hidden", "depth", "dropout", "batch_size", "max_epochs", "patience", "mc_passes"):
        v = getattr(args, k, None)
        if v is not None:
            setattr(cfg, k, v[0] if isinstance(v, list) else v)
    cfg.ffn_hidden = cfg.hidden
    return cfg


_DATA: dict[str, GraphTensors] = {}


def graph_data(aggregation: str = "mean") -> GraphTensors:
    """Dataset-wide graph tensors (featurised once, cached on disk and in memory)."""
    if aggregation not in _DATA:
        _DATA[aggregation] = GraphTensors(dataset_store(DATASET, CACHE_DIR), aggregation)
    return _DATA[aggregation]


# --------------------------------------------------------------------------- one run
def split_parts(data: GraphTensors, split: dict, model_name: str) -> dict[str, SplitPart]:
    """Store positions, labels and (dmpnn_desc) standardised descriptors for train / val / test."""
    y = labels(DATASET)
    parts = {}
    for k in ("train", "val", "test"):
        ids = [str(i) for i in split[k]]
        parts[k] = SplitPart(ids, GraphSet(data.store.positions(ids), y.loc[ids].to_numpy(dtype=float)))
    if model_name == "dmpnn_desc":
        X, all_ids, _ = load_or_compute(DATASET, "rdkit2d")
        ds = DescriptorScaler(clip=10.0).fit(subset(X, all_ids, parts["train"].ids))
        for p in parts.values():
            p.gs.desc = ds.transform(subset(X, all_ids, p.ids))
    return parts


def run_paths(out: Path, stype: str, seed: int, model_name: str) -> dict[str, Path]:
    split_name = f"seed{seed}"
    return {"pred": prediction_path(out, stype, split_name, model_name, MODELS[model_name]),
            "run": out / "runs" / stype / split_name / f"{model_name}.json",
            "curve": out / "curves" / stype / split_name / f"{model_name}.csv"}


def run_done(out: Path, stype: str, seed: int, model_name: str) -> bool:
    p = run_paths(out, stype, seed, model_name)
    return p["pred"].exists() and p["run"].exists()


def evaluate(model, data: GraphTensors, part: SplitPart, scaler, cfg: GNNConfig, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic prediction and MC-dropout std, sharing one encoder pass."""
    z = embed(model, data.precollate(part.gs, cfg.eval_batch_size))
    pred = predict_from_embedding(model, z, scaler)
    std = mc_dropout_std(model, data, part.gs, scaler, cfg.mc_passes, seed=seed, z=z)
    return pred, std


def run_one(dirs: Dirs, stype: str, seed: int, model_name: str, cfg: GNNConfig) -> dict:
    """Train, predict (deterministic + MC-dropout std) and persist one run; returns the metrics row."""
    data = graph_data(cfg.aggregation)
    parts = split_parts(data, load_split(stype, seed), model_name)
    tr, va, te = parts["train"], parts["val"], parts["test"]
    log(f"{stype} seed{seed} {model_name}: n_train={len(tr.ids)} n_val={len(va.ids)} n_test={len(te.ids)} "
        f"threads={torch.get_num_threads()}")
    model, scaler, hist, info = train_dmpnn(cfg, data, tr.gs, va.gs, seed=seed, log=log)
    t0 = time.time()
    pv, sv = evaluate(model, data, va, scaler, cfg, seed)
    pt, st = evaluate(model, data, te, scaler, cfg, seed + 1)
    infer_s = time.time() - t0
    preds = pd.concat([
        pd.DataFrame({"compound_id": va.ids, "subset": "val", "y_true": va.gs.y, "y_pred": pv, "y_std": sv}),
        pd.DataFrame({"compound_id": te.ids, "subset": "test", "y_true": te.gs.y, "y_pred": pt, "y_std": st}),
    ], ignore_index=True)
    m = regression_metrics(te.gs.y, pt)
    row = {"experiment": EXPERIMENT, "train_dataset": DATASET, "test_dataset": DATASET, "split_type": stype,
           "split_name": f"seed{seed}", "seed": seed, "featurizer": MODELS[model_name], "model": model_name,
           "n_train": len(tr.ids), "n_val": len(va.ids), "n_test": len(te.ids),
           **{k: m[k] for k in ("MAE", "RMSE", "R2", "Pearson", "Spearman")},
           "val_MAE": regression_metrics(va.gs.y, pv)["MAE"], "fit_seconds": round(info["train_seconds"], 2)}
    mc = {f"spearman_std_abs_err_{k}": float(stats.spearmanr(s, np.abs(p - y))[0])
          for k, s, p, y in (("val", sv, pv, va.gs.y), ("test", st, pt, te.gs.y))}
    paths = run_paths(dirs.out, stype, seed, model_name)
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    hist.insert(0, "model", model_name)
    hist.insert(0, "seed", seed)
    hist.insert(0, "split_type", stype)
    hist.to_csv(paths["curve"], index=False)
    if (stype, seed, model_name) == CHECKPOINT_RUN:
        save_checkpoint(model, scaler, cfg, dirs.model, f"dmpnn_{stype}_seed{seed}",
                        extra={"split": f"data/splits/aqsoldb/{stype}/seed{seed}.json", "training": info,
                               "test_metrics": {k: m[k] for k in ("MAE", "RMSE", "R2")},
                               "usage": "solpred.models.gnn.load_checkpoint(dir, tag); featurise with "
                                        "solpred.features.graph (GraphStore / GraphTensors, mean readout)"})
    save_json({"row": row, "info": {**info, "inference_seconds": infer_s},
               "mc_dropout": {"passes": cfg.mc_passes, **mc, "mean_std_test": float(np.mean(st))},
               "config": cfg.to_dict()}, paths["run"])
    preds.to_csv(paths["pred"], index=False)  # written last: marks the run as complete
    log(f"  done {stype} seed{seed} {model_name}: best_epoch={info['best_epoch']}/{info['epochs_run']} "
        f"val_MAE={row['val_MAE']:.3f} test MAE={row['MAE']:.3f} RMSE={row['RMSE']:.3f} R2={row['R2']:.3f} "
        f"rho(std,|err|)={mc['spearman_std_abs_err_test']:.3f} ({info['train_seconds']:.0f}s + {infer_s:.1f}s)")
    return row


# --------------------------------------------------------------------------- job control
def job_list(args: argparse.Namespace) -> list[tuple[str, str, int]]:
    """(model, split_type, seed) in priority order: all dmpnn jobs first, then dmpnn_desc (desc seeds only)."""
    jobs = []
    for model in [m for m in MODELS if m in args.models]:
        seeds = [s for s in args.seeds if model != "dmpnn_desc" or s in args.desc_seeds]
        jobs += [(model, st, s) for s in seeds for st in args.split_types]
    return jobs


def write_metrics(out: Path) -> pd.DataFrame:
    """Rebuild metrics_runs.csv from the finished runs' JSON files (safe with concurrent workers)."""
    recs = finished_runs(out)
    if not recs:
        return pd.DataFrame(columns=METRIC_COLUMNS)
    order = {s: i for i, s in enumerate(SPLIT_TYPES)}
    df = pd.DataFrame([r["row"] for r in recs])[METRIC_COLUMNS]
    df = df.assign(_o=df["split_type"].map(order)).sort_values(["model", "_o", "seed"]).drop(columns="_o")
    tmp = out / f"metrics_runs.{os.getpid()}.tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, out / "metrics_runs.csv")
    return df.reset_index(drop=True)


def finished_runs(out: Path) -> list[dict]:
    recs = [load_json(f) for f in sorted((out / "runs").glob("*/*/*.json"))]
    return [r for r in recs if run_done(out, r["row"]["split_type"], r["row"]["seed"], r["row"]["model"])]


def forward_argv(args: argparse.Namespace, worker: int, threads: int) -> list[str]:
    """Command line of one worker process (same filters, its own index and thread count)."""
    argv = [sys.executable, str(Path(__file__).resolve()), "--threads", str(threads), "--workers", str(args.workers),
            "--worker-id", str(worker), "--split-types", *args.split_types, "--seeds", *map(str, args.seeds),
            "--desc-seeds", *map(str, args.desc_seeds), "--models", *args.models,
            "--out-dir", str(args.out_dir), "--fig-dir", str(args.fig_dir), "--model-dir", str(args.model_dir)]
    for k in ("hidden", "depth", "dropout", "batch_size", "max_epochs", "patience", "mc_passes"):
        v = getattr(args, k)
        if v is not None:
            argv += [f"--{k.replace('_', '-')}", str(v)]
    return argv + (["--force"] if args.force else [])


def spawn_workers(args: argparse.Namespace, dirs: Dirs) -> None:
    """Start ``args.workers`` processes with ``threads // workers`` threads each and wait for them."""
    threads = max(1, args.threads // args.workers)
    graph_data()  # build the graph cache once before the workers start
    for f in (dirs.out / "locks").glob("*.lock"):  # stale claims of an interrupted earlier invocation
        f.unlink()
    (dirs.out / "logs").mkdir(parents=True, exist_ok=True)
    procs = []
    for k in range(args.workers):
        logf = open(dirs.out / "logs" / f"train_worker{k}.log", "a", encoding="utf-8")
        procs.append((subprocess.Popen(forward_argv(args, k, threads), stdout=logf, stderr=subprocess.STDOUT,
                                       cwd=str(ROOT)), logf))
        log(f"worker {k}: pid {procs[-1][0].pid}, {threads} threads, log {dirs.out / 'logs' / f'train_worker{k}.log'}")
    codes = []
    for p, f in procs:
        codes.append(p.wait())
        f.close()
    write_metrics(dirs.out)
    shutil.rmtree(dirs.out / "locks", ignore_errors=True)
    log(f"workers finished with exit codes {codes}; {len(finished_runs(dirs.out))} finished runs")
    if any(codes):
        sys.exit(1)


def lock_path(out: Path, stype: str, seed: int, model_name: str) -> Path:
    return out / "locks" / f"{stype}_seed{seed}_{model_name}.lock"


def claim(out: Path, stype: str, seed: int, model_name: str) -> bool:
    """Atomically create the job's lock file; False if another worker already holds it."""
    f = lock_path(out, stype, seed, model_name)
    f.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.close(os.open(f, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        return True
    except FileExistsError:
        return False


def cmd_train(args: argparse.Namespace, dirs: Dirs) -> None:
    jobs = job_list(args)
    if args.workers > 1 and args.worker_id is None:
        return spawn_workers(args, dirs)
    torch.set_num_interop_threads(1)
    set_determinism(0, args.threads)
    cfg = build_config(args)
    worker = args.worker_id is not None
    log(f"{len(jobs)} jobs in list, worker={args.worker_id}, threads={torch.get_num_threads()}, config={cfg.to_dict()}")
    for model_name, stype, seed in jobs:
        if run_done(dirs.out, stype, seed, model_name) and not args.force:
            if not worker:
                log(f"skip {stype} seed{seed} {model_name} (finished)")
            continue
        if worker and not claim(dirs.out, stype, seed, model_name):
            continue
        run_one(dirs, stype, seed, model_name, cfg)
        write_metrics(dirs.out)


# --------------------------------------------------------------------------- profile
def cmd_profile(args: argparse.Namespace, dirs: Dirs) -> None:
    """Time epochs / inference for candidate sizes on random seed0 and extrapolate the full plan."""
    torch.set_num_threads(args.threads)
    t0 = time.time()
    store = dataset_store(DATASET, CACHE_DIR)
    data = GraphTensors(store)
    load_s = time.time() - t0
    log(f"graph store: {len(store)} molecules, {len(store.x)} atoms, {len(store.e)} directed edges ({load_s:.1f}s)")
    results = []
    for model_name in args.models:
        parts = split_parts(data, load_split("random", 0), model_name)
        for hidden in (args.hidden or [DEFAULT_CFG.hidden]):
            for depth in (args.depth or [DEFAULT_CFG.depth]):
                cfg = build_config(args)
                cfg.hidden, cfg.ffn_hidden, cfg.depth = hidden, hidden, depth
                set_determinism(0, args.threads)
                model, sc, hist, info = train_dmpnn(cfg, data, parts["train"].gs, parts["val"].gs, seed=0,
                                                    max_epochs=args.profile_epochs, log=log)
                t1 = time.time()
                evaluate(model, data, parts["val"], sc, cfg, 0)
                evaluate(model, data, parts["test"], sc, cfg, 1)
                r = {"model": model_name, "hidden": hidden, "depth": depth, "batch_size": cfg.batch_size,
                     "threads": args.threads, "epochs_profiled": len(hist),
                     "epoch_seconds": float(hist["epoch_seconds"].mean()),
                     "train_seconds_per_epoch": float(hist["train_seconds"].mean()),
                     "val_eval_seconds_per_epoch": float((hist["epoch_seconds"] - hist["train_seconds"]).mean()),
                     "final_inference_seconds_val_test_mc": time.time() - t1,
                     "n_parameters": info["n_parameters"], "val_MAE_last_epoch": float(hist["val_MAE"].iloc[-1])}
                results.append(r)
                log(f"PROFILE {r}")
    prof_file = dirs.out / "profile.json"
    prof = load_json(prof_file) if prof_file.exists() else {}
    prof.setdefault("runs", [])
    prof["runs"] += [{**r, "timestamp": time.strftime("%Y-%m-%d %H:%M"), "note": args.note} for r in results]
    prof["graph_store_load_seconds"] = load_s
    prof["plan_estimate"] = plan_estimates(prof["runs"], len(job_list(args_for_plan())), args.total_threads)
    save_json(prof, prof_file)
    for e in prof["plan_estimate"]["layouts"]:
        log(f"PLAN hidden={e['hidden']} depth={e['depth']} {e['workers']}x{e['threads_per_worker']} threads: "
            f"{e['wall_hours_expected']:.2f} h expected ({e['wall_hours_worst']:.2f} h worst case)")


def args_for_plan() -> argparse.Namespace:
    """Filters of the full plan (all splits, seeds 0-4, dmpnn_desc on seeds 0-2)."""
    return argparse.Namespace(models=list(MODELS), split_types=list(SPLIT_TYPES), seeds=[0, 1, 2, 3, 4],
                              desc_seeds=[0, 1, 2])


def plan_estimates(runs: list[dict], n_jobs: int, total_threads: int, expected_epochs: int = 45,
                   per_run_overhead_s: float = 10.0) -> dict:
    """Wall-clock estimate of the plan for every profiled (hidden, depth, threads) setting.

    ``workers = total_threads // threads``; each worker runs ``ceil(n_jobs / workers)`` jobs of
    ``epochs x epoch_seconds + final inference + overhead``. Assumes the per-thread speed measured
    during profiling (i.e. a similar CPU load) - an extrapolation, not a measurement.
    """
    latest = {}
    for r in runs:
        if r.get("model") == "dmpnn" and "epoch_seconds" in r:
            latest[(r["hidden"], r["depth"], r["threads"])] = r
    layouts = []
    for (hidden, depth, threads), r in sorted(latest.items()):
        workers = max(1, total_threads // threads)
        per_worker = -(-n_jobs // workers)
        run_s = lambda ep: ep * r["epoch_seconds"] + r["final_inference_seconds_val_test_mc"] + per_run_overhead_s  # noqa: E731
        layouts.append({"hidden": hidden, "depth": depth, "threads_per_worker": threads, "workers": workers,
                        "jobs_per_worker": per_worker, "epoch_seconds_measured": r["epoch_seconds"],
                        "run_minutes_expected": run_s(expected_epochs) / 60,
                        "wall_hours_expected": per_worker * run_s(expected_epochs) / 3600,
                        "wall_hours_worst": per_worker * run_s(DEFAULT_CFG.max_epochs) / 3600,
                        "measured_at": r.get("timestamp"), "load_note": r.get("note")})
    return {"n_jobs": n_jobs, "total_threads": total_threads, "expected_epochs": expected_epochs,
            "worst_case_epochs": DEFAULT_CFG.max_epochs, "per_run_overhead_seconds": per_run_overhead_s,
            "layouts": layouts}


# --------------------------------------------------------------------------- analysis helpers
def mean_sd(x: pd.Series) -> dict:
    x = pd.Series(x, dtype=float)
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else float("nan"), "n": int(len(x))}


def load_preds(root: Path, stype: str, seed: int, model: str, feat: str) -> pd.DataFrame | None:
    p = prediction_path(root, stype, f"seed{seed}", model, feat)
    return pd.read_csv(p) if p.exists() else None


def pooled_test(out: Path, recs: list[dict], model: str, stype: str) -> pd.DataFrame | None:
    ps = [load_preds(out, stype, r["row"]["seed"], model, MODELS[model]) for r in recs
          if r["row"]["model"] == model and r["row"]["split_type"] == stype]
    ps = [p[p.subset == "test"] for p in ps if p is not None]
    return pd.concat(ps, ignore_index=True) if ps else None


def comparison_table(out: Path, gnn: pd.DataFrame) -> pd.DataFrame:
    """Per split and seed: GNN test MAE vs the LightGBM reference (paired bootstrap, same test molecules)."""
    rows = []
    broot = path("results") / "baseline"
    for (stype, model, feat, seed), _ in gnn.groupby(["split_type", "model", "featurizer", "seed"]):
        a, b = load_preds(out, stype, seed, model, feat), load_preds(broot, stype, seed, *BASELINE_REF)
        if a is None or b is None:
            continue
        mrg = a[a.subset == "test"].merge(b[b.subset == "test"], on="compound_id", suffixes=("_gnn", "_ref"))
        if len(mrg) != (a.subset == "test").sum() or not np.allclose(mrg.y_true_gnn, mrg.y_true_ref):
            raise ValueError(f"test sets of {model} and the reference differ ({stype} seed{seed})")
        r = paired_bootstrap_mae_diff(mrg.y_true_gnn, mrg.y_pred_gnn, mrg.y_pred_ref, n_boot=2000, seed=int(seed))
        rows.append({"split_type": stype, "seed": int(seed), "model": model, "featurizer": feat,
                     "reference": "__".join(BASELINE_REF), "n_test": len(mrg),
                     "MAE_gnn": float(np.mean(np.abs(mrg.y_pred_gnn - mrg.y_true_gnn))),
                     "MAE_ref": float(np.mean(np.abs(mrg.y_pred_ref - mrg.y_true_gnn))), **r})
    return pd.DataFrame(rows)


def mc_dropout_stats(out: Path, recs: list[dict]) -> dict:
    """Spearman(MC-dropout std, |error|) on test: per run (mean +- sd) and pooled over seeds."""
    res = {}
    for model in MODELS:
        for stype in SPLIT_TYPES:
            per_run = [r["mc_dropout"]["spearman_std_abs_err_test"] for r in recs
                       if r["row"]["model"] == model and r["row"]["split_type"] == stype]
            te = pooled_test(out, recs, model, stype)
            if te is None:
                continue
            err = np.abs(te.y_pred - te.y_true)
            rho, pval = stats.spearmanr(te.y_std, err)
            q = pd.qcut(te.y_std.rank(method="first"), 5, labels=False)
            mae_q = err.groupby(q).mean()
            res[f"{model}|{stype}"] = {
                "spearman_test_per_run": mean_sd(pd.Series(per_run)),
                "spearman_test_pooled": float(rho), "p_value_pooled": float(pval), "n_pooled": int(len(te)),
                "mean_mc_std_test": float(te.y_std.mean()),
                "test_MAE_lowest_std_quintile": float(mae_q.iloc[0]),
                "test_MAE_highest_std_quintile": float(mae_q.iloc[-1])}
    return res


# --------------------------------------------------------------------------- figures
def _style(ax) -> None:
    ax.grid(alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_training_curves(curves: pd.DataFrame, fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for j, stype in enumerate(SPLIT_TYPES):
        for model, ls in (("dmpnn", "-"), ("dmpnn_desc", "--")):
            c = curves[(curves.split_type == stype) & (curves.model == model)]
            for k, (_, cs) in enumerate(c.groupby("seed")):
                lab = MODEL_LABELS[model] if k == 0 else None
                axes[0, j].plot(cs.epoch, cs.val_MAE, ls, color=PALETTE[stype], alpha=0.8, lw=1.2, label=lab)
                axes[1, j].plot(cs.epoch, cs.train_loss_mse_std, ls, color=PALETTE[stype], alpha=0.8, lw=1.2)
                b = cs.loc[cs.val_MAE.idxmin()]
                axes[0, j].plot(b.epoch, b.val_MAE, "o", color=PALETTE[stype], ms=4.5,
                                mfc="white" if model == "dmpnn_desc" else PALETTE[stype])
        axes[0, j].set_title(f"{stype} split")
        axes[0, j].set_ylabel("validation MAE (log mol/L)")
        axes[1, j].set_ylabel("training loss (MSE, standardised logS)")
        axes[1, j].set_xlabel("epoch")
        axes[1, j].set_yscale("log")
        if axes[0, j].get_legend_handles_labels()[0]:
            axes[0, j].legend(fontsize=8, frameon=False)
        for a in axes[:, j]:
            _style(a)
    fig.suptitle("D-MPNN training curves, one line per seed (dots: best validation epoch, weights restored)",
                 fontsize=11)
    fig.tight_layout()
    f = fig_dir / "training_curves.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    return f


def plot_comparison(gnn: pd.DataFrame, base: pd.DataFrame | None, comp: pd.DataFrame, fig_dir: Path) -> Path:
    """(a) test MAE mean +- sd per split for LightGBM and the D-MPNNs; (b) paired per-seed MAE differences."""
    import matplotlib.pyplot as plt

    groups = []
    if base is not None:
        groups.append(("LightGBM\n(rdkit2d+Morgan)", base[(base.model == BASELINE_REF[0]) &
                                                       (base.featurizer == BASELINE_REF[1])]))
    groups += [(MODEL_LABELS[m].replace(" (", "\n(").replace(" + ", " +\n"), gnn[gnn.model == m])
               for m in MODELS if (gnn.model == m).any()]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.25, 1]})
    w = 0.26
    for gi, (lab, df) in enumerate(groups):
        for j, stype in enumerate(SPLIT_TYPES):
            v = df[df.split_type == stype]["MAE"]
            if v.empty:
                continue
            x, sd = gi + (j - 1) * w, (v.std(ddof=1) if len(v) > 1 else 0.0)
            ax.bar(x, v.mean(), w * 0.9, yerr=sd, color=PALETTE[stype], capsize=3, ecolor="0.25",
                   error_kw={"lw": 1}, label=stype if gi == 0 else None)
            ax.scatter(np.full(len(v), x) + np.linspace(-0.05, 0.05, len(v)), v, s=9, color="0.15", zorder=3, lw=0)
            ax.text(x, v.mean() + sd + 0.012, f"{v.mean():.2f}\n(n={len(v)})", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=9)
    ax.set_ylabel("test MAE (log mol/L)")
    ax.set_title("(a) test MAE, mean ± sd over seeds (dots: seeds)", fontsize=10)
    ax.legend(title="split", frameon=False, fontsize=8, title_fontsize=8, loc="upper left")
    _style(ax)
    if not comp.empty:
        markers = {"dmpnn": "o", "dmpnn_desc": "s"}
        for j, stype in enumerate(SPLIT_TYPES):
            for mi, model in enumerate(MODELS):
                c = comp[(comp.split_type == stype) & (comp.model == model)].sort_values("seed")
                if c.empty:
                    continue
                x = j + (mi - 0.5) * 0.36 + (c.seed.to_numpy() - 2) * 0.05
                bx.errorbar(x, c.mae_diff, yerr=[c.mae_diff - c.ci_low, c.ci_high - c.mae_diff], fmt=markers[model],
                            color=PALETTE[stype], ms=5, lw=1, capsize=2,
                            mfc=PALETTE[stype] if model == "dmpnn" else "white",
                            label=MODEL_LABELS[model] if j == 0 else None)
        bx.axhline(0, color="0.3", lw=0.9)
        bx.set_xticks(range(len(SPLIT_TYPES)))
        bx.set_xticklabels(SPLIT_TYPES)
        bx.set_xlim(-0.5, len(SPLIT_TYPES) - 0.5)
        bx.set_ylabel("MAE(D-MPNN) − MAE(LightGBM) (log mol/L)")
        bx.set_title("(b) paired difference per seed, 95% bootstrap CI (<0: D-MPNN better)", fontsize=10)
        if bx.get_legend_handles_labels()[0]:
            leg = bx.legend(frameon=False, fontsize=8)
            for h in leg.legend_handles:
                h.set_color("0.3")
        _style(bx)
    fig.tight_layout()
    f = fig_dir / "mae_vs_lgbm.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    return f


def plot_mc_dropout(out: Path, recs: list[dict], fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    models = [m for m in MODELS if any(r["row"]["model"] == m for r in recs)]
    fig, axes = plt.subplots(1, len(models), figsize=(5.6 * len(models), 4.3), squeeze=False)
    for ax, model in zip(axes[0], models):
        for stype in SPLIT_TYPES:
            te = pooled_test(out, recs, model, stype)
            if te is None:
                continue
            err = np.abs(te.y_pred - te.y_true)
            q = pd.qcut(te.y_std.rank(method="first"), 10, labels=False)
            rho = stats.spearmanr(te.y_std, err)[0]
            ax.plot(te.y_std.groupby(q).mean(), err.groupby(q).mean(), "o-", color=PALETTE[stype], lw=1.5, ms=5,
                    label=f"{stype} (Spearman {rho:.2f}, n={len(te)})")
        ax.set_xlabel("MC-dropout std, decile mean (log mol/L)")
        ax.set_ylabel("test |error|, decile mean (log mol/L)")
        ax.set_title(f"{MODEL_LABELS[model]}: error vs MC-dropout spread (test, seeds pooled)", fontsize=10)
        ax.legend(frameon=False, fontsize=8)
        _style(ax)
    fig.tight_layout()
    f = fig_dir / "mc_dropout_std_vs_error.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    return f


def plot_parity(out: Path, recs: list[dict], fig_dir: Path) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharex=True, sharey=True)
    lo, hi = -13.5, 2.5
    for ax, stype in zip(axes, SPLIT_TYPES):
        te = pooled_test(out, recs, "dmpnn", stype)
        if te is not None:
            m = regression_metrics(te.y_true, te.y_pred)
            ax.scatter(te.y_true, te.y_pred, s=5, alpha=0.35, color=PALETTE[stype], lw=0)
            ax.set_title(f"{stype}: MAE {m['MAE']:.2f}, R² {m['R2']:.2f} (n={m['n']})", fontsize=10)
        ax.plot([lo, hi], [lo, hi], color="0.3", ls="--", lw=0.9)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_xlabel("measured logS (log mol/L)")
        _style(ax)
    axes[0].set_ylabel("predicted logS (log mol/L)")
    fig.suptitle("D-MPNN (graph only) test predictions, seeds pooled", fontsize=11)
    fig.tight_layout()
    f = fig_dir / "parity_dmpnn.png"
    fig.savefig(f, dpi=200)
    plt.close(fig)
    return f


# --------------------------------------------------------------------------- analyze
def rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def reference_metrics(base: pd.DataFrame | None, gnn: pd.DataFrame) -> dict:
    """LightGBM test metrics per split: all 5 seeds and restricted to the seeds each GNN model has."""
    ref: dict = {}
    if base is None:
        return ref
    b = base[(base.model == BASELINE_REF[0]) & (base.featurizer == BASELINE_REF[1])]
    for stype in SPLIT_TYPES:
        g = b[b.split_type == stype]
        if g.empty:
            continue
        ref[stype] = {"all_seeds": {k: mean_sd(g[k]) for k in ("MAE", "RMSE", "R2")}}
        for model in MODELS:
            seeds = gnn[(gnn.model == model) & (gnn.split_type == stype)].seed
            if len(seeds):
                ref[stype][f"seeds_of_{model}"] = {"seeds": sorted(int(s) for s in seeds),
                                                   "MAE": mean_sd(g[g.seed.isin(seeds)]["MAE"])}
    return ref


def cmd_analyze(args: argparse.Namespace, dirs: Dirs) -> None:
    out, fig_dir = dirs.out, dirs.fig
    gnn = write_metrics(out)
    recs = finished_runs(out)
    if gnn.empty:
        raise SystemExit(f"no finished runs under {out}")
    log(f"{len(gnn)} finished runs")
    curves = pd.concat([pd.read_csv(f) for f in sorted((out / "curves").glob("*/*/*.csv"))], ignore_index=True)
    curves.to_csv(out / "training_curves.csv", index=False)
    s = summarize(gnn)
    val = gnn.groupby(["split_type", "featurizer", "model"])["val_MAE"].agg(["mean", "std"]).add_prefix("val_MAE_")
    s = s.merge(val.reset_index(), on=["split_type", "featurizer", "model"])
    s.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(s.round(3).to_markdown(index=False), encoding="utf-8")
    bfile = path("results") / "baseline" / "metrics_runs.csv"
    base = pd.read_csv(bfile) if bfile.exists() else None
    comp = comparison_table(out, gnn)
    comp.to_csv(out / "comparison_vs_lgbm.csv", index=False)
    if not comp.empty:
        (out / "comparison_vs_lgbm.md").write_text(comp.round(3).to_markdown(index=False), encoding="utf-8")
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = [plot_training_curves(curves, fig_dir), plot_comparison(gnn, base, comp, fig_dir),
            plot_mc_dropout(out, recs, fig_dir), plot_parity(out, recs, fig_dir)]

    metrics: dict = {}
    for (model, stype), g in gnn.groupby(["model", "split_type"]):
        metrics.setdefault(model, {})[stype] = {"seeds": sorted(int(x) for x in g.seed),
                                                **{k: mean_sd(g[k]) for k in
                                                   ("MAE", "RMSE", "R2", "Pearson", "Spearman", "val_MAE")}}
    vs: dict = {}
    for (model, stype), g in (comp.groupby(["model", "split_type"]) if not comp.empty else []):
        vs.setdefault(model, {})[stype] = {
            "MAE_gnn": mean_sd(g["MAE_gnn"]), "MAE_lgbm_same_seeds": mean_sd(g["MAE_ref"]),
            "MAE_diff_gnn_minus_lgbm": mean_sd(g["mae_diff"]),
            "n_seeds_gnn_better": int((g["mae_diff"] < 0).sum()),
            "n_seeds_gnn_significantly_better": int((g["ci_high"] < 0).sum()),
            "n_seeds_gnn_significantly_worse": int((g["ci_low"] > 0).sum()),
            "seeds_compared": [int(x) for x in g["seed"]]}
    info = pd.DataFrame([{**r["info"], "split_type": r["row"]["split_type"], "model": r["row"]["model"]}
                         for r in recs])
    curves_per_run = curves.groupby(["split_type", "seed", "model"])["epoch_seconds"].mean()
    prof = load_json(out / "profile.json") if (out / "profile.json").exists() else None
    cfg = recs[0]["config"]
    total_s = float(info["train_seconds"].sum() + info["inference_seconds"].sum())
    summary = {
        "experiment": EXPERIMENT,
        "n_runs": int(len(gnn)),
        "runs_per_model_split": {f"{m}|{s}": int(n) for (m, s), n in gnn.groupby(["model", "split_type"]).size().items()},
        "test_metrics_mean_sd_over_seeds": metrics,
        "lgbm_reference_test_metrics": reference_metrics(base, gnn),
        "paired_vs_lgbm_rdkit2d+morgan_counts": vs,
        "mc_dropout_spearman_std_vs_abs_error": mc_dropout_stats(out, recs),
        "hyperparameters": {**cfg, "atom_fdim": ATOM_FDIM, "bond_fdim": BOND_FDIM,
                            "descriptor_block": "rdkit2d (217), median-imputed and standardised on train, "
                                                "clipped to +-10 (dmpnn_desc only)",
                            "rationale": RATIONALE},
        "training": {
            "best_epoch": {f"{m}|{s}": mean_sd(g["best_epoch"]) for (m, s), g in info.groupby(["model", "split_type"])},
            "epochs_run": {f"{m}|{s}": mean_sd(g["epochs_run"]) for (m, s), g in info.groupby(["model", "split_type"])},
            "n_early_stopped": int(info["early_stopped"].sum()),
            "n_parameters": {m: int(g["n_parameters"].iloc[0]) for m, g in info.groupby("model")},
        },
        "compute": {
            "total_train_seconds": float(info["train_seconds"].sum()),
            "total_inference_seconds_incl_mc_dropout": float(info["inference_seconds"].sum()),
            "total_cpu_process_hours": total_s / 3600,
            "mean_train_seconds_per_run": {m: float(g["train_seconds"].mean()) for m, g in info.groupby("model")},
            "mean_epoch_seconds": float(curves_per_run.mean()),
            "threads_per_process": sorted({int(t) for t in info["threads"]}),
            "profile": prof,
        },
        "notes": {
            "y_pred": "deterministic forward pass with dropout disabled (best-validation-epoch weights)",
            "y_std": f"std over {cfg['mc_passes']} Monte-Carlo dropout passes (p={cfg['dropout']} in the FFN head; "
                     "encoder deterministic)",
            "early_stopping": "validation MAE (log units), best weights restored; test never used for any choice",
            "dmpnn_desc_seeds": "dmpnn_desc was run on fewer seeds than dmpnn (compute budget); compare it with "
                                "LightGBM on the same seeds (paired_vs_lgbm / lgbm_reference seeds_of_dmpnn_desc)",
        },
        "files": {"metrics_runs": rel(out / "metrics_runs.csv"), "summary_table": rel(out / "summary.csv"),
                  "comparison": rel(out / "comparison_vs_lgbm.csv"),
                  "training_curves": rel(out / "training_curves.csv"), "figures": [rel(f) for f in figs],
                  "checkpoint": sorted(rel(p) for p in dirs.model.glob("dmpnn_*"))},
    }
    save_json(summary, out / "summary.json")
    log(f"wrote {rel(out / 'summary.json')} and {len(figs)} figures")
    print(s[["split_type", "model", "n_runs", "MAE_mean", "MAE_std", "R2_mean", "val_MAE_mean"]].round(3).to_string())


# --------------------------------------------------------------------------- CLI
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--analyze", action="store_true", help="tables, figures and summary.json from finished runs")
    mode.add_argument("--profile", action="store_true", help="time epochs / inference (random seed0)")
    ap.add_argument("--threads", type=int, default=3, help="total torch intra-op threads (split across workers)")
    ap.add_argument("--workers", type=int, default=1, help="parallel worker processes (training mode)")
    ap.add_argument("--worker-id", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--split-types", nargs="+", default=list(SPLIT_TYPES), choices=list(SPLIT_TYPES))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--desc-seeds", type=int, nargs="+", default=[0, 1, 2], help="seeds on which dmpnn_desc runs")
    ap.add_argument("--models", nargs="+", default=list(MODELS), choices=list(MODELS))
    ap.add_argument("--force", action="store_true", help="re-run even when outputs exist")
    ap.add_argument("--hidden", type=int, nargs="+", default=None)
    ap.add_argument("--depth", type=int, nargs="+", default=None)
    ap.add_argument("--dropout", type=float, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--max-epochs", type=int, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--mc-passes", type=int, default=None)
    ap.add_argument("--profile-epochs", type=int, default=2)
    ap.add_argument("--total-threads", type=int, default=6, help="thread budget assumed by the --profile plan estimate")
    ap.add_argument("--note", default="", help="free text stored with --profile results (e.g. CPU load)")
    ap.add_argument("--out-dir", type=Path, default=path("results") / EXPERIMENT)
    ap.add_argument("--fig-dir", type=Path, default=path("figures") / EXPERIMENT)
    ap.add_argument("--model-dir", type=Path, default=path("models") / EXPERIMENT)
    args = ap.parse_args()
    if not args.profile and ((args.hidden and len(args.hidden) > 1) or (args.depth and len(args.depth) > 1)):
        ap.error("several --hidden / --depth values are only allowed with --profile")
    if args.hidden and not args.profile:
        args.hidden = args.hidden[0]
    if args.depth and not args.profile:
        args.depth = args.depth[0]
    return args


def main() -> None:
    args = parse_args()
    dirs = Dirs(Path(args.out_dir), Path(args.fig_dir), Path(args.model_dir))
    dirs.out.mkdir(parents=True, exist_ok=True)
    if args.analyze:
        cmd_analyze(args, dirs)
    elif args.profile:
        cmd_profile(args, dirs)
    else:
        cmd_train(args, dirs)


if __name__ == "__main__":
    main()
