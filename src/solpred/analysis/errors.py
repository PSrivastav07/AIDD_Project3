"""Similarity-aware error analysis of stored test-set predictions.

The analysis works on the prediction files written by :func:`solpred.experiment.save_run`
(``<pred_root>/predictions/<split_type>/<split_name>/<model_tag>.csv``) and is therefore model
agnostic: any model whose predictions follow that layout can be analysed.

Method
------
* **Nearest-neighbour similarity.** For every test compound we compute the maximum Tanimoto
  similarity (ECFP4 = Morgan radius 2, 2048 bits, computed on the full standardised SMILES exactly as
  in :mod:`solpred.splits`) to the *training* compounds of the same split, together with the id and
  label of that nearest neighbour (NN). This is the quantity that separates random from scaffold /
  cluster splits. Similarity is measured to the TRAINING set only; validation compounds are excluded
  (they are used only for early stopping, the model is not fit on them). Binary ECFP4 saturates for
  homologues (chain length / ring size are nearly invisible at radius 2), so the top similarity bin
  (>= 0.9) of the random split contains many fingerprint-identical but chemically different neighbours.
* **Binned errors with compound-level bootstrap CIs.** Errors are pooled over split seeds. Because the
  same compound can be a test compound in several seeds (random split), confidence intervals are
  obtained with a *cluster bootstrap* over unique compounds (each resampled compound brings all its
  pooled rows), which is the conservative choice. Signed error is ``y_pred - y_true`` (positive =
  model predicts the compound to be *more soluble* than measured).
* **Disentangling label shift from structural novelty.** Scaffold test sets are shifted toward
  insoluble compounds (and contain no acyclic molecules). We (i) compare split types inside matched
  similarity bins and matched true-logS bins and (ii) use *direct standardisation*: the random-split
  error in each stratum (similarity bin, logS bin, or their cross product) is re-weighted to the
  stratum composition of the scaffold / cluster test sets. If the re-weighted random MAE reproduces
  the scaffold MAE, the penalty is accounted for by the stratifying variable - under the assumption
  that, WITHIN a stratum, random- and scaffold-split test compounds are exchangeable (same expected
  error); this is violated if scaffold compounds are harder at equal NN similarity, which they are
  (e.g. 0.6-0.7 bin: scaffold MAE 0.63 vs random 0.51). The fraction of the penalty accounted for gets a
  JOINT cluster-bootstrap CI (source and target compounds both resampled, weights recomputed), which
  is wide because the penalty itself is small. (iii) An additive OLS model
  ``|error| ~ split + sim_bin + logS_bin`` with compound-clustered standard errors gives the
  residual split penalty after adjustment (an alternative estimator; it typically accounts for ~2/3 of
  the penalty). The two estimators bracket the answer: similarity and label shift together account
  for most (roughly 2/3 to all) of the penalty, and a residual scaffold-specific penalty remains.
* **Error by chemistry.** Physicochemical bins (RDKit descriptors from the shared feature cache) and
  ~20 SMARTS-defined functional-group classes. A class is flagged as *systematically biased* when it
  has at least ``min_compounds`` unique compounds, the 95 % CI of its mean signed error excludes 0 and
  ``|mean signed error| >= 0.2`` log units.
* **Worst predictions** with rule-based hypotheses (label inconsistency with a near-identical
  neighbour, inter-measurement disagreement, out-of-domain, salts, surfactants, extreme labels, ...).
* **Scaffold-level errors** in scaffold splits (Bemis-Murcko scaffold of the largest fragment).
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs

from solpred.features import load_dataset, load_or_compute
from solpred.splits import load_split, morgan_bits

SIGNED_ERROR_NOTE = "signed error = y_pred - y_true (positive: predicted more soluble than measured)"
SIM_EDGES = (0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)
SIM_EDGES_COARSE = (0.0, 0.4, 0.5, 0.6, 0.7, 1.0001)
LOGS_EDGES = (-np.inf, -6.0, -4.0, -2.0, 0.0, np.inf)
SPLIT_ORDER = ("random", "scaffold", "cluster", "external")


# --------------------------------------------------------------------------- loading
def prediction_file(pred_root: Path, split_type: str, split_name: str, model_tag: str) -> Path:
    """Path of one stored prediction file (see :mod:`solpred.experiment`)."""
    return Path(pred_root) / "predictions" / split_type / split_name / f"{model_tag}.csv"


def split_names(split_type: str, seeds: list[int]) -> list[str]:
    """Split names stored under a split type (``seed<k>`` for the seeded AqSolDB splits)."""
    if split_type == "external":
        return ["aqsoldb_to_esol"]
    return [f"seed{s}" for s in seeds]


def load_split_by_name(split_type: str, split_name: str) -> dict:
    if split_type == "external":
        return load_split("external", name=split_name)
    if split_type == "loso":
        return load_split("loso", source=split_name.replace("source_", ""))
    return load_split(split_type, int(split_name.replace("seed", "")))


def load_predictions(pred_root: Path, model_tag: str, split_types: list[str], seeds: list[int],
                     subset: str = "test") -> pd.DataFrame:
    """Concatenate the ``subset`` rows of all available prediction files of one model.

    Returns columns split_type, split_name, seed, compound_id, y_true, y_pred, error, abs_error.
    Missing files are skipped with a warning (so the analysis can run while experiments finish).
    """
    frames = []
    for st in split_types:
        for name in split_names(st, seeds):
            f = prediction_file(pred_root, st, name, model_tag)
            if not f.exists():
                warnings.warn(f"missing predictions: {f}")
                continue
            d = pd.read_csv(f, dtype={"compound_id": str})
            d = d[d["subset"] == subset].copy()
            d.insert(0, "split_type", st)
            d.insert(1, "split_name", name)
            d.insert(2, "seed", int(name[4:]) if name.startswith("seed") else -1)
            frames.append(d)
    if not frames:
        raise FileNotFoundError(f"no prediction files for {model_tag} under {pred_root}")
    df = pd.concat(frames, ignore_index=True)
    df["error"] = df["y_pred"] - df["y_true"]
    df["abs_error"] = df["error"].abs()
    return df.drop(columns=[c for c in ("subset",) if c in df.columns])


# --------------------------------------------------------------------------- similarity
def _fps_by_id(dataset: str) -> dict:
    df = load_dataset(dataset)
    return dict(zip(df["compound_id"], morgan_bits(df["smiles"].tolist())))


def nearest_train_neighbours(split_type: str, split_name: str, fp_cache: dict | None = None) -> pd.DataFrame:
    """Max ECFP4 Tanimoto of every test compound to the training set, with the NN id and label."""
    fp_cache = {} if fp_cache is None else fp_cache
    split = load_split_by_name(split_type, split_name)
    tr_ds = split.get("train_dataset", "aqsoldb")
    te_ds = split.get("test_dataset", tr_ds)
    for ds in {tr_ds, te_ds}:
        if ds not in fp_cache:
            fp_cache[ds] = _fps_by_id(ds)
    tr_ids = list(map(str, split["train"]))
    tr_fps = [fp_cache[tr_ds][i] for i in tr_ids]
    tr_lab = load_dataset(tr_ds).set_index("compound_id")["logS"]
    rows = []
    for cid in map(str, split["test"]):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp_cache[te_ds][cid], tr_fps))
        j = int(np.argmax(sims))
        rows.append((cid, float(sims[j]), tr_ids[j], float(tr_lab.loc[tr_ids[j]]),
                     float(np.sort(sims)[-5:].mean())))
    out = pd.DataFrame(rows, columns=["compound_id", "nn_sim", "nn_id", "nn_logS", "sim_top5_mean"])
    out.insert(0, "split_type", split_type)
    out.insert(1, "split_name", split_name)
    return out


def nn_similarity_table(split_types: list[str], seeds: list[int], cache_file: Path) -> pd.DataFrame:
    """Nearest-neighbour table for all requested splits, cached in ``cache_file`` (model independent)."""
    cache_file = Path(cache_file)
    have = pd.read_csv(cache_file, dtype={"compound_id": str, "nn_id": str}) if cache_file.exists() else None
    done = set() if have is None else set(zip(have["split_type"], have["split_name"]))
    new, fp_cache = [], {}
    for st in split_types:
        for name in split_names(st, seeds):
            if (st, name) not in done:
                new.append(nearest_train_neighbours(st, name, fp_cache))
    if new:
        have = pd.concat(([have] if have is not None else []) + new, ignore_index=True)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        have.to_csv(cache_file, index=False)
    return have


# --------------------------------------------------------------------------- statistics
def grouped_bootstrap_mean(values, groups, n_boot: int = 1000, seed: int = 0,
                           alpha: float = 0.05) -> tuple[float, float, float]:
    """Mean of ``values`` with a percentile CI from a cluster bootstrap over ``groups``."""
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return np.nan, np.nan, np.nan
    codes, _ = pd.factorize(np.asarray(groups))
    sums = np.bincount(codes, weights=v)
    cnts = np.bincount(codes).astype(float)
    g = len(sums)
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    chunk = max(1, int(2e6 // max(g, 1)))
    for s in range(0, n_boot, chunk):
        idx = rng.integers(0, g, (min(chunk, n_boot - s), g))
        boots[s:s + len(idx)] = sums[idx].sum(1) / cnts[idx].sum(1)
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(v.mean()), float(lo), float(hi)


def error_summary(df: pd.DataFrame, by: list[str], n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Per-group n, MAE and mean signed error (MSE) with cluster-bootstrap CIs, RMSE and label means."""
    rows = []
    for key, g in df.groupby(by, observed=True, sort=True):
        key = key if isinstance(key, tuple) else (key,)
        mae, mlo, mhi = grouped_bootstrap_mean(g["abs_error"], g["compound_id"], n_boot, seed)
        mse, slo, shi = grouped_bootstrap_mean(g["error"], g["compound_id"], n_boot, seed + 1)
        rows.append({**dict(zip(by, key)), "n": len(g), "n_compounds": g["compound_id"].nunique(),
                     "MAE": mae, "MAE_lo": mlo, "MAE_hi": mhi,
                     "mean_signed_error": mse, "MSE_lo": slo, "MSE_hi": shi,
                     "RMSE": float(np.sqrt(np.mean(g["error"] ** 2))),
                     "mean_true": float(g["y_true"].mean()), "mean_pred": float(g["y_pred"].mean())})
    return pd.DataFrame(rows)


def add_bins(df: pd.DataFrame, col: str, edges, name: str | None = None, labels=None,
             fmt: str = "{:g}") -> pd.DataFrame:
    """Add an ordered categorical bin column ``name`` (left-closed intervals)."""
    name = name or f"{col}_bin"
    if labels is None:
        labels = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            if np.isinf(lo):
                labels.append(f"< {fmt.format(hi)}")
            elif np.isinf(hi):
                labels.append(f">= {fmt.format(lo)}")
            else:
                labels.append(f"[{fmt.format(lo)}, {fmt.format(hi)})")
    df[name] = pd.cut(df[col], bins=list(edges), labels=labels, right=False, include_lowest=True)
    return df


def sim_labels(edges=SIM_EDGES) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        out.append(f"{lo:.1f}-{min(hi, 1.0):.1f}")
    return out


def standardized_mae(source: pd.DataFrame, target: pd.DataFrame, strata: list[str], min_n: int = 10,
                     fallback: str | None = None, n_boot: int = 500, seed: int = 0) -> dict:
    """MAE of ``source`` errors re-weighted to the stratum composition of ``target`` (direct standardisation).

    Strata of the target with fewer than ``min_n`` source rows fall back to the source MAE of the
    coarser stratum ``fallback`` (a column name) or, if that is missing too, to the overall source MAE.
    The CI comes from a cluster bootstrap of the source compounds with target weights held fixed.
    """
    w = target.groupby(strata, observed=True).size()
    w = w / w.sum()

    def estimate(src: pd.DataFrame) -> tuple[float, float]:
        cell = src.groupby(strata, observed=True)["abs_error"].agg(["mean", "size"])
        fb = src.groupby(fallback, observed=True)["abs_error"].mean() if fallback else None
        tot, covered = 0.0, 0.0
        for key, wk in w.items():
            kk = key if isinstance(key, tuple) else (key,)
            if key in cell.index and cell.loc[key, "size"] >= min_n:
                tot += wk * cell.loc[key, "mean"]
                covered += wk
            elif fb is not None and kk[strata.index(fallback)] in fb.index:
                tot += wk * fb.loc[kk[strata.index(fallback)]]
            else:
                tot += wk * src["abs_error"].mean()
        return tot, covered

    point, covered = estimate(source)
    rng = np.random.default_rng(seed)
    pos = source.groupby("compound_id").indices
    comp = np.array(list(pos))
    boots = []
    for _ in range(n_boot):
        pick = comp[rng.integers(0, len(comp), len(comp))]
        idx = np.concatenate([pos[c] for c in pick])
        boots.append(estimate(source.iloc[idx])[0])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"MAE_standardized": float(point), "lo": float(lo), "hi": float(hi),
            "weight_covered_by_cells": float(covered)}


def _stratum_codes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return df[cols].astype(str).agg("|".join, axis=1)


def standardization_joint(source: pd.DataFrame, target: pd.DataFrame, strata: list[str], min_n: int = 10,
                          fallback: str | None = None, n_boot: int = 1000, seed: int = 0) -> dict:
    """Direct standardisation with a JOINT cluster bootstrap of source and target compounds.

    Same estimator as :func:`standardized_mae` (identical point estimate), but every bootstrap replicate
    resamples the source (random-split) compounds AND the target (scaffold/cluster) compounds, recomputes
    the target stratum weights, the standardised source MAE, the raw source MAE and the observed target
    MAE, and hence ``frac = (MAE_std - MAE_raw) / (MAE_obs - MAE_raw)``. Also reports the sparsest
    stratum (source rows) carrying target weight and the target weight that falls back to a coarser
    stratum (fewer than ``min_n`` source rows).
    """
    ks, kt = _stratum_codes(source, strata), _stratum_codes(target, strata)
    cats = pd.Index(sorted(set(ks) | set(kt)))
    cs, ct = cats.get_indexer(ks), cats.get_indexer(kt)
    K = len(cats)
    if fallback:
        fcol = strata.index(fallback)
        fvals = pd.Index(sorted({c.split("|")[fcol] for c in cats}))
        cat2f = fvals.get_indexer([c.split("|")[fcol] for c in cats])
        F = len(fvals)
    ae_s, ae_t = source["abs_error"].to_numpy(float), target["abs_error"].to_numpy(float)
    comp_s, ucs = pd.factorize(source["compound_id"])
    comp_t, uct = pd.factorize(target["compound_id"])

    def est(rw_s: np.ndarray, rw_t: np.ndarray) -> tuple[float, float, float, float]:
        n_k = np.bincount(cs, weights=rw_s, minlength=K)
        s_k = np.bincount(cs, weights=rw_s * ae_s, minlength=K)
        w_k = np.bincount(ct, weights=rw_t, minlength=K)
        w_k = w_k / w_k.sum()
        raw = float((rw_s * ae_s).sum() / rw_s.sum())
        obs = float((rw_t * ae_t).sum() / rw_t.sum())
        ok = n_k >= min_n
        fill = np.full(K, raw)
        if fallback:
            fn = np.bincount(cat2f[cs], weights=rw_s, minlength=F)
            fs = np.bincount(cat2f[cs], weights=rw_s * ae_s, minlength=F)
            fm = np.where(fn > 0, fs / np.maximum(fn, 1e-12), raw)
            fill = fm[cat2f]
        val = np.where(ok, s_k / np.maximum(n_k, 1e-12), fill)
        return float((w_k * val).sum()), raw, obs, float(w_k[~ok].sum())

    point = est(np.ones(len(ae_s)), np.ones(len(ae_t)))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        rw_s = np.bincount(rng.integers(0, len(ucs), len(ucs)), minlength=len(ucs))[comp_s].astype(float)
        rw_t = np.bincount(rng.integers(0, len(uct), len(uct)), minlength=len(uct))[comp_t].astype(float)
        std_b, raw_b, obs_b, fbw = est(rw_s, rw_t)
        pen = obs_b - raw_b
        boots.append((std_b, (std_b - raw_b) / pen if pen != 0 else np.nan, fbw))
    boots = np.asarray(boots)
    n_k = np.bincount(cs, minlength=K)
    w_k = np.bincount(ct, minlength=K)
    used = w_k > 0
    std, raw, obs, fbw = point
    frac = (std - raw) / (obs - raw) if obs != raw else np.nan
    flo, fhi = np.nanpercentile(boots[:, 1], [2.5, 97.5])
    slo, shi = np.percentile(boots[:, 0], [2.5, 97.5])
    return {"MAE_standardized": std, "MAE_std_joint_lo": float(slo), "MAE_std_joint_hi": float(shi),
            "frac_of_penalty_explained": float(frac), "frac_joint_ci_lo": float(flo), "frac_joint_ci_hi": float(fhi),
            "n_strata_with_target_weight": int(used.sum()),
            "min_source_rows_per_stratum": int(n_k[used].min()),
            "min_target_rows_per_stratum": int(w_k[used].min()),
            "frac_target_weight_fallback": fbw,
            "frac_target_weight_fallback_boot_mean": float(boots[:, 2].mean()),
            "n_boot_joint": n_boot}


def adjusted_split_penalty(df: pd.DataFrame, covariates: list[str]) -> pd.DataFrame:
    """OLS ``abs_error ~ C(split_type) + covariates`` with compound-clustered SEs (reference: random)."""
    import statsmodels.formula.api as smf

    d = df.dropna(subset=covariates).copy()
    for c in covariates:
        d[c] = d[c].astype(str)
    rhs = " + ".join(["C(split_type, Treatment('random'))"] + [f"C({c})" for c in covariates])
    groups = pd.factorize(d["compound_id"])[0]
    fit = smf.ols(f"abs_error ~ {rhs}", data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    rows = []
    for name in fit.params.index:
        if name.startswith("C(split_type"):
            st = name.split("[T.")[1].rstrip("]")
            ci = fit.conf_int().loc[name]
            rows.append({"split_type": st, "adjusted_for": "+".join(covariates) or "none",
                         "penalty_vs_random": float(fit.params[name]), "lo": float(ci[0]),
                         "hi": float(ci[1]), "p_value": float(fit.pvalues[name]), "n": int(fit.nobs)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- chemistry
PHYSCHEM_BINS = {
    "MolWt": ((0, 150, 250, 350, 500, 750, np.inf), "molecular weight (g/mol)"),
    "MolLogP": ((-np.inf, 0, 2, 4, 6, 8, np.inf), "Crippen cLogP"),
    "TPSA": ((0, 20, 40, 60, 90, 140, np.inf), "TPSA (A^2)"),
    "HeavyAtomCount": ((0, 10, 20, 30, 40, 60, np.inf), "heavy atoms"),
    "NumRotatableBonds": ((0, 1, 3, 6, 11, np.inf), "rotatable bonds"),
    "NumAromaticRings": ((0, 1, 2, 3, 4, np.inf), "aromatic rings"),
    "formal_charge_atoms": ((0, 1, 2, 3, np.inf), "atoms with formal charge"),
    "is_multicomponent": (None, "multicomponent (salt / mixture)"),
}


def physchem_frame(dataset: str = "aqsoldb") -> pd.DataFrame:
    """Descriptors used for chemistry binning (from the shared rdkit2d cache) plus curation flags."""
    X, ids, names = load_or_compute(dataset, "rdkit2d")
    cols = ["MolWt", "MolLogP", "TPSA", "HeavyAtomCount", "NumRotatableBonds", "NumAromaticRings"]
    out = pd.DataFrame(X[:, [names.index(c) for c in cols]], columns=cols)
    out.insert(0, "compound_id", ids.astype(str))
    meta = load_dataset(dataset)[["compound_id", "formal_charge_atoms", "is_multicomponent"]]
    out = out.merge(meta, on="compound_id", how="left")
    out["is_multicomponent"] = out["is_multicomponent"].astype(str).str.lower().map({"true": "salt/multi",
                                                                                      "false": "single"})
    return out


def physchem_long(errors: pd.DataFrame, phys: pd.DataFrame) -> pd.DataFrame:
    """Long table (property, bin) for every error row."""
    dup = [c for c in phys.columns if c != "compound_id" and c in errors.columns]
    d = errors.drop(columns=dup).merge(phys, on="compound_id", how="left")
    parts = []
    for prop, (edges, _) in PHYSCHEM_BINS.items():
        x = d.copy()
        if edges is None:
            x["bin"] = x[prop].astype(str)
            x["bin_order"] = x["bin"].map({"single": 0, "salt/multi": 1})
        else:
            labels = _int_bin_labels(edges) if prop in ("NumRotatableBonds", "NumAromaticRings",
                                                        "formal_charge_atoms", "HeavyAtomCount") else None
            add_bins(x, prop, edges, "bin", labels=labels)
            x["bin_order"] = x["bin"].cat.codes
            x["bin"] = x["bin"].astype(str)
        x["property"] = prop
        parts.append(x)
    return pd.concat(parts, ignore_index=True)


def _int_bin_labels(edges) -> list[str]:
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        if np.isinf(hi):
            out.append(f">= {int(lo)}")
        elif hi - lo == 1:
            out.append(f"{int(lo)}")
        else:
            out.append(f"{int(lo)}-{int(hi) - 1}")
    return out


# SMARTS definitions; a class matches when ANY of its patterns matches.
FUNCTIONAL_GROUPS: dict[str, list[str]] = {
    "carboxylic acid / carboxylate": ["[CX3](=O)[OX2H1,OX1-]"],
    "aliphatic amine": ["[NX3;H2,H1,H0;+0;!$(N~[!#6;!#1]);!$(N-[#6]=[#7,#8,#16]);!$(N-a);!$(N-C#N)]",
                        "[NX4+;H1,H2,H3]"],
    "quaternary ammonium": ["[NX4+;H0;!$([NX4+]-[OX1-])]"],   # N-oxides excluded
    "amide / urea / carbamate": ["[CX3](=[OX1])[NX3]"],
    "ester": ["[#6][CX3](=O)[OX2H0][#6]"],
    "alcohol (aliphatic OH)": ["[OX2H][CX4]"],
    "phenol": ["[OX2H]c"],
    "nitro": ["[$([NX3](=O)=O),$([NX3+](=O)[O-])]"],
    "sulfonamide": ["[SX4](=O)(=O)[NX3]"],
    "sulfonic acid / sulfonate / sulfate monoester": ["[SX4](=O)(=O)[OX2H,OX1-]"],
    "phosphate / phosphonate (incl. thio)": ["[PX4](=[OX1,SX1])[OX2,OX1-,SX2]"],
    "halogenated aromatic": ["c[F,Cl,Br,I]"],
    "polyhalogenated (>=3 halogens)": [],      # counted programmatically
    "PAH / fused aromatic (>=3 rings)": [],    # ring-fusion analysis
    "steroid core": ["[#6]1~[#6]~[#6]~[#6]2~[#6](~[#6]~1)~[#6]~[#6]~[#6]1~[#6]~2~[#6]~[#6]~[#6]2~[#6]~[#6]~[#6]~[#6]~1~2"],
    "azo": ["[#6][NX2]=[NX2][#6]"],
    "long alkyl chain (>=C10)": ["[CH3,CH2;!R]" + "[CH2;!R]" * 8 + "[CH3,CH2;!R]"],
    "organosilicon": ["[#14]"],
    "boron": ["[#5]"],
    "amino acid / peptide": ["[NX3,NX4+;!$(N-C=O)][CX4][CX3](=O)[OX2H,OX1-]",
                             "[NX3][CX4][CX3](=O)[NX3][CX4][CX3]=O"],
    "heteroaromatic N": ["[n]"],
    "hydrocarbon (C/H only)": [],              # element check
}


def _largest_fused_aromatic_system(mol) -> int:
    rings = [set(r) for r in mol.GetRingInfo().BondRings()
             if all(mol.GetBondWithIdx(b).GetIsAromatic() for b in r)]
    if not rings:
        return 0
    parent = list(range(len(rings)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(rings)):
        for j in range(i + 1, len(rings)):
            if rings[i] & rings[j]:
                parent[find(i)] = find(j)
    roots = [find(i) for i in range(len(rings))]
    return max(roots.count(r) for r in set(roots))


def functional_group_matrix(smiles: list[str]) -> pd.DataFrame:
    """Boolean matrix (compounds x functional-group classes)."""
    pats = {k: [Chem.MolFromSmarts(s) for s in v] for k, v in FUNCTIONAL_GROUPS.items()}
    for k, v in pats.items():
        assert all(p is not None for p in v), f"bad SMARTS for {k}"
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        r = {k: any(m.HasSubstructMatch(p) for p in v) for k, v in pats.items() if v}
        n_hal = sum(1 for a in m.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I"))
        r["polyhalogenated (>=3 halogens)"] = n_hal >= 3
        r["PAH / fused aromatic (>=3 rings)"] = _largest_fused_aromatic_system(m) >= 3
        r["hydrocarbon (C/H only)"] = all(a.GetSymbol() in ("C", "H") for a in m.GetAtoms())
        rows.append(r)
    return pd.DataFrame(rows)[list(FUNCTIONAL_GROUPS)]


def functional_group_long(errors: pd.DataFrame, fg: pd.DataFrame) -> pd.DataFrame:
    """Long table: one row per (error row, matched functional-group class)."""
    melted = fg.melt(id_vars="compound_id", var_name="group", value_name="has")
    melted = melted[melted["has"]].drop(columns="has")
    return errors.merge(melted, on="compound_id", how="inner")


def flag_bias(tab: pd.DataFrame, min_compounds: int = 20, min_abs_bias: float = 0.2) -> pd.DataFrame:
    """Add ``biased`` (CI of signed error excludes 0, |bias| >= threshold, enough compounds)."""
    tab = tab.copy()
    excl = (tab["MSE_lo"] > 0) | (tab["MSE_hi"] < 0)
    tab["biased"] = excl & (tab["mean_signed_error"].abs() >= min_abs_bias) & (tab["n_compounds"] >= min_compounds)
    tab["bias_direction"] = np.where(~tab["biased"], "",
                                     np.where(tab["mean_signed_error"] > 0, "over-predicts solubility",
                                              "under-predicts solubility"))
    return tab


# --------------------------------------------------------------------------- worst predictions
IONISABLE_SMARTS = {
    "carboxylic acid": "[CX3](=O)[OX2H1,OX1-]",
    "sulfonic/sulfuric acid": "[SX4](=O)(=O)[OX2H,OX1-]",
    "P-acid": "[PX4](=[OX1,SX1])[OX2H,OX1-]",
    "aliphatic amine": FUNCTIONAL_GROUPS["aliphatic amine"][0],
    "protonated amine": "[NX4+;H1,H2,H3]",
    "quaternary ammonium": "[NX4+;H0;!$([NX4+]-[OX1-])]",
    "amidine/guanidine": "[NX3;!$(NC=O)][CX3]=[NX2;!$(N-O)]",
    "4-amino-azaarene base": "[NX3;H2,H1;!$(N[C,S]=O)]-c:c:c:[nX2]",
}
REACTIVE_SMARTS = {
    "isocyanate": "N=C=O", "acyl halide": "C(=O)[Cl,Br,I]", "benzylic halide": "c[CH2][Cl,Br,I]",
    "anhydride": "C(=O)OC(=O)", "sulfonyl halide": "S(=O)(=O)[Cl,Br,F]", "epoxide": "C1OC1",
}
_PATTERN_CACHE: dict = {}


def _matches(smiles: str, patterns: dict) -> list[str]:
    m = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if m is None:
        return []
    out = []
    for name, sma in patterns.items():
        p = _PATTERN_CACHE.setdefault(sma, Chem.MolFromSmarts(sma))
        if m.HasSubstructMatch(p):
            out.append(name)
    return out


def hypothesis(row: pd.Series) -> str:
    """Rule-based explanation(s) for a large error; rules are listed in order of evidential weight."""
    h = []
    gap_true = abs(row["y_true"] - row["nn_logS"])
    gap_pred = abs(row["y_pred"] - row["nn_logS"])
    multi = str(row.get("is_multicomponent")).lower() == "true"
    chain = bool(row.get("long alkyl chain (>=C10)", False))
    if row["nn_sim"] >= 0.95 and chain:
        h.append("NN is a chain-length homologue (ECFP4 radius 2 cannot see chain length)")
    if row["nn_sim"] >= 0.7 and gap_true >= 1.5 and gap_pred < gap_true:
        h.append(f"label disagrees with a close analogue (NN sim {row['nn_sim']:.2f}, "
                 f"|dlogS| {gap_true:.1f}); possible label error or activity cliff")
    if row.get("aqsol_sd", 0) > 0.5 or str(row.get("reliability_group")) in ("G2", "G4") \
            or str(row.get("dedup_conflict")).lower() == "true":
        h.append("conflicting replicate measurements")
    if row["nn_sim"] < 0.35:
        h.append("out of domain (NN sim < 0.35)")
    if multi:
        h.append("salt (counter-ion not modelled explicitly)" if row.get("formal_charge_atoms", 0) > 0
                 else "neutral multi-component SMILES (mixture/isomers): descriptors summed over components")
    react = _matches(row.get("smiles"), REACTIVE_SMARTS)
    if react:
        h.append(f"hydrolytically reactive ({', '.join(react)}): label may reflect hydrolysis products")
    ion = _matches(row.get("smiles"), IONISABLE_SMARTS)
    if ion:
        h.append(f"ionisable ({', '.join(ion)}): pH/salt-form dependent apparent solubility")
    if chain:
        h.append("surfactant-like long alkyl chain")
    if row["y_true"] < -8 or row["y_true"] > 1:
        h.append("extreme label (regression to the mean)")
    if row.get("organosilicon", False) or row.get("boron", False):
        h.append("sparse chemistry (Si/B)")
    if row.get("MolWt", 0) > 600 or row.get("NumRotatableBonds", 0) > 15:
        h.append("large / flexible molecule")
    if row["error"] > 0 and (row.get("PAH / fused aromatic (>=3 rings)", False)
                             or row.get("NumAromaticRings", 0) >= 3):
        h.append("rigid polyaromatic: crystal-packing (melting point) penalty underestimated")
    if not h:
        h.append("no rule fired (direction: " + ("too soluble" if row["error"] > 0 else "too insoluble") + ")")
    return "; ".join(h)


def worst_predictions(errors: pd.DataFrame, meta: pd.DataFrame, k: int = 30) -> pd.DataFrame:
    """Top-``k`` absolute errors, one row per compound (its worst occurrence across the given splits)."""
    d = errors.merge(meta, on="compound_id", how="left")
    d = d.sort_values("abs_error", ascending=False).drop_duplicates("compound_id").head(k).copy()
    d["hypothesis"] = d.apply(hypothesis, axis=1)
    d.insert(0, "rank", np.arange(1, len(d) + 1))
    return d


# --------------------------------------------------------------------------- scaffolds
def scaffold_errors(errors: pd.DataFrame, scaffolds: pd.DataFrame, min_compounds: int = 5,
                    n_boot: int = 1000) -> pd.DataFrame:
    """MAE per Bemis-Murcko scaffold over pooled seeds.

    Only scaffolds with >= ``min_compounds`` distinct test compounds (pooled over the seeds) are kept;
    a compound that is a test compound in several seeds contributes one row per seed.
    """
    d = errors.merge(scaffolds, on="compound_id", how="left")
    counts = d.groupby("scaffold")["compound_id"].nunique()
    keep = counts[counts >= min_compounds].index
    d = d[d["scaffold"].isin(keep)]
    tab = error_summary(d, ["scaffold"], n_boot=n_boot)
    extra = d.groupby("scaffold").agg(n_seeds=("split_name", "nunique"), sd_true=("y_true", "std"),
                                      example_names=("name", lambda s: "; ".join(pd.unique(s)[:3])))
    return tab.merge(extra.reset_index(), on="scaffold").sort_values("MAE", ascending=False)
