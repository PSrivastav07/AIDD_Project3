"""Adversarial QA audit of the foundation data, splits and features (evidence for foundation_review.md).

Method
------
Every check is computed from the stored artefacts (``data/processed``, ``data/splits``,
``data/features``, and - when present - the Phase-1 predictions in ``results/baseline``); nothing in
``data/`` or ``src/`` is modified. Outputs go to ``results/qa/``:

* ``audit_numbers.json``           every headline number quoted in the review
* ``tables/*.csv``                 per-item evidence (leakage per split, flagged labels, samples, ...)
* ``figures/*.png``                two summary figures (dpi 200)

Checks
------
1. **Near-duplicate leakage.** Three structure keys of decreasing strictness are compared between the
   training and the test part of every stored split:
   ``smiles_2d`` (the de-duplication key), the InChIKey connectivity block of the full standardised
   structure (``ik14``; catches charge states without counter-ion, tautomers, isotopologues) and the
   connectivity block of the *neutralised parent* (largest organic fragment + RDKit ``Uncharger``;
   catches salt vs. free acid/base and different salt forms). For matched test compounds the label
   difference to the matched training compounds is reported, and - if Phase-1 predictions exist - the
   LightGBM error on matched vs. unmatched test compounds.
2. **Scaffold leakage.** Stored scaffolds are recomputed; generic (CSK) scaffold overlap and
   label/composition shift of every split type are quantified (train-mean predictor MAE).
3. **Butina.** A hand-checked toy example, equality with ``rdkit.ML.Cluster.Butina`` on a 1,500
   compound subset, and an exact re-computation of ``data/splits/aqsoldb_butina_clusters.csv``.
4. **Removals.** A seeded sample of <=30 removed rows per reason with an automatic chemistry
   annotation (metal-carbon bond, ionic metal salt, ...), and the carbon-containing inorganics kept.
5. **Labels.** Extreme values, a physical upper bound (``logS <= log10(1000 * rho / MW)``), residuals
   from Delaney's published ESOL equation, conflicting duplicates and near-duplicate pairs.
6. **Representation artefacts.** The 17 InChIKey-inconsistent rows (classified automatically by
   comparing source InChIKey, raw-SMILES InChIKey and standardised InChIKey), carbanions / alkoxides
   created by the standardiser, net-charged single species.
7. **Descriptor NaNs** by column and compound class.
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from rdkit import Chem, DataStructs, RDLogger  # noqa: E402
from rdkit.Chem import Crippen  # noqa: E402
from rdkit.Chem.MolStandardize import rdMolStandardize  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402
from rdkit.ML.Cluster import Butina  # noqa: E402

from solpred.config import ROOT, load_config, load_json, save_json  # noqa: E402
from solpred.features import load_dataset, load_or_compute  # noqa: E402
from solpred.splits import butina_clusters, load_split, max_similarity, morgan_bits  # noqa: E402

RDLogger.DisableLog("rdApp.*")

OUT = ROOT / "results" / "qa"
BASELINE = ROOT / "results" / "baseline" / "predictions"
PALETTE = {"random": "#4C72B0", "scaffold": "#DD8452", "cluster": "#55A868", "external": "#C44E52",
           "loso": "#8172B3"}
SEEDS = [0, 1, 2, 3, 4]
GROUPED = ["random", "scaffold", "cluster"]
LOSO = ["A", "B", "C", "E", "F"]
LGBM = "lgbm__rdkit2d+morgan_counts.csv"
_UNCHARGER = rdMolStandardize.Uncharger()


# =========================================================================== structure keys
def _ik14(mol) -> str:
    try:
        return (Chem.MolToInchiKey(mol) or "")[:14]
    except Exception:  # noqa: BLE001
        return ""


def neutral_parent_ik14(parent_smiles: str) -> str:
    """InChIKey connectivity block of the neutralised largest organic fragment."""
    mol = Chem.MolFromSmiles(parent_smiles)
    if mol is None:
        return ""
    return _ik14(_UNCHARGER.uncharge(mol))


def generic_scaffold(scaffold: str) -> str:
    """Cyclic skeleton (all atoms C, all bonds single) of a Murcko scaffold; '' for acyclics."""
    if not scaffold:
        return ""
    mol = Chem.MolFromSmiles(scaffold)
    if mol is None:
        return "?"
    try:
        return Chem.MolToSmiles(MurckoScaffold.MakeScaffoldGeneric(mol))
    except Exception:  # noqa: BLE001
        return "?"


def structure_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Per-compound keys used by the leakage audit (index = compound_id)."""
    out = pd.DataFrame(index=df["compound_id"].to_numpy())
    out["smiles_2d"] = df["smiles_2d"].to_numpy()
    out["ik14"] = df["inchikey"].str[:14].to_numpy()
    out["parent_ik14"] = [neutral_parent_ik14(s) for s in df["parent_smiles"]]
    out["generic_scaffold"] = [generic_scaffold(s) for s in df["scaffold"].fillna("")]
    out["scaffold"] = df["scaffold"].fillna("").to_numpy()
    out["logS"] = df["logS"].to_numpy(dtype=float)
    out["is_multicomponent"] = df["is_multicomponent"].to_numpy(dtype=bool)
    return out


# =========================================================================== 1. near-duplicate leakage
def within_dataset_groups(keys: pd.DataFrame, key: str) -> pd.DataFrame:
    """Groups of >1 compound sharing ``key`` (but, by construction, not ``smiles_2d``)."""
    k = keys[keys[key] != ""]
    dup = k[k[key].duplicated(keep=False)].copy()
    dup["group_size"] = dup.groupby(key)[key].transform("size")
    dup["group_range_logS"] = dup.groupby(key)["logS"].transform(lambda x: x.max() - x.min())
    return dup.sort_values([key, "logS"])


def _baseline_test_preds(stype: str, name: str) -> pd.Series | None:
    p = BASELINE / stype / name / LGBM
    if not p.exists():
        return None
    df = pd.read_csv(p)
    df = df[df["subset"] == "test"]
    return pd.Series((df["y_pred"] - df["y_true"]).abs().to_numpy(), index=df["compound_id"].astype(str))


def leakage_for_split(split: dict, train_keys: pd.DataFrame, test_keys: pd.DataFrame,
                      stype: str, name: str) -> tuple[dict, pd.DataFrame]:
    """Key matches between test and *train* (val is used for early stopping only)."""
    tr = train_keys.loc[split["train"]]
    te = test_keys.loc[split["test"]]
    row = {"split_type": stype, "split_name": name, "n_train": len(tr), "n_test": len(te)}
    errs = _baseline_test_preds(stype, name)
    pairs = []
    for key in ("smiles_2d", "ik14", "parent_ik14"):
        lab_by_key = tr[tr[key] != ""].groupby(key)["logS"].mean()
        hit = te[key].isin(lab_by_key.index) & (te[key] != "")
        row[f"n_match_{key}"] = int(hit.sum())
        row[f"frac_match_{key}"] = float(hit.mean()) if len(te) else np.nan
        if hit.any():
            d = (te.loc[hit, "logS"] - te.loc[hit, key].map(lab_by_key)).abs()
            row[f"match_{key}_abs_dlogS_mean"] = float(d.mean())
            row[f"match_{key}_abs_dlogS_median"] = float(d.median())
            if key != "smiles_2d":
                pairs.append(pd.DataFrame({"split_type": stype, "split_name": name, "key": key,
                                           "test_id": d.index, "abs_dlogS_to_train_match": d.to_numpy(),
                                           "test_is_multicomponent": te.loc[hit, "is_multicomponent"].to_numpy()}))
        if errs is not None and key != "smiles_2d":
            e = errs.reindex(te.index)
            row[f"lgbm_mae_matched_{key}"] = float(e[hit].mean()) if hit.any() else np.nan
            row[f"lgbm_mae_unmatched_{key}"] = float(e[~hit].mean())
            row[f"lgbm_mae_all"] = float(e.mean())
            row[f"lgbm_mae_all_minus_matched_{key}"] = float(e[~hit].mean() - e.mean())
    return row, (pd.concat(pairs) if pairs else pd.DataFrame())


def audit_leakage(aq_keys: pd.DataFrame, es_keys: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, pairs = [], []
    for stype in GROUPED:
        for seed in SEEDS:
            r, p = leakage_for_split(load_split(stype, seed), aq_keys, aq_keys, stype, f"seed{seed}")
            rows.append(r)
            pairs.append(p)
    for src in LOSO:
        r, p = leakage_for_split(load_split("loso", source=src), aq_keys, aq_keys, "loso", src)
        rows.append(r)
        pairs.append(p)
    r, p = leakage_for_split(load_split("external", name="aqsoldb_to_esol"), aq_keys, es_keys,
                             "external", "aqsoldb_to_esol")
    rows.append(r)
    pairs.append(p)
    r, p = leakage_for_split(load_split("external", name="esol_to_aqsoldb"), es_keys, aq_keys,
                             "external", "esol_to_aqsoldb")
    rows.append(r)
    pairs.append(p)
    return pd.DataFrame(rows), pd.concat([p for p in pairs if len(p)], ignore_index=True)


def leakage_summary(by_split: pd.DataFrame, pairs: pd.DataFrame) -> dict:
    out = {}
    for stype, g in by_split.groupby("split_type", sort=False):
        d = {"n_splits": int(len(g))}
        for key in ("smiles_2d", "ik14", "parent_ik14"):
            d[f"n_match_{key}_mean"] = float(g[f"n_match_{key}"].mean())
            d[f"n_match_{key}_range"] = [int(g[f"n_match_{key}"].min()), int(g[f"n_match_{key}"].max())]
            d[f"pct_match_{key}_mean"] = float(100 * g[f"frac_match_{key}"].mean())
        for col in ("lgbm_mae_all", "lgbm_mae_matched_parent_ik14", "lgbm_mae_unmatched_parent_ik14",
                    "lgbm_mae_matched_ik14", "lgbm_mae_all_minus_matched_parent_ik14"):
            if col in g and g[col].notna().any():
                d[col + "_mean"] = float(g[col].mean())
                d[col + "_n_splits"] = int(g[col].notna().sum())
        out[stype] = d
    if len(pairs):
        pk = pairs[pairs["key"] == "parent_ik14"]
        ik = pairs[pairs["key"] == "ik14"]
        # parent-only matches = salt-form variants (not the same full structure)
        merged = pk.merge(ik[["split_name", "split_type", "test_id"]].assign(ik=True),
                          on=["split_type", "split_name", "test_id"], how="left")
        parent_only = merged[merged["ik"].isna()]
        out["pairs"] = {
            "ik14_pairs_pooled": int(len(ik)),
            "ik14_abs_dlogS_median": float(ik["abs_dlogS_to_train_match"].median()) if len(ik) else None,
            "ik14_frac_abs_dlogS_lt_0.3": float((ik["abs_dlogS_to_train_match"] < 0.3).mean()) if len(ik) else None,
            "parent_only_pairs_pooled": int(len(parent_only)),
            "parent_only_abs_dlogS_median": float(parent_only["abs_dlogS_to_train_match"].median()),
            "parent_only_abs_dlogS_mean": float(parent_only["abs_dlogS_to_train_match"].mean()),
            "parent_only_frac_abs_dlogS_gt_1": float((parent_only["abs_dlogS_to_train_match"] > 1).mean()),
        }
    return out


# =========================================================================== 2. scaffolds and shift
def recompute_scaffold(parent_smiles: str) -> str:
    mol = Chem.MolFromSmiles(parent_smiles)
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)
    except Exception:  # noqa: BLE001
        return ""


def audit_scaffolds_and_shift(aq: pd.DataFrame, keys: pd.DataFrame, fps_by_id: dict) -> tuple[dict, pd.DataFrame]:
    recomputed = [recompute_scaffold(s) for s in aq["parent_smiles"]]
    n_mismatch = int((pd.Series(recomputed, index=aq.index) != aq["scaffold"]).sum())
    rows = []
    for stype in GROUPED:
        for seed in SEEDS:
            sp = load_split(stype, seed)
            tr, te = keys.loc[sp["train"]], keys.loc[sp["test"]]
            sims = max_similarity([fps_by_id[i] for i in sp["test"]], [fps_by_id[i] for i in sp["train"]])
            rows.append({
                "split_type": stype, "seed": seed,
                "test_frac_scaffold_in_train": float(te["scaffold"].isin(set(tr["scaffold"])).mean()),
                "test_frac_generic_scaffold_in_train": float(te["generic_scaffold"].isin(set(tr["generic_scaffold"])).mean()),
                "test_frac_cyclic_generic_scaffold_in_train": float(
                    te.loc[te["scaffold"] != "", "generic_scaffold"].isin(set(tr["generic_scaffold"])).mean()),
                "test_frac_acyclic": float((te["scaffold"] == "").mean()),
                "test_frac_multicomponent": float(te["is_multicomponent"].mean()),
                "train_logS_mean": float(tr["logS"].mean()), "test_logS_mean": float(te["logS"].mean()),
                "test_logS_sd": float(te["logS"].std()),
                "train_mean_predictor_test_MAE": float((te["logS"] - tr["logS"].mean()).abs().mean()),
                "test_maxsim_median": float(np.median(sims)),
                "test_frac_maxsim_ge_0.4": float((sims >= 0.4).mean()),
                "test_frac_maxsim_ge_0.7": float((sims >= 0.7).mean()),
                "test_frac_maxsim_eq_1": float((sims >= 1.0 - 1e-12).mean()),
            })
    df = pd.DataFrame(rows)
    agg = df.drop(columns="seed").groupby("split_type", sort=False).agg(["mean", "min", "max"])
    summary = {"stored_scaffold_recompute_mismatches": n_mismatch,
               "by_split_type": {st: {f"{c}_{s}": float(agg.loc[st, (c, s)]) for c in agg.columns.levels[0]
                                      for s in ("mean", "min", "max") if (c, s) in agg.columns}
                                 for st in agg.index}}
    return summary, df


# =========================================================================== 3. Butina
def _bv(bits, n=64):
    fp = DataStructs.ExplicitBitVect(n)
    for b in bits:
        fp.SetBit(int(b))
    return fp


def _partition(labels) -> set:
    groups = defaultdict(set)
    for i, lab in enumerate(labels):
        groups[int(lab)].add(i)
    return {frozenset(g) for g in groups.values()}


def rdkit_butina_partition(fps: list, cutoff: float) -> set:
    """RDKit reference on the reversed list (aligns RDKit's high-index tie-break with solpred's low-index)."""
    rev = fps[::-1]
    n = len(rev)
    dists = []
    for i in range(1, n):
        dists.extend(1.0 - s for s in DataStructs.BulkTanimotoSimilarity(rev[i], rev[:i]))
    return {frozenset(n - 1 - j for j in cl) for cl in Butina.ClusterData(dists, n, cutoff, isDistData=True)}


def audit_butina(aq: pd.DataFrame, fps: list, cutoff: float, full: bool) -> dict:
    x, y, z, w = _bv(range(0, 6)), _bv(range(0, 10)), _bv(range(4, 14)), _bv(range(8, 18))
    toy = butina_clusters([x, y, z, w], cutoff).tolist()
    out = {"toy_chain_labels": toy, "toy_expected": [0, 0, 0, 1], "toy_ok": toy == [0, 0, 0, 1]}
    sub = fps[:1500]
    out["rdkit_reference_subset_n"] = len(sub)
    out["rdkit_reference_identical_partition"] = _partition(butina_clusters(sub, cutoff)) == rdkit_butina_partition(sub, cutoff)
    stored = pd.read_csv(ROOT / "data" / "splits" / "aqsoldb_butina_clusters.csv", dtype={"compound_id": str})
    sizes = stored["cluster"].value_counts()
    out.update({"n_clusters": int(sizes.size), "n_singletons": int((sizes == 1).sum()),
                "frac_compounds_in_singletons": float((sizes == 1).sum() / len(stored)),
                "largest_cluster": int(sizes.max()), "n_clusters_ge_100": int((sizes >= 100).sum()),
                "frac_compounds_in_top10_clusters": float(sizes.iloc[:10].sum() / len(stored))})
    if full:
        t = time.time()
        labels = butina_clusters(fps, cutoff)
        out["full_recompute_seconds"] = round(time.time() - t, 1)
        out["full_recompute_identical_to_stored"] = bool(
            stored["compound_id"].tolist() == aq["compound_id"].tolist()
            and np.array_equal(labels, stored["cluster"].to_numpy()))
        # centroid property: every member within the cutoff of the cluster's centroid (first assigned = most nbrs)
    return out


# =========================================================================== 4. removals
def _parse_any(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        m = Chem.MolFromSmiles(smi, sanitize=False)
    return m


def annotate_removed(row, organic: set, counter: set) -> str:
    mol = _parse_any(row["smiles_raw"]) if isinstance(row["smiles_raw"], str) else None
    if mol is None:
        return "unparseable"
    elems = {a.GetSymbol() for a in mol.GetAtoms()}
    n_c = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "C")
    if row["reason"] == "no_carbon":
        return "inorganic (no C)" if n_c == 0 else "C lost in standardisation"
    bad = elems - organic - counter
    mc = any({b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} & bad and
             "C" in {b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} for b in mol.GetBonds())
    if mc:
        return "organometallic (metal-C bond)"
    bonded = any({b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()} & bad for b in mol.GetBonds())
    return "metal complex / covalent metal-heteroatom" if bonded else "ionic salt of disallowed ion"


def audit_removed(cfg: dict, n_per_reason: int) -> tuple[dict, pd.DataFrame]:
    organic, counter = set(cfg["curation"]["organic_elements"]), set(cfg["curation"]["counterion_elements"])
    rem = pd.read_csv(ROOT / "data" / "processed" / "aqsoldb_removed.csv", keep_default_na=False, na_values=[""])
    rem["annotation"] = [annotate_removed(r, organic, counter) for _, r in rem.iterrows()]
    bad_el = Counter()
    for e in rem.loc[rem["reason"] == "disallowed_element", "elements"].fillna(""):
        for el in set(e.split(";")) - organic - counter - {""}:
            bad_el[el] += 1
    # explicit concat: pandas>=3 drops the grouping column inside groupby.apply
    sample = pd.concat([g.sample(min(len(g), n_per_reason), random_state=0)
                        for _, g in rem.groupby("reason", sort=True)], ignore_index=True)
    summary = {"n_removed": int(len(rem)), "by_reason": rem["reason"].value_counts().to_dict(),
               "annotation_by_reason": {r: g["annotation"].value_counts().to_dict() for r, g in rem.groupby("reason")},
               "disallowed_element_counts": dict(bad_el.most_common()),
               "removed_logS_mean_by_reason": rem.groupby("reason")["logS"].mean().round(3).to_dict()}
    return summary, sample


def inorganic_carbon_kept(aq: pd.DataFrame) -> pd.DataFrame:
    """Kept compounds with <=1 carbon atom and no C-H bond (cyanides, carbonates, CO, CS2 ...)."""
    rows = []
    for _, r in aq.iterrows():
        mol = Chem.MolFromSmiles(r["smiles"])
        cs = [a for a in mol.GetAtoms() if a.GetSymbol() == "C"]
        if len(cs) <= 1 and all(a.GetTotalNumHs() == 0 for a in cs):
            rows.append({"compound_id": r["compound_id"], "name": r["name"], "smiles": r["smiles"], "logS": r["logS"]})
    return pd.DataFrame(rows)


# =========================================================================== 5. labels
DELANEY = {"intercept": 0.16, "clogp": -0.63, "mw": -0.0062, "rotatable_bonds": 0.066, "aromatic_proportion": -0.74}


def audit_labels(aq: pd.DataFrame, keys: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X, ids, names = load_or_compute("aqsoldb", "esol4")
    f = pd.DataFrame(X, index=ids, columns=names).loc[aq["compound_id"]]
    esol = DELANEY["intercept"] + sum(DELANEY[c] * f[c].to_numpy() for c in names)
    df = aq[["compound_id", "name", "smiles", "logS", "mol_wt", "source", "n_occurrences", "aqsol_sd",
             "reliability_group", "is_multicomponent", "n_merged", "merged_logS_range"]].copy()
    df["delaney_esol"] = esol
    df["delaney_residual"] = df["logS"] - esol
    for rho in (1.2, 2.0):
        df[f"logS_max_rho{rho}"] = np.log10(1000 * rho / df["mol_wt"])
    flags = {
        "logS_gt_1.5": df["logS"] > 1.5,
        "logS_lt_-10": df["logS"] < -10,
        "above_physical_bound_rho2.0": df["logS"] > df["logS_max_rho2.0"],
        "above_bound_rho1.2": df["logS"] > df["logS_max_rho1.2"],
        "abs_delaney_residual_gt_3_single_component": (df["delaney_residual"].abs() > 3) & ~df["is_multicomponent"],
        "aqsol_sd_gt_1": df["aqsol_sd"] > 1.0,
        "dedup_merged_range_gt_1": df["merged_logS_range"] > 1.0,
    }
    for k, v in flags.items():
        df[k] = v
    flagged = df[np.column_stack(list(flags.values())).any(1)].copy()
    vc = df["logS"].round(4).value_counts()
    sc = ~df["is_multicomponent"]
    res = df.loc[sc, "delaney_residual"]
    summary = {k: int(v.sum()) for k, v in flags.items()}
    summary.update({
        "delaney_residual_single_component": {"n": int(sc.sum()), "mean": float(res.mean()),
                                              "MAE": float(res.abs().mean()), "sd": float(res.std())},
        "most_repeated_logS_values": {str(k): int(v) for k, v in vc.head(5).items()},
        "n_logS_values_shared_by_ge5_compounds": int((vc >= 5).sum()),
        "logS_gt_1.5_ids": df.loc[flags["logS_gt_1.5"], "compound_id"].tolist(),
        "above_physical_bound_rho2.0_ids": df.loc[flags["above_physical_bound_rho2.0"], "compound_id"].tolist(),
        "frac_dataset_flagged_any": float(len(flagged) / len(df)),
    })
    return summary, flagged


def classify_dedup_conflicts(aq: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Merged groups: do members carry *different defined* stereo (true isomers) or one undefined record?"""
    raw = pd.read_csv(ROOT / "data" / "raw" / "aqsoldb_curated-solubility-dataset.csv").set_index("ID")
    rows = []
    for _, r in aq[aq["n_merged"] > 1].iterrows():
        isos, n_def = set(), 0
        for mid in r["merged_ids"].split(";"):
            m = Chem.MolFromSmiles(raw.loc[mid, "SMILES"])
            if m is None:
                continue
            iso = Chem.MolToSmiles(m)
            isos.add(iso)
            n_def += int(("@" in iso) or ("/" in iso) or ("\\" in iso))
        n = r["n_merged"]
        kind = ("distinct defined stereoisomers" if n_def == n and len(isos) == n
                else "same stereo / representation" if len(isos) == 1
                else "defined + undefined stereo record")
        rows.append({"compound_id": r["compound_id"], "name": r["name"], "smiles_2d": r["smiles_2d"],
                     "n_merged": n, "merged_ids": r["merged_ids"], "logS_median": r["logS"],
                     "range": r["merged_logS_range"], "sd": r["merged_logS_sd"], "conflict": r["dedup_conflict"],
                     "kind": kind})
    df = pd.DataFrame(rows).sort_values("range", ascending=False)
    summary = {"n_groups": int(len(df)), "kind_counts": df["kind"].value_counts().to_dict(),
               "conflict_kind_counts": df.loc[df["conflict"], "kind"].value_counts().to_dict(),
               "median_range_by_kind": df.groupby("kind")["range"].median().round(3).to_dict()}
    return summary, df


# =========================================================================== 6. representation artefacts
_NEUTRAL_METAL = re.compile(r"\[(?:Li|Na|K|Mg|Ca)\](?!\+)")
_ZWITTER_EXCLUDE = [Chem.MolFromSmarts(s) for s in ("[N+](=O)[O-]", "[#7+][O-]", "[N-]=[N+]", "[N+]=[N-]",
                                                     "[C-]#[N+]", "[C-]#[O+]")]


def _charge_profile(smi: str) -> dict:
    mol = Chem.MolFromSmiles(smi)
    frags = Chem.GetMolFrags(mol, asMols=True)
    excl = set()
    for patt in _ZWITTER_EXCLUDE:
        for match in mol.GetSubstructMatches(patt):
            excl.update(match)
    charged = [a for a in mol.GetAtoms() if a.GetFormalCharge() != 0 and a.GetIdx() not in excl]
    return {
        "net_charge": Chem.GetFormalCharge(mol), "n_frag": len(frags),
        "carbanion": any(a.GetSymbol() == "C" and a.GetFormalCharge() < 0 for a in mol.GetAtoms()),
        "alkoxide": mol.HasSubstructMatch(Chem.MolFromSmarts("[OX1-][CX4]")),
        "bare_proton": any(a.GetSymbol() == "H" and a.GetDegree() == 0 and a.GetFormalCharge() == 1 for a in mol.GetAtoms()),
        "zwitterion_single": len(frags) == 1 and Chem.GetFormalCharge(mol) == 0
        and any(a.GetFormalCharge() > 0 for a in charged) and any(a.GetFormalCharge() < 0 for a in charged),
    }


def audit_charges(aq: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    prof = pd.DataFrame([_charge_profile(s) for s in aq["smiles"]], index=aq.index)
    df = pd.concat([aq[["compound_id", "name", "smiles_raw", "smiles", "source", "logS"]], prof], axis=1)
    df["raw_neutral_metal"] = aq["smiles_raw"].str.contains(_NEUTRAL_METAL)
    df["raw_has_alkoxide"] = [Chem.MolFromSmiles(s).HasSubstructMatch(Chem.MolFromSmarts("[OX1-][CX4]"))
                              if Chem.MolFromSmiles(s) is not None else False for s in aq["smiles_raw"]]
    df["alkoxide_created"] = df["alkoxide"] & ~df["raw_has_alkoxide"]
    df["charged_single_species"] = (df["n_frag"] == 1) & (df["net_charge"] != 0)
    df["unbalanced_salt"] = (df["n_frag"] > 1) & (df["net_charge"] != 0)
    # descriptor impact of the missing neutralisation (Crippen logP charged vs. uncharged)
    dlogp = []
    for s in df.loc[df["charged_single_species"] | df["zwitterion_single"], "smiles"]:
        m = Chem.MolFromSmiles(s)
        dlogp.append(abs(Crippen.MolLogP(m) - Crippen.MolLogP(_UNCHARGER.uncharge(m))))
    by_src = df.groupby("source")[["charged_single_species", "zwitterion_single"]].mean().mul(100).round(2)
    flagged = df[df[["carbanion", "alkoxide_created", "bare_proton", "charged_single_species", "unbalanced_salt",
                     "raw_neutral_metal"]].any(axis=1)]
    summary = {k: int(df[k].sum()) for k in ("carbanion", "alkoxide_created", "bare_proton", "raw_neutral_metal",
                                             "charged_single_species", "zwitterion_single", "unbalanced_salt")}
    summary["net_charge_nonzero"] = int((df["net_charge"] != 0).sum())
    summary["carbanion_ids"] = df.loc[df["carbanion"], "compound_id"].tolist()
    summary["alkoxide_created_ids"] = df.loc[df["alkoxide_created"], "compound_id"].tolist()
    summary["bare_proton_ids"] = df.loc[df["bare_proton"], "compound_id"].tolist()
    summary["crippen_logp_abs_shift_charged_vs_neutral"] = {"n": len(dlogp), "mean": float(np.mean(dlogp)),
                                                            "median": float(np.median(dlogp))}
    summary["pct_charged_or_zwitterion_by_source"] = by_src.to_dict(orient="index")
    return summary, flagged


def audit_inchikey(aq: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    bad = aq[~aq["inchikey_consistent"]].copy()
    raw_ik = []
    for s in bad["smiles_raw"]:
        m = Chem.MolFromSmiles(s)
        raw_ik.append(_ik14(m) if m is not None else "")
    bad["raw_smiles_ik14"] = raw_ik
    bad["source_ik14"] = bad["inchikey_source"].str[:14]
    bad["std_ik14"] = bad["inchikey"].str[:14]
    bad["category"] = np.select(
        [bad["raw_smiles_ik14"] == bad["source_ik14"], bad["raw_smiles_ik14"] == bad["std_ik14"]],
        ["standardiser changed structure (raw SMILES matched source InChIKey)",
         "AqSolDB SMILES disagrees with its own InChIKey (standardiser neutral)"],
        "both: source SMILES/InChIKey disagree AND standardiser changed structure")
    cols = ["compound_id", "name", "smiles_raw", "smiles", "logS", "source_ik14", "raw_smiles_ik14", "std_ik14", "category"]
    return {"n": int(len(bad)), "category_counts": bad["category"].value_counts().to_dict()}, bad[cols]


# =========================================================================== 7. descriptor NaNs
def audit_descriptor_nans(aq: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    X, ids, names = load_or_compute("aqsoldb", "rdkit2d")
    nan = np.isnan(X)
    rows_nan = nan.any(1)
    metal = aq["elements"].str.contains(r"\b(?:Li|Na|K|Mg|Ca)\b", regex=True).to_numpy()
    cols = {names[j]: int(nan[:, j].sum()) for j in np.where(nan.any(0))[0]}
    const = [names[j] for j in range(X.shape[1]) if np.nanstd(X[:, j]) == 0]
    summary = {
        "n_rows_with_nan": int(rows_nan.sum()), "frac_rows_with_nan": float(rows_nan.mean()),
        "frac_cells_nan": float(nan.mean()), "nan_by_column": cols,
        "frac_nan_rows_metal_salts": float(rows_nan[metal].mean()) if metal.any() else None,
        "frac_nan_rows_non_metal": float(rows_nan[~metal].mean()),
        "n_metal_containing": int(metal.sum()),
        "frac_nan_rows_that_are_multicomponent": float(aq.loc[rows_nan, "is_multicomponent"].mean()),
        "logS_mean_nan_rows": float(aq.loc[rows_nan, "logS"].mean()),
        "logS_mean_other_rows": float(aq.loc[~rows_nan, "logS"].mean()),
        "constant_columns": const,
    }
    detail = aq.loc[rows_nan, ["compound_id", "name", "smiles", "elements", "is_multicomponent", "logS"]].copy()
    detail["n_nan_descriptors"] = nan[rows_nan].sum(1)
    return summary, detail


# =========================================================================== 8. other checks
def audit_other(aq: pd.DataFrame) -> dict:
    out = {}
    occ = aq.set_index("compound_id")["n_occurrences"]
    out["loso_test_frac_label_from_multiple_measurements"] = {
        src: float((occ.loc[load_split("loso", source=src)["test"]] > 1).mean()) for src in LOSO}
    summ = pd.read_csv(ROOT / "data" / "splits" / "split_summary.csv")
    out["split_summary_rows"] = int(len(summ))
    return out


# =========================================================================== figures
def figure_leakage(by_split: pd.DataFrame, path: Path) -> None:
    order = ["random", "scaffold", "cluster", "loso", "external"]
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    width = 0.38
    for i, (key, label, hatch) in enumerate([("ik14", "same InChIKey connectivity (charge state / tautomer)", ""),
                                             ("parent_ik14", "same neutral parent (incl. other salt forms)", "//")]):
        means = [100 * by_split.loc[by_split["split_type"] == s, f"frac_match_{key}"].mean() for s in order]
        sds = [100 * by_split.loc[by_split["split_type"] == s, f"frac_match_{key}"].std() for s in order]
        xs = np.arange(len(order)) + (i - 0.5) * width
        ax.bar(xs, means, width, yerr=np.nan_to_num(sds), color=[PALETTE[s] for s in order], hatch=hatch,
               edgecolor="black", linewidth=0.6, capsize=3, label=label)
    ax.set_xticks(np.arange(len(order)), ["random\n(5 seeds)", "scaffold\n(5 seeds)", "cluster\n(5 seeds)",
                                          "LOSO\n(5 sources)", "external\n(2 directions)"])
    ax.set_ylabel("test compounds with a train match (%)")
    ax.set_title("Near-duplicates surviving smiles_2d de-duplication (test vs. train)")
    leg = ax.legend(frameon=False, fontsize=8)
    for h in leg.legend_handles:
        h.set_facecolor("white")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_label_bounds(lab_all: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    mw = lab_all["mol_wt"]
    above = lab_all["logS"] > np.log10(2000 / mw)
    ax.scatter(mw[~above], lab_all.loc[~above, "logS"], s=3, alpha=0.3, color="#4C72B0", label="AqSolDB compound")
    ax.scatter(mw[above], lab_all.loc[above, "logS"], s=14, color="#C44E52", label=f"above bound (n={int(above.sum())})")
    grid = np.logspace(np.log10(mw.min()), np.log10(mw.max()), 200)
    ax.plot(grid, np.log10(2000 / grid), "k--", lw=1, label=r"$\log_{10}(1000\,\rho/MW)$, $\rho$=2 g/mL")
    ax.set_xscale("log")
    ax.set_xlabel("molecular weight (g/mol)")
    ax.set_ylabel("logS (log mol/L)")
    ax.set_title("Physical upper bound on molar solubility")
    ax.legend(frameon=False, fontsize=7)
    ax = axes[1]
    sc = ~lab_all["is_multicomponent"]
    lim = 12.0
    res = lab_all["delaney_residual"]
    n_out = int((res.abs() > lim).sum())
    bins = np.linspace(-lim, lim, 97)
    ax.hist(res[sc].clip(-lim, lim), bins=bins, color="#4C72B0", alpha=0.8, label="single-component")
    ax.hist(res[~sc].clip(-lim, lim), bins=bins, color="#DD8452", alpha=0.6, label="multicomponent (salts)")
    ax.text(0.02, 0.97, f"|residual| > {lim:g}: n={n_out} (clipped)", transform=ax.transAxes, fontsize=7, va="top")
    for v in (-3, 3):
        ax.axvline(v, color="k", ls="--", lw=0.8)
    ax.set_xlabel("logS - Delaney ESOL equation (log mol/L)")
    ax.set_ylabel("compounds")
    ax.set_title("Residual from the published ESOL equation")
    ax.legend(frameon=False, fontsize=8)
    for a in axes:
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# =========================================================================== driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(OUT), help="output directory (default results/qa)")
    ap.add_argument("--skip-full-butina", action="store_true", help="skip the full 9k re-clustering (~1 min)")
    ap.add_argument("--n-per-reason", type=int, default=30, help="removed rows sampled per reason")
    args = ap.parse_args()
    out = Path(args.out)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    t0 = time.time()
    aq, es = load_dataset("aqsoldb"), load_dataset("esol")
    numbers: dict = {"dataset_version": load_json(ROOT / "data" / "processed" / "DATASET_VERSION.json"),
                     "baseline_lgbm_available": {st: sum((BASELINE / st / f"seed{s}" / LGBM).exists() for s in SEEDS)
                                                 for st in GROUPED}}

    print("[1] structure keys + near-duplicate leakage", flush=True)
    aq_keys, es_keys = structure_keys(aq), structure_keys(es)
    for name, keys in (("aqsoldb", aq_keys), ("esol", es_keys)):
        for key in ("ik14", "parent_ik14"):
            g = within_dataset_groups(keys, key)
            numbers.setdefault("within_dataset", {}).setdefault(name, {})[key] = {
                "n_compounds_in_groups": int(len(g)), "n_groups": int(g[key].nunique()),
                "n_groups_range_gt_1": int(g.loc[g["group_range_logS"] > 1, key].nunique())}
            if name == "aqsoldb":
                g.join(aq.set_index("compound_id")[["name", "smiles"]]).to_csv(
                    out / "tables" / f"within_aqsoldb_groups_{key}.csv", index_label="compound_id")
    by_split, pairs = audit_leakage(aq_keys, es_keys)
    by_split.to_csv(out / "tables" / "leakage_by_split.csv", index=False)
    pairs.to_csv(out / "tables" / "leakage_matched_pairs.csv", index=False)
    numbers["leakage"] = leakage_summary(by_split, pairs)
    figure_leakage(by_split, out / "figures" / "near_duplicate_leakage.png")

    print("[2] scaffolds, similarity and label shift", flush=True)
    fps = morgan_bits(aq["smiles"].tolist(), cfg["splits"]["cluster"]["fp_radius"], cfg["splits"]["cluster"]["fp_nbits"])
    fps_by_id = dict(zip(aq["compound_id"], fps))
    numbers["scaffold_and_shift"], shift = audit_scaffolds_and_shift(aq, aq_keys, fps_by_id)
    shift.to_csv(out / "tables" / "split_shift_and_similarity.csv", index=False)

    print("[3] Butina", flush=True)
    numbers["butina"] = audit_butina(aq, fps, cfg["splits"]["cluster"]["butina_distance_cutoff"],
                                     full=not args.skip_full_butina)

    print("[4] removals", flush=True)
    numbers["removed"], sample = audit_removed(cfg, args.n_per_reason)
    sample.to_csv(out / "tables" / "removed_sample.csv", index=False)
    inorg = inorganic_carbon_kept(aq)
    inorg.to_csv(out / "tables" / "kept_inorganic_carbon.csv", index=False)
    numbers["kept_inorganic_carbon"] = {"n": int(len(inorg)), "ids": inorg["compound_id"].tolist()}

    print("[5] labels", flush=True)
    numbers["labels"], flagged = audit_labels(aq, aq_keys)
    flagged.to_csv(out / "tables" / "suspicious_labels.csv", index=False)
    numbers["dedup_groups"], dedup = classify_dedup_conflicts(aq)
    dedup.to_csv(out / "tables" / "dedup_merged_groups.csv", index=False)
    lab_all = aq[["compound_id", "logS", "mol_wt", "is_multicomponent"]].merge(
        flagged[["compound_id"]], how="left", on="compound_id")
    X, ids, names = load_or_compute("aqsoldb", "esol4")
    f = pd.DataFrame(X, index=ids, columns=names).loc[aq["compound_id"]]
    lab_all["delaney_residual"] = aq["logS"].to_numpy() - (DELANEY["intercept"] + sum(DELANEY[c] * f[c].to_numpy() for c in names))
    figure_label_bounds(lab_all, out / "figures" / "label_plausibility.png")

    print("[6] representation artefacts", flush=True)
    numbers["inchikey_inconsistent"], ik = audit_inchikey(aq)
    ik.to_csv(out / "tables" / "inchikey_inconsistent.csv", index=False)
    numbers["charges"], ch = audit_charges(aq)
    ch.to_csv(out / "tables" / "charge_artefacts.csv", index=False)

    print("[7] descriptor NaNs", flush=True)
    numbers["descriptor_nans"], nanrows = audit_descriptor_nans(aq)
    nanrows.to_csv(out / "tables" / "rdkit2d_nan_rows.csv", index=False)

    print("[8] other", flush=True)
    numbers["other"] = audit_other(aq)
    numbers["runtime_seconds"] = round(time.time() - t0, 1)
    save_json(numbers, out / "audit_numbers.json")
    print(f"done in {numbers['runtime_seconds']} s -> {out / 'audit_numbers.json'}")


if __name__ == "__main__":
    main()
