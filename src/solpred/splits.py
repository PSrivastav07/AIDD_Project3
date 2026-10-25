"""Leakage-safe data splits.

* ``random``   - uniform random 80/10/10 split.
* ``scaffold`` - Bemis-Murcko scaffold split (scaffold of the largest organic fragment), using the
  "balanced" assignment of Chemprop: scaffold sets larger than half the test size go to train
  first, then the remaining sets are shuffled (seeded) and assigned greedily to train/val/test.
  All acyclic molecules share the empty scaffold and therefore form one large group (in train).
* ``cluster``  - Butina clustering on ECFP4 (Morgan r=2, 2048 bits) with a Tanimoto distance
  cutoff; whole clusters are assigned with the same balanced procedure. This is the most
  stringent similarity-based split (no near neighbours of a test compound in training).
* ``loso``     - leave-one-source-out over the AqSolDB source letter (A-I).
* external     - train on one dataset with all overlapping compounds removed, test on the other.

Every split is stored as JSON: ``{"train": [ids], "val": [ids], "test": [ids], "meta": {...}}``.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from solpred.config import load_config, load_json, path, save_json

SPLIT_NAMES = ("train", "val", "test")


# --------------------------------------------------------------------------- fingerprints
def morgan_bits(smiles: list[str], radius: int = 2, nbits: int = 2048) -> list:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in smiles]


def max_similarity(query_fps: list, ref_fps: list, k: int = 1) -> np.ndarray:
    """Max (k=1) or mean top-k Tanimoto similarity of each query to the reference set."""
    out = np.empty(len(query_fps), dtype=float)
    for i, fp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
        out[i] = sims.max() if k == 1 else np.sort(sims)[-k:].mean()
    return out


def butina_clusters(fps: list, distance_cutoff: float) -> np.ndarray:
    """Butina (sphere-exclusion) clustering; returns a cluster label per molecule.

    Implemented with a boolean neighbour matrix (O(N^2) memory as bits/bytes) rather than RDKit's
    list-based ``Butina.ClusterData`` to keep memory low for ~10k molecules.
    """
    n = len(fps)
    sim_cutoff = 1.0 - distance_cutoff
    nbr = np.zeros((n, n), dtype=bool)
    for i in range(n):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        nbr[i] = sims >= sim_cutoff
    labels = np.full(n, -1, dtype=int)
    counts = nbr.sum(1)
    order = np.lexsort((np.arange(n), -counts))  # most neighbours first, ties by index
    cid = 0
    for i in order:
        if labels[i] != -1:
            continue
        members = np.where(nbr[i] & (labels == -1))[0]
        labels[members] = cid
        labels[i] = cid
        cid += 1
    return labels


# --------------------------------------------------------------------------- assignment
def balanced_group_split(groups: dict, n_total: int, fractions, seed: int) -> dict:
    """Chemprop-style balanced group split; ``groups`` maps group key -> list of ids."""
    f_train, f_val, f_test = fractions
    train_size, val_size, test_size = f_train * n_total, f_val * n_total, f_test * n_total
    sets = list(groups.values())
    big = [s for s in sets if len(s) > val_size / 2 or len(s) > test_size / 2]
    small = [s for s in sets if not (len(s) > val_size / 2 or len(s) > test_size / 2)]
    rng = np.random.RandomState(seed)
    rng.shuffle(big)
    rng.shuffle(small)
    out = {"train": [], "val": [], "test": []}
    for s in big + small:
        if len(out["train"]) + len(s) <= train_size:
            out["train"] += s
        elif len(out["val"]) + len(s) <= val_size:
            out["val"] += s
        else:
            out["test"] += s
    return out


def random_split(ids: list, fractions, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    perm = list(np.asarray(ids)[rng.permutation(len(ids))])
    n_train = int(round(fractions[0] * len(ids)))
    n_val = int(round(fractions[1] * len(ids)))
    return {"train": perm[:n_train], "val": perm[n_train:n_train + n_val], "test": perm[n_train + n_val:]}


def scaffold_split(df: pd.DataFrame, fractions, seed: int) -> dict:
    groups = defaultdict(list)
    for cid, scaf in zip(df["compound_id"], df["scaffold"].fillna("")):
        groups[scaf].append(cid)
    return balanced_group_split(groups, len(df), fractions, seed)


def cluster_split(df: pd.DataFrame, labels: np.ndarray, fractions, seed: int) -> dict:
    groups = defaultdict(list)
    for cid, lab in zip(df["compound_id"], labels):
        groups[int(lab)].append(cid)
    return balanced_group_split(groups, len(df), fractions, seed)


# --------------------------------------------------------------------------- diagnostics
def split_diagnostics(df: pd.DataFrame, split: dict, fps_by_id: dict) -> dict:
    """Sizes, label statistics, scaffold overlap and test->train nearest-neighbour similarity."""
    meta = {}
    lab = df.set_index("compound_id")["logS"]
    scaf = df.set_index("compound_id")["scaffold"].fillna("")
    for k in SPLIT_NAMES:
        y = lab.loc[split[k]]
        meta[f"n_{k}"] = len(split[k])
        meta[f"logS_mean_{k}"] = float(y.mean()) if len(y) else None
        meta[f"logS_std_{k}"] = float(y.std()) if len(y) > 1 else None
    tr_scaf = set(scaf.loc[split["train"]])
    te_scaf = scaf.loc[split["test"]]
    meta["test_frac_scaffold_seen_in_train"] = float(te_scaf.isin(tr_scaf).mean()) if len(te_scaf) else None
    meta["test_frac_acyclic"] = float((te_scaf == "").mean()) if len(te_scaf) else None
    if fps_by_id is not None and len(split["test"]):
        sims = max_similarity([fps_by_id[i] for i in split["test"]], [fps_by_id[i] for i in split["train"]])
        meta["test_maxsim_to_train_mean"] = float(sims.mean())
        meta["test_maxsim_to_train_median"] = float(np.median(sims))
        meta["test_frac_maxsim_ge_0.7"] = float((sims >= 0.7).mean())
        meta["test_frac_maxsim_lt_0.4"] = float((sims < 0.4).mean())
    return meta


def _check_disjoint(split: dict) -> None:
    sets = [set(split[k]) for k in SPLIT_NAMES]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2]), "split leakage"


# --------------------------------------------------------------------------- driver
def make_all_splits(cfg: dict | None = None) -> pd.DataFrame:
    cfg = cfg or load_config()
    proc = path("processed", cfg)
    aq = pd.read_csv(proc / "aqsoldb_clean.csv", keep_default_na=False, na_values=[""])
    es = pd.read_csv(proc / "esol_clean.csv", keep_default_na=False, na_values=[""])
    scfg = cfg["splits"]
    fr = scfg["fractions"]
    out_root = path("splits", cfg)
    fps = morgan_bits(aq["smiles"].tolist(), scfg["cluster"]["fp_radius"], scfg["cluster"]["fp_nbits"])
    fps_by_id = dict(zip(aq["compound_id"], fps))

    labels = butina_clusters(fps, scfg["cluster"]["butina_distance_cutoff"])
    pd.DataFrame({"compound_id": aq["compound_id"], "cluster": labels}).to_csv(
        out_root / "aqsoldb_butina_clusters.csv", index=False)

    summary = []
    ids = aq["compound_id"].tolist()
    for stype in scfg["types"]:
        for seed in scfg["seeds"]:
            if stype == "random":
                sp = random_split(ids, fr, seed)
            elif stype == "scaffold":
                sp = scaffold_split(aq, fr, seed)
            elif stype == "cluster":
                sp = cluster_split(aq, labels, fr, seed)
            else:
                raise ValueError(stype)
            _check_disjoint(sp)
            assert sum(len(sp[k]) for k in SPLIT_NAMES) == len(aq)
            meta = {"dataset": "aqsoldb", "split_type": stype, "seed": seed, "fractions": fr,
                    **split_diagnostics(aq, sp, fps_by_id)}
            save_json({**{k: list(map(str, sp[k])) for k in SPLIT_NAMES}, "meta": meta},
                      out_root / "aqsoldb" / stype / f"seed{seed}.json")
            summary.append(meta)

    # leave-one-source-out: validation = random 10% of the remaining sources (seed 0)
    counts = aq["source"].value_counts()
    for src in sorted(counts[counts >= scfg["loso_min_compounds"]].index):
        test = aq.loc[aq["source"] == src, "compound_id"].tolist()
        rest = aq.loc[aq["source"] != src, "compound_id"].tolist()
        inner = random_split(rest, [0.9, 0.1, 0.0], 0)
        sp = {"train": inner["train"], "val": inner["val"] + inner["test"], "test": test}
        _check_disjoint(sp)
        meta = {"dataset": "aqsoldb", "split_type": "loso", "seed": 0, "held_out_source": src,
                **split_diagnostics(aq, sp, fps_by_id)}
        save_json({**sp, "meta": meta}, out_root / "aqsoldb" / "loso" / f"source_{src}.json")
        summary.append(meta)

    # external transfer: overlapping compounds (same smiles_2d) are removed from the training set
    es_fps = morgan_bits(es["smiles"].tolist(), scfg["cluster"]["fp_radius"], scfg["cluster"]["fp_nbits"])
    aq_no_overlap = aq.loc[~aq["in_esol"].astype(bool), "compound_id"].tolist()
    inner = random_split(aq_no_overlap, [0.9, 0.1, 0.0], 0)
    ext = {"train_dataset": "aqsoldb", "test_dataset": "esol",
           "train": inner["train"], "val": inner["val"] + inner["test"], "test": es["compound_id"].tolist()}
    sims = max_similarity(es_fps, [fps_by_id[i] for i in ext["train"]])
    ext["meta"] = {"split_type": "external", "name": "aqsoldb_to_esol", "n_train": len(ext["train"]),
                   "n_val": len(ext["val"]), "n_test": len(ext["test"]),
                   "removed_overlap_from_train": int(aq["in_esol"].astype(bool).sum()),
                   "test_maxsim_to_train_mean": float(sims.mean()),
                   "test_maxsim_to_train_median": float(np.median(sims))}
    save_json(ext, out_root / "external" / "aqsoldb_to_esol.json")
    summary.append(ext["meta"])

    es_ids = es["compound_id"].tolist()
    inner = random_split(es_ids, [0.9, 0.1, 0.0], 0)
    ext2 = {"train_dataset": "esol", "test_dataset": "aqsoldb",
            "train": inner["train"], "val": inner["val"] + inner["test"], "test": aq_no_overlap}
    es_fps_by_id = dict(zip(es["compound_id"], es_fps))
    sims = max_similarity([fps_by_id[i] for i in aq_no_overlap], [es_fps_by_id[i] for i in ext2["train"]])
    ext2["meta"] = {"split_type": "external", "name": "esol_to_aqsoldb", "n_train": len(ext2["train"]),
                    "n_val": len(ext2["val"]), "n_test": len(ext2["test"]),
                    "test_maxsim_to_train_mean": float(sims.mean()),
                    "test_maxsim_to_train_median": float(np.median(sims))}
    save_json(ext2, out_root / "external" / "esol_to_aqsoldb.json")
    summary.append(ext2["meta"])

    summ = pd.DataFrame(summary)
    summ.to_csv(out_root / "split_summary.csv", index=False)
    return summ


def load_split(split_type: str, seed: int | None = None, source: str | None = None,
               name: str | None = None, cfg: dict | None = None) -> dict:
    """Load a stored split.

    ``load_split("scaffold", 0)``; ``load_split("loso", source="A")``; ``load_split("external", name="aqsoldb_to_esol")``.
    """
    root = path("splits", cfg)
    if split_type == "loso":
        return load_json(root / "aqsoldb" / "loso" / f"source_{source}.json")
    if split_type == "external":
        return load_json(root / "external" / f"{name}.json")
    return load_json(root / "aqsoldb" / split_type / f"seed{seed}.json")
