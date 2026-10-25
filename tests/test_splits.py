"""Tests for ``solpred.splits``: split generators, Butina clustering and the stored split files.

Generic properties (disjoint, complete, deterministic, group-respecting) are tested on synthetic
inputs; the stored JSON files under ``data/splits`` are additionally checked for the leakage
guarantees each split type promises (no shared scaffold / Butina cluster / structure between
train and test). Tests over stored files skip when the splits have not been generated.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd
import pytest
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

from conftest import SPLITS, require
from solpred.splits import (SPLIT_NAMES, balanced_group_split, butina_clusters, cluster_split, load_split,
                            max_similarity, morgan_bits, random_split, scaffold_split)

FRACTIONS = [0.8, 0.1, 0.1]
SEEDS = [0, 1, 2, 3, 4]
GROUPED = ["random", "scaffold", "cluster"]
LOSO_SOURCES = ["A", "B", "C", "E", "F"]


def _bitvect(on_bits, n=64):
    fp = DataStructs.ExplicitBitVect(n)
    for b in on_bits:
        fp.SetBit(int(b))
    return fp


def _assert_partition(split: dict, universe: set) -> None:
    sets = [set(split[k]) for k in SPLIT_NAMES]
    for k, s in zip(SPLIT_NAMES, sets):
        assert len(s) == len(split[k]), f"duplicate ids inside {k}"
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sets[0] | sets[1] | sets[2] == universe


# --------------------------------------------------------------------------- generators (synthetic)
def test_random_split_sizes_partition_and_determinism():
    ids = [f"X-{i}" for i in range(1001)]
    a, b, c = random_split(ids, FRACTIONS, 0), random_split(ids, FRACTIONS, 0), random_split(ids, FRACTIONS, 1)
    _assert_partition(a, set(ids))
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (801, 100, 100)
    assert a == b
    assert a["test"] != c["test"]


def test_balanced_group_split_keeps_groups_intact():
    rng = np.random.RandomState(0)
    groups, k = {"big": [f"b{i}" for i in range(60)]}, 0
    while k < 340:
        size = int(rng.randint(1, 5))
        groups[f"g{k}"] = [f"s{k}_{j}" for j in range(size)]
        k += 1
    n = sum(len(v) for v in groups.values())
    out = balanced_group_split(groups, n, FRACTIONS, seed=3)
    _assert_partition(out, {i for v in groups.values() for i in v})
    where = {i: k for k in SPLIT_NAMES for i in out[k]}
    for members in groups.values():
        assert len({where[i] for i in members}) == 1, "a group was split across subsets"
    assert set(groups["big"]) <= set(out["train"])                 # sets > test_size/2 go to train first
    assert len(out["train"]) <= FRACTIONS[0] * n
    assert len(out["val"]) <= FRACTIONS[1] * n
    assert abs(len(out["test"]) - FRACTIONS[2] * n) <= 4 + 1       # off by at most the largest small group
    assert out == balanced_group_split(groups, n, FRACTIONS, seed=3)
    assert out != balanced_group_split(groups, n, FRACTIONS, seed=4)


def test_balanced_group_split_singletons_exact():
    groups = {i: [f"m{i}"] for i in range(200)}
    out = balanced_group_split(groups, 200, FRACTIONS, seed=0)
    assert (len(out["train"]), len(out["val"]), len(out["test"])) == (160, 20, 20)


def test_scaffold_split_synthetic_no_shared_scaffold():
    scaf = ["", "", "", "c1ccccc1", "c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1", "c1ccsc1", "C1CC1"] * 3
    df = pd.DataFrame({"compound_id": [f"S-{i}" for i in range(len(scaf))], "scaffold": scaf})
    out = scaffold_split(df, FRACTIONS, 0)
    s = df.set_index("compound_id")["scaffold"]
    for a, b in [("train", "test"), ("train", "val"), ("val", "test")]:
        assert not set(s.loc[out[a]]) & set(s.loc[out[b]])


# --------------------------------------------------------------------------- Butina clustering
def test_butina_toy_example_matches_hand_calculation():
    """Chain x-y-z-w: sim(x,y)=.60, sim(y,z)=.43, sim(z,w)=.43, all other pairs < .4.

    y and z both have 3 neighbours (incl. self); ties go to the lower index, so y becomes the first
    centroid and takes {x, y, z}; w is left alone although it is a neighbour of z (sphere exclusion,
    not connected components).
    """
    x, y, z, w = _bitvect(range(0, 6)), _bitvect(range(0, 10)), _bitvect(range(4, 14)), _bitvect(range(8, 18))
    assert DataStructs.TanimotoSimilarity(x, y) == pytest.approx(0.6)
    assert DataStructs.TanimotoSimilarity(y, z) == pytest.approx(6 / 14)
    assert DataStructs.TanimotoSimilarity(x, z) < 0.4
    labels = butina_clusters([x, y, z, w], distance_cutoff=0.6)
    assert labels.tolist() == [0, 0, 0, 1]
    # a stricter cutoff (sim >= 0.5) breaks the y-z link: {x, y}, {z}, {w}
    assert butina_clusters([x, y, z, w], distance_cutoff=0.5).tolist() == [0, 0, 1, 2]


def test_butina_two_islands_and_singleton():
    a, b, c = _bitvect(range(0, 10)), _bitvect(range(0, 8)), _bitvect(list(range(0, 6)) + list(range(10, 14)))
    d, e, f = _bitvect(range(30, 40)), _bitvect(list(range(30, 36)) + list(range(40, 44))), _bitvect(range(50, 56))
    labels = butina_clusters([a, b, c, d, e, f], distance_cutoff=0.6)
    assert labels.tolist() == [0, 0, 0, 1, 1, 2]


def _partition(labels) -> set:
    groups = defaultdict(set)
    for i, lab in enumerate(labels):
        groups[int(lab)].add(i)
    return {frozenset(g) for g in groups.values()}


def test_butina_matches_rdkit_reference(aq):
    """Same partition as ``rdkit.ML.Cluster.Butina`` (RDKit breaks neighbour-count ties towards the
    *higher* index, solpred towards the lower one, so RDKit is run on the reversed list)."""
    smiles = aq["smiles"].iloc[:400].tolist()
    fps = morgan_bits(smiles)
    ours = _partition(butina_clusters(fps, 0.6))
    rev = fps[::-1]
    n = len(rev)
    dists = []
    for i in range(1, n):
        sims = DataStructs.BulkTanimotoSimilarity(rev[i], rev[:i])
        dists.extend(1.0 - s for s in sims)
    ref = Butina.ClusterData(dists, n, 0.6, isDistData=True)
    ref_part = {frozenset(n - 1 - j for j in cl) for cl in ref}
    assert ours == ref_part
    # every member is within the cutoff of the cluster's first (centroid) element in RDKit's output
    for cl in ref:
        cen = rev[cl[0]]
        assert all(DataStructs.TanimotoSimilarity(cen, rev[j]) >= 0.4 - 1e-12 for j in cl)


def test_max_similarity_bruteforce():
    rng = np.random.RandomState(0)
    q = [_bitvect(rng.choice(64, 12, replace=False)) for _ in range(5)]
    r = [_bitvect(rng.choice(64, 12, replace=False)) for _ in range(9)]
    brute = np.array([[DataStructs.TanimotoSimilarity(a, b) for b in r] for a in q])
    np.testing.assert_allclose(max_similarity(q, r), brute.max(1))
    np.testing.assert_allclose(max_similarity(q, r, k=3), np.sort(brute, 1)[:, -3:].mean(1))


def test_morgan_bits_deterministic():
    a, b = morgan_bits(["CCO", "c1ccccc1"]), morgan_bits(["CCO", "c1ccccc1"])
    assert all(x == y for x, y in zip(a, b))
    assert a[0].GetNumBits() == 2048


# --------------------------------------------------------------------------- stored split files
@pytest.fixture(scope="module")
def stored(aq):
    require(SPLITS / "aqsoldb" / "random" / "seed0.json")
    out = {(t, s): load_split(t, s) for t in GROUPED for s in SEEDS}
    for src in LOSO_SOURCES:
        out[("loso", src)] = load_split("loso", source=src)
    for name in ("aqsoldb_to_esol", "esol_to_aqsoldb"):
        out[("external", name)] = load_split("external", name=name)
    return out


@pytest.mark.parametrize("stype", GROUPED)
@pytest.mark.parametrize("seed", SEEDS)
def test_stored_splits_disjoint_complete(stored, aq, stype, seed):
    sp = stored[(stype, seed)]
    _assert_partition(sp, set(aq["compound_id"]))
    n = len(aq)
    assert abs(len(sp["test"]) - 0.1 * n) <= 0.01 * n
    assert abs(len(sp["val"]) - 0.1 * n) <= 0.01 * n
    assert sp["meta"]["split_type"] == stype and sp["meta"]["seed"] == seed


@pytest.mark.parametrize("seed", SEEDS)
def test_stored_splits_reproducible(stored, aq, butina_labels, cfg, seed):
    fr = cfg["splits"]["fractions"]
    ids = aq["compound_id"].tolist()
    strip = lambda d: {k: list(map(str, d[k])) for k in SPLIT_NAMES}  # noqa: E731
    assert strip(random_split(ids, fr, seed)) == strip(stored[("random", seed)])
    assert strip(scaffold_split(aq, fr, seed)) == strip(stored[("scaffold", seed)])
    labels = butina_labels.loc[aq["compound_id"]].to_numpy()
    assert strip(cluster_split(aq, labels, fr, seed)) == strip(stored[("cluster", seed)])


@pytest.mark.parametrize("stype", GROUPED)
def test_stored_seeds_differ(stored, stype):
    tests = [frozenset(stored[(stype, s)]["test"]) for s in SEEDS]
    assert len(set(tests)) == len(SEEDS)


@pytest.mark.parametrize("seed", SEEDS)
def test_scaffold_split_shares_no_scaffold(stored, aq, seed):
    sp = stored[("scaffold", seed)]
    scaf = aq.set_index("compound_id")["scaffold"]
    tr, va, te = (set(scaf.loc[sp[k]]) for k in SPLIT_NAMES)
    assert not tr & te and not tr & va and not va & te
    assert "" in tr                                                 # acyclic block sits in train


@pytest.mark.parametrize("seed", SEEDS)
def test_cluster_split_shares_no_cluster(stored, butina_labels, seed):
    sp = stored[("cluster", seed)]
    tr, va, te = (set(butina_labels.loc[sp[k]]) for k in SPLIT_NAMES)
    assert not tr & te and not tr & va and not va & te


def test_butina_cluster_file_aligned(aq, butina_labels):
    assert butina_labels.index.tolist() == aq["compound_id"].tolist()
    assert butina_labels.min() == 0
    assert butina_labels.nunique() == butina_labels.max() + 1


@pytest.mark.parametrize("src", LOSO_SOURCES)
def test_loso_holds_out_exactly_one_source(stored, aq, src):
    sp = stored[("loso", src)]
    source = aq.set_index("compound_id")["source"]
    _assert_partition(sp, set(aq["compound_id"]))
    assert set(sp["test"]) == set(aq.loc[aq["source"] == src, "compound_id"])
    assert not (source.loc[sp["train"] + sp["val"]] == src).any()
    assert sp["meta"]["held_out_source"] == src


def test_external_aqsoldb_to_esol_no_structure_overlap(stored, aq, es):
    sp = stored[("external", "aqsoldb_to_esol")]
    assert sp["train_dataset"] == "aqsoldb" and sp["test_dataset"] == "esol"
    assert set(sp["test"]) == set(es["compound_id"])
    trval = sp["train"] + sp["val"]
    assert not set(sp["train"]) & set(sp["val"])
    assert set(trval) == set(aq.loc[~aq["in_esol"], "compound_id"])
    s_aq = aq.set_index("compound_id")["smiles_2d"]
    s_es = es.set_index("compound_id")["smiles_2d"]
    assert not set(s_aq.loc[trval]) & set(s_es.loc[sp["test"]])


def test_external_esol_to_aqsoldb_no_structure_overlap(stored, aq, es):
    sp = stored[("external", "esol_to_aqsoldb")]
    assert sp["train_dataset"] == "esol" and sp["test_dataset"] == "aqsoldb"
    trval = sp["train"] + sp["val"]
    assert set(trval) == set(es["compound_id"]) and not set(sp["train"]) & set(sp["val"])
    s_aq = aq.set_index("compound_id")["smiles_2d"]
    s_es = es.set_index("compound_id")["smiles_2d"]
    assert set(sp["test"]) <= set(aq["compound_id"])
    assert not set(s_es.loc[trval]) & set(s_aq.loc[sp["test"]])
