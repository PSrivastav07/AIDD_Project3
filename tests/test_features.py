"""Tests for ``solpred.features`` (featurisers, cache, subset) and ``solpred.applicability``.

Featuriser tests use a handful of SMILES; cache tests run against a toy dataset written into a
temporary ``processed`` directory (``isolated_cfg``) so nothing in ``data/`` is touched. The
``*_real`` tests compare the stored ``data/features`` caches with fresh computations for a random
sample of rows (skipped when the caches are missing).
"""
from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from conftest import FEATURES, require
from solpred.applicability import ApplicabilityDomain
from solpred.features import (DESCRIPTOR_NAMES, featurize_smiles, load_dataset, load_or_compute,
                              subset)

SMILES = ["CCO", "c1ccccc1", "CC(=O)[O-].[Na+]", "Clc1ccc(Cl)cc1", "OC(=O)c1ccccc1O"]


# --------------------------------------------------------------------------- featurisers
@pytest.mark.parametrize("kind, width", [("rdkit2d", len(DESCRIPTOR_NAMES)), ("morgan_counts", 2048),
                                         ("morgan_bits", 2048), ("maccs", 166), ("esol4", 4)])
def test_shapes_and_dtype(kind, width, cfg):
    X, names = featurize_smiles(SMILES, kind, cfg)
    assert X.shape == (len(SMILES), width)
    assert len(names) == width and len(set(names)) == width
    assert X.dtype == np.float32


def test_rdkit2d_descriptor_count(cfg):
    X, names = featurize_smiles(["CCO"], "rdkit2d", cfg)
    assert names == DESCRIPTOR_NAMES
    assert len(names) >= 200


def test_determinism(cfg):
    for kind in ("rdkit2d", "morgan_counts", "maccs", "esol4"):
        a, _ = featurize_smiles(SMILES, kind, cfg)
        b, _ = featurize_smiles(list(reversed(SMILES)), kind, cfg)
        np.testing.assert_array_equal(a, b[::-1])


def test_rdkit2d_nan_handling(cfg):
    """Non-finite values become NaN (never inf); salts lose the BCUT2D block; Ipc is log-scaled."""
    X, names = featurize_smiles(SMILES + ["CCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCCC"], "rdkit2d", cfg)
    assert not np.isinf(X).any()
    assert np.nanmax(np.abs(X)) <= 1e8
    bcut = [j for j, n in enumerate(names) if n.startswith("BCUT2D")]
    assert np.isnan(X[2, bcut]).all()                       # sodium acetate: no BCUT parameters for Na+
    assert not np.isnan(X[[0, 1, 3, 4]]).any()             # neutral organics: complete rows
    ipc = names.index("Ipc")
    assert (X[:, ipc] >= 0).all() and X[-1, ipc] < 1e3      # log1p keeps huge Ipc values small


def test_morgan_bits_are_binarised_counts(cfg):
    c, _ = featurize_smiles(SMILES, "morgan_counts", cfg)
    b, _ = featurize_smiles(SMILES, "morgan_bits", cfg)
    np.testing.assert_array_equal(b, (c > 0).astype(np.float32))
    assert c[0].sum() > 0 and set(np.unique(b)) <= {0.0, 1.0}


def test_esol4_values(cfg):
    X, names = featurize_smiles(["c1ccccc1", "CCCCO"], "esol4", cfg)
    assert names == ["clogp", "mw", "rotatable_bonds", "aromatic_proportion"]
    np.testing.assert_allclose(X[0], [1.6866, 78.114, 0, 1.0], atol=1e-3)
    assert X[1, 2] == 2 and X[1, 3] == 0.0


def test_maccs_drops_unused_bit0(cfg):
    X, names = featurize_smiles(SMILES, "maccs", cfg)
    assert names[0] == "maccs1" and names[-1] == "maccs166"
    assert set(np.unique(X)) <= {0.0, 1.0}


def test_composite_kind_is_ordered_hstack(cfg):
    a, na = featurize_smiles(SMILES, "esol4", cfg)
    b, nb = featurize_smiles(SMILES, "maccs", cfg)
    ab, nab = featurize_smiles(SMILES, "esol4+maccs", cfg)
    np.testing.assert_array_equal(ab, np.hstack([a, b]))
    assert nab == na + nb
    ba, _ = featurize_smiles(SMILES, "maccs+esol4", cfg)
    np.testing.assert_array_equal(ba, np.hstack([b, a]))


def test_bad_input(cfg):
    with pytest.raises(ValueError, match="could not be parsed"):
        featurize_smiles(["CCO", "C1CC"], "esol4", cfg)
    with pytest.raises(ValueError, match="unknown feature kind"):
        featurize_smiles(["CCO"], "ecfp99", cfg)


# --------------------------------------------------------------------------- cache + subset
def _write_toy(cfg, smiles=SMILES) -> pd.DataFrame:
    from solpred.config import path

    df = pd.DataFrame({"compound_id": [f"T-{i}" for i in range(len(smiles))], "smiles": smiles,
                       "scaffold": ["", "c1ccccc1", "", "c1ccccc1", "c1ccccc1"][: len(smiles)],
                       "logS": np.linspace(-3, 0, len(smiles))})
    df.to_csv(path("processed", cfg) / "toy_clean.csv", index=False)
    return df


def test_load_or_compute_writes_and_reuses_cache(isolated_cfg):
    from solpred.config import path

    df = _write_toy(isolated_cfg)
    X1, ids1, names1 = load_or_compute("toy", "esol4", isolated_cfg)
    cache = path("features", isolated_cfg) / "toy__esol4.npz"
    assert cache.exists()
    assert ids1.tolist() == df["compound_id"].tolist()
    mtime = cache.stat().st_mtime_ns
    X2, ids2, names2 = load_or_compute("toy", "esol4", isolated_cfg)
    assert cache.stat().st_mtime_ns == mtime                 # served from cache
    np.testing.assert_array_equal(X1, X2)
    assert names1 == names2
    np.testing.assert_array_equal(X1, featurize_smiles(df["smiles"].tolist(), "esol4", isolated_cfg)[0])


def test_load_or_compute_recomputes_when_ids_change(isolated_cfg):
    _write_toy(isolated_cfg)
    load_or_compute("toy", "esol4", isolated_cfg)
    df = _write_toy(isolated_cfg, SMILES[:3])
    X, ids, _ = load_or_compute("toy", "esol4", isolated_cfg)
    assert X.shape[0] == 3 and ids.tolist() == df["compound_id"].tolist()


@pytest.mark.xfail(strict=True, reason="cache validity is checked on compound ids only: a structure change "
                                       "with unchanged ids silently returns stale features")
def test_known_cache_detects_structure_change(isolated_cfg):
    _write_toy(isolated_cfg)
    X_old, _, _ = load_or_compute("toy", "esol4", isolated_cfg)
    changed = ["CCCCCCCCO"] + SMILES[1:]
    _write_toy(isolated_cfg, changed)
    X_new, _, _ = load_or_compute("toy", "esol4", isolated_cfg)
    assert not np.array_equal(X_old[0], X_new[0])


def test_load_or_compute_composite(isolated_cfg):
    _write_toy(isolated_cfg)
    a, ids, na = load_or_compute("toy", "esol4", isolated_cfg)
    b, _, nb = load_or_compute("toy", "maccs", isolated_cfg)
    ab, ids_ab, nab = load_or_compute("toy", "esol4+maccs", isolated_cfg)
    np.testing.assert_array_equal(ab, np.hstack([a, b]))
    assert nab == na + nb and ids_ab.tolist() == ids.tolist()


def test_subset_ordering():
    X = np.arange(12, dtype=np.float32).reshape(6, 2)
    ids = np.array(["a", "b", "c", "d", "e", "f"])
    np.testing.assert_array_equal(subset(X, ids, ["d", "a", "d"]), X[[3, 0, 3]])
    assert subset(X, ids, []).shape == (0, 2)
    with pytest.raises(KeyError):
        subset(X, ids, ["zz"])


def test_load_dataset_keeps_empty_scaffold(isolated_cfg):
    _write_toy(isolated_cfg)
    df = load_dataset("toy", isolated_cfg)
    assert df.loc[0, "scaffold"] == ""                           # not NaN
    assert all(isinstance(c, str) for c in df["compound_id"])


# --------------------------------------------------------------------------- stored caches
@pytest.mark.parametrize("dataset", ["aqsoldb", "esol"])
@pytest.mark.parametrize("kind", ["rdkit2d", "morgan_counts", "maccs", "esol4"])
def test_stored_cache_matches_fresh_computation_real(dataset, kind, cfg):
    require(FEATURES / f"{dataset}__{kind}.npz")
    df = load_dataset(dataset)
    z = np.load(FEATURES / f"{dataset}__{kind}.npz", allow_pickle=False)
    assert z["ids"].tolist() == df["compound_id"].tolist()          # row order = curated file order
    assert z["X"].shape[0] == len(df)
    rows = np.random.RandomState(0).choice(len(df), 25, replace=False)
    fresh, names = featurize_smiles(df["smiles"].iloc[rows].tolist(), kind, cfg)
    assert list(z["names"]) == names
    np.testing.assert_allclose(z["X"][rows], fresh, rtol=1e-5, atol=1e-5, equal_nan=True)


# --------------------------------------------------------------------------- applicability domain
@pytest.fixture
def ad():
    train = ["CCO", "CCCO", "c1ccccc1", "Cc1ccccc1", "CC(=O)O"]
    return ApplicabilityDomain(k=2, sim_threshold=0.35).fit(train, ids=list("abcde"), labels=[0.1, -0.2, -1.6, -2.2, 1.0])


def test_ad_scores(ad):
    out = ad.score(["Cc1ccccc1", "CCCCCCCCCCCCCCCCCCCCCl", "not_a_smiles"])
    assert out.loc[0, "ad_maxsim"] == pytest.approx(1.0)
    assert bool(out.loc[0, "ad_in_domain"]) is True
    assert out.loc[0, "nearest_train_id"] == "d" and out.loc[0, "nearest_train_logS"] == pytest.approx(-2.2)
    assert out.loc[0, "ad_meansim_k"] <= out.loc[0, "ad_maxsim"]
    assert out.loc[1, "ad_maxsim"] < 0.35 and not out.loc[1, "ad_in_domain"]
    assert np.isnan(out.loc[2, "ad_maxsim"]) and not out.loc[2, "ad_in_domain"]
    assert pd.isna(out.loc[2, "nearest_train_id"])           # None (pandas may store it as NaN)


def test_ad_skips_invalid_training_smiles():
    ad = ApplicabilityDomain().fit(["CCO", "C1CC", "c1ccccc1"], ids=["x", "y", "z"], labels=[1.0, 2.0, 3.0])
    assert ad.train_ids_ == ["x", "z"]
    np.testing.assert_array_equal(ad.train_labels_, [1.0, 3.0])
    assert ad.score(["c1ccccc1"]).loc[0, "nearest_train_logS"] == 3.0


def test_ad_pickle_roundtrip(ad):
    q = ["CCCCO", "c1ccncc1", "CC(=O)OC"]
    clone = pickle.loads(pickle.dumps(ad))
    pd.testing.assert_frame_equal(ad.score(q), clone.score(q))
    assert clone.train_ids_ == ad.train_ids_
