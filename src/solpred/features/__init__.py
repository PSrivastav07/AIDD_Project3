"""Molecular featurisations with an on-disk cache.

Kinds
-----
``rdkit2d``        all RDKit 2D physicochemical descriptors (~210; ``Ipc`` is log-transformed,
                   non-finite values become NaN - imputation happens inside model pipelines)
``morgan_counts``  ECFP4-like Morgan count fingerprint (radius 2, 2048 folded bits)
``morgan_bits``    binary Morgan fingerprint (radius 2, 2048 bits)
``maccs``          166 MACCS structural keys
``esol4``          Delaney's four ESOL descriptors: cLogP (Crippen), MW, rotatable bonds, aromatic proportion
``a+b``            horizontal concatenation, e.g. ``rdkit2d+morgan_counts``
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator

from solpred.config import load_config, path

RDLogger.DisableLog("rdApp.*")

DESCRIPTOR_NAMES = [n for n, _ in Descriptors._descList]
_DESC_FUNCS = dict(Descriptors._descList)
BASE_KINDS = ("rdkit2d", "morgan_counts", "morgan_bits", "maccs", "esol4")


def _mols(smiles: list[str]) -> list:
    mols = [Chem.MolFromSmiles(s) for s in smiles]
    bad = [s for s, m in zip(smiles, mols) if m is None]
    if bad:
        raise ValueError(f"{len(bad)} SMILES could not be parsed, e.g. {bad[:3]}")
    return mols


def rdkit2d(mols) -> tuple[np.ndarray, list[str]]:
    X = np.full((len(mols), len(DESCRIPTOR_NAMES)), np.nan, dtype=np.float64)
    for i, m in enumerate(mols):
        for j, name in enumerate(DESCRIPTOR_NAMES):
            try:
                X[i, j] = _DESC_FUNCS[name](m)
            except Exception:  # noqa: BLE001 - a few descriptors fail on exotic molecules
                pass
    if "Ipc" in DESCRIPTOR_NAMES:
        j = DESCRIPTOR_NAMES.index("Ipc")
        X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    X[~np.isfinite(X)] = np.nan
    X = np.clip(X, -1e8, 1e8)
    return X.astype(np.float32), list(DESCRIPTOR_NAMES)


def morgan(mols, radius: int = 2, nbits: int = 2048, counts: bool = True) -> tuple[np.ndarray, list[str]]:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    fn = gen.GetCountFingerprintAsNumPy if counts else gen.GetFingerprintAsNumPy
    X = np.stack([fn(m) for m in mols]).astype(np.float32)
    tag = "mc" if counts else "mb"
    return X, [f"{tag}{i}" for i in range(nbits)]


def maccs(mols) -> tuple[np.ndarray, list[str]]:
    X = np.stack([np.array(list(MACCSkeys.GenMACCSKeys(m)), dtype=np.float32) for m in mols])
    return X[:, 1:], [f"maccs{i}" for i in range(1, 167)]


def esol4(mols) -> tuple[np.ndarray, list[str]]:
    rows = []
    for m in mols:
        heavy = m.GetNumHeavyAtoms()
        arom = sum(1 for a in m.GetAtoms() if a.GetIsAromatic())
        rows.append([Crippen.MolLogP(m), Descriptors.MolWt(m), Descriptors.NumRotatableBonds(m),
                     arom / heavy if heavy else 0.0])
    return np.asarray(rows, dtype=np.float32), ["clogp", "mw", "rotatable_bonds", "aromatic_proportion"]


def featurize_smiles(smiles: list[str], kind: str, cfg: dict | None = None) -> tuple[np.ndarray, list[str]]:
    """Featurise a list of (standardised) SMILES without caching."""
    cfg = cfg or load_config()
    mols = _mols(list(smiles))
    return _featurize_mols(mols, kind, cfg)


def _featurize_mols(mols, kind: str, cfg: dict) -> tuple[np.ndarray, list[str]]:
    parts = kind.split("+")
    if len(parts) > 1:
        blocks = [_featurize_mols(mols, p, cfg) for p in parts]
        return np.hstack([b[0] for b in blocks]), sum((b[1] for b in blocks), [])
    mc = cfg["features"]["morgan"]
    if kind == "rdkit2d":
        return rdkit2d(mols)
    if kind == "morgan_counts":
        return morgan(mols, mc["radius"], mc["nbits"], counts=True)
    if kind == "morgan_bits":
        return morgan(mols, mc["radius"], mc["nbits"], counts=False)
    if kind == "maccs":
        return maccs(mols)
    if kind == "esol4":
        return esol4(mols)
    raise ValueError(f"unknown feature kind {kind!r}")


def load_dataset(dataset: str, cfg: dict | None = None) -> pd.DataFrame:
    """Load a curated dataset (``aqsoldb`` or ``esol``); empty strings are preserved (acyclic scaffold)."""
    cfg = cfg or load_config()
    df = pd.read_csv(path("processed", cfg) / f"{dataset}_clean.csv", keep_default_na=False, na_values=[""])
    df["scaffold"] = df["scaffold"].fillna("")
    df["compound_id"] = df["compound_id"].astype(str)
    return df


def load_or_compute(dataset: str, kind: str, cfg: dict | None = None) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return ``(X, compound_ids, feature_names)`` for a curated dataset, cached in data/features.

    Composite kinds are assembled from cached base blocks. Row order = order in ``{dataset}_clean.csv``.
    """
    cfg = cfg or load_config()
    parts = kind.split("+")
    if len(parts) > 1:
        blocks = [load_or_compute(dataset, p, cfg) for p in parts]
        ids = blocks[0][1]
        for b in blocks[1:]:
            assert np.array_equal(b[1], ids)
        return np.hstack([b[0] for b in blocks]), ids, sum((b[2] for b in blocks), [])
    cache = path("features", cfg) / f"{dataset}__{kind}.npz"
    df = load_dataset(dataset, cfg)
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if np.array_equal(z["ids"], df["compound_id"].to_numpy(dtype=str)):
            return z["X"], z["ids"], list(z["names"])
    X, names = featurize_smiles(df["smiles"].tolist(), kind, cfg)
    ids = df["compound_id"].to_numpy(dtype=str)
    np.savez_compressed(cache, X=X, ids=ids, names=np.asarray(names))
    return X, ids, names


def subset(X: np.ndarray, ids: np.ndarray, wanted: list[str]) -> np.ndarray:
    """Rows of ``X`` for the requested compound ids (in the requested order)."""
    pos = {c: i for i, c in enumerate(ids)}
    return X[[pos[w] for w in wanted]]
