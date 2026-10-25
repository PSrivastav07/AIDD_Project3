"""Molecular graphs for directed message-passing neural networks (Chemprop-style D-MPNN).

Every molecule becomes a graph of heavy atoms (hydrogens implicit) with *directed* edges: each
chemical bond u-v yields two consecutive edges, ``2k: u->v`` and ``2k+1: v->u``, so the reverse of
edge ``j`` is ``j ^ 1`` (equivalently: swap the two rows of every edge pair). Disconnected
components (salts, counter-ions such as Na+ or Br-) are kept in the same graph; an atom without
bonds simply has no edges and is still represented at the atom level.

Atom features (``ATOM_FDIM`` = 52; one-hot blocks end with an "other" slot)
    element (C N O S F Cl Br I P B Si Se H Li Na K Mg Ca | other)    19
    degree (0-5 | other)                                                7
    formal charge (-2..+2 | other)                                      6
    chiral tag (unspecified, CW, CCW, other-chirality | other)          5
    total number of hydrogens (0-4 | other)                             6
    hybridisation (S, SP, SP2, SP3, SP3D, SP3D2 | other/unspecified)    7
    aromatic flag                                                       1
    atomic mass / 100                                                   1
Bond features (``BOND_FDIM`` = 13)
    bond type (single, double, triple, aromatic)                        4
    conjugated, in ring                                                 2
    stereo (none, any, Z, E, cis, trans | other)                        7

Storage and batching (built for CPU throughput)
    ``GraphStore`` holds a whole dataset as flat arrays - all atom rows, all directed-edge rows,
    molecule-local edge endpoints and CSR-style pointers (``atom_ptr``, ``edge_ptr``). It is
    featurised once with RDKit and cached as an ``.npz`` keyed by a hash of the SMILES list and
    the feature definition. ``GraphStore.batch(mols)`` builds the index arrays of a mini-batch
    (a disconnected union of the selected molecules) with a handful of vectorised NumPy
    operations - no per-molecule Python work - so it can be called inside the training loop.
    Because every molecule contributes an even number of edges, edge pairs stay aligned in a
    batch (reverse of edge j is j ^ 1). The batch also carries the non-zeros of the
    edge-to-edge message operator K (``message_pairs``): K[j, k] = 1 when edge k ends at the
    start atom of edge j and k is not the reverse of j, i.e. the D-MPNN neighbourhood sum with
    reverse-edge subtraction expressed as one sparse matrix.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "B", "Si", "Se", "H", "Li", "Na", "K", "Mg", "Ca"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-2, -1, 0, 1, 2]
CHIRAL_TAGS = [Chem.ChiralType.CHI_UNSPECIFIED, Chem.ChiralType.CHI_TETRAHEDRAL_CW,
               Chem.ChiralType.CHI_TETRAHEDRAL_CCW, Chem.ChiralType.CHI_OTHER]
NUM_HS = [0, 1, 2, 3, 4]
HYBRIDISATIONS = [Chem.HybridizationType.S, Chem.HybridizationType.SP, Chem.HybridizationType.SP2,
                  Chem.HybridizationType.SP3, Chem.HybridizationType.SP3D, Chem.HybridizationType.SP3D2]
BOND_TYPES = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE, Chem.BondType.AROMATIC]
BOND_STEREO = [Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY, Chem.BondStereo.STEREOZ,
               Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOTRANS]

ATOM_FDIM = (len(ELEMENTS) + 1) + (len(DEGREES) + 1) + (len(FORMAL_CHARGES) + 1) + (len(CHIRAL_TAGS) + 1) \
    + (len(NUM_HS) + 1) + (len(HYBRIDISATIONS) + 1) + 2
BOND_FDIM = len(BOND_TYPES) + 2 + (len(BOND_STEREO) + 1)
FEATURE_VERSION = "graph-v1"  # bump when the feature definition above changes (invalidates caches)


# --------------------------------------------------------------------------- RDKit featurisation
def one_hot(value, choices: list) -> list[float]:
    """One-hot encoding of ``value`` over ``choices`` plus a trailing "other" slot."""
    out = [0.0] * (len(choices) + 1)
    out[choices.index(value) if value in choices else len(choices)] = 1.0
    return out


def atom_features(atom: Chem.Atom) -> list[float]:
    """``ATOM_FDIM`` features of one heavy atom."""
    return (one_hot(atom.GetSymbol(), ELEMENTS)
            + one_hot(atom.GetTotalDegree(), DEGREES)
            + one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
            + one_hot(atom.GetChiralTag(), CHIRAL_TAGS)
            + one_hot(atom.GetTotalNumHs(), NUM_HS)
            + one_hot(atom.GetHybridization(), HYBRIDISATIONS)
            + [1.0 if atom.GetIsAromatic() else 0.0, atom.GetMass() / 100.0])


def bond_features(bond: Chem.Bond) -> list[float]:
    """``BOND_FDIM`` features of one bond (shared by its two directed edges)."""
    bt = bond.GetBondType()
    return ([1.0 if bt == t else 0.0 for t in BOND_TYPES]
            + [1.0 if bond.GetIsConjugated() else 0.0, 1.0 if bond.IsInRing() else 0.0]
            + one_hot(bond.GetStereo(), BOND_STEREO))


def mol_arrays(mol: Chem.Mol) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(x [n_atoms, ATOM_FDIM], e [n_edges, BOND_FDIM], src, dst)`` with molecule-local endpoints."""
    x = np.asarray([atom_features(a) for a in mol.GetAtoms()], dtype=np.float32).reshape(-1, ATOM_FDIM)
    feats, src, dst = [], [], []
    for b in mol.GetBonds():
        u, v = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        f = bond_features(b)
        feats += [f, f]
        src += [u, v]
        dst += [v, u]
    n_e = len(src)
    return (x, np.asarray(feats, dtype=np.float32).reshape(n_e, BOND_FDIM),
            np.asarray(src, dtype=np.int32), np.asarray(dst, dtype=np.int32))


# --------------------------------------------------------------------------- batching helpers
def ragged_arange(starts: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Concatenation of ``arange(s, s + l)`` for every (s, l) pair, without a Python loop."""
    lengths = np.asarray(lengths, dtype=np.int64)
    total = int(lengths.sum())
    excl = np.cumsum(lengths) - lengths
    return np.repeat(np.asarray(starts, dtype=np.int64) - excl, lengths) + np.arange(total, dtype=np.int64)


def message_pairs(src: np.ndarray, dst: np.ndarray, n_atoms: int) -> tuple[np.ndarray, np.ndarray]:
    """Non-zeros ``(rows, cols)`` of the D-MPNN edge-to-edge message operator K.

    ``K[j, k] = 1`` iff edge k ends where edge j starts (``dst[k] == src[j]``) and k is not the
    reverse of j (``k != j ^ 1``), so ``(K h)_j = sum_{k in N(u)} h_ku - h_vu`` for ``j = u->v``.
    Rows are returned sorted (CSR order).
    """
    indeg = np.bincount(dst, minlength=n_atoms)
    in_ptr = np.concatenate([[0], np.cumsum(indeg)])
    in_edges = np.argsort(dst, kind="stable")
    cnt = indeg[src]
    rows = np.repeat(np.arange(len(src), dtype=np.int64), cnt)
    cols = in_edges[ragged_arange(in_ptr[src], cnt)]
    keep = cols != (rows ^ 1)
    return rows[keep], cols[keep]


@dataclass
class BatchIndex:
    """Index arrays of one mini-batch (a disconnected union of ``n_mols`` molecules).

    ``atom_idx`` / ``edge_idx`` select rows of the store's ``x`` / ``e``; ``src`` / ``dst`` are
    batch-local atom indices of every directed edge; ``atom2mol`` maps atoms to batch molecules;
    ``msg_rows`` / ``msg_cols`` are the non-zeros of the edge-to-edge message operator
    (``message_pairs``). Edges come in aligned (u->v, v->u) pairs, so ``rev(j) = j ^ 1``.
    """
    atom_idx: np.ndarray
    edge_idx: np.ndarray
    src: np.ndarray
    dst: np.ndarray
    atom2mol: np.ndarray
    msg_rows: np.ndarray
    msg_cols: np.ndarray
    n_mols: int

    @property
    def n_atoms(self) -> int:
        return len(self.atom_idx)

    @property
    def n_edges(self) -> int:
        return len(self.edge_idx)


@dataclass
class GraphStore:
    """All molecular graphs of a dataset as flat arrays with CSR pointers (see module docstring)."""
    ids: np.ndarray       # [N] compound ids (str)
    x: np.ndarray         # [A, ATOM_FDIM] float32, atoms of molecule i in rows atom_ptr[i]:atom_ptr[i+1]
    e: np.ndarray         # [E, BOND_FDIM] float32, directed edges of molecule i in edge_ptr[i]:edge_ptr[i+1]
    src: np.ndarray       # [E] int32 molecule-local source atom
    dst: np.ndarray       # [E] int32 molecule-local target atom
    atom_ptr: np.ndarray  # [N+1] int64
    edge_ptr: np.ndarray  # [N+1] int64

    def __post_init__(self) -> None:
        self._pos = {str(c): i for i, c in enumerate(self.ids)}
        self.n_atoms_per_mol = np.diff(self.atom_ptr)
        self.n_edges_per_mol = np.diff(self.edge_ptr)

    def __len__(self) -> int:
        return len(self.ids)

    def positions(self, ids) -> np.ndarray:
        """Row positions of compound ``ids`` (KeyError for unknown ids)."""
        return np.fromiter((self._pos[str(i)] for i in ids), dtype=np.int64, count=len(ids))

    def batch(self, mols: np.ndarray) -> BatchIndex:
        """Vectorised collation of the molecules at positions ``mols`` into one disconnected graph."""
        mols = np.asarray(mols, dtype=np.int64)
        na = self.n_atoms_per_mol[mols]
        ne = self.n_edges_per_mol[mols]
        atom_idx = ragged_arange(self.atom_ptr[mols], na)
        edge_idx = ragged_arange(self.edge_ptr[mols], ne)
        a_off = np.repeat(np.cumsum(na) - na, ne)
        src = self.src[edge_idx].astype(np.int64) + a_off
        dst = self.dst[edge_idx].astype(np.int64) + a_off
        rows, cols = message_pairs(src, dst, len(atom_idx))
        return BatchIndex(atom_idx=atom_idx, edge_idx=edge_idx, src=src, dst=dst,
                          atom2mol=np.repeat(np.arange(len(mols), dtype=np.int64), na),
                          msg_rows=rows, msg_cols=cols, n_mols=len(mols))

    def edge_src_global(self) -> np.ndarray:
        """Row of ``x`` holding the source atom of every directed edge (dataset-wide index)."""
        mol_of_edge = np.repeat(np.arange(len(self), dtype=np.int64), self.n_edges_per_mol)
        return self.atom_ptr[mol_of_edge] + self.src.astype(np.int64)

    # ---------------------------------------------------------------- persistence
    def save(self, file: Path, key: str) -> None:
        """Atomic ``.npz`` write (safe when two processes build the cache simultaneously)."""
        file = Path(file)
        file.parent.mkdir(parents=True, exist_ok=True)
        tmp = file.with_name(f"{file.stem}.{os.getpid()}.tmp.npz")
        np.savez(tmp, ids=self.ids.astype(str), x=self.x, e=self.e, src=self.src, dst=self.dst,
                 atom_ptr=self.atom_ptr, edge_ptr=self.edge_ptr, key=np.array(key))
        os.replace(tmp, file)

    @classmethod
    def load(cls, file: Path, key: str | None = None) -> "GraphStore | None":
        """Load a cached store; ``None`` if missing or built from different SMILES / features."""
        file = Path(file)
        if not file.exists():
            return None
        with np.load(file, allow_pickle=False) as z:
            if key is not None and str(z["key"]) != key:
                return None
            return cls(ids=z["ids"].astype(str), x=z["x"], e=z["e"], src=z["src"], dst=z["dst"],
                       atom_ptr=z["atom_ptr"], edge_ptr=z["edge_ptr"])


def build_store(ids: list[str], smiles: list[str]) -> GraphStore:
    """Featurise ``smiles`` with RDKit into a ``GraphStore`` (raises on unparsable SMILES)."""
    xs, es, srcs, dsts = [], [], [], []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            raise ValueError(f"cannot parse SMILES {s!r}")
        x, e, src, dst = mol_arrays(m)
        xs.append(x), es.append(e), srcs.append(src), dsts.append(dst)
    ptr = lambda arrs: np.concatenate([[0], np.cumsum([len(a) for a in arrs])]).astype(np.int64)  # noqa: E731
    return GraphStore(ids=np.asarray(ids, dtype=str), x=np.concatenate(xs), e=np.concatenate(es),
                      src=np.concatenate(srcs), dst=np.concatenate(dsts), atom_ptr=ptr(xs), edge_ptr=ptr(es))


def cache_key(ids: list[str], smiles: list[str]) -> str:
    """Hash of the feature definition and the (id, SMILES) list - a stale cache is rebuilt."""
    h = hashlib.sha256(f"{FEATURE_VERSION}|{ATOM_FDIM}|{BOND_FDIM}".encode())
    for i, s in zip(ids, smiles):
        h.update(f"{i}\t{s}\n".encode())
    return h.hexdigest()


def dataset_store(dataset: str, cache_dir: Path | None = None) -> GraphStore:
    """``GraphStore`` of a curated dataset, read from / written to ``cache_dir/<dataset>__graph.npz``."""
    from solpred.features import load_dataset

    df = load_dataset(dataset)
    ids, smiles = df["compound_id"].astype(str).tolist(), df["smiles"].tolist()
    key = cache_key(ids, smiles)
    file = Path(cache_dir) / f"{dataset}__graph.npz" if cache_dir is not None else None
    if file is not None:
        store = GraphStore.load(file, key)
        if store is not None:
            return store
    store = build_store(ids, smiles)
    if file is not None:
        store.save(file, key)
    return store
