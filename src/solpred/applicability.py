"""Applicability domain (AD) based on Tanimoto similarity to the training set.

A query compound is *in domain* when its maximum ECFP4 Tanimoto similarity to any training compound
is at least ``sim_threshold``. The nearest training neighbour (id, SMILES, label) is reported so that a
user can sanity-check a prediction against measured data.

Stable API (used by the production predictor)::

    ad = ApplicabilityDomain(k=5, sim_threshold=0.35).fit(train_smiles, ids=train_ids, labels=train_logS)
    df = ad.score(query_smiles)   # ad_maxsim, ad_meansim_k, ad_in_domain, nearest_train_*
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


class ApplicabilityDomain:
    def __init__(self, k: int = 5, sim_threshold: float = 0.35, radius: int = 2, nbits: int = 2048):
        self.k = k
        self.sim_threshold = sim_threshold
        self.radius = radius
        self.nbits = nbits

    def _fps(self, smiles: list[str]) -> list:
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=self.radius, fpSize=self.nbits)
        out = []
        for s in smiles:
            m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
            out.append(gen.GetFingerprint(m) if m is not None else None)
        return out

    def fit(self, smiles: list[str], ids: list[str] | None = None, labels=None) -> "ApplicabilityDomain":
        fps = self._fps(list(smiles))
        keep = [i for i, f in enumerate(fps) if f is not None]
        self.train_fps_ = [fps[i] for i in keep]
        self.train_smiles_ = [list(smiles)[i] for i in keep]
        self.train_ids_ = [list(ids)[i] for i in keep] if ids is not None else [str(i) for i in keep]
        self.train_labels_ = np.asarray(labels, dtype=float)[keep] if labels is not None else None
        return self

    def similarities(self, fp) -> np.ndarray:
        return np.asarray(DataStructs.BulkTanimotoSimilarity(fp, self.train_fps_))

    def score(self, smiles: list[str]) -> pd.DataFrame:
        rows = []
        for fp in self._fps(list(smiles)):
            if fp is None:
                rows.append({"ad_maxsim": np.nan, "ad_meansim_k": np.nan, "ad_in_domain": False,
                             "nearest_train_id": None, "nearest_train_smiles": None, "nearest_train_logS": np.nan})
                continue
            sims = self.similarities(fp)
            top = np.argsort(sims)[::-1][: self.k]
            j = int(top[0])
            rows.append({
                "ad_maxsim": float(sims[j]),
                "ad_meansim_k": float(sims[top].mean()),
                "ad_in_domain": bool(sims[j] >= self.sim_threshold),
                "nearest_train_id": self.train_ids_[j],
                "nearest_train_smiles": self.train_smiles_[j],
                "nearest_train_logS": float(self.train_labels_[j]) if self.train_labels_ is not None else np.nan,
            })
        return pd.DataFrame(rows)

    # RDKit bit vectors are stored as binary strings so the object pickles portably
    def __getstate__(self):
        state = self.__dict__.copy()
        if "train_fps_" in state:
            state["train_fps_"] = [fp.ToBinary() for fp in state["train_fps_"]]
        return state

    def __setstate__(self, state):
        if "train_fps_" in state:
            state["train_fps_"] = [DataStructs.ExplicitBitVect(b) for b in state["train_fps_"]]
        self.__dict__.update(state)


# =========================================================================== extensions (AD study)
# Everything below is additive: ApplicabilityDomain above is unchanged.
def tanimoto_neighbours(query_fps: list, ref_fps: list, k: int = 5) -> pd.DataFrame:
    """Max similarity, mean similarity of the ``k`` nearest neighbours and the nearest-neighbour index of
    every query fingerprint against a reference set (ECFP4 bit vectors)."""
    k = min(k, len(ref_fps))
    rows = np.empty((len(query_fps), 3), dtype=float)
    for i, fp in enumerate(query_fps):
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fp, ref_fps))
        top = np.partition(sims, -k)[-k:]
        j = int(np.argmax(sims))
        rows[i] = (sims[j], top.mean(), j)
    return pd.DataFrame({"maxsim": rows[:, 0], f"meansim_{k}": rows[:, 1], "nn_index": rows[:, 2].astype(int)})


class DescriptorAD:
    """Descriptor-space applicability domain: Mahalanobis distance in a PCA subspace.

    The training descriptors are median-imputed and standardised (training mean / std; constant columns
    dropped), projected on the first ``n_components`` principal components, and every query is scored by
    its Mahalanobis distance ``d_M = sqrt(sum_k t_k^2 / lambda_k)`` in that space. The leverage
    ``h = 1/n + d_M^2 / (n - 1)`` (hat value of a linear model on the PC scores) is reported as well; the
    query is *in domain* when ``h <= h* = 3 (p + 1) / n`` (the usual Williams-plot warning leverage, p =
    ``n_components``). ``ad_mahal_pct`` is the percentile of ``d_M`` within the training distribution.

    Usage::

        dad = DescriptorAD(n_components=20).fit(X_train)
        df = dad.score(X_query)   # ad_mahalanobis, ad_leverage, ad_mahal_pct, ad_desc_in_domain
    """

    def __init__(self, n_components: int = 20, leverage_factor: float = 3.0):
        self.n_components = n_components
        self.leverage_factor = leverage_factor

    def _standardise(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=float)[:, self.keep_]
        X = np.where(np.isfinite(X), X, self.median_)
        return (X - self.mean_) / self.std_

    def fit(self, X) -> "DescriptorAD":
        X = np.asarray(X, dtype=float)
        X = np.where(np.isfinite(X), X, np.nan)
        median = np.nanmedian(X, axis=0)
        Xi = np.where(np.isnan(X), median, X)
        std = Xi.std(axis=0)
        self.keep_ = np.where(np.isfinite(median) & (std > 1e-12))[0]
        self.median_, self.mean_, self.std_ = median[self.keep_], Xi[:, self.keep_].mean(0), std[self.keep_]
        Z = self._standardise(X)
        _, s, vt = np.linalg.svd(Z, full_matrices=False)
        p = min(self.n_components, len(s))
        self.components_ = vt[:p]
        self.eigenvalues_ = s[:p] ** 2 / (len(Z) - 1)
        self.explained_variance_ratio_ = self.eigenvalues_ / ((s ** 2).sum() / (len(Z) - 1))
        self.n_train_ = len(Z)
        self.h_star_ = self.leverage_factor * (p + 1) / self.n_train_
        self.train_dist_ = np.sort(self._distance(Z))
        return self

    def _distance(self, Z: np.ndarray) -> np.ndarray:
        T = Z @ self.components_.T
        return np.sqrt(np.sum(T ** 2 / self.eigenvalues_, axis=1))

    def score(self, X) -> pd.DataFrame:
        d = self._distance(self._standardise(X))
        h = 1.0 / self.n_train_ + d ** 2 / (self.n_train_ - 1)
        pct = 100.0 * np.searchsorted(self.train_dist_, d, side="right") / len(self.train_dist_)
        return pd.DataFrame({"ad_mahalanobis": d, "ad_leverage": h, "ad_mahal_pct": pct,
                             "ad_desc_in_domain": h <= self.h_star_})
