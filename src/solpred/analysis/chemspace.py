"""Chemical-space maps, high-similarity / large-dlogS pairs and descriptor-level explainability.

Method
------
* **Chemical-space map.** UMAP on binary ECFP4 fingerprints (Morgan radius 2, 2048 bits) with the
  Jaccard metric (1 - Tanimoto), so distances in the embedding approximate the Tanimoto geometry used
  for the splits and the applicability domain. The embedding is fitted once on the union of AqSolDB
  and the ESOL compounds absent from AqSolDB; ESOL compounds that are also in AqSolDB (same
  ``smiles_2d``) reuse the AqSolDB coordinates.
* **High-similarity, large-dlogS pairs.** All AqSolDB pairs with ECFP4 Tanimoto >= ``sim_min`` and
  ``|dlogS| >= dlogs_min``. Such pairs bound the accuracy of any learner on binary ECFP4: when two
  identical/near-identical fingerprints carry labels 2 log units apart, at least one prediction must be
  wrong by >= 1 log unit. They are NOT generally activity cliffs, near-duplicates or label noise: in
  AqSolDB most of them are fingerprint-identical (Tanimoto 1.0) and/or homologues (bit-vector
  saturation, e.g. cyclosiloxanes); a count-based Morgan Tanimoto (:func:`count_tanimoto`) separates
  them. Only same-size pairs are candidates for label noise or true cliffs. Pairs are typed as salt vs free form (same ``parent_smiles``), homologues
  (heavy-atom counts differ by >= 2: binary ECFP4 barely sees chain length or ring size, so these are a
  representation limit rather than label noise) and same-size analogues (label-noise / activity-cliff
  suspects); fingerprint-identical pairs (Tanimoto 1) are flagged separately. The distribution of
  ``|dlogS|`` over all high-similarity pairs gives an empirical noise floor for nearest-neighbour
  reasoning on binary ECFP4.
* **Explainability.** TreeSHAP values of LightGBM models; features are aggregated into physically
  meaningful families (lipophilicity, size, polarity/H-bonding, charge, aromaticity/rigidity,
  topology, fragment counts, Morgan bits) so that the importance of "what drives solubility" can be
  discussed at the level of the general solubility equation (logS = 0.5 - logP - 0.01 (mp - 25)).
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


# --------------------------------------------------------------------------- fingerprints / UMAP
def fingerprint_matrix(smiles: list[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Binary ECFP matrix (n x nbits, bool)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    return np.stack([gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)).astype(bool) for s in smiles])


def umap_embedding(X: np.ndarray, n_neighbors: int = 30, min_dist: float = 0.1, random_state: int = 0,
                   metric: str = "jaccard") -> np.ndarray:
    """2-D UMAP embedding (seeded, hence single-threaded and reproducible)."""
    import umap

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric,
                        random_state=random_state, n_components=2, low_memory=True)
    return reducer.fit_transform(X)


# --------------------------------------------------------------------------- similar pairs
def similar_pairs(fps: list, sim_min: float = 0.9) -> pd.DataFrame:
    """All unordered pairs (i < j) with Tanimoto >= ``sim_min`` (indices into ``fps``)."""
    rows_i, rows_j, sims = [], [], []
    for i in range(len(fps) - 1):
        s = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:]))
        hit = np.nonzero(s >= sim_min)[0]
        rows_i.append(np.full(len(hit), i))
        rows_j.append(hit + i + 1)
        sims.append(s[hit])
    return pd.DataFrame({"i": np.concatenate(rows_i), "j": np.concatenate(rows_j), "sim": np.concatenate(sims)})


def count_tanimoto(smiles_a: list[str], smiles_b: list[str], radius: int = 2, nbits: int = 2048) -> np.ndarray:
    """Tanimoto on Morgan COUNT fingerprints (radius 2) for aligned SMILES pairs (NaN if unparsable)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    cache: dict = {}

    def fp(smi):
        if smi not in cache:
            m = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
            cache[smi] = gen.GetCountFingerprint(m) if m is not None else None
        return cache[smi]

    out = []
    for a, b in zip(smiles_a, smiles_b):
        fa, fb = fp(a), fp(b)
        out.append(DataStructs.TanimotoSimilarity(fa, fb) if fa is not None and fb is not None else np.nan)
    return np.asarray(out, dtype=float)


def annotate_pairs(pairs: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Attach both members' metadata and ``abs_dlogS``; ``df`` rows are indexed like the fingerprints."""
    cols = ["compound_id", "name", "smiles", "logS", "source", "reliability_group", "n_occurrences",
            "aqsol_sd", "parent_smiles", "is_multicomponent", "n_heavy_atoms"]
    cols = [c for c in cols if c in df.columns]
    a = df.iloc[pairs["i"].to_numpy()][cols].reset_index(drop=True).add_suffix("_1")
    b = df.iloc[pairs["j"].to_numpy()][cols].reset_index(drop=True).add_suffix("_2")
    out = pd.concat([pairs.reset_index(drop=True), a, b], axis=1)
    out["abs_dlogS"] = (out["logS_1"] - out["logS_2"]).abs()
    out["same_parent"] = out["parent_smiles_1"] == out["parent_smiles_2"]
    out["fp_identical"] = out["sim"] >= 0.9999
    out["same_source"] = out["source_1"] == out["source_2"]
    out["both_single_measurement"] = (out["reliability_group_1"] == "G1") & (out["reliability_group_2"] == "G1")
    if "n_heavy_atoms_1" in out:
        out["d_heavy_atoms"] = (out["n_heavy_atoms_1"] - out["n_heavy_atoms_2"]).abs()
    else:
        out["d_heavy_atoms"] = np.nan
    # ECFP4 bits (radius 2) barely change with chain length / ring size, so pairs that differ by >= 2
    # heavy atoms are mostly homologues whose solubility legitimately differs (a representation limit),
    # whereas only pairs of (almost) equal size with |dlogS| >= 2 are label-noise / true-cliff candidates.
    homologue = out["d_heavy_atoms"] >= 2
    out["pair_type"] = np.select(
        [out["same_parent"], homologue],
        ["salt vs parent (same parent)", "homologues (|d heavy atoms| >= 2)"],
        "same-size analogues (|d heavy atoms| <= 1)")
    return out


def nn_label_floor(pairs: pd.DataFrame, labels: np.ndarray, n: int) -> dict:
    """Error of predicting each compound by the label of its most similar partner (pairs >= sim_min).

    Only compounds that have at least one partner in ``pairs`` contribute. This is an empirical lower
    bound on what a purely similarity-based predictor can achieve in the high-similarity regime.
    """
    best_sim = np.full(n, -1.0)
    best_lab = np.full(n, np.nan)
    for i, j, s in pairs[["i", "j", "sim"]].itertuples(index=False):
        i, j = int(i), int(j)
        if s > best_sim[i]:
            best_sim[i], best_lab[i] = s, labels[j]
        if s > best_sim[j]:
            best_sim[j], best_lab[j] = s, labels[i]
    has = best_sim >= 0
    err = np.abs(labels[has] - best_lab[has])
    return {"n_compounds_with_partner": int(has.sum()), "MAE_nn_label": float(err.mean()),
            "median_abs_diff": float(np.median(err)), "frac_abs_diff_ge_1": float((err >= 1).mean()),
            "frac_abs_diff_ge_2": float((err >= 2).mean())}


# --------------------------------------------------------------------------- explainability
FAMILY_RULES: list[tuple[str, str]] = [
    ("lipophilicity", r"^(MolLogP|MolMR|SlogP_VSA\d+|SMR_VSA\d+|BCUT2D_LOGP\w+|BCUT2D_MR\w+)$"),
    ("polarity / H-bonding", r"^(TPSA|NumHDonors|NumHAcceptors|NHOHCount|NOCount|NumHeteroatoms|"
                             r"PEOE_VSA\d+|EState_VSA\d+|VSA_EState\d+|\w*EStateIndex)$"),
    ("charge", r"^(MaxPartialCharge|MinPartialCharge|MaxAbsPartialCharge|MinAbsPartialCharge|BCUT2D_CHG\w+)$"),
    ("size", r"^(MolWt|HeavyAtomMolWt|ExactMolWt|NumValenceElectrons|HeavyAtomCount|LabuteASA|"
             r"BCUT2D_MW\w+|Chi\d\w*)$"),
    ("aromaticity / rigidity", r"^(NumAromatic\w+|NumAliphatic\w+|NumSaturated\w+|RingCount|FractionCSP3|"
                               r"NumRotatableBonds|NumBridgeheadAtoms|NumSpiroAtoms|NumHeterocycles|"
                               r"HallKierAlpha|fr_benzene|fr_bicyclic)$"),
    ("topology / shape", r"^(BalabanJ|BertzCT|Ipc|AvgIpc|Kappa\d|Phi|FpDensityMorgan\d|SPS|qed|"
                         r"NumAtomStereoCenters|NumUnspecifiedAtomStereoCenters|NumAmideBonds|NumRadicalElectrons)$"),
    ("fragment counts", r"^fr_\w+$"),
    ("Morgan count bits", r"^m[cb]\d+$"),
]


def feature_family(name: str) -> str:
    """Physically motivated family of an RDKit descriptor / fingerprint bit name."""
    for fam, pat in FAMILY_RULES:
        if re.match(pat, name):
            return fam
    return "other"


def shap_importance(shap_values: np.ndarray, names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature with its family, sorted descending."""
    imp = np.abs(shap_values).mean(0)
    df = pd.DataFrame({"feature": names, "mean_abs_shap": imp,
                       "family": [feature_family(n) for n in names]})
    df["fraction"] = df["mean_abs_shap"] / df["mean_abs_shap"].sum()
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def family_importance(imp: pd.DataFrame) -> pd.DataFrame:
    g = imp.groupby("family").agg(mean_abs_shap=("mean_abs_shap", "sum"), n_features=("feature", "size"))
    g["fraction"] = g["mean_abs_shap"] / g["mean_abs_shap"].sum()
    return g.sort_values("mean_abs_shap", ascending=False).reset_index()


def morgan_bit_environment(bit: int, smiles: list[str], radius: int = 2, nbits: int = 2048,
                           max_mols: int = 2000) -> tuple[str, int]:
    """Most frequent substructure (SMILES of the atom environment) that sets a folded Morgan bit."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=nbits)
    counts: dict[str, int] = {}
    for s in smiles[:max_mols]:
        m = Chem.MolFromSmiles(s)
        ao = rdFingerprintGenerator.AdditionalOutput()
        ao.AllocateBitInfoMap()
        gen.GetCountFingerprint(m, additionalOutput=ao)  # folded: bit ids match the feature columns
        info = ao.GetBitInfoMap().get(bit)
        if not info:
            continue
        atom, rad = info[0]
        if rad == 0:
            a = m.GetAtomWithIdx(atom)
            frag = f"[{a.GetSymbol()}{'H' + str(a.GetTotalNumHs()) if a.GetTotalNumHs() else ''}]" \
                   + (" (arom)" if a.GetIsAromatic() else "")
        else:
            env = Chem.FindAtomEnvironmentOfRadiusN(m, rad, atom)
            amap: dict = {}
            sub = Chem.PathToSubmol(m, env, atomMap=amap)
            frag = Chem.MolToSmiles(sub, canonical=True) if sub.GetNumAtoms() else "?"
        counts[frag] = counts.get(frag, 0) + 1
    if not counts:
        return "", 0
    best = max(counts, key=counts.get)
    return best, counts[best]
