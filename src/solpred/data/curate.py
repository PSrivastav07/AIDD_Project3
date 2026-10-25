"""Structure standardisation, filtering and de-duplication for AqSolDB and ESOL.

Pipeline (each step is counted in ``curation_log.json``):

1. parse the SMILES with RDKit;
2. standardise with ``rdMolStandardize.Cleanup`` (sanitise, remove explicit H, disconnect metals,
   normalise functional groups, re-ionise). Salts are *kept as salts*: solubility of a salt is a
   property of the salt, so counter-ions are not stripped;
3. drop compounds without carbon (inorganic);
4. drop compounds with elements outside the organic set, except for monoatomic counter-ions
   (Li/Na/K/Mg/Ca) - this removes organometallics and heavy-metal complexes;
5. drop rows with a missing / non-finite label;
6. de-duplicate on the non-isomeric canonical SMILES (``smiles_2d``). 2D featurisations cannot
   distinguish stereoisomers, so stereoisomers would otherwise be identical inputs with
   different labels and could leak between train and test. Duplicates are merged with the median
   label and the within-group spread is recorded (and flagged when SD > threshold).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

from solpred.config import load_config, path, save_json

RDLogger.DisableLog("rdApp.*")

_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser(preferOrganic=True)


@dataclass
class StdResult:
    ok: bool
    reason: str = ""
    smiles: str = ""
    smiles_2d: str = ""
    inchikey: str = ""
    parent_smiles: str = ""
    scaffold: str = ""
    n_fragments: int = 0
    n_heavy_atoms: int = 0
    formal_charge_atoms: int = 0
    mol_wt: float = float("nan")
    elements: list = field(default_factory=list)


def standardize_smiles(smi: str, organic: set[str], counterions: set[str]) -> StdResult:
    """Standardise one SMILES; ``ok=False`` with a ``reason`` when the compound is rejected."""
    if not isinstance(smi, str) or not smi.strip():
        return StdResult(False, "empty_smiles")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return StdResult(False, "parse_error")
    try:
        mol = rdMolStandardize.Cleanup(mol)
        Chem.SanitizeMol(mol)
    except Exception:  # noqa: BLE001 - RDKit raises several exception types
        return StdResult(False, "standardize_error")

    elements = sorted({a.GetSymbol() for a in mol.GetAtoms()})
    if "C" not in elements:
        return StdResult(False, "no_carbon", elements=elements)
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    for frag in frags:
        frag_elems = {a.GetSymbol() for a in frag.GetAtoms()}
        if frag_elems - organic - counterions:
            return StdResult(False, "disallowed_element", elements=elements)
        if frag_elems & counterions and frag.GetNumAtoms() > 1:
            return StdResult(False, "metal_in_covalent_fragment", elements=elements)

    parent = _FRAGMENT_CHOOSER.choose(mol)
    try:
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    except Exception:  # noqa: BLE001
        scaffold = ""
    return StdResult(
        ok=True,
        smiles=Chem.MolToSmiles(mol),
        smiles_2d=Chem.MolToSmiles(mol, isomericSmiles=False),
        inchikey=Chem.MolToInchiKey(mol) or "",
        parent_smiles=Chem.MolToSmiles(parent),
        scaffold=scaffold,
        n_fragments=len(frags),
        n_heavy_atoms=mol.GetNumHeavyAtoms(),
        formal_charge_atoms=sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() != 0),
        mol_wt=Descriptors.MolWt(mol),
        elements=elements,
    )


def _standardize_frame(df: pd.DataFrame, smiles_col: str, cfg: dict) -> pd.DataFrame:
    organic = set(cfg["curation"]["organic_elements"])
    counterions = set(cfg["curation"]["counterion_elements"])
    rows = [standardize_smiles(s, organic, counterions) for s in df[smiles_col]]
    std = pd.DataFrame([r.__dict__ for r in rows], index=df.index)
    std["elements"] = std["elements"].map(lambda e: ";".join(e))
    return pd.concat([df, std], axis=1)


def _deduplicate(df: pd.DataFrame, key: str, priority: list[str], ascending: list[bool], sd_thr: float) -> pd.DataFrame:
    """Merge rows sharing ``key``: representative row by ``priority``, label = median of members."""
    df = df.sort_values(priority, ascending=ascending, kind="mergesort")
    grp = df.groupby(key, sort=False)
    rep = grp.head(1).set_index(key)
    agg = grp.agg(
        logS_median=("logS", "median"),
        n_merged=("logS", "size"),
        merged_logS_sd=("logS", lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else 0.0),
        merged_logS_range=("logS", lambda x: float(x.max() - x.min())),
        merged_ids=("compound_id", lambda x: ";".join(map(str, x))),
    )
    out = rep.join(agg)
    out["logS_representative"] = out["logS"]
    out["logS"] = out.pop("logS_median")
    out["dedup_conflict"] = out["merged_logS_sd"] > sd_thr
    return out.reset_index()


def _funnel_step(log: list, name: str, n_before: int, n_after: int, detail: dict | None = None) -> None:
    log.append({"step": name, "n_before": int(n_before), "n_after": int(n_after),
                "removed": int(n_before - n_after), **({"detail": detail} if detail else {})})


KEEP_STD_COLS = ["smiles", "smiles_2d", "inchikey", "parent_smiles", "scaffold", "n_fragments",
                 "n_heavy_atoms", "formal_charge_atoms", "mol_wt", "elements"]


def curate_aqsoldb(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    cfg = cfg or load_config()
    raw = pd.read_csv(path("raw", cfg) / cfg["data"]["aqsoldb"]["filename"])
    log: list = []
    df = raw.rename(columns={"ID": "compound_id", "Name": "name", "SMILES": "smiles_raw",
                             "Solubility": "logS", "SD": "aqsol_sd", "Ocurrences": "n_occurrences",
                             "Group": "reliability_group", "InChIKey": "inchikey_source"})
    df = df[["compound_id", "name", "smiles_raw", "inchikey_source", "logS", "aqsol_sd",
             "n_occurrences", "reliability_group"]].copy()
    df["source"] = df["compound_id"].str.split("-").str[0]
    n0 = len(df)
    df = _standardize_frame(df, "smiles_raw", cfg)
    removed = df[~df["ok"]].copy()
    reasons = removed["reason"].value_counts().to_dict()
    df = df[df["ok"]].copy()
    _funnel_step(log, "standardise_and_filter", n0, len(df), reasons)

    n1 = len(df)
    bad = ~np.isfinite(df["logS"].astype(float))
    removed = pd.concat([removed, df[bad].assign(reason="missing_label")])
    df = df[~bad].copy()
    _funnel_step(log, "finite_label", n1, len(df))

    df["inchikey_consistent"] = df["inchikey"].str[:14] == df["inchikey_source"].astype(str).str[:14]
    n2 = len(df)
    df = _deduplicate(df[["compound_id", "name", "smiles_raw", "inchikey_source", "logS", "aqsol_sd",
                          "n_occurrences", "reliability_group", "source", "inchikey_consistent"] + KEEP_STD_COLS],
                      key=cfg["curation"]["dedup_key"], priority=["n_occurrences", "compound_id"],
                      ascending=[False, True], sd_thr=cfg["curation"]["conflict_sd_threshold"])
    _funnel_step(log, "deduplicate_smiles_2d", n2, len(df),
                 {"groups_merged": int((df["n_merged"] > 1).sum()),
                  "conflicting_groups": int(df["dedup_conflict"].sum())})
    df["is_multicomponent"] = df["n_fragments"] > 1
    df["dataset"] = "aqsoldb"
    removed = removed[["compound_id", "name", "smiles_raw", "logS", "reason", "elements"]]
    return df, removed, log


def curate_esol(cfg: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    cfg = cfg or load_config()
    raw = pd.read_csv(path("raw", cfg) / cfg["data"]["esol"]["filename"])
    log: list = []
    df = pd.DataFrame({
        "compound_id": [f"ESOL-{i:04d}" for i in range(len(raw))],
        "name": raw["Compound ID"],
        "smiles_raw": raw["smiles"],
        "logS": raw["measured log solubility in mols per litre"].astype(float),
        "esol_pred": raw["ESOL predicted log solubility in mols per litre"].astype(float),
    })
    n0 = len(df)
    df = _standardize_frame(df, "smiles_raw", cfg)
    removed = df[~df["ok"]].copy()
    reasons = removed["reason"].value_counts().to_dict()
    df = df[df["ok"]].copy()
    _funnel_step(log, "standardise_and_filter", n0, len(df), reasons)
    n1 = len(df)
    bad = ~np.isfinite(df["logS"])
    df = df[~bad].copy()
    _funnel_step(log, "finite_label", n1, len(df))
    n2 = len(df)
    df = _deduplicate(df[["compound_id", "name", "smiles_raw", "logS", "esol_pred"] + KEEP_STD_COLS],
                      key=cfg["curation"]["dedup_key"], priority=["compound_id"], ascending=[True],
                      sd_thr=cfg["curation"]["conflict_sd_threshold"])
    _funnel_step(log, "deduplicate_smiles_2d", n2, len(df),
                 {"groups_merged": int((df["n_merged"] > 1).sum()),
                  "conflicting_groups": int(df["dedup_conflict"].sum())})
    df["is_multicomponent"] = df["n_fragments"] > 1
    df["dataset"] = "esol"
    removed = removed[["compound_id", "name", "smiles_raw", "logS", "reason", "elements"]]
    return df, removed, log


def link_overlap(aq: pd.DataFrame, es: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Annotate the ESOL/AqSolDB overlap (same ``smiles_2d``) and return a per-compound comparison."""
    m = es[["compound_id", "smiles_2d", "name", "logS"]].merge(
        aq[["compound_id", "smiles_2d", "logS", "source", "reliability_group", "n_occurrences", "aqsol_sd"]],
        on="smiles_2d", suffixes=("_esol", "_aqsoldb"))
    m["diff_aqsoldb_minus_esol"] = m["logS_aqsoldb"] - m["logS_esol"]
    aq = aq.copy()
    es = es.copy()
    aq["in_esol"] = aq["smiles_2d"].isin(es["smiles_2d"])
    es["in_aqsoldb"] = es["smiles_2d"].isin(aq["smiles_2d"])
    lookup = m.set_index("compound_id_esol")
    es["aqsoldb_id"] = es["compound_id"].map(lookup["compound_id_aqsoldb"])
    es["aqsoldb_logS"] = es["compound_id"].map(lookup["logS_aqsoldb"])
    return aq, es, m


def file_sha256(p) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run_curation(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    out = path("processed", cfg)
    aq, aq_removed, aq_log = curate_aqsoldb(cfg)
    es, es_removed, es_log = curate_esol(cfg)
    aq, es, overlap = link_overlap(aq, es)

    aq_cols = ["compound_id", "name", "smiles", "smiles_2d", "inchikey", "logS", "source", "aqsol_sd",
               "n_occurrences", "reliability_group", "n_merged", "merged_ids", "merged_logS_sd",
               "merged_logS_range", "dedup_conflict", "parent_smiles", "scaffold", "n_fragments",
               "is_multicomponent", "n_heavy_atoms", "formal_charge_atoms", "mol_wt", "elements",
               "inchikey_source", "inchikey_consistent", "in_esol", "smiles_raw"]
    es_cols = ["compound_id", "name", "smiles", "smiles_2d", "inchikey", "logS", "esol_pred", "n_merged",
               "merged_ids", "merged_logS_sd", "merged_logS_range", "dedup_conflict", "parent_smiles",
               "scaffold", "n_fragments", "is_multicomponent", "n_heavy_atoms", "formal_charge_atoms",
               "mol_wt", "elements", "in_aqsoldb", "aqsoldb_id", "aqsoldb_logS", "smiles_raw"]
    aq = aq[aq_cols].sort_values("compound_id", key=lambda s: s.str.split("-").map(lambda t: (t[0], int(t[1]))))
    es = es[es_cols].sort_values("compound_id")
    files = {
        "aqsoldb_clean.csv": aq, "esol_clean.csv": es, "esol_aqsoldb_overlap.csv": overlap,
        "aqsoldb_removed.csv": aq_removed, "esol_removed.csv": es_removed,
    }
    for name, frame in files.items():
        frame.to_csv(out / name, index=False)

    d = overlap["diff_aqsoldb_minus_esol"]
    log = {
        "aqsoldb": {"n_raw": int(aq_log[0]["n_before"]), "n_final": int(len(aq)), "steps": aq_log,
                    "inchikey_inconsistent": int((~aq["inchikey_consistent"]).sum()),
                    "multicomponent": int(aq["is_multicomponent"].sum()),
                    "acyclic (empty scaffold)": int((aq["scaffold"] == "").sum()),
                    "unique_scaffolds": int(aq["scaffold"].nunique())},
        "esol": {"n_raw": int(es_log[0]["n_before"]), "n_final": int(len(es)), "steps": es_log,
                 "unique_scaffolds": int(es["scaffold"].nunique())},
        "overlap": {"n_compounds": int(len(overlap)),
                    "mae_between_sources": float(d.abs().mean()),
                    "rmse_between_sources": float(np.sqrt((d ** 2).mean())),
                    "median_abs_diff": float(d.abs().median()),
                    "frac_abs_diff_gt_0.5": float((d.abs() > 0.5).mean()),
                    "frac_abs_diff_gt_1.0": float((d.abs() > 1.0).mean()),
                    "exact_match_frac(|d|<0.01)": float((d.abs() < 0.01).mean())},
        "config": cfg["curation"],
    }
    save_json(log, out / "curation_log.json")
    save_json({"dataset_version": "1.0.0", "rdkit": Chem.rdBase.rdkitVersion,
               "files": {n: file_sha256(out / n) for n in files}}, out / "DATASET_VERSION.json")
    return log
