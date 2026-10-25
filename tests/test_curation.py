"""Tests for ``solpred.data.curate``: standardisation, filtering, de-duplication and ESOL linkage.

Unit tests run on hand-written SMILES / tiny frames; the ``*_real`` tests check invariants of the
stored curated files (skipped when they have not been generated). Tests marked ``xfail(strict=True)``
document *known* limitations of the read-only curation code that are discussed in
``results/qa/foundation_review.md``; they will start failing (XPASS) once the limitation is fixed,
which is the signal to drop the marker.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem

from conftest import PROCESSED, RAW, require
from solpred.data.curate import _deduplicate, link_overlap, standardize_smiles


@pytest.fixture
def std(element_sets):
    organic, counter = element_sets
    return lambda smi: standardize_smiles(smi, organic, counter)


# --------------------------------------------------------------------------- rejection reasons
@pytest.mark.parametrize("smi, reason", [
    ("", "empty_smiles"),
    ("   ", "empty_smiles"),
    (None, "empty_smiles"),
    ("C1CC", "parse_error"),                       # unclosed ring
    ("not_a_smiles", "parse_error"),
    ("O", "no_carbon"),                            # water
    ("[Na+].[Cl-]", "no_carbon"),                  # inorganic salt
    ("OS(=O)(=O)O", "no_carbon"),
    ("C[Hg]Cl", "disallowed_element"),             # organomercury
    ("CCCC[Sn](CCCC)(CCCC)CCCC", "disallowed_element"),  # organotin
    ("[Zn+2].CC(=O)[O-].CC(=O)[O-]", "disallowed_element"),  # zinc salt: Zn not an allowed counter-ion
    ("[Fe+2].c1cc[cH-]c1.c1cc[cH-]c1", "disallowed_element"),  # ferrocene
    ("C[Mg]Br", "metal_in_covalent_fragment"),     # Grignard: Mg-C bond survives disconnection
    ("CCCC[Li]", "metal_in_covalent_fragment"),    # organolithium
])
def test_rejection_reasons(std, smi, reason):
    r = std(smi)
    assert not r.ok
    assert r.reason == reason
    assert r.smiles == "" and r.smiles_2d == ""


def test_accepts_organic_elements(std):
    for smi in ["CB(O)O", "C[Si](C)(C)C", "c1cc[se]c1", "CP(=O)(O)O", "FC(F)(F)c1ccc(Br)cc1I", "CS(C)=O"]:
        r = std(smi)
        assert r.ok, (smi, r.reason)


# --------------------------------------------------------------------------- salts and counter-ions
def test_sodium_salt_kept_as_salt(std):
    r = std("CC(=O)[O-].[Na+]")
    assert r.ok
    assert r.n_fragments == 2
    assert "[Na+]" in r.smiles
    assert r.parent_smiles == "CC(=O)[O-]"          # largest organic fragment, charge kept
    assert r.formal_charge_atoms == 2


@pytest.mark.parametrize("smi", ["CC(=O)[O-].[K+]", "CC(=O)[O-].[Li+]", "CC(=O)[O-].CC(=O)[O-].[Ca+2]",
                                 "CC(=O)[O-].CC(=O)[O-].[Mg+2]"])
def test_allowed_counterions(std, smi):
    assert std(smi).ok


def test_covalent_metal_salt_is_disconnected(std):
    """Metal-oxygen bonds are broken by the MetalDisconnector -> ionic salt, not rejected."""
    r = std("CC(=O)O[Na]")
    assert r.ok
    assert r.smiles == std("CC(=O)[O-].[Na+]").smiles


def test_neutral_metal_atom_salt_is_reionised(std):
    """AqSolDB writes many salts as ``[Na].acid``; Cleanup charges the metal and deprotonates the acid."""
    assert std("[Na].OC(=O)c1ccccc1").smiles == std("[Na+].[O-]C(=O)c1ccccc1").smiles


def test_hydrochloride_kept_with_parent(std):
    r = std("Cl.NCCC(=O)O")
    assert r.ok and r.n_fragments == 2
    assert r.parent_smiles == "NCCC(=O)O"
    assert r.smiles_2d == "Cl.NCCC(=O)O"               # HCl kept as a neutral fragment


def test_quaternary_ammonium_salt(std):
    r = std("CC[N+](C)(C)C.[Br-]")
    assert r.ok and r.n_fragments == 2
    assert r.parent_smiles == "CC[N+](C)(C)C"
    assert r.scaffold == ""


# --------------------------------------------------------------------------- stereo / canonicalisation
def test_stereo_removed_in_smiles_2d(std):
    l_ala, d_ala, rac = std("N[C@@H](C)C(=O)O"), std("N[C@H](C)C(=O)O"), std("NC(C)C(=O)O")
    assert "@" in l_ala.smiles and "@" in d_ala.smiles
    assert l_ala.smiles != d_ala.smiles
    assert l_ala.smiles_2d == d_ala.smiles_2d == rac.smiles_2d
    assert "@" not in l_ala.smiles_2d


def test_double_bond_stereo_removed_in_smiles_2d(std):
    e, z = std("C/C=C/C(=O)O"), std("C/C=C\\C(=O)O")
    assert e.smiles != z.smiles
    assert e.smiles_2d == z.smiles_2d
    assert "/" not in e.smiles_2d and "\\" not in e.smiles_2d


def test_canonical_independent_of_input_order(std):
    assert std("OCC").smiles == std("C(O)C").smiles == std("[CH3][CH2][OH]").smiles
    assert std("c1ccccc1O").inchikey == std("Oc1ccccc1").inchikey


def test_descriptive_fields(std):
    r = std("Cc1ccccc1")
    assert r.scaffold == "c1ccccc1"
    assert r.n_heavy_atoms == 7
    assert r.mol_wt == pytest.approx(92.141, abs=0.01)
    assert len(r.inchikey) == 27 and r.inchikey == "YXFVVABEGXRONW-UHFFFAOYSA-N"
    assert r.elements == ["C"]
    assert std("CCCCCC").scaffold == ""              # acyclic -> empty scaffold


# --------------------------------------------------------------------------- known limitations (documented)
@pytest.mark.xfail(strict=True, reason="Cleanup charges isolated neutral metals and deprotonates a C-H "
                                       "when no acid is present (carbanion artefact, see foundation_review.md)")
def test_known_no_carbanion_from_neutral_metal(std):
    r = std("[Na].CCCCCCCCCCCCOS(=O)(=O)c1ccccc1")
    mol = Chem.MolFromSmiles(r.smiles)
    assert not any(a.GetSymbol() == "C" and a.GetFormalCharge() < 0 for a in mol.GetAtoms())


@pytest.mark.xfail(strict=True, reason="no neutralisation step: acid and carboxylate (or HCl salt and "
                                       "ammonium chloride) get different smiles_2d and are not merged")
def test_known_charge_states_not_merged(std):
    assert std("Cc1ccc(C(=O)O)cc1C").smiles_2d == std("Cc1ccc(C(=O)[O-])cc1C").smiles_2d


@pytest.mark.xfail(strict=True, reason="isotope labels are dropped but the explicit [H] remains, so "
                                       "CDCl3 and CHCl3 get different smiles_2d")
def test_known_isotopologues_not_merged(std):
    assert std("[2H]C(Cl)(Cl)Cl").smiles_2d == std("ClC(Cl)Cl").smiles_2d


# --------------------------------------------------------------------------- de-duplication
def _dup_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "compound_id": ["A-2", "B-1", "C-7", "A-9", "D-3"],
        "smiles_2d": ["CCO", "CCO", "CCO", "CCN", "CCC"],
        "logS": [-1.0, -2.0, -4.0, 0.5, -1.5],
        "n_occurrences": [1, 3, 1, 1, 2],
    })


def test_deduplicate_median_and_counts():
    out = _deduplicate(_dup_frame(), key="smiles_2d", priority=["n_occurrences", "compound_id"],
                       ascending=[False, True], sd_thr=0.5).set_index("smiles_2d")
    assert len(out) == 3
    g = out.loc["CCO"]
    assert g["logS"] == pytest.approx(-2.0)                       # median of -1, -2, -4
    assert g["n_merged"] == 3
    assert g["merged_logS_sd"] == pytest.approx(np.std([-1, -2, -4], ddof=1))
    assert g["merged_logS_range"] == pytest.approx(3.0)
    assert bool(g["dedup_conflict"]) is True
    assert g["compound_id"] == "B-1"                              # most occurrences wins
    assert g["logS_representative"] == pytest.approx(-2.0)
    assert g["merged_ids"].split(";")[0] == "B-1"
    assert set(g["merged_ids"].split(";")) == {"A-2", "B-1", "C-7"}
    for key in ("CCN", "CCC"):
        s = out.loc[key]
        assert s["n_merged"] == 1 and s["merged_logS_sd"] == 0.0 and not s["dedup_conflict"]


def test_deduplicate_even_group_median_and_threshold():
    df = pd.DataFrame({"compound_id": ["X-1", "X-2"], "smiles_2d": ["C", "C"], "logS": [-1.0, -1.6],
                       "n_occurrences": [1, 1]})
    out = _deduplicate(df, "smiles_2d", ["n_occurrences", "compound_id"], [False, True], sd_thr=0.5)
    assert out.loc[0, "logS"] == pytest.approx(-1.3)
    assert out.loc[0, "compound_id"] == "X-1"                      # tie broken by id
    assert out.loc[0, "merged_logS_sd"] == pytest.approx(0.6 / np.sqrt(2))
    assert not out.loc[0, "dedup_conflict"]                        # SD 0.42 <= 0.5


def test_link_overlap():
    aq = pd.DataFrame({"compound_id": ["A-1", "A-2", "A-3"], "smiles_2d": ["CCO", "CCN", "CCC"],
                       "logS": [0.5, -1.0, -2.0], "source": list("AAB"), "reliability_group": ["G1"] * 3,
                       "n_occurrences": [1, 1, 1], "aqsol_sd": [0.0] * 3})
    es = pd.DataFrame({"compound_id": ["ESOL-0000", "ESOL-0001"], "smiles_2d": ["CCC", "c1ccccc1"],
                       "name": ["propane", "benzene"], "logS": [-1.5, -1.6]})
    aq2, es2, m = link_overlap(aq, es)
    assert aq2["in_esol"].tolist() == [False, False, True]
    assert es2["in_aqsoldb"].tolist() == [True, False]
    assert es2.loc[0, "aqsoldb_id"] == "A-3" and pd.isna(es2.loc[1, "aqsoldb_id"])
    assert es2.loc[0, "aqsoldb_logS"] == pytest.approx(-2.0)
    assert len(m) == 1 and m.loc[0, "diff_aqsoldb_minus_esol"] == pytest.approx(-0.5)
    assert "in_esol" not in aq.columns                              # inputs not mutated


# --------------------------------------------------------------------------- invariants of the stored files
def test_curated_aqsoldb_invariants_real(aq, curation_log):
    assert aq["compound_id"].is_unique
    assert aq["smiles_2d"].is_unique
    assert not aq["smiles_2d"].str.contains(r"[@/\\]", regex=True).any()
    assert len(aq) == curation_log["aqsoldb"]["n_final"]
    assert int(aq["is_multicomponent"].sum()) == curation_log["aqsoldb"]["multicomponent"]
    assert int((aq["scaffold"] == "").sum()) == curation_log["aqsoldb"]["acyclic (empty scaffold)"]
    assert np.isfinite(aq["logS"]).all()
    assert (aq["n_merged"] == aq["merged_ids"].str.split(";").str.len()).all()
    assert (aq["dedup_conflict"] == (aq["merged_logS_sd"] > 0.5)).all()
    assert (aq["is_multicomponent"] == (aq["n_fragments"] > 1)).all()
    allowed = set(curation_log["config"]["organic_elements"]) | set(curation_log["config"]["counterion_elements"])
    used = set(";".join(aq["elements"]).split(";"))
    assert used <= allowed


def test_every_raw_row_accounted_for_real(aq):
    require(RAW / "aqsoldb_curated-solubility-dataset.csv", PROCESSED / "aqsoldb_removed.csv")
    raw = pd.read_csv(RAW / "aqsoldb_curated-solubility-dataset.csv")
    removed = pd.read_csv(PROCESSED / "aqsoldb_removed.csv", keep_default_na=False, na_values=[""])
    kept = aq["merged_ids"].str.split(";").explode()
    accounted = pd.concat([kept, removed["compound_id"]])
    assert accounted.is_unique
    assert set(accounted) == set(raw["ID"])


def test_merged_label_is_median_of_raw_real(aq):
    require(RAW / "aqsoldb_curated-solubility-dataset.csv")
    raw = pd.read_csv(RAW / "aqsoldb_curated-solubility-dataset.csv").set_index("ID")["Solubility"]
    ex = aq[["compound_id", "merged_ids", "logS"]].assign(mid=aq["merged_ids"].str.split(";")).explode("mid")
    med = ex.assign(raw=ex["mid"].map(raw)).groupby("compound_id")["raw"].median()
    np.testing.assert_allclose(aq.set_index("compound_id")["logS"].loc[med.index], med, atol=1e-9)


def test_esol_overlap_consistent_real(aq, es, curation_log):
    assert int(aq["in_esol"].sum()) == int(es["in_aqsoldb"].sum()) == curation_log["overlap"]["n_compounds"]
    lab = aq.set_index("compound_id")["logS"]
    linked = es[es["in_aqsoldb"]]
    np.testing.assert_allclose(linked["aqsoldb_logS"], lab.loc[linked["aqsoldb_id"]].to_numpy())
    assert set(linked["smiles_2d"]) <= set(aq["smiles_2d"])
    assert es["smiles_2d"].is_unique
