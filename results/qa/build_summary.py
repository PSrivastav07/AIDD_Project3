"""Collect the QA headline numbers into ``results/qa/summary.json``.

Reads ``results/qa/audit_numbers.json`` (written by ``foundation_audit.py``) and the JUnit XML of the
full pytest run (``results/qa/pytest_results.xml``, produced by
``.venv/Scripts/python.exe -m pytest tests -p no:cacheprovider --junitxml=results/qa/pytest_results.xml``)
and writes a compact, machine-readable summary that the report can cite. Nothing is recomputed here.
"""
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

QA = Path(__file__).resolve().parent
QA_OWNED = {"test_curation", "test_splits", "test_features", "test_metrics", "test_experiment"}


def pytest_counts(xml_path: Path) -> dict:
    """Per-file and total pass / xfail / skip / failure counts from a JUnit XML file."""
    root = ET.parse(xml_path).getroot()
    suite = root if root.tag == "testsuite" else root[0]
    per_file: Counter = Counter()
    for case in suite.iter("testcase"):
        module = case.attrib["classname"].split(".")[-1]
        status = "passed"
        for child in case:
            if child.tag == "skipped":
                text = child.attrib.get("type", "") + child.attrib.get("message", "")
                status = "xfailed" if "xfail" in text else "skipped"
            elif child.tag in ("failure", "error"):
                status = "failed" if child.tag == "failure" else "error"
        per_file[(module, status)] += 1
    files = sorted({m for m, _ in per_file})
    by_file = {m: {s: per_file[(m, s)] for s in ("passed", "xfailed", "skipped", "failed", "error")
                   if per_file[(m, s)]} for m in files}
    total = Counter()
    owned = Counter()
    for (m, s), n in per_file.items():
        total[s] += n
        if m in QA_OWNED:
            owned[s] += n
    return {"n_tests": int(suite.attrib["tests"]), "wall_time_s": float(suite.attrib["time"]),
            "total": dict(total), "qa_owned_files": dict(owned), "by_file": by_file}


def build(audit: dict, tests: dict) -> dict:
    lk, sh = audit["leakage"], audit["scaffold_and_shift"]["by_split_type"]
    lab, nan = audit["labels"], audit["descriptor_nans"]
    return {
        "pytest": tests,
        "leakage_test_vs_train": {
            st: {"pct_same_smiles_2d": lk[st]["pct_match_smiles_2d_mean"],
                 "pct_same_inchikey_connectivity": round(lk[st]["pct_match_ik14_mean"], 3),
                 "pct_same_neutral_parent": round(lk[st]["pct_match_parent_ik14_mean"], 3),
                 **({"lgbm_test_mae_all": round(lk[st]["lgbm_mae_all_mean"], 3),
                     "lgbm_test_mae_without_parent_matches": round(lk[st]["lgbm_mae_unmatched_parent_ik14_mean"], 3),
                     "lgbm_mae_on_parent_matches": round(lk[st]["lgbm_mae_matched_parent_ik14_mean"], 3)}
                    if "lgbm_mae_all_mean" in lk[st] else {})}
            for st in ("random", "scaffold", "cluster", "loso", "external")},
        "matched_pairs": lk["pairs"],
        "within_aqsoldb_surviving_duplicates": audit["within_dataset"]["aqsoldb"],
        "split_shift": {st: {"test_frac_generic_scaffold_in_train": round(sh[st]["test_frac_generic_scaffold_in_train_mean"], 3),
                             "test_frac_acyclic": round(sh[st]["test_frac_acyclic_mean"], 3),
                             "train_logS_mean": round(sh[st]["train_logS_mean_mean"], 3),
                             "test_logS_mean": round(sh[st]["test_logS_mean_mean"], 3),
                             "train_mean_predictor_test_MAE": round(sh[st]["train_mean_predictor_test_MAE_mean"], 3),
                             "test_maxsim_median": round(sh[st]["test_maxsim_median_mean"], 3),
                             "test_frac_maxsim_eq_1": round(sh[st]["test_frac_maxsim_eq_1_mean"], 4)}
                        for st in ("random", "scaffold", "cluster")},
        "butina": {k: audit["butina"][k] for k in ("toy_ok", "rdkit_reference_identical_partition",
                                                    "full_recompute_identical_to_stored", "n_clusters",
                                                    "n_singletons", "largest_cluster")},
        "removed": {"n": audit["removed"]["n_removed"], "by_reason": audit["removed"]["by_reason"],
                    "annotation_by_reason": audit["removed"]["annotation_by_reason"]},
        "kept_inorganic_carbon_n": audit["kept_inorganic_carbon"]["n"],
        "labels": {k: lab[k] for k in ("logS_gt_1.5", "logS_lt_-10", "above_physical_bound_rho2.0",
                                       "abs_delaney_residual_gt_3_single_component", "aqsol_sd_gt_1",
                                       "dedup_merged_range_gt_1", "frac_dataset_flagged_any")},
        "dedup_groups": audit["dedup_groups"],
        "inchikey_inconsistent": audit["inchikey_inconsistent"],
        "charges": {k: audit["charges"][k] for k in ("carbanion", "alkoxide_created", "bare_proton",
                                                     "charged_single_species", "zwitterion_single",
                                                     "unbalanced_salt")},
        "descriptor_nans": {k: nan[k] for k in ("n_rows_with_nan", "frac_rows_with_nan", "frac_nan_rows_metal_salts",
                                                "logS_mean_nan_rows", "logS_mean_other_rows", "constant_columns")},
        "loso_test_frac_label_from_multiple_measurements": audit["other"]["loso_test_frac_label_from_multiple_measurements"],
        "sources": {"audit": "results/qa/audit_numbers.json", "pytest": "results/qa/pytest_results.xml",
                    "review": "results/qa/foundation_review.md"},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--audit", default=str(QA / "audit_numbers.json"))
    ap.add_argument("--junit", default=str(QA / "pytest_results.xml"))
    ap.add_argument("--out", default=str(QA / "summary.json"))
    args = ap.parse_args()
    audit = json.loads(Path(args.audit).read_text(encoding="utf-8"))
    summary = build(audit, pytest_counts(Path(args.junit)))
    Path(args.out).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["pytest"]["total"]), "->", args.out)


if __name__ == "__main__":
    main()
