"""Triage-ranking demo: turn ``solpred-predict`` output into a measure / don't-measure shortlist.

Compounds are ranked by predicted logS and assigned a tier from the conformal interval relative to a
project-specific solubility threshold (default logS = -4, i.e. 100 umol/L):

* ``soluble``   - the whole ``1 - alpha`` interval lies above the threshold;
* ``insoluble`` - the whole interval lies below it;
* ``uncertain: measure`` - the interval straddles the threshold, so the model cannot decide.

Flags: ``OOD`` (outside the applicability domain), ``train`` (structure is in the training set: use the measured
value, the prediction is in-sample), ``salt`` (multicomponent input).

Usage::

    .venv/Scripts/python.exe examples/triage_demo.py --input examples/example_predictions.csv \
        --threshold -4 --output examples/triage_ranking.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def tier(pi_low: float, pi_high: float, threshold: float) -> str:
    """Decision tier from the prediction interval relative to ``threshold``."""
    if pi_low > threshold:
        return "soluble"
    if pi_high < threshold:
        return "insoluble"
    return "uncertain: measure"


def triage_table(pred: pd.DataFrame, threshold: float, name_col: str = "name") -> pd.DataFrame:
    """Ranked table of the valid predictions with tier and flags."""
    v = pred[pred["valid"].astype(bool)].copy()
    v["tier"] = [tier(lo, hi, threshold) for lo, hi in zip(v["pi_low"], v["pi_high"])]
    v["flags"] = [" ".join(f for f, on in (("OOD", not bool(a)), ("train", bool(t)), ("salt", "multicomponent" in str(w)))
                           if on)
                  for a, t, w in zip(v["ad_in_domain"], v["in_training_set"], v["warnings"])]
    v = v.sort_values("logS_pred", ascending=False)
    v.insert(0, "rank", range(1, len(v) + 1))
    label = name_col if name_col in v.columns else "input_smiles"
    level = f"{float(v['pi_level'].iloc[0]):.0%}" if len(v) else "PI"
    t = v[["rank", label, "logS_pred", "pi_low", "pi_high", "solubility_class", "solubility_mg_per_L", "ad_maxsim",
           "tier", "flags"]].rename(columns={
        label: "compound", "logS_pred": "logS pred", "pi_low": f"{level} PI low", "pi_high": f"{level} PI high",
        "solubility_class": "class", "solubility_mg_per_L": "mg/L", "ad_maxsim": "AD max sim"})
    t["mg/L"] = t["mg/L"].map(lambda x: f"{x:.3g}")
    return t


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="examples/example_predictions.csv", help="CSV written by solpred-predict")
    ap.add_argument("--threshold", type=float, default=-4.0, help="decision threshold in logS (log mol/L)")
    ap.add_argument("--name-col", default="name")
    ap.add_argument("--output", default=None, help="optional Markdown output")
    args = ap.parse_args()
    pred = pd.read_csv(args.input, keep_default_na=False, na_values=[""])
    t = triage_table(pred, args.threshold, args.name_col)
    md = t.to_markdown(index=False, floatfmt=".2f")
    counts = t["tier"].value_counts().to_dict()
    n_invalid = int((~pred["valid"].astype(bool)).sum())
    footer = f"\nTiers at threshold logS = {args.threshold:g}: {counts}; {n_invalid} input(s) not predicted.\n"
    print(md + footer)
    if args.output:
        Path(args.output).write_text(md + "\n" + footer, encoding="utf-8")


if __name__ == "__main__":
    main()
