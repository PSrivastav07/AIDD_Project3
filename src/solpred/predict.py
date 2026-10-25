"""Production aqueous-solubility (logS) predictor with conformal intervals and applicability-domain flags.

Pipeline for every input SMILES
-------------------------------
1. **Standardise** with the curation rules of the training data
   (:func:`solpred.data.curate.standardize_smiles`, element sets from the model bundle). Unparseable input
   -> ``invalid SMILES``; parseable but outside the curation scope (no carbon, disallowed element,
   organometallic) -> ``rejected by curation rules``. Neither gets a prediction, and neither crashes a batch.
2. **Featurise** the standardised structure exactly as at training time (feature kind stored in the bundle).
3. **Predict** with the bagged ensemble: ``logS_pred`` = member mean, ``logS_std`` = member std.
4. **Interval**: normalised split-conformal interval ``[pi_low, pi_high]`` at level ``1 - alpha``, calibrated on
   scaffold-grouped out-of-fold predictions with score |y - mean| / (std + beta) (see :mod:`solpred.final_model`;
   beta is stored as the calibrator's ``eps``). The guarantee is *marginal* over
   compounds exchangeable with the calibration set; it is weaker outside the applicability domain.
5. **Applicability domain**: ECFP4 max-Tanimoto to the training set, nearest training compound with its
   measured logS.
6. **Warnings** (``"; "``-joined): invalid SMILES, rejected by curation rules, outside applicability domain,
   multicomponent (salt / mixture), prediction outside training logS range, and the informational
   ``in training set`` (measured value available - the prediction is then in-sample).

Python::

    from solpred.predict import SolubilityPredictor
    df = SolubilityPredictor.load().predict(["CCO", "c1ccc2ccccc2c1"], alpha=0.1)

CLI (console script ``solpred-predict``)::

    solpred-predict --smiles CCO c1ccccc1
    solpred-predict --input examples/example_smiles.csv --smiles-col smiles --output preds.csv --alpha 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from solpred.config import ROOT, load_json
from solpred.data.curate import StdResult, standardize_smiles
from solpred.features import featurize_smiles
from solpred.final_model import ensemble_predict, load_bundle
from solpred.metrics import solubility_class

DEFAULT_MODEL_DIR = "models/final"
OUTPUT_COLUMNS = ["input_smiles", "standardized_smiles", "valid", "error", "logS_pred", "logS_std", "pi_low",
                  "pi_high", "pi_level", "solubility_class", "solubility_mg_per_L", "ad_maxsim", "ad_in_domain",
                  "nearest_train_id", "nearest_train_smiles", "nearest_train_logS", "in_training_set", "warnings"]
INVALID_REASONS = {"empty_smiles", "parse_error", "standardize_error"}
W_INVALID = "invalid SMILES"
W_REJECTED = "rejected by curation rules"
W_OOD = "outside applicability domain"
W_MULTI = "multicomponent"
W_RANGE = "prediction outside training logS range"
W_TRAIN = "in training set"


def _resolve_dir(d: str | Path) -> Path:
    p = Path(d)
    if not p.is_absolute() and not p.exists():
        p = ROOT / p
    return p


class SolubilityPredictor:
    """Wraps a trained ``models/final`` bundle; see the module docstring for the pipeline."""

    def __init__(self, bundle: dict, metadata: dict | None = None):
        self.bundle = bundle
        self.metadata = metadata or {}
        self.members = bundle["members"]
        self.features = bundle["features"]
        self.conformal = bundle["conformal"]
        self.ad = bundle["ad"]
        self.logS_range = tuple(bundle["training_logS_range"])
        self.train_index: dict = bundle.get("train_index", {})
        self._organic = set(bundle["curation"]["organic_elements"])
        self._counterions = set(bundle["curation"]["counterion_elements"])
        self._feat_cfg = {"features": bundle["features_cfg"]}
        self._n_features = len(bundle["feature_names"])

    @classmethod
    def load(cls, dir: str | Path = DEFAULT_MODEL_DIR) -> "SolubilityPredictor":  # noqa: A002 - public API name
        """Load ``model.joblib`` (+ ``metadata.json`` when present) from a model directory."""
        d = _resolve_dir(dir)
        if not (d / "model.joblib").exists():
            raise FileNotFoundError(f"no model.joblib in {d}; train it with scripts/11_train_final_model.py")
        meta = load_json(d / "metadata.json") if (d / "metadata.json").exists() else {}
        return cls(load_bundle(d), meta)

    @property
    def sim_threshold(self) -> float:
        return float(self.ad.sim_threshold)

    # ------------------------------------------------------------------ steps
    def standardize(self, smiles: Sequence) -> list[StdResult]:
        out = []
        for s in smiles:
            try:
                out.append(standardize_smiles(s if isinstance(s, str) else "", self._organic, self._counterions))
            except Exception as exc:  # noqa: BLE001 - never let one input kill the batch
                out.append(StdResult(False, f"standardize_error: {exc}"))
        return out

    def featurize(self, smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """Feature matrix for standardised SMILES and a mask of rows that featurised successfully."""
        X = np.full((len(smiles), self._n_features), np.nan, dtype=np.float32)
        ok = np.zeros(len(smiles), dtype=bool)
        if not smiles:
            return X, ok
        try:
            Xb, _ = featurize_smiles(smiles, self.features, self._feat_cfg)
            X[:], ok[:] = Xb, True
        except Exception:  # noqa: BLE001 - fall back to one molecule at a time to isolate failures
            for i, s in enumerate(smiles):
                try:
                    X[i] = featurize_smiles([s], self.features, self._feat_cfg)[0][0]
                    ok[i] = True
                except Exception:  # noqa: BLE001
                    pass
        return X, ok

    def predict(self, smiles: Sequence[str], alpha: float = 0.1) -> pd.DataFrame:
        """Predict logS with a ``1 - alpha`` conformal interval and AD flags (one row per input, same order)."""
        if not 0.0 < float(alpha) < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        smiles = list(smiles)
        n = len(smiles)
        std = self.standardize(smiles)
        res = pd.DataFrame({
            "input_smiles": smiles,
            "standardized_smiles": [r.smiles if r.ok else None for r in std],
            "valid": [r.ok for r in std],
            "error": [None if r.ok else r.reason for r in std],
        })
        for c in ["logS_pred", "logS_std", "pi_low", "pi_high", "solubility_mg_per_L", "ad_maxsim", "nearest_train_logS"]:
            res[c] = np.nan
        res["pi_level"] = 1.0 - float(alpha)
        res["solubility_class"] = None
        res["ad_in_domain"] = pd.array([pd.NA] * n, dtype="boolean")
        res["in_training_set"] = pd.array([pd.NA] * n, dtype="boolean")
        res["nearest_train_id"] = None
        res["nearest_train_smiles"] = None
        warnings: list[list[str]] = [[] for _ in range(n)]
        for i, r in enumerate(std):
            if not r.ok:
                if r.reason.split(":")[0] in INVALID_REASONS:
                    warnings[i].append(W_INVALID)
                    res.at[i, "error"] = f"invalid SMILES ({r.reason})"
                else:
                    warnings[i].append(f"{W_REJECTED} ({r.reason})")
                    res.at[i, "error"] = f"rejected by curation rules ({r.reason}); outside model scope"

        idx = np.flatnonzero(res["valid"].to_numpy())
        if len(idx):
            X, ok = self.featurize([std[i].smiles for i in idx])
            for j in np.flatnonzero(~ok):
                i = idx[j]
                res.at[i, "valid"] = False
                res.at[i, "error"] = "featurization failed"
                warnings[i].append(W_INVALID)
            idx, X = idx[ok], X[ok]
        if len(idx):
            self._fill_predictions(res, idx, X, [std[i] for i in idx], alpha, warnings)
        res["warnings"] = ["; ".join(w) for w in warnings]
        return res[OUTPUT_COLUMNS]

    # ------------------------------------------------------------------ internals
    def _fill_predictions(self, res: pd.DataFrame, idx: np.ndarray, X: np.ndarray, std: list[StdResult],
                          alpha: float, warnings: list[list[str]]) -> None:
        mean, sd = ensemble_predict(self.members, X)
        scale = sd if self.conformal.normalized else None
        lo, hi = self.conformal.interval(mean, scale=scale, alpha=alpha)
        mw = np.array([r.mol_wt for r in std], dtype=float)
        res.loc[idx, "logS_pred"] = mean
        res.loc[idx, "logS_std"] = sd
        res.loc[idx, "pi_low"] = lo
        res.loc[idx, "pi_high"] = hi
        res.loc[idx, "solubility_class"] = solubility_class(mean)
        res.loc[idx, "solubility_mg_per_L"] = np.power(10.0, mean) * mw * 1000.0
        ad = self.ad.score([r.smiles for r in std])
        res.loc[idx, "ad_maxsim"] = ad["ad_maxsim"].to_numpy()
        res.loc[idx, "ad_in_domain"] = ad["ad_in_domain"].to_numpy(bool)
        res.loc[idx, "nearest_train_id"] = ad["nearest_train_id"].to_numpy(object)
        res.loc[idx, "nearest_train_smiles"] = ad["nearest_train_smiles"].to_numpy(object)
        res.loc[idx, "nearest_train_logS"] = ad["nearest_train_logS"].to_numpy(float)
        lo_y, hi_y = self.logS_range
        for j, i in enumerate(idx):
            r = std[j]
            hit = self.train_index.get(r.smiles_2d)
            res.at[i, "in_training_set"] = hit is not None
            if not ad["ad_in_domain"].iat[j]:
                warnings[i].append(f"{W_OOD} (max Tanimoto {ad['ad_maxsim'].iat[j]:.2f} < {self.sim_threshold:g})")
            if r.n_fragments > 1:
                warnings[i].append(f"{W_MULTI} ({r.n_fragments} fragments: salt/mixture; logS refers to the whole species)")
            if not lo_y <= mean[j] <= hi_y:
                warnings[i].append(f"{W_RANGE} [{lo_y:.2f}, {hi_y:.2f}]")
            if hit is not None:
                warnings[i].append(f"{W_TRAIN} ({hit[0]}, measured logS {hit[1]:.2f}; prediction is in-sample)")


# --------------------------------------------------------------------------- CLI
def _read_input(args) -> tuple[pd.DataFrame, list[str]]:
    if args.smiles:
        return pd.DataFrame(index=range(len(args.smiles))), list(args.smiles)
    df = pd.read_csv(args.input, keep_default_na=False, na_values=[""], dtype=str)
    if args.smiles_col not in df.columns:
        raise SystemExit(f"column {args.smiles_col!r} not in {args.input} (columns: {list(df.columns)})")
    smiles = df[args.smiles_col].fillna("").tolist()
    return df.drop(columns=[args.smiles_col]).reset_index(drop=True), smiles


def main(argv: Sequence[str] | None = None) -> int:
    """``solpred-predict``: predict logS (+ interval, class, AD flags) for SMILES given inline or in a CSV."""
    ap = argparse.ArgumentParser(prog="solpred-predict", description=main.__doc__)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--smiles", nargs="+", help="one or more SMILES strings")
    src.add_argument("--input", help="CSV file with a SMILES column (other columns are carried through)")
    ap.add_argument("--smiles-col", default="smiles", help="SMILES column of --input (default: smiles)")
    ap.add_argument("--output", help="output CSV (default: print a compact table to stdout)")
    ap.add_argument("--alpha", type=float, default=0.1, help="miscoverage; interval level = 1 - alpha (default 0.1)")
    ap.add_argument("--model-dir", default=DEFAULT_MODEL_DIR, help="trained model directory (default models/final)")
    args = ap.parse_args(argv)

    extra, smiles = _read_input(args)
    pred = SolubilityPredictor.load(args.model_dir).predict(smiles, alpha=args.alpha)
    out = pd.concat([extra, pred], axis=1)
    n_valid = int(pred["valid"].sum())
    n_flag = int((pred["warnings"].str.len() > 0).sum())
    print(f"[solpred-predict] {len(pred)} inputs, {n_valid} predicted, {n_flag} with warnings "
          f"(interval level {1 - args.alpha:.0%})", file=sys.stderr)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False, float_format="%.4g")
        print(f"[solpred-predict] wrote {args.output}", file=sys.stderr)
    else:
        cols = [c for c in extra.columns] + ["input_smiles", "logS_pred", "pi_low", "pi_high", "solubility_class",
                                             "ad_maxsim", "warnings"]
        with pd.option_context("display.max_colwidth", 70, "display.width", 200):
            print(out[cols].round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
