"""Step 4 - compute and cache every base featurisation for AqSolDB and ESOL (data/features/*.npz)."""
import argparse
import time

import numpy as np

from solpred.config import load_config
from solpred.features import BASE_KINDS, load_or_compute

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    for ds in ("aqsoldb", "esol"):
        for kind in BASE_KINDS:
            t = time.time()
            X, ids, names = load_or_compute(ds, kind, cfg)
            print(f"{ds:8s} {kind:14s} shape={X.shape} nan_frac={np.isnan(X).mean():.5f} ({time.time() - t:.1f}s)")
