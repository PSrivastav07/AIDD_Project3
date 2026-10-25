"""Step 3 - generate random / scaffold / cluster (x5 seeds), leave-one-source-out and external splits."""
import argparse

import pandas as pd

from solpred.config import load_config
from solpred.splits import make_all_splits

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    summ = make_all_splits(load_config(args.config))
    with pd.option_context("display.max_columns", 30, "display.width", 250):
        print(summ)
