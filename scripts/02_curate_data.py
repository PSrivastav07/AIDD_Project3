"""Step 2 - standardise, filter and de-duplicate AqSolDB and ESOL; link their overlap."""
import argparse
import json

from solpred.config import load_config
from solpred.data.curate import run_curation

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    log = run_curation(load_config(args.config))
    print(json.dumps(log, indent=2))
