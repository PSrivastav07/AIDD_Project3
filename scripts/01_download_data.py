"""Step 1 - download AqSolDB (Harvard Dataverse) and ESOL (DeepChem mirror) into data/raw."""
import argparse
import json

from solpred.config import load_config
from solpred.data.download import download_all

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()
    manifest = download_all(force=args.force, cfg=load_config(args.config))
    print(json.dumps(manifest, indent=2))
