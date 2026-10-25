"""Download the raw AqSolDB and ESOL files and record provenance (URL, date, SHA-256)."""
from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import requests

from solpred.config import load_config, load_json, path, save_json


def sha256(file: Path) -> str:
    h = hashlib.sha256()
    with open(file, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, force: bool = False) -> Path:
    if dest.exists() and not force:
        return dest
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def download_all(force: bool = False, cfg: dict | None = None) -> dict:
    """Fetch every dataset in ``cfg['data']`` into data/raw and write data/raw/MANIFEST.json."""
    cfg = cfg or load_config()
    raw = path("raw", cfg)
    manifest_file = raw / "MANIFEST.json"
    manifest = load_json(manifest_file) if manifest_file.exists() else {}
    for name, spec in cfg["data"].items():
        dest = raw / spec["filename"]
        fresh = force or not dest.exists()
        download(spec["url"], dest, force=force)
        entry = {
            "url": spec["url"],
            "file": str(dest.relative_to(raw.parent.parent)).replace("\\", "/"),
            "sha256": sha256(dest),
            "bytes": dest.stat().st_size,
        }
        if fresh or name not in manifest:
            entry["downloaded_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        else:
            entry["downloaded_utc"] = manifest[name].get("downloaded_utc")
        for k in ("doi", "license"):
            if k in spec:
                entry[k] = spec[k]
        manifest[name] = entry
    save_json(manifest, manifest_file)
    return manifest
