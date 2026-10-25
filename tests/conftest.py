"""Shared pytest fixtures for the ``solpred`` test-suite.

Design
------
* Thread limits are set *before* numpy/BLAS are imported by the test modules, so the suite respects
  the shared-machine CPU budget (2 threads; override with ``SOLPRED_TEST_THREADS``).
* Unit tests use small synthetic inputs. Tests that check stored artefacts (curated CSVs, split JSON
  files, feature caches) use the session fixtures below, which ``pytest.skip`` instead of failing when
  an artefact has not been generated yet (fresh checkout).
* Nothing here writes into the project tree: ``isolated_cfg`` returns a deep copy of the project
  configuration whose ``processed`` / ``features`` / ``results`` paths point into ``tmp_path``.
"""
from __future__ import annotations

import copy
import os
from pathlib import Path

_THREADS = os.environ.get("SOLPRED_TEST_THREADS", "2")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, _THREADS)

import pandas as pd  # noqa: E402  (import after the thread limits are set)
import pytest  # noqa: E402

from solpred.config import ROOT, load_config  # noqa: E402

N_THREADS = int(_THREADS)
PROCESSED = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
FEATURES = ROOT / "data" / "features"
RAW = ROOT / "data" / "raw"


def require(*paths: Path) -> None:
    """Skip the calling test when a generated artefact is missing."""
    missing = [str(p.relative_to(ROOT)) for p in paths if not Path(p).exists()]
    if missing:
        pytest.skip(f"artefact(s) not generated yet: {missing}")


@pytest.fixture(scope="session")
def cfg() -> dict:
    """A private deep copy of the project configuration (never mutate the cached original)."""
    return copy.deepcopy(load_config())


@pytest.fixture(scope="session")
def element_sets(cfg) -> tuple[set[str], set[str]]:
    """``(organic_elements, counterion_elements)`` from the configuration."""
    return set(cfg["curation"]["organic_elements"]), set(cfg["curation"]["counterion_elements"])


@pytest.fixture(scope="session")
def aq() -> pd.DataFrame:
    """Curated AqSolDB (skips when ``data/processed/aqsoldb_clean.csv`` is missing)."""
    require(PROCESSED / "aqsoldb_clean.csv")
    from solpred.features import load_dataset

    return load_dataset("aqsoldb")


@pytest.fixture(scope="session")
def es() -> pd.DataFrame:
    """Curated ESOL (skips when ``data/processed/esol_clean.csv`` is missing)."""
    require(PROCESSED / "esol_clean.csv")
    from solpred.features import load_dataset

    return load_dataset("esol")


@pytest.fixture(scope="session")
def curation_log() -> dict:
    require(PROCESSED / "curation_log.json")
    from solpred.config import load_json

    return load_json(PROCESSED / "curation_log.json")


@pytest.fixture(scope="session")
def butina_labels(aq) -> pd.Series:
    """Stored Butina cluster label per AqSolDB compound id."""
    require(SPLITS / "aqsoldb_butina_clusters.csv")
    df = pd.read_csv(SPLITS / "aqsoldb_butina_clusters.csv", dtype={"compound_id": str})
    return df.set_index("compound_id")["cluster"]


@pytest.fixture
def isolated_cfg(cfg, tmp_path) -> dict:
    """Config copy whose data/results directories live in ``tmp_path`` (absolute paths win in ``path()``)."""
    c = copy.deepcopy(cfg)
    for key in ("processed", "features", "results", "figures", "tables", "models"):
        c["paths"][key] = str(tmp_path / key)
    return c
