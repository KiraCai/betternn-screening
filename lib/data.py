"""Data loading and sampling utilities."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd


def load_dataset(
    csv_path: str | Path,
    smiles_col: str = "smiles",
    target_cols: list[str] | None = None,
    id_col: str = "zincid",
) -> pd.DataFrame:
    """Load CSV, drop rows with missing SMILES or targets, reset index."""
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col]).reset_index(drop=True)
    if target_cols:
        mask = df[target_cols].notna().all(axis=1)
        df = df[mask].reset_index(drop=True)
    return df


def load_binders(binders_path: str | Path, id_col: str = "zincid") -> set[str]:
    """Load binder IDs from a CSV file."""
    df = pd.read_csv(binders_path)
    return set(df[id_col].astype(str))


def find_binder_indices(
    df: pd.DataFrame, binder_ids: set[str], id_col: str = "zincid"
) -> np.ndarray:
    """Return integer indices of rows whose id_col is in binder_ids."""
    return np.array(
        [i for i, z in enumerate(df[id_col].astype(str)) if z in binder_ids],
        dtype=np.int64,
    )


def cluster_initial(
    X: np.ndarray, clusters: np.ndarray, frac: float, seed: int
) -> np.ndarray:
    """Select initial training indices with one molecule per cluster, then random fill."""
    rng = np.random.RandomState(seed)
    n = len(X)
    n_sample = int(n * frac)
    k = int(clusters.max()) + 1
    n_per = max(1, n_sample // k)
    chosen: list[int] = []
    for c in range(k):
        idx = np.where(clusters == c)[0]
        if len(idx):
            chosen.extend(rng.choice(idx, size=min(n_per, len(idx)), replace=False).tolist())
    rem = n_sample - len(chosen)
    if rem > 0:
        pool = np.array(sorted(set(range(n)) - set(chosen)))
        chosen.extend(rng.choice(pool, size=min(rem, len(pool)), replace=False).tolist())
    return np.array(sorted(chosen[:n_sample]), dtype=np.int64)


def reservoir_sample(iterable, k: int, seed: int = 42) -> list:
    """Algorithm R reservoir sampling for streaming large files."""
    rng = random.Random(seed)
    reservoir: list = []
    for i, item in enumerate(iterable):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)
            if j < k:
                reservoir[j] = item
    return reservoir
