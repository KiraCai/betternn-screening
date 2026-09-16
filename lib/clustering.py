"""Clustering utilities: KMeans, sphere exclusion, singleton/binder isolation."""

from __future__ import annotations

import numpy as np
from sklearn.cluster import MiniBatchKMeans


# ---------------------------------------------------------------------------
# KMeans clustering
# ---------------------------------------------------------------------------
def kmeans_cluster(
    X: np.ndarray,
    n_clusters: int = 200,
    seed: int = 42,
    batch_size: int = 4096,
) -> np.ndarray:
    """Cluster feature matrix X with MiniBatchKMeans. Returns integer labels."""
    km = MiniBatchKMeans(
        n_clusters=min(n_clusters, len(X)),
        batch_size=batch_size,
        random_state=seed,
        n_init=3,
        max_iter=200,
    )
    return km.fit_predict(X)


# ---------------------------------------------------------------------------
# Tanimoto helpers (numpy, no RDKit dependency)
# ---------------------------------------------------------------------------
def _tanimoto_one_vs_all(a: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Tanimoto similarity of one binary vector *a* against rows of *B*."""
    inter = B @ a  # (n,) dot products = intersection counts
    ca = float(a.sum())
    cb = B.sum(axis=1).astype(np.float64)
    union = ca + cb - inter
    return np.where(union > 0, inter / union, 0.0)


def tanimoto_nearest_neighbor(fps: np.ndarray) -> np.ndarray:
    """Nearest-neighbor Tanimoto for every row. O(n^2) - use on <=100k molecules."""
    n = len(fps)
    nn = np.zeros(n, dtype=np.float64)
    fp_f = fps.astype(np.float64)
    for i in range(n):
        sim = _tanimoto_one_vs_all(fp_f[i], fp_f)
        sim[i] = 0.0
        nn[i] = sim.max()
    return nn.astype(np.float32)


# ---------------------------------------------------------------------------
# Sphere exclusion (Butina algorithm)
# ---------------------------------------------------------------------------
def sphere_exclusion(
    fps: np.ndarray,
    threshold: float = 0.4,
    scores: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sphere-exclusion clustering on binary fingerprints.

    Args:
        fps: (n, d) binary fingerprint array (0/1 values).
        threshold: Tanimoto similarity cutoff. Molecules within this
            similarity of a leader are assigned to its cluster.
        scores: optional (n,) array to order leader selection (descending).
            If None, processes molecules in original order.

    Returns:
        leaders: 1D array of leader (centroid) indices.
        labels: (n,) cluster assignment for each molecule.
    """
    n = len(fps)
    fp_f = fps.astype(np.float64)
    labels = -np.ones(n, dtype=np.int64)

    if scores is not None:
        order = np.argsort(-scores)
    else:
        order = np.arange(n)

    leader_list: list[int] = []
    lid = 0
    for i in order:
        if labels[i] >= 0:
            continue
        sim = _tanimoto_one_vs_all(fp_f[i], fp_f)
        members = (sim >= threshold) & (labels < 0)
        labels[members] = lid
        leader_list.append(int(i))
        lid += 1

    return np.array(leader_list, dtype=np.int64), labels


# ---------------------------------------------------------------------------
# Singleton analysis
# ---------------------------------------------------------------------------
def count_singletons(
    fps: np.ndarray,
    threshold: float = 0.4,
    nn_distances: np.ndarray | None = None,
) -> dict[str, float]:
    """Count structurally isolated molecules (no neighbor above Tanimoto threshold).

    Args:
        fps: (n, d) binary fingerprint array.
        threshold: Tanimoto similarity cutoff.
        nn_distances: optional pre-computed nearest-neighbor Tanimoto array.
            If provided, fps is ignored and singletons are defined as
            molecules with nn_distance < threshold.
    """
    if nn_distances is not None:
        singletons = nn_distances < threshold
    else:
        nn = tanimoto_nearest_neighbor(fps)
        singletons = nn < threshold

    n = len(singletons)
    n_sing = int(singletons.sum())
    return {
        "n_total": n,
        "n_singletons": n_sing,
        "pct_singletons": round(n_sing / n * 100, 2) if n > 0 else 0.0,
    }


def binder_isolation(
    fps: np.ndarray,
    binder_idx: np.ndarray,
    threshold: float = 0.4,
    nn_distances: np.ndarray | None = None,
) -> dict[str, float]:
    """How many binders have no structural neighbor within Tanimoto threshold.

    Args:
        fps: (n, d) binary fingerprint array for the full library.
        binder_idx: indices of known binders within fps.
        threshold: Tanimoto similarity cutoff.
        nn_distances: optional pre-computed nearest-neighbor Tanimoto for
            the full library (same length as fps).
    """
    if len(binder_idx) == 0:
        return {"n_isolated": 0, "n_total": 0, "pct_isolated": 0.0}

    if nn_distances is not None:
        isolated = nn_distances[binder_idx] < threshold
    else:
        fp_f = fps.astype(np.float64)
        isolated = np.zeros(len(binder_idx), dtype=bool)
        for k, bi in enumerate(binder_idx):
            sim = _tanimoto_one_vs_all(fp_f[bi], fp_f)
            sim[bi] = 0.0
            isolated[k] = sim.max() < threshold

    n_iso = int(isolated.sum())
    n_total = len(binder_idx)
    return {
        "n_isolated": n_iso,
        "n_total": n_total,
        "pct_isolated": round(n_iso / n_total * 100, 2) if n_total > 0 else 0.0,
    }
