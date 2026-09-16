"""
Evaluation metrics for surrogate model quality.

Covers ranking correlation, recall-at-k, binder ROC-AUC, enrichment factors,
and the sustained-crossing budget estimator used throughout the project.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr as _spearmanr
from sklearn.metrics import roc_auc_score


# ---------------------------------------------------------------------------
# Ranking correlation
# ---------------------------------------------------------------------------
def spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Spearman rank correlation between true and predicted scores."""
    rho, _ = _spearmanr(y_true, y_pred)
    return float(rho)


# ---------------------------------------------------------------------------
# Recall at top-k%
# ---------------------------------------------------------------------------
def recall_at_k(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10),
) -> dict[str, float]:
    """Fraction of the true top-k% that appears in the predicted top-k%.

    Returns dict with keys like ``"R@1%"``, ``"R@5%"``, ``"R@10%"``.
    """
    n = len(y_true)
    out: dict[str, float] = {}
    for pct in fractions:
        n_top = max(1, int(n * pct))
        true_top = set(np.argsort(y_true)[-n_top:])
        pred_top = set(np.argsort(y_pred)[-n_top:])
        out[f"R@{int(pct * 100)}%"] = len(true_top & pred_top) / n_top
    return out


# ---------------------------------------------------------------------------
# Binder ROC-AUC
# ---------------------------------------------------------------------------
def binder_roc_auc(pred_all: np.ndarray, binder_idx: np.ndarray) -> float:
    """ROC-AUC treating experimental binders as positives.

    Returns ``nan`` if there are fewer than 2 classes present.
    """
    if len(binder_idx) == 0 or len(binder_idx) >= len(pred_all):
        return float("nan")
    labels = np.zeros(len(pred_all), dtype=np.int32)
    labels[binder_idx] = 1
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, pred_all))


# ---------------------------------------------------------------------------
# Binder counts in top-k%
# ---------------------------------------------------------------------------
def binder_counts(
    pred_all: np.ndarray,
    binder_idx: np.ndarray,
    fractions: tuple[float, ...] = (0.01, 0.05, 0.10, 0.20),
) -> dict[str, int]:
    """Number of known binders in the predicted top-k%.

    Returns dict with keys like ``"binders_top1%"``, ``"binders_top5%"``.
    """
    out: dict[str, int] = {}
    if len(binder_idx) == 0:
        for pct in fractions:
            out[f"binders_top{int(pct * 100)}%"] = 0
        return out
    ranks = np.argsort(np.argsort(-pred_all))
    br = ranks[binder_idx]
    n = len(pred_all)
    for pct in fractions:
        top_n = int(n * pct)
        out[f"binders_top{int(pct * 100)}%"] = int((br < top_n).sum())
    return out


# ---------------------------------------------------------------------------
# Enrichment factor
# ---------------------------------------------------------------------------
def enrichment_factor(
    pred_all: np.ndarray,
    binder_idx: np.ndarray,
    cutoff_fracs: tuple[float, ...] = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.50),
) -> dict[str, float]:
    """Enrichment factor at each cutoff: (binders in top-x%) / (expected by random).

    Returns dict with keys like ``"EF@0.1%"``, ``"EF@1%"``.
    """
    n = len(pred_all)
    n_binders = len(binder_idx)
    out: dict[str, float] = {}
    if n_binders == 0 or n == 0:
        for frac in cutoff_fracs:
            key = f"EF@{frac * 100:g}%"
            out[key] = 0.0
        return out

    base_rate = n_binders / n
    ranks = np.argsort(np.argsort(-pred_all))
    br = ranks[binder_idx]

    for frac in cutoff_fracs:
        top_n = max(1, int(n * frac))
        hit = int((br < top_n).sum())
        observed_rate = hit / top_n
        ef = observed_rate / base_rate if base_rate > 0 else 0.0
        key = f"EF@{frac * 100:g}%"
        out[key] = ef
    return out


# ---------------------------------------------------------------------------
# Sustained crossing
# ---------------------------------------------------------------------------
def sustained_crossing(
    values: np.ndarray | list[float],
    budgets: np.ndarray | list[int],
    threshold: float,
) -> int | None:
    """First budget where *values* >= *threshold* and stays >= for all larger budgets.

    Used to determine the minimum training budget at which the surrogate
    reliably matches a target fraction of the oracle's quality.

    Returns the budget value (not index), or ``None`` if the threshold is
    never sustained.
    """
    values = np.asarray(values, dtype=np.float64)
    budgets = np.asarray(budgets)
    assert len(values) == len(budgets)

    order = np.argsort(budgets)
    values = values[order]
    budgets = budgets[order]

    for i in range(len(values)):
        if np.all(values[i:] >= threshold):
            return int(budgets[i])
    return None


# ---------------------------------------------------------------------------
# Convenience: compute all standard metrics in one call
# ---------------------------------------------------------------------------
def evaluate_surrogate(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    binder_idx: np.ndarray | None = None,
    pred_all: np.ndarray | None = None,
) -> dict[str, float]:
    """Compute Spearman + recall-at-k on (y_true, y_pred) and optionally
    binder ROC-AUC + binder counts on (pred_all, binder_idx).

    Parameters
    ----------
    y_true, y_pred : arrays over the *unlabeled* pool
    binder_idx : indices of known binders in the *full* pool
    pred_all : predictions over the *full* pool (needed for binder metrics)
    """
    out: dict[str, float] = {}
    out["spearman"] = spearman(y_true, y_pred)
    out.update(recall_at_k(y_true, y_pred))
    if binder_idx is not None and pred_all is not None:
        out["binder_roc_auc"] = binder_roc_auc(pred_all, binder_idx)
        out.update({k: float(v) for k, v in binder_counts(pred_all, binder_idx).items()})
    return out
