"""
04 — One-shot vs iterative Active Learning comparison (Report §2.4.4).

Runs six selection strategies on the same dataset and compares binder
ROC-AUC and Spearman correlation across AL cycles:

  1. One-shot + KMeans-200 stratification
  2. One-shot + purely random selection
  3. AL + cluster init + UCB acquisition
  4. AL + random init  + UCB acquisition
  5. AL + random init  + greedy acquisition
  6. AL + cluster init + greedy acquisition

Key finding: one-shot training matches or beats iterative AL on binder
ROC-AUC at equal label budget.  The two one-shot variants are nearly
identical (Δ ≈ 0.001), proving that clustering quality barely matters.

Usage:
    python scripts/04_oneshot_vs_al.py --data-dir data/sample
    python scripts/04_oneshot_vs_al.py --data-dir /path/to/full \\
        --seeds 42 7 13 100 2024 99 1 5 21 77
"""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import train_ensemble
from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices, cluster_initial
from lib.metrics import evaluate_surrogate, binder_roc_auc
from lib.clustering import kmeans_cluster
from lib.plotting import setup_style, save_figure, add_oracle_line


# ---------------------------------------------------------------------------
# Acquisition helpers
# ---------------------------------------------------------------------------
def _ucb_pick(mean: np.ndarray, std: np.ndarray, beta: float, n_batch: int) -> np.ndarray:
    score = mean + beta * std
    return np.argsort(score)[-n_batch:]


def _greedy_pick(mean: np.ndarray, n_batch: int) -> np.ndarray:
    return np.argsort(mean)[-n_batch:]


def _cluster_ucb_pick(
    mean: np.ndarray, std: np.ndarray, beta: float, n_batch: int,
    clusters: np.ndarray, n_clusters: int,
) -> np.ndarray:
    score = mean + beta * std
    order = np.argsort(score)[::-1]
    counts = np.zeros(n_clusters, dtype=np.int32)
    cap = max(1, n_batch // n_clusters + 1)
    chosen: list[int] = []
    while len(chosen) < n_batch:
        progressed = False
        for idx in order:
            if idx in set(chosen):
                continue
            c = int(clusters[idx])
            if counts[c] < cap:
                chosen.append(int(idx))
                counts[c] += 1
                progressed = True
                if len(chosen) >= n_batch:
                    break
        if not progressed:
            cap += 1
    return np.array(chosen, dtype=np.int64)


# ---------------------------------------------------------------------------
# Method definitions
# ---------------------------------------------------------------------------
METHODS = [
    {"name": "OneShot+KMeans",    "oneshot": True,  "init": "cluster", "acq": None},
    {"name": "OneShot+Random",    "oneshot": True,  "init": "random",  "acq": None},
    {"name": "AL+Cluster+UCB",    "oneshot": False, "init": "cluster", "acq": "ucb"},
    {"name": "AL+Random+UCB",     "oneshot": False, "init": "random",  "acq": "ucb"},
    {"name": "AL+Random+Greedy",  "oneshot": False, "init": "random",  "acq": "greedy"},
    {"name": "AL+Cluster+Greedy", "oneshot": False, "init": "cluster", "acq": "greedy"},
]


def _initial_selection(
    X: np.ndarray, y: np.ndarray, clusters: np.ndarray,
    frac: float, seed: int, mode: str,
) -> np.ndarray:
    n = len(y)
    n_sample = int(n * frac)
    if mode == "cluster":
        return cluster_initial(X, clusters, frac, seed)
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n, size=n_sample, replace=False)).astype(np.int64)


def _acquire(
    mean_pool: np.ndarray, std_pool: np.ndarray,
    clusters_pool: np.ndarray, n_clusters: int,
    n_batch: int, beta: float, acq: str,
) -> np.ndarray:
    if acq == "ucb":
        return _ucb_pick(mean_pool, std_pool, beta, n_batch)
    if acq == "greedy":
        return _greedy_pick(mean_pool, n_batch)
    if acq == "cluster_ucb":
        return _cluster_ucb_pick(mean_pool, std_pool, beta, n_batch, clusters_pool, n_clusters)
    raise ValueError(f"Unknown acquisition: {acq}")


# ---------------------------------------------------------------------------
# Single method run
# ---------------------------------------------------------------------------
def run_method(
    method: dict, X: np.ndarray, y: np.ndarray,
    binder_idx: np.ndarray, clusters: np.ndarray, n_clusters: int,
    init_frac: float, batch_frac: float, n_cycles: int,
    beta: float, seed: int, model_type: str,
) -> list[dict]:
    n = len(y)
    labeled = set(
        _initial_selection(X, y, clusters, init_frac, seed, method["init"]).tolist()
    )
    rows: list[dict] = []

    total_cycles = 1 if method["oneshot"] else n_cycles + 1
    for cycle in range(total_cycles):
        labeled_arr = np.array(sorted(labeled), dtype=np.int64)
        unlabeled_arr = np.array(sorted(set(range(n)) - labeled), dtype=np.int64)

        t0 = time.time()
        result = train_ensemble(model_type, X[labeled_arr], y[labeled_arr], X, seed=seed)
        elapsed = time.time() - t0

        m = evaluate_surrogate(
            y[unlabeled_arr], result.mean[unlabeled_arr],
            binder_idx=binder_idx, pred_all=result.mean,
        )
        row = {
            "method": method["name"], "seed": seed, "cycle": cycle,
            "pct": len(labeled) / n * 100, "n_labeled": len(labeled),
            "train_seconds": elapsed, **m,
        }
        rows.append(row)
        sp = m["spearman"]
        auc = m.get("binder_roc_auc", float("nan"))
        print(f"  [{method['name']:>20}] cycle={cycle} pct={row['pct']:5.1f}% "
              f"Sp={sp:.4f} AUC={auc:.3f} ({elapsed:.1f}s)", flush=True)

        if not method["oneshot"] and cycle < n_cycles:
            n_batch = max(1, int(n * batch_frac))
            pool_mean = result.mean[unlabeled_arr]
            pool_std = result.std[unlabeled_arr]
            pool_clusters = clusters[unlabeled_arr]
            local = _acquire(pool_mean, pool_std, pool_clusters, n_clusters,
                             n_batch, beta, method["acq"])
            new_idx = unlabeled_arr[local]
            labeled.update(new_idx.tolist())

        torch.cuda.empty_cache()
        gc.collect()

    return rows


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
METHOD_COLORS = {
    "OneShot+KMeans":    "#d62728",
    "OneShot+Random":    "#ff7f0e",
    "AL+Cluster+UCB":    "#1f77b4",
    "AL+Random+UCB":     "#2ca02c",
    "AL+Random+Greedy":  "#9467bd",
    "AL+Cluster+Greedy": "#8c564b",
}


def plot_results(df: pd.DataFrame, oracle_auc: float, output_dir: Path) -> None:
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for metric, ax, ylabel in [
        ("binder_roc_auc", axes[0], "Binder ROC-AUC"),
        ("spearman", axes[1], "Spearman ρ"),
    ]:
        for method_name in dict.fromkeys(df["method"]):
            sub = df[df["method"] == method_name]
            grouped = sub.groupby("pct")[metric].agg(["mean", "std"]).reset_index()
            color = METHOD_COLORS.get(method_name, "gray")
            is_oneshot = "OneShot" in method_name
            ax.errorbar(
                grouped["pct"], grouped["mean"], yerr=grouped["std"],
                marker="o" if is_oneshot else "s", markersize=5,
                linewidth=2.0 if is_oneshot else 1.2,
                color=color, label=method_name, capsize=3,
            )
        if metric == "binder_roc_auc":
            add_oracle_line(ax, oracle_auc)
        ax.set_xlabel("% labeled")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=8)

    fig.suptitle("One-shot vs Iterative Active Learning", fontsize=14)
    save_figure(fig, output_dir / "04_oneshot_vs_al.png")
    print(f"Saved plot -> {output_dir / '04_oneshot_vs_al.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="One-shot vs iterative AL (Report §2.4.4)")
    ap.add_argument("--data-dir", type=str, default="data/sample")
    ap.add_argument("--score-col", type=str, default="consensus_score")
    ap.add_argument("--binders", type=str, default=None)
    ap.add_argument("--init-frac", type=float, default=0.05)
    ap.add_argument("--batch-frac", type=float, default=0.02)
    ap.add_argument("--n-cycles", type=int, default=4)
    ap.add_argument("--beta", type=float, default=2.0)
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--model", type=str, default="betternn",
                    choices=["deepens", "boltznn", "betternn"])
    ap.add_argument("--output-dir", type=str, default="results")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    df = load_dataset(data_dir / "library.csv", target_cols=[args.score_col])
    y = df[args.score_col].values.astype(np.float32)
    n = len(y)

    binders_path = Path(args.binders) if args.binders else data_dir / "binders.csv"
    binder_ids = load_binders(binders_path)
    binder_idx = find_binder_indices(df, binder_ids)

    print("Loading fingerprints ...")
    fp = load_fingerprints(data_dir / "fingerprints")
    X = concat_fingerprints(fp.morgan, fp.atompair)
    print(f"  {n} molecules, X={X.shape}, {len(binder_idx)} binders")

    print("Clustering ...")
    clusters = kmeans_cluster(X, n_clusters=args.n_clusters, seed=42)
    n_clusters = int(clusters.max()) + 1

    oracle_auc = binder_roc_auc(y, binder_idx)
    print(f"Oracle binder ROC-AUC: {oracle_auc:.3f}")

    all_rows: list[dict] = []
    total = len(METHODS) * len(args.seeds)
    done = 0

    for seed in args.seeds:
        for method in METHODS:
            print(f"\n{'='*60}")
            print(f"seed={seed}  method={method['name']}  [{done+1}/{total}]")
            print(f"{'='*60}")
            rows = run_method(
                method, X, y, binder_idx, clusters, n_clusters,
                args.init_frac, args.batch_frac, args.n_cycles,
                args.beta, seed, args.model,
            )
            all_rows.extend(rows)
            done += 1
            pd.DataFrame(all_rows).to_csv(output_dir / "04_oneshot_vs_al_long.csv", index=False)

    long_df = pd.DataFrame(all_rows)
    csv_path = output_dir / "04_oneshot_vs_al_long.csv"
    long_df.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")

    summary = (
        long_df.groupby("method")
        .agg(
            spearman_mean=("spearman", "mean"),
            spearman_std=("spearman", "std"),
            auc_mean=("binder_roc_auc", "mean"),
            auc_std=("binder_roc_auc", "std"),
        )
        .round(4)
    )
    print(f"\n{'='*60}\nSummary (mean across all seeds and cycles)\n{'='*60}")
    print(summary.to_string())

    plot_results(long_df, oracle_auc, output_dir)


if __name__ == "__main__":
    main()
