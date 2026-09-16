"""
05 - Budget estimation and acceleration factor (Report Ch 2.4.5)

Scans training budgets across multiple library sizes to answer: how many
oracle calls does the surrogate need, and how does that scale with the pool?

Outputs:
    {output_dir}/budget_scan_long.csv          per-(pool_size, budget, seed) metrics
    {output_dir}/acceleration_summary.csv      sustained-crossing acceleration per pool size
    {output_dir}/budget_roc_vs_budget.png       ROC-AUC vs budget, one panel per pool size
    {output_dir}/budget_acceleration.png        acceleration vs pool size
"""

import argparse
import gc
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import train_ensemble
from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices, cluster_initial
from lib.metrics import binder_roc_auc, evaluate_surrogate, sustained_crossing
from lib.clustering import kmeans_cluster
from lib.plotting import setup_style, save_figure, add_oracle_line, MODEL_COLORS


def subsample_pool(n_full: int, pool_size: int, binder_idx: np.ndarray,
                   seed: int) -> np.ndarray:
    """Random subsample of the pool that always includes all binders."""
    rng = np.random.RandomState(seed)
    non_binder = np.array(sorted(set(range(n_full)) - set(binder_idx.tolist())))
    n_need = pool_size - len(binder_idx)
    if n_need <= 0:
        return binder_idx.copy()
    chosen = rng.choice(non_binder, size=min(n_need, len(non_binder)), replace=False)
    return np.sort(np.concatenate([binder_idx, chosen]))


def main():
    ap = argparse.ArgumentParser(description="Budget scan and acceleration factor")
    ap.add_argument("--data-dir", type=str, default="data/sample",
                    help="Directory with library.csv and fingerprints")
    ap.add_argument("--binders", type=str, default=None,
                    help="Binder CSV path (default: data-dir/binders.csv)")
    ap.add_argument("--score-col", type=str, default="consensus_score",
                    help="Column name for the oracle score")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20])
    ap.add_argument("--pool-sizes", type=int, nargs="+",
                    default=[10000, 20000, 50000, 100000])
    ap.add_argument("--threshold", type=float, default=0.90,
                    help="Fraction of oracle ROC-AUC to consider 'good enough'")
    ap.add_argument("--model", choices=["deepens", "boltznn", "betternn"],
                    default="betternn")
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--output-dir", type=str, default="results")
    args = ap.parse_args()

    setup_style()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binders_path = args.binders or str(data_dir / "binders.csv")

    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("Loading data ...")
    df = load_dataset(data_dir / "library.csv", target_cols=[args.score_col])
    y = df[args.score_col].values.astype(np.float32)
    n_full = len(y)

    binder_ids = load_binders(binders_path)
    binder_idx = find_binder_indices(df, binder_ids)
    print(f"  Rows: {n_full};  binders in pool: {len(binder_idx)}")

    fp = load_fingerprints(data_dir / "fingerprints")
    X_full = concat_fingerprints(fp.morgan, fp.atompair)

    all_rows: list[dict] = []
    pool_sizes = [ps for ps in args.pool_sizes if ps <= n_full]
    if n_full not in pool_sizes:
        pool_sizes.append(n_full)
    pool_sizes = sorted(pool_sizes)

    for ps in pool_sizes:
        if ps >= n_full:
            pool_idx = np.arange(n_full)
        else:
            pool_idx = subsample_pool(n_full, ps, binder_idx, seed=0)

        y_pool = y[pool_idx]
        X_pool = X_full[pool_idx]
        local_binder = np.array([
            i for i, gi in enumerate(pool_idx) if gi in set(binder_idx.tolist())
        ], dtype=np.int64)

        oracle_auc = binder_roc_auc(y_pool, local_binder)
        target_auc = args.threshold * oracle_auc
        print(f"\n=== pool_size={ps}  oracle_AUC={oracle_auc:.4f}  "
              f"target={target_auc:.4f} ===")

        clusters = kmeans_cluster(X_pool, n_clusters=min(args.n_clusters, ps))

        for budget in args.budgets:
            n_train = max(1, int(ps * budget))
            if n_train >= ps:
                continue
            for seed in args.seeds:
                labeled = cluster_initial(X_pool, clusters, budget, seed)
                unlabeled = np.array(sorted(set(range(ps)) - set(labeled.tolist())))

                t0 = time.time()
                result = train_ensemble(
                    args.model, X_pool[labeled], y_pool[labeled],
                    X_pool, seed=seed, n_models=5,
                )
                elapsed = time.time() - t0

                auc = binder_roc_auc(result.mean, local_binder)
                m = evaluate_surrogate(y_pool[unlabeled], result.mean[unlabeled],
                                       local_binder, result.mean)
                row = {
                    "pool_size": ps, "budget": budget, "n_train": len(labeled),
                    "seed": seed, "binder_auc": auc, "oracle_auc": oracle_auc,
                    "train_seconds": elapsed, **m,
                }
                all_rows.append(row)
                print(f"  ps={ps} b={budget:.3f} n={len(labeled):>6} s={seed} "
                      f"AUC={auc:.4f} Sp={m['spearman']:.4f} ({elapsed:.1f}s)",
                      flush=True)

                torch.cuda.empty_cache()
                gc.collect()

        pd.DataFrame(all_rows).to_csv(out_dir / "budget_scan_long.csv", index=False)

    long_df = pd.DataFrame(all_rows)
    long_df.to_csv(out_dir / "budget_scan_long.csv", index=False)

    # --- Acceleration summary ---
    accel_rows: list[dict] = []
    for ps in pool_sizes:
        sub = long_df[long_df["pool_size"] == ps]
        if sub.empty:
            continue
        mean_by_budget = sub.groupby("budget")["binder_auc"].agg(["mean", "std"])
        oracle_auc = sub["oracle_auc"].iloc[0]
        target_auc = args.threshold * oracle_auc

        budgets_arr = mean_by_budget.index.values
        values_arr = mean_by_budget["mean"].values
        low_arr = values_arr - mean_by_budget["std"].values

        n_cross = sustained_crossing(low_arr, budgets_arr, target_auc)
        if n_cross is not None:
            n_train_cross = max(1, int(ps * n_cross))
            acceleration = ps / n_train_cross
        else:
            n_train_cross = None
            acceleration = None
        accel_rows.append({
            "pool_size": ps, "oracle_auc": oracle_auc,
            "crossing_budget": n_cross, "n_train": n_train_cross,
            "acceleration": acceleration,
        })

    accel_df = pd.DataFrame(accel_rows)
    accel_df.to_csv(out_dir / "acceleration_summary.csv", index=False)
    print(f"\nAcceleration summary:\n{accel_df.to_string(index=False)}")

    # --- Plot 1: ROC-AUC vs budget per pool size ---
    fig, axes = plt.subplots(1, len(pool_sizes), figsize=(5 * len(pool_sizes), 4.5),
                             squeeze=False)
    for ax, ps in zip(axes[0], pool_sizes):
        sub = long_df[long_df["pool_size"] == ps]
        grp = sub.groupby("budget")["binder_auc"].agg(["mean", "std"])
        ax.errorbar(grp.index * 100, grp["mean"], yerr=grp["std"],
                    marker="o", capsize=3, label=args.model)
        oracle_auc = sub["oracle_auc"].iloc[0]
        add_oracle_line(ax, oracle_auc)
        add_oracle_line(ax, args.threshold * oracle_auc, label=f"{args.threshold:.0%} Oracle",
                        color="gray")
        ax.set_xlabel("Budget (% of pool)")
        ax.set_ylabel("Binder ROC-AUC")
        ax.set_title(f"Pool = {ps:,}")
        ax.legend(fontsize=8)
        ax.set_xscale("log")
    save_figure(fig, out_dir / "budget_roc_vs_budget.png")

    # --- Plot 2: acceleration vs pool size ---
    valid = accel_df.dropna(subset=["acceleration"])
    if not valid.empty:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        ax.plot(valid["pool_size"], valid["acceleration"], marker="s", linewidth=2)
        ax.set_xlabel("Library size")
        ax.set_ylabel(f"Acceleration (@ {args.threshold:.0%} of oracle)")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title("Surrogate acceleration vs library size")
        save_figure(fig, out_dir / "budget_acceleration.png")

    print("Done.")


if __name__ == "__main__":
    main()
