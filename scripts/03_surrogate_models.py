"""
Chapter 2.4.3 — Development and comparison of surrogate models.

Trains three architectures (DeepEns_concat, BoltzNN_v1_wide, BetterNN) at
multiple training budgets in one-shot mode (no AL cycles).  Evaluates with
Spearman, R@k%, binder ROC-AUC, and binder counts.  Outputs a long-form CSV,
a summary table, and two comparison plots.

Usage:
    python scripts/03_surrogate_models.py --data-dir data/sample
    python scripts/03_surrogate_models.py --data-dir /path/to/full/adra2b \\
        --budgets 0.001 0.005 0.01 0.02 0.05 0.11 0.20 \\
        --seeds 42 7 13 100 2024 99 1 5 21 77
"""

from __future__ import annotations

import argparse
import gc
import time
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import train_ensemble
from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices, cluster_initial
from lib.metrics import evaluate_surrogate, binder_roc_auc
from lib.clustering import kmeans_cluster
from lib.plotting import setup_style, save_figure, add_oracle_line, MODEL_COLORS

ARM_DISPLAY = {
    "deepens": "DeepEns_concat",
    "boltznn": "BoltzNN_v1_wide",
    "betternn": "BetterNN",
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data/sample",
                    help="Directory with library.csv, binders.csv, fingerprints/")
    ap.add_argument("--score-col", default="consensus_score",
                    help="Target score column")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--id-col", default="zincid")
    ap.add_argument("--binders", default=None,
                    help="Path to binders CSV (default: <data-dir>/binders.csv)")
    ap.add_argument("--budgets", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.11, 0.20])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--arms", nargs="+", choices=["deepens", "boltznn", "betternn"],
                    default=["deepens", "boltznn", "betternn"])
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--n-models", type=int, default=5)
    ap.add_argument("--output-dir", default="results",
                    help="Directory for CSV and plots")
    return ap.parse_args()


def load_data(args: argparse.Namespace):
    data_dir = Path(args.data_dir)
    csv_path = data_dir / "library.csv"
    binders_path = Path(args.binders) if args.binders else data_dir / "binders.csv"
    fp_dir = data_dir / "fingerprints"

    df = load_dataset(csv_path, smiles_col=args.smiles_col,
                      target_cols=[args.score_col], id_col=args.id_col)
    y = df[args.score_col].values.astype(np.float32)

    binder_ids = load_binders(binders_path, id_col=args.id_col)
    binder_idx = find_binder_indices(df, binder_ids, id_col=args.id_col)

    fp = load_fingerprints(fp_dir)
    valid = fp.valid
    X_concat = concat_fingerprints(fp.morgan[valid], fp.atompair[valid])
    X_desc = fp.descriptors[valid]
    X_full = np.hstack([X_concat, X_desc]).astype(np.float32)

    y = y[valid]
    binder_idx_set = set(binder_idx.tolist())
    valid_indices = np.where(valid)[0]
    remap = {old: new for new, old in enumerate(valid_indices)}
    binder_idx = np.array([remap[b] for b in binder_idx if b in remap], dtype=np.int64)

    return X_concat, X_full, y, binder_idx


def run_budget_scan(args: argparse.Namespace):
    print(f"Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    X_concat, X_full, y, binder_idx = load_data(args)
    n = len(y)
    print(f"Loaded {n:,} molecules, {len(binder_idx)} binders")

    oracle_auc = binder_roc_auc(y, binder_idx)
    print(f"Oracle binder ROC-AUC: {oracle_auc:.4f}")

    print(f"Clustering (k={args.n_clusters}) ...")
    clusters = kmeans_cluster(X_concat, n_clusters=args.n_clusters, seed=42)

    input_map = {
        "deepens": X_concat,
        "boltznn": X_full,
        "betternn": X_concat,
    }

    total = len(args.budgets) * len(args.seeds) * len(args.arms)
    rows: list[dict] = []
    pbar = tqdm(total=total, desc="Training")

    for budget in args.budgets:
        for seed in args.seeds:
            labeled = cluster_initial(X_concat, clusters, budget, seed)
            unlabeled = np.array(sorted(set(range(n)) - set(labeled.tolist())),
                                 dtype=np.int64)

            for arm_key in args.arms:
                arm_name = ARM_DISPLAY[arm_key]
                X_in = input_map[arm_key]

                t0 = time.time()
                result = train_ensemble(
                    arm_key, X_in[labeled], y[labeled], X_in,
                    seed=seed, n_models=args.n_models,
                )
                elapsed = time.time() - t0

                m = evaluate_surrogate(
                    y[unlabeled], result.mean[unlabeled],
                    binder_idx=binder_idx, pred_all=result.mean,
                )
                row = {
                    "arm": arm_name,
                    "seed": seed,
                    "budget": budget,
                    "n_labeled": len(labeled),
                    "pct": len(labeled) / n * 100,
                    "train_seconds": round(elapsed, 1),
                    **m,
                }
                rows.append(row)

                pbar.set_postfix_str(
                    f"{arm_name} b={budget:.0%} Sp={m['spearman']:.3f}"
                )
                pbar.update(1)

                torch.cuda.empty_cache()
                gc.collect()

    pbar.close()
    return pd.DataFrame(rows), oracle_auc


def make_plots(df: pd.DataFrame, oracle_auc: float, output_dir: Path):
    setup_style()

    for metric, ylabel in [("binder_roc_auc", "Binder ROC-AUC"),
                            ("spearman", "Spearman ρ")]:
        fig, ax = plt.subplots()
        for arm_name in df["arm"].unique():
            sub = df[df["arm"] == arm_name]
            agg = sub.groupby("budget")[metric].agg(["mean", "std"]).reset_index()
            color = MODEL_COLORS.get(arm_name, None)
            ax.errorbar(agg["budget"] * 100, agg["mean"], yerr=agg["std"],
                        marker="o", capsize=3, label=arm_name, color=color)
        if metric == "binder_roc_auc":
            add_oracle_line(ax, oracle_auc)
        ax.set_xlabel("Training budget (% of library)")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.set_title("Surrogate model comparison — one-shot training")
        tag = "roc_auc" if metric == "binder_roc_auc" else metric
        save_figure(fig, output_dir / f"03_model_comparison_{tag}.png")
        print(f"Saved {output_dir / f'03_model_comparison_{tag}.png'}")


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    long_df, oracle_auc = run_budget_scan(args)

    csv_path = output_dir / "03_surrogate_models_long.csv"
    long_df.to_csv(csv_path, index=False)
    print(f"\nLong-form results -> {csv_path}")

    summary = (long_df.groupby(["arm", "budget"])
               .agg(spearman_mean=("spearman", "mean"),
                    spearman_std=("spearman", "std"),
                    roc_auc_mean=("binder_roc_auc", "mean"),
                    roc_auc_std=("binder_roc_auc", "std"),
                    train_s=("train_seconds", "mean"))
               .round(4))
    summary_path = output_dir / "03_surrogate_models_summary.csv"
    summary.to_csv(summary_path)
    print(f"Summary -> {summary_path}")
    print(summary.to_string())

    make_plots(long_df, oracle_auc, output_dir)


if __name__ == "__main__":
    main()
