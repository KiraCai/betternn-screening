"""
08 - Scaling to very large libraries (report chapter 2.4.8).

Demonstrates:
  1. Surrogate training at different sample sizes
  2. Streaming prediction in batches (memory-bounded)
  3. Enrichment vs training size
  4. Difficulty probe: self-consistency recall predicts learning difficulty
"""

import argparse
import gc
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import train_ensemble, build_model, train_single, predict_single, DEFAULT_CONFIGS, DEVICE
from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices
from lib.metrics import binder_roc_auc, enrichment_factor, recall_at_k
from lib.plotting import setup_style, save_figure, add_oracle_line


@torch.no_grad()
def streaming_predict(models, X_path_or_array, y_mean, y_std, batch_size=8192):
    """Predict in batches, accumulating only the result array."""
    if isinstance(X_path_or_array, np.ndarray):
        X = X_path_or_array
    else:
        X = np.load(X_path_or_array, mmap_mode="r")
    n = len(X)
    preds = np.zeros(n, dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        chunk = X[start:end].astype(np.float32)
        Xt = torch.from_numpy(chunk).to(DEVICE)
        batch_preds = []
        for model in models:
            model.eval()
            out = model(Xt)
            if isinstance(out, tuple):
                out = out[0]
            batch_preds.append(out.cpu().numpy() * y_std + y_mean)
        preds[start:end] = np.mean(batch_preds, axis=0)
        del Xt
    return preds


def difficulty_probe(X, y, train_frac=0.01, recall_frac=0.0001, seed=42, n_models=5):
    """Self-consistency probe: train on train_frac, measure recall of top recall_frac."""
    n = len(y)
    rng = np.random.RandomState(seed)
    n_train = max(10, int(n * train_frac))
    train_idx = rng.choice(n, size=n_train, replace=False)
    test_idx = np.array(sorted(set(range(n)) - set(train_idx)))

    ens = train_ensemble("betternn", X[train_idx], y[train_idx], X[test_idx],
                         seed=seed, n_models=n_models)
    pred = ens.mean
    y_test = y[test_idx]

    n_top = max(1, int(len(y_test) * recall_frac))
    true_top = set(np.argsort(y_test)[-n_top:])
    pred_top = set(np.argsort(pred)[-n_top:])
    recall = len(true_top & pred_top) / n_top if n_top > 0 else 0.0
    return recall


def main():
    ap = argparse.ArgumentParser(description="Large-scale screening and difficulty probes")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--binders", type=str, required=True)
    ap.add_argument("--score-col", type=str, default="consensus_score")
    ap.add_argument("--id-col", type=str, default="zincid")
    ap.add_argument("--smiles-col", type=str, default="smiles")
    ap.add_argument("--train-sizes", type=int, nargs="+",
                    default=[100, 500, 1000, 5000, 50000])
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--probe-train-frac", type=float, default=0.01)
    ap.add_argument("--probe-recall-frac", type=float, default=0.0001)
    ap.add_argument("--model", type=str, default="betternn")
    ap.add_argument("--output-dir", type=str, default="results/08_large_scale")
    args = ap.parse_args()

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("Loading data ...")
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
    df = load_dataset(csv_files[0], smiles_col=args.smiles_col,
                      target_cols=[args.score_col])
    y = df[args.score_col].values.astype(np.float32)
    n = len(y)
    binder_ids = load_binders(args.binders, id_col=args.id_col)
    binder_idx = find_binder_indices(df, binder_ids, id_col=args.id_col)
    print(f"  {n:,} molecules, {len(binder_idx)} binders")

    print("Loading fingerprints ...")
    fp = load_fingerprints(data_dir / "fingerprints")
    X = concat_fingerprints(fp.morgan, fp.atompair)

    oracle_auc = binder_roc_auc(y, binder_idx)
    oracle_ef = enrichment_factor(y, binder_idx, (0.001, 0.01, 0.10))
    print(f"Oracle ROC-AUC: {oracle_auc:.3f}, EF: {oracle_ef}")

    # --- Part 1: EF vs training size ---
    print("\n=== Part 1: Enrichment vs training size ===")
    all_rows = []
    for n_train in args.train_sizes:
        if n_train >= n:
            print(f"  Skipping n_train={n_train} (>= pool size {n})")
            continue
        for seed in args.seeds:
            rng = np.random.RandomState(seed)
            train_idx = rng.choice(n, size=n_train, replace=False)

            ens = train_ensemble(args.model, X[train_idx], y[train_idx], X,
                                 seed=seed, n_models=5)
            pred = ens.mean

            auc = binder_roc_auc(pred, binder_idx)
            ef = enrichment_factor(pred, binder_idx, (0.001, 0.01, 0.10))
            row = {"n_train": n_train, "seed": seed, "binder_roc_auc": auc, **ef}
            all_rows.append(row)
            print(f"  n_train={n_train:>6}, seed={seed:>3}: "
                  f"AUC={auc:.3f}, EF@1%={ef.get('EF@1%', 0):.1f}")
            gc.collect()
            torch.cuda.empty_cache()

    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "large_scale_results.csv", index=False)

    # Plot: ROC-AUC vs training size
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    summary = results.groupby("n_train").agg(
        auc_mean=("binder_roc_auc", "mean"), auc_std=("binder_roc_auc", "std")
    )
    axes[0].errorbar(summary.index, summary["auc_mean"], yerr=summary["auc_std"],
                     marker="o", capsize=3)
    add_oracle_line(axes[0], oracle_auc)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Training molecules")
    axes[0].set_ylabel("Binder ROC-AUC")
    axes[0].set_title("ROC-AUC vs training size")

    ef_col = "EF@1%"
    if ef_col in results.columns:
        ef_summary = results.groupby("n_train").agg(
            ef_mean=(ef_col, "mean"), ef_std=(ef_col, "std")
        )
        axes[1].errorbar(ef_summary.index, ef_summary["ef_mean"],
                         yerr=ef_summary["ef_std"], marker="o", capsize=3)
        add_oracle_line(axes[1], oracle_ef.get(ef_col, 0))
        axes[1].set_xscale("log")
        axes[1].set_xlabel("Training molecules")
        axes[1].set_ylabel("Enrichment Factor @ 1%")
        axes[1].set_title("Enrichment vs training size")
    save_figure(fig, out_dir / "large_scale_ef_vs_trainsize.png")

    # --- Part 2: Difficulty probe ---
    print("\n=== Part 2: Difficulty probe (self-consistency) ===")
    probe_rows = []
    for seed in args.seeds:
        recall = difficulty_probe(
            X, y,
            train_frac=args.probe_train_frac,
            recall_frac=args.probe_recall_frac,
            seed=seed,
        )
        probe_rows.append({"seed": seed, "probe_recall": recall})
        print(f"  seed={seed}: probe recall@{args.probe_recall_frac*100:.2f}% = {recall:.3f}")

    probe_df = pd.DataFrame(probe_rows)
    probe_df.to_csv(out_dir / "difficulty_probe.csv", index=False)
    print(f"Mean probe recall: {probe_df['probe_recall'].mean():.3f} "
          f"(+/- {probe_df['probe_recall'].std():.3f})")
    print(f"\nAll results saved to {out_dir}")


if __name__ == "__main__":
    main()
