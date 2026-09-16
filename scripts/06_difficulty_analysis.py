"""
06 - Learning difficulty analysis (Report Ch 2.4.6)

Analyzes why some targets are harder to learn than others using structural
diversity metrics: sphere exclusion clustering, singleton fraction, and
binder isolation at different score-depth cuts.

Outputs:
    {output_dir}/difficulty_metrics.csv          per-(depth, threshold) metrics
    {output_dir}/difficulty_singleton_depth.png   singleton% vs score depth
    {output_dir}/difficulty_binder_isolation.png  binder isolation scatter
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices
from lib.clustering import sphere_exclusion, count_singletons, binder_isolation
from lib.plotting import setup_style, save_figure


def main():
    ap = argparse.ArgumentParser(description="Learning difficulty analysis")
    ap.add_argument("--data-dir", type=str, default="data/sample",
                    help="Directory with library.csv and fingerprints")
    ap.add_argument("--binders", type=str, default=None,
                    help="Binder CSV path (default: data-dir/binders.csv)")
    ap.add_argument("--score-col", type=str, default="consensus_score",
                    help="Column name for the oracle score")
    ap.add_argument("--id-col", type=str, default="zincid")
    ap.add_argument("--smiles-col", type=str, default="smiles")
    ap.add_argument("--thresholds", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                    help="Tanimoto similarity thresholds for sphere exclusion")
    ap.add_argument("--depth-cuts", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 1.0],
                    help="Score percentile cuts (fraction of library)")
    ap.add_argument("--output-dir", type=str, default="results")
    args = ap.parse_args()

    setup_style()
    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    binders_path = args.binders or str(data_dir / "binders.csv")

    print("Loading data ...")
    df = load_dataset(data_dir / "library.csv", smiles_col=args.smiles_col,
                      target_cols=[args.score_col], id_col=args.id_col)
    y = df[args.score_col].values.astype(np.float32)
    n = len(y)

    binder_ids = load_binders(binders_path, id_col=args.id_col)
    binder_idx = find_binder_indices(df, binder_ids, id_col=args.id_col)
    binder_set = set(binder_idx.tolist())
    print(f"  Rows: {n};  binders: {len(binder_idx)}")

    fp = load_fingerprints(data_dir / "fingerprints")
    X_morgan = fp.morgan

    score_order = np.argsort(-y)

    all_rows: list[dict] = []

    for depth in args.depth_cuts:
        n_cut = max(1, int(n * depth))
        cut_idx = score_order[:n_cut]
        binder_in_cut = np.array([
            i for i, gi in enumerate(cut_idx) if gi in binder_set
        ], dtype=np.int64)

        fps_cut = X_morgan[cut_idx]
        print(f"\n--- depth={depth*100:.1f}% ({n_cut} molecules, "
              f"{len(binder_in_cut)} binders) ---")

        for threshold in args.thresholds:
            leaders, labels = sphere_exclusion(fps_cut, threshold=threshold)
            n_clusters = len(leaders)

            sing = count_singletons(fps_cut, threshold=threshold)
            iso = binder_isolation(fps_cut, binder_in_cut, threshold=threshold)

            row = {
                "depth": depth,
                "depth_pct": f"{depth*100:.1f}%",
                "n_molecules": n_cut,
                "tanimoto_threshold": threshold,
                "n_clusters": n_clusters,
                "n_singletons": sing["n_singletons"],
                "pct_singletons": sing["pct_singletons"],
                "n_binders_in_cut": len(binder_in_cut),
                "n_binders_isolated": iso["n_isolated"],
                "pct_binders_isolated": iso["pct_isolated"],
            }
            all_rows.append(row)
            print(f"  T={threshold:.1f}: {n_clusters} clusters, "
                  f"{sing['pct_singletons']:.1f}% singletons, "
                  f"{iso['n_isolated']}/{len(binder_in_cut)} binders isolated")

    results = pd.DataFrame(all_rows)
    results.to_csv(out_dir / "difficulty_metrics.csv", index=False)
    print(f"\nMetrics saved -> {out_dir / 'difficulty_metrics.csv'}")

    # --- Plot 1: singleton% vs depth for each threshold ---
    fig, ax = plt.subplots(figsize=(7, 5))
    for threshold in args.thresholds:
        sub = results[results["tanimoto_threshold"] == threshold]
        ax.plot(sub["depth"] * 100, sub["pct_singletons"],
                marker="o", label=f"T={threshold:.1f}")
    ax.set_xlabel("Score depth (top % of library)")
    ax.set_ylabel("Singleton fraction (%)")
    ax.set_title("Structural diversity by score depth")
    ax.legend()
    ax.set_xscale("log")
    save_figure(fig, out_dir / "difficulty_singleton_depth.png")

    # --- Plot 2: binder isolation at each depth ---
    fig, ax = plt.subplots(figsize=(7, 5))
    ref_t = args.thresholds[len(args.thresholds) // 2]
    sub = results[results["tanimoto_threshold"] == ref_t]
    ax.bar(range(len(sub)), sub["pct_binders_isolated"],
           tick_label=[f"{d*100:.0f}%" for d in sub["depth"]])
    ax.set_xlabel("Score depth cut")
    ax.set_ylabel(f"Binders isolated (%, Tanimoto < {ref_t})")
    ax.set_title("Binder isolation by score depth")
    save_figure(fig, out_dir / "difficulty_binder_isolation.png")

    # --- Summary table ---
    print("\n=== Summary (middle threshold) ===")
    cols = ["depth_pct", "n_molecules", "n_clusters", "pct_singletons",
            "n_binders_isolated", "pct_binders_isolated"]
    print(sub[cols].to_string(index=False))
    print("\nDone.")


if __name__ == "__main__":
    main()
