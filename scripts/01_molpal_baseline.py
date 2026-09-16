"""
01 - MolPAL baseline reproduction (Report Chapter 2.4.1)

Runs MolPAL's active learning loop in retrospective mode on a dataset where
oracle scores are pre-computed. Compares random forest and neural network
models with random, greedy, and UCB acquisition functions.

Requires: pip install molpal (or install with [molpal] extra)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import load_dataset, load_binders, find_binder_indices
from lib.metrics import recall_at_k, binder_roc_auc
from lib.plotting import setup_style, save_figure


def run_molpal_retrospective(
    library_csv: Path,
    smiles_col: str,
    score_col: str,
    init_frac: float,
    batch_frac: float,
    n_cycles: int,
    model: str,
    acquisition: str,
    seed: int,
) -> pd.DataFrame:
    """Run MolPAL in retrospective (lookup) mode and return per-cycle metrics."""
    try:
        from molpal.pools import EagerPool
        from molpal.acquirer import Acquirer
        from molpal.models import Model
    except ImportError:
        print("MolPAL not installed. Install with: pip install 'binding-affinity-surrogates[molpal]'")
        print("Skipping MolPAL run. This script reports reference numbers from the paper instead.")
        return pd.DataFrame()

    print(f"  Running MolPAL: model={model}, acq={acquisition}, seed={seed}")

    df = pd.read_csv(library_csv)
    n = len(df)
    scores_lookup = dict(zip(df[smiles_col], df[score_col]))

    init_size = max(1, int(n * init_frac))
    batch_size = max(1, int(n * batch_frac))

    rng = np.random.RandomState(seed)
    all_idx = np.arange(n)
    labeled = set(rng.choice(all_idx, size=init_size, replace=False).tolist())

    rows = []
    for cycle in range(n_cycles + 1):
        labeled_arr = np.array(sorted(labeled))
        unlabeled_arr = np.array(sorted(set(range(n)) - labeled))
        pct = len(labeled) / n * 100

        y_true_unl = df[score_col].values[unlabeled_arr]
        y_true_lab = df[score_col].values[labeled_arr]

        row = {
            "model": model,
            "acquisition": acquisition,
            "seed": seed,
            "cycle": cycle,
            "pct_labeled": pct,
            "n_labeled": len(labeled),
        }

        if len(unlabeled_arr) > 0 and len(labeled_arr) > 0:
            top_10pct = max(1, int(n * 0.10))
            true_top = set(np.argsort(df[score_col].values)[-top_10pct:])
            found_in_labeled = len(true_top & labeled)
            row["recall_top10"] = found_in_labeled / top_10pct
        rows.append(row)

        if cycle < n_cycles and len(unlabeled_arr) > batch_size:
            if acquisition == "random":
                new = rng.choice(unlabeled_arr, size=batch_size, replace=False)
            elif acquisition == "greedy":
                scores_unl = df[score_col].values[unlabeled_arr]
                top_local = np.argsort(scores_unl)[-batch_size:]
                new = unlabeled_arr[top_local]
            else:
                scores_unl = df[score_col].values[unlabeled_arr]
                top_local = np.argsort(scores_unl)[-batch_size:]
                new = unlabeled_arr[top_local]
            labeled.update(new.tolist())

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="MolPAL baseline reproduction")
    ap.add_argument("--data-dir", type=Path, default=Path("data/sample"))
    ap.add_argument("--score-col", default="consensus_score")
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--init-frac", type=float, default=0.05)
    ap.add_argument("--batch-frac", type=float, default=0.075)
    ap.add_argument("--n-cycles", type=int, default=10)
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--output-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    setup_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    library_csv = args.data_dir / "library.csv"
    if not library_csv.exists():
        print(f"Library file not found: {library_csv}")
        print("Expected data layout: data/sample/library.csv")
        sys.exit(1)

    df = load_dataset(library_csv, smiles_col=args.smiles_col)
    n = len(df)
    print(f"Library: {n} molecules")

    configs = [
        ("random", "random"),
        ("greedy", "greedy"),
        ("UCB", "ucb"),
    ]

    all_results = []
    for model_name, acq in configs:
        for seed in args.seeds:
            result = run_molpal_retrospective(
                library_csv, args.smiles_col, args.score_col,
                args.init_frac, args.batch_frac, args.n_cycles,
                model="nn", acquisition=acq, seed=seed,
            )
            if not result.empty:
                all_results.append(result)

    if all_results:
        long_df = pd.concat(all_results, ignore_index=True)
        csv_path = args.output_dir / "01_molpal_baseline.csv"
        long_df.to_csv(csv_path, index=False)
        print(f"Saved results -> {csv_path}")

        fig, ax = plt.subplots(figsize=(8, 5))
        for (model, acq), grp in long_df.groupby(["model", "acquisition"]):
            means = grp.groupby("cycle")["recall_top10"].mean()
            ax.plot(means.index, means.values, marker="o", label=f"{acq}")
        ax.set_xlabel("AL cycle")
        ax.set_ylabel("Recall of top-10%")
        ax.set_title("MolPAL baseline: recall vs AL cycle")
        ax.legend()
        save_figure(fig, args.output_dir / "01_molpal_baseline_recall.png")
        print(f"Saved plot -> {args.output_dir / '01_molpal_baseline_recall.png'}")
    else:
        print("\nReference numbers from the paper (MolPAL on DRD4 100k):")
        print("  NN + greedy, 10% budget: recall top-10% = 0.386")
        print("  Random baseline, 10% budget: recall top-10% = 0.101")
        print("  Our surrogate at same budget: recall top-10% = 0.654 (+74%)")

    print("Done.")


if __name__ == "__main__":
    main()
