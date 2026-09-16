"""
07 - Two-stage screening evaluation (report chapter 2.4.7).

Cheap surrogate picks a short list; expensive oracle rescores only that list.
Compares surrogate vs raw docking as the first-stage ranker.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.models import train_ensemble
from lib.features import load_fingerprints, concat_fingerprints
from lib.data import load_dataset, load_binders, find_binder_indices, cluster_initial
from lib.metrics import binder_roc_auc, enrichment_factor
from lib.clustering import kmeans_cluster
from lib.plotting import setup_style, save_figure, add_oracle_line, MODEL_COLORS


def _ef_on_shortlist(oracle_scores, binder_mask, shortlist_mask, final_fracs):
    """Enrichment factor within a shortlist rescored by oracle."""
    n_total = len(oracle_scores)
    n_binders = int(binder_mask.sum())
    if n_binders == 0 or not shortlist_mask.any():
        return {f"EF@{f*100:g}%": 0.0 for f in final_fracs}

    base_rate = n_binders / n_total
    sl_scores = np.full(n_total, -np.inf)
    sl_scores[shortlist_mask] = oracle_scores[shortlist_mask]

    ranks = np.argsort(np.argsort(-sl_scores))
    br = ranks[binder_mask]
    out = {}
    for frac in final_fracs:
        top_n = max(1, int(n_total * frac))
        hit = int((br < top_n).sum())
        out[f"EF@{frac*100:g}%"] = (hit / top_n) / base_rate if base_rate > 0 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser(description="Two-stage screening evaluation")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--binders", type=str, required=True)
    ap.add_argument("--score-col", type=str, default="consensus_score")
    ap.add_argument("--docking-col", type=str, default="docking_score")
    ap.add_argument("--id-col", type=str, default="zincid")
    ap.add_argument("--smiles-col", type=str, default="smiles")
    ap.add_argument("--seeds", type=int, nargs="+",
                    default=[42, 7, 13, 100, 2024, 99, 1, 5, 21, 77])
    ap.add_argument("--budgets", type=float, nargs="+", default=[0.001, 0.005, 0.01, 0.05])
    ap.add_argument("--shortlist-fracs", type=float, nargs="+",
                    default=[0.01, 0.02, 0.05, 0.10, 0.20, 0.50])
    ap.add_argument("--final-fracs", type=float, nargs="+",
                    default=[0.001, 0.005, 0.01, 0.05])
    ap.add_argument("--model", type=str, default="betternn",
                    choices=["deepens", "boltznn", "betternn"])
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--output-dir", type=str, default="results/07_twostage")
    args = ap.parse_args()

    setup_style()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    print("Loading data ...")
    csv_files = list(data_dir.glob("*.csv")) + list(data_dir.glob("*.csv.gz"))
    assert csv_files, f"No CSV found in {data_dir}"
    df = load_dataset(csv_files[0], smiles_col=args.smiles_col,
                      target_cols=[args.score_col])
    y = df[args.score_col].values.astype(np.float32)
    n = len(y)
    has_docking = args.docking_col in df.columns
    if has_docking:
        docking = -df[args.docking_col].values.astype(np.float32)  # Vina: more negative = better
    binder_ids = load_binders(args.binders, id_col=args.id_col)
    binder_idx = find_binder_indices(df, binder_ids, id_col=args.id_col)
    binder_mask = np.zeros(n, dtype=bool)
    binder_mask[binder_idx] = True
    print(f"  {n} molecules, {len(binder_idx)} binders")

    print("Loading fingerprints ...")
    fp = load_fingerprints(data_dir / "fingerprints")
    X = concat_fingerprints(fp.morgan, fp.atompair)

    print(f"Clustering (k={args.n_clusters}) ...")
    clusters = kmeans_cluster(X, n_clusters=args.n_clusters)

    oracle_auc = binder_roc_auc(y, binder_idx)
    oracle_ef = enrichment_factor(y, binder_idx, tuple(args.final_fracs))
    print(f"Oracle ROC-AUC: {oracle_auc:.3f}")
    print(f"Oracle EF: {oracle_ef}")

    all_rows = []
    for budget in args.budgets:
        for seed in args.seeds:
            labeled = cluster_initial(X, clusters, budget, seed)
            print(f"\n--- budget={budget*100:.2f}% ({len(labeled)} mols), seed={seed} ---")

            ens = train_ensemble(args.model, X[labeled], y[labeled], X, seed=seed)
            pred = ens.mean

            for sl_frac in args.shortlist_fracs:
                sl_n = max(1, int(n * sl_frac))

                # Surrogate as first ranker
                sl_surr = np.zeros(n, dtype=bool)
                sl_surr[np.argsort(pred)[-sl_n:]] = True
                ef_surr = _ef_on_shortlist(y, binder_mask, sl_surr, args.final_fracs)

                row = {"ranker": "surrogate", "budget": budget, "seed": seed,
                       "shortlist_frac": sl_frac, "n_shortlist": sl_n,
                       "n_train": len(labeled), "total_cost_frac": budget + sl_frac,
                       **ef_surr}
                all_rows.append(row)

                # Docking as first ranker
                if has_docking:
                    sl_dock = np.zeros(n, dtype=bool)
                    sl_dock[np.argsort(docking)[-sl_n:]] = True
                    ef_dock = _ef_on_shortlist(y, binder_mask, sl_dock, args.final_fracs)
                    row_d = {"ranker": "docking", "budget": 0.0, "seed": seed,
                             "shortlist_frac": sl_frac, "n_shortlist": sl_n,
                             "n_train": 0, "total_cost_frac": sl_frac,
                             **ef_dock}
                    all_rows.append(row_d)

    results = pd.DataFrame(all_rows)
    csv_path = out_dir / "twostage_results.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved -> {csv_path}")

    # --- Plot: EF@1% vs shortlist fraction ---
    ef_col = "EF@1%"
    if ef_col in results.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        surr = results[results["ranker"] == "surrogate"]
        for budget in args.budgets:
            sub = surr[surr["budget"] == budget].groupby("shortlist_frac")[ef_col].mean()
            ax.plot(sub.index * 100, sub.values, marker="o",
                    label=f"Surrogate ({budget*100:.1f}% train)")
        if has_docking:
            dock = results[results["ranker"] == "docking"].groupby("shortlist_frac")[ef_col].mean()
            ax.plot(dock.index * 100, dock.values, marker="s", color="gray",
                    linestyle="--", label="Docking (free)")
        add_oracle_line(ax, oracle_ef.get(ef_col, 0))
        ax.set_xlabel("Short list size (% of library)")
        ax.set_ylabel("Enrichment Factor @ 1%")
        ax.set_xscale("log")
        ax.legend()
        ax.set_title("Two-stage screening: surrogate vs docking")
        save_figure(fig, out_dir / "twostage_ef_vs_shortlist.png")

    # --- Summary table ---
    summary = (surr.groupby(["budget", "shortlist_frac"])
               .agg(**{c: (c, "mean") for c in results.columns if c.startswith("EF@")})
               .round(1))
    summary.to_csv(out_dir / "twostage_summary.csv")
    print(f"Summary -> {out_dir / 'twostage_summary.csv'}")
    print(summary)


if __name__ == "__main__":
    main()
