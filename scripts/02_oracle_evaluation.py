"""
02 - Oracle evaluation: where does Boltz-2 add value over docking?

Report chapter 2.4.2.  Reproduces Experiment C: four background sets of
increasing docking "difficulty", all binders mixed in, ROC-AUC of Boltz-2
vs docking measured on each.

Usage:
    python scripts/02_oracle_evaluation.py --data-dir data/sample --target ADRA2B
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.data import load_binders, find_binder_indices
from lib.metrics import binder_roc_auc, enrichment_factor
from lib.plotting import setup_style, save_figure, TARGET_COLORS


def build_backgrounds(
    docking_scores: np.ndarray,
    bg_size: int,
    n_total: int,
) -> dict[str, np.ndarray]:
    """Return indices for 4 background sets of increasing docking weakness.

    exp1: best bg_size by docking
    exp2: worst bg_size from top min(10*bg_size, n)
    exp3: worst bg_size from top min(100*bg_size, n)
    exp4: worst bg_size from the full library
    """
    order = np.argsort(-docking_scores)  # best docking first (most negative for Vina)

    cuts = {
        "exp1_top": bg_size,
        "exp2_tail_10x": min(10 * bg_size, n_total),
        "exp3_tail_100x": min(100 * bg_size, n_total),
        "exp4_tail_all": n_total,
    }

    backgrounds: dict[str, np.ndarray] = {}
    for name, cut in cuts.items():
        if "top" in name:
            backgrounds[name] = order[:bg_size]
        else:
            pool = order[:cut]
            backgrounds[name] = pool[-bg_size:] if len(pool) >= bg_size else pool
    return backgrounds


def evaluate_background(
    boltz_scores: np.ndarray,
    docking_scores: np.ndarray,
    bg_idx: np.ndarray,
    binder_idx_full: np.ndarray,
) -> dict[str, float]:
    """Compute ROC-AUC of Boltz-2 and docking on a background + binders subset."""
    binder_set = set(binder_idx_full.tolist())
    bg_set = set(bg_idx.tolist())
    combined = sorted(bg_set | binder_set)
    combined_arr = np.array(combined, dtype=np.int64)

    local_binder = np.array(
        [i for i, g in enumerate(combined_arr) if g in binder_set], dtype=np.int64
    )

    boltz_sub = boltz_scores[combined_arr]
    dock_sub = docking_scores[combined_arr]

    return {
        "n_bg": len(bg_idx),
        "n_combined": len(combined_arr),
        "n_binders": len(local_binder),
        "roc_auc_boltz": binder_roc_auc(boltz_sub, local_binder),
        "roc_auc_docking": binder_roc_auc(dock_sub, local_binder),
    }


def plot_roc_bars(results: pd.DataFrame, target: str, output_dir: Path) -> None:
    """Bar chart: ROC-AUC of Boltz-2 vs docking across 4 backgrounds."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(results))
    w = 0.35
    ax.bar(x - w / 2, results["roc_auc_boltz"], w, label="Boltz-2", color="#1f77b4")
    ax.bar(x + w / 2, results["roc_auc_docking"], w, label="Docking", color="#ff7f0e")
    ax.set_xticks(x)
    ax.set_xticklabels(
        ["Best\n(top bg_size)", "Tail of\ntop 10x", "Tail of\ntop 100x", "Tail of\nfull library"],
        fontsize=9,
    )
    ax.set_ylabel("Binder ROC-AUC")
    ax.set_title(f"{target}: Boltz-2 vs Docking on backgrounds of increasing difficulty")
    ax.axhline(0.5, ls=":", color="gray", alpha=0.5, label="Random")
    ax.legend()
    ax.set_ylim(0, 1.05)
    save_figure(fig, output_dir / f"02_expC_roc_bar_{target}.png")


def plot_enrichment_curves(
    boltz_scores: np.ndarray,
    docking_scores: np.ndarray,
    backgrounds: dict[str, np.ndarray],
    binder_idx_full: np.ndarray,
    target: str,
    output_dir: Path,
) -> None:
    """Enrichment curves per background panel."""
    bg_names = list(backgrounds.keys())
    fig, axes = plt.subplots(1, len(bg_names), figsize=(4 * len(bg_names), 4), sharey=True)
    if len(bg_names) == 1:
        axes = [axes]

    cutoffs = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
    binder_set = set(binder_idx_full.tolist())

    for ax, name in zip(axes, bg_names):
        bg_set = set(backgrounds[name].tolist())
        combined = np.array(sorted(bg_set | binder_set), dtype=np.int64)
        local_binder = np.array(
            [i for i, g in enumerate(combined) if g in binder_set], dtype=np.int64
        )
        boltz_sub = boltz_scores[combined]
        dock_sub = docking_scores[combined]

        ef_boltz = enrichment_factor(boltz_sub, local_binder, cutoffs)
        ef_dock = enrichment_factor(dock_sub, local_binder, cutoffs)

        xs = [c * 100 for c in cutoffs]
        ax.plot(xs, list(ef_boltz.values()), "o-", label="Boltz-2", color="#1f77b4")
        ax.plot(xs, list(ef_dock.values()), "s--", label="Docking", color="#ff7f0e")
        ax.axhline(1.0, ls=":", color="gray", alpha=0.5)
        ax.set_xlabel("Top x%")
        ax.set_title(name.replace("_", " "))
        if ax == axes[0]:
            ax.set_ylabel("Enrichment Factor")
        ax.legend(fontsize=8)

    fig.suptitle(f"{target}: Enrichment curves by background difficulty", y=1.02)
    save_figure(fig, output_dir / f"02_expC_enrichment_{target}.png")


def main():
    ap = argparse.ArgumentParser(description="Ch 2.4.2: Oracle evaluation (Experiment C)")
    ap.add_argument("--data-dir", type=str, required=True)
    ap.add_argument("--target", type=str, default="ADRA2B")
    ap.add_argument("--score-col", type=str, default="consensus_score",
                    help="Boltz-2 or consensus score column")
    ap.add_argument("--docking-col", type=str, default="docking_score")
    ap.add_argument("--smiles-col", type=str, default="smiles")
    ap.add_argument("--id-col", type=str, default="zincid")
    ap.add_argument("--binders", type=str, default=None,
                    help="Binders CSV (default: <data-dir>/binders.csv)")
    ap.add_argument("--bg-size", type=int, default=10_000)
    ap.add_argument("--output-dir", type=str, default="results")
    args = ap.parse_args()

    setup_style()
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    binders_path = Path(args.binders) if args.binders else data_dir / "binders.csv"

    print(f"Loading dataset from {data_dir} ...")
    csv_candidates = list(data_dir.glob("*.csv"))
    lib_csv = [f for f in csv_candidates if "binder" not in f.stem.lower()]
    if not lib_csv:
        sys.exit(f"No library CSV found in {data_dir}")
    df = pd.read_csv(lib_csv[0]).dropna(subset=[args.smiles_col]).reset_index(drop=True)

    for col in (args.score_col, args.docking_col):
        if col not in df.columns:
            sys.exit(f"Column '{col}' not found. Available: {list(df.columns)}")

    mask = df[[args.score_col, args.docking_col]].notna().all(axis=1)
    df = df[mask].reset_index(drop=True)
    n = len(df)
    print(f"  {n} molecules with both scores")

    binder_ids = load_binders(binders_path, id_col=args.id_col)
    binder_idx = find_binder_indices(df, binder_ids, id_col=args.id_col)
    print(f"  {len(binder_idx)} binders found in pool")

    boltz = df[args.score_col].values.astype(np.float32)
    docking = df[args.docking_col].values.astype(np.float32)

    bg_size = min(args.bg_size, n // 4) if n < 4 * args.bg_size else args.bg_size
    print(f"  Background size: {bg_size}")

    backgrounds = build_backgrounds(docking, bg_size, n)

    rows = []
    for name, bg_idx in backgrounds.items():
        res = evaluate_background(boltz, docking, bg_idx, binder_idx)
        res["background"] = name
        rows.append(res)
        print(f"  {name:>20s}: Boltz AUC={res['roc_auc_boltz']:.3f}  "
              f"Dock AUC={res['roc_auc_docking']:.3f}  "
              f"({res['n_binders']} binders in {res['n_combined']} mols)")

    results = pd.DataFrame(rows)
    csv_path = output_dir / f"02_expC_results_{args.target}.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    plot_roc_bars(results, args.target, output_dir)
    plot_enrichment_curves(boltz, docking, backgrounds, binder_idx, args.target, output_dir)
    print(f"Figures saved to {output_dir}/")


if __name__ == "__main__":
    main()
