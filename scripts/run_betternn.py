#!/usr/bin/env python
"""
run_betternn.py — train the BetterNN deep-ensemble surrogate on ONE target and
rank its library.

What it does, in order:
  1. Load a target folder:  <data-dir>/library.csv  (+ optional binders.csv)
  2. Fingerprints:  load <data-dir>/fingerprints/ if present, otherwise GENERATE
     them on CPU (asks how many CPU cores to use, unless --cpus is given).
  3. Prediction pool: use the whole valid library, or a random subset of size
     --predict-size (sampled with --pool-seed for reproducibility).
  4. For each seed: draw a random training set (--budget fraction or --n-train
     absolute), train the 5-model BetterNN ensemble on GPU, predict the pool.
  5. Report Spearman / Recall / binder ROC-AUC / enrichment (mean±std over seeds)
     and write a ranked CSV of the top molecules.

GPU is chosen interactively (or with --gpus). Everything can be made
non-interactive with --no-interactive (uses safe defaults).

Run `python scripts/run_betternn.py --help` for the full flag list.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train BetterNN on one target and rank its library.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # --- data ---
    g = p.add_argument_group("data")
    g.add_argument("--data-dir", default="data/sample",
                   help="Folder with library.csv, binders.csv, fingerprints/")
    g.add_argument("--score-col", default="consensus_score",
                   help="Column in library.csv used as the oracle label to predict")
    g.add_argument("--smiles-col", default="smiles", help="SMILES column")
    g.add_argument("--id-col", default="zincid", help="Molecule id column")
    g.add_argument("--binders", default=None,
                   help="Binders CSV (default: <data-dir>/binders.csv if it exists)")

    # --- training budget ---
    g = p.add_argument_group("training budget")
    g.add_argument("--budget", type=float, default=0.02,
                   help="Training-set size as a FRACTION of the prediction pool")
    g.add_argument("--n-train", type=int, default=None,
                   help="Training-set size as an ABSOLUTE count (overrides --budget)")
    g.add_argument("--n-models", type=int, default=5, help="Ensemble members")

    # --- seeds ---
    g = p.add_argument_group("seeds (choose ONE way)")
    g.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Explicit list of seeds, e.g. --seeds 42 7 13")
    g.add_argument("--n-seeds", type=int, default=None,
                   help="Number of seeds to draw RANDOMLY (reproducible via --seed-base)")
    g.add_argument("--seed-base", type=int, default=12345,
                   help="RNG base used when --n-seeds draws random seeds")

    # --- prediction pool ---
    g = p.add_argument_group("prediction pool")
    g.add_argument("--predict-size", type=int, default=None,
                   help="Randomly sample this many molecules from the library as the "
                        "prediction pool (default: use the whole valid library)")
    g.add_argument("--pool-seed", type=int, default=0,
                   help="RNG seed for the random prediction-pool subset")

    # --- screen mode: train on scored library, rank a separate (unscored) file ---
    g = p.add_argument_group("screen mode (train on library, rank another file)")
    g.add_argument("--screen-file", default=None,
                   help="CSV of molecules to RANK. Trains BetterNN on the whole scored "
                        "library.csv, then predicts this file (its score column is "
                        "optional). Fingerprints for it are cached next to it.")

    # --- compute ---
    g = p.add_argument_group("compute")
    g.add_argument("--cpus", type=int, default=None,
                   help="CPU workers for fingerprint generation (asks if omitted)")
    g.add_argument("--gpus", default=None,
                   help='GPU index(es) for training, e.g. "0" or "0,1". '
                        '"cpu" forces CPU. Asks if omitted.')
    g.add_argument("--no-interactive", action="store_true",
                   help="Never prompt; use defaults (all CPUs, GPU 0 if present)")

    # --- output ---
    g = p.add_argument_group("output")
    g.add_argument("--output-dir", default="results", help="Where to write CSVs")
    g.add_argument("--top-k", type=int, default=1000,
                   help="How many top-ranked molecules to save")
    g.add_argument("--tag", default=None, help="Optional name prefix for output files")
    return p


# ---------------------------------------------------------------------------
# Interactive compute selection (must run BEFORE importing torch)
# ---------------------------------------------------------------------------
def list_gpus() -> list[tuple[int, str, str]]:
    """Query nvidia-smi without initializing CUDA. Returns [(idx, name, mem)]."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,memory.total",
             "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    gpus = []
    for line in out.strip().splitlines():
        parts = [x.strip() for x in line.split(",")]
        if len(parts) >= 3:
            gpus.append((int(parts[0]), parts[1], parts[2]))
    return gpus


def choose_gpus(args) -> str:
    """Return the CUDA_VISIBLE_DEVICES string to set ('' means CPU)."""
    if args.gpus is not None:
        return "" if args.gpus.lower() == "cpu" else args.gpus
    gpus = list_gpus()
    if not gpus:
        print("[compute] No NVIDIA GPU detected -> training on CPU (slow).")
        return ""
    print("\n[compute] Available GPUs:")
    for idx, name, mem in gpus:
        print(f"    [{idx}] {name}  ({mem})")
    if args.no_interactive:
        print(f"[compute] --no-interactive -> using GPU {gpus[0][0]}")
        return str(gpus[0][0])
    ans = input(f"Which GPU(s) to use for training? "
                f"(e.g. 0 or 0,1; 'cpu' for CPU) [default {gpus[0][0]}]: ").strip()
    if ans.lower() == "cpu":
        return ""
    return ans if ans else str(gpus[0][0])


def choose_cpus(args) -> int:
    """Number of CPU workers for fingerprint generation."""
    total = os.cpu_count() or 1
    if args.cpus is not None:
        return max(1, min(args.cpus, total))
    if args.no_interactive:
        return total
    ans = input(f"\n[compute] {total} CPU cores available. "
                f"How many to use for fingerprint generation? [default {total}]: ").strip()
    if not ans:
        return total
    try:
        return max(1, min(int(ans), total))
    except ValueError:
        return total


def get_or_make_fps(fp_dir, smiles_list, n_cpu: int):
    """Load fingerprints from fp_dir, or generate + cache them. Returns (concat X, valid)."""
    import numpy as np
    from lib import features as Fx
    fp_dir = Path(fp_dir)
    have = (fp_dir / "morgan_2048.npz").exists() or (fp_dir / "morgan_2048.npy").exists()
    if have:
        bf = Fx.load_fingerprints(fp_dir)
        print(f"[features] loaded fingerprints from {fp_dir}")
    else:
        print(f"[features] generating fingerprints for {len(smiles_list):,} mols on {n_cpu} CPU workers ...")
        bf = Fx.featurize_batch([str(s) for s in smiles_list], n_workers=n_cpu)
        fp_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fp_dir / "morgan_2048.npz", data=bf.morgan)
        np.savez_compressed(fp_dir / "atompair.npz", data=bf.atompair)
        np.save(fp_dir / "descriptors.npy", bf.descriptors)
        np.save(fp_dir / "valid.npy", bf.valid)
        np.save(fp_dir / "canonical_smiles.npy", bf.canonical_smiles)
        print(f"[features] saved fingerprints to {fp_dir}")
    return Fx.concat_fingerprints(bf.morgan, bf.atompair), bf.valid.astype(bool)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    args = build_parser().parse_args()

    # 1) resolve seeds
    import numpy as np
    if args.seeds is not None:
        seeds = list(args.seeds)
    elif args.n_seeds is not None:
        rng = np.random.RandomState(args.seed_base)
        seeds = sorted(int(s) for s in rng.choice(1_000_000, size=args.n_seeds, replace=False))
    else:
        seeds = [42, 7, 13, 100, 2024, 99, 1, 5, 21, 77]  # project default (10)
    print(f"[seeds] {len(seeds)} seed(s): {seeds}")

    # 2) choose GPU (set CUDA_VISIBLE_DEVICES BEFORE importing torch)
    visible = choose_gpus(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = visible
    print(f"[compute] CUDA_VISIBLE_DEVICES={visible!r}")

    # heavy imports AFTER the env var is set
    import pandas as pd
    import torch
    from lib import data as D
    from lib import features as Fx
    from lib import metrics as M
    from lib.models import train_ensemble

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[compute] torch device = {dev}"
          + (f"  ({torch.cuda.get_device_name(0)})" if dev == "cuda" else "  (CPU — slow)"))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tag = args.tag or data_dir.name

    # 3) load library (keep raw order for fingerprint alignment)
    lib_csv = data_dir / "library.csv"
    df = pd.read_csv(lib_csv)
    print(f"[data] {lib_csv}: {len(df):,} rows; predicting column '{args.score_col}'")
    if args.score_col not in df.columns:
        sys.exit(f"ERROR: column '{args.score_col}' not in {lib_csv}. "
                 f"Available: {list(df.columns)}")

    # 4) fingerprints — load or generate
    fp_dir = data_dir / "fingerprints"
    need = not ((fp_dir / "morgan_2048.npz").exists() or (fp_dir / "morgan_2048.npy").exists())
    if need:
        n_cpu = choose_cpus(args)
        print(f"[features] no fingerprints found -> generating on {n_cpu} CPU workers ...")
        bf = Fx.featurize_batch(df[args.smiles_col].astype(str).tolist(), n_workers=n_cpu)
        fp_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fp_dir / "morgan_2048.npz", data=bf.morgan)
        np.savez_compressed(fp_dir / "atompair.npz", data=bf.atompair)
        np.save(fp_dir / "descriptors.npy", bf.descriptors)
        np.save(fp_dir / "valid.npy", bf.valid)
        np.save(fp_dir / "canonical_smiles.npy", bf.canonical_smiles)
        print(f"[features] saved fingerprints to {fp_dir}")
    else:
        bf = Fx.load_fingerprints(fp_dir)
        print(f"[features] loaded fingerprints from {fp_dir}")

    # 5) align + keep valid rows that also have a score
    valid = bf.valid.astype(bool)
    score_ok = df[args.score_col].notna().values
    if len(valid) != len(df):
        sys.exit(f"ERROR: fingerprints ({len(valid)}) and library ({len(df)}) length "
                 f"mismatch — regenerate fingerprints for this library.")
    keep = valid & score_ok
    df = df[keep].reset_index(drop=True)
    X = Fx.concat_fingerprints(bf.morgan[keep], bf.atompair[keep])  # (N, 4096)
    y = df[args.score_col].values.astype(np.float32)
    N = len(y)
    print(f"[data] usable molecules: {N:,}  (fingerprint dim {X.shape[1]})")

    # 5b) SCREEN MODE: train on the whole scored library, rank a separate file
    if args.screen_file:
        screen_path = Path(args.screen_file)
        sdf = pd.read_csv(screen_path)
        print(f"[screen] {screen_path}: {len(sdf):,} molecules to rank")
        # training set = the whole scored library, unless --n-train caps it
        if args.n_train:
            tr = np.random.RandomState(seeds[0]).choice(N, min(args.n_train, N), replace=False)
        else:
            tr = np.arange(N)
        print(f"[screen] training on {len(tr):,} scored molecules "
              f"({args.n_models} members × {len(seeds)} seeds)")
        # fingerprints for the screen file (cached next to it)
        s_fp = screen_path.parent / f"{screen_path.stem}_fingerprints"
        n_cpu_s = choose_cpus(args) if not (
            (s_fp / "morgan_2048.npz").exists() or (s_fp / "morgan_2048.npy").exists()) else 1
        Xs, svalid = get_or_make_fps(s_fp, sdf[args.smiles_col].astype(str).tolist(), n_cpu_s)
        if len(svalid) == len(sdf):
            sdf = sdf[svalid].reset_index(drop=True); Xs = Xs[svalid]
        pred_accum = np.zeros(len(Xs), dtype=np.float64)
        for si, seed in enumerate(seeds, 1):
            res = train_ensemble("betternn", X[tr], y[tr], Xs, seed=seed, n_models=args.n_models)
            pred_accum += res.mean
            print(f"  [{si}/{len(seeds)}] seed={seed} done", flush=True)
        pred = pred_accum / len(seeds)
        sdf["betternn_pred"] = pred
        # optional metrics if the screen file itself carries a score / binder flag
        if args.score_col in sdf.columns and sdf[args.score_col].notna().any():
            m = sdf[args.score_col].notna().values
            print(f"[screen] Spearman vs '{args.score_col}' on {int(m.sum()):,} scored rows: "
                  f"{M.spearman(sdf[args.score_col].values[m], pred[m]):.3f}")
        ranked = sdf.sort_values("betternn_pred", ascending=False).head(args.top_k)
        cols = [c for c in (args.id_col, args.smiles_col, args.score_col, "betternn_pred")
                if c in ranked.columns]
        rank_path = out_dir / f"{tag}_screen_top{args.top_k}.csv"
        ranked[cols].to_csv(rank_path, index=False)
        print(f"\nSaved ranked screen shortlist:\n  {rank_path}")
        return

    # 6) prediction pool (optional random subset of the whole library)
    if args.predict_size and args.predict_size < N:
        sub = np.sort(np.random.RandomState(args.pool_seed).choice(N, args.predict_size, replace=False))
        df = df.iloc[sub].reset_index(drop=True)
        X = X[sub]; y = y[sub]; N = len(y)
        print(f"[pool] random prediction pool of {N:,} molecules (pool-seed={args.pool_seed})")

    # binder indices within the current pool
    binders_path = Path(args.binders) if args.binders else (data_dir / "binders.csv")
    if binders_path.exists():
        binder_ids = D.load_binders(binders_path, id_col=args.id_col)
        binder_idx = D.find_binder_indices(df, binder_ids, id_col=args.id_col)
        print(f"[data] {len(binder_idx)} known binders present in the pool")
    else:
        binder_idx = np.array([], dtype=np.int64)
        print("[data] no binders.csv -> skipping binder ROC-AUC / enrichment")

    # 7) training size
    n_train = args.n_train if args.n_train else max(1, int(round(args.budget * N)))
    n_train = min(n_train, N - 1)
    print(f"[train] n_train = {n_train:,}  ({n_train / N * 100:.2f}% of pool), "
          f"{args.n_models} ensemble members, {len(seeds)} seeds\n")

    # 8) run over seeds
    rows = []
    pred_accum = np.zeros(N, dtype=np.float64)
    for si, seed in enumerate(seeds, 1):
        tr = np.random.RandomState(seed).choice(N, size=n_train, replace=False)
        res = train_ensemble("betternn", X[tr], y[tr], X, seed=seed, n_models=args.n_models)
        pred = res.mean
        pred_accum += pred

        unlab = np.setdiff1d(np.arange(N), tr, assume_unique=False)
        met = M.evaluate_surrogate(y[unlab], pred[unlab],
                                   binder_idx=binder_idx if len(binder_idx) else None,
                                   pred_all=pred if len(binder_idx) else None)
        if len(binder_idx):
            met.update(M.enrichment_factor(pred, binder_idx))
        met = {"seed": seed, **met}
        rows.append(met)
        msg = f"  [{si}/{len(seeds)}] seed={seed}  Sp={met['spearman']:.3f}  R@5%={met['R@5%']:.3f}"
        if "binder_roc_auc" in met:
            msg += f"  binderAUC={met['binder_roc_auc']:.3f}"
        print(msg, flush=True)

    # 9) aggregate + save
    res_df = pd.DataFrame(rows)
    long_path = out_dir / f"{tag}_betternn_perseed.csv"
    res_df.to_csv(long_path, index=False)

    num = res_df.drop(columns=["seed"]).select_dtypes("number")
    summary = pd.DataFrame({"metric": num.columns,
                            "mean": num.mean().values,
                            "std": num.std().values})
    summ_path = out_dir / f"{tag}_betternn_summary.csv"
    summary.to_csv(summ_path, index=False)

    # 10) ranked top molecules (from the seed-averaged prediction)
    df_out = df.copy()
    df_out["betternn_pred"] = pred_accum / len(seeds)
    ranked = df_out.sort_values("betternn_pred", ascending=False).head(args.top_k)
    cols = [c for c in (args.id_col, args.smiles_col, args.score_col, "betternn_pred") if c in ranked.columns]
    rank_path = out_dir / f"{tag}_betternn_top{args.top_k}.csv"
    ranked[cols].to_csv(rank_path, index=False)

    print("\n=== summary (mean ± std over seeds) ===")
    for _, r in summary.iterrows():
        print(f"  {r['metric']:<16} {r['mean']:.4f} ± {r['std']:.4f}")
    print(f"\nSaved:\n  {long_path}\n  {summ_path}\n  {rank_path}")


if __name__ == "__main__":
    main()
