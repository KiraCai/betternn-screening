# betternn-screening

Train the **BetterNN** deep-ensemble surrogate on **one target** (protein + its
ligand library), rank the library, and report screening metrics. Built on top of
the `deep-ensemble-models` codebase; the entry point for everyday use is
**`scripts/run_betternn.py`**.

The surrogate learns to reproduce an expensive oracle label (e.g. a Boltz-2 /
BoltzDock consensus score) from 4096-bit Morgan⊕AtomPair fingerprints, so you can
rank a whole library after scoring only a small fraction with the oracle.

---

## 1. Install

```bash
uv sync                       # creates .venv with the CUDA-12.4 torch build
```

The `torch` pin in `pyproject.toml` uses the **cu124** wheel so training runs on
GPUs with NVIDIA driver 12.4. If your driver supports a newer CUDA, change the
index URL in `pyproject.toml` (`pytorch-cu124` → e.g. `cu128`) and re-`uv sync`.

Run scripts with either `uv run python scripts/run_betternn.py ...` **or**
directly `.venv/bin/python scripts/run_betternn.py ...`.

---

## 2. Data layout — one folder per target

Put each target in its own folder:

```
data/<target>/
├── library.csv        # required: columns  smiles, zincid, <score column>
├── binders.csv        # optional: column  zincid  (experimental binders)
└── fingerprints/      # optional: auto-generated on first run if missing
```

- **`library.csv`** must contain a SMILES column (`--smiles-col`, default
  `smiles`), an id column (`--id-col`, default `zincid`), and the oracle-label
  column you want to predict (`--score-col`, default `consensus_score`).
- **`binders.csv`** (optional) lists experimentally validated binder ids; if
  present you get binder ROC-AUC and enrichment factors.
- **`fingerprints/`** — if absent, it is generated automatically (see CPU prompt
  below) and cached, so subsequent runs are fast.

A ready **ADRA2B** demo lives in `data/sample/` (10k molecules, 21 binders).

---

## 3. Quick start (ADRA2B demo)

```bash
# fully interactive: asks which GPU, and how many CPUs if it has to build FPs
uv run python scripts/run_betternn.py --data-dir data/sample

# scripted / no prompts: 3 seeds, 2% budget, GPU 0
uv run python scripts/run_betternn.py --data-dir data/sample \
    --seeds 42 7 13 --budget 0.02 --gpus 0 --no-interactive
```

Your own target:

```bash
uv run python scripts/run_betternn.py \
    --data-dir data/my_protein --score-col consensus_score --n-seeds 10
```

---

## 4. All flags

### data
| flag | default | meaning |
|---|---|---|
| `--data-dir` | `data/sample` | target folder with `library.csv`, `binders.csv`, `fingerprints/` |
| `--score-col` | `consensus_score` | column in `library.csv` used as the oracle label to predict |
| `--smiles-col` | `smiles` | SMILES column |
| `--id-col` | `zincid` | molecule id column |
| `--binders` | `<data-dir>/binders.csv` | path to the binders CSV |

### training budget
| flag | default | meaning |
|---|---|---|
| `--budget` | `0.02` | training-set size as a **fraction** of the prediction pool |
| `--n-train` | — | training-set size as an **absolute count** (overrides `--budget`) |
| `--n-models` | `5` | number of ensemble members |

### seeds — choose ONE way
| flag | default | meaning |
|---|---|---|
| `--seeds` | — | explicit list, e.g. `--seeds 42 7 13` |
| `--n-seeds` | — | just say **how many** seeds; they are drawn randomly |
| `--seed-base` | `12345` | RNG base so the random `--n-seeds` draw is reproducible |
| *(none of the above)* | | falls back to the 10 project-default seeds |

### prediction pool
| flag | default | meaning |
|---|---|---|
| `--predict-size` | whole library | randomly sample this many molecules from the library as the prediction pool |
| `--pool-seed` | `0` | RNG seed for that random subset |

### compute
| flag | default | meaning |
|---|---|---|
| `--cpus` | *ask* | CPU workers for fingerprint generation; if omitted and FPs are missing, it shows the available core count and asks |
| `--gpus` | *ask* | GPU index(es) for training, e.g. `0` or `0,1`; `cpu` forces CPU; if omitted it lists GPUs and asks |
| `--no-interactive` | off | never prompt — use all CPUs and GPU 0 (or CPU if none) |

### output
| flag | default | meaning |
|---|---|---|
| `--output-dir` | `results` | where CSVs are written |
| `--top-k` | `1000` | how many top-ranked molecules to save |
| `--tag` | folder name | filename prefix for the outputs |

Full list any time: `python scripts/run_betternn.py --help`.

---

## 5. Interactive CPU / GPU prompts

**Fingerprints (CPU).** If `fingerprints/` is missing, the script prints the
number of available cores and asks how many to use, then generates them in
parallel and caches them:

```
[compute] 32 CPU cores available. How many to use for fingerprint generation? [default 32]:
```

Skip the question with `--cpus 16` or `--no-interactive`.

**Training (GPU).** The script lists the GPUs (via `nvidia-smi`, before CUDA is
initialized) and asks which to use:

```
[compute] Available GPUs:
    [0] NVIDIA GeForce RTX 3080  (10240 MiB)
    [1] NVIDIA GeForce RTX 3080  (10240 MiB)
Which GPU(s) to use for training? (e.g. 0 or 0,1; 'cpu' for CPU) [default 0]:
```

Skip with `--gpus 0` (or `--gpus cpu`) or `--no-interactive`.

---

## 6. Outputs

Written to `--output-dir` (prefixed by `--tag`):

- `<tag>_betternn_perseed.csv` — metrics for every seed.
- `<tag>_betternn_summary.csv` — mean ± std across seeds.
- `<tag>_betternn_top<K>.csv` — top-K molecules by the seed-averaged prediction
  (`id, smiles, score, betternn_pred`), i.e. your shortlist for the oracle.

Metrics: Spearman (global ranking), Recall@1/5/10 %, binder ROC-AUC, binder
counts and enrichment factors at several cutoffs.

---

## 7. Reproducing the report experiments

The original numbered scripts from `deep-ensemble-models` are kept in `scripts/`
(`01_…08_…`) and still run via `just` on the bundled ADRA2B set. `run_betternn.py`
is the simplified, per-target, GPU-aware entry point layered on top of the same
`lib/` code.

---

## 8. Planned improvements (suggested)

- **Multi-GPU ensemble** — train the 5 members in parallel across GPUs (currently
  members run sequentially on the chosen device).
- **Bit-packed fingerprints** for 10M–1B libraries (uint8 → bits) to cut disk/RAM.
- **Streaming prediction** for libraries too large to hold in memory.
- **Auto oracle-budget suggestion** via the sustained-crossing estimator in
  `lib/metrics.py` (`sustained_crossing`).
- **Config files** (`--config run.yaml`) to snapshot a full experiment.
