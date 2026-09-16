# Experiment runner for the binding-affinity surrogate report.
#
# All recipes run on the bundled ADRA2B demo set by default. Point them at a
# full dataset with `just data-dir=/path/to/adra2b exp04`.

data-dir := "data/sample"
output-dir := "results"

# List available recipes.
default:
    @just --list

# Install/refresh the virtual environment.
sync:
    uv sync

# Check that every script parses and its --help works.
check:
    uv run python -c "import ast, pathlib; [ast.parse(f.read_text()) for f in pathlib.Path('scripts').glob('*.py')]"
    @echo "all scripts parse"

# Print the seed defaults of every script (expects 10 for stochastic ones).
seeds:
    @uv run python -c "\
    import ast, pathlib; \
    [print(f'{f.name}: ' + (', '.join(f'{len(ast.literal_eval(k.value))} seeds' \
      for n in ast.walk(ast.parse(f.read_text())) \
      if isinstance(n, ast.Call) and getattr(n.func, 'attr', '') == 'add_argument' \
      and n.args and getattr(n.args[0], 'value', '') == '--seeds' \
      for k in n.keywords if k.arg == 'default') or 'deterministic')) \
     for f in sorted(pathlib.Path('scripts').glob('*.py'))]"

# --- Individual experiments (chapter numbers follow the report) ---

# 2.4.1  MolPAL active learning baseline.
exp01 *ARGS:
    uv run python scripts/01_molpal_baseline.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.2  Where Boltz-2 beats docking (4 difficulty levels).
exp02 *ARGS:
    uv run python scripts/02_oracle_evaluation.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.3  Compare DeepEns / BoltzNN / BetterNN at multiple budgets. Slow: hours.
exp03 *ARGS:
    uv run python scripts/03_surrogate_models.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.4  6-way comparison: one-shot vs iterative AL.
exp04 *ARGS:
    uv run python scripts/04_oneshot_vs_al.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.5  Budget scan, acceleration factor by library size.
exp05 *ARGS:
    uv run python scripts/05_budget_and_acceleration.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.6  Sphere exclusion, singleton / binder isolation analysis. Slow: hours.
exp06 *ARGS:
    uv run python scripts/06_difficulty_analysis.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.7  Enrichment factors, two-stage surrogate+oracle rescore.
exp07 *ARGS:
    uv run python scripts/07_twostage_screening.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# 2.4.8  Scaling to 10M+, self-consistency difficulty probes.
exp08 *ARGS:
    uv run python scripts/08_large_scale.py --data-dir {{data-dir}} --output-dir {{output-dir}} {{ARGS}}

# --- Batches ---

# Run the experiments that finish in reasonable time on the demo set.
fast: exp01 exp02 exp04 exp05 exp07 exp08

# Run everything, including the two slow clustering-bound scripts (03, 06).
all: exp01 exp02 exp03 exp04 exp05 exp06 exp07 exp08

# Delete generated CSVs and figures.
clean:
    rm -rf {{output-dir}}/*.csv {{output-dir}}/figures/*.png
