# Contributing to PlastBurstAlign

Thank you for helping! Bug reports, benchmark results, documentation fixes and
code are all welcome. By participating you agree to the
[Code of Conduct](CODE_OF_CONDUCT.md). Security problems go through
[SECURITY.md](SECURITY.md), not the public issue tracker.

## Development setup

PlastBurstAlign needs Python >= 3.11 (CI covers 3.11, 3.12 and 3.13). The
external aligners are optional at run time (a slow built-in fallback aligner
exists) but you want them for development.

With conda/mamba (installs MAFFT, MUSCLE 5 and Clustal Omega too):

```bash
conda env create -f environment.yml        # or: mamba env create -f environment.yml
conda activate plastburstalign
pip install -e ".[dev]"                     # pytest, pytest-cov, ruff, mypy, build, twine, pre-commit
pre-commit install                          # optional: runs ruff + mypy on every commit
```

With plain pip (install the aligners with your package manager, e.g.
`apt install mafft muscle clustalo` or `brew install mafft`):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Check what the package can find on your machine with
`plastburstalign --doctor` (host profile, resolved aligners, recommendations).
To exercise the pure-Python fallback aligner even when MAFFT is installed, set
`PBA_FORCE_BUILTIN_ALIGNER=1`; tests that need a real MAFFT skip themselves when
it is not usable.

## Everyday commands

```bash
# tests (fast; keep new tests tiny and unittest-style under tests/)
python -m pytest -q -p no:cacheprovider tests
python -m pytest -q -p no:cacheprovider tests/test_plastbench.py   # one file

# coverage, as CI runs it (the pytest addopts disable pytest-cov by default,
# so replace them; CI additionally passes --cov-fail-under)
python -m pytest -o addopts="-v" --cov --cov-report=term-missing tests

# lint, format and types: all three are CI gates
ruff check .
ruff format --check .        # `ruff format <files>` to fix your own files
mypy plastburstalign         # whole package, legacy modules are listed in pyproject.toml

# packaging smoke test (writes to a temp dir, not to the repository)
python -m build --outdir "$(mktemp -d)" && echo built
```

Code style: `from __future__ import annotations`, type hints on public functions,
ruff-clean and ruff-formatted (line length 88). Do not format files you did not
change, so diffs stay reviewable.

## The golden parity gate

Refactors must not change outputs. `benchmarking/golden_parity.py` runs the real
CLI on a fixed 16-genome subset of the Asteraceae benchmark in the `cds`, `igs`,
`int` and `rna` modes and compares canonicalized SHA-256 hashes with
`tests/golden/asteraceae_parity.json`:

```bash
python benchmarking/golden_parity.py check               # all four modes (1-2 minutes)
python benchmarking/golden_parity.py check --modes rna   # the fast one
```

It needs the dataset directory `benchmarking/benchmark_YangEtAl2022_Asteraceae`
(download it once with
`python benchmarking/plastbench/plastbench.py fetch benchmark_YangEtAl2022_Asteraceae --dest benchmarking`)
and the aligner versions recorded in the golden file (a mismatch is printed as a
warning because a different MAFFT can legitimately change alignments). It prints
`SKIP` and exits 0 if the dataset is absent. Only run `capture` on a known-good
revision, and say so in the pull request if a legitimate behaviour change
requires a new golden file.

## Branches, commits and pull requests

* Branch from `main`; use short descriptive branch names such as
  `feat/slurm-export` or `fix/prot-backtranslation`. The v3 work lives on
  `feat/v3-gold-standard`.
* Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  as in `git log`: `type(scope): imperative summary`, with type one of `feat`,
  `fix`, `perf`, `refactor`, `test`, `docs`, `style`, `ci`, `build`, `chore`
  and an optional scope such as `io`, `scanner`, `heuristics`, `cluster`,
  `storage`, `scheduling` (e.g. `fix(extraction): honor min_seq_length`).
* Keep pull requests focused; one concern per PR is much easier to review.
* Fill in the pull request template. Before requesting review make sure that
  `pytest`, `ruff check .`, `ruff format --check .`, `mypy plastburstalign` and
  the parity gate pass, and add a line to the *Unreleased* section of
  [CHANGELOG.md](CHANGELOG.md).
* Performance claims need evidence: attach a PlastBench result (below).

## Benchmarks (PlastBench)

`benchmarking/plastbench/` holds the benchmark harness; see its
[README](benchmarking/plastbench/README.md) for the tiers, the metrics, the
results schema and the maintainer workflow for publishing datasets. The smoke
tier (100 genomes) runs on a laptop in a few minutes:

```bash
PB=benchmarking/plastbench/plastbench.py
python $PB fetch benchmark_YangEtAl2022_Asteraceae --dest data
python $PB subsample --src data/benchmark_YangEtAl2022_Asteraceae --tier smoke --out data/smoke100
python $PB run --input data/smoke100 --mode igs --out results/smoke.json
python $PB report results/smoke.json
```

Do not run the 1k/5k/15k tiers on a laptop. Share results through the
"Benchmark submission" issue template.

## Reporting bugs

Use the bug-report issue form and paste the output of `plastburstalign --doctor`.
If a GenBank file triggers the problem, attach a minimal record (a few
kilobases are usually enough) and the exact command line.
