# PlastBurstAlign v3.0rc1 — STATUS / LIMITS

Verified on branch `feat/v3-gold-standard` (2026-11-01), host ~6C/12T, ~3 GiB MemAvailable, `/dev/shm` ~7.5 GiB. Env: `plastburst-dev` (Py 3.11, MAFFT 7.526).

## Gate results

| Gate | Result |
|------|--------|
| `pytest -q -p no:cacheprovider tests` | **859 passed** in ~46 s (no deselects; signal/timeout resolver tests OK on this host) |
| `ruff check --no-cache .` | clean |
| `ruff format --no-cache --check .` | 107 files already formatted |
| `mypy plastburstalign` | clean (41 files) |
| `python benchmarking/golden_parity.py check` | **PARITY OK** (cds/igs/int/rna) |
| same + `--reverse-input --threads 3 --strict-order` | **PARITY OK** |

No feature rewrites; no gate blockers.

## Smoke timings

| Run | Wall |
|-----|------|
| Synth 16 taxa × 20 genes, CLI `cds` (`-n 4`) | ~1.1 s |
| Synth same, API `align(..., mode="cds+igs", num_threads=4)` | ~2.8–3.2 s |
| Asteraceae 155 `.gb`, CLI `igs` (`-n 6`) | ~8 m 41 s (~522 s); 341/341 loci OK, 2 chunked; outputs deleted |

## What shipped (v3 gold-standard surface)

- Strategy work model + provisional tier constants; CPU-set leases / LocusScheduler / chunking; dense matrices + vectorized trim + streaming supermatrix; HostProfile / autotune; aligner resolver + builtin center-star fallback; `align` / `autotune` / `estimate` API + run manifest; rich dashboard; CI / fuzz / determinism / PlastBench scaffolding; docs + `3.0.0rc1` in `pyproject.toml`.

## Uncommitted working tree (at verification)

Modified: `CHANGELOG.md`, `README.md`, `docs/exemplary_usage.md`, package/CLI/alignment/Slurm paths, several tests. Untracked: `docs/calibration.md`, `align_runtime.py`, `api.py`, `manifest.py`, `tests/test_api.py`, `tests/test_pipeline_determinism.py`, `benchmarking/results/`.

## Provisional constants

`DEFAULT_CONSTANTS.calibrated=False` (`strategy_constants.py`): conservative analytic tiers (`work_fast_tree=3e10`, `work_high_scale=7e10`, `work_extreme_scale=1.5e11`, `beta=0.5`, `N_FORCE_PARTTREE=5000`). Memory model / SMT default also logged as PROVISIONAL. Fit via `python benchmarking/calibrate_strategy.py --fit` (see `docs/calibration.md`).

## Known gaps / limits

- **`--time-budget`**: stub only (value recorded + non-reproducible warning; host-adaptive path not wired).
- **Manifest loci order**: sorted by `(region, mode)` in `manifest.py` (stable across threads).
- **Package version**: source/`pyproject` is `3.0.0rc1`, but an older install still reports `importlib.metadata` **1.3**. Reinstall editable: `pip install -e .` in `plastburst-dev` before release checks.
- **13k / 15k PlastBench tiers**: defined; this laptop cannot run them (RAM). Use HPC.
- **`gappyout`**: trimAl-compatible in intent, **not** claimed bit-identical (trimAl not available here to diff).
- **Builtin aligner**: deterministic center-star fallback with hard size guards; quality warning by design—prefer MAFFT/MUSCLE/Clustal for production.
- **Multi-mode API smoke**: `cds+igs` completed, but merged `AlignmentResult.ntax` reported **576** for a 16-taxon set (likely merge bookkeeping quirk)—single-mode paths looked correct; investigate before relying on multi-mode `ntax`/`summary`.
- Asteraceae IGS: two rare loci failed FASTA→NEXUS conversion warnings; alignments otherwise OK.

## How to run gates / benches

```bash
export PATH=$HOME/miniforge3/envs/plastburst-dev/bin:$PATH
python -m pytest -q -p no:cacheprovider tests
ruff check --no-cache . && ruff format --no-cache --check . && mypy plastburstalign
python benchmarking/golden_parity.py check
python benchmarking/golden_parity.py check --reverse-input --threads 3 --strict-order
python benchmarking/calibrate_strategy.py --smoke   # or --fit on a quiet machine
python -m benchmarking.plastbench.plastbench list   # see benchmarking/plastbench/README.md
```
