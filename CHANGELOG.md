# Changelog

All notable changes to PlastBurstAlign are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-releases follow [PEP 440](https://peps.python.org/pep-0440/), e.g. `3.0.0rc1`).

## [Unreleased]

Add entries for pull requests here (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## [3.0.0rc1] - Unreleased

Release-candidate packaging for the v3 “gold standard” line. Modules and the
monolithic / Slurm pipeline are wired; strategy and memory-model constants are
still **provisional** (`calibrated=False`). The public library API ships:
`pba.align` / `pba.estimate` / `pba.autotune` and `AlignmentResult`
(`plastburstalign/api.py`). CLI includes `--estimate`, `--dashboard` /
`--no-dashboard`, and a `--time-budget` stub (value recorded; host-adaptive
scheduling not fully wired).

### Added

- **Public API**: `align`, `estimate`, `autotune`, and `AlignmentResult`
  (exported from `plastburstalign` / `import plastburstalign as pba`).
- **New command-line flags** (existing flags unchanged): `--trim`,
  `--min-occupancy`, `--partitions`, `--codon-partitions`,
  `--keep-unaligned/--no-keep-unaligned`, `--subdivide`, `--smt`,
  `--memory-limit`, `--seed`, `--phylip`, `--doctor`, `--estimate`,
  `--dashboard` / `--no-dashboard`, `--time-budget` (stub); console script
  alias `pba` for `plastburstalign`.
- **Rich dashboard / ProgressSink** modules for pre-flight, live progress and
  post-flight scaling reports (library + TTY; plain logging in CI).
- **Autonomic host profiling** (`HostProfile`) and `TuningPlan` (`autotune`)
  with a calibrated-shaped (still provisional) memory model and admission
  control; `-n auto` resolves to usable physical cores.
- **Aligner resolver** (explicit path, `PBA_MAFFT`, `PATH`, active venv/conda,
  HPC module variables; version validation) and a deterministic pure-Python
  **built-in fallback aligner**; `plastburstalign --doctor` reports host,
  resolved tools and recommendations.
- **Data engine**: dense `uint8` locus matrices, vectorized site filters
  (occupancy, entropy, informative, `gappyout`), streaming
  `SupermatrixBuilder`, partition writers (RAxML-NG / IQ-TREE / NEXUS, optional
  codon positions), FASTA / PHYLIP / NEXUS writers.
- **Tiered scratch** (tmpfs / `$SLURM_TMPDIR` / disk) with byte-level admission
  and unlink-after-parse unless `keep_unaligned`.
- **Scheduler**: CPU-set leases, memory admission, straggler promotion, and
  deterministic anchor-based sub-locus chunking (`--subdivide`).
- **Run manifest** (`run_manifest.json`: versions, params, per-locus strategy /
  chunks, host/tuning, output paths).
- **Hardening**: per-feature / record / chunk / locus error isolation; GenBank
  fuzz corruptors and synthetic plastome generator (`tests/synth.py`).
- **Determinism tests**: identical FASTA / NEXUS / partition SHA-256 across
  repeated runs, different thread counts (`--subdivide off`), and different
  `PYTHONHASHSEED`.
- **PlastBench** (`benchmarking/plastbench/`), golden parity gate
  (`benchmarking/golden_parity.py`), and strategy smoke calibration
  (`benchmarking/calibrate_strategy.py --smoke`).
- **CI**: Ubuntu and macOS × Python 3.11–3.13 × aligner (`mafft` | `builtin`);
  ruff, format, mypy, coverage, `build` + `twine check`.
- **Governance / packaging**: `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue/PR templates, `environment.yml`,
  `py.typed`, `dev` / `test` extras; docs rewrite and
  [docs/calibration.md](docs/calibration.md).

### Changed

- **Version `3.0.0rc1`**; development status Beta; Python 3.11–3.13.
- **Dependencies**: `numpy>=1.26`, `rich>=13`, `psutil`; removed unused
  `requests`; SPDX license `BSD-3-Clause`.
- **Alignment strategy** is a pure function of the data (work model
  `N·L^α + β·N²·L`); hardware changes scheduling / scratch only. Legacy
  PartTree/retree N-count gates removed (old config keys deprecated overrides).
  Constants in `strategy_constants.py` remain provisional until calibration.
- **Slurm**: partition weights from region work estimates; workers export
  unaligned FASTA from the SQLite store into node-local scratch.
- README license badge corrected to BSD-3-Clause; publish workflow builds on
  Python ≥ 3.11 and marks pre-releases appropriately.

### Fixed

- Slurm export / worker path for real runs (unaligned FASTA, `supports_max_len`,
  protein subset back-translation).
- Non-deterministic taxon order (store and read by `file_index`).
- `MAFFT.align` / `min_len` plumbing; `helpers.resolve_scratch_dir` imports;
  `concat_MSAs` loading every locus at once; stray `__init__` issues.
- Lint / format / type-check CI failures.

### Removed

- The unused `requests` dependency.

[Unreleased]: https://github.com/michaelgruenstaeudl/PlastBurstAlign/compare/v3.0.0rc1...HEAD
[3.0.0rc1]: https://github.com/michaelgruenstaeudl/PlastBurstAlign/releases/tag/v3.0.0rc1
