# Changelog

All notable changes to PlastBurstAlign are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
(pre-releases follow [PEP 440](https://peps.python.org/pep-0440/), e.g. `3.0.0rc1`).

## [Unreleased]

Add entries for pull requests here (see [CONTRIBUTING.md](CONTRIBUTING.md)).

## [3.0.0rc1] - Unreleased

> DRAFT. This section was written from the v3.0 implementation plan before the
> work was finished; the release manager must verify every bullet against the
> shipped code and the final test results, tighten the wording and set the date.

### Added

- **Python API** (`plastburstalign.api`): `pba.align(input_dir, mode="cds+igs+int", ...)`
  returning an `AlignmentResult` (`ntax`, `nchar`, informative sites, per-locus
  report, `write_nexus/phylip/fasta`, `summary()`), plus `pba.autotune()` and the
  pre-flight `pba.estimate()`. Multi-mode runs merge into one supermatrix with
  mode-prefixed partition names. A run manifest (JSON: versions, parameters,
  per-locus strategy/chunks, host profile, output SHA-256) is written with every
  run. The package is typed (`py.typed`) with full exports in `__init__.py`.
- **New command-line flags** (existing flags are unchanged): `--trim`,
  `--min-occupancy`, `--partitions`, `--keep-unaligned/--no-keep-unaligned`,
  `--subdivide`, `--smt`, `--memory-limit`, `--time-budget`, `--seed`,
  `--estimate`, `--doctor`, `--dashboard/--no-dashboard`; console script alias
  `pba` for `plastburstalign`.
- **Rich dashboard** with a pre-flight estimate, live per-locus progress and a
  post-flight scaling report (parallel efficiency, straggler tax, data density),
  driven by a `ProgressSink` event bus (silent sink for library use, plain
  logging in CI).
- **Autonomic host profiling** (`HostProfile`: physical/logical cores, cgroup
  and Slurm limits, NUMA, SIMD, memory, storage tiers) and `TuningPlan`
  (`autotune`) with a calibrated memory model and admission control.
- **Aligner resolver** (explicit path, `PBA_MAFFT`, `PATH`, active venv/conda,
  HPC module variables; version validation) and a deterministic pure-Python
  **built-in fallback aligner** used as a last resort (set
  `PBA_FORCE_BUILTIN_ALIGNER=1` to force it); `plastburstalign --doctor` reports
  the host, the resolved tools and recommendations.
- **Data engine**: dense `uint8` locus matrices with a fast FASTA parser,
  vectorized site filters (occupancy, entropy, parsimony-informative,
  `gappyout`), a streaming `SupermatrixBuilder`, partition files for
  RAxML-NG/IQ-TREE/NEXUS including codon positions, and FASTA/PHYLIP/NEXUS
  writers.
- **Tiered scratch** (tmpfs / `$SLURM_TMPDIR` / disk) with byte-level admission
  control and unlink-after-parse of intermediates.
- **Scheduler**: CPU-set leases, memory admission, straggler promotion and
  deterministic anchor-based sub-locus chunking for very long loci.
- **Hardening**: per-feature, per-record, per-chunk and per-locus error
  isolation counted in the run report instead of aborting runs; seeded GenBank
  fuzz corruptors and a synthetic plastome generator for tests.
- **Determinism guarantees**: identical FASTA/NEXUS/partition SHA-256 across
  repeated runs, different thread counts and different `PYTHONHASHSEED`.
- **PlastBench** (`benchmarking/plastbench/`): dataset manifest with checksums,
  deterministic subsampler (100 / 1,000 / 5,000 / 15,000 genomes), runner for
  PBA vs sequential MAFFT, MUSCLE 5, Clustal Omega and a bash pipeline (wall time,
  peak process-tree RSS, optional Robinson-Foulds distance), shields.io badge
  JSON, and a `benchmark.yml` workflow.
- **Golden parity gate** (`benchmarking/golden_parity.py`,
  `tests/golden/asteraceae_parity.json`) that pins canonicalized output hashes of
  the `cds`, `igs`, `int` and `rna` modes.
- **CI**: Ubuntu and macOS x Python 3.11-3.13 x aligner (`mafft` | `builtin`)
  matrix, ruff, ruff format, mypy, coverage gate, `build` + `twine check` and a
  clean-venv install test; publishing now requires the test workflow to pass.
- **Project governance**: `CITATION.cff`, `.zenodo.json`, `CONTRIBUTING.md`,
  `SECURITY.md`, `CODE_OF_CONDUCT.md`, issue forms and a pull request template;
  `environment.yml` and `dev`/`test`/`docs` extras.

### Changed

- **Version `3.0.0rc1`**; development status Beta; Python 3.11-3.13 supported
  (floor stays 3.11).
- **Dependencies**: added `numpy>=1.26` and `rich>=13`; removed the unused
  `requests`; license metadata is now the SPDX expression `BSD-3-Clause`.
- **Alignment strategy** is a pure function of the data (taxa, lengths, alphabet,
  configuration) and uses a calibrated work model (`N*L^alpha` plus an
  `N^2*L` term); hardware only changes scheduling, parallelism, memory admission
  and the scratch tier. The old taxon-count gates (`N >= 1000/5000`) are removed;
  the legacy PartTree/retree threshold keys are still honored as explicit
  overrides with a deprecation warning.
- **Slurm export**: partition weights come from the region summary (`N*L^alpha`)
  instead of file sizes, and workers export their own unaligned FASTA from the
  SQLite feature store, so no shared unaligned files are needed.
- `num_threads="auto"` means the physical, usable cores.
- README license badge now correctly shows BSD-3-Clause; `build-and-publish`
  builds on Python 3.12 and marks pre-releases as GitHub pre-releases.

### Fixed

- Slurm path for real runs: `_execute_export` now writes the unaligned FASTA
  files and the worker functions no longer reference the undefined
  `supports_max_len`; the protein subset path calls `BackTranslation` correctly
  and back-translates.
- `items()` hydration of every region just to test for emptiness in
  `_nuc_MSA`/`_prot_MSA`.
- Non-deterministic taxon order (extraction chunks were stored in completion
  order): rows are stored by `file_index` and read back `ORDER BY file_index, id`.
- `MAFFT.align` ignored `max_len`, and `_build_command` hard-coded `min_len=6`
  when both counts were given, silently disabling the short-sequence guard.
- `helpers.resolve_scratch_dir` used `os`/`tempfile` without importing them;
  missing `Any` imports; a stray `__name__` override in `__init__`.
- `concat_MSAs` loaded every locus at once, defeating the staged accumulator.
- Lint, format and type-check failures (CI was red); `build-and-publish.yaml`
  built on Python 3.9 although `requires-python>=3.11`.

### Removed

- The unused `requests` dependency.

[Unreleased]: https://github.com/michaelgruenstaeudl/PlastBurstAlign/compare/v3.0.0rc1...HEAD
[3.0.0rc1]: https://github.com/michaelgruenstaeudl/PlastBurstAlign/releases/tag/v3.0.0rc1
