# PlastBench

PlastBench is the benchmark suite of PlastBurstAlign (PBA). It answers one
question reproducibly: *how much faster, and how much leaner in memory, is PBA
than the pipelines people usually script around MAFFT, MUSCLE or Clustal Omega,
on the same plastome collection?*

Everything lives in this directory and needs only the Python standard library
plus `psutil`:

| File | Purpose |
|---|---|
| `plastbench.py` | the harness (`list`, `fetch`, `subsample`, `run`, `report`, `badges`, `pack`) |
| `datasets.json` | dataset manifest: Zenodo location, checksums, status, tier definitions |
| `README.md` | this document |

## Tiers

A *tier* is a fixed number of genomes drawn deterministically from a *master*
dataset (see [Subsampling](#subsampling)). Tiers are nested: the 100 genomes of
`smoke` are also part of every larger tier drawn from the same master.

| Tier | N genomes | Master dataset (`datasets.json`) | Status | Intended machine |
|---|---:|---|---|---|
| `smoke` | 100 | `benchmark_YangEtAl2022_Asteraceae` (155 genomes) | published | laptop / GitHub runner (the CI `benchmark.yml` runs this tier) |
| `1k` | 1,000 | `benchmark_YangEtAl2022_angiosperms` (2,585 genomes) | published | workstation / HPC node |
| `5k` | 5,000 | `benchmark_ZhangNNEtAl2025_13724angiosperms` (13,724 genomes) | published | HPC node (tens of GB RAM) |
| `15k` | 15,000 | `plastbench_15k_master` | **unpublished** | HPC node; needs a master with at least 15,000 genomes (the largest published one has 13,724) |

The published masters are the five archives of Zenodo record
[17137737](https://doi.org/10.5281/zenodo.17137737) (concept DOI
`10.5281/zenodo.13403720`). The larger tiers cannot be reproduced on a typical
laptop (about 6 cores, 16 GB RAM); they are defined and runnable but are meant
to be executed on an HPC node. Nothing in this repository claims results for
them.

## Quick start (smoke tier)

```bash
export PB=benchmarking/plastbench/plastbench.py

python $PB list                                   # validates datasets.json
python $PB fetch benchmark_YangEtAl2022_Asteraceae --dest data
python $PB subsample --src data/benchmark_YangEtAl2022_Asteraceae --tier smoke --out data/smoke100
python $PB run --input data/smoke100 --mode igs --threads 8 --out results/smoke.json
python $PB report results/smoke.json                          # Markdown table
python $PB badges results/smoke.json --out-dir results/badges  # shields.io JSON
```

`run` needs `python -m plastburstalign` to work (install the package with
`pip install -e .`, or run from a checkout: the harness puts the checkout on
`PYTHONPATH` for the PBA subprocesses). Tools that are not installed are
recorded as `skipped`, so a partial installation still produces a result file.
Restrict the run with `--tools pba mafft` and choose the speed-up reference with
`--baseline` (default `mafft`).

Useful `run` options: `--mode cds|igs|int|rna` (default `igs`), `--threads N|auto`
(threads of PBA; the baselines are single-threaded by design), `--min-taxa`,
`--min-len` (default `5` and `6`, as in `benchmarking_IGS.py`), `--timeout`
(seconds per tool), `--pba-arg FLAG` (repeatable, extra flags for PBA),
`--rss-metric rss|pss`, `--workdir DIR` (keep intermediates), `--tier`/`--dataset-name`
(labels stored in the result), `--reference-tree PATH|baseline`.

## What is measured

| Key | What runs |
|---|---|
| `pba` | `python -m plastburstalign -i SUBSET -o OUT -s MODE -t T -l L -n THREADS -c` |
| `mafft` | one locus after the other: `mafft --auto --thread 1`, then a concatenation (sequential baseline, **the default speed-up reference**) |
| `muscle` | the same loop with MUSCLE 5: `muscle -super5 IN -threads 1 -output OUT` |
| `clustalo` | the same loop with Clustal Omega: `clustalo --auto --threads=1` |
| `shell` | a generated bash script (`mafft --auto` loop + `awk` concatenation), the way many labs script it |

Per tool the result records:

* **wall time** of the whole pipeline (`wall_seconds`);
* **peak memory of the whole process tree**: a background thread polls
  `psutil` every 50 ms and keeps the maximum of the summed RSS (or PSS with
  `--rss-metric pss` on Linux) of the launched process and all its descendants
  (`peak_rss_bytes`). RSS double-counts pages shared between forked workers; use
  `pss` for a fair number when PBA runs many workers. Spikes shorter than the
  polling interval can be missed;
* `ntax` and `nchar` of the resulting concatenated matrix, the number of aligned
  loci and a SHA-256 of the matrix;
* optionally a **topology metric**: if `FastTree` or `iqtree2` is on `PATH` and
  `--reference-tree` is given (a Newick file, or `baseline` to compare every
  pipeline against the baseline's tree), a quick ML tree is built from each
  matrix and the normalized Robinson-Foulds distance
  (`RF / (|splits_a| + |splits_b|)`, on the shared taxa) is stored. Without a
  reference the harness skips this and says so in `notes`.

### How the baselines get their input (and why it is fair)

All baselines align **exactly the loci PBA extracts**. The GenBank files are
scanned once by PBA itself with a no-op "aligner"
(`--alignmentcommand 'cp %INPUT% %OUTPUT%'`); the unaligned per-locus FASTA files
it writes (`nucl_<locus>.unalign.fasta`) are then fed to each baseline. That
needs no second GenBank parser and no access to PBA internals. It is *favourable
to the baselines*: their extraction runs at PBA speed, and its time is added to
each baseline as phase `extract` (`wall_seconds = extract + align_concat`;
both phases are stored under `phases`). In `cds` mode PBA aligns proteins and
back-translates, while the baselines align the nucleotide CDS directly.

### Results JSON (`schema_version` 1)

```json
{
  "schema_version": 1,
  "generator": "plastbench 1",
  "created_utc": "2026-01-01T12:00:00Z",
  "dataset": {"name": "...", "tier": "smoke", "n_genomes": 100, "seed": "plastbench-v1",
              "listing_sha256": "...", "mode": "igs", "min_num_taxa": 5, "min_seq_length": 6},
  "host": {"system": "Linux", "release": "...", "machine": "x86_64", "python": "3.12.1",
           "cpu_model": "...", "cpu_count_logical": 12, "cpu_count_physical": 6,
           "ram_total_bytes": 0, "ram_available_bytes": 0, "in_ci": false},
  "tool_versions": {"python": "...", "plastburstalign": "...", "pba_git_head": "...",
                    "pba_git_dirty": "no", "mafft": "...", "muscle": "...", "clustalo": "...",
                    "fasttree": null, "iqtree": null},
  "baseline": "mafft",
  "settings": {"threads": 8, "rss_metric": "rss", "poll_interval_s": 0.05,
               "timeout_s": null, "pba_extra_args": []},
  "tools": {
    "pba":   {"status": "ok", "reason": null, "command": ["..."], "threads": 8,
              "wall_seconds": 41.2, "peak_rss_bytes": 912000000, "phases": null,
              "n_loci": 120, "ntax": 100, "nchar": 150000, "matrix_sha256": "...", "tree": null},
    "mafft": {"status": "ok", "wall_seconds": 310.5, "peak_rss_bytes": 310000000,
              "phases": {"extract": {"wall_seconds": 5.1, "peak_rss_bytes": 0},
                         "align_concat": {"wall_seconds": 305.4, "peak_rss_bytes": 0}},
              "tree": {"method": "fasttree", "rf": 12, "normalized_rf": 0.06, "n_shared_taxa": 100}}
  },
  "notes": ["..."]
}
```

`status` is one of `ok`, `failed`, `skipped`, `timeout`; failed or skipped tools
carry a `reason`. The host block deliberately contains no host name or user
name. `plastbench.py run` exits non-zero only if the PBA run itself fails
(or if any tool fails with `--strict`), so baseline failures are data, not
crashes.

### Report and badges

`report` merges one or more result files into one Markdown table (sorted by N)
with the speed-up of every tool relative to the baseline
(`baseline wall time / tool wall time`). `badges` writes two
[shields.io endpoint](https://shields.io/badges/endpoint-badge) files for the
largest successful result: `speedup.json` and `peak_memory.json`, each
`{"schemaVersion": 1, "label": ..., "message": ..., "color": ...}`.

## Subsampling

`plastbench.py subsample --src DIR --n N|--tier T --out DIR` selects the first N
files (default extension `.gb`, non-recursive) when ordered by

```
sha1(f"{filename}\0{seed}")  ascending, ties broken by filename
```

with the fixed default seed `plastbench-v1`. The order does not depend on
`PYTHONHASHSEED`, the file system, the locale or the platform, so everybody gets
the same genomes. The output directory holds symlinks (or copies with `--copy`)
and a `subset.json` with the listing, its SHA-256 (`listing_sha256`), the seed
and the method; it contains no absolute paths, so it is byte-reproducible.
Cite the tier, the seed and `listing_sha256` when publishing numbers.

## Running the large tiers on an HPC node

```bash
#!/bin/bash
#SBATCH --job-name=plastbench-5k
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=48:00:00
module load python mafft muscle clustalo fasttree   # site specific
python benchmarking/plastbench/plastbench.py fetch benchmark_ZhangNNEtAl2025_13724angiosperms --dest $SCRATCH/pb
python benchmarking/plastbench/plastbench.py subsample --tier 5k \
  --src $SCRATCH/pb/benchmark_ZhangNNEtAl2025_13724angiosperms --out $SCRATCH/pb/5k
python benchmarking/plastbench/plastbench.py run --input $SCRATCH/pb/5k --tier 5k \
  --threads $SLURM_CPUS_PER_TASK --rss-metric pss --timeout 129600 \
  --reference-tree baseline --out results/5k.json
```

Note that the Python baselines hold the whole supermatrix in memory while
concatenating, like most hand-written pipelines; that cost is part of their
measured peak RSS. Run the baselines in a separate job (`--tools mafft muscle`) if their
sequential run would exceed the PBA job's time limit, then merge with
`report results/*.json`. Use `--workdir` on node-local scratch to keep
intermediates off the shared file system.

## Publishing datasets to Zenodo (maintainers)

This requires a Zenodo account and cannot be automated from this repository.

1. Assemble the master directory of GenBank flat files (`*.gb`, one plastome per
   file, e.g. with `efetch`; see `docs/generating_benchmarking_data.md`) and keep
   the accession list next to it.
2. Create a reproducible archive and its checksums:
   `python benchmarking/plastbench/plastbench.py pack MASTER_DIR --out NAME.tar.gz`
   (sorted entries, no timestamps or owners; it prints `sha256`, `md5`, size and
   file count).
3. On Zenodo, add the archive as a new file of the PlastBurstAlign benchmark
   record (or create a new version/record), publish, and note the record id,
   the file URL (`https://zenodo.org/records/<ID>/files/NAME.tar.gz?download=1`)
   and the md5 that Zenodo displays (it must equal the one printed by `pack`).
4. Edit `datasets.json`: set `zenodo_record`, `url`, `sha256`, `md5`,
   `size_bytes`, `n_genomes`, `status: "published"` (and the same in `tiers`),
   clear `todo`.
5. Run `python benchmarking/plastbench/plastbench.py list` and
   `python -m pytest -q tests/test_plastbench.py`; the manifest validator
   rejects missing checksums, unknown tiers and inconsistent statuses.
6. Mention the new dataset in `CHANGELOG.md`. Software releases are archived
   separately through the Zenodo-GitHub integration (see `.zenodo.json` and
   `CITATION.cff` at the repository root).

Nothing in `datasets.json` is a placeholder or an invented value. Every
published entry carries the SHA-256 that was computed by streaming the archive
from Zenodo, the MD5 that Zenodo reports for the same file (the two agree), its
size and its genome count; the unpublished entry has `sha256: null` and cannot
be fetched. `fetch` verifies the SHA-256 (or, for a new entry without one, the
Zenodo MD5) and refuses to use an archive it cannot verify unless
`--allow-unverified` is given.

## Continuous benchmarking

`.github/workflows/benchmark.yml` (manual dispatch and published releases) runs
the smoke tier on a GitHub-hosted runner (`fetch` -> `subsample --tier smoke` ->
`run --reference-tree baseline` -> `report` -> `badges`), uploads the results as
an artifact and, for `main` or a release, commits `speedup.json`,
`peak_memory.json` and the latest result to the orphan `badges` branch. Hosted
runners are shared and noisy; treat those badges as an indicator, not as a
measurement. One-time setup (nothing to create by hand; the first run creates
the branch) and the README badge URLs are described at the top of the workflow
file.

## Tests

`tests/test_plastbench.py` covers the subsampler (determinism across seeds and
`PYTHONHASHSEED`), the manifest schema, badge/report rendering, the RSS sampler,
concatenation (including the generated bash/awk pipeline), the Robinson-Foulds
code and `fetch` through `file://` URLs. It needs no network and runs no
benchmark.
