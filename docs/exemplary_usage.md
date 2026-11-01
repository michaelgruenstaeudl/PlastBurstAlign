### Example usage

#### Install from a local checkout

From the repository root, install the package and its Python dependencies into
the active environment:

```bash
python -m pip install -e .
```

PlastBurstAlign also requires an external alignment program. Install MAFFT,
MUSCLE, or Clustal Omega separately and make its executable available on `PATH`,
or provide its path with `--alignmenttoolpath`. Without an external aligner the
built-in center-star fallback is used (slower; emits a quality warning).

#### Prepare input data

Put the input plastid genome GenBank flatfiles in a directory. The default
extension is `.gb`; use `--fileext` (or `-f`) if your files use another
extension. For example:

```text
Input_dataset/
  genome_1.gb
  genome_2.gb
```

#### Run from the command line

The installed commands `plastburstalign` / `pba` and `python -m plastburstalign`
accept the same options. This example extracts coding regions and aligns them
with MAFFT:

```bash
plastburstalign -i Input_dataset -o Output_CDS -s cds -a mafft -c --seed 1
```

The output directory is created if it does not exist. The main selection modes
are:

| Mode | Regions |
|------|---------|
| `cds` | Coding sequences |
| `int` | Introns |
| `igs` | Intergenic spacers |
| `rna` | Annotated tRNAs and rRNAs, aligned as nucleotides |

Use a separate output directory for each run if you want to keep its results
separate. For example, to align tRNAs and rRNAs:

```bash
plastburstalign -i Input_dataset -o Output_RNA -s rna -a mafft -c --seed 1
```

RNA mode includes both types by default. Add `--no-rrna` for tRNAs only or
`--no-trna` for rRNAs only. At least one type must remain enabled. RNA features
are aligned directly as nucleotide sequences; joined feature parts are
concatenated in strand-correct order, omitting intervening introns.

Common options include:

| Option | Description | Example |
|--------|-------------|---------|
| `-i`, `--inpd` | Input directory containing GenBank files | `-i Input_dataset` |
| `-o`, `--outd` | Output directory | `-o Output_CDS` |
| `-s`, `--selectmode` | Region type: `cds`, `int`, `igs`, or `rna` | `-s igs` |
| `-a`, `--alignmenttool` | Alignment program: `mafft`, `muscle`, or `clustal` | `-a mafft` |
| `--alignmenttoolpath` | Path to the alignment executable when it is not on `PATH` | `--alignmenttoolpath /path/to/mafft` |
| `-f`, `--fileext` | Input file extension (default: `.gb`) | `-f .gbk` |
| `-l`, `--minseqlength` | Minimum region length in base pairs | `-l 9` |
| `-t`, `--minnumtaxa` | Minimum number of taxa, or a relative frequency from 0 to 1 | `-t 0.1` |
| `-n`, `--numthreads` | Number of CPUs to use, or `auto` (physical usable cores) | `-n 8` |
| `-c`, `--concat` | Also write a concatenation of successful alignments | `-c` |
| `--config` | Path to a YAML configuration file | `--config config.yaml` |
| `--no-trna` | Exclude tRNAs in RNA mode | `--no-trna` |
| `--no-rrna` | Exclude rRNAs in RNA mode | `--no-rrna` |
| `--trim` | In-flight site filter | `--trim occupancy=0.5` |
| `--min-occupancy` | Occupancy threshold (paired with `--trim`) | `--min-occupancy 0.5` |
| `--partitions` | Partition formats (`raxml`, `nexus`, `iqtree`, …) | `--partitions raxml,nexus` |
| `--codon-partitions` | CDS codon-position partitions | `--codon-partitions` |
| `--subdivide` | Sub-locus chunking: `auto`, `off`, or `force` | `--subdivide off` |
| `--keep-unaligned` / `--no-keep-unaligned` | Keep unaligned FASTA in `out_dir` | `--no-keep-unaligned` |
| `--smt` / `--no-smt` | Include SMT siblings in the CPU budget | `--smt` |
| `--memory-limit` | Memory admission cap | `--memory-limit 8G` |
| `--seed` | MAFFT `--randomseed` for deterministic runs | `--seed 1` |
| `--phylip` | Also write relaxed PHYLIP concatenation | `--phylip` |
| `--doctor` | Diagnose host, aligners and scratch; exit | `plastburstalign --doctor` |
| `--export-slurm` | Emit Slurm array scripts instead of aligning locally | `--export-slurm` |

Run `plastburstalign --help` for the full list. YAML configuration keys use the
parameter names shown in `config.yaml`, such as `in_dir`, `out_dir`,
`select_mode`, `alignment_tool`, `trim`, `subdivide`, and `seed`.

#### Output files

Unaligned nucleotide FASTA files are written to the output directory when
`--keep-unaligned` is enabled (the CLI default). For coding regions, unaligned
protein FASTA files and intermediate protein alignments may also appear there
(or only in scratch when unaligned files are not kept). Aligned nucleotide
FASTA files are placed in `fasta/`, and their NEXUS conversions are placed in
`nexus/` when individual NEXUS writing is enabled. With `--concat`, the
concatenated FASTA and NEXUS files plus partition sidecars
(`*.partitions.txt`, `*.partitions.nex`, …) are written directly to the output
directory. Every successful run also writes `run_manifest.json` (parameters,
strategy per locus, host/tuning summary, versions).

#### Slurm sketch

```bash
plastburstalign -i Input_dataset -o Out_IGS -s igs -a mafft -c \
  --export-slurm --slurm-partitions 10 --seed 1
# submit the generated array / finalizer scripts, then:
# plastburstalign --finalize-slurm --manifest Out_IGS/slurm_manifest.json ...
```

Workers export unaligned FASTA from the SQLite feature store into node-local
scratch; shared unaligned files are not required.

#### Run the benchmarking script

The repository includes [`benchmarking/test_script.py`](../benchmarking/test_script.py)
to run PlastBurstAlign once on a dataset with a chosen thread count and selection
mode. First, place or unpack the GenBank dataset in a directory under
`benchmarking/` (for example, `benchmarking/benchmarking1/`). From the
repository root, run:

```bash
python benchmarking/test_script.py benchmarking1 8 cds
```

The positional arguments are the dataset directory, number of CPUs, and
selection mode (`cds`, `int`, `igs`, or `rna`), in that order. A relative
dataset path is resolved from the `benchmarking/` directory; an absolute path
can also be supplied. The script runs `python -m plastburstalign` and writes
results under `benchmarking/` in a directory named
`<dataset-name>_output_<mode>_<threads>threads` under `benchmarking/`. For
example, the command above writes to
`benchmarking/benchmarking1_output_cds_8threads/`.

Install the package dependencies and an external alignment program before
running the benchmark. The script imports Python's `resource` module, so run it
on Linux or macOS (or in a compatible Unix-like environment such as WSL); it
does not run as-is on native Windows Python.

#### Python API

Preferred entry point:

```python
import plastburstalign as pba

result = pba.align(
    "Input_dataset",
    mode="cds",
    out_dir="Output_CDS",
    seed=1,
    trim="occupancy=0.5",
)
print(result.summary())
```

`pba.estimate(...)` returns a pre-flight cost estimate; `pba.autotune(...)`
returns a `TuningPlan`. Lower-level `UserParameters` + `PlastBurstAlign` remains
available:

```python
from plastburstalign import PlastBurstAlign, UserParameters

params = UserParameters(
    in_dir="Input_dataset",
    out_dir="Output_CDS",
    select_mode="cds",
    alignment_tool="mafft",
    concat=True,
    seed=1,
    subdivide="off",
    trim="occupancy=0.5",
    num_threads="auto",
)
PlastBurstAlign(params).execute()
```
