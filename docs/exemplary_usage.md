### Example usage

#### Install from a local checkout

From the repository root, install the package and its Python dependencies into
the active environment:

```bash
python -m pip install .
```

PlastBurstAlign also requires an external alignment program. Install MAFFT,
MUSCLE, or Clustal Omega separately and make its executable available on `PATH`,
or provide its path with `--alignmenttoolpath`.

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

The installed command and `python -m plastburstalign` accept the same options.
This example extracts coding regions and aligns them with MAFFT:

```bash
plastburstalign -i Input_dataset -o Output_CDS -s cds -a mafft
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
plastburstalign -i Input_dataset -o Output_RNA -s rna --trna --rrna -a mafft
```

RNA types are disabled by default in the command line. Pass `--trna` and
`--rrna` to include both, or pass only one switch to extract only that type.
At least one type must be enabled. For example:

```bash
# tRNAs only
plastburstalign -i Input_dataset -o Output_tRNA -s rna --trna -a mafft
# rRNAs only
plastburstalign -i Input_dataset -o Output_rRNA -s rna --rrna -a mafft
```

RNA features are aligned directly as nucleotide sequences; joined feature parts
are concatenated in strand-correct order, omitting intervening introns.

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
| `-n`, `--numthreads` | Number of CPUs to use, or `auto` | `-n 8` |
| `-c`, `--concat` | Also write a concatenation of successful alignments | `-c` |
| `--config` | Path to a YAML configuration file | `--config config.yaml` |
| `--trna` | Include tRNAs in RNA mode (disabled unless specified) | `--trna` |
| `--rrna` | Include rRNAs in RNA mode (disabled unless specified) | `--rrna` |

Run `plastburstalign --help` for the full list of command-line options. YAML
configuration keys use the parameter names shown in `config.yaml`, such as
`in_dir`, `out_dir`, `select_mode`, and `alignment_tool`. In CLI runs, pass
`--trna` and/or `--rrna` explicitly for RNA mode; these switches default to
disabled and override the corresponding YAML values.

#### Output files

Unaligned nucleotide FASTA files are written directly to the output directory.
For coding regions, unaligned protein FASTA files and intermediate protein
alignments are also written there. Aligned nucleotide FASTA files are placed in
`fasta/`, and their NEXUS conversions are placed in `nexus/`. With `--concat`,
the concatenated FASTA and NEXUS files are written directly to the output
directory. PlastBurstAlign creates these output folders as needed; there is no
need to create or move them manually.

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

The package can also be run from Python by supplying a `UserParameters`
instance:

```python
from plastburstalign import PlastBurstAlign, UserParameters

params = UserParameters(
    in_dir="Input_dataset",
    out_dir="Output_CDS",
    select_mode="cds",
    alignment_tool="mafft",
)
PlastBurstAlign(params).execute()
```

#### Additional CLI Flags & High-Level API

| Option | Description | Example |
| :--- | :--- | :--- |
| `--trim` / `--min-occupancy` | In-flight site filtering (`occupancy=<f>`, `entropy`, `informative-only`, `gappyout`) | `--trim occupancy=0.5` |
| `--partitions` / `--codon-partitions` | Emit `raxml`, `nexus`, `iqtree`, or `partitionfinder` files | `--partitions raxml,nexus` |
| `--subdivide` | Sub-locus chunking (`auto`, `off`, or `force`) | `--subdivide off` |
| `--seed` | Random seed for reproducible aligner runs | `--seed 1` |
| `--memory-limit` / `--smt` / `--no-smt` | Memory admission cap and SMT core budget control | `--memory-limit 8G` |
| `--doctor` / `--estimate` | Print host diagnostics or preflight cost estimate and exit | `plastburstalign --doctor` |
| `--export-slurm` / `--align-partition` / `--finalize-slurm` | Distributed Slurm array workflow | `--export-slurm` |

```python
import plastburstalign as pba

result = pba.align(
    "Input_dataset", mode="cds", out_dir="Output_CDS", seed=1, trim="occupancy=0.5"
)
print(result.summary())
```

