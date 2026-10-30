# PlastBurstAlign

[![CI](https://github.com/michaelgruenstaeudl/PlastBurstAlign/actions/workflows/ci.yml/badge.svg)](https://github.com/michaelgruenstaeudl/PlastBurstAlign/actions/workflows/ci.yml)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)
[![Type Checked: mypy](https://img.shields.io/badge/type%20checked-mypy-blue.svg)](https://mypy-lang.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

A high-performance Python tool to extract and align genes, introns, and intergenic spacers across thousands of plastid genomes without memory exhaustion.

---

### Purpose
This software tool is designed for large-scale quality assessment of organellar genome annotations. It detects annotation discrepancies in large genome datasets by comparing sequence and annotation features through automated multiple sequence alignments across homologous regions.

### Background
The multiple sequence alignment (MSA) of a set of plastid genomes is challenging. At least five factors contribute to this challenge:
- First, the plastid genome is a mosaic of individual genome regions. An MSA procedure must identify, extract, group, and align homologous regions across the input genomes.
- Second, many plastid genomes contain annotation errors in gene positions and/or gene names. An MSA procedure must automatically exclude incorrectly annotated regions from the alignment procedure.
- Third, plastid genomes comprise both coding and noncoding regions, which require different alignment strategies (e.g., amino acid-based for genes, nucleotide-based for introns and intergenic spacers). An MSA procedure must apply the appropriate strategy automatically.
- Fourth, modern plastid genome studies often involve hundreds, if not thousands, of complete genomes. An MSA procedure must perform sequence alignment within practical time frames (e.g., hours rather than days) and within bounded system memory limits.
- Fifth, manually excluding user-specified genome regions after alignment is prohibitively complex. An MSA procedure must support the automatic exclusion of user-specified regions before the alignment starts.

The software `plastburstalign` addresses these and other challenges: it provides an MSA procedure that extracts and aligns genes, introns, and intergenic spacers across hundreds or thousands of plastid genomes in an autonomous, memory-safe fashion.

### Overview of Process
![Depiction of plastomes being split according to specified marker type; the extracted sequences are then aligned and concatenated](docs/PlastBurstAlign_ProcessOverview.png)

### Key Features

#### Scalable Architecture & Memory Safety
- **Disk-Backed Feature Store:** Uses an indexed SQLite WAL (Write-Ahead Logging) storage engine to stream genomic records directly to disk, keeping coordinator memory footprint below 50 MB regardless of dataset size (tested on >13,700 angiosperm plastomes).
- **Bounded Concurrency Engine:** Uses `BoundedTaskQueue` workers to dynamically decouple memory consumption from core count, preventing Out-Of-Memory (OOM) crashes on high-core workstations and HPC clusters.
- **Staged Matrix Streaming:** Concatenates multigenic NEXUS and FASTA alignments via temporary accumulator storage, avoiding massive in-memory matrix overhead.
- **Cross-Platform Multiprocessing:** Safely handles process initialization (`spawn` on macOS/Windows, `fork` on Linux) with environment override capabilities (`PLASTBURST_MP_METHOD`).

#### Extraction & Quality Control
- **Extraction of All Primary Region Types:**
  - Coding sequences (`cds`)
  - Introns (`int`)
  - Intergenic spacers (`igs`)
  - tRNAs and rRNAs (`rna`; nucleotide-only alignment)
- **Automatic Exon Splicing:** Merges adjacent cis-spliced exons and groups/repositions trans-spliced exons (e.g., *rps12*) via `ExonSpliceHandler`.
- **Gene Completeness Checks:** Verifies valid start and stop codons for protein-coding genes.
- **Inverted Repeat Deduplication:** Automatically normalizes region names and ignores reverse-flanking duplicates so IR-mirrored spacers are not counted twice.
- **Flexible Region Filtering:**
  - Filter by minimum sequence length (`-l`).
  - Filter by absolute taxon count or relative dataset frequency (`-t`).
  - Target-exclude user-specified genes or entire regions (`--exclcds`, `--exclreg`).
- **Standardized Naming Schemes:**
  - Homogenizes idiosyncratic tRNA names to canonical formats (e.g., `trnA_UAA`).
  - Maps rRNA products to standard plastid gene labels (`rrn16`, `rrn23`, `rrn4.5`, `rrn5`).

#### Supported Aligners
- [MAFFT](https://mafft.cbrc.jp/alignment/software/)
- [MUSCLE](https://www.drive5.com/muscle/)
- [Clustal Omega](http://www.clustal.org/omega/)
- Custom aligner commands via CLI parameter `--alignment_command`

---

### Installation

#### Option 1: Conda Environment (Recommended)
This environment installs Python, all dependencies, and external alignment binaries (`mafft`, `muscle`, `clustalo`):
```bash
git clone [https://github.com/michaelgruenstaeudl/PlastBurstAlign.git](https://github.com/michaelgruenstaeudl/PlastBurstAlign.git)
cd PlastBurstAlign
conda env create -f environment.yml
conda activate plastburstalign
```

#### Option 2: Pip with System Aligners

```bash
pip install plastburstalign

# Debian/Ubuntu systems: install alignment binaries
sudo apt-get update && sudo apt-get install -y mafft muscle clustalo

```

---

### Usage

#### Command-Line Interface

```bash
plastburstalign \
  -i Input_dataset \
  -o Output_dataset \
  -s cds \
  -a mafft \
  -n 8

```

#### Detailed Parameter Reference

| Option | Description | Example |
| --- | --- | --- |
| `-i` | Input directory containing GenBank flat files (`.gb`, `.gbk`) | `-i Input_dataset` |
| `-o` | Output destination directory | `-o Output_dataset` |
| `-s` | Sequence type to extract (`cds`, `int`, `igs`, or `rna`) | `-s cds` |
| `-a` | Alignment tool (`mafft`, `muscle`, `clustalo`, `custom`) | `-a mafft` |
| `-n` | Worker threads/processes for extraction and alignment | `-n 16` |
| `-l` | Minimum sequence length (bp); shorter regions are dropped | `-l 9` |
| `-t` | Taxon threshold: absolute count (`int >= 1`) or relative frequency (`0.0 < float <= 1.0`) | `-t 0.1` |
| `--no-trna` | Disable tRNAs in `rna` mode (default: enabled) | `--no-trna` |
| `--no-rrna` | Disable rRNAs in `rna` mode (default: enabled) | `--no-rrna` |
| `--config` | Path to YAML configuration file | `--config config.yaml` |

#### Extracting tRNAs and rRNAs

Extract annotated non-coding RNAs as DNA nucleotide sequences without translation:

```bash
# Both tRNAs and rRNAs (default)
plastburstalign -i Input_dataset -o Output_RNA -s rna -a mafft

# tRNAs only
plastburstalign -i Input_dataset -o Output_tRNA -s rna --no-rrna -a mafft

# rRNAs only
plastburstalign -i Input_dataset -o Output_rRNA -s rna --no-trna -a mafft

```

---

### Python API

```python
from plastburstalign import PlastBurstAlign, UserParameters

params = UserParameters(
    in_dir="Input_dataset",
    out_dir="Output_dataset",
    select_mode="cds",
    align_tool="mafft",
    num_threads=8,
)

burst = PlastBurstAlign(params)
burst.execute()
```

---

### Development & Testing

#### Running the Test Suite

The repository includes a 46-test regression suite covering storage, bounded concurrency, custom aligners, and staged matrix output:

```bash
pytest -v tests/

```

#### Code Quality & Pre-commit Hooks

This project enforces code quality and formatting via [Ruff](https://github.com/astral-sh/ruff) and [Mypy](https://mypy-lang.org/):

```bash
# Install pre-commit
pip install pre-commit
pre-commit install

# Run checks across all files
pre-commit run --all-files

```

---

### Benchmarking

Automated benchmarking harnesses are provided in `benchmarking/`:

```bash
cd benchmarking
# CDS mode on Asteraceae benchmark with 8 cores
python benchmarking_CDS.py benchmark_YangEtAl2022_Asteraceae 8

# Intron mode
python benchmarking_INT.py benchmark_YangEtAl2022_Asteraceae 8

# Intergenic spacer mode
python benchmarking_IGS.py benchmark_YangEtAl2022_Asteraceae 8

```

---

### Details on Biological Operations

#### Exon Splicing

Plastid genes often consist of multiple exons. To align genes based on amino acid sequence information, all exons must be extracted and concatenated prior to alignment. `plastburstalign` differentiates between cis- and trans-spliced genes:

* **Cis-spliced genes:** Exons are adjacent and concatenated directly in genomic order.
* **Trans-spliced genes (e.g., *rps12*):** Exons undergo a two-step repositioning procedure: contiguous overlapping exon groups are merged, and distant exon segments are combined at the position of the first exon occurrence.

#### Gene Name Normalization

* **tRNAs:** Homogenized to canonical `trnA_UAA` notation using translation table 11 anticodons. This prevents identical genes with differing nomenclature (`trnA-UAA`, `trnA_uaa`, `trnA-Leu-UAA`) from being treated as distinct partitions, decreasing artificial taxa dropout by ~25%.
* **rRNAs:** Normalized to canonical forms (`rrn16`, `rrn23`, `rrn4.5`, `rrn5`).
