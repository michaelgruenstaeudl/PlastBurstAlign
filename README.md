# PlastBurstAlign
A Python tool to extract and align genes, introns, and intergenic spacers across thousands of plastid genomes using associative arrays

---

### Purpose
This software tool is designed for large-scale quality assessment of organellar genome annotations. It detects annotation discrepancies in large genome datasets by comparing sequence and annotation features through automated multiple sequence alignments across homologous regions.

### Overview of process
![Depiction of plastomes being split according to specified marker type; the extracted sequences are then aligned and concatenated](docs/PlastBurstAlign_ProcessOverview.png)

### Main features
- Extraction of all genome regions from set of input plastid genomes, followed by grouping and alignment of the extracted regions:
  - genes (cds)
  - introns (int)
  - intergenic spacers (igs)
  - tRNAs and rRNAs (rna; nucleotide-only alignment)
- Support for multiple alignment tools:
  - MAFFT
  - MUSCLE
  - Clustal Omega
- Automatic exon splicing:
  - automatic merging of all exons of any cis-spliced gene
  - automatic grouping of all exons of any trans-spliced gene (e.g., _rps12_), followed by merging of adjacent exons [see `ExonSpliceHandler` for both]
- Automatic quality control to evaluate if extracted genes are complete (i.e., valid start and stop codon present)
- Duplicate regions are stored by normalized region name, with at most one copy of each named region retained per genome; for intergenic spacers, the software also checks the reverse-flanking-gene name so that IR-mirrored spacers are not counted twice.
- Removal of user-specified regions:
  - regions that do not fulfill a minimum, user-specified sequence length
  - regions that do not fulfill a minimum, user-specified number of taxa of the dataset that the region must be found in [see `DataCleaning` for both]
  - any user-specified genome region (i.e., gene, intron, or intergenic spacer)
-  Automatic determination if DNA sequence alignment based on amino acid (for CDS) or nucleotide (for introns, intergenic spacers, tRNAs, and rRNAs) sequence information
- Parallelized processing for faster extraction and alignment across multiple CPUs

### Additional features
- Concatenation of all genome regions alignment either in alphabetic order or based on location in genome (first input genome used as reference)
- Automatic standardization of tRNA gene names to accommodate letter case differences among the gene annotations of different input genomes (e.g., for anticodon and amino acid abbreviations of tRNAs)
- Automatic standardization of plastid rRNA names to `rrn16`, `rrn23`, `rrn4.5`, or `rrn5`
- Flexible configuration of alignment tools via user-defined parameters
- Production of informative logs; two detail levels:
  - default (suitable for regular software execution)
  - verbose (suitable for debugging)
- Clear reporting when genome regions cannot be extracted

### Input/output
#### Input
- Set of complete plastid genomes (each in GenBank flatfile format)
#### Output
- DNA sequence alignments of individual genome regions (FASTA format)
- Concatenation of all individual DNA sequence alignments (FASTA and NEXUS format)

### Installation on Linux (Debian)
```bash
# Installation
#pip install git+https://github.com/michaelgruenstaeudl/PlastBurstAlign.git  
#pip3 install git+ssh://git@github.com/michaelgruenstaeudl/PlastBurstAlign.git

pip install plastburstalign
```
You must manually install one of the supported alignment tools (MAFFT, MUSCLE, Clustal Omega). If the executable is NOT installed in PATH, provide tool path in input.

### Installation of External alignment tools
### Option 1 — Conda environment (recommended)
```bash
git clone https://github.com/michaelgruenstaeudl/PlastBurstAlign.git
cd PlastBurstAlign
conda env create -f environment.yml
conda activate plastburstalign
```
This environment installs Python dependencies and external alignment tools (MAFFT, MUSCLE, Clustal Omega).
### Option 2 — Manual installation
```bash
sudo apt install mafft muscle clustalo
```
### Usage

### Command-line (recommended)
After installation via `pip`, run:

```bash
plastburstalign
```
### Example run

```bash
plastburstalign \
  -i Input_dataset \
  -o Output_dataset \
  -s cds \
  -a mafft
```

### Detailed examples
See [this document](https://github.com/michaelgruenstaeudl/PlastBurstAlign/blob/main/docs/exemplary_usage.md)

### Benchmarking
```bash
cd benchmarking
# CDS
python benchmarking_CDS.py benchmark_YangEtAl2022_Asteraceae 5
# INT
python benchmarking_INT.py benchmark_YangEtAl2022_Asteraceae 5
# IGS
python benchmarking_IGS.py benchmark_YangEtAl2022_Asteraceae 5
```
- Dataset `benchmark_ZhangNNEtAl2025_53angiosperms`: the same dataset (angiosperms; n=53) as used in [Zhang N.-N. et al. 2025](https://doi.org/10.1016/j.pld.2025.05.005)
- Dataset `benchmark_YangEtAl2022_Asteraceae.tar.gz`: all Asteraceae (n=155) listed in [Yang et al. 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.808156)
- Dataset `benchmark_YangEtAl2022_monocots`: all monocots (n=733) listed in [Yang et al. 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.808156)
- Dataset `benchmark_YangEtAl2022_angiosperms`: all angiosperms (n=2585) listed in [Yang et al. 2022](https://www.frontiersin.org/journals/plant-science/articles/10.3389/fpls.2022.808156)
- Dataset `benchmark_ZhangNNEtAl2025_13724angiosperms`: the same dataset (angiosperms; n=13724 instead of n=13731) as used in [Zhang N.-N. et al. 2025](https://doi.org/10.1016/j.pld.2025.05.005) but not including records AC_000188, NC_029253, NC_029254, NC_029331, NC_029333, NC_071962, and NC_085618, which have been removed by NCBI from NCBI RefSeq as a result of standard genome annotation processing.

---

### Parameters overview

| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Input dataset directory | `-i Input_dataset` |
| `-o` | Output directory | `-o Output_dataset` |
| `-s` | Sequence type to extract (`cds`, `int`, `igs`, or `rna`) | `-s cds` |
| `-a` | Alignment tool to use | `-a mafft` |
| `-l` | Minimum sequence length (bp); regions shorter than this are excluded| `-l 9` |
| `-t` | Minimum number of taxa (int) or relative frequency (0<value<=1) in which a region must be present to be extracted | `-t 0.1` |
| `-n` | Number of threads to use| `-n 8` |
| `--trna` | Include tRNAs in `rna` mode (disabled unless specified) | `--trna` |
| `--rrna` | Include rRNAs in `rna` mode (disabled unless specified) | `--rrna` |
| `--config` | Path to YAML config file containing parameters | `--config config.yaml` |

---

### Extracting tRNAs and rRNAs

Use `-s rna` (or `select_mode: "rna"` in a YAML configuration) to extract annotated
`tRNA` and `rRNA` features and align them directly as DNA nucleotide sequences:

```bash
plastburstalign -i Input_dataset -o Output_RNA -s rna --trna --rrna -a mafft
```

RNA types are disabled by default in the command line. Specify both switches
to include both types, or just one switch to extract only that type:

```bash
# tRNAs only
plastburstalign -i Input_dataset -o Output_tRNA -s rna --trna -a mafft
# rRNAs only
plastburstalign -i Input_dataset -o Output_rRNA -s rna --rrna -a mafft
```

The corresponding Python API parameters are `include_trna` and `include_rrna` (both default to `false`). In CLI runs, the `--trna` and `--rrna` switches override YAML values for these parameters; because both switches default to disabled, pass the desired switch or switches when using RNA mode with a YAML configuration. If neither type is enabled, `rna` mode reports an error. These switches do not affect `cds`, `int`, or `igs` mode.

RNA mode does not translate sequences, require start/stop codons, or trim sequences to a multiple of three. It extracts all annotated parts in strand-correct order; for joined features, the exons are concatenated and intervening introns are omitted. Features must have a `gene` qualifier, as in the other modes. Gene-name  normalization (including tRNA names), removal of duplicate copies per genome, minimum sequence length and taxon thresholds, ordering, and optional concatenation all apply. Both `--exclcds` (gene exclusions) and `--exclreg` (region exclusions) can exclude RNA genes by their normalized names. Output uses the existing `nucl_` FASTA and NEXUS formats, with no protein files. The `cds`, `int`, and `igs` modes are unchanged.

### Source usage
If you cloned the repository:

```bash
git clone https://github.com/michaelgruenstaeudl/PlastBurstAlign.git
cd PlastBurstAlign
python -m plastburstalign
```

---

### Python API

You can also use the package directly in Python:

```python
from plastburstalign import PlastBurstAlign

burst = PlastBurstAlign()
burst.execute()
```


#### Usage of individual package components
Individual components can be used as well. For example, to use the class `MAFFT` by itself (e.g., instantiate a configuration of MAFFT that will execute with 1 thread; institute another that will execute with 10 threads), type:

```python
from plastburstalign import MAFFT, MUSCLE, ClustalOmega

mafft = MAFFT({"num_threads": 4})
muscle = MUSCLE()
clustal = ClustalOmega()
```

### Details on exon splicing
The plastid genome is a mosaic of individual genome regions, with many of its genes consisting of multiple exons. To align genes based on their amino acid sequence information, all exons of a gene must be extracted and concatenated prior to alignment. `plastburstalign` conducts this exon splicing through an automated process that differentiates between cis- or trans-spliced genes: the exons of cis-spliced genes are adjacent to each other, those of trans-spliced genes are not. The software concatenates the exons of any cis-spliced gene in place (i.e., no repositioning of the exons necessary). The exons of any trans-spliced gene (e.g., _rps12_), by contrast, undergo a two-step repositioning procedure before being concatenated. First, groups of contiguous exons are formed based on their location information: if an exon is adjacent to or even overlaps with another exon of the same gene name, they are merged. Second, exons of the same gene name are merged at the location of the first exon occurrence.

### Details on removal of user-specified genome regions from alignment
Due to the size and complexity of large DNA sequence alignments, individual genome regions can barely be removed from a concatenated sequence alignment; instead, any user-specified exclusion of a genome region must be performed before the actual sequence alignment. `plastburstalign` contains two functions for such an exclusion: commandline-parameter `exclude_region` excludes any user-specified region by exact name match from the dataset; commandline-parameter `exclude_fullcds` removes entire user-specified genes as well as any introns inside, and any intergenic spacers immediately adjacent to, the specified genes from the dataset.

### Details on automatic standardization of tRNA and rRNA gene names
tRNA gene annotations vary in format: they may include both an amino-acid abbreviation and an anticodon (e.g., `trnA-Leu-UAA`), only an anticodon (e.g., `trnA-UAA`), lowercase rather than uppercase anticodons (e.g., `trnA-uaa` versus `trnA-UAA`), or hyphens rather than underscores (e.g., `trnA-UAA` versus `trnA_UAA`).

To ensure that only homologous genes are grouped together and aligned, `plastburstalign` automatically standardizes tRNA gene names across input genomes. Specifically, the software homogenizes incongruent tRNAs gene names to a single format: _tRNAabbreviation_anticodon_  (e.g., `trnA_UAA`). This format is (i) the most commonly used tRNA naming scheme among plastid genomes and (ii) the least problematic scheme for nucleotide sequence alignment operations, which typically interpret dashes as sequence characters. During the standardization operations, `plastburstalign` utilizes the [three-letter amino acid abbreviations](https://www.insdc.org/submitting-standards/feature-table/#7.4.3) and the [anticodon definitions](https://www.insdc.org/submitting-standards/genetic-code-tables/) of [translation table 11](https://www.insdc.org/submitting-standards/genetic-code-tables/) of the International Nucleotide Sequence Database Collaboration (INSDC). tRNAs with more than one possible codon but for which neither amino acid nor anticodon abbreviations are given in the gene name (e.g., `trnA` can be any of the following: `trnA_UAA`, `trnA_CAA`, `trnA_AAG`, `trnA_GAG`, `trnA_UAG`, and `trnA_CAG`), by contrast, are not changed by `plastburstalign` to avoid the incorrect designations.

For rRNAs, recognized gene names and RNA-product labels are standardized to the canonical plastid gene names: 16S rRNA to `rrn16`, 23S rRNA to `rrn23`, 4.5S rRNA to `rrn4.5`, and 5S rRNA to `rrn5`.

---

### High-Throughput Architecture & Additional Options

- **Scalable execution & determinism:** SQLite WAL feature store (`DiskBackedFeatureStore`), CPU-set core leases, LPT `LocusScheduler`, optional sub-locus chunking (`--subdivide auto|off|force`), deterministic `--seed`, dense `uint8` site filters (`--trim`), streaming supermatrix builder (`--phylip`), and RAxML-NG / IQ-TREE 2 / NEXUS partition exports (`--partitions`, `--codon-partitions`).
- **Host profiling & Slurm:** Automatic core/memory/scratch tuning (`-n auto`), preflight diagnostics (`--doctor`, `--estimate`), and Slurm array workflows (`--export-slurm`, `--align-partition`, `--finalize-slurm`). See [docs/calibration.md](docs/calibration.md) and [docs/v3_status.md](docs/v3_status.md).
- **High-level Python API:** `plastburstalign.align(...)`, `plastburstalign.autotune(...)`, and `plastburstalign.estimate(...)` emit structured `AlignmentResult` objects and `run_manifest.json`.
