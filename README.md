# PlastBurstAlign
Python tool for extracting and aligning genes, introns, and intergenic spacers across thousands of plastid genomes.

---

### Purpose
PlastBurstAlign supports large-scale quality assessment of organellar genome annotations. It detects discrepancies by comparing sequence and annotation features across homologous regions with automated multiple sequence alignments.

### Features
- Extracts and groups coding sequences, introns, intergenic spacers, and tRNAs/rRNAs
- Standardizes tRNA/rRNA names and deduplicates normalized regions per genome, detecting IR-mirrored spacers from their flanking genes.
- Filters user-selected regions and enforces minimum sequence-length and taxon-representation thresholds.
- Splices cis-exons in place, groups and repositions trans-spliced exons, and trims one or two trailing bases from CDS whose length is not divisible by three when they start with `ATG`; skips other such CDS without start/stop validation.
- Aligns each marker group with protein alignment and nucleotide back-translation for CDS, and nucleotide alignment for other modes; supports MAFFT, MUSCLE, Clustal Omega, and generic or custom commands.
- Concatenates successful alignments alphabetically or by genomic position in the first genome processed.
- Parallelizes extraction and marker alignment, logs progress, warnings, failures, and extraction issues, and provides additional diagnostics in verbose mode.

![Depiction of plastomes being split according to specified marker type; the extracted sequences are then aligned and concatenated](docs/PlastBurstAlign_ProcessOverview.png)

---

### Installation

See the [Installation page on the GitHub Wiki](https://github.com/michaelgruenstaeudl/PlastBurstAlign/wiki/Installation) for prerequisites and installation instructions.

---

### Command-line parameters

See the [Command-line parameters section on the GitHub Wiki](https://github.com/michaelgruenstaeudl/PlastBurstAlign/wiki/Command%E2%80%90line-parameters#command-line-parameters) for the available options and examples.

---

### Input and output

#### Input data

Place complete plastid genome GenBank flatfiles in an input directory. The
default file extension is `.gb`. For example:

```text
Input_dataset/
  genome_1.gb
  genome_2.gb
```

#### Output files

Use a separate output directory for each run; it and its subfolders are created
as needed. Unaligned nucleotide FASTAs (plus protein FASTAs and intermediate
protein alignments for CDS) are written to its root. Aligned nucleotide FASTAs
go in `fasta/`, with NEXUS files in `nexus/`. When concatenation is enabled,
the concatenated FASTA and NEXUS files are written to the output directory.

---

### Benchmarking

See the [Benchmarking section on the GitHub Wiki](https://github.com/michaelgruenstaeudl/PlastBurstAlign/wiki/Benchmarking#benchmarking-datasets) for benchmark datasets, usage, and preparation details.
