# PlastBurstAlign
PlastBurstAlign is an open-source, modular Python command-line tool for extracting and aligning all coding and noncoding genome regions across thousands of plastid genomes.

[![PyPI version](https://img.shields.io/pypi/v/plastburstalign)](https://pypi.org/project/plastburstalign/)
[![Python versions](https://img.shields.io/pypi/pyversions/plastburstalign)](https://pypi.org/project/plastburstalign/)
[![License](https://img.shields.io/github/license/michaelgruenstaeudl/PlastBurstAlign)](LICENSE)
[![GitHub stars](https://img.shields.io/github/stars/michaelgruenstaeudl/PlastBurstAlign)](https://github.com/michaelgruenstaeudl/PlastBurstAlign)

---

### Overview
PlastBurstAlign deconstructs complete plastomes into protein-coding sequences, tRNAs, rRNAs, introns, and intergenic spacers. It automatically standardizes gene names, reconstructs exons, filters features, handles duplicated regions, and groups homologous sequences across genomes by gene or marker name. Coding sequences are aligned via amino acid translations and noncoding markers as nucleotides using user-selected alignment software. Most importantly, PlastBurstAlign addresses the runtime and memory demands of large plastid phylogenomic datasets through task parallelization that accounts for marker alignment complexity and explicit memory management.

---

### Features
- Extracts coding sequences, introns, tRNAs/rRNAs, or intergenic spacers in selectable modes and groups sequences by gene or marker name.
- Standardizes tRNA/rRNA names and deduplicates regions by name per genome, including inverted repeat copies and reverse-named intergenic spacers.
- Filters user-excluded markers and enforces minimum sequence length and taxon representation thresholds.
- Reconstructs cis- and trans-spliced coding sequences.
- Confirms presence of start and stop codons among coding sequences and excludes sequences with invalid start/stop codons or translation errors.
- Aligns coding sequences via translated amino acids and nucleotide back-translation, whereas RNA, intron, and spacer sequences are aligned as nucleotides.
- Supports MAFFT, MUSCLE, and Clustal Omega out of the box; additional command-line aligners can be configured via YAML.
- Writes individual and optional concatenated alignments in FASTA/NEXUS, ordering concatenated markers alphabetically or by feature order in the first genome.
- ⭐ Runs marker alignments concurrently and allocates more threads to higher-cost alignments when supported, reducing runtime.
- ⭐ Implements disk-backed sequence storage and a bounded task queue to limit memory use.
- Logs progress, warnings, and failures, with extra verbose diagnostics.

---

### Installation

See the [Installation page on the GitHub Wiki](https://github.com/michaelgruenstaeudl/PlastBurstAlign/wiki/Installation) for prerequisites and installation instructions.

---

### Quick start

After installing PlastBurstAlign and MAFFT, place complete GenBank `.gb` files in an input directory and run:

```bash
plastburstalign -i Input_dataset -o Output_cds -s cds -c
```

This extracts and aligns coding sequences, then writes concatenated FASTA and NEXUS files to the output directory.

---

### Input and output

#### Input data

Place complete plastid genome GenBank flatfiles in an input directory. The default file extension is `.gb`. For example:

```text
Input_dataset/
  genome_1.gb
  genome_2.gb
```

#### Output files

Use a separate output directory for each run; it and its subfolders are created automatically as needed. Unaligned nucleotide FASTAs (plus protein FASTAs and intermediate protein alignments for CDS) are written to its root. Aligned nucleotide FASTAs go in `fasta/`, with NEXUS files in `nexus/`. When concatenation is enabled, the concatenated FASTA and NEXUS files are additionally written to the output directory.

---

### Command-line parameters

See the [Command-line parameters section on the GitHub Wiki](https://github.com/michaelgruenstaeudl/PlastBurstAlign/wiki/Command%E2%80%90line-parameters#command-line-parameters) for the available options and examples.