# PlastBurstAlign
A Python tool to extract and align genes, introns, and intergenic spacers across thousands of plastid genomes using associative arrays

---

### Purpose
This software tool is designed for large-scale quality assessment of organellar genome annotations. It detects annotation discrepancies in large genome datasets by comparing sequence and annotation features through automated multiple sequence alignments across homologous regions.

### Background
The multiple sequence alignment (MSA) of a set of plastid genomes is challenging. At least five factors contribute to this challenge:
- First, the plastid genome is a mosaic of individual genome regions. An MSA procedure must identify, extract, group, and align homologous regions across the input genomes.
- Second, many plastid genomes contain annotation errors in gene positions and/or gene names. An MSA procedure must automatically exclude incorrectly annotated regions from the alignment procedure.
- Third, plastid genomes comprise both coding and noncoding regions, which require different alignment strategies (e.g., amino acid-based for genes, nucleotide-based for introns and intergenic spacers). An MSA procedure must apply the appropriate strategy automatically.
- Fourth, modern plastid genome studies often involve hundreds, if not thousands, of complete genomes. An MSA procedure must perform sequence alignment within practical time frames (e.g., hours rather than days).
- Fifth, manually excluding user-specified genome regions after alignment is prohibitively complex. An MSA procedure must support the automatic exclusion of user-specified regions before the alignment starts.

The software `plastburstalign` addresses these and other challenges: it provides an MSA procedure that extracts and aligns genes, introns, and intergenic spacers across hundreds or thousands of plastid genomes in an autonomous fashion.

### Overview of process
![Depiction of plastomes being split according to specified marker type; the extracted sequences are then aligned and concatenated](docs/PlastBurstAlign_ProcessOverview.png)

### Main features
- Extraction of all genome regions from set of input plastid genomes, followed by grouping and alignment of the extracted regions:
  - genes (cds)
  - introns (int)
  - intergenic spacers (igs)
- Support for multiple alignment tools:
  - MAFFT
  - MUSCLE
  - Clustal Omega
- Automatic exon splicing:
  - automatic merging of all exons of any cis-spliced gene
  - automatic grouping of all exons of any trans-spliced gene (e.g., _rps12_), followed by merging of adjacent exons [see `ExonSpliceHandler` for both]
- Automatic quality control to evaluate if extracted genes are complete (i.e., valid start and stop codon present)
- Automatic removal of any duplicate regions (i.e., relevant for regions duplicated through the IRs)
- Removal of user-specified regions:
  - regions that do not fulfill a minimum, user-specified sequence length
  - regions that do not fulfill a minimum, user-specified number of taxa of the dataset that the region must be found in [see `DataCleaning` for both]
  - any user-specified genome region (i.e., gene, intron, or intergenic spacer)
-  Automatic determination if DNA sequence alignment based on amino acid (for genes) or nucleotide (for introns and intergenic spacers) sequence information 
- Parallelized processing for faster extraction and alignment across multiple CPUs

### Additional features
- Concatenation of all genome regions alignment either in alphabetic order or based on location in genome (first input genome used as reference)
- Automatic standardization of tRNA gene names to accommodate letter case differences among the gene annotations of different input genomes (e.g., for anticodon and amino acid abbreviations of tRNAs) [see `clean_gene()`]
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

---

### Parameters overview

| Option | Description | Example |
|--------|-------------|---------|
| `-i` | Input dataset directory | `Input_dataset` |
| `-o` | Output directory | `Output_dataset` |
| `-s` | Sequence type to extract (e.g., cds, int, igs) | `cds` |
| `-a` | Alignment tool to use | `mafft` |
| `-l` | Minimum sequence length (bp); regions shorter than this are excluded| `9` |
| `-t` | Minimum number of taxa (int) or relative frequency (0<value<=1) in which a region must be present to be extracted | `0.1` |
| `-n` | Number of threads to use| `8` |
| `--config` | Path to YAML config file containing parameters | `config.yaml` |

---

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

### Details on automatic standardization of tRNA gene names
The names of all tRNAs are automatically standardized across the input genomes to counteract the accumulation of idiosyncratic gene names. tRNAs are often labeled differently by different researchers. For example, researcher A may label tRNAs with both amino acid abbreviations and anticodons (e.g., `trnA-Leu-UAA`), whereas researcher B may label them with the respective anticodons only (e.g., `trnA-UAA`). Similarly, researcher C may label tRNA genes with lower-case anticodons (e.g., `trnA-uaa`) but researcher D with upper-case anticodons (e.g., `trnA-UAA`). Differences in tRNA gene names may also originate from the idiosyncratic use of dashes versus underscores (e.g., `trnA-UAA` versus `trnA_UAA`). Leaving the names of tRNAs that code for the same gene unadjusted and, thus, incongruent across different input genomes risks the artificial increase in the number of unique genes, introns, and intergenic spacers in the dataset.

To ensure that only homologous genes are grouped together and aligned, `plastburstalign` automatically standardizes tRNA gene names across input genomes. Specifically, the software homogenizes incongruent tRNAs gene names to a single format: _tRNAabbreviation_anticodon_  (e.g., `trnA_UAA`). This format is (i) the most commonly used tRNA naming scheme among plastid genomes and (ii) the least problematic scheme for nucleotide sequence alignment operations, which typically interpret dashes as sequence characters. During the standardization operations, `plastburstalign` utilizes the [three-letter amino acid abbreviations](https://www.insdc.org/submitting-standards/feature-table/#7.4.3) and the [anticodon definitions](https://www.insdc.org/submitting-standards/genetic-code-tables/) of [translation table 11](https://www.insdc.org/submitting-standards/genetic-code-tables/) of the International Nucleotide Sequence Database Collaboration (INSDC). tRNAs with more than one possible codon but for which neither amino acid nor anticodon abbreviations are given in the gene name (e.g., `trnA` can be any of the following: `trnA_UAA`, `trnA_CAA`, `trnA_AAG`, `trnA_GAG`, `trnA_UAG`, and `trnA_CAG`), by contrast, are not changed by `plastburstalign` to avoid the incorrect designations.

As a side effect, the automatic standardization of tRNA gene names also decreases the number of annotated genome regions that need to be removed from the dataset for not reaching the minimum number of taxa defined. Without the standardization, the intergenic spacer between the genes `trnA_CAA` and `ndhB`, for example, may be grouped under two different names (e.g., `trnA_CAA_ndhB` and `trnA_caa_ndhB`), with the latter group being less common and eventually removed from the dataset for not reaching the minimum number of taxa. By implementing a gene name standardization, the same intergenic spacer is grouped under only one name (i.e., `trnA_CAA_ndhB`) and not discarded. Preliminary tests indicated that the number of annotated genome regions that were removed due to not reaching the minimum number of taxa was decreased by approximately 25% through the tRNA gene name standardization.


### Exemplary usage
See [this document](https://github.com/michaelgruenstaeudl/PlastBurstAlign/blob/main/docs/exemplary_usage.md)
