"""Deterministic synthetic plastome GenBank generator (test fixture, not collected).

Produces valid, realistic GenBank flatfiles that the project's own scanner
(:func:`plastburstalign.genbank_scanner.fast_read_genbank`) and
:func:`Bio.SeqIO.read` both parse, and that the extractor can use in all four
modes (``cds``, ``igs``, ``int``, ``rna``).

Public API
----------
``make_template(seed, n_genes=24)``
    The ancestral genome: gene catalogue subset, layout and root sequences.
``make_plastome(rng_seed, taxon_id, template, divergence=0.01, indel_rate=0.002)``
    One GenBank flatfile (``str``) = the template mutated by one branch.
``write_genbank_dataset(out_dir, n_taxa=8, n_genes=24, seed=0, ...)``
    ``n_taxa`` files related by a random tree, plus a :class:`DatasetManifest`
    with the taxa, files and expected per-mode region names / record counts.

Genome model
------------
``LSC | IRb | SSC | IRa`` where ``IRa`` is the reverse complement of ``IRb``
(so every IR gene is annotated twice, on opposite strands, exercising the
extractor's de-duplication). Layout (gene order, strands) follows real plastomes;
sizes are compressed (a 24-gene genome is about 15 kb) so that thousands of
alignments stay fast. Features:

* protein-coding genes: ``ATG`` start, in frame, stop codon, ``/transl_table=11``,
  ``/codon_start``, ``/product``, ``/translation``; both strands;
* cis-spliced genes with 1 or 2 introns (``join(a..b,c..d)`` /
  ``complement(join(...))``), tRNAs with introns (``trnK-UUU``, ``trnL-UAA`` ...);
* a trans-spliced ``rps12``: 5' exon in the LSC (minus strand) and exons 2+3
  (with a cis intron) in the IR, annotated as a 3-part mixed-strand ``join``;
* tRNA and rRNA genes with idiosyncratic spellings (``trnA-UGC``, ``trnA_ugc``,
  ``trnA(UGC)``, ``16S rRNA`` ...) that all normalise to one region name;
* intergenic spacers with indels, wrapped ``ORIGIN`` (60 per line, 10-char blocks,
  lowercase), ``LOCUS``/``DEFINITION``/``ACCESSION``/``VERSION`` headers.

Variation along the tree: substitutions (codon-preserving in CDS: the reading
frame, start and stop codons stay intact, no premature stops) plus short indels
in spacers and introns. A child taxon is the state of a random earlier taxon
mutated along one branch.

Speed: see :func:`write_genbank_dataset` (about 400 genomes/s for 24 genes on a
laptop core, a few hundred genomes in about a second).
"""

from __future__ import annotations

import functools
import random
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from Bio.Data import CodonTable
from Bio.SeqUtils import seq3

MODES = ("cds", "igs", "int", "rna")

_COMP = str.maketrans("ACGTacgt", "TGCAtgca")
_BASES = "ACGT"
_STOPS = ("TAA", "TAG", "TGA")
_SENSE_CODONS = [
    a + b + c for a in _BASES for b in _BASES for c in _BASES if a + b + c not in _STOPS
]
_FORWARD_TABLE: dict[str, str] = dict(
    CodonTable.unambiguous_dna_by_id[11].forward_table
)
# AT-rich plastid-like base composition
_BASE_WEIGHTS = (0.31, 0.19, 0.19, 0.31)


def revcomp(seq: str) -> str:
    """Reverse complement of an ``ACGT`` string."""
    return seq.translate(_COMP)[::-1]


@functools.lru_cache(maxsize=1 << 16)
def translate_table11(cds: str) -> str:
    """Translation (table 11) of an in-frame coding sequence, stop shown as ``*``."""
    return "".join(
        _FORWARD_TABLE.get(cds[i : i + 3], "*") for i in range(0, len(cds) - 2, 3)
    )


# ---------------------------------------------------------------------------
# Gene catalogue
# ---------------------------------------------------------------------------

# name, kind, segment, strand, number of introns. Order = genomic order
# (IR = IRb orientation, listed from the LSC side). Names are the *canonical*
# region names the extractor produces (tRNAs are ``trnX_ANT`` with an RNA anticodon).
_CATALOG: tuple[tuple[str, str, str, int, int], ...] = (
    ("trnH_GUG", "trna", "LSC", 1, 0),
    ("psbA", "cds", "LSC", -1, 0),
    ("trnK_UUU", "trna", "LSC", -1, 1),
    ("matK", "cds", "LSC", -1, 0),
    ("rps16", "cds", "LSC", -1, 1),
    ("trnQ_UUG", "trna", "LSC", -1, 0),
    ("psbK", "cds", "LSC", 1, 0),
    ("psbI", "cds", "LSC", 1, 0),
    ("trnS_GCU", "trna", "LSC", 1, 0),
    ("trnG_UCC", "trna", "LSC", 1, 1),
    ("trnR_UCU", "trna", "LSC", -1, 0),
    ("atpA", "cds", "LSC", -1, 0),
    ("atpF", "cds", "LSC", -1, 1),
    ("atpH", "cds", "LSC", -1, 0),
    ("atpI", "cds", "LSC", -1, 0),
    ("rps2", "cds", "LSC", -1, 0),
    ("rpoC2", "cds", "LSC", -1, 0),
    ("rpoC1", "cds", "LSC", -1, 1),
    ("rpoB", "cds", "LSC", -1, 0),
    ("trnC_GCA", "trna", "LSC", 1, 0),
    ("petN", "cds", "LSC", 1, 0),
    ("psbM", "cds", "LSC", -1, 0),
    ("trnD_GUC", "trna", "LSC", -1, 0),
    ("trnY_GUA", "trna", "LSC", -1, 0),
    ("trnE_UUC", "trna", "LSC", -1, 0),
    ("trnT_GGU", "trna", "LSC", 1, 0),
    ("psbD", "cds", "LSC", -1, 0),
    ("psbC", "cds", "LSC", -1, 0),
    ("trnS_UGA", "trna", "LSC", 1, 0),
    ("psbZ", "cds", "LSC", 1, 0),
    ("trnG_GCC", "trna", "LSC", 1, 0),
    ("rps14", "cds", "LSC", -1, 0),
    ("psaB", "cds", "LSC", -1, 0),
    ("psaA", "cds", "LSC", -1, 0),
    ("ycf3", "cds", "LSC", -1, 2),
    ("trnS_GGA", "trna", "LSC", 1, 0),
    ("rps4", "cds", "LSC", -1, 0),
    ("trnT_UGU", "trna", "LSC", -1, 0),
    ("trnL_UAA", "trna", "LSC", -1, 1),
    ("trnF_GAA", "trna", "LSC", -1, 0),
    ("ndhJ", "cds", "LSC", -1, 0),
    ("ndhK", "cds", "LSC", -1, 0),
    ("ndhC", "cds", "LSC", -1, 0),
    ("trnV_UAC", "trna", "LSC", -1, 1),
    ("trnM_CAU", "trna", "LSC", 1, 0),
    ("atpE", "cds", "LSC", -1, 0),
    ("atpB", "cds", "LSC", -1, 0),
    ("rbcL", "cds", "LSC", 1, 0),
    ("accD", "cds", "LSC", 1, 0),
    ("psaI", "cds", "LSC", 1, 0),
    ("ycf4", "cds", "LSC", 1, 0),
    ("cemA", "cds", "LSC", 1, 0),
    ("petA", "cds", "LSC", 1, 0),
    ("psbJ", "cds", "LSC", -1, 0),
    ("psbL", "cds", "LSC", -1, 0),
    ("psbF", "cds", "LSC", -1, 0),
    ("psbE", "cds", "LSC", -1, 0),
    ("petL", "cds", "LSC", 1, 0),
    ("petG", "cds", "LSC", 1, 0),
    ("trnW_CCA", "trna", "LSC", 1, 0),
    ("trnP_UGG", "trna", "LSC", 1, 0),
    ("psaJ", "cds", "LSC", 1, 0),
    ("rpl33", "cds", "LSC", 1, 0),
    ("rps18", "cds", "LSC", 1, 0),
    ("rpl20", "cds", "LSC", -1, 0),
    ("clpP", "cds", "LSC", -1, 2),
    ("psbB", "cds", "LSC", 1, 0),
    ("psbT", "cds", "LSC", 1, 0),
    ("psbN", "cds", "LSC", -1, 0),
    ("psbH", "cds", "LSC", 1, 0),
    ("petB", "cds", "LSC", 1, 1),
    ("petD", "cds", "LSC", 1, 1),
    ("rpoA", "cds", "LSC", -1, 0),
    ("rps11", "cds", "LSC", -1, 0),
    ("rpl36", "cds", "LSC", -1, 0),
    ("infA", "cds", "LSC", -1, 0),
    ("rps8", "cds", "LSC", -1, 0),
    ("rpl14", "cds", "LSC", -1, 0),
    ("rpl16", "cds", "LSC", -1, 1),
    ("rps3", "cds", "LSC", -1, 0),
    ("rpl22", "cds", "LSC", -1, 0),
    ("rps19", "cds", "LSC", -1, 0),
    # rps12 contributes two elements: 5' exon in the LSC (see ``_TRANS_GENE``)
    ("rpl2", "cds", "IR", -1, 1),
    ("rpl23", "cds", "IR", -1, 0),
    ("trnI_CAU", "trna", "IR", -1, 0),
    ("ycf2", "cds", "IR", -1, 0),
    ("trnL_CAA", "trna", "IR", 1, 0),
    ("ndhB", "cds", "IR", -1, 1),
    ("rps7", "cds", "IR", -1, 0),
    ("rps12", "cds", "IR", 1, 1),
    ("trnV_GAC", "trna", "IR", 1, 0),
    ("rrn16", "rrna", "IR", 1, 0),
    ("trnI_GAU", "trna", "IR", 1, 1),
    ("trnA_UGC", "trna", "IR", 1, 1),
    ("rrn23", "rrna", "IR", 1, 0),
    ("rrn4.5", "rrna", "IR", 1, 0),
    ("rrn5", "rrna", "IR", 1, 0),
    ("trnR_ACG", "trna", "IR", 1, 0),
    ("trnN_GUU", "trna", "IR", -1, 0),
    ("ycf1", "cds", "SSC", -1, 0),
    ("ndhF", "cds", "SSC", -1, 0),
    ("rpl32", "cds", "SSC", -1, 0),
    ("trnL_UAG", "trna", "SSC", -1, 0),
    ("ccsA", "cds", "SSC", -1, 0),
    ("ndhD", "cds", "SSC", 1, 0),
    ("psaC", "cds", "SSC", 1, 0),
    ("ndhE", "cds", "SSC", 1, 0),
    ("ndhG", "cds", "SSC", 1, 0),
    ("ndhI", "cds", "SSC", 1, 0),
    ("ndhA", "cds", "SSC", -1, 1),
    ("ndhH", "cds", "SSC", -1, 0),
    ("rps15", "cds", "SSC", -1, 0),
)

# Genes that sit inside the first intron of another gene (as ``matK`` in ``trnK``).
_NESTED = {"matK": "trnK_UUU"}

# The trans-spliced gene: exon 1 closes the LSC.
_TRANS_GENE = "rps12"
_TRANS_EXON_LENGTHS = (114, 26, 232)  # 372 nt = 124 codons

# Which genes a small ``n_genes`` keeps, in priority order. The first five cover
# LSC, IR (rRNA), SSC, a trans-spliced gene and the strands.
_PRIORITY = (
    "psbA",
    "rbcL",
    "rps12",
    "rrn16",
    "ndhF",
    "rps16",
    "trnK_UUU",
    "atpF",
    "ndhB",
    "trnL_UAA",
    "clpP",
    "trnA_UGC",
    "rrn23",
    "ycf3",
    "ndhA",
    "rpl2",
    "trnV_GAC",
    "petB",
    "matK",
    "rpoC1",
    "atpB",
    "psbD",
    "rps4",
    "ycf1",
    "trnH_GUG",
    "trnI_GAU",
    "rps7",
    "trnN_GUU",
    "psaA",
    "petD",
    "rpl16",
    "ndhH",
)

# canonical -> spellings found in real files that the extractor's gene-name
# cleaning maps back onto the canonical name (checked in ``tests/test_synth.py``).
_RRNA_SPELLINGS = {
    "rrn16": ("rrn16", "rrn16", "rrn16S", "16S rRNA"),
    "rrn23": ("rrn23", "rrn23", "rrn23S", "23S rRNA"),
    "rrn4.5": ("rrn4.5", "rrn4.5", "rrn4.5S", "4.5S rRNA"),
    "rrn5": ("rrn5", "rrn5", "rrn5S", "5S rRNA"),
}


def spellings_for(name: str, kind: str) -> tuple[str, ...]:
    """``/gene`` spellings that all clean to the canonical region ``name``.

    Weighted towards the canonical spelling (first entries repeat).
    """
    if kind == "trna":
        base, anticodon = name.split("_")
        return (
            f"{base}-{anticodon}",
            f"{base}-{anticodon}",
            f"{base}-{anticodon}",
            f"{base}_{anticodon.lower()}",
            f"{base}-{anticodon.lower()}",
            f"{base}({anticodon})",
        )
    if kind == "rrna":
        return _RRNA_SPELLINGS[name]
    return (name, name, name, name.lower(), name.upper())


@dataclass(frozen=True)
class GeneSpec:
    """Static description of one gene of the template."""

    name: str  # canonical region name
    kind: str  # "cds" | "trna" | "rrna"
    segment: str  # "LSC" | "IR" | "SSC"
    strand: int  # orientation in the LSC / IRb / SSC coordinates
    n_introns: int
    trans_spliced: bool = False
    spellings: tuple[str, ...] = ()
    nested_in: str | None = None  # host gene whose first intron contains this gene

    @property
    def n_exons(self) -> int:
        return self.n_introns + 1

    @property
    def product(self) -> str:
        if self.kind == "trna":
            return f"tRNA-{seq3(self.name[3])}"
        if self.kind == "rrna":
            return {
                "rrn16": "16S ribosomal RNA",
                "rrn23": "23S ribosomal RNA",
                "rrn4.5": "4.5S ribosomal RNA",
                "rrn5": "5S ribosomal RNA",
            }[self.name]
        return f"{self.name} protein"


@dataclass(frozen=True)
class Element:
    """One slot of the layout: a whole gene, or one half of the trans-spliced gene."""

    gene: str
    segment: str  # "LSC" | "IR" | "SSC"
    part: str = "all"  # "all" | "exon1" (LSC piece of rps12) | "exon23" (IR piece)


@dataclass
class GeneSeq:
    """Sequences of a gene in gene (5'->3') orientation."""

    exons: list[str]
    introns: list[str]


@dataclass
class GenomeState:
    genes: dict[str, GeneSeq]
    spacers: list[str]  # spacers[i] precedes layout element i; two more at the end


@dataclass
class Template:
    """Ancestral genome shared by every taxon of a dataset."""

    seed: int
    specs: dict[str, GeneSpec]
    layout: list[Element]
    root: GenomeState
    nested: dict[str, list[str]] = field(default_factory=dict)  # host -> nested genes

    @property
    def gene_names(self) -> list[str]:
        """Canonical names in layout order (IR genes once, nested after their host)."""
        seen: dict[str, None] = {}
        for el in self.layout:
            seen.setdefault(el.gene, None)
            for inner in self.nested.get(el.gene, ()):
                seen.setdefault(inner, None)
        return list(seen)


def _rand_seq(rng: random.Random, n: int) -> str:
    return "".join(rng.choices(_BASES, _BASE_WEIGHTS, k=n))


def _rand_cds(rng: random.Random, n_codons: int) -> str:
    body = "".join(rng.choices(_SENSE_CODONS, k=n_codons - 2))
    return "ATG" + body + rng.choice(_STOPS)


def _split(seq: str, lengths: Sequence[int]) -> list[str]:
    out, pos = [], 0
    for n in lengths:
        out.append(seq[pos : pos + n])
        pos += n
    return out


def _make_gene_seq(rng: random.Random, spec: GeneSpec, rrna_scale: float) -> GeneSeq:
    if spec.trans_spliced:
        exon_lengths = list(_TRANS_EXON_LENGTHS)
        coding = _rand_cds(rng, sum(exon_lengths) // 3)
        introns = [_rand_seq(rng, rng.randint(150, 260))]
        return GeneSeq(_split(coding, exon_lengths), introns)
    if spec.kind == "cds":
        n_codons = rng.randint(45, 190)
        coding = _rand_cds(rng, n_codons)
        # exon boundaries need not be codon aligned
        cuts = sorted(rng.sample(range(24, len(coding) - 24), spec.n_introns))
        lengths = [b - a for a, b in zip([0, *cuts], [*cuts, len(coding)])]
        introns = [_rand_seq(rng, rng.randint(90, 320)) for _ in range(spec.n_introns)]
        return GeneSeq(_split(coding, lengths), introns)
    if spec.kind == "trna":
        n = rng.randint(70, 88)
        if spec.n_introns:
            exon_lengths = [n // 2, n - n // 2]
            introns = [_rand_seq(rng, rng.randint(80, 220))]
        else:
            exon_lengths, introns = [n], []
        return GeneSeq(_split(_rand_seq(rng, n), exon_lengths), introns)
    base = {"rrn16": 520, "rrn23": 700, "rrn4.5": 100, "rrn5": 120}[spec.name]
    n = max(40, int(base * rrna_scale))
    return GeneSeq([_rand_seq(rng, n)], [])


def make_template(
    seed: int = 0, n_genes: int = 24, rrna_scale: float = 1.0
) -> Template:
    """Build the ancestral genome.

    Args:
        seed: Seed of the template (layout is a function of ``n_genes`` only;
            sequences also depend on ``seed``).
        n_genes: Number of genes (``rps12`` counts once, although it contributes
            two layout elements). Beyond the catalogue size synthetic ``synNNN``
            filler genes are appended to the LSC and SSC.
        rrna_scale: Multiplier for the (already compressed) rRNA lengths.
    """
    if n_genes < 5:
        raise ValueError("n_genes must be at least 5")
    rng = random.Random(f"template:{seed}")
    chosen: list[str] = list(_PRIORITY[:n_genes])
    if n_genes > len(_PRIORITY):
        for c in _CATALOG:
            if len(chosen) >= n_genes:
                break
            if c[0] not in chosen:
                chosen.append(c[0])
    n_fillers = max(0, n_genes - len(chosen))
    chosen_set = set(chosen)

    specs: dict[str, GeneSpec] = {}
    layout: list[Element] = []
    for name, kind, segment, strand, n_introns in _CATALOG:
        if name not in chosen_set:
            continue
        trans = name == _TRANS_GENE
        host = _NESTED.get(name)
        specs[name] = GeneSpec(
            name,
            kind,
            segment,
            strand,
            n_introns,
            trans_spliced=trans,
            spellings=spellings_for(name, kind),
            nested_in=host if host in chosen_set else None,
        )
    # layout in catalogue order: LSC (closed by exon 1 of rps12), IR, SSC
    for segment in ("LSC", "IR", "SSC"):
        for name, _kind, seg, _strand, _n in _CATALOG:
            if name not in chosen_set or seg != segment:
                continue
            if specs[name].nested_in:
                continue  # laid out inside its host gene
            if name == _TRANS_GENE:
                layout.append(Element(name, "IR", "exon23"))
            else:
                layout.append(Element(name, segment))
        if segment == "LSC" and _TRANS_GENE in chosen_set:
            layout.append(Element(_TRANS_GENE, "LSC", "exon1"))
    # filler genes (large n_genes): LSC fillers precede the IR, SSC ones follow it
    for k in range(n_fillers):
        name = f"syn{k:03d}"
        segment = "LSC" if k % 3 else "SSC"
        specs[name] = GeneSpec(
            name,
            "cds",
            segment,
            1 if k % 2 else -1,
            0,
            spellings=spellings_for(name, "cds"),
        )
        idx = next((i for i, e in enumerate(layout) if e.segment == "IR"), len(layout))
        layout.insert(idx if segment == "LSC" else len(layout), Element(name, segment))

    genes = {
        name: _make_gene_seq(rng, spec, rrna_scale) for name, spec in specs.items()
    }
    spacers = [_rand_seq(rng, rng.randint(25, 160)) for _ in range(len(layout) + 2)]
    nested: dict[str, list[str]] = {}
    for name, spec in specs.items():
        if spec.nested_in:
            nested.setdefault(spec.nested_in, []).append(name)
    return Template(seed, specs, layout, GenomeState(genes, spacers), nested)


# ---------------------------------------------------------------------------
# Mutation (tree model)
# ---------------------------------------------------------------------------


def _count(rng: random.Random, lam: float) -> int:
    """Mean-preserving small count: floor(lam) plus a Bernoulli remainder."""
    k = int(lam)
    return k + (1 if rng.random() < lam - k else 0)


def _mutate_noncoding(
    rng: random.Random, seq: str, sub: float, indel: float, min_len: int
) -> str:
    n = len(seq)
    if n == 0:
        return seq
    chars = list(seq)
    for pos in rng.sample(range(n), min(n, _count(rng, n * sub))):
        chars[pos] = rng.choice(_BASES.replace(chars[pos], ""))
    seq = "".join(chars)
    for _ in range(_count(rng, n * indel)):
        pos = rng.randrange(len(seq) + 1)
        size = 1 + int(rng.expovariate(0.6))
        if rng.random() < 0.5:
            seq = seq[:pos] + _rand_seq(rng, size) + seq[pos:]
        elif len(seq) - size >= min_len:
            seq = seq[:pos] + seq[pos + size :]
    return seq


def _mutate_cds(rng: random.Random, exons: list[str], sub: float) -> list[str]:
    """Codon-preserving substitutions over the spliced coding sequence."""
    coding = "".join(exons)
    n_codons = len(coding) // 3
    if n_codons < 3:
        return exons
    chars = list(coding)
    lam = n_codons * 3 * sub
    for _ in range(_count(rng, lam)):
        codon = rng.randrange(1, n_codons - 1)  # keep the start and stop codon
        offset = rng.choices((0, 1, 2), (0.25, 0.15, 0.6))[0]
        pos = codon * 3 + offset
        old = chars[pos]
        new = rng.choice(_BASES.replace(old, ""))
        mutated = "".join(chars[codon * 3 : codon * 3 + 3])
        mutated = mutated[:offset] + new + mutated[offset + 1 :]
        if mutated in _STOPS:
            continue
        chars[pos] = new
    return _split("".join(chars), [len(e) for e in exons])


def mutate_state(
    state: GenomeState,
    template: Template,
    rng: random.Random,
    divergence: float,
    indel_rate: float,
) -> GenomeState:
    """One branch of the tree: substitutions everywhere, indels in non-coding parts."""
    genes: dict[str, GeneSeq] = {}
    for name, gs in state.genes.items():
        spec = template.specs[name]
        if spec.kind == "cds":
            exons = _mutate_cds(rng, gs.exons, divergence)
        else:  # tRNA / rRNA: slower, substitutions only
            exons = [
                _mutate_noncoding(rng, e, divergence * 0.3, 0.0, len(e))
                for e in gs.exons
            ]
        introns = [
            _mutate_noncoding(rng, i, divergence * 1.5, indel_rate, 40)
            for i in gs.introns
        ]
        genes[name] = GeneSeq(exons, introns)
    spacers = [
        _mutate_noncoding(rng, s, divergence * 2.0, indel_rate * 2, 12)
        for s in state.spacers
    ]
    return GenomeState(genes, spacers)


# ---------------------------------------------------------------------------
# Assembly: GenomeState -> sequence + placed features
# ---------------------------------------------------------------------------


@dataclass
class PlacedGene:
    """A gene as annotated in the genome (one copy)."""

    name: str  # canonical
    spelled: str  # /gene qualifier
    kind: str
    copy: str  # "LSC" | "IRb" | "SSC" | "IRa"
    start: int  # 0-based half-open range of the gene feature (union of exons)
    end: int
    strand: int
    exons: list[tuple[int, int]]  # genomic ranges, ascending
    location: str  # location of the CDS / tRNA / rRNA feature
    gene_location: str  # location of the ``gene`` feature
    coding: str = ""  # spliced sequence in transcript orientation
    trans_spliced: bool = False


def _fmt_range(a: int, b: int) -> str:
    return f"{a + 1}..{b}"


def _fmt_part(rng_: tuple[int, int], strand: int) -> str:
    base = _fmt_range(*rng_)
    return base if strand >= 0 else f"complement({base})"


def _location(exons: list[tuple[int, int]], strand: int) -> str:
    """GenBank location string for a single-strand multi-exon feature."""
    if len(exons) == 1:
        return _fmt_part(exons[0], strand)
    inner = "join(" + ",".join(_fmt_range(a, b) for a, b in exons) + ")"
    return inner if strand >= 0 else f"complement({inner})"


def _gene_pieces(gs: GeneSeq) -> list[str]:
    """exon, intron, exon, ... in gene orientation."""
    out: list[str] = []
    for i, ex in enumerate(gs.exons):
        out.append(ex)
        if i < len(gs.introns):
            out.append(gs.introns[i])
    return out


NestedGene = tuple[GeneSpec, GeneSeq, str]  # spec, sequences, /gene spelling


def _place(
    spec: GeneSpec,
    gs: GeneSeq,
    spelled: str,
    strand: int,
    copy: str,
    offset: int,
    nested: Sequence[NestedGene] = (),
) -> tuple[str, list[PlacedGene]]:
    """Genomic sequence of a gene body at ``offset`` and its annotation(s).

    ``nested`` genes (``matK`` inside the ``trnK`` intron) are inserted into the
    first intron of this gene; the result lists the host gene first.
    """
    pieces: list[str] = []
    rel: list[tuple[int, int]] = []  # exon ranges in gene orientation
    inner: list[tuple[NestedGene, int, int]] = []  # nested gene, body offset, length
    pos = 0
    for i, ex in enumerate(gs.exons):
        rel.append((pos, pos + len(ex)))
        pieces.append(ex)
        pos += len(ex)
        if i >= len(gs.introns):
            continue
        intron = gs.introns[i]
        if i == 0 and nested:
            cut = len(intron) // 3
            pieces.append(intron[:cut])
            pos += cut
            for item in nested:
                nseq, _ = _place(item[0], item[1], item[2], item[0].strand, copy, 0)
                # the body is in host orientation; nested sequences are genomic
                pieces.append(nseq if strand >= 0 else revcomp(nseq))
                inner.append((item, pos, len(nseq)))
                pos += len(nseq)
            pieces.append(intron[cut:])
            pos += len(intron) - cut
        else:
            pieces.append(intron)
            pos += len(intron)
    body = "".join(pieces)
    length = len(body)
    if strand >= 0:
        seq = body
        exons = [(offset + a, offset + b) for a, b in rel]
    else:
        seq = revcomp(body)
        exons = [(offset + length - b, offset + length - a) for a, b in reversed(rel)]
    host = PlacedGene(
        name=spec.name,
        spelled=spelled,
        kind=spec.kind,
        copy=copy,
        start=offset,
        end=offset + length,
        strand=strand,
        exons=exons,
        location=_location(exons, strand),
        gene_location=_fmt_part((offset, offset + length), strand),
        coding="".join(gs.exons),
    )
    placed = [host]
    for item, body_pos, n in inner:
        genomic = offset + body_pos if strand >= 0 else offset + length - body_pos - n
        _, sub = _place(item[0], item[1], item[2], item[0].strand, copy, genomic)
        placed.extend(sub)
    return seq, placed


@dataclass
class Assembly:
    sequence: str
    genes: list[PlacedGene]


def assemble(
    template: Template,
    state: GenomeState,
    spelled: dict[str, str],
    drop: frozenset[str] = frozenset(),
) -> Assembly:
    """Lay the genome out as ``LSC | IRb | SSC | IRa`` and place every gene copy."""
    specs, layout = template.specs, template.layout
    parts: list[str] = []
    placed: list[PlacedGene] = []
    pos = 0
    rps12_exon1: tuple[int, int] | None = None

    def add(seq: str) -> int:
        nonlocal pos
        start = pos
        parts.append(seq)
        pos += len(seq)
        return start

    def segment_indices(seg: str) -> list[int]:
        return [
            i for i, e in enumerate(layout) if e.segment == seg and e.gene not in drop
        ]

    def element_seq_and_genes(
        el: Element, offset: int, copy: str, strand: int
    ) -> tuple[str, list[PlacedGene]]:
        spec = specs[el.gene]
        gs = state.genes[el.gene]
        nonlocal rps12_exon1
        if el.part == "exon1":
            # 5' exon of the trans-spliced gene: a bare exon on the minus strand
            seq = revcomp(gs.exons[0])
            rps12_exon1 = (offset, offset + len(seq))
            return seq, []
        if el.part == "exon23":
            sub = GeneSeq(gs.exons[1:], gs.introns)
            seq, pgs = _place(spec, sub, spelled[el.gene], strand, copy, offset)
            pgs[0].trans_spliced = True
            pgs[0].coding = "".join(gs.exons)
            return seq, pgs
        nested = [
            (specs[n], state.genes[n], spelled[n])
            for n in template.nested.get(el.gene, ())
            if n not in drop
        ]
        return _place(spec, gs, spelled[el.gene], strand, copy, offset, nested)

    # --- LSC and SSC are laid out left to right -----------------------------
    def lay_single(seg: str, copy: str) -> None:
        for i in segment_indices(seg):
            add(state.spacers[i])
            el = layout[i]
            strand = specs[el.gene].strand if el.part != "exon1" else -1
            seq, pgs = element_seq_and_genes(el, pos, copy, strand)
            add(seq)
            placed.extend(pgs)

    lay_single("LSC", "LSC")

    # --- IRb, then SSC, then IRa = reverse complement of IRb ---------------
    ir_idx = segment_indices("IR")
    irb_blocks: list[str] = []
    irb_placed: list[PlacedGene] = []
    irb_start = pos
    if ir_idx:
        add(state.spacers[ir_idx[0]])
        irb_start = pos
        local = 0
        for k, i in enumerate(ir_idx):
            if k:
                sp = state.spacers[i]
                irb_blocks.append(sp)
                local += len(sp)
            el = layout[i]
            seq, pgs = element_seq_and_genes(
                el, irb_start + local, "IRb", specs[el.gene].strand
            )
            irb_blocks.append(seq)
            local += len(seq)
            irb_placed.extend(pgs)
        irb_seq = "".join(irb_blocks)
        add(irb_seq)
        placed.extend(irb_placed)
    else:
        irb_seq = ""

    lay_single("SSC", "SSC")

    if ir_idx:
        add(state.spacers[len(layout)])  # SSC / IRa junction
        ira_start = pos
        add(revcomp(irb_seq))
        block_len = len(irb_seq)
        for pg in irb_placed:
            spec = specs[pg.name]
            exons = sorted(
                (
                    (ira_start + block_len - (b - irb_start)),
                    (ira_start + block_len - (a - irb_start)),
                )
                for a, b in pg.exons
            )
            exons = [(a, b) for a, b in exons]
            strand = -pg.strand
            placed.append(
                PlacedGene(
                    name=pg.name,
                    spelled=pg.spelled,
                    kind=spec.kind,
                    copy="IRa",
                    start=exons[0][0],
                    end=exons[-1][1],
                    strand=strand,
                    exons=exons,
                    location=_location(exons, strand),
                    gene_location=_fmt_part((exons[0][0], exons[-1][1]), strand),
                    coding=pg.coding,
                    trans_spliced=pg.trans_spliced,
                )
            )
    add(state.spacers[len(layout) + 1])

    # --- trans-spliced gene: join(exon 1 in the LSC, exons 2+3 in the IR) ---
    if rps12_exon1 is not None:
        for pg in placed:
            if not pg.trans_spliced:
                continue
            exon1 = _fmt_part(rps12_exon1, -1)
            if pg.copy == "IRb":
                rest = ",".join(_fmt_range(a, b) for a, b in pg.exons)
            else:  # IRa: transcript order = descending coordinates, minus strand
                rest = ",".join(
                    f"complement({_fmt_range(a, b)})" for a, b in reversed(pg.exons)
                )
            pg.location = f"join({exon1},{rest})"
            pg.gene_location = pg.location
            pg.start = min(rps12_exon1[0], pg.start)
            pg.end = max(rps12_exon1[1], pg.end)
            pg.exons = [rps12_exon1, *pg.exons]

    placed.sort(key=lambda g: (g.start, g.end))
    return Assembly("".join(parts), placed)


# ---------------------------------------------------------------------------
# GenBank rendering
# ---------------------------------------------------------------------------

_LOC_WIDTH = 58  # 79 columns minus the 21 column indent
_INDENT = " " * 21


def _wrap_location(loc: str) -> list[str]:
    """Wrap a location string at commas like NCBI does."""
    lines: list[str] = []
    cur = ""
    for tok in re.split(r"(?<=,)", loc):
        if cur and len(cur) + len(tok) > _LOC_WIDTH:
            lines.append(cur)
            cur = tok
        else:
            cur += tok
    lines.append(cur)
    return lines


def _wrap_qualifier(key: str, value: str | None) -> list[str]:
    if value is None:
        return [f"{_INDENT}/{key}"]
    quoted = not value.lstrip("-").isdigit()
    text = f'/{key}="{value}"' if quoted else f"/{key}={value}"
    out: list[str] = []
    while len(text) > _LOC_WIDTH:
        cut = text.rfind(" ", 0, _LOC_WIDTH + 1)
        if cut <= 0:
            cut = _LOC_WIDTH
        out.append(text[:cut])
        text = text[cut + 1 :] if text[cut : cut + 1] == " " else text[cut:]
    out.append(text)
    return [_INDENT + line for line in out]


def _feature(
    ftype: str, loc: str, quals: Iterable[tuple[str, str | None]]
) -> list[str]:
    loc_lines = _wrap_location(loc)
    lines = [f"     {ftype:<16}{loc_lines[0]}"]
    lines.extend(_INDENT + ln for ln in loc_lines[1:])
    for key, value in quals:
        lines.extend(_wrap_qualifier(key, value))
    return lines


def _origin(seq: str) -> list[str]:
    lower = seq.lower()
    lines = ["ORIGIN"]
    for i in range(0, len(lower), 60):
        chunk = lower[i : i + 60]
        blocks = " ".join(chunk[j : j + 10] for j in range(0, len(chunk), 10))
        lines.append(f"{i + 1:>9} {blocks}")
    return lines


def render_genbank(
    asm: Assembly,
    specs: dict[str, GeneSpec],
    taxon_id: str,
    organism: str,
    date: str = "01-JAN-2020",
    taxid: int = 4210,
) -> str:
    """Render an :class:`Assembly` as a GenBank flatfile."""
    n = len(asm.sequence)
    accession = taxon_id
    lines = [
        f"LOCUS       {taxon_id:<16} {n:>11} bp    DNA     circular PLN {date}",
        f"DEFINITION  {organism} chloroplast, complete genome.",
        f"ACCESSION   {accession}",
        f"VERSION     {accession}.1",
        "KEYWORDS    .",
        f"SOURCE      chloroplast {organism}",
        f"  ORGANISM  {organism}",
        "            Eukaryota; Viridiplantae; Streptophyta; Embryophyta;",
        "            Tracheophyta; Spermatophyta; Magnoliopsida; Asterales;",
        "            Asteraceae.",
        "FEATURES             Location/Qualifiers",
    ]
    lines += _feature(
        "source",
        f"1..{n}",
        [
            ("organism", organism),
            ("organelle", "plastid:chloroplast"),
            ("mol_type", "genomic DNA"),
            ("db_xref", f"taxon:{taxid}"),
        ],
    )
    for pg in asm.genes:
        spec = specs[pg.name]
        lines += _feature("gene", pg.gene_location, [("gene", pg.spelled)])
        if pg.kind == "cds":
            quals: list[tuple[str, str | None]] = [("gene", pg.spelled)]
            if pg.trans_spliced:
                quals += [("trans_splicing", None), ("exception", "trans-splicing")]
            quals += [
                ("codon_start", "1"),
                ("transl_table", "11"),
                ("product", spec.product),
                ("translation", translate_table11(pg.coding).rstrip("*")),
            ]
            lines += _feature("CDS", pg.location, quals)
        elif pg.kind == "trna":
            lines += _feature(
                "tRNA", pg.location, [("gene", pg.spelled), ("product", spec.product)]
            )
        else:
            lines += _feature(
                "rRNA", pg.location, [("gene", pg.spelled), ("product", spec.product)]
            )
    lines += _origin(asm.sequence)
    lines.append("//")
    return "\n".join(lines) + "\n"


def _spell(rng: random.Random, template: Template, variation: float) -> dict[str, str]:
    """Pick the /gene spelling of each gene for one genome."""
    out: dict[str, str] = {}
    for name, spec in template.specs.items():
        out[name] = (
            rng.choice(spec.spellings)
            if spec.spellings and rng.random() < variation
            else spec.spellings[0]
            if spec.spellings
            else name
        )
    return out


def make_plastome(
    rng_seed: int,
    taxon_id: str,
    template: Template,
    divergence: float = 0.01,
    indel_rate: float = 0.002,
    spelling_variation: float = 0.3,
    drop_genes: Iterable[str] = (),
) -> str:
    """One synthetic plastome GenBank flatfile.

    The genome is the template state mutated along a single branch of length
    ``divergence`` (substitutions per site; codon-preserving in CDS, with
    ``indel_rate`` indels per site in spacers and introns). Everything is a pure
    function of the arguments.

    Args:
        rng_seed: Seed of this genome's randomness.
        taxon_id: ``LOCUS`` / ``ACCESSION`` name (also the extractor's record name).
        template: Ancestral genome from :func:`make_template`.
        divergence: Substitutions per site along the branch.
        indel_rate: Indels per site in non-coding sequence.
        spelling_variation: Probability that a gene uses a non-canonical spelling.
        drop_genes: Canonical names of (non-IR, non-trans-spliced) genes to delete
            from this genome (gene loss).
    """
    rng = random.Random(f"genome:{rng_seed}:{taxon_id}")
    state = mutate_state(template.root, template, rng, divergence, indel_rate)
    return _render_state(
        template, state, taxon_id, rng, spelling_variation, drop_genes
    )[0]


def _render_state(
    template: Template,
    state: GenomeState,
    taxon_id: str,
    rng: random.Random,
    spelling_variation: float,
    drop_genes: Iterable[str] = (),
) -> tuple[str, Assembly]:
    spelled = _spell(rng, template, spelling_variation)
    asm = assemble(template, state, spelled, frozenset(drop_genes))
    organism = f"Synthetica {taxon_id.lower()}"
    return render_genbank(asm, template.specs, taxon_id, organism), asm


# ---------------------------------------------------------------------------
# Expected extraction results
# ---------------------------------------------------------------------------


def _first_per_name(genes: Iterable[PlacedGene]) -> list[PlacedGene]:
    seen: set[str] = set()
    out: list[PlacedGene] = []
    for g in genes:
        if g.name not in seen:
            seen.add(g.name)
            out.append(g)
    return out


DEFAULT_EXCLUDE_FULLCDS = ("rps12",)  # == UserParameters default


def excluded_names(mode: str, exclude_fullcds: Iterable[str]) -> frozenset[str]:
    """Names ``UserParameters`` makes the extractor skip (igs mode adds ``matK``)."""
    names = set(exclude_fullcds or DEFAULT_EXCLUDE_FULLCDS)
    if mode == "igs":
        names.add("matK")  # nested inside the trnK intron
    return frozenset(names)


def expected_regions_for_genome(
    asm: Assembly,
    mode: str,
    min_seq_length: int = 3,
    exclude_fullcds: Iterable[str] = DEFAULT_EXCLUDE_FULLCDS,
) -> list[str]:
    """Region names the extractor should produce for one genome, in order.

    Models the extractor's documented semantics independently of its code (genes
    named in ``exclude_fullcds`` are ignored, as are ``matK`` in igs mode):

    * ``cds``: every CDS once (IR copies are de-duplicated by name; the first copy
      in annotation order wins);
    * ``rna``: every tRNA / rRNA once;
    * ``int``: ``<gene>_intron<k>`` for CDS / tRNA features with 2 or 3 location
      parts, k = 1..parts-1 (the trans-spliced gene counts the LSC / IR gap as
      ``intron1``);
    * ``igs``: ``<a>_<b>`` for consecutive genes (the exons of a cis-spliced gene
      form one interval; the LSC exon and the merged IR exons of the trans-spliced
      gene are separate intervals) that are at least ``min_seq_length`` bp apart.
      The second IR copy yields the reverse pairs ``<b>_<a>``.
    """
    skip = excluded_names(mode, exclude_fullcds)
    genes = [g for g in asm.genes if g.name not in skip]
    if mode == "cds":
        return [g.name for g in _first_per_name(g for g in genes if g.kind == "cds")]
    if mode == "rna":
        return [g.name for g in _first_per_name(g for g in genes if g.kind != "cds")]
    if mode == "int":
        out: list[str] = []
        seen: set[str] = set()
        for g in genes:
            if g.kind == "rrna" or g.name in seen:
                continue
            seen.add(g.name)
            parts = len(g.exons)
            if parts in (2, 3):
                out.extend(f"{g.name}_intron{k}" for k in range(1, parts))
        return out
    if mode == "igs":
        return [name for name, _inv in _igs_pairs(genes, min_seq_length)]
    raise ValueError(f"unknown mode {mode!r}")


def _intervals(genes: Iterable[PlacedGene]) -> list[tuple[int, int, str]]:
    """Gene intervals as the IGS extractor sees them: (start, end, name)."""
    out: list[tuple[int, int, str]] = []
    for g in genes:
        if g.trans_spliced:
            for a, b in g.exons:
                out.append((a, b, g.name))
        else:
            out.append((g.start, g.end, g.name))
    out.sort(key=lambda t: (t[1], t[0]))
    # adjacent intervals of the same gene are merged (exon 2 + exon 3 of rps12)
    merged: list[tuple[int, int, str]] = []
    for iv in out:
        if merged and merged[-1][2] == iv[2]:
            prev = merged.pop()
            merged.append((min(prev[0], iv[0]), max(prev[1], iv[1]), iv[2]))
        else:
            merged.append(iv)
    return merged


def _igs_pairs(
    genes: Iterable[PlacedGene], min_seq_length: int = 3
) -> list[tuple[str, str]]:
    ivs = _intervals(genes)
    pairs: list[tuple[str, str]] = []
    for cur, nxt in zip(ivs, ivs[1:]):
        gap = nxt[0] - cur[1]
        if gap >= min_seq_length:
            pairs.append((f"{cur[2]}_{nxt[2]}", f"{nxt[2]}_{cur[2]}"))
    return pairs


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetManifest:
    """What :func:`write_genbank_dataset` wrote and what extraction should find."""

    out_dir: Path
    seed: int
    n_genes: int
    exclude_fullcds: tuple[str, ...]  # the exclusion the expectations assume
    taxa: list[str]  # taxon ids == LOCUS names, in sorted file order
    files: list[Path]  # sorted like ``PlastidData.files``
    gene_names: list[str]  # canonical names in layout order
    # mode -> region -> number of genomes expected to contribute a record
    expected: dict[str, dict[str, int]] = field(default_factory=dict)
    # mode -> region names in first-genome (``order="seq"``) registration order
    expected_order: dict[str, list[str]] = field(default_factory=dict)
    # per taxon: region names per mode (for loss / partial-dataset comparisons)
    per_taxon: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    # taxon -> canonical CDS name -> spliced coding sequence (only if keep_truth)
    truth_cds: dict[str, dict[str, str]] = field(default_factory=dict)

    @property
    def n_taxa(self) -> int:
        return len(self.taxa)

    def regions(self, mode: str) -> list[str]:
        """Expected region names of ``mode`` (first-genome order)."""
        return list(self.expected_order[mode])

    def to_dict(self) -> dict[str, Any]:
        return {
            "out_dir": str(self.out_dir),
            "seed": self.seed,
            "n_genes": self.n_genes,
            "taxa": self.taxa,
            "files": [str(p) for p in self.files],
            "gene_names": self.gene_names,
            "expected": self.expected,
        }


def _distinct(names: list[str]) -> list[str]:
    return list(dict.fromkeys(names))


def write_genbank_dataset(
    out_dir: str | Path,
    n_taxa: int = 8,
    n_genes: int = 24,
    seed: int = 0,
    divergence: float = 0.012,
    indel_rate: float = 0.002,
    spelling_variation: float = 0.3,
    gene_loss_rate: float = 0.0,
    fileext: str = ".gb",
    keep_truth: bool = False,
    rrna_scale: float = 1.0,
    min_seq_length: int = 3,
    exclude_fullcds: Iterable[str] = DEFAULT_EXCLUDE_FULLCDS,
) -> DatasetManifest:
    """Write ``n_taxa`` related synthetic plastomes into ``out_dir``.

    Taxa are ``SYN000000`` ... (zero padded; file ``SYN000000.gb``), related by a
    random tree: taxon 0 is the template mutated along one branch, every later
    taxon is a randomly chosen earlier taxon mutated along one branch. All
    randomness derives from ``seed``.

    Speed (``n_genes=24``, one core of a 2020 laptop): about 400 genomes/s;
    300 genomes in under a second, 3000 in about 8 s (16 kb each). Scales
    linearly in ``n_taxa * n_genes``.

    Args:
        out_dir: Directory to write into (created).
        n_taxa: Number of genomes.
        n_genes: Genes per genome (see :func:`make_template`).
        seed: Master seed.
        divergence: Substitutions per site per branch.
        indel_rate: Indels per site per branch in spacers and introns.
        spelling_variation: Probability of a non-canonical ``/gene`` spelling.
        gene_loss_rate: Per genome and gene probability of deleting a gene (never
            IR genes or the trans-spliced gene); the manifest accounts for it.
        fileext: File extension.
        keep_truth: Also store each genome's coding sequences in the manifest.
        rrna_scale: Scales rRNA lengths.
        min_seq_length: ``min_seq_length`` the expectations assume.
        exclude_fullcds: ``exclude_fullcds`` the expectations assume (the
            ``UserParameters`` default excludes the trans-spliced ``rps12``; pass
            the same list to ``UserParameters``; ``["none"]`` keeps every gene).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    exclusion = tuple(exclude_fullcds)
    template = make_template(seed, n_genes, rrna_scale)
    rng = random.Random(f"dataset:{seed}")
    width = max(6, len(str(max(n_taxa - 1, 0))))
    states: list[GenomeState] = []
    taxa: list[str] = []
    files: list[Path] = []
    per_taxon: dict[str, dict[str, list[str]]] = {}
    truth: dict[str, dict[str, str]] = {}
    expected: dict[str, dict[str, int]] = {m: {} for m in MODES}
    first_order: dict[str, list[str]] = {}
    lossy = sorted(
        n
        for n, s in template.specs.items()
        if s.segment != "IR" and not s.trans_spliced
    )
    for i in range(n_taxa):
        taxon = f"SYN{i:0{width}d}"
        parent = template.root if i == 0 else states[rng.randrange(i)]
        state = mutate_state(parent, template, rng, divergence, indel_rate)
        states.append(state)
        drop = frozenset(
            n for n in lossy if gene_loss_rate and rng.random() < gene_loss_rate
        )
        text, asm = _render_state(template, state, taxon, rng, spelling_variation, drop)
        path = out / f"{taxon}{fileext}"
        path.write_text(text, encoding="ascii")
        taxa.append(taxon)
        files.append(path)
        names: dict[str, list[str]] = {}
        for mode in MODES:
            regs = expected_regions_for_genome(asm, mode, min_seq_length, exclusion)
            names[mode] = regs
            for reg in regs:
                expected[mode][reg] = expected[mode].get(reg, 0) + 1
            if i == 0:
                first_order[mode] = _distinct(regs)
        per_taxon[taxon] = names
        if keep_truth:
            truth[taxon] = {
                g.name: g.coding for g in _first_per_name(asm.genes) if g.kind == "cds"
            }
    # regions first seen after genome 0 (gene loss) come later, in file order
    for mode in MODES:
        seen = set(first_order[mode])
        for taxon in taxa[1:]:
            for reg in per_taxon[taxon][mode]:
                if reg not in seen:
                    seen.add(reg)
                    first_order[mode].append(reg)
    return DatasetManifest(
        out_dir=out,
        seed=seed,
        n_genes=n_genes,
        exclude_fullcds=exclusion,
        taxa=taxa,
        files=files,
        gene_names=template.gene_names,
        expected=expected,
        expected_order=first_order,
        per_taxon=per_taxon,
        truth_cds=truth,
    )
