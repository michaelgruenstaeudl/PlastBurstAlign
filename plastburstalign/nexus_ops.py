"""Domain operations and incremental data structures for NEXUS alignment data."""

from __future__ import annotations

import copy

from Bio.Nexus.Nexus import Nexus as NexusAlignment
from Bio.Seq import Seq


class ConcatenationError(RuntimeError):
    """Raised when alignment concatenation fails."""


class IncrementalNexusAccumulator:
    """Incrementally accumulates individual Nexus alignment matrices into a single

    concatenated Nexus alignment. Produces output byte-equivalent to Nexus.Nexus.combine()
    without holding all individual alignments in memory simultaneously.
    """

    def __init__(self) -> None:
        self.combined: NexusAlignment | None = None
        self.chunk_buffer: dict[str, list[str]] = {}

    def add(self, name: str, m: NexusAlignment) -> None:
        if self.combined is None:
            self.combined = copy.deepcopy(m)
            self.combined.charlabels = None
            self.combined.statelabels = None
            self.combined.interleave = False
            self.combined.translate = None

            for cn, cs in list(self.combined.charsets.items()):
                self.combined.charsets[f"{name}.{cn}"] = cs
                del self.combined.charsets[cn]
            for tn, ts in list(self.combined.taxsets.items()):
                self.combined.taxsets[f"{name}.{tn}"] = ts
                del self.combined.taxsets[tn]

            self.combined.charpartitions = {
                "combined": {name: list(range(self.combined.nchar))}
            }
            # Initialize O(1) append buffer with string representations
            self.chunk_buffer = {
                t: [str(seq)] for t, seq in self.combined.matrix.items()
            }
            return

        comb = self.combined
        assert comb is not None

        # Handle mixed datatypes across partitions
        if comb.datatype != m.datatype:
            comb.datatype = "None"

        comb_taxlabels: list[str] = comb.taxlabels
        m_taxlabels: list[str] = m.taxlabels

        # O(1) set membership replaces O(N^2) list scans
        comb_tax_set = set(comb_taxlabels)
        m_tax_set = set(m_taxlabels)

        both = [t for t in comb_taxlabels if t in m_tax_set]
        combined_only = [t for t in comb_taxlabels if t not in m_tax_set]
        m_only = [t for t in m_taxlabels if t not in comb_tax_set]

        assert isinstance(comb.matrix, dict)
        assert isinstance(m.matrix, dict)
        m_matrix = m.matrix

        m_gap = m.gap
        comb_gap = comb.gap
        m_missing = m.missing
        comb_missing = comb.missing
        needs_replace = (m_gap != comb_gap) or (m_missing != comb_missing)

        def clean_m_seq(raw_seq) -> str:
            s = str(raw_seq)
            if needs_replace:
                s = s.replace(m_gap, comb_gap).replace(m_missing, comb_missing)
            return s

        # Precompute padding strings once per locus instead of per-taxon
        missing_padding = comb_missing * m.nchar
        comb_missing_padding = comb_missing * comb.nchar

        # O(1) list appends avoid quadratic string copying
        for t in both:
            self.chunk_buffer[t].append(clean_m_seq(m_matrix[t]))
        for t in combined_only:
            self.chunk_buffer[t].append(missing_padding)
        for t in m_only:
            self.chunk_buffer[t] = [comb_missing_padding, clean_m_seq(m_matrix[t])]

        comb_taxlabels.extend(m_only)
        for cn, cs in m.charsets.items():
            comb.charsets[f"{name}.{cn}"] = [x + comb.nchar for x in cs]
        if m.taxsets:
            if not comb.taxsets:
                comb.taxsets = {}
            comb.taxsets.update({f"{name}.{tn}": ts for tn, ts in m.taxsets.items()})

        comb.charpartitions["combined"][name] = list(
            range(comb.nchar, comb.nchar + m.nchar)
        )
        if m.charlabels:
            if not comb.charlabels:
                comb.charlabels = {}
            comb.charlabels.update(
                {comb.nchar + i: label for i, label in m.charlabels.items()}
            )
        comb.nchar += m.nchar
        comb.ntax += len(m_only)

    def finalize(self) -> NexusAlignment | None:
        if self.combined is None:
            return None

        # Assemble final matrix in a single pass per taxon
        self.combined.matrix = {
            t: Seq("".join(chunks)) for t, chunks in self.chunk_buffer.items()
        }
        self.chunk_buffer.clear()

        charpartitions: dict[str, dict[str, list[int]]] = self.combined.charpartitions
        charsets: dict[str, list[int]] = self.combined.charsets
        for c, partition in charpartitions.get("combined", {}).items():
            charsets[c] = partition
        return self.combined