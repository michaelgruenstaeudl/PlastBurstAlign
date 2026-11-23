"""Shared helpers for the M2 data-engine tests (not collected by pytest).

* ``simulate_locus`` / ``align_with_mafft`` produce small, real MAFFT-aligned
  FASTA files (wrapped, lowercase, optional ``_R_`` direction markers).
* ``legacy_load_fasta`` and ``legacy_concat`` reproduce, line by line, the FASTA
  path of ``AlignmentCoordination._load_alignment_fast`` and the
  accumulator/``fast_write_*`` tail of ``concat_MSAs`` so the new engine can be
  compared byte for byte against the legacy pipeline.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from plastburstalign.nexus_ops import (
    IncrementalNexusAccumulator,
    fast_write_fasta,
    fast_write_nexus,
    fast_write_nexus_partitions,
    fast_write_raxml_partitions,
)


def mafft_available() -> bool:
    """Whether a ``mafft`` executable is on PATH."""
    return shutil.which("mafft") is not None


_COMPLEMENT = str.maketrans("ACGT", "TGCA")


def reverse_complement(seq: str) -> str:
    """Reverse complement of an ``ACGT`` string."""
    return seq.translate(_COMPLEMENT)[::-1]


def simulate_locus(
    rng: random.Random,
    taxa: Sequence[str],
    length: int = 160,
    divergence: float = 0.08,
    indel_rate: float = 0.02,
    reversed_taxa: Sequence[str] = (),
) -> dict[str, str]:
    """Simulate unaligned homologous sequences (substitutions and indels)."""
    ancestor = [rng.choice("ACGT") for _ in range(length)]
    out: dict[str, str] = {}
    for taxon in taxa:
        seq: list[str] = []
        for base in ancestor:
            roll = rng.random()
            if roll < indel_rate / 2:
                continue  # deletion
            seq.append(rng.choice("ACGT") if rng.random() < divergence else base)
            if roll > 1 - indel_rate / 2:
                seq.extend(rng.choice("ACGT") for _ in range(rng.randint(1, 4)))
        text = "".join(seq)
        out[taxon] = reverse_complement(text) if taxon in reversed_taxa else text
    return out


def align_with_mafft(
    locus: str,
    sequences: dict[str, str],
    workdir: Path,
    adjustdirection: bool = False,
) -> Path:
    """Align ``{taxon: seq}`` with MAFFT; ids are written as ``<locus>_<taxon>``.

    Returns the path of the (wrapped, lowercase) aligned FASTA file, named like
    the pipeline's ``nucl_<locus>.aligned.fasta``.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    unaligned = workdir / f"nucl_{locus}.unalign.fasta"
    with open(unaligned, "w") as fh:
        for taxon, seq in sequences.items():
            fh.write(f">{locus}_{taxon}\n{seq}\n")
    cmd = ["mafft", "--auto", "--quiet", "--thread", "1"]
    if adjustdirection:
        cmd.append("--adjustdirection")
    cmd.append(str(unaligned))
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    aligned = workdir / f"nucl_{locus}.aligned.fasta"
    aligned.write_text(proc.stdout)
    return aligned


class LegacyMatrix:
    """Duck-typed stand-in for ``alignment_ops.FastNexusMatrix``."""

    def __init__(self, matrix: dict[str, str], taxlabels: list[str], nchar: int):
        self.matrix = matrix
        self.taxlabels = taxlabels
        self.nchar = nchar
        self.datatype = "dna"
        self.gap = "-"
        self.missing = "?"
        self.charsets: dict[str, Any] = {}
        self.taxsets: dict[str, Any] = {}
        self.charlabels: list[str] | None = None
        self.statelabels: list[str] | None = None
        self.charpartitions: dict[str, Any] = {}


def legacy_load_fasta(msa_name: str, file_path: str | os.PathLike[str]) -> Any:
    """Verbatim copy of the FASTA branch of ``_load_alignment_fast``."""
    prefix = f"{msa_name}_"
    r_prefix = f"_R_{msa_name}_"
    file_path = str(file_path)
    if not (os.path.exists(file_path) and os.path.getsize(file_path) > 0):
        return None
    matrix: dict[str, str] = {}
    taxlabels: list[str] = []
    try:
        with open(file_path, "r", buffering=1024 * 1024) as f:
            curr_id = None
            chunks: list[str] = []
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if curr_id:
                        matrix[curr_id] = "".join(chunks)
                        taxlabels.append(curr_id)
                        chunks.clear()
                    raw_id = line[1:].split()[0]
                    if raw_id.startswith(r_prefix):
                        curr_id = "concat_" + raw_id[len(r_prefix) :]
                    elif raw_id.startswith(prefix):
                        curr_id = "concat_" + raw_id[len(prefix) :]
                    else:
                        curr_id = raw_id
                else:
                    chunks.append(line)
            if curr_id:
                matrix[curr_id] = "".join(chunks)
                taxlabels.append(curr_id)
        if not taxlabels:
            return None
        nchar = len(matrix[taxlabels[0]])
        return LegacyMatrix(matrix, taxlabels, nchar)
    except Exception:
        return None


def legacy_concat(
    sources: Sequence[tuple[str, str | os.PathLike[str]]],
    out_prefix: str | os.PathLike[str],
) -> dict[str, Path]:
    """Run the legacy load/accumulate/write pipeline; returns the written files.

    File names follow the legacy convention
    (``<prefix>.aligned.fasta|nexus``, ``<prefix>.partitions.txt|nex``).
    """
    acc = IncrementalNexusAccumulator()
    for name, path in sources:
        loaded = legacy_load_fasta(name, path)
        if loaded is not None and getattr(loaded, "nchar", 0) > 0:
            acc.add(name, loaded)
    combined = acc.finalize()
    if combined is None:
        raise RuntimeError("legacy pipeline: no alignment could be loaded")
    base = str(out_prefix)
    paths = {
        "fasta": Path(f"{base}.aligned.fasta"),
        "nexus": Path(f"{base}.aligned.nexus"),
        "raxml": Path(f"{base}.partitions.txt"),
        "nexus_partitions": Path(f"{base}.partitions.nex"),
    }
    fast_write_fasta(paths["fasta"], combined.matrix, combined.taxlabels)
    fast_write_nexus(
        paths["nexus"],
        combined.matrix,
        datatype=combined.datatype,
        gap=combined.gap,
        missing=combined.missing,
        charpartitions=combined.charpartitions,
    )
    fast_write_raxml_partitions(
        paths["raxml"], combined.charpartitions, datatype=combined.datatype
    )
    fast_write_nexus_partitions(paths["nexus_partitions"], combined.charpartitions)
    return paths
