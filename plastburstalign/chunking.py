"""Deterministic anchor-based sub-locus chunking.

Very long loci (L >= ~5 kb) dominate the critical path of a run because
progressive alignment is superlinear in the length. This module splits such a
locus at *shared anchors* into shorter pieces that can be aligned
independently (and concurrently), then splices the per-chunk alignments back
into one alignment.

The module is **pure**: it knows nothing about schedulers, threads or cores.
The plan depends only on the data and on :class:`ChunkConfig`. Neither the
number of cores, nor the iteration order of sets/dicts, nor ``PYTHONHASHSEED``
can change it (tests enforce this with subprocesses).

Algorithm
---------
1. A deterministic *reference* is chosen: the sequence of median length (ties
   broken by id, then input index).
2. For each ideal boundary ``j * L_ref / m`` (``m = round(L_ref /
   target_chunk_len)``) the reference k-mers around it are examined, nearest
   first. A k-mer becomes an *anchor* only if it is not low-complexity and
   occurs **exactly once in every sequence** (``str.find`` twice, early exit on
   the first failing sequence) and keeps the chain **strictly collinear** (its
   position in every sequence lies after the previous anchor).
3. Every sequence is cut at its anchor positions; the anchor k-mer is the
   first residues of the right-hand chunk.
4. Chunks are aligned independently. A sequence with an empty piece in a chunk
   is left out of that chunk's alignment (aligners cannot align empty input)
   and receives an all-gap row of that chunk's width on splicing.
5. :func:`splice_alignments` concatenates, per taxon and in the original input
   order, the rows of the chunk alignments. The residues of every output row
   are the original residues (case included), so the ungapped row *is* the
   input sequence.

The plan is skipped (``None``) if the locus is shorter than the minimum, has
fewer than two sequences, ``adjust_direction`` is set (reverse-complemented
input breaks collinearity), or no valid collinear anchor exists.

Typical use::

    plan = plan_chunks("nucl_ycf1.unalign.fasta", "nucl", ChunkConfig())
    if plan is not None:
        paths = plan.write_chunk_fastas(scratch)
        # ... align every plan.chunks[c] with needs_alignment -> aligned[c]
        splice_alignments(plan, aligned_paths, "nucl_ycf1.aligned.fasta")
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .logging_ops import logger as log

__all__ = [
    "Anchor",
    "ChunkConfig",
    "ChunkPlan",
    "ChunkSpec",
    "ChunkSpliceError",
    "SpliceResult",
    "plan_chunks",
    "plan_chunks_with_reason",
    "splice_alignments",
]

_NUCL_LETTERS = frozenset("ACGT")
_PROT_LETTERS = frozenset("ACDEFGHIKLMNPQRSTVWY")
GAP = "-"


class ChunkSpliceError(ValueError):
    """Aligned chunks do not match the plan (the caller should not splice)."""


@dataclass(frozen=True)
class ChunkConfig:
    """Parameters of the chunker. The plan is a pure function of data + config.

    Attributes:
        min_locus_len: Nucleotide loci whose longest sequence is shorter than
            this are not chunked.
        target_chunk_len: Desired chunk length (nucleotides).
        min_locus_len_prot / target_chunk_len_prot: The same for protein loci
            (residues).
        k_nucl / k_prot: Anchor k-mer length per alphabet.
        window_fraction: Search window around each ideal boundary, as a
            fraction of the target chunk length (each side).
        min_chunk_fraction: Smallest allowed chunk (in the reference), as a
            fraction of the target chunk length.
        min_distinct: Minimum number of distinct symbols in an anchor.
        max_run: Longest allowed homopolymer run inside an anchor.
        min_sequences: Fewer sequences than this are not chunked.
        max_candidates: Upper bound of k-mers examined per boundary.
        adjust_direction: ``True`` when sequences may be reverse-complemented
            (``--adjustdirection``); chunking is then disabled.
    """

    min_locus_len: int = 5000
    target_chunk_len: int = 2500
    min_locus_len_prot: int = 1700
    target_chunk_len_prot: int = 850
    k_nucl: int = 15
    k_prot: int = 6
    window_fraction: float = 0.4
    min_chunk_fraction: float = 0.25
    min_distinct: int = 3
    max_run: int = 4
    min_sequences: int = 2
    max_candidates: int = 4000
    adjust_direction: bool = False

    def for_alphabet(self, alphabet: str) -> tuple[int, int, int]:
        """``(min_locus_len, target_chunk_len, k)`` for ``"nucl"`` or ``"prot"``."""
        if alphabet == "prot":
            return self.min_locus_len_prot, self.target_chunk_len_prot, self.k_prot
        return self.min_locus_len, self.target_chunk_len, self.k_nucl


@dataclass(frozen=True)
class Anchor:
    """A chain anchor: a k-mer unique in every sequence.

    Attributes:
        index: 0-based position in the chain (anchor ``j`` starts chunk ``j+1``).
        kmer: The (upper-case) anchor k-mer.
        ref_pos: Start of the k-mer in the reference sequence.
        positions: Start of the k-mer in every sequence (input order).
    """

    index: int
    kmer: str
    ref_pos: int
    positions: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class ChunkSpec:
    """Summary of one chunk, usable as hints for the aligner.

    Attributes:
        index: Chunk number (left to right).
        n_seq: Number of sequences with a non-empty piece (rows to align).
        n_empty: Number of sequences with an empty piece (all-gap rows).
        min_len / max_len: Piece length range (residues) over non-empty pieces.
        needs_alignment: ``False`` when fewer than two non-empty pieces exist.
    """

    index: int
    n_seq: int
    n_empty: int
    min_len: int
    max_len: int
    needs_alignment: bool


@dataclass(frozen=True)
class SpliceResult:
    """Outcome of :func:`splice_alignments`."""

    path: Path
    n_seq: int
    width: int
    chunk_widths: tuple[int, ...]


@dataclass(frozen=True)
class ChunkPlan:
    """A chunking plan for one locus (see the module docstring).

    The plan keeps the sequences in memory (``chunk_sequences`` and the splice
    need them); they are excluded from ``repr``, equality and :meth:`to_dict`.
    """

    name: str
    alphabet: str
    config: ChunkConfig
    ids: tuple[str, ...]
    headers: tuple[str, ...]
    reference_index: int
    anchors: tuple[Anchor, ...]
    chunks: tuple[ChunkSpec, ...]
    cuts: tuple[tuple[int, ...], ...] = field(repr=False)
    sequences: tuple[str, ...] = field(repr=False, compare=False)

    @property
    def n_seq(self) -> int:
        return len(self.ids)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def chunk_piece(self, row: int, chunk: int) -> str:
        """The piece of sequence ``row`` that belongs to ``chunk`` (may be empty)."""
        bounds = self.cuts[row]
        return self.sequences[row][bounds[chunk] : bounds[chunk + 1]]

    def chunk_sequences(self, chunk: int) -> list[tuple[int, str]]:
        """``[(row_index, piece), ...]`` of the non-empty pieces of a chunk."""
        out: list[tuple[int, str]] = []
        for row in range(self.n_seq):
            piece = self.chunk_piece(row, chunk)
            if piece:
                out.append((row, piece))
        return out

    def chunk_path(self, directory: str | os.PathLike[str], chunk: int) -> Path:
        """Path of the unaligned FASTA of ``chunk`` inside ``directory``."""
        return Path(directory) / f"{self.name}.chunk{chunk:02d}.unalign.fasta"

    def write_chunk_fastas(self, directory: str | os.PathLike[str]) -> list[Path]:
        """Write one unaligned FASTA per chunk; return the paths (chunk order).

        Records are named ``s<row_index>`` (the input row), so the aligned
        chunks can be mapped back without relying on the original ids.
        """
        target = Path(directory)
        target.mkdir(parents=True, exist_ok=True)
        paths: list[Path] = []
        for chunk in range(self.n_chunks):
            path = self.chunk_path(target, chunk)
            with open(path, "w", encoding="utf-8") as handle:
                for row, piece in self.chunk_sequences(chunk):
                    handle.write(f">s{row}\n{piece}\n")
            paths.append(path)
        return paths

    @property
    def fingerprint(self) -> str:
        """SHA-256 over the config, ids, sequences and cuts (data-only, stable)."""
        digest = hashlib.sha256()
        digest.update(json.dumps(asdict(self.config), sort_keys=True).encode())
        digest.update(self.alphabet.encode())
        for ident, seq in zip(self.ids, self.sequences):
            digest.update(ident.encode())
            digest.update(b"\0")
            digest.update(seq.encode())
            digest.update(b"\1")
        for bounds in self.cuts:
            digest.update(",".join(map(str, bounds)).encode())
            digest.update(b"\n")
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable description for the run manifest."""
        return {
            "name": self.name,
            "alphabet": self.alphabet,
            "n_seq": self.n_seq,
            "n_chunks": self.n_chunks,
            "reference_id": self.ids[self.reference_index],
            "config": asdict(self.config),
            "anchors": [
                {"index": a.index, "kmer": a.kmer, "ref_pos": a.ref_pos}
                for a in self.anchors
            ],
            "chunks": [asdict(c) for c in self.chunks],
            "fingerprint": self.fingerprint,
        }


# --------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------


def _read_fasta(path: str | os.PathLike[str]) -> list[tuple[str, str]]:
    """``[(header_line, sequence), ...]`` from a FASTA file (header without '>')."""
    records: list[tuple[str, list[str]]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                records.append((line[1:], []))
            elif records:
                records[-1][1].append(line)
    return [(header, "".join(parts)) for header, parts in records]


def _normalize_records(records: Any) -> list[tuple[str, str, str]]:
    """``[(id, header, sequence), ...]`` from a path, tuples or SeqRecord-likes."""
    if isinstance(records, (str, os.PathLike)):
        out: list[tuple[str, str, str]] = []
        for header, seq in _read_fasta(records):
            ident = header.split(None, 1)[0] if header else ""
            out.append((ident, header, seq))
        return out
    out = []
    for rec in records:
        if isinstance(rec, tuple):
            ident, seq = str(rec[0]), str(rec[1])
            out.append((ident, ident, seq))
        else:
            ident = str(rec.id)
            header = str(getattr(rec, "description", "") or ident)
            out.append((ident, header, str(rec.seq)))
    return out


def _is_low_complexity(kmer: str, config: ChunkConfig) -> bool:
    """Too few symbols or dinucleotides, homopolymer runs, or tandem repeats."""
    if len(set(kmer)) < config.min_distinct:
        return True
    if len({kmer[i : i + 2] for i in range(len(kmer) - 1)}) < max(2, len(kmer) // 3):
        return True
    run = 1
    for prev, cur in zip(kmer, kmer[1:]):
        run = run + 1 if cur == prev else 1
        if run > config.max_run:
            return True
    k = len(kmer)
    for period in (1, 2, 3):
        if period < k and kmer == (kmer[:period] * (k // period + 1))[:k]:
            return True
    return False


def _median_reference(lengths: Sequence[int], ids: Sequence[str]) -> int:
    """Index of the median-length sequence (ties: id, then input index)."""
    order = sorted(range(len(lengths)), key=lambda i: (lengths[i], ids[i], i))
    return order[(len(order) - 1) // 2]


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------


def plan_chunks_with_reason(
    records: str | os.PathLike[str] | Iterable[Any],
    alphabet: str = "nucl",
    config: ChunkConfig | None = None,
    *,
    name: str = "locus",
) -> tuple[ChunkPlan | None, str]:
    """Like :func:`plan_chunks`, additionally returning the reason for ``None``.

    The reason (``"chunked into 3"`` or e.g. ``"no collinear anchor chain"``)
    is meant for the run manifest.
    """
    cfg = config or ChunkConfig()
    alpha = "prot" if str(alphabet).lower() in ("prot", "protein", "aa") else "nucl"
    min_locus, target, k = cfg.for_alphabet(alpha)

    if cfg.adjust_direction and alpha == "nucl":
        return None, "adjust_direction enabled (reverse-complement input)"
    rows = _normalize_records(records)
    n_seq = len(rows)
    if n_seq < max(2, cfg.min_sequences):
        return None, f"too few sequences ({n_seq})"
    ids = tuple(r[0] for r in rows)
    headers = tuple(r[1] for r in rows)
    seqs = tuple(r[2] for r in rows)
    lengths = [len(s) for s in seqs]
    max_len = max(lengths)
    if max_len < min_locus:
        return None, f"longest sequence {max_len} < min_locus_len {min_locus}"

    ref_idx = _median_reference(lengths, ids)
    upper = [s.upper() for s in seqs]
    ref = upper[ref_idx]
    ref_len = len(ref)
    n_chunks_target = max(2, round(ref_len / target))
    window = max(1, int(cfg.window_fraction * target))
    min_chunk = max(k, int(cfg.min_chunk_fraction * target))
    allowed = _PROT_LETTERS if alpha == "prot" else _NUCL_LETTERS

    # Sequences that fail often are tried first (deterministic move-to-front);
    # the order never changes which anchors are accepted.
    check_order = list(range(n_seq))
    anchors: list[Anchor] = []
    prev_ref_pos = 0
    # start of the previous anchor per sequence (-k: no anchor yet, so the first
    # permitted start is 0); the next anchor must begin at or after prev + k
    prev_positions: list[int] = [-k] * n_seq

    for j in range(1, n_chunks_target):
        ideal = round(j * ref_len / n_chunks_target)
        lo = max(ideal - window, prev_ref_pos + min_chunk)
        hi = min(ideal + window, ref_len - min_chunk)
        if lo > hi:
            continue
        found: Anchor | None = None
        tried = 0
        for dist in range(0, window + 1):
            for rp in (ideal - dist, ideal + dist) if dist else (ideal,):
                if rp < lo or rp > hi or rp + k > ref_len:
                    continue
                tried += 1
                kmer = ref[rp : rp + k]
                if not set(kmer) <= allowed or _is_low_complexity(kmer, cfg):
                    continue
                first = ref.find(kmer)
                if first != rp or ref.find(kmer, rp + 1) != -1:
                    continue
                positions = [-1] * n_seq
                failed_at = -1
                for rank, i in enumerate(check_order):
                    seq = upper[i]
                    p = seq.find(kmer)
                    min_pos = prev_positions[i] + k
                    if p < min_pos or seq.find(kmer, p + 1) != -1:
                        failed_at = rank
                        break
                    positions[i] = p
                if failed_at >= 0:
                    if failed_at > 0:  # move the offender to the front
                        check_order.insert(0, check_order.pop(failed_at))
                    continue
                found = Anchor(len(anchors), kmer, rp, tuple(positions))
                break
            if found is not None or tried >= cfg.max_candidates:
                break
        if found is None:
            continue
        anchors.append(found)
        prev_ref_pos = found.ref_pos
        prev_positions = list(found.positions)

    if not anchors:
        return None, "no collinear anchor chain (no k-mer unique in every sequence)"

    cuts: list[tuple[int, ...]] = []
    for i in range(n_seq):
        bounds = [0] + [a.positions[i] for a in anchors] + [lengths[i]]
        if any(b2 < b1 for b1, b2 in zip(bounds, bounds[1:])):  # pragma: no cover
            return None, "anchor chain is not collinear"  # defensive; by construction
        cuts.append(tuple(bounds))

    n_chunks = len(anchors) + 1
    specs: list[ChunkSpec] = []
    for c in range(n_chunks):
        piece_lens = [b[c + 1] - b[c] for b in cuts]
        non_empty = [x for x in piece_lens if x > 0]
        specs.append(
            ChunkSpec(
                index=c,
                n_seq=len(non_empty),
                n_empty=n_seq - len(non_empty),
                min_len=min(non_empty) if non_empty else 0,
                max_len=max(non_empty) if non_empty else 0,
                needs_alignment=len(non_empty) >= 2,
            )
        )
    plan = ChunkPlan(
        name=name,
        alphabet=alpha,
        config=cfg,
        ids=ids,
        headers=headers,
        reference_index=ref_idx,
        anchors=tuple(anchors),
        chunks=tuple(specs),
        cuts=tuple(cuts),
        sequences=seqs,
    )
    reason = (
        f"chunked into {n_chunks} (anchors at reference {[a.ref_pos for a in anchors]})"
    )
    log.debug(f"chunking {name}: {reason}")
    return plan, reason


def plan_chunks(
    records: str | os.PathLike[str] | Iterable[Any],
    alphabet: str = "nucl",
    config: ChunkConfig | None = None,
    *,
    name: str = "locus",
) -> ChunkPlan | None:
    """Plan the sub-locus chunking of one locus, or return ``None``.

    Args:
        records: A FASTA path, ``[(id, sequence), ...]`` or Biopython
            ``SeqRecord``-like objects (``.id`` and ``.seq``).
        alphabet: ``"nucl"`` or ``"prot"``.
        config: Parameters (default :class:`ChunkConfig`).
        name: Stem of the chunk file names.

    Returns:
        The :class:`ChunkPlan`, or ``None`` when chunking does not apply
        (short locus, too few sequences, ``adjust_direction``, no valid
        collinear anchor chain). Use :func:`plan_chunks_with_reason` for the
        reason.
    """
    return plan_chunks_with_reason(records, alphabet, config, name=name)[0]


# --------------------------------------------------------------------------
# splicing
# --------------------------------------------------------------------------


def _read_aligned(path: str | os.PathLike[str]) -> dict[str, str]:
    rows: dict[str, list[str]] = {}
    current: list[str] | None = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                key = line[1:].split(None, 1)[0] if len(line) > 1 else ""
                if key in rows:
                    raise ChunkSpliceError(f"duplicate record {key!r} in {path}")
                current = rows[key] = []
            elif current is not None:
                current.append(line)
    return {key: "".join(parts) for key, parts in rows.items()}


def _conform_row(aligned: str, original: str, label: str) -> str:
    """Return ``aligned`` with the residues (and case) of ``original``."""
    stripped = aligned.replace(GAP, "")
    if stripped == original:
        return aligned
    if len(stripped) != len(original) or stripped.upper() != original.upper():
        raise ChunkSpliceError(f"aligned residues of {label} differ from the input")
    if original.islower():
        return aligned.lower()
    if original.isupper():
        return aligned.upper()
    residues = iter(original)
    return "".join(GAP if ch == GAP else next(residues) for ch in aligned)


def splice_alignments(
    plan: ChunkPlan,
    aligned_chunk_paths: Sequence[str | os.PathLike[str] | None],
    out_path: str | os.PathLike[str],
    *,
    line_width: int = 60,
) -> SpliceResult:
    """Splice independently aligned chunks into one alignment FASTA.

    Rows are written in the original input order with the original headers.
    For a chunk in which a sequence has no residues the row is all gaps of the
    chunk's width. The residues (and their case) of every row are those of the
    input sequence, so ungapping a row gives back the input exactly.

    Args:
        plan: The plan the chunk FASTAs were written from.
        aligned_chunk_paths: One aligned FASTA per chunk (chunk order). Entries
            of chunks with ``needs_alignment == False`` may be ``None``.
        out_path: Destination FASTA.
        line_width: FASTA line width (0 = one line per sequence).

    Raises:
        ChunkSpliceError: If an aligned chunk is inconsistent with the plan
            (missing/extra rows, residues changed, unequal widths). Nothing is
            written in that case.
    """
    if len(aligned_chunk_paths) != plan.n_chunks:
        raise ChunkSpliceError(
            f"expected {plan.n_chunks} aligned chunks, got {len(aligned_chunk_paths)}"
        )
    chunk_rows: list[dict[int, str]] = []
    widths: list[int] = []
    for c, spec in enumerate(plan.chunks):
        members = plan.chunk_sequences(c)
        if not spec.needs_alignment:
            rows = {row: piece for row, piece in members}
            width = max((len(p) for p in rows.values()), default=0)
        else:
            path = aligned_chunk_paths[c]
            if path is None:
                raise ChunkSpliceError(f"chunk {c} needs an alignment but has no path")
            aligned = _read_aligned(path)
            expected = {f"s{row}" for row, _ in members}
            if set(aligned) != expected:
                raise ChunkSpliceError(
                    f"chunk {c}: aligned records do not match the plan "
                    f"({len(aligned)} found, {len(expected)} expected)"
                )
            rows = {}
            width = len(next(iter(aligned.values()))) if aligned else 0
            for row, piece in members:
                text = aligned[f"s{row}"]
                if len(text) != width:
                    raise ChunkSpliceError(f"chunk {c}: rows differ in length")
                rows[row] = _conform_row(text, piece, f"{plan.ids[row]} chunk {c}")
        chunk_rows.append(rows)
        widths.append(width)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    gap_rows = [GAP * w for w in widths]
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "w", encoding="utf-8") as handle:
        for row in range(plan.n_seq):
            text = "".join(
                chunk_rows[c].get(row, gap_rows[c]) for c in range(plan.n_chunks)
            )
            handle.write(f">{plan.headers[row]}\n")
            if line_width > 0:
                for start in range(0, len(text), line_width):
                    handle.write(text[start : start + line_width] + "\n")
            else:
                handle.write(text + "\n")
    os.replace(tmp, out)
    return SpliceResult(
        path=out, n_seq=plan.n_seq, width=sum(widths), chunk_widths=tuple(widths)
    )
