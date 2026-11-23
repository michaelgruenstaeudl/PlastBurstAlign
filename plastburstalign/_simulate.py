"""Pure-Python simulator of divergent homologous loci.

Used by the tests and by ``benchmarking/calibrate_strategy.py`` to produce
unaligned FASTA with a known evolutionary history: a random root sequence
evolves down a random binary tree by substitutions and indels. Block-wise rate
heterogeneity (``conserved_fraction`` of the blocks are invariant) leaves
conserved stretches so that loci contain shared unique k-mers, as real
plastid loci do (needed to test anchor-based chunking).

Everything is deterministic for a given ``seed`` and independent of the
interpreter's hash seed (only ``random.Random`` and ordered containers are
used). No numpy.
"""

from __future__ import annotations

import bisect
import itertools
import math
import random
from collections.abc import Iterable, Sequence
from pathlib import Path

__all__ = [
    "NUCLEOTIDES",
    "AMINO_ACIDS",
    "reverse_complement",
    "simulate_locus",
    "write_fasta",
]

NUCLEOTIDES = "ACGT"
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

_COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")
_MAX_INDEL = 12


def reverse_complement(seq: str) -> str:
    """Reverse complement of an ``ACGT`` string (case preserved)."""
    return seq.translate(_COMPLEMENT)[::-1]


def _poisson(rng: random.Random, lam: float) -> int:
    """Poisson variate (Knuth for small, normal approximation for large lambda)."""
    if lam <= 0.0:
        return 0
    if lam > 30.0:
        return max(0, round(rng.gauss(lam, math.sqrt(lam))))
    limit = math.exp(-lam)
    k = 0
    p = 1.0
    while True:
        p *= rng.random()
        if p <= limit:
            return k
        k += 1


class _Node:
    """A tree node. Leaves have ``leaf_index`` set and no children."""

    __slots__ = ("children", "height", "leaf_index", "n_leaves")

    def __init__(self) -> None:
        self.children: list[_Node] = []
        self.height = 0.0
        self.leaf_index = -1
        self.n_leaves = 1


def _build_tree(n_leaves: int, divergence: float, rng: random.Random) -> _Node:
    """Random binary tree; heights give a root-to-tip distance of divergence/2."""
    total = max(1, n_leaves - 1)
    root = _Node()
    root.n_leaves = n_leaves
    counter = itertools.count()
    stack: list[_Node] = [root]
    while stack:
        node = stack.pop()
        if node.n_leaves == 1:
            node.leaf_index = next(counter)
            node.height = 0.0
            continue
        node.height = 0.5 * divergence * ((node.n_leaves - 1) / total) ** 0.5
        left_size = rng.randint(1, node.n_leaves - 1)
        for size in (left_size, node.n_leaves - left_size):
            child = _Node()
            child.n_leaves = size
            node.children.append(child)
        # push the larger child first so leaf indices are assigned
        # deterministically (smaller subtree first)
        for child in sorted(node.children, key=lambda c: -c.n_leaves):
            stack.append(child)
    return root


def _random_root(
    length: int,
    alphabet: str,
    block_len: int,
    conserved_fraction: float,
    rng: random.Random,
) -> tuple[list[str], list[float]]:
    chars = [rng.choice(alphabet) for _ in range(length)]
    rates: list[float] = []
    multipliers = (0.4, 1.0, 2.5)
    while len(rates) < length:
        if rng.random() < conserved_fraction:
            rate = 0.0
        else:
            rate = rng.choice(multipliers)
        rates.extend([rate] * rng.randint(max(2, block_len // 2), block_len * 2))
    return chars, rates[:length]


def _evolve(
    chars: list[str],
    rates: list[float],
    branch: float,
    indel_per_sub: float,
    alphabet: str,
    rng: random.Random,
) -> tuple[list[str], list[float]]:
    """Apply substitutions and indels of one branch (rates weight the sites)."""
    if branch <= 0.0 or not chars:
        return list(chars), list(rates)
    total_rate = sum(rates)
    if total_rate <= 0.0:
        return list(chars), list(rates)
    mean_rate = total_rate / len(chars)
    n_sub = _poisson(rng, branch * total_rate)
    n_indel = _poisson(rng, branch * total_rate * indel_per_sub)
    out = list(chars)
    out_rates = list(rates)
    cumulative = list(itertools.accumulate(rates))

    for _ in range(n_sub):
        pos = bisect.bisect_left(cumulative, rng.random() * total_rate)
        pos = min(pos, len(out) - 1)
        if rates[pos] <= 0.0:
            continue
        new = rng.choice(alphabet)
        while new == out[pos]:
            new = rng.choice(alphabet)
        out[pos] = new

    # indels at sites drawn by rate; applied from the right so indices stay valid
    sites = sorted(
        (
            min(
                bisect.bisect_left(cumulative, rng.random() * total_rate),
                len(chars) - 1,
            )
            for _ in range(n_indel)
        ),
        reverse=True,
    )
    for site in sites:
        if rates[site] <= 0.0 or site >= len(out):
            continue
        size = min(_MAX_INDEL, 1 + int(rng.expovariate(0.5)))
        if rng.random() < 0.5:
            del out[site : site + size]
            del out_rates[site : site + size]
        else:
            ins = [rng.choice(alphabet) for _ in range(size)]
            out[site:site] = ins
            out_rates[site:site] = [mean_rate] * size
    return out, out_rates


def simulate_locus(
    n_seq: int,
    length: int,
    *,
    alphabet: str = "nucl",
    divergence: float = 0.2,
    indel_per_sub: float = 0.05,
    conserved_fraction: float = 0.25,
    block_len: int = 60,
    seed: int = 0,
    id_prefix: str = "t",
) -> list[tuple[str, str]]:
    """Simulate ``n_seq`` homologous sequences of a locus.

    Args:
        n_seq: Number of sequences (leaves of the random tree).
        length: Root sequence length (leaf lengths differ through indels).
        alphabet: ``"nucl"`` or ``"prot"``.
        divergence: Expected substitutions per site between two distant leaves
            (``divergence / 2`` from the root to a tip), before rate weights.
        indel_per_sub: Indel events per substitution event.
        conserved_fraction: Fraction of ``block_len``-sized blocks without
            any substitution or indel.
        block_len: Mean rate-block length.
        seed: Seed of the private ``random.Random``.
        id_prefix: Prefix of the sequence ids (``t00000``...).

    Returns:
        ``[(id, sequence), ...]``, ids unique, in a seed-determined order.
    """
    if n_seq < 1 or length < 1:
        raise ValueError("n_seq and length must be positive")
    letters = AMINO_ACIDS if alphabet in ("prot", "protein") else NUCLEOTIDES
    rng = random.Random(seed)
    root = _build_tree(n_seq, divergence, rng)
    chars, rates = _random_root(length, letters, block_len, conserved_fraction, rng)

    leaves: dict[int, str] = {}
    stack: list[tuple[_Node, list[str], list[float]]] = [(root, chars, rates)]
    while stack:
        node, seq, rate = stack.pop()
        if not node.children:
            leaves[node.leaf_index] = "".join(seq)
            continue
        for child in node.children:
            branch = max(0.0, node.height - child.height) * rng.uniform(0.5, 1.5)
            c_seq, c_rate = _evolve(seq, rate, branch, indel_per_sub, letters, rng)
            stack.append((child, c_seq, c_rate))

    order = list(range(n_seq))
    rng.shuffle(order)
    width = max(5, len(str(n_seq - 1)))
    return [(f"{id_prefix}{i:0{width}d}", leaves[leaf]) for i, leaf in enumerate(order)]


def write_fasta(
    path: str | Path,
    records: Iterable[tuple[str, str]] | Sequence[tuple[str, str]],
    *,
    width: int = 0,
) -> Path:
    """Write ``(id, sequence)`` records to ``path`` (``width`` 0 = one line)."""
    out = Path(path)
    with open(out, "w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n")
            if width and width > 0:
                for start in range(0, len(seq), width):
                    handle.write(seq[start : start + width] + "\n")
            else:
                handle.write(seq + "\n")
    return out
