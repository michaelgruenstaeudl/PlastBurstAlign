"""Tunable constants of the alignment-strategy work model.

This file is *data*: ``plastburstalign.strategy`` contains the logic and reads
every number from :data:`DEFAULT_CONSTANTS`. The calibration script
(``benchmarking/calibrate_strategy.py --fit``) prints a ready-to-paste
replacement for the ``DEFAULT_CONSTANTS`` block below; a calibration run is
the *only* thing that should change the numbers (and it also updates the
provenance fields).

Work model
==========
::

    W = scale * N * L_max^alpha(L_max)  +  beta * N^2 * L_max

* The first term is the progressive (guide-tree) alignment cost: ``N - 1``
  profile merges of cost about ``L^alpha`` (``alpha`` 2.0 below 1 kb, 2.2 below
  2.5 kb, 2.4 above, the exponent growing because alignment width inflates).
* The second term is MAFFT's all-vs-all 6-mer distance stage, which the
  blueprint's ``W = N * L^alpha`` omits. It is what makes huge-N / short-L
  loci expensive even though ``L^alpha`` is tiny.

Tier boundaries (``W`` units, ``ACCURATE`` below the first)::

    W <  work_fast_tree      ACCURATE       mafft --auto
    W <  work_high_scale     FAST_TREE      --parttree --retree 2 --maxiterate 0
    W <  work_extreme_scale  HIGH_SCALE     ... --nofft
    W >= work_extreme_scale  EXTREME_SCALE  --parttree --retree 1 ... --nofft

A fixed cap :data:`N_FORCE_PARTTREE` additionally forces (at least) the
``FAST_TREE`` tier. It guards the O(N^2) distance matrix of the non-PartTree
algorithms without consulting the host, so the choice stays a pure function of
the data.

Provisional values (``calibrated=False``)
=========================================
Until the calibration phase has run on a quiet machine the numbers are
derived analytically; they are deliberately *conservative* (they err towards
the accurate ``--auto`` algorithm) because a wrongly chosen heuristic tier is a
silent accuracy loss while a wrongly chosen ``--auto`` is only slow:

``work_fast_tree = 3.0e10``
    The pre-v3 code used 2.5e9, which already moved 16 taxa x 4 kb
    (``W = 7.5e9``) and 50 taxa x 3 kb (``W = 1.1e10``) to PartTree. The v3
    requirement is that ordinary loci stay ``ACCURATE``: 100 taxa x 3 kb
    (``W = 2.2e10``) must keep ``--auto``, which MAFFT handles in seconds, so
    the first boundary has to lie above 2.2e10. 3.0e10 is the next round value
    (1.4x margin; 12x the pre-v3 boundary). **This deliberately deviates from
    the pre-v3 anchor 2.5e9**; the calibration run decides the final value.
``work_extreme_scale = 1.5e11``
    Anchored to the cases that stall a progressive run and need the cheapest
    tier: the 700-taxon 3.5-kb Monocot IGS (``W = 2.2e11``) and 13.7k taxa x
    1.5 kb (``W = 2.7e11`` incl. the pairwise term). Both land above it; the
    pre-v3 value 2.0e10 was reached already by 50 taxa x 4 kb.
``work_high_scale = 7.0e10``
    The geometric midpoint of the two boundaries above
    (``sqrt(3.0e10 * 1.5e11) = 6.7e10``), giving three roughly equally wide
    (on a log scale) heuristic bands.
``beta = 0.5``
    The distance stage costs about ``N^2 / 2 * L`` 6-mer operations, roughly
    one DP cell each, so in DP-cell equivalents ``beta ~ 0.5``.
``scale = 1.0`` (both alphabets) and equal protein exponents
    The protein model is a separate set of constants (the per-cell cost of a
    20-letter alphabet differs), but no protein measurements exist yet, so it
    starts equal to the nucleotide model and is fitted separately by the
    calibration.
``N_FORCE_PARTTREE = 5000``
    ``8 * N^2`` bytes is 200 MB for the distance matrix at N = 5000 (800 MB at
    10 000, 3 GB at 20 000), per concurrently running locus. The pre-v3
    "retree 1" count gate was 5000 as well.
``addfragments_max_n = 5000``
    Short sequences (< 6 residues) are added to the PartTree alignment of the
    others with ``mafft --addfragments``. Measured peak RSS of that step
    (MAFFT 7.526, simulated loci) grows as ``~25 B * N^2`` (67 MB at N = 1500,
    238 MB at 3000, 917 MB at 6000), i.e. ~4.7 GB extrapolated to 13.7k, which
    defeats the purpose of PartTree. ``--add`` needs ``~4.4 B * N^2`` (160 MB at
    6000) and is used above this N. The switch depends on N only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

#: Sequences shorter than this (in residues) make MAFFT's PartTree abort
#: (its 6-mer index needs at least 6 residues).
MIN_PARTTREE_SEQ_LEN: int = 6

#: Fixed reference cap on N: at or above it PartTree is always used (memory
#: safety for the O(N^2) all-vs-all distance matrix). It never depends on the
#: host.
N_FORCE_PARTTREE: int = 5000


@dataclass(frozen=True)
class AlphabetModel:
    """Work-model constants for one alphabet.

    Attributes:
        alpha_breaks: Length breakpoints ``(b1, b2)`` between the exponents.
        alphas: Exponents used for ``L < b1``, ``L < b2`` and ``L >= b2``.
        scale: Multiplier of the progressive term ``N * L^alpha``.
        beta: Coefficient of the all-vs-all term ``N^2 * L``.
    """

    alpha_breaks: tuple[int, int] = (1000, 2500)
    alphas: tuple[float, float, float] = (2.0, 2.2, 2.4)
    scale: float = 1.0
    beta: float = 0.5

    def alpha_for(self, max_len: int) -> float:
        """Exponent for a locus whose longest sequence has ``max_len`` residues."""
        if max_len < self.alpha_breaks[0]:
            return self.alphas[0]
        if max_len < self.alpha_breaks[1]:
            return self.alphas[1]
        return self.alphas[2]


@dataclass(frozen=True)
class StrategyConstants:
    """All numbers of the strategy model plus their provenance.

    The three ``work_*`` boundaries must be non-decreasing; ``strategy``
    enforces this cumulatively when overrides break it.
    """

    nucl: AlphabetModel = field(default_factory=AlphabetModel)
    prot: AlphabetModel = field(default_factory=AlphabetModel)
    work_fast_tree: float = 3.0e10
    work_high_scale: float = 7.0e10
    work_extreme_scale: float = 1.5e11
    n_force_parttree: int = N_FORCE_PARTTREE
    min_parttree_seq_len: int = MIN_PARTTREE_SEQ_LEN
    addfragments_max_n: int = 5000
    # --- provenance -------------------------------------------------------
    calibrated: bool = False
    source: str = (
        "provisional: analytic, conservative (prefers --auto); 2026-11-01 reduced "
        "grid on Ryzen 5650U supported keeping these numbers (naive --fit rejected; "
        "see docs/calibration.md)"
    )
    mafft_version: str = "v7.526 (2024/Apr/26)"
    cpu: str = (
        "AMD Ryzen 5 PRO 5650U with Radeon Graphics (12 logical cpus); "
        "reduced-grid check only, calibrated=False"
    )
    date: str = "2026-11-01"
    target_seconds: float = 45.0
    target_threads: int = 8

    def model_for(self, alphabet: str) -> AlphabetModel:
        """Return the model of a normalized alphabet (``"nucl"`` or ``"prot"``)."""
        return self.prot if alphabet == "prot" else self.nucl

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable dict (for the run manifest)."""
        return asdict(self)


#: Constants used by default. A calibration run overwrites this instance.
DEFAULT_CONSTANTS = StrategyConstants()
