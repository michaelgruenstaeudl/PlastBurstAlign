"""Work-based alignment strategy selection.

The choice of the aligner algorithm is a *pure, deterministic function of the
data*: ``(N, L_min, L_max, alphabet)`` plus explicit user overrides. Hardware
(cores, RAM, tmpfs) never enters here; it only influences scheduling elsewhere.
This is what makes runs reproducible across machines.

Model (see :mod:`plastburstalign.strategy_constants` for the numbers)::

    W = scale * N * L_max^alpha  +  beta * N^2 * L_max

The first term is the progressive alignment cost, the second MAFFT's
all-vs-all distance stage (omitted by the blueprint's ``W = N * L^alpha``).
``W`` selects one of four tiers:

========================  ===================================================
``ACCURATE``              ``--auto``
``FAST_TREE``             ``--parttree --retree 2 --maxiterate 0``
``HIGH_SCALE``            ``--parttree --retree 2 --maxiterate 0 --nofft``
``EXTREME_SCALE``         ``--parttree --retree 1 --maxiterate 0 --nofft``
========================  ===================================================

PartTree cannot index sequences shorter than 6 residues. When a PartTree tier
is chosen and the shortest sequence is below that, the strategy reports
``needs_fragment_split=True``; the aligner then aligns the long sequences with
the tier and adds the short ones with ``mafft --addfragments`` instead of
falling back to the O(N^2) ``--auto``.

Explicit overrides (``overrides`` mapping, see :data:`OVERRIDE_KEYS`):

* ``tier``: force a tier (name or :class:`AlignmentTier`).
* ``work_fast_tree_threshold`` / ``work_high_scale_threshold`` /
  ``work_extreme_scale_threshold``: move the tier boundaries.
  ``work_parttree_threshold`` (= the first boundary) and
  ``work_retree1_threshold`` (= the extreme-scale boundary) are the pre-v3
  names and keep working.
* ``n_force_parttree``: change the fixed N cap.
* ``parttree_threshold`` / ``retree1_threshold`` (deprecated): the pre-v3 N
  count gates. They are honoured *only* when given explicitly, and emit a
  one-time deprecation warning.
"""

from __future__ import annotations

import enum
import threading
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .logging_ops import logger as log
from .strategy_constants import DEFAULT_CONSTANTS, StrategyConstants

__all__ = [
    "AlignmentStrategy",
    "AlignmentTier",
    "LocusProfile",
    "OVERRIDE_KEYS",
    "TIER_MAFFT_ARGS",
    "WorkTerms",
    "determine_alignment_args",
    "determine_alignment_strategy",
    "estimate_work",
    "estimate_work_terms",
    "normalize_alphabet",
]


class AlignmentTier(enum.IntEnum):
    """Algorithm tiers ordered by increasing speed (and decreasing accuracy)."""

    ACCURATE = 0
    FAST_TREE = 1
    HIGH_SCALE = 2
    EXTREME_SCALE = 3


#: MAFFT arguments (without ``--thread`` and file names) of every tier.
TIER_MAFFT_ARGS: dict[AlignmentTier, tuple[str, ...]] = {
    AlignmentTier.ACCURATE: ("--auto",),
    AlignmentTier.FAST_TREE: ("--parttree", "--retree", "2", "--maxiterate", "0"),
    AlignmentTier.HIGH_SCALE: (
        "--parttree",
        "--retree",
        "2",
        "--maxiterate",
        "0",
        "--nofft",
    ),
    AlignmentTier.EXTREME_SCALE: (
        "--parttree",
        "--retree",
        "1",
        "--maxiterate",
        "0",
        "--nofft",
    ),
}

#: Override keys understood by :func:`determine_alignment_strategy`.
OVERRIDE_KEYS: frozenset[str] = frozenset(
    {
        "tier",
        "work_fast_tree_threshold",
        "work_high_scale_threshold",
        "work_extreme_scale_threshold",
        "work_parttree_threshold",
        "work_retree1_threshold",
        "n_force_parttree",
        "parttree_threshold",
        "retree1_threshold",
    }
)

_DEPRECATED_KEYS: dict[str, str] = {
    "parttree_threshold": (
        "'parttree_threshold' (N-count gate) is deprecated and ignored unless set "
        "explicitly; the strategy is now chosen from the work model W = N*L^alpha "
        "+ beta*N^2*L. Use 'work_fast_tree_threshold' instead."
    ),
    "retree1_threshold": (
        "'retree1_threshold' (N-count gate) is deprecated and ignored unless set "
        "explicitly; use 'work_extreme_scale_threshold' instead."
    ),
}
_warned_lock = threading.Lock()
_warned_keys: set[str] = set()

_NUCL_NAMES = frozenset({"nucl", "nuc", "nucleotide", "dna", "rna"})
_PROT_NAMES = frozenset({"prot", "protein", "aa", "amino"})


def normalize_alphabet(alphabet: str) -> str:
    """Return ``"nucl"`` or ``"prot"`` for a user-facing alphabet name."""
    key = str(alphabet).strip().lower()
    if key in _NUCL_NAMES:
        return "nucl"
    if key in _PROT_NAMES:
        return "prot"
    raise ValueError(f"unknown alphabet {alphabet!r}; expected 'nucl' or 'prot'")


class WorkTerms:
    """The two terms of the work model (``total`` is their sum)."""

    __slots__ = ("pairwise", "primary")

    def __init__(self, primary: float, pairwise: float) -> None:
        self.primary = primary
        self.pairwise = pairwise

    @property
    def total(self) -> float:
        return self.primary + self.pairwise


def estimate_work_terms(
    n_seq: int,
    max_len: int,
    *,
    alphabet: str = "nucl",
    constants: StrategyConstants | None = None,
) -> WorkTerms:
    """Both terms of ``W`` (all zero for an empty or zero-length locus)."""
    if n_seq <= 0 or max_len <= 0:
        return WorkTerms(0.0, 0.0)
    model = (constants or DEFAULT_CONSTANTS).model_for(normalize_alphabet(alphabet))
    n = float(n_seq)
    length = float(max_len)
    primary = model.scale * n * length ** model.alpha_for(max_len)
    pairwise = model.beta * n * n * length
    return WorkTerms(primary, pairwise)


def estimate_work(
    n_seq: int,
    max_len: int,
    *,
    alphabet: str = "nucl",
    constants: StrategyConstants | None = None,
) -> float:
    """Alignment work ``W = scale*N*L^alpha + beta*N^2*L`` of a locus."""
    return estimate_work_terms(
        n_seq, max_len, alphabet=alphabet, constants=constants
    ).total


@dataclass(frozen=True)
class LocusProfile:
    """The data-dependent inputs of the strategy for one locus."""

    n_seq: int
    min_len: int | None
    max_len: int
    alphabet: str = "nucl"

    def work(self, constants: StrategyConstants | None = None) -> float:
        """``W`` of this locus."""
        return estimate_work(
            self.n_seq, self.max_len, alphabet=self.alphabet, constants=constants
        )

    def strategy(
        self,
        overrides: Mapping[str, Any] | None = None,
        constants: StrategyConstants | None = None,
    ) -> AlignmentStrategy:
        """Strategy for this locus."""
        return determine_alignment_strategy(
            self.n_seq,
            self.max_len,
            self.min_len,
            alphabet=self.alphabet,
            overrides=overrides,
            constants=constants,
        )


@dataclass(frozen=True)
class AlignmentStrategy:
    """Result of the strategy decision.

    Attributes:
        tier: Chosen tier.
        mafft_args: MAFFT algorithm arguments (no thread/file arguments).
        work: The work estimate ``W``.
        reason: Human-readable justification (also stored in the manifest).
        needs_fragment_split: ``True`` when a PartTree tier was chosen but
            sequences shorter than ``min_parttree_seq_len`` exist; those must
            be added with ``--addfragments`` rather than aligned by PartTree.
    """

    tier: AlignmentTier
    mafft_args: tuple[str, ...]
    work: float
    reason: str
    needs_fragment_split: bool = False

    @property
    def uses_parttree(self) -> bool:
        return "--parttree" in self.mafft_args

    def to_dict(self) -> dict[str, Any]:
        """JSON-serializable summary for the run manifest."""
        return {
            "tier": self.tier.name,
            "mafft_args": list(self.mafft_args),
            "work": self.work,
            "reason": self.reason,
            "needs_fragment_split": self.needs_fragment_split,
        }


def _warn_deprecated_once(key: str) -> None:
    with _warned_lock:
        if key in _warned_keys:
            return
        _warned_keys.add(key)
    message = _DEPRECATED_KEYS[key]
    log.warning(message)
    warnings.warn(message, DeprecationWarning, stacklevel=4)


def _reset_deprecation_warnings() -> None:
    """Forget which deprecation warnings were emitted (for tests)."""
    with _warned_lock:
        _warned_keys.clear()


def _coerce_tier(value: Any) -> AlignmentTier:
    if isinstance(value, AlignmentTier):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return AlignmentTier(value)
    key = str(value).strip().upper().replace("-", "_")
    try:
        return AlignmentTier[key]
    except KeyError:
        names = ", ".join(t.name.lower() for t in AlignmentTier)
        raise ValueError(f"unknown tier {value!r}; expected one of: {names}") from None


def _number(overrides: Mapping[str, Any], key: str, default: float) -> float:
    value = overrides.get(key)
    return default if value is None else float(value)


def determine_alignment_strategy(
    n_seq: int,
    max_len: int,
    min_len: int | None = None,
    *,
    alphabet: str = "nucl",
    overrides: Mapping[str, Any] | None = None,
    constants: StrategyConstants | None = None,
) -> AlignmentStrategy:
    """Choose the alignment tier for a locus.

    Pure and deterministic: the result depends only on the arguments and on
    ``constants`` (default :data:`DEFAULT_CONSTANTS`), never on the host.

    Args:
        n_seq: Number of sequences in the locus.
        max_len: Residue count of the longest sequence.
        min_len: Residue count of the shortest sequence, if known. Only used
            to set ``needs_fragment_split``.
        alphabet: ``"nucl"`` or ``"prot"``.
        overrides: Explicit user overrides, see the module docstring.
        constants: Model constants (tests/calibration only).

    Returns:
        The :class:`AlignmentStrategy`.

    Raises:
        ValueError: On an unknown alphabet, tier or override key.
    """
    const = constants or DEFAULT_CONSTANTS
    alpha_name = normalize_alphabet(alphabet)
    ov: Mapping[str, Any] = overrides or {}
    unknown = sorted(set(ov) - OVERRIDE_KEYS)
    if unknown:
        raise ValueError(f"unknown strategy override(s): {', '.join(unknown)}")
    for key in sorted(_DEPRECATED_KEYS):
        if ov.get(key) is not None:
            _warn_deprecated_once(key)

    terms = estimate_work_terms(n_seq, max_len, alphabet=alpha_name, constants=const)
    work = terms.total

    # Tier boundaries, with the pre-v3 key names as aliases. Overrides may break
    # the ordering; the extreme-scale boundary is lifted to the first one and the
    # middle one is clamped between them, so explicitly given values win.
    b_fast = _number(
        ov,
        "work_fast_tree_threshold",
        _number(ov, "work_parttree_threshold", const.work_fast_tree),
    )
    b_extreme = max(
        _number(
            ov,
            "work_extreme_scale_threshold",
            _number(ov, "work_retree1_threshold", const.work_extreme_scale),
        ),
        b_fast,
    )
    b_high = min(
        max(_number(ov, "work_high_scale_threshold", const.work_high_scale), b_fast),
        b_extreme,
    )
    n_cap = int(_number(ov, "n_force_parttree", float(const.n_force_parttree)))

    forced = ov.get("tier")
    tier: AlignmentTier
    if forced is not None:
        tier = _coerce_tier(forced)
        reason = f"tier {tier.name} forced by override"
    else:
        tier = AlignmentTier(
            int(work >= b_fast) + int(work >= b_high) + int(work >= b_extreme)
        )
        if tier is AlignmentTier.ACCURATE:
            reason = f"W={work:.3e} < {b_fast:.3e}"
        elif tier is AlignmentTier.EXTREME_SCALE:
            reason = f"W={work:.3e} >= {b_extreme:.3e}"
        else:
            lo = (b_fast, b_high)[tier - 1]
            hi = (b_high, b_extreme)[tier - 1]
            reason = f"{lo:.3e} <= W={work:.3e} < {hi:.3e}"
        reason += (
            f" (N={n_seq}, Lmax={max_len}, {alpha_name}; progressive "
            f"{terms.primary:.3e} + pairwise {terms.pairwise:.3e})"
        )

        # Global Scale Guard: For N >= 10,000 taxa, force EXTREME_SCALE (--retree 1) unless overridden
        if (
            n_seq >= 10000
            and tier < AlignmentTier.EXTREME_SCALE
            and "n_force_parttree" not in ov
        ):
            tier = AlignmentTier.EXTREME_SCALE
            reason = f"N={n_seq} >= 10000 (forced EXTREME_SCALE); " + reason

        # Legacy explicit N-count gates (deprecated, honoured only when given).
        legacy_pt = ov.get("parttree_threshold")
        legacy_r1 = ov.get("retree1_threshold")
        if legacy_r1 is not None and n_seq >= int(legacy_r1):
            if tier < AlignmentTier.EXTREME_SCALE:
                tier = AlignmentTier.EXTREME_SCALE
                reason = (
                    f"N={n_seq} >= legacy retree1_threshold={int(legacy_r1)}; " + reason
                )
        elif legacy_pt is not None and n_seq >= int(legacy_pt):
            if tier < AlignmentTier.HIGH_SCALE:
                tier = AlignmentTier.HIGH_SCALE
                reason = (
                    f"N={n_seq} >= legacy parttree_threshold={int(legacy_pt)}; "
                    + reason
                )

        # Fixed reference cap: O(N^2) distance matrix memory safety.
        if n_seq >= n_cap and tier < AlignmentTier.FAST_TREE:
            tier = AlignmentTier.FAST_TREE
            reason = (
                f"N={n_seq} >= N_FORCE_PARTTREE={n_cap} (O(N^2) distance matrix); "
                + reason
            )

    args = TIER_MAFFT_ARGS[tier]
    needs_split = (
        tier is not AlignmentTier.ACCURATE
        and min_len is not None
        and min_len < const.min_parttree_seq_len
    )
    if needs_split:
        reason += (
            f"; shortest sequence has {min_len} residues "
            f"(< {const.min_parttree_seq_len}): short sequences are added with "
            "--addfragments"
        )
    return AlignmentStrategy(
        tier=tier,
        mafft_args=args,
        work=work,
        reason=reason,
        needs_fragment_split=needs_split,
    )


def determine_alignment_args(
    n_seq: int,
    max_len: int,
    min_len: int | None = None,
    *,
    alphabet: str = "nucl",
    overrides: Mapping[str, Any] | None = None,
) -> list[str]:
    """MAFFT algorithm arguments for a locus (blueprint-compatible wrapper)."""
    return list(
        determine_alignment_strategy(
            n_seq, max_len, min_len, alphabet=alphabet, overrides=overrides
        ).mafft_args
    )
