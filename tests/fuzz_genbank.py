"""Seeded GenBank corruptors for extraction fuzzing.

Each entry of :data:`ALL_CORRUPTIONS` is ``(name, fn)`` with
``fn(text, rng) -> str``. The string is Latin-1 (code points 0-255);
:func:`write_flatfile` writes those bytes. :func:`apply` picks a corruptor
by name or, when ``name`` is omitted, at random from ``rng``.

The corruptors only edit the flatfile. They do not run alignment. A later
full-pipeline fuzz agent can reuse :func:`apply` / :func:`mutate_bytes` on
synthetic plastomes from :mod:`tests.synth` and should treat extraction as
already isolated:

* feature failures are skipped and named on ``PlastidData.extraction_report``
  (stages ``scan`` and ``feature:cds|igs|int|rna``);
* a file that cannot be parsed is a file failure (stages ``read`` / ``parse``)
  and does not drop its siblings;
* :class:`plastburstalign.extraction_report.ExtractionError` is the only
  expected fatal exception, and only when every file yields nothing;
* :data:`FAILS_THE_FILE` lists corruptors that are supposed to make the whole
  flatfile unreadable. The others are feature-level or tolerated on purpose
  (CRLF, tabs, IUPAC, internal stops, duplicates, a real origin-wrapping
  join, a missing ``//``). Tolerated edits must not change records that came
  from an uncorrupted file.

Do not point these at the aligner. Locus-level fuzzing belongs in a later pass.
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from pathlib import Path

from plastburstalign.genbank_scanner import MAX_LOCATION_PARTS, MAX_QUALIFIER_CHARS

Corruptor = Callable[[str, random.Random], str]

# Whole-file failures: fast_read_genbank raises GenBankParseError.
FAILS_THE_FILE = frozenset(
    {
        "truncate",
        "missing_origin",
        "missing_features",
        "empty_file",
        "binary_nul",
    }
)


def _insert_before_origin(text: str, block: str) -> str:
    marker = "\nORIGIN"
    idx = text.find(marker)
    if idx == -1:
        return text + "\n" + block
    return text[:idx] + "\n" + block + text[idx:]


def _mutate_origin_bases(text: str, repl: str) -> str:
    """Overwrite the first origin bases, leaving the column layout in place."""
    marker = "\nORIGIN"
    idx = text.find(marker)
    if idx == -1:
        return text + repl
    head, tail = text[: idx + 1], text[idx + 1 :]
    out: list[str] = []
    k = 0
    for ch in tail:
        if k < len(repl) and ch in "acgtACGT":
            out.append(repl[k])
            k += 1
        else:
            out.append(ch)
    return head + "".join(out)


def truncate(text: str, rng: random.Random) -> str:
    """Keep a short prefix so FEATURES/ORIGIN/`//` cannot all be present."""
    del rng
    if len(text) < 8:
        return text[:1]
    return text[: max(24, len(text) // 7)]


def missing_origin(text: str, rng: random.Random) -> str:
    del rng
    idx = text.find("\nORIGIN")
    if idx == -1:
        return text.replace("ORIGIN", "DELETED_ORIGIN")
    return text[:idx] + "\n//\n"


def missing_features(text: str, rng: random.Random) -> str:
    del rng
    start = text.find("\nFEATURES ")
    origin = text.find("\nORIGIN")
    if start == -1 or origin == -1 or origin < start:
        return text.replace("FEATURES", "DELETED_FEATURES")
    return text[:start] + text[origin:]


def missing_terminator(text: str, rng: random.Random) -> str:
    """Drop the record terminator. The scanner accepts this and reads to EOF."""
    del rng
    stripped = text.rstrip()
    if stripped.endswith("//"):
        return stripped[:-2].rstrip() + "\n"
    return text.replace("//", "")


def origin_wrap_join(text: str, rng: random.Random) -> str:
    """A real origin-wrapping join, which a plastome is allowed to contain."""
    del rng
    # Length is unknown without parsing; a join that cites both ends of a
    # typical synthetic plastome still exercises the location parser.
    loc = "join(15000..15200,1..80)"
    block = (
        f"     gene            {loc}\n"
        '                     /gene="wrapGene"\n'
        f"     CDS             {loc}\n"
        '                     /gene="wrapGene"\n'
        "                     /codon_start=1\n"
        "                     /transl_table=11\n"
    )
    return _insert_before_origin(text, block)


def unbalanced_complement(text: str, rng: random.Random) -> str:
    del rng
    block = (
        "     CDS             complement(10..40\n"
        '                     /gene="badComp"\n'
        "                     /transl_table=11\n"
    )
    return _insert_before_origin(text, block)


def unbalanced_join(text: str, rng: random.Random) -> str:
    del rng
    block = (
        "     CDS             join(10..40,80..120\n"
        '                     /gene="badJoin"\n'
        "                     /transl_table=11\n"
    )
    return _insert_before_origin(text, block)


def out_of_range(text: str, rng: random.Random) -> str:
    del rng
    block = (
        "     CDS             99999999..999999999\n"
        '                     /gene="outOfRange"\n'
        "                     /transl_table=11\n"
    )
    return _insert_before_origin(text, block)


def partial_boundary(text: str, rng: random.Random) -> str:
    del rng
    block = '     CDS             10..\n                     /gene="partial"\n'
    return _insert_before_origin(text, block)


def internal_stops(text: str, rng: random.Random) -> str:
    """In-frame stops inside the origin. Table 11 translates these to ``*``."""
    del rng
    return _mutate_origin_bases(text, "TAATAGTGATAA" * 8)


def iupac_bases(text: str, rng: random.Random) -> str:
    """Ambiguous bases. They are legal nucleotide characters and must be kept."""
    del rng
    return _mutate_origin_bases(text, "RYSWKMBDHV" * 12)


def non_acgt(text: str, rng: random.Random) -> str:
    del rng
    return _mutate_origin_bases(text, "XXXX00001111")


def empty_gene(text: str, rng: random.Random) -> str:
    del rng
    return re.sub(r'/gene="[^"]*"', '/gene=""', text, count=3)


def duplicates(text: str, rng: random.Random) -> str:
    """Repeat the first CDS block so the same gene is annotated twice."""
    del rng
    marker = "\n     CDS "
    start = text.find(marker)
    if start == -1:
        return text + text[: min(len(text), 40)]
    nxt = text.find("\n     ", start + 1)
    if nxt == -1:
        return text
    return text[:nxt] + text[start:nxt] + text[nxt:]


def crlf(text: str, rng: random.Random) -> str:
    del rng
    return text.replace("\n", "\r\n")


def tabs(text: str, rng: random.Random) -> str:
    del rng
    return text.replace("                     /", "\t/", 4)


def binary_nul(text: str, rng: random.Random) -> str:
    """Break the ORIGIN marker with a NUL so the record structure is gone."""
    del rng
    idx = text.find("\nORIGIN")
    if idx == -1:
        return "\x00" + text
    return text[:idx] + "\nORI\x00GIN" + text[idx + len("\nORIGIN") :]


def unterminated_quote(text: str, rng: random.Random) -> str:
    del rng
    key = '/gene="'
    start = text.find(key)
    if start == -1:
        return text + '\n/gene="unterminated\n'
    close = text.find('"', start + len(key))
    if close == -1:
        return text
    return text[:close] + text[close + 1 :]


def huge_qualifier(text: str, rng: random.Random) -> str:
    """A qualifier longer than the scanner's truncation cap."""
    del rng
    blob = "Q" * (MAX_QUALIFIER_CHARS + 64)
    block = (
        "     CDS             1..9\n"
        '                     /gene="hugeQual"\n'
        f'                     /note="{blob}"\n'
    )
    return _insert_before_origin(text, block)


def huge_join(text: str, rng: random.Random) -> str:
    """More location parts than the scanner will parse."""
    del rng
    n_parts = MAX_LOCATION_PARTS + 2
    parts = ",".join(f"{i}..{i + 1}" for i in range(1, n_parts * 2, 2))
    block = (
        f'     CDS             join({parts})\n                     /gene="hugeJoin"\n'
    )
    return _insert_before_origin(text, block)


def empty_file(text: str, rng: random.Random) -> str:
    del text, rng
    return ""


def random_bytes(text: str, rng: random.Random) -> str:
    return mutate_bytes(text, rng, rng.randint(1, 8))


def mutate_bytes(text: str, rng: random.Random, n: int = 1) -> str:
    """Flip ``n`` bytes anywhere in the flatfile, including NULs."""
    if n < 1:
        return text
    if not text:
        return rng.choice(("", "\x00", "LOCUS"))
    data = bytearray(text.encode("latin-1", errors="replace"))
    for _ in range(n):
        data[rng.randrange(len(data))] = rng.randrange(256)
    return data.decode("latin-1")


ALL_CORRUPTIONS: list[tuple[str, Corruptor]] = [
    ("truncate", truncate),
    ("missing_origin", missing_origin),
    ("missing_features", missing_features),
    ("missing_terminator", missing_terminator),
    ("origin_wrap_join", origin_wrap_join),
    ("unbalanced_complement", unbalanced_complement),
    ("unbalanced_join", unbalanced_join),
    ("out_of_range", out_of_range),
    ("partial_boundary", partial_boundary),
    ("internal_stops", internal_stops),
    ("iupac_bases", iupac_bases),
    ("non_acgt", non_acgt),
    ("empty_gene", empty_gene),
    ("duplicates", duplicates),
    ("crlf", crlf),
    ("tabs", tabs),
    ("binary_nul", binary_nul),
    ("unterminated_quote", unterminated_quote),
    ("huge_qualifier", huge_qualifier),
    ("huge_join", huge_join),
    ("empty_file", empty_file),
    ("random_bytes", random_bytes),
]

_BY_NAME = dict(ALL_CORRUPTIONS)


def apply(text: str, rng: random.Random, name: str | None = None) -> tuple[str, str]:
    """Return ``(corruptor_name, corrupted_text)``.

    ``name=None`` selects a corruptor with ``rng``.
    """
    if name is None:
        chosen, fn = ALL_CORRUPTIONS[rng.randrange(len(ALL_CORRUPTIONS))]
    else:
        chosen = name
        try:
            fn = _BY_NAME[name]
        except KeyError:
            known = ", ".join(_BY_NAME)
            raise KeyError(f"unknown corruptor {name!r}; known: {known}") from None
    return chosen, fn(text, rng)


def write_flatfile(path: str | Path, text: str) -> None:
    """Write a corruptor result as Latin-1 bytes (NULs included)."""
    Path(path).write_bytes(text.encode("latin-1", errors="replace"))
