"""Fast-path GenBank flatfile scanner.

High-throughput, low-allocation scanner for GenBank flatfiles.
Extracts record metadata (id, name, description), nucleotide sequence,
and target genomic features (CDS, gene, tRNA, rRNA) with minimal object
instantiation and zero-copy string slicing.

Robustness
----------
Valid input is parsed exactly as before: the strict fast path first, then
(transparently) BioPython's ``SeqIO.read`` if the fast path rejects the file.
Only when *both* parsers fail is the input treated as damaged:

* the fast scanner is re-run in *lenient* mode, in which a feature whose
  location cannot be parsed is skipped instead of failing the whole file. The
  number of skipped features is recorded in ``record.annotations`` as
  ``pba_skipped_features`` (with up to ``MAX_SKIP_DETAILS`` descriptions, one per
  line, in ``pba_skipped_feature_details``); the keys are absent for clean records;
* if that does not help either (no ``LOCUS``/``FEATURES``/``ORIGIN``, empty or
  unreadable file, ...) :class:`GenBankParseError` is raised with the path and a
  one-line reason.

Pathological input is bounded instead of being allowed to hang or blow up
(documented module constants, far above anything a plastome needs):

* ``MAX_FILE_BYTES``: larger files are rejected (:class:`GenBankParseError`,
  stage ``read``);
* ``MAX_LOCATION_PARTS`` / ``MAX_LOCATION_CHARS``: a feature location with more
  parts or characters is skipped (and BioPython's unbounded location parser is
  not consulted for it);
* ``MAX_QUALIFIER_CHARS``: longer qualifier values are truncated (counted in
  ``pba_truncated_qualifiers``).

Further annotation keys (present only when non-zero): ``pba_extra_records``
(additional records of a multi-record file, which are ignored as before).
"""

from __future__ import annotations

import logging
import os
from os import PathLike
from typing import TextIO

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import ExactPosition, Location, SeqFeature, SimpleLocation
from Bio.SeqRecord import SeqRecord

log = logging.getLogger(__name__)

_DEFAULT_TARGET_FEATURES = frozenset({"CDS", "gene", "tRNA", "rRNA"})
_STRIP_SEQ_TRANS = str.maketrans("", "", " 0123456789\r\n\t")

#: Files larger than this are rejected (a plastome flatfile is about 0.4 MB).
MAX_FILE_BYTES = 128 * 1024 * 1024
#: Maximum number of parts (``..`` ranges) of one feature location.
MAX_LOCATION_PARTS = 1000
#: Maximum length of one feature location string.
MAX_LOCATION_CHARS = 100_000
#: Maximum length of one qualifier value; the rest is dropped.
MAX_QUALIFIER_CHARS = 1_000_000
#: Maximum number of skipped-feature descriptions kept in the record annotations.
MAX_SKIP_DETAILS = 50


class GenBankParseError(ValueError):
    """The whole file is unusable as a GenBank record.

    Attributes:
        path: The file (or stream name).
        reason: One-line explanation.
        stage: ``"read"`` if the file could not be read at all (missing,
            directory, permissions, too large), else ``"parse"``.
    """

    def __init__(self, path: str, reason: str, *, stage: str = "parse") -> None:
        self.path = str(path)
        self.reason = reason
        self.stage = stage
        super().__init__(f"{self.path}: {reason}")

    def __reduce__(self):  # keeps path/reason/stage across process boundaries
        return (_rebuild_parse_error, (self.path, self.reason, self.stage))


def _rebuild_parse_error(path: str, reason: str, stage: str) -> GenBankParseError:
    return GenBankParseError(path, reason, stage=stage)


class _LimitExceeded(ValueError):
    """A documented parser limit was hit (feature skipped, not retried elsewhere)."""


def _describe(source: str | PathLike | TextIO) -> str:
    if isinstance(source, (str, PathLike)):
        return str(os.fspath(source))
    return str(getattr(source, "name", "<stream>"))


def _has_genbank_section(text: str, keyword: str) -> bool:
    """True when a record line begins with ``keyword`` (``FEATURES``, ``ORIGIN``)."""
    for line in text.splitlines():
        if line.startswith(keyword) and (
            len(line) == len(keyword) or line[len(keyword)] in " \t"
        ):
            return True
    return False


def _short(exc: BaseException, limit: int = 160) -> str:
    text = " ".join(f"{type(exc).__name__}: {exc}".split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def fast_read_genbank(
    source: str | PathLike | TextIO,
    target_types: frozenset[str] | set[str] | None = _DEFAULT_TARGET_FEATURES,
) -> SeqRecord:
    """Fast scanner reading a single GenBank record.

    Extracts record ID, locus name, definition, sequence, and requested
    features. If parsing fails or non-standard formatting is encountered,
    transparently falls back to BioPython's `SeqIO.read()`. If that fails too,
    damaged features are skipped where possible (see the module docstring).

    Args:
        source: File path or open text file handle.
        target_types: Feature types to extract. If None, extracts all features.

    Returns:
        A populated Bio.SeqRecord.SeqRecord.

    Raises:
        GenBankParseError: The whole file is unusable (unreadable, empty, no
            record structure, ...).
    """
    name = _describe(source)
    try:
        text = _read_content(source)
    except GenBankParseError:
        raise
    except Exception as exc:
        detail = (
            exc.strerror
            if isinstance(exc, OSError) and exc.strerror
            else f"{type(exc).__name__}: {exc}"
        )
        raise GenBankParseError(
            name, f"cannot read file ({detail})", stage="read"
        ) from exc
    if not text.strip():
        raise GenBankParseError(
            name, "file is empty" if not text else "file contains only whitespace"
        )

    try:
        return _scan_genbank_block(text, target_types)
    except Exception as exc:
        fast_exc = exc
        log.debug("FastGenBankScanner fallback on %s: %s", name, exc)

    bio_exc: Exception | None = None
    if not isinstance(fast_exc, _LimitExceeded):
        if hasattr(source, "seek"):
            try:
                source.seek(0)
            except Exception:
                pass
        try:
            record = SeqIO.read(source, "genbank")
        except Exception as exc:
            bio_exc = exc
        else:
            # BioPython accepts a sequence-only record, so a flatfile whose
            # feature table was deleted still parses. A plastome with no
            # FEATURES or ORIGIN section is not usable (module docstring);
            # keep the fast path's rejection instead of that fallback.
            if not _has_genbank_section(text, "FEATURES") or not _has_genbank_section(
                text, "ORIGIN"
            ):
                bio_exc = ValueError("Missing FEATURES or ORIGIN section")
            else:
                return record

    # Both parsers failed: keep whatever can be read, feature by feature.
    try:
        return _scan_genbank_block(text, target_types, skip_bad_features=True)
    except Exception:
        reason = _short(fast_exc)
        if bio_exc is not None:
            reason += f"; BioPython: {_short(bio_exc)}"
        raise GenBankParseError(name, reason) from (bio_exc or fast_exc)


def _read_content(source: str | PathLike | TextIO) -> str:
    if isinstance(source, (str, PathLike)):
        size = os.path.getsize(source)
        if size > MAX_FILE_BYTES:
            raise GenBankParseError(
                str(os.fspath(source)),
                f"file is {size} bytes, larger than the {MAX_FILE_BYTES} byte limit",
                stage="read",
            )
        with open(source, "r", encoding="latin-1", errors="replace") as f:
            return f.read()
    content = source.read()
    if len(content) > MAX_FILE_BYTES:
        raise GenBankParseError(
            _describe(source),
            f"input is {len(content)} characters, larger than the limit",
            stage="read",
        )
    return content


def _parse_location(loc_str: str, seq_len: int) -> Location:
    """Fast-path location parsing for simple, complement, and compound features."""
    if loc_str.count(",") >= MAX_LOCATION_PARTS:
        raise _LimitExceeded(f"location has more than {MAX_LOCATION_PARTS} parts")
    if ".." in loc_str and "(" not in loc_str:
        start_str, end_str = loc_str.split("..", 1)
        if start_str.isdigit() and end_str.isdigit():
            return SimpleLocation(
                ExactPosition(int(start_str) - 1),
                ExactPosition(int(end_str)),
                strand=1,
            )
    elif (
        loc_str.startswith("complement(")
        and loc_str.endswith(")")
        and "join" not in loc_str
    ):
        inner = loc_str[11:-1]
        if ".." in inner:
            start_str, end_str = inner.split("..", 1)
            if start_str.isdigit() and end_str.isdigit():
                return SimpleLocation(
                    ExactPosition(int(start_str) - 1),
                    ExactPosition(int(end_str)),
                    strand=-1,
                )
    return Location.fromstring(loc_str, seq_len)


def _scan_genbank_block(
    text: str,
    target_types: frozenset[str] | set[str] | None,
    skip_bad_features: bool = False,
) -> SeqRecord:
    """Scans the first record of ``text``.

    With ``skip_bad_features`` a feature whose location cannot be parsed is
    dropped (and counted) instead of raising.
    """
    if not text.startswith("LOCUS"):
        # Not a standard GenBank file; trigger fallback
        raise ValueError("File does not start with LOCUS")

    feat_idx = text.find("\nFEATURES ")
    orig_idx = text.find("\nORIGIN", feat_idx if feat_idx != -1 else 0)
    end_idx = text.find("\n//", orig_idx if orig_idx != -1 else 0)

    if feat_idx == -1 or orig_idx == -1:
        raise ValueError("Missing FEATURES or ORIGIN section")
    if end_idx == -1:
        end_idx = len(text)

    # 1. Header parsing
    rec_name = ""
    rec_id = ""
    description = ""
    topology: str | None = None
    def_parts: list[str] = []
    in_def = False

    header_block = text[:feat_idx]
    for line in header_block.splitlines():
        if line.startswith("LOCUS"):
            parts = line.split()
            if len(parts) >= 2:
                rec_name = parts[1]
                rec_id = rec_name
            # GenBank LOCUS: ... bp  mol_type  topology  division  date
            lowered = {p.lower() for p in parts}
            if "circular" in lowered:
                topology = "circular"
            elif "linear" in lowered:
                topology = "linear"
            in_def = False
        elif line.startswith("DEFINITION"):
            def_parts.append(line[10:].strip())
            in_def = True
        elif line.startswith("ACCESSION"):
            parts = line.split()
            if len(parts) >= 2 and (not rec_id or rec_id == rec_name):
                rec_id = parts[1]
            in_def = False
        elif line.startswith("VERSION"):
            parts = line.split()
            if len(parts) >= 2:
                rec_id = parts[1]
            in_def = False
        elif in_def:
            if line.startswith("   ") or line.startswith("\t"):
                def_parts.append(line.strip())
            else:
                in_def = False

    if def_parts:
        description = " ".join(def_parts).strip()
        if description.endswith("."):
            description = description[:-1]

    # 2. Sequence translation
    seq_start = text.find("\n", orig_idx + 1)
    if seq_start == -1:
        seq_start = orig_idx + 7
    seq_str = text[seq_start:end_idx].translate(_STRIP_SEQ_TRANS).upper()
    seq_len = len(seq_str)

    # 3. Features extraction
    features: list[SeqFeature] = []
    skipped: list[str] = []
    n_skipped = 0
    truncated_quals = 0
    feat_text = text[feat_idx:orig_idx]
    feat_lines = feat_text.splitlines()

    curr_type: str | None = None
    curr_loc_str: str = ""
    loc_overflow = False
    curr_quals: dict[str, list[str]] = {}
    curr_qual_key: str | None = None
    curr_qual_val_parts: list[str] = []
    qual_chars = 0
    in_quoted_qualifier = False
    keep_curr_feature = False

    def clip(value: str) -> str:
        nonlocal truncated_quals
        if len(value) > MAX_QUALIFIER_CHARS:
            truncated_quals += 1
            return value[:MAX_QUALIFIER_CHARS]
        return value

    def add_qual_part(part: str) -> None:
        nonlocal qual_chars, truncated_quals
        if qual_chars > MAX_QUALIFIER_CHARS:
            return
        qual_chars += len(part)
        if qual_chars > MAX_QUALIFIER_CHARS:
            truncated_quals += 1
            return
        curr_qual_val_parts.append(part)

    def flush_qual():
        nonlocal curr_qual_key, curr_qual_val_parts, in_quoted_qualifier, qual_chars
        if curr_qual_key is not None:
            val = " ".join(curr_qual_val_parts).strip()
            if val.startswith('"') and val.endswith('"') and len(val) >= 2:
                val = val[1:-1]
            elif val.startswith('"'):
                val = val[1:]
            elif val.endswith('"'):
                val = val[:-1]
            curr_quals.setdefault(curr_qual_key, []).append(val)
            curr_qual_key = None
            curr_qual_val_parts = []
            qual_chars = 0
            in_quoted_qualifier = False

    def flush_feat():
        nonlocal curr_type, curr_loc_str, curr_quals, keep_curr_feature
        nonlocal loc_overflow, n_skipped
        flush_qual()
        if curr_type is not None and keep_curr_feature:
            try:
                if loc_overflow:
                    raise _LimitExceeded(
                        f"location longer than {MAX_LOCATION_CHARS} characters"
                    )
                loc = _parse_location(curr_loc_str, seq_len)
            except Exception as exc:
                if not skip_bad_features:
                    raise
                n_skipped += 1
                if len(skipped) < MAX_SKIP_DETAILS:
                    shown = (
                        curr_loc_str
                        if len(curr_loc_str) <= 60
                        else (curr_loc_str[:57] + "...")
                    )
                    skipped.append(f"{curr_type} {shown}: {_short(exc, 120)}")
            else:
                features.append(
                    SeqFeature(location=loc, type=curr_type, qualifiers=curr_quals)
                )
        curr_type = None
        curr_loc_str = ""
        loc_overflow = False
        curr_quals = {}
        keep_curr_feature = False

    for line in feat_lines[1:]:
        if line.startswith("     ") and len(line) > 5 and line[5] != " ":
            flush_feat()
            feature_type = line[5:21].strip()
            keep_curr_feature = target_types is None or feature_type in target_types
            curr_type = feature_type
            first_loc = line[21:].strip()
            if len(first_loc) > MAX_LOCATION_CHARS:
                loc_overflow = True
            else:
                curr_loc_str = first_loc
            continue

        if not keep_curr_feature:
            continue

        stripped = line[21:].strip() if len(line) >= 21 else line.strip()
        if not stripped:
            continue

        if in_quoted_qualifier:
            if stripped.endswith('"') and not stripped.endswith('\\"'):
                add_qual_part(stripped[:-1])
                flush_qual()
            else:
                add_qual_part(stripped)
            continue

        if stripped.startswith("/"):
            flush_qual()
            if "=" in stripped:
                k, v = stripped[1:].split("=", 1)
                k = k.strip()
                v = v.strip()
                if v.startswith('"'):
                    if v.endswith('"') and len(v) > 1:
                        curr_quals.setdefault(k, []).append(clip(v[1:-1]))
                    else:
                        curr_qual_key = k
                        curr_qual_val_parts = [clip(v[1:])]
                        qual_chars = len(curr_qual_val_parts[0])
                        in_quoted_qualifier = True
                else:
                    curr_quals.setdefault(k, []).append(clip(v))
            else:
                curr_quals.setdefault(stripped[1:].strip(), []).append("")
        elif curr_qual_key is None:
            if len(curr_loc_str) + len(stripped) > MAX_LOCATION_CHARS:
                loc_overflow = True
            else:
                curr_loc_str += stripped

    flush_feat()

    annotations: dict[str, str | int] = {}
    if topology is not None:
        annotations["topology"] = topology
    if n_skipped:
        annotations["pba_skipped_features"] = n_skipped
        annotations["pba_skipped_feature_details"] = "\n".join(skipped)
    if truncated_quals:
        annotations["pba_truncated_qualifiers"] = truncated_quals
    extra_records = text.count("\nLOCUS ", end_idx)
    if extra_records:
        annotations["pba_extra_records"] = extra_records

    return SeqRecord(
        Seq(seq_str),
        id=rec_id,
        name=rec_name,
        description=description,
        features=features,
        annotations=annotations,
    )
