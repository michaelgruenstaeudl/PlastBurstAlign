"""Structured record of what GenBank extraction could and could not read.

Extraction is isolated at four levels: a *feature* that cannot be parsed or
extracted is skipped, a *file* (record) that cannot be read is skipped, a
*worker process* that dies takes only its chunk down (which is then re-run one
file at a time), and only a run in which **no** file yields any region is fatal.
Everything that was skipped is recorded in an :class:`ExtractionReport` that is
exposed as ``PlastidData.extraction_report`` after ``ExtractAndCollect.extract()``.

Stages used in the reports (``FileFailure.stage`` / ``FeatureFailure.stage``):

``read``
    the file could not be opened/read (missing, directory, permissions, too large)
``parse``
    the file is not a parseable GenBank record (both the fast scanner and
    BioPython failed)
``record``
    the file parsed but its record is unusable (for example no sequence)
``extract``
    the extractor failed for the record as a whole (features already extracted
    from that record are kept)
``worker``
    the worker process died (crash, out-of-memory kill) or raised unexpectedly
``scan``
    (feature level) the scanner dropped a feature whose location it could not parse
``feature:<mode>``
    (feature level) the extractor dropped a feature in ``cds``/``igs``/``int``/``rna``
    mode (exception, invalid nucleotide characters, unusable gene name, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# The lists are bounded so that a run over thousands of corrupt files cannot
# blow up memory; the counters keep counting beyond the cap.
MAX_LISTED_FILE_FAILURES = 5000
MAX_LISTED_FILE_WARNINGS = 5000
MAX_LISTED_FEATURE_FAILURES = 2000
MAX_REASON_CHARS = 300


def clip_reason(reason: str) -> str:
    """Single line, bounded length."""
    text = " ".join(str(reason).split())
    if len(text) > MAX_REASON_CHARS:
        text = text[: MAX_REASON_CHARS - 3] + "..."
    return text


@dataclass(frozen=True)
class FileFailure:
    """A file that yielded no (or only partial) results."""

    file: str
    stage: str
    reason: str
    file_index: int = -1


@dataclass(frozen=True)
class FeatureFailure:
    """A single feature that was skipped."""

    file: str
    stage: str
    feature: str
    reason: str


@dataclass(frozen=True)
class FileWarning:
    """A file that was processed but with a caveat (empty result, extra records)."""

    file: str
    message: str


@dataclass
class ExtractionReport:
    """What happened to the input files during extraction.

    Attributes:
        files_total: Files that were attempted.
        files_ok: Files read, parsed and extracted without a file-level failure
            (individual features of such a file may still have been skipped).
        files_with_regions: Files that contributed at least one extracted region.
        file_failures: Files that failed, with stage and reason (capped).
        file_warnings: Non-fatal caveats per file (capped).
        features_skipped: Number of skipped features (all, also beyond the cap).
        feature_failures: Skipped features with stage and reason (capped).
        stage_counts: Failures per stage (files and features).
        chunks_retried: Chunks that had to be re-run to isolate a failure.
        worker_crashes: Worker pools that broke (process died).
    """

    files_total: int = 0
    files_ok: int = 0
    files_with_regions: int = 0
    file_failures: list[FileFailure] = field(default_factory=list)
    file_warnings: list[FileWarning] = field(default_factory=list)
    features_skipped: int = 0
    feature_failures: list[FeatureFailure] = field(default_factory=list)
    stage_counts: dict[str, int] = field(default_factory=dict)
    chunks_retried: int = 0
    worker_crashes: int = 0
    # counters beyond the capped lists
    file_failures_total: int = 0
    file_warnings_total: int = 0

    # -- recording ---------------------------------------------------------------

    def add_file_failure(
        self, file: str, stage: str, reason: str, file_index: int = -1
    ) -> None:
        self.file_failures_total += 1
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        if len(self.file_failures) < MAX_LISTED_FILE_FAILURES:
            self.file_failures.append(
                FileFailure(str(file), stage, clip_reason(reason), file_index)
            )

    def add_feature_failure(
        self, file: str, stage: str, feature: str, reason: str
    ) -> None:
        self.features_skipped += 1
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + 1
        if len(self.feature_failures) < MAX_LISTED_FEATURE_FAILURES:
            self.feature_failures.append(
                FeatureFailure(
                    str(file), stage, clip_reason(feature), clip_reason(reason)
                )
            )

    def add_warning(self, file: str, message: str) -> None:
        self.file_warnings_total += 1
        if len(self.file_warnings) < MAX_LISTED_FILE_WARNINGS:
            self.file_warnings.append(FileWarning(str(file), clip_reason(message)))

    def merge(self, other: ExtractionReport) -> None:
        """Adds ``other`` (chunk reports are merged in submission order)."""
        self.files_total += other.files_total
        self.files_ok += other.files_ok
        self.files_with_regions += other.files_with_regions
        self.features_skipped += other.features_skipped
        self.chunks_retried += other.chunks_retried
        self.worker_crashes += other.worker_crashes
        self.file_failures_total += other.file_failures_total
        self.file_warnings_total += other.file_warnings_total
        for stage, n in other.stage_counts.items():
            self.stage_counts[stage] = self.stage_counts.get(stage, 0) + n
        room = MAX_LISTED_FILE_FAILURES - len(self.file_failures)
        self.file_failures.extend(other.file_failures[: max(room, 0)])
        room = MAX_LISTED_FILE_WARNINGS - len(self.file_warnings)
        self.file_warnings.extend(other.file_warnings[: max(room, 0)])
        room = MAX_LISTED_FEATURE_FAILURES - len(self.feature_failures)
        self.feature_failures.extend(other.feature_failures[: max(room, 0)])

    # -- queries -----------------------------------------------------------------

    @property
    def files_failed(self) -> int:
        return self.file_failures_total

    @property
    def ok(self) -> bool:
        """True if nothing at all was skipped or failed."""
        return not (
            self.file_failures_total
            or self.features_skipped
            or self.worker_crashes
            or self.chunks_retried
        )

    @property
    def failed_files(self) -> list[str]:
        """Paths of the failed files, in input order."""
        return [f.file for f in self.file_failures]

    def failure_for(self, file: str) -> FileFailure | None:
        """The failure of ``file`` (full path or base name), if any."""
        for failure in self.file_failures:
            if failure.file == file or failure.file.endswith("/" + file):
                return failure
        return None

    def summary(self, max_items: int = 10) -> str:
        """A concise multi-line, human readable summary."""
        lines = [
            f"extraction report: {self.files_ok}/{self.files_total} file(s) read"
            f" without error, {self.files_failed} failed,"
            f" {self.features_skipped} feature(s) skipped"
        ]
        for failure in self.file_failures[:max_items]:
            name = failure.file.rsplit("/", 1)[-1]
            lines.append(f"  - {name} [{failure.stage}]: {failure.reason}")
        if self.file_failures_total > max_items:
            lines.append(f"  - ... and {self.file_failures_total - max_items} more")
        if self.features_skipped:
            by_stage = ", ".join(
                f"{stage}={n}" for stage, n in sorted(self.stage_counts.items())
            )
            lines.append(f"  stages: {by_stage}")
            for ff in self.feature_failures[: min(max_items, 5)]:
                name = ff.file.rsplit("/", 1)[-1]
                lines.append(f"    * {name} [{ff.stage}] {ff.feature}: {ff.reason}")
        if self.worker_crashes or self.chunks_retried:
            lines.append(
                f"  worker crashes: {self.worker_crashes}; chunks re-run file by"
                f" file: {self.chunks_retried}"
            )
        return "\n".join(lines)

    def add_unlisted_feature_failures(self, stage: str, count: int) -> None:
        """Count feature failures that are not stored individually.

        The scanner can skip more features than it keeps descriptions for.
        Those still have to move ``features_skipped`` and ``stage_counts``.
        """
        if count <= 0:
            return
        self.features_skipped += count
        self.stage_counts[stage] = self.stage_counts.get(stage, 0) + count

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable form (for the run manifest)."""
        return {
            "files_total": self.files_total,
            "files_ok": self.files_ok,
            "files_failed": self.files_failed,
            "files_with_regions": self.files_with_regions,
            "features_skipped": self.features_skipped,
            "stage_counts": dict(sorted(self.stage_counts.items())),
            "chunks_retried": self.chunks_retried,
            "worker_crashes": self.worker_crashes,
            "file_failures": [
                {"file": f.file, "stage": f.stage, "reason": f.reason}
                for f in self.file_failures
            ],
            "file_warnings": [
                {"file": w.file, "message": w.message} for w in self.file_warnings
            ],
            "feature_failures": [
                {
                    "file": f.file,
                    "stage": f.stage,
                    "feature": f.feature,
                    "reason": f.reason,
                }
                for f in self.feature_failures
            ],
        }


class ExtractionError(RuntimeError):
    """No input file yielded a usable region.

    The run report is attached so callers can see which files failed and why
    without parsing the message.

    Attributes:
        report: The :class:`ExtractionReport` for the run (possibly partial).
    """

    def __init__(self, message: str, report: ExtractionReport | None = None) -> None:
        super().__init__(message)
        self.report = report if report is not None else ExtractionReport()
