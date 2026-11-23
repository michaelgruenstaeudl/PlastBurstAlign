#!/usr/bin/env python3
"""PlastBench: a small, reproducible benchmark harness for PlastBurstAlign.

PlastBench compares PlastBurstAlign (PBA) with conventional pipelines on the
same GenBank input: sequential per-locus MAFFT, MUSCLE5, Clustal Omega and a
generated bash "custom shell pipeline". It measures wall time and the *peak
memory of the whole process tree* and (optionally) topology agreement.

Only the standard library and ``psutil`` are required, so the script can run on
a bare benchmark node. Subcommands::

    plastbench.py list                              # datasets and tiers
    plastbench.py fetch NAME --dest DIR             # download + sha256 + extract
    plastbench.py subsample --src DIR -n 100 --out DIR
    plastbench.py run --input DIR --out results.json
    plastbench.py report results.json [...]         # Markdown table
    plastbench.py badges results.json --out-dir badges/
    plastbench.py pack DIR --out NAME.tar.gz        # deterministic archive

How the baselines get their input
---------------------------------
Every baseline aligns *exactly the loci PBA extracts*. The input GenBank files
are scanned once by ``python -m plastburstalign`` with a no-op "aligner"
(``--alignmentcommand 'cp %INPUT% %OUTPUT%'``); the unaligned per-locus FASTA
files that PBA writes (``nucl_<locus>.unalign.fasta``) are then fed to each
baseline. This is the simplest approach that needs no second GenBank parser and
no access to PBA internals, and it is *favourable to the baselines*: their
extraction runs at PBA speed and is charged to them as phase ``extract``. The
baseline totals are ``extract + align_concat``. The baselines are sequential
(one locus at a time, one thread) by design: that is the "conventional pipeline"
being compared against. In ``cds`` mode PBA aligns protein and back-translates,
while the baselines align the nucleotide CDS directly.

Results JSON (``schema_version`` 1)
-----------------------------------
::

    {
      "schema_version": 1,
      "generator": "plastbench 1",
      "created_utc": "2026-01-01T12:00:00Z",
      "dataset": {"name": str|null, "tier": str|null, "n_genomes": int,
                  "seed": str|null, "listing_sha256": str|null,
                  "mode": str, "min_num_taxa": int, "min_seq_length": int},
      "host": {"system", "release", "machine", "python", "cpu_model",
               "cpu_count_logical", "cpu_count_physical",
               "ram_total_bytes", "ram_available_bytes", "in_ci"},
      "tool_versions": {"python": str, "plastburstalign": str|null,
                        "pba_git_head": str|null, "pba_git_dirty": "yes"|"no"|null, "mafft": str|null, ...},
      "baseline": "mafft",
      "settings": {"threads": str|int, "rss_metric": "rss"|"pss",
                   "poll_interval_s": float, "timeout_s": float|null,
                   "pba_extra_args": [str]},
      "tools": {
        "<pba|mafft|muscle|clustalo|shell>": {
          "status": "ok"|"failed"|"skipped"|"timeout",
          "reason": str|null,
          "command": [str]|null,
          "threads": str|int|null,
          "wall_seconds": float|null,
          "peak_rss_bytes": int|null,
          "phases": {"extract": {"wall_seconds", "peak_rss_bytes"},
                     "align_concat": {...}}|null,
          "n_loci": int|null, "ntax": int|null, "nchar": int|null,
          "matrix_sha256": str|null,
          "tree": {"method", "rf", "normalized_rf", "n_shared_taxa"}|null
        }
      },
      "notes": [str]
    }

``peak_rss_bytes`` is the maximum, over polling samples, of the summed RSS (or
PSS with ``--rss-metric pss`` on Linux) of the launched process and all its
descendants. Polling can miss spikes shorter than ``poll_interval_s``, and the
RSS sum double-counts pages shared between forked workers; PSS avoids that.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "datasets.json"

RESULTS_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
GENERATOR = "plastbench 1"

#: Seed of the deterministic subsampler (part of the benchmark definition).
DEFAULT_SEED = "plastbench-v1"
#: Tier name -> number of genomes.
TIERS: dict[str, int] = {"smoke": 100, "1k": 1000, "5k": 5000, "15k": 15000}

TOOLS: tuple[str, ...] = ("pba", "mafft", "muscle", "clustalo", "shell")
BASELINE_TOOLS: tuple[str, ...] = ("mafft", "muscle", "clustalo", "shell")
DEFAULT_BASELINE = "mafft"
TOOL_LABELS = {
    "pba": "PlastBurstAlign",
    "mafft": "MAFFT --auto (sequential)",
    "muscle": "MUSCLE5 -super5 (sequential)",
    "clustalo": "Clustal Omega (sequential)",
    "shell": "bash pipeline (MAFFT + awk)",
}
STATUSES = ("ok", "failed", "skipped", "timeout")
DATASET_STATUSES = ("published", "unpublished")

REQUIRED_DATASET_FIELDS = (
    "name",
    "n_genomes",
    "source",
    "zenodo_record",
    "url",
    "sha256",
    "status",
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_MD5_RE = re.compile(r"^[0-9a-f]{32}$")
_USER_AGENT = "plastbench/1 (+https://github.com/michaelgruenstaeudl/PlastBurstAlign)"

# Command used by the shell baseline for alignment (override with the env var).
SHELL_DEFAULT_ALIGNER = "mafft --auto --thread 1 --quiet"

# The bash "custom shell pipeline" baseline: a sequential per-locus MAFFT loop
# followed by an awk supermatrix concatenation, the way many labs script it.
SHELL_PIPELINE = r"""#!/usr/bin/env bash
# PlastBench baseline: "custom shell pipeline" (generated; do not edit).
# usage: shell_pipeline.sh UNALIGNED_DIR OUT_DIR
# Aligns every *.fasta in UNALIGNED_DIR one at a time, then concatenates the
# alignments into OUT_DIR/concat.aligned.fasta (missing taxa -> gaps).
# The aligner command can be overridden with PLASTBENCH_ALIGNER.
set -uo pipefail
export LC_ALL=C

IN_DIR="${1:?usage: $0 UNALIGNED_DIR OUT_DIR}"
OUT_DIR="${2:?usage: $0 UNALIGNED_DIR OUT_DIR}"
mkdir -p "$OUT_DIR/aligned"

read -r -a ALIGNER <<< "${PLASTBENCH_ALIGNER:-__DEFAULT_ALIGNER__}"

n_ok=0
n_failed=0
aligned=()
for f in "$IN_DIR"/*.fasta; do
    [ -s "$f" ] || continue
    b="${f##*/}"
    b="${b%.fasta}"
    out="$OUT_DIR/aligned/$b.aligned.fasta"
    if "${ALIGNER[@]}" "$f" > "$out" 2> /dev/null && [ -s "$out" ]; then
        aligned+=("$out")
        n_ok=$((n_ok + 1))
    else
        rm -f "$out"
        n_failed=$((n_failed + 1))
        echo "alignment failed: $f" >&2
    fi
done
echo "aligned=$n_ok failed=$n_failed" >&2
[ "$n_ok" -gt 0 ] || exit 1

awk '
function gaps(n,    g) {
    if (n in G) return G[n]
    g = "-"
    while (length(g) * 2 <= n) g = g g
    while (length(g) < n) g = g "-"
    G[n] = g
    return g
}
FNR == 1 {
    nf++
    n = split(FILENAME, parts, "/")
    b = parts[n]
    sub(/\.fasta$/, "", b)
    sub(/\.aligned$/, "", b)
    sub(/\.unalign$/, "", b)
    sub(/^nucl_/, "", b)
    locus[nf] = b
}
/^>/ {
    name = substr($1, 2)
    pre = locus[nf] "_"
    if (index(name, pre) == 1) name = substr(name, length(pre) + 1)
    if (!(name in seen)) { seen[name] = 1; order[++nt] = name }
    cur = name
    next
}
{
    gsub(/[ \t\r]/, "")
    piece[cur, nf] = piece[cur, nf] $0
    l = length(piece[cur, nf])
    if (l > L[nf]) L[nf] = l
}
END {
    for (i = 1; i <= nt; i++) {
        printf(">concat_%s\n", order[i])
        for (k = 1; k <= nf; k++) {
            s = piece[order[i], k]
            if (length(s) < L[k]) s = s gaps(L[k] - length(s))
            printf("%s", s)
        }
        printf("\n")
    }
}' "${aligned[@]}" > "$OUT_DIR/concat.aligned.fasta"
"""


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def log(message: str) -> None:
    print(f"[plastbench] {message}", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------- #
# Dataset manifest
# --------------------------------------------------------------------------- #
def load_manifest(path: Path | str = DEFAULT_MANIFEST) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        doc: dict[str, Any] = json.load(handle)
    return doc


def validate_manifest(doc: Any) -> list[str]:
    """Return a list of schema problems (empty when the manifest is valid)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["manifest must be a JSON object"]
    if doc.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        problems.append(f"schema_version must be {MANIFEST_SCHEMA_VERSION}")
    datasets = doc.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        problems.append("`datasets` must be a non-empty list")
        datasets = []
    names: set[str] = set()
    by_name: dict[str, dict[str, Any]] = {}
    for i, entry in enumerate(datasets):
        where = f"datasets[{i}]"
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object")
            continue
        for key in REQUIRED_DATASET_FIELDS:
            if key not in entry:
                problems.append(f"{where}: missing field `{key}`")
        name = entry.get("name")
        if not isinstance(name, str) or not name:
            problems.append(f"{where}: `name` must be a non-empty string")
            continue
        where = f"dataset {name!r}"
        if name in names:
            problems.append(f"{where}: duplicate name")
        names.add(name)
        by_name[name] = entry
        n = entry.get("n_genomes")
        if n is not None and (not isinstance(n, int) or isinstance(n, bool) or n < 1):
            problems.append(f"{where}: `n_genomes` must be a positive integer or null")
        status = entry.get("status")
        if status not in DATASET_STATUSES:
            problems.append(f"{where}: `status` must be one of {DATASET_STATUSES}")
        sha = entry.get("sha256")
        if sha is not None and not (isinstance(sha, str) and _SHA256_RE.match(sha)):
            problems.append(f"{where}: `sha256` must be 64 lowercase hex chars or null")
        md5 = entry.get("md5")
        if md5 is not None and not (isinstance(md5, str) and _MD5_RE.match(md5)):
            problems.append(f"{where}: `md5` must be 32 lowercase hex chars or null")
        if status == "published":
            if not isinstance(entry.get("url"), str) or not entry.get("url"):
                problems.append(f"{where}: published datasets need a `url`")
            if not entry.get("zenodo_record"):
                problems.append(f"{where}: published datasets need `zenodo_record`")
            if sha is None and md5 is None:
                problems.append(f"{where}: published datasets need a checksum")
        elif status == "unpublished":
            if sha is not None:
                problems.append(f"{where}: unpublished datasets must have sha256=null")
            if not entry.get("todo"):
                problems.append(f"{where}: unpublished datasets need a `todo` note")
    tiers = doc.get("tiers")
    if not isinstance(tiers, dict) or not tiers:
        problems.append("`tiers` must be a non-empty object")
        tiers = {}
    for tier, spec in tiers.items():
        where = f"tier {tier!r}"
        if tier not in TIERS:
            problems.append(f"{where}: unknown tier (expected one of {list(TIERS)})")
            continue
        if not isinstance(spec, dict):
            problems.append(f"{where}: must be an object")
            continue
        if spec.get("n_genomes") != TIERS[tier]:
            problems.append(f"{where}: n_genomes must be {TIERS[tier]}")
        master = by_name.get(str(spec.get("master")))
        if master is None:
            problems.append(f"{where}: `master` must name a dataset in the manifest")
            continue
        if spec.get("status") not in DATASET_STATUSES:
            problems.append(f"{where}: `status` must be one of {DATASET_STATUSES}")
        if spec.get("status") == "published":
            n_master = master.get("n_genomes")
            if master.get("status") != "published":
                problems.append(f"{where}: published tier needs a published master")
            elif not isinstance(n_master, int) or n_master < TIERS[tier]:
                problems.append(f"{where}: master has fewer than {TIERS[tier]} genomes")
        elif not spec.get("todo"):
            problems.append(f"{where}: unpublished tiers need a `todo` note")
    return problems


def find_dataset(manifest: dict[str, Any], name: str) -> dict[str, Any]:
    for entry in manifest["datasets"]:
        if entry["name"] == name:
            found: dict[str, Any] = entry
            return found
    known = ", ".join(e["name"] for e in manifest["datasets"])
    raise SystemExit(f"unknown dataset {name!r}; known datasets: {known}")


# --------------------------------------------------------------------------- #
# fetch: resumable download, checksum verification, safe extraction
# --------------------------------------------------------------------------- #
def download(url: str, dest: Path, *, timeout: float = 60.0) -> Path:
    """Download ``url`` to ``dest`` (resuming ``dest + '.part'`` when possible).

    The data is written to ``<dest>.part`` and atomically renamed on success, so
    ``dest`` never holds a truncated file. A server that ignores ``Range`` (or a
    ``file://`` URL) simply restarts the download.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    for attempt in (0, 1):
        offset = part.stat().st_size if part.exists() and attempt == 0 else 0
        headers = {"User-Agent": _USER_AGENT}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        request = urllib.request.Request(url, headers=headers)
        try:
            response = urllib.request.urlopen(request, timeout=timeout)
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and attempt == 0:  # stale/complete .part
                part.unlink(missing_ok=True)
                continue
            raise
        with response:
            status = getattr(response, "status", 200)
            mode = "ab" if offset and status == 206 else "wb"
            with open(part, mode) as handle:
                shutil.copyfileobj(response, handle, 1 << 20)
        break
    os.replace(part, dest)
    return dest


def verify_checksums(
    path: Path, *, sha256: str | None, md5: str | None, allow_unverified: bool
) -> dict[str, str]:
    """Verify ``path`` against the manifest checksums; return the computed ones."""
    computed: dict[str, str] = {"sha256": sha256_file(path)}
    if sha256 is not None:
        if computed["sha256"] != sha256:
            raise SystemExit(
                f"sha256 mismatch for {path.name}: expected {sha256}, "
                f"got {computed['sha256']}"
            )
        return computed
    if md5 is not None:
        computed["md5"] = md5_file(path)
        if computed["md5"] != md5:
            raise SystemExit(
                f"md5 mismatch for {path.name}: expected {md5}, got {computed['md5']}"
            )
        return computed
    if not allow_unverified:
        raise SystemExit(
            f"no checksum recorded for {path.name}; refusing to use it. "
            f"Computed sha256={computed['sha256']} (pass --allow-unverified to proceed)"
        )
    log(f"WARNING: unverified archive, sha256={computed['sha256']}")
    return computed


def _safe_members(tar: tarfile.TarFile, dest: Path) -> list[tarfile.TarInfo]:
    """Members of ``tar`` that are safe to extract below ``dest``.

    Only regular files and directories are allowed; absolute paths, ``..``
    components, links and device nodes make the whole archive be rejected.
    """
    root = dest.resolve()
    members: list[tarfile.TarInfo] = []
    for member in tar.getmembers():
        target = (root / member.name).resolve()
        if member.name.startswith("/") or not target.is_relative_to(root):
            raise SystemExit(f"unsafe path in archive: {member.name!r}")
        if not (member.isfile() or member.isdir()):
            raise SystemExit(f"unsupported member type in archive: {member.name!r}")
        members.append(member)
    return members


def safe_extract(archive: Path, dest: Path) -> Path:
    """Extract ``archive`` below ``dest`` and return the extracted top directory."""
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive, "r:*") as tar:
        members = _safe_members(tar, dest)
        tops = {m.name.split("/")[0] for m in members if m.name not in ("", ".")}
        kwargs: dict[str, Any] = {}
        if hasattr(tarfile, "data_filter"):  # Python >= 3.11.4 / 3.12
            kwargs["filter"] = "data"
        tar.extractall(dest, members=members, **kwargs)
    if len(tops) != 1:
        return dest
    return dest / next(iter(tops))


def fetch_dataset(
    entry: dict[str, Any],
    dest: Path,
    *,
    url: str | None = None,
    sha256: str | None = None,
    allow_unverified: bool = False,
    force: bool = False,
    delete_archive: bool = False,
) -> Path:
    """Download, verify and extract one manifest dataset; return its directory."""
    name = entry["name"]
    url = url or entry.get("url")
    if entry.get("status") != "published" and not url:
        raise SystemExit(
            f"dataset {name!r} is not published yet: {entry.get('todo') or 'no URL'}"
        )
    if not url:
        raise SystemExit(f"dataset {name!r} has no URL")
    # A caller-supplied sha256 replaces the manifest checksums entirely.
    md5 = None if sha256 else entry.get("md5")
    sha256 = sha256 or entry.get("sha256")
    if sha256 is None and md5 is None and not allow_unverified:
        raise SystemExit(
            f"dataset {name!r} has no checksum in the manifest ({entry.get('todo')}); "
            "pass --sha256 or --allow-unverified"
        )
    dest.mkdir(parents=True, exist_ok=True)
    archive = dest / f"{name}.tar.gz"
    marker = dest / f"{name}.fetched.json"
    extracted = dest / entry.get("archive_root", name)
    if not force and marker.exists() and extracted.is_dir():
        info = json.loads(marker.read_text(encoding="utf-8"))
        if info.get("sha256") == (sha256 or info.get("sha256")):
            log(f"{name}: already fetched ({extracted})")
            return extracted
    if force:
        archive.unlink(missing_ok=True)
        archive.with_name(archive.name + ".part").unlink(missing_ok=True)
    if archive.exists():
        try:
            verify_checksums(
                archive, sha256=sha256, md5=md5, allow_unverified=allow_unverified
            )
            log(f"{name}: reusing verified archive {archive.name}")
        except SystemExit:
            log(f"{name}: existing archive is corrupt, downloading again")
            archive.unlink()
    if not archive.exists():
        log(f"{name}: downloading {url}")
        download(url, archive)
    computed = verify_checksums(
        archive, sha256=sha256, md5=md5, allow_unverified=allow_unverified
    )
    staging = Path(tempfile.mkdtemp(prefix=f".extract-{name}-", dir=dest))
    try:
        log(f"{name}: extracting")
        top = safe_extract(archive, staging)
        if extracted.exists():
            shutil.rmtree(extracted)
        os.replace(top, extracted)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    n_files = sum(1 for p in extracted.iterdir() if p.is_file())
    marker.write_text(
        json.dumps({"name": name, "url": url, "n_files": n_files, **computed}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    if delete_archive:
        archive.unlink()
    log(f"{name}: ready at {extracted} ({n_files} files)")
    return extracted


# --------------------------------------------------------------------------- #
# subsample: deterministic, nested subsets
# --------------------------------------------------------------------------- #
def rank_key(filename: str, seed: str | int) -> tuple[str, str]:
    """Sort key of a file: ``sha1(filename NUL seed)`` (hex), ties by filename.

    Independent of ``PYTHONHASHSEED``, locale, filesystem order and platform.
    """
    digest = hashlib.sha1(f"{filename}\0{seed}".encode(), usedforsecurity=False)
    return digest.hexdigest(), filename


def list_inputs(src: Path, ext: str = ".gb") -> list[str]:
    return sorted(
        p.name
        for p in src.iterdir()
        if p.is_file() and p.name.endswith(ext) and not p.name.startswith(".")
    )


def select_subset(
    names: Iterable[str], n: int, seed: str | int = DEFAULT_SEED
) -> list[str]:
    """First ``n`` names by ``rank_key``. Subsets for increasing ``n`` are nested."""
    ranked = sorted(set(names), key=lambda name: rank_key(name, seed))
    return ranked[:n]


def listing_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def write_subset(
    src: Path,
    out: Path,
    n: int,
    *,
    seed: str | int = DEFAULT_SEED,
    ext: str = ".gb",
    copy: bool = False,
    force: bool = False,
    allow_fewer: bool = False,
) -> dict[str, Any]:
    """Materialize the subset in ``out`` and write ``out/subset.json``."""
    available = list_inputs(src, ext)
    if len(available) < n and not allow_fewer:
        raise SystemExit(
            f"{src} has only {len(available)} '{ext}' files, fewer than the "
            f"requested {n} (use --allow-fewer to take all of them)"
        )
    selected = select_subset(available, n, seed)
    out.mkdir(parents=True, exist_ok=True)
    existing = [p for p in out.iterdir() if p.name.endswith(ext) or p.is_symlink()]
    if existing and not force:
        raise SystemExit(f"{out} is not empty; use --force to replace its contents")
    for path in existing:
        path.unlink()
    (out / "subset.json").unlink(missing_ok=True)
    for name in selected:
        source = (src / name).resolve()
        target = out / name
        if copy:
            shutil.copy2(source, target)
            continue
        try:
            target.symlink_to(source)
        except OSError:  # e.g. filesystems without symlink support
            shutil.copy2(source, target)
    doc = {
        "schema_version": 1,
        "kind": "plastbench-subset",
        "source_name": src.resolve().name,
        "seed": str(seed),
        "extension": ext,
        "n_requested": n,
        "n_selected": len(selected),
        "n_available": len(available),
        "method": "files ascending by sha1(filename + NUL + seed), ties by filename",
        "listing_sha256": listing_sha256(selected),
        "files": selected,
    }
    (out / "subset.json").write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return doc


# --------------------------------------------------------------------------- #
# Peak RSS sampling of a process tree
# --------------------------------------------------------------------------- #
def _process_memory(proc: psutil.Process, metric: str) -> int:
    if metric == "pss":
        try:
            pss = getattr(proc.memory_full_info(), "pss", None)
            if pss is not None:
                return int(pss)
        except (psutil.Error, OSError):
            pass
    return int(proc.memory_info().rss)


def tree_memory(root: psutil.Process, metric: str = "rss") -> int:
    """Summed memory (bytes) of ``root`` and all of its living descendants."""
    total = 0
    try:
        procs = [root, *root.children(recursive=True)]
    except psutil.Error:
        return 0
    for proc in procs:
        try:
            total += _process_memory(proc, metric)
        except (psutil.Error, OSError):
            continue  # the process exited between listing and sampling
    return total


class PeakRssSampler(threading.Thread):
    """Background thread that tracks the peak memory of a process tree."""

    def __init__(self, pid: int, interval: float = 0.05, metric: str = "rss") -> None:
        super().__init__(daemon=True, name=f"plastbench-rss-{pid}")
        self.interval = interval
        self.metric = metric
        self.peak = 0
        self.samples = 0
        self._stop_event = threading.Event()
        try:
            self._root: psutil.Process | None = psutil.Process(pid)
        except psutil.Error:
            self._root = None

    def run(self) -> None:
        if self._root is None:
            return
        while True:
            self.peak = max(self.peak, tree_memory(self._root, self.metric))
            self.samples += 1
            if self._stop_event.wait(self.interval):
                break

    def stop(self) -> int:
        self._stop_event.set()
        if self.is_alive():
            self.join(timeout=5)
        return self.peak


@dataclass
class Measurement:
    """Outcome of one measured command."""

    returncode: int | None
    wall_seconds: float
    peak_rss_bytes: int
    samples: int
    timed_out: bool
    log_tail: str


def _tail(path: Path, limit: int = 2000) -> str:
    try:
        data = path.read_bytes()[-limit:]
    except OSError:
        return ""
    return data.decode("utf-8", errors="replace")


def _kill_tree(proc: subprocess.Popen[bytes]) -> None:
    try:
        parent = psutil.Process(proc.pid)
        victims = [*parent.children(recursive=True), parent]
    except psutil.Error:
        victims = []
    for victim in victims:
        try:
            victim.kill()
        except psutil.Error:
            pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
        try:
            proc.kill()
        except OSError:  # e.g. signals denied by a sandbox: report, do not crash
            log(f"WARNING: could not kill process {proc.pid} after a timeout")


def run_measured(
    cmd: Sequence[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    timeout: float | None = None,
    stdout_path: Path | None = None,
    log_path: Path | None = None,
    metric: str = "rss",
    interval: float = 0.05,
) -> Measurement:
    """Run ``cmd``, measuring wall time and the peak memory of its process tree.

    ``stdout_path`` redirects stdout to a file (for tools that print the
    alignment); stderr (and stdout otherwise) goes to ``log_path``.
    """
    own_log = log_path is None
    if log_path is None:
        handle, name = tempfile.mkstemp(prefix="plastbench-", suffix=".log")
        os.close(handle)
        log_path = Path(name)
    out_handle = open(stdout_path, "wb") if stdout_path else None
    log_handle = open(log_path, "wb")
    try:
        started = time.perf_counter()
        proc = subprocess.Popen(
            list(cmd),
            cwd=cwd,
            env=env,
            stdout=out_handle if out_handle else log_handle,
            stderr=log_handle,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
        sampler = PeakRssSampler(proc.pid, interval=interval, metric=metric)
        sampler.start()
        timed_out = False
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill_tree(proc)
        wall = time.perf_counter() - started
        peak = sampler.stop()
    finally:
        log_handle.close()
        if out_handle:
            out_handle.close()
    result = Measurement(
        returncode=None if timed_out else proc.returncode,
        wall_seconds=wall,
        peak_rss_bytes=peak,
        samples=sampler.samples,
        timed_out=timed_out,
        log_tail=_tail(log_path),
    )
    if own_log:
        log_path.unlink(missing_ok=True)
    return result


# --------------------------------------------------------------------------- #
# FASTA helpers and concatenation
# --------------------------------------------------------------------------- #
def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Read a (small) FASTA file as ``[(header, sequence), ...]``."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    parts: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.strip()
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(parts)))
                header, parts = line[1:], []
            elif line:
                parts.append("".join(line.split()))
    if header is not None:
        records.append((header, "".join(parts)))
    return records


def fasta_dimensions(path: Path) -> tuple[int, int]:
    """``(ntax, nchar)`` of an alignment FASTA, streaming (nchar = longest row)."""
    ntax = nchar = current = 0
    with open(path, "rb") as handle:
        for raw in handle:
            if raw.startswith(b">"):
                ntax += 1
                nchar = max(nchar, current)
                current = 0
            else:
                current += len(raw.strip())
    return ntax, max(nchar, current)


def locus_name(path: Path | str) -> str:
    """``nucl_trnL_UAA.unalign.fasta`` -> ``trnL_UAA``."""
    name = Path(path).name
    for suffix in (".fasta", ".fa", ".aligned", ".unalign"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name[5:] if name.startswith("nucl_") else name


def taxon_from_header(header: str, locus: str) -> str:
    """First header token without PBA's ``<locus>_`` prefix."""
    token = header.split()[0] if header.split() else header
    prefix = f"{locus}_"
    return token[len(prefix) :] if token.startswith(prefix) else token


def concatenate_alignments(files: Sequence[Path], out_path: Path) -> tuple[int, int]:
    """Concatenate per-locus alignments into one supermatrix FASTA.

    Taxa are keyed by their header (minus the ``<locus>_`` prefix) and listed in
    order of first appearance; a taxon missing from a locus is filled with gaps;
    rows shorter than the longest row of their locus are padded with ``-``.
    Output headers are ``>concat_<taxon>`` (same as PBA). Returns ``(ntax, nchar)``.
    """
    order: dict[str, None] = {}
    pieces: list[dict[str, str]] = []
    lengths: list[int] = []
    for path in files:
        locus = locus_name(path)
        rows: dict[str, str] = {}
        for header, seq in read_fasta(path):
            taxon = taxon_from_header(header, locus)
            order.setdefault(taxon, None)
            rows[taxon] = rows.get(taxon, "") + seq
        pieces.append(rows)
        lengths.append(max((len(s) for s in rows.values()), default=0))
    with open(out_path, "w", encoding="utf-8") as out:
        for taxon in order:
            out.write(f">concat_{taxon}\n")
            for rows, length in zip(pieces, lengths, strict=True):
                out.write(rows.get(taxon, "").ljust(length, "-"))
            out.write("\n")
    return len(order), sum(lengths)


# --------------------------------------------------------------------------- #
# Topology metrics (Robinson-Foulds) with a tiny Newick parser
# --------------------------------------------------------------------------- #
_NEWICK_TOKEN = re.compile(
    r"""\s*(\(|\)|,|;|\[[^\]]*\]|:[^,();\[\]\s]*|'(?:[^']|'')*'|[^,();:\[\]\s']+)"""
)


def normalize_taxon_label(label: str) -> str:
    """Strip PBA's ``concat_`` prefix so matrix and reference labels agree."""
    return label[len("concat_") :] if label.startswith("concat_") else label


def parse_newick(
    text: str, label_index: dict[str, int] | None = None
) -> tuple[list[str], list[int]]:
    """Parse a Newick string (iteratively, so deep trees are fine).

    Returns ``(tip_labels, clade_masks)``; ``clade_masks`` holds one integer
    bit-mask per non-root internal node, built from ``label_index`` (tips that
    are not in the index contribute no bits).
    """
    tips: list[str] = []
    clades: list[int] = []
    stack: list[int] = []
    previous = ""
    for match in _NEWICK_TOKEN.finditer(text):
        token = match.group(1)
        if token.startswith("["):
            continue
        if token == "(":
            stack.append(0)
            previous = "("
        elif token == ",":
            previous = ","
        elif token == ")":
            if not stack:
                raise ValueError("unbalanced parentheses in Newick string")
            mask = stack.pop()
            if stack:
                stack[-1] |= mask
                clades.append(mask)
            previous = ")"
        elif token == ";":
            break
        elif token.startswith(":"):
            continue
        else:
            if previous in ("(", ","):  # a leaf; after ')' it is a support label
                label = token[1:-1].replace("''", "'") if token[0] == "'" else token
                label = normalize_taxon_label(label)
                tips.append(label)
                bit = label_index.get(label) if label_index else None
                if bit is not None and stack:
                    stack[-1] |= 1 << bit
            previous = "label"
    if stack:
        raise ValueError("unbalanced parentheses in Newick string")
    return tips, clades


def robinson_foulds(tree_a: str, tree_b: str) -> dict[str, Any]:
    """Unrooted Robinson-Foulds distance of two Newick trees on shared tips.

    ``normalized_rf`` is ``RF / (|splits_a| + |splits_b|)`` (0 = identical
    topologies, 1 = no shared non-trivial split), equal to ``RF / (2(n-3))`` for
    fully resolved trees.
    """
    tips_a, _ = parse_newick(tree_a)
    tips_b, _ = parse_newick(tree_b)
    shared = sorted(set(tips_a) & set(tips_b))
    n = len(shared)
    if n < 4:
        return {"rf": None, "normalized_rf": None, "n_shared_taxa": n}
    index = {label: i for i, label in enumerate(shared)}
    full = (1 << n) - 1

    def splits(text: str) -> set[int]:
        _, clades = parse_newick(text, index)
        found: set[int] = set()
        for mask in clades:
            size = mask.bit_count()
            if size < 2 or n - size < 2:
                continue
            found.add(mask ^ full if mask & 1 else mask)
        return found

    set_a, set_b = splits(tree_a), splits(tree_b)
    distance = len(set_a ^ set_b)
    denominator = len(set_a) + len(set_b)
    return {
        "rf": distance,
        "normalized_rf": (distance / denominator) if denominator else 0.0,
        "n_shared_taxa": n,
    }


def _find_exe(candidates: Sequence[str]) -> str | None:
    for name in candidates:
        found = shutil.which(name)
        if found:
            return found
    return None


def tree_builder(method: str = "auto") -> tuple[str, str] | None:
    """``(kind, executable)`` of the quick tree builder to use, if any."""
    if method == "none":
        return None
    fasttree = _find_exe(("FastTree", "FastTreeMP", "fasttree"))
    iqtree = _find_exe(("iqtree2", "iqtree"))
    if method == "fasttree":
        return ("fasttree", fasttree) if fasttree else None
    if method == "iqtree":
        return ("iqtree", iqtree) if iqtree else None
    if fasttree:
        return "fasttree", fasttree
    if iqtree:
        return "iqtree", iqtree
    return None


def build_tree(builder: tuple[str, str], matrix: Path, out_nwk: Path) -> bool:
    """Build a quick ML tree from a nucleotide FASTA matrix; True on success."""
    kind, exe = builder
    if kind == "fasttree":
        m = run_measured(
            [exe, "-nt", "-gtr", "-quiet", str(matrix)], stdout_path=out_nwk
        )
        return m.returncode == 0 and out_nwk.exists() and out_nwk.stat().st_size > 0
    prefix = out_nwk.with_suffix("")
    m = run_measured(
        [exe, "-s", str(matrix), "-m", "GTR+G", "-fast", "-nt", "1", "-seed", "1"]
        + ["-pre", str(prefix), "-redo", "-quiet"]
    )
    treefile = Path(f"{prefix}.treefile")
    if m.returncode == 0 and treefile.exists():
        shutil.copyfile(treefile, out_nwk)
        return True
    return False


# --------------------------------------------------------------------------- #
# Tool discovery and host description
# --------------------------------------------------------------------------- #
def _first_line(text: str) -> str | None:
    for line in text.splitlines():
        if line.strip():
            return line.strip()
    return None


def _version_of(cmd: Sequence[str]) -> str | None:
    try:
        proc = subprocess.run(
            list(cmd),
            capture_output=True,
            text=True,
            timeout=20,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return _first_line(proc.stdout + "\n" + proc.stderr)


def tool_versions(pba_python: str, env: dict[str, str]) -> dict[str, str | None]:
    versions: dict[str, str | None] = {"python": platform.python_version()}
    code = "import plastburstalign as p; print(p.__version__)"
    try:
        proc = subprocess.run(
            [pba_python, "-c", code],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
            stdin=subprocess.DEVNULL,
        )
        versions["plastburstalign"] = (
            _first_line(proc.stdout) if proc.returncode == 0 else None
        )
    except (OSError, subprocess.SubprocessError):
        versions["plastburstalign"] = None
    versions["pba_git_head"] = None
    versions["pba_git_dirty"] = None
    if shutil.which("git"):
        head = _version_of(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"])
        versions["pba_git_head"] = head if head and len(head) == 40 else None
        if versions["pba_git_head"]:
            dirty = _version_of(
                ["git", "-C", str(REPO_ROOT), "status", "--porcelain", "--untracked=no"]
            )
            versions["pba_git_dirty"] = "yes" if dirty else "no"
    probes: dict[str, list[str]] = {
        "mafft": ["mafft", "--version"],
        "muscle": ["muscle", "-version"],
        "clustalo": ["clustalo", "--version"],
    }
    for name, cmd in probes.items():
        versions[name] = _version_of(cmd) if shutil.which(cmd[0]) else None
    fasttree = _find_exe(("FastTree", "FastTreeMP", "fasttree"))
    versions["fasttree"] = _version_of([fasttree]) if fasttree else None
    iqtree = _find_exe(("iqtree2", "iqtree"))
    versions["iqtree"] = _version_of([iqtree, "--version"]) if iqtree else None
    return versions


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="replace").splitlines():
            if line.lower().startswith(("model name", "hardware", "cpu model")):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    if platform.system() == "Darwin":
        version = _version_of(["sysctl", "-n", "machdep.cpu.brand_string"])
        if version:
            return version
    return platform.processor() or "unknown"


def host_profile() -> dict[str, Any]:
    """Privacy-friendly summary of the machine (no host name or user name)."""
    memory = psutil.virtual_memory()
    return {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "cpu_model": _cpu_model(),
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_bytes": int(memory.total),
        "ram_available_bytes": int(memory.available),
        "in_ci": bool(os.environ.get("CI")),
    }


# --------------------------------------------------------------------------- #
# run: PBA and baselines
# --------------------------------------------------------------------------- #
def pba_environment() -> dict[str, str]:
    """Environment that makes ``python -m plastburstalign`` use this checkout."""
    env = dict(os.environ)
    if (REPO_ROOT / "plastburstalign" / "__init__.py").is_file():
        extra = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + extra if extra else "")
    return env


def pba_command(
    python: str,
    input_dir: Path,
    out_dir: Path,
    *,
    mode: str,
    threads: str | int,
    min_taxa: int,
    min_len: int,
    concat: bool,
    extra: Sequence[str] = (),
) -> list[str]:
    cmd = [
        python,
        "-m",
        "plastburstalign",
        "-i",
        str(input_dir),
        "-o",
        str(out_dir),
        "-s",
        mode,
        "-t",
        str(min_taxa),
        "-l",
        str(min_len),
        "-n",
        str(threads),
    ]
    if concat:
        cmd.append("-c")
    cmd.extend(extra)
    return cmd


def find_unaligned_fastas(out_dir: Path) -> list[Path]:
    """Unaligned per-locus FASTA files written by a PBA extraction run."""
    found = sorted(out_dir.glob("nucl_*.unalign.fasta"))
    if not found:
        found = sorted(out_dir.rglob("nucl_*.unalign*.fasta"))
    if not found:  # the no-op aligner copies the unaligned data to *.aligned.fasta
        found = sorted(
            p for p in out_dir.rglob("nucl_*.aligned.fasta") if "concat" not in p.name
        )
    return found


def find_concat_matrix(out_dir: Path) -> Path | None:
    """The concatenated supermatrix FASTA written by PBA, if any."""
    for pattern in ("*concat*.aligned.fasta", "*concat*.fasta", "*supermatrix*.fasta"):
        candidates = sorted(out_dir.rglob(pattern))
        if candidates:
            return candidates[0]
    return None


def count_aligned_loci(out_dir: Path) -> int:
    return sum(
        1 for p in out_dir.rglob("nucl_*.aligned.fasta") if "concat" not in p.name
    )


def _phase(m: Measurement) -> dict[str, Any]:
    return {
        "wall_seconds": round(m.wall_seconds, 4),
        "peak_rss_bytes": m.peak_rss_bytes,
    }


def _empty_tool(
    status: str,
    reason: str | None,
    command: Sequence[str] | None = None,
    threads: str | int | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "reason": reason,
        "command": list(command) if command else None,
        "threads": threads,
        "wall_seconds": None,
        "peak_rss_bytes": None,
        "phases": None,
        "n_loci": None,
        "ntax": None,
        "nchar": None,
        "matrix_sha256": None,
        "tree": None,
    }


def _fill_matrix(entry: dict[str, Any], matrix: Path | None) -> Path | None:
    if matrix is not None and matrix.exists():
        entry["ntax"], entry["nchar"] = fasta_dimensions(matrix)
        entry["matrix_sha256"] = sha256_file(matrix)
        return matrix
    return None


def run_pba(
    args: argparse.Namespace, input_dir: Path, work: Path
) -> tuple[dict[str, Any], Path | None]:
    out_dir = work / "pba_out"
    cmd = pba_command(
        args.pba_python,
        input_dir,
        out_dir,
        mode=args.mode,
        threads=args.threads,
        min_taxa=args.min_taxa,
        min_len=args.min_len,
        concat=True,
        extra=args.pba_arg,
    )
    entry = _empty_tool("failed", None, cmd, args.threads)
    log("pba: " + shlex.join(cmd))
    m = run_measured(
        cmd,
        env=pba_environment(),
        timeout=args.timeout,
        log_path=work / "pba.log",
        metric=args.rss_metric,
        interval=args.poll_interval,
    )
    entry["wall_seconds"] = round(m.wall_seconds, 4)
    entry["peak_rss_bytes"] = m.peak_rss_bytes
    if m.timed_out:
        entry.update(status="timeout", reason=f"timed out after {args.timeout}s")
        return entry, None
    if m.returncode != 0:
        entry["reason"] = f"exit status {m.returncode}: {m.log_tail[-400:]}"
        return entry, None
    matrix = _fill_matrix(entry, find_concat_matrix(out_dir))
    entry["n_loci"] = count_aligned_loci(out_dir)
    if matrix is None:
        entry["reason"] = "PBA finished but wrote no concatenated matrix"
        return entry, None
    entry.update(status="ok", reason=None)
    return entry, matrix


def run_extraction(
    args: argparse.Namespace, input_dir: Path, work: Path
) -> tuple[Measurement | None, Path | None, str | None]:
    """Run PBA with a no-op aligner to produce the unaligned per-locus FASTAs."""
    out_dir = work / "extract_out"
    cmd = pba_command(
        args.pba_python,
        input_dir,
        out_dir,
        mode=args.mode,
        threads=args.threads,
        min_taxa=args.min_taxa,
        min_len=args.min_len,
        concat=False,
        extra=["--alignmentcommand", "cp %INPUT% %OUTPUT%"],
    )
    log("extract (shared by baselines): " + shlex.join(cmd))
    m = run_measured(
        cmd,
        env=pba_environment(),
        timeout=args.timeout,
        log_path=work / "extract.log",
        metric=args.rss_metric,
        interval=args.poll_interval,
    )
    fastas = find_unaligned_fastas(out_dir)
    if m.timed_out or not fastas:
        why = "timed out" if m.timed_out else f"no unaligned FASTA: {m.log_tail[-300:]}"
        return m, None, f"extraction failed ({why})"
    unaligned = work / "unaligned"
    unaligned.mkdir(parents=True, exist_ok=True)
    for path in fastas:
        target = unaligned / f"{locus_name(path)}.fasta"
        if not target.exists():
            target.symlink_to(path.resolve())
    return m, unaligned, None


def _aligner_command(tool: str, src: Path) -> list[str]:
    if tool == "mafft":
        return ["mafft", "--auto", "--thread", "1", "--quiet", str(src)]
    if tool == "muscle":
        return ["muscle", "-super5", str(src), "-threads", "1", "-output"]
    if tool == "clustalo":
        return [
            "clustalo",
            "--auto",
            "-i",
            str(src),
            "--outfmt=fasta",
            "--threads=1",
            "--force",
            "-o",
        ]
    raise ValueError(f"not a Python-driven baseline: {tool}")


def baseline_loop(tool: str, unaligned_dir: Path, out_dir: Path) -> dict[str, Any]:
    """Align every locus one after the other, then concatenate (runs in a child)."""
    aligned_dir = out_dir / "aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    aligned: list[Path] = []
    failed: list[str] = []
    for src in sorted(unaligned_dir.glob("*.fasta")):
        if src.stat().st_size == 0:
            continue
        dst = aligned_dir / f"{src.stem}.aligned.fasta"
        cmd = _aligner_command(tool, src)
        if tool == "mafft":
            with open(dst, "wb") as handle:
                proc = subprocess.run(
                    cmd,
                    stdout=handle,
                    stderr=subprocess.DEVNULL,
                    stdin=subprocess.DEVNULL,
                )
        else:
            proc = subprocess.run(
                [*cmd, str(dst)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        if proc.returncode == 0 and dst.exists() and dst.stat().st_size > 0:
            aligned.append(dst)
        else:
            dst.unlink(missing_ok=True)
            failed.append(src.name)
    matrix = out_dir / "concat.aligned.fasta"
    ntax, nchar = concatenate_alignments(aligned, matrix) if aligned else (0, 0)
    summary = {
        "tool": tool,
        "n_loci": len(aligned),
        "n_failed_loci": len(failed),
        "failed_loci": failed[:20],
        "ntax": ntax,
        "nchar": nchar,
    }
    (out_dir / "baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def write_shell_pipeline(path: Path) -> Path:
    script = SHELL_PIPELINE.replace("__DEFAULT_ALIGNER__", SHELL_DEFAULT_ALIGNER)
    path.write_text(script, encoding="utf-8")
    path.chmod(0o755)
    return path


def baseline_available(tool: str) -> tuple[bool, str | None]:
    needed = {"shell": ["bash", "awk", "mafft"]}.get(tool, [tool])
    missing = [exe for exe in needed if shutil.which(exe) is None]
    if missing:
        return False, f"{', '.join(missing)} not found on PATH"
    return True, None


def run_baseline(
    args: argparse.Namespace,
    tool: str,
    unaligned: Path,
    work: Path,
    extract: Measurement,
) -> tuple[dict[str, Any], Path | None]:
    out_dir = work / f"{tool}_out"
    out_dir.mkdir(parents=True, exist_ok=True)
    if tool == "shell":
        script = write_shell_pipeline(work / "shell_pipeline.sh")
        cmd = ["bash", str(script), str(unaligned), str(out_dir)]
    else:
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "_baseline",
            "--tool",
            tool,
            "--unaligned",
            str(unaligned),
            "--out",
            str(out_dir),
        ]
    entry = _empty_tool("failed", None, cmd, 1)
    log(f"{tool}: " + shlex.join(cmd))
    m = run_measured(
        cmd,
        timeout=args.timeout,
        log_path=work / f"{tool}.log",
        metric=args.rss_metric,
        interval=args.poll_interval,
    )
    entry["phases"] = {"extract": _phase(extract), "align_concat": _phase(m)}
    entry["wall_seconds"] = round(extract.wall_seconds + m.wall_seconds, 4)
    entry["peak_rss_bytes"] = max(extract.peak_rss_bytes, m.peak_rss_bytes)
    if m.timed_out:
        entry.update(status="timeout", reason=f"timed out after {args.timeout}s")
        return entry, None
    if m.returncode != 0:
        entry["reason"] = f"exit status {m.returncode}: {m.log_tail[-400:]}"
        return entry, None
    matrix = _fill_matrix(entry, out_dir / "concat.aligned.fasta")
    entry["n_loci"] = len(list((out_dir / "aligned").glob("*.aligned.fasta")))
    if matrix is None:
        entry["reason"] = "no concatenated matrix was written"
        return entry, None
    entry.update(status="ok", reason=None)
    return entry, matrix


def add_topology_metrics(
    results: dict[str, dict[str, Any]],
    matrices: dict[str, Path],
    work: Path,
    *,
    method: str,
    reference: str | None,
    baseline: str,
    notes: list[str],
) -> None:
    """Quick trees for every matrix and RF distance to the reference tree."""
    builder = tree_builder(method)
    if reference is None:
        notes.append("topology metrics skipped: no --reference-tree supplied")
        return
    if builder is None:
        notes.append("topology metrics skipped: neither FastTree nor iqtree2 on PATH")
        return
    trees: dict[str, str] = {}
    for tool, matrix in matrices.items():
        out = work / f"{tool}.nwk"
        if build_tree(builder, matrix, out):
            trees[tool] = out.read_text(encoding="utf-8")
        else:
            notes.append(f"topology: {builder[0]} failed on the {tool} matrix")
    if reference == "baseline":
        ref_text = trees.get(baseline)
        if ref_text is None:
            notes.append(f"topology skipped: no tree for baseline {baseline!r}")
            return
    else:
        ref_text = Path(reference).read_text(encoding="utf-8")
    for tool, text in trees.items():
        metrics = robinson_foulds(text, ref_text)
        results[tool]["tree"] = {"method": builder[0], **metrics}


def count_inputs(input_dir: Path, ext: str = ".gb") -> int:
    return len(list_inputs(input_dir, ext))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if isinstance(args.threads, str) and args.threads.isdigit():
        args.threads = int(args.threads)
    input_dir = Path(args.input).resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"--input {input_dir} is not a directory")
    n_inputs = count_inputs(input_dir)
    if n_inputs == 0:
        raise SystemExit(f"no .gb files in {input_dir}")
    subset: dict[str, Any] = {}
    subset_json = input_dir / "subset.json"
    if subset_json.exists():
        subset = json.loads(subset_json.read_text(encoding="utf-8"))
    tools = list(dict.fromkeys(args.tools))
    baseline = args.baseline
    if baseline not in tools and baseline in BASELINE_TOOLS:
        log(f"note: baseline {baseline!r} is not in --tools; speedups need it")

    keep_work = args.keep_work or args.workdir is not None
    work = Path(args.workdir or tempfile.mkdtemp(prefix="plastbench_")).resolve()
    work.mkdir(parents=True, exist_ok=True)
    notes: list[str] = [
        "baselines reuse PBA extraction (no-op aligner); extraction time is charged "
        "to every baseline as phase `extract`",
        "baselines run sequentially with one thread per aligner process",
    ]
    results: dict[str, dict[str, Any]] = {}
    matrices: dict[str, Path] = {}
    try:
        env = pba_environment()
        if "pba" in tools:
            results["pba"], matrix = run_pba(args, input_dir, work)
            if matrix:
                matrices["pba"] = matrix
        wanted = [t for t in tools if t in BASELINE_TOOLS]
        runnable: list[str] = []
        for tool in wanted:
            ok, why = baseline_available(tool)
            if ok:
                runnable.append(tool)
            else:
                results[tool] = _empty_tool("skipped", why)
        if runnable:
            extract, unaligned, error = run_extraction(args, input_dir, work)
            if extract is None or unaligned is None or error:
                for tool in runnable:
                    results[tool] = _empty_tool("failed", error)
            else:
                for tool in runnable:
                    results[tool], matrix = run_baseline(
                        args, tool, unaligned, work, extract
                    )
                    if matrix:
                        matrices[tool] = matrix
        if args.tree_method != "none" or args.reference_tree:
            add_topology_metrics(
                results,
                matrices,
                work,
                method=args.tree_method,
                reference=args.reference_tree,
                baseline=baseline,
                notes=notes,
            )
        versions = tool_versions(args.pba_python, env)
    finally:
        if not keep_work:
            shutil.rmtree(work, ignore_errors=True)
    ordered = {tool: results[tool] for tool in TOOLS if tool in results}
    return {
        "schema_version": RESULTS_SCHEMA_VERSION,
        "generator": GENERATOR,
        "created_utc": utc_now(),
        "dataset": {
            "name": args.dataset_name or subset.get("source_name"),
            "tier": args.tier,
            "n_genomes": n_inputs,
            "seed": subset.get("seed"),
            "listing_sha256": subset.get("listing_sha256"),
            "mode": args.mode,
            "min_num_taxa": args.min_taxa,
            "min_seq_length": args.min_len,
        },
        "host": host_profile(),
        "tool_versions": versions,
        "baseline": baseline,
        "settings": {
            "threads": args.threads,
            "rss_metric": args.rss_metric,
            "poll_interval_s": args.poll_interval,
            "timeout_s": args.timeout,
            "pba_extra_args": list(args.pba_arg),
        },
        "tools": ordered,
        "notes": notes,
    }


# --------------------------------------------------------------------------- #
# Results validation, report and badges
# --------------------------------------------------------------------------- #
def validate_results(doc: Any) -> list[str]:
    """Return a list of schema problems (empty when ``doc`` is valid)."""
    problems: list[str] = []
    if not isinstance(doc, dict):
        return ["results must be a JSON object"]
    if doc.get("schema_version") != RESULTS_SCHEMA_VERSION:
        problems.append(f"schema_version must be {RESULTS_SCHEMA_VERSION}")
    for key in ("dataset", "host", "tool_versions", "tools"):
        if not isinstance(doc.get(key), dict):
            problems.append(f"`{key}` must be an object")
    tools = doc.get("tools") if isinstance(doc.get("tools"), dict) else {}
    for name, entry in tools.items():
        where = f"tools.{name}"
        if name not in TOOLS:
            problems.append(f"{where}: unknown tool")
        if not isinstance(entry, dict):
            problems.append(f"{where}: must be an object")
            continue
        status = entry.get("status")
        if status not in STATUSES:
            problems.append(f"{where}: `status` must be one of {STATUSES}")
        if status == "ok":
            wall, rss = entry.get("wall_seconds"), entry.get("peak_rss_bytes")
            if not isinstance(wall, (int, float)) or wall <= 0:
                problems.append(f"{where}: ok tools need positive wall_seconds")
            if not isinstance(rss, int) or rss < 0:
                problems.append(f"{where}: ok tools need integer peak_rss_bytes")
    dataset = doc.get("dataset")
    if isinstance(dataset, dict) and not isinstance(dataset.get("n_genomes"), int):
        problems.append("dataset.n_genomes must be an integer")
    return problems


def _ok(doc: dict[str, Any], tool: str) -> dict[str, Any] | None:
    entry = doc.get("tools", {}).get(tool)
    if isinstance(entry, dict) and entry.get("status") == "ok":
        found: dict[str, Any] = entry
        return found
    return None


def speedup(doc: dict[str, Any], tool: str, baseline: str) -> float | None:
    """``baseline wall time / tool wall time`` or ``None`` if either is missing."""
    mine, ref = _ok(doc, tool), _ok(doc, baseline)
    if mine is None or ref is None or not mine["wall_seconds"]:
        return None
    return float(ref["wall_seconds"]) / float(mine["wall_seconds"])


def _fmt_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.1f}" if value >= 10 else f"{value:.2f}"


def _fmt_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    if value >= 1 << 30:
        return f"{value / (1 << 30):.2f} GiB"
    return f"{value / (1 << 20):.0f} MiB"


def _tier_sort_key(doc: dict[str, Any]) -> tuple[int, str]:
    dataset = doc.get("dataset", {})
    return int(dataset.get("n_genomes") or 0), str(dataset.get("name") or "")


def render_report(docs: Sequence[dict[str, Any]], baseline: str | None = None) -> str:
    """Render one or more results documents as a single Markdown table."""
    lines: list[str] = []
    ordered = sorted(docs, key=_tier_sort_key)
    ref_names = {baseline or d.get("baseline") or DEFAULT_BASELINE for d in ordered}
    head = (
        f"Speedup vs {next(iter(ref_names))}"
        if len(ref_names) == 1
        else "Speedup vs baseline"
    )
    lines.append(
        f"| Dataset | N | Mode | Tool | Status | Wall (s) | Peak RSS | {head} "
        "| ntax | nchar | loci | nRF |"
    )
    lines.append("|---|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for doc in ordered:
        dataset = doc.get("dataset", {})
        label = dataset.get("tier") or dataset.get("name") or "?"
        ref = baseline or doc.get("baseline") or DEFAULT_BASELINE
        for tool in TOOLS:
            entry = doc.get("tools", {}).get(tool)
            if not entry:
                continue
            ok = entry.get("status") == "ok"
            factor = speedup(doc, tool, ref) if ok else None
            tree = entry.get("tree") or {}
            rf = tree.get("normalized_rf")
            cells = [
                str(label),
                str(dataset.get("n_genomes", "?")),
                str(dataset.get("mode", "?")),
                TOOL_LABELS.get(tool, tool),
                entry.get("status", "?"),
                _fmt_seconds(entry.get("wall_seconds")) if ok else "n/a",
                _fmt_bytes(entry.get("peak_rss_bytes")) if ok else "n/a",
                f"{factor:.2f}x" if factor is not None else "n/a",
                str(entry.get("ntax")) if ok else "n/a",
                str(entry.get("nchar")) if ok else "n/a",
                str(entry.get("n_loci")) if ok else "n/a",
                f"{rf:.3f}" if isinstance(rf, (int, float)) else "n/a",
            ]
            lines.append("| " + " | ".join(cells) + " |")
    notes = sorted({n for d in ordered for n in d.get("notes", [])})
    if notes:
        lines.append("")
        lines.extend(f"- {note}" for note in notes)
    return "\n".join(lines) + "\n"


def _speedup_color(value: float | None) -> str:
    if value is None:
        return "lightgrey"
    for bound, color in ((10.0, "brightgreen"), (3.0, "green"), (1.5, "yellowgreen")):
        if value >= bound:
            return color
    return "yellow" if value >= 1.0 else "orange"


def _memory_color(value: int | None) -> str:
    if value is None:
        return "lightgrey"
    gib = value / (1 << 30)
    if gib < 1:
        return "brightgreen"
    if gib < 4:
        return "green"
    return "yellow" if gib < 16 else "orange"


def make_badges(doc: dict[str, Any], baseline: str | None = None) -> dict[str, Any]:
    """shields.io endpoint JSON documents keyed by output file name."""
    ref = baseline or doc.get("baseline") or DEFAULT_BASELINE
    n = doc.get("dataset", {}).get("n_genomes", "?")
    factor = speedup(doc, "pba", ref)
    entry = _ok(doc, "pba")
    rss = entry["peak_rss_bytes"] if entry else None
    return {
        "speedup.json": {
            "schemaVersion": 1,
            "label": f"PBA speedup vs {ref} (N={n})",
            "message": f"{factor:.1f}x" if factor is not None else "n/a",
            "color": _speedup_color(factor),
        },
        "peak_memory.json": {
            "schemaVersion": 1,
            "label": f"PBA peak RAM (N={n})",
            "message": _fmt_bytes(rss),
            "color": _memory_color(rss),
        },
    }


def write_badges(
    docs: Sequence[dict[str, Any]], out_dir: Path, baseline: str | None = None
) -> list[Path]:
    """Write the badge files for the largest successful results document."""
    usable = [d for d in docs if _ok(d, "pba")] or list(docs)
    chosen = max(usable, key=_tier_sort_key)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, payload in make_badges(chosen, baseline).items():
        path = out_dir / name
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        written.append(path)
    return written


def load_results(paths: Iterable[str]) -> list[dict[str, Any]]:
    docs: list[dict[str, Any]] = []
    for item in paths:
        with open(item, encoding="utf-8") as handle:
            doc = json.load(handle)
        problems = validate_results(doc)
        if problems:
            raise SystemExit(f"{item}: invalid results file: " + "; ".join(problems))
        docs.append(doc)
    return docs


# --------------------------------------------------------------------------- #
# pack: deterministic archives for Zenodo
# --------------------------------------------------------------------------- #
def pack_directory(src: Path, out: Path) -> dict[str, Any]:
    """Create a byte-reproducible ``.tar.gz`` of ``src`` (sorted, no timestamps)."""
    src = src.resolve()
    files = sorted(p for p in src.rglob("*") if p.is_file() and not p.is_symlink())
    with open(out, "wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tar:
                root = tarfile.TarInfo(src.name)
                root.type, root.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(root)
                for path in files:
                    info = tarfile.TarInfo(
                        f"{src.name}/{path.relative_to(src).as_posix()}"
                    )
                    info.size = path.stat().st_size
                    info.mode = 0o644
                    with open(path, "rb") as handle:
                        tar.addfile(info, handle)
    return {
        "archive": out.name,
        "n_files": len(files),
        "bytes": out.stat().st_size,
        "sha256": sha256_file(out),
        "md5": md5_file(out),
    }


# --------------------------------------------------------------------------- #
# Command line interface
# --------------------------------------------------------------------------- #
def _cmd_list(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    problems = validate_manifest(manifest)
    for problem in problems:
        print(f"manifest problem: {problem}", file=sys.stderr)
    print("Datasets:")
    for entry in manifest["datasets"]:
        print(f"  {entry['name']:<45} N={entry['n_genomes']!s:<7} {entry['status']}")
    print("Tiers:")
    for tier, spec in manifest["tiers"].items():
        print(
            f"  {tier:<6} N={spec['n_genomes']:<6} master={spec['master']:<42} "
            f"{spec['status']}"
        )
    return 1 if problems else 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    entry = find_dataset(manifest, args.name)
    path = fetch_dataset(
        entry,
        Path(args.dest),
        url=args.url,
        sha256=args.sha256,
        allow_unverified=args.allow_unverified,
        force=args.force,
        delete_archive=args.delete_archive,
    )
    print(path)
    return 0


def _cmd_subsample(args: argparse.Namespace) -> int:
    n = TIERS[args.tier] if args.tier else args.n
    if n is None:
        raise SystemExit("give -n/--n or --tier")
    doc = write_subset(
        Path(args.src),
        Path(args.out),
        n,
        seed=args.seed,
        ext=args.ext,
        copy=args.copy,
        force=args.force,
        allow_fewer=args.allow_fewer,
    )
    log(f"selected {doc['n_selected']} of {doc['n_available']} files -> {args.out}")
    print(Path(args.out) / "subset.json")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    doc = run_benchmark(args)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(render_report([doc]))
    log(f"results written to {out}")
    bad = [t for t, e in doc["tools"].items() if e["status"] in ("failed", "timeout")]
    if bad:
        log("failed tools: " + ", ".join(bad))
    # Baseline failures are data (recorded in the JSON); only a failing PBA run,
    # or any failure under --strict, makes the command fail.
    return 1 if ("pba" in bad or (args.strict and bad)) else 0


def _cmd_report(args: argparse.Namespace) -> int:
    text = render_report(load_results(args.results), args.baseline)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_badges(args: argparse.Namespace) -> int:
    for path in write_badges(
        load_results(args.results), Path(args.out_dir), args.baseline
    ):
        print(path)
    return 0


def _cmd_pack(args: argparse.Namespace) -> int:
    info = pack_directory(Path(args.directory), Path(args.out))
    print(json.dumps(info, indent=2))
    return 0


def _cmd_baseline(args: argparse.Namespace) -> int:
    """Internal: run one Python-driven baseline (measured by the parent)."""
    summary = baseline_loop(args.tool, Path(args.unaligned), Path(args.out))
    print(json.dumps(summary))
    return 0 if summary["n_loci"] > 0 else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plastbench",
        description="PlastBench: benchmark PlastBurstAlign against conventional "
        "alignment pipelines.",
    )
    parser.add_argument(
        "--manifest", default=str(DEFAULT_MANIFEST), help=argparse.SUPPRESS
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("list", help="show the dataset manifest and tiers")
    p.set_defaults(func=_cmd_list)

    p = sub.add_parser("fetch", help="download, verify (sha256) and extract a dataset")
    p.add_argument("name", help="dataset name from datasets.json")
    p.add_argument("--dest", default="plastbench_data", help="download directory")
    p.add_argument("--url", help="override the manifest URL (e.g. a mirror)")
    p.add_argument("--sha256", help="override the manifest sha256")
    p.add_argument("--allow-unverified", action="store_true")
    p.add_argument("--force", action="store_true", help="re-download and re-extract")
    p.add_argument("--delete-archive", action="store_true")
    p.set_defaults(func=_cmd_fetch)

    p = sub.add_parser("subsample", help="deterministic subset of a genome directory")
    p.add_argument(
        "--src", required=True, help="directory with the master GenBank files"
    )
    p.add_argument(
        "--out", required=True, help="output directory (symlinks + subset.json)"
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument("-n", "--n", type=int, help="number of genomes")
    group.add_argument("--tier", choices=list(TIERS), help="named tier")
    p.add_argument(
        "--seed", default=DEFAULT_SEED, help="subsampling seed (fixed by default)"
    )
    p.add_argument("--ext", default=".gb", help="input file extension")
    p.add_argument(
        "--copy", action="store_true", help="copy files instead of symlinking"
    )
    p.add_argument("--force", action="store_true")
    p.add_argument("--allow-fewer", action="store_true")
    p.set_defaults(func=_cmd_subsample)

    p = sub.add_parser("run", help="run PBA and the baselines, write results JSON")
    p.add_argument("--input", required=True, help="directory of GenBank files")
    p.add_argument("--out", default="plastbench_results.json")
    p.add_argument("--tools", nargs="+", choices=TOOLS, default=list(TOOLS))
    p.add_argument("--baseline", choices=BASELINE_TOOLS, default=DEFAULT_BASELINE)
    p.add_argument("--mode", default="igs", choices=("cds", "igs", "int", "rna"))
    p.add_argument(
        "--threads", default="auto", help="PBA threads (-n); baselines use 1"
    )
    p.add_argument("--min-taxa", type=int, default=5)
    p.add_argument("--min-len", type=int, default=6)
    p.add_argument("--tier", help="label stored in the results (e.g. smoke)")
    p.add_argument("--dataset-name", help="dataset label stored in the results")
    p.add_argument("--timeout", type=float, default=None, help="seconds per tool")
    p.add_argument("--workdir", help="keep all intermediate files here")
    p.add_argument("--keep-work", action="store_true")
    p.add_argument("--pba-python", default=sys.executable)
    p.add_argument("--pba-arg", action="append", default=[], help="extra PBA flag")
    p.add_argument("--rss-metric", choices=("rss", "pss"), default="rss")
    p.add_argument("--poll-interval", type=float, default=0.05)
    p.add_argument("--strict", action="store_true", help="fail if any tool fails")
    p.add_argument(
        "--tree-method", choices=("auto", "fasttree", "iqtree", "none"), default="auto"
    )
    p.add_argument(
        "--reference-tree", help="Newick file, or `baseline`, for RF distances"
    )
    p.set_defaults(func=_cmd_run)

    p = sub.add_parser("report", help="render results JSON files as a Markdown table")
    p.add_argument("results", nargs="+")
    p.add_argument("--baseline", choices=BASELINE_TOOLS)
    p.add_argument("--output", help="write to a file instead of stdout")
    p.set_defaults(func=_cmd_report)

    p = sub.add_parser("badges", help="write shields.io endpoint badge JSON files")
    p.add_argument("results", nargs="+")
    p.add_argument("--out-dir", default="badges")
    p.add_argument("--baseline", choices=BASELINE_TOOLS)
    p.set_defaults(func=_cmd_badges)

    p = sub.add_parser("pack", help="build a reproducible .tar.gz for Zenodo")
    p.add_argument("directory")
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_pack)

    p = sub.add_parser("_baseline", help=argparse.SUPPRESS)
    p.add_argument("--tool", required=True, choices=("mafft", "muscle", "clustalo"))
    p.add_argument("--unaligned", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=_cmd_baseline)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    func = args.func
    code: int = func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
