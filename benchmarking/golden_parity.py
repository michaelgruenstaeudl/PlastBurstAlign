#!/usr/bin/env python
"""Golden parity gate for PlastBurstAlign refactors.

The gate runs the *real* command line interface
(``python -m plastburstalign``) on a small, fixed subset of the Asteraceae
benchmark (the first K files, sorted by name, of
``benchmarking/benchmark_YangEtAl2022_Asteraceae``) in every extraction mode
(``cds``, ``igs``, ``int``, ``rna``) with MAFFT, and compares *canonicalized*
outputs against a stored golden JSON (``tests/golden/asteraceae_parity.json``).

Canonicalization (so that legitimate later changes do not trip the gate):

* per-locus aligned FASTA (``<out>/fasta/nucl_<locus>.aligned.fasta``): rows
  are sorted by taxon id (first header token) and sequences are upper-cased
  before SHA-256 hashing (the row order of the pre-v3 pipeline depends on
  process completion order and is therefore not reproducible);
* the concatenated supermatrix (``nucl_<N>concat.aligned.fasta``) is split
  into its partitions using ``nucl_<N>concat.partitions.txt``; rows are
  sorted by taxon id, sequences are upper-cased and the partitions are
  re-assembled in *alphabetical partition order* before hashing, so the hash
  does not depend on the (file-order dependent) region order. Per-partition
  lengths are stored too; the literal partition order and a hash that depends
  on it are stored as informational fields (compared only with
  ``--strict-order``);
* every locus also stores a hash of its *ungapped* sequences. If an aligned
  hash differs but the ungapped hash is identical, the report says so
  ("extraction identical, alignment differs"), which points at MAFFT input
  order sensitivity rather than extraction/data-path changes.

Subcommands::

    python benchmarking/golden_parity.py capture [--modes cds rna] [--repeats 2]
    python benchmarking/golden_parity.py check   [--modes cds rna] [--strict-order]

``capture`` must be run on a known-good revision *before* a refactor. It
writes the golden JSON. Because pre-v3 revisions read the input folder with a
bare ``os.listdir`` (filesystem-dependent order) and MAFFT output can depend
on input row order, ``capture`` runs the CLI with a ``sitecustomize`` shim
that sorts ``os.listdir`` (``--no-sort-listdir`` disables it); ``check``
does not use the shim by default, so it also verifies that the pipeline
itself is order deterministic. With ``--repeats R`` (default 2) every mode is run R
times; loci whose canonical hash differs between repeats are stored as
``volatile`` and are skipped (and reported) by ``check``.

``check`` re-runs the modes and diffs against the golden, printing which
loci/modes differ; the exit status is 0 only if every compared item matches.
If the dataset directory is absent the command prints ``SKIP`` and exits 0.

Environment: use the dev conda env (``plastburst-dev``: Python 3.11, MAFFT
7.526) so that the aligner version matches the golden; a MAFFT version
mismatch is printed as a warning because it can legitimately change
alignments.

Runtime (6C/12T laptop, ``-n 2``, 16 files): roughly 10-40 s per mode; the
``rna`` and ``int`` modes are the fastest. ``tests/test_golden_parity.py``
runs the ``rna`` mode only; run the full four-mode gate by hand::

    python benchmarking/golden_parity.py check

All work is done in a temporary directory that is removed afterwards
(``--keep`` keeps it for debugging).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = REPO_ROOT / "benchmarking" / "benchmark_YangEtAl2022_Asteraceae"
DEFAULT_GOLDEN = REPO_ROOT / "tests" / "golden" / "asteraceae_parity.json"
DEFAULT_NUM_FILES = 16
DEFAULT_THREADS = 2
DEFAULT_REPEATS = 2
SCHEMA_VERSION = 1
FILE_EXT = ".gb"

ALL_MODES = ("cds", "igs", "int", "rna")

# Fixed per-mode CLI arguments: small enough to stay quick, non-trivial in
# locus count. `-l` is the minimum sequence length, `-t` the minimum number
# of taxa (of K=16) in which a region must occur.
MODE_ARGS: dict[str, list[str]] = {
    "cds": ["-l", "300", "-t", "12"],
    "igs": ["-l", "6", "-t", "12"],
    "int": ["-l", "6", "-t", "12"],
    "rna": ["-l", "30", "-t", "12"],
}


# ---------------------------------------------------------------------------
# FASTA canonicalization
# ---------------------------------------------------------------------------


def read_fasta(path: str | os.PathLike[str]) -> list[tuple[str, str]]:
    """Parse a FASTA file into (taxon id, sequence) tuples.

    The taxon id is the first whitespace-delimited header token. Sequence
    lines are concatenated; whitespace is dropped.
    """
    rows: list[tuple[str, str]] = []
    curr_id: str | None = None
    chunks: list[str] = []
    with open(path, encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = raw.strip()
            if not line:
                continue
            if line.startswith(">"):
                if curr_id is not None:
                    rows.append((curr_id, "".join(chunks)))
                    chunks = []
                header = line[1:].split()
                curr_id = header[0] if header else ""
            else:
                chunks.append(line)
    if curr_id is not None:
        rows.append((curr_id, "".join(chunks)))
    return rows


def _sha256_rows(rows: list[tuple[str, str]]) -> str:
    digest = hashlib.sha256()
    for taxon, seq in rows:
        digest.update(f">{taxon}\n{seq}\n".encode())
    return digest.hexdigest()


def canonical_rows(rows: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Rows sorted by taxon id (then sequence) with upper-cased sequences."""
    return sorted(((t, s.upper()) for t, s in rows), key=lambda r: (r[0], r[1]))


def canonical_alignment(rows: list[tuple[str, str]]) -> dict[str, Any]:
    """Canonical fingerprint of an alignment (order- and case-insensitive)."""
    canon = canonical_rows(rows)
    lengths = {len(seq) for _, seq in canon}
    ungapped = [(t, s.replace("-", "").replace(".", "")) for t, s in canon]
    return {
        "sha256": _sha256_rows(canon),
        "ungapped_sha256": _sha256_rows(ungapped),
        "ntax": len(canon),
        "nchar": max(lengths) if lengths else 0,
        "ragged": len(lengths) > 1,
    }


_PARTITION_RE = re.compile(r"^\s*[^,]+,\s*(?P<name>\S+)\s*=\s*(\d+)-(\d+)\s*$")


def read_partitions(path: str | os.PathLike[str]) -> list[tuple[str, int, int]]:
    """Parse a RAxML-NG partition file into (name, start0, end0_exclusive)."""
    parts: list[tuple[str, int, int]] = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            match = _PARTITION_RE.match(line)
            if match:
                parts.append(
                    (match.group("name"), int(match.group(2)) - 1, int(match.group(3)))
                )
    return parts


def canonical_concat(
    fasta_path: str | os.PathLike[str], partitions_path: str | os.PathLike[str] | None
) -> dict[str, Any]:
    """Canonical fingerprint of a concatenated supermatrix plus its partitions."""
    rows = canonical_rows(read_fasta(fasta_path))
    lengths = {len(seq) for _, seq in rows}
    info: dict[str, Any] = {
        "ntax": len(rows),
        "nchar": max(lengths) if lengths else 0,
        "ragged": len(lengths) > 1,
        "ordered_sha256": _sha256_rows(rows),
    }
    parts: list[tuple[str, int, int]] = []
    if partitions_path is not None and os.path.exists(partitions_path):
        parts = read_partitions(partitions_path)

    info["partition_order"] = [name for name, _, _ in parts]
    info["partitions"] = {name: end - start for name, start, end in parts}
    covered = 0
    contiguous = True
    for _, start, end in parts:
        contiguous = contiguous and start == covered
        covered = end
    info["partitions_cover_matrix"] = (
        bool(parts) and contiguous and (covered == info["nchar"])
    )

    if parts:
        by_name = sorted(parts, key=lambda p: p[0])
        reordered: list[tuple[str, str]] = []
        for taxon, seq in rows:
            reordered.append((taxon, "".join(seq[s:e] for _, s, e in by_name)))
        info["sha256"] = _sha256_rows(reordered)
        info["partition_sha256"] = {
            name: _sha256_rows([(taxon, seq[s:e]) for taxon, seq in rows])
            for name, s, e in parts
        }
    else:
        info["sha256"] = info["ordered_sha256"]
        info["partition_sha256"] = {}
    return info


def canonicalize_run(out_dir: str | os.PathLike[str]) -> dict[str, Any]:
    """Canonical description of one finished CLI run's output directory."""
    out = Path(out_dir)
    fasta_dir = out / "fasta"
    loci: dict[str, Any] = {}
    locus_files: list[str] = []
    if fasta_dir.is_dir():
        for path in sorted(fasta_dir.glob("*.aligned.fasta")):
            locus_files.append(path.name)
            loci[path.name] = canonical_alignment(read_fasta(path))

    nexus_dir = out / "nexus"
    nexus_files = (
        sorted(p.name for p in nexus_dir.glob("*.nexus")) if nexus_dir.is_dir() else []
    )
    unalign_files = sorted(p.name for p in out.glob("*.unalign.fasta"))

    concat: dict[str, Any] | None = None
    concat_fastas = sorted(out.glob("nucl_*concat.aligned.fasta"))
    if concat_fastas:
        fasta_path = concat_fastas[0]
        parts_path = out / fasta_path.name.replace(".aligned.fasta", ".partitions.txt")
        concat = canonical_concat(fasta_path, parts_path)
        concat["nloci"] = int(re.search(r"nucl_(\d+)concat", fasta_path.name).group(1))  # type: ignore[union-attr]

    run_digest = hashlib.sha256()
    run_digest.update(
        json.dumps(
            {
                "loci": {k: v["sha256"] for k, v in loci.items()},
                "concat": concat["sha256"] if concat else None,
            },
            sort_keys=True,
        ).encode()
    )
    return {
        "locus_files": locus_files,
        "nexus_files": nexus_files,
        "unalign_files": unalign_files,
        "loci": loci,
        "concat": concat,
        "run_sha256": run_digest.hexdigest(),
    }


# ---------------------------------------------------------------------------
# Running the CLI
# ---------------------------------------------------------------------------

# Pre-v3 revisions read the input folder with a bare ``os.listdir``, whose order
# is filesystem dependent, and MAFFT output can depend on input row order. To
# make a capture from such a revision portable (and equal to what the
# order-deterministic v3 pipeline produces), ``capture`` runs the CLI with a
# ``sitecustomize`` shim that makes ``os.listdir`` return sorted names.
_SORTED_LISTDIR_SHIM = """\
import os as _os

_orig_listdir = _os.listdir


def _sorted_listdir(path="."):
    return sorted(_orig_listdir(path))


_os.listdir = _sorted_listdir
"""


def install_listdir_shim(shim_dir: Path) -> Path:
    shim_dir.mkdir(parents=True, exist_ok=True)
    (shim_dir / "sitecustomize.py").write_text(_SORTED_LISTDIR_SHIM, encoding="utf-8")
    return shim_dir


def dataset_files(dataset_dir: Path, num_files: int) -> list[Path]:
    """The first ``num_files`` GenBank files of the dataset, sorted by name."""
    return sorted(dataset_dir.glob(f"*{FILE_EXT}"), key=lambda p: p.name)[:num_files]


def stage_input(files: list[Path], in_dir: Path, reverse: bool = False) -> None:
    """Symlink (or copy) the selected files into ``in_dir``."""
    in_dir.mkdir(parents=True, exist_ok=True)
    for src in reversed(files) if reverse else files:
        dst = in_dir / src.name
        try:
            os.symlink(src.resolve(), dst)
        except OSError:
            shutil.copy2(src, dst)


def aligner_version() -> str:
    exe = shutil.which("mafft")
    if not exe:
        return "mafft-not-found"
    proc = subprocess.run([exe, "--version"], capture_output=True, text=True)
    return (proc.stderr or proc.stdout).strip().splitlines()[0]


def run_mode(
    mode: str,
    in_dir: Path,
    out_dir: Path,
    threads: int,
    tmp_dir: Path,
    timeout: int = 1800,
    shim_dir: Path | None = None,
) -> dict[str, Any]:
    """Run the real CLI for one mode and canonicalize its output."""
    cmd = [
        sys.executable,
        "-m",
        "plastburstalign",
        "-i",
        str(in_dir),
        "-o",
        str(out_dir),
        "-s",
        mode,
        "-n",
        str(threads),
        "-c",
        "-a",
        "mafft",
        *MODE_ARGS[mode],
    ]
    env = dict(os.environ)
    env["TMPDIR"] = str(tmp_dir)
    if shim_dir is not None:
        env["PYTHONPATH"] = os.pathsep.join(
            [str(shim_dir)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = time.perf_counter() - start
    result: dict[str, Any] = {
        "returncode": proc.returncode,
        "seconds": round(elapsed, 1),
        "args": cmd[3:],
    }
    if proc.returncode != 0:
        result["stderr_tail"] = proc.stderr.strip().splitlines()[-8:]
    result.update(canonicalize_run(out_dir))
    return result


def execute_modes(
    modes: list[str],
    dataset_dir: Path,
    num_files: int,
    threads: int,
    repeats: int,
    keep: bool,
    reverse_input: bool = False,
    sort_listdir: bool = False,
) -> tuple[dict[str, list[dict[str, Any]]], list[Path]]:
    """Run each mode ``repeats`` times in a fresh temp dir."""
    files = dataset_files(dataset_dir, num_files)
    work = Path(tempfile.mkdtemp(prefix="pba_golden_"))
    results: dict[str, list[dict[str, Any]]] = {}
    try:
        in_dir = work / "input"
        stage_input(files, in_dir, reverse=reverse_input)
        tmp_dir = work / "tmp"
        tmp_dir.mkdir()
        shim_dir = install_listdir_shim(work / "shim") if sort_listdir else None
        for mode in modes:
            results[mode] = []
            for rep in range(repeats):
                out_dir = work / f"out_{mode}_{rep}"
                print(
                    f"[golden] running mode={mode} rep={rep + 1}/{repeats} "
                    f"({len(files)} files, -n {threads}) ...",
                    flush=True,
                )
                results[mode].append(
                    run_mode(mode, in_dir, out_dir, threads, tmp_dir, shim_dir=shim_dir)
                )
                last = results[mode][-1]
                print(
                    f"[golden]   rc={last['returncode']} "
                    f"{len(last['locus_files'])} loci, {last['seconds']} s",
                    flush=True,
                )
    finally:
        if keep:
            print(f"[golden] kept work dir: {work}")
        else:
            shutil.rmtree(work, ignore_errors=True)
    return results, [work] if keep else []


# ---------------------------------------------------------------------------
# capture / check
# ---------------------------------------------------------------------------


def merge_repeats(runs: list[dict[str, Any]]) -> dict[str, Any]:
    """Collapse repeated runs of one mode into one golden entry."""
    base = runs[0]
    volatile: set[str] = set()
    for other in runs[1:]:
        for name in set(base["loci"]) | set(other["loci"]):
            a = base["loci"].get(name, {}).get("sha256")
            b = other["loci"].get(name, {}).get("sha256")
            if a != b:
                volatile.add(name)
    entry = dict(base)
    entry["volatile_loci"] = sorted(volatile)
    entry["repeats"] = len(runs)
    entry["concat_volatile"] = (
        len({(r["concat"] or {}).get("sha256") for r in runs}) > 1
    )
    return entry


def capture(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"SKIP: dataset directory not found: {dataset_dir}")
        return 0
    modes = list(args.modes)
    results, _ = execute_modes(
        modes,
        dataset_dir,
        args.num_files,
        args.threads,
        args.repeats,
        args.keep,
        sort_listdir=args.sort_listdir,
    )
    golden_path = Path(args.golden)
    existing: dict[str, Any] = {}
    if golden_path.exists() and args.modes_subset:
        existing = json.loads(golden_path.read_text(encoding="utf-8"))
    golden: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "description": (
            "Canonicalized outputs of the real PlastBurstAlign CLI on a fixed "
            "Asteraceae subset. See benchmarking/golden_parity.py."
        ),
        "dataset": DEFAULT_DATASET.name,
        "files": [p.name for p in dataset_files(dataset_dir, args.num_files)],
        "threads": args.threads,
        "sort_listdir_shim": args.sort_listdir,
        "git_head": _git_head(),
        "aligner_version": aligner_version(),
        "python_version": platform.python_version(),
        "modes": existing.get("modes", {}),
    }
    for mode in modes:
        golden["modes"][mode] = merge_repeats(results[mode])
    golden_path.parent.mkdir(parents=True, exist_ok=True)
    golden_path.write_text(
        json.dumps(golden, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )
    failed = [m for m in modes if golden["modes"][m]["returncode"] != 0]
    for mode in modes:
        entry = golden["modes"][mode]
        print(
            f"[golden] {mode}: rc={entry['returncode']} "
            f"loci={len(entry['locus_files'])} "
            f"concat={(entry['concat'] or {}).get('ntax')}x"
            f"{(entry['concat'] or {}).get('nchar')} "
            f"volatile={len(entry['volatile_loci'])} "
            f"run_sha256={entry['run_sha256'][:12]}"
        )
    print(f"[golden] wrote {golden_path}")
    return 1 if failed else 0


def _git_head() -> str:
    try:
        proc = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return proc.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def diff_mode(
    mode: str,
    golden: dict[str, Any],
    current: dict[str, Any],
    strict_order: bool = False,
) -> list[str]:
    """Return human-readable differences (empty list = parity)."""
    problems: list[str] = []
    if current["returncode"] != 0:
        problems.append(
            f"CLI exited with status {current['returncode']}: "
            + " | ".join(current.get("stderr_tail", []))
        )
    volatile = set(golden.get("volatile_loci", []))

    g_files, c_files = set(golden["locus_files"]), set(current["locus_files"])
    for name in sorted(g_files - c_files):
        problems.append(f"locus missing in current run: {name}")
    for name in sorted(c_files - g_files):
        problems.append(f"locus not in golden (new): {name}")

    for name in sorted(g_files & c_files):
        if name in volatile:
            continue
        g, c = golden["loci"][name], current["loci"][name]
        if g["sha256"] == c["sha256"]:
            continue
        what = (
            "extraction identical, ALIGNMENT differs"
            if g["ungapped_sha256"] == c["ungapped_sha256"]
            else "extracted SEQUENCES differ"
        )
        problems.append(
            f"locus {name}: {what} "
            f"(ntax {g['ntax']}->{c['ntax']}, nchar {g['nchar']}->{c['nchar']})"
        )

    g_concat, c_concat = golden.get("concat"), current.get("concat")
    if (g_concat is None) != (c_concat is None):
        problems.append("concatenated supermatrix present in only one run")
    elif g_concat and c_concat:
        for key in ("ntax", "nchar"):
            if g_concat[key] != c_concat[key] and not volatile:
                problems.append(f"concat {key}: {g_concat[key]} -> {c_concat[key]}")
        gp, cp = g_concat["partitions"], c_concat["partitions"]
        if set(gp) != set(cp):
            problems.append(
                f"concat partition set differs: only golden {sorted(set(gp) - set(cp))}"
                f", only current {sorted(set(cp) - set(gp))}"
            )
        shared = sorted(set(gp) & set(cp))
        for name in shared:
            if name in volatile:
                continue
            if gp[name] != cp[name]:
                problems.append(
                    f"concat partition {name}: length {gp[name]} -> {cp[name]}"
                )
            elif g_concat["partition_sha256"].get(name) != c_concat[
                "partition_sha256"
            ].get(name):
                problems.append(f"concat partition {name}: column content differs")
        if not volatile and g_concat["sha256"] != c_concat["sha256"]:
            if not any(p.startswith("concat") for p in problems):
                problems.append("concat supermatrix hash differs (canonicalized)")
        if strict_order:
            if g_concat["partition_order"] != c_concat["partition_order"]:
                problems.append("concat partition ORDER differs (--strict-order)")
            elif (
                not volatile
                and g_concat["ordered_sha256"] != c_concat["ordered_sha256"]
            ):
                problems.append("concat ordered hash differs (--strict-order)")
        if not c_concat.get("partitions_cover_matrix", False):
            problems.append("partitions file does not tile the concatenated matrix")

    if golden["nexus_files"] != current["nexus_files"]:
        print(
            f"  note[{mode}]: individual NEXUS file set differs "
            f"({len(golden['nexus_files'])} vs {len(current['nexus_files'])})"
        )
    if golden["unalign_files"] != current["unalign_files"]:
        print(
            f"  note[{mode}]: unaligned FASTA file set differs "
            f"({len(golden['unalign_files'])} vs {len(current['unalign_files'])})"
        )
    if volatile:
        print(
            f"  note[{mode}]: {len(volatile)} loci were volatile in the golden "
            "capture and are not compared"
        )
    return problems


def check(args: argparse.Namespace) -> int:
    dataset_dir = Path(args.dataset)
    if not dataset_dir.is_dir():
        print(f"SKIP: dataset directory not found: {dataset_dir}")
        return 0
    golden_path = Path(args.golden)
    if not golden_path.exists():
        print(f"ERROR: golden file not found: {golden_path} (run `capture` first)")
        return 2
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    modes = [m for m in args.modes if m in golden["modes"]]
    missing = [m for m in args.modes if m not in golden["modes"]]
    if missing:
        print(f"ERROR: modes not in golden: {missing}")
        return 2

    expected_files = golden["files"]
    num_files = args.num_files or len(expected_files)
    actual_files = [p.name for p in dataset_files(dataset_dir, num_files)]
    if actual_files != expected_files:
        print("ERROR: dataset file selection differs from the golden capture")
        return 2

    current_ver = aligner_version()
    if current_ver != golden.get("aligner_version"):
        print(
            f"WARNING: aligner version {current_ver!r} differs from golden "
            f"{golden.get('aligner_version')!r}; differences may be legitimate"
        )

    results, _ = execute_modes(
        modes,
        dataset_dir,
        num_files,
        args.threads or golden.get("threads", DEFAULT_THREADS),
        1,
        args.keep,
        reverse_input=args.reverse_input,
        sort_listdir=args.sort_listdir,
    )
    exit_code = 0
    for mode in modes:
        problems = diff_mode(
            mode, golden["modes"][mode], results[mode][0], args.strict_order
        )
        if problems:
            exit_code = 1
            print(f"[golden] {mode}: FAIL ({len(problems)} difference(s))")
            for problem in problems:
                print(f"    - {problem}")
        else:
            cur = results[mode][0]
            print(
                f"[golden] {mode}: OK "
                f"({len(cur['locus_files'])} loci, "
                f"run_sha256={cur['run_sha256'][:12]})"
            )
    print("[golden] PARITY OK" if exit_code == 0 else "[golden] PARITY FAILED")
    return exit_code


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Golden parity gate for PlastBurstAlign (see module docstring)."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("capture", "check"):
        p = sub.add_parser(name, help=f"{name} the golden parity data")
        p.add_argument("--modes", nargs="+", choices=ALL_MODES, default=list(ALL_MODES))
        p.add_argument("--dataset", default=str(DEFAULT_DATASET))
        p.add_argument("--golden", default=str(DEFAULT_GOLDEN))
        p.add_argument(
            "--keep", action="store_true", help="keep the temporary work directory"
        )
        p.add_argument(
            "--sort-listdir",
            action=argparse.BooleanOptionalAction,
            default=(name == "capture"),
            help=(
                "run the CLI with a sitecustomize shim that sorts os.listdir "
                "(default: on for capture, off for check; see module docstring)"
            ),
        )
        if name == "capture":
            p.add_argument("--num-files", type=int, default=DEFAULT_NUM_FILES)
            p.add_argument("--threads", type=int, default=DEFAULT_THREADS)
            p.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
        else:
            p.add_argument("--num-files", type=int, default=None)
            p.add_argument("--threads", type=int, default=None)
            p.add_argument(
                "--strict-order",
                action="store_true",
                help="also compare the literal partition order of the supermatrix",
            )
            p.add_argument(
                "--reverse-input",
                action="store_true",
                help="stage the input symlinks in reverse order (stress test)",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.modes_subset = set(args.modes) != set(ALL_MODES)
    if args.command == "capture":
        return capture(args)
    return check(args)


if __name__ == "__main__":
    sys.exit(main())
