"""
Generic alignment tool interface and concrete implementations.
Supports multiple alignment tools (MAFFT, Clustal Omega, etc.)
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping, Sequence
from typing import IO, Any, Dict, List, NamedTuple, Optional

from .ephemeral_engine import EphemeralScratchManager
from .logging_ops import logger as log
from .strategy import (
    OVERRIDE_KEYS,
    AlignmentStrategy,
    AlignmentTier,
    determine_alignment_strategy,
    normalize_alphabet,
)
from .strategy_constants import DEFAULT_CONSTANTS
from .task_queue import pin_command


class AlignmentTimeout(TimeoutError):
    """Raised when an aligner subprocess exceeds its deadline."""

    def __init__(self, timeout: float, cmd: Sequence[str] | str) -> None:
        self.timeout = float(timeout)
        if isinstance(cmd, str):
            cmd_str = cmd
        else:
            cmd_str = " ".join(str(part) for part in cmd)
        self.cmd = cmd_str
        super().__init__(f"alignment timed out after {self.timeout:.1f}s: {cmd_str}")


def _kill_process_tree(proc: subprocess.Popen[Any]) -> None:
    """SIGTERM then SIGKILL the process group / tree (plastbench-style)."""
    if proc.pid is None:
        return
    try:
        import psutil

        try:
            parent = psutil.Process(proc.pid)
            victims = [*parent.children(recursive=True), parent]
        except psutil.Error:
            victims = []
        for victim in victims:
            try:
                victim.terminate()
            except psutil.Error:
                pass
        _, alive = psutil.wait_procs(victims, timeout=5) if victims else ([], [])
        for victim in alive:
            try:
                victim.kill()
            except psutil.Error:
                pass
    except Exception:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass


def run_aligner_subprocess(
    cmd: Sequence[str] | str,
    *,
    stdout: Any = subprocess.DEVNULL,
    stderr: Any = subprocess.PIPE,
    text: bool = True,
    env: Mapping[str, str] | None = None,
    timeout: float | None = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[Any]:
    """Run an aligner command; on timeout kill the tree and raise AlignmentTimeout."""
    popen_cmd: Any = cmd if shell or isinstance(cmd, str) else list(cmd)
    proc = subprocess.Popen(
        popen_cmd,
        stdout=stdout,
        stderr=stderr,
        text=text,
        env=dict(env) if env is not None else None,
        shell=shell,
        start_new_session=True,
    )
    try:
        out, err = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _kill_process_tree(proc)
        try:
            out, err = proc.communicate(timeout=5)
        except Exception:
            out, err = getattr(exc, "stdout", None), getattr(exc, "stderr", None)
        raise AlignmentTimeout(float(timeout or 0.0), cmd) from exc
    return subprocess.CompletedProcess(
        args=cmd if not isinstance(cmd, list) else list(cmd),
        returncode=proc.returncode,
        stdout=out,
        stderr=err,
    )


#: Translation table dropping everything that is not a residue (stop, gaps, space).
_NON_RESIDUES = str.maketrans("", "", "*-. \t\r\n")

# Resolver kinds understood by ``get_alignment_tool``. ``generic`` / unknown
# names stay on ``GenericAlignmentTool`` and never fall through to the builtin.
_RESOLVER_KINDS = {
    "mafft": "mafft",
    "clustal": "clustalo",
    "clustalo": "clustalo",
    "clustal-omega": "clustalo",
    "muscle": "muscle",
}


def mafft_auto_explicit_args(n_seq: int, max_len: int) -> tuple[str, ...]:
    """Explicit MAFFT 7.526 flags equivalent to ``--auto`` (never contains ``--auto``).

    The decision tree is the one in the MAFFT 7.526 ``mafft`` wrapper
    (``nlen`` / ``nseq`` branches around the ``auto=1`` block). It was checked
    by running that binary on a simulated 8-sequence locus:

    * 8000 bp, ``mafft --auto`` prints ``FFT-NS-i (Standard)``, and so does
      ``mafft --retree 2 --maxiterate 2``;
    * 2500 bp, ``mafft --auto`` prints ``L-INS-i``.

    Chunk alignments must be pinned to the full-locus flags. Passing ``--auto``
    to a ~2.5 kb chunk would silently switch an 8 kb FFT-NS-i locus to L-INS-i.
    """
    nlen = max(0, int(max_len))
    nseq = max(0, int(n_seq))
    if nlen < 3000 and nseq < 100:
        return ("--localpair", "--maxiterate", "1000")
    if nlen < 1000 and nseq < 200:
        return ("--localpair", "--maxiterate", "2")
    if nlen < 10000 and nseq < 500:
        return ("--retree", "2", "--maxiterate", "2")
    if nseq < 200000:
        # FFT-NS-2. Above 20k sequences MAFFT also enables its memsave guide
        # tree; the algorithm flags stay ``--retree 2`` until 100k, then
        # ``--retree 1``.
        return ("--retree", "1") if nseq >= 100000 else ("--retree", "2")
    if nlen < 3000:
        return ("--parttree",)
    return ("--parttree", "--retree", "1")


def mafft_auto_is_fft(n_seq: int, max_len: int) -> bool:
    """True when ``--auto`` would *not* select a pairwise (L/G/E-INS) algorithm.

    ``subdivide=auto`` chunks only in this regime: that is the 8 kb measurement
    above, where the full locus is FFT-NS-i and a short chunk would otherwise
    become L-INS-i. Shorter loci that ``--auto`` already aligns with L-INS-i
    stay whole.
    """
    flags = mafft_auto_explicit_args(n_seq, max_len)
    return not any(
        flag in flags for flag in ("--localpair", "--globalpair", "--genafpair")
    )


def _read_fasta_records(path: str) -> list[tuple[str, str]]:
    """``[(header_line, sequence), ...]`` of a FASTA file (header without '>')."""
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


def _write_fasta_records(path: str, records: Iterable[tuple[str, str]]) -> None:
    """Write ``(name, sequence)`` records, one line per sequence."""
    with open(path, "w", encoding="utf-8") as handle:
        for name, seq in records:
            handle.write(f">{name}\n{seq}\n")


def _read_aligned_fasta(path: str) -> dict[str, str]:
    """``{first header token: aligned row}`` of a FASTA alignment."""
    rows: dict[str, list[str]] = {}
    current: list[str] | None = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                key = line[1:].split(None, 1)[0] if len(line) > 1 else ""
                current = rows.setdefault(key, [])
            elif current is not None:
                current.append(line)
    return {key: "".join(parts) for key, parts in rows.items()}


def _write_wrapped(handle: IO[str], header: str, row: str, width: int = 60) -> None:
    """Write one FASTA record wrapped like MAFFT does (60 columns)."""
    handle.write(f">{header}\n")
    for start in range(0, len(row), width):
        handle.write(row[start : start + width] + "\n")


class AlignmentTool(ABC):
    """
    Abstract base class for alignment tools.
    Defines the interface that all alignment tool implementations must follow.
    """

    def __init__(
        self,
        user_params: Optional[Dict[str, Any]] = None,
        tool_path: Optional[str] = None,
    ):
        """
        Initialize the alignment tool.

        Args:
            user_params: User parameters dict containing configuration
            tool_path: Optional path to the alignment tool executable
        """
        self.user_params = user_params or {}
        self.exec_path: str | None = None
        # self.num_threads = self._set_num_threads(user_params)
        # self.num_threads = "1"  # For debugging, set to 1 to avoid multiprocessing issues"
        self.tool_params = self._get_tool_params(user_params)
        custom_scratch = self.tool_params.get(
            "scratch_dir", self.user_params.get("scratch_dir")
        )
        self.scratch_mgr = EphemeralScratchManager(custom_scratch_dir=custom_scratch)
        self.resolved_version: str | None = None
        self.resolved_path: str | None = None

        if tool_path:
            self._validate_tool_path(tool_path)
        else:
            self._find_tool()

        self._log_configuration()

    @abstractmethod
    def _find_tool(self):
        """Find or locate the alignment tool."""
        pass

    @abstractmethod
    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
    ) -> bool:
        """Align sequences from input_file and write result to output_file.

        Args:
            input_file: Path to unaligned FASTA.
            output_file: Destination path for aligned FASTA.
            threads: Optional dynamic thread count to override default
              configuration.

        Returns:
            True if alignment succeeded, False otherwise.
        """
        pass

    def execute(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
    ) -> bool:
        """Alias for align() for backward compatibility."""
        return self.align(
            input_file, output_file, threads=threads, sequence_count=sequence_count
        )

    def _get_tool_params(self, user_params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract tool-specific parameters from user parameters."""
        if user_params and isinstance(user_params, dict):
            tool_config = user_params.get("alignment_tool_config", {})
            return tool_config
        return {}

    def _validate_tool_path(self, tool_path: str):
        """Validate that the tool path exists."""
        if os.path.exists(tool_path) and os.access(tool_path, os.X_OK):
            self.exec_path = tool_path
            log.info(f"  using specified tool path: {tool_path}")
        else:
            log.error(f"  specified tool path is not executable: {tool_path}")
            self.exec_path = None

    def _get_subprocess_env(self) -> dict[str, str]:
        """Provides subprocess environment with TMPDIR mapped to ephemeral RAM-disk."""
        return self.scratch_mgr.get_subprocess_env()

    def _log_configuration(self):
        """Log the tool configuration."""
        tool_name = self.__class__.__name__
        if self.exec_path:
            log.info(f"configured {tool_name} with executable: {self.exec_path}")
        else:
            log.error(f" failed to configure {tool_name}")

    def _validate_input(self, input_file: str, output_file: str) -> bool:
        """Validate input and output files."""
        if not self.exec_path:
            log.error("alignment tool is not properly configured")
            return False
        if not os.path.exists(input_file) or not isinstance(input_file, str):
            log.error(f"input file does not exist: {input_file}")
            return False
        if not isinstance(output_file, str):
            log.error(f"output file path is invalid: {output_file}")
            return False
        return True


class _ResolvedStrategy(NamedTuple):
    """Sequence statistics and strategy resolved for one MAFFT call."""

    n_seq: int
    max_len: int
    min_len: int | None
    strategy: AlignmentStrategy


class MAFFT(AlignmentTool):
    """MAFFT alignment tool with a deterministic, work-based strategy.

    The algorithm comes from ``strategy.determine_alignment_strategy`` and is a
    pure function of ``(N, L_min, L_max, alphabet)`` plus explicit overrides;
    hardware never influences it. When a PartTree tier is chosen and the locus
    contains sequences shorter than 6 residues (PartTree aborts on those),
    :meth:`align` aligns the others with the tier and adds the short ones with
    ``mafft --addfragments`` (``--add`` for very large N, see
    :meth:`_align_with_fragments`) instead of falling back to the O(N^2)
    ``--auto``.
    """

    def _find_tool(self):
        """Search for MAFFT in system PATH."""
        if shutil.which("mafft"):
            self.exec_path = "mafft"
            log.info("found MAFFT in system PATH")
        else:
            log.info("MAFFT not found in system PATH")

    @staticmethod
    def _scan_fasta_meta(input_file: str) -> tuple[int, int, int]:
        """Fast binary scan to count records and find min and max sequence length
        without parsing BioPython objects. Excludes stop asterisks (*), gaps (-, .),
        and whitespace. Returns (record_count, min_seq_length, max_seq_length)."""
        count = 0
        min_len = float("inf")
        max_len = 0
        curr_len = 0
        in_seq = False
        non_residues = bytes(
            [42, 32, 9, 13, 10, 45, 46]
        )  # ASCII for * space \t \r \n - .
        try:
            with open(input_file, "rb") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith(b">"):
                        count += 1
                        if in_seq:
                            if curr_len < min_len:
                                min_len = curr_len
                            if curr_len > max_len:
                                max_len = curr_len
                            curr_len = 0
                        in_seq = True
                    else:
                        cleaned = line.translate(None, non_residues)
                        curr_len += len(cleaned)
                if in_seq:
                    if curr_len < min_len:
                        min_len = curr_len
                    if curr_len > max_len:
                        max_len = curr_len
        except OSError:
            pass
        if min_len == float("inf"):
            min_len = 0
        return count, int(min_len), int(max_len)

    @staticmethod
    def _count_sequences_fast(input_file: str) -> int:
        """Backward-compatible helper returning sequence count."""
        count, _, _ = MAFFT._scan_fasta_meta(input_file)
        return count

    # ------------------------------------------------------------------
    # strategy resolution
    # ------------------------------------------------------------------

    def _adaptive_enabled(self) -> bool:
        """Whether adaptive heuristics are on (default: True)."""
        return bool(
            self.tool_params.get(
                "adaptive_heuristics",
                self.user_params.get("adaptive_heuristics", True),
            )
        )

    def _explicit_algorithm(self) -> str | None:
        """The explicitly requested algorithm (anything but ``auto``), if any."""
        explicit = self.tool_params.get("algorithm")
        if explicit and explicit != "auto":
            return str(explicit)
        return None

    def _strategy_overrides(self) -> dict[str, Any]:
        """Strategy overrides explicitly present in the tool configuration."""
        return {
            key: self.tool_params[key]
            for key in sorted(OVERRIDE_KEYS)
            if self.tool_params.get(key) is not None
        }

    @staticmethod
    def _detect_alphabet(input_file: str, alphabet: str | None) -> str:
        """``"prot"`` or ``"nucl"``: the explicit alphabet, else the file prefix."""
        if alphabet is not None:
            return normalize_alphabet(alphabet)
        return "prot" if os.path.basename(input_file).startswith("prot_") else "nucl"

    def _resolve_strategy(
        self,
        input_file: str,
        sequence_count: int | None,
        max_len: int | None,
        min_len: int | None,
        alphabet: str | None,
    ) -> _ResolvedStrategy:
        """Fill in unknown sequence statistics (one scan at most) and decide.

        A single scan of ``input_file`` supplies whatever the caller did not
        pass. ``min_len`` is only needed for the short-sequence guard and is
        therefore only looked up when a PartTree tier was chosen.
        """
        n_seq, max_l, min_l = sequence_count, max_len, min_len
        file_exists = os.path.exists(input_file)
        if (n_seq is None or max_l is None) and file_exists:
            scanned_count, scanned_min, scanned_max = self._scan_fasta_meta(input_file)
            if n_seq is None:
                n_seq = scanned_count
            if max_l is None:
                max_l = scanned_max
            if min_l is None:
                min_l = scanned_min
        n = n_seq if n_seq is not None else 0
        longest = max_l if max_l is not None else 0
        alpha = self._detect_alphabet(input_file, alphabet)
        overrides = self._strategy_overrides()
        strategy = determine_alignment_strategy(
            n, longest, min_l, alphabet=alpha, overrides=overrides
        )
        if strategy.uses_parttree and min_l is None and file_exists:
            # Counts were supplied without a minimum length: scan once for it.
            min_l = self._scan_fasta_meta(input_file)[1]
            strategy = determine_alignment_strategy(
                n, longest, min_l, alphabet=alpha, overrides=overrides
            )
        return _ResolvedStrategy(n, longest, min_l, strategy)

    # ------------------------------------------------------------------
    # command assembly
    # ------------------------------------------------------------------

    def _assemble_command(
        self,
        algorithm_args: Sequence[str],
        input_file: str,
        threads: int | None,
        alphabet: str,
    ) -> list[str]:
        """``mafft <algorithm> --thread T [--adjustdirection] [--quiet] <input>``."""
        cmd: list[str] = [self.exec_path or "mafft", *algorithm_args]
        # Prioritize dynamically leased threads, falling back to tool_params
        mafft_threads = (
            threads if threads is not None else self.tool_params.get("threads", 1)
        )
        cmd.extend(["--thread", str(mafft_threads)])
        # adjustdirection is strictly nucleotide-only and only applied if enabled
        if alphabet != "prot" and self.tool_params.get("adjustdirection", False):
            cmd.append("--adjustdirection")
        if self.tool_params.get("quiet"):
            cmd.append("--quiet")
        seed = self.user_params.get("seed")
        if seed is not None:
            cmd.extend(["--randomseed", str(int(seed))])
        cmd.append(input_file)
        return cmd

    def _explicit_algorithm_args(self, explicit_algorithm: str) -> list[str]:
        """MAFFT arguments of an explicitly requested algorithm."""
        args: list[str] = []
        if explicit_algorithm == "localpair":
            args.append("--localpair")
        elif explicit_algorithm == "globalpair":
            args.append("--globalpair")
        elif explicit_algorithm == "genafpair":
            args.append("--genafpair")
        elif explicit_algorithm == "retree":
            args.extend(["--retree", str(self.tool_params.get("retree", 1))])
        elif explicit_algorithm == "maxiterate":
            args.extend(["--maxiterate", str(self.tool_params.get("maxiterate", 0))])
        return args

    def _build_command(
        self,
        input_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        max_len: int | None = None,
        min_len: int | None = None,
        alphabet: str | None = None,
    ) -> list[str]:
        """Build one complete MAFFT command line.

        The algorithm comes from the work-based strategy (see class docstring);
        an explicit ``algorithm`` in the tool configuration or
        ``adaptive_heuristics=False`` bypasses it. ``sequence_count``,
        ``max_len`` and ``min_len`` (shortest sequence) may be supplied by the
        caller (for example from the feature store's region summary) to avoid
        scanning ``input_file``; ``alphabet`` (``"nucl"``/``"prot"``) defaults
        to the ``prot_`` filename-prefix detection.

        A single command cannot express the short-sequence path (PartTree tier
        plus ``--addfragments``); for such inputs this method returns the
        legacy ``--auto`` bypass command. :meth:`align` never uses it: it runs
        :meth:`_align_with_fragments` instead.
        """
        alpha = self._detect_alphabet(input_file, alphabet)
        explicit = self._explicit_algorithm()
        if explicit:
            # If an algorithm was explicitly requested, honor it directly
            algorithm_args = self._explicit_algorithm_args(explicit)
        elif not self._adaptive_enabled():
            algorithm_args = ["--auto"]
        else:
            resolved = self._resolve_strategy(
                input_file, sequence_count, max_len, min_len, alpha
            )
            strategy = resolved.strategy
            if strategy.needs_fragment_split:
                log.warning(
                    f"      adaptive heuristic: tier {strategy.tier.name} (N="
                    f"{resolved.n_seq}, max_len={resolved.max_len}, "
                    f"W={strategy.work:.2e}) needs the short-sequence path (min "
                    f"length={resolved.min_len} < 6); a single command can only "
                    "use --auto (align() uses --addfragments instead)."
                )
                algorithm_args = ["--auto"]
            else:
                algorithm_args = list(strategy.mafft_args)
                if strategy.tier is not AlignmentTier.ACCURATE:
                    log.info(
                        f"      adaptive heuristic (N={resolved.n_seq}, "
                        f"max_len={resolved.max_len}, W={strategy.work:.2e}): tier "
                        f"{strategy.tier.name} ({' '.join(algorithm_args)})"
                    )
        return self._assemble_command(algorithm_args, input_file, threads, alpha)

    # ------------------------------------------------------------------
    # running
    # ------------------------------------------------------------------

    def _run_to_file(
        self,
        cmd: list[str],
        output_file: str,
        env: Mapping[str, str] | None = None,
        timeout: float | None = None,
    ) -> bool:
        """Run ``cmd`` with stdout redirected to ``output_file``.

        Raises :class:`AlignmentTimeout` when ``timeout`` is exceeded.
        """
        with open(output_file, "w") as out_handle:
            process = run_aligner_subprocess(
                cmd,
                stdout=out_handle,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(env) if env is not None else self._get_subprocess_env(),
                timeout=timeout,
            )
        if process.returncode == 0:
            return True
        err_msg = (process.stderr or "").strip()
        cmd_str = " ".join(cmd)
        log.error(
            f"    MAFFT failed [cmd: {cmd_str}] (rc={process.returncode}):\n{err_msg}"
        )
        return False

    def _prepare_cmd(
        self,
        cmd: list[str],
        *,
        cpu_ids: Sequence[int] | None,
        pin: bool,
    ) -> list[str]:
        """Prefix ``taskset -c`` when the scheduler asked for pinning."""
        if pin:
            return pin_command(cmd, cpu_ids)
        return list(cmd)

    def _fragment_scratch(self, env: Mapping[str, str] | None) -> str:
        """Directory for the short-sequence split, inside the task TMPDIR when set."""
        base = None
        if env is not None and env.get("TMPDIR"):
            base = env["TMPDIR"]
        else:
            base = str(self.scratch_mgr.path)
        return tempfile.mkdtemp(prefix="pba_frag_", dir=base)

    def _align_with_fragments(
        self,
        input_file: str,
        output_file: str,
        threads: int | None,
        strategy: AlignmentStrategy,
        alphabet: str,
        *,
        cpu_ids: Sequence[int] | None = None,
        env: Mapping[str, str] | None = None,
        pin: bool = False,
        timeout: float | None = None,
    ) -> bool:
        """Align a locus that has sequences too short for PartTree.

        1. Split the input into *main* (>= 6 residues), *short* (1-5 residues)
           and *empty* (0 residues) sequences.
        2. Align the main sequences with the chosen PartTree tier.
        3. Add the short ones with ``mafft --addfragments short.fa main.aln``
           (or ``--add`` when the main set exceeds ``addfragments_max_n``
           sequences, because ``--addfragments`` needs about 25 B * N^2 of
           memory versus 4.4 B * N^2 for ``--add``). No ``--keeplength``: it
           would delete residues of the short sequences.
        4. Empty sequences (which MAFFT cannot read) become all-gap rows.
        5. Write the rows in the ORIGINAL input order with the original
           headers (so the output is deterministic and every ungapped row is
           the input sequence).

        Temporary files live in this aligner's scratch directory. With fewer
        than 2 main sequences PartTree cannot run (it crashes on 1): the whole
        set is then aligned with ``--auto`` if N is below the fixed N cap and
        a ``ValueError`` is raised otherwise.
        """
        records = _read_fasta_records(input_file)
        n_total = len(records)
        min_ok = DEFAULT_CONSTANTS.min_parttree_seq_len
        main: list[int] = []
        short: list[int] = []
        empty: list[int] = []
        for i, (_, seq) in enumerate(records):
            n_res = len(seq.translate(_NON_RESIDUES))
            (empty if n_res == 0 else short if n_res < min_ok else main).append(i)

        if len(main) < 2:
            cap = int(
                self._strategy_overrides().get(
                    "n_force_parttree", DEFAULT_CONSTANTS.n_force_parttree
                )
            )
            if n_total >= cap:
                raise ValueError(
                    f"cannot align {os.path.basename(input_file)}: PartTree needs at "
                    f"least 2 sequences of >= {min_ok} residues but only "
                    f"{len(main)} of {n_total} qualify, and --auto is refused for "
                    f"N >= {cap} (O(N^2) memory)"
                )
            log.warning(
                f"      fewer than 2 sequences of >= {min_ok} residues in "
                f"{os.path.basename(input_file)}; using --auto for this small locus"
            )
            cmd = self._prepare_cmd(
                self._assemble_command(["--auto"], input_file, threads, alphabet),
                cpu_ids=cpu_ids,
                pin=pin,
            )
            return self._run_to_file(cmd, output_file, env=env, timeout=timeout)

        scratch = self._fragment_scratch(env)
        try:
            main_fa = os.path.join(scratch, "main.fa")
            short_fa = os.path.join(scratch, "short.fa")
            main_aln = os.path.join(scratch, "main.aln")
            combined_aln = os.path.join(scratch, "combined.aln")
            _write_fasta_records(main_fa, ((f"m{i}", records[i][1]) for i in main))
            _write_fasta_records(short_fa, ((f"m{i}", records[i][1]) for i in short))

            cmd = self._prepare_cmd(
                self._assemble_command(strategy.mafft_args, main_fa, threads, alphabet),
                cpu_ids=cpu_ids,
                pin=pin,
            )
            log.info(
                f"      short-sequence path: aligning {len(main)} sequences with "
                f"{strategy.tier.name}, then adding {len(short)} short sequence(s)"
            )
            if not self._run_to_file(cmd, main_aln, env=env, timeout=timeout):
                return False
            aligned_path = main_aln
            if short:
                cap_add = int(
                    self.tool_params.get(
                        "addfragments_max_n", DEFAULT_CONSTANTS.addfragments_max_n
                    )
                )
                add_flag = "--addfragments" if len(main) <= cap_add else "--add"
                mafft_threads = (
                    threads
                    if threads is not None
                    else self.tool_params.get("threads", 1)
                )
                add_cmd = [self.exec_path or "mafft", add_flag, short_fa]
                add_cmd.extend(["--thread", str(mafft_threads)])
                if self.tool_params.get("quiet"):
                    add_cmd.append("--quiet")
                add_cmd.append(main_aln)
                add_cmd = self._prepare_cmd(add_cmd, cpu_ids=cpu_ids, pin=pin)
                if not self._run_to_file(
                    add_cmd, combined_aln, env=env, timeout=timeout
                ):
                    if add_flag == "--addfragments":
                        log.warning(
                            "    --addfragments failed/segfaulted on short fragments; falling back to --add"
                        )
                        fallback_cmd = [self.exec_path or "mafft", "--add", short_fa]
                        fallback_cmd.extend(["--thread", str(mafft_threads)])
                        if self.tool_params.get("quiet"):
                            fallback_cmd.append("--quiet")
                        fallback_cmd.append(main_aln)
                        fallback_cmd = self._prepare_cmd(
                            fallback_cmd, cpu_ids=cpu_ids, pin=pin
                        )
                        if not self._run_to_file(
                            fallback_cmd, combined_aln, env=env, timeout=timeout
                        ):
                            return False
                    else:
                        return False
                aligned_path = combined_aln

            combined = _read_aligned_fasta(aligned_path)
            widths = {len(row) for row in combined.values()}
            if len(widths) > 1:
                raise ValueError("MAFFT returned rows of unequal length")
            width = widths.pop() if widths else 0
            mismatched = 0
            empty_set = set(empty)
            with open(output_file, "w") as out_handle:
                for i, (header, seq) in enumerate(records):
                    reverse_complemented = False
                    if i in empty_set:
                        row = "-" * width
                    elif f"m{i}" in combined:
                        row = combined[f"m{i}"]
                    elif f"_R_m{i}" in combined:  # reverse-complemented by MAFFT
                        row = combined[f"_R_m{i}"]
                        reverse_complemented = True
                    else:
                        raise ValueError(f"MAFFT output lacks the record {header!r}")
                    if (
                        not reverse_complemented
                        and row.replace("-", "").upper() != "".join(seq.split()).upper()
                    ):
                        mismatched += 1
                    _write_wrapped(
                        out_handle,
                        f"_R_{header}" if reverse_complemented else header,
                        row,
                    )
            if mismatched:
                log.warning(
                    f"      {mismatched} row(s) of {os.path.basename(output_file)} "
                    "differ from their input sequence after the short-sequence path"
                )
            return True
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        max_len: int | None = None,
        min_len: int | None = None,
        alphabet: str | None = None,
        algorithm_args: Sequence[str] | None = None,
        cpu_ids: Sequence[int] | None = None,
        env: Mapping[str, str] | None = None,
        pin: bool = False,
        timeout: float | None = None,
        **kwargs: Any,
    ) -> bool:
        """Perform alignment using MAFFT with dynamic threads and adaptive heuristics.

        `sequence_count`, `max_len` and `min_len` are optional hints (typically from
        the feature store) that spare the strategy resolution a scan of the input
        file. `alphabet` (``"nucl"``/``"prot"``) overrides the ``prot_`` filename
        detection. `algorithm_args` pins the algorithm (used for sub-locus chunks);
        it must not contain ``--auto``. `cpu_ids` / `pin` prefix ``taskset`` when
        the scheduler is pinning, and `env` is the child environment (``TMPDIR``).
        ``timeout`` is a subprocess wall-clock deadline in seconds.
        """
        if not self._validate_input(input_file, output_file):
            return False
        # kwargs may carry timeout from generic plumbing
        if timeout is None and kwargs.get("timeout") is not None:
            timeout = float(kwargs["timeout"])

        try:
            alpha = self._detect_alphabet(input_file, alphabet)
            if algorithm_args is not None:
                pinned = tuple(str(arg) for arg in algorithm_args)
                if "--auto" in pinned:
                    raise ValueError(
                        "chunk alignments must not pass --auto; pin the full-locus "
                        "algorithm instead"
                    )
                cmd = self._prepare_cmd(
                    self._assemble_command(pinned, input_file, threads, alpha),
                    cpu_ids=cpu_ids,
                    pin=pin,
                )
                if self._run_to_file(cmd, output_file, env=env, timeout=timeout):
                    log.info(
                        f"      successfully aligned {os.path.basename(input_file)} "
                        f"using MAFFT ({' '.join(cmd[1:])})"
                    )
                    return True
                return False
            if self._explicit_algorithm() is None and self._adaptive_enabled():
                resolved = self._resolve_strategy(
                    input_file, sequence_count, max_len, min_len, alpha
                )
                if resolved.strategy.needs_fragment_split:
                    log.info(
                        f"      adaptive heuristic (N={resolved.n_seq}, max_len="
                        f"{resolved.max_len}, W={resolved.strategy.work:.2e}): "
                        f"{resolved.strategy.reason}"
                    )
                    return self._align_with_fragments(
                        input_file,
                        output_file,
                        threads,
                        resolved.strategy,
                        alpha,
                        cpu_ids=cpu_ids,
                        env=env,
                        pin=pin,
                        timeout=timeout,
                    )
                # forward the resolved statistics so the command build never rescans
                sequence_count, max_len, min_len = (
                    resolved.n_seq,
                    resolved.max_len,
                    resolved.min_len,
                )
            cmd = self._prepare_cmd(
                self._build_command(
                    input_file,
                    threads=threads,
                    sequence_count=sequence_count,
                    max_len=max_len,
                    min_len=min_len,
                    alphabet=alpha,
                ),
                cpu_ids=cpu_ids,
                pin=pin,
            )
            if self._run_to_file(cmd, output_file, env=env, timeout=timeout):
                log.info(
                    f"      successfully aligned {os.path.basename(input_file)} using"
                    f" MAFFT ({' '.join(cmd[1:])})"
                )
                return True
            return False
        except AlignmentTimeout:
            raise
        except Exception as e:
            log.error(f"   error during MAFFT alignment: {e}")
            return False


class ClustalOmega(AlignmentTool):
    """Clustal Omega alignment tool implementation."""

    def _find_tool(self):
        """Search for Clustal Omega in system PATH."""
        if shutil.which("clustalo"):
            self.exec_path = "clustalo"
            log.info("  found Clustal Omega in system PATH")
        else:
            log.info("  Clustal Omega not found in system PATH")

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Perform alignment using Clustal Omega."""
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd: list[str] = [self.exec_path or "clustalo"]

            # Add input/output files
            cmd.extend(["--infile", input_file])
            cmd.extend(["--outfile", output_file])

            # Add output format (default: fasta)
            outfmt = self.tool_params.get("outfmt", "fasta")
            cmd.extend(["--outfmt", outfmt])

            clustalo_threads = (
                threads if threads is not None else self.tool_params.get("threads", 1)
            )
            cmd.extend(["--thread", str(clustalo_threads)])

            if self.tool_params.get("dealign"):
                cmd.append("--dealign")

            if self.tool_params.get("verbose"):
                cmd.append("--verbose")

            if kwargs.get("pin"):
                cmd = pin_command(cmd, kwargs.get("cpu_ids"))
            child_env = kwargs.get("env")
            timeout = kwargs.get("timeout")
            log.debug(f"   running Clustal Omega with command: {' '.join(cmd)}")
            process = run_aligner_subprocess(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(child_env) if child_env else self._get_subprocess_env(),
                timeout=float(timeout) if timeout is not None else None,
            )
            if process.returncode == 0:
                log.info(
                    f"      successfully aligned {os.path.basename(input_file)} using"
                    f" Clustal Omega with {clustalo_threads} threads"
                )
                return True
            else:
                log.error(
                    f"   Clustal Omega failed with return code {process.returncode}"
                )
                return False
        except AlignmentTimeout:
            raise
        except Exception as e:
            log.error(f"   error during Clustal Omega alignment: {e}")
            return False


class MUSCLE(AlignmentTool):
    """MUSCLE alignment tool implementation."""

    def _find_tool(self):
        """Search for MUSCLE in system PATH."""
        if shutil.which("muscle"):
            self.exec_path = "muscle"
            log.info("  found MUSCLE in system PATH")
        else:
            log.info("  MUSCLE not found in system PATH")

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        **kwargs: Any,
    ) -> bool:
        """Perform alignment using MUSCLE v5."""
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd: list[str] = [self.exec_path or "muscle"]

            # Add input and output files
            cmd.extend(["-super5", input_file])
            cmd.extend(["-output", output_file])

            # Add MUSCLE v5 options
            perm = self.tool_params.get("perm")
            if perm is not None:
                cmd.extend(["-perm", str(perm)])

            perturb = self.tool_params.get("perturb")
            if perturb is not None:
                cmd.extend(["-perturb", str(perturb)])

            eff_threads = (
                threads if threads is not None else self.tool_params.get("threads")
            )
            if eff_threads is not None and int(eff_threads) > 1:
                cmd.extend(["-threads", str(eff_threads)])

            if kwargs.get("pin"):
                cmd = pin_command(cmd, kwargs.get("cpu_ids"))
            child_env = kwargs.get("env")
            timeout = kwargs.get("timeout")
            log.debug(f"   running MUSCLE with command: {' '.join(cmd)}")
            process = run_aligner_subprocess(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(child_env) if child_env else self._get_subprocess_env(),
                timeout=float(timeout) if timeout is not None else None,
            )
            if process.returncode == 0:
                log.info(
                    f"      successfully aligned {os.path.basename(input_file)} using"
                    " MUSCLE"
                )
                return True
            else:
                log.error(f"   MUSCLE failed with return code {process.returncode}")
                return False
        except AlignmentTimeout:
            raise
        except Exception as e:
            log.error(f"   error during MUSCLE alignment: {e}")
            return False


class GenericAlignmentTool(AlignmentTool):
    """Generic command-line alignment tool wrapper."""

    def _find_tool(self):
        """Search for the tool name in system PATH if a path was not provided."""
        tool_name = self.user_params.get("alignment_tool")
        if tool_name and shutil.which(tool_name):
            self.exec_path = shutil.which(tool_name)
            log.info(f"  found {tool_name} in system PATH")
        else:
            log.info(f"  {tool_name} not found in system PATH")

    def _format_extra_args(self) -> str:
        """Convert configured extra args into a shell-safe string for templates."""
        extra_args = self.tool_params.get("extra_args")
        if isinstance(extra_args, list):
            return " ".join(str(arg) for arg in extra_args)
        if isinstance(extra_args, dict):
            parts: List[str] = []
            for key, value in extra_args.items():
                flag = str(key)
                if not flag.startswith("-"):
                    flag = f"-{flag}"
                parts.append(flag)
                if value is not None:
                    parts.append(str(value))
            return " ".join(parts)
        return ""

    def _append_extra_args(self, cmd: List[str]) -> List[str]:
        """Append user-configured extra arguments to the command."""
        extra_args = self.tool_params.get("extra_args")
        if isinstance(extra_args, list):
            cmd.extend([str(arg) for arg in extra_args])
        elif isinstance(extra_args, dict):
            for key, value in extra_args.items():
                flag = str(key)
                if not flag.startswith("-"):
                    flag = f"-{flag}"
                cmd.append(flag)
                if value is not None:
                    cmd.append(str(value))
        return cmd

    def _build_command(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
    ) -> list[str]:
        """Build the command line for a generic tool based on configuration."""
        cmd: list[str] = [
            self.exec_path or str(self.user_params.get("alignment_tool", ""))
        ]
        input_flag = self.tool_params.get("input_flag")
        output_flag = self.tool_params.get("output_flag")
        if input_flag:
            cmd.extend([input_flag, input_file])
        else:
            cmd.append(input_file)
        if output_flag:
            cmd.extend([output_flag, output_file])
        else:
            cmd.append(output_file)

        thread_flag = self.tool_params.get("thread_flag")
        eff_threads = (
            threads if threads is not None else self.tool_params.get("threads")
        )
        if thread_flag and eff_threads:
            cmd.extend([thread_flag, str(eff_threads)])

        cmd = self._append_extra_args(cmd)
        return cmd

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        **kwargs: Any,
    ) -> bool:
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd = self._build_command(input_file, output_file, threads=threads)
            if kwargs.get("pin"):
                cmd = pin_command(cmd, kwargs.get("cpu_ids"))
            child_env = kwargs.get("env")
            timeout = kwargs.get("timeout")
            log.debug(f"   running generic aligner with command: {' '.join(cmd)}")
            process = run_aligner_subprocess(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(child_env) if child_env else self._get_subprocess_env(),
                timeout=float(timeout) if timeout is not None else None,
            )
            if process.returncode == 0:
                log.info(
                    f"      successfully aligned {os.path.basename(input_file)} using"
                    f" {self.exec_path}"
                )
                return True
            else:
                log.error(
                    f"   generic aligner failed with return code {process.returncode}"
                )
                return False
        except AlignmentTimeout:
            raise
        except Exception as e:
            log.error(f"   error during generic alignment: {e}")
            return False


def get_alignment_tool(
    tool_name: str,
    user_params: Optional[Dict[str, Any]] = None,
    tool_path: Optional[str] = None,
) -> Optional[AlignmentTool]:
    """Build the aligner for this run.

    Known external tools go through :func:`aligner_resolver.resolve_aligner`.
    When that returns ``None`` (nothing on ``PATH``, or
    ``PBA_FORCE_BUILTIN_ALIGNER``), the built-in center-star aligner is used.
    An explicit command template still selects :class:`CustomAligner`.
    """
    user_params = user_params or {}
    custom_cmd = user_params.get("alignment_command") or user_params.get(
        "alignment_tool_config", {}
    ).get("command_template")

    # If an explicit command template is given, prioritize CustomAligner
    if custom_cmd or (
        isinstance(tool_name, str) and tool_name.lower().strip() == "custom"
    ):
        return CustomAligner(user_params, tool_path, command_template=custom_cmd)

    if not isinstance(tool_name, str) or not tool_name.strip():
        log.warning("no alignment tool specified; defaulting to mafft")
        tool_name = "mafft"
    else:
        tool_name = tool_name.lower().strip()

    kind = _RESOLVER_KINDS.get(tool_name)
    if kind is None:
        if tool_name != "generic":
            log.warning(
                f"unknown alignment tool: {tool_name}; attempting generic wrapper"
            )
        return GenericAlignmentTool(user_params, tool_path)

    from .aligner_resolver import resolve_aligner

    explicit = tool_path or user_params.get("alignment_tool_path")
    resolved = resolve_aligner(kind, explicit if isinstance(explicit, str) else None)
    if resolved is None:
        # Imported lazily: builtin_aligner imports AlignmentTool from this module.
        from .builtin_aligner import BuiltinAligner

        log.warning(
            f"no working {kind} executable found; using the built-in center-star "
            "aligner (lower accuracy; set PBA_FORCE_BUILTIN_ALIGNER to force this)"
        )
        aligner: AlignmentTool = BuiltinAligner(user_params)
        return aligner

    log.info(
        f"resolved {resolved.kind} {resolved.version} at {resolved.path} "
        f"({resolved.source})"
    )
    if kind == "mafft":
        tool: AlignmentTool = MAFFT(user_params, resolved.path)
    elif kind == "muscle":
        tool = MUSCLE(user_params, resolved.path)
    else:
        tool = ClustalOmega(user_params, resolved.path)
    tool.resolved_version = resolved.version
    tool.resolved_path = resolved.path
    return tool


class CustomAligner(AlignmentTool):
    """
    Executes sequence alignment via a user-defined command template.

    Supports placeholders:
      - %INPUT% / {input}: Path to input unaligned FASTA
      - %OUTPUT% / {output}: Path to output aligned FASTA
      - %THREADS% / {threads}: Dynamically budgeted threads for this invocation
    """

    def __init__(
        self,
        user_params: Optional[Dict[str, Any]] = None,
        tool_path: Optional[str] = None,
        command_template: Optional[str] = None,
    ):
        self.command_template = (
            command_template
            or (user_params or {}).get("alignment_command")
            or (user_params or {})
            .get("alignment_tool_config", {})
            .get("command_template")
        )
        if not self.command_template:
            raise ValueError("CustomAligner requires a non-empty command template.")
        super().__init__(user_params, tool_path)

    def _find_tool(self):
        """Custom commands specify their own binary; mark exec_path as valid."""
        self.exec_path = "custom_command"

    def _get_budgeted_threads(self) -> int:
        """
        Calculates per-worker thread budget to prevent CPU oversubscription:
            threads_per_worker = max(1, total_threads // num_workers)
        """
        total_threads = int(self.user_params.get("num_threads", 1))
        # If tool_params explicitly overrides thread count, honor it
        configured_threads = self.tool_params.get("threads")
        if configured_threads is not None:
            return max(1, int(configured_threads))
        return max(1, total_threads)

    def _build_command_string(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
    ) -> str:
        eff_threads = threads if threads is not None else self._get_budgeted_threads()
        cmd = self.command_template
        # Replace uppercase placeholders
        cmd = cmd.replace("%INPUT%", input_file)
        cmd = cmd.replace("%OUTPUT%", output_file)
        cmd = cmd.replace("%THREADS%", str(eff_threads))
        # Support brace format placeholders
        cmd = cmd.replace("{input}", input_file)
        cmd = cmd.replace("{output}", output_file)
        cmd = cmd.replace("{threads}", str(eff_threads))
        return cmd

    def align(
        self,
        input_file: str,
        output_file: str,
        threads: int | None = None,
        sequence_count: int | None = None,
        **kwargs: Any,
    ) -> bool:
        if not os.path.exists(input_file):
            log.error(f"Input file does not exist: {input_file}")
            return False

        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        cmd_str = self._build_command_string(input_file, output_file, threads=threads)
        log.info(f"Executing custom alignment command: {cmd_str}")

        try:
            child_env = kwargs.get("env")
            timeout = kwargs.get("timeout")
            process = run_aligner_subprocess(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=dict(child_env) if child_env else self._get_subprocess_env(),
                timeout=float(timeout) if timeout is not None else None,
            )

            if process.returncode != 0:
                log.error(
                    "Custom alignment failed with returncode"
                    f" {process.returncode}: {(process.stderr or '').strip()}"
                )
                return False

            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                log.error(
                    "Alignment completed but output file is missing or empty:"
                    f" {output_file}"
                )
                return False

            log.info(
                f"Successfully aligned {os.path.basename(input_file)} ->"
                f" {os.path.basename(output_file)}"
            )
            return True

        except AlignmentTimeout:
            raise
        except Exception as e:
            log.error(f"Exception encountered during custom alignment execution: {e}")
            return False
