"""
Generic alignment tool interface and concrete implementations.
Supports multiple alignment tools (MAFFT, Clustal Omega, etc.)
"""

import os
import shutil
import subprocess
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, List

from .logging_ops import logger as log


class AlignmentTool(ABC):
    """
    Abstract base class for alignment tools.
    Defines the interface that all alignment tool implementations must follow.
    """

    def __init__(
        self, user_params: Optional[Dict[str, Any]] = None, tool_path: Optional[str] = None
    ):
        """
        Initialize the alignment tool.

        Args:
            user_params: User parameters dict containing configuration
            tool_path: Optional path to the alignment tool executable
        """
        self.user_params = user_params or {}
        self.exec_path = None
        # self.num_threads = self._set_num_threads(user_params)
        # self.num_threads = "1"  # For debugging, set to 1 to avoid multiprocessing issues"
        self.tool_params = self._get_tool_params(user_params)

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
    def align(self, input_file: str, output_file: str) -> bool:
        """
        Perform sequence alignment.

        Args:
            input_file: Path to input sequence file
            output_file: Path to output alignment file

        Returns:
            True if alignment succeeded, False otherwise
        """
        pass

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


class MAFFT(AlignmentTool):
    """MAFFT alignment tool implementation."""

    def _find_tool(self):
        """Search for MAFFT in system PATH."""
        if shutil.which("mafft"):
            self.exec_path = "mafft"
            log.info("found MAFFT in system PATH")
        else:
            log.info("MAFFT not found in system PATH")

    def align(self, input_file: str, output_file: str) -> bool:
        """
        Perform alignment using MAFFT.

        Args:
            input_file: Path to input sequence file
            output_file: Path to output alignment file

        Returns:
            True if alignment succeeded, False otherwise
        """
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd = [self.exec_path]

            # Add algorithm option (default: --auto)
            algorithm = self.tool_params.get("algorithm", "auto")
            if algorithm == "auto":
                cmd.append("--auto")
            elif algorithm == "localpair":
                cmd.append("--localpair")
            elif algorithm == "globalpair":
                cmd.append("--globalpair")
            elif algorithm == "genafpair":
                cmd.append("--genafpair")
            elif algorithm == "retree":
                retree_num = self.tool_params.get("retree", 1)
                cmd.extend(["--retree", str(retree_num)])
            elif algorithm == "maxiterate":
                maxiterate_num = self.tool_params.get("maxiterate", 0)
                cmd.extend(["--maxiterate", str(maxiterate_num)])

            # Add thread option
            mafft_threads = self.tool_params.get("threads", 1)
            cmd.extend(["--thread", str(mafft_threads)])

            # Add other MAFFT options
            if self.tool_params.get("adjustdirection", True):
                cmd.append("--adjustdirection")

            if self.tool_params.get("quiet"):
                cmd.append("--quiet")

            # Add input file
            cmd.append(input_file)

            with open(output_file, "w") as out_handle:
                process = subprocess.run(
                    cmd, stdout=out_handle, stderr=subprocess.DEVNULL, text=True, check=False
                )
            if process.returncode == 0:
                log.info(
                    f"     successfully aligned {os.path.basename(input_file)} using MAFFT with {mafft_threads} threads"
                )
                return True
            else:
                log.error(f"  MAFFT failed with return code {process.returncode}")
                return False
        except Exception as e:
            log.error(f"  error during MAFFT alignment: {e}")
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

    def align(self, input_file: str, output_file: str) -> bool:
        """
        Perform alignment using Clustal Omega.

        Args:
            input_file: Path to input sequence file
            output_file: Path to output alignment file

        Returns:
            True if alignment succeeded, False otherwise
        """
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd = [self.exec_path]

            # Add threads

            # Add input/output files
            cmd.extend(["--infile", input_file])
            cmd.extend(["--outfile", output_file])

            # Add output format (default: fasta)
            outfmt = self.tool_params.get("outfmt", "fasta")
            cmd.extend(["--outfmt", outfmt])

            clustalo_threads = self.tool_params.get("threads", 1)
            cmd.extend(["--thread", str(clustalo_threads)])

            if self.tool_params.get("dealign"):
                cmd.append("--dealign")

            if self.tool_params.get("verbose"):
                cmd.append("--verbose")

            log.debug(f"  running Clustal Omega with command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, check=False
            )
            if process.returncode == 0:
                log.info(
                    f"     successfully aligned {os.path.basename(input_file)} using Clustal Omega with {clustalo_threads} threads"
                )
                return True
            else:
                log.error(f"  Clustal Omega failed with return code {process.returncode}")
                return False
        except Exception as e:
            log.error(f"  error during Clustal Omega alignment: {e}")
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

    def align(self, input_file: str, output_file: str) -> bool:
        """
        Perform alignment using MUSCLE v5.

        Args:
            input_file: Path to input sequence file
            output_file: Path to output alignment file

        Returns:
            True if alignment succeeded, False otherwise
        """
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd = [self.exec_path]

            # Add input and output files
            cmd.extend(["-super5", input_file])
            cmd.extend(["-output", output_file])

            # Add MUSCLE v5 options
            # Guide tree permutation (none, abc, acb, bca)
            perm = self.tool_params.get("perm")
            if perm is not None:
                cmd.extend(["-perm", str(perm)])

            # Perturbation seed
            perturb = self.tool_params.get("perturb")
            if perturb is not None:
                cmd.extend(["-perturb", str(perturb)])

            # Maximum gap fraction (0..1)
            # maxgapfract = self.tool_params.get("maxgapfract")
            # if maxgapfract is not None:
            #     cmd.extend(["-maxgapfract", str(maxgapfract)])

            # # Minimum column confidence (0..1)
            # minconf = self.tool_params.get("minconf")
            # if minconf is not None:
            #     cmd.extend(["-minconf", str(minconf)])

            log.debug(f"  running MUSCLE with command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, text=True, check=False
            )
            if process.returncode == 0:
                log.info(f"     successfully aligned {os.path.basename(input_file)} using MUSCLE")
                return True
            else:
                log.error(f"  MUSCLE failed with return code {process.returncode}")
                return False
        except Exception as e:
            log.error(f"  error during MUSCLE alignment: {e}")
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

    def _build_command(self, input_file: str, output_file: str) -> List[str]:
        """Build the command line for a generic tool based on configuration."""
        # cmd_template = self.tool_params.get("command_template")
        # if cmd_template:
        #     if isinstance(cmd_template, str):
        #         uses_extra_placeholder = "{extra_args}" in cmd_template
        #         formatted = cmd_template.format(
        #             tool=self.exec_path,
        #             input=input_file,
        #             output=output_file,
        #             extra_args=self._format_extra_args(),
        #         )
        #         cmd = shlex.split(formatted)
        #         if not uses_extra_placeholder:
        #             cmd = self._append_extra_args(cmd)
        #         print(f"  generic alignment command: {' '.join(cmd)}")
        #         return cmd
        #     elif isinstance(cmd_template, list):
        #         uses_extra_placeholder = any("{extra_args}" in str(item) for item in cmd_template)
        #         cmd = [
        #             str(item).format(
        #                 tool=self.exec_path,
        #                 input=input_file,
        #                 output=output_file,
        #                 extra_args=self._format_extra_args(),
        #             )
        #             for item in cmd_template
        #         ]
        #         if not uses_extra_placeholder:
        #             cmd = self._append_extra_args(cmd)
        #         print(f"  generic alignment command: {' '.join(cmd)}")
        #         return cmd

        cmd = [self.exec_path]
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

        cmd = self._append_extra_args(cmd)
        print(f"  generic alignment command: {' '.join(cmd)}")
        return cmd

    def align(self, input_file: str, output_file: str) -> bool:
        """
        Perform alignment using a generic external aligner.

        Args:
            input_file: Path to input sequence file
            output_file: Path to output alignment file

        Returns:
            True if alignment succeeded, False otherwise
        """
        if not self._validate_input(input_file, output_file):
            return False

        try:
            cmd = self._build_command(input_file, output_file)
            log.debug(f"  running generic aligner with command: {' '.join(cmd)}")
            process = subprocess.run(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
                check=False,
            )
            if process.returncode == 0:
                log.info(
                    f"     successfully aligned {os.path.basename(input_file)} using {self.exec_path}"
                )
                return True
            else:
                log.error(f"  generic aligner failed with return code {process.returncode}")
                return False
        except Exception as e:
            log.error(f"  error during generic alignment: {e}")
            return False


def get_alignment_tool(
    tool_name: str, user_params: Optional[Dict[str, Any]] = None, tool_path: Optional[str] = None
) -> Optional[AlignmentTool]:
    user_params = user_params or {}
    custom_cmd = user_params.get("alignment_command") or user_params.get(
        "alignment_tool_config", {}
    ).get("command_template")

    # If an explicit command template is given, prioritize CustomAligner
    if custom_cmd or (isinstance(tool_name, str) and tool_name.lower().strip() == "custom"):
        return CustomAligner(user_params, tool_path, command_template=custom_cmd)

    if not isinstance(tool_name, str) or not tool_name.strip():
        log.warning("no alignment tool specified; defaulting to mafft")
        tool_name = "mafft"
    else:
        tool_name = tool_name.lower().strip()

    if tool_name == "mafft":
        return MAFFT(user_params, tool_path)
    elif tool_name in ["clustal", "clustalo", "clustal-omega"]:
        return ClustalOmega(user_params, tool_path)
    elif tool_name == "muscle":
        return MUSCLE(user_params, tool_path)
    elif tool_name == "generic":
        return GenericAlignmentTool(user_params, tool_path)
    else:
        log.warning(f"unknown alignment tool: {tool_name}; attempting generic wrapper")
        return GenericAlignmentTool(user_params, tool_path)


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
            or (user_params or {}).get("alignment_tool_config", {}).get("command_template")
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

    def _build_command_string(self, input_file: str, output_file: str) -> str:
        threads = self._get_budgeted_threads()
        cmd = self.command_template
        # Replace uppercase placeholders
        cmd = cmd.replace("%INPUT%", input_file)
        cmd = cmd.replace("%OUTPUT%", output_file)
        cmd = cmd.replace("%THREADS%", str(threads))
        # Support brace format placeholders
        cmd = cmd.replace("{input}", input_file)
        cmd = cmd.replace("{output}", output_file)
        cmd = cmd.replace("{threads}", str(threads))
        return cmd

    def align(self, input_file: str, output_file: str) -> bool:
        if not os.path.exists(input_file):
            log.error(f"Input file does not exist: {input_file}")
            return False

        # Ensure parent output directory exists
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)

        cmd_str = self._build_command_string(input_file, output_file)
        log.info(f"Executing custom alignment command: {cmd_str}")

        try:
            # shell=True allows pipe/redirection syntax (e.g. mafft ... > output.fasta)
            process = subprocess.run(
                cmd_str,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if process.returncode != 0:
                log.error(
                    f"Custom alignment failed with returncode {process.returncode}: {process.stderr.strip()}"
                )
                return False

            if not os.path.exists(output_file) or os.path.getsize(output_file) == 0:
                log.error(f"Alignment completed but output file is missing or empty: {output_file}")
                return False

            log.info(
                f"Successfully aligned {os.path.basename(input_file)} -> {os.path.basename(output_file)}"
            )
            return True

        except Exception as e:
            log.error(f"Exception encountered during custom alignment execution: {e}")
            return False
