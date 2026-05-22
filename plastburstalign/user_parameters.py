import argparse
import multiprocessing
import os
from enum import Enum
from typing import List, Union, Any, Dict, Optional
import yaml

# Package imports
from .logging_ops import Logger, logger as log


class SelectMode(str, Enum):
    """
    An enumeration class used to keep track of selection and alignment modes.
    """
    CDS = "cds"
    INT = "int"
    IGS = "igs"

    def collects_proteins(self):
        return self == type(self).CDS


class UserParameters(Dict[str, Any]):
    def __init__(
            self,
            in_dir: str = "./input",
            out_dir: str = "./output",
            select_mode: str = "cds",
            fileext: str = ".gb",
            exclude_fullcds: Optional[List[str]] = None,
            exclude_region: Optional[List[str]] = None,
            min_seq_length: int = 3,
            min_num_taxa: int = 2,
            num_threads: Union[str, int] = "auto",
            verbose: bool = False,
            order: str = "seq",
            concat: bool = False,
            alignment_tool: str = "mafft",
            alignment_tool_path: Optional[str] = None,
            alignment_tool_config: Optional[Dict[str, Any]] = None,
            config_file: Optional[str] = None
    ):
        """
        A dictionary that validates user parameters that are to be used within the package.


        Args:
            in_dir: Path to input directory (which contains the GenBank files)
            out_dir: Path to output directory
            select_mode: Type of regions to be extracted (i.e. `cds`, `int`, or `igs`)
            fileext: File extension of input files
            exclude_fullcds: List of genes to be excluded (CDS as well as INT/IGS that involve these genes)
            exclude_region: List of regions to be excluded (specific regions by name).
                Using the parameter will lead to an exception for select mode `cds` as
                `exclude_fullcds` should be used for that purpose.
            min_seq_length: Minimal sequence length (in bp) below which regions will not be extracted
            min_num_taxa: Minimum number of taxa in which a region must be present to be extracted
            num_threads: Number of CPUs to use; can be any positive integer or 'auto'
            verbose: Enable verbose logging
            order: Order that the alignments should be saved ('seq' or 'alpha')
            concat: Enable sequential writing of concatenation files
            alignment_tool: Alignment tool to use ('mafft', 'clustal', etc.)
            alignment_tool_path: Path to the alignment tool executable
            alignment_tool_config: Dictionary of tool-specific parameters
            config_file: Path to a YAML config file containing parameters
        """
        super().__init__()
        self._set_select_mode(select_mode)
        self._set_in_dir(in_dir)
        self._set_out_dir(out_dir)
        self._set_fileext(fileext)
        self._set_exclude_fullcds(exclude_fullcds)
        self._set_exclude_region(exclude_region)
        self._set_min_seq_length(min_seq_length)
        self._set_min_num_taxa(min_num_taxa)
        self._set_num_threads(num_threads)
        self._set_verbose(verbose)
        self._set_order(order)
        self._set_concat(concat)
        self._set_alignment_tool(alignment_tool)
        self._set_alignment_tool_path(alignment_tool_path)
        self._set_alignment_tool_config(alignment_tool_config)
        self._set_config_file(config_file)

    def _set_select_mode(self, select_mode: str):
        try:
            self["select_mode"] = SelectMode(select_mode.lower())
        except ValueError:
            log.critical(f"Invalid selection mode provided: `{select_mode}`")
            raise ValueError

    def _set_in_dir(self, in_dir: str):
        if not os.path.exists(in_dir) or not isinstance(in_dir, str):
            log.critical(f"Input directory `{in_dir}` does not exist.")
            raise FileNotFoundError
        self["in_dir"] = in_dir

    def _set_out_dir(self, out_dir: str):
        if not os.path.exists(out_dir) or not isinstance(out_dir, str):
            log.critical(f"Output directory `{out_dir}` does not exist.")
            raise FileNotFoundError
        self["out_dir"] = out_dir

    def _set_fileext(self, fileext: str):
        self._test_type("fileext", fileext, str)
        self["fileext"] = fileext

    def _set_exclude_fullcds(self, exclude_fullcds: Optional[List[str]]):
        if not exclude_fullcds:
            exclude_fullcds = ["rps12"]
        self._test_type("exclude_fullcds", exclude_fullcds, list)
        self["exclude_fullcds"] = exclude_fullcds
        if self["select_mode"] == "igs":
            # Excluding matK is necessary, as matK is located inside trnK
            self["exclude_fullcds"].append("matK")

    def _set_exclude_region(self, exclude_region: Optional[List[str]]):
        if not exclude_region:
            exclude_region = []
        self._test_type("exclude_region", exclude_region, list)
        if self["select_mode"] == "cds" and exclude_region:
            log.critical("Excluded region list can not be used for select mode `cds`")
            raise ValueError
        self["exclude_region"] = exclude_region

    def _set_min_seq_length(self, min_seq_length: int):
        self._test_type("min_seq_length", min_seq_length, int)
        self["min_seq_length"] = min_seq_length

    def _set_min_num_taxa(self, min_num_taxa: int):
        self._test_type("min_num_taxa", min_num_taxa, int)
        self["min_num_taxa"] = min_num_taxa

    def _set_num_threads(self, num_threads: Union[str, int]):
        if isinstance(num_threads, str):
            num_threads_clean = num_threads.strip().lower()
            if num_threads_clean == "auto":
                try:
                    num_threads = os.cpu_count()
                except NotImplementedError:
                    num_threads = multiprocessing.cpu_count()
            elif num_threads_clean.isnumeric():
                num_threads = int(num_threads_clean)
            else:
                log.critical(f"Invalid `num_threads` value provided: {num_threads}")
                raise ValueError

        if num_threads is None:
            num_threads = 1

        self._test_type("num_threads", num_threads, int)
        if num_threads < 1:
            log.critical(f"Invalid `num_threads` value provided: {num_threads}")
            raise ValueError
        self["num_threads"] = num_threads

    def _set_verbose(self, verbose: bool):
        self._test_type("verbose", verbose, bool)
        self["verbose"] = verbose
        Logger.set_logger(verbose)

    def _set_order(self, order: str):
        self._test_type("order", order, str)
        if order == "alpha":
            self["order"] = order
        else:
            self["order"] = "seq"

    def _set_concat(self, concat: bool):
        self._test_type("concat", concat, bool)
        self["concat"] = concat

    def _set_alignment_tool(self, alignment_tool: str):
        self._test_type("alignment_tool", alignment_tool, str)
        self["alignment_tool"] = alignment_tool.lower()

    def _set_alignment_tool_path(self, alignment_tool_path: Optional[str]):
        if alignment_tool_path and not os.path.exists(alignment_tool_path):
            log.warning(f"Specified alignment tool path does not exist: {alignment_tool_path}")
        self["alignment_tool_path"] = alignment_tool_path

    def _set_alignment_tool_config(self, alignment_tool_config: Optional[Dict[str, Any]]):
        if alignment_tool_config is None:
            alignment_tool_config = {}
        self._test_type("alignment_tool_config", alignment_tool_config, dict)
        self["alignment_tool_config"] = alignment_tool_config

    def _set_config_file(self, config_file: Optional[str]):
        if config_file and not os.path.exists(config_file):
            log.warning(f"Specified config file does not exist: {config_file}")
        self["config_file"] = config_file

    @staticmethod
    def _test_type(param_name: str, param_val: Any, expected_type: Any):
        if not isinstance(param_val, expected_type):
            log.critical(f"Invalid `{param_name}` type provided: {type(param_val)}")
            raise TypeError


class UserParametersScript(UserParameters):
    _construct_name_map: Dict[str, str] = {
        "inpd": "in_dir",
        "outd": "out_dir",
        "selectmode": "select_mode",
        "fileext": "fileext",
        "exclcds": "exclude_fullcds",
        "exclreg": "exclude_region",
        "minseqlength": "min_seq_length",
        "minnumtaxa": "min_num_taxa",
        "numthreads": "num_threads",
        "verbose": "verbose",
        "order": "order",
        "concat": "concat",
        "alignmenttool": "alignment_tool",
        "alignmenttoolpath": "alignment_tool_path",
        "config": "config_file"
    }

    def __init__(self):
        """
        An interface to create `UserParameters` within the context of a script execution.
        """
        parser = self._get_parser()
        args = parser.parse_args()
        
        # Load config file if specified
        config_params = {}
        if hasattr(args, 'config') and args.config:
            config_params = self._load_config_file(args.config)
        
        # Merge command line args with config file, command line takes precedence
        args_dict = self._get_construct_dict(args)
        merged_params = {**config_params, **args_dict}
        
        super().__init__(**merged_params)

    @staticmethod
    def _load_config_file(config_path: str) -> Dict[str, Any]:
        """
        Load parameters from a YAML config file.
        
        Args:
            config_path: Path to the YAML config file
            
        Returns:
            Dictionary of parameters from config file
        """
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            log.info(f"Loaded configuration from {config_path}")
            return config or {}
        except Exception as e:
            log.error(f"Failed to load config file {config_path}: {e}")
            return {}

    @staticmethod
    def _get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--inpd",
            "-i",
            type=str,
            required=False,
            help="Path to input directory (which contains the GenBank files)",
        )
        parser.add_argument(
            "--outd",
            "-o",
            type=str,
            required=False,
            help="(Optional) Path to output directory",
        )
        parser.add_argument(
            "--selectmode",
            "-s",
            type=str,
            required=False,
            help="(Optional) Type of regions to be extracted (i.e. `cds`, `int`, or `igs`)",
        )
        parser.add_argument(
            "--fileext",
            "-f",
            type=str,
            required=False,
            help="(Optional) File extension of input files",
        )
        parser.add_argument(
            "--exclcds",
            "-e",
            type=str,
            nargs="+",
            required=False,
            help="(Optional) List of genes to be excluded (CDS as well as INT/IGS that involve these genes)",
        )
        parser.add_argument(
            "--exclreg",
            "-x",
            type=str,
            nargs="+",
            required=False,
            help=(
                "(Optional) List of regions to be excluded (specific regions by name). "
                "Using the argument will lead to an exception for select mode `cds` as "
                "`exclcds` should be used for that purpose."
            )
        )
        parser.add_argument(
            "--minseqlength",
            "-l",
            type=int,
            required=False,
            help="(Optional) Minimal sequence length (in bp) below which regions will not be extracted",
        )
        parser.add_argument(
            "--minnumtaxa",
            "-t",
            type=int,
            required=False,
            help="(Optional) Minimum number of taxa in which a region must be present to be extracted",
        )
        parser.add_argument(
            "--numthreads",
            "-n",
            type=str,
            required=False,
            help="(Optional) Number of CPUs to use; can be any positive integer or 'auto' (default)",
        )
        parser.add_argument(
            "--verbose",
            "-v",
            action="store_true",
            required=False,
            help="(Optional) Enable verbose logging",
        )
        parser.add_argument(
            "--order",
            "-r",
            type=str,
            required=False,
            help="(Optional) Order that the alignments should be saved ('seq' or 'alpha')",
        )
        parser.add_argument(
            "--concat",
            "-c",
            action="store_true",
            required=False,
            #help="(Optional) Enable sequential writing of alignment concatenation"
            help="(Optional) Enable concatenation of individual alignments and writing of concatenation file to disk"

        )
        parser.add_argument(
            "--alignmenttool",
            "-a",
            type=str,
            required=False,
            default=argparse.SUPPRESS,
            help="(Optional) Alignment tool to use (default: mafft). Options: mafft, clustal, muscle",
        )
        parser.add_argument(
            "--alignmenttoolpath",
            type=str,
            required=False,
            help="(Optional) Path to alignment tool executable",
        )
        parser.add_argument(
            "--config",
            type=str,
            required=False,
            help="(Optional) Path to YAML config file containing parameters",
        )
        return parser

    @classmethod
    def _get_construct_dict(cls, args: argparse.Namespace) -> Dict[str, Any]:
        # map the cmd argument names to class constructor parameters,
        # and remove empty argument values (`None`)
        args_dict = vars(args)
        args_dict = {
            cls._construct_name_map[k]: v for k, v in args_dict.items() if v is not None
        }
        return args_dict
