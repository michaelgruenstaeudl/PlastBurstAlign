#!/usr/bin/env python3

# Example usage: python benchmarking_INT.py benchmark_YangEtAl2022_Asteraceae 5

import argparse
import glob
import os
import subprocess
import sys
import tarfile
import time
import urllib.request
from pathlib import Path


def _resolve_dataset(dataset: str, script_dir: Path) -> Path:
    supplied_path = Path(dataset).expanduser()
    candidates = [
        supplied_path,
        Path.cwd() / supplied_path,
        script_dir / supplied_path,
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()

    archive = script_dir / f"{dataset}.tar.gz"
    url = f"https://zenodo.org/records/17137737/files/{dataset}.tar.gz?download=1"
    print(f"Downloading benchmark dataset from {url}")
    urllib.request.urlretrieve(url, archive)

    print(f"Extracting {archive.name}")
    with tarfile.open(archive, "r:gz") as tar:
        tar.extractall(path=script_dir)

    expected_dir = script_dir / dataset
    if expected_dir.is_dir():
        return expected_dir.resolve()

    extracted_dirs = [
        Path(path)
        for path in glob.glob(str(script_dir / "*"))
        if Path(path).is_dir() and glob.glob(os.path.join(path, "*.gb"))
    ]
    if len(extracted_dirs) == 1:
        return extracted_dirs[0].resolve()
    raise FileNotFoundError(
        f"Could not identify the extracted dataset directory for {dataset!r}. "
        "Expected one directory containing .gb files."
    )


def main() -> None:
    if len(sys.argv) < 3 and not any(arg in {"-h", "--help"} for arg in sys.argv[1:]):
        print(
            "Example usage: python benchmarking_INT.py "
            "benchmark_YangEtAl2022_Asteraceae 5",
            file=sys.stderr,
        )

    parser = argparse.ArgumentParser(
        description="Download (if needed) and benchmark PlastBurstAlign in intron mode."
    )
    parser.add_argument(
        "dataset",
        help="Dataset directory, or Zenodo dataset name to download",
    )
    parser.add_argument("cpus", type=int, help="Number of CPUs to use")
    args = parser.parse_args()
    if args.cpus < 1:
        parser.error("cpus must be at least 1")

    script_dir = Path(__file__).resolve().parent
    repo_dir = script_dir.parent
    input_dir = _resolve_dataset(args.dataset, script_dir)
    gb_files = glob.glob(str(input_dir / "*.gb"))
    if not gb_files:
        parser.error(f"No .gb files found directly in {input_dir}")
    print(f"Using dataset folder: {input_dir}")
    print(f"Size of input dataset: {len(gb_files)} GenBank files")

    output_dir = script_dir / f"{input_dir.name}_output_INT_{args.cpus}threads"
    command = [
        sys.executable,
        "-m",
        "plastburstalign",
        "-i",
        str(input_dir),
        "-o",
        str(output_dir),
        "-s",
        "int",
        "-t",
        "5",
        "-l",
        "6",
        "-n",
        str(args.cpus),
    ]

    print(f"Running PlastBurstAlign in intron mode; output: {output_dir}")
    run_start = time.time()
    subprocess.run(command, check=True, cwd=repo_dir)
    print(f"Time required for analysis: {time.time() - run_start:.2f} seconds")


if __name__ == "__main__":
    main()
