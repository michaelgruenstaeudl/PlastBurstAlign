#!/usr/bin/env python3

import glob
import os
import shutil
import sys
import time
import urllib.request
import tarfile
import subprocess
import pandas as pd
import psutil
from datetime import datetime
import resource
import json


def test(dataset, nCPUs, mode):
    """Run a single test case and return performance metrics"""

    script_dir = os.path.dirname(os.path.abspath(__file__))
    mod_rel = ".."
    mod_path = os.path.join(script_dir, mod_rel)
    print(f"Script directory: {script_dir}")
    main_folder = os.path.join(script_dir, dataset)
    print(f"Dataset folder: {main_folder}")

    folder = os.path.join(script_dir, f"{os.path.basename(dataset)}_output_{mode}_{nCPUs}threads")
    print(f"Output folder: {folder}")
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    try:
        
        print(f"Running plastburstalign with {nCPUs} threads, mode: {mode}...")
        os.chdir(mod_path)

        # Set environment for thread control
        env = os.environ.copy()
        env['OMP_NUM_THREADS'] = str(nCPUs)
        env['OPENBLAS_NUM_THREADS'] = str(nCPUs)
        env['MKL_NUM_THREADS'] = str(nCPUs)

        process = subprocess.run([
            sys.executable, "-m", "plastburstalign",
            "-i", main_folder, "-o", folder,
            "-s", mode,"-n", str(nCPUs)
        ], check=True)

    except Exception as e:
        print(f"Error running plastburstalign: {e}")
        return



if __name__ == "__main__":
    dataset = sys.argv[1]
    nCPUs = sys.argv[2]
    mode = sys.argv[3]
    
    test(dataset, nCPUs, mode)
