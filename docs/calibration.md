# Strategy and memory calibration

PlastBurstAlign chooses a MAFFT heuristic tier from a work estimate

```text
W = scale · N · L_max^α(L_max)  +  β · N² · L_max
```

Tier boundaries and related knobs live in
[`plastburstalign/strategy_constants.py`](../plastburstalign/strategy_constants.py)
as `DEFAULT_CONSTANTS`. Until a full calibration run lands on a quiet machine,
`calibrated=False` and the numbers are **provisional** (intentionally
conservative toward `--auto`). The memory model used by `autotune` is likewise
marked provisional in run logs and manifests.

## Smoke calibration

From the repository root (MAFFT on `PATH`):

```bash
export PATH=$HOME/miniforge3/envs/plastburst-dev/bin:$PATH
python benchmarking/calibrate_strategy.py --smoke
```

`--smoke` times a tiny `(N, L)` grid in a few seconds and exercises the
pipeline end to end. A fitting pass that prints a ready-to-paste
`DEFAULT_CONSTANTS` block uses `--fit` (see the script docstring); only that
workflow should rewrite the constants and provenance fields.

## Design rule

Algorithm choice is a pure function of the data (`N`, lengths, alphabet,
config). Host resources change scheduling, worker counts, scratch tier, and
memory admission only—so identical inputs with a fixed `--seed` stay bit-stable
across machines when comparing alignment outputs (not wall-clock metrics).

**Hard gate:** `N = 100`, `L = 3 kb` must remain `ACCURATE` (`mafft --auto`).
A fit that lowers `work_fast_tree` enough to send that locus to PartTree is
wrong for shipping, even if it meets a wall-time target on some cells.

## 2026-11-01 reduced-grid run (this machine)

Host: AMD Ryzen 5 PRO 5650U (6C/12T), ~15 GiB RAM (~3 GiB MemAvailable during
the run), `/dev/shm` 7.5 GiB, MAFFT `v7.526 (2024/Apr/26)`. Runs were outside
the Cursor sandbox so `/dev/shm` and process accounting were real. The host was
**not** quiet (substantial background RAM use); treat timings as directional.

### Commands

```bash
export PATH=$HOME/miniforge3/envs/plastburst-dev/bin:$PATH
cd /path/to/PlastBurstAlign
mkdir -p benchmarking/results

python benchmarking/calibrate_strategy.py --smoke \
  --out benchmarking/results/smoke.json

# Reduced grid (not the default 13k-scale tier): 4×3×4×2 = 96 cells, 90 s cap
python benchmarking/calibrate_strategy.py --grid \
  --ns 50,100,200,300 \
  --lengths 500,1500,3000 \
  --tiers all \
  --threads 1,8 \
  --timeout 90 \
  --out benchmarking/results/reduced_grid.json

python benchmarking/calibrate_strategy.py --fit \
  benchmarking/results/reduced_grid.json \
  --target-seconds 45 --target-threads 8 --min-seconds 0.5
```

Wall clock for the reduced grid was ~26 minutes. JSON lives under
`benchmarking/results/` (local artefact; not required in git).

### Selected timings (8 threads unless noted)

| N | L | ACCURATE | FAST_TREE | Notes |
|---|---|----------|-----------|-------|
| 100 | 3000 | 23.9 s (1t: 12.8 s) | 34.9 s | Design point: `--auto` faster and under 45 s |
| 200 | 3000 | 66.4 s | 43.4 s | PartTree wins; current `W≈4.4e10` already `FAST_TREE` |
| 300 | 3000 | TIMEOUT (>90 s) | 63.9 s | Heuristic tier appropriate |
| 50 | 3000 | 3.2 s | 11.1 s | PartTree slower at small N |
| 100 | 1500 | 2.6 s | 5.2 s | PartTree slower |
| 300 | 1500 | 32.2 s | 14.9 s | PartTree helps |

Peak ACCURATE RSS at 8 threads was roughly 200–440 MB on this grid (e.g. 439 MB
at `N=100 L=3000`). PartTree peaks stayed ~70–75 MB at `L=3000`.

### Fit decision: **do not ship**

`--fit` on this grid proposed roughly

```text
alphas=(3.0, 3.0, 1.6), beta=100, work_fast_tree≈1.1e10, calibrated=True
```

That model is rejected:

1. Exponents are pathological (α drops with length; β jumps two orders of
   magnitude)—a symptom of a sparse grid where `--auto` wall time does not
   track analytic `W` smoothly (e.g. some `L=3000` cells finish faster than
   `L=1500`).
2. `work_fast_tree≈1.1e10` would move **`N=100 L=3 kb` into PartTree**,
   violating the design gate and `tests/test_strategy.py`.
3. At the sizes that matter for ordinary plastome loci, PartTree is often
   *slower* than `--auto`; a 45 s / 8-thread κ-based boundary therefore
   over-promotes heuristics.

**Shipped constants stay provisional:** `work_fast_tree=3.0e10`,
`work_high_scale=7.0e10`, `work_extreme_scale=1.5e11`, `alphas=(2.0, 2.2, 2.4)`,
`beta=0.5`, `calibrated=False`. The measurements *support* keeping
`N=100 L=3 kb` on `--auto` and using PartTree once `W` crosses ~3e10
(`N=200 L=3 kb` already there).

`TimeModel.sec_per_work` / `MemoryModel` / `DEFAULT_USE_SMT` were **not**
flipped to `calibrated=True`. Observed ACCURATE κ (`seconds / W` with the
current work model, 8 threads, ≥0.5 s) spanned ~3e-10 … 1e-7 across cells
(median ~1.2e-8; upper quartile ~5.9e-8) versus the provisional
`5.0e-9`—useful as a range, not a trusted least-squares slope.

### SMT hint (6 vs 12 workers)

Tiny RNA set (12 taxa, 24 genes, 6 loci): wall ~0.53 s vs ~0.52 s (straggler-
bound; no SMT signal).

Larger CDS set (24 taxa, 48 genes):

```bash
# after write_genbank_dataset(... n_taxa=24, n_genes=48 ...)
python -m plastburstalign -i … -o … -s cds -a mafft -n 6 --no-smt --seed 1 --subdivide off -t 2
python -m plastburstalign -i … -o … -s cds -a mafft -n 12 --smt --seed 1 --subdivide off -t 2
```

Manifest wall ~1.92 s (6 workers) vs ~1.65 s (12 workers), ~1.16×—modest.
`DEFAULT_USE_SMT` remains `False` (physical cores by default).

### Limits of this calibration

* Reduced `(N,L)` only—no protein grid, no monocot `N≈700` scenario, no 13k
  taxa tier.
* Host RAM headroom was low; 8-thread ACCURATE sometimes *slower* than
  1-thread (memory / contention).
* Serial MAFFT cells only; multi-locus SMT sample is small and short.
* A quiet machine with a denser length grid (and separate protein runs) is
  still required before setting `calibrated=True`.
