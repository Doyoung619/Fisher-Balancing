# Reproducing and extending experiments

## Environment

Use Linux, Python 3.12, and one BLAS thread per worker. Python 3.12.10 was tested. Create a new environment inside this checkout; no existing research environment needs modification.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

`requirements.txt` pins direct scientific dependencies. `requirements-cpu-lock.txt` also pins the transitive packages from the clean tested CPU environment. To install that exact package set instead:

```bash
python -m pip install -r requirements-cpu-lock.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

This CPU lock is packaging-validation provenance, not the original historical GPU environment. For sampled CUDA runs use a compatible PyTorch 2.4.1 CUDA build (historically cu124 with Triton3.0.0). Do not install the CPU lock over a CUDA environment. Hardware and software changes can affect floating-point tie decisions; resume compatibility is stricter than merely installing the same top-level package names.

## Three levels of validation

| Level | Command from repository root | What it establishes |
|---|---|---|
| Data-free tests | `python scripts/check.py` | Numerical objectives, constraints, acquisition checks, risk aggregation, checkpoint/recovery logic on available fixtures |
| Tiny workflow | `python scripts/smoke.py --output outputs/smoke_001` | All engines produce usable local artifacts on artificial data; real engine also aggregates, plots and verifies |
| Actual real-data integration | `python scripts/check.py --include-data` | Also exercises marked real-response checks, after preparing the local datasets and references |

Without PyTorch, use `--skip-sampled`. Missing CUDA, archived snapshots, and measured production artifacts are reported as skips. Those skips are not validated hardware/results. The explicit `requires_data` marker makes the boundary visible; tests never fetch datasets automatically.

The root runner launches the experiment suites in separate interpreters because the synthetic packages share the name `src`. For targeted work, enter one experiment directory and run `python -m pytest tests/... -q` there.

## Continuous-ball protocol

See [the experiment README](../experiments/continuous_ofat/README.md). `python -m src.common --prepare --smoke --output ABSOLUTE_RUN` initializes all 98 smoke combinations. Run `python -m src.supervisor run --run-dir ABSOLUTE_RUN --workers 2`. For production omit `--smoke` and deliberately allocate resources for N20000 and 512/256 repeats.

Inputs are the specification, exclusion amendment, and executable resolver; YAML is a generated view. Preparation creates immutable scientific configuration and independent reference values before trajectories. The supervisor creates previews and final analysis. Gaussian shared-method aliases must remain with their source files; copying only an alias is insufficient.

Stop at a safe batch boundary with `scripts/stop.sh RUN`, then use `scripts/resume.sh RUN WORKERS`. The same source/config/environment hashes are required. For a new horizon/repeat count, use the explicit continuation command in the README rather than editing old JSON.

## Sampled-ball protocol

See [the experiment README](../experiments/sampled_ball/README.md). Initialize with `python -m src.control init --run-dir ABSOLUTE_RUN --preset preferred_full_scan --N 10000`. The production default scans all K4096 candidates at every virtual query. The obsolete shortlist preset is available only for explicit historical comparisons.

Foreground CPU-only execution is `python -m src.control supervise --run-dir ABSOLUTE_RUN --cpu-workers 4 --gpu-workers 0`. Original mixed execution used 28 CPU and four GPU workers. No runtime estimate from that host is a portable guarantee. Full scan and cumulative nonlinear refitting can be expensive on CPU.

Stop/resume and plotting wrappers accept the run path, or read the local ignored `CURRENT_RUN` pointer. Keep config/jobs/manifest/checkpoints/results together. Final analysis uses saved states and does not require raw external data. K-sensitivity compares numerical clouds; it does not certify the continuous optimum.

## Recorded-response protocol

See [the experiment README](../experiments/real_response/README.md). Download/prepare actual source data via the CLI, then run `scripts/run_profile.py` with a chosen YAML profile and new output directory. `--data-root` is a **raw input root**, not an output root. Prepared data and references stay local under `experiments/real_response/data/`.

The profile runner supports the preserved batch, temperature, FB-opt/FB-opt2, and family-specific configurations without importing historical campaign outputs. Prepare the corresponding profile's data scope before running it. Reference preparation adds any required matched budgets. Rerunning the same command resumes committed trajectory state; use the same scientific input and prepared-data hashes. Keep datasets fixed when comparing methods.

The original `pipeline` CLI includes a real-data throughput pilot and automatic resource planning tailored to full pools. It is a production orchestration path. The portable runner uses a chosen worker count and avoids that pilot for small profiles.

## Results, storage and reproducibility limits

Full experiments generate NPZ/NPY arrays, checkpoints, selected IDs, references, CSV/JSON reports and images. They are ignored by Git; provision storage separately. The original GLM workspace is preserved independently of this checkout. The publication contains only small aggregate examples and selected figures.

For code changes, seed changes, altered dimensions/budgets, or different data transforms, use a new run directory. Historical checkpoint hashes intentionally differ from the curated publication. A data-free checkout can reproduce implementations and small demonstrations; reproducing a historical numerical result additionally needs its full original data, source version, configuration and compatible environment.
