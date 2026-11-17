# Sampled-ball synthetic experiments

The action domain is a continuous Euclidean ball, approximated by a fresh shared candidate cloud each batch. This is a numerical Monte Carlo acquisition oracle. Evaluating every candidate is exact within the cloud, not exact optimization over the whole ball.

## Environment and smoke

Install root dependencies plus PyTorch 2.4.1. CPU wheels support all data-free checks; CUDA runs also use Triton 3.0.0 from the compatible PyTorch CUDA installation. Original GPU work used torch 2.4.1+cu124. CPU and CUDA runtime validation are separate.

```bash
cd experiments/sampled_ball
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
```

For an offline 14-job CPU example with K=64, N=40 and four repeats, run `python scripts/smoke.py --output outputs/demo_001` from the repository root. It tests all seven methods and both families, exporting raw uncentered risks.

## Fresh production run

```bash
python -m src.control init --run-dir "$PWD/runs/production_001" --preset preferred_full_scan --N 10000
# Foreground CPU-only execution (can be expensive at full K/N/repetitions):
python -m src.control supervise --run-dir "$PWD/runs/production_001" --cpu-workers 4 --gpu-workers 0
```

For the original mixed CPU/four-GPU allocation, use a host with four visible CUDA devices:

```bash
bash scripts/launch.sh "$PWD/runs/production_001" 28 4
bash scripts/status.sh "$PWD/runs/production_001"
bash scripts/stop.sh "$PWD/runs/production_001"
bash scripts/resume.sh "$PWD/runs/production_001" 28 4
bash scripts/plot.sh "$PWD/runs/production_001" --preview
bash scripts/plot.sh "$PWD/runs/production_001"
```

The detached launcher requires tmux. Worker counts are runtime choices; adjust them to your machine. With zero GPU workers all methods route to CPUs. CLI workers can also execute an individual initialized job with `python -m src.control worker --run-dir RUN --job-id JOB --device cpu`.

## Authoritative protocol

`src/control.py::make_config` is the executable preset definition. `configs/production.json` records its publication snapshot. The default is `preferred_full_scan`: K=Q=4096, N=10000, seven conditions and seven methods for Gaussian/Bernoulli. FB has 256 center/128 noncentral response repeats; comparators have 64/32. Total: 98 jobs, 5120 trajectories, 51200000 responses.

Gaussian candidates lie on the sphere. Bernoulli candidates use independent directions at radial levels .25, .5, .75 and 1. Cloud seeds are shared across methods/repeats within family/condition/batch. Unlike the continuous Gaussian engine, sampled FB/A/D/E trajectories are not aliased. All calculations use float64.

The earlier `preferred` shortlist preset remains in code for explicit historical comparisons, but is superseded. Independent historical audits found material score loss at unaudited query positions; it is not the final protocol. Do not substitute Q=128 for a full scan and retain the same experiment label.

The constrained estimator, feedback, full-history plug-in Fisher, Q aggregation and fixed N_cal=400 temperature convention match the documented synthetic definitions. UCB/TS conventions are reimplemented locally; no other experiment package is imported.

## Output and interpretation

Each run has config/jobs/manifest, atomic PyTorch checkpoints, `results/*.npz`, progress/status JSON, logs, reference calculations, figures, endpoint/metric CSV tables, and K-sensitivity diagnostics. `status.json` must reach `COMPLETE` with `stage=finished`; inspect failures/postprocessing errors too. Full saved states are needed to regenerate all scientific analysis; the small published endpoint CSV is only an example.

Resume validates scientific source/config fingerprints and restores policy RNG state. Keep the original package/versioned source for old runs. Query and response streams are deterministic by indexed seed, but bitwise agreement across hardware/software versions is not guaranteed.

The final historical example in [../../docs/RESULTS.md](../../docs/RESULTS.md) is from the corrected full-scan run. Neither finite-cloud scans nor K=8192 sensitivity references supply a full continuous-oracle theorem certificate.
