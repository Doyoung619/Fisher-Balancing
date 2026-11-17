# Continuous-ball OFAT

Gaussian and Bernoulli responses on the closed Euclidean ball, with no fixed candidate pool. Active methods: Ours (FB), A-opt, D-opt, E-greedy, UCB, TS, Random. E-relax code remains for historical checks but is excluded from the active queue by `sources/USER_AMENDMENT_20261005.md`.

## Setup and quick validation

Install the root `requirements.txt`; PyTorch is not required. Activate the repository environment, then:

```bash
cd experiments/continuous_ofat
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q
python -m src.common --prepare --smoke --output "$PWD/runs/smoke_001"
python -m src.supervisor run --run-dir "$PWD/runs/smoke_001" --workers 2
```

The standard smoke is still 98 logical trajectories at N=400. For a much smaller first execution, use the repository's `python scripts/smoke.py` from its root instead. Neither smoke estimates the production risk reliably.

## Production and recovery

```bash
python -m src.common --prepare --output "$PWD/runs/production_001"
bash scripts/launch_detached.sh "$PWD/runs/production_001" 8
bash scripts/status.sh "$PWD/runs/production_001"
bash scripts/stop.sh "$PWD/runs/production_001"
# After graceful stop, reuse the identical code/environment:
bash scripts/resume.sh "$PWD/runs/production_001" 8
python -m src.plotting --run-dir "$PWD/runs/production_001" --phase final
```

Detached execution uses tmux when available, otherwise nohup/setsid. The foreground supervisor is also supported. Launch wrappers use the repository `.venv` or `PYTHON=/absolute/interpreter`. Set the worker count to fit your machine; the original 64-worker default is a historical host setting, not a requirement.

Resolved configuration comes from `sources/OFAT_SPEC.json`, the amendment, and `src/common.py::resolved_config`. `configs/ofat.yaml` and `smoke.yaml` are snapshots; regenerate them with `python -m src.common --write-configs`. `--prepare` does not load arbitrary edits to these YAML snapshots.

Production has N=20000, 512 center/256 noncentral repeats, 7 conditions, 7 methods, 2 families: 28672 logical trajectories. Gaussian Ours/A-opt/D-opt/E-greedy share an exact response-independent design and paired noise, giving 22528 physical trajectories. Those aliases are not independent repetitions. All other paths retain their own policy streams.

## Scientific definitions

The estimator minimizes cumulative sum loss plus rho=1 ridge on `||theta|| <= 4`. Initial observations: zero. Feedback arrives after the entire batch. Nonlinear plug-in information reevaluates all observations at the current estimate. Truth is `(kappa/L) * u` with kappa=1.5; radius sweeps hold kappa fixed. N_cal=400 fixes the temperature coefficient and is independent of the run horizon.

The theoretical Gaussian reference is gamma=L²/d. The Bernoulli reference has an independent analytic dual certificate; see [audit_theory.md](audit_theory.md). Bernoulli acquisition uses scalar/multistart numerical searches and is not globally certified. The exact-oracle theorem does not automatically apply to those numerical oracles. [method_audit.md](method_audit.md) documents objectives and qualifications.

## Outputs and continuation

Runs contain immutable config/jobs/reference/manifests, atomic checkpoints, per-trajectory NPZ exports, summaries, progress/status JSON, and preview/final figures with tables. Plotting uses saved states. Source/config/job hashes, the environment manifest, and key installed package versions are checked on resume. Preserve the entire run directory; do not change a stored manifest to bypass compatibility checks.

For a deliberately extended protocol, stop safely and create a new child run:

```bash
python -m src.common --parent "$PWD/runs/production_001" --N 40000 --extra-repetitions 128
```

The command prints the new run path and preserves parent provenance. Continuation is explicit; no automatic extension depends on observed rankings.
