# Working on Fisher Balancing

Read this file, the selected experiment's README, and `docs/DEVELOPMENT.md` before editing. This is research software: preserve the statistical protocol as carefully as the implementation.

## Start here

- `python scripts/check.py` runs independent pytest processes in the three experiment directories. Set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- `python scripts/smoke.py` exercises all engines on tiny, explicitly artificial data. Use `--output outputs/NEW_NAME` to retain artifacts. This is execution evidence, not a research replication.
- Without PyTorch, add `--skip-sampled` to both commands.
- Real-data tests use `@pytest.mark.requires_data`; `scripts/check.py --include-data` includes them when prepared data/references are present. Never download data implicitly in tests. Archived hardware/result checks may skip with explicit reasons.
- Do not run `pytest experiments` in one process: two independent packages are both named `src`.

## Sources of truth

| Area | Scientific inputs / implementation |
|---|---|
| Continuous | `sources/OFAT_SPEC.json` plus the E-relax exclusion amendment; executable resolution in `src/common.py::resolved_config`; YAML files are generated snapshots |
| Sampled | `src/control.py::make_config` and `make_jobs`; published default is `preferred_full_scan`; `configs/production.json` is a readable snapshot |
| Real | `configs/*.yaml`; `real_response/selectors.py`, `family_selectors.py`, `numerics.py`, `evaluation.py`; scalar schema in `configs/required_scalar_columns.json` |
| Definitions | `docs/RESULTS.md`, `continuous_ofat/audit_theory.md`, `continuous_ofat/method_audit.md`, `real_response/loss_audit.md` |

## Invariants to preserve

1. Float64; cumulative **sum** negative log likelihood plus fixed ridge; the real-data intercept is penalized too. Changing to mean loss requires a deliberate regularization redesign.
2. Choose the entire batch before revealing its responses. Selectors must not receive hidden labels, full-pool reference coefficients, test outcomes, or synthetic truth.
3. Recompute nonlinear plug-in curvature on the entire selected history at the current fit. Acquisition-only numerical floors must not leak into likelihood fitting or primary information matrices.
4. `Q = E.T @ E / S` is uncentered. `Rdir_squared = lambda_max(Q)` and `R2_squared = trace(Q)`; take square roots only for root-risk displays.
5. Paired seeds, RNG streams, temperature schedule, candidate generation, and checkpoint state are scientific inputs. Do not change them silently.
6. Continuous Gaussian Ours/A-opt/D-opt/E-greedy share one exact physical path. This aliasing does not apply to sampled-ball acquisition.
7. Full scanning a sampled candidate cloud does not certify exact optimization over the continuous ball. Do not promote numerical agreement to a theorem guarantee.
8. Real full-pool estimates are empirical references, not known generating parameters. Tiny examples, incomplete seeds, and previews must stay labeled.

## Safe development workflow

- Keep generated data/runs/checkpoints out of Git. Only intentionally small, reviewed examples belong in `examples/`, and only selected images belong in `docs/figures/`.
- Preserve original datasets and historical outputs. Use a new output directory for a changed protocol or implementation. Never alter hashes/manifests to force an incompatible resume.
- New selectors belong in the corresponding experiment's methods/oracles or selectors module; add independent objective/constraint/leakage tests when algorithmic behavior changes.
- Record meaningful protocol changes and validation in the relevant README. Separate numerical/backend changes from statistical changes.
- Shell launchers resolve the repository `.venv`; `PYTHON=/absolute/path/to/python` overrides it. Do not add personal server paths.
- Run the relevant suite and workflow first; use the full data-free check before sharing. GPU-specific changes also need CUDA validation; CPU success alone cannot establish GPU parity.
