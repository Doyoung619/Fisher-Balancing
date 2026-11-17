# Recorded real-response GLM experiments

OBD clicks (Bernoulli), MovieLens ratings minus 3 (Gaussian working model), and archived Last.fm weekly scrobble counts (Poisson working model). Selection is without replacement from fixed recorded rows. Unobserved user/item labels are never synthesized.

## Start without downloading datasets

Install root `requirements.txt`. PyTorch is optional for measured CUDA benchmarks and unnecessary for the primary NumPy CPU engine.

```bash
cd experiments/real_response
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
python -m pytest tests -q -m 'not requires_data'
```

From the repository root, `python scripts/smoke.py --skip-sampled --output outputs/demo_001` runs continuous and real-response engine demos. The real-response demo uses the explicitly artificial `examples/toy_pool.csv` in an isolated workspace, including checkpoints, all three GLM families, aggregation, plots and verification. It does not populate or impersonate the actual data cache.

## Download, prepare and run actual data

These commands download public source archives and build local data; network/source availability is required and no source archives are distributed in Git.

```bash
cd experiments/real_response
python -m real_response.cli download --profile smoke
python -m real_response.cli prepare --profile smoke
python scripts/run_profile.py --config configs/smoke.yaml --output outputs/smoke_001 --workers 2
```

For preexisting raw archives, use `python -m real_response.cli prepare --profile smoke --offline --data-root /absolute/raw/root`. The raw root contains `obd/`, `movielens/`, and `lastfm/` with the filenames specified in `real_response/data.py::download_all`. `--data-root` refers to raw inputs; prepared features and references are written under this experiment's local `data/`.

The smoke profile still uses actual real-data pools: two dimensions, one seed, a small adaptive budget, and every applicable core baseline. For a full run, prepare the production pools first:

```bash
python -m real_response.cli prepare --profile server_main
python scripts/run_profile.py --config configs/server_main.yaml --output outputs/main_001 --workers 8
```

`server_main` has 930 trajectories: d=5/10/20, seeds 0–9, ten common methods plus Bernoulli-only BADGE. Initial labels n0=80; OBD uses additional budget4000, m200, K512; MovieLens/Last.fm use budget400, m20, K256. Runtime and storage can be substantial. Worker counts are explicit resource choices, not scientific parameters.

The portable runner extends matched-reference caches for the selected profile, runs/resumes the trajectories, summarizes, and verifies. Use a new output path for a changed protocol. The original auto-benchmark pipeline remains accessible as `python -m real_response.cli pipeline --profile server_main`; its pilot was designed for full real pools. Prefer `scripts/run_profile.py` for smoke runs, copied profiles, fixed worker allocations, and the preserved batch/temperature/FB-opt/family presets.

`fb_opt` and `fb_opt2` support Gaussian/Bernoulli. The family-specific profiles add Gaussian B-EMCM/LCMD/RegVol or Bernoulli Entropy/FIRAL. The published MovieLens example uses `fb_opt2`, whereas `server_main` includes `fb_sample` and `fb_range`. See `selectors.py`, `family_selectors.py`, and each YAML's method list.

## References and outputs

Pool-only preprocessing fits imputation, categorical encodings and centered unwhitened PCA19. d=5/10/20 are prefixes of a shared maximum-dimension transform including the scaled intercept. The maximum-dimension label-free D-opt initialization is shared across dimensions/methods for each seed. Reference coefficients are fitted separately for every dimension; slicing a d20 coefficient vector is invalid.

`theta_full` is a full-pool fixed-ridge estimate. Matched references use the prescribed budget-dependent penalty and are evaluation-only. Neither is an identified generating parameter. See [loss_audit.md](loss_audit.md) and [data_provenance.md](data_provenance.md) for definitions and historical dataset limitations.

Outputs contain protocol/execution manifests, per-task committed batch NPZ/JSON checkpoints, scalar `trajectory.csv`, signed errors, all four information matrices/spectra, selected row IDs, slot/solver audits, seed aggregate Q matrices, endpoint bootstrap samples, tables and figures. Re-run the same command after interruption: compatible completed states are reused. `scripts/run_profile.py` always completes reporting/verification; a smoke may pass all requested tasks with `MAIN_COMPLETE=false`, since that field is reserved for the original ten-seed/three-dimension production scope.

To regenerate reports without refitting:

```bash
python -m real_response.cli summarize --config configs/server_main.yaml --output outputs/main_001
python -m real_response.cli verify-artifacts --config configs/server_main.yaml --output outputs/main_001
python -m real_response.cli export-paper --config configs/server_main.yaml --output outputs/main_001
```

Use the exact config/prepared data from that run. After preparing all datasets/references, `python -m pytest tests -q` also runs the explicitly marked integration checks. Archived GPU/production-output checks skip when their original artifacts are absent.
