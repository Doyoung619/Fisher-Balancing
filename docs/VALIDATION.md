# Publication validation

Validated on November 16, 2026 (America/New_York), using a newly created, isolated CPython 3.12.10 environment installed from the published requirements. PyTorch was the CPU 2.4.1 build. The complete installed CPU package set is in `requirements-cpu-lock.txt`. `python -m pip check` found no broken requirements.

## Public, data-free checkout

`OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python scripts/check.py`:

| Suite | Passed | Skipped | Deselected |
|---|---:|---:|---:|
| continuous_ofat | 48 | 0 | 0 |
| real_response | 79 | 9 | 37 |
| sampled_ball | 51 | 3 | 0 |
| Total | 178 | 12 | 37 |

Real skips: six archived-selector comparisons without the historical source snapshot; one saved CUDA benchmark check; two production-output checks. The 37 deselected cases require prepared real datasets/references. Sampled skips require CUDA devices. Continuous results include the retained inactive-method numerical tests; E-relax remains excluded from production.

`python scripts/smoke.py --output outputs/publication_smoke` completed two continuous trajectories, fourteen sampled CPU jobs, and eighteen artificial-data trajectories through the recorded-response engine. The last workflow also generated reports/figures and passed task and aggregate verification. `MAIN_COMPLETE` correctly stayed false for this small demo.

## Additional local real-data checks

Prepared datasets, references, initial IDs, and the old selector code snapshot were **copied** from the preserved original workspace into ignored directories in the new checkout. With these local artifacts, the full real-response suite passed **122 tests**, with three saved hardware/production-output tests skipped. Those inputs are not distributed; this count is not the fresh-clone baseline above.

The portable `scripts/run_profile.py` also ran the actual `configs/smoke.yaml` with two workers. All **62/62 trajectories**, **186/186 checkpoint records**, and requested aggregate outputs verified, with zero failed/incomplete tasks. It regenerated reports and figures from actual prepared real-response data. The included MovieLens plotting CSV also regenerated its representative figure successfully without raw data. Continuous standalone `--prepare --smoke` and strict `validate_run` checks passed. The CPU-only sampled supervisor completed a Bernoulli FB job (the queue that previously needed CUDA routing). A regression test confirms that changed scientific identity cannot overwrite existing real-run metadata.

`MAIN_COMPLETE=false` is expected: the original field requires the ten-seed, three-dimension production scope.

## Scope

These checks validate packaging, the existing tests, and small executable workflows. The historical full production grids were not rerun during publication. CPU checks do not establish CUDA parity. The two published figures and example aggregates remain historical artifacts with their own SHA256 provenance, not figures produced by the publication smoke.

A second validation from an export containing only publishable Git files reproduced the 178-pass baseline and all three tiny workflows. Source-workspace inventory before/after publication matched exactly: 509455 files, 49959059604 bytes, with zero added, removed or modified size/mtime records. The original data were preserved.
