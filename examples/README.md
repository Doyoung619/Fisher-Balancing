# Small examples only

`toy_pool.csv` is explicitly artificial: 160 seeded rows with `x0`–`x4`, `y_gaussian`, `y_bernoulli`, and `y_poisson`. It contains no recorded user data. `scripts/smoke.py` uses the first120 rows as a pool and last40 as heldout inputs to exercise the real-response engine in a separate synthetic workspace. It is not the actual OBD/MovieLens/Last.fm preprocessing format.

`sampled_endpoints.csv` selects only the central condition from the existing final sampled-ball endpoint table. `real_movielens_curve.csv` contains one historical condition's aggregate plotting curves, all twelve methods and all checkpoints. These tables contain aggregate results, not raw datasets.

See [the result guide](../docs/RESULTS.md) for metric definitions and regeneration commands. [provenance.json](provenance.json) identifies the original relative source paths, SHA256 values and selection rules. No additional large data files belong in this directory.
