# Publication scope and provenance

This repository was curated from the existing GLM workspace into a separate checkout. Original data, checkpoints, results, environments, and source files remain in that workspace. No source data were moved or deleted during publication.

Included: three experiment implementations, their tests and scientific presets, the synthetic manuscript and specification inputs used by the engines, concise reproducibility/development documents, one 160-row artificial fixture, selected aggregate CSV examples, and exactly two historical PNGs. The original experiment requirement prompts, terminal/process state, virtual environments, data downloads, prepared datasets, duplicate delivery bundles, and full result trees are excluded.

## Packaging changes

- Replaced personal absolute interpreter paths with repository `.venv` resolution and the `PYTHON` override. Runtime Python provenance records the actual interpreter.
- Added one tested, scoped dependency list and a clean CPU environment lock instead of publishing an unrelated server-wide package freeze.
- Extracted the real-response required scalar column list from the original requirements document into `configs/required_scalar_columns.json`. Verification and execution metadata now use that explicit schema rather than parsing a prompt at runtime.
- Removed absolute reference-cache paths from real profiles. Added a profile runner that prepares the requested matched budgets locally and runs the core engine without requiring prior campaign outputs.
- Omitted historical campaign orchestration and duplicate packaging scripts whose purpose was to merge already-completed local output trees. Their selector implementations and scientific YAML profiles remain available through the portable runner.
- Made `preferred_full_scan` the sampled-ball initialization default, matching the corrected final historical protocol. The previous default selected the superseded shortlist path. Added CPU routing when zero GPU workers are requested and early validation of requested GPU availability.
- Marked real-data integration tests explicitly. Missing external datasets, archived snapshots, and measured hardware artifacts produce visible skips, not fabricated substitutes. The offline workflow uses an explicitly artificial fixture in a separate workspace.

Core likelihoods, information matrices, selectors, risk definitions, and RNG formulas were retained. Source/config hashing intentionally gives new runs new provenance; old checkpoints should be resumed only with their original code/environment, not by editing hashes.

## Historical examples

`examples/provenance.json` records each selected artifact's relative path in the original workspace, original SHA256, published SHA256, and selection rule. The two images are unmodified historical figures. The sampled endpoint CSV contains only central-condition rows; the MovieLens CSV contains the aggregate curves for one condition. Neither includes raw recorded responses or full experimental trajectories.

The supplied synthetic manuscript is preserved under each synthetic experiment's `sources/` directory so both experiment packages remain standalone. Its SHA256 is `40824afd7eb4f817bf5a98a86f4364498fa5ab8fb4e880fec5fd41e3c283027b`. It is a research input, not a claim that real-data empirical references satisfy its true-parameter theorem.

Local validation details are recorded in [VALIDATION.md](VALIDATION.md). The historical experiment metadata and current packaging validation are separate evidence.
