# Historical real-data provenance and interpretation

This note records measured dataset properties from the original local execution; the full manifests and source caches remain in the preserved GLM workspace. The published repository excludes those archives and does not claim to have downloaded them again. Fresh runs create their own data manifests and `data/raw/` caches. The historical real-data experiments used actual archived responses; the separately labeled offline smoke uses artificial data.

| Dataset | Pool / test rows | Response | Important limitation |
|---|---:|---|---|
| OBD-small random/all | 8,000 / 2,000 | Original click indicator | Pool has 32 clicks; test has 6. `LOW_EVENT_REFERENCE` applies at d=10 and d=20. |
| MovieLens 100K | 10,000 / 5,000 | Original rating minus fixed constant 3 | Gaussian squared loss is a working model for recorded ratings. Unobserved user–movie responses are never synthesized. |
| Last.fm 1K, 50-user release | 10,000 / 5,000 | Archived scrobbles per user/artist/seven-day bin | 27 users qualify from prior history. Zero means no archived scrobble, with possible logging attrition. It does not establish an exposure or dislike. |

OBD is pinned to commit `8cbd5fa4558b7ad2ba4781546d6604e4cc3e07c4`. The official loader confirmed nested `random/all/all.csv`, `item_context.csv`, original `click`, and `propensity_score`. The official README identifies user–item affinity as based on past clicks. Anonymized categorical predictors remain categorical; `item_feature_0` follows the official loader's numeric treatment. Clicks, propensity, raw index and timestamp are excluded from predictors. Equal timestamps stay on one side of the chronological boundary.

MovieLens comes from the official GroupLens ZIP. Its README, genre and occupation descriptions and Latin-1 item metadata were inspected. Profile and movie joins preserve the observed rating records. User/movie IDs and target-derived aggregates are excluded from features. Pool-standardized age × genre and gender indicator × genre interactions are defined before fitting.

Last.fm uses the `eifuentes/lastfm-dataset-1K` v1.0 processed mirror, not the full 1K file. The downloaded parquet contains 776,336 scrobbles and 50 users. The original collector website was accessible during this run. Eligibility uses only the 180 days preceding 2008-05-05 UTC (at least 100 scrobbles and four active weeks), yielding 27 users. The 20 artists are selected by eligible users' history counts with normalized-key tie breaking. This explicitly resolves the protocol's artist-history population to eligible users. Weeks 0–41 form the pool period; weeks 42–51 form the test period. Artist keys use MBID when available, otherwise Unicode-normalized artist names. No fuzzy merging, count clipping, positive-only filtering, or response resampling occurs. Label-independent hashing selects the capped row pools. Pool/test have 341/173 nonzero cells; pool maximum count is 405.

All imputation values, numeric scales, categorical dictionaries, rare-level grouping, interactions and centered PCA are fitted using pool covariates. Test covariates do not determine those fits. PCA retains 19 unwhitened coordinates; each loading's largest absolute entry is positive. A single maximum-dimension pool scale bounds pool rows. d=5/10/20 are exact prefixes, including the scaled intercept. All nine designs have full column rank. Test maximum norms above one (MovieLens 1.105; Last.fm 1.031) are preserved without clipping or refitting.

The source caches contain distribution terms. MovieLens imposes research-use conditions and restricts raw redistribution; Last.fm's original release permits noncommercial use and attributes collection to Oscar Celma. The protocol attributes OBD data to CC BY 4.0; the repository loader's Apache license applies to code and is not substituted for a data license. The small-distribution README itself does not state that data license, so its license attribution is not represented as independently verified from that README. Raw profiles and user records remain in the local cache and are not embedded in paper tables or figures.

`prepared_data_summary.csv`, `feature_rank_audit.csv` and `data_integrity_audit.json` in the current output directory record measured values. The independent audit reconstructs every prepared response from source records, validates foreign keys and chronological partitions, checks weekly conservation and fixed history eligibility, and checks all feature ranks. Re-run without changing training artifacts with:

```bash
OPENBLAS_NUM_THREADS=1 python3 scripts/audit_prepared_data.py --output outputs/data_audit
```

At the time of the original real-response implementation, its manuscript PDF and separate PAT feedback file had not been provided. Equation mappings and interpretation instructions come from the supplied execution prompt; the experiment does not claim independent inspection of absent attachments. `theta_full` is a full-pool ridge reference, never an identified true parameter. Poisson and Gaussian working-model results do not establish the manuscript's asymptotic or distributional assumptions.
