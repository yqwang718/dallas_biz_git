# DCM Library

A JAX-based Discrete Choice Model (DCM) framework for estimating spatial location choice via conditional logit. Agents (e.g., offenders or victims) choose among census blocks; the model captures how distance, demographic similarity, and local features drive those choices. Unlike prior crime-location-choice studies that rely on sampled alternatives, this framework estimates over the **full choice set of 15,000+ census blocks**, enabled by JAX's CPU/GPU acceleration and memory-efficient chunked computation.

## Citation

If you use this code, please cite:

> Hipp, J. R., & Wang, Y. (2026). How does the business environment shape mobility by offenders and mobile targets? *Crime & Delinquency*. https://doi.org/10.1177/00111287261423080

### Library structure (`dcm/`)

| Module | Purpose |
|--------|---------|
| `protocols.py` | Pydantic data schemas (`AgentFeatures`, `BlockFeatures`, `BlockAggregatedBizFeatures`), JSONL loader, and array construction utilities |
| `interactions.py` | Configurable interaction functions — distance metrics (L2, log-distance), race dissimilarity, income difference, and triplet interactions |
| `models.py` | Three model variants, each with chunked-sum and per-sample versions: **base** (distance + race + income + features), **interactions** (distance x feature interactions with controls), and **network_interactions** (distance x network-feature interactions) |
| `mle_utils.py` | MLE utilities — standard errors via Hessian inversion, BIC computation |
| `tests.py` | Consistency and load tests (chunked vs. vectorized, gradient checks, 1M-sample stress test) |

### Key design choices

- **JAX** for automatic differentiation and GPU acceleration
- **`lax.scan` + gradient checkpointing** for memory-efficient optimization over large datasets (chunked negative log-likelihood)
- **Pydantic** for type-safe, self-documenting data schemas
- **JSONL** streaming format for agent and block features
- **YAML config** to specify model type, feature sets, and optimizer settings without code changes

## Pipeline

The estimation pipeline has 8 steps. Data files are not included in this repository (licensed); only the framework code is provided.

### 1. Prepare Block Features

**Script:** `scripts/1_prepare.ipynb`

Merges census block shapefiles with business environment data into a single GeoPackage, then extracts block-level features to JSONL.

- **Input:** Census shapefiles + aggregated business data (`.dta`)
- **Output:** `blocks_fine.jsonl` — one record per census block with fields:

| Field | Description |
|-------|-------------|
| `block_id` | Unique block identifier (row index) |
| `home_coord` | Block centroid coordinates |
| `log_consumer`, `log_white_collar`, `log_blue_collar` | Log establishment counts by sector |
| `log_consumer_control`, `log_white_collar_control`, `log_blue_collar_control` | Buffer-based control counts (1/4 mile) |
| `consumer_hetero` | Consumer heterogeneity index |
| `log_emp_*` | Log employment variants of the above |
| `extra_features` | 31 fine-grained business types |

### 2. Extract Agent Features

**Script:** `scripts/2_feature_extraction.ipynb`

Loads geocoded agent records and incident records, filters by date range, assigns each record to its census block, and exports agent-level JSONL files.

- **Input:** Geocoded agent/incident CSVs + GeoPackage from step 1
- **Output:** `offenders_fine.jsonl` and `victims_fine.jsonl` — one record per agent with fields:
  - `agent_id`, `home_block_id`, `home_coord`, `race`, `crime_type`, `incident_block_id`

Together with `blocks_fine.jsonl`, these three files are the core input to the DCM estimation pipeline.

### 3. Run DCM Models

**Scripts:** `scripts/run.sh` orchestrates 12 model estimations via `scripts/main.py`

Each run varies the agent type, feature set, and whether distance-feature interactions are included:

| File | Agent | Interaction | Feature Set |
|------|-------|-------------|-------------|
| `{agent}_biz.json` | offenders / victims | no | Aggregated business types |
| `{agent}_biz_int.json` | offenders / victims | yes | Aggregated business types |
| `{agent}_full_biz.json` | offenders / victims | no | 31 separate business types |
| `{agent}_full_biz_int.json` | offenders / victims | yes | 31 separate business types |
| `{agent}_lnemp.json` | offenders / victims | no | Log-employment variants |
| `{agent}_lnemp_int.json` | offenders / victims | yes | Log-employment variants |

For each configuration, `main.py`:
1. Loads agent and block features from JSONL
2. Runs DCM estimation for each crime type + all types combined
3. Computes estimates, standard errors, and BIC
4. Saves results to JSON

### 4. Export Result Tables

**Script:** `scripts/saver.py`

Converts JSON estimator results into formatted CSV tables with coefficients, significance stars, standard errors in parentheses, BIC, and sample size (N).

### 5. Study Area Map

**Script:** `scripts/dallas_biz_mapping.ipynb`

Generates a study area map showing census blocks overlaid on an OpenStreetMap basemap with the city boundary.

### 6. Summary Statistics

**Script:** `scripts/summarizer.py` then `scripts/clean table.ipynb`

`summarizer.py` computes descriptive statistics (mean, std, percentiles) for block features and agent-block distances by crime type. `clean table.ipynb` reformats the output for publication (e.g., `0.337 ± 0.374` becomes `0.337 (0.374)`).

### 7. Z-test Comparisons

**Script:** `scripts/compute_ztest_comparisons.py`

Compares offender vs. victim model coefficients using Paternoster et al.'s z-test:

`z = (β_offender - β_victim) / √(SE₁² + SE₂²)`

Runs across all 6 model pairs (biz, biz_int, lnemp, lnemp_int, full_biz, full_biz_int).

### 8. Confidence Intervals

**Script:** `scripts/compute_confidence_intervals.py`

Computes 95% confidence intervals (`CI = estimate ± 1.96 * SE`) with significance stars for all estimator files.
