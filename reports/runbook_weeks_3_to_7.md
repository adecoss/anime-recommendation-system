# Anime Recommender Project Runbook, Weeks 3 to 7

This runbook documents the reproducible path from raw/cached sources to the Week 3, Week 5, and Week 7 milestone artifacts.

## 1. Build the Catalog Dataset

Dry run:

```bash
python src/01_run_dataset_ingestion.py
```

Execute ingestion notebook:

```bash
python src/01_run_dataset_ingestion.py --execute
```

Primary outputs:

- `data/processed/anime_dataset.csv`
- `data/processed/anime_dataset.json`
- `data/caches/anidb_metadata_cache.json`
- `data/build/failed_api_requests.json`
- `data/build/skipped_invalid_type_ids.json`
- `data/build/skipped_permanent_http_ids.json`

## 2. Build the Ratings Interaction Layer

Dry run:

```bash
python src/02_run_ratings_ingestion.py
```

Execute ratings notebook:

```bash
python src/02_run_ratings_ingestion.py --execute
```

Primary output:

- `data/processed/ratings_processed.csv`

## 3. Exploratory Data Analysis

Open and run:

```text
notebooks/03_anime_dataset_eda.ipynb
```

The EDA notebook saves images only under:

- `artifacts/plots/eda/`

It does not save EDA tables or rewrite the processed dataset. The EDA graph section now includes:

- recommendation and relation edge-count diagnostics
- Bakemonogatari relation-tree connectivity as a concrete franchise-navigation example
- central anime by incoming recommendation weight and relation degree
- readable top-popularity subgraph for recommendation connections
- chunked ratings quantity distribution for ratings per user and ratings per anime

## 4. Dataset Improvements

Open:

```text
notebooks/04_dataset_improvements.ipynb
```

This notebook is the EDA-driven cleanup step after the retry run finishes. It fills season from air month, converts duration into numeric minutes, adds total watch time, refills AniDB-backed tags/demographics where possible, and leaves optional manual cells for live AniDB repair or dropping negligible episode gaps.

To refresh the enriched AniDB XML cache from Shoko and immediately re-apply dataset improvements:

```bash
python src/08_enrich_anidb_cache_and_dataset.py
```

The enriched AniDB cache stores raw tags, filtered animation-work creators for studio fallback, episode-length summaries, similar-anime links, and allowed scalar metadata. It excludes heavy or noisy AniDB sections such as characters, titles, resources, pictures, ratings, recommendations, and related-anime blocks.

## 5. Week 3 Deliverable

Open and run:

```text
notebooks/05_week3_dataset_charter.ipynb
```

This notebook covers:

- project proposal
- source inventory
- schema draft
- processed dataset evidence
- data dictionary draft
- scale analysis
- ethics and access note
- reproducible ingestion command
- defense preparation

## 6. Week 5 Feature and Dimensionality Pipeline

Build the catalog feature matrix:

```bash
python src/03_build_catalog_features.py
```

Optional script-level SVD summary:

```bash
python src/05_run_week5_dimensionality.py
```

Refresh the milestone/report visualizations:

```bash
python src/09_create_project_visualizations.py
```

Open and run the report notebook:

```text
notebooks/06_week5_representation_dimensionality.ipynb
```

If Jupyter keeps spinning before the first cell executes on Windows, use the project-local notebook runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_week_notebooks.ps1
```

The runner sets `JUPYTER_CONFIG_DIR`, `JUPYTER_RUNTIME_DIR`, `IPYTHONDIR`, and `LOKY_MAX_CPU_COUNT` inside the project so execution does not depend on writing to `C:\Users\CHAMPUX\.jupyter`.

Important point for defense: PCA is applied to the full mixed catalog feature matrix with a capped TF-IDF vocabulary, while Truncated SVD is used for the larger sparse representation.

## 7. Week 7 Clustering Pipeline

Run the script-level clustering artifact:

```bash
python src/06_run_week7_clustering.py
```

Open and run the report notebook:

```text
notebooks/07_week7_clustering_validation.ipynb
```

This notebook covers:

- K-means parameter sweep
- DBSCAN parameter sweep
- OPTICS parameter sweep as a second density-method check
- validation metrics
- cluster profiles
- SVD-space cluster visualization
- cluster genre-profile heatmap
- failure analysis
- defense preparation

## 8. Graph Layer

Build recommendation graph metrics:

```bash
python src/07_build_recommendation_graph.py
```

Primary outputs:

- `artifacts/graph_exports/recommendation_graph_metrics.csv`
- `artifacts/graph_exports/recommendation_graph_summary.json`

## 9. ICIS-Style Report

The Overleaf-compatible report is:

```text
reports/document/doc.tex
```

Figures used by the report are stored in:

```text
reports/document/figures/
```
