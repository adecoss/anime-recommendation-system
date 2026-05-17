# Anime Recommendation and Graph Intelligence

Semester project for an anime discovery and recommendation system. The project builds a reproducible anime catalog from MyAnimeList/Jikan metadata, AniDB metadata/cache enrichment, and a Kaggle user-rating interaction dataset.

## Current Focus

The repository is organized for the Week 3, Week 5, and Week 7 course deliverables:

- Week 3: dataset charter, source inventory, schema draft, processed dataset V1/V3, data dictionary, scale analysis, and ethics/access note.
- Week 5: feature representation, PCA/SVD dimensionality reduction, retained-variance/energy analysis, and t-SNE visualization for comparison only.
- Week 7: K-means, DBSCAN, and OPTICS clustering with parameter sweeps, validation metrics, cluster profiles, and failure analysis.

## Repository Structure

```text
data/
  processed/        processed anime catalog; large ratings file is local-only
  build/            build logs, retry registries, skipped-ID registries
notebooks/
  01_create_dataset.ipynb
  02_get_user_ratings.ipynb
  03_anime_dataset_eda.ipynb
  04_dataset_improvements.ipynb
  05_week3_dataset_charter.ipynb
  06_week5_representation_dimensionality.ipynb
  07_week7_clustering_validation.ipynb
src/
  01_run_dataset_ingestion.py
  02_run_ratings_ingestion.py
  03_build_catalog_features.py
  04_apply_dataset_improvements.py
  05_run_week5_dimensionality.py
  06_run_week7_clustering.py
  07_build_recommendation_graph.py
  09_create_project_visualizations.py
reports/
  BigData_Doc.pdf
  runbook_weeks_3_to_7.md
  video/week7_clustering_video_deck.pptx
artifacts/
  plots/            generated EDA, Week 5, and Week 7 figures
```

## Reproducible Commands

Run the main pipeline pieces in order:

```bash
python src/01_run_dataset_ingestion.py
python src/02_run_ratings_ingestion.py
python src/03_build_catalog_features.py
python src/04_apply_dataset_improvements.py
python src/05_run_week5_dimensionality.py
python src/06_run_week7_clustering.py
python src/07_build_recommendation_graph.py
python src/09_create_project_visualizations.py
```

On Windows, if Jupyter spins before executing Week 5 or Week 7, use the local runner:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_week_notebooks.ps1
```

That command redirects Jupyter/IPython runtime folders into the project so it does not need write access to the user-level `.jupyter` directory.

## Data Notes

The processed anime catalog is kept in `data/processed/anime_dataset.csv` and `data/processed/anime_dataset.json`.

Large or easily regenerated files are ignored:

- raw Kaggle files
- AniDB cache files
- the 2GB+ `ratings_processed.csv`
- temporary model/matrix files

This keeps the GitHub repository usable while preserving reproducibility through scripts and notebooks.

## Week 7 Video

The video deck is in `reports/video/`:

- `week7_clustering_video_deck.pptx`
