# Anime Recommendation and Graph Intelligence System

## Overview
This project builds a large-scale anime recommendation and graph analysis system using user interaction data and metadata.

[User Animelist Dataset]: https://www.kaggle.com/datasets/ramazanturann/user-animelist-dataset?select=ratings.csv

[MyAnimeList Anime Metadata Dataset]: https://www.kaggle.com/datasets/battos/myanimelist-animes?select=animes.csv

## Datasets
This project uses publicly available datasets:

- [User Animelist Dataset] – user ratings (~148M interactions)
- [MyAnimeList Anime Dataset] – anime metadata

> Note: Due to size constraints, raw datasets are not included in this repository.

## Reproducibility

1. Download datasets from Kaggle ([User Animelist Dataset])
2. Place them in `data/raw/`
3. Run:

```bash
python src/build_catalog.py
python src/build_interactions.py
