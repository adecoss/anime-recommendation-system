# Anime Recommendation and Graph Intelligence System

## Overview
This project builds a large-scale anime recommendation and graph analysis system using user interaction data and metadata.

## Dataset
- User Animelist Dataset (Kaggle)
- MyAnimeList Metadata Dataset (Kaggle)

## Reproducibility

1. Download datasets from Kaggle (https://www.kaggle.com/datasets/ramazanturann/user-animelist-dataset?select=animes.csv)
2. Place them in `data/raw/`
3. Run:

```bash
python src/build_catalog.py
python src/build_interactions.py
