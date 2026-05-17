from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "feature_matrices"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

ANIME_PATH = PROCESSED_DIR / "anime_dataset.csv"
MATRIX_PATH = ARTIFACT_DIR / "catalog_feature_matrix.npz"
METADATA_PATH = ARTIFACT_DIR / "catalog_feature_metadata.csv"
SUMMARY_PATH = ARTIFACT_DIR / "catalog_feature_summary.json"


def split_pipe(value: object) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_duration_minutes(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return np.nan

    if isinstance(value, (int, float)) and not pd.isna(value):
        return float(value) if float(value) > 0 else np.nan

    text = str(value).strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        numeric = float(text)
        return numeric if numeric > 0 else np.nan

    total = 0.0
    hours = re.search(r"(\d+(?:\.\d+)?)\s*(?:hr|hour)", text)
    minutes = re.search(r"(\d+(?:\.\d+)?)\s*min", text)
    seconds = re.search(r"(\d+(?:\.\d+)?)\s*sec", text)

    if hours:
        total += float(hours.group(1)) * 60
    if minutes:
        total += float(minutes.group(1))
    if seconds:
        total += float(seconds.group(1)) / 60

    return total if total > 0 else np.nan


def infer_season(month: object) -> object:
    if pd.isna(month):
        return np.nan

    month = int(month)
    if month in [1, 2, 3]:
        return "winter"
    if month in [4, 5, 6]:
        return "spring"
    if month in [7, 8, 9]:
        return "summer"
    if month in [10, 11, 12]:
        return "fall"
    return np.nan


def main() -> None:
    anime_df = pd.read_csv(ANIME_PATH)
    anime_df = anime_df.rename(
        columns={
            "explicit_genres": "explicit_tags",
            "explicit_genre_weights": "explicit_tag_weights",
        }
    )

    anime_df["duration_minutes"] = anime_df["duration"].apply(parse_duration_minutes)
    anime_df["season_final"] = anime_df["season"].fillna(
        anime_df["aired_month"].apply(infer_season)
    )
    anime_df["total_watch_minutes"] = (
        anime_df["episodes"] * anime_df["duration_minutes"]
    )

    for col in [
        "genres",
        "tags",
        "explicit_tags",
        "demographics",
        "studios",
        "synopsis",
        "type",
        "rating",
        "season_final",
    ]:
        if col not in anime_df:
            anime_df[col] = ""
        anime_df[col] = anime_df[col].fillna("")

    numeric_cols = [
        "score",
        "scored_by",
        "rank",
        "popularity",
        "members",
        "favorites",
        "episodes",
        "duration_minutes",
        "total_watch_minutes",
        "aired_year",
        "aired_month",
    ]
    numeric_cols = [col for col in numeric_cols if col in anime_df.columns]
    numeric_matrix = sparse.csr_matrix(
        StandardScaler().fit_transform(anime_df[numeric_cols].fillna(0))
    )

    categorical_specs = {
        "genres": anime_df["genres"].apply(split_pipe).tolist(),
        "demographics": anime_df["demographics"].apply(split_pipe).tolist(),
        "type": anime_df["type"].apply(lambda value: [value] if value else []).tolist(),
        "rating": anime_df["rating"].apply(lambda value: [value] if value else []).tolist(),
        "season": anime_df["season_final"].apply(lambda value: [value] if value else []).tolist(),
    }

    categorical_matrices = []
    categorical_feature_count = 0
    for values in categorical_specs.values():
        encoder = MultiLabelBinarizer()
        matrix = encoder.fit_transform(values)
        categorical_matrices.append(sparse.csr_matrix(matrix))
        categorical_feature_count += len(encoder.classes_)

    text = (
        anime_df["synopsis"]
        + " "
        + anime_df["genres"]
        + " "
        + anime_df["tags"]
        + " "
        + anime_df["explicit_tags"]
        + " "
        + anime_df["studios"]
        + " "
        + anime_df["demographics"]
    )

    vectorizer = TfidfVectorizer(
        max_features=4000,
        min_df=3,
        max_df=0.85,
        stop_words="english",
    )
    text_matrix = vectorizer.fit_transform(text)

    feature_matrix = sparse.hstack(
        [numeric_matrix, *categorical_matrices, text_matrix],
        format="csr",
    )

    save_npz(MATRIX_PATH, feature_matrix)
    anime_df[
        ["mal_id", "title", "type", "score", "members", "genres", "tags"]
    ].to_csv(METADATA_PATH, index=False)

    summary = {
        "rows": int(feature_matrix.shape[0]),
        "columns": int(feature_matrix.shape[1]),
        "numeric_features": len(numeric_cols),
        "categorical_features": int(categorical_feature_count),
        "tfidf_features": int(len(vectorizer.get_feature_names_out())),
        "nonzero_values": int(feature_matrix.nnz),
        "sparsity": float(1 - feature_matrix.nnz / (feature_matrix.shape[0] * feature_matrix.shape[1])),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Saved matrix to {MATRIX_PATH}")
    print(f"Saved metadata to {METADATA_PATH}")


if __name__ == "__main__":
    main()
