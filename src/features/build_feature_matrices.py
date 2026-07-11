from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
# =========================================================
# LOAD DATASET
# =========================================================

anime_df = pd.read_csv(
    PROCESSED_DIR / "anime_dataset.csv"
)

if "tags" not in anime_df.columns and "themes" in anime_df.columns:
    anime_df = anime_df.rename(columns={"themes": "tags"})

if "tags" in anime_df.columns and "themes" in anime_df.columns:
    anime_df["tags"] = (
        anime_df["tags"].fillna("") + "|" + anime_df["themes"].fillna("")
    ).str.strip("|")
    anime_df = anime_df.drop(columns=["themes"])

if "explicit_genres" in anime_df.columns and "explicit_tags" not in anime_df.columns:
    anime_df = anime_df.rename(columns={"explicit_genres": "explicit_tags"})

if "explicit_genre_weights" in anime_df.columns and "explicit_tag_weights" not in anime_df.columns:
    anime_df = anime_df.rename(columns={"explicit_genre_weights": "explicit_tag_weights"})

if "explicit_tags" not in anime_df.columns:
    anime_df["explicit_tags"] = ""

if "tag_weights" not in anime_df.columns:
    anime_df["tag_weights"] = ""

if "explicit_tag_weights" not in anime_df.columns:
    anime_df["explicit_tag_weights"] = ""

print(f"Loaded anime dataset: {anime_df.shape}")


def expand_weighted_tags(value):
    if pd.isna(value) or not value:
        return ""

    weighted_terms = []

    for item in str(value).split("|"):
        if not item:
            continue

        tag, _, weight_text = item.rpartition(":")

        if not tag:
            tag = item

        try:
            weight = int(weight_text)
        except ValueError:
            weight = 0

        repeats = max(1, weight // 100)
        weighted_terms.extend([tag] * repeats)

    return " ".join(weighted_terms)

# =========================================================
# CLEAN TEXT FIELDS
# =========================================================

text_cols = [
    "genres",
    "explicit_tags",
    "tags",
    "tag_weights",
    "explicit_tag_weights",
    "studios",
    "demographics"
]

for col in text_cols:
    anime_df[col] = anime_df[col].fillna("")

# =========================================================
# COMBINED TEXT REPRESENTATION
# =========================================================

anime_df["weighted_tags_text"] = anime_df["tag_weights"].apply(
    expand_weighted_tags
)

anime_df["weighted_explicit_tags_text"] = anime_df[
    "explicit_tag_weights"
].apply(
    expand_weighted_tags
)

anime_df["combined_text"] = (
    anime_df["genres"] + " " +
    anime_df["explicit_tags"] + " " +
    anime_df["tags"] + " " +
    anime_df["weighted_tags_text"] + " " +
    anime_df["weighted_explicit_tags_text"] + " " +
    anime_df["studios"] + " " +
    anime_df["demographics"]
)

print("Combined text created")
# =========================================================
# TF-IDF MATRIX
# =========================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words="english",
    min_df=3,
    max_df=0.85
)

X_tfidf = vectorizer.fit_transform(
    anime_df["combined_text"]
)

print(
    f"TF-IDF matrix shape: {X_tfidf.shape}"
)

save_npz(
    EMBEDDINGS_DIR / "tfidf_matrix.npz",
    X_tfidf
)

joblib.dump(
    vectorizer,
    EMBEDDINGS_DIR / "tfidf_vectorizer.pkl"
)
# =========================================================
# MULTI-HOT GENRE ENCODING
# =========================================================

anime_df["genre_list"] = anime_df[
    "genres"
].apply(
    lambda x: x.split("|") if x else []
)

mlb = MultiLabelBinarizer()

X_genres = mlb.fit_transform(
    anime_df["genre_list"]
)

genre_df = pd.DataFrame(
    X_genres,
    columns=mlb.classes_,
    index=anime_df.index
)

genre_df.to_csv(
    EMBEDDINGS_DIR / "genre_features.csv",
    index=False
)

joblib.dump(
    mlb,
    EMBEDDINGS_DIR / "genre_mlb.pkl"
)

print(
    f"Genre matrix shape: {X_genres.shape}"
)
# =========================================================
# NUMERIC FEATURES
# =========================================================
  
numeric_cols = [
      "score",
      "members",
      "favorites",
      "episodes",
      "popularity",
      "rank",
      "scored_by",
  ]
  
numeric_df = anime_df[numeric_cols].copy()
  
numeric_df = numeric_df.fillna(0)
  
scaler = StandardScaler()
  
X_numeric = scaler.fit_transform(numeric_df)
  
numeric_output = pd.DataFrame(
      X_numeric,
      columns=numeric_cols
)
  
numeric_output.to_csv(
      EMBEDDINGS_DIR / "numeric_features.csv",
      index=False
)
  
joblib.dump(
      scaler,
      EMBEDDINGS_DIR / "numeric_scaler.pkl"
)
  
print(
      f"Numeric feature matrix shape: {X_numeric.shape}"
)

# =========================================================
# SAVE METADATA INDEX
# =========================================================

anime_df[[
    "mal_id",
    "title",
    "type",
    "score"
]].to_csv(
    EMBEDDINGS_DIR / "anime_index.csv",
    index=False
)

print("Feature matrices generated successfully")
