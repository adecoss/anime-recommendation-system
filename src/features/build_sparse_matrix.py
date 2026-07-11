from pathlib import Path
import joblib
import numpy as np
import pandas as pd

from scipy.sparse import csr_matrix
from scipy.sparse import save_npz

# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
EMBEDDINGS_DIR = ARTIFACTS_DIR / "embeddings"

EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
# =========================================================
# LOAD RATINGS
# =========================================================

print("Loading ratings dataset...")

ratings = pd.read_csv(PROCESSED_DIR / "current_user_ratings.csv")
if "animeID" not in ratings.columns and "mal_id" in ratings.columns:
    ratings = ratings.rename(columns={"mal_id": "animeID"})

print(f"Ratings shape: {ratings.shape}")
# =========================================================
# FILTER ACTIVE USERS
# =========================================================

MIN_USER_RATINGS = 50
MIN_ANIME_RATINGS = 200

print("Filtering active users...")

user_counts = ratings.groupby("userID").size()

active_users = user_counts[
    user_counts >= MIN_USER_RATINGS
].index

ratings = ratings[
    ratings["userID"].isin(active_users)
]

print(
    f"Remaining after user filtering: {ratings.shape}"
)

# =========================================================
# FILTER POPULAR ANIME
# =========================================================

print("Filtering popular anime...")

anime_counts = ratings.groupby("animeID").size()

popular_anime = anime_counts[
    anime_counts >= MIN_ANIME_RATINGS
].index

ratings = ratings[
    ratings["animeID"].isin(popular_anime)
]

print(
    f"Remaining after anime filtering: {ratings.shape}"
)

# =========================================================
# CREATE INDEX MAPPINGS
# =========================================================

print("Creating index mappings...")

unique_users = ratings["userID"].unique()
unique_anime = ratings["animeID"].unique()

user_to_idx = {
    user_id: idx
    for idx, user_id in enumerate(unique_users)
}

anime_to_idx = {
    anime_id: idx
    for idx, anime_id in enumerate(unique_anime)
}

idx_to_user = {
    idx: user_id
    for user_id, idx in user_to_idx.items()
}

idx_to_anime = {
    idx: anime_id
    for anime_id, idx in anime_to_idx.items()
}

# =========================================================
# CONVERT TO MATRIX INDICES
# =========================================================

ratings["user_idx"] = ratings[
    "userID"
].map(user_to_idx)

ratings["anime_idx"] = ratings[
    "animeID"
].map(anime_to_idx)

# =========================================================
# BUILD SPARSE MATRIX
# =========================================================

print("Building sparse matrix...")

user_item_matrix = csr_matrix(
    (
        ratings["rating"].astype(np.float32),
        (
            ratings["user_idx"],
            ratings["anime_idx"]
        )
    )
)

print(
    f"Sparse matrix shape: {user_item_matrix.shape}"
)

print(
    f"Non-zero entries: {user_item_matrix.nnz}"
)

# =========================================================
# SPARSITY ANALYSIS
# =========================================================

num_users, num_items = user_item_matrix.shape

possible_interactions = num_users * num_items

sparsity = (
    1 - (user_item_matrix.nnz / possible_interactions)
)

print(f"Matrix sparsity: {sparsity:.6f}")

# =========================================================
# SAVE MATRIX
# =========================================================

save_npz(
    EMBEDDINGS_DIR / "user_item_matrix.npz",
    user_item_matrix
)

# =========================================================
# SAVE FILTERED RATINGS
# =========================================================

ratings.to_csv(
    EMBEDDINGS_DIR / "ratings_filtered.csv",
    index=False
)

# =========================================================
# SAVE MAPPINGS
# =========================================================

joblib.dump(
    user_to_idx,
    EMBEDDINGS_DIR / "user_to_idx.pkl"
)

joblib.dump(
    anime_to_idx,
    EMBEDDINGS_DIR / "anime_to_idx.pkl"
)

joblib.dump(
    idx_to_user,
    EMBEDDINGS_DIR / "idx_to_user.pkl"
)

joblib.dump(
    idx_to_anime,
    EMBEDDINGS_DIR / "idx_to_anime.pkl"
)

# =========================================================
# SAVE MATRIX STATS
# =========================================================

stats = {
    "num_users": int(num_users),
    "num_anime": int(num_items),
    "num_ratings": int(user_item_matrix.nnz),
    "sparsity": float(sparsity)
}

joblib.dump(
    stats,
    EMBEDDINGS_DIR / "matrix_stats.pkl"
)

print("Sparse matrix pipeline complete")
