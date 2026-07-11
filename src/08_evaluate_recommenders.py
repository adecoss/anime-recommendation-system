from __future__ import annotations

import json
import math
import os
import argparse
import gc
import importlib.util
import time
import warnings
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names.*",
    category=UserWarning,
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler


# Recommender is a ranking/recommendation experiment, not a classic supervised
# predictor. The script uses historical MAL user ratings as implicit feedback:
# if a user gave an anime a score >= LIKE_THRESHOLD, we treat that as a "liked"
# interaction. The goal is to rank unseen candidate anime so that one hidden
# liked anime for each user appears as high as possible.
#
# The evaluated methods are intentionally practical recommender families:
#
# 1. popularity_baseline
#    Non-personalized baseline. It ranks anime by how often they appear as
#    liked in the training ratings. This is strong for beginners because many
#    users like famous, high-visibility titles, but it repeats obvious picks.
#
# 2. metadata_content
#    Content-based profile matcher. It uses catalog genres, tags,
#    demographics, studios, source, rating, and type. This is useful for sparse
#    profiles and filter-like behavior.
#
# 3. graph_related
#    Graph recommender. It follows relation/recommendation links from liked
#    titles, so it captures sequel/navigation and "because users linked these"
#    evidence.
#
# 4. people_staff_affinity
#    People-signal recommender. It uses shared voice actors and key staff
#    edges, which supports "more anime with this actor/director/creator" rows.
#
# 5. latent_svd
#    Personalized collaborative model. It factorizes the user-anime interaction
#    matrix with TruncatedSVD. Users and anime are represented in the same
#    latent taste space, and a user score for an anime is the dot product
#    between the user vector and the anime vector.
#
# 6. full_product_hybrid, tuned_product_hybrid, and level_tuned_product_hybrid
#    Product-safe blends. They combine collaborative, popularity, metadata,
#    graph, and people/staff signals. The level-tuned variant chooses different
#    blend percentages for Beginner, Casual, Fan, and Veteran users.
#

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "recommendation"
PLOT_DIR = BASE_DIR / "artifacts" / "plots" / "recommender"

CURRENT_RATINGS_PATH = DATA_DIR / "current_user_ratings.csv"
RATINGS_PATH = CURRENT_RATINGS_PATH
CATALOG_PATH = DATA_DIR / "anime_dataset.csv"
MY_LIST_PATH = BASE_DIR / "data" / "raw" / "MyList.xml"
VOICE_ACTOR_EDGES_PATH = DATA_DIR / "anime_voice_actor_edges.csv"
STAFF_EDGES_PATH = DATA_DIR / "anime_staff_edges.csv"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


LIKE_THRESHOLD = 7
MIN_USER_POSITIVES = 5
MIN_ITEM_POSITIVES = 10
MAX_MODEL_USERS = 100_000
MAX_EVAL_USERS: int | None = None
NEGATIVES_PER_USER = 100
EVALUATION_PROGRESS_INTERVAL = 2_000
PRODUCT_ROW_K = 12
NEGATIVE_SENSITIVITY_SIZES = [100, 500, 1_000]
MAX_SENSITIVITY_USERS = 3_000
FULL_CATALOG_EVAL_USERS = 1_000
SVD_COMPONENTS = 64
FULL_HYBRID_WEIGHTS = {
    "latent_svd": 0.50,
    "popularity": 0.15,
    "metadata_content": 0.15,
    "graph_related": 0.12,
    "people_staff_affinity": 0.08,
}
DEFAULT_TUNED_HYBRID_WEIGHTS = {
    "latent_svd": 0.45,
    "item_knn_collaborative": 0.15,
    "popularity": 0.12,
    "metadata_content": 0.12,
    "graph_related": 0.11,
    "people_staff_affinity": 0.05,
}
TUNED_HYBRID_TUNING_USERS = 3_000
TUNED_HYBRID_RANDOM_CANDIDATES = 120
ADVANCED_RERANKER_TRAIN_CASES = 6_000
ADVANCED_RERANKER_EVAL_CASES = 10_000
ADVANCED_TORCH_TRAIN_PAIRS = 320_000
ADVANCED_TORCH_EPOCHS = 6
RANDOM_STATE = 42
CHUNK_SIZE = 2_000_000


EVALUATION_PATH = ARTIFACT_DIR / "evaluation_metrics.csv"
RUN_SUMMARY_PATH = ARTIFACT_DIR / "recommendation_summary.json"
EXAMPLES_PATH = ARTIFACT_DIR / "recommendation_examples.csv"
ALIGNMENT_PATH = ARTIFACT_DIR / "data_alignment.csv"
USER_EVAL_PATH = ARTIFACT_DIR / "user_level_eval_sample.csv"
LEVEL_METRICS_PATH = ARTIFACT_DIR / "metrics_by_user_level.csv"
BEGINNER_CANDIDATES_PATH = ARTIFACT_DIR / "beginner_entrypoint_candidates.csv"
MY_LIST_RECOMMENDATIONS_PATH = ARTIFACT_DIR / "mylist_recommendation_example.csv"
MY_LIST_GUARDED_RECOMMENDATIONS_PATH = ARTIFACT_DIR / "mylist_guarded_recommendation_example.csv"
MY_LIST_GUARDRAIL_COMPARISON_PATH = ARTIFACT_DIR / "mylist_guardrail_comparison.csv"
MY_LIST_GUARDRAIL_SUMMARY_PATH = ARTIFACT_DIR / "mylist_guardrail_block_summary.csv"
HYBRID_COMPONENT_SEARCH_PATH = ARTIFACT_DIR / "hybrid_component_weight_search.csv"
LEVEL_HYBRID_COMPONENT_SEARCH_PATH = ARTIFACT_DIR / "level_hybrid_component_weight_search.csv"
ADVANCED_RANKER_METRICS_PATH = ARTIFACT_DIR / "advanced_ranker_metrics.csv"
ADVANCED_RANKER_LEVEL_METRICS_PATH = ARTIFACT_DIR / "advanced_ranker_metrics_by_user_level.csv"
ADVANCED_RANKER_EVAL_PATH = ARTIFACT_DIR / "advanced_ranker_eval_rows.csv"
ADVANCED_ARCHITECTURE_INVENTORY_PATH = ARTIFACT_DIR / "advanced_architecture_inventory.csv"
ADVANCED_TRAINING_LOG_PATH = ARTIFACT_DIR / "advanced_training_log.csv"
ADVANCED_MODEL_CONFIG_PATH = ARTIFACT_DIR / "advanced_model_configs.csv"
CLASSICAL_TRAINING_LOG_PATH = ARTIFACT_DIR / "classical_training_log.csv"
NEGATIVE_SENSITIVITY_PATH = ARTIFACT_DIR / "negative_sampling_sensitivity.csv"
FULL_CATALOG_EVAL_PATH = ARTIFACT_DIR / "full_catalog_eval_sample.csv"
ERROR_CASES_PATH = ARTIFACT_DIR / "systematic_error_cases.csv"
ADVANCED_ERROR_CASES_PATH = ARTIFACT_DIR / "advanced_systematic_error_cases.csv"
ALIGNMENT_CONTRACT_PATH = ARTIFACT_DIR / "alignment_contract.csv"
RANK_DISTRIBUTION_PLOT = PLOT_DIR / "rank_distribution_by_method.png"
LEVEL_METHOD_PLOT = PLOT_DIR / "method_hit_by_user_level.png"
MYLIST_PLOT = PLOT_DIR / "mylist_top_recommendations.png"
MYLIST_GUARDED_PLOT = PLOT_DIR / "mylist_guarded_top_recommendations.png"
DISCOVERY_METRICS_PLOT = PLOT_DIR / "discovery_metrics.png"

ADVANCED_TRAINING_LOGS: list[dict[str, object]] = []
CLASSICAL_TRAINING_LOGS: list[dict[str, object]] = []

USER_LEVELS = [
    {
        "level": "Beginner",
        "min_known_entries": 1,
        "max_known_entries": 49,
        "primary_signal": "popular, high-score, short entry points aligned with early liked genres",
        "risk": "too little signal for aggressive novelty; keep recommendations safe and recognizable",
    },
    {
        "level": "Casual",
        "min_known_entries": 50,
        "max_known_entries": 149,
        "primary_signal": "similar anime, direct relations, popularity prior, and high-rated anchors",
        "risk": "profile is usable but still easy to overfit to a few early favorites",
    },
    {
        "level": "Fan",
        "min_known_entries": 150,
        "max_known_entries": 499,
        "primary_signal": "collaborative ranking, relation continuation, current shows, people/studio signals, controlled novelty",
        "risk": "many obvious titles are known; recommendations must balance familiarity and discovery",
    },
    {
        "level": "Veteran",
        "min_known_entries": 500,
        "max_known_entries": None,
        "primary_signal": "long-tail discovery, graph expansion, staff/VA paths, novelty, obscure catalog coverage",
        "risk": "hardest group: obvious catalog items are saturated and evaluation recovery is strict",
    },
]

CONTROL_FILTERS = {
    "genre_filter": "include/exclude broad MAL genres",
    "demographic_filter": "Josei, Kodomo, Seinen, Shoujo, Shounen, 18+",
    "content_rating_filter": "G, PG, PG-13, R, R+, Rx",
    "explicit_toggle": "exclude or include Ecchi/Erotica/Hentai and explicit tags",
    "score_floor": "minimum MAL score when a user wants safer picks",
    "episode_length_filter": "episode count and total watchtime ranges",
    "year_filter": "release year or season recency range",
}

PREREQUISITE_RELATIONS = {"Prequel", "Parent Story", "Full Story"}
SIDE_CONTENT_TYPES = {"Special", "TV Special", "OVA"}


@dataclass(frozen=True)
class EvalCase:
    user_id: int
    holdout_item: int
    train_items: tuple[int, ...]
    candidate_items: np.ndarray


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(path)


def project_path(value: object) -> object:
    """Return paths relative to the project folder for notebook/user-facing output."""
    if not isinstance(value, (str, Path)):
        return value
    text = str(value)
    try:
        path = Path(text)
        if path.is_absolute():
            return str(path.relative_to(BASE_DIR))
    except (ValueError, OSError):
        pass
    base_text = str(BASE_DIR)
    if text.startswith(base_text):
        return text[len(base_text) :].lstrip("\\/")
    return text


def relativize_payload(value: object) -> object:
    if isinstance(value, dict):
        return {key: relativize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_payload(item) for item in value]
    return project_path(value)


def log_step(message: str) -> None:
    print(message, flush=True)


def progress_bar(index: int, total: int, width: int = 22) -> str:
    filled = int(round(width * index / max(total, 1)))
    return "[" + "#" * filled + "." * (width - filled) + "]"


def log_stage(index: int, total: int, message: str) -> None:
    print(f"STAGE {index:02d}/{total:02d} {progress_bar(index, total)} | {message}", flush=True)


def record_training_result(
    model_name: str,
    family: str,
    started_at: float,
    train_rows: int,
    eval_rows: int,
    status: str = "trained",
    note: str = "",
) -> None:
    elapsed = time.perf_counter() - started_at
    ADVANCED_TRAINING_LOGS.append(
        {
            "model": model_name,
            "family": family,
            "status": status,
            "train_rows": int(train_rows),
            "eval_rows": int(eval_rows),
            "elapsed_seconds": round(float(elapsed), 3),
            "note": note,
        }
    )
    print(f"{model_name}: {status} in {elapsed:.1f}s {note}".strip(), flush=True)


def record_classical_result(
    component_name: str,
    family: str,
    started_at: float,
    train_rows: int,
    eval_rows: int,
    status: str = "built",
    note: str = "",
) -> None:
    elapsed = time.perf_counter() - started_at
    CLASSICAL_TRAINING_LOGS.append(
        {
            "component": component_name,
            "family": family,
            "status": status,
            "train_rows": int(train_rows),
            "eval_rows": int(eval_rows),
            "elapsed_seconds": round(float(elapsed), 3),
            "note": note,
        }
    )
    print(f"{component_name}: {status} in {elapsed:.1f}s {note}".strip(), flush=True)


def single_holdout_metric_fields(rank: int, k: int = PRODUCT_ROW_K) -> dict[str, float | int]:
    """Metrics for the one-held-out-liked-anime evaluation setup.

    With one relevant item per user, Recall@K is the same as Hit@K. MAP@K is
    also a simple reciprocal-rank value when the holdout is inside the first K.
    Keeping the names explicit makes the offline report match the product rows,
    which show 12 anime per row.
    """
    hit = int(rank <= k)
    return {
        f"hit_at_{k}": hit,
        f"recall_at_{k}": hit,
        f"ndcg_at_{k}": float(1.0 / math.log2(rank + 1) if hit else 0.0),
        f"map_at_{k}": float(1.0 / rank if hit else 0.0),
    }


def load_catalog() -> pd.DataFrame:
    cols = [
        "mal_id",
        "title",
        "type",
        "score",
        "members",
        "popularity",
        "genres",
        "tags",
        "explicit_tags",
        "demographics",
        "studios",
        "rating",
        "source",
        "episodes",
        "duration",
        "total_watch_minutes",
        "aired_year",
        "relations",
        "recommendations",
    ]
    available = pd.read_csv(CATALOG_PATH, nrows=0).columns
    usecols = [col for col in cols if col in available]
    catalog = pd.read_csv(CATALOG_PATH, usecols=usecols)
    catalog["mal_id"] = catalog["mal_id"].astype(np.int32)
    return catalog


def load_positive_ratings(catalog_ids: set[int]) -> pd.DataFrame:
    """Load only useful implicit-feedback rows from the ratings file.

    The ratings table can be large, so it is streamed in chunks. We keep rows
    only when:
    - the rating is >= LIKE_THRESHOLD, meaning the user probably liked it;
    - the anime id exists in our current catalog, so it can be recommended.
    - when list status exists, the row is not dropped.

    Low ratings and dropped rows are not used as positive seeds. They are still
    useful product information and should be used by the product guardrail, but
    this collaborative matrix is "liked or not observed", not "like/dislike".
    """
    if not RATINGS_PATH.exists():
        raise FileNotFoundError(f"Ratings file not found: {RATINGS_PATH}")
    available = pd.read_csv(RATINGS_PATH, nrows=0).columns
    usecols = ["userID", "animeID", "rating"] + (["status"] if "status" in available else [])
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_kept = 0

    for chunk in pd.read_csv(
        RATINGS_PATH,
        usecols=usecols,
        dtype={"userID": "string", "animeID": "string", "rating": "string", "status": "string"},
        chunksize=CHUNK_SIZE,
    ):
        rows_read += len(chunk)
        chunk["userID"] = pd.to_numeric(chunk["userID"], errors="coerce")
        chunk["animeID"] = pd.to_numeric(chunk["animeID"], errors="coerce")
        chunk["rating"] = pd.to_numeric(chunk["rating"], errors="coerce")
        chunk.dropna(subset=["userID", "animeID", "rating"], inplace=True)
        chunk["userID"] = chunk["userID"].astype(np.int64)
        chunk["animeID"] = chunk["animeID"].astype(np.int32)
        chunk["rating"] = chunk["rating"].astype(np.float32)
        chunk = chunk[(chunk["rating"] >= LIKE_THRESHOLD) & (chunk["animeID"].isin(catalog_ids))]
        if "status" in chunk.columns:
            chunk = chunk[~chunk["status"].fillna("").str.casefold().eq("dropped")]
        if not chunk.empty:
            chunks.append(chunk[["userID", "animeID"]].copy())
            rows_kept += len(chunk)
        log_step(f"streamed ratings rows={rows_read:,}; positives kept={rows_kept:,}")

    if not chunks:
        raise RuntimeError("No positive ratings found. Check current_user_ratings.csv and catalog ids.")

    positives = pd.concat(chunks, ignore_index=True)
    positives.drop_duplicates(["userID", "animeID"], inplace=True)
    positives["userID"] = positives["userID"].astype(np.int64)
    positives["animeID"] = positives["animeID"].astype(np.int32)
    return positives


def deterministic_holdout(user_id: int, items: Iterable[int]) -> tuple[int, tuple[int, ...]]:
    unique_items = sorted(set(int(item) for item in items))
    index = (user_id * 1_103_515_245 + 12_345) % len(unique_items)
    holdout = unique_items[index]
    train_items = tuple(item for item in unique_items if item != holdout)
    return holdout, train_items


def stable_user_sample(user_ids: np.ndarray, size: int | None, random_state: int) -> np.ndarray:
    user_ids = np.array(sorted(set(int(user_id) for user_id in user_ids)), dtype=np.int64)
    if size is None or len(user_ids) <= size:
        return user_ids
    rng = np.random.default_rng(random_state)
    selected = rng.choice(user_ids, size=size, replace=False)
    return np.array(sorted(selected), dtype=np.int64)


def user_level_from_count(known_entries: int) -> str:
    for level in USER_LEVELS:
        maximum = level["max_known_entries"]
        if known_entries >= level["min_known_entries"] and (maximum is None or known_entries <= maximum):
            return level["level"]
    return USER_LEVELS[-1]["level"]


def prepare_split(
    positives: pd.DataFrame,
    candidate_items: np.ndarray,
    negatives_per_user: int = NEGATIVES_PER_USER,
    max_eval_users: int = MAX_EVAL_USERS,
) -> tuple[pd.DataFrame, dict[int, EvalCase], np.ndarray]:
    """Create a leave-one-liked-anime-out ranking test.

    For each eligible user, one liked anime is hidden as the holdout. The model
    trains/ranks from the rest of the user's liked anime. At evaluation time the
    candidate list contains:
    - the hidden liked anime, which should be ranked highly;
    - NEGATIVES_PER_USER sampled catalog anime the user did not like/record.

    This is a standard top-k recommender test: "Can the system recover a known
    future/held-out positive item from a noisy candidate set?"
    """
    candidate_set = set(int(item) for item in candidate_items)
    positives = positives[positives["animeID"].isin(candidate_set)].copy()

    user_counts = positives.groupby("userID")["animeID"].nunique()
    eligible_users = user_counts[user_counts >= MIN_USER_POSITIVES].index.to_numpy(dtype=np.int64)
    eval_users = stable_user_sample(eligible_users, max_eval_users, RANDOM_STATE)

    model_user_pool = stable_user_sample(eligible_users, MAX_MODEL_USERS, RANDOM_STATE + 7)
    model_users = np.array(sorted(set(model_user_pool).union(set(eval_users))), dtype=np.int64)

    grouped = positives[positives["userID"].isin(eval_users)].groupby("userID")["animeID"].apply(list)
    all_candidate_items = np.array(sorted(candidate_set), dtype=np.int32)
    eval_cases: dict[int, EvalCase] = {}
    rng = np.random.default_rng(RANDOM_STATE + 101)

    for user_id, items in grouped.items():
        holdout, train_items = deterministic_holdout(int(user_id), items)
        blocked = set(train_items)
        blocked.add(holdout)
        negative_pool = np.array([item for item in all_candidate_items if item not in blocked], dtype=np.int32)
        if len(negative_pool) < negatives_per_user:
            continue
        negatives = rng.choice(negative_pool, size=negatives_per_user, replace=False)
        candidate_list = np.array([holdout, *negatives.tolist()], dtype=np.int32)
        eval_cases[int(user_id)] = EvalCase(
            user_id=int(user_id),
            holdout_item=int(holdout),
            train_items=tuple(int(item) for item in train_items),
            candidate_items=candidate_list,
        )

    return positives, eval_cases, model_users


def rebuild_candidate_sets(
    eval_cases: dict[int, EvalCase],
    candidate_items: np.ndarray,
    negatives_per_user: int | None,
    max_users: int,
    random_state: int,
) -> dict[int, EvalCase]:
    """Reuse the same holdouts while changing candidate-pool difficulty.

    The Recommender feedback correctly noted that 100 sampled negatives is easier
    than ranking against the whole catalog. This helper keeps the training split
    fixed and only changes the alternatives shown to each method:
    - if negatives_per_user is an integer, sample that many unseen negatives;
    - if it is None, rank against all candidate-pool anime not in the user's
      training profile, which approximates a full-catalog ranking test.
    """
    all_items = np.array(sorted(int(item) for item in candidate_items), dtype=np.int32)
    rng = np.random.default_rng(random_state)
    selected_cases = list(eval_cases.values())[:max_users]
    rebuilt: dict[int, EvalCase] = {}

    for case in selected_cases:
        blocked = set(int(item) for item in case.train_items)
        blocked.add(int(case.holdout_item))
        negative_pool = np.array([item for item in all_items if int(item) not in blocked], dtype=np.int32)
        if negatives_per_user is None:
            negatives = negative_pool
        else:
            if len(negative_pool) < negatives_per_user:
                continue
            negatives = rng.choice(negative_pool, size=negatives_per_user, replace=False)
        rebuilt[case.user_id] = EvalCase(
            user_id=case.user_id,
            holdout_item=case.holdout_item,
            train_items=case.train_items,
            candidate_items=np.array([case.holdout_item, *negatives.tolist()], dtype=np.int32),
        )

    return rebuilt


def build_training_matrix(
    positives: pd.DataFrame,
    model_users: np.ndarray,
    candidate_items: np.ndarray,
    eval_cases: dict[int, EvalCase],
) -> tuple[csr_matrix, dict[int, int], dict[int, int], pd.DataFrame]:
    """Build the sparse user-anime matrix used by collaborative filtering.

    Rows are users, columns are anime, and values are 1 when the user liked the
    anime. This is implicit feedback: the value does not store the exact rating,
    only that the rating crossed the "good enough to count as liked" threshold.

    The held-out anime for evaluation users is removed before training. That
    prevents leakage: the model cannot be rewarded for memorizing the answer.
    """
    user_to_row = {int(user_id): idx for idx, user_id in enumerate(model_users)}
    item_to_col = {int(item_id): idx for idx, item_id in enumerate(candidate_items)}

    train = positives[positives["userID"].isin(user_to_row.keys())].copy()
    holdouts = pd.DataFrame(
        [{"userID": case.user_id, "animeID": case.holdout_item} for case in eval_cases.values()],
        columns=["userID", "animeID"],
    )
    if not holdouts.empty:
        train = train.merge(holdouts.assign(_holdout=1), on=["userID", "animeID"], how="left")
        train = train[train["_holdout"].isna()].drop(columns=["_holdout"])

    train = train[train["animeID"].isin(item_to_col.keys())]
    row = train["userID"].map(user_to_row).to_numpy(dtype=np.int32)
    col = train["animeID"].map(item_to_col).to_numpy(dtype=np.int32)
    data = np.ones(len(train), dtype=np.float32)
    matrix = csr_matrix((data, (row, col)), shape=(len(model_users), len(candidate_items)), dtype=np.float32)
    matrix.eliminate_zeros()
    return matrix, user_to_row, item_to_col, train


def rank_of_holdout(scores: np.ndarray) -> int:
    holdout_score = scores[0]
    better = np.sum(scores[1:] > holdout_score)
    ties_before = np.sum(scores[1:] == holdout_score)
    return int(better + ties_before + 1)


def evaluate_rankings(
    eval_cases: dict[int, EvalCase],
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_scores: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    tuned_hybrid_weights: dict[str, float] | None = None,
    level_tuned_hybrid_weights: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score the same candidate set with each recommendation method.

    The candidate_items array inside each EvalCase is ordered with the holdout
    item first, followed by sampled negatives. Each method produces one score
    per candidate. We then compute the rank of the holdout. A good recommender
    gives the hidden liked anime a low rank number, ideally 1-10.
    """
    rows = []
    example_rows = []

    catalog_item_ids = np.array(list(item_to_col.keys()), dtype=np.int32)

    total_cases = len(eval_cases)
    for processed, case in enumerate(eval_cases.values(), start=1):
        if case.user_id not in user_to_row:
            continue
        candidate_cols = np.array([item_to_col[int(item)] for item in case.candidate_items], dtype=np.int32)

        component_scores = compute_component_scores(
            case,
            candidate_cols,
            item_to_col,
            user_to_row,
            user_factors,
            item_factors,
            popularity_by_item,
            metadata_context,
            graph_scores,
            people_scores,
            item_knn_vectors,
        )
        method_scores = {
            # popularity_baseline: global crowd preference baseline.
            "popularity_baseline": component_scores["popularity"],
            # metadata_content: content-based similarity from tags/genres/etc.
            "metadata_content": component_scores["metadata_content"],
            # graph_related: relation/recommendation edges from liked titles.
            "graph_related": component_scores["graph_related"],
            # people_staff_affinity: shared voice actors and key staff.
            "people_staff_affinity": component_scores["people_staff_affinity"],
            # item_knn_collaborative: item-neighborhood collaborative signal.
            "item_knn_collaborative": component_scores["item_knn_collaborative"],
            # latent_svd: pure personalized collaborative signal.
            "latent_svd": component_scores["latent_svd"],
            "full_product_hybrid": blend_components(component_scores, FULL_HYBRID_WEIGHTS),
        }
        if tuned_hybrid_weights:
            method_scores["tuned_product_hybrid"] = blend_components(component_scores, tuned_hybrid_weights)
        if level_tuned_hybrid_weights:
            user_level = user_level_from_count(len(case.train_items))
            weights = level_tuned_hybrid_weights.get(user_level) or tuned_hybrid_weights
            if weights:
                method_scores["level_tuned_product_hybrid"] = blend_components(component_scores, weights)

        ranks = {method: rank_of_holdout(scores) for method, scores in method_scores.items()}
        for method, rank in ranks.items():
            profile_size = len(case.train_items)
            top_idx = np.argsort(-method_scores[method])[:PRODUCT_ROW_K]
            top12_ids = [int(case.candidate_items[i]) for i in top_idx]
            top1_idx = int(top_idx[0]) if len(top_idx) else 0
            rows.append(
                {
                    "userID": case.user_id,
                    "method": method,
                    "holdout_animeID": case.holdout_item,
                    "profile_size": profile_size,
                    "user_level": user_level_from_count(profile_size),
                    "candidate_pool_size": len(case.candidate_items),
                    "rank": rank,
                    **single_holdout_metric_fields(rank),
                    "mrr": float(1.0 / rank),
                    "holdout_score": float(method_scores[method][0]),
                    "top1_animeID": int(case.candidate_items[top1_idx]),
                    "top1_score": float(method_scores[method][top1_idx]),
                    "top12_anime_ids": "|".join(str(item) for item in top12_ids),
                }
            )

        if len(example_rows) < 200:
            if "level_tuned_product_hybrid" in method_scores:
                example_method = "level_tuned_product_hybrid"
            elif "tuned_product_hybrid" in method_scores:
                example_method = "tuned_product_hybrid"
            else:
                example_method = "full_product_hybrid"
            hybrid_scores = method_scores[example_method]
            top_idx = np.argsort(-hybrid_scores)[:PRODUCT_ROW_K]
            example_rows.append(
                {
                    "userID": case.user_id,
                    "holdout_animeID": case.holdout_item,
                    "train_anime_ids": "|".join(str(item) for item in case.train_items[:20]),
                    "profile_size": len(case.train_items),
                    "user_level": user_level_from_count(len(case.train_items)),
                    "popularity_rank": ranks["popularity_baseline"],
                    "metadata_rank": ranks["metadata_content"],
                    "graph_rank": ranks["graph_related"],
                    "people_staff_rank": ranks["people_staff_affinity"],
                    "item_knn_rank": ranks["item_knn_collaborative"],
                    "latent_svd_rank": ranks["latent_svd"],
                    "full_hybrid_rank": ranks["full_product_hybrid"],
                    "tuned_hybrid_rank": ranks.get("tuned_product_hybrid", np.nan),
                    "level_tuned_hybrid_rank": ranks.get("level_tuned_product_hybrid", np.nan),
                    "example_method": example_method,
                    "product_hybrid_top12_anime_ids": "|".join(str(int(case.candidate_items[i])) for i in top_idx),
                    "candidate_pool_size": len(case.candidate_items),
                }
            )

        if EVALUATION_PROGRESS_INTERVAL and processed % EVALUATION_PROGRESS_INTERVAL == 0:
            print(
                f"classical/product evaluation progress: processed={processed:,}/{total_cases:,}; "
                f"rows={len(rows):,}",
                flush=True,
            )
            gc.collect()

    eval_rows = pd.DataFrame(rows)
    examples = pd.DataFrame(example_rows)
    return eval_rows, examples


def generate_hybrid_weight_candidates() -> list[dict[str, float]]:
    components = list(DEFAULT_TUNED_HYBRID_WEIGHTS.keys())
    presets = [
        FULL_HYBRID_WEIGHTS | {"item_knn_collaborative": 0.0},
        DEFAULT_TUNED_HYBRID_WEIGHTS,
        {"latent_svd": 0.65, "item_knn_collaborative": 0.10, "popularity": 0.10, "metadata_content": 0.05, "graph_related": 0.07, "people_staff_affinity": 0.03},
        {"latent_svd": 0.40, "item_knn_collaborative": 0.15, "popularity": 0.10, "metadata_content": 0.15, "graph_related": 0.15, "people_staff_affinity": 0.05},
        {"latent_svd": 0.35, "item_knn_collaborative": 0.10, "popularity": 0.10, "metadata_content": 0.15, "graph_related": 0.25, "people_staff_affinity": 0.05},
        {"latent_svd": 0.40, "item_knn_collaborative": 0.10, "popularity": 0.10, "metadata_content": 0.10, "graph_related": 0.10, "people_staff_affinity": 0.20},
    ]
    rng = np.random.default_rng(RANDOM_STATE + 606)
    random_weights = rng.dirichlet(np.array([5.0, 1.6, 1.5, 1.5, 1.3, 0.9]), size=TUNED_HYBRID_RANDOM_CANDIDATES)
    candidates: list[dict[str, float]] = []
    for preset in presets:
        total = sum(max(float(preset.get(component, 0.0)), 0.0) for component in components)
        if total <= 0:
            continue
        candidates.append({component: max(float(preset.get(component, 0.0)), 0.0) / total for component in components})
    for row in random_weights:
        candidates.append({component: float(weight) for component, weight in zip(components, row)})

    unique: dict[tuple[float, ...], dict[str, float]] = {}
    for weights in candidates:
        key = tuple(round(weights.get(component, 0.0), 4) for component in components)
        unique[key] = weights
    return list(unique.values())


def tune_product_hybrid_weights(
    eval_cases: dict[int, EvalCase],
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Search useful percentage blends for the product hybrid."""
    tuning_cases = list(eval_cases.values())[: min(TUNED_HYBRID_TUNING_USERS, len(eval_cases))]
    component_cache: list[dict[str, object]] = []
    for case in tuning_cases:
        if case.user_id not in user_to_row:
            continue
        candidate_cols = np.array([item_to_col[int(item)] for item in case.candidate_items], dtype=np.int32)
        raw_components = compute_component_scores(
            case,
            candidate_cols,
            item_to_col,
            user_to_row,
            user_factors,
            item_factors,
            popularity_by_item,
            metadata_context,
            graph_scores,
            people_scores,
            item_knn_vectors,
        )
        component_cache.append(
            {
                "user_level": user_level_from_count(len(case.train_items)),
                "components": {name: normalized_vector(values) for name, values in raw_components.items()},
            }
        )

    rows = []
    candidates = generate_hybrid_weight_candidates()
    for idx, weights in enumerate(candidates, start=1):
        ranks = []
        levels = []
        for cached in component_cache:
            score = blend_components(cached["components"], weights)
            ranks.append(rank_of_holdout(score))
            levels.append(str(cached["user_level"]))
        if not ranks:
            continue
        ranks_arr = np.array(ranks, dtype=np.float32)
        row = {
            "candidate_id": idx,
            "evaluated_users": len(ranks),
            "hit_rate_at_12": float(np.mean(ranks_arr <= PRODUCT_ROW_K)),
            "recall_at_12": float(np.mean(ranks_arr <= PRODUCT_ROW_K)),
            "ndcg_at_12": float(np.mean([1.0 / math.log2(rank + 1) if rank <= PRODUCT_ROW_K else 0.0 for rank in ranks])),
            "map_at_12": float(np.mean([1.0 / rank if rank <= PRODUCT_ROW_K else 0.0 for rank in ranks])),
            "mean_reciprocal_rank": float(np.mean(1.0 / ranks_arr)),
            "median_rank": float(np.median(ranks_arr)),
        }
        for component, weight in weights.items():
            row[f"weight_{component}"] = round(float(weight), 4)
        for level in sorted(set(levels)):
            mask = np.array([item == level for item in levels], dtype=bool)
            if mask.any():
                row[f"hit_at_12_{level}"] = float(np.mean(ranks_arr[mask] <= PRODUCT_ROW_K))
        rows.append(row)

    search = pd.DataFrame(rows)
    if search.empty:
        return DEFAULT_TUNED_HYBRID_WEIGHTS, search
    search["balanced_profile_hit_at_12"] = search[
        [col for col in search.columns if col.startswith("hit_at_12_")]
    ].mean(axis=1)
    search = search.sort_values(
        ["balanced_profile_hit_at_12", "hit_rate_at_12", "ndcg_at_12", "mean_reciprocal_rank"],
        ascending=False,
    ).reset_index(drop=True)
    best = search.iloc[0]
    weights = {
        col.replace("weight_", ""): float(best[col])
        for col in search.columns
        if col.startswith("weight_")
    }
    return weights, search


def tune_product_hybrid_weights_by_level(
    eval_cases: dict[int, EvalCase],
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    fallback_weights: dict[str, float],
) -> tuple[dict[str, dict[str, float]], pd.DataFrame]:
    """Tune product-hybrid percentages separately for each user profile band.

    A Beginner, Casual, Fan, and Veteran profile do not need the same balance of
    collaborative, graph, popularity, metadata, and people/staff evidence. This
    search keeps the same candidate weight vectors as the global tuner, but
    chooses the best vector within each profile band.
    """
    tuning_cases = list(eval_cases.values())[: min(TUNED_HYBRID_TUNING_USERS, len(eval_cases))]
    component_cache: list[dict[str, object]] = []
    for case in tuning_cases:
        if case.user_id not in user_to_row:
            continue
        candidate_cols = np.array([item_to_col[int(item)] for item in case.candidate_items], dtype=np.int32)
        raw_components = compute_component_scores(
            case,
            candidate_cols,
            item_to_col,
            user_to_row,
            user_factors,
            item_factors,
            popularity_by_item,
            metadata_context,
            graph_scores,
            people_scores,
            item_knn_vectors,
        )
        component_cache.append(
            {
                "user_level": user_level_from_count(len(case.train_items)),
                "components": {name: normalized_vector(values) for name, values in raw_components.items()},
            }
        )

    candidates = generate_hybrid_weight_candidates()
    rows: list[dict[str, float | int | str]] = []
    best_by_level: dict[str, dict[str, float]] = {}
    level_names = [level["level"] for level in USER_LEVELS]
    components = list(DEFAULT_TUNED_HYBRID_WEIGHTS.keys())

    for level in level_names:
        level_cache = [cached for cached in component_cache if cached["user_level"] == level]
        if not level_cache:
            best_by_level[level] = dict(fallback_weights)
            continue
        level_rows = []
        for idx, weights in enumerate(candidates, start=1):
            ranks = []
            for cached in level_cache:
                score = blend_components(cached["components"], weights)
                ranks.append(rank_of_holdout(score))
            ranks_arr = np.array(ranks, dtype=np.float32)
            row: dict[str, float | int | str] = {
                "user_level": level,
                "candidate_id": idx,
                "evaluated_users": len(ranks),
                "hit_rate_at_12": float(np.mean(ranks_arr <= PRODUCT_ROW_K)),
                "recall_at_12": float(np.mean(ranks_arr <= PRODUCT_ROW_K)),
                "ndcg_at_12": float(np.mean([1.0 / math.log2(rank + 1) if rank <= PRODUCT_ROW_K else 0.0 for rank in ranks])),
                "map_at_12": float(np.mean([1.0 / rank if rank <= PRODUCT_ROW_K else 0.0 for rank in ranks])),
                "mean_reciprocal_rank": float(np.mean(1.0 / ranks_arr)),
                "median_rank": float(np.median(ranks_arr)),
            }
            for component in components:
                row[f"weight_{component}"] = round(float(weights.get(component, 0.0)), 4)
            level_rows.append(row)
        level_df = pd.DataFrame(level_rows).sort_values(["hit_rate_at_12", "ndcg_at_12", "mean_reciprocal_rank"], ascending=False)
        best = level_df.iloc[0]
        best_by_level[level] = {
            component: float(best[f"weight_{component}"])
            for component in components
        }
        rows.extend(level_rows)

    search = pd.DataFrame(rows)
    if not search.empty:
        search = search.sort_values(["user_level", "hit_rate_at_12", "ndcg_at_12"], ascending=[True, False, False])
    return best_by_level, search


def summarize_metrics(eval_rows: pd.DataFrame, train_matrix: csr_matrix, item_ids: np.ndarray) -> pd.DataFrame:
    """Aggregate ranking quality into Recommender-friendly metrics.

    Hit@12 asks whether the hidden liked anime appears in the visible product row.
    The product-facing row size is 12, so Hit/Recall/NDCG/MAP@12 are emitted.
    With a single held-out relevant item, Recall@12 equals Hit@12.
    MRR is 1/rank, another direct measure of early ranking quality.
    Median/mean rank help explain failures even when Hit@12 looks good.
    """
    metrics = (
        eval_rows.groupby("method")
        .agg(
            evaluated_users=("userID", "nunique"),
            hit_rate_at_12=("hit_at_12", "mean"),
            recall_at_12=("recall_at_12", "mean"),
            ndcg_at_12=("ndcg_at_12", "mean"),
            map_at_12=("map_at_12", "mean"),
            mean_reciprocal_rank=("mrr", "mean"),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
            median_candidate_pool=("candidate_pool_size", "median"),
        )
        .reset_index()
    )
    metrics["candidate_pool_per_user"] = metrics["median_candidate_pool"].round().astype(int)
    metrics.drop(columns=["median_candidate_pool"], inplace=True)
    metrics["like_threshold"] = LIKE_THRESHOLD
    metrics["min_user_positives"] = MIN_USER_POSITIVES
    metrics["min_item_positives"] = MIN_ITEM_POSITIVES
    return metrics.sort_values(["hit_rate_at_12", "ndcg_at_12", "map_at_12"], ascending=False)


def split_pipe_ids(value: object) -> list[int]:
    ids: list[int] = []
    for raw in str(value or "").split("|"):
        if not raw:
            continue
        try:
            ids.append(int(float(raw)))
        except ValueError:
            continue
    return ids


def genre_set(value: object) -> set[str]:
    if pd.isna(value):
        return set()
    return {part.strip() for part in str(value).split("|") if part.strip()}


def split_label_field(value: object) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def parse_weighted_id_pairs(value: object) -> dict[int, float]:
    """Parse fields like '1535:22|1:4' into target-weight pairs."""
    pairs: dict[int, float] = {}
    if pd.isna(value):
        return pairs
    for raw in str(value).split("|"):
        if not raw or ":" not in raw:
            continue
        left, right = raw.split(":", 1)
        try:
            target = int(float(left.strip()))
            weight = float(str(right).split(":", 1)[0].strip())
        except ValueError:
            continue
        pairs[target] = max(pairs.get(target, 0.0), weight)
    return pairs


def parse_relation_pairs(value: object) -> dict[int, float]:
    """Parse relation edges and give prerequisite/continuation edges extra weight."""
    relation_weights = {
        "Sequel": 4.0,
        "Prequel": 4.0,
        "Parent Story": 3.5,
        "Full Story": 3.5,
        "Side Story": 2.4,
        "Alternative Version": 2.0,
        "Alternative Setting": 2.0,
        "Summary": 1.2,
        "Other": 1.0,
    }
    pairs: dict[int, float] = {}
    if pd.isna(value):
        return pairs
    for raw in str(value).split("|"):
        if not raw or ":" not in raw:
            continue
        left, relation = raw.split(":", 1)
        try:
            target = int(float(left.strip()))
        except ValueError:
            continue
        weight = relation_weights.get(relation.strip(), 1.0)
        pairs[target] = max(pairs.get(target, 0.0), weight)
    return pairs


def normalized_vector(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if len(values) == 0 or values.max() <= values.min():
        return np.zeros_like(values, dtype=np.float32)
    return (values - values.min()) / (values.max() - values.min())


def build_metadata_feature_context(catalog: pd.DataFrame, candidate_items: np.ndarray) -> tuple[dict[int, dict[str, float]], dict[str, float]]:
    """Create lightweight item-token vectors from catalog metadata."""
    candidate_set = set(int(item) for item in candidate_items)
    item_tokens: dict[int, dict[str, float]] = {}
    document_frequency: dict[str, int] = {}
    weighted_fields = {
        "genres": 2.2,
        "tags": 1.3,
        "explicit_tags": 1.0,
        "demographics": 1.2,
        "studios": 1.2,
    }

    for row in catalog.itertuples(index=False):
        mal_id = int(getattr(row, "mal_id"))
        if mal_id not in candidate_set:
            continue
        tokens: dict[str, float] = {}
        for field, weight in weighted_fields.items():
            if not hasattr(row, field):
                continue
            for label in split_label_field(getattr(row, field)):
                token = f"{field}:{label.casefold()}"
                tokens[token] = max(tokens.get(token, 0.0), weight)
        for field, weight in [("type", 0.6), ("source", 0.5), ("rating", 0.5)]:
            if hasattr(row, field):
                value = getattr(row, field)
                if not pd.isna(value) and str(value).strip():
                    token = f"{field}:{str(value).strip().casefold()}"
                    tokens[token] = max(tokens.get(token, 0.0), weight)
        item_tokens[mal_id] = tokens
        for token in tokens:
            document_frequency[token] = document_frequency.get(token, 0) + 1

    n_items = max(len(item_tokens), 1)
    idf = {token: math.log1p(n_items / (1 + count)) for token, count in document_frequency.items()}
    return item_tokens, idf


def build_graph_score_context(catalog: pd.DataFrame, candidate_items: np.ndarray) -> dict[int, dict[int, float]]:
    """Build source -> target graph scores from recommendations and relations."""
    candidate_set = set(int(item) for item in candidate_items)
    graph: dict[int, dict[int, float]] = {}
    for row in catalog.itertuples(index=False):
        source = int(getattr(row, "mal_id"))
        if source not in candidate_set:
            continue
        scores: dict[int, float] = {}
        for target, weight in parse_weighted_id_pairs(getattr(row, "recommendations", np.nan)).items():
            if target in candidate_set and target != source:
                scores[target] = max(scores.get(target, 0.0), math.log1p(weight))
        for target, weight in parse_relation_pairs(getattr(row, "relations", np.nan)).items():
            if target in candidate_set and target != source:
                scores[target] = max(scores.get(target, 0.0), weight)
        graph[source] = scores
    return graph


def build_people_score_context(candidate_items: np.ndarray) -> dict[int, dict[str, float]]:
    """Build item -> people/staff weights for VA and staff affinity ranking."""
    candidate_set = set(int(item) for item in candidate_items)
    item_people: dict[int, dict[str, float]] = {int(item): {} for item in candidate_items}

    if VOICE_ACTOR_EDGES_PATH.exists():
        va_cols = ["mal_id", "voice_actor_group_key", "voice_actor_favorites", "character_relevance", "role_weight"]
        available = pd.read_csv(VOICE_ACTOR_EDGES_PATH, nrows=0).columns
        usecols = [col for col in va_cols if col in available]
        for chunk in pd.read_csv(VOICE_ACTOR_EDGES_PATH, usecols=usecols, chunksize=200_000):
            chunk["mal_id"] = pd.to_numeric(chunk["mal_id"], errors="coerce")
            chunk.dropna(subset=["mal_id"], inplace=True)
            chunk = chunk[chunk["mal_id"].astype(int).isin(candidate_set)]
            for row in chunk.itertuples(index=False):
                mal_id = int(getattr(row, "mal_id"))
                key = str(getattr(row, "voice_actor_group_key", "") or "").strip()
                if not key:
                    continue
                fav = safe_float(getattr(row, "voice_actor_favorites", 0), 0.0)
                role_weight = safe_float(getattr(row, "role_weight", 0.4), 0.4)
                relevance = safe_float(getattr(row, "character_relevance", 0.0), 0.0)
                weight = (1.0 + math.log1p(max(fav, 0.0))) * max(role_weight, 0.2) * (0.5 + relevance)
                item_people.setdefault(mal_id, {})[f"va:{key}"] = max(item_people.setdefault(mal_id, {}).get(f"va:{key}", 0.0), weight)

    if STAFF_EDGES_PATH.exists():
        staff_cols = ["mal_id", "staff_group_key", "staff_role_group", "staff_favorites"]
        available = pd.read_csv(STAFF_EDGES_PATH, nrows=0).columns
        usecols = [col for col in staff_cols if col in available]
        role_weights = {"director": 1.2, "original_creator": 1.1, "original_story": 1.0}
        for chunk in pd.read_csv(STAFF_EDGES_PATH, usecols=usecols, chunksize=200_000):
            chunk["mal_id"] = pd.to_numeric(chunk["mal_id"], errors="coerce")
            chunk.dropna(subset=["mal_id"], inplace=True)
            chunk = chunk[chunk["mal_id"].astype(int).isin(candidate_set)]
            for row in chunk.itertuples(index=False):
                mal_id = int(getattr(row, "mal_id"))
                key = str(getattr(row, "staff_group_key", "") or "").strip()
                role = str(getattr(row, "staff_role_group", "") or "").strip().casefold()
                if not key or role not in role_weights:
                    continue
                fav = safe_float(getattr(row, "staff_favorites", 0), 0.0)
                weight = (1.0 + math.log1p(max(fav, 0.0))) * role_weights[role]
                item_people.setdefault(mal_id, {})[f"staff:{key}"] = max(
                    item_people.setdefault(mal_id, {}).get(f"staff:{key}", 0.0),
                    weight,
                )

    return item_people


def safe_float(value: object, default: float = 0.0) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    if pd.isna(parsed):
        return default
    return float(parsed)


def profile_token_scores(train_items: Iterable[int], candidate_items: np.ndarray, item_tokens: dict[int, dict[str, float]], idf: dict[str, float]) -> np.ndarray:
    profile: dict[str, float] = {}
    for item in train_items:
        for token, weight in item_tokens.get(int(item), {}).items():
            profile[token] = profile.get(token, 0.0) + weight * idf.get(token, 1.0)
    scores = []
    for item in candidate_items:
        score = 0.0
        for token, weight in item_tokens.get(int(item), {}).items():
            score += profile.get(token, 0.0) * weight * idf.get(token, 1.0)
        scores.append(score)
    return np.array(scores, dtype=np.float32)


def profile_graph_scores(train_items: Iterable[int], candidate_items: np.ndarray, graph_scores: dict[int, dict[int, float]]) -> np.ndarray:
    profile_scores: dict[int, float] = {}
    for item in train_items:
        for target, weight in graph_scores.get(int(item), {}).items():
            profile_scores[target] = profile_scores.get(target, 0.0) + weight
    return np.array([profile_scores.get(int(item), 0.0) for item in candidate_items], dtype=np.float32)


def profile_people_scores(train_items: Iterable[int], candidate_items: np.ndarray, item_people: dict[int, dict[str, float]]) -> np.ndarray:
    profile: dict[str, float] = {}
    for item in train_items:
        for key, weight in item_people.get(int(item), {}).items():
            profile[key] = profile.get(key, 0.0) + weight
    scores = []
    for item in candidate_items:
        score = 0.0
        for key, weight in item_people.get(int(item), {}).items():
            score += profile.get(key, 0.0) * weight
        scores.append(score)
    return np.array(scores, dtype=np.float32)


def build_item_knn_vectors(item_factors: np.ndarray) -> np.ndarray:
    """Normalize item latent vectors for item-item collaborative scoring."""
    vectors = np.asarray(item_factors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def profile_item_knn_scores(
    train_items: Iterable[int],
    candidate_cols: np.ndarray,
    item_to_col: dict[int, int],
    item_knn_vectors: np.ndarray,
) -> np.ndarray:
    """Score candidates by similarity to the user's liked item neighborhood."""
    seed_cols = [item_to_col[int(item)] for item in train_items if int(item) in item_to_col]
    if not seed_cols:
        return np.zeros(len(candidate_cols), dtype=np.float32)
    profile = item_knn_vectors[np.array(seed_cols, dtype=np.int32)].mean(axis=0)
    norm = np.linalg.norm(profile)
    if norm > 0:
        profile = profile / norm
    return item_knn_vectors[candidate_cols] @ profile


def blend_components(component_scores: dict[str, np.ndarray], weights: dict[str, float]) -> np.ndarray:
    score = np.zeros_like(next(iter(component_scores.values())), dtype=np.float32)
    for name, weight in weights.items():
        if weight <= 0 or name not in component_scores:
            continue
        score += float(weight) * normalized_vector(component_scores[name])
    return score


def compute_component_scores(
    case: EvalCase,
    candidate_cols: np.ndarray,
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
) -> dict[str, np.ndarray]:
    item_tokens, token_idf = metadata_context
    return {
        "latent_svd": user_factors[user_to_row[case.user_id]] @ item_factors[candidate_cols].T,
        "item_knn_collaborative": profile_item_knn_scores(case.train_items, candidate_cols, item_to_col, item_knn_vectors),
        "popularity": np.array([popularity_by_item.get(int(item), 0.0) for item in case.candidate_items], dtype=np.float32),
        "metadata_content": profile_token_scores(case.train_items, case.candidate_items, item_tokens, token_idf),
        "graph_related": profile_graph_scores(case.train_items, case.candidate_items, graph_scores),
        "people_staff_affinity": profile_people_scores(case.train_items, case.candidate_items, people_scores),
    }


def detect_advanced_architecture_inventory() -> pd.DataFrame:
    """Record which advanced recommender families are usable locally.

    The project should not fail just because an optional recommender library is
    absent on the current machine. This inventory makes the comparison explicit:
    sklearn pointwise rerankers always run here, while LightGBM, XGBoost,
    CatBoost, implicit ALS, and Torch models are enabled only when the local
    environment has those optional dependencies installed.
    """
    rows = [
        {
            "architecture": "pointwise_logistic_reranker",
            "family": "learning_to_rank",
            "available": True,
            "dependency": "scikit-learn",
            "role": "linear learned reranker over collaborative/content/graph/people features",
        },
        {
            "architecture": "gbdt_signal_reranker",
            "family": "learning_to_rank",
            "available": True,
            "dependency": "scikit-learn HistGradientBoostingClassifier",
            "role": "nonlinear tree reranker over the same product features",
        },
        {
            "architecture": "neural_mlp_reranker",
            "family": "deep_learning",
            "available": True,
            "dependency": "scikit-learn MLPClassifier",
            "role": "small neural reranker over product features",
        },
        {
            "architecture": "torch_two_tower_cf",
            "family": "deep_learning",
            "available": bool(importlib.util.find_spec("torch")),
            "dependency": "torch",
            "role": "user and anime embedding towers trained from implicit positives plus sampled negatives",
        },
        {
            "architecture": "implicit_als_cf",
            "family": "matrix_factorization",
            "available": bool(importlib.util.find_spec("implicit")),
            "dependency": "implicit",
            "role": "implicit-feedback ALS collaborative factorization trained on the sparse user-anime matrix",
        },
        {
            "architecture": "lightgbm_lambdarank",
            "family": "learning_to_rank",
            "available": bool(importlib.util.find_spec("lightgbm")),
            "dependency": "lightgbm",
            "role": "group-aware LambdaRank reranker over product candidate features",
        },
        {
            "architecture": "xgboost_pairwise_ranker",
            "family": "learning_to_rank",
            "available": bool(importlib.util.find_spec("xgboost")),
            "dependency": "xgboost",
            "role": "group-aware pairwise ranking reranker over product candidate features",
        },
        {
            "architecture": "catboost_yetirank",
            "family": "learning_to_rank",
            "available": bool(importlib.util.find_spec("catboost")),
            "dependency": "catboost",
            "role": "group-aware YetiRank reranker over product candidate features",
        },
    ]
    for package, family, role in [
        ("surprise", "matrix_factorization", "rating-prediction matrix factorization if installed"),
    ]:
        rows.append(
            {
                "architecture": package,
                "family": family,
                "available": bool(importlib.util.find_spec(package)),
                "dependency": package,
                "role": role,
            }
        )
    return pd.DataFrame(rows)


ADVANCED_FEATURE_COLUMNS = [
    "latent_svd",
    "item_knn_collaborative",
    "popularity",
    "metadata_content",
    "graph_related",
    "people_staff_affinity",
    "svd_popularity_agreement",
    "metadata_graph_agreement",
    "people_graph_agreement",
    "full_product_hybrid",
    "tuned_product_hybrid",
    "level_tuned_product_hybrid",
    "profile_size_log",
    "is_beginner",
    "is_casual",
    "is_fan",
    "is_veteran",
]


def build_advanced_model_config_table() -> pd.DataFrame:
    """Document the trainable model families and the features/layers they use."""
    feature_summary = f"{len(ADVANCED_FEATURE_COLUMNS)} candidate features: " + ", ".join(ADVANCED_FEATURE_COLUMNS)
    rows = [
        {
            "model": "pointwise_logistic_reranker",
            "family": "base learned reranker",
            "objective": "balanced pointwise classification of held-out liked item vs sampled negatives",
            "architecture_or_layers": "LogisticRegression(max_iter=500, class_weight=balanced)",
            "input_features": feature_summary,
        },
        {
            "model": "gbdt_signal_reranker",
            "family": "tree learned reranker",
            "objective": "pointwise classification over component scores and profile features",
            "architecture_or_layers": "HistGradientBoostingClassifier(max_iter=260, learning_rate=0.045, max_leaf_nodes=31)",
            "input_features": feature_summary,
        },
        {
            "model": "neural_mlp_reranker",
            "family": "neural learned reranker",
            "objective": "pointwise classification over scaled component features",
            "architecture_or_layers": "MLPClassifier(hidden_layer_sizes=(128,64,32), early_stopping=True, max_iter=80)",
            "input_features": feature_summary,
        },
        {
            "model": "lightgbm_lambdarank",
            "family": "groupwise learning-to-rank",
            "objective": "LambdaRank/NDCG within each user's candidate group",
            "architecture_or_layers": "LGBMRanker(n_estimators=360, learning_rate=0.035, num_leaves=45)",
            "input_features": feature_summary,
        },
        {
            "model": "xgboost_pairwise_ranker",
            "family": "groupwise learning-to-rank",
            "objective": "pairwise ranking within each user's candidate group",
            "architecture_or_layers": "XGBRanker(objective=rank:pairwise, n_estimators=360, max_depth=6)",
            "input_features": feature_summary,
        },
        {
            "model": "catboost_yetirank",
            "family": "groupwise learning-to-rank",
            "objective": "YetiRank candidate ordering",
            "architecture_or_layers": "CatBoostRanker(loss_function=YetiRank, iterations=360, depth=7)",
            "input_features": feature_summary,
        },
        {
            "model": "torch_feature_mlp_reranker",
            "family": "neural learned reranker",
            "objective": "binary relevance over component features",
            "architecture_or_layers": f"Linear({len(ADVANCED_FEATURE_COLUMNS)}->160)->ReLU->Dropout(.10)->Linear(160->80)->ReLU->Dropout(.06)->Linear(80->32)->ReLU->Linear(32->1), epochs={ADVANCED_TORCH_EPOCHS}",
            "input_features": feature_summary,
        },
        {
            "model": "implicit_als_cf",
            "family": "collaborative factorization",
            "objective": "implicit-feedback matrix factorization",
            "architecture_or_layers": "AlternatingLeastSquares(factors=128, regularization=0.08, iterations=16)",
            "input_features": "user-anime positive interaction matrix",
        },
        {
            "model": "torch_two_tower_cf",
            "family": "neural collaborative filtering",
            "objective": "binary relevance with sampled negatives",
            "architecture_or_layers": f"user embedding x item embedding dot product, dim=96, epochs={ADVANCED_TORCH_EPOCHS}",
            "input_features": "user_id|anime_id|positive_or_sampled_negative_label",
        },
    ]
    out = pd.DataFrame(rows)
    out["advanced_train_cases"] = ADVANCED_RERANKER_TRAIN_CASES
    out["advanced_eval_cases"] = ADVANCED_RERANKER_EVAL_CASES
    out["product_row_k"] = PRODUCT_ROW_K
    out["candidate_pool_negatives_per_user"] = NEGATIVES_PER_USER
    out["feature_groups"] = np.where(
        out["input_features"].str.contains("candidate features", na=False),
        "collaborative + popularity + metadata + graph + people/staff + profile-band flags",
        out["input_features"],
    )
    return out


def case_component_feature_frame(
    case: EvalCase,
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    tuned_hybrid_weights: dict[str, float],
    level_tuned_hybrid_weights: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    """Build one candidate-level feature table for learned rerankers."""
    candidate_cols = np.array([item_to_col[int(item)] for item in case.candidate_items], dtype=np.int32)
    raw = compute_component_scores(
        case,
        candidate_cols,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
    )
    normalized = {name: normalized_vector(values) for name, values in raw.items()}
    level = user_level_from_count(len(case.train_items))
    level_weights = (level_tuned_hybrid_weights or {}).get(level) or tuned_hybrid_weights
    frame = pd.DataFrame(
        {
            "userID": case.user_id,
            "candidate_animeID": case.candidate_items.astype(np.int32),
            "holdout_animeID": case.holdout_item,
            "label": [1] + [0] * (len(case.candidate_items) - 1),
            "profile_size": len(case.train_items),
            "user_level": level,
        }
    )
    for name, values in normalized.items():
        frame[name] = values.astype(np.float32)
    frame["svd_popularity_agreement"] = frame["latent_svd"] * frame["popularity"]
    frame["metadata_graph_agreement"] = frame["metadata_content"] * frame["graph_related"]
    frame["people_graph_agreement"] = frame["people_staff_affinity"] * frame["graph_related"]
    frame["full_product_hybrid"] = blend_components(normalized, FULL_HYBRID_WEIGHTS)
    frame["tuned_product_hybrid"] = blend_components(normalized, tuned_hybrid_weights)
    frame["level_tuned_product_hybrid"] = blend_components(normalized, level_weights)
    frame["profile_size_log"] = math.log1p(len(case.train_items))
    frame["is_beginner"] = int(level == "Beginner")
    frame["is_casual"] = int(level == "Casual")
    frame["is_fan"] = int(level == "Fan")
    frame["is_veteran"] = int(level == "Veteran")
    return frame


def build_advanced_reranker_tables(
    cases: list[EvalCase],
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    tuned_hybrid_weights: dict[str, float],
    level_tuned_hybrid_weights: dict[str, dict[str, float]] | None,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for case in cases:
        if case.user_id not in user_to_row:
            continue
        frames.append(
            case_component_feature_frame(
                case,
                item_to_col,
                user_to_row,
                user_factors,
                item_factors,
                popularity_by_item,
                metadata_context,
                graph_scores,
                people_scores,
                item_knn_vectors,
                tuned_hybrid_weights,
                level_tuned_hybrid_weights,
            )
        )
    if not frames:
        return pd.DataFrame(columns=["userID", "candidate_animeID", "label", *ADVANCED_FEATURE_COLUMNS])
    return pd.concat(frames, ignore_index=True)


def eval_rows_from_scored_candidates(scored: pd.DataFrame, method: str) -> pd.DataFrame:
    rows = []
    for user_id, group in scored.groupby("userID", sort=False):
        group = group.reset_index(drop=True)
        scores = group["model_score"].to_numpy(dtype=np.float32)
        labels = group["label"].to_numpy(dtype=np.int8)
        if not labels.any():
            continue
        holdout_pos = int(np.flatnonzero(labels == 1)[0])
        holdout_score = scores[holdout_pos]
        rank = int(np.sum(scores > holdout_score) + np.sum((scores == holdout_score) & (np.arange(len(scores)) < holdout_pos)) + 1)
        top_idx = np.argsort(-scores)[:PRODUCT_ROW_K]
        top1_idx = int(top_idx[0]) if len(top_idx) else 0
        profile_size = int(group.loc[holdout_pos, "profile_size"])
        rows.append(
            {
                "userID": int(user_id),
                "method": method,
                "holdout_animeID": int(group.loc[holdout_pos, "candidate_animeID"]),
                "profile_size": profile_size,
                "user_level": str(group.loc[holdout_pos, "user_level"]),
                "candidate_pool_size": len(group),
                "rank": rank,
                **single_holdout_metric_fields(rank),
                "mrr": float(1.0 / rank),
                "holdout_score": float(holdout_score),
                "top1_animeID": int(group.loc[top1_idx, "candidate_animeID"]),
                "top1_score": float(scores[top1_idx]),
                "top12_anime_ids": "|".join(str(int(group.loc[idx, "candidate_animeID"])) for idx in top_idx),
            }
        )
    return pd.DataFrame(rows)


def train_predict_sklearn_rerankers(train_table: pd.DataFrame, eval_table: pd.DataFrame) -> pd.DataFrame:
    if train_table.empty or eval_table.empty:
        return pd.DataFrame()
    x_train = train_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_table["label"].to_numpy(dtype=np.int8)
    x_eval = eval_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_eval_scaled = scaler.transform(x_eval)
    models = {
        "pointwise_logistic_reranker": LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "gbdt_signal_reranker": HistGradientBoostingClassifier(
            max_iter=260,
            learning_rate=0.045,
            max_leaf_nodes=31,
            l2_regularization=0.02,
            random_state=RANDOM_STATE,
        ),
        "neural_mlp_reranker": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            alpha=0.001,
            batch_size=2048,
            learning_rate_init=0.001,
            max_iter=80,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }
    frames = []
    for method, model in models.items():
        features_for_train = x_train_scaled if method != "gbdt_signal_reranker" else x_train
        features_for_eval = x_eval_scaled if method != "gbdt_signal_reranker" else x_eval
        started_at = time.perf_counter()
        print(f"Training advanced model: {method}", flush=True)
        try:
            model.fit(features_for_train, y_train)
            scored = eval_table.copy()
            if hasattr(model, "predict_proba"):
                scored["model_score"] = model.predict_proba(features_for_eval)[:, 1]
            else:
                scored["model_score"] = model.decision_function(features_for_eval)
            frames.append(eval_rows_from_scored_candidates(scored, method))
            record_training_result(method, "sklearn_pointwise", started_at, len(train_table), len(eval_table))
        except Exception as exc:
            print(f"Advanced reranker skipped: {method} failed with {exc}")
            record_training_result(method, "sklearn_pointwise", started_at, len(train_table), len(eval_table), "skipped", str(exc)[:180])
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def group_sizes(table: pd.DataFrame) -> list[int]:
    return table.groupby("userID", sort=False).size().astype(int).tolist()


def train_predict_external_rankers(train_table: pd.DataFrame, eval_table: pd.DataFrame) -> pd.DataFrame:
    """Train installed group-aware ranking libraries when available."""
    if train_table.empty or eval_table.empty:
        return pd.DataFrame()
    x_train = train_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_table["label"].to_numpy(dtype=np.float32)
    x_eval = eval_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    train_groups = group_sizes(train_table)
    frames: list[pd.DataFrame] = []

    if importlib.util.find_spec("lightgbm"):
        started_at = time.perf_counter()
        method = "lightgbm_lambdarank"
        print(f"Training advanced model: {method}", flush=True)
        try:
            from lightgbm import LGBMRanker

            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                n_estimators=360,
                learning_rate=0.035,
                num_leaves=45,
                min_child_samples=25,
                subsample=0.9,
                colsample_bytree=0.9,
                random_state=RANDOM_STATE,
                verbose=-1,
            )
            model.fit(x_train, y_train, group=train_groups)
            scored = eval_table.copy()
            scored["model_score"] = model.predict(x_eval)
            frames.append(eval_rows_from_scored_candidates(scored, method))
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table))
        except Exception as exc:
            print(f"LightGBM ranker skipped: {exc}", flush=True)
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table), "skipped", str(exc)[:180])

    if importlib.util.find_spec("xgboost"):
        started_at = time.perf_counter()
        method = "xgboost_pairwise_ranker"
        print(f"Training advanced model: {method}", flush=True)
        try:
            from xgboost import XGBRanker

            model = XGBRanker(
                objective="rank:pairwise",
                n_estimators=360,
                learning_rate=0.035,
                max_depth=6,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=RANDOM_STATE,
                tree_method="hist",
            )
            model.fit(x_train, y_train, group=np.array(train_groups, dtype=np.uint32), verbose=False)
            scored = eval_table.copy()
            scored["model_score"] = model.predict(x_eval)
            frames.append(eval_rows_from_scored_candidates(scored, method))
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table))
        except Exception as exc:
            print(f"XGBoost ranker skipped: {exc}", flush=True)
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table), "skipped", str(exc)[:180])

    if importlib.util.find_spec("catboost"):
        started_at = time.perf_counter()
        method = "catboost_yetirank"
        print(f"Training advanced model: {method}", flush=True)
        try:
            from catboost import CatBoostRanker, Pool

            train_pool = Pool(
                data=x_train,
                label=y_train,
                group_id=train_table["userID"].to_numpy(),
            )
            eval_pool = Pool(
                data=x_eval,
                group_id=eval_table["userID"].to_numpy(),
            )
            model = CatBoostRanker(
                loss_function="YetiRank",
                iterations=360,
                learning_rate=0.035,
                depth=7,
                l2_leaf_reg=3.0,
                random_seed=RANDOM_STATE,
                verbose=False,
                allow_writing_files=False,
            )
            model.fit(train_pool)
            scored = eval_table.copy()
            scored["model_score"] = model.predict(eval_pool)
            frames.append(eval_rows_from_scored_candidates(scored, method))
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table))
        except Exception as exc:
            print(f"CatBoost ranker skipped: {exc}", flush=True)
            record_training_result(method, "groupwise_ranker", started_at, len(train_table), len(eval_table), "skipped", str(exc)[:180])

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def train_predict_torch_feature_mlp(train_table: pd.DataFrame, eval_table: pd.DataFrame) -> pd.DataFrame:
    method = "torch_feature_mlp_reranker"
    if not importlib.util.find_spec("torch") or train_table.empty or eval_table.empty:
        return pd.DataFrame()
    started_at = time.perf_counter()
    print(f"Training advanced model: {method}", flush=True)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print(f"Torch feature MLP skipped: {exc}")
        record_training_result(method, "neural_feature_reranker", started_at, len(train_table), len(eval_table), "skipped", str(exc)[:180])
        return pd.DataFrame()

    x_train = train_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    y_train = train_table["label"].to_numpy(dtype=np.float32)
    x_eval = eval_table[ADVANCED_FEATURE_COLUMNS].fillna(0.0).to_numpy(dtype=np.float32)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train).astype(np.float32)
    x_eval = scaler.transform(x_eval).astype(np.float32)

    torch.manual_seed(RANDOM_STATE)
    model = nn.Sequential(
        nn.Linear(x_train.shape[1], 160),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(160, 80),
        nn.ReLU(),
        nn.Dropout(0.06),
        nn.Linear(80, 32),
        nn.ReLU(),
        nn.Linear(32, 1),
    )
    pos_weight = torch.tensor([(len(y_train) - y_train.sum()) / max(y_train.sum(), 1.0)], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0015, weight_decay=0.002)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train).view(-1, 1)),
        batch_size=4096,
        shuffle=True,
    )
    model.train()
    for _ in range(ADVANCED_TORCH_EPOCHS):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(torch.from_numpy(x_eval))).view(-1).numpy()
    scored = eval_table.copy()
    scored["model_score"] = scores
    record_training_result(method, "neural_feature_reranker", started_at, len(train_table), len(eval_table))
    return eval_rows_from_scored_candidates(scored, method)


def train_predict_implicit_als(
    train_matrix: csr_matrix,
    eval_cases: list[EvalCase],
    user_to_row: dict[int, int],
    item_to_col: dict[int, int],
) -> pd.DataFrame:
    method = "implicit_als_cf"
    if not importlib.util.find_spec("implicit") or train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
        return pd.DataFrame()
    started_at = time.perf_counter()
    print(f"Training advanced model: {method}", flush=True)
    try:
        from implicit.als import AlternatingLeastSquares
    except Exception as exc:
        print(f"implicit ALS skipped: {exc}", flush=True)
        record_training_result(method, "collaborative_factorization", started_at, train_matrix.nnz, len(eval_cases), "skipped", str(exc)[:180])
        return pd.DataFrame()
    try:
        model = AlternatingLeastSquares(
            factors=128,
            regularization=0.08,
            iterations=16,
            random_state=RANDOM_STATE,
        )
        # implicit expects item-user confidence. In this orientation,
        # model.user_factors are anime/item factors and model.item_factors
        # are user factors, despite the attribute names.
        confidence = (train_matrix.T * 8.0).tocsr().astype(np.float32)
        try:
            from threadpoolctl import threadpool_limits

            with threadpool_limits(1, "blas"):
                model.fit(confidence, show_progress=False)
        except Exception:
            model.fit(confidence, show_progress=False)
        rows = []
        for case in eval_cases:
            if case.user_id not in user_to_row:
                continue
            candidate_cols = [item_to_col[int(item)] for item in case.candidate_items if int(item) in item_to_col]
            if len(candidate_cols) != len(case.candidate_items):
                continue
            user_vec = model.item_factors[user_to_row[case.user_id]]
            scores = model.user_factors[np.array(candidate_cols, dtype=np.int32)] @ user_vec
            scored = pd.DataFrame(
                {
                    "userID": case.user_id,
                    "candidate_animeID": case.candidate_items.astype(np.int32),
                    "label": [1] + [0] * (len(case.candidate_items) - 1),
                    "profile_size": len(case.train_items),
                    "user_level": user_level_from_count(len(case.train_items)),
                    "model_score": scores,
                }
            )
            rows.append(eval_rows_from_scored_candidates(scored, method))
        if not rows:
            record_training_result(method, "collaborative_factorization", started_at, train_matrix.nnz, len(eval_cases), "skipped", "no scored eval rows")
            return pd.DataFrame()
        record_training_result(method, "collaborative_factorization", started_at, train_matrix.nnz, len(eval_cases))
        return pd.concat(rows, ignore_index=True)
    except Exception as exc:
        print(f"implicit ALS skipped: {exc}", flush=True)
        record_training_result(method, "collaborative_factorization", started_at, train_matrix.nnz, len(eval_cases), "skipped", str(exc)[:180])
        return pd.DataFrame()


def train_predict_torch_two_tower(
    train_matrix: csr_matrix,
    eval_cases: list[EvalCase],
    user_to_row: dict[int, int],
    item_to_col: dict[int, int],
) -> pd.DataFrame:
    method = "torch_two_tower_cf"
    if not importlib.util.find_spec("torch") or train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
        return pd.DataFrame()
    started_at = time.perf_counter()
    print(f"Training advanced model: {method}", flush=True)
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception as exc:
        print(f"Torch two-tower skipped: {exc}")
        record_training_result(method, "neural_collaborative", started_at, train_matrix.nnz, len(eval_cases), "skipped", str(exc)[:180])
        return pd.DataFrame()

    rng = np.random.default_rng(RANDOM_STATE + 909)
    positive_users, positive_items = train_matrix.nonzero()
    if len(positive_users) == 0:
        record_training_result(method, "neural_collaborative", started_at, train_matrix.nnz, len(eval_cases), "skipped", "no positive pairs")
        return pd.DataFrame()
    sample_size = min(ADVANCED_TORCH_TRAIN_PAIRS, len(positive_users))
    sample_idx = rng.choice(len(positive_users), size=sample_size, replace=False)
    pos_users = positive_users[sample_idx].astype(np.int64)
    pos_items = positive_items[sample_idx].astype(np.int64)
    neg_users = pos_users.copy()
    neg_items = rng.integers(0, train_matrix.shape[1], size=sample_size, dtype=np.int64)
    # Resample a few times when a sampled negative is actually positive.
    train_csr = train_matrix.tocsr()
    for _ in range(3):
        collision = np.array(train_csr[neg_users, neg_items]).ravel() > 0
        if not collision.any():
            break
        neg_items[collision] = rng.integers(0, train_matrix.shape[1], size=int(collision.sum()), dtype=np.int64)
    users = np.concatenate([pos_users, neg_users])
    items = np.concatenate([pos_items, neg_items])
    labels = np.concatenate([np.ones(sample_size, dtype=np.float32), np.zeros(sample_size, dtype=np.float32)])
    shuffle = rng.permutation(len(labels))
    users = users[shuffle]
    items = items[shuffle]
    labels = labels[shuffle]

    class TwoTower(nn.Module):
        def __init__(self, n_users: int, n_items: int, dim: int = 96) -> None:
            super().__init__()
            self.user_embedding = nn.Embedding(n_users, dim)
            self.item_embedding = nn.Embedding(n_items, dim)
            self.user_bias = nn.Embedding(n_users, 1)
            self.item_bias = nn.Embedding(n_items, 1)

        def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
            score = (self.user_embedding(user_ids) * self.item_embedding(item_ids)).sum(dim=1, keepdim=True)
            return score + self.user_bias(user_ids) + self.item_bias(item_ids)

    torch.manual_seed(RANDOM_STATE)
    model = TwoTower(train_matrix.shape[0], train_matrix.shape[1])
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.004, weight_decay=0.0005)
    loader = DataLoader(
        TensorDataset(
            torch.from_numpy(users),
            torch.from_numpy(items),
            torch.from_numpy(labels).view(-1, 1),
        ),
        batch_size=8192,
        shuffle=True,
    )
    model.train()
    for _ in range(ADVANCED_TORCH_EPOCHS):
        for batch_user, batch_item, batch_label in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch_user, batch_item), batch_label)
            loss.backward()
            optimizer.step()

    rows = []
    model.eval()
    with torch.no_grad():
        for case in eval_cases:
            if case.user_id not in user_to_row:
                continue
            candidate_cols = [item_to_col[int(item)] for item in case.candidate_items if int(item) in item_to_col]
            if len(candidate_cols) != len(case.candidate_items):
                continue
            user_tensor = torch.full((len(candidate_cols),), user_to_row[case.user_id], dtype=torch.long)
            item_tensor = torch.tensor(candidate_cols, dtype=torch.long)
            scores = torch.sigmoid(model(user_tensor, item_tensor)).view(-1).numpy()
            scored = pd.DataFrame(
                {
                    "userID": case.user_id,
                    "candidate_animeID": case.candidate_items.astype(np.int32),
                    "label": [1] + [0] * (len(case.candidate_items) - 1),
                    "profile_size": len(case.train_items),
                    "user_level": user_level_from_count(len(case.train_items)),
                    "model_score": scores,
                }
            )
            rows.append(eval_rows_from_scored_candidates(scored, method))
    if not rows:
        record_training_result(method, "neural_collaborative", started_at, train_matrix.nnz, len(eval_cases), "skipped", "no scored eval rows")
        return pd.DataFrame()
    record_training_result(method, "neural_collaborative", started_at, train_matrix.nnz, len(eval_cases))
    return pd.concat(rows, ignore_index=True)


def evaluate_advanced_architectures(
    eval_cases: dict[int, EvalCase],
    train_matrix: csr_matrix,
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    tuned_hybrid_weights: dict[str, float],
    level_tuned_hybrid_weights: dict[str, dict[str, float]] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Train/evaluate advanced rerankers on a separate case split.

    Existing methods are evaluated on all held-out cases. Learned rerankers need
    supervision, so this function uses early cases as reranker-training groups
    and later cases as reranker-evaluation groups. This avoids judging the
    rerankers on the exact candidate lists used to train them.
    """
    inventory = detect_advanced_architecture_inventory()
    ordered_cases = list(eval_cases.values())
    if len(ordered_cases) < 200:
        return pd.DataFrame(), pd.DataFrame(), inventory
    train_cases = ordered_cases[: min(ADVANCED_RERANKER_TRAIN_CASES, max(len(ordered_cases) // 3, 1))]
    eval_start = len(train_cases)
    eval_cases_list = ordered_cases[eval_start : eval_start + min(ADVANCED_RERANKER_EVAL_CASES, max(len(ordered_cases) - eval_start, 0))]
    if not eval_cases_list:
        return pd.DataFrame(), pd.DataFrame(), inventory
    train_table = build_advanced_reranker_tables(
        train_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
        level_tuned_hybrid_weights,
    )
    eval_table = build_advanced_reranker_tables(
        eval_cases_list,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
        level_tuned_hybrid_weights,
    )
    eval_frames: list[pd.DataFrame] = []

    # Add the strongest existing score columns on the same advanced eval split,
    # so the learned models are compared against the hand-tuned product systems
    # on exactly the same users and candidate pools.
    for column, method in [
        ("popularity", "advanced_split_popularity_baseline"),
        ("latent_svd", "advanced_split_latent_svd"),
        ("item_knn_collaborative", "advanced_split_item_knn"),
        ("level_tuned_product_hybrid", "advanced_split_level_tuned_hybrid"),
    ]:
        scored = eval_table.copy()
        scored["model_score"] = scored[column].astype(np.float32)
        eval_frames.append(eval_rows_from_scored_candidates(scored, method))

    eval_frames.append(train_predict_sklearn_rerankers(train_table, eval_table))
    eval_frames.append(train_predict_external_rankers(train_table, eval_table))
    eval_frames.append(train_predict_implicit_als(train_matrix, eval_cases_list, user_to_row, item_to_col))
    eval_frames.append(train_predict_torch_feature_mlp(train_table, eval_table))
    eval_frames.append(train_predict_torch_two_tower(train_matrix, eval_cases_list, user_to_row, item_to_col))

    eval_rows = pd.concat([frame for frame in eval_frames if frame is not None and not frame.empty], ignore_index=True)
    if eval_rows.empty:
        return eval_rows, pd.DataFrame(), inventory
    metrics = summarize_metrics(eval_rows, train_matrix, np.array(list(item_to_col.keys()), dtype=np.int32))
    metrics["advanced_train_cases"] = len(train_cases)
    metrics["advanced_eval_cases"] = len(eval_cases_list)
    metrics["feature_count"] = len(ADVANCED_FEATURE_COLUMNS)
    return eval_rows, metrics, inventory


def summarize_discovery_metrics(
    eval_rows: pd.DataFrame,
    catalog: pd.DataFrame,
    popularity_by_item: dict[int, float],
    candidate_items: np.ndarray,
) -> pd.DataFrame:
    catalog_genres = catalog.set_index("mal_id")["genres"].fillna("").map(genre_set).to_dict()
    rows = []
    candidate_count = max(len(candidate_items), 1)

    for method, group in eval_rows.groupby("method"):
        def discovery_for_column(column: str) -> dict[str, float]:
            all_top_items: list[int] = []
            novelty_values: list[float] = []
            unique_genre_counts: list[int] = []
            diversity_ratios: list[float] = []

            for value in group[column]:
                top_ids = split_pipe_ids(value)
                all_top_items.extend(top_ids)
                novelty_values.extend(1.0 - popularity_by_item.get(item, 0.0) for item in top_ids)

                genres: list[str] = []
                for item in top_ids:
                    genres.extend(sorted(catalog_genres.get(item, set())))
                unique_genres = set(genres)
                unique_genre_counts.append(len(unique_genres))
                diversity_ratios.append(len(unique_genres) / max(len(genres), 1))

            return {
                "coverage": len(set(all_top_items)) / candidate_count,
                "unique_recommended": len(set(all_top_items)),
                "mean_novelty": float(np.mean(novelty_values)) if novelty_values else np.nan,
                "mean_unique_genres": float(np.mean(unique_genre_counts)) if unique_genre_counts else np.nan,
                "mean_genre_diversity_ratio": float(np.mean(diversity_ratios)) if diversity_ratios else np.nan,
            }

        top12 = discovery_for_column("top12_anime_ids")

        rows.append(
            {
                "method": method,
                "coverage_at_12": top12["coverage"],
                "unique_recommended_at_12": top12["unique_recommended"],
                "mean_novelty_at_12": top12["mean_novelty"],
                "mean_unique_genres_at_12": top12["mean_unique_genres"],
                "mean_genre_diversity_ratio_at_12": top12["mean_genre_diversity_ratio"],
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics_by_level(eval_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        eval_rows.groupby(["user_level", "method"])
        .agg(
            evaluated_users=("userID", "nunique"),
            median_profile_size=("profile_size", "median"),
            hit_rate_at_12=("hit_at_12", "mean"),
            recall_at_12=("recall_at_12", "mean"),
            ndcg_at_12=("ndcg_at_12", "mean"),
            map_at_12=("map_at_12", "mean"),
            mean_reciprocal_rank=("mrr", "mean"),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    level_order = {level["level"]: idx for idx, level in enumerate(USER_LEVELS)}
    method_order = {
        "level_tuned_product_hybrid": 0,
        "tuned_product_hybrid": 1,
        "full_product_hybrid": 2,
        "latent_svd": 3,
        "item_knn_collaborative": 4,
        "metadata_content": 5,
        "graph_related": 6,
        "people_staff_affinity": 7,
        "popularity_baseline": 8,
    }
    metrics["_level_order"] = metrics["user_level"].map(level_order).fillna(999)
    metrics["_method_order"] = metrics["method"].map(method_order).fillna(999)
    return metrics.sort_values(["_level_order", "_method_order"]).drop(columns=["_level_order", "_method_order"])


def add_balanced_profile_metrics(metrics: pd.DataFrame, level_metrics: pd.DataFrame) -> pd.DataFrame:
    balanced = (
        level_metrics.groupby("method")
        .agg(
            balanced_profile_hit_at_12=("hit_rate_at_12", "mean"),
            balanced_profile_ndcg_at_12=("ndcg_at_12", "mean"),
            balanced_profile_map_at_12=("map_at_12", "mean"),
            balanced_profile_mrr=("mean_reciprocal_rank", "mean"),
        )
        .reset_index()
    )
    return metrics.merge(balanced, on="method", how="left")


def add_titles(examples: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    titles = catalog.set_index("mal_id")["title"].to_dict()

    def names(ids: str, limit: int = 10) -> str:
        out = []
        for raw in str(ids).split("|"):
            if not raw:
                continue
            anime_id = int(raw)
            out.append(str(titles.get(anime_id, anime_id)))
            if len(out) >= limit:
                break
        return " | ".join(out)

    examples = examples.copy()
    examples["holdout_title"] = examples["holdout_animeID"].map(titles)
    examples["train_titles"] = examples["train_anime_ids"].apply(lambda value: names(value, limit=8))
    top12_col = "product_hybrid_top12_anime_ids" if "product_hybrid_top12_anime_ids" in examples.columns else "full_hybrid_top12_anime_ids"
    examples["product_hybrid_top12_titles"] = examples[top12_col].apply(lambda value: names(value, limit=PRODUCT_ROW_K))
    primary_rank = examples.get("level_tuned_hybrid_rank", examples["tuned_hybrid_rank"]).fillna(
        examples["tuned_hybrid_rank"].fillna(examples["full_hybrid_rank"])
    )
    examples["case_type"] = np.select(
        [
            (primary_rank <= PRODUCT_ROW_K) & (examples["popularity_rank"] > PRODUCT_ROW_K),
            (primary_rank > PRODUCT_ROW_K) & (examples["popularity_rank"] <= PRODUCT_ROW_K),
            (primary_rank <= PRODUCT_ROW_K) & (examples["popularity_rank"] <= PRODUCT_ROW_K),
        ],
        ["hybrid_win", "hybrid_failure_vs_popularity", "both_hit"],
        default="both_miss",
    )
    return examples


def summarize_candidate_sensitivity(
    base_eval_cases: dict[int, EvalCase],
    candidate_items: np.ndarray,
    item_to_col: dict[int, int],
    user_to_row: dict[int, int],
    user_factors: np.ndarray,
    item_factors: np.ndarray,
    popularity_scaled: np.ndarray,
    popularity_by_item: dict[int, float],
    metadata_context: tuple[dict[int, dict[str, float]], dict[str, float]],
    graph_scores: dict[int, dict[int, float]],
    people_scores: dict[int, dict[str, float]],
    item_knn_vectors: np.ndarray,
    tuned_hybrid_weights: dict[str, float],
    level_tuned_hybrid_weights: dict[str, dict[str, float]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate whether the result survives harder candidate pools."""
    summary_frames: list[pd.DataFrame] = []

    for negatives in NEGATIVE_SENSITIVITY_SIZES:
        sensitivity_cases = rebuild_candidate_sets(
            base_eval_cases,
            candidate_items,
            negatives_per_user=negatives,
            max_users=MAX_SENSITIVITY_USERS,
            random_state=RANDOM_STATE + negatives,
        )
        eval_rows, _ = evaluate_rankings(
            sensitivity_cases,
            item_to_col,
            user_to_row,
            user_factors,
            item_factors,
            popularity_scaled,
            popularity_by_item,
            metadata_context,
            graph_scores,
            people_scores,
            item_knn_vectors,
            tuned_hybrid_weights,
            level_tuned_hybrid_weights,
        )
        metrics = summarize_metrics(eval_rows, csr_matrix((0, 0)), candidate_items)
        metrics["candidate_pool_definition"] = f"1 positive + {negatives} sampled negatives"
        summary_frames.append(metrics)

    full_catalog_cases = rebuild_candidate_sets(
        base_eval_cases,
        candidate_items,
        negatives_per_user=None,
        max_users=FULL_CATALOG_EVAL_USERS,
        random_state=RANDOM_STATE + 9_999,
    )
    full_catalog_rows, _ = evaluate_rankings(
        full_catalog_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_scaled,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
        level_tuned_hybrid_weights,
    )
    full_metrics = summarize_metrics(full_catalog_rows, csr_matrix((0, 0)), candidate_items)
    full_metrics["candidate_pool_definition"] = "held-out positive ranked against all eligible catalog candidates"
    summary_frames.append(full_metrics)

    return pd.concat(summary_frames, ignore_index=True), full_catalog_rows


def build_alignment_contract(
    catalog: pd.DataFrame,
    positives: pd.DataFrame,
    candidate_items: np.ndarray,
    model_users: np.ndarray,
    eval_cases: dict[int, EvalCase],
    train: pd.DataFrame,
    eval_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Write explicit data-contract checks for the recommender pipeline."""
    catalog_ids = set(catalog["mal_id"].astype(int))
    candidate_set = set(int(item) for item in candidate_items)
    holdout_pairs = {(case.user_id, case.holdout_item) for case in eval_cases.values()}
    train_pairs = set(zip(train["userID"].astype(int), train["animeID"].astype(int)))
    eval_candidate_ids: set[int] = set()
    for value in eval_rows["top12_anime_ids"].dropna():
        eval_candidate_ids.update(split_pipe_ids(value))

    checks = [
        {
            "check": "catalog_ids_unique",
            "value": int(catalog["mal_id"].nunique() == len(catalog)),
            "status": "pass" if catalog["mal_id"].nunique() == len(catalog) else "fail",
            "detail": "Each anime row must map to one MAL id.",
        },
        {
            "check": "positive_interactions_aligned_to_catalog",
            "value": int(positives["animeID"].isin(catalog_ids).sum()),
            "status": "pass" if positives["animeID"].isin(catalog_ids).all() else "fail",
            "detail": "Every retained positive rating must be recommendable from the catalog.",
        },
        {
            "check": "candidate_pool_subset_of_catalog",
            "value": int(len(candidate_set - catalog_ids)),
            "status": "pass" if not (candidate_set - catalog_ids) else "fail",
            "detail": "Candidate ids are restricted to catalog ids.",
        },
        {
            "check": "model_matrix_shape",
            "value": f"{len(model_users)} users x {len(candidate_items)} anime",
            "status": "pass",
            "detail": "Rows are model users; columns are eligible anime candidate ids.",
        },
        {
            "check": "holdout_removed_from_train",
            "value": int(len(holdout_pairs & train_pairs)),
            "status": "pass" if not (holdout_pairs & train_pairs) else "fail",
            "detail": "Held-out liked anime must not be present in the training matrix.",
        },
        {
            "check": "evaluation_top12_targets_in_catalog",
            "value": int(len(eval_candidate_ids - catalog_ids)),
            "status": "pass" if not (eval_candidate_ids - catalog_ids) else "fail",
            "detail": "All emitted recommendation ids must resolve back to catalog rows.",
        },
        {
            "check": "evaluation_users_have_train_profile",
            "value": int(sum(len(case.train_items) >= MIN_USER_POSITIVES - 1 for case in eval_cases.values())),
            "status": "pass",
            "detail": "Every evaluated user has a non-empty profile after one liked title is held out.",
        },
    ]
    return pd.DataFrame(checks)


def build_error_case_table(
    eval_rows: pd.DataFrame,
    catalog: pd.DataFrame,
    limit: int = 240,
    primary_method: str | None = None,
    baseline_method: str = "popularity_baseline",
) -> pd.DataFrame:
    """Create concrete strong/failure cases with held-out titles and ranks."""
    if eval_rows.empty:
        return pd.DataFrame()
    title_lookup = catalog.set_index("mal_id")["title"].to_dict()
    score_lookup = catalog.set_index("mal_id")["score"].to_dict() if "score" in catalog.columns else {}

    rank_pivot = eval_rows.pivot_table(index="userID", columns="method", values="rank", aggfunc="first")
    score_pivot = eval_rows.pivot_table(index="userID", columns="method", values="holdout_score", aggfunc="first")
    methods = set(eval_rows["method"])
    if primary_method is None:
        for candidate in ["level_tuned_product_hybrid", "tuned_product_hybrid", "full_product_hybrid"]:
            if candidate in methods:
                primary_method = candidate
                break
    if primary_method is None or primary_method not in methods:
        primary_method = str(eval_rows["method"].iloc[0])
    if baseline_method not in methods:
        baseline_method = "popularity_baseline" if "popularity_baseline" in methods else primary_method
    primary_rank_col = f"{primary_method}_rank"
    baseline_rank_col = f"{baseline_method}_rank"
    base = eval_rows[eval_rows["method"].eq(primary_method)].copy()
    if base.empty:
        return pd.DataFrame()
    base = base.merge(rank_pivot.add_suffix("_rank").reset_index(), on="userID", how="left")
    base = base.merge(score_pivot.add_suffix("_holdout_model_score").reset_index(), on="userID", how="left")
    base["heldout_title"] = base["holdout_animeID"].map(title_lookup)
    base["heldout_catalog_score"] = base["holdout_animeID"].map(score_lookup)
    base["top1_title"] = base["top1_animeID"].map(title_lookup)
    base["top12_titles"] = base["top12_anime_ids"].map(
        lambda value: " | ".join(title_lookup.get(item, str(item)) for item in split_pipe_ids(value)[:PRODUCT_ROW_K])
    )

    conditions = [
        (base[primary_rank_col] <= PRODUCT_ROW_K) & (base[baseline_rank_col] > PRODUCT_ROW_K),
        (base[primary_rank_col] > PRODUCT_ROW_K) & (base[baseline_rank_col] <= PRODUCT_ROW_K),
        (base[primary_rank_col] > 50),
        (base[primary_rank_col] <= 3),
    ]
    labels = [
        "strong_personalization_case",
        "hybrid_failure_vs_popularity",
        "hard_failure_rank_over_50",
        "strong_top3_recovery",
    ]
    base["case_type"] = np.select(conditions, labels, default="ordinary_case")
    base["interpretation_hint"] = np.select(
        conditions,
        [
            "Profile-level tuned product hybrid recovered a liked title that popularity did not place in the visible 12-item row.",
            "Popularity found the holdout but the profile-level tuned product hybrid missed the visible row.",
            "The held-out liked title was buried; this is a systematic failure candidate.",
            "The held-out liked title was recovered almost immediately.",
        ],
        default="Included for rank distribution context.",
    )

    selected = []
    per_case = max(limit // len(labels), 1)
    for case_type in labels:
        chunk = base[base["case_type"].eq(case_type)].sort_values(primary_rank_col).head(per_case)
        selected.append(chunk)
    out = pd.concat(selected, ignore_index=True) if selected else base.head(limit)
    cols = [
        "case_type",
        "interpretation_hint",
        "userID",
        "user_level",
        "profile_size",
        "holdout_animeID",
        "heldout_title",
        "heldout_catalog_score",
        baseline_rank_col,
        "metadata_content_rank",
        "graph_related_rank",
        "people_staff_affinity_rank",
        "latent_svd_rank",
        "full_product_hybrid_rank",
        "tuned_product_hybrid_rank",
        "level_tuned_product_hybrid_rank",
        f"{primary_method}_rank",
        "metadata_content_holdout_model_score",
        "graph_related_holdout_model_score",
        "people_staff_affinity_holdout_model_score",
        "latent_svd_holdout_model_score",
        "full_product_hybrid_holdout_model_score",
        "tuned_product_hybrid_holdout_model_score",
        "level_tuned_product_hybrid_holdout_model_score",
        f"{primary_method}_holdout_model_score",
        "top1_animeID",
        "top1_title",
        "top12_titles",
        "candidate_pool_size",
    ]
    return out[[col for col in cols if col in out.columns]].head(limit)


def build_beginner_entrypoint_candidates(catalog: pd.DataFrame, limit: int = 150) -> pd.DataFrame:
    work = catalog.copy()
    for col in ["score", "members", "episodes", "duration", "total_watch_minutes", "aired_year"]:
        if col not in work.columns:
            work[col] = np.nan
        work[col] = pd.to_numeric(work[col], errors="coerce")
    for col in ["genres", "tags", "demographics", "rating", "type"]:
        if col not in work.columns:
            work[col] = ""
        work[col] = work[col].fillna("").astype(str)
    if "relations" not in work.columns:
        work["relations"] = ""
    work["relations"] = work["relations"].fillna("").astype(str)

    explicit_mask = (
        work["genres"].str.contains(r"\b(?:Hentai|Erotica)\b", case=False, na=False)
        | work["rating"].str.contains("Rx", case=False, na=False)
        | work["demographics"].str.contains("18\\+", case=False, na=False)
    )
    entry_type_mask = work["type"].isin(["TV", "Movie", "ONA", "OVA"])
    continuation_mask = work["relations"].str.contains(
        r"(?:^|\|)(?:Prequel|Parent Story|Full Story):",
        case=False,
        na=False,
    )
    viable_runtime = work["episodes"].fillna(1).between(1, 75) | work["type"].eq("Movie")
    viable_score = work["score"].fillna(0) >= 7.0
    viable_members = work["members"].fillna(0) >= 50_000
    candidates = work[
        ~explicit_mask & ~continuation_mask & entry_type_mask & viable_runtime & viable_score & viable_members
    ].copy()

    score_norm = (candidates["score"] - candidates["score"].min()) / max(
        candidates["score"].max() - candidates["score"].min(),
        1e-9,
    )
    members_log = np.log1p(candidates["members"].fillna(0))
    members_norm = (members_log - members_log.min()) / max(members_log.max() - members_log.min(), 1e-9)
    long_penalty = np.clip((candidates["episodes"].fillna(1) - 26) / 74, 0, 1)
    recent_bonus = np.clip((candidates["aired_year"].fillna(2000) - 2000) / 30, 0, 1)
    candidates["beginner_entry_score"] = 0.42 * score_norm + 0.38 * members_norm + 0.10 * recent_bonus - 0.10 * long_penalty
    candidates["why_entry_level"] = np.where(
        candidates["type"].eq("Movie"),
        "high-score popular movie; low commitment",
        "high-score popular short/medium-length series",
    )
    cols = [
        "mal_id",
        "title",
        "type",
        "score",
        "members",
        "episodes",
        "duration",
        "total_watch_minutes",
        "aired_year",
        "genres",
        "demographics",
        "rating",
        "beginner_entry_score",
        "why_entry_level",
    ]
    return candidates.sort_values("beginner_entry_score", ascending=False)[cols].head(limit)


def parse_mylist_xml(path: Path) -> tuple[pd.DataFrame, dict]:
    if not path.exists():
        return pd.DataFrame(columns=["mal_id", "title", "status", "score"]), {
            "available": False,
            "path": str(path),
            "reason": "MyList.xml not found",
        }

    root = ET.parse(path).getroot()
    myinfo = root.find("myinfo")
    info = {"available": True, "path": str(path)}
    if myinfo is not None:
        for tag in [
            "user_total_anime",
            "user_total_completed",
            "user_total_watching",
            "user_total_onhold",
            "user_total_dropped",
            "user_total_plantowatch",
        ]:
            node = myinfo.find(tag)
            if node is not None and node.text:
                try:
                    info[tag] = int(node.text)
                except ValueError:
                    info[tag] = node.text

    rows = []
    for node in root.findall("anime"):
        anime_id_node = node.find("series_animedb_id")
        title_node = node.find("series_title")
        status_node = node.find("my_status")
        score_node = node.find("my_score")
        if anime_id_node is None or not anime_id_node.text:
            continue
        try:
            anime_id = int(anime_id_node.text)
        except ValueError:
            continue
        try:
            score = int(score_node.text) if score_node is not None and score_node.text else 0
        except ValueError:
            score = 0
        rows.append(
            {
                "mal_id": anime_id,
                "title": title_node.text if title_node is not None else "",
                "status": status_node.text if status_node is not None else "",
                "score": score,
            }
        )
    return pd.DataFrame(rows), info


def build_mylist_recommendations(
    catalog: pd.DataFrame,
    candidate_items: np.ndarray,
    item_to_col: dict[int, int],
    item_factors: np.ndarray,
    popularity_by_item: dict[int, float],
    limit: int = 200,
) -> tuple[pd.DataFrame, dict]:
    """Build the qualitative personal recommendation demo from MyList.xml.

    This is not part of the large offline evaluation. It shows how the same
    latent item space can score a real profile:
    - take scored, non-plan-to-watch anime with score >= LIKE_THRESHOLD;
    - average their item factors into one profile vector;
    - rank unseen catalog anime by hybrid latent + popularity score.

    The first version deliberately exposes failure modes, such as recommending
    sequels without prerequisites or side content. The guarded version below
    demonstrates how graph/product rules can repair those issues.
    """
    mylist, info = parse_mylist_xml(MY_LIST_PATH)
    if mylist.empty:
        return pd.DataFrame(), info

    candidate_set = set(int(item) for item in candidate_items)
    known_ids = set(mylist["mal_id"].astype(int))
    scored_seed = mylist[
        ~mylist["status"].eq("Plan to Watch")
        & (mylist["score"] >= LIKE_THRESHOLD)
        & (mylist["mal_id"].isin(candidate_set))
    ].copy()
    seed_ids = sorted(set(scored_seed["mal_id"].astype(int)).intersection(item_to_col.keys()))
    completed_count = int(info.get("user_total_completed", len(mylist[mylist["status"].eq("Completed")])))

    info.update(
        {
            "rows_in_mylist_xml": int(len(mylist)),
            "known_or_planned_ids_excluded": int(len(known_ids)),
            "liked_scored_catalog_seed_count": int(len(seed_ids)),
            "completed_count_for_level": completed_count,
            "level": user_level_from_count(completed_count),
            "like_threshold": LIKE_THRESHOLD,
        }
    )

    if not seed_ids:
        info["reason"] = "No liked scored seed ids overlap the candidate pool"
        return pd.DataFrame(), info

    seed_cols = np.array([item_to_col[item] for item in seed_ids], dtype=np.int32)
    profile_vector = item_factors[seed_cols].mean(axis=0)
    candidate_cols = np.array([item_to_col[int(item)] for item in candidate_items], dtype=np.int32)
    latent_scores = profile_vector @ item_factors[candidate_cols].T
    pop_scores = np.array([popularity_by_item.get(int(item), 0.0) for item in candidate_items], dtype=np.float32)

    latent_norm = (latent_scores - latent_scores.min()) / max(latent_scores.max() - latent_scores.min(), 1e-9)
    pop_norm = (pop_scores - pop_scores.min()) / max(pop_scores.max() - pop_scores.min(), 1e-9)
    demo_latent_weight = 0.5
    hybrid_scores = demo_latent_weight * latent_norm + (1 - demo_latent_weight) * pop_norm

    rows = pd.DataFrame(
        {
            "mal_id": candidate_items.astype(int),
            "mylist_latent_score": latent_scores,
            "mylist_hybrid_score": hybrid_scores,
            "popularity_score": pop_scores,
        }
    )
    rows = rows[~rows["mal_id"].isin(known_ids)].copy()
    out = rows.merge(catalog, on="mal_id", how="left")

    cols = [
        "mal_id",
        "title",
        "type",
        "score",
        "members",
        "episodes",
        "duration",
        "total_watch_minutes",
        "aired_year",
        "genres",
        "demographics",
        "rating",
        "relations",
        "mylist_hybrid_score",
        "mylist_latent_score",
        "popularity_score",
    ]
    cols = [col for col in cols if col in out.columns]
    return out.sort_values("mylist_hybrid_score", ascending=False)[cols].head(limit), info


def relation_pairs(value: object) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    if pd.isna(value):
        return pairs
    for part in str(value).split("|"):
        if ":" not in part:
            continue
        label, raw_id = part.split(":", 1)
        try:
            pairs.append((label.strip(), int(float(raw_id))))
        except ValueError:
            continue
    return pairs


def build_mylist_guarded_recommendations(
    mylist_recommendations: pd.DataFrame,
    catalog: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Apply product guardrails to the personal demo recommendations.

    The collaborative model only knows taste similarity. It does not inherently
    know watch order, recap movies, prerequisite seasons, or disliked franchise
    branches. This guardrail layer uses catalog relations and MyList scores to
    block recommendations that are technically similar but operationally bad:
    - sequels/prequels/full-story entries whose prerequisites are missing;
    - entries related to titles the user scored below LIKE_THRESHOLD;
    - summaries or side content when the full/main entry is more appropriate.
    """
    mylist, _ = parse_mylist_xml(MY_LIST_PATH)
    if mylist_recommendations.empty or mylist.empty:
        return pd.DataFrame(), pd.DataFrame(), {"available": False, "reason": "missing MyList recommendations or XML"}

    mylist = mylist.copy()
    mylist["score"] = pd.to_numeric(mylist["score"], errors="coerce").fillna(0).astype(int)
    known_ids = set(mylist["mal_id"].astype(int))
    score_by_id = dict(zip(mylist["mal_id"].astype(int), mylist["score"]))
    status_by_id = dict(zip(mylist["mal_id"].astype(int), mylist["status"].fillna("").astype(str)))
    liked_ids = {
        int(row.mal_id)
        for row in mylist.itertuples()
        if int(row.score) >= LIKE_THRESHOLD and str(row.status) != "Plan to Watch"
    }
    low_score_ids = {
        int(row.mal_id)
        for row in mylist.itertuples()
        if 0 < int(row.score) < LIKE_THRESHOLD and str(row.status) != "Plan to Watch"
    }
    planned_ids = {
        int(row.mal_id)
        for row in mylist.itertuples()
        if str(row.status) == "Plan to Watch"
    }

    catalog_relations = catalog.set_index("mal_id")["relations"].fillna("").to_dict()
    summary_of_known: dict[int, list[int]] = {}
    for known_id in known_ids:
        for label, related_id in relation_pairs(catalog_relations.get(known_id, "")):
            if label == "Summary":
                summary_of_known.setdefault(related_id, []).append(known_id)

    rows = []
    for rank, row in enumerate(mylist_recommendations.itertuples(index=False), start=1):
        anime_id = int(row.mal_id)
        relations = relation_pairs(getattr(row, "relations", ""))
        prerequisite_ids = [related_id for label, related_id in relations if label in PREREQUISITE_RELATIONS]
        reasons = []

        missing = [related_id for related_id in prerequisite_ids if related_id not in known_ids]
        planned = [related_id for related_id in prerequisite_ids if related_id in planned_ids]
        disliked = [related_id for related_id in prerequisite_ids if related_id in low_score_ids]
        liked_prereqs = [related_id for related_id in prerequisite_ids if related_id in liked_ids]

        if anime_id in summary_of_known:
            reasons.append(
                "summary/recap of known title "
                + ",".join(str(known_id) for known_id in summary_of_known[anime_id][:3])
            )
        if missing:
            reasons.append("missing prerequisite " + ",".join(str(item) for item in missing[:3]))
        if planned:
            reasons.append("prerequisite is only plan-to-watch " + ",".join(str(item) for item in planned[:3]))
        if disliked:
            reasons.append(
                "low-rated prerequisite "
                + ",".join(f"{item}:{score_by_id.get(item)}" for item in disliked[:3])
            )

        item_type = str(getattr(row, "type", ""))
        if item_type in SIDE_CONTENT_TYPES and not liked_prereqs:
            reasons.append("side/special content without liked parent")

        action = "kept" if not reasons else "blocked"
        rows.append(
            {
                "raw_rank": rank,
                "mal_id": anime_id,
                "title": getattr(row, "title", ""),
                "type": item_type,
                "score": getattr(row, "score", np.nan),
                "mylist_hybrid_score": getattr(row, "mylist_hybrid_score", np.nan),
                "guardrail_action": action,
                "guardrail_reason": " | ".join(reasons),
                "known_status": status_by_id.get(anime_id, ""),
                "known_score": score_by_id.get(anime_id, ""),
            }
        )

    comparison = pd.DataFrame(rows)
    guarded_ids = set(comparison.loc[comparison["guardrail_action"].eq("kept"), "mal_id"].astype(int))
    guarded = mylist_recommendations[mylist_recommendations["mal_id"].astype(int).isin(guarded_ids)].copy()
    guarded["guarded_rank"] = range(1, len(guarded) + 1)
    cols = ["guarded_rank"] + [col for col in guarded.columns if col != "guarded_rank"]
    guarded = guarded[cols]
    summary = {
        "available": True,
        "raw_candidates_reviewed": int(len(comparison)),
        "blocked_candidates": int(comparison["guardrail_action"].eq("blocked").sum()),
        "kept_candidates": int(comparison["guardrail_action"].eq("kept").sum()),
        "guardrails": [
            "block summaries/recaps of known titles",
            "block candidates with missing prerequisites",
            "block candidates whose prerequisite is only plan-to-watch",
            "block candidates whose prerequisite was rated below the like threshold",
            "block side/special content unless the parent prerequisite is liked",
        ],
    }
    return guarded, comparison, summary


def summarize_guardrail_blocks(comparison: pd.DataFrame) -> pd.DataFrame:
    categories = [
        ("summary_or_recap", "summary/recap"),
        ("missing_prerequisite", "missing prerequisite"),
        ("plan_to_watch_prerequisite", "prerequisite is only plan-to-watch"),
        ("low_rated_prerequisite", "low-rated prerequisite"),
        ("side_content_without_liked_parent", "side/special content"),
    ]
    rows = []
    blocked = comparison[comparison["guardrail_action"].eq("blocked")].copy()
    for category, needle in categories:
        mask = blocked["guardrail_reason"].fillna("").str.contains(needle, case=False, regex=False)
        rows.append(
            {
                "block_category": category,
                "blocked_count": int(mask.sum()),
                "blocked_pct_of_reviewed": float(mask.mean()) if len(comparison) else 0.0,
                "example_titles": " | ".join(blocked.loc[mask, "title"].head(5).astype(str)),
            }
        )
    rows.append(
        {
            "block_category": "any_block",
            "blocked_count": int(len(blocked)),
            "blocked_pct_of_reviewed": float(len(blocked) / len(comparison)) if len(comparison) else 0.0,
            "example_titles": " | ".join(blocked["title"].head(5).astype(str)),
        }
    )
    return pd.DataFrame(rows)


def plot_metrics(metrics: pd.DataFrame) -> None:
    order = metrics.sort_values("hit_rate_at_12", ascending=False)["method"].tolist()
    plot_df = metrics.set_index("method").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    bar_color = ["#2CA02C", "#59A14F", "#F28E2B", "#4C78A8", "#1F77B4", "#9C755F", "#76B7B2", "#EDC948", "#B07AA1"]
    for ax, metric, title in [
        (axes[0], "hit_rate_at_12", "Hit Rate@12"),
        (axes[1], "ndcg_at_12", "NDCG@12"),
        (axes[2], "mean_reciprocal_rank", "MRR"),
    ]:
        bars = ax.bar(plot_df["method"], plot_df[metric], color=bar_color[: len(plot_df)])
        ax.set_title(title)
        ax.set_ylim(0, max(0.01, plot_df[metric].max() * 1.2))
        ax.tick_params(axis="x", rotation=35, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.bar(plot_df["method"], plot_df["median_rank"], color=bar_color[: len(plot_df)])
    ax.set_title("Median Holdout Rank Lower Is Better")
    ax.set_ylabel("Median rank among 101 candidates")
    ax.tick_params(axis="x", rotation=35, labelsize=8)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(plot_df["median_rank"]):
        ax.text(idx, value, f"{value:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "median_rank.png", dpi=180)
    plt.close(fig)


def plot_level_metrics(eval_rows: pd.DataFrame, level_metrics: pd.DataFrame) -> None:
    profile_sizes = eval_rows.drop_duplicates("userID")[["userID", "profile_size", "user_level"]]
    level_order = [level["level"] for level in USER_LEVELS]
    counts = profile_sizes["user_level"].value_counts().reindex(level_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(counts.index, counts.values, color="#4C78A8")
    ax.set_title("Evaluation Users by Recommendation Profile Band")
    ax.set_ylabel("Users")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{int(value):,}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "user_level_distribution.png", dpi=180)
    plt.close(fig)

    plotted_method = "level_tuned_product_hybrid" if "level_tuned_product_hybrid" in set(level_metrics["method"]) else "tuned_product_hybrid"
    hybrid = level_metrics[level_metrics["method"].eq(plotted_method)].copy()
    hybrid["user_level"] = pd.Categorical(hybrid["user_level"], categories=level_order, ordered=True)
    hybrid = hybrid.sort_values("user_level")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(hybrid["user_level"].astype(str), hybrid["hit_rate_at_12"], color="#59A14F")
    ax.set_title(f"{plotted_method} Hit@12 by Recommendation Profile Band")
    ax.set_ylabel("Hit@12")
    ax.set_ylim(0, max(0.05, hybrid["hit_rate_at_12"].max() * 1.2))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "hit_at_12_by_user_level.png", dpi=180)
    plt.close(fig)

    method_order = [
        "level_tuned_product_hybrid",
        "tuned_product_hybrid",
        "full_product_hybrid",
        "latent_svd",
        "item_knn_collaborative",
        "popularity_baseline",
        "metadata_content",
        "graph_related",
        "people_staff_affinity",
    ]
    pivot = level_metrics.pivot_table(index="user_level", columns="method", values="hit_rate_at_12", aggfunc="first")
    pivot = pivot.reindex(level_order)[[method for method in method_order if method in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(pivot.index))
    width = min(0.12, 0.8 / max(len(pivot.columns), 1))
    colors = ["#006D2C", "#2CA02C", "#59A14F", "#F28E2B", "#4C78A8", "#1F77B4", "#9C755F", "#76B7B2", "#EDC948", "#B07AA1"]
    for idx, method in enumerate(pivot.columns):
        offsets = x + (idx - (len(pivot.columns) - 1) / 2) * width
        bars = ax.bar(offsets, pivot[method].values, width=width, label=method, color=colors[idx % len(colors)])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Hit@12 by User Level and Ranking Method")
    ax.set_ylabel("Hit@12")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_ylim(0, max(0.05, np.nanmax(pivot.values) * 1.22))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(LEVEL_METHOD_PLOT, dpi=180)
    plt.close(fig)


def plot_rank_distributions(eval_rows: pd.DataFrame) -> None:
    method_order = [
        "level_tuned_product_hybrid",
        "tuned_product_hybrid",
        "full_product_hybrid",
        "latent_svd",
        "item_knn_collaborative",
        "popularity_baseline",
        "metadata_content",
        "graph_related",
        "people_staff_affinity",
    ]
    data = [
        eval_rows.loc[eval_rows["method"].eq(method), "rank"].to_numpy()
        for method in method_order
        if method in set(eval_rows["method"])
    ]
    labels = [method for method in method_order if method in set(eval_rows["method"])]
    fig, ax = plt.subplots(figsize=(11, 4.8))
    version_parts = tuple(int(part) for part in plt.matplotlib.__version__.split(".")[:2])
    boxplot_kwargs = {"showfliers": False, "patch_artist": True}
    if version_parts >= (3, 9):
        boxplot_kwargs["tick_labels"] = labels
    else:
        boxplot_kwargs["labels"] = labels
    box = ax.boxplot(data, **boxplot_kwargs)
    colors = ["#006D2C", "#2CA02C", "#59A14F", "#F28E2B", "#4C78A8", "#1F77B4", "#9C755F", "#76B7B2", "#EDC948", "#B07AA1"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title("Held-Out Liked Anime Rank Distribution")
    ax.set_ylabel("Rank among 101 candidates, lower is better")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RANK_DISTRIBUTION_PLOT, dpi=180)
    plt.close(fig)


def plot_mylist_recommendations(mylist_recommendations: pd.DataFrame) -> None:
    if mylist_recommendations.empty or "mylist_hybrid_score" not in mylist_recommendations.columns:
        return
    top = mylist_recommendations.head(12).iloc[::-1].copy()
    titles = top["title"].fillna(top["mal_id"].astype(str)).astype(str)
    titles = titles.str.slice(0, 48)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(titles, top["mylist_hybrid_score"], color="#4C78A8")
    ax.set_title("MyList Demo: Top Hybrid Recommendations")
    ax.set_xlabel("Hybrid score")
    ax.grid(axis="x", alpha=0.25)
    for bar in bars:
        value = bar.get_width()
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(MYLIST_PLOT, dpi=180)
    plt.close(fig)


def plot_mylist_guarded_recommendations(guarded_recommendations: pd.DataFrame) -> None:
    if guarded_recommendations.empty or "mylist_hybrid_score" not in guarded_recommendations.columns:
        return
    top = guarded_recommendations.head(12).iloc[::-1].copy()
    titles = top["title"].fillna(top["mal_id"].astype(str)).astype(str)
    titles = titles.str.slice(0, 48)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.barh(titles, top["mylist_hybrid_score"], color="#59A14F")
    ax.set_title("MyList Demo: Guarded Hybrid Recommendations")
    ax.set_xlabel("Hybrid score after relation/negative filters")
    ax.grid(axis="x", alpha=0.25)
    for bar in bars:
        value = bar.get_width()
        ax.text(value, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", ha="left", fontsize=8)
    fig.tight_layout()
    fig.savefig(MYLIST_GUARDED_PLOT, dpi=180)
    plt.close(fig)


def plot_discovery_metrics(metrics: pd.DataFrame) -> None:
    cols = ["coverage_at_12", "mean_novelty_at_12", "mean_genre_diversity_ratio_at_12"]
    if not set(cols).issubset(metrics.columns):
        return
    order = metrics.sort_values("hit_rate_at_12", ascending=False)["method"].tolist()
    plot_df = metrics.set_index("method").loc[order].reset_index()
    labels = {
        "coverage_at_12": "Coverage@12",
        "mean_novelty_at_12": "Novelty@12",
        "mean_genre_diversity_ratio_at_12": "Genre diversity@12",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#2CA02C", "#59A14F", "#F28E2B", "#4C78A8", "#1F77B4", "#9C755F", "#76B7B2", "#EDC948", "#B07AA1"]
    for ax, col in zip(axes, cols):
        bars = ax.bar(plot_df["method"], plot_df[col], color=colors[: len(plot_df)])
        ax.set_title(labels[col])
        ax.set_ylim(0, max(0.01, plot_df[col].max() * 1.22))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(DISCOVERY_METRICS_PLOT, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate anime recommenders with product-row @12 metrics.")
    parser.add_argument("--max-eval-users", type=int, default=MAX_EVAL_USERS, help="Maximum held-out users to evaluate. Omit for all eligible users.")
    parser.add_argument("--min-item-positives", type=int, default=MIN_ITEM_POSITIVES, help="Candidate pool threshold: anime must have at least this many positive interactions.")
    parser.add_argument("--max-model-users", type=int, default=MAX_MODEL_USERS, help="Maximum users used to build the training matrix.")
    parser.add_argument("--negatives-per-user", type=int, default=NEGATIVES_PER_USER, help="Sampled negative candidates per evaluated user.")
    parser.add_argument("--progress-interval", type=int, default=EVALUATION_PROGRESS_INTERVAL, help="Print classical/product evaluation progress every N users. Use 0 to disable.")
    parser.add_argument("--advanced-train-cases", type=int, default=ADVANCED_RERANKER_TRAIN_CASES, help="Held-out candidate groups used to train learned rerankers.")
    parser.add_argument("--advanced-eval-cases", type=int, default=ADVANCED_RERANKER_EVAL_CASES, help="Held-out candidate groups used to evaluate learned rerankers.")
    parser.add_argument("--skip-advanced", action="store_true", help="Skip learned reranker/deep model training.")
    return parser.parse_args()


def configure_from_args(args: argparse.Namespace) -> None:
    global MAX_EVAL_USERS, MIN_ITEM_POSITIVES, MAX_MODEL_USERS, NEGATIVES_PER_USER
    global ADVANCED_RERANKER_TRAIN_CASES, ADVANCED_RERANKER_EVAL_CASES, EVALUATION_PROGRESS_INTERVAL
    MAX_EVAL_USERS = args.max_eval_users
    MIN_ITEM_POSITIVES = int(args.min_item_positives)
    MAX_MODEL_USERS = int(args.max_model_users)
    NEGATIVES_PER_USER = int(args.negatives_per_user)
    EVALUATION_PROGRESS_INTERVAL = max(int(args.progress_interval), 0)
    ADVANCED_RERANKER_TRAIN_CASES = int(args.advanced_train_cases)
    ADVANCED_RERANKER_EVAL_CASES = int(args.advanced_eval_cases)


def main() -> None:
    args = parse_args()
    configure_from_args(args)
    ADVANCED_TRAINING_LOGS.clear()
    CLASSICAL_TRAINING_LOGS.clear()
    total_stages = 10

    stage_started = time.perf_counter()
    log_stage(1, total_stages, "Loading catalog and current user ratings")
    catalog = load_catalog()
    catalog_ids = set(catalog["mal_id"].astype(int))
    positives = load_positive_ratings(catalog_ids)
    record_classical_result("data_alignment_load", "input_layer", stage_started, len(positives), 0, "loaded")

    stage_started = time.perf_counter()
    log_stage(2, total_stages, "Preparing candidate pool, holdouts, and training matrix")
    positive_item_counts = positives["animeID"].value_counts()
    candidate_items = np.array(sorted(positive_item_counts[positive_item_counts >= MIN_ITEM_POSITIVES].index), dtype=np.int32)
    positives, eval_cases, model_users = prepare_split(positives, candidate_items, max_eval_users=MAX_EVAL_USERS)
    train_matrix, user_to_row, item_to_col, train = build_training_matrix(
        positives,
        model_users,
        candidate_items,
        eval_cases,
    )
    record_classical_result("candidate_pool_and_matrix", "data_split", stage_started, len(train), len(eval_cases), "built")

    log_stage(3, total_stages, "Building classical/product signal contexts")
    signal_started = time.perf_counter()
    train_item_counts = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float32)
    popularity_raw = np.log1p(train_item_counts)
    popularity_scaled = MinMaxScaler().fit_transform(popularity_raw.reshape(-1, 1)).ravel()
    popularity_by_item = {int(item): float(popularity_scaled[idx]) for idx, item in enumerate(candidate_items)}
    record_classical_result("popularity_baseline", "classical_baseline", signal_started, int(train_matrix.nnz), len(eval_cases), "built")

    signal_started = time.perf_counter()
    metadata_context = build_metadata_feature_context(catalog, candidate_items)
    record_classical_result("metadata_content", "content_ranker", signal_started, len(catalog), len(eval_cases), "built")

    signal_started = time.perf_counter()
    graph_scores = build_graph_score_context(catalog, candidate_items)
    record_classical_result("graph_related", "graph_ranker", signal_started, len(catalog), len(eval_cases), "built")

    signal_started = time.perf_counter()
    people_scores = build_people_score_context(candidate_items)
    record_classical_result("people_staff_affinity", "people_staff_ranker", signal_started, len(candidate_items), len(eval_cases), "built")

    stage_started = time.perf_counter()
    log_stage(4, total_stages, "Training latent SVD and item-neighborhood vectors")
    n_components = min(SVD_COMPONENTS, train_matrix.shape[1] - 1, train_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_.T
    item_knn_vectors = build_item_knn_vectors(item_factors)
    record_classical_result("latent_svd_and_item_knn", "collaborative_factorization", stage_started, int(train_matrix.nnz), len(eval_cases), "trained")

    stage_started = time.perf_counter()
    log_stage(5, total_stages, "Tuning global and profile-level product hybrid weights")
    tuned_hybrid_weights, hybrid_component_search = tune_product_hybrid_weights(
        eval_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
    )
    level_tuned_hybrid_weights, level_hybrid_component_search = tune_product_hybrid_weights_by_level(
        eval_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
    )
    record_classical_result("product_hybrid_weight_search", "hybrid_search", stage_started, len(hybrid_component_search), len(eval_cases), "tuned")

    stage_started = time.perf_counter()
    log_stage(6, total_stages, "Evaluating baseline, content, graph, collaborative, and product-hybrid methods")
    print(
        f"classical/product evaluation workload: users={len(eval_cases):,}; "
        f"candidate_pool_per_user={NEGATIVES_PER_USER + 1:,}; "
        f"progress_interval={EVALUATION_PROGRESS_INTERVAL:,}",
        flush=True,
    )
    eval_rows, examples = evaluate_rankings(
        eval_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_scaled,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
        level_tuned_hybrid_weights,
    )
    record_classical_result("classical_product_evaluation", "offline_ranking_eval", stage_started, len(eval_rows), len(eval_cases), "evaluated")
    advanced_inventory = detect_advanced_architecture_inventory()
    if args.skip_advanced:
        log_stage(7, total_stages, "Skipping advanced learned rerankers by request")
        advanced_eval_rows = pd.DataFrame()
        advanced_ranker_metrics = pd.DataFrame()
    else:
        log_stage(7, total_stages, "Training and evaluating advanced learned rerankers")
        advanced_eval_rows, advanced_ranker_metrics, advanced_inventory = evaluate_advanced_architectures(
            eval_cases,
            train_matrix,
            item_to_col,
            user_to_row,
            user_factors,
            item_factors,
            popularity_by_item,
            metadata_context,
            graph_scores,
            people_scores,
            item_knn_vectors,
            tuned_hybrid_weights,
            level_tuned_hybrid_weights,
        )
    log_stage(8, total_stages, "Running candidate-pool sensitivity checks")
    metrics = summarize_metrics(eval_rows, train_matrix, candidate_items)
    discovery_metrics = summarize_discovery_metrics(eval_rows, catalog, popularity_by_item, candidate_items)
    metrics = metrics.merge(discovery_metrics, on="method", how="left")
    negative_sensitivity, full_catalog_eval_rows = summarize_candidate_sensitivity(
        eval_cases,
        candidate_items,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_scaled,
        popularity_by_item,
        metadata_context,
        graph_scores,
        people_scores,
        item_knn_vectors,
        tuned_hybrid_weights,
        level_tuned_hybrid_weights,
    )
    stage_started = time.perf_counter()
    log_stage(9, total_stages, "Building reports, error cases, plots, and artifacts")
    level_metrics = summarize_metrics_by_level(eval_rows)
    metrics = add_balanced_profile_metrics(metrics, level_metrics)
    if not advanced_eval_rows.empty:
        advanced_level_metrics = summarize_metrics_by_level(advanced_eval_rows)
        advanced_ranker_metrics = add_balanced_profile_metrics(advanced_ranker_metrics, advanced_level_metrics)
    else:
        if advanced_eval_rows.empty and len(advanced_eval_rows.columns) == 0:
            advanced_eval_rows = pd.DataFrame(columns=eval_rows.columns)
        if advanced_ranker_metrics.empty and len(advanced_ranker_metrics.columns) == 0:
            advanced_ranker_metrics = pd.DataFrame(columns=metrics.columns)
        advanced_level_metrics = pd.DataFrame(columns=level_metrics.columns)
    examples = add_titles(examples, catalog)
    error_cases = build_error_case_table(eval_rows, catalog)
    if not advanced_eval_rows.empty and not advanced_ranker_metrics.empty:
        best_advanced_method = str(
            advanced_ranker_metrics.sort_values(
                ["balanced_profile_hit_at_12", "hit_rate_at_12", "map_at_12"],
                ascending=False,
            ).iloc[0]["method"]
        )
        advanced_error_cases = build_error_case_table(
            advanced_eval_rows,
            catalog,
            primary_method=best_advanced_method,
            baseline_method="advanced_split_popularity_baseline",
        )
    else:
        best_advanced_method = None
        advanced_error_cases = pd.DataFrame()
    alignment_contract = build_alignment_contract(
        catalog,
        positives,
        candidate_items,
        model_users,
        eval_cases,
        train,
        eval_rows,
    )
    beginner_candidates = build_beginner_entrypoint_candidates(catalog)
    mylist_recommendations = pd.DataFrame()
    mylist_guarded_recommendations = pd.DataFrame()
    mylist_guardrail_comparison = pd.DataFrame()
    mylist_guardrail_block_summary = pd.DataFrame()
    mylist_summary = {"available": False, "reason": "MyList demo moved to notebook 10 product-row evaluator"}
    mylist_guardrail_summary = {"available": False, "reason": "MyList demo moved to notebook 10 product-row evaluator"}

    metrics.to_csv(EVALUATION_PATH, index=False)
    level_metrics.to_csv(LEVEL_METRICS_PATH, index=False)
    eval_rows.to_csv(USER_EVAL_PATH, index=False)
    examples.to_csv(EXAMPLES_PATH, index=False)
    beginner_candidates.to_csv(BEGINNER_CANDIDATES_PATH, index=False)
    mylist_recommendations.to_csv(MY_LIST_RECOMMENDATIONS_PATH, index=False)
    mylist_guarded_recommendations.to_csv(MY_LIST_GUARDED_RECOMMENDATIONS_PATH, index=False)
    mylist_guardrail_comparison.to_csv(MY_LIST_GUARDRAIL_COMPARISON_PATH, index=False)
    mylist_guardrail_block_summary.to_csv(MY_LIST_GUARDRAIL_SUMMARY_PATH, index=False)
    hybrid_component_search.to_csv(HYBRID_COMPONENT_SEARCH_PATH, index=False)
    level_hybrid_component_search.to_csv(LEVEL_HYBRID_COMPONENT_SEARCH_PATH, index=False)
    advanced_ranker_metrics.to_csv(ADVANCED_RANKER_METRICS_PATH, index=False)
    advanced_level_metrics.to_csv(ADVANCED_RANKER_LEVEL_METRICS_PATH, index=False)
    advanced_eval_rows.to_csv(ADVANCED_RANKER_EVAL_PATH, index=False)
    advanced_inventory.to_csv(ADVANCED_ARCHITECTURE_INVENTORY_PATH, index=False)
    advanced_training_log = pd.DataFrame(
        ADVANCED_TRAINING_LOGS,
        columns=["model", "family", "status", "train_rows", "eval_rows", "elapsed_seconds", "note"],
    )
    advanced_training_log.to_csv(ADVANCED_TRAINING_LOG_PATH, index=False)
    build_advanced_model_config_table().to_csv(ADVANCED_MODEL_CONFIG_PATH, index=False)
    negative_sensitivity.to_csv(NEGATIVE_SENSITIVITY_PATH, index=False)
    full_catalog_eval_rows.to_csv(FULL_CATALOG_EVAL_PATH, index=False)
    error_cases.to_csv(ERROR_CASES_PATH, index=False)
    advanced_error_cases.to_csv(ADVANCED_ERROR_CASES_PATH, index=False)
    alignment_contract.to_csv(ALIGNMENT_CONTRACT_PATH, index=False)

    alignment = pd.DataFrame(
        [
            {"layer": "catalog", "rows": len(catalog), "definition": "anime_dataset rows"},
            {"layer": "positive_interactions", "rows": len(positives), "definition": f"ratings >= {LIKE_THRESHOLD} aligned to catalog"},
            {"layer": "candidate_pool", "rows": len(candidate_items), "definition": f"catalog anime with >= {MIN_ITEM_POSITIVES} positive interactions"},
            {"layer": "model_users", "rows": len(model_users), "definition": "eligible users used for matrix factorization training"},
            {"layer": "evaluation_users", "rows": len(eval_cases), "definition": "users with one positive item held out"},
        ]
    )
    alignment.to_csv(ALIGNMENT_PATH, index=False)

    plot_metrics(metrics)
    plot_level_metrics(eval_rows, level_metrics)
    plot_rank_distributions(eval_rows)
    plot_mylist_recommendations(mylist_recommendations)
    plot_mylist_guarded_recommendations(mylist_guarded_recommendations)
    plot_discovery_metrics(metrics)
    record_classical_result("artifact_writing_and_plots", "reporting", stage_started, len(metrics), len(eval_cases), "written")
    classical_training_log = pd.DataFrame(
        CLASSICAL_TRAINING_LOGS,
        columns=["component", "family", "status", "train_rows", "eval_rows", "elapsed_seconds", "note"],
    )
    classical_training_log.to_csv(CLASSICAL_TRAINING_LOG_PATH, index=False)
    log_stage(10, total_stages, "Writing final summary")

    summary = {
        "task": "personalized anime recommendation/ranking",
        "decision_supported": "rank candidate anime for a user based on scored liked anime, while allowing product filters and profile-band-aware recommendation modes",
        "baseline": "popularity_baseline: globally common liked anime in training data",
        "content_system": "metadata_content: user-profile similarity from AniList/MAL catalog labels, demographics, studios, rating, source, and type",
        "graph_system": "graph_related: relation and recommendation edges from liked titles",
        "people_system": "people_staff_affinity: shared voice actors plus director/original creator/original story signals",
        "collaborative_system": "latent_svd: personalized collaborative filtering from scored liked interactions",
        "product_safe_system": "level_tuned_product_hybrid: profile-band-specific tuned blend of SVD, item-item collaborative, popularity, metadata, graph, and people/staff signals",
        "advanced_architecture_systems": (
            "separate train/eval comparison of pointwise logistic reranking, gradient-boosted tree reranking, "
            "neural MLP reranking, and Torch two-tower collaborative embeddings when available"
        ),
        "full_hybrid_weights": FULL_HYBRID_WEIGHTS,
        "tuned_hybrid_weights": tuned_hybrid_weights,
        "level_tuned_hybrid_weights": level_tuned_hybrid_weights,
        "hybrid_component_weight_search": project_path(HYBRID_COMPONENT_SEARCH_PATH),
        "level_hybrid_component_weight_search": project_path(LEVEL_HYBRID_COMPONENT_SEARCH_PATH),
        "advanced_ranker_metrics": project_path(ADVANCED_RANKER_METRICS_PATH),
        "advanced_ranker_level_metrics": project_path(ADVANCED_RANKER_LEVEL_METRICS_PATH),
        "advanced_architecture_inventory": project_path(ADVANCED_ARCHITECTURE_INVENTORY_PATH),
        "advanced_training_log": project_path(ADVANCED_TRAINING_LOG_PATH),
        "classical_training_log": project_path(CLASSICAL_TRAINING_LOG_PATH),
        "advanced_model_configs": project_path(ADVANCED_MODEL_CONFIG_PATH),
        "advanced_available_models": advanced_inventory.loc[advanced_inventory["available"], "architecture"].tolist(),
        "best_advanced_method": best_advanced_method,
        "user_levels": USER_LEVELS,
        "control_filters": CONTROL_FILTERS,
        "level_policy_note": (
            "Level thresholds are product rules based on completed/known entries. "
            "Offline evaluation uses scored catalog-matched ratings as interaction evidence; when status exists, dropped rows are excluded from positive seeds."
        ),
        "ratings_source": project_path(RATINGS_PATH),
        "ratings_source_note": (
            "The evaluator uses current_user_ratings.csv from the public-list collector. "
            "The collector includes scored completed, watching, on-hold, dropped, and plan-to-watch rows; positives are ratings >= 7, excluding dropped entries."
        ),
        "catalog_source": project_path(CATALOG_PATH),
        "like_threshold": LIKE_THRESHOLD,
        "candidate_pool": {
            "definition": f"catalog anime with at least {MIN_ITEM_POSITIVES} positive ratings in the filtered interaction layer",
            "items": int(len(candidate_items)),
            "sampled_negatives_per_user": NEGATIVES_PER_USER,
            "max_eval_users": MAX_EVAL_USERS,
            "advanced_train_cases": ADVANCED_RERANKER_TRAIN_CASES,
            "advanced_eval_cases": ADVANCED_RERANKER_EVAL_CASES,
        },
        "train_test_logic": "per eligible user, one liked anime is deterministically held out; the user's remaining liked anime stay in training",
        "leakage_note": "held-out user-item pairs are removed from the training matrix; no timestamp claims are made because the rating source has no event time",
        "model_users": int(len(model_users)),
        "evaluated_users": int(len(eval_cases)),
        "positive_interactions": int(len(positives)),
        "svd_components": int(n_components),
        "svd_retained_energy_proxy": float(np.sum(svd.explained_variance_ratio_)),
        "mylist_example": mylist_summary,
        "mylist_guardrail_example": mylist_guardrail_summary,
        "outputs": {
            "metrics": project_path(EVALUATION_PATH),
            "level_metrics": project_path(LEVEL_METRICS_PATH),
            "examples": project_path(EXAMPLES_PATH),
            "beginner_entrypoint_candidates": project_path(BEGINNER_CANDIDATES_PATH),
            "alignment": project_path(ALIGNMENT_PATH),
            "user_eval_rows": project_path(USER_EVAL_PATH),
            "negative_sampling_sensitivity": project_path(NEGATIVE_SENSITIVITY_PATH),
            "full_catalog_eval_sample": project_path(FULL_CATALOG_EVAL_PATH),
            "level_hybrid_component_weight_search": project_path(LEVEL_HYBRID_COMPONENT_SEARCH_PATH),
            "advanced_ranker_metrics": project_path(ADVANCED_RANKER_METRICS_PATH),
            "advanced_ranker_level_metrics": project_path(ADVANCED_RANKER_LEVEL_METRICS_PATH),
            "advanced_ranker_eval_rows": project_path(ADVANCED_RANKER_EVAL_PATH),
            "advanced_architecture_inventory": project_path(ADVANCED_ARCHITECTURE_INVENTORY_PATH),
            "advanced_training_log": project_path(ADVANCED_TRAINING_LOG_PATH),
            "classical_training_log": project_path(CLASSICAL_TRAINING_LOG_PATH),
            "advanced_model_configs": project_path(ADVANCED_MODEL_CONFIG_PATH),
            "systematic_error_cases": project_path(ERROR_CASES_PATH),
            "advanced_systematic_error_cases": project_path(ADVANCED_ERROR_CASES_PATH),
            "alignment_contract": project_path(ALIGNMENT_CONTRACT_PATH),
            "metric_plot": project_path(PLOT_DIR / "metric_comparison.png"),
            "rank_plot": project_path(PLOT_DIR / "median_rank.png"),
            "level_distribution_plot": project_path(PLOT_DIR / "user_level_distribution.png"),
            "level_hit_plot": project_path(PLOT_DIR / "hit_at_12_by_user_level.png"),
            "rank_distribution_plot": project_path(RANK_DISTRIBUTION_PLOT),
            "level_method_plot": project_path(LEVEL_METHOD_PLOT),
            "mylist_plot": project_path(MYLIST_PLOT),
            "mylist_guarded_plot": project_path(MYLIST_GUARDED_PLOT),
            "discovery_metrics_plot": project_path(DISCOVERY_METRICS_PLOT),
        },
        "feedback_fixes": {
            "candidate_pool_sensitivity": project_path(NEGATIVE_SENSITIVITY_PATH),
            "full_catalog_sample_users": FULL_CATALOG_EVAL_USERS,
            "error_cases": project_path(ERROR_CASES_PATH),
            "advanced_error_cases": project_path(ADVANCED_ERROR_CASES_PATH),
            "alignment_contract": project_path(ALIGNMENT_CONTRACT_PATH),
            "note": "These artifacts address the feedback about sampled negatives, concrete strong/failure cases, and explicit matrix/id validation.",
        },
    }
    summary = relativize_payload(summary)
    atomic_write_json(RUN_SUMMARY_PATH, summary)

    print("Recommender evaluation complete.", flush=True)
    print(f"Summary: {project_path(RUN_SUMMARY_PATH)}", flush=True)
    print(f"Classical timings: {project_path(CLASSICAL_TRAINING_LOG_PATH)}", flush=True)
    print(f"Advanced timings: {project_path(ADVANCED_TRAINING_LOG_PATH)}", flush=True)
    leaderboard_cols = [
        "method",
        "hit_rate_at_12",
        "ndcg_at_12",
        "map_at_12",
        "balanced_profile_hit_at_12",
    ]
    print("Classical/product leaderboard:", flush=True)
    print(
        metrics.sort_values(["balanced_profile_hit_at_12", "hit_rate_at_12", "map_at_12"], ascending=False)[leaderboard_cols]
        .head(8)
        .to_string(index=False),
        flush=True,
    )
    if not advanced_ranker_metrics.empty:
        print("Advanced leaderboard:", flush=True)
        print(
            advanced_ranker_metrics.sort_values(
                ["balanced_profile_hit_at_12", "hit_rate_at_12", "map_at_12"],
                ascending=False,
            )[leaderboard_cols]
            .head(8)
            .to_string(index=False),
            flush=True,
        )


if __name__ == "__main__":
    main()
