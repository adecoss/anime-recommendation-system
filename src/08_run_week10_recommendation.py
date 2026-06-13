from __future__ import annotations

import json
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
ARTIFACT_DIR = BASE_DIR / "artifacts" / "recommendation"
PLOT_DIR = BASE_DIR / "artifacts" / "plots" / "week10"

RATINGS_PATH = DATA_DIR / "ratings_processed.csv"
CATALOG_PATH = DATA_DIR / "anime_dataset.csv"
MY_LIST_PATH = BASE_DIR / "data" / "raw" / "MyList.xml"

ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)


LIKE_THRESHOLD = 7
MIN_USER_POSITIVES = 5
MIN_ITEM_POSITIVES = 20
MAX_MODEL_USERS = 100_000
MAX_EVAL_USERS = 15_000
NEGATIVES_PER_USER = 100
SVD_COMPONENTS = 64
HYBRID_LATENT_WEIGHT = 0.85
RANDOM_STATE = 42
CHUNK_SIZE = 2_000_000


EVALUATION_PATH = ARTIFACT_DIR / "week10_evaluation_metrics.csv"
RUN_SUMMARY_PATH = ARTIFACT_DIR / "week10_recommendation_summary.json"
EXAMPLES_PATH = ARTIFACT_DIR / "week10_recommendation_examples.csv"
ALIGNMENT_PATH = ARTIFACT_DIR / "week10_data_alignment.csv"
USER_EVAL_PATH = ARTIFACT_DIR / "week10_user_level_eval_sample.csv"
LEVEL_METRICS_PATH = ARTIFACT_DIR / "week10_metrics_by_user_level.csv"
BEGINNER_CANDIDATES_PATH = ARTIFACT_DIR / "week10_beginner_entrypoint_candidates.csv"
MY_LIST_RECOMMENDATIONS_PATH = ARTIFACT_DIR / "week10_mylist_recommendation_example.csv"
MY_LIST_GUARDED_RECOMMENDATIONS_PATH = ARTIFACT_DIR / "week10_mylist_guarded_recommendation_example.csv"
MY_LIST_GUARDRAIL_COMPARISON_PATH = ARTIFACT_DIR / "week10_mylist_guardrail_comparison.csv"
MY_LIST_GUARDRAIL_SUMMARY_PATH = ARTIFACT_DIR / "week10_mylist_guardrail_block_summary.csv"
RANK_DISTRIBUTION_PLOT = PLOT_DIR / "week10_rank_distribution_by_method.png"
LEVEL_METHOD_PLOT = PLOT_DIR / "week10_method_hit_by_user_level.png"
MYLIST_PLOT = PLOT_DIR / "week10_mylist_top_recommendations.png"
MYLIST_GUARDED_PLOT = PLOT_DIR / "week10_mylist_guarded_top_recommendations.png"
DISCOVERY_METRICS_PLOT = PLOT_DIR / "week10_discovery_metrics.png"

USER_LEVELS = [
    {
        "level": "Newcomer",
        "min_known_entries": 0,
        "max_known_entries": 19,
        "primary_signal": "cold-start onboarding, genre filters, popularity, score, short entry points",
        "risk": "no reliable personal vector; collaborative filtering needs a popularity/content fallback",
    },
    {
        "level": "Casual Viewer",
        "min_known_entries": 20,
        "max_known_entries": 49,
        "primary_signal": "genre onboarding, recognizable popular titles, short lists, early content similarity",
        "risk": "some watched anime exist, but the profile is still too small for a stable latent taste vector",
    },
    {
        "level": "Explorer",
        "min_known_entries": 50,
        "max_known_entries": 99,
        "primary_signal": "similar anime, relation navigation, popularity prior, high-rated anchors",
        "risk": "taste is still forming; the model can overfit to a few early favorites",
    },
    {
        "level": "Regular Fan",
        "min_known_entries": 100,
        "max_known_entries": 249,
        "primary_signal": "collaborative filtering, relation continuation, content/tag similarity, current discovery",
        "risk": "model should balance familiar taste with controlled exploration",
    },
    {
        "level": "Dedicated Fan",
        "min_known_entries": 250,
        "max_known_entries": 499,
        "primary_signal": "collaborative ranking, niche clusters, relation navigation, controlled novelty",
        "risk": "the user knows many common recommendations, so repetition becomes more visible",
    },
    {
        "level": "Veteran Fan",
        "min_known_entries": 500,
        "max_known_entries": 749,
        "primary_signal": "collaborative ranking, niche clusters, graph-aware expansion, seasonal/current discovery",
        "risk": "many obvious items are already known; recommendations need novelty and coverage",
    },
    {
        "level": "Completionist",
        "min_known_entries": 750,
        "max_known_entries": None,
        "primary_signal": "long-tail discovery, novelty, obscure catalog coverage, graph-aware exploration",
        "risk": "hardest group: exact recovery is difficult and obvious catalog items are saturated",
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
        "demographics",
        "rating",
        "episodes",
        "duration",
        "total_watch_minutes",
        "aired_year",
        "relations",
    ]
    available = pd.read_csv(CATALOG_PATH, nrows=0).columns
    usecols = [col for col in cols if col in available]
    catalog = pd.read_csv(CATALOG_PATH, usecols=usecols)
    catalog["mal_id"] = catalog["mal_id"].astype(np.int32)
    return catalog


def load_positive_ratings(catalog_ids: set[int]) -> pd.DataFrame:
    chunks: list[pd.DataFrame] = []
    rows_read = 0
    rows_kept = 0

    for chunk in pd.read_csv(
        RATINGS_PATH,
        usecols=["userID", "animeID", "rating"],
        dtype={"userID": np.int32, "animeID": np.int32, "rating": np.int8},
        chunksize=CHUNK_SIZE,
    ):
        rows_read += len(chunk)
        chunk = chunk[(chunk["rating"] >= LIKE_THRESHOLD) & (chunk["animeID"].isin(catalog_ids))]
        if not chunk.empty:
            chunks.append(chunk[["userID", "animeID"]].copy())
            rows_kept += len(chunk)
        print(f"streamed ratings rows={rows_read:,}; positives kept={rows_kept:,}")

    if not chunks:
        raise RuntimeError("No positive ratings found. Check ratings_processed.csv and catalog ids.")

    positives = pd.concat(chunks, ignore_index=True)
    positives.drop_duplicates(["userID", "animeID"], inplace=True)
    positives["userID"] = positives["userID"].astype(np.int32)
    positives["animeID"] = positives["animeID"].astype(np.int32)
    return positives


def deterministic_holdout(user_id: int, items: Iterable[int]) -> tuple[int, tuple[int, ...]]:
    unique_items = sorted(set(int(item) for item in items))
    index = (user_id * 1_103_515_245 + 12_345) % len(unique_items)
    holdout = unique_items[index]
    train_items = tuple(item for item in unique_items if item != holdout)
    return holdout, train_items


def stable_user_sample(user_ids: np.ndarray, size: int, random_state: int) -> np.ndarray:
    user_ids = np.array(sorted(set(int(user_id) for user_id in user_ids)), dtype=np.int32)
    if len(user_ids) <= size:
        return user_ids
    rng = np.random.default_rng(random_state)
    selected = rng.choice(user_ids, size=size, replace=False)
    return np.array(sorted(selected), dtype=np.int32)


def user_level_from_count(known_entries: int) -> str:
    for level in USER_LEVELS:
        maximum = level["max_known_entries"]
        if known_entries >= level["min_known_entries"] and (maximum is None or known_entries <= maximum):
            return level["level"]
    return USER_LEVELS[-1]["level"]


def prepare_split(positives: pd.DataFrame, candidate_items: np.ndarray) -> tuple[pd.DataFrame, dict[int, EvalCase], np.ndarray]:
    candidate_set = set(int(item) for item in candidate_items)
    positives = positives[positives["animeID"].isin(candidate_set)].copy()

    user_counts = positives.groupby("userID")["animeID"].nunique()
    eligible_users = user_counts[user_counts >= MIN_USER_POSITIVES].index.to_numpy(dtype=np.int32)
    eval_users = stable_user_sample(eligible_users, MAX_EVAL_USERS, RANDOM_STATE)

    model_user_pool = stable_user_sample(eligible_users, MAX_MODEL_USERS, RANDOM_STATE + 7)
    model_users = np.array(sorted(set(model_user_pool).union(set(eval_users))), dtype=np.int32)

    grouped = positives[positives["userID"].isin(eval_users)].groupby("userID")["animeID"].apply(list)
    all_candidate_items = np.array(sorted(candidate_set), dtype=np.int32)
    eval_cases: dict[int, EvalCase] = {}
    rng = np.random.default_rng(RANDOM_STATE + 101)

    for user_id, items in grouped.items():
        holdout, train_items = deterministic_holdout(int(user_id), items)
        blocked = set(train_items)
        blocked.add(holdout)
        negative_pool = np.array([item for item in all_candidate_items if item not in blocked], dtype=np.int32)
        if len(negative_pool) < NEGATIVES_PER_USER:
            continue
        negatives = rng.choice(negative_pool, size=NEGATIVES_PER_USER, replace=False)
        candidate_list = np.array([holdout, *negatives.tolist()], dtype=np.int32)
        eval_cases[int(user_id)] = EvalCase(
            user_id=int(user_id),
            holdout_item=int(holdout),
            train_items=tuple(int(item) for item in train_items),
            candidate_items=candidate_list,
        )

    return positives, eval_cases, model_users


def build_training_matrix(
    positives: pd.DataFrame,
    model_users: np.ndarray,
    candidate_items: np.ndarray,
    eval_cases: dict[int, EvalCase],
) -> tuple[csr_matrix, dict[int, int], dict[int, int], pd.DataFrame]:
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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    example_rows = []

    catalog_item_ids = np.array(list(item_to_col.keys()), dtype=np.int32)

    for case in eval_cases.values():
        if case.user_id not in user_to_row:
            continue
        candidate_cols = np.array([item_to_col[int(item)] for item in case.candidate_items], dtype=np.int32)
        latent_scores = user_factors[user_to_row[case.user_id]] @ item_factors[candidate_cols].T
        pop_scores = np.array([popularity_by_item.get(int(item), 0.0) for item in case.candidate_items], dtype=np.float32)

        if latent_scores.max() > latent_scores.min():
            latent_norm = (latent_scores - latent_scores.min()) / (latent_scores.max() - latent_scores.min())
        else:
            latent_norm = np.zeros_like(latent_scores)
        if pop_scores.max() > pop_scores.min():
            pop_norm = (pop_scores - pop_scores.min()) / (pop_scores.max() - pop_scores.min())
        else:
            pop_norm = np.zeros_like(pop_scores)
        rng = np.random.default_rng(RANDOM_STATE + case.user_id)
        random_scores = rng.random(len(case.candidate_items))

        method_scores = {
            "random_uniform": random_scores,
            "popularity_baseline": pop_scores,
            "latent_svd": latent_scores,
            "hybrid_svd_popularity": HYBRID_LATENT_WEIGHT * latent_norm + (1 - HYBRID_LATENT_WEIGHT) * pop_norm,
        }

        ranks = {method: rank_of_holdout(scores) for method, scores in method_scores.items()}
        for method, rank in ranks.items():
            profile_size = len(case.train_items)
            top_idx = np.argsort(-method_scores[method])[:10]
            top10_ids = [int(case.candidate_items[i]) for i in top_idx]
            rows.append(
                {
                    "userID": case.user_id,
                    "method": method,
                    "holdout_animeID": case.holdout_item,
                    "profile_size": profile_size,
                    "user_level": user_level_from_count(profile_size),
                    "rank": rank,
                    "hit_at_10": int(rank <= 10),
                    "ndcg_at_10": float(1.0 / math.log2(rank + 1) if rank <= 10 else 0.0),
                    "mrr": float(1.0 / rank),
                    "top10_anime_ids": "|".join(str(item) for item in top10_ids),
                }
            )

        if len(example_rows) < 200:
            hybrid_scores = method_scores["hybrid_svd_popularity"]
            top_idx = np.argsort(-hybrid_scores)[:10]
            example_rows.append(
                {
                    "userID": case.user_id,
                    "holdout_animeID": case.holdout_item,
                    "train_anime_ids": "|".join(str(item) for item in case.train_items[:20]),
                    "profile_size": len(case.train_items),
                    "user_level": user_level_from_count(len(case.train_items)),
                    "popularity_rank": ranks["popularity_baseline"],
                    "latent_svd_rank": ranks["latent_svd"],
                    "hybrid_rank": ranks["hybrid_svd_popularity"],
                    "hybrid_top10_anime_ids": "|".join(str(int(case.candidate_items[i])) for i in top_idx),
                    "candidate_pool_size": len(case.candidate_items),
                }
            )

    eval_rows = pd.DataFrame(rows)
    examples = pd.DataFrame(example_rows)
    return eval_rows, examples


def summarize_metrics(eval_rows: pd.DataFrame, train_matrix: csr_matrix, item_ids: np.ndarray) -> pd.DataFrame:
    metrics = (
        eval_rows.groupby("method")
        .agg(
            evaluated_users=("userID", "nunique"),
            hit_rate_at_10=("hit_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mean_reciprocal_rank=("mrr", "mean"),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    metrics["candidate_pool_per_user"] = NEGATIVES_PER_USER + 1
    metrics["like_threshold"] = LIKE_THRESHOLD
    metrics["min_user_positives"] = MIN_USER_POSITIVES
    metrics["min_item_positives"] = MIN_ITEM_POSITIVES
    return metrics.sort_values("hit_rate_at_10", ascending=False)


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
        all_top_items: list[int] = []
        novelty_values: list[float] = []
        unique_genre_counts: list[int] = []
        diversity_ratios: list[float] = []

        for value in group["top10_anime_ids"]:
            top_ids = split_pipe_ids(value)
            all_top_items.extend(top_ids)
            novelty_values.extend(1.0 - popularity_by_item.get(item, 0.0) for item in top_ids)

            genres: list[str] = []
            for item in top_ids:
                genres.extend(sorted(catalog_genres.get(item, set())))
            unique_genres = set(genres)
            unique_genre_counts.append(len(unique_genres))
            diversity_ratios.append(len(unique_genres) / max(len(genres), 1))

        rows.append(
            {
                "method": method,
                "coverage_at_10": len(set(all_top_items)) / candidate_count,
                "unique_recommended_at_10": len(set(all_top_items)),
                "mean_novelty_at_10": float(np.mean(novelty_values)) if novelty_values else np.nan,
                "mean_unique_genres_at_10": float(np.mean(unique_genre_counts)) if unique_genre_counts else np.nan,
                "mean_genre_diversity_ratio_at_10": float(np.mean(diversity_ratios)) if diversity_ratios else np.nan,
            }
        )

    return pd.DataFrame(rows)


def summarize_metrics_by_level(eval_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = (
        eval_rows.groupby(["user_level", "method"])
        .agg(
            evaluated_users=("userID", "nunique"),
            median_profile_size=("profile_size", "median"),
            hit_rate_at_10=("hit_at_10", "mean"),
            ndcg_at_10=("ndcg_at_10", "mean"),
            mean_reciprocal_rank=("mrr", "mean"),
            median_rank=("rank", "median"),
            mean_rank=("rank", "mean"),
        )
        .reset_index()
    )
    level_order = {level["level"]: idx for idx, level in enumerate(USER_LEVELS)}
    method_order = {
        "hybrid_svd_popularity": 0,
        "latent_svd": 1,
        "popularity_baseline": 2,
        "random_uniform": 3,
    }
    metrics["_level_order"] = metrics["user_level"].map(level_order).fillna(999)
    metrics["_method_order"] = metrics["method"].map(method_order).fillna(999)
    return metrics.sort_values(["_level_order", "_method_order"]).drop(columns=["_level_order", "_method_order"])


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
    examples["hybrid_top10_titles"] = examples["hybrid_top10_anime_ids"].apply(lambda value: names(value, limit=10))
    examples["case_type"] = np.select(
        [
            (examples["hybrid_rank"] <= 10) & (examples["popularity_rank"] > 10),
            (examples["hybrid_rank"] > 10) & (examples["popularity_rank"] <= 10),
            (examples["hybrid_rank"] <= 10) & (examples["popularity_rank"] <= 10),
        ],
        ["hybrid_win", "hybrid_failure_vs_popularity", "both_hit"],
        default="both_miss",
    )
    return examples


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
    hybrid_scores = HYBRID_LATENT_WEIGHT * latent_norm + (1 - HYBRID_LATENT_WEIGHT) * pop_norm

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
    order = metrics.sort_values("hit_rate_at_10", ascending=False)["method"].tolist()
    plot_df = metrics.set_index("method").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bar_color = ["#4C78A8", "#F28E2B", "#59A14F", "#9C755F", "#B07AA1"]
    for ax, metric, title in [
        (axes[0], "hit_rate_at_10", "Hit Rate@10"),
        (axes[1], "ndcg_at_10", "NDCG@10"),
        (axes[2], "mean_reciprocal_rank", "MRR"),
    ]:
        bars = ax.bar(plot_df["method"], plot_df[metric], color=bar_color[: len(plot_df)])
        ax.set_title(title)
        ax.set_ylim(0, max(0.01, plot_df[metric].max() * 1.2))
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "week10_metric_comparison.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(plot_df["method"], plot_df["median_rank"], color="#B07AA1")
    ax.set_title("Median Holdout Rank Lower Is Better")
    ax.set_ylabel("Median rank among 101 candidates")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(plot_df["median_rank"]):
        ax.text(idx, value, f"{value:.0f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "week10_median_rank.png", dpi=180)
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
    fig.savefig(PLOT_DIR / "week10_user_level_distribution.png", dpi=180)
    plt.close(fig)

    hybrid = level_metrics[level_metrics["method"].eq("hybrid_svd_popularity")].copy()
    hybrid["user_level"] = pd.Categorical(hybrid["user_level"], categories=level_order, ordered=True)
    hybrid = hybrid.sort_values("user_level")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(hybrid["user_level"].astype(str), hybrid["hit_rate_at_10"], color="#59A14F")
    ax.set_title("Hybrid Hit@10 by Recommendation Profile Band")
    ax.set_ylabel("Hit@10")
    ax.set_ylim(0, max(0.05, hybrid["hit_rate_at_10"].max() * 1.2))
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.25)
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "week10_hit_at_10_by_user_level.png", dpi=180)
    plt.close(fig)

    method_order = ["random_uniform", "popularity_baseline", "latent_svd", "hybrid_svd_popularity"]
    pivot = level_metrics.pivot_table(index="user_level", columns="method", values="hit_rate_at_10", aggfunc="first")
    pivot = pivot.reindex(level_order)[[method for method in method_order if method in pivot.columns]]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    x = np.arange(len(pivot.index))
    width = 0.18
    colors = ["#9C755F", "#59A14F", "#4C78A8", "#F28E2B"]
    for idx, method in enumerate(pivot.columns):
        offsets = x + (idx - (len(pivot.columns) - 1) / 2) * width
        bars = ax.bar(offsets, pivot[method].values, width=width, label=method, color=colors[idx % len(colors)])
        for bar in bars:
            value = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.2f}", ha="center", va="bottom", fontsize=7)
    ax.set_title("Hit@10 by User Level and Ranking Method")
    ax.set_ylabel("Hit@10")
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index, rotation=20)
    ax.set_ylim(0, max(0.05, np.nanmax(pivot.values) * 1.22))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(LEVEL_METHOD_PLOT, dpi=180)
    plt.close(fig)


def plot_rank_distributions(eval_rows: pd.DataFrame) -> None:
    method_order = ["random_uniform", "popularity_baseline", "latent_svd", "hybrid_svd_popularity"]
    data = [
        eval_rows.loc[eval_rows["method"].eq(method), "rank"].to_numpy()
        for method in method_order
        if method in set(eval_rows["method"])
    ]
    labels = [method for method in method_order if method in set(eval_rows["method"])]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    version_parts = tuple(int(part) for part in plt.matplotlib.__version__.split(".")[:2])
    boxplot_kwargs = {"showfliers": False, "patch_artist": True}
    if version_parts >= (3, 9):
        boxplot_kwargs["tick_labels"] = labels
    else:
        boxplot_kwargs["labels"] = labels
    box = ax.boxplot(data, **boxplot_kwargs)
    colors = ["#9C755F", "#59A14F", "#4C78A8", "#F28E2B"]
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
    cols = ["coverage_at_10", "mean_novelty_at_10", "mean_genre_diversity_ratio_at_10"]
    if not set(cols).issubset(metrics.columns):
        return
    order = metrics.sort_values("hit_rate_at_10", ascending=False)["method"].tolist()
    plot_df = metrics.set_index("method").loc[order].reset_index()
    labels = {
        "coverage_at_10": "Coverage@10",
        "mean_novelty_at_10": "Novelty@10",
        "mean_genre_diversity_ratio_at_10": "Genre diversity@10",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#4C78A8", "#F28E2B", "#59A14F", "#9C755F"]
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


def main() -> None:
    catalog = load_catalog()
    catalog_ids = set(catalog["mal_id"].astype(int))
    positives = load_positive_ratings(catalog_ids)

    positive_item_counts = positives["animeID"].value_counts()
    candidate_items = np.array(sorted(positive_item_counts[positive_item_counts >= MIN_ITEM_POSITIVES].index), dtype=np.int32)
    positives, eval_cases, model_users = prepare_split(positives, candidate_items)
    train_matrix, user_to_row, item_to_col, train = build_training_matrix(
        positives,
        model_users,
        candidate_items,
        eval_cases,
    )

    train_item_counts = np.asarray(train_matrix.sum(axis=0)).ravel().astype(np.float32)
    popularity_raw = np.log1p(train_item_counts)
    popularity_scaled = MinMaxScaler().fit_transform(popularity_raw.reshape(-1, 1)).ravel()
    popularity_by_item = {int(item): float(popularity_scaled[idx]) for idx, item in enumerate(candidate_items)}

    n_components = min(SVD_COMPONENTS, train_matrix.shape[1] - 1, train_matrix.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    user_factors = svd.fit_transform(train_matrix)
    item_factors = svd.components_.T

    eval_rows, examples = evaluate_rankings(
        eval_cases,
        item_to_col,
        user_to_row,
        user_factors,
        item_factors,
        popularity_scaled,
        popularity_by_item,
    )
    metrics = summarize_metrics(eval_rows, train_matrix, candidate_items)
    discovery_metrics = summarize_discovery_metrics(eval_rows, catalog, popularity_by_item, candidate_items)
    metrics = metrics.merge(discovery_metrics, on="method", how="left")
    level_metrics = summarize_metrics_by_level(eval_rows)
    examples = add_titles(examples, catalog)
    beginner_candidates = build_beginner_entrypoint_candidates(catalog)
    mylist_recommendations, mylist_summary = build_mylist_recommendations(
        catalog=catalog,
        candidate_items=candidate_items,
        item_to_col=item_to_col,
        item_factors=item_factors,
        popularity_by_item=popularity_by_item,
    )
    mylist_guarded_recommendations, mylist_guardrail_comparison, mylist_guardrail_summary = (
        build_mylist_guarded_recommendations(mylist_recommendations, catalog)
    )
    mylist_guardrail_block_summary = summarize_guardrail_blocks(mylist_guardrail_comparison)

    metrics.to_csv(EVALUATION_PATH, index=False)
    level_metrics.to_csv(LEVEL_METRICS_PATH, index=False)
    eval_rows.sample(min(len(eval_rows), 15_000), random_state=RANDOM_STATE).to_csv(USER_EVAL_PATH, index=False)
    examples.to_csv(EXAMPLES_PATH, index=False)
    beginner_candidates.to_csv(BEGINNER_CANDIDATES_PATH, index=False)
    mylist_recommendations.to_csv(MY_LIST_RECOMMENDATIONS_PATH, index=False)
    mylist_guarded_recommendations.to_csv(MY_LIST_GUARDED_RECOMMENDATIONS_PATH, index=False)
    mylist_guardrail_comparison.to_csv(MY_LIST_GUARDRAIL_COMPARISON_PATH, index=False)
    mylist_guardrail_block_summary.to_csv(MY_LIST_GUARDRAIL_SUMMARY_PATH, index=False)

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

    summary = {
        "task": "personalized anime recommendation/ranking",
        "decision_supported": "rank candidate anime for a user based on scored liked anime, while allowing product filters and profile-band-aware recommendation modes",
        "chance_floor": "random_uniform: sanity baseline for top-k ranking under sampled candidates",
        "baseline": "popularity_baseline: globally common liked anime in training data",
        "stronger_system": "latent_svd and hybrid_svd_popularity: personalized collaborative models that should beat random and popularity baselines",
        "product_safe_system": "hybrid_svd_popularity: latent collaborative SVD score blended with popularity for safer sparse-profile recommendations",
        "user_levels": USER_LEVELS,
        "control_filters": CONTROL_FILTERS,
        "level_policy_note": (
            "Level thresholds are product rules based on completed/known entries. "
            "Offline evaluation approximates this with known liked ratings because the Kaggle source has ratings but no chronological watch history."
        ),
        "ratings_source": str(RATINGS_PATH),
        "ratings_source_note": (
            "The stable interaction layer is the 2020 Kaggle MAL ratings dataset. "
            "A separate current-user ratings collector is being built for a later deliverable, but it is not yet used for this benchmark."
        ),
        "catalog_source": str(CATALOG_PATH),
        "like_threshold": LIKE_THRESHOLD,
        "candidate_pool": {
            "definition": f"catalog anime with at least {MIN_ITEM_POSITIVES} positive ratings in the filtered interaction layer",
            "items": int(len(candidate_items)),
            "sampled_negatives_per_user": NEGATIVES_PER_USER,
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
            "metrics": str(EVALUATION_PATH),
            "level_metrics": str(LEVEL_METRICS_PATH),
            "examples": str(EXAMPLES_PATH),
            "beginner_entrypoint_candidates": str(BEGINNER_CANDIDATES_PATH),
            "mylist_recommendation_example": str(MY_LIST_RECOMMENDATIONS_PATH),
            "mylist_guarded_recommendation_example": str(MY_LIST_GUARDED_RECOMMENDATIONS_PATH),
            "mylist_guardrail_comparison": str(MY_LIST_GUARDRAIL_COMPARISON_PATH),
            "mylist_guardrail_block_summary": str(MY_LIST_GUARDRAIL_SUMMARY_PATH),
            "alignment": str(ALIGNMENT_PATH),
            "user_eval_sample": str(USER_EVAL_PATH),
            "metric_plot": str(PLOT_DIR / "week10_metric_comparison.png"),
            "rank_plot": str(PLOT_DIR / "week10_median_rank.png"),
            "level_distribution_plot": str(PLOT_DIR / "week10_user_level_distribution.png"),
            "level_hit_plot": str(PLOT_DIR / "week10_hit_at_10_by_user_level.png"),
            "rank_distribution_plot": str(RANK_DISTRIBUTION_PLOT),
            "level_method_plot": str(LEVEL_METHOD_PLOT),
            "mylist_plot": str(MYLIST_PLOT),
            "mylist_guarded_plot": str(MYLIST_GUARDED_PLOT),
            "discovery_metrics_plot": str(DISCOVERY_METRICS_PLOT),
        },
    }
    atomic_write_json(RUN_SUMMARY_PATH, summary)

    print("Week 10 recommendation run complete.")
    print(json.dumps(summary, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
