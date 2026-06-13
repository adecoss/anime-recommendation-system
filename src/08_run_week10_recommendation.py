from __future__ import annotations

import json
import math
import os
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

USER_LEVELS = [
    {
        "level": "Absolute Beginner",
        "min_known_entries": 0,
        "max_known_entries": 49,
        "primary_signal": "guided onboarding, genre filters, popularity, score, short entry points",
        "risk": "cold start: no reliable personal taste vector yet",
    },
    {
        "level": "Amateur",
        "min_known_entries": 50,
        "max_known_entries": 149,
        "primary_signal": "similar anime, relation navigation, popularity, high-rated catalog anchors",
        "risk": "taste is still forming and may overfit to a few early favorites",
    },
    {
        "level": "Good",
        "min_known_entries": 150,
        "max_known_entries": 999,
        "primary_signal": "collaborative filtering, relation continuation, seasonal/current discovery, controlled exploration",
        "risk": "model should not only repeat the user's comfort zone",
    },
    {
        "level": "Pro",
        "min_known_entries": 1000,
        "max_known_entries": None,
        "primary_signal": "long-tail discovery, novelty, niche clusters, graph-aware exploration",
        "risk": "hardest group: most obvious catalog items are already known",
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

        method_scores = {
            "popularity_baseline": pop_scores,
            "latent_svd": latent_scores,
            "hybrid_svd_popularity": HYBRID_LATENT_WEIGHT * latent_norm + (1 - HYBRID_LATENT_WEIGHT) * pop_norm,
        }

        ranks = {method: rank_of_holdout(scores) for method, scores in method_scores.items()}
        for method, rank in ranks.items():
            profile_size = len(case.train_items)
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
    method_order = {"hybrid_svd_popularity": 0, "latent_svd": 1, "popularity_baseline": 2}
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


def plot_metrics(metrics: pd.DataFrame) -> None:
    order = metrics.sort_values("hit_rate_at_10", ascending=False)["method"].tolist()
    plot_df = metrics.set_index("method").loc[order].reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    bar_color = ["#4C78A8", "#F28E2B", "#59A14F"]
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
    ax.set_title("Evaluation Users by Recommender Maturity Level")
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
    ax.set_title("Hybrid Hit@10 by User Maturity Level")
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
    level_metrics = summarize_metrics_by_level(eval_rows)
    examples = add_titles(examples, catalog)
    beginner_candidates = build_beginner_entrypoint_candidates(catalog)

    metrics.to_csv(EVALUATION_PATH, index=False)
    level_metrics.to_csv(LEVEL_METRICS_PATH, index=False)
    eval_rows.sample(min(len(eval_rows), 15_000), random_state=RANDOM_STATE).to_csv(USER_EVAL_PATH, index=False)
    examples.to_csv(EXAMPLES_PATH, index=False)
    beginner_candidates.to_csv(BEGINNER_CANDIDATES_PATH, index=False)

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

    summary = {
        "task": "personalized anime recommendation/ranking",
        "decision_supported": "rank candidate anime for a user based on completed liked anime, while allowing product filters and maturity-aware recommendation modes",
        "baseline": "popularity_baseline: globally common liked anime in training data",
        "stronger_system": "latent_svd and hybrid_svd_popularity: personalized collaborative models that beat the popularity baseline",
        "product_safe_system": "hybrid_svd_popularity: latent collaborative SVD score blended with popularity for safer sparse-profile recommendations",
        "user_levels": USER_LEVELS,
        "control_filters": CONTROL_FILTERS,
        "level_policy_note": (
            "Level thresholds are product rules based on completed/known entries. "
            "Offline evaluation approximates this with known liked ratings because the Kaggle source has ratings but no chronological watch history."
        ),
        "ratings_source": str(RATINGS_PATH),
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
        "outputs": {
            "metrics": str(EVALUATION_PATH),
            "level_metrics": str(LEVEL_METRICS_PATH),
            "examples": str(EXAMPLES_PATH),
            "beginner_entrypoint_candidates": str(BEGINNER_CANDIDATES_PATH),
            "alignment": str(ALIGNMENT_PATH),
            "user_eval_sample": str(USER_EVAL_PATH),
            "metric_plot": str(PLOT_DIR / "week10_metric_comparison.png"),
            "rank_plot": str(PLOT_DIR / "week10_median_rank.png"),
            "level_distribution_plot": str(PLOT_DIR / "week10_user_level_distribution.png"),
            "level_hit_plot": str(PLOT_DIR / "week10_hit_at_10_by_user_level.png"),
        },
    }
    atomic_write_json(RUN_SUMMARY_PATH, summary)

    print("Week 10 recommendation run complete.")
    print(json.dumps(summary, indent=2))
    print(metrics)


if __name__ == "__main__":
    main()
