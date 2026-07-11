from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
BUILD_DIR = ROOT / "data" / "build"
ARTIFACT_TABLE_DIR = ROOT / "data" / "build" / "audits"
ARTIFACT_PLOT_DIR = ROOT / "artifacts" / "plots" / "eda"

INPUT_CSV = PROCESSED_DIR / "anime_dataset.csv"
USER_RATINGS_CSV = PROCESSED_DIR / "current_user_ratings.csv"
USER_PROFILE_CSV = PROCESSED_DIR / "current_user_profile_features.csv"
VOICE_ACTOR_INDEX_CSV = PROCESSED_DIR / "voice_actor_index.csv"
STAFF_INDEX_CSV = PROCESSED_DIR / "staff_index.csv"
DISCREPANCY_CSV = BUILD_DIR / "dataset_source_discrepancies.csv"
EDA_SUMMARY_FILE = BUILD_DIR / "dataset_eda_summary.json"
EMPTY_AUDIT_CSV = ARTIFACT_TABLE_DIR / "dataset_empty_field_audit.csv"
CHOICE_AUDIT_CSV = ARTIFACT_TABLE_DIR / "dataset_source_choice_audit.csv"
DISCREPANCY_EXAMPLES_CSV = ARTIFACT_TABLE_DIR / "dataset_source_discrepancy_examples.csv"
RECAP_AUDIT_CSV = ARTIFACT_TABLE_DIR / "dataset_recap_flag_audit.csv"
MISSINGNESS_PLOT = ARTIFACT_PLOT_DIR / "dataset_missingness_audit.png"
CHOICE_PLOT = ARTIFACT_PLOT_DIR / "dataset_choice_disagreements.png"
TYPE_PLOT = ARTIFACT_PLOT_DIR / "anime_type_counts.png"
YEAR_PLOT = ARTIFACT_PLOT_DIR / "aired_year_counts.png"
GENRE_PLOT = ARTIFACT_PLOT_DIR / "top_genres.png"
TAG_PLOT = ARTIFACT_PLOT_DIR / "top_tags.png"
EXPLICIT_TAG_PLOT = ARTIFACT_PLOT_DIR / "top_explicit_tags.png"
SCORE_MEMBER_PLOT = ARTIFACT_PLOT_DIR / "score_members_distributions.png"
RATING_ACTIVITY_PLOT = ARTIFACT_PLOT_DIR / "user_rating_activity.png"
PROFILE_BAND_PLOT = ARTIFACT_PLOT_DIR / "user_profile_bands.png"
GRAPH_EDGE_PLOT = ARTIFACT_PLOT_DIR / "edge_count_buckets.png"
PEOPLE_SIGNAL_PLOT = ARTIFACT_PLOT_DIR / "top_people_signals.png"

COMPARABLE_FIELDS = {
    "type": ["type_mal", "type_anilist"],
    "source": ["source_mal", "source_anilist"],
    "episodes": ["episodes_mal", "episodes_anilist", "episodes_anidb"],
    "duration": ["duration_mal", "duration_anilist", "duration_anidb"],
    "aired_year": ["aired_year_mal", "aired_year_anilist"],
    "aired_month": ["aired_month_mal", "aired_month_anilist"],
    "season": ["season_mal", "season_anilist", "season_from_month"],
    "demographics": ["demographics_mal", "demographics_anilist", "demographics_anidb"],
    "studios": ["studios_mal", "studios_anilist", "studios_anidb"],
    "characters": ["characters_jikan", "characters_anilist"],
    "voice_actors": ["voice_actors_jikan", "voice_actors_anilist"],
}

SOURCE_ALIASES = {
    "game": "Game",
    "video game": "Game",
    "video_game": "Game",
    "visual novel": "Visual Novel",
    "visual_novel": "Visual Novel",
    "light novel": "Light Novel",
    "light_novel": "Light Novel",
    "web manga": "Web Manga",
    "web_manga": "Web Manga",
    "web novel": "Web Novel",
    "web_novel": "Web Novel",
    "4-koma manga": "4-koma Manga",
    "4_koma_manga": "4-koma Manga",
    "mixed media": "Mixed Media",
    "mixed_media": "Mixed Media",
    "picture book": "Picture Book",
    "picture_book": "Picture Book",
    "other": "Other",
    "original": "Original",
    "manga": "Manga",
    "novel": "Novel",
    "music": "Music",
    "radio": "Radio",
    "unknown": "",
}

DEMOGRAPHIC_NAMES = {
    "18 restricted": "18+",
    "18+": "18+",
    "josei": "Josei",
    "kodomo": "Kodomo",
    "kids": "Kodomo",
    "mina": "Kodomo",
    "seinen": "Seinen",
    "shoujo": "Shoujo",
    "shojo": "Shoujo",
    "shounen": "Shounen",
    "shonen": "Shounen",
}

IMPORTANT_FIELDS = [
    "mal_id",
    "anilist_id",
    "anidb_id",
    "title",
    "type",
    "source",
    "episodes",
    "duration",
    "total_watch_minutes",
    "score",
    "members",
    "aired_year",
    "aired_month",
    "season",
    "genres",
    "tags",
    "demographics",
    "studios",
    "characters",
    "voice_actors",
    "relations",
    "recommendations",
]


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    tmp.replace(path)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def normalize_source_label(value: Any) -> str:
    if is_missing(value):
        return ""
    text = re.sub(r"\s+", " ", str(value).replace("_", " ").strip()).casefold()
    return SOURCE_ALIASES.get(text, str(value).replace("_", " ").title())


def normalize_studio_token(value: str) -> str:
    text = re.sub(r"[^\w\s]", " ", value.casefold())
    text = re.sub(r"\s+", " ", text).strip()
    if text.startswith("studio "):
        text = text.removeprefix("studio ").strip()
    return text


def split_multi_value(value: Any) -> list[str]:
    if is_missing(value):
        return []
    return [part.strip() for part in re.split(r"[|;]", str(value)) if part.strip()]


def normalize_demographics(value: Any) -> str:
    if is_missing(value):
        return ""
    values = []
    seen = set()
    for part in split_multi_value(value):
        clean = DEMOGRAPHIC_NAMES.get(part.casefold(), part)
        key = clean.casefold()
        if key not in seen:
            seen.add(key)
            values.append(clean)
    return "|".join(values)


def choice_key(value: Any, field: str | None = None) -> str | None:
    if is_missing(value):
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if value == 0:
            return None
        return str(int(value)) if float(value).is_integer() else f"{float(value):.4f}"
    text = str(value).strip()
    if not text:
        return None

    if field == "type":
        normalized = text.casefold()
        if normalized in {"special", "tv special"}:
            return "special"
        return normalized

    if field == "source":
        normalized = normalize_source_label(text)
        return normalized.casefold() if normalized else None

    if field == "studios":
        parts = sorted(normalize_studio_token(part) for part in split_multi_value(text))
        parts = [part for part in parts if part]
        return "|".join(parts) if parts else None

    if field == "demographics":
        parts = sorted(part.casefold() for part in split_multi_value(normalize_demographics(text)))
        return "|".join(parts) if parts else None

    if "|" in text or ";" in text:
        return "|".join(sorted(part.strip().casefold() for part in split_multi_value(text)))
    return re.sub(r"\s+", " ", text).casefold()


def pipe_count(value: Any) -> int:
    if is_missing(value):
        return 0
    return len([part for part in str(value).split("|") if part.strip()])


def explode_pipe_counts(series: pd.Series) -> pd.DataFrame:
    counter: dict[str, int] = {}
    for value in series.dropna():
        for label in split_multi_value(value):
            counter[label] = counter.get(label, 0) + 1
    return (
        pd.DataFrame(counter.items(), columns=["label", "count"])
        .sort_values(["count", "label"], ascending=[False, True])
        .reset_index(drop=True)
    )


def profile_band(scored_count: int) -> str:
    if scored_count <= 0:
        return "Cold Starter"
    if scored_count < 50:
        return "Beginner"
    if scored_count < 200:
        return "Casual"
    if scored_count < 750:
        return "Fan"
    return "Veteran"


def save_bar(
    data: pd.DataFrame,
    label_col: str,
    value_col: str,
    title: str,
    output: Path,
    *,
    top_n: int = 20,
    color: str = "#4c78a8",
) -> None:
    ARTIFACT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    if data.empty:
        return
    plot_df = data.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(11, max(5, 0.32 * len(plot_df))))
    ax.barh(plot_df[label_col].astype(str), plot_df[value_col], color=color)
    for index, value in enumerate(plot_df[value_col]):
        ax.text(value, index, f" {int(value):,}", va="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel(value_col.replace("_", " ").title())
    fig.tight_layout()
    fig.savefig(output, dpi=170)
    plt.close(fig)


def summarize_unique_values(field: str, values: set[str], max_items: int = 25, max_chars: int = 800) -> str:
    if field in {"characters", "voice_actors"}:
        return f"{len(values):,} unique entity sets; see discrepancy examples for popular rows"
    text = "|".join(sorted(values)[:max_items])
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


def empty_field_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    fields = [field for field in IMPORTANT_FIELDS if field in df.columns]
    for field in fields:
        missing_mask = df[field].apply(is_missing)
        zero_mask = pd.Series(False, index=df.index)
        if pd.api.types.is_numeric_dtype(df[field]):
            zero_mask = df[field].fillna(-1).eq(0)
        example_columns = ["mal_id", "title"] + (["popularity"] if "popularity" in df.columns else [])
        examples = df.loc[missing_mask, example_columns]
        if "popularity" in examples.columns:
            examples = examples.sort_values("popularity", na_position="last")
        examples = examples.head(5)
        example_text = " | ".join(f"{int(row.mal_id)}:{row.title}" for row in examples.itertuples(index=False))
        rows.append(
            {
                "field": field,
                "missing_count": int(missing_mask.sum()),
                "missing_pct": round(float(missing_mask.mean() * 100), 3),
                "zero_count": int(zero_mask.sum()),
                "non_empty_count": int((~missing_mask).sum()),
                "example_missing_titles": example_text,
            }
        )
    return pd.DataFrame(rows).sort_values(["missing_count", "field"], ascending=[False, True])


def source_choice_audit(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for field, source_columns in COMPARABLE_FIELDS.items():
        if field not in df.columns:
            continue
        available_sources = [column for column in source_columns if column in df.columns]
        if not available_sources:
            continue

        source_wins = {column: 0 for column in available_sources}
        unique_values: set[str] = set()
        consensus_rows = 0
        no_consensus_rows = 0
        any_source_rows = 0
        selected_missing_rows = 0
        disagreement_rows = 0

        for _, row in df.iterrows():
            selected_key = choice_key(row.get(field), field)
            keys = [(column, choice_key(row.get(column), field)) for column in available_sources]
            keys = [(column, key) for column, key in keys if key is not None]
            unique_values.update(key for _, key in keys)
            if keys:
                any_source_rows += 1
            if selected_key is None:
                selected_missing_rows += 1

            counts: dict[str, int] = {}
            for _, key in keys:
                counts[key] = counts.get(key, 0) + 1
            if counts and max(counts.values()) >= 2:
                consensus_rows += 1
            elif len(keys) >= 2:
                no_consensus_rows += 1

            unique_keys = {key for _, key in keys}
            if len(unique_keys) > 1:
                disagreement_rows += 1

            for column, key in keys:
                if key == selected_key:
                    source_wins[column] += 1
                    break

        rows.append(
            {
                "field": field,
                "source_columns": "|".join(available_sources),
                "rows": int(len(df)),
                "rows_with_any_source": any_source_rows,
                "selected_missing_rows": selected_missing_rows,
                "consensus_rows": consensus_rows,
                "no_consensus_rows": no_consensus_rows,
                "source_disagreement_rows": disagreement_rows,
                "selected_source_counts": "|".join(f"{column}:{count}" for column, count in source_wins.items()),
                "unique_normalized_values": summarize_unique_values(field, unique_values),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "field",
                "source_columns",
                "rows",
                "rows_with_any_source",
                "selected_missing_rows",
                "consensus_rows",
                "no_consensus_rows",
                "source_disagreement_rows",
                "selected_source_counts",
                "unique_normalized_values",
            ]
        )
    return pd.DataFrame(rows).sort_values(["source_disagreement_rows", "field"], ascending=[False, True])


def discrepancy_examples(discrepancy_path: Path) -> pd.DataFrame:
    if not discrepancy_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(discrepancy_path)
    if df.empty:
        return df
    if "popularity" in df.columns:
        df = df.sort_values(["field", "popularity"], na_position="last")
    keep = [
        "field",
        "mal_id",
        "title",
        "popularity",
        "selected_value",
        "unique_raw_values",
        "unique_normalized_values",
        "mal_value",
        "anilist_value",
        "anidb_value",
        "derived_value",
    ]
    keep = [column for column in keep if column in df.columns]
    return df[keep].head(250)


def recap_flag_audit(df: pd.DataFrame) -> pd.DataFrame:
    if "is_recap_like" not in df.columns:
        return pd.DataFrame()
    mask = df["is_recap_like"].astype(str).str.lower().isin({"true", "1"})
    if not mask.any():
        return pd.DataFrame()
    keep = [
        "mal_id",
        "title",
        "type",
        "episodes",
        "duration",
        "total_watch_minutes",
        "members",
        "favorites",
        "recap_reason",
        "recap_action",
        "relations",
        "genres",
        "tags",
    ]
    keep = [column for column in keep if column in df.columns]
    return df.loc[mask, keep].sort_values(["members", "favorites"], ascending=[False, False], na_position="last")


def plot_missingness(empty_df: pd.DataFrame) -> None:
    ARTIFACT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = empty_df.head(18).sort_values("missing_count")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(plot_df["field"], plot_df["missing_count"], color="#4c78a8")
    for index, value in enumerate(plot_df["missing_count"]):
        ax.text(value, index, f" {value:,}", va="center", fontsize=9)
    ax.set_title("Fields With the Most Missing Values")
    ax.set_xlabel("Missing rows")
    fig.tight_layout()
    fig.savefig(MISSINGNESS_PLOT, dpi=160)
    plt.close(fig)


def plot_choice_disagreements(choice_df: pd.DataFrame) -> None:
    ARTIFACT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    plot_df = choice_df.head(14).sort_values("source_disagreement_rows")
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.barh(plot_df["field"], plot_df["source_disagreement_rows"], color="#b2795f")
    for index, value in enumerate(plot_df["source_disagreement_rows"]):
        ax.text(value, index, f" {value:,}", va="center", fontsize=9)
    ax.set_title("Source Choice Disagreements")
    ax.set_xlabel("Rows where available sources disagree")
    fig.tight_layout()
    fig.savefig(CHOICE_PLOT, dpi=160)
    plt.close(fig)


def relation_count(value: Any) -> int:
    return pipe_count(value)


def recommendation_weight_sum(value: Any) -> int:
    total = 0
    for edge in split_multi_value(value):
        if ":" not in edge:
            continue
        _, weight = edge.split(":", 1)
        try:
            total += int(float(weight))
        except ValueError:
            total += 1
    return total


def save_catalog_plots(df: pd.DataFrame) -> dict[str, Any]:
    type_counts = df["type"].fillna("Missing").value_counts().rename_axis("label").reset_index(name="count")
    save_bar(type_counts, "label", "count", "Anime Type Counts", TYPE_PLOT, color="#59a14f")

    year_counts = (
        pd.to_numeric(df["aired_year"], errors="coerce")
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename_axis("year")
        .reset_index(name="count")
    )
    if not year_counts.empty:
        ARTIFACT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(year_counts["year"], year_counts["count"], color="#4c78a8", linewidth=2)
        ax.fill_between(year_counts["year"], year_counts["count"], color="#4c78a8", alpha=0.18)
        ax.set_title("Anime Entries by Aired Year")
        ax.set_xlabel("Aired year")
        ax.set_ylabel("Entries")
        fig.tight_layout()
        fig.savefig(YEAR_PLOT, dpi=170)
        plt.close(fig)

    genre_counts = explode_pipe_counts(df["genres"])
    tag_counts = explode_pipe_counts(df["tags"])
    explicit_counts = explode_pipe_counts(df["explicit_tags"]) if "explicit_tags" in df else pd.DataFrame(columns=["label", "count"])
    save_bar(genre_counts, "label", "count", "Top Genres", GENRE_PLOT, color="#4e79a7")
    save_bar(tag_counts, "label", "count", "Top Tags", TAG_PLOT, color="#f28e2b")
    save_bar(explicit_counts, "label", "count", "Top Explicit Tags", EXPLICIT_TAG_PLOT, color="#e15759")

    ARTIFACT_PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    df["score"].dropna().hist(ax=axes[0], bins=35, color="#4e79a7")
    axes[0].set_title("Score Distribution")
    axes[0].set_xlabel("MAL score")
    axes[0].set_ylabel("Entries")
    pd.Series(pd.to_numeric(df["members"], errors="coerce").fillna(0).pipe(lambda s: s[s > 0])).apply(lambda x: x).plot(
        kind="hist",
        bins=45,
        logx=True,
        ax=axes[1],
        color="#59a14f",
    )
    axes[1].set_title("Members Distribution")
    axes[1].set_xlabel("Members, log scale")
    fig.tight_layout()
    fig.savefig(SCORE_MEMBER_PLOT, dpi=170)
    plt.close(fig)

    graph_counts = pd.DataFrame(
        {
            "label": ["relation_edges", "recommendation_edges"],
            "count": [
                int(df["relations"].apply(relation_count).sum()) if "relations" in df else 0,
                int(df["recommendations"].apply(relation_count).sum()) if "recommendations" in df else 0,
            ],
        }
    )
    save_bar(graph_counts, "label", "count", "Catalog Graph Edge Counts", GRAPH_EDGE_PLOT, top_n=2, color="#af7aa1")

    return {
        "top_genres": genre_counts.head(15).to_dict(orient="records"),
        "top_tags": tag_counts.head(15).to_dict(orient="records"),
        "top_explicit_tags": explicit_counts.head(15).to_dict(orient="records"),
        "relation_edges": int(graph_counts.loc[graph_counts["label"].eq("relation_edges"), "count"].iloc[0]),
        "recommendation_edges": int(graph_counts.loc[graph_counts["label"].eq("recommendation_edges"), "count"].iloc[0]),
    }


def save_user_and_people_plots() -> dict[str, Any]:
    summary: dict[str, Any] = {}
    if USER_RATINGS_CSV.exists():
        ratings = pd.read_csv(USER_RATINGS_CSV, usecols=["userID", "animeID", "rating", "status"])
        status_counts = ratings["status"].fillna("missing").value_counts().rename_axis("label").reset_index(name="count")
        save_bar(status_counts, "label", "count", "Current User Rating Status Counts", RATING_ACTIVITY_PLOT, color="#edc948")
        summary["current_user_ratings_rows"] = int(len(ratings))
        summary["current_user_count"] = int(ratings["userID"].nunique())
        summary["rated_anime_count"] = int(ratings["animeID"].nunique())
    if USER_PROFILE_CSV.exists():
        profiles = pd.read_csv(USER_PROFILE_CSV)
        profiles["profile_band"] = pd.to_numeric(profiles["scored_count"], errors="coerce").fillna(0).astype(int).apply(profile_band)
        band_order = ["Cold Starter", "Beginner", "Casual", "Fan", "Veteran"]
        band_counts = profiles["profile_band"].value_counts().reindex(band_order, fill_value=0).rename_axis("label").reset_index(name="count")
        save_bar(band_counts, "label", "count", "Collected User Profile Bands", PROFILE_BAND_PLOT, top_n=10, color="#76b7b2")
        summary["profile_rows"] = int(len(profiles))
        summary["profile_bands"] = band_counts.to_dict(orient="records")
    people_frames = []
    if VOICE_ACTOR_INDEX_CSV.exists():
        va = pd.read_csv(VOICE_ACTOR_INDEX_CSV)
        if {"voice_actor_name", "voice_actor_favorites"}.issubset(va.columns):
            people_frames.append(
                va[["voice_actor_name", "voice_actor_favorites"]]
                .rename(columns={"voice_actor_name": "label", "voice_actor_favorites": "count"})
                .assign(kind="voice_actor")
            )
    if STAFF_INDEX_CSV.exists():
        staff = pd.read_csv(STAFF_INDEX_CSV)
        if {"staff_name", "staff_favorites"}.issubset(staff.columns):
            people_frames.append(
                staff[["staff_name", "staff_favorites"]]
                .rename(columns={"staff_name": "label", "staff_favorites": "count"})
                .assign(kind="staff")
            )
    if people_frames:
        people = pd.concat(people_frames, ignore_index=True)
        people["label"] = people["label"].astype(str).str.replace(r"\s+", " ", regex=True).str.strip()
        people["count"] = pd.to_numeric(people["count"], errors="coerce").fillna(0).astype(int)
        people = (
            people.groupby("label", as_index=False)
            .agg(count=("count", "max"), kind=("kind", lambda values: "|".join(sorted(set(map(str, values))))))
            .sort_values("count", ascending=False)
        )
        save_bar(people[["label", "count"]], "label", "count", "Top People Signals by Favorites", PEOPLE_SIGNAL_PLOT, color="#b07aa1")
        summary["top_people_signals"] = people.head(20).to_dict(orient="records")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit empties and source choices in the raw-built anime dataset.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input_csv}. Run src/02_build_anime_dataset.py first.")

    ARTIFACT_TABLE_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)

    empty_df = empty_field_audit(df)
    choice_df = source_choice_audit(df)
    examples_df = discrepancy_examples(DISCREPANCY_CSV)
    recap_df = recap_flag_audit(df)

    empty_df.to_csv(EMPTY_AUDIT_CSV, index=False)
    choice_df.to_csv(CHOICE_AUDIT_CSV, index=False)
    examples_df.to_csv(DISCREPANCY_EXAMPLES_CSV, index=False)
    recap_df.to_csv(RECAP_AUDIT_CSV, index=False)
    plot_missingness(empty_df)
    plot_choice_disagreements(choice_df)
    catalog_plot_summary = save_catalog_plots(df)
    user_people_summary = save_user_and_people_plots()

    summary = {
        "updated_at": now_iso(),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "empty_audit_csv": str(EMPTY_AUDIT_CSV),
        "choice_audit_csv": str(CHOICE_AUDIT_CSV),
        "discrepancy_examples_csv": str(DISCREPANCY_EXAMPLES_CSV),
        "recap_audit_csv": str(RECAP_AUDIT_CSV),
        "missingness_plot": str(MISSINGNESS_PLOT),
        "choice_plot": str(CHOICE_PLOT),
        "largest_missing_fields": empty_df.head(8).to_dict(orient="records"),
        "largest_choice_disagreements": choice_df.head(8).to_dict(orient="records"),
        "recap_like_rows_kept": int(len(recap_df)),
        "catalog_plots": {
            "type_counts": str(TYPE_PLOT),
            "aired_year_counts": str(YEAR_PLOT),
            "top_genres": str(GENRE_PLOT),
            "top_tags": str(TAG_PLOT),
            "top_explicit_tags": str(EXPLICIT_TAG_PLOT),
            "score_members": str(SCORE_MEMBER_PLOT),
            "graph_edges": str(GRAPH_EDGE_PLOT),
        },
        "user_and_people_plots": {
            "rating_activity": str(RATING_ACTIVITY_PLOT),
            "profile_bands": str(PROFILE_BAND_PLOT),
            "people_signals": str(PEOPLE_SIGNAL_PLOT),
        },
        **catalog_plot_summary,
        **user_people_summary,
    }
    atomic_write_json(EDA_SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
