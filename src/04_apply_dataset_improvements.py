from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

try:
    from anidb_metadata_utils import extract_anidb_payload, atomic_write_json
except ImportError:  # Imported from a notebook with project root on sys.path.
    from src.anidb_metadata_utils import extract_anidb_payload, atomic_write_json

try:
    import requests
except ImportError:  # pragma: no cover - live AniDB fill is optional
    requests = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "anime_dataset.csv"
DATASET_JSON = ROOT / "data" / "processed" / "anime_dataset.json"
ANIDB_CACHE_FILE = ROOT / "data" / "caches" / "anidb_metadata_cache.json"

ANIDB_CLIENT = "matuki"
ANIDB_CLIENTVER = 3
ANIDB_REQUEST_DELAY_SECONDS = 4
ANIDB_REQUEST_JITTER_SECONDS = 0.5

DEMOGRAPHIC_BRANCH = "target audience"
IGNORED_ANIDB_BRANCHES = {
    "maintenance tags",
    "origin",
    "original work",
    "technical aspects",
    "setting",
}
DROPPED_ANIDB_TAGS = {"warning"}
ALWAYS_EXPLICIT_BRANCHES = {"pornography", "sexual abuse", "rape"}
RATING_GATED_EXPLICIT_BRANCHES = {
    "fetishes",
    "ecchi",
    "incest",
    "brainwashing",
    "harem",
    "content indicators",
}
RATING_GATED_EXPLICIT_TAGS = {
    "nudity",
    "sex",
    "animal abuse",
    "gore",
    "mutilation",
    "ecchi",
    "incest",
    "brainwashing",
    "harem",
}
EXPLICIT_RATINGS = {"R+ - Mild Nudity", "Rx - Hentai"}
DEMOGRAPHIC_GENRE_INFERENCE = {
    "hentai": "18+",
    "erotica": "18+",
}
DEMOGRAPHIC_TAG_INFERENCE = {
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

ORIGIN_TAGS_FOR_STUDIOS = {
    "American-Japanese co-production",
    "Chinese production",
    "development hell",
    "fan-made",
    "French-Chinese co-production",
    "French-Japanese co-production",
    "Indo-Japanese co-production",
    "Italian-Japanese co-production",
    "Japanese production",
    "Korean-Japanese co-production",
    "North Korean production",
    "Polish-Japanese co-production",
    "remake",
    "Russian-Japanese co-production",
    "Saudi Arabian-Japanese co-production",
    "Singaporean production",
    "Sino-Japanese co-production",
    "South Korean production",
    "Taiwanese production",
    "Thai production",
}

GENRE_ALIASES = {
    "science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "daily life": "Slice of Life",
    "slice of life": "Slice of Life",
    "shounen ai": "Boys Love",
    "boys love": "Boys Love",
    "shoujo ai": "Girls Love",
    "girls love": "Girls Love",
    "cooking": "Gourmet",
    "food": "Gourmet",
    "thriller": "Suspense",
    "suspense": "Suspense",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        if pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def is_missing_text(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def split_pipe_values(value: Any) -> list[str]:
    if is_missing_text(value):
        return []
    return [item.strip() for item in str(value).split("|") if item.strip()]


def merge_pipe_values(values: list[str]) -> str:
    seen: set[str] = set()
    merged: list[str] = []
    for value in values:
        clean = str(value).strip()
        key = clean.casefold()
        if clean and key not in seen:
            seen.add(key)
            merged.append(clean)
    return "|".join(merged)


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[\W_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_demographic_name(name: str) -> str | None:
    mapping = {
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
    return mapping.get(normalize_name(name))


def has_explicit_rating(rating: Any) -> bool:
    return str(rating or "").strip() in EXPLICIT_RATINGS


def infer_season(month: Any) -> str | None:
    month_int = parse_int(month, default=None)
    if month_int is None:
        return None
    if month_int in {1, 2, 3}:
        return "winter"
    if month_int in {4, 5, 6}:
        return "spring"
    if month_int in {7, 8, 9}:
        return "summer"
    if month_int in {10, 11, 12}:
        return "fall"
    return None


def parse_duration_minutes(value: Any) -> float | None:
    if is_missing_text(value):
        return None

    if isinstance(value, (int, float)) and not pd.isna(value):
        numeric = float(value)
        return numeric if numeric > 0 else None

    text = str(value).strip().lower()
    if text == "unknown":
        return None

    numeric_text = re.fullmatch(r"\d+(?:\.\d+)?", text)
    if numeric_text:
        numeric = float(text)
        return numeric if numeric > 0 else None

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

    if total <= 0:
        return None
    return round(total, 3)


def normalize_demographic_values(value: Any) -> str:
    normalized = []
    for item in split_pipe_values(value):
        demographic = normalize_demographic_name(item)
        normalized.append(demographic or item)
    return merge_pipe_values(normalized)


def infer_demographics_from_row(row: pd.Series) -> str:
    inferred = []
    rating = str(row.get("rating") or "").strip()

    if rating in {"Rx - Hentai", "R+ - Mild Nudity"}:
        inferred.append("18+")
    if rating == "PG - Children":
        inferred.append("Kodomo")

    for genre in split_pipe_values(row.get("genres")):
        demographic = DEMOGRAPHIC_GENRE_INFERENCE.get(normalize_name(genre))
        if demographic:
            inferred.append(demographic)

    for tag in split_pipe_values(row.get("tags")) + split_pipe_values(row.get("explicit_tags")):
        demographic = DEMOGRAPHIC_TAG_INFERENCE.get(normalize_name(tag))
        if demographic:
            inferred.append(demographic)

    return merge_pipe_values(inferred)


def should_have_explicit_tags(row: pd.Series) -> bool:
    if has_explicit_rating(row.get("rating")):
        return True
    genres = {normalize_name(item) for item in split_pipe_values(row.get("genres"))}
    return bool(genres & set(DEMOGRAPHIC_GENRE_INFERENCE))


def load_anidb_cache() -> dict[str, Any]:
    if not ANIDB_CACHE_FILE.exists():
        return {"updated_at": None, "items": {}}
    with ANIDB_CACHE_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("items", {})
    return payload


def save_anidb_cache(payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    atomic_write_json(ANIDB_CACHE_FILE, payload)


def tag_ancestor_names(tag_id: int, tags_by_id: dict[int, dict[str, Any]]) -> list[str]:
    ancestors: list[str] = []
    tag = tags_by_id.get(tag_id)
    parent_id = tag.get("parent_id") if tag else None
    seen: set[int] = set()

    while parent_id and parent_id not in seen:
        seen.add(parent_id)
        parent = tags_by_id.get(parent_id)
        if not parent:
            break
        ancestors.append(parent["name"])
        parent_id = parent.get("parent_id")

    return ancestors


def parse_anidb_tag_records(raw_tags: list[dict[str, Any]] | None, rating: Any = None) -> dict[str, str]:
    tags_by_id = {
        int(tag["id"]): {
            "id": int(tag["id"]),
            "parent_id": parse_int(tag.get("parent_id"), default=None),
            "name": str(tag.get("name", "")).strip(),
            "weight": parse_int(tag.get("weight"), default=0) or 0,
        }
        for tag in raw_tags or []
        if tag.get("id") is not None and tag.get("name")
    }

    parsed = {
        "tags": [],
        "tag_weights": [],
        "explicit_tags": [],
        "explicit_tag_weights": [],
        "demographics": [],
    }
    explicit_rating = has_explicit_rating(rating)

    for tag_id, tag in tags_by_id.items():
        name = tag["name"]
        name_key = normalize_name(name)
        parent_id = tag.get("parent_id")

        if not parent_id:
            continue
        if name_key in DROPPED_ANIDB_TAGS:
            continue

        ancestors = tag_ancestor_names(tag_id, tags_by_id)
        lineage_keys = {name_key} | {normalize_name(item) for item in ancestors}

        if lineage_keys & IGNORED_ANIDB_BRANCHES:
            continue

        if DEMOGRAPHIC_BRANCH in lineage_keys:
            if name_key != DEMOGRAPHIC_BRANCH:
                demographic = normalize_demographic_name(name)
                if demographic:
                    parsed["demographics"].append(demographic)
            continue

        is_always_explicit = bool(lineage_keys & ALWAYS_EXPLICIT_BRANCHES)
        is_rating_gated_explicit = explicit_rating and bool(
            (lineage_keys & RATING_GATED_EXPLICIT_BRANCHES)
            or (name_key in RATING_GATED_EXPLICIT_TAGS)
        )

        if is_always_explicit or is_rating_gated_explicit:
            parsed["explicit_tags"].append(name)
            parsed["explicit_tag_weights"].append(f"{name}:{tag['weight']}")
        else:
            parsed["tags"].append(name)
            parsed["tag_weights"].append(f"{name}:{tag['weight']}")

    return {key: merge_pipe_values(values) for key, values in parsed.items()}


def origin_tags_from_payload(payload: dict[str, Any] | None) -> str:
    if not payload:
        return ""
    animation_work_studios = payload.get("animation_work_studios") or []
    if animation_work_studios:
        return merge_pipe_values([str(item) for item in animation_work_studios])

    allowed = {normalize_name(item): item for item in ORIGIN_TAGS_FOR_STUDIOS}
    matches = []
    for tag in payload.get("raw_tags", []) or []:
        canonical = allowed.get(normalize_name(tag.get("name")))
        if canonical:
            matches.append(canonical)
    return merge_pipe_values(matches)


def episode_average_length_from_payload(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    summary = payload.get("episode_summary") or {}
    preferred_average = parse_duration_minutes(summary.get("preferred_average_length_minutes"))
    if preferred_average is not None:
        return preferred_average
    average = parse_duration_minutes(summary.get("average_length_minutes"))
    if average is not None:
        return average
    total = parse_duration_minutes(summary.get("total_length_minutes"))
    count = parse_int(summary.get("length_count"), default=None)
    if total is not None and count:
        return round(total / count, 3)
    return None


def episode_total_length_from_payload(payload: dict[str, Any] | None) -> float | None:
    if not payload:
        return None
    summary = payload.get("episode_summary") or {}
    preferred_total = parse_duration_minutes(summary.get("preferred_total_length_minutes"))
    if preferred_total is not None:
        return preferred_total
    return parse_duration_minutes(summary.get("total_length_minutes"))


def preferred_episode_count_from_payload(payload: dict[str, Any] | None) -> int | None:
    if not payload:
        return None
    summary = payload.get("episode_summary") or {}
    preferred_count = parse_int(summary.get("preferred_episode_count"), default=None)
    if preferred_count and preferred_count > 0:
        return preferred_count
    regular_count = parse_int(summary.get("regular_episode_count"), default=None)
    if regular_count and regular_count > 0:
        return regular_count
    return None


def merge_recommendation_edges(existing: Any, new_edges: list[str]) -> str:
    merged: dict[str, int] = {}
    for edge in split_pipe_values(existing):
        if ":" in edge:
            target, weight = edge.split(":", 1)
        else:
            target, weight = edge, "1"
        target = target.strip()
        if not target:
            continue
        merged[target] = max(merged.get(target, 0), parse_int(weight, default=1) or 1)

    for edge in new_edges:
        if ":" in edge:
            target, weight = edge.split(":", 1)
        else:
            target, weight = edge, "1"
        target = target.strip()
        if not target:
            continue
        merged[target] = max(merged.get(target, 0), parse_int(weight, default=1) or 1)

    return "|".join(f"{target}:{weight}" for target, weight in merged.items())


def similar_anime_edges_from_payload(
    payload: dict[str, Any] | None,
    anidb_to_mal: dict[int, int],
) -> list[str]:
    if not payload:
        return []

    edges = []
    for item in payload.get("similar_anime", []) or []:
        anidb_id = parse_int(
            item.get("anidb_id") or item.get("aid") or item.get("id"),
            default=None,
        )
        if anidb_id is None:
            continue
        mal_id = anidb_to_mal.get(anidb_id)
        if mal_id is None:
            continue

        weight = (
            parse_int(item.get("approval"), default=None)
            or parse_int(item.get("total"), default=None)
            or 1
        )
        edges.append(f"{mal_id}:{max(1, int(weight))}")
    return edges


def fetch_anidb_live_payload(anidb_id: int) -> dict[str, Any] | None:
    if requests is None:
        raise RuntimeError("requests is not installed; live AniDB fill is unavailable")

    time.sleep(ANIDB_REQUEST_DELAY_SECONDS + random.uniform(0, ANIDB_REQUEST_JITTER_SECONDS))
    url = (
        "http://api.anidb.net:9001/httpapi"
        f"?request=anime&client={ANIDB_CLIENT}&clientver={ANIDB_CLIENTVER}"
        f"&protover=1&aid={int(anidb_id)}"
    )
    response = requests.get(url, timeout=20)
    if response.status_code != 200:
        return None

    root = ET.fromstring(response.content)
    if root.tag.lower() == "error":
        return None
    payload = extract_anidb_payload(root)
    payload["source_url"] = url
    payload["source_downloaded_at"] = now_iso()
    return payload


def update_cache_with_live_payloads(
    cache_payload: dict[str, Any],
    anidb_ids: list[int],
    limit: int | None = None,
    label: str = "anidb_live_repair",
    show_progress: bool = True,
) -> int:
    updated = 0
    selected_ids = anidb_ids if limit is None else anidb_ids[:limit]
    total = len(selected_ids)

    for position, anidb_id in enumerate(selected_ids, start=1):
        if show_progress:
            print(
                f"[{position}/{total}] {label} | AniDB {int(anidb_id)} | request_start",
                flush=True,
            )

        payload = fetch_anidb_live_payload(anidb_id)
        if payload is None:
            if show_progress:
                print(
                    f"[{position}/{total}] {label} | AniDB {int(anidb_id)} | no_update",
                    flush=True,
                )
            continue

        cache_payload["items"][str(int(anidb_id))] = {
            "cached_at": now_iso(),
            "source": "live_http",
            **payload,
        }
        updated += 1
        save_anidb_cache(cache_payload)

        if show_progress:
            print(
                f"[{position}/{total}] {label} | AniDB {int(anidb_id)} | "
                f"saved | episode_count={payload.get('episode_count')} | "
                f"raw_tags={len(payload.get('raw_tags', []) or [])}",
                flush=True,
            )

    return updated


def episode_live_candidate_frame(df: pd.DataFrame, cache_payload: dict[str, Any]) -> pd.DataFrame:
    items = cache_payload.get("items", {})
    episode_gap = df["episodes"].isna() | (
        pd.to_numeric(df["episodes"], errors="coerce").fillna(-1) == 0
    )
    candidates = df.loc[
        episode_gap & df["anidb_id"].notna(),
        ["mal_id", "anidb_id", "title", "type", "status", "episodes", "popularity"],
    ].copy()
    candidates["cached_episode_count"] = candidates["anidb_id"].apply(
        lambda value: items.get(str(int(value)), {}).get("episode_count")
    )
    cached_episode_count = pd.to_numeric(candidates["cached_episode_count"], errors="coerce")
    candidates = candidates.loc[cached_episode_count.isna() | (cached_episode_count <= 0)].copy()
    return candidates.sort_values("popularity", na_position="last")


def metadata_live_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    tag_gap = df["tags"].apply(is_missing_text)
    demographic_gap = df["demographics"].apply(is_missing_text)
    explicit_gap = df["explicit_tags"].apply(is_missing_text) & df.apply(
        should_have_explicit_tags,
        axis=1,
    )
    duration_gap = df["duration"].apply(parse_duration_minutes).isna()
    runtime_gap = (
        ~duration_gap
        & (
            df.get("total_watch_minutes", pd.Series(index=df.index, dtype=float)).isna()
            if "total_watch_minutes" in df.columns
            else True
        )
    )
    studio_gap = df["studios"].apply(is_missing_text)

    missing_counts = {
        "needs_explicit_tags": int(explicit_gap.sum()),
        "needs_duration": int(duration_gap.sum()),
        "needs_total_watch_minutes": int(runtime_gap.sum()) if not isinstance(runtime_gap, bool) else 0,
        "needs_studios": int(studio_gap.sum()),
        "needs_tags": int(tag_gap.sum()),
        "needs_demographics": int(demographic_gap.sum()),
    }

    candidates = df.loc[
        (tag_gap | demographic_gap | explicit_gap | duration_gap | runtime_gap | studio_gap)
        & df["anidb_id"].notna(),
        [
            "mal_id",
            "anidb_id",
            "title",
            "type",
            "rating",
            "popularity",
            "tags",
            "explicit_tags",
            "demographics",
            "duration",
            "total_watch_minutes",
            "studios",
        ],
    ].copy()
    candidates["needs_tags"] = tag_gap.loc[candidates.index].to_numpy()
    candidates["needs_explicit_tags"] = explicit_gap.loc[candidates.index].to_numpy()
    candidates["needs_demographics"] = demographic_gap.loc[candidates.index].to_numpy()
    candidates["needs_duration"] = duration_gap.loc[candidates.index].to_numpy()
    candidates["needs_total_watch_minutes"] = (
        runtime_gap.loc[candidates.index].to_numpy()
        if not isinstance(runtime_gap, bool)
        else False
    )
    candidates["needs_studios"] = studio_gap.loc[candidates.index].to_numpy()

    need_columns = [
        "needs_explicit_tags",
        "needs_duration",
        "needs_total_watch_minutes",
        "needs_studios",
        "needs_tags",
        "needs_demographics",
    ]
    candidates["scarcity_priority"] = candidates[need_columns].apply(
        lambda row: min(
            [
                missing_counts[column]
                for column, needs_value in row.items()
                if bool(needs_value) and missing_counts[column] > 0
            ]
            or [999999]
        ),
        axis=1,
    )
    candidates["need_count"] = candidates[need_columns].sum(axis=1)
    return candidates.sort_values(
        ["scarcity_priority", "popularity"],
        na_position="last",
    )


def duration_live_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    duration_gap = df["duration"].apply(parse_duration_minutes).isna()
    runtime_gap = (
        df["total_watch_minutes"].isna()
        if "total_watch_minutes" in df.columns
        else pd.Series(True, index=df.index)
    )
    candidates = df.loc[
        (duration_gap | runtime_gap) & df["anidb_id"].notna(),
        ["mal_id", "anidb_id", "title", "type", "status", "episodes", "duration", "total_watch_minutes", "popularity"],
    ].copy()
    candidates["needs_duration"] = duration_gap.loc[candidates.index].to_numpy()
    candidates["needs_total_watch_minutes"] = runtime_gap.loc[candidates.index].to_numpy()
    return candidates.sort_values("popularity", na_position="last")


def currently_airing_update_candidate_frame(df: pd.DataFrame) -> pd.DataFrame:
    status = df["status"].fillna("").astype(str).str.casefold()
    candidates = df.loc[
        status.eq("currently airing") & df["anidb_id"].notna(),
        ["mal_id", "anidb_id", "title", "type", "status", "episodes", "duration", "total_watch_minutes", "popularity"],
    ].copy()
    return candidates.sort_values("popularity", na_position="last")


def build_genre_lookup(df: pd.DataFrame) -> dict[str, str]:
    lookup = dict(GENRE_ALIASES)
    for value in df["genres"].dropna():
        for genre in split_pipe_values(value):
            lookup[normalize_name(genre)] = genre
    return lookup


def infer_genres_from_tags(row: pd.Series, genre_lookup: dict[str, str]) -> str:
    candidates = split_pipe_values(row.get("tags")) + split_pipe_values(row.get("explicit_tags"))
    inferred = []
    for tag in candidates:
        genre = genre_lookup.get(normalize_name(tag))
        if genre:
            inferred.append(genre)
    return merge_pipe_values(inferred)


def apply_improvements(df: pd.DataFrame, cache_payload: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    summary: dict[str, Any] = {
        "started_rows": int(len(df)),
        "dropped_missing_air_date": 0,
        "seasons_filled_from_aired_month": 0,
        "duration_values_parsed_to_minutes": 0,
        "duration_filled_from_anidb_episode_lengths": 0,
        "total_watch_minutes_filled": 0,
        "total_watch_minutes_filled_from_anidb_episode_lengths": 0,
        "episodes_filled_from_anidb_cache": 0,
        "genres_filled_from_tags": 0,
        "studios_filled_from_origin_tags": 0,
        "recommendations_augmented_from_anidb_similar_anime": 0,
        "tags_filled_from_anidb_cache": 0,
        "explicit_tags_filled_from_anidb_cache": 0,
        "demographics_filled_from_anidb_cache": 0,
        "demographics_normalized": 0,
        "demographics_inferred_from_rating_genres_tags": 0,
        "remaining_episode_live_candidates": [],
        "remaining_tag_demographic_or_explicit_live_candidates": [],
    }

    items = cache_payload.get("items", {})
    df = df.copy()
    anidb_to_mal = {
        int(row["anidb_id"]): int(row["mal_id"])
        for _, row in df[df["anidb_id"].notna()].iterrows()
    }

    missing_air_date = df["aired_year"].isna() & df["aired_month"].isna()
    summary["dropped_missing_air_date"] = int(missing_air_date.sum())
    df = df.loc[~missing_air_date].copy()

    if "season" in df.columns:
        inferred_season = df["aired_month"].apply(infer_season)
        missing_season = df["season"].apply(is_missing_text)
        fillable_season = missing_season & inferred_season.notna()
        df.loc[fillable_season, "season"] = inferred_season.loc[fillable_season]
        summary["seasons_filled_from_aired_month"] = int(fillable_season.sum())

    duration_minutes = df["duration"].apply(parse_duration_minutes)
    df["duration"] = duration_minutes

    missing_duration = df["duration"].isna()
    for idx, row in df.loc[missing_duration & df["anidb_id"].notna()].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        average_length = episode_average_length_from_payload(payload)
        if average_length is not None:
            df.at[idx, "duration"] = average_length
            summary["duration_filled_from_anidb_episode_lengths"] += 1

    summary["duration_values_parsed_to_minutes"] = int(pd.to_numeric(df["duration"], errors="coerce").notna().sum())

    episode_values = pd.to_numeric(df["episodes"], errors="coerce")
    df["total_watch_minutes"] = episode_values * pd.to_numeric(df["duration"], errors="coerce")

    missing_runtime = df["total_watch_minutes"].isna()
    for idx, row in df.loc[missing_runtime & df["anidb_id"].notna()].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        total_length = episode_total_length_from_payload(payload)
        if total_length is not None:
            df.at[idx, "total_watch_minutes"] = total_length
            summary["total_watch_minutes_filled_from_anidb_episode_lengths"] += 1

    summary["total_watch_minutes_filled"] = int(df["total_watch_minutes"].notna().sum())

    before_demographics = df["demographics"].copy()
    df["demographics"] = df["demographics"].apply(normalize_demographic_values)
    summary["demographics_normalized"] = int(
        (before_demographics.fillna("").astype(str) != df["demographics"].fillna("").astype(str)).sum()
    )

    episode_missing = df["episodes"].isna() | (pd.to_numeric(df["episodes"], errors="coerce").fillna(-1) == 0)
    for idx, row in df.loc[episode_missing].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        episode_count = parse_int((payload or {}).get("episode_count"), default=None)
        if not episode_count or episode_count <= 0:
            episode_count = preferred_episode_count_from_payload(payload)
        if episode_count and episode_count > 0:
            df.at[idx, "episodes"] = episode_count
            summary["episodes_filled_from_anidb_cache"] += 1

    still_episode_missing = df["episodes"].isna() | (pd.to_numeric(df["episodes"], errors="coerce").fillna(-1) == 0)
    episode_live_candidates = df.loc[
        still_episode_missing & df["anidb_id"].notna(),
        ["mal_id", "anidb_id", "title", "status", "episodes"],
    ].copy()
    summary["remaining_episode_live_candidates"] = episode_live_candidates.to_dict("records")

    genre_lookup = build_genre_lookup(df)
    genre_missing = df["genres"].apply(is_missing_text)
    for idx, row in df.loc[genre_missing].iterrows():
        inferred = infer_genres_from_tags(row, genre_lookup)
        if inferred:
            df.at[idx, "genres"] = inferred
            summary["genres_filled_from_tags"] += 1

    studio_missing = df["studios"].apply(is_missing_text)
    for idx, row in df.loc[studio_missing].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        origins = origin_tags_from_payload(payload)
        if origins:
            df.at[idx, "studios"] = origins
            summary["studios_filled_from_origin_tags"] += 1

    if "recommendations" not in df.columns:
        df["recommendations"] = ""
    for idx, row in df[df["anidb_id"].notna()].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        new_edges = similar_anime_edges_from_payload(payload, anidb_to_mal)
        if not new_edges:
            continue
        before = split_pipe_values(row.get("recommendations"))
        merged = merge_recommendation_edges(row.get("recommendations"), new_edges)
        after = split_pipe_values(merged)
        if len(after) > len(before):
            df.at[idx, "recommendations"] = merged
            summary["recommendations_augmented_from_anidb_similar_anime"] += 1

    needs_tags = df["tags"].apply(is_missing_text)
    needs_demographics = df["demographics"].apply(is_missing_text)
    needs_explicit_tags = df["explicit_tags"].apply(is_missing_text) & df.apply(
        should_have_explicit_tags,
        axis=1,
    )
    for idx, row in df.loc[needs_tags | needs_demographics | needs_explicit_tags].iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        payload = items.get(str(anidb_id)) if anidb_id is not None else None
        if not payload:
            continue
        parsed = parse_anidb_tag_records(payload.get("raw_tags", []), rating=row.get("rating"))
        if needs_tags.loc[idx] and parsed.get("tags"):
            df.at[idx, "tags"] = parsed["tags"]
            df.at[idx, "tag_weights"] = parsed["tag_weights"]
            summary["tags_filled_from_anidb_cache"] += 1
        if needs_explicit_tags.loc[idx] and parsed.get("explicit_tags"):
            df.at[idx, "explicit_tags"] = parsed["explicit_tags"]
            df.at[idx, "explicit_tag_weights"] = parsed["explicit_tag_weights"]
            summary["explicit_tags_filled_from_anidb_cache"] += 1
        if needs_demographics.loc[idx] and parsed.get("demographics"):
            df.at[idx, "demographics"] = parsed["demographics"]
            summary["demographics_filled_from_anidb_cache"] += 1

    needs_demographics = df["demographics"].apply(is_missing_text)
    for idx, row in df.loc[needs_demographics].iterrows():
        inferred = infer_demographics_from_row(row)
        if inferred:
            df.at[idx, "demographics"] = inferred
            summary["demographics_inferred_from_rating_genres_tags"] += 1

    remaining_tag_or_demo = df.loc[
        (
            df["tags"].apply(is_missing_text)
            | df["demographics"].apply(is_missing_text)
            | (df["explicit_tags"].apply(is_missing_text) & df.apply(should_have_explicit_tags, axis=1))
        ) & df["anidb_id"].notna(),
        ["mal_id", "anidb_id", "title", "tags", "explicit_tags", "demographics"],
    ].copy()
    summary["remaining_tag_demographic_or_explicit_live_candidates"] = remaining_tag_or_demo.to_dict("records")
    summary["finished_rows"] = int(len(df))

    return df, summary


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass
    return value


def save_dataset(df: pd.DataFrame, csv_path: Path = DATASET_CSV, json_path: Path = DATASET_JSON) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    records = json_safe(df.to_dict(orient="records"))
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2, allow_nan=False)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Apply EDA-driven anime dataset improvements.")
    parser.add_argument("--input-csv", type=Path, default=DATASET_CSV)
    parser.add_argument("--output-csv", type=Path, default=DATASET_CSV)
    parser.add_argument("--output-json", type=Path, default=DATASET_JSON)
    parser.add_argument("--live-anidb", action="store_true", help="Try live AniDB calls for remaining cache gaps.")
    parser.add_argument("--live-limit", type=int, default=None, help="Maximum live AniDB calls when --live-anidb is used.")
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv)
    cache_payload = load_anidb_cache()

    if args.live_anidb:
        episode_gaps = df[df["episodes"].isna() | (pd.to_numeric(df["episodes"], errors="coerce").fillna(-1) == 0)]
        tag_demo_gaps = df[
            df["tags"].apply(is_missing_text)
            | df["demographics"].apply(is_missing_text)
            | (
                df["explicit_tags"].apply(is_missing_text)
                & df.apply(should_have_explicit_tags, axis=1)
            )
        ]
        candidate_ids = (
            pd.concat([episode_gaps["anidb_id"], tag_demo_gaps["anidb_id"]])
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        updated = update_cache_with_live_payloads(cache_payload, candidate_ids, limit=args.live_limit)
        print(f"Live AniDB cache updates: {updated}")

    improved, summary = apply_improvements(df, cache_payload)
    save_dataset(improved, csv_path=args.output_csv, json_path=args.output_json)

    cli_summary = {
        key: value
        for key, value in summary.items()
        if not key.startswith("remaining_")
    }
    cli_summary["remaining_episode_live_candidates_count"] = len(
        summary["remaining_episode_live_candidates"]
    )
    cli_summary["remaining_tag_or_demographic_live_candidates_count"] = len(
        summary["remaining_tag_demographic_or_explicit_live_candidates"]
    )
    cli_summary["remaining_episode_live_candidates_sample"] = summary[
        "remaining_episode_live_candidates"
    ][:10]
    cli_summary["remaining_tag_or_demographic_live_candidates_sample"] = summary[
        "remaining_tag_demographic_or_explicit_live_candidates"
    ][:10]

    print(json.dumps(json_safe(cli_summary), ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
