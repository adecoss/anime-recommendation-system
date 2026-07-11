from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
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
    from anilist_metadata_utils import (
        ANILIST_CACHE_FILE,
        classify_anilist_media,
        get_anilist_media,
        load_anilist_cache,
        merge_recommendation_sources,
        save_anilist_cache,
        seed_anilist_cache_from_sample,
        update_anilist_cache_for_mal_ids,
    )
except ImportError:  # Imported from a notebook with project root on sys.path.
    from src.anilist_metadata_utils import (
        ANILIST_CACHE_FILE,
        classify_anilist_media,
        get_anilist_media,
        load_anilist_cache,
        merge_recommendation_sources,
        save_anilist_cache,
        seed_anilist_cache_from_sample,
        update_anilist_cache_for_mal_ids,
    )

try:
    import requests
except ImportError:  # pragma: no cover - live AniDB fill is optional
    requests = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_CSV = ROOT / "data" / "processed" / "anime_dataset.csv"
DATASET_JSON = ROOT / "data" / "processed" / "anime_dataset.json"
ANIDB_CACHE_FILE = ROOT / "data" / "caches" / "anidb_metadata_cache.json"
SKIPPED_INVALID_TYPE_IDS_FILE = ROOT / "data" / "build" / "skipped_invalid_type_ids.json"
TAG_QUALITY_REVIEW_CSV = ROOT / "artifacts" / "eda_tables" / "tag_quality_review.csv"
REMOVED_DUPLICATE_SPECIALS_CSV = ROOT / "artifacts" / "eda_tables" / "removed_duplicate_specials.csv"
REMOVED_SPECIALS_WITHOUT_ANIDB_CSV = ROOT / "artifacts" / "eda_tables" / "removed_specials_without_anidb.csv"


def normalize_secret_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value).strip().upper()).strip("_")


def parse_secret_file(path: Path) -> dict[str, str]:
    parsed: dict[str, str] = {}
    if not path.exists():
        return parsed
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        parsed[normalize_secret_key(key)] = value.strip().strip('"').strip("'")
    return parsed


def read_local_secret(*names: str, default: str | None = None, filename: str | None = None) -> str | None:
    for name in names:
        value = os.environ.get(name) or os.environ.get(normalize_secret_key(name))
        if value:
            return value

    secret_cache = parse_secret_file(ROOT / "secrets" / "secret.txt")
    for name in names:
        value = secret_cache.get(normalize_secret_key(name))
        if value:
            return value

    if filename:
        path = ROOT / "secrets" / filename
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    return default


ANIDB_CLIENT = read_local_secret("ANIDB_CLIENT", "ANIDB_CLIENT_NAME", filename="anidb_client.txt", default="")
ANIDB_CLIENTVER = int(read_local_secret("ANIDB_CLIENTVER", "ANIDB_CLIENT_VERSION", filename="anidb_clientver.txt", default="0") or 0)
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

SPECIAL_TYPES = {"Special", "TV Special"}
RECAP_TITLE_PATTERNS = (
    r"\brecap\b",
    r"\brecaps\b",
    r"\bsummary\b",
    r"\bdigest\b",
    r"\bsoushuuhen\b",
    r"\bsoushuu-hen\b",
)
MINOR_SHARED_ANIDB_TITLE_PATTERNS = RECAP_TITLE_PATTERNS + (
    r"\bmanner\b",
    r"\bomake\b",
    r"\bpreview\b",
    r"\btrailer\b",
    r"\bteaser\b",
    r"\bcommercial\b",
    r"\bcm\b",
    r"\bpv\b",
)
CORE_FRANCHISE_RELATIONS = {"Prequel", "Sequel", "Parent Story", "Alternative Setting", "Alternative Version"}
SUMMARY_RELATION = "Summary"
FULL_STORY_RELATION = "Full Story"
MINOR_SHARED_ANIDB_RELATIONS = {FULL_STORY_RELATION}
TAG_MIN_GLOBAL_COUNT = 5
TAG_MIN_POSITIVE_WEIGHT = 1
TAG_LOW_COUNT_FALLBACK_MIN = 3
GENRE_PROMOTION_MIN_TAG_WEIGHT = 400
ANIDB_SIMILAR_MIN_APPROVAL_RATIO = 0.25
ANIDB_SIMILAR_SHARED_MIN_MEMBER_RATIO = 0.10
ANIDB_SIMILAR_SHARED_MIN_DURATION_MINUTES = 10

MAL_GENRES = {
    "Action",
    "Adventure",
    "Avant Garde",
    "Award Winning",
    "Boys Love",
    "Comedy",
    "Drama",
    "Fantasy",
    "Girls Love",
    "Gourmet",
    "Horror",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Slice of Life",
    "Sports",
    "Supernatural",
    "Suspense",
}
MAL_EXPLICIT_GENRES = {"Ecchi", "Erotica", "Hentai"}
MAL_THEMES = {
    "Adult Cast",
    "Anthropomorphic",
    "CGDCT",
    "Childcare",
    "Combat Sports",
    "Crossdressing",
    "Delinquents",
    "Detective",
    "Educational",
    "Gag Humor",
    "Gore",
    "Harem",
    "High Stakes Game",
    "Historical",
    "Idols (Female)",
    "Idols (Male)",
    "Isekai",
    "Iyashikei",
    "Love Polygon",
    "Love Status Quo",
    "Magical Sex Shift",
    "Mahou Shoujo",
    "Martial Arts",
    "Mecha",
    "Medical",
    "Military",
    "Music",
    "Mythology",
    "Organized Crime",
    "Otaku Culture",
    "Parody",
    "Performing Arts",
    "Pets",
    "Psychological",
    "Racing",
    "Reincarnation",
    "Reverse Harem",
    "Samurai",
    "School",
    "Showbiz",
    "Space",
    "Strategy Game",
    "Super Power",
    "Survival",
    "Team Sports",
    "Time Travel",
    "Urban Fantasy",
    "Vampire",
    "Video Game",
    "Villainess",
    "Visual Arts",
    "Workplace",
}
PROTECTED_TAGS: set[str] = set()
GENRE_TAG_ALIASES = {
    "action": "Action",
    "adventure": "Adventure",
    "avant garde": "Avant Garde",
    "award winning": "Award Winning",
    "boys love": "Boys Love",
    "shounen ai": "Boys Love",
    "comedy": "Comedy",
    "drama": "Drama",
    "fantasy": "Fantasy",
    "girls love": "Girls Love",
    "shoujo ai": "Girls Love",
    "gourmet": "Gourmet",
    "cooking": "Gourmet",
    "horror": "Horror",
    "mystery": "Mystery",
    "romance": "Romance",
    "hard science fiction": "Sci-Fi",
    "science fiction": "Sci-Fi",
    "soft science fiction": "Sci-Fi",
    "sci fi": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "slice of life": "Slice of Life",
    "daily life": "Slice of Life",
    "sports": "Sports",
    "supernatural": "Supernatural",
    "suspense": "Suspense",
    "thriller": "Suspense",
}
THEME_TAG_ALIASES = {
    "adult cast": "Adult Cast",
    "predominantly adult cast": "Adult Cast",
    "anthropomorphism": "Anthropomorphic",
    "anthropomorphic": "Anthropomorphic",
    "cute girls doing cute things": "CGDCT",
    "childcare": "Childcare",
    "combat sports": "Combat Sports",
    "cross dressing": "Crossdressing",
    "cross-dressing": "Crossdressing",
    "crossdressing": "Crossdressing",
    "delinquents": "Delinquents",
    "detective": "Detective",
    "educational": "Educational",
    "gag humor": "Gag Humor",
    "slapstick": "Gag Humor",
    "gore": "Gore",
    "harem": "Harem",
    "high stakes game": "High Stakes Game",
    "historical": "Historical",
    "idols female": "Idols (Female)",
    "female idol": "Idols (Female)",
    "female idols": "Idols (Female)",
    "idols male": "Idols (Male)",
    "male idol": "Idols (Male)",
    "male idols": "Idols (Male)",
    "isekai": "Isekai",
    "iyashikei": "Iyashikei",
    "love polygon": "Love Polygon",
    "love status quo": "Love Status Quo",
    "magical sex shift": "Magical Sex Shift",
    "gender bender": "Magical Sex Shift",
    "mahou shoujo": "Mahou Shoujo",
    "magical girl": "Mahou Shoujo",
    "martial arts": "Martial Arts",
    "mecha": "Mecha",
    "medical": "Medical",
    "military": "Military",
    "music": "Music",
    "mythology": "Mythology",
    "organized crime": "Organized Crime",
    "otaku culture": "Otaku Culture",
    "parody": "Parody",
    "performing arts": "Performing Arts",
    "pets": "Pets",
    "psychological": "Psychological",
    "racing": "Racing",
    "reincarnation": "Reincarnation",
    "reverse harem": "Reverse Harem",
    "samurai": "Samurai",
    "school life": "School",
    "school": "School",
    "showbiz": "Showbiz",
    "space": "Space",
    "strategy game": "Strategy Game",
    "card games": "Strategy Game",
    "super power": "Super Power",
    "superpower": "Super Power",
    "survival": "Survival",
    "team sports": "Team Sports",
    "time travel": "Time Travel",
    "urban fantasy": "Urban Fantasy",
    "vampire": "Vampire",
    "video game": "Video Game",
    "visual novel": "Video Game",
    "villainess": "Villainess",
    "visual arts": "Visual Arts",
    "workplace": "Workplace",
    "association football": "football",
}
NOISY_TAGS = {
    "cast",
    "cast-free",
    "ending",
    "ending tags that need merging",
    "old animetags op ed",
    "old animetags - op ed",
    "only makes sense with original work knowledge",
    "some weird shit goin on",
    "some weird shit goin` on",
    "slow when it comes to love",
    "speculative fiction",
    "storytelling",
    "technical aspects",
    "tropes",
    "violent retribution for accidental infringement",
}
VAGUE_TAGS = {
    "air",
    "earth",
    "fire",
    "water",
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


def parse_weight_map(value: Any) -> dict[str, int]:
    weights: dict[str, int] = {}
    for item in split_pipe_values(value):
        if ":" not in item:
            continue
        name, weight = item.rsplit(":", 1)
        name = name.strip()
        parsed_weight = parse_int(weight, default=None)
        if name and parsed_weight is not None:
            weights[normalize_name(name)] = parsed_weight
    return weights


def count_pipe_values(series: pd.Series) -> Counter[str]:
    counts: Counter[str] = Counter()
    for value in series.fillna(""):
        counts.update(split_pipe_values(value))
    return counts


def normalize_name(value: Any) -> str:
    text = str(value or "").strip().casefold()
    text = re.sub(r"[\W_]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


PROTECTED_TAGS = {normalize_name(tag) for tag in MAL_THEMES}
NOISY_TAG_KEYS = {normalize_name(tag) for tag in NOISY_TAGS}
VAGUE_TAG_KEYS = {normalize_name(tag) for tag in VAGUE_TAGS}


def strip_anidb_maintenance_suffix(value: Any) -> str:
    text = str(value or "").strip()
    text = re.sub(r"\s*--\s*TO BE (?:SPLIT|MOVED|DELETED|MERGED).*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*-\s*TO BE (?:SPLIT|MOVED|DELETED|MERGED).*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


def canonical_tag_name(value: Any) -> str:
    clean = strip_anidb_maintenance_suffix(value)
    key = normalize_name(clean)
    if key in THEME_TAG_ALIASES:
        return THEME_TAG_ALIASES[key]
    if key == "sexual abuse":
        return "sexual abuse"
    return clean


def should_drop_normal_tag(value: Any) -> bool:
    key = normalize_name(strip_anidb_maintenance_suffix(value))
    return key in NOISY_TAG_KEYS or key in VAGUE_TAG_KEYS or "need deleting" in key or "needs merging" in key


def canonicalize_pipe_column(value: Any) -> str:
    return merge_pipe_values([canonical_tag_name(item) for item in split_pipe_values(value)])


def canonicalize_weight_column(value: Any) -> str:
    pairs = []
    for item in split_pipe_values(value):
        if ":" not in item:
            continue
        tag, weight = item.rsplit(":", 1)
        tag = canonical_tag_name(tag)
        parsed_weight = parse_int(weight, default=None)
        if tag and parsed_weight is not None:
            pairs.append(f"{tag}:{parsed_weight}")
    return merge_pipe_values(pairs)


def normalize_tags_and_promote_genres(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    summary = {
        "genre_values_added_from_tags": 0,
        "normal_tags_removed_as_genre_equivalents": 0,
        "normal_tags_removed_as_noisy_or_vague": 0,
        "normal_tags_canonicalized": 0,
        "explicit_tags_canonicalized": 0,
    }
    df = df.copy()

    if "genres" not in df.columns:
        df["genres"] = ""

    for idx, row in df.iterrows():
        genres = split_pipe_values(row.get("genres"))
        tags = split_pipe_values(row.get("tags"))
        tag_weights = parse_weight_map(row.get("tag_weights"))
        new_tags: list[str] = []
        new_weights: list[str] = []
        added_genres: list[str] = []
        dropped_genre_tags = 0
        dropped_noisy = 0

        for tag in tags:
            canonical = canonical_tag_name(tag)
            canonical_key = normalize_name(canonical)
            genre = GENRE_TAG_ALIASES.get(canonical_key)
            weight = tag_weights.get(normalize_name(tag), tag_weights.get(canonical_key))
            if genre and weight is not None and weight >= GENRE_PROMOTION_MIN_TAG_WEIGHT:
                added_genres.append(genre)
                dropped_genre_tags += 1
                continue
            if should_drop_normal_tag(canonical):
                dropped_noisy += 1
                continue

            new_tags.append(canonical)
            if weight is not None:
                new_weights.append(f"{canonical}:{weight}")

        merged_genres = merge_pipe_values(genres + added_genres)
        merged_tags = merge_pipe_values(new_tags)
        merged_weights = merge_pipe_values(new_weights)

        if merged_genres != ("" if is_missing_text(row.get("genres")) else str(row.get("genres"))):
            summary["genre_values_added_from_tags"] += len(set(map(normalize_name, added_genres)) - set(map(normalize_name, genres)))
            df.at[idx, "genres"] = merged_genres
        if merged_tags != ("" if is_missing_text(row.get("tags")) else str(row.get("tags"))) or merged_weights != (
            "" if is_missing_text(row.get("tag_weights")) else str(row.get("tag_weights"))
        ):
            summary["normal_tags_canonicalized"] += 1
            df.at[idx, "tags"] = merged_tags
            df.at[idx, "tag_weights"] = merged_weights
        summary["normal_tags_removed_as_genre_equivalents"] += dropped_genre_tags
        summary["normal_tags_removed_as_noisy_or_vague"] += dropped_noisy

    for tag_col, weight_col in [("explicit_tags", "explicit_tag_weights")]:
        if tag_col not in df.columns:
            continue
        for idx, row in df.iterrows():
            old_tags = "" if is_missing_text(row.get(tag_col)) else str(row.get(tag_col))
            old_weights = "" if is_missing_text(row.get(weight_col)) else str(row.get(weight_col))
            new_tags = canonicalize_pipe_column(old_tags)
            new_weights = canonicalize_weight_column(old_weights)
            if new_tags != old_tags or new_weights != old_weights:
                df.at[idx, tag_col] = new_tags
                if weight_col in df.columns:
                    df.at[idx, weight_col] = new_weights
                summary["explicit_tags_canonicalized"] += 1

    return df, summary


def row_sort_for_canonical(df: pd.DataFrame) -> pd.DataFrame:
    working = df.copy()
    working["_type_priority"] = working["type"].apply(
        lambda value: 1 if str(value).strip() in SPECIAL_TYPES else 0
    )
    working["_members_sort"] = pd.to_numeric(
        working["members"] if "members" in working.columns else pd.Series(-1, index=working.index),
        errors="coerce",
    ).fillna(-1)
    working["_episodes_sort"] = pd.to_numeric(
        working["episodes"] if "episodes" in working.columns else pd.Series(-1, index=working.index),
        errors="coerce",
    ).fillna(-1)
    working["_popularity_sort"] = pd.to_numeric(
        working["popularity"] if "popularity" in working.columns else pd.Series(999999999, index=working.index),
        errors="coerce",
    ).fillna(999999999)
    return working.sort_values(
        ["_type_priority", "_members_sort", "_episodes_sort", "_popularity_sort"],
        ascending=[True, False, False, True],
    )


def canonical_mal_by_anidb(df: pd.DataFrame) -> dict[int, int]:
    mapping: dict[int, int] = {}
    candidates = df[df["anidb_id"].notna()].copy()
    if candidates.empty:
        return mapping
    sorted_candidates = row_sort_for_canonical(candidates)
    for anidb_id, group in sorted_candidates.groupby("anidb_id", sort=False):
        parsed_anidb_id = parse_int(anidb_id, default=None)
        if parsed_anidb_id is None or group.empty:
            continue
        mapping[parsed_anidb_id] = int(group.iloc[0]["mal_id"])
    return mapping


def anidb_group_stats(df: pd.DataFrame) -> dict[int, dict[str, Any]]:
    stats: dict[int, dict[str, Any]] = {}
    if "anidb_id" not in df.columns:
        return stats

    for anidb_id, group in df[df["anidb_id"].notna()].groupby("anidb_id"):
        parsed_anidb_id = parse_int(anidb_id, default=None)
        if parsed_anidb_id is None:
            continue
        members = pd.to_numeric(group.get("members"), errors="coerce").fillna(0)
        group_mal_ids = {int(value) for value in group["mal_id"].dropna().astype(int)}
        minor_relation_sources: set[int] = set()
        minor_relation_targets: set[int] = set()
        for _idx, row in group.iterrows():
            source_mal_id = parse_int(row.get("mal_id"), default=None)
            for relation, target in parse_relation_edges(row.get("relations")):
                if relation == SUMMARY_RELATION and target in group_mal_ids:
                    minor_relation_targets.add(int(target))
                if relation == FULL_STORY_RELATION and target in group_mal_ids and source_mal_id is not None:
                    minor_relation_sources.add(int(source_mal_id))
        stats[parsed_anidb_id] = {
            "row_count": int(len(group)),
            "canonical_mal_id": int(row_sort_for_canonical(group).iloc[0]["mal_id"]),
            "max_members": int(members.max()) if not members.empty else 0,
            "minor_relation_sources": minor_relation_sources,
            "minor_relation_targets": minor_relation_targets,
        }
    return stats


def has_core_franchise_relation(value: Any) -> bool:
    for relation, _target in parse_relation_edges(value):
        if relation in CORE_FRANCHISE_RELATIONS:
            return True
    return False


def has_minor_shared_anidb_relation(value: Any) -> bool:
    for relation, _target in parse_relation_edges(value):
        if relation in MINOR_SHARED_ANIDB_RELATIONS:
            return True
    return False


def is_minor_shared_anidb_title(value: Any) -> bool:
    title = str(value or "")
    return any(re.search(pattern, title, flags=re.IGNORECASE) for pattern in MINOR_SHARED_ANIDB_TITLE_PATTERNS)


def should_receive_anidb_similar_edges(row: pd.Series, group_stats: dict[int, dict[str, Any]]) -> bool:
    anidb_id = parse_int(row.get("anidb_id"), default=None)
    if anidb_id is None:
        return False

    stats = group_stats.get(anidb_id, {"row_count": 1, "canonical_mal_id": parse_int(row.get("mal_id"), default=None)})
    if stats.get("row_count", 1) <= 1:
        return True

    mal_id = parse_int(row.get("mal_id"), default=None)
    if mal_id == stats.get("canonical_mal_id"):
        return True

    if mal_id in stats.get("minor_relation_targets", set()):
        return False

    if mal_id in stats.get("minor_relation_sources", set()):
        return False

    if is_minor_shared_anidb_title(row.get("title")):
        return False

    duration = parse_duration_minutes(row.get("duration"))
    members = parse_int(row.get("members"), default=0) or 0
    max_members = int(stats.get("max_members") or 0)
    member_ratio = members / max_members if max_members else 0

    if duration is not None and duration < ANIDB_SIMILAR_SHARED_MIN_DURATION_MINUTES:
        return False

    if has_minor_shared_anidb_relation(row.get("relations")):
        return False

    if has_core_franchise_relation(row.get("relations")):
        return True

    if not is_missing_text(row.get("recommendations")):
        return True

    return member_ratio >= ANIDB_SIMILAR_SHARED_MIN_MEMBER_RATIO


def parse_recommendation_edges(value: Any) -> dict[int, int]:
    edges: dict[int, int] = {}
    for item in split_pipe_values(value):
        if ":" in item:
            target, weight = item.split(":", 1)
        else:
            target, weight = item, "1"
        target_id = parse_int(target, default=None)
        if target_id is None:
            continue
        edges[target_id] = max(edges.get(target_id, 0), parse_int(weight, default=1) or 1)
    return edges


def format_recommendation_edges(edges: dict[int, int]) -> str:
    return "|".join(f"{target}:{weight}" for target, weight in edges.items())


def parse_relation_edges(value: Any) -> list[tuple[str, int]]:
    edges: list[tuple[str, int]] = []
    for item in split_pipe_values(value):
        if ":" not in item:
            continue
        relation, target = item.rsplit(":", 1)
        target_id = parse_int(target, default=None)
        relation = relation.strip()
        if relation and target_id is not None:
            edges.append((relation, target_id))
    return edges


def format_relation_edges(edges: list[tuple[str, int]]) -> str:
    seen: set[tuple[str, int]] = set()
    formatted: list[str] = []
    for relation, target in edges:
        key = (relation, int(target))
        if key in seen:
            continue
        seen.add(key)
        formatted.append(f"{relation}:{int(target)}")
    return "|".join(formatted)


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


def has_adult_demographic_rating(rating: Any) -> bool:
    return str(rating or "").strip() == "Rx - Hentai"


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

    if has_adult_demographic_rating(rating):
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
    """Classify AniDB raw tags using the same cleanup rules used by the dataset.

    This intentionally does not return normal tags that are equivalent to MAL
    genres. Values such as ``action`` and ``science fiction`` belong in
    ``genres`` after promotion, not back in the cleaned descriptive ``tags``.
    """
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

        canonical = canonical_tag_name(name)
        canonical_key = normalize_name(canonical)

        if is_always_explicit or is_rating_gated_explicit:
            if canonical:
                parsed["explicit_tags"].append(canonical)
                parsed["explicit_tag_weights"].append(f"{canonical}:{tag['weight']}")
        else:
            if not canonical:
                continue
            if should_drop_normal_tag(canonical):
                continue
            parsed["tags"].append(canonical)
            parsed["tag_weights"].append(f"{canonical}:{tag['weight']}")

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
    merged = parse_recommendation_edges(existing)

    for edge in new_edges:
        for target, weight in parse_recommendation_edges(edge).items():
            merged[target] = max(merged.get(target, 0), weight)

    return format_recommendation_edges(merged)


def remove_duplicate_specials_with_shared_anidb(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int], pd.DataFrame]:
    if "anidb_id" not in df.columns:
        return df, {}, pd.DataFrame()

    removals: list[dict[str, Any]] = []
    removed_to_canonical: dict[int, int] = {}
    duplicate_groups = df[df["anidb_id"].notna()].groupby("anidb_id")

    for anidb_id, group in duplicate_groups:
        if len(group) <= 1:
            continue

        sorted_group = row_sort_for_canonical(group)
        canonical_row = sorted_group.iloc[0]
        canonical_mal_id = int(canonical_row["mal_id"])
        canonical_members = parse_int(canonical_row.get("members"), default=0) or 0
        canonical_episodes = parse_int(canonical_row.get("episodes"), default=0) or 0

        for _, row in group.iterrows():
            mal_id = int(row["mal_id"])
            if mal_id == canonical_mal_id:
                continue

            row_type = str(row.get("type") or "").strip()
            row_title = str(row.get("title") or "")
            is_special_type = row_type in SPECIAL_TYPES
            is_recap_like = any(
                re.search(pattern, row_title, flags=re.IGNORECASE)
                for pattern in RECAP_TITLE_PATTERNS
            )
            if not is_special_type and not is_recap_like:
                continue

            row_members = parse_int(row.get("members"), default=0) or 0
            row_episodes = parse_int(row.get("episodes"), default=0) or 0
            clearly_smaller = canonical_members > row_members or canonical_episodes > row_episodes
            if not is_recap_like and not clearly_smaller:
                continue

            removed_to_canonical[mal_id] = canonical_mal_id
            removals.append(
                {
                    "removed_mal_id": mal_id,
                    "removed_title": row.get("title"),
                    "removed_type": row.get("type"),
                    "removed_members": row_members,
                    "removed_episodes": row_episodes,
                    "shared_anidb_id": parse_int(anidb_id, default=None),
                    "canonical_mal_id": canonical_mal_id,
                    "canonical_title": canonical_row.get("title"),
                    "canonical_type": canonical_row.get("type"),
                    "canonical_members": canonical_members,
                    "canonical_episodes": canonical_episodes,
                }
            )

    if not removed_to_canonical:
        return df, {}, pd.DataFrame()

    cleaned = df.loc[~df["mal_id"].astype(int).isin(removed_to_canonical)].copy()
    return cleaned, removed_to_canonical, pd.DataFrame(removals)


def remove_specials_without_anidb(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "anidb_id" not in df.columns:
        return df, pd.DataFrame()
    mask = df["type"].astype(str).isin(SPECIAL_TYPES) & df["anidb_id"].isna()
    removed = df.loc[
        mask,
        ["mal_id", "title", "type", "members", "popularity", "relations", "recommendations"],
    ].copy()
    if removed.empty:
        return df, removed
    cleaned = df.loc[~mask].copy()
    return cleaned, removed


def load_invalid_type_registry(path: Path = SKIPPED_INVALID_TYPE_IDS_FILE) -> dict[int, dict[str, Any]]:
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}

    rows = payload.get("invalid_type_ids", []) if isinstance(payload, dict) else []
    registry: dict[int, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        mal_id = parse_int(row.get("mal_id"), default=None)
        if mal_id is None:
            continue
        registry[mal_id] = {
            "mal_id": mal_id,
            "anime_type": row.get("anime_type"),
            "index": parse_int(row.get("index"), default=None),
        }
    return registry


def save_invalid_type_registry(
    registry: dict[int, dict[str, Any]],
    path: Path = SKIPPED_INVALID_TYPE_IDS_FILE,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "invalid_type_ids": [
            {
                "mal_id": int(row["mal_id"]),
                "anime_type": row.get("anime_type"),
                "index": row.get("index"),
            }
            for _, row in sorted(registry.items())
        ]
    }
    atomic_write_json(path, payload)


def register_removed_specials_as_invalid_type(
    removed_no_anidb_specials: pd.DataFrame,
    removed_duplicate_specials: pd.DataFrame,
    path: Path = SKIPPED_INVALID_TYPE_IDS_FILE,
) -> int:
    registry = load_invalid_type_registry(path)
    before = len(registry)

    for _, row in removed_no_anidb_specials.iterrows():
        mal_id = parse_int(row.get("mal_id"), default=None)
        if mal_id is None:
            continue
        registry[mal_id] = {
            "mal_id": mal_id,
            "anime_type": row.get("type"),
            "index": None,
        }

    for _, row in removed_duplicate_specials.iterrows():
        mal_id = parse_int(row.get("removed_mal_id"), default=None)
        if mal_id is None:
            continue
        registry[mal_id] = {
            "mal_id": mal_id,
            "anime_type": row.get("removed_type"),
            "index": None,
        }

    if len(registry) != before:
        save_invalid_type_registry(registry, path)

    return len(registry) - before


def rewrite_edges_after_removals(df: pd.DataFrame, removed_to_canonical: dict[int, int]) -> tuple[pd.DataFrame, int]:
    valid_ids = set(df["mal_id"].dropna().astype(int))
    changed = 0

    for idx, row in df.iterrows():
        if "recommendations" in df.columns:
            rec_edges = parse_recommendation_edges(row.get("recommendations"))
            rewritten_rec: dict[int, int] = {}
            for target, weight in rec_edges.items():
                target = removed_to_canonical.get(target, target)
                if target == int(row["mal_id"]) or target not in valid_ids:
                    continue
                rewritten_rec[target] = max(rewritten_rec.get(target, 0), weight)
            formatted_rec = format_recommendation_edges(rewritten_rec)
            if formatted_rec != ("" if is_missing_text(row.get("recommendations")) else str(row.get("recommendations"))):
                df.at[idx, "recommendations"] = formatted_rec
                changed += 1

        if "relations" in df.columns:
            rewritten_rel: list[tuple[str, int]] = []
            for relation, target in parse_relation_edges(row.get("relations")):
                target = removed_to_canonical.get(target, target)
                if target == int(row["mal_id"]) or target not in valid_ids:
                    continue
                rewritten_rel.append((relation, target))
            formatted_rel = format_relation_edges(rewritten_rel)
            if formatted_rel != ("" if is_missing_text(row.get("relations")) else str(row.get("relations"))):
                df.at[idx, "relations"] = formatted_rel
                changed += 1

    return df, changed


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
        approval = parse_int(item.get("approval"), default=None)
        total = parse_int(item.get("total"), default=None)
        if approval is None or total is None or total <= 0:
            continue
        if approval / total < ANIDB_SIMILAR_MIN_APPROVAL_RATIO:
            continue

        weight = (
            approval
            or parse_int(item.get("total"), default=None)
            or 1
        )
        edges.append(f"{mal_id}:{max(1, int(weight))}")
    return edges


def fetch_anidb_live_payload(anidb_id: int) -> dict[str, Any] | None:
    if requests is None:
        raise RuntimeError("requests is not installed; live AniDB fill is unavailable")
    if not ANIDB_CLIENT or not ANIDB_CLIENTVER:
        raise RuntimeError("missing AniDB client/clientver; set secrets/secret.txt or ANIDB_CLIENT and ANIDB_CLIENTVER")

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


def prune_weighted_tags_for_row(
    tags_value: Any,
    weights_value: Any,
    global_counts: Counter[str],
    min_global_count: int = TAG_MIN_GLOBAL_COUNT,
    low_count_fallback_min: int = TAG_LOW_COUNT_FALLBACK_MIN,
) -> tuple[str, str, int]:
    tags = split_pipe_values(tags_value)
    if not tags:
        return "", "", 0

    weight_map = parse_weight_map(weights_value)
    primary: list[str] = []
    fallback: list[str] = []
    dropped = 0

    for tag in tags:
        count = global_counts.get(tag, 0)
        weight = weight_map.get(normalize_name(tag))
        tag_key = normalize_name(tag)

        if tag_key in PROTECTED_TAGS:
            primary.append(tag)
            continue

        if should_drop_normal_tag(tag):
            dropped += 1
            continue

        if count < min_global_count:
            dropped += 1
            continue

        if weight is None or weight >= TAG_MIN_POSITIVE_WEIGHT:
            primary.append(tag)
        elif weight == 0:
            fallback.append(tag)
            dropped += 1
        else:
            dropped += 1

    kept = primary[:]
    if len(kept) < low_count_fallback_min:
        for tag in fallback:
            if tag not in kept:
                kept.append(tag)
            if len(kept) >= low_count_fallback_min:
                break

    actual_dropped = len(tags) - len(kept)
    kept_weights = []
    for tag in kept:
        weight = weight_map.get(normalize_name(tag))
        if weight is not None:
            kept_weights.append(f"{tag}:{weight}")

    return merge_pipe_values(kept), merge_pipe_values(kept_weights), actual_dropped


def export_tag_quality_review(
    df: pd.DataFrame,
    path: Path = TAG_QUALITY_REVIEW_CSV,
    after_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    after_df = df if after_df is None else after_df
    for tag_column, weight_column, tag_type in [
        ("tags", "tag_weights", "tag"),
        ("explicit_tags", "explicit_tag_weights", "explicit_tag"),
    ]:
        if tag_column not in df.columns:
            continue

        counts = count_pipe_values(df[tag_column])
        after_counts = count_pipe_values(after_df[tag_column]) if tag_column in after_df.columns else Counter()
        all_tags = set(counts) | set(after_counts)
        weight_values: dict[str, list[int]] = defaultdict(list)
        example_titles: dict[str, list[str]] = defaultdict(list)

        for _, row in df.iterrows():
            weights = parse_weight_map(row.get(weight_column))
            for tag in split_pipe_values(row.get(tag_column)):
                key = normalize_name(tag)
                if key in weights:
                    weight_values[tag].append(weights[key])
                if len(example_titles[tag]) < 5:
                    title = str(row.get("title") or "").strip()
                    if title and title not in example_titles[tag]:
                        example_titles[tag].append(title)

        for tag in all_tags:
            count = counts.get(tag, 0)
            after_count = after_counts.get(tag, 0)
            weights = weight_values.get(tag, [])
            zero_weight_rows = sum(1 for value in weights if value == 0)
            positive_weight_rows = sum(1 for value in weights if value and value > 0)
            avg_weight = round(sum(weights) / len(weights), 2) if weights else None
            protected = normalize_name(tag) in PROTECTED_TAGS
            rows.append(
                {
                    "tag_type": tag_type,
                    "tag": tag,
                    "count": int(count),
                    "count_after_pruning": int(after_count),
                    "weight_rows": len(weights),
                    "positive_weight_rows": positive_weight_rows,
                    "zero_weight_rows": zero_weight_rows,
                    "avg_weight": avg_weight,
                    "protected_mal_theme": protected,
                    "drop_candidate": bool(
                        tag_type == "tag"
                        and not protected
                        and (
                            count < TAG_MIN_GLOBAL_COUNT
                            or after_count == 0
                            or should_drop_normal_tag(tag)
                            or (weights and positive_weight_rows == 0)
                        )
                    ),
                    "example_titles": " | ".join(example_titles[tag]),
                }
            )

    review = pd.DataFrame(rows)
    if not review.empty:
        review["_tag_type_sort"] = review["tag_type"].map({"tag": 0, "explicit_tag": 1}).fillna(2)
        review = review.sort_values(
            ["_tag_type_sort", "drop_candidate", "count_after_pruning", "count", "tag"],
            ascending=[True, False, True, True, True],
        ).drop(columns=["_tag_type_sort"])
    path.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(path, index=False)
    return review


def prune_dataset_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    summary = {
        "unique_tags_before": 0,
        "unique_tags_after": 0,
        "tag_values_removed": 0,
        "rows_with_tags_changed": 0,
    }
    if "tags" not in df.columns:
        export_tag_quality_review(df)
        return df, summary

    before_df = df.copy()
    counts_before = count_pipe_values(df["tags"])
    summary["unique_tags_before"] = len(counts_before)
    rows_changed = 0
    removed = 0

    for idx, row in df.iterrows():
        new_tags, new_weights, dropped = prune_weighted_tags_for_row(
            row.get("tags"),
            row.get("tag_weights"),
            counts_before,
        )
        old_tags = "" if is_missing_text(row.get("tags")) else str(row.get("tags"))
        old_weights = "" if is_missing_text(row.get("tag_weights")) else str(row.get("tag_weights"))
        if new_tags != old_tags or new_weights != old_weights:
            df.at[idx, "tags"] = new_tags
            df.at[idx, "tag_weights"] = new_weights
            rows_changed += 1
        removed += dropped

    counts_after_weight_prune = count_pipe_values(df["tags"])
    rare_tags = {
        tag
        for tag, count in counts_after_weight_prune.items()
        if count < TAG_MIN_GLOBAL_COUNT and normalize_name(tag) not in PROTECTED_TAGS
    }
    if rare_tags:
        for idx, row in df.iterrows():
            tags = split_pipe_values(row.get("tags"))
            if not tags:
                continue
            filtered_tags = [tag for tag in tags if tag not in rare_tags]
            if len(filtered_tags) == len(tags):
                continue
            weights = parse_weight_map(row.get("tag_weights"))
            filtered_weights = [
                f"{tag}:{weights[normalize_name(tag)]}"
                for tag in filtered_tags
                if normalize_name(tag) in weights
            ]
            df.at[idx, "tags"] = merge_pipe_values(filtered_tags)
            df.at[idx, "tag_weights"] = merge_pipe_values(filtered_weights)
            rows_changed += 1
            removed += len(tags) - len(filtered_tags)

    counts_after = count_pipe_values(df["tags"])
    summary["unique_tags_after"] = len(counts_after)
    summary["tag_values_removed"] = int(removed)
    summary["rows_with_tags_changed"] = int(rows_changed)
    export_tag_quality_review(before_df, after_df=df)
    return df, summary


def anilist_season_to_dataset(value: Any) -> str:
    return str(value or "").strip().lower()


def anilist_date_parts(media: dict[str, Any] | None) -> tuple[int | None, int | None]:
    if not media:
        return None, None
    start = media.get("startDate") or {}
    return parse_int(start.get("year"), default=None), parse_int(start.get("month"), default=None)


def consensus_fill_from_anilist(df: pd.DataFrame, anilist_cache: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Patch catalog metadata from AniList while preserving MAL/AniDB strengths.

    AniList is used as the preferred source for semantic ``genres`` and
    descriptive ``tags`` because its tag taxonomy has ranks, categories, and
    adult flags. MAL remains the source for popularity, score, and the public
    age-rating label. AniDB remains useful for episode/duration repairs and
    content-indicator safety signals.
    """
    summary = {
        "anilist_cache_file": str(ANILIST_CACHE_FILE),
        "anilist_cache_entries": len(anilist_cache.get("items", {})),
        "anilist_rows_seen": 0,
        "anilist_rows_missing": 0,
        "anilist_id_filled": 0,
        "anilist_genre_rows_replaced": 0,
        "anilist_tag_rows_replaced": 0,
        "anilist_explicit_tag_rows_replaced": 0,
        "anilist_demographic_rows_merged": 0,
        "anilist_recommendation_rows_merged": 0,
        "anilist_episode_fills": 0,
        "anilist_duration_fills": 0,
        "anilist_source_fills": 0,
        "anilist_aired_year_fills": 0,
        "anilist_aired_month_fills": 0,
        "anilist_season_fills": 0,
    }

    df = df.copy()
    for column in [
        "anilist_id",
        "anilist_format",
        "anilist_is_adult",
        "tag_weights",
        "explicit_tags",
        "explicit_tag_weights",
        "recommendations",
    ]:
        if column not in df.columns:
            df[column] = ""

    for idx, row in df.iterrows():
        mal_id = parse_int(row.get("mal_id"), default=None)
        if mal_id is None:
            continue
        media, source = get_anilist_media(mal_id, anilist_cache, live=False)
        if not media:
            summary["anilist_rows_missing"] += 1
            continue

        summary["anilist_rows_seen"] += 1
        parsed = classify_anilist_media(row, media)
        anilist_id = parse_int(parsed.get("anilist_id"), default=None)
        if anilist_id is not None and is_missing_text(row.get("anilist_id")):
            df.at[idx, "anilist_id"] = anilist_id
            summary["anilist_id_filled"] += 1
        df.at[idx, "anilist_format"] = media.get("format") or ""
        df.at[idx, "anilist_is_adult"] = bool(media.get("isAdult"))

        if parsed.get("genres"):
            old = "" if is_missing_text(row.get("genres")) else str(row.get("genres"))
            if old != parsed["genres"]:
                df.at[idx, "genres"] = parsed["genres"]
                summary["anilist_genre_rows_replaced"] += 1

        if parsed.get("tags"):
            old = "" if is_missing_text(row.get("tags")) else str(row.get("tags"))
            if old != parsed["tags"]:
                df.at[idx, "tags"] = parsed["tags"]
                df.at[idx, "tag_weights"] = parsed["tag_weights"]
                summary["anilist_tag_rows_replaced"] += 1

        old_explicit = "" if is_missing_text(row.get("explicit_tags")) else str(row.get("explicit_tags"))
        if parsed.get("explicit_tags") and old_explicit != parsed["explicit_tags"]:
            df.at[idx, "explicit_tags"] = parsed["explicit_tags"]
            df.at[idx, "explicit_tag_weights"] = parsed["explicit_tag_weights"]
            summary["anilist_explicit_tag_rows_replaced"] += 1
        elif not parsed.get("explicit_tags") and old_explicit:
            # If AniList has a non-adult consensus and MAL is not Rx, remove
            # old broad explicit tags created by AniDB sexual/fetish branches.
            if not has_explicit_rating(row.get("rating")):
                df.at[idx, "explicit_tags"] = ""
                df.at[idx, "explicit_tag_weights"] = ""
                summary["anilist_explicit_tag_rows_replaced"] += 1

        if parsed.get("demographics"):
            keep_age_demographics = {"Kodomo"}
            if has_adult_demographic_rating(row.get("rating")):
                keep_age_demographics.add("18+")
            existing_demographics = [
                value for value in split_pipe_values(row.get("demographics")) if value in keep_age_demographics
            ]
            merged_demographics = merge_pipe_values(
                existing_demographics + split_pipe_values(parsed["demographics"])
            )
            if merged_demographics != ("" if is_missing_text(row.get("demographics")) else str(row.get("demographics"))):
                df.at[idx, "demographics"] = merged_demographics
                summary["anilist_demographic_rows_merged"] += 1
        else:
            keep_age_demographics = {"Kodomo"}
            if has_adult_demographic_rating(row.get("rating")):
                keep_age_demographics.add("18+")
            existing_demographics = split_pipe_values(row.get("demographics"))
            retained_demographics = [value for value in existing_demographics if value in keep_age_demographics]
            retained = merge_pipe_values(retained_demographics)
            if retained != ("" if is_missing_text(row.get("demographics")) else str(row.get("demographics"))):
                df.at[idx, "demographics"] = retained
                summary["anilist_demographic_rows_merged"] += 1

        if parsed.get("recommendations"):
            merged_recommendations = merge_recommendation_sources(row.get("recommendations"), parsed["recommendations"])
            if merged_recommendations != ("" if is_missing_text(row.get("recommendations")) else str(row.get("recommendations"))):
                df.at[idx, "recommendations"] = merged_recommendations
                summary["anilist_recommendation_rows_merged"] += 1

        anilist_episodes = parse_int(media.get("episodes"), default=None)
        next_airing_episode = parse_int(media.get("next_airing_episode"), default=None)
        if anilist_episodes is None and next_airing_episode and next_airing_episode > 1:
            anilist_episodes = next_airing_episode - 1
        current_episodes = parse_int(row.get("episodes"), default=None)
        if anilist_episodes and (current_episodes is None or current_episodes <= 0):
            df.at[idx, "episodes"] = anilist_episodes
            summary["anilist_episode_fills"] += 1

        anilist_duration = parse_int(media.get("duration"), default=None)
        current_duration = parse_duration_minutes(row.get("duration"))
        if anilist_duration and current_duration is None:
            df.at[idx, "duration"] = anilist_duration
            summary["anilist_duration_fills"] += 1

        if media.get("source") and is_missing_text(row.get("source")):
            df.at[idx, "source"] = str(media["source"]).replace("_", " ").title()
            summary["anilist_source_fills"] += 1

        start_year, start_month = anilist_date_parts(media)
        if start_year and pd.isna(row.get("aired_year")):
            df.at[idx, "aired_year"] = start_year
            summary["anilist_aired_year_fills"] += 1
        if start_month and pd.isna(row.get("aired_month")):
            df.at[idx, "aired_month"] = start_month
            summary["anilist_aired_month_fills"] += 1

        season = anilist_season_to_dataset(media.get("season"))
        if season and is_missing_text(row.get("season")):
            df.at[idx, "season"] = season
            summary["anilist_season_fills"] += 1

    return df, summary


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
        "studios_filled_from_origin_tags": 0,
        "recommendations_augmented_from_anidb_similar_anime": 0,
        "anidb_similar_recommendation_rows_skipped_shared_minor": 0,
        "tags_filled_from_anidb_cache": 0,
        "explicit_tags_filled_from_anidb_cache": 0,
        "demographics_filled_from_anidb_cache": 0,
        "demographics_normalized": 0,
        "demographics_inferred_from_rating_genres_tags": 0,
        "anilist_cache_file": str(ANILIST_CACHE_FILE),
        "anilist_cache_entries": 0,
        "anilist_rows_seen": 0,
        "anilist_rows_missing": 0,
        "anilist_id_filled": 0,
        "anilist_genre_rows_replaced": 0,
        "anilist_tag_rows_replaced": 0,
        "anilist_explicit_tag_rows_replaced": 0,
        "anilist_demographic_rows_merged": 0,
        "anilist_recommendation_rows_merged": 0,
        "anilist_episode_fills": 0,
        "anilist_duration_fills": 0,
        "anilist_source_fills": 0,
        "anilist_aired_year_fills": 0,
        "anilist_aired_month_fills": 0,
        "anilist_season_fills": 0,
        "duplicate_special_rows_removed": 0,
        "special_rows_without_anidb_removed": 0,
        "removed_special_ids_added_to_invalid_type_registry": 0,
        "edge_rows_rewritten_after_duplicate_special_removal": 0,
        "duplicate_special_removal_audit_csv": str(REMOVED_DUPLICATE_SPECIALS_CSV),
        "special_without_anidb_removal_audit_csv": str(REMOVED_SPECIALS_WITHOUT_ANIDB_CSV),
        "tag_quality_review_csv": str(TAG_QUALITY_REVIEW_CSV),
        "genre_values_added_from_tags": 0,
        "normal_tags_removed_as_genre_equivalents": 0,
        "normal_tags_removed_as_noisy_or_vague": 0,
        "normal_tags_canonicalized": 0,
        "explicit_tags_canonicalized": 0,
        "unique_tags_before_pruning": 0,
        "unique_tags_after_pruning": 0,
        "tag_values_removed_by_pruning": 0,
        "rows_with_tags_changed_by_pruning": 0,
        "remaining_episode_live_candidates": [],
        "remaining_tag_demographic_or_explicit_live_candidates": [],
    }

    items = cache_payload.get("items", {})
    anilist_cache = load_anilist_cache()
    if seed_anilist_cache_from_sample(anilist_cache):
        save_anilist_cache(anilist_cache)
    df = df.copy()

    missing_air_date = df["aired_year"].isna() & df["aired_month"].isna()
    summary["dropped_missing_air_date"] = int(missing_air_date.sum())
    df = df.loc[~missing_air_date].copy()

    df, removed_no_anidb_specials = remove_specials_without_anidb(df)
    summary["special_rows_without_anidb_removed"] = int(len(removed_no_anidb_specials))
    REMOVED_SPECIALS_WITHOUT_ANIDB_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not removed_no_anidb_specials.empty or not REMOVED_SPECIALS_WITHOUT_ANIDB_CSV.exists():
        removed_no_anidb_specials.to_csv(REMOVED_SPECIALS_WITHOUT_ANIDB_CSV, index=False)

    df, removed_to_canonical, removed_specials = remove_duplicate_specials_with_shared_anidb(df)
    summary["duplicate_special_rows_removed"] = int(len(removed_specials))
    REMOVED_DUPLICATE_SPECIALS_CSV.parent.mkdir(parents=True, exist_ok=True)
    if not removed_specials.empty or not REMOVED_DUPLICATE_SPECIALS_CSV.exists():
        removed_specials.to_csv(REMOVED_DUPLICATE_SPECIALS_CSV, index=False)
    summary["removed_special_ids_added_to_invalid_type_registry"] = register_removed_specials_as_invalid_type(
        removed_no_anidb_specials,
        removed_specials,
    )
    df, edge_rewrite_count = rewrite_edges_after_removals(df, removed_to_canonical)
    summary["edge_rows_rewritten_after_duplicate_special_removal"] = int(edge_rewrite_count)

    anidb_to_mal = canonical_mal_by_anidb(df)
    shared_anidb_stats = anidb_group_stats(df)

    df, tag_normalization_summary = normalize_tags_and_promote_genres(df)
    for key, value in tag_normalization_summary.items():
        summary[key] = int(value)

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
        if not should_receive_anidb_similar_edges(row, shared_anidb_stats):
            summary["anidb_similar_recommendation_rows_skipped_shared_minor"] += 1
            continue
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

    df, anilist_summary = consensus_fill_from_anilist(df, anilist_cache)
    for key, value in anilist_summary.items():
        if isinstance(value, int):
            summary[key] = int(value)
        else:
            summary[key] = value

    needs_demographics = df["demographics"].apply(is_missing_text)
    for idx, row in df.loc[needs_demographics].iterrows():
        inferred = infer_demographics_from_row(row)
        if inferred:
            df.at[idx, "demographics"] = inferred
            summary["demographics_inferred_from_rating_genres_tags"] += 1

    if summary["anilist_rows_seen"] == 0:
        df, tag_normalization_summary = normalize_tags_and_promote_genres(df)
        for key, value in tag_normalization_summary.items():
            summary[key] += int(value)

    df, tag_pruning_summary = prune_dataset_tags(df)
    summary["unique_tags_before_pruning"] = tag_pruning_summary["unique_tags_before"]
    summary["unique_tags_after_pruning"] = tag_pruning_summary["unique_tags_after"]
    summary["tag_values_removed_by_pruning"] = tag_pruning_summary["tag_values_removed"]
    summary["rows_with_tags_changed_by_pruning"] = tag_pruning_summary["rows_with_tags_changed"]

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
    df = df.drop(columns=[column for column in ["synopsis", "description"] if column in df.columns])
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
    parser.add_argument("--live-anilist", action="store_true", help="Fetch missing AniList media payloads before applying corrections.")
    parser.add_argument("--anilist-limit", type=int, default=None, help="Maximum live AniList calls when --live-anilist is used.")
    parser.add_argument("--anilist-sleep", type=float, default=1.0, help="Seconds to sleep between live AniList calls.")
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

    if args.live_anilist:
        anilist_cache = load_anilist_cache()
        if seed_anilist_cache_from_sample(anilist_cache):
            save_anilist_cache(anilist_cache)
        cached_ids = {int(key) for key in anilist_cache.get("items", {}) if str(key).isdigit()}
        anilist_candidates = (
            df.loc[~df["mal_id"].astype(int).isin(cached_ids)]
            .sort_values("popularity", na_position="last")["mal_id"]
            .dropna()
            .astype(int)
            .drop_duplicates()
            .tolist()
        )
        updated = update_anilist_cache_for_mal_ids(
            anilist_candidates,
            anilist_cache,
            limit=args.anilist_limit,
            sleep_seconds=args.anilist_sleep,
        )
        print(f"Live AniList cache updates: {updated}")

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
