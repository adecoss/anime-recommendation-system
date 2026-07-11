from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from anidb_metadata_utils import parse_int
except ImportError:
    from src.anidb_metadata_utils import parse_int

try:
    from anilist_metadata_utils import classify_anilist_media, merge_recommendation_sources
except ImportError:
    from src.anilist_metadata_utils import classify_anilist_media, merge_recommendation_sources


ROOT = Path(__file__).resolve().parents[1]
RAW_SOURCE_DIR = ROOT / "data" / "raw_sources"
MAL_RAW_DIR = RAW_SOURCE_DIR / "mal_jikan"
ANILIST_RAW_DIR = RAW_SOURCE_DIR / "anilist"
ANIDB_RAW_DIR = RAW_SOURCE_DIR / "anidb"
PROCESSED_DIR = ROOT / "data" / "processed"
BUILD_DIR = ROOT / "data" / "build"

JIKAN_ANIME_CACHE_FILE = MAL_RAW_DIR / "jikan_anime_full_cache.json"
JIKAN_RECOMMENDATION_CACHE_FILE = MAL_RAW_DIR / "jikan_recommendation_cache.json"
JIKAN_CHARACTER_CACHE_FILE = MAL_RAW_DIR / "jikan_character_voice_actor_cache.json"
ANILIST_RAW_CACHE_FILE = ANILIST_RAW_DIR / "anilist_media_cache.json"
ANIDB_RAW_CACHE_FILE = ANIDB_RAW_DIR / "anidb_metadata_cache.json"
RAW_SOURCE_INDEX_CSV = RAW_SOURCE_DIR / "raw_source_index.csv"

OUTPUT_CSV = PROCESSED_DIR / "anime_dataset.csv"
OUTPUT_JSON = PROCESSED_DIR / "anime_dataset.json"
DISCREPANCY_CSV = BUILD_DIR / "dataset_source_discrepancies.csv"
BUILD_SUMMARY_FILE = BUILD_DIR / "dataset_build_summary.json"

VALID_TYPES = {"TV", "Movie", "OVA", "ONA", "Special", "TV Special"}
NOT_YET_AIRED_STATUS = "Not yet aired"
ANIDB_SIMILAR_MIN_APPROVAL_RATIO = 0.25
MAX_CHARACTER_LABELS = 30
MAX_VOICE_ACTOR_LABELS = 30
LOW_SIGNAL_FAVORITE_THRESHOLD = 10
LOW_SIGNAL_MEMBER_THRESHOLD = 2500
RECAP_KEEP_RUNTIME_THRESHOLD = 60
RECAP_COMPLETE_COLUMNS = ["genres", "tags", "episodes", "duration", "rating", "studios"]
CLOSE_RELATIONS = {"Sequel", "Prequel", "Side Story", "Parent Story", "Summary", "Full Story"}
INHERITABLE_FIELDS = ["genres", "tags", "tag_weights", "explicit_tags", "explicit_tag_weights", "demographics", "rating", "studios"]
RATING_VIOLENCE_TAGS = {
    "gore",
    "violence",
    "torture",
    "body horror",
    "war",
    "blood",
    "crime",
}
RECAP_TITLE_PATTERNS = {
    "recap",
    "recaps",
    "manner movie",
    "manner movies",
    "digest",
    "summary",
    "summaries",
    "compilation",
    "soushuu",
    "soushuuhen",
    "soushuu-hen",
    "kaisou",
    "kaisouroku",
    "climax chokuzen",
}
RECAP_DESCRIPTION_PATTERNS = {
    "recap",
    "digest",
    "summary",
    "summarizes",
    "summarise",
    "summarize",
    "compilation",
    "compiles",
    "retelling",
    "recapitulat",
}

MAL_GENRE_TO_ANILIST = {
    "Action": "Action",
    "Adventure": "Adventure",
    "Avant Garde": None,
    "Award Winning": None,
    "Boys Love": None,
    "Comedy": "Comedy",
    "Drama": "Drama",
    "Ecchi": "Ecchi",
    "Erotica": None,
    "Fantasy": "Fantasy",
    "Girls Love": None,
    "Gourmet": None,
    "Hentai": "Hentai",
    "Horror": "Horror",
    "Mystery": "Mystery",
    "Romance": "Romance",
    "Sci-Fi": "Sci-Fi",
    "Slice of Life": "Slice of Life",
    "Sports": "Sports",
    "Supernatural": "Supernatural",
    "Suspense": "Thriller",
}

MAL_THEME_TO_ANILIST_TAG = {
    "Adult Cast": ("Primarily Adult Cast", 88),
    "Anthropomorphic": ("Anthropomorphism", 86),
    "CGDCT": ("Cute Girls Doing Cute Things", 92),
    "Childcare": ("Family Life", 80),
    "Combat Sports": ("Martial Arts", 84),
    "Crossdressing": ("Crossdressing", 88),
    "Delinquents": ("Delinquents", 84),
    "Detective": ("Detective", 90),
    "Educational": ("Educational", 85),
    "Gag Humor": ("Slapstick", 83),
    "Gore": ("Gore", 95),
    "Harem": ("Female Harem", 88),
    "High Stakes Game": ("Gambling", 82),
    "Historical": ("Historical", 92),
    "Idols (Female)": ("Idol", 87),
    "Idols (Male)": ("Idol", 87),
    "Isekai": ("Isekai", 94),
    "Iyashikei": ("Iyashikei", 90),
    "Love Polygon": ("Love Triangle", 80),
    "Love Status Quo": ("Unrequited Love", 76),
    "Magical Sex Shift": ("Gender Bending", 85),
    "Mahou Shoujo": ("Magic", 92),
    "Martial Arts": ("Martial Arts", 94),
    "Mecha": ("Real Robot", 90),
    "Medical": ("Medicine", 82),
    "Military": ("Military", 92),
    "Music": ("Music", 92),
    "Mythology": ("Mythology", 90),
    "Organized Crime": ("Crime", 85),
    "Otaku Culture": ("Otaku Culture", 86),
    "Parody": ("Parody", 90),
    "Performing Arts": ("Acting", 80),
    "Pets": ("Animals", 80),
    "Psychological": ("Psychological", 92),
    "Racing": ("Cars", 82),
    "Reincarnation": ("Reincarnation", 92),
    "Reverse Harem": ("Male Harem", 88),
    "Samurai": ("Samurai", 92),
    "School": ("School", 90),
    "Showbiz": ("Acting", 80),
    "Space": ("Space", 90),
    "Strategy Game": ("Board Game", 82),
    "Super Power": ("Super Power", 92),
    "Survival": ("Survival", 90),
    "Team Sports": ("Primarily Male Cast", 75),
    "Time Travel": ("Time Manipulation", 92),
    "Urban Fantasy": ("Urban Fantasy", 92),
    "Vampire": ("Vampire", 92),
    "Video Game": ("Video Games", 90),
    "Villainess": ("Villainess", 90),
    "Visual Arts": ("Drawing", 80),
    "Workplace": ("Work", 80),
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

ANILIST_FORMAT_TO_MAL_TYPE = {
    "TV": "TV",
    "TV_SHORT": "TV",
    "MOVIE": "Movie",
    "SPECIAL": "Special",
    "OVA": "OVA",
    "ONA": "ONA",
    "MUSIC": "Special",
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


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    last_error: PermissionError | None = None
    for attempt in range(8):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(2.0, 0.15 * (attempt + 1)))

    pending = path.with_suffix(path.suffix + f".pending_{time.time_ns()}.json")
    try:
        tmp.replace(pending)
        print(
            f"[WARN] Could not replace locked JSON file {path}; wrote pending copy {pending}: {last_error}",
            flush=True,
        )
    except Exception as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[WARN] Could not persist JSON file {path}: {exc}", flush=True)


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_cache(path: Path) -> dict[str, Any]:
    payload = load_json(path, {"items": {}})
    payload.setdefault("items", {})
    return payload


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def merge_pipe(values: list[Any]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        if is_missing(value):
            continue
        for part in str(value).split("|"):
            clean = part.strip()
            key = clean.casefold()
            if clean and key not in seen:
                seen.add(key)
                out.append(clean)
    return "|".join(out)


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
    values = []
    for part in split_multi_value(value):
        values.append(DEMOGRAPHIC_NAMES.get(part.casefold(), part))
    return merge_pipe(values)


def parse_duration_minutes(value: Any) -> int | None:
    if is_missing(value):
        return None
    text = str(value)
    hours = 0
    minutes = 0
    h_match = re.search(r"(\d+)\s*hr", text)
    m_match = re.search(r"(\d+)\s*min", text)
    if h_match:
        hours = int(h_match.group(1))
    if m_match:
        minutes = int(m_match.group(1))
    if hours or minutes:
        return hours * 60 + minutes
    number = re.search(r"\d+", text)
    return int(number.group(0)) if number else None


def date_parts(data: dict[str, Any]) -> tuple[int | None, int | None]:
    aired = data.get("aired") or {}
    prop = aired.get("prop") or {}
    from_date = prop.get("from") or {}
    year = from_date.get("year") or data.get("year")
    month = from_date.get("month")
    return parse_int(year, default=None), parse_int(month, default=None)


def season_from_month(month: int | None) -> str:
    if month in {1, 2, 3}:
        return "winter"
    if month in {4, 5, 6}:
        return "spring"
    if month in {7, 8, 9}:
        return "summer"
    if month in {10, 11, 12}:
        return "fall"
    return ""


def list_names(items: Any) -> str:
    if not isinstance(items, list):
        return ""
    return merge_pipe([item.get("name") for item in items if isinstance(item, dict)])


def mal_genre_fallback(data: dict[str, Any]) -> str:
    names = split_multi_value(merge_pipe([list_names(data.get("genres")), list_names(data.get("explicit_genres"))]))
    mapped = []
    for name in names:
        label = MAL_GENRE_TO_ANILIST.get(name)
        if label:
            mapped.append(label)
    return merge_pipe(mapped)


def mal_theme_fallback(data: dict[str, Any]) -> tuple[str, str]:
    names = split_multi_value(list_names(data.get("themes")))
    tags = []
    weights = []
    for name in names:
        mapped = MAL_THEME_TO_ANILIST_TAG.get(name)
        if not mapped:
            continue
        label, weight = mapped
        tags.append(label)
        weights.append(f"{label}:{weight}")
    return merge_pipe(tags), merge_pipe(weights)


def parse_anidb_id(data: dict[str, Any]) -> int | None:
    direct_id = data.get("anidb_id")
    if direct_id is not None:
        try:
            return int(direct_id)
        except (TypeError, ValueError):
            pass
    for item in data.get("external") or []:
        if item.get("name") == "AniDB":
            match = re.search(r"(?:aid=|/anime/)(\d+)", str(item.get("url") or ""))
            if match:
                return int(match.group(1))
    return None


def accepted_jikan_data(data: dict[str, Any]) -> bool:
    if data.get("type") not in VALID_TYPES:
        return False
    if data.get("status") == NOT_YET_AIRED_STATUS:
        return False
    if data.get("status") != NOT_YET_AIRED_STATUS and data.get("score") is None:
        return False
    return True


def jikan_recommendations(cache_item: dict[str, Any] | None) -> str:
    response = (cache_item or {}).get("response") or {}
    edges = []
    for item in response.get("data") or []:
        entry = item.get("entry") or {}
        mal_id = item.get("mal_id") or entry.get("mal_id")
        votes = item.get("votes") or 1
        try:
            mal_id = int(mal_id)
            votes = max(1, int(votes))
        except (TypeError, ValueError):
            continue
        edges.append(f"{mal_id}:{votes}")
    return merge_pipe(edges)


def jikan_relations(data: dict[str, Any]) -> str:
    edges = []
    for group in data.get("relations") or []:
        relation = group.get("relation")
        for entry in group.get("entry") or []:
            if entry.get("type") != "anime":
                continue
            mal_id = entry.get("mal_id")
            try:
                mal_id = int(mal_id)
            except (TypeError, ValueError):
                continue
            edges.append(f"{mal_id}:{relation}")
    return merge_pipe(edges)


def character_label(character: dict[str, Any]) -> str:
    character_id = character.get("id")
    name = str(character.get("name") or "").strip()
    role = str(character.get("role") or "").strip()
    parts = [str(character_id) if character_id is not None else "", role, name]
    return ":".join(part for part in parts if part)


def voice_actor_label(actor: dict[str, Any]) -> str:
    actor_id = actor.get("id")
    name = str(actor.get("name") or "").strip()
    language = str(actor.get("language") or "").strip()
    parts = [str(actor_id) if actor_id is not None else "", language, name]
    return ":".join(part for part in parts if part)


def format_characters(characters: Any) -> str:
    if not isinstance(characters, list):
        return ""
    return merge_pipe([character_label(item) for item in characters[:MAX_CHARACTER_LABELS] if isinstance(item, dict)])


def format_voice_actors(characters: Any) -> str:
    if not isinstance(characters, list):
        return ""
    actors: list[str] = []
    for character in characters:
        if not isinstance(character, dict):
            continue
        for actor in character.get("voice_actors") or []:
            if isinstance(actor, dict):
                actors.append(voice_actor_label(actor))
            if len(merge_pipe(actors).split("|")) >= MAX_VOICE_ACTOR_LABELS:
                return merge_pipe(actors)
    return merge_pipe(actors)


def count_voice_actors(characters: Any) -> int:
    if not isinstance(characters, list):
        return 0
    actor_ids: set[str] = set()
    actor_names: set[str] = set()
    for character in characters:
        if not isinstance(character, dict):
            continue
        for actor in character.get("voice_actors") or []:
            if not isinstance(actor, dict):
                continue
            if actor.get("id") is not None:
                actor_ids.add(str(actor.get("id")))
            elif actor.get("name"):
                actor_names.add(str(actor.get("name")))
    return len(actor_ids) + len(actor_names)


def anidb_demographics(metadata: dict[str, Any] | None) -> str:
    weighted_values: list[tuple[str, int]] = []
    for tag in (metadata or {}).get("raw_tags") or []:
        key = str(tag.get("name") or "").strip().casefold()
        if key in DEMOGRAPHIC_NAMES:
            weighted_values.append((DEMOGRAPHIC_NAMES[key], parse_int(tag.get("weight"), default=0) or 0))
    if not weighted_values:
        return ""
    max_weight = max(weight for _, weight in weighted_values)
    return merge_pipe([value for value, weight in weighted_values if weight == max_weight])


def anidb_content_indicators(metadata: dict[str, Any] | None) -> str:
    indicators = {"nudity", "sex", "violence"}
    values = []
    for tag in (metadata or {}).get("raw_tags") or []:
        name = str(tag.get("name") or "").strip()
        if name.casefold() in indicators:
            values.append(name)
    return merge_pipe(values)


def anidb_loli_weight(metadata: dict[str, Any] | None) -> str:
    for tag in (metadata or {}).get("raw_tags") or []:
        if str(tag.get("name") or "").strip().casefold() == "loli":
            weight = parse_int(tag.get("weight"), default=600) or 600
            return f"Loli:{max(1, min(100, round(weight / 6)))}"
    return ""


def anidb_episode_count(metadata: dict[str, Any] | None) -> int | None:
    return parse_int((metadata or {}).get("episode_count"), default=None)


def anilist_next_episode_number(media: dict[str, Any] | None) -> int | None:
    next_airing = (media or {}).get("next_airing_episode")
    if isinstance(next_airing, dict):
        return parse_int(next_airing.get("episode"), default=None)
    return parse_int(next_airing, default=None)


def anidb_duration(metadata: dict[str, Any] | None) -> int | None:
    summary = (metadata or {}).get("episode_summary") or {}
    return parse_int(
        summary.get("preferred_average_length_minutes")
        or summary.get("regular_average_length_minutes")
        or summary.get("average_length_minutes"),
        default=None,
    )


def anidb_studios(metadata: dict[str, Any] | None) -> str:
    return merge_pipe((metadata or {}).get("animation_work_studios") or [])


def build_anidb_to_mal_map(rows: list[dict[str, Any]]) -> dict[int, int]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("anidb_id"):
            grouped[int(row["anidb_id"])].append(row)
    out = {}
    for anidb_id, items in grouped.items():
        best = sorted(
            items,
            key=lambda item: (
                item.get("members") or 0,
                item.get("scored_by") or 0,
                -(item.get("mal_id") or 0),
            ),
            reverse=True,
        )[0]
        out[anidb_id] = int(best["mal_id"])
    return out


def anidb_similar_recommendations(metadata: dict[str, Any] | None, anidb_to_mal: dict[int, int]) -> str:
    edges = []
    for item in (metadata or {}).get("similar_anime") or []:
        anidb_id = parse_int(item.get("anidb_id"), default=None)
        if anidb_id is None or anidb_id not in anidb_to_mal:
            continue
        approval = parse_int(item.get("approval"), default=0) or 0
        total = parse_int(item.get("total"), default=0) or 0
        if total <= 0 or approval / total < ANIDB_SIMILAR_MIN_APPROVAL_RATIO:
            continue
        edges.append(f"{anidb_to_mal[anidb_id]}:{approval}")
    return merge_pipe(edges)


def parse_relation_edges(value: Any) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for part in split_multi_value(value):
        if ":" not in part:
            continue
        mal_text, relation = part.split(":", 1)
        mal_id = parse_int(mal_text, default=None)
        if mal_id is not None:
            edges.append((int(mal_id), relation.strip()))
    return edges


def normalize_type_label(value: Any) -> Any:
    if is_missing(value):
        return value
    text = str(value).strip()
    if text.casefold() in {"special", "tv special"}:
        return "Special"
    return value


def text_contains_pattern(text: Any, patterns: set[str]) -> list[str]:
    haystack = re.sub(r"\s+", " ", str(text or "").casefold())
    return sorted(pattern for pattern in patterns if pattern in haystack)


def recap_reasons(data: dict[str, Any], relations: str) -> str:
    reasons: list[str] = []
    title_text = " ".join(
        str(data.get(field) or "")
        for field in ["title", "title_english", "title_japanese", "title_synonyms"]
    )
    title_hits = text_contains_pattern(title_text, RECAP_TITLE_PATTERNS)
    if title_hits:
        reasons.extend(f"title:{hit}" for hit in title_hits)

    relation_names = {relation for _, relation in parse_relation_edges(relations)}
    if "Full Story" in relation_names:
        reasons.append("relation:Full Story")

    description_text = " ".join(str(data.get(field) or "") for field in ["synopsis", "background"])
    description_hits = text_contains_pattern(description_text, RECAP_DESCRIPTION_PATTERNS)
    is_regular_tv_without_other_signal = str(data.get("type") or "").casefold() == "tv" and not reasons
    if description_hits and not is_regular_tv_without_other_signal:
        reasons.extend(f"description:{hit}" for hit in description_hits)

    return merge_pipe(reasons)


def has_core_metadata(row: pd.Series) -> bool:
    return all(not is_missing(row.get(field)) for field in RECAP_COMPLETE_COLUMNS)


def recap_should_be_deleted(row: pd.Series) -> bool:
    if is_missing(row.get("recap_reason")):
        return False
    if "manner movie" in str(row.get("recap_reason") or "").casefold():
        return True
    runtime = parse_int(row.get("total_watch_minutes"), default=0) or 0
    if str(row.get("type") or "").casefold() == "movie" and runtime >= RECAP_KEEP_RUNTIME_THRESHOLD:
        return False
    if has_core_metadata(row):
        return False
    return True


def close_relative_ids(row: dict[str, Any] | pd.Series, valid_ids: set[int]) -> list[int]:
    ids = []
    for target_id, relation in parse_relation_edges(row.get("relations")):
        if relation in CLOSE_RELATIONS and target_id in valid_ids:
            ids.append(target_id)
    return ids


def adjust_split_anilist_episode_counts(base_rows: list[dict[str, Any]]) -> int:
    """AniList sometimes aggregates MAL-split parts into one media row.

    If the AniList episode count equals the sum of MAL episodes across close
    related parts, keep the per-MAL episode count for source comparison.
    """
    adjusted = 0
    by_mal = {int(row["mal_id"]): row for row in base_rows}

    grouped_by_anilist: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in base_rows:
        anilist_id = parse_int(row.get("anilist_id"), default=None)
        if anilist_id is not None:
            grouped_by_anilist[int(anilist_id)].append(row)
    for group in grouped_by_anilist.values():
        if len(group) < 2:
            continue
        anilist_episode = parse_int(group[0].get("episodes_anilist"), default=None)
        mal_total = sum(parse_int(row.get("episodes_mal"), default=0) or 0 for row in group)
        if anilist_episode and mal_total == anilist_episode:
            for row in group:
                if row.get("episodes_mal") and row.get("episodes_anilist") != row.get("episodes_mal"):
                    row["episodes_anilist"] = row.get("episodes_mal")
                    adjusted += 1

    valid_ids = set(by_mal)
    for row in base_rows:
        mal_episode = parse_int(row.get("episodes_mal"), default=None)
        anilist_episode = parse_int(row.get("episodes_anilist"), default=None)
        if not mal_episode or not anilist_episode or anilist_episode <= mal_episode:
            continue
        current_anilist_id = parse_int(row.get("anilist_id"), default=None)
        relatives = []
        for rel_id in close_relative_ids(row, valid_ids):
            rel = by_mal[rel_id]
            rel_anilist_id = parse_int(rel.get("anilist_id"), default=None)
            if rel_anilist_id is None or rel_anilist_id == current_anilist_id:
                relatives.append(rel)
        related_total = mal_episode + sum(parse_int(rel.get("episodes_mal"), default=0) or 0 for rel in relatives)
        if related_total == anilist_episode:
            row["episodes_anilist"] = mal_episode
            adjusted += 1
    return adjusted


def text_has_any(value: Any, terms: set[str]) -> bool:
    text = f"|{str(value or '').casefold()}|"
    return any(f"|{term.casefold()}|" in text for term in terms)


def infer_demographics(row: dict[str, Any]) -> str:
    if not is_missing(row.get("demographics")):
        return str(row.get("demographics"))
    rating = str(row.get("rating") or "").casefold()
    genres = row.get("genres")
    tags = row.get("tags")
    explicit_tags = row.get("explicit_tags")
    combined = merge_pipe([genres, tags, explicit_tags])
    if "rx" in rating or text_has_any(genres, {"Hentai", "Erotica"}) or not is_missing(explicit_tags):
        return "18+"
    for demographic in ["Josei", "Kodomo", "Seinen", "Shoujo", "Shounen"]:
        if text_has_any(combined, {demographic}):
            return demographic
    if "g - all ages" in rating and text_has_any(combined, {"Primarily Child Cast", "Educational", "Animals"}):
        return "Kodomo"
    return ""


def infer_rating(row: dict[str, Any]) -> str:
    if not is_missing(row.get("rating")):
        return str(row.get("rating"))
    genres = row.get("genres")
    tags = row.get("tags")
    explicit_tags = row.get("explicit_tags")
    demographics = row.get("demographics")
    combined = merge_pipe([genres, tags, explicit_tags])
    if text_has_any(genres, {"Hentai"}) or text_has_any(demographics, {"18+"}) and text_has_any(genres, {"Hentai"}):
        return "Rx - Hentai"
    if text_has_any(genres, {"Erotica"}) or not is_missing(explicit_tags):
        return "R+ - Mild Nudity"
    if text_has_any(combined, RATING_VIOLENCE_TAGS):
        return "R - 17+ (violence & profanity)"
    if text_has_any(demographics, {"Kodomo"}):
        return "G - All Ages"
    if text_has_any(demographics, {"Shounen", "Shoujo", "Seinen", "Josei"}):
        return "PG-13 - Teens 13 or older"
    return ""


def inherit_from_close_relatives(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    by_id = {int(row.mal_id): idx for idx, row in df[["mal_id"]].iterrows()}
    valid_ids = set(by_id)
    inherited_counts = {field: 0 for field in INHERITABLE_FIELDS}

    reverse_relatives: dict[int, set[int]] = defaultdict(set)
    for row in df.itertuples(index=False):
        source_id = int(row.mal_id)
        for target_id, relation in parse_relation_edges(getattr(row, "relations", "")):
            if relation in CLOSE_RELATIONS and target_id in valid_ids:
                reverse_relatives[target_id].add(source_id)

    for idx, row in df.iterrows():
        related_ids = set(close_relative_ids(row, valid_ids)) | reverse_relatives.get(int(row["mal_id"]), set())
        if not related_ids:
            continue
        related_rows = (
            df.loc[[by_id[rel_id] for rel_id in related_ids]]
            .sort_values(["members", "favorites"], ascending=[False, False], na_position="last")
        )
        for field in INHERITABLE_FIELDS:
            if field not in df.columns or not is_missing(df.at[idx, field]):
                continue
            donor = related_rows.loc[~related_rows[field].apply(is_missing), field] if field in related_rows.columns else pd.Series(dtype=object)
            if not donor.empty:
                df.at[idx, field] = donor.iloc[0]
                inherited_counts[field] += 1

    inferred_demographics = 0
    inferred_ratings = 0
    for idx, row in df.iterrows():
        demographic = infer_demographics(row.to_dict())
        if demographic and is_missing(df.at[idx, "demographics"]):
            df.at[idx, "demographics"] = demographic
            inferred_demographics += 1
        rating = infer_rating(df.loc[idx].to_dict())
        if rating and is_missing(df.at[idx, "rating"]):
            df.at[idx, "rating"] = rating
            inferred_ratings += 1

    inherited_counts["demographics_inferred"] = inferred_demographics
    inherited_counts["rating_inferred"] = inferred_ratings
    return df, inherited_counts


def has_close_metadata_relative(row: pd.Series, df: pd.DataFrame, by_id: dict[int, int], reverse_relatives: dict[int, set[int]]) -> bool:
    related_ids = set(close_relative_ids(row, set(by_id))) | reverse_relatives.get(int(row["mal_id"]), set())
    for rel_id in related_ids:
        rel = df.loc[by_id[rel_id]]
        if not is_missing(rel.get("genres")) or not is_missing(rel.get("tags")):
            return True
    return False


def selected_value(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            if isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0:
                continue
            return value
    return None


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
        parts = sorted(part.strip().casefold() for part in split_multi_value(text))
        return "|".join(parts)
    return re.sub(r"\s+", " ", text).casefold()


def choose_consensus(
    mal_value: Any = None,
    anilist_value: Any = None,
    anidb_value: Any = None,
    *,
    field: str | None = None,
) -> Any:
    ordered = [("mal", mal_value), ("anilist", anilist_value), ("anidb", anidb_value)]
    non_missing = [(source, value, choice_key(value, field)) for source, value in ordered]
    non_missing = [(source, value, key) for source, value, key in non_missing if key is not None]
    if not non_missing:
        return None

    counts: dict[str, int] = defaultdict(int)
    for _, _, key in non_missing:
        counts[key] += 1
    majority_keys = {key for key, count in counts.items() if count >= 2}
    for _, value, key in non_missing:
        if key in majority_keys:
            return value
    return non_missing[0][1]


def reorder_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    ordered = [
        "mal_id",
        "anilist_id",
        "anidb_id",
        "url",
        "image_url",
        "title",
        "title_english",
        "type",
        "type_mal",
        "type_anilist",
        "is_recap_like",
        "recap_reason",
        "recap_action",
        "source",
        "source_mal",
        "source_anilist",
        "episodes",
        "episodes_mal",
        "episodes_anilist",
        "episodes_anidb",
        "duration",
        "duration_mal",
        "duration_anilist",
        "duration_anidb",
        "total_watch_minutes",
        "status",
        "rating",
        "score",
        "scored_by",
        "rank",
        "popularity",
        "members",
        "favorites",
        "aired_year",
        "aired_year_mal",
        "aired_year_anilist",
        "aired_month",
        "aired_month_mal",
        "aired_month_anilist",
        "season",
        "season_mal",
        "season_anilist",
        "season_from_month",
        "genres",
        "tags",
        "tag_weights",
        "explicit_tags",
        "explicit_tag_weights",
        "demographics",
        "demographics_mal",
        "demographics_anilist",
        "demographics_anidb",
        "studios",
        "studios_mal",
        "studios_anilist",
        "studios_anidb",
        "characters",
        "characters_jikan",
        "characters_anilist",
        "voice_actors",
        "voice_actors_jikan",
        "voice_actors_anilist",
        "character_count",
        "character_count_jikan",
        "character_count_anilist",
        "voice_actor_count",
        "voice_actor_count_jikan",
        "voice_actor_count_anilist",
        "relations",
        "recommendations",
        "recommendations_jikan",
        "recommendations_anilist",
        "recommendations_anidb",
    ]
    present = [column for column in ordered if column in df.columns]
    rest = [column for column in df.columns if column not in present]
    return df[present + rest]


def build_rows() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    jikan_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    rec_cache = load_cache(JIKAN_RECOMMENDATION_CACHE_FILE)
    character_cache = load_cache(JIKAN_CHARACTER_CACHE_FILE)
    anilist_cache = load_cache(ANILIST_RAW_CACHE_FILE)
    anidb_cache = load_cache(ANIDB_RAW_CACHE_FILE)

    base_rows: list[dict[str, Any]] = []
    source_payloads: dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]] = {}

    for key, item in jikan_cache.get("items", {}).items():
        response = item.get("response") or {}
        data = response.get("data") or {}
        if not accepted_jikan_data(data):
            continue
        mal_id = int(key)
        anidb_id = parse_anidb_id(data)
        anilist_entry = anilist_cache.get("items", {}).get(key, {})
        anilist_media = anilist_entry.get("media") or None
        jikan_characters = (character_cache.get("items", {}).get(key, {}) or {}).get("characters") or []
        anilist_characters = (anilist_media or {}).get("characters") or []
        anidb_item = anidb_cache.get("items", {}).get(str(anidb_id), {}) if anidb_id else {}
        anidb_metadata = anidb_item.get("metadata") if isinstance(anidb_item, dict) and "metadata" in anidb_item else anidb_item

        year, month = date_parts(data)
        relations = jikan_relations(data)
        row = {
            "mal_id": mal_id,
            "anilist_id": (anilist_media or {}).get("id"),
            "anidb_id": anidb_id,
            "url": data.get("url"),
            "image_url": (((data.get("images") or {}).get("jpg") or {}).get("large_image_url")),
            "title": data.get("title"),
            "title_english": data.get("title_english"),
            "type_mal": data.get("type"),
            "type_anilist": ANILIST_FORMAT_TO_MAL_TYPE.get(str((anilist_media or {}).get("format") or ""), ""),
            "source_mal": data.get("source"),
            "source_anilist": normalize_source_label((anilist_media or {}).get("source")),
            "episodes_mal": parse_int(data.get("episodes"), default=None),
            "episodes_anilist": parse_int((anilist_media or {}).get("episodes"), default=None)
            or (
                anilist_next_episode_number(anilist_media) - 1
                if anilist_next_episode_number(anilist_media)
                else None
            ),
            "episodes_anidb": anidb_episode_count(anidb_metadata),
            "duration_mal": parse_duration_minutes(data.get("duration")),
            "duration_anilist": parse_int((anilist_media or {}).get("duration"), default=None),
            "duration_anidb": anidb_duration(anidb_metadata),
            "status": data.get("status"),
            "rating": data.get("rating"),
            "score": data.get("score"),
            "scored_by": data.get("scored_by"),
            "rank": data.get("rank"),
            "popularity": data.get("popularity"),
            "members": data.get("members"),
            "favorites": data.get("favorites"),
            "aired_year_mal": year,
            "aired_month_mal": month,
            "aired_year_anilist": ((anilist_media or {}).get("startDate") or {}).get("year"),
            "aired_month_anilist": ((anilist_media or {}).get("startDate") or {}).get("month"),
            "season_mal": str(data.get("season") or "").lower(),
            "season_anilist": str((anilist_media or {}).get("season") or "").lower(),
            "season_from_month": season_from_month(month),
            "demographics_mal": normalize_demographics(list_names(data.get("demographics"))),
            "demographics_anidb": anidb_demographics(anidb_metadata),
            "studios_mal": list_names(data.get("studios")),
            "studios_anilist": merge_pipe((anilist_media or {}).get("studios") or []),
            "studios_anidb": anidb_studios(anidb_metadata),
            "characters_jikan": format_characters(jikan_characters),
            "voice_actors_jikan": format_voice_actors(jikan_characters),
            "character_count_jikan": len(jikan_characters),
            "voice_actor_count_jikan": count_voice_actors(jikan_characters),
            "characters_anilist": format_characters(anilist_characters),
            "voice_actors_anilist": format_voice_actors(anilist_characters),
            "character_count_anilist": len(anilist_characters),
            "voice_actor_count_anilist": count_voice_actors(anilist_characters),
            "relations": relations,
            "recap_reason": recap_reasons(data, relations),
        }
        base_rows.append(row)
        source_payloads[mal_id] = (data, anilist_media or {}, anidb_metadata)

    anilist_split_episode_repairs = adjust_split_anilist_episode_counts(base_rows)
    anidb_to_mal = build_anidb_to_mal_map(base_rows)
    rows = []
    discrepancy_rows = []
    dropped_no_episode_count = 0

    for base in base_rows:
        mal_id = int(base["mal_id"])
        data, anilist_media, anidb_metadata = source_payloads[mal_id]
        anidb_indicators = anidb_content_indicators(anidb_metadata)
        classifier_seed = pd.Series(
            {
                "genres": list_names(data.get("genres")) + "|" + list_names(data.get("explicit_genres")),
                "rating": data.get("rating"),
                "tags": anidb_indicators,
                "explicit_tags": "",
                "explicit_tag_weights": anidb_loli_weight(anidb_metadata),
            }
        )
        labels = classify_anilist_media(classifier_seed, anilist_media)
        mal_fallback_genres = mal_genre_fallback(data)
        mal_fallback_tags, mal_fallback_tag_weights = mal_theme_fallback(data)
        final_genres = labels.get("genres", "") or mal_fallback_genres
        final_tags = labels.get("tags", "") or mal_fallback_tags
        final_tag_weights = labels.get("tag_weights", "") or mal_fallback_tag_weights

        jikan_recs = jikan_recommendations(rec_cache.get("items", {}).get(str(mal_id)))
        anilist_recs = labels.get("recommendations", "")
        anidb_recs = anidb_similar_recommendations(anidb_metadata, anidb_to_mal)

        chosen_type = normalize_type_label(choose_consensus(base.get("type_mal"), base.get("type_anilist"), field="type"))
        chosen_source = choose_consensus(base.get("source_mal"), base.get("source_anilist"), field="source")
        chosen_episodes = choose_consensus(base.get("episodes_mal"), base.get("episodes_anilist"), base.get("episodes_anidb"))
        chosen_duration = choose_consensus(base.get("duration_mal"), base.get("duration_anilist"), base.get("duration_anidb"))
        chosen_aired_year = choose_consensus(base.get("aired_year_mal"), base.get("aired_year_anilist"))
        chosen_aired_month = choose_consensus(base.get("aired_month_mal"), base.get("aired_month_anilist"))
        chosen_season = choose_consensus(base.get("season_mal"), base.get("season_anilist"), base.get("season_from_month"))
        chosen_demographics = choose_consensus(
            base.get("demographics_mal"),
            normalize_demographics(labels.get("demographics")),
            base.get("demographics_anidb"),
        )
        chosen_studios = choose_consensus(
            base.get("studios_mal"),
            base.get("studios_anilist"),
            base.get("studios_anidb"),
            field="studios",
        )
        chosen_characters = choose_consensus(base.get("characters_jikan"), base.get("characters_anilist"))
        chosen_voice_actors = choose_consensus(base.get("voice_actors_jikan"), base.get("voice_actors_anilist"))
        chosen_character_count = choose_consensus(base.get("character_count_jikan"), base.get("character_count_anilist"))
        chosen_voice_actor_count = choose_consensus(base.get("voice_actor_count_jikan"), base.get("voice_actor_count_anilist"))

        if not chosen_episodes:
            dropped_no_episode_count += 1
            continue

        row = dict(base)
        row.update(
            {
                "type": chosen_type,
                "is_recap_like": bool(not is_missing(base.get("recap_reason"))),
                "recap_reason": base.get("recap_reason"),
                "recap_action": "",
                "source": chosen_source,
                "episodes": chosen_episodes,
                "duration": chosen_duration,
                "aired_year": chosen_aired_year,
                "aired_month": chosen_aired_month,
                "season": chosen_season,
                "genres": final_genres,
                "tags": final_tags,
                "tag_weights": final_tag_weights,
                "explicit_tags": labels.get("explicit_tags", ""),
                "explicit_tag_weights": labels.get("explicit_tag_weights", ""),
                "demographics_anilist": normalize_demographics(labels.get("demographics", "")),
                "demographics": chosen_demographics,
                "studios": chosen_studios,
                "characters": chosen_characters,
                "voice_actors": chosen_voice_actors,
                "character_count": chosen_character_count,
                "voice_actor_count": chosen_voice_actor_count,
                "recommendations_jikan": jikan_recs,
                "recommendations_anilist": anilist_recs,
                "recommendations_anidb": anidb_recs,
                "recommendations": merge_recommendation_sources(jikan_recs, anilist_recs, anidb_recs),
                "total_watch_minutes": (
                    float(chosen_episodes) * float(chosen_duration)
                    if chosen_episodes and chosen_duration
                    else None
                ),
            }
        )
        rows.append(row)

        comparison_values = {
            "type": {
                "mal": base.get("type_mal"),
                "anilist": base.get("type_anilist"),
            },
            "episodes": {
                "mal": base.get("episodes_mal"),
                "anilist": base.get("episodes_anilist"),
                "anidb": base.get("episodes_anidb"),
            },
            "duration": {
                "mal": base.get("duration_mal"),
                "anilist": base.get("duration_anilist"),
                "anidb": base.get("duration_anidb"),
            },
            "source": {
                "mal": base.get("source_mal"),
                "anilist": base.get("source_anilist"),
            },
            "season": {
                "mal": base.get("season_mal"),
                "anilist": base.get("season_anilist"),
                "derived": base.get("season_from_month"),
            },
            "demographics": {
                "mal": base.get("demographics_mal"),
                "anilist": normalize_demographics(labels.get("demographics")),
                "anidb": base.get("demographics_anidb"),
            },
            "studios": {
                "mal": base.get("studios_mal"),
                "anilist": base.get("studios_anilist"),
                "anidb": base.get("studios_anidb"),
            },
        }
        for field, values in comparison_values.items():
            non_missing = {k: str(v).strip() for k, v in values.items() if not is_missing(v)}
            normalized_values = {k: choice_key(v, field) for k, v in values.items() if choice_key(v, field) is not None}
            if len(set(normalized_values.values())) > 1:
                discrepancy_rows.append(
                    {
                        "mal_id": mal_id,
                        "title": base.get("title"),
                        "popularity": base.get("popularity"),
                        "field": field,
                        "selected_value": row.get(field),
                        **{f"{k}_value": v for k, v in values.items()},
                        "unique_raw_values": "|".join(sorted({str(v).strip() for v in non_missing.values() if str(v).strip()})),
                        "unique_normalized_values": "|".join(sorted(set(normalized_values.values()))),
                    }
                )

    df = pd.DataFrame(rows).sort_values(["mal_id"])
    if df.empty:
        raise RuntimeError(
            "No accepted Jikan raw rows were found. Run src/01_gather_raw_sources.py --jikan first, "
            "or verify data/raw_sources/mal_jikan/jikan_anime_full_cache.json."
        )

    df, inherited_counts = inherit_from_close_relatives(df)

    recap_mask = df["recap_reason"].apply(lambda value: not is_missing(value)) if "recap_reason" in df.columns else pd.Series(False, index=df.index)
    delete_recap_mask = df.apply(recap_should_be_deleted, axis=1) if "recap_reason" in df.columns else pd.Series(False, index=df.index)
    dropped_recap_like_rows = int(delete_recap_mask.sum())
    if dropped_recap_like_rows:
        df = df.loc[~delete_recap_mask].copy()
    if "recap_action" in df.columns:
        df.loc[df["recap_reason"].apply(lambda value: not is_missing(value)), "recap_action"] = "flagged_kept_review"
        df.loc[
            df["recap_reason"].apply(lambda value: not is_missing(value))
            & df.apply(has_core_metadata, axis=1),
            "recap_action",
        ] = "flagged_kept_complete_core_metadata"
        df.loc[
            df["recap_reason"].apply(lambda value: not is_missing(value))
            & df["type"].astype(str).str.casefold().eq("movie")
            & df["total_watch_minutes"].fillna(0).ge(RECAP_KEEP_RUNTIME_THRESHOLD)
            ,
            "recap_action",
        ] = "flagged_kept_summary_movie"

    by_id = {int(row.mal_id): idx for idx, row in df[["mal_id"]].iterrows()}
    reverse_relatives: dict[int, set[int]] = defaultdict(set)
    valid_ids = set(by_id)
    for _, row in df.iterrows():
        source_id = int(row["mal_id"])
        for target_id, relation in parse_relation_edges(row.get("relations")):
            if relation in CLOSE_RELATIONS and target_id in valid_ids:
                reverse_relatives[target_id].add(source_id)
    sparse_metadata_mask = df["genres"].apply(is_missing) | df["tags"].apply(is_missing)
    low_signal_mask = df["favorites"].fillna(0).lt(LOW_SIGNAL_FAVORITE_THRESHOLD) & df["members"].fillna(0).lt(
        LOW_SIGNAL_MEMBER_THRESHOLD
    )
    no_metadata_relative_mask = df.apply(
        lambda row: not has_close_metadata_relative(row, df, by_id, reverse_relatives),
        axis=1,
    )
    delete_sparse_low_signal = sparse_metadata_mask & low_signal_mask & no_metadata_relative_mask
    dropped_sparse_low_signal = int(delete_sparse_low_signal.sum())
    if dropped_sparse_low_signal:
        df = df.loc[~delete_sparse_low_signal].copy()

    discrepancy_df = pd.DataFrame(discrepancy_rows)
    if not discrepancy_df.empty:
        final_ids = set(df["mal_id"].astype(int))
        discrepancy_df = discrepancy_df.loc[discrepancy_df["mal_id"].astype(int).isin(final_ids)].copy()
    df = reorder_output_columns(df)
    summary = {
        "updated_at": now_iso(),
        "rows": int(len(df)),
        "raw_jikan_entries": len(jikan_cache.get("items", {})),
        "raw_jikan_character_entries": len(character_cache.get("items", {})),
        "raw_anilist_entries": len(anilist_cache.get("items", {})),
        "raw_anidb_entries": len(anidb_cache.get("items", {})),
        "anilist_cache_used": str(ANILIST_RAW_CACHE_FILE),
        "anidb_cache_used": str(ANIDB_RAW_CACHE_FILE),
        "anilist_rows": int(df["anilist_id"].notna().sum()) if "anilist_id" in df else 0,
        "anidb_rows": int(df["anidb_id"].notna().sum()) if "anidb_id" in df else 0,
        "discrepancies": int(len(discrepancy_df)),
        "anilist_split_episode_repairs": int(anilist_split_episode_repairs),
        "dropped_no_episode_count": int(dropped_no_episode_count),
        "recap_like_rows_detected": int(recap_mask.sum()),
        "dropped_recap_like_rows": dropped_recap_like_rows,
        "kept_recap_like_rows": int(df["recap_reason"].apply(lambda value: not is_missing(value)).sum()) if "recap_reason" in df else 0,
        "dropped_sparse_low_signal_rows": dropped_sparse_low_signal,
        "inherited_or_inferred_fields": inherited_counts,
        "note": "Genres/tags primarily come from AniList; MAL genres/themes fill only blank AniList labels, with MAL Erotica/Hentai and AniDB Loli caveats handled by classifier seed.",
    }
    return df, discrepancy_df, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build anime_dataset from raw source caches.")
    parser.add_argument("--output-csv", type=Path, default=OUTPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=OUTPUT_JSON)
    args = parser.parse_args()

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    df, discrepancy_df, summary = build_rows()
    df.to_csv(args.output_csv, index=False)
    args.output_json.write_text(df.to_json(orient="records", indent=2, force_ascii=False), encoding="utf-8")
    discrepancy_df.to_csv(DISCREPANCY_CSV, index=False)
    summary.update(
        {
            "output_csv": str(args.output_csv),
            "output_json": str(args.output_json),
            "discrepancy_csv": str(DISCREPANCY_CSV),
        }
    )
    atomic_write_json(BUILD_SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
