from __future__ import annotations

import argparse
import json
import math
import re
import time
from collections import defaultdict
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - live enrichment is optional
    requests = None

try:
    from anilist_metadata_utils import (
        ANILIST_MEDIA_CACHE_VERSION,
        get_anilist_media,
        load_anilist_cache,
        request_anilist_media,
        save_anilist_cache,
    )
except ImportError:  # pragma: no cover - imported from notebooks
    from src.anilist_metadata_utils import (
        ANILIST_MEDIA_CACHE_VERSION,
        get_anilist_media,
        load_anilist_cache,
        request_anilist_media,
        save_anilist_cache,
    )


ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = ROOT / "data" / "processed"
BUILD_DIR = ROOT / "data" / "build"
MAL_RAW_DIR = ROOT / "data" / "raw_sources" / "mal_jikan"
ANILIST_RAW_DIR = ROOT / "data" / "raw_sources" / "anilist"
RAW_ANIDB_CACHE = ROOT / "data" / "raw_sources" / "anidb" / "anidb_metadata_cache.json"
ANILIST_REF_DIR = ROOT / "data" / "reference" / "anilist"
ANILIST_GENRES_FILE = ANILIST_REF_DIR / "genres.json"
ANILIST_TAGS_FILE = ANILIST_REF_DIR / "tags.json"
ANIDB_LABEL_MAP_CSV = ANILIST_REF_DIR / "anidb_to_anilist_label_map.csv"
JIKAN_CHARACTER_CACHE_FILE = MAL_RAW_DIR / "jikan_character_voice_actor_cache.json"
JIKAN_PERSON_DETAIL_CACHE_FILE = MAL_RAW_DIR / "jikan_person_detail_cache.json"
JIKAN_CHARACTER_DETAIL_CACHE_FILE = MAL_RAW_DIR / "jikan_character_detail_cache.json"
JIKAN_TOP_CHARACTER_CACHE_FILE = MAL_RAW_DIR / "jikan_top_character_cache.json"
JIKAN_TOP_PEOPLE_CACHE_FILE = MAL_RAW_DIR / "jikan_top_people_cache.json"
ANILIST_TOP_CHARACTER_CACHE_FILE = ANILIST_RAW_DIR / "anilist_top_character_cache.json"
ANILIST_TOP_STAFF_CACHE_FILE = ANILIST_RAW_DIR / "anilist_top_staff_cache.json"
JIKAN_CHARACTER_DETAIL_FAILED_FILE = BUILD_DIR / "jikan_character_detail_failed_requests.json"
ANILIST_CHARACTER_DETAIL_FAILED_FILE = BUILD_DIR / "anilist_character_detail_failed_requests.json"

INPUT_CSV = PROCESSED_DIR / "anime_dataset.csv"
INPUT_JSON = PROCESSED_DIR / "anime_dataset.json"
VOICE_ACTOR_EDGE_CSV = PROCESSED_DIR / "anime_voice_actor_edges.csv"
VOICE_ACTOR_INDEX_CSV = PROCESSED_DIR / "voice_actor_index.csv"
CHARACTER_INDEX_CSV = PROCESSED_DIR / "character_index.csv"
STAFF_EDGE_CSV = PROCESSED_DIR / "anime_staff_edges.csv"
STAFF_INDEX_CSV = PROCESSED_DIR / "staff_index.csv"
IMPROVEMENT_SUMMARY_FILE = BUILD_DIR / "dataset_improvement_summary.json"

MIN_ANIDB_GENRE_WEIGHT = 200
MIN_ANIDB_TAG_WEIGHT = 200
SPARSE_TAG_COUNT = 3
BASE_CHARACTER_VA_DETAILS_PER_ENTRY = 30
MAX_DYNAMIC_CHARACTER_VA_DETAILS_PER_ENTRY = 150
MAX_CHARACTER_VA_DETAILS_PER_ENTRY = BASE_CHARACTER_VA_DETAILS_PER_ENTRY
JIKAN_BASE_URL = "https://api.jikan.moe/v4"
JIKAN_DETAIL_DELAY_SECONDS = 1
JIKAN_MAX_RETRIES = 4
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
ANILIST_TOP_DELAY_SECONDS = 2.2
ANILIST_TOP_MAX_RETRIES = 4
ANILIST_CHARACTER_FAVORITE_QUERY_VERSION = 3
TARGET_STAFF_ROLES = {"Original Creator", "Original Story", "Director", "Original Character Design"}
STAFF_ROLE_GROUPS = {
    "original creator": "original_creator",
    "original story": "original_creator",
    "director": "director",
    "original character design": "original_character_design",
}

DEMOGRAPHIC_VALUES = {"18+", "Josei", "Kodomo", "Seinen", "Shoujo", "Shounen"}
DEMOGRAPHIC_TAG_HINTS = {
    "Josei": "Josei",
    "Seinen": "Seinen",
    "Shoujo": "Shoujo",
    "Shounen": "Shounen",
    "Kids": "Kodomo",
    "Children": "Kodomo",
    "Primarily Child Cast": "Kodomo",
}
ADULT_RATINGS = {"Rx - Hentai"}
ADULT_GENRES = {"Hentai"}
ADULT_TAG_HINTS = {"Hentai", "Pornographic", "Sexual Content"}
SEINEN_HINT_TAGS = {"Primarily Adult Cast", "Psychological", "Noir", "Politics", "Crime", "Philosophy", "Work"}
KODOMO_HINT_TAGS = {"Primarily Child Cast", "Educational", "Anthropomorphism", "Animals", "Pets"}
CLOSE_RELATIONS = {"Sequel", "Prequel", "Side Story", "Parent Story", "Summary", "Full Story"}
ORIGIN_TAGS = {
    "american-japanese co-production",
    "chinese production",
    "development hell",
    "fan-made",
    "french-chinese co-production",
    "french-japanese co-production",
    "indo-japanese co-production",
    "italian-japanese co-production",
    "japanese production",
    "korean-japanese co-production",
    "north korean production",
    "polish-japanese co-production",
    "russian-japanese co-production",
    "saudi arabian-japanese co-production",
    "singaporean production",
    "sino-japanese co-production",
    "south korean production",
    "taiwanese production",
    "thai production",
}
DEMOGRAPHIC_NAMES = {
    "18 restricted": "18+",
    "adult": "18+",
    "josei": "Josei",
    "kodomo": "Kodomo",
    "kids": "Kodomo",
    "mina": "Kodomo",
    "seinen": "Seinen",
    "shoujo": "Shoujo",
    "shonen": "Shounen",
    "shounen": "Shounen",
}
STUDIO_ALIASES = {
    "10gauge": "10Gauge",
    "asread": "Asread.",
    "gallop": "Gallop",
    "studiogallop": "Gallop",
    "j c staff": "J.C.Staff",
    "jcstaff": "J.C.Staff",
    "madhouse": "Madhouse",
    "gainax": "Gainax",
    "tatsunokoproduction": "Tatsunoko Production",
    "toei animation": "Toei Animation",
    "toeianimation": "Toei Animation",
}
CATALOG_STUDIO_OVERRIDES = {
    # Fuyu no Hi is a collaborative animation anthology. Source APIs may list
    # the contributing directors/animators as studios; keep the catalog field
    # empty rather than polluting studio features with people names.
    3215: "",
}
PERSON_NAME_ALIASES = {
    "gackt": "GACKT",
}

ANIDB_GENRE_MAP = {
    "action": "Action",
    "adventure": "Adventure",
    "comedy": "Comedy",
    "drama": "Drama",
    "ecchi": "Ecchi",
    "fantasy": "Fantasy",
    "hentai": "Hentai",
    "horror": "Horror",
    "mahou shoujo": "Mahou Shoujo",
    "magical girl": "Mahou Shoujo",
    "mecha": "Mecha",
    "music": "Music",
    "mystery": "Mystery",
    "psychological": "Psychological",
    "romance": "Romance",
    "science fiction": "Sci-Fi",
    "sci-fi": "Sci-Fi",
    "slice of life": "Slice of Life",
    "sports": "Sports",
    "supernatural": "Supernatural",
    "thriller": "Thriller",
}

ANIDB_TAG_MAP = {
    "4-koma": "4-koma",
    "acting": "Acting",
    "adoption": "Adoption",
    "afterlife": "Afterlife",
    "age gap": "Age Gap",
    "age regression": "Age Regression",
    "agriculture": "Agriculture",
    "aliens": "Aliens",
    "alternate universe": "Alternate Universe",
    "amnesia": "Amnesia",
    "anachronism": "Anachronism",
    "angels": "Angels",
    "animals": "Animals",
    "anthropomorphism": "Anthropomorphism",
    "anti-hero": "Anti-Hero",
    "archery": "Archery",
    "arranged marriage": "Arranged Marriage",
    "artificial intelligence": "Artificial Intelligence",
    "assassins": "Assassins",
    "astronomy": "Astronomy",
    "augmented reality": "Augmented Reality",
    "band": "Band",
    "baseball": "Baseball",
    "basketball": "Basketball",
    "battle royale": "Battle Royale",
    "bicycle": "Cycling",
    "boxing": "Boxing",
    "bullying": "Bullying",
    "butler": "Butler",
    "cars": "Cars",
    "chibi": "Chibi",
    "chuunibyou": "Chuunibyou",
    "classic literature": "Classic Literature",
    "clone": "Clone",
    "college": "College",
    "conspiracy": "Conspiracy",
    "cosmic horror": "Cosmic Horror",
    "crime": "Crime",
    "crossdressing": "Crossdressing",
    "cross-dressing": "Crossdressing",
    "cultivation": "Cultivation",
    "cyberpunk": "Cyberpunk",
    "cyborg": "Cyborg",
    "cute girls doing cute things": "Cute Girls Doing Cute Things",
    "dancing": "Dancing",
    "death game": "Death Game",
    "delinquents": "Delinquents",
    "demons": "Demons",
    "denpa": "Denpa",
    "desert": "Desert",
    "detective": "Detective",
    "dinosaurs": "Dinosaurs",
    "disability": "Disability",
    "drawing": "Drawing",
    "drugs": "Drugs",
    "dystopia": "Dystopian",
    "educational": "Educational",
    "ensemble cast": "Ensemble Cast",
    "environmental": "Environmental",
    "environmentalism": "Environmental",
    "episodic": "Episodic",
    "espionage": "Espionage",
    "exorcism": "Exorcism",
    "fairy": "Fairy",
    "fairy tale": "Fairy Tale",
    "family life": "Family Life",
    "female harem": "Female Harem",
    "female protagonist": "Female Protagonist",
    "fishing": "Fishing",
    "food": "Food",
    "football": "Football",
    "foreign": "Foreign",
    "found family": "Found Family",
    "fugitive": "Fugitive",
    "gambling": "Gambling",
    "gender bender": "Gender Bending",
    "gender bending": "Gender Bending",
    "ghost": "Ghost",
    "gore": "Gore",
    "gods": "Gods",
    "guns": "Guns",
    "gyaru": "Gyaru",
    "heterosexual": "Heterosexual",
    "hikikomori": "Hikikomori",
    "historical": "Historical",
    "idol": "Idol",
    "idols": "Idol",
    "isekai": "Isekai",
    "iyashikei": "Iyashikei",
    "josei": "Josei",
    "kaiju": "Kaiju",
    "kemonomimi": "Kemonomimi",
    "kuudere": "Kuudere",
    "language barrier": "Language Barrier",
    "lost civilization": "Lost Civilization",
    "love triangle": "Love Triangle",
    "magic": "Magic",
    "maids": "Maids",
    "male harem": "Male Harem",
    "male protagonist": "Male Protagonist",
    "martial arts": "Martial Arts",
    "medicine": "Medicine",
    "memory manipulation": "Memory Manipulation",
    "mermaid": "Mermaid",
    "military": "Military",
    "mixed media": "Mixed Media",
    "monster boy": "Monster Boy",
    "monster girl": "Monster Girl",
    "motorcycles": "Motorcycles",
    "musical theater": "Musical Theater",
    "mythology": "Mythology",
    "nekomimi": "Nekomimi",
    "ninja": "Ninja",
    "noir": "Noir",
    "nudity": "Nudity",
    "office lady": "Office Lady",
    "orphan": "Orphan",
    "otaku culture": "Otaku Culture",
    "parody": "Parody",
    "philosophy": "Philosophy",
    "pirates": "Pirates",
    "police": "Police",
    "politics": "Politics",
    "post-apocalyptic": "Post-Apocalyptic",
    "prison": "Prison",
    "primarily adult cast": "Primarily Adult Cast",
    "primarily child cast": "Primarily Child Cast",
    "primarily female cast": "Primarily Female Cast",
    "primarily male cast": "Primarily Male Cast",
    "primarily teen cast": "Primarily Teen Cast",
    "proxy battles": "Proxy Battle",
    "rehabilitation": "Rehabilitation",
    "reincarnation": "Reincarnation",
    "religion": "Religion",
    "revenge": "Revenge",
    "robots": "Robots",
    "royal affairs": "Royal Affairs",
    "rural": "Rural",
    "samurai": "Samurai",
    "satire": "Satire",
    "school": "School",
    "school club": "School Club",
    "school life": "School",
    "seinen": "Seinen",
    "shapeshifting": "Shapeshifting",
    "ships": "Ships",
    "shoujo": "Shoujo",
    "shounen": "Shounen",
    "shrine maiden": "Shrine Maiden",
    "slapstick": "Slapstick",
    "slavery": "Slavery",
    "software development": "Software Development",
    "space": "Space",
    "space opera": "Space Opera",
    "swordplay": "Swordplay",
    "tanned skin": "Tanned Skin",
    "teacher": "Teacher",
    "tennis": "Tennis",
    "terrorism": "Terrorism",
    "time manipulation": "Time Manipulation",
    "time travel": "Time Manipulation",
    "tomboy": "Tomboy",
    "torture": "Torture",
    "tragedy": "Tragedy",
    "trains": "Trains",
    "travel": "Travel",
    "tsundere": "Tsundere",
    "twins": "Twins",
    "urban": "Urban",
    "urban fantasy": "Urban Fantasy",
    "vampire": "Vampire",
    "video game": "Video Games",
    "video games": "Video Games",
    "villainess": "Villainess",
    "war": "War",
    "werewolf": "Werewolf",
    "work": "Work",
    "writing": "Writing",
    "yakuza": "Yakuza",
    "yandere": "Yandere",
    "youkai": "Youkai",
    "yuri": "Yuri",
    "zombie": "Zombie",
}

IGNORE_ANIDB_TAGS = {
    "cast",
    "content indicators",
    "dynamic",
    "elements",
    "ending",
    "fetishes",
    "maintenance tags",
    "origin",
    "original work",
    "place",
    "present",
    "season",
    "setting",
    "storytelling",
    "target audience",
    "technical aspects",
    "themes",
    "time",
    "tropes",
    "unsorted",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{time.time_ns()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    last_error: PermissionError | None = None
    for attempt in range(1, 9):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(min(2.0, 0.15 * attempt))
    pending = path.with_suffix(path.suffix + f".pending_{time.time_ns()}.json")
    try:
        tmp.replace(pending)
        print(
            f"[WARN] Could not overwrite locked JSON {path}; wrote pending copy {pending}: {last_error}",
            flush=True,
        )
    except OSError as exc:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        print(f"[WARN] Could not persist JSON {path}: {exc}", flush=True)


def write_csv_with_pending_fallback(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_csv(path, index=False)
        return str(path)
    except PermissionError:
        pending = path.with_suffix(path.suffix + f".pending_{time.time_ns()}.csv")
        df.to_csv(pending, index=False)
        print(f"[WARN] Could not overwrite locked CSV {path}; wrote pending copy {pending}", flush=True)
        return str(pending)


def write_json_with_pending_fallback(df: pd.DataFrame, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = df.to_json(orient="records", indent=2, force_ascii=False)
    try:
        path.write_text(payload, encoding="utf-8")
        return str(path)
    except PermissionError:
        pending = path.with_suffix(path.suffix + f".pending_{time.time_ns()}.json")
        pending.write_text(payload, encoding="utf-8")
        print(f"[WARN] Could not overwrite locked JSON {path}; wrote pending copy {pending}", flush=True)
        return str(pending)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except TypeError:
        pass
    return str(value).strip().lower() in {"", "nan", "none", "null"}


def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        if is_missing(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def normalize_key(value: Any) -> str:
    text = re.sub(r"[^\w\s+-]", " ", str(value or "").casefold())
    return re.sub(r"\s+", " ", text).strip()


def split_pipe(value: Any) -> list[str]:
    if is_missing(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def merge_pipe(values: list[Any]) -> str:
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        for part in split_pipe(value):
            key = part.casefold()
            if key not in seen:
                seen.add(key)
                out.append(part)
    return "|".join(out)


def split_edge(value: Any) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for part in split_pipe(value):
        if ":" not in part:
            continue
        target_text, payload = part.split(":", 1)
        target_id = parse_int(target_text, default=None)
        if target_id is None:
            continue
        edges.append((target_id, payload.strip()))
    return edges


def split_relation_edges(value: Any) -> list[tuple[int, str]]:
    return split_edge(value)


def split_recommendation_edges(value: Any) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for target_id, payload in split_edge(value):
        weight = parse_int(payload, default=1) or 1
        out.append((target_id, weight))
    return out


def normalize_studio_key(value: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value or "").casefold())
    text = re.sub(r"\bstudio\b", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return re.sub(r"\s+", "", text)


def pretty_studio_name(value: Any) -> str:
    text = str(value or "").strip()
    key = normalize_studio_key(text)
    if not key:
        return ""
    if key in STUDIO_ALIASES:
        return STUDIO_ALIASES[key]
    if text.isupper() or text.islower():
        return text.title()
    return text.rstrip(".") + "." if key == "asread" else text


def normalize_studio_list(value: Any) -> str:
    values = []
    for part in re.split(r"[|;]", str(value or "")):
        clean = pretty_studio_name(part)
        if clean:
            values.append(clean)
    return merge_pipe(values)


def filter_edge_column(df: pd.DataFrame, column: str, valid_ids: set[int]) -> tuple[pd.DataFrame, int]:
    if column not in df.columns:
        return df, 0
    removed = 0
    for idx, value in df[column].items():
        kept: list[str] = []
        for target_id, payload in split_edge(value):
            if target_id in valid_ids and payload:
                kept.append(f"{target_id}:{payload}")
            else:
                removed += 1
        df.at[idx, column] = merge_pipe(kept)
    return df, removed


def filter_graph_edges_to_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    valid_ids = {int(value) for value in df["mal_id"].dropna().astype(int).tolist()}
    counts: dict[str, int] = {}
    for column in [
        "relations",
        "recommendations",
        "recommendations_jikan",
        "recommendations_anilist",
        "recommendations_anidb",
    ]:
        df, removed = filter_edge_column(df, column, valid_ids)
        counts[f"{column}_edges_removed_outside_dataset"] = removed
    return df, counts


def build_reverse_close_relations(df: pd.DataFrame) -> dict[int, set[int]]:
    reverse: dict[int, set[int]] = defaultdict(set)
    valid_ids = {int(value) for value in df["mal_id"].dropna().astype(int).tolist()}
    for row in df.itertuples(index=False):
        source_id = int(getattr(row, "mal_id"))
        for target_id, relation in split_relation_edges(getattr(row, "relations", "")):
            if target_id in valid_ids and relation in CLOSE_RELATIONS:
                reverse[target_id].add(source_id)
    return reverse


def close_neighbor_ids(row: pd.Series, reverse: dict[int, set[int]], valid_ids: set[int]) -> list[int]:
    source_id = parse_int(row.get("mal_id"), default=None)
    ids: list[int] = []
    for target_id, relation in split_relation_edges(row.get("relations")):
        if target_id in valid_ids and relation in CLOSE_RELATIONS:
            ids.append(target_id)
    if source_id is not None:
        ids.extend(sorted(reverse.get(source_id, set())))
    seen: set[int] = set()
    out = []
    for item in ids:
        if item != source_id and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def best_neighbor_value(df: pd.DataFrame, ids: list[int], field: str) -> Any:
    if field not in df.columns or not ids:
        return None
    by_id = df.set_index("mal_id", drop=False)
    candidates = []
    for item_id in ids:
        if item_id in by_id.index:
            value = by_id.at[item_id, field]
            if not is_missing(value):
                candidates.append(value)
    if not candidates:
        return None
    counts: dict[str, int] = defaultdict(int)
    display_by_key: dict[str, Any] = {}
    for value in candidates:
        key = normalize_key(value)
        counts[key] += 1
        display_by_key.setdefault(key, value)
    chosen_key = sorted(counts, key=lambda key: (-counts[key], key))[0]
    return display_by_key[chosen_key]


def best_recommendation_value(df: pd.DataFrame, row: pd.Series, field: str) -> Any:
    if field not in df.columns:
        return None
    by_id = df.set_index("mal_id", drop=False)
    candidates = []
    for target_id, weight in sorted(split_recommendation_edges(row.get("recommendations")), key=lambda edge: -edge[1]):
        if target_id in by_id.index:
            value = by_id.at[target_id, field]
            if not is_missing(value):
                candidates.append((value, weight))
    if not candidates:
        return None
    scores: dict[str, int] = defaultdict(int)
    display_by_key: dict[str, Any] = {}
    for value, weight in candidates:
        key = normalize_key(value)
        scores[key] += int(weight)
        display_by_key.setdefault(key, value)
    chosen_key = sorted(scores, key=lambda key: (-scores[key], key))[0]
    return display_by_key[chosen_key]


def repair_missing_from_neighbors(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    valid_ids = {int(value) for value in df["mal_id"].dropna().astype(int).tolist()}
    reverse = build_reverse_close_relations(df)
    counts = {
        "source_filled_from_close_neighbor": 0,
        "studios_filled_from_close_neighbor": 0,
        "rating_filled_from_close_neighbor": 0,
        "rating_filled_from_recommendations": 0,
    }
    for idx, row in df.iterrows():
        neighbor_ids = close_neighbor_ids(row, reverse, valid_ids)
        if is_missing(row.get("source")):
            value = best_neighbor_value(df, neighbor_ids, "source")
            if not is_missing(value):
                df.at[idx, "source"] = value
                counts["source_filled_from_close_neighbor"] += 1
        if is_missing(row.get("studios")):
            value = best_neighbor_value(df, neighbor_ids, "studios")
            if not is_missing(value):
                df.at[idx, "studios"] = value
                counts["studios_filled_from_close_neighbor"] += 1
        if is_missing(row.get("rating")):
            value = best_neighbor_value(df, neighbor_ids, "rating")
            if not is_missing(value):
                df.at[idx, "rating"] = value
                counts["rating_filled_from_close_neighbor"] += 1
            else:
                value = best_recommendation_value(df, row, "rating")
                if not is_missing(value):
                    df.at[idx, "rating"] = value
                    counts["rating_filled_from_recommendations"] += 1
    return df, counts


def anidb_weighted_demographic(metadata: dict[str, Any] | None) -> str:
    weighted_values: list[tuple[str, int]] = []
    for tag in (metadata or {}).get("raw_tags") or []:
        name = str(tag.get("name") or "").strip().casefold()
        if name in DEMOGRAPHIC_NAMES:
            weighted_values.append((DEMOGRAPHIC_NAMES[name], parse_int(tag.get("weight"), default=0) or 0))
    if not weighted_values:
        return ""
    max_weight = max(weight for _, weight in weighted_values)
    if max_weight <= 0:
        return ""
    return merge_pipe([value for value, weight in weighted_values if weight == max_weight])


def anidb_origin(metadata: dict[str, Any] | None) -> str:
    values = []
    for tag in (metadata or {}).get("raw_tags") or []:
        name = str(tag.get("name") or "").strip()
        if name.casefold() in ORIGIN_TAGS:
            values.append(name)
    return merge_pipe(values)


def build_anidb_extra_lookup(cache: dict[str, Any]) -> dict[int, dict[str, str]]:
    lookup: dict[int, dict[str, str]] = {}
    for key, item in cache.get("items", {}).items():
        metadata = item.get("metadata") if isinstance(item, dict) and "metadata" in item else item
        anidb_id = parse_int(key, default=None)
        if anidb_id is None or not isinstance(metadata, dict):
            continue
        lookup[anidb_id] = {
            "demographics_weighted": anidb_weighted_demographic(metadata),
            "production_origin": anidb_origin(metadata),
        }
    return lookup


def add_anidb_extra_fields(df: pd.DataFrame, anidb_extra_lookup: dict[int, dict[str, str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    if "production_origin" not in df.columns:
        df["production_origin"] = ""
    if "demographics_anidb_weighted" not in df.columns:
        df["demographics_anidb_weighted"] = ""
    origin_filled = 0
    weighted_demo_filled = 0
    for idx, row in df.iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        if anidb_id is None:
            continue
        extra = anidb_extra_lookup.get(anidb_id, {})
        if is_missing(row.get("production_origin")) and extra.get("production_origin"):
            df.at[idx, "production_origin"] = extra["production_origin"]
            origin_filled += 1
        if is_missing(row.get("demographics_anidb_weighted")) and extra.get("demographics_weighted"):
            df.at[idx, "demographics_anidb_weighted"] = extra["demographics_weighted"]
            weighted_demo_filled += 1
    return df, {
        "production_origin_filled_from_anidb": origin_filled,
        "demographics_anidb_weighted_filled": weighted_demo_filled,
    }


def choose_single_demographic(row: pd.Series) -> tuple[str, str]:
    scores: dict[str, float] = defaultdict(float)
    source_weights = {
        "demographics": 1.5,
        "demographics_mal": 3.0,
        "demographics_anilist": 3.0,
        "demographics_anidb_weighted": 2.5,
        "demographics_anidb": 0.35,
    }
    first_source_order: dict[str, int] = {}
    order = 0
    for column, weight in source_weights.items():
        for value in split_pipe(row.get(column)):
            if value in DEMOGRAPHIC_VALUES:
                scores[value] += weight
                first_source_order.setdefault(value, order)
                order += 1

    rating = str(row.get("rating") or "").strip()
    genres = set(split_pipe(row.get("genres")))
    tags = set(split_pipe(row.get("tags")))
    explicit_tags = set(split_pipe(row.get("explicit_tags")))

    if rating in ADULT_RATINGS or genres.intersection(ADULT_GENRES) or explicit_tags.intersection(ADULT_TAG_HINTS):
        scores["18+"] += 10.0
        first_source_order.setdefault("18+", -10)

    for tag, demographic in DEMOGRAPHIC_TAG_HINTS.items():
        if tag in tags or tag in genres:
            scores[demographic] += 2.0
            first_source_order.setdefault(demographic, 50)

    if tags.intersection(SEINEN_HINT_TAGS) or genres.intersection({"Psychological", "Thriller", "Suspense"}):
        scores["Seinen"] += 2.2
        first_source_order.setdefault("Seinen", 40)
    if tags.intersection(KODOMO_HINT_TAGS) and rating in {"G - All Ages", "PG - Children"}:
        scores["Kodomo"] += 2.2
        first_source_order.setdefault("Kodomo", 40)

    if "18+" in scores and scores["18+"] < 10 and len(scores) > 1:
        scores["18+"] -= 3.0

    if not scores:
        return "", "missing"

    chosen = sorted(scores, key=lambda value: (-scores[value], first_source_order.get(value, 999), value))[0]
    source = "inferred_or_resolved" if chosen not in split_pipe(row.get("demographics")) else "collapsed_existing"
    return chosen, source


def resolve_demographics(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "demographics" not in df.columns:
        df["demographics"] = ""
    filled = 0
    collapsed = 0
    unchanged = 0
    for idx, row in df.iterrows():
        before = merge_pipe(split_pipe(row.get("demographics")))
        chosen, _ = choose_single_demographic(row)
        if not chosen:
            continue
        if is_missing(before):
            filled += 1
        elif before != chosen:
            collapsed += 1
        else:
            unchanged += 1
        df.at[idx, "demographics"] = chosen
    return df, {
        "demographics_filled": filled,
        "demographics_collapsed_to_single": collapsed,
        "demographics_unchanged_single": unchanged,
    }


def drop_rows_without_genres_and_tags(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if "genres" not in df.columns:
        df["genres"] = ""
    if "tags" not in df.columns:
        df["tags"] = ""
    mask = df["genres"].apply(is_missing) & df["tags"].apply(is_missing)
    dropped = int(mask.sum())
    if dropped:
        df = df.loc[~mask].copy()
    return df, dropped


def add_weighted_tag(weight_text: Any, tag: str, weight: int) -> str:
    existing = []
    found = False
    for part in split_pipe(weight_text):
        if ":" not in part:
            existing.append(part)
            continue
        name, old_weight = part.rsplit(":", 1)
        if name.casefold() == tag.casefold():
            old = parse_int(old_weight, default=0) or 0
            existing.append(f"{name}:{max(old, weight)}")
            found = True
        else:
            existing.append(part)
    if not found:
        existing.append(f"{tag}:{weight}")
    return merge_pipe(existing)


def apply_explicit_safeguards(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    for column in ["genres", "tags", "tag_weights", "explicit_tags", "explicit_tag_weights", "demographics"]:
        if column not in df.columns:
            df[column] = ""
    rx_mask = df["rating"].fillna("").astype(str).str.strip().eq("Rx - Hentai")
    hentai_genre_added = 0
    hentai_demo_added = 0
    explicit_tag_added = 0
    for idx in df.index[rx_mask]:
        if "Hentai" not in split_pipe(df.at[idx, "genres"]) and "Erotica" not in split_pipe(df.at[idx, "genres"]):
            df.at[idx, "genres"] = merge_pipe([df.at[idx, "genres"], "Hentai"])
            hentai_genre_added += 1
        if df.at[idx, "demographics"] != "18+":
            df.at[idx, "demographics"] = "18+"
            hentai_demo_added += 1
        if "Nudity" not in split_pipe(df.at[idx, "explicit_tags"]):
            df.at[idx, "explicit_tags"] = merge_pipe([df.at[idx, "explicit_tags"], "Nudity"])
            df.at[idx, "explicit_tag_weights"] = add_weighted_tag(df.at[idx, "explicit_tag_weights"], "Nudity", 100)
            explicit_tag_added += 1
    return df, {
        "rx_hentai_genre_added": hentai_genre_added,
        "rx_hentai_demographic_18_added": hentai_demo_added,
        "rx_hentai_explicit_tag_added": explicit_tag_added,
    }


def normalize_studios(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    normalized_rows = 0
    for column in ["studios", "studios_mal", "studios_anilist", "studios_anidb"]:
        if column not in df.columns:
            continue
        for idx, value in df[column].items():
            normalized = normalize_studio_list(value)
            if normalized != ("" if is_missing(value) else str(value)):
                df.at[idx, column] = normalized
                normalized_rows += 1
    if "mal_id" in df.columns and "studios" in df.columns:
        for mal_id, studio_value in CATALOG_STUDIO_OVERRIDES.items():
            mask = pd.to_numeric(df["mal_id"], errors="coerce").eq(int(mal_id))
            if mask.any():
                changed = int((df.loc[mask, "studios"].fillna("").astype(str) != str(studio_value)).sum())
                df.loc[mask, "studios"] = studio_value
                normalized_rows += changed
    return df, {"studio_cells_normalized": normalized_rows}


def fill_missing_studios_from_origin(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "studios" not in df.columns:
        df["studios"] = ""
    if "production_origin" not in df.columns:
        df["production_origin"] = ""
    filled = 0
    for idx, row in df.iterrows():
        if is_missing(row.get("studios")) and not is_missing(row.get("production_origin")):
            df.at[idx, "studios"] = row.get("production_origin")
            filled += 1
    return df, {"studios_filled_from_production_origin": filled}


def rerank_dataset(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    if "score" not in df.columns:
        return df, {"rank_recomputed_rows": 0, "hentai_rank_rows": 0}
    scores = pd.to_numeric(df["score"], errors="coerce")
    df["rank"] = scores.rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    hentai_mask = df["genres"].fillna("").astype(str).str.contains(r"(?:^|\|)Hentai(?:\||$)", regex=True)
    df["hentai_rank"] = pd.NA
    if hentai_mask.any():
        df.loc[hentai_mask, "hentai_rank"] = scores.loc[hentai_mask].rank(method="min", ascending=False, na_option="bottom").astype("Int64")
    return df, {"rank_recomputed_rows": int(len(df)), "hentai_rank_rows": int(hentai_mask.sum())}


def drop_unrepairable_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, int]]:
    counts: dict[str, int] = {}
    for column, label in [
        ("duration", "missing_duration"),
        ("aired_year", "missing_aired_year"),
        ("rating", "missing_rating"),
    ]:
        if column not in df.columns:
            counts[f"dropped_{label}_rows"] = 0
            continue
        mask = df[column].apply(is_missing)
        if column == "duration":
            numeric = pd.to_numeric(df[column], errors="coerce")
            mask = mask | numeric.isna() | numeric.le(0)
        dropped = int(mask.sum())
        counts[f"dropped_{label}_rows"] = dropped
        if dropped:
            df = df.loc[~mask].copy()
    return df, counts


def pipe_count(value: Any) -> int:
    return len(split_pipe(value))


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        corrupt_path = path.with_suffix(path.suffix + f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            path.replace(corrupt_path)
            print(f"[WARN] Corrupt JSON moved to {corrupt_path}: {exc}", flush=True)
        except OSError as move_exc:
            print(f"[WARN] Corrupt JSON could not be moved from {path}: {move_exc}", flush=True)
        return default


def record_detail_failure(payload: dict[str, Any], mal_id: int, error: str, *, retryable: bool = True) -> None:
    payload.setdefault("items", {})[str(int(mal_id))] = {
        "mal_id": int(mal_id),
        "error": str(error)[:500],
        "retryable": bool(retryable),
        "last_attempt_at": now_iso(),
    }


def clear_detail_failure(payload: dict[str, Any], mal_id: int) -> None:
    payload.setdefault("items", {}).pop(str(int(mal_id)), None)


def load_anilist_refs() -> tuple[set[str], dict[str, dict[str, Any]]]:
    genres = set(load_json(ANILIST_GENRES_FILE, []))
    tag_rows = load_json(ANILIST_TAGS_FILE, [])
    tags = {row.get("name"): row for row in tag_rows if isinstance(row, dict) and row.get("name")}
    return genres, tags


def iter_anidb_metadata(cache: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in cache.get("items", {}).values():
        if not isinstance(item, dict):
            continue
        metadata = item.get("metadata") if "metadata" in item else item
        if isinstance(metadata, dict):
            out.append(metadata)
    return out


def collect_anidb_tag_stats(cache: dict[str, Any]) -> pd.DataFrame:
    stats: dict[tuple[int | None, str], dict[str, Any]] = {}
    for metadata in iter_anidb_metadata(cache):
        seen_in_anime: set[tuple[int | None, str]] = set()
        for tag in metadata.get("raw_tags") or []:
            if not isinstance(tag, dict):
                continue
            name = str(tag.get("name") or "").strip()
            if not name:
                continue
            tag_id = parse_int(tag.get("id"), default=None)
            key = (tag_id, name.casefold())
            weight = parse_int(tag.get("weight"), default=0) or 0
            row = stats.setdefault(
                key,
                {
                    "anidb_tag_id": tag_id,
                    "anidb_tag_name": name,
                    "parent_id": parse_int(tag.get("parent_id"), default=None),
                    "anime_count": 0,
                    "tag_occurrences": 0,
                    "max_weight": 0,
                    "weight_sum": 0,
                },
            )
            row["tag_occurrences"] += 1
            row["max_weight"] = max(row["max_weight"], weight)
            row["weight_sum"] += weight
            seen_in_anime.add(key)
        for key in seen_in_anime:
            stats[key]["anime_count"] += 1
    df = pd.DataFrame(stats.values())
    if df.empty:
        return df
    df["avg_weight"] = (df["weight_sum"] / df["tag_occurrences"]).round(2)
    return df.sort_values(["anime_count", "max_weight", "anidb_tag_name"], ascending=[False, False, True])


def build_anidb_label_map(cache: dict[str, Any]) -> pd.DataFrame:
    genres, tags = load_anilist_refs()
    tag_names = set(tags)
    tag_is_adult = {name: bool(row.get("isAdult")) for name, row in tags.items()}
    stats = collect_anidb_tag_stats(cache)
    rows = []
    for row in stats.to_dict(orient="records"):
        name = row["anidb_tag_name"]
        key = normalize_key(name)
        target_kind = "ignore"
        target_label = ""
        mapping_source = "unmapped"

        if key in IGNORE_ANIDB_TAGS:
            mapping_source = "explicit_ignore"
        elif key in ANIDB_GENRE_MAP and ANIDB_GENRE_MAP[key] in genres:
            target_kind = "genre"
            target_label = ANIDB_GENRE_MAP[key]
            mapping_source = "curated_genre"
        elif key in ANIDB_TAG_MAP and ANIDB_TAG_MAP[key] in tag_names and not tag_is_adult.get(ANIDB_TAG_MAP[key], False):
            target_kind = "tag"
            target_label = ANIDB_TAG_MAP[key]
            mapping_source = "curated_tag"
        elif name in tag_names and not tag_is_adult.get(name, False):
            target_kind = "tag"
            target_label = name
            mapping_source = "exact_anilist_tag"

        fallback_weight = ""
        if target_kind == "tag":
            fallback_weight = max(75, min(100, round((parse_int(row.get("max_weight"), default=0) or 0) / 6)))

        rows.append(
            {
                **row,
                "normalized_name": key,
                "target_kind": target_kind,
                "target_label": target_label,
                "fallback_weight": fallback_weight,
                "mapping_source": mapping_source,
            }
        )
    mapped = pd.DataFrame(rows)
    ANILIST_REF_DIR.mkdir(parents=True, exist_ok=True)
    mapped.to_csv(ANIDB_LABEL_MAP_CSV, index=False)
    return mapped


def mapped_labels_for_metadata(
    metadata: dict[str, Any],
    by_id: dict[int, Any],
    by_name: dict[str, Any],
    *,
    min_genre_weight: int = MIN_ANIDB_GENRE_WEIGHT,
    min_tag_weight: int = MIN_ANIDB_TAG_WEIGHT,
) -> tuple[str, str, str]:
    genres: list[str] = []
    tags: list[str] = []
    weights: list[str] = []
    for tag in metadata.get("raw_tags") or []:
        if not isinstance(tag, dict):
            continue
        weight = parse_int(tag.get("weight"), default=0) or 0
        tag_id = parse_int(tag.get("id"), default=None)
        row = by_id.get(tag_id) or by_name.get(normalize_key(tag.get("name")))
        if row is None or row.target_kind == "ignore" or is_missing(row.target_label):
            continue
        if row.target_kind == "genre" and weight >= min_genre_weight:
            genres.append(row.target_label)
        elif row.target_kind == "tag" and weight >= min_tag_weight:
            tag_weight = max(75, min(100, round(weight / 6)))
            tags.append(row.target_label)
            weights.append(f"{row.target_label}:{tag_weight}")
    return merge_pipe(genres), merge_pipe(tags), merge_pipe(weights)


def build_anidb_label_lookup(cache: dict[str, Any], label_map: pd.DataFrame) -> dict[int, tuple[str, str, str]]:
    if label_map.empty:
        return {}
    by_id = {
        parse_int(row.anidb_tag_id, default=None): row
        for row in label_map.itertuples(index=False)
        if not is_missing(row.anidb_tag_id)
    }
    by_name = {normalize_key(row.anidb_tag_name): row for row in label_map.itertuples(index=False)}
    lookup = {}
    for key, item in cache.get("items", {}).items():
        metadata = item.get("metadata") if isinstance(item, dict) and "metadata" in item else item
        if not isinstance(metadata, dict):
            continue
        anidb_id = parse_int(key, default=None)
        if anidb_id is None:
            continue
        lookup[anidb_id] = mapped_labels_for_metadata(metadata, by_id, by_name)
    return lookup


def apply_anidb_fallback(df: pd.DataFrame, anidb_lookup: dict[int, tuple[str, str, str]]) -> tuple[pd.DataFrame, dict[str, int]]:
    df = df.copy()
    summary = {
        "genre_rows_filled_from_anidb": 0,
        "tag_rows_filled_from_anidb": 0,
        "tag_rows_augmented_from_anidb": 0,
    }
    for idx, row in df.iterrows():
        anidb_id = parse_int(row.get("anidb_id"), default=None)
        if anidb_id is None or anidb_id not in anidb_lookup:
            continue
        fallback_genres, fallback_tags, fallback_weights = anidb_lookup[anidb_id]
        if is_missing(row.get("genres")) and fallback_genres:
            df.at[idx, "genres"] = fallback_genres
            summary["genre_rows_filled_from_anidb"] += 1
        current_tag_count = pipe_count(row.get("tags"))
        if fallback_tags and current_tag_count == 0:
            df.at[idx, "tags"] = fallback_tags
            df.at[idx, "tag_weights"] = fallback_weights
            summary["tag_rows_filled_from_anidb"] += 1
        elif fallback_tags and current_tag_count < SPARSE_TAG_COUNT:
            merged_tags = merge_pipe([row.get("tags"), fallback_tags])
            if merged_tags != row.get("tags"):
                df.at[idx, "tags"] = merged_tags
                df.at[idx, "tag_weights"] = merge_pipe([row.get("tag_weights"), fallback_weights])
                summary["tag_rows_augmented_from_anidb"] += 1
    return df, summary


def compact_name_japanese_order(name: Any) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    alias = PERSON_NAME_ALIASES.get(canonical_name_key(text))
    if alias:
        return alias
    if "," in text:
        return text
    parts = text.split()
    if len(parts) == 2:
        return f"{parts[1]}, {parts[0]}"
    if len(parts) == 3:
        return f"{parts[-1]}, {' '.join(parts[:-1])}"
    return text


def person_group_key(row: pd.Series, id_columns: list[str], name_column: str) -> str:
    for column in id_columns:
        value = parse_int(row.get(column), default=None)
        if value is not None:
            return f"{column}:{int(value)}"
    return f"name:{canonical_name_key(row.get(name_column))}"


def preferred_person_name(values: pd.Series) -> str:
    names = [str(value).strip() for value in values.dropna() if str(value).strip()]
    if not names:
        return ""
    for name in names:
        alias = PERSON_NAME_ALIASES.get(canonical_name_key(name))
        if alias:
            return alias
    return sorted(names, key=lambda name: (-sum(1 for char in name if char.isupper()), len(name), name.casefold()))[0]


def name_match_keys(name: Any) -> set[str]:
    text = str(name or "").strip()
    if not text:
        return set()
    keys = {normalize_key(text)}
    if "," in text:
        left, right = [part.strip() for part in text.split(",", 1)]
        if left and right:
            keys.add(normalize_key(f"{right} {left}"))
    else:
        parts = [part for part in text.split() if part]
        if len(parts) == 2:
            keys.add(normalize_key(f"{parts[1]} {parts[0]}"))
        elif len(parts) == 3:
            keys.add(normalize_key(f"{parts[-1]} {' '.join(parts[:-1])}"))
    return {key for key in keys if key}


def add_favorite_by_name(index: dict[str, int], name: Any, favorites: Any) -> None:
    favorite_count = parse_int(favorites, default=0) or 0
    if favorite_count <= 0:
        return
    for key in name_match_keys(name):
        index[key] = max(index.get(key, 0), favorite_count)


def add_favorite_by_id(index: dict[int, int], item_id: Any, favorites: Any) -> None:
    parsed_id = parse_int(item_id, default=None)
    favorite_count = parse_int(favorites, default=0) or 0
    if parsed_id is None or favorite_count <= 0:
        return
    index[int(parsed_id)] = max(index.get(int(parsed_id), 0), favorite_count)


def indexed_favorites(
    favorite_indexes: dict[str, dict[Any, int]] | None,
    kind: str,
    item_id: Any,
    name: Any,
) -> int:
    if not favorite_indexes:
        return 0
    id_index = favorite_indexes.get(f"{kind}_by_id", {})
    name_index = favorite_indexes.get(f"{kind}_by_name", {})
    parsed_id = parse_int(item_id, default=None)
    best = id_index.get(int(parsed_id), 0) if parsed_id is not None else 0
    for key in name_match_keys(name):
        best = max(best, name_index.get(key, 0))
    return int(best or 0)


def best_indexed_favorites(
    current_value: Any,
    favorite_indexes: dict[str, dict[Any, int]] | None,
    kind: str,
    item_id: Any,
    name: Any,
) -> int:
    current = parse_int(current_value, default=0) or 0
    return max(current, indexed_favorites(favorite_indexes, kind, item_id, name))


def safe_label(value: Any) -> str:
    return str(value or "").replace("|", " ").replace(":", " ").strip()


def jikan_request(path: str, *, sleep_seconds: float) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for live Jikan character/person enrichment")
    url = f"{JIKAN_BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(1, JIKAN_MAX_RETRIES + 1):
        try:
            response = requests.get(url, timeout=45)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(90, sleep_seconds * (2**attempt)))
            continue
        if response.status_code == 200:
            time.sleep(sleep_seconds)
            return response.json()
        if response.status_code in {404, 410}:
            return {"data": None, "missing": True, "status_code": response.status_code}
        last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            time.sleep(max(float(retry_after), sleep_seconds))
        else:
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(last_error or f"Jikan request failed: {url}")


def get_jikan_detail(
    cache: dict[str, Any],
    detail_kind: str,
    item_id: int,
    *,
    sleep_seconds: float,
) -> dict[str, Any]:
    key = str(int(item_id))
    if key in cache.get("items", {}):
        return cache["items"][key]
    if detail_kind == "person":
        payload = jikan_request(f"/people/{int(item_id)}", sleep_seconds=sleep_seconds)
    elif detail_kind == "character":
        payload = jikan_request(f"/characters/{int(item_id)}", sleep_seconds=sleep_seconds)
    else:
        raise ValueError(f"Unknown Jikan detail kind: {detail_kind}")
    data = payload.get("data") if isinstance(payload, dict) else None
    item = {
        "id": int(item_id),
        "name": compact_name_japanese_order((data or {}).get("name")),
        "favorites": parse_int((data or {}).get("favorites"), default=0) or 0,
        "missing": bool((payload or {}).get("missing")),
        "fetched_at": now_iso(),
    }
    cache.setdefault("items", {})[key] = item
    return item


def refresh_jikan_top_favorites(kind: str, *, pages: int, sleep_seconds: float) -> dict[str, Any]:
    if kind == "characters":
        cache_path = JIKAN_TOP_CHARACTER_CACHE_FILE
        endpoint = "/top/characters"
    elif kind == "people":
        cache_path = JIKAN_TOP_PEOPLE_CACHE_FILE
        endpoint = "/top/people"
    else:
        raise ValueError(f"Unknown Jikan top favorite kind: {kind}")

    cache = load_json(cache_path, {"updated_at": None, "pages": {}, "items": {}})
    cache.setdefault("pages", {})
    cache.setdefault("failed_pages", {})
    cache.setdefault("items", {})
    pages_requested = 0
    pages_skipped = 0
    pages_failed = 0
    rows_added = 0
    consecutive_failures = 0

    for page in range(1, max(0, int(pages)) + 1):
        page_key = str(page)
        if page_key in cache["pages"]:
            pages_skipped += 1
            continue
        try:
            payload = jikan_request(f"{endpoint}?page={page}", sleep_seconds=sleep_seconds)
        except Exception as exc:
            pages_failed += 1
            consecutive_failures += 1
            cache["failed_pages"][page_key] = {
                "page": page,
                "error": str(exc)[:500],
                "last_attempt_at": now_iso(),
                "retryable": True,
            }
            cache["updated_at"] = now_iso()
            atomic_write_json(cache_path, cache)
            print(
                f"Jikan top {kind} failed: page={page:,}/{pages:,}, "
                f"pages_failed={pages_failed:,}, consecutive_failures={consecutive_failures:,}, error={exc}",
                flush=True,
            )
            if consecutive_failures >= 5:
                print(
                    f"Jikan top {kind} stopping early after {consecutive_failures} consecutive failures; rerun later to retry.",
                    flush=True,
                )
                break
            continue
        consecutive_failures = 0
        cache["failed_pages"].pop(page_key, None)
        page_ids: list[int] = []
        for row in payload.get("data") or []:
            if not isinstance(row, dict):
                continue
            item_id = parse_int(row.get("mal_id"), default=None)
            if item_id is None:
                continue
            item = {
                "id": int(item_id),
                "name": compact_name_japanese_order(row.get("name")),
                "favorites": parse_int(row.get("favorites"), default=0) or 0,
            }
            cache["items"][str(item_id)] = item
            page_ids.append(int(item_id))
            rows_added += 1
        cache["pages"][page_key] = page_ids
        cache["updated_at"] = now_iso()
        atomic_write_json(cache_path, cache)
        pages_requested += 1
        print(
            f"Jikan top {kind} progress: page={page:,}/{pages:,}, "
            f"pages_requested={pages_requested:,}, pages_skipped={pages_skipped:,}, "
            f"items_cached={len(cache['items']):,}",
            flush=True,
        )
        pagination = payload.get("pagination") or {}
        if pagination and not pagination.get("has_next_page", True):
            break

    cache["updated_at"] = now_iso()
    atomic_write_json(cache_path, cache)
    return {
        f"jikan_top_{kind}_pages_requested": pages_requested,
        f"jikan_top_{kind}_pages_skipped": pages_skipped,
        f"jikan_top_{kind}_pages_failed": pages_failed,
        f"jikan_top_{kind}_rows_added_or_refreshed": rows_added,
        f"jikan_top_{kind}_items_cached": len(cache.get("items", {})),
    }


def anilist_top_request(query: str, variables: dict[str, Any], *, sleep_seconds: float) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is required for live AniList top favorite enrichment")
    last_error: Exception | None = None
    for attempt in range(1, ANILIST_TOP_MAX_RETRIES + 1):
        try:
            response = requests.post(
                ANILIST_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                timeout=45,
            )
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(90, sleep_seconds * (2**attempt)))
            continue
        if response.status_code == 200:
            time.sleep(sleep_seconds)
            return response.json()
        last_error = RuntimeError(f"HTTP {response.status_code}: {response.text[:300]}")
        retry_after = response.headers.get("Retry-After")
        if retry_after and retry_after.isdigit():
            time.sleep(max(float(retry_after), sleep_seconds))
        else:
            time.sleep(sleep_seconds * attempt)
    raise RuntimeError(last_error or "AniList top favorite request failed")


ANILIST_TOP_CHARACTERS_QUERY = """
query TopCharacters($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    characters(sort: [FAVOURITES_DESC, ID]) {
      id
      favourites
      name {
        full
      }
    }
  }
}
"""


ANILIST_TOP_STAFF_QUERY = """
query TopStaff($page: Int, $perPage: Int) {
  Page(page: $page, perPage: $perPage) {
    pageInfo {
      hasNextPage
    }
    staff(sort: [FAVOURITES_DESC, ID]) {
      id
      favourites
      name {
        full
      }
      languageV2
    }
  }
}
"""


def refresh_anilist_top_favorites(kind: str, *, pages: int, per_page: int, sleep_seconds: float) -> dict[str, Any]:
    if kind == "characters":
        cache_path = ANILIST_TOP_CHARACTER_CACHE_FILE
        query = ANILIST_TOP_CHARACTERS_QUERY
        node_key = "characters"
    elif kind == "staff":
        cache_path = ANILIST_TOP_STAFF_CACHE_FILE
        query = ANILIST_TOP_STAFF_QUERY
        node_key = "staff"
    else:
        raise ValueError(f"Unknown AniList top favorite kind: {kind}")

    cache = load_json(cache_path, {"updated_at": None, "pages": {}, "items": {}})
    cache.setdefault("pages", {})
    cache.setdefault("failed_pages", {})
    cache.setdefault("items", {})
    pages_requested = 0
    pages_skipped = 0
    pages_failed = 0
    rows_added = 0
    consecutive_failures = 0

    for page in range(1, max(0, int(pages)) + 1):
        page_key = str(page)
        if page_key in cache["pages"]:
            pages_skipped += 1
            continue
        try:
            payload = anilist_top_request(
                query,
                {"page": page, "perPage": int(per_page)},
                sleep_seconds=sleep_seconds,
            )
        except Exception as exc:
            pages_failed += 1
            consecutive_failures += 1
            cache["failed_pages"][page_key] = {
                "page": page,
                "error": str(exc)[:500],
                "last_attempt_at": now_iso(),
                "retryable": True,
            }
            cache["updated_at"] = now_iso()
            atomic_write_json(cache_path, cache)
            print(
                f"AniList top {kind} failed: page={page:,}/{pages:,}, "
                f"pages_failed={pages_failed:,}, consecutive_failures={consecutive_failures:,}, error={exc}",
                flush=True,
            )
            if consecutive_failures >= 5:
                print(
                    f"AniList top {kind} stopping early after {consecutive_failures} consecutive failures; rerun later to retry.",
                    flush=True,
                )
                break
            continue
        consecutive_failures = 0
        cache["failed_pages"].pop(page_key, None)
        page_payload = ((payload.get("data") or {}).get("Page") or {})
        page_ids: list[int] = []
        for row in page_payload.get(node_key) or []:
            if not isinstance(row, dict):
                continue
            if kind == "staff" and str(row.get("languageV2") or "").casefold() not in {"", "japanese"}:
                continue
            item_id = parse_int(row.get("id"), default=None)
            if item_id is None:
                continue
            item = {
                "id": int(item_id),
                "name": compact_name_japanese_order(((row.get("name") or {}).get("full") or "")),
                "favorites": parse_int(row.get("favourites"), default=0) or 0,
            }
            cache["items"][str(item_id)] = item
            page_ids.append(int(item_id))
            rows_added += 1
        cache["pages"][page_key] = page_ids
        cache["updated_at"] = now_iso()
        atomic_write_json(cache_path, cache)
        pages_requested += 1
        print(
            f"AniList top {kind} progress: page={page:,}/{pages:,}, "
            f"pages_requested={pages_requested:,}, pages_skipped={pages_skipped:,}, "
            f"items_cached={len(cache['items']):,}",
            flush=True,
        )
        page_info = page_payload.get("pageInfo") or {}
        if page_info and not page_info.get("hasNextPage", True):
            break

    cache["updated_at"] = now_iso()
    atomic_write_json(cache_path, cache)
    return {
        f"anilist_top_{kind}_pages_requested": pages_requested,
        f"anilist_top_{kind}_pages_skipped": pages_skipped,
        f"anilist_top_{kind}_pages_failed": pages_failed,
        f"anilist_top_{kind}_rows_added_or_refreshed": rows_added,
        f"anilist_top_{kind}_items_cached": len(cache.get("items", {})),
    }


def add_character_label_to_indexes(indexes: dict[str, dict[Any, int]], character: dict[str, Any]) -> None:
    add_favorite_by_id(indexes["character_by_id"], character.get("id"), character.get("favorites"))
    add_favorite_by_name(indexes["character_by_name"], character.get("name"), character.get("favorites"))


def add_staff_label_to_indexes(indexes: dict[str, dict[Any, int]], staff: dict[str, Any]) -> None:
    add_favorite_by_id(indexes["staff_by_id"], staff.get("id"), staff.get("favorites"))
    add_favorite_by_name(indexes["staff_by_name"], staff.get("name"), staff.get("favorites"))


def add_anilist_media_to_favorite_indexes(indexes: dict[str, dict[Any, int]], media: dict[str, Any]) -> dict[str, int]:
    stats = defaultdict(int)
    for character in media.get("characters") or []:
        if not isinstance(character, dict):
            continue
        before = len(indexes["character_by_id"]) + len(indexes["character_by_name"])
        add_character_label_to_indexes(indexes, character)
        after = len(indexes["character_by_id"]) + len(indexes["character_by_name"])
        if after > before:
            stats["anilist_media_character_index_rows"] += 1
        for actor in character.get("voice_actors") or []:
            if not isinstance(actor, dict):
                continue
            before_staff = len(indexes["staff_by_id"]) + len(indexes["staff_by_name"])
            add_staff_label_to_indexes(indexes, actor)
            after_staff = len(indexes["staff_by_id"]) + len(indexes["staff_by_name"])
            if after_staff > before_staff:
                stats["anilist_media_staff_index_rows"] += 1
    return dict(stats)


def build_character_favorite_indexes(
    anilist_cache: dict[str, Any],
    df: pd.DataFrame | None = None,
) -> tuple[dict[str, dict[Any, int]], dict[str, int]]:
    indexes: dict[str, dict[Any, int]] = {
        "character_by_id": {},
        "character_by_name": {},
        "staff_by_id": {},
        "staff_by_name": {},
    }
    stats = defaultdict(int)

    for item in anilist_cache.get("items", {}).values():
        media = item.get("media") if isinstance(item, dict) else None
        if not isinstance(media, dict):
            continue
        for key, value in add_anilist_media_to_favorite_indexes(indexes, media).items():
            stats[key] += value

    if df is not None and "voice_actor_character_favorites" in df.columns:
        for value in df["voice_actor_character_favorites"]:
            for part in split_pipe(value):
                bits = part.split(":")
                if len(bits) < 6:
                    continue
                character = {"id": bits[0], "name": bits[1], "favorites": bits[2]}
                staff = {"id": bits[3], "name": bits[4], "favorites": bits[5]}
                add_character_label_to_indexes(indexes, character)
                add_staff_label_to_indexes(indexes, staff)
                stats["existing_label_index_rows"] += 1

    for path, kind in [
        (JIKAN_TOP_CHARACTER_CACHE_FILE, "character"),
        (ANILIST_TOP_CHARACTER_CACHE_FILE, "character"),
        (JIKAN_CHARACTER_DETAIL_CACHE_FILE, "character"),
        (JIKAN_TOP_PEOPLE_CACHE_FILE, "staff"),
        (ANILIST_TOP_STAFF_CACHE_FILE, "staff"),
        (JIKAN_PERSON_DETAIL_CACHE_FILE, "staff"),
    ]:
        payload = load_json(path, {"items": {}})
        for row in payload.get("items", {}).values():
            if not isinstance(row, dict):
                continue
            if kind == "character":
                add_favorite_by_name(indexes["character_by_name"], row.get("name"), row.get("favorites"))
                stats[f"{path.stem}_character_name_index_rows"] += 1
            else:
                add_favorite_by_name(indexes["staff_by_name"], row.get("name"), row.get("favorites"))
                stats[f"{path.stem}_staff_name_index_rows"] += 1

    stats["character_favorite_index_ids"] = len(indexes["character_by_id"])
    stats["character_favorite_index_names"] = len(indexes["character_by_name"])
    stats["staff_favorite_index_ids"] = len(indexes["staff_by_id"])
    stats["staff_favorite_index_names"] = len(indexes["staff_by_name"])
    return indexes, dict(stats)


def top_cache_indexes(path: Path) -> tuple[dict[int, int], dict[str, int]]:
    payload = load_json(path, {"items": {}})
    by_id: dict[int, int] = {}
    by_name: dict[str, int] = {}
    for row in payload.get("items", {}).values():
        if not isinstance(row, dict):
            continue
        add_favorite_by_id(by_id, row.get("id"), row.get("favorites"))
        add_favorite_by_name(by_name, row.get("name"), row.get("favorites"))
    return by_id, by_name


def favorite_from_source_indexes(
    by_id: dict[int, int],
    by_name: dict[str, int],
    item_id: Any,
    name: Any,
) -> int:
    parsed_id = parse_int(item_id, default=None)
    best = by_id.get(int(parsed_id), 0) if parsed_id is not None else 0
    for key in name_match_keys(name):
        best = max(best, by_name.get(key, 0))
    return int(best or 0)


def language_priority(language: Any) -> int:
    text = str(language or "").strip().casefold()
    if text == "japanese":
        return 0
    if text == "english":
        return 1
    if not text:
        return 2
    return 3


def role_weight(role: Any) -> float:
    priority = role_priority(role)
    if priority == 0:
        return 1.0
    if priority == 1:
        return 0.6
    if priority == 3:
        return 0.15
    return 0.35


def is_generic_narrator(row: dict[str, Any]) -> bool:
    name_key = canonical_name_key(row.get("character_name"))
    if name_key not in {"narrator", "narration"}:
        return False
    # AniList character 36309 is explicitly a generic bucket for narrators.
    # MAL also has many per-anime narrator pages, but those are usually role
    # credits rather than characters a user would seek recommendations from.
    return True


def make_edge_key(row: dict[str, Any]) -> tuple[int, str, str]:
    return (
        int(row["mal_id"]),
        canonical_name_key(row.get("character_name")),
        canonical_name_key(row.get("voice_actor_name")),
    )


def first_present(*values: Any) -> Any:
    for value in values:
        if not is_missing(value):
            return value
    return ""


def canonical_name_key(name: Any) -> str:
    keys = name_match_keys(name)
    if not keys:
        return ""
    return sorted(keys)[0]


GENERIC_CHARACTER_KEYS = {
    "announcer",
    "boy",
    "child",
    "doctor",
    "girl",
    "man",
    "narration",
    "narrator",
    "old man",
    "student",
    "teacher",
    "woman",
}


def compact_alias_key(name: Any) -> str:
    text = normalize_key(name)
    if not text:
        return ""
    parts = [part for part in text.split() if len(part) > 1]
    return " ".join(sorted(parts))


def character_names_alias_match(left: Any, right: Any) -> bool:
    left_key = canonical_name_key(left)
    right_key = canonical_name_key(right)
    if not left_key or not right_key:
        return False
    if left_key in GENERIC_CHARACTER_KEYS or right_key in GENERIC_CHARACTER_KEYS:
        return False
    if left_key == right_key:
        return True
    if len(left_key) >= 4 and f" {left_key} " in f" {right_key} ":
        return True
    if len(right_key) >= 4 and f" {right_key} " in f" {left_key} ":
        return True
    compact_left = compact_alias_key(left)
    compact_right = compact_alias_key(right)
    return bool(compact_left and compact_left == compact_right)


def compatible_role(left: Any, right: Any) -> bool:
    left_priority = role_priority(left)
    right_priority = role_priority(right)
    return abs(left_priority - right_priority) <= 1


def should_alias_merge_edges(left: dict[str, Any], right: dict[str, Any]) -> bool:
    if int(left["mal_id"]) != int(right["mal_id"]):
        return False
    if canonical_name_key(left.get("voice_actor_name")) != canonical_name_key(right.get("voice_actor_name")):
        return False
    if not compatible_role(left.get("character_role"), right.get("character_role")):
        return False
    left_source = str(left.get("source") or "").casefold()
    right_source = str(right.get("source") or "").casefold()
    if "anilist" in left_source and "anilist" in right_source:
        return False
    if "jikan" in left_source and "jikan" in right_source:
        return False
    return character_names_alias_match(left.get("character_name"), right.get("character_name"))


def merge_edge_values(current: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    current["source"] = merge_pipe([current.get("source"), row.get("source")])
    for column in ["character_id_anilist", "character_id_mal", "voice_actor_id_anilist", "voice_actor_id_mal"]:
        if is_missing(current.get(column)) and not is_missing(row.get(column)):
            current[column] = row.get(column)
    for column in [
        "character_favorites_mal",
        "character_favorites_anilist",
        "voice_actor_favorites_mal",
        "voice_actor_favorites_anilist",
    ]:
        current[column] = max(parse_int(current.get(column), default=0) or 0, parse_int(row.get(column), default=0) or 0)
    if len(str(row.get("character_name") or "")) > len(str(current.get("character_name") or "")):
        current["character_name"] = row.get("character_name")
    if "anilist" in str(row.get("source") or "").casefold():
        current["character_role"] = row.get("character_role")
    elif "anilist" not in str(current.get("source") or "").casefold() and role_priority(row.get("character_role")) < role_priority(current.get("character_role")):
        current["character_role"] = row.get("character_role")
    if language_priority(row.get("voice_actor_language")) < language_priority(current.get("voice_actor_language")):
        current["voice_actor_language"] = row.get("voice_actor_language")
    return current


def merge_edge_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    merged: dict[tuple[int, str, str], dict[str, Any]] = {}
    for row in rows:
        key = make_edge_key(row)
        current = merged.get(key)
        if current is None:
            merged[key] = dict(row)
            continue
        merged[key] = merge_edge_values(current, row)

    rows_out = list(merged.values())
    alias_merged: list[dict[str, Any]] = []
    for row in rows_out:
        match_index = None
        for idx, current in enumerate(alias_merged):
            if should_alias_merge_edges(current, row):
                match_index = idx
                break
        if match_index is None:
            alias_merged.append(dict(row))
        else:
            alias_merged[match_index] = merge_edge_values(alias_merged[match_index], row)
    return alias_merged


def relevant_extra_voice_actor_edge(row: dict[str, Any]) -> bool:
    priority = role_priority(row.get("character_role"))
    character_favorites = parse_int(row.get("character_favorites"), default=0) or 0
    character_relevance = float(row.get("character_relevance") or 0.0)
    if priority == 0:
        return True
    if priority == 1:
        return character_favorites >= 10 or character_relevance >= 0.01
    if priority == 2:
        return character_favorites >= 25 or character_relevance >= 0.03
    if priority == 3:
        return character_favorites >= 100 or character_relevance >= 0.05
    return False


def select_voice_actor_edges_for_anime(
    rows: list[dict[str, Any]],
    base_items: int,
    *,
    dynamic: bool = True,
    dynamic_cap: int = MAX_DYNAMIC_CHARACTER_VA_DETAILS_PER_ENTRY,
) -> list[dict[str, Any]]:
    by_character: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if is_generic_narrator(row):
            continue
        character_key = (
            str(first_present(row.get("character_id_anilist"), row.get("character_id_mal"))),
            canonical_name_key(row.get("character_name")),
        )
        by_character[character_key].append(row)

    selected: list[dict[str, Any]] = []
    for character_rows in by_character.values():
        best = sorted(
            character_rows,
            key=lambda row: (
                language_priority(row.get("voice_actor_language")),
                -(parse_int(row.get("voice_actor_favorites_mal"), default=0) or 0),
                -(parse_int(row.get("voice_actor_favorites_anilist"), default=0) or 0),
                normalize_key(row.get("voice_actor_name")),
            ),
        )[0]
        selected.append(best)

    max_character_favorites = max(
        [max(parse_int(row.get("character_favorites_mal"), default=0) or 0, parse_int(row.get("character_favorites_anilist"), default=0) or 0) for row in selected]
        or [0]
    )
    for row in selected:
        row["character_favorites"] = max(
            parse_int(row.get("character_favorites_mal"), default=0) or 0,
            parse_int(row.get("character_favorites_anilist"), default=0) or 0,
        )
        row["voice_actor_favorites"] = max(
            parse_int(row.get("voice_actor_favorites_mal"), default=0) or 0,
            parse_int(row.get("voice_actor_favorites_anilist"), default=0) or 0,
        )
        if max_character_favorites > 0:
            character_relevance = row["character_favorites"] / max_character_favorites
        else:
            character_relevance = 0.0
        row["character_relevance"] = round(character_relevance, 6)
        row["role_weight"] = role_weight(row.get("character_role"))
        row["relevance_score"] = round(
            row["role_weight"] * (1.0 + 0.25 * character_relevance) * math.log1p(float(row["voice_actor_favorites"] or 0)),
            6,
        )

    sorted_rows = sorted(
        selected,
        key=lambda row: (
            role_priority(row.get("character_role")),
            -int(row.get("character_favorites") or 0),
            -int(row.get("voice_actor_favorites") or 0),
            -float(row.get("relevance_score") or 0.0),
            canonical_name_key(row.get("character_name")),
        ),
    )
    if not dynamic:
        return sorted_rows[:base_items]

    chosen = list(sorted_rows[:base_items])
    seen_keys = {
        (
            canonical_name_key(row.get("character_name")),
            canonical_name_key(row.get("voice_actor_name")),
        )
        for row in chosen
    }
    for row in sorted_rows[base_items:]:
        if len(chosen) >= dynamic_cap:
            break
        key = (
            canonical_name_key(row.get("character_name")),
            canonical_name_key(row.get("voice_actor_name")),
        )
        if key in seen_keys:
            continue
        if relevant_extra_voice_actor_edge(row):
            chosen.append(row)
            seen_keys.add(key)
    return chosen


def build_voice_actor_edge_tables(
    df: pd.DataFrame,
    *,
    max_items: int = BASE_CHARACTER_VA_DETAILS_PER_ENTRY,
    dynamic: bool = True,
    dynamic_cap: int = MAX_DYNAMIC_CHARACTER_VA_DETAILS_PER_ENTRY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anilist_cache = load_anilist_cache()
    jikan_character_cache = load_json(JIKAN_CHARACTER_CACHE_FILE, {"items": {}})
    mal_character_by_id, mal_character_by_name = top_cache_indexes(JIKAN_TOP_CHARACTER_CACHE_FILE)
    mal_staff_by_id, mal_staff_by_name = top_cache_indexes(JIKAN_TOP_PEOPLE_CACHE_FILE)
    anilist_character_by_id, anilist_character_by_name = top_cache_indexes(ANILIST_TOP_CHARACTER_CACHE_FILE)
    anilist_staff_by_id, anilist_staff_by_name = top_cache_indexes(ANILIST_TOP_STAFF_CACHE_FILE)

    all_rows: list[dict[str, Any]] = []
    processed = 0
    for _, anime in df.iterrows():
        mal_id = parse_int(anime.get("mal_id"), default=None)
        if mal_id is None:
            continue
        processed += 1
        base = {
            "mal_id": int(mal_id),
            "anilist_id": parse_int(anime.get("anilist_id"), default=None),
            "title": anime.get("title"),
            "type": anime.get("type"),
        }
        candidates: list[dict[str, Any]] = []

        media = (anilist_cache.get("items", {}).get(str(mal_id), {}) or {}).get("media")
        if isinstance(media, dict):
            for character in media.get("characters") or []:
                if not isinstance(character, dict):
                    continue
                character_name = compact_name_japanese_order(character.get("name"))
                if not character_name:
                    continue
                character_id = parse_int(character.get("id"), default=None)
                character_favorites_anilist = max(
                    parse_int(character.get("favorites"), default=0) or 0,
                    favorite_from_source_indexes(anilist_character_by_id, anilist_character_by_name, character_id, character_name),
                )
                for actor in character.get("voice_actors") or []:
                    if not isinstance(actor, dict):
                        continue
                    actor_name = compact_name_japanese_order(actor.get("name"))
                    if not actor_name:
                        continue
                    actor_id = parse_int(actor.get("id"), default=None)
                    candidates.append(
                        {
                            **base,
                            "character_id_anilist": character_id,
                            "character_id_mal": pd.NA,
                            "character_name": character_name,
                            "character_role": str(character.get("role") or "").strip() or "Unknown",
                            "character_favorites_mal": favorite_from_source_indexes(mal_character_by_id, mal_character_by_name, None, character_name),
                            "character_favorites_anilist": character_favorites_anilist,
                            "voice_actor_id_anilist": actor_id,
                            "voice_actor_id_mal": pd.NA,
                            "voice_actor_name": actor_name,
                            "voice_actor_language": str(actor.get("language") or "").strip() or "Unknown",
                            "voice_actor_favorites_mal": favorite_from_source_indexes(mal_staff_by_id, mal_staff_by_name, None, actor_name),
                            "voice_actor_favorites_anilist": max(
                                parse_int(actor.get("favorites"), default=0) or 0,
                                favorite_from_source_indexes(anilist_staff_by_id, anilist_staff_by_name, actor_id, actor_name),
                            ),
                            "source": "AniList",
                        }
                    )

        jikan_item = (jikan_character_cache.get("items", {}).get(str(mal_id), {}) or {})
        for character in jikan_item.get("characters") or []:
            if not isinstance(character, dict):
                continue
            character_name = compact_name_japanese_order(character.get("name"))
            if not character_name:
                continue
            character_id = parse_int(character.get("id"), default=None)
            for actor in character.get("voice_actors") or []:
                if not isinstance(actor, dict):
                    continue
                actor_name = compact_name_japanese_order(actor.get("name"))
                if not actor_name:
                    continue
                actor_id = parse_int(actor.get("id"), default=None)
                candidates.append(
                    {
                        **base,
                        "character_id_anilist": pd.NA,
                        "character_id_mal": character_id,
                        "character_name": character_name,
                        "character_role": str(character.get("role") or "").strip() or "Unknown",
                        "character_favorites_mal": favorite_from_source_indexes(mal_character_by_id, mal_character_by_name, character_id, character_name),
                        "character_favorites_anilist": favorite_from_source_indexes(anilist_character_by_id, anilist_character_by_name, None, character_name),
                        "voice_actor_id_anilist": pd.NA,
                        "voice_actor_id_mal": actor_id,
                        "voice_actor_name": actor_name,
                        "voice_actor_language": str(actor.get("language") or "").strip() or "Japanese",
                        "voice_actor_favorites_mal": favorite_from_source_indexes(mal_staff_by_id, mal_staff_by_name, actor_id, actor_name),
                        "voice_actor_favorites_anilist": favorite_from_source_indexes(anilist_staff_by_id, anilist_staff_by_name, None, actor_name),
                        "source": "Jikan",
                    }
                )

        merged = merge_edge_rows(candidates)
        all_rows.extend(
            select_voice_actor_edges_for_anime(
                merged,
                base_items=max_items,
                dynamic=dynamic,
                dynamic_cap=dynamic_cap,
            )
        )
        if processed % 1000 == 0:
            print(
                f"VA edge table progress: processed={processed:,}, edge_rows={len(all_rows):,}",
                flush=True,
            )

    edge_df = pd.DataFrame(all_rows)
    if edge_df.empty:
        return edge_df, pd.DataFrame(), pd.DataFrame(), {
            "voice_actor_edge_anime_processed": processed,
            "voice_actor_edge_rows": 0,
            "voice_actor_index_rows": 0,
            "character_index_rows": 0,
        }

    edge_df["_role_priority"] = edge_df["character_role"].apply(role_priority)
    edge_df = edge_df.sort_values(
        ["mal_id", "_role_priority", "character_favorites", "voice_actor_favorites", "relevance_score", "character_name"],
        ascending=[True, True, False, False, False, True],
    ).drop(columns=["_role_priority"])
    edge_df["voice_actor_group_key"] = edge_df.apply(
        lambda row: person_group_key(row, ["voice_actor_id_mal", "voice_actor_id_anilist"], "voice_actor_name"),
        axis=1,
    )
    voice_actor_index = (
        edge_df.groupby(["voice_actor_group_key"], dropna=False)
        .agg(
            voice_actor_name=("voice_actor_name", preferred_person_name),
            voice_actor_id_anilist=("voice_actor_id_anilist", "first"),
            voice_actor_id_mal=("voice_actor_id_mal", "first"),
            voice_actor_favorites_mal=("voice_actor_favorites_mal", "max"),
            voice_actor_favorites_anilist=("voice_actor_favorites_anilist", "max"),
            voice_actor_favorites=("voice_actor_favorites", "max"),
            anime_count=("mal_id", "nunique"),
            main_role_count=("character_role", lambda values: int(sum(str(value).casefold() == "main" for value in values))),
            supporting_role_count=("character_role", lambda values: int(sum(str(value).casefold() == "supporting" for value in values))),
            anime_ids=("mal_id", lambda values: merge_pipe([str(int(value)) for value in sorted(set(values))])),
            anime_titles=("title", lambda values: merge_pipe([str(value) for value in values.dropna().head(25)])),
        )
        .reset_index()
        .drop(columns=["voice_actor_group_key"])
        .sort_values(["voice_actor_favorites", "main_role_count", "anime_count"], ascending=[False, False, False])
    )
    character_index = (
        edge_df.groupby(["character_name"], dropna=False)
        .agg(
            character_id_anilist=("character_id_anilist", "first"),
            character_id_mal=("character_id_mal", "first"),
            character_favorites_mal=("character_favorites_mal", "max"),
            character_favorites_anilist=("character_favorites_anilist", "max"),
            character_favorites=("character_favorites", "max"),
            anime_count=("mal_id", "nunique"),
            anime_ids=("mal_id", lambda values: merge_pipe([str(int(value)) for value in sorted(set(values))])),
            voice_actor_names=("voice_actor_name", lambda values: merge_pipe([str(value) for value in values.dropna().head(25)])),
        )
        .reset_index()
        .sort_values(["character_favorites", "anime_count"], ascending=[False, False])
    )
    return edge_df, voice_actor_index, character_index, {
        "voice_actor_edge_anime_processed": processed,
        "voice_actor_edge_rows": int(len(edge_df)),
        "voice_actor_edge_anime_with_rows": int(edge_df["mal_id"].nunique()),
        "voice_actor_index_rows": int(len(voice_actor_index)),
        "character_index_rows": int(len(character_index)),
    }


def normalized_staff_role(role: Any) -> str:
    text = str(role or "").strip()
    for target in TARGET_STAFF_ROLES:
        if text.casefold() == target.casefold():
            return target
    return ""


def staff_role_group(role: Any) -> str:
    return STAFF_ROLE_GROUPS.get(str(role or "").strip().casefold(), "")


def build_staff_edge_tables(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    anilist_cache = load_anilist_cache()
    mal_staff_by_id, mal_staff_by_name = top_cache_indexes(JIKAN_TOP_PEOPLE_CACHE_FILE)
    anilist_staff_by_id, anilist_staff_by_name = top_cache_indexes(ANILIST_TOP_STAFF_CACHE_FILE)
    rows: list[dict[str, Any]] = []
    processed = 0
    missing_staff_payload = 0
    skipped_non_target_roles = 0
    seen_edges: set[tuple[int, int, str]] = set()

    for _, anime in df.iterrows():
        mal_id = parse_int(anime.get("mal_id"), default=None)
        if mal_id is None:
            continue
        processed += 1
        media = (anilist_cache.get("items", {}).get(str(mal_id), {}) or {}).get("media")
        if not isinstance(media, dict) or not isinstance(media.get("staff"), list):
            missing_staff_payload += 1
            continue
        for staff in media.get("staff") or []:
            if not isinstance(staff, dict):
                continue
            role = normalized_staff_role(staff.get("role"))
            if not role:
                skipped_non_target_roles += 1
                continue
            staff_id = parse_int(staff.get("id"), default=None)
            staff_name = compact_name_japanese_order(staff.get("name"))
            if staff_id is None or not staff_name:
                continue
            edge_key = (int(mal_id), int(staff_id), role)
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            favorites_mal = favorite_from_source_indexes(mal_staff_by_id, mal_staff_by_name, None, staff_name)
            favorites_anilist = max(
                parse_int(staff.get("favorites"), default=0) or 0,
                favorite_from_source_indexes(anilist_staff_by_id, anilist_staff_by_name, staff_id, staff_name),
            )
            rows.append(
                {
                    "mal_id": int(mal_id),
                    "anilist_id": parse_int(anime.get("anilist_id"), default=None),
                    "title": anime.get("title"),
                    "type": anime.get("type"),
                    "staff_id_anilist": int(staff_id),
                    "staff_name": staff_name,
                    "staff_role": role,
                    "staff_role_group": staff_role_group(role),
                    "staff_language": str(staff.get("language") or "").strip(),
                    "staff_favorites_mal": int(favorites_mal),
                    "staff_favorites_anilist": int(favorites_anilist),
                    "staff_favorites": max(int(favorites_mal), int(favorites_anilist)),
                    "source": "AniList",
                }
            )
        if processed % 1000 == 0:
            print(
                f"Staff edge table progress: processed={processed:,}, edge_rows={len(rows):,}, missing_staff_payload={missing_staff_payload:,}",
                flush=True,
            )

    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        return edge_df, pd.DataFrame(), {
            "staff_edge_anime_processed": processed,
            "staff_edge_rows": 0,
            "staff_index_rows": 0,
            "staff_missing_payload_rows": missing_staff_payload,
            "staff_non_target_roles_skipped": skipped_non_target_roles,
        }

    role_order = {"original_creator": 0, "director": 1, "original_character_design": 2}
    edge_df["_role_order"] = edge_df["staff_role_group"].map(role_order).fillna(99)
    edge_df = edge_df.sort_values(
        ["mal_id", "_role_order", "staff_favorites", "staff_name"],
        ascending=[True, True, False, True],
    ).drop(columns=["_role_order"])
    edge_df["staff_group_key"] = edge_df.apply(
        lambda row: person_group_key(row, ["staff_id_anilist"], "staff_name"),
        axis=1,
    )

    staff_index = (
        edge_df.groupby(["staff_group_key"], dropna=False)
        .agg(
            staff_name=("staff_name", preferred_person_name),
            staff_id_anilist=("staff_id_anilist", "first"),
            staff_favorites_mal=("staff_favorites_mal", "max"),
            staff_favorites_anilist=("staff_favorites_anilist", "max"),
            staff_favorites=("staff_favorites", "max"),
            anime_count=("mal_id", "nunique"),
            original_creator_count=("staff_role_group", lambda values: int(sum(str(value) == "original_creator" for value in values))),
            director_count=("staff_role_group", lambda values: int(sum(str(value) == "director" for value in values))),
            original_character_design_count=("staff_role_group", lambda values: int(sum(str(value) == "original_character_design" for value in values))),
            staff_roles=("staff_role", lambda values: merge_pipe([str(value) for value in values.dropna()])),
            anime_ids=("mal_id", lambda values: merge_pipe([str(int(value)) for value in sorted(set(values))])),
            anime_titles=("title", lambda values: merge_pipe([str(value) for value in values.dropna().head(25)])),
        )
        .reset_index()
        .drop(columns=["staff_group_key"])
        .sort_values(
            ["staff_favorites", "original_creator_count", "director_count", "anime_count"],
            ascending=[False, False, False, False],
        )
    )
    return edge_df, staff_index, {
        "staff_edge_anime_processed": processed,
        "staff_edge_rows": int(len(edge_df)),
        "staff_edge_anime_with_rows": int(edge_df["mal_id"].nunique()),
        "staff_index_rows": int(len(staff_index)),
        "staff_missing_payload_rows": int(missing_staff_payload),
        "staff_non_target_roles_skipped": int(skipped_non_target_roles),
    }


def role_priority(role: Any) -> int:
    text = str(role or "").strip().casefold()
    if text == "main":
        return 0
    if text == "supporting":
        return 1
    if text == "background":
        return 3
    return 2


def build_va_character_details_for_anime(
    raw_characters: list[dict[str, Any]],
    person_cache: dict[str, Any],
    character_detail_cache: dict[str, Any],
    *,
    sleep_seconds: float,
    max_items: int = MAX_CHARACTER_VA_DETAILS_PER_ENTRY,
) -> str:
    if not raw_characters:
        return ""

    non_background = [item for item in raw_characters if role_priority(item.get("role")) < 3]
    selected_pool = non_background if len(non_background) >= max_items else list(raw_characters)
    candidate_rows: list[dict[str, Any]] = []

    for item in selected_pool:
        if not isinstance(item, dict):
            continue
        character_id = parse_int(item.get("id"), default=None)
        if character_id is None:
            continue
        char_detail = get_jikan_detail(character_detail_cache, "character", character_id, sleep_seconds=sleep_seconds)
        character_name = compact_name_japanese_order(char_detail.get("name") or item.get("name"))
        character_favorites = parse_int(char_detail.get("favorites"), default=0) or 0

        actor_choices = []
        for actor in item.get("voice_actors") or []:
            if not isinstance(actor, dict):
                continue
            actor_id = parse_int(actor.get("id"), default=None)
            if actor_id is None:
                continue
            actor_detail = get_jikan_detail(person_cache, "person", actor_id, sleep_seconds=sleep_seconds)
            actor_choices.append(
                {
                    "person_id": actor_id,
                    "person_name": compact_name_japanese_order(actor_detail.get("name") or actor.get("name")),
                    "person_favorites": parse_int(actor_detail.get("favorites"), default=0) or 0,
                }
            )
        if not actor_choices:
            continue
        actor_choice = sorted(actor_choices, key=lambda row: (-row["person_favorites"], row["person_id"]))[0]
        candidate_rows.append(
            {
                "role": str(item.get("role") or "").strip() or "Unknown",
                "role_priority": role_priority(item.get("role")),
                "character_id": character_id,
                "character_name": character_name,
                "character_favorites": character_favorites,
                **actor_choice,
            }
        )

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (row["role_priority"], -row["character_favorites"], -row["person_favorites"], row["character_id"]),
    )[:max_items]
    labels = []
    for row in candidate_rows:
        labels.append(
            ":".join(
                [
                    str(row["character_id"]),
                    safe_label(row["character_name"]),
                    str(row["character_favorites"]),
                    str(row["person_id"]),
                    safe_label(row["person_name"]),
                    str(row["person_favorites"]),
                    safe_label(row["role"]),
                ]
            )
        )
    return merge_pipe(labels)


def anilist_media_has_character_favorite_payload(media: dict[str, Any] | None) -> bool:
    if not isinstance(media, dict):
        return False
    characters = media.get("characters")
    if not isinstance(characters, list) or not characters:
        return False
    for character in characters:
        if not isinstance(character, dict) or "favorites" not in character:
            return False
        for actor in character.get("voice_actors") or []:
            if isinstance(actor, dict) and "favorites" not in actor:
                return False
    return True


def get_anilist_media_for_character_enrichment(
    mal_id: int,
    cache: dict[str, Any],
    *,
    sleep_seconds: float,
    live_if_missing: bool = True,
    save_immediately: bool = True,
) -> tuple[dict[str, Any] | None, str]:
    key = str(int(mal_id))
    cached_item = cache.get("items", {}).get(key) or {}
    cached_media = cached_item.get("media")
    if cached_item.get("character_favorite_query_version") == ANILIST_CHARACTER_FAVORITE_QUERY_VERSION:
        return cached_media if isinstance(cached_media, dict) else None, "cache"

    if not live_if_missing:
        return cached_media if isinstance(cached_media, dict) else None, "missing_character_payload"

    media = request_anilist_media(int(mal_id), sleep_seconds=sleep_seconds)
    cache.setdefault("items", {})[key] = {
        "fetched_at": now_iso(),
        "source": "anilist_graphql" if media else "anilist_graphql_missing",
        "query_version": ANILIST_MEDIA_CACHE_VERSION,
        "character_favorite_query_version": ANILIST_CHARACTER_FAVORITE_QUERY_VERSION,
        "media": media,
    }
    if save_immediately:
        save_anilist_cache(cache)
    return media, "live"


def build_va_character_details_from_anilist_media(
    media: dict[str, Any] | None,
    *,
    max_items: int = MAX_CHARACTER_VA_DETAILS_PER_ENTRY,
    favorite_indexes: dict[str, dict[Any, int]] | None = None,
) -> str:
    if not isinstance(media, dict):
        return ""

    raw_characters = media.get("characters") or []
    if not raw_characters:
        return ""

    non_background = [item for item in raw_characters if role_priority(item.get("role")) < 3]
    background = [item for item in raw_characters if role_priority(item.get("role")) >= 3]
    selected_pool = list(non_background)
    if len(selected_pool) < max_items:
        selected_pool.extend(background[: max_items - len(selected_pool)])

    candidate_rows: list[dict[str, Any]] = []
    for item in selected_pool:
        if not isinstance(item, dict):
            continue
        character_id = parse_int(item.get("id"), default=None)
        if character_id is None:
            continue
        character_name = compact_name_japanese_order(item.get("name"))
        character_favorites = best_indexed_favorites(
            item.get("favorites"),
            favorite_indexes,
            "character",
            character_id,
            character_name,
        )
        actor_choices = []
        for actor in item.get("voice_actors") or []:
            if not isinstance(actor, dict):
                continue
            actor_id = parse_int(actor.get("id"), default=None)
            if actor_id is None:
                continue
            actor_name = compact_name_japanese_order(actor.get("name"))
            actor_choices.append(
                {
                    "person_id": actor_id,
                    "person_name": actor_name,
                    "person_favorites": best_indexed_favorites(
                        actor.get("favorites"),
                        favorite_indexes,
                        "staff",
                        actor_id,
                        actor_name,
                    ),
                }
            )
        if not actor_choices:
            continue
        actor_choice = sorted(actor_choices, key=lambda row: (-row["person_favorites"], row["person_id"]))[0]
        candidate_rows.append(
            {
                "role": str(item.get("role") or "").strip() or "Unknown",
                "role_priority": role_priority(item.get("role")),
                "character_id": character_id,
                "character_name": character_name,
                "character_favorites": character_favorites,
                **actor_choice,
            }
        )

    candidate_rows = sorted(
        candidate_rows,
        key=lambda row: (row["role_priority"], -row["character_favorites"], -row["person_favorites"], row["character_id"]),
    )[:max_items]
    labels = []
    for row in candidate_rows:
        labels.append(
            ":".join(
                [
                    str(row["character_id"]),
                    safe_label(row["character_name"]),
                    str(row["character_favorites"]),
                    str(row["person_id"]),
                    safe_label(row["person_name"]),
                    str(row["person_favorites"]),
                    safe_label(row["role"]),
                ]
            )
        )
    return merge_pipe(labels)


def enrich_voice_actor_character_favorites(
    df: pd.DataFrame,
    *,
    limit: int | None,
    sleep_seconds: float,
) -> tuple[pd.DataFrame, dict[str, int]]:
    anilist_cache = load_anilist_cache()
    favorite_indexes, index_stats = build_character_favorite_indexes(anilist_cache, df)
    failures = load_json(ANILIST_CHARACTER_DETAIL_FAILED_FILE, {"updated_at": None, "items": {}})
    rows_updated = 0
    missing_anilist_character_rows = 0
    live_requests = 0
    cache_hits = 0
    processed = 0
    skipped_existing = 0
    failed_rows = 0
    if "voice_actor_character_favorites" not in df.columns:
        df["voice_actor_character_favorites"] = ""
    print(
        "VA/character favorite indexes ready: "
        f"character_ids={index_stats.get('character_favorite_index_ids', 0):,}, "
        f"character_names={index_stats.get('character_favorite_index_names', 0):,}, "
        f"staff_ids={index_stats.get('staff_favorite_index_ids', 0):,}, "
        f"staff_names={index_stats.get('staff_favorite_index_names', 0):,}",
        flush=True,
    )

    for idx, row in df.iterrows():
        if limit is not None and processed >= limit:
            break
        mal_id = parse_int(row.get("mal_id"), default=None)
        if mal_id is None:
            continue
        existing_label = row.get("voice_actor_character_favorites")
        retryable_failure = (failures.get("items", {}).get(str(mal_id)) or {}).get("retryable")
        cached_item = anilist_cache.get("items", {}).get(str(mal_id)) or {}
        cache_current = cached_item.get("character_favorite_query_version") == ANILIST_CHARACTER_FAVORITE_QUERY_VERSION
        if not is_missing(existing_label) and not retryable_failure and cache_current:
            skipped_existing += 1
            continue
        processed += 1
        try:
            media, source = get_anilist_media_for_character_enrichment(
                mal_id,
                anilist_cache,
                sleep_seconds=sleep_seconds,
                save_immediately=False,
            )
            if source == "live":
                live_requests += 1
            elif source == "cache":
                cache_hits += 1
            if source == "live" and isinstance(media, dict):
                for key, value in add_anilist_media_to_favorite_indexes(favorite_indexes, media).items():
                    index_stats[key] = index_stats.get(key, 0) + value
            label = build_va_character_details_from_anilist_media(media, favorite_indexes=favorite_indexes)
        except Exception as exc:
            failed_rows += 1
            record_detail_failure(failures, mal_id, str(exc), retryable=True)
            print(f"AniList VA/character enrichment failed: processed={processed:,}, MAL {mal_id}, error={exc}", flush=True)
            if processed % 25 == 0:
                save_anilist_cache(anilist_cache)
            continue
        if label:
            df.at[idx, "voice_actor_character_favorites"] = label
            rows_updated += 1
        else:
            missing_anilist_character_rows += 1
        clear_detail_failure(failures, mal_id)
        if processed % 25 == 0:
            save_anilist_cache(anilist_cache)
        if processed % 10 == 0 or rows_updated <= 3:
            print(
                "AniList VA/character enrichment progress: "
                f"processed={processed:,}, rows_updated={rows_updated:,}, "
                f"skipped_existing={skipped_existing:,}, failed={failed_rows:,}, "
                f"cache_hits={cache_hits:,}, live_requests={live_requests:,}, "
                f"missing_character_rows={missing_anilist_character_rows:,}",
                flush=True,
            )

    save_anilist_cache(anilist_cache)
    failures["updated_at"] = now_iso()
    atomic_write_json(ANILIST_CHARACTER_DETAIL_FAILED_FILE, failures)
    index_stats["character_favorite_index_ids"] = len(favorite_indexes["character_by_id"])
    index_stats["character_favorite_index_names"] = len(favorite_indexes["character_by_name"])
    index_stats["staff_favorite_index_ids"] = len(favorite_indexes["staff_by_id"])
    index_stats["staff_favorite_index_names"] = len(favorite_indexes["staff_by_name"])
    return df, {
        "voice_actor_character_rows_processed": processed,
        "voice_actor_character_rows_updated": rows_updated,
        "voice_actor_character_missing_raw_rows": missing_anilist_character_rows,
        "voice_actor_character_rows_skipped_existing": skipped_existing,
        "voice_actor_character_rows_failed": failed_rows,
        "anilist_character_cache_hits": cache_hits,
        "anilist_character_live_requests": live_requests,
        "anilist_media_cache_size": len(anilist_cache.get("items", {})),
        **{key: int(value) for key, value in index_stats.items()},
    }


def drop_auxiliary_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    keep_columns = {
        "mal_id",
        "anilist_id",
        "anidb_id",
        "url",
        "image_url",
        "title",
        "title_english",
        "type",
        "source",
        "episodes",
        "duration",
        "total_watch_minutes",
        "status",
        "rating",
        "score",
        "scored_by",
        "rank",
        "hentai_rank",
        "popularity",
        "members",
        "favorites",
        "aired_year",
        "aired_month",
        "season",
        "genres",
        "tags",
        "tag_weights",
        "explicit_tags",
        "explicit_tag_weights",
        "demographics",
        "studios",
        "characters",
        "voice_actors",
        "relations",
        "recommendations",
    }
    explicit_drop_columns = {
        "season_from_month",
        "demographics_anidb_weighted",
        "production_origin",
        "is_recap_like",
        "recap_reason",
        "recap_action",
        "voice_actor_character_favorites",
    }
    drop_columns = [
        column
        for column in df.columns
        if (
            column in explicit_drop_columns
            or (
                column not in keep_columns
                and (
                    column.endswith("_mal")
                    or column.endswith("_anilist")
                    or column.endswith("_anidb")
                    or column.endswith("_jikan")
                    or column.endswith("_weighted")
                    or column.endswith("_count")
                    or column.startswith("character_count")
                    or column.startswith("voice_actor_count")
                )
            )
        )
    ]
    # Keep-column protection for selected final fields that happen to match a
    # broad suffix rule, e.g. production_origin if a future suffix is added.
    drop_columns = [
        column
        for column in drop_columns
        if column not in keep_columns
    ]
    return df.drop(columns=drop_columns, errors="ignore"), drop_columns


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply post-build dataset improvements after 002 and 003.")
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-json", type=Path, default=INPUT_JSON)
    parser.add_argument("--anidb-cache", type=Path, default=RAW_ANIDB_CACHE)
    parser.add_argument("--enrich-character-favorites", action="store_true", help="Use AniList media character edges to fill VA-character favorite features.")
    parser.add_argument("--character-favorite-limit", type=int, default=None, help="Optional row limit for the live/cache VA-character favorite enrichment pass.")
    parser.add_argument("--jikan-detail-sleep", type=float, default=JIKAN_DETAIL_DELAY_SECONDS, help="Backward-compatible name for the AniList request delay.")
    parser.add_argument("--refresh-jikan-top-favorites", action="store_true", help="Cache Jikan /top/characters and /top/people pages before VA-character enrichment.")
    parser.add_argument("--jikan-top-pages", type=int, default=25, help="Number of Jikan top pages to cache for characters and people.")
    parser.add_argument("--jikan-top-character-pages", type=int, default=None, help="Override Jikan top character pages; useful because character favorites need a deeper crawl.")
    parser.add_argument("--jikan-top-people-pages", type=int, default=None, help="Override Jikan top people pages.")
    parser.add_argument("--refresh-anilist-top-favorites", action="store_true", help="Cache AniList top character/staff favorites before VA-character enrichment.")
    parser.add_argument("--anilist-top-pages", type=int, default=25, help="Number of AniList top pages to cache for characters and staff.")
    parser.add_argument("--anilist-top-per-page", type=int, default=50, help="AniList Page perPage value for top favorite cache refresh.")
    parser.add_argument("--build-va-character-tables", action="store_true", help="Write normalized anime-VA-character edge and index tables.")
    parser.add_argument("--va-character-edge-csv", type=Path, default=VOICE_ACTOR_EDGE_CSV)
    parser.add_argument("--voice-actor-index-csv", type=Path, default=VOICE_ACTOR_INDEX_CSV)
    parser.add_argument("--character-index-csv", type=Path, default=CHARACTER_INDEX_CSV)
    parser.add_argument("--build-staff-tables", action="store_true", help="Write normalized anime-staff edge and index tables for Original Creator, Director, and Original Character Design.")
    parser.add_argument("--staff-edge-csv", type=Path, default=STAFF_EDGE_CSV)
    parser.add_argument("--staff-index-csv", type=Path, default=STAFF_INDEX_CSV)
    parser.add_argument("--max-va-character-edges-per-anime", type=int, default=BASE_CHARACTER_VA_DETAILS_PER_ENTRY, help="Base VA-character edge count kept for every anime.")
    parser.add_argument("--max-dynamic-va-character-edges-per-anime", type=int, default=MAX_DYNAMIC_CHARACTER_VA_DETAILS_PER_ENTRY, help="Hard cap for dynamic VA-character extras on long-cast anime.")
    parser.add_argument("--disable-dynamic-va-character-edges", action="store_true", help="Use the base VA-character edge limit only.")
    parser.add_argument("--drop-auxiliary-columns", action="store_true", help="Write a slim final dataset by dropping source-comparison, flag, and count columns.")
    args = parser.parse_args()

    if not args.input_csv.exists():
        raise FileNotFoundError(f"Dataset not found: {args.input_csv}. Run 002 first.")
    if not args.anidb_cache.exists():
        raise FileNotFoundError(f"AniDB cache not found: {args.anidb_cache}. Run 001 first.")

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.input_csv)
    before = {
        "rows": int(len(df)),
        "missing_source": int(df["source"].apply(is_missing).sum()) if "source" in df else None,
        "missing_duration": int(df["duration"].apply(is_missing).sum()) if "duration" in df else None,
        "missing_rating": int(df["rating"].apply(is_missing).sum()) if "rating" in df else None,
        "missing_rank": int(df["rank"].apply(is_missing).sum()) if "rank" in df else None,
        "missing_aired_year": int(df["aired_year"].apply(is_missing).sum()) if "aired_year" in df else None,
        "missing_studios": int(df["studios"].apply(is_missing).sum()) if "studios" in df else None,
        "missing_genres": int(df["genres"].apply(is_missing).sum()) if "genres" in df else None,
        "missing_tags": int(df["tags"].apply(is_missing).sum()) if "tags" in df else None,
        "missing_demographics": int(df["demographics"].apply(is_missing).sum()) if "demographics" in df else None,
        "sparse_tag_rows": int(df["tags"].apply(lambda value: pipe_count(value) < SPARSE_TAG_COUNT).sum()) if "tags" in df else None,
    }

    anidb_cache = load_json(args.anidb_cache, {"items": {}})
    label_map = build_anidb_label_map(anidb_cache)
    anidb_lookup = build_anidb_label_lookup(anidb_cache, label_map)
    anidb_extra_lookup = build_anidb_extra_lookup(anidb_cache)
    df, improvement_counts = apply_anidb_fallback(df, anidb_lookup)
    df, anidb_extra_counts = add_anidb_extra_fields(df, anidb_extra_lookup)
    df, explicit_counts = apply_explicit_safeguards(df)
    df, neighbor_counts = repair_missing_from_neighbors(df)
    df, studio_counts = normalize_studios(df)
    df, origin_studio_counts = fill_missing_studios_from_origin(df)
    df, demographic_counts = resolve_demographics(df)
    df, dropped_empty_label_rows = drop_rows_without_genres_and_tags(df)
    df, drop_counts = drop_unrepairable_rows(df)
    df, rank_counts = rerank_dataset(df)
    df, graph_counts = filter_graph_edges_to_dataset(df)
    enrichment_counts: dict[str, int] = {}
    top_favorite_counts: dict[str, int] = {}
    if args.refresh_jikan_top_favorites:
        jikan_character_pages = args.jikan_top_character_pages if args.jikan_top_character_pages is not None else args.jikan_top_pages
        jikan_people_pages = args.jikan_top_people_pages if args.jikan_top_people_pages is not None else args.jikan_top_pages
        top_favorite_counts.update(
            refresh_jikan_top_favorites(
                "characters",
                pages=jikan_character_pages,
                sleep_seconds=args.jikan_detail_sleep,
            )
        )
        top_favorite_counts.update(
            refresh_jikan_top_favorites(
                "people",
                pages=jikan_people_pages,
                sleep_seconds=args.jikan_detail_sleep,
            )
        )
    if args.refresh_anilist_top_favorites:
        top_favorite_counts.update(
            refresh_anilist_top_favorites(
                "characters",
                pages=args.anilist_top_pages,
                per_page=args.anilist_top_per_page,
                sleep_seconds=ANILIST_TOP_DELAY_SECONDS,
            )
        )
        top_favorite_counts.update(
            refresh_anilist_top_favorites(
                "staff",
                pages=args.anilist_top_pages,
                per_page=args.anilist_top_per_page,
                sleep_seconds=ANILIST_TOP_DELAY_SECONDS,
            )
        )
    if args.enrich_character_favorites:
        df, enrichment_counts = enrich_voice_actor_character_favorites(
            df,
            limit=args.character_favorite_limit,
            sleep_seconds=args.jikan_detail_sleep,
        )
    va_table_counts: dict[str, int | str] = {}
    if args.build_va_character_tables:
        edge_df, voice_actor_index, character_index, va_table_counts_int = build_voice_actor_edge_tables(
            df,
            max_items=args.max_va_character_edges_per_anime,
            dynamic=not args.disable_dynamic_va_character_edges,
            dynamic_cap=args.max_dynamic_va_character_edges_per_anime,
        )
        args.va_character_edge_csv.parent.mkdir(parents=True, exist_ok=True)
        edge_df.to_csv(args.va_character_edge_csv, index=False)
        voice_actor_index.to_csv(args.voice_actor_index_csv, index=False)
        character_index.to_csv(args.character_index_csv, index=False)
        va_table_counts = {
            **va_table_counts_int,
            "voice_actor_edge_csv": str(args.va_character_edge_csv),
            "voice_actor_index_csv": str(args.voice_actor_index_csv),
            "character_index_csv": str(args.character_index_csv),
        }
    staff_table_counts: dict[str, int | str] = {}
    if args.build_staff_tables:
        staff_edge_df, staff_index, staff_table_counts_int = build_staff_edge_tables(df)
        args.staff_edge_csv.parent.mkdir(parents=True, exist_ok=True)
        staff_edge_df.to_csv(args.staff_edge_csv, index=False)
        staff_index.to_csv(args.staff_index_csv, index=False)
        staff_table_counts = {
            **staff_table_counts_int,
            "staff_edge_csv": str(args.staff_edge_csv),
            "staff_index_csv": str(args.staff_index_csv),
        }
    dropped_auxiliary_columns: list[str] = []
    if args.drop_auxiliary_columns:
        df, dropped_auxiliary_columns = drop_auxiliary_columns(df)

    after = {
        "rows": int(len(df)),
        "missing_source": int(df["source"].apply(is_missing).sum()) if "source" in df else None,
        "missing_duration": int(df["duration"].apply(is_missing).sum()) if "duration" in df else None,
        "missing_rating": int(df["rating"].apply(is_missing).sum()) if "rating" in df else None,
        "missing_rank": int(df["rank"].apply(is_missing).sum()) if "rank" in df else None,
        "missing_aired_year": int(df["aired_year"].apply(is_missing).sum()) if "aired_year" in df else None,
        "missing_studios": int(df["studios"].apply(is_missing).sum()) if "studios" in df else None,
        "missing_genres": int(df["genres"].apply(is_missing).sum()) if "genres" in df else None,
        "missing_tags": int(df["tags"].apply(is_missing).sum()) if "tags" in df else None,
        "missing_demographics": int(df["demographics"].apply(is_missing).sum()) if "demographics" in df else None,
        "sparse_tag_rows": int(df["tags"].apply(lambda value: pipe_count(value) < SPARSE_TAG_COUNT).sum()) if "tags" in df else None,
    }

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    written_csv = write_csv_with_pending_fallback(df, args.output_csv)
    written_json = write_json_with_pending_fallback(df, args.output_json)
    summary = {
        "updated_at": now_iso(),
        "input_csv": str(args.input_csv),
        "output_csv": written_csv,
        "output_json": written_json,
        "anidb_label_map_csv": str(ANIDB_LABEL_MAP_CSV),
        "anidb_tags_total": int(len(label_map)),
        "anidb_tags_mapped": int((label_map["target_kind"] != "ignore").sum()) if not label_map.empty else 0,
        "before": before,
        "after": after,
        "dropped_empty_genre_and_tag_rows": dropped_empty_label_rows,
        "dropped_auxiliary_columns": dropped_auxiliary_columns,
        **improvement_counts,
        **anidb_extra_counts,
        **explicit_counts,
        **neighbor_counts,
        **studio_counts,
        **origin_studio_counts,
        **demographic_counts,
        **drop_counts,
        **rank_counts,
        **graph_counts,
        **top_favorite_counts,
        **enrichment_counts,
        **va_table_counts,
        **staff_table_counts,
    }
    atomic_write_json(IMPROVEMENT_SUMMARY_FILE, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
