from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
for import_path in (SCRIPT_DIR, ROOT):
    import_path_text = str(import_path)
    if import_path_text not in sys.path:
        sys.path.insert(0, import_path_text)

try:
    from anilist_metadata_utils import (
        ANILIST_CACHE_FILE,
        get_anilist_media,
        load_anilist_cache,
        save_anilist_cache,
        seed_anilist_cache_from_sample,
        update_anilist_cache_for_mal_ids,
    )
except ImportError:  # Imported from notebooks.
    from src.anilist_metadata_utils import (
        ANILIST_CACHE_FILE,
        get_anilist_media,
        load_anilist_cache,
        save_anilist_cache,
        seed_anilist_cache_from_sample,
        update_anilist_cache_for_mal_ids,
    )

try:
    from anidb_metadata_utils import extract_anidb_payload
except ImportError:
    from src.anidb_metadata_utils import extract_anidb_payload


RAW_DIR = ROOT / "data" / "raw"
RAW_SOURCE_DIR = ROOT / "data" / "raw_sources"
MAL_RAW_DIR = RAW_SOURCE_DIR / "mal_jikan"
ANILIST_RAW_DIR = RAW_SOURCE_DIR / "anilist"
ANIDB_RAW_DIR = RAW_SOURCE_DIR / "anidb"
BUILD_DIR = ROOT / "data" / "build"
LOG_DIR = ROOT / "logs"

MAL_CANDIDATE_IDS_FILE = RAW_DIR / "mal_candidate_ids.json"
MAL_IDS_CACHE_URL = "https://raw.githubusercontent.com/purarue/mal-id-cache/master/cache/anime_cache.json"
ANIDB_XML_CACHE_URL = "https://files.shokoanime.com/files/shoko-server/other/Anime_HTTP.zip"
ANIDB_XML_CACHE_ZIP = RAW_DIR / "Anime_HTTP.zip"
JIKAN_ANIME_CACHE_FILE = MAL_RAW_DIR / "jikan_anime_full_cache.json"
JIKAN_RECOMMENDATION_CACHE_FILE = MAL_RAW_DIR / "jikan_recommendation_cache.json"
JIKAN_CHARACTER_CACHE_FILE = MAL_RAW_DIR / "jikan_character_voice_actor_cache.json"
RAW_SOURCE_INDEX_CSV = RAW_SOURCE_DIR / "raw_source_index.csv"
RAW_GATHER_CHECKPOINT_FILE = BUILD_DIR / "raw_dataset_gather_checkpoint.json"
RAW_GATHER_FAILED_FILE = BUILD_DIR / "raw_dataset_failed_api_requests.json"
RAW_GATHER_INVALID_FILE = BUILD_DIR / "raw_dataset_invalid_ids.json"
RAW_GATHER_SUMMARY_FILE = BUILD_DIR / "raw_dataset_gather_summary.json"
SEASONAL_REFRESH_CANDIDATES_CSV = BUILD_DIR / "seasonal_refresh_candidates.csv"
SEASONAL_DISCOVERY_CSV = BUILD_DIR / "seasonal_discovered_jikan_ids.csv"
SEASONAL_MAL_DISCOVERY_CSV = BUILD_DIR / "seasonal_discovered_mal_html_ids.csv"
RAW_GATHER_LOG_FILE = LOG_DIR / "raw_dataset_gather_log.txt"
ANILIST_RAW_CACHE_FILE = ANILIST_RAW_DIR / "anilist_media_cache.json"
ANIDB_RAW_CACHE_FILE = ANIDB_RAW_DIR / "anidb_metadata_cache.json"
ANIDB_SOURCE_CACHE_FILE = ANIDB_RAW_CACHE_FILE
ANIDB_RECENT_BACKFILL_SINCE = "2025-09-14"
ANIDB_REQUEST_DELAY_SECONDS = 4.0
ANIDB_REQUEST_JITTER_SECONDS = 0.75

JIKAN_API_BASE = "https://api.jikan.moe/v4"
MAL_WEB_BASE = "https://myanimelist.net"
VALID_TYPES = {"TV", "Movie", "OVA", "ONA", "Special", "TV Special"}
NOT_YET_AIRED_STATUS = "Not yet aired"
CURRENTLY_AIRING_STATUS = "Currently Airing"
REQUEST_HEADERS = {
    "User-Agent": "anime-recommender-course-project/1.0 (+local research notebook)",
}


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def normalize_secret_key(name: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(name).upper())


def parse_secret_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[normalize_secret_key(key)] = value.strip().strip("\"'")
    return values


def read_local_secret(*names: str, filename: str | None = None, default: str = "") -> str:
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


def ensure_dirs() -> None:
    for path in [MAL_RAW_DIR, ANILIST_RAW_DIR, ANIDB_RAW_DIR, BUILD_DIR, LOG_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def write_log(message: str) -> None:
    ensure_dirs()
    line = f"[{now_iso()}] {message}"
    print(line, flush=True)
    with RAW_GATHER_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


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
    payload = load_json(path, {"updated_at": None, "items": {}})
    payload.setdefault("items", {})
    return payload


def save_cache(path: Path, payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    atomic_write_json(path, payload)


def download_file(url: str, path: Path, timeout: int = 120) -> None:
    if requests is None:
        raise RuntimeError("requests is not installed; live downloads are unavailable")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    write_log(f"DOWNLOAD_START | {url} -> {path}")
    with requests.get(url, headers=REQUEST_HEADERS, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
    tmp.replace(path)
    write_log(f"DOWNLOAD_COMPLETE | {path}")


def download_mal_candidate_ids(force: bool = True) -> None:
    if force or not MAL_CANDIDATE_IDS_FILE.exists():
        download_file(MAL_IDS_CACHE_URL, MAL_CANDIDATE_IDS_FILE)
    payload = load_json(MAL_CANDIDATE_IDS_FILE, {})
    if not isinstance(payload, (dict, list)):
        raise RuntimeError(f"Unexpected MAL candidate id cache shape: {type(payload).__name__}")
    write_log(f"MAL_CANDIDATE_IDS_READY | {MAL_CANDIDATE_IDS_FILE}")


def load_candidate_ids() -> list[int]:
    payload = load_json(MAL_CANDIDATE_IDS_FILE, {})
    values: list[int] = []
    if isinstance(payload, dict):
        for key in ["sfw", "nsfw"]:
            values.extend(int(value) for value in payload.get(key, []) if str(value).isdigit())
    elif isinstance(payload, list):
        values = [int(value) for value in payload if str(value).isdigit()]
    return sorted(set(values))


def zip_info_datetime(info: zipfile.ZipInfo) -> str | None:
    try:
        return datetime(*info.date_time).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def compact_anidb_entry(entry: dict[str, Any]) -> dict[str, Any]:
    summary = entry.get("episode_summary") or {}
    keep_summary = {
        key: summary[key]
        for key in [
            "listed_episode_count",
            "length_count",
            "regular_episode_count",
            "preferred_episode_count",
            "preferred_length_count",
            "preferred_basis",
            "compilation_episode_count",
            "part_episode_count",
            "average_length_minutes",
            "total_length_minutes",
            "preferred_average_length_minutes",
            "preferred_total_length_minutes",
            "regular_length_count",
            "regular_average_length_minutes",
            "regular_total_length_minutes",
        ]
        if key in summary and summary[key] is not None
    }
    entry["episode_summary"] = keep_summary
    return entry


def refresh_shoko_anidb_cache(delete_zip_after_import: bool = True) -> dict[str, Any]:
    download_file(ANIDB_XML_CACHE_URL, ANIDB_XML_CACHE_ZIP, timeout=180)
    cache = load_cache(ANIDB_SOURCE_CACHE_FILE)
    items = cache.setdefault("items", {})
    source_downloaded_at = now_iso()
    pattern = re.compile(r"AnimeDoc_(\d+)\.xml$", re.IGNORECASE)
    imported = 0
    parse_errors = 0

    try:
        with zipfile.ZipFile(ANIDB_XML_CACHE_ZIP) as zf:
            infos = [info for info in zf.infolist() if pattern.search(info.filename)]
            for position, info in enumerate(infos, start=1):
                match = pattern.search(info.filename)
                if not match:
                    continue
                anidb_id = match.group(1)
                try:
                    root = ET.fromstring(zf.read(info))
                except ET.ParseError:
                    parse_errors += 1
                    continue
                if root.tag.lower() == "error":
                    parse_errors += 1
                    continue

                payload = extract_anidb_payload(root)
                payload["cached_at"] = zip_info_datetime(info) or source_downloaded_at
                payload["source"] = "shoko_xml_cache"
                payload["xml_file_modified_at"] = zip_info_datetime(info)
                payload["source_url"] = ANIDB_XML_CACHE_URL
                payload["source_downloaded_at"] = source_downloaded_at
                existing = items.get(anidb_id) or {}
                if existing.get("source") == "live_http" and existing.get("episode_count"):
                    payload["episode_count"] = existing["episode_count"]
                items[anidb_id] = compact_anidb_entry({**existing, **payload})
                imported += 1
                if imported % 500 == 0:
                    save_cache(ANIDB_SOURCE_CACHE_FILE, cache)
                    write_log(f"SHOKO_PROGRESS | imported={imported}/{len(infos)} | parse_errors={parse_errors}")
    finally:
        if delete_zip_after_import and ANIDB_XML_CACHE_ZIP.exists():
            ANIDB_XML_CACHE_ZIP.unlink()
            write_log(f"DISPOSABLE_FILE_DELETED | {ANIDB_XML_CACHE_ZIP}")

    save_cache(ANIDB_SOURCE_CACHE_FILE, cache)
    write_log(f"SHOKO_COMPLETE | imported={imported} | parse_errors={parse_errors} | total_entries={len(items)}")
    return {"shoko_imported": imported, "shoko_parse_errors": parse_errors, "anidb_cache_entries": len(items)}


def parse_anidb_id(data: dict[str, Any]) -> int | None:
    direct_id = data.get("anidb_id")
    if direct_id is not None:
        try:
            return int(direct_id)
        except (TypeError, ValueError):
            pass
    for item in data.get("external") or []:
        if item.get("name") == "AniDB":
            url = str(item.get("url") or "")
            match = re.search(r"(?:aid=|/anime/)(\d+)", url)
            if match:
                return int(match.group(1))
    return None


def compact_jikan_name_list(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    for item in items:
        if not isinstance(item, dict):
            continue
        compact = {}
        if item.get("mal_id") is not None:
            compact["mal_id"] = item.get("mal_id")
        if item.get("type") is not None:
            compact["type"] = item.get("type")
        if item.get("name") is not None:
            compact["name"] = item.get("name")
        if compact:
            out.append(compact)
    return out


def compact_jikan_relations(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    out = []
    for group in items:
        if not isinstance(group, dict):
            continue
        entries = compact_jikan_name_list(group.get("entry"))
        if entries:
            out.append({"relation": group.get("relation"), "entry": entries})
    return out


def compact_jikan_images(images: Any) -> dict[str, Any]:
    jpg = ((images or {}).get("jpg") or {}) if isinstance(images, dict) else {}
    large = jpg.get("large_image_url") or jpg.get("image_url")
    return {"jpg": {"large_image_url": large}} if large else {}


def compact_jikan_anime_response(response: dict[str, Any]) -> dict[str, Any]:
    data = (response or {}).get("data") or {}
    if not isinstance(data, dict):
        return response
    keep_keys = [
        "mal_id",
        "url",
        "title",
        "title_english",
        "title_japanese",
        "title_synonyms",
        "type",
        "source",
        "episodes",
        "status",
        "duration",
        "rating",
        "score",
        "scored_by",
        "rank",
        "popularity",
        "members",
        "favorites",
        "synopsis",
        "background",
        "season",
        "year",
        "aired",
        "airing",
        "approved",
    ]
    compact = {key: data.get(key) for key in keep_keys if key in data}
    compact["images"] = compact_jikan_images(data.get("images"))
    compact["genres"] = compact_jikan_name_list(data.get("genres"))
    compact["explicit_genres"] = compact_jikan_name_list(data.get("explicit_genres"))
    compact["themes"] = compact_jikan_name_list(data.get("themes"))
    compact["demographics"] = compact_jikan_name_list(data.get("demographics"))
    compact["studios"] = compact_jikan_name_list(data.get("studios"))
    compact["relations"] = compact_jikan_relations(data.get("relations"))
    anidb_id = parse_anidb_id(data)
    if anidb_id is not None:
        compact["anidb_id"] = int(anidb_id)
    return {"data": compact}


def compact_jikan_recommendation_response(response: dict[str, Any]) -> dict[str, Any]:
    compact_items = []
    for item in (response or {}).get("data") or []:
        if not isinstance(item, dict):
            continue
        entry = item.get("entry") or {}
        if not isinstance(entry, dict):
            continue
        mal_id = item.get("mal_id") or entry.get("mal_id")
        try:
            mal_id = int(mal_id)
        except (TypeError, ValueError):
            continue
        compact_items.append(
            {
                "mal_id": mal_id,
                "title": item.get("title") or entry.get("title"),
                "votes": item.get("votes") or 1,
            }
        )
    return {"data": compact_items}


def parse_mal_aired_date(data: dict[str, Any]) -> datetime | None:
    prop = ((data.get("aired") or {}).get("prop") or {}).get("from") or {}
    year = prop.get("year")
    month = prop.get("month")
    day = prop.get("day") or 1
    try:
        if not year or not month:
            return None
        return datetime(int(year), int(month), int(day))
    except (TypeError, ValueError):
        return None


def parse_iso_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for candidate in [text, text.replace("Z", "+00:00")]:
        try:
            parsed = datetime.fromisoformat(candidate)
            return parsed.replace(tzinfo=None)
        except ValueError:
            continue
    return None


def cache_age_hours(value: Any, reference: datetime | None = None) -> float | None:
    fetched_at = parse_iso_datetime(value)
    if fetched_at is None:
        return None
    reference = reference or datetime.now()
    return max(0.0, (reference - fetched_at).total_seconds() / 3600)


def season_for_month(month: int) -> str:
    if month in {1, 2, 3}:
        return "winter"
    if month in {4, 5, 6}:
        return "spring"
    if month in {7, 8, 9}:
        return "summer"
    return "fall"


def next_season(season: str, year: int) -> tuple[str, int]:
    order = ["winter", "spring", "summer", "fall"]
    idx = order.index(season)
    if idx == len(order) - 1:
        return "winter", year + 1
    return order[idx + 1], year


def previous_season(season: str, year: int) -> tuple[str, int]:
    order = ["winter", "spring", "summer", "fall"]
    idx = order.index(season)
    if idx == 0:
        return "fall", year - 1
    return order[idx - 1], year


def seasonal_target_window(reference_date: datetime | None = None, *, include_previous: bool = False) -> list[tuple[str, int]]:
    """Return the project seasonal window: current season plus prior seasons.

    For recommendation freshness we do not want upcoming/Fall placeholders to
    dominate the refresh just as Summer starts.  The useful window is the
    current season and, when requested, the two immediately preceding seasons.
    On 2026-07-11 this is Summer 2026, Spring 2026, and Winter 2026.
    """

    reference_date = reference_date or datetime.now()
    current = (season_for_month(reference_date.month), reference_date.year)
    if not include_previous:
        return [current]
    previous = previous_season(*current)
    previous_two = previous_season(*previous)
    return [current, previous, previous_two]


def season_year_from_jikan(data: dict[str, Any]) -> tuple[str | None, int | None]:
    season = str(data.get("season") or "").strip().casefold() or None
    year = data.get("year")
    if year is None:
        aired_date = parse_mal_aired_date(data)
        year = aired_date.year if aired_date else None
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return season, year_int


def season_year_from_anilist(media: dict[str, Any] | None) -> tuple[str | None, int | None]:
    if not isinstance(media, dict):
        return None, None
    season = str(media.get("season") or "").strip().casefold() or None
    year = media.get("seasonYear")
    if year is None:
        start = media.get("startDate") or {}
        year = start.get("year")
        month = start.get("month")
        if season is None and month:
            try:
                season = season_for_month(int(month))
            except (TypeError, ValueError):
                pass
    try:
        year_int = int(year) if year is not None else None
    except (TypeError, ValueError):
        year_int = None
    return season, year_int


def seasonal_refresh_candidates(
    reference_date: datetime | None = None,
    *,
    max_age_hours: float = 12.0,
    include_previous: bool = False,
) -> list[dict[str, Any]]:
    reference_date = reference_date or datetime.now()
    target_window = seasonal_target_window(reference_date, include_previous=include_previous)
    current = target_window[0]
    previous = target_window[1] if len(target_window) > 1 else previous_season(*current)
    previous_two = target_window[2] if len(target_window) > 2 else previous_season(*previous)
    target_seasons = set(target_window)

    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    anilist_cache = load_anilist_cache()
    candidates: list[dict[str, Any]] = []
    for key, item in anime_cache.get("items", {}).items():
        if not str(key).isdigit():
            continue
        data = (item.get("response") or {}).get("data") or {}
        accepted, reason = anime_acceptance(data)
        if not accepted:
            continue
        mal_id = int(key)
        anilist_item = anilist_cache.get("items", {}).get(key) or {}
        media = anilist_item.get("media") or {}

        jikan_status = str(data.get("status") or "").strip()
        anilist_status = str(media.get("status") or "").strip().upper()
        is_currently_airing = (
            jikan_status == CURRENTLY_AIRING_STATUS
            or bool(data.get("airing"))
            or anilist_status == "RELEASING"
            or bool((media.get("next_airing_episode") or {}))
        )
        is_not_yet_aired = jikan_status == NOT_YET_AIRED_STATUS or anilist_status == "NOT_YET_RELEASED"

        jikan_season, jikan_year = season_year_from_jikan(data)
        anilist_season, anilist_year = season_year_from_anilist(media)
        aired_date = parse_mal_aired_date(data)
        near_air_window = (
            aired_date is not None
            and (reference_date - aired_date).days <= 30
            and (aired_date - reference_date).days <= 180
        )
        in_target_season = (
            (jikan_season, jikan_year) in target_seasons
            or (anilist_season, anilist_year) in target_seasons
            or near_air_window
        )
        if not is_currently_airing and not (is_not_yet_aired and in_target_season):
            continue

        jikan_age = cache_age_hours(item.get("fetched_at"), reference_date)
        anilist_age = cache_age_hours(anilist_item.get("fetched_at"), reference_date)
        needs_jikan = jikan_age is None or jikan_age >= max_age_hours
        needs_anilist = anilist_age is None or anilist_age >= max_age_hours
        if not needs_jikan and not needs_anilist:
            continue

        candidates.append(
            {
                "mal_id": mal_id,
                "title": data.get("title"),
                "type": data.get("type"),
                "jikan_status": jikan_status,
                "anilist_status": anilist_status,
                "score": data.get("score"),
                "popularity": data.get("popularity") if data.get("popularity") is not None else 999999999,
                "members": data.get("members") if data.get("members") is not None else 0,
                "jikan_season": jikan_season,
                "jikan_year": jikan_year,
                "anilist_season": anilist_season,
                "anilist_year": anilist_year,
                "current_season": current[0],
                "current_year": current[1],
                "previous_season": previous[0],
                "previous_year": previous[1],
                "previous_two_season": previous_two[0],
                "previous_two_year": previous_two[1],
                "is_currently_airing": is_currently_airing,
                "is_not_yet_aired": is_not_yet_aired,
                "jikan_cache_age_hours": round(jikan_age, 2) if jikan_age is not None else None,
                "anilist_cache_age_hours": round(anilist_age, 2) if anilist_age is not None else None,
                "needs_jikan": needs_jikan,
                "needs_anilist": needs_anilist,
            }
        )

    return sorted(
        candidates,
        key=lambda row: (
            0 if row["is_currently_airing"] else 1,
            int(row["popularity"] or 999999999),
            -int(row["members"] or 0),
            int(row["mal_id"]),
        ),
    )


def paged_jikan_get(endpoint: str, *, page: int) -> dict[str, Any]:
    separator = "&" if "?" in endpoint else "?"
    return jikan_get(f"{endpoint}{separator}page={page}")


def discover_jikan_season_ids(
    reference_date: datetime | None = None,
    *,
    include_previous: bool = False,
    include_current: bool = True,
    include_next: bool = False,
    page_limit: int | None = None,
) -> list[dict[str, Any]]:
    """Discover seasonal MAL ids directly from Jikan season endpoints.

    The normal MAL id cache is broad but not season-aware, and the local full
    Jikan cache can miss very new seasonal entries until a full crawl reaches
    them. Jikan's season endpoints are a small, targeted source of seasonal MAL
    ids, so we use them before the full-cache seasonal refresh.
    """
    reference_date = reference_date or datetime.now()
    target_window = seasonal_target_window(reference_date, include_previous=include_previous)
    current = target_window[0]
    upcoming = next_season(*current)
    endpoints: list[tuple[str, str, int]] = []
    if include_current:
        endpoints.append(("/seasons/now", current[0], current[1]))
    for season, year in target_window:
        endpoints.append((f"/seasons/{year}/{season}", season, year))
    if include_next:
        endpoints.append((f"/seasons/{upcoming[1]}/{upcoming[0]}", upcoming[0], upcoming[1]))

    rows: list[dict[str, Any]] = []
    seen_endpoint_page: set[tuple[str, int]] = set()
    seen_row_key: set[tuple[int, str]] = set()
    for endpoint, endpoint_season, endpoint_year in endpoints:
        page = 1
        while True:
            if page_limit is not None and page > int(page_limit):
                break
            if (endpoint, page) in seen_endpoint_page:
                break
            seen_endpoint_page.add((endpoint, page))
            try:
                payload = paged_jikan_get(endpoint, page=page)
            except Exception as exc:
                write_log(f"SEASONAL_DISCOVERY_FAILED | endpoint={endpoint} | page={page} | {exc}")
                break
            data_rows = payload.get("data") or []
            for data in data_rows:
                try:
                    mal_id = int(data.get("mal_id"))
                except (TypeError, ValueError):
                    continue
                key = (mal_id, endpoint)
                if key in seen_row_key:
                    continue
                seen_row_key.add(key)
                accepted, reason = anime_acceptance(data)
                rows.append(
                    {
                        "mal_id": mal_id,
                        "title": data.get("title"),
                        "type": data.get("type"),
                        "status": data.get("status"),
                        "airing": bool(data.get("airing")),
                        "score": data.get("score"),
                        "scored_by": data.get("scored_by"),
                        "members": data.get("members") if data.get("members") is not None else 0,
                        "popularity": data.get("popularity") if data.get("popularity") is not None else 999999999,
                        "season": data.get("season") or endpoint_season,
                        "year": data.get("year") or endpoint_year,
                        "endpoint": endpoint,
                        "page": page,
                        "accepted_by_project_rules": accepted,
                        "acceptance_reason": reason,
                    }
                )
            pagination = payload.get("pagination") or {}
            if not pagination.get("has_next_page"):
                break
            page += 1

    deduped: dict[int, dict[str, Any]] = {}
    for row in sorted(
        rows,
        key=lambda item: (
            0 if item["accepted_by_project_rules"] else 1,
            0 if item["status"] == CURRENTLY_AIRING_STATUS or item["airing"] else 1,
            int(item["popularity"] or 999999999),
            -int(item["members"] or 0),
            int(item["mal_id"]),
        ),
    ):
        deduped.setdefault(int(row["mal_id"]), row)
    return list(deduped.values())


def stale_or_missing_jikan_ids(
    rows: list[dict[str, Any]],
    reference_date: datetime,
    *,
    max_age_hours: float,
) -> list[int]:
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    ids: list[int] = []
    for row in rows:
        if not row.get("accepted_by_project_rules"):
            continue
        mal_id = int(row["mal_id"])
        item = anime_cache.get("items", {}).get(str(mal_id)) or {}
        age = cache_age_hours(item.get("fetched_at"), reference_date)
        if age is None or age >= max_age_hours:
            ids.append(mal_id)
    return sorted(set(ids))


def mal_season_url(season: str, year: int, *, current: bool = False) -> str:
    if current:
        return f"{MAL_WEB_BASE}/anime/season"
    return f"{MAL_WEB_BASE}/anime/season/{int(year)}/{season}"


def discover_mal_season_page_ids(
    reference_date: datetime | None = None,
    *,
    include_previous: bool = False,
) -> list[dict[str, Any]]:
    """Scrape MAL season pages for anime IDs when Jikan season discovery fails.

    This intentionally gathers only IDs and light page metadata.  The actual
    canonical fields still come from the normal Jikan `/anime/{id}/full`,
    AniList, and improvement-table pipeline after these IDs are queued.
    """

    if requests is None:
        write_log("MAL_SEASON_HTML_DISCOVERY_SKIPPED | requests unavailable")
        return []

    reference_date = reference_date or datetime.now()
    target_window = seasonal_target_window(reference_date, include_previous=include_previous)
    endpoints: list[tuple[str, str, int]] = [(mal_season_url(target_window[0][0], target_window[0][1], current=True), *target_window[0])]
    endpoints.extend((mal_season_url(season, year), season, year) for season, year in target_window)

    rows: list[dict[str, Any]] = []
    seen_pages: set[str] = set()
    seen_pairs: set[tuple[int, str]] = set()
    anime_href = re.compile(r'href=["\'](?:https://myanimelist\.net)?/anime/(\d+)(?:/([^"\'#?<> ]+))?', re.I)
    for url, season, year in endpoints:
        if url in seen_pages:
            continue
        seen_pages.add(url)
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=40)
            response.raise_for_status()
            html = response.text
        except Exception as exc:
            write_log(f"MAL_SEASON_HTML_DISCOVERY_FAILED | url={url} | {type(exc).__name__}: {exc}")
            continue

        page_ids = 0
        for match in anime_href.finditer(html):
            try:
                mal_id = int(match.group(1))
            except (TypeError, ValueError):
                continue
            key = (mal_id, url)
            if key in seen_pairs:
                continue
            seen_pairs.add(key)
            slug = (match.group(2) or "").replace("_", " ").strip()
            rows.append(
                {
                    "mal_id": mal_id,
                    "title_hint": slug,
                    "type": None,
                    "status": None,
                    "airing": None,
                    "score": None,
                    "scored_by": None,
                    "members": 0,
                    "popularity": 999999999,
                    "season": season,
                    "year": year,
                    "endpoint": url,
                    "page": None,
                    "accepted_by_project_rules": True,
                    "acceptance_reason": "mal_season_html_id_discovery",
                    "source": "mal_html",
                }
            )
            page_ids += 1
        write_log(f"MAL_SEASON_HTML_DISCOVERY_PAGE | url={url} | ids={page_ids}")

    deduped: dict[int, dict[str, Any]] = {}
    for row in rows:
        deduped.setdefault(int(row["mal_id"]), row)
    return list(deduped.values())


def parse_date_arg(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d")


def is_hentai_jikan(data: dict[str, Any]) -> bool:
    rating = str(data.get("rating") or "")
    explicit = {str(item.get("name") or "").casefold() for item in data.get("explicit_genres") or []}
    genres = {str(item.get("name") or "").casefold() for item in data.get("genres") or []}
    return "Rx - Hentai".casefold() in rating.casefold() or "hentai" in explicit or "hentai" in genres


def recent_anidb_live_candidates(since: str = ANIDB_RECENT_BACKFILL_SINCE) -> list[dict[str, Any]]:
    since_date = parse_date_arg(since)
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    source_cache = load_cache(ANIDB_SOURCE_CACHE_FILE)
    raw_cache = load_cache(ANIDB_RAW_CACHE_FILE)
    cached_items = {}
    cached_items.update(source_cache.get("items", {}))
    cached_items.update(raw_cache.get("items", {}))

    candidates: list[dict[str, Any]] = []
    seen_anidb: set[int] = set()
    for key, item in anime_cache.get("items", {}).items():
        data = (item.get("response") or {}).get("data") or {}
        accepted, _ = anime_acceptance(data)
        if not accepted:
            continue
        anidb_id = parse_anidb_id(data)
        if not anidb_id or anidb_id in seen_anidb:
            continue
        aired_date = parse_mal_aired_date(data)
        if aired_date is None or aired_date < since_date:
            continue
        cached = cached_items.get(str(anidb_id)) or {}
        candidates.append(
            {
                "mal_id": int(key),
                "anidb_id": int(anidb_id),
                "title": data.get("title"),
                "status": data.get("status"),
                "rating": data.get("rating"),
                "score": data.get("score"),
                "popularity": data.get("popularity") if data.get("popularity") is not None else 999999999,
                "members": data.get("members") if data.get("members") is not None else 0,
                "aired_date": aired_date.date().isoformat(),
                "is_hentai": is_hentai_jikan(data),
                "cached_source": cached.get("source"),
                "cached_at": cached.get("cached_at") or cached.get("source_downloaded_at"),
                "cached_episode_count": cached.get("episode_count"),
            }
        )
        seen_anidb.add(int(anidb_id))

    return sorted(
        candidates,
        key=lambda row: (
            0 if row["is_hentai"] else 1,
            int(row["popularity"] or 999999999),
            -int(row["members"] or 0),
            int(row["mal_id"]),
        ),
    )


def anidb_credentials() -> tuple[str, int]:
    client = read_local_secret("ANIDB_CLIENT", "ANIDB_CLIENT_NAME", filename="anidb_client.txt", default="")
    clientver = int(
        read_local_secret("ANIDB_CLIENTVER", "ANIDB_CLIENT_VERSION", filename="anidb_clientver.txt", default="0")
        or 0
    )
    if not client or not clientver:
        raise RuntimeError("missing AniDB client/clientver; set secrets/secret.txt or ANIDB_CLIENT and ANIDB_CLIENTVER")
    return client, clientver


def anidb_http_get(anidb_id: int, delay_seconds: float = ANIDB_REQUEST_DELAY_SECONDS) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; live AniDB calls are unavailable")
    client, clientver = anidb_credentials()
    time.sleep(float(delay_seconds) + random.uniform(0, ANIDB_REQUEST_JITTER_SECONDS))
    url = (
        "http://api.anidb.net:9001/httpapi"
        f"?request=anime&client={client}&clientver={clientver}"
        f"&protover=1&aid={int(anidb_id)}"
    )
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
    if response.status_code != 200:
        raise RuntimeError(f"HTTP {response.status_code}: {response.text[:220]}")

    root = ET.fromstring(response.content)
    if root.tag.lower() == "error":
        message = (root.text or "").strip()
        code = root.get("code")
        raise RuntimeError(f"AniDB error response: {message or 'unknown'}; code={code}")

    payload = extract_anidb_payload(root)
    payload["cached_at"] = now_iso()
    payload["source"] = "live_http"
    payload["source_url"] = url
    payload["source_downloaded_at"] = now_iso()
    return compact_anidb_entry(payload)


def is_permanent_anidb_error(reason: str) -> bool:
    text = reason.casefold()
    return (
        "anime not found" in text
        or "unknown anime id" in text
        or "code=320" in text
        or "code=330" in text
        or "http 404" in text
    )


def is_ban_or_rate_limit(reason: str) -> bool:
    text = reason.casefold()
    return "banned" in text or "rate" in text or "code=500" in text or "code=501" in text


def jikan_get(endpoint: str, max_retries: int = 4, base_delay: float = 1.0) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed; live Jikan calls are unavailable")
    url = f"{JIKAN_API_BASE}{endpoint}"
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
        except requests.RequestException as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            wait = min(120, (2**attempt) * base_delay)
            write_log(f"JIKAN_REQUEST_EXCEPTION | {endpoint} | {last_error} | sleep={wait:.1f}s")
            time.sleep(wait)
            continue
        if response.status_code == 200:
            time.sleep(base_delay)
            return response.json()
        last_error = f"HTTP {response.status_code}: {response.text[:220]}"
        if response.status_code in {429, 500, 502, 503, 504}:
            wait = min(180, (2**attempt) * base_delay + random.uniform(0, base_delay))
            write_log(f"JIKAN_RETRYABLE | {endpoint} | {last_error} | sleep={wait:.1f}s")
            time.sleep(wait)
            continue
        raise RuntimeError(last_error)
    raise RuntimeError(last_error or f"Jikan request failed for {endpoint}")


def anime_acceptance(data: dict[str, Any]) -> tuple[bool, str]:
    anime_type = data.get("type")
    status = data.get("status")
    score = data.get("score")
    if anime_type not in VALID_TYPES:
        return False, f"invalid_type:{anime_type}"
    if status != NOT_YET_AIRED_STATUS and score is None:
        return False, "missing_score_non_not_yet_aired"
    return True, "accepted"


def is_not_found_error(error: Exception | str) -> bool:
    return "HTTP 404" in str(error)


def record_invalid_id(
    payload: dict[str, Any],
    mal_id: int,
    index: int | None,
    reason: str,
    data: dict[str, Any] | None = None,
) -> None:
    data = data or {}
    invalid_kind = "invalid_mal_id" if reason.startswith("invalid_mal_id") else "filtered_candidate"
    payload.setdefault("items", {})[str(int(mal_id))] = {
        "mal_id": int(mal_id),
        "index": index,
        "invalid_kind": invalid_kind,
        "type": data.get("type"),
        "status": data.get("status"),
        "score": data.get("score"),
        "reason": reason,
        "recorded_at": now_iso(),
    }
    atomic_write_json(RAW_GATHER_INVALID_FILE, payload)


def record_failure(payload: dict[str, Any], mal_id: int, stage: str, error: str, retryable: bool = True) -> None:
    payload.setdefault("items", {})[f"{mal_id}:{stage}"] = {
        "mal_id": int(mal_id),
        "stage": stage,
        "error": str(error)[:500],
        "retryable": bool(retryable),
        "last_attempt_at": now_iso(),
    }
    atomic_write_json(RAW_GATHER_FAILED_FILE, payload)


def clear_failure(payload: dict[str, Any], mal_id: int, stage: str) -> None:
    payload.setdefault("items", {}).pop(f"{mal_id}:{stage}", None)
    atomic_write_json(RAW_GATHER_FAILED_FILE, payload)


def write_checkpoint(stage: str, **fields: Any) -> None:
    payload = {
        "updated_at": now_iso(),
        "stage": stage,
        **fields,
    }
    atomic_write_json(RAW_GATHER_CHECKPOINT_FILE, payload)


def compact_jikan_characters(response: dict[str, Any]) -> list[dict[str, Any]]:
    compact: list[dict[str, Any]] = []
    for item in response.get("data") or []:
        if not isinstance(item, dict):
            continue
        character = item.get("character") or {}
        if not isinstance(character, dict) or character.get("mal_id") is None:
            continue
        try:
            character_id = int(character.get("mal_id"))
        except (TypeError, ValueError):
            continue
        if character_id <= 0:
            continue

        voice_actors: list[dict[str, Any]] = []
        for actor in item.get("voice_actors") or []:
            if not isinstance(actor, dict):
                continue
            person = actor.get("person") or {}
            if not isinstance(person, dict) or person.get("mal_id") is None:
                continue
            try:
                person_id = int(person.get("mal_id"))
            except (TypeError, ValueError):
                continue
            if person_id <= 0:
                continue
            voice_actors.append(
                {
                    "id": person_id,
                    "name": str(person.get("name") or "").strip(),
                    "language": str(actor.get("language") or "").strip(),
                }
            )

        compact.append(
            {
                "id": character_id,
                "name": str(character.get("name") or "").strip(),
                "role": item.get("role"),
                "voice_actors": voice_actors,
            }
        )
    return compact


def gather_jikan(
    limit: int | None = None,
    retry_failed: bool = False,
    sleep_seconds: float = 1.0,
    anilist_sleep_seconds: float = 2.1,
    only_ids: list[int] | None = None,
    refresh_ids: list[int] | None = None,
    with_anilist: bool = True,
    with_characters: bool = True,
    with_recommendations: bool = True,
) -> dict[str, Any]:
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    rec_cache = load_cache(JIKAN_RECOMMENDATION_CACHE_FILE)
    character_cache = load_cache(JIKAN_CHARACTER_CACHE_FILE)
    anilist_cache = load_anilist_cache()
    if seed_anilist_cache_from_sample(anilist_cache):
        save_anilist_cache(anilist_cache)
    failures = load_json(RAW_GATHER_FAILED_FILE, {"updated_at": None, "items": {}})
    invalid = load_json(RAW_GATHER_INVALID_FILE, {"updated_at": None, "items": {}})

    refresh_set = {int(value) for value in (refresh_ids or [])}
    for mal_id in refresh_set:
        anime_cache.get("items", {}).pop(str(mal_id), None)
        if with_recommendations:
            rec_cache.get("items", {}).pop(str(mal_id), None)
        character_cache.get("items", {}).pop(str(mal_id), None)
        anilist_cache.get("items", {}).pop(str(mal_id), None)

    candidate_ids = sorted(set(int(value) for value in only_ids)) if only_ids else load_candidate_ids()
    if retry_failed:
        retry_ids = [
            int(item["mal_id"])
            for item in failures.get("items", {}).values()
            if item.get("retryable") and item.get("stage") in {"jikan_anime", "jikan_recommendations", "jikan_characters"}
        ]
        candidate_ids = sorted(set(candidate_ids) | set(retry_ids))
    if limit is not None:
        candidate_ids = candidate_ids[: int(limit)]

    accepted = 0
    anime_live = 0
    rec_live = 0
    character_live = 0
    anilist_live = 0

    write_checkpoint(
        "jikan_gather_started",
        total_candidates=len(candidate_ids),
        accepted_seen=accepted,
        jikan_anime_live_requests=anime_live,
        jikan_character_live_requests=character_live,
        anilist_live_requests=anilist_live,
        jikan_recommendation_live_requests=rec_live,
    )

    for index, mal_id in enumerate(candidate_ids, start=1):
        key = str(int(mal_id))
        if key not in anime_cache["items"]:
            try:
                response = jikan_get(f"/anime/{int(mal_id)}/full", base_delay=sleep_seconds)
            except Exception as exc:
                retryable = not is_not_found_error(exc)
                if retryable:
                    record_failure(failures, int(mal_id), "jikan_anime", str(exc), retryable=True)
                else:
                    record_invalid_id(invalid, int(mal_id), index, "invalid_mal_id:jikan_404")
                    clear_failure(failures, int(mal_id), "jikan_anime")
                write_log(f"JIKAN_ANIME_FAILED | {index}/{len(candidate_ids)} | MAL {mal_id} | {exc}")
                continue
            anime_cache["items"][key] = {
                "fetched_at": now_iso(),
                "source": "jikan:/anime/{id}/full",
                "response": compact_jikan_anime_response(response),
            }
            save_cache(JIKAN_ANIME_CACHE_FILE, anime_cache)
            clear_failure(failures, int(mal_id), "jikan_anime")
            anime_live += 1
            write_log(f"JIKAN_ANIME_READY | {index}/{len(candidate_ids)} | MAL {mal_id} | live")

        data = (anime_cache["items"][key].get("response") or {}).get("data") or {}
        accepted_row, reason = anime_acceptance(data)
        if not accepted_row:
            record_invalid_id(invalid, int(mal_id), index, reason, data)
            write_log(f"JIKAN_SKIPPED | {index}/{len(candidate_ids)} | MAL {mal_id} | {reason}")
            write_checkpoint(
                "jikan_gather_running",
                index=index,
                total_candidates=len(candidate_ids),
                last_mal_id=int(mal_id),
                last_status="skipped",
                last_reason=reason,
                accepted_seen=accepted,
                jikan_anime_live_requests=anime_live,
                jikan_character_live_requests=character_live,
                anilist_live_requests=anilist_live,
                jikan_recommendation_live_requests=rec_live,
                failure_count=len(failures.get("items", {})),
            )
            continue
        invalid.setdefault("items", {}).pop(key, None)
        accepted += 1

        if with_anilist and key not in anilist_cache.get("items", {}):
            try:
                media, source = get_anilist_media(
                    int(mal_id),
                    anilist_cache,
                    live=True,
                    sleep_seconds=anilist_sleep_seconds,
                )
            except Exception as exc:
                record_failure(failures, int(mal_id), "anilist_media", str(exc), retryable=True)
                write_log(f"ANILIST_FAILED | {index}/{len(candidate_ids)} | MAL {mal_id} | {exc}")
            else:
                if source == "live":
                    anilist_live += 1
                clear_failure(failures, int(mal_id), "anilist_media")
                write_log(
                    f"ANILIST_READY | {index}/{len(candidate_ids)} | MAL {mal_id} | "
                    f"{source} | anilist_id={(media or {}).get('id')}"
                )

        if with_characters and key not in character_cache["items"]:
            try:
                response = jikan_get(f"/anime/{int(mal_id)}/characters", base_delay=sleep_seconds)
            except Exception as exc:
                retryable = not is_not_found_error(exc)
                if retryable:
                    record_failure(failures, int(mal_id), "jikan_characters", str(exc), retryable=True)
                write_log(f"JIKAN_CHARACTERS_FAILED | {index}/{len(candidate_ids)} | MAL {mal_id} | {exc}")
            else:
                character_cache["items"][key] = {
                    "fetched_at": now_iso(),
                    "source": "jikan:/anime/{id}/characters",
                    "characters": compact_jikan_characters(response),
                }
                save_cache(JIKAN_CHARACTER_CACHE_FILE, character_cache)
                clear_failure(failures, int(mal_id), "jikan_characters")
                character_live += 1
                write_log(
                    f"JIKAN_CHARACTERS_READY | {index}/{len(candidate_ids)} | MAL {mal_id} | "
                    f"live | characters={len(character_cache['items'][key]['characters'])}"
                )

        if with_recommendations and data.get("status") != NOT_YET_AIRED_STATUS and key not in rec_cache["items"]:
            try:
                response = jikan_get(f"/anime/{int(mal_id)}/recommendations", base_delay=sleep_seconds)
            except Exception as exc:
                retryable = "HTTP 404" not in str(exc)
                record_failure(failures, int(mal_id), "jikan_recommendations", str(exc), retryable=retryable)
                write_log(f"JIKAN_RECS_FAILED | {index}/{len(candidate_ids)} | MAL {mal_id} | {exc}")
            else:
                rec_cache["items"][key] = {
                    "fetched_at": now_iso(),
                    "source": "jikan:/anime/{id}/recommendations",
                    "response": compact_jikan_recommendation_response(response),
                }
                save_cache(JIKAN_RECOMMENDATION_CACHE_FILE, rec_cache)
                clear_failure(failures, int(mal_id), "jikan_recommendations")
                rec_live += 1
                write_log(f"JIKAN_RECS_READY | {index}/{len(candidate_ids)} | MAL {mal_id} | live")

        if index % 25 == 0:
            write_checkpoint(
                "jikan_gather_running",
                index=index,
                total_candidates=len(candidate_ids),
                last_mal_id=int(mal_id),
                last_status="accepted",
                accepted_seen=accepted,
                jikan_anime_live_requests=anime_live,
                jikan_character_live_requests=character_live,
                anilist_live_requests=anilist_live,
                jikan_recommendation_live_requests=rec_live,
                jikan_anime_cache_entries=len(anime_cache["items"]),
                jikan_character_cache_entries=len(character_cache["items"]),
                anilist_cache_entries=len(anilist_cache.get("items", {})),
                jikan_recommendation_cache_entries=len(rec_cache["items"]),
                failure_count=len(failures.get("items", {})),
            )
            write_log(
                f"CHECKPOINT | {index}/{len(candidate_ids)} | last_mal_id={mal_id} | "
                f"accepted={accepted} | failures={len(failures.get('items', {}))}"
            )

        if index % 100 == 0:
            mirror_anilist_cache()
            write_log(
                f"JIKAN_PROGRESS | {index}/{len(candidate_ids)} | "
                f"accepted={accepted} | anime_live={anime_live} | "
                f"anilist_live={anilist_live} | rec_live={rec_live}"
            )

    save_cache(JIKAN_ANIME_CACHE_FILE, anime_cache)
    save_cache(JIKAN_RECOMMENDATION_CACHE_FILE, rec_cache)
    save_cache(JIKAN_CHARACTER_CACHE_FILE, character_cache)
    save_anilist_cache(anilist_cache)
    mirror_anilist_cache()
    atomic_write_json(RAW_GATHER_INVALID_FILE, invalid)
    write_checkpoint(
        "jikan_gather_complete",
        total_candidates=len(candidate_ids),
        accepted_seen=accepted,
        jikan_anime_live_requests=anime_live,
        jikan_character_live_requests=character_live,
        anilist_live_requests=anilist_live,
        jikan_recommendation_live_requests=rec_live,
        jikan_anime_cache_entries=len(anime_cache["items"]),
        jikan_character_cache_entries=len(character_cache["items"]),
        anilist_cache_entries=len(anilist_cache.get("items", {})),
        jikan_recommendation_cache_entries=len(rec_cache["items"]),
        failure_count=len(failures.get("items", {})),
    )
    return {
        "candidate_ids": len(candidate_ids),
        "accepted_seen": accepted,
        "jikan_anime_live_requests": anime_live,
        "jikan_character_live_requests": character_live,
        "anilist_live_requests": anilist_live,
        "jikan_recommendation_live_requests": rec_live,
        "jikan_recommendations_enabled": with_recommendations,
        "jikan_anime_cache_entries": len(anime_cache["items"]),
        "jikan_character_cache_entries": len(character_cache["items"]),
        "anilist_cache_entries": len(anilist_cache.get("items", {})),
        "jikan_recommendation_cache_entries": len(rec_cache["items"]),
    }


def accepted_mal_ids_from_jikan() -> list[int]:
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    ids = []
    for key, item in anime_cache.get("items", {}).items():
        data = (item.get("response") or {}).get("data") or {}
        accepted, _ = anime_acceptance(data)
        if accepted:
            ids.append(int(key))
    return sorted(set(ids))


def gather_jikan_characters(
    limit: int | None = None,
    retry_failed: bool = False,
    sleep_seconds: float = 1.0,
    only_ids: list[int] | None = None,
    refresh_existing: bool = False,
) -> dict[str, Any]:
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    character_cache = load_cache(JIKAN_CHARACTER_CACHE_FILE)
    failures = load_json(RAW_GATHER_FAILED_FILE, {"updated_at": None, "items": {}})
    accepted_ids = sorted(set(int(value) for value in only_ids)) if only_ids else accepted_mal_ids_from_jikan()
    retry_ids = {
        int(item["mal_id"])
        for item in failures.get("items", {}).values()
        if item.get("retryable") and item.get("stage") == "jikan_characters"
    }
    if retry_failed:
        accepted_ids = sorted(set(accepted_ids) | retry_ids)
    if refresh_existing:
        pending_ids = accepted_ids
    else:
        pending_ids = [
            mal_id
            for mal_id in accepted_ids
            if str(mal_id) not in character_cache.get("items", {}) or mal_id in retry_ids
        ]
    selected_ids = pending_ids[: int(limit)] if limit is not None else pending_ids

    fetched = 0
    failed = 0
    cached_missing = 0
    write_log(
        "JIKAN_CHARACTERS_BACKFILL_START | "
        f"accepted={len(accepted_ids)} | pending={len(pending_ids)} | "
        f"selected={len(selected_ids)} | refresh_existing={refresh_existing}"
    )

    for position, mal_id in enumerate(selected_ids, start=1):
        key = str(int(mal_id))
        if key not in anime_cache.get("items", {}) and not only_ids:
            continue
        try:
            response = jikan_get(f"/anime/{int(mal_id)}/characters", base_delay=sleep_seconds)
        except Exception as exc:
            retryable = not is_not_found_error(exc)
            if retryable:
                failed += 1
                record_failure(failures, int(mal_id), "jikan_characters", str(exc), retryable=True)
                write_log(f"JIKAN_CHARACTERS_BACKFILL_FAILED | {position}/{len(selected_ids)} | MAL {mal_id} | {exc}")
                continue
            character_cache.setdefault("items", {})[key] = {
                "fetched_at": now_iso(),
                "source": "jikan:/anime/{id}/characters",
                "missing": True,
                "characters": [],
            }
            cached_missing += 1
            clear_failure(failures, int(mal_id), "jikan_characters")
        else:
            character_cache.setdefault("items", {})[key] = {
                "fetched_at": now_iso(),
                "source": "jikan:/anime/{id}/characters",
                "characters": compact_jikan_characters(response),
            }
            fetched += 1
            clear_failure(failures, int(mal_id), "jikan_characters")

        save_cache(JIKAN_CHARACTER_CACHE_FILE, character_cache)
        if position <= 3 or position % 10 == 0:
            write_log(
                f"JIKAN_CHARACTERS_BACKFILL_PROGRESS | {position}/{len(selected_ids)} | "
                f"MAL {mal_id} | fetched={fetched} | failed={failed} | "
                f"cached_missing={cached_missing} | cache_entries={len(character_cache.get('items', {}))}"
            )
        if position % 25 == 0:
            write_checkpoint(
                "jikan_characters_backfill_running",
                index=position,
                total_candidates=len(selected_ids),
                last_mal_id=int(mal_id),
                fetched=fetched,
                failed=failed,
                cached_missing=cached_missing,
                cache_entries=len(character_cache.get("items", {})),
            )

    save_cache(JIKAN_CHARACTER_CACHE_FILE, character_cache)
    atomic_write_json(RAW_GATHER_FAILED_FILE, failures)
    write_checkpoint(
        "jikan_characters_backfill_complete",
        total_candidates=len(selected_ids),
        fetched=fetched,
        failed=failed,
        cached_missing=cached_missing,
        cache_entries=len(character_cache.get("items", {})),
    )
    return {
        "accepted_mal_ids": len(accepted_ids),
        "pending_character_ids": len(pending_ids),
        "selected_character_ids": len(selected_ids),
        "jikan_character_live_requests": fetched,
        "jikan_character_failed": failed,
        "jikan_character_cached_missing": cached_missing,
        "jikan_character_cache_entries": len(character_cache.get("items", {})),
        "jikan_character_cache_file": str(JIKAN_CHARACTER_CACHE_FILE),
    }


def mirror_anilist_cache() -> int:
    cache_path = ANILIST_RAW_CACHE_FILE if ANILIST_RAW_CACHE_FILE.exists() else ANILIST_CACHE_FILE
    return len(load_cache(cache_path).get("items", {})) if cache_path.exists() else 0


def gather_anilist(limit: int | None = None, sleep_seconds: float = 2.1) -> dict[str, Any]:
    mal_ids = accepted_mal_ids_from_jikan()
    cache = load_anilist_cache()
    if seed_anilist_cache_from_sample(cache):
        save_anilist_cache(cache)
    cached_ids = {int(key) for key in cache.get("items", {}) if str(key).isdigit()}
    missing = [mal_id for mal_id in mal_ids if mal_id not in cached_ids]
    updated = update_anilist_cache_for_mal_ids(missing, cache, limit=limit, sleep_seconds=sleep_seconds)
    mirrored_entries = mirror_anilist_cache()
    return {
        "accepted_mal_ids": len(mal_ids),
        "anilist_missing_before_limit": len(missing),
        "anilist_live_updates": updated,
        "anilist_cache_entries": mirrored_entries,
    }


def parse_id_list(values: list[str] | None) -> list[int]:
    ids: list[int] = []
    for value in values or []:
        for part in str(value).split(","):
            text = part.strip()
            if text.isdigit():
                ids.append(int(text))
    return sorted(set(ids))


def mirror_anidb_cache() -> dict[str, Any]:
    cache = load_cache(ANIDB_RAW_CACHE_FILE)
    return {"anidb_cache_entries": len(cache.get("items", {})), "anidb_cache_file": str(ANIDB_RAW_CACHE_FILE)}


def save_anidb_live_caches(cache: dict[str, Any]) -> None:
    save_cache(ANIDB_RAW_CACHE_FILE, cache)


def gather_anidb_recent_live(
    since: str = ANIDB_RECENT_BACKFILL_SINCE,
    limit: int | None = None,
    sleep_seconds: float = ANIDB_REQUEST_DELAY_SECONDS,
    refresh_existing: bool = False,
    stop_on_ban: bool = True,
) -> dict[str, Any]:
    candidates = recent_anidb_live_candidates(since=since)
    cache = load_cache(ANIDB_SOURCE_CACHE_FILE)
    raw_cache = load_cache(ANIDB_RAW_CACHE_FILE)
    cache.setdefault("items", {})
    for key, raw_item in (raw_cache.get("items") or {}).items():
        existing_item = cache["items"].get(key) or {}
        if raw_item.get("source") == "live_http" and existing_item.get("source") != "live_http":
            cache["items"][key] = raw_item
    failures = load_json(RAW_GATHER_FAILED_FILE, {"updated_at": None, "items": {}})
    invalid = load_json(RAW_GATHER_INVALID_FILE, {"updated_at": None, "items": {}})
    if refresh_existing:
        pending_candidates = candidates
    else:
        pending_candidates = [
            row
            for row in candidates
            if (cache.get("items", {}).get(str(row["anidb_id"])) or {}).get("source") != "live_http"
        ]
    selected = pending_candidates[: int(limit)] if limit is not None else pending_candidates
    updated = 0
    skipped_cached = 0
    failed = 0
    invalid_count = 0

    write_log(
        "ANIDB_LIVE_RECENT_START | "
        f"since={since} | candidates={len(candidates)} | pending={len(pending_candidates)} | "
        f"selected={len(selected)} | "
        f"refresh_existing={refresh_existing}"
    )

    for position, row in enumerate(selected, start=1):
        anidb_id = int(row["anidb_id"])
        mal_id = int(row["mal_id"])
        key = str(anidb_id)
        existing = cache.get("items", {}).get(key)
        if existing and not refresh_existing and existing.get("source") == "live_http":
            skipped_cached += 1
            write_log(
                f"ANIDB_LIVE_SKIPPED | {position}/{len(selected)} | "
                f"MAL {mal_id} | AniDB {anidb_id} | already_live"
            )
            continue

        try:
            payload = anidb_http_get(anidb_id, delay_seconds=sleep_seconds)
        except Exception as exc:
            reason = str(exc)
            failed += 1
            if is_permanent_anidb_error(reason):
                invalid_key = f"{mal_id}:anidb:{anidb_id}"
                invalid.setdefault("items", {})[invalid_key] = {
                    "mal_id": mal_id,
                    "anidb_id": anidb_id,
                    "index": position,
                    "reason": f"invalid_anidb_id:{reason[:240]}",
                    "title": row.get("title"),
                    "recorded_at": now_iso(),
                }
                invalid_count += 1
                clear_failure(failures, mal_id, "anidb_live")
            else:
                record_failure(
                    failures,
                    mal_id,
                    "anidb_live",
                    f"AniDB {anidb_id}: {reason[:300]}",
                    retryable=True,
                )
            atomic_write_json(RAW_GATHER_FAILED_FILE, failures)
            atomic_write_json(RAW_GATHER_INVALID_FILE, invalid)
            write_log(
                f"ANIDB_LIVE_FAILED | {position}/{len(selected)} | "
                f"MAL {mal_id} | AniDB {anidb_id} | {reason[:300]}"
            )
            if stop_on_ban and is_ban_or_rate_limit(reason):
                write_log(
                    f"ANIDB_LIVE_STOPPED | ban_or_rate_limit | "
                    f"MAL {mal_id} | AniDB {anidb_id} | updated={updated}"
                )
                break
            continue

        existing = cache.get("items", {}).get(key) or {}
        cache.setdefault("items", {})[key] = compact_anidb_entry({**existing, **payload})
        save_anidb_live_caches(cache)
        clear_failure(failures, mal_id, "anidb_live")
        atomic_write_json(RAW_GATHER_FAILED_FILE, failures)
        updated += 1
        write_log(
            f"ANIDB_LIVE_READY | {position}/{len(selected)} | "
            f"MAL {mal_id} | AniDB {anidb_id} | hentai={row['is_hentai']} | "
            f"popularity={row['popularity']} | episodes={payload.get('episode_count')}"
        )

        if position % 10 == 0:
            write_checkpoint(
                "anidb_live_recent_running",
                index=position,
                total_candidates=len(selected),
                updated=updated,
                failed=failed,
                skipped_cached=skipped_cached,
                invalid_anidb_ids=invalid_count,
                last_mal_id=mal_id,
                last_anidb_id=anidb_id,
            )

    save_anidb_live_caches(cache)
    atomic_write_json(RAW_GATHER_FAILED_FILE, failures)
    atomic_write_json(RAW_GATHER_INVALID_FILE, invalid)
    write_checkpoint(
        "anidb_live_recent_complete",
        total_candidates=len(selected),
        updated=updated,
        failed=failed,
        skipped_cached=skipped_cached,
        invalid_anidb_ids=invalid_count,
    )
    return {
        "since": since,
        "candidate_count": len(candidates),
        "pending_candidate_count": len(pending_candidates),
        "selected_count": len(selected),
        "updated": updated,
        "failed": failed,
        "skipped_cached": skipped_cached,
        "invalid_anidb_ids": invalid_count,
        "anidb_cache_entries": len(cache.get("items", {})),
        "priority": "hentai_first_then_lowest_mal_popularity",
        "anidb_cache_file": str(ANIDB_RAW_CACHE_FILE),
    }


def build_raw_source_index() -> pd.DataFrame:
    anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
    character_cache = load_cache(JIKAN_CHARACTER_CACHE_FILE)
    anilist_cache = load_cache(ANILIST_RAW_CACHE_FILE)
    rows = []
    for key, item in anime_cache.get("items", {}).items():
        data = (item.get("response") or {}).get("data") or {}
        accepted, reason = anime_acceptance(data)
        if not accepted:
            continue
        mal_id = int(key)
        anilist_item = anilist_cache.get("items", {}).get(str(mal_id), {})
        media = anilist_item.get("media") or {}
        character_item = character_cache.get("items", {}).get(str(mal_id), {})
        jikan_characters = character_item.get("characters") or []
        anilist_characters = media.get("characters") or []
        rows.append(
            {
                "mal_id": mal_id,
                "anilist_id": media.get("id"),
                "anidb_id": parse_anidb_id(data),
                "title": data.get("title"),
                "type": data.get("type"),
                "status": data.get("status"),
                "score": data.get("score"),
                "jikan_fetched_at": item.get("fetched_at"),
                "jikan_character_fetched_at": character_item.get("fetched_at"),
                "jikan_character_count": len(jikan_characters),
                "anilist_fetched_at": anilist_item.get("fetched_at"),
                "anilist_character_count": len(anilist_characters),
            }
        )
    df = pd.DataFrame(rows).sort_values(["mal_id"]) if rows else pd.DataFrame()
    RAW_SOURCE_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RAW_SOURCE_INDEX_CSV, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Gather raw MAL/Jikan, AniList, and AniDB source caches.")
    parser.add_argument("--jikan", action="store_true", help="Fetch missing Jikan anime/recommendation payloads.")
    parser.add_argument("--jikan-characters", action="store_true", help="Fetch missing compact Jikan character/Japanese voice-actor payloads for accepted MAL ids.")
    parser.add_argument("--anilist", action="store_true", help="Fetch missing AniList payloads for accepted MAL ids.")
    parser.add_argument("--anidb", action="store_true", help="Report the canonical raw-source AniDB cache status.")
    parser.add_argument("--anidb-live-recent", action="store_true", help="Live-fetch AniDB metadata for recent MAL entries with AniDB ids, prioritized by hentai then popularity.")
    parser.add_argument("--all", action="store_true", help="Run all available raw-source stages.")
    parser.add_argument("--limit", type=int, default=None, help="Limit live calls for testing.")
    parser.add_argument("--ids", nargs="*", default=None, help="Only gather these MAL ids. Accepts spaces or comma-separated chunks.")
    parser.add_argument("--refresh-ids", nargs="*", default=None, help="Clear cached Jikan/AniList entries for these MAL ids before gathering.")
    parser.add_argument("--seasonal-refresh", action="store_true", help="Auto-refresh current-season and recent-season titles whose raw cache is stale.")
    parser.add_argument("--seasonal-refresh-date", default=None, help="Reference date for seasonal refresh, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--seasonal-refresh-max-age-hours", type=float, default=12.0, help="Refresh seasonal rows when Jikan or AniList cache is older than this many hours.")
    parser.add_argument("--include-previous-season", action="store_true", help="Also discover/refresh the previous two seasons; useful when Summer just started.")
    parser.add_argument("--skip-seasonal-discovery", action="store_true", help="Do not call Jikan /seasons endpoints before selecting seasonal refresh ids.")
    parser.add_argument("--skip-mal-season-scrape", action="store_true", help="Do not scrape MAL season HTML pages as a seasonal discovery fallback.")
    parser.add_argument("--seasonal-discovery-page-limit", type=int, default=None, help="Optional page limit for Jikan /seasons discovery calls.")
    parser.add_argument("--refresh-missing-anidb-id", action="store_true", help="Clear cached Jikan anime rows with no direct anidb_id so a Jikan rerun can recover the external AniDB link.")
    parser.add_argument("--skip-mal-id-download", action="store_true", help="Use existing data/raw/mal_candidate_ids.json instead of redownloading it.")
    parser.add_argument("--skip-anilist-with-jikan", action="store_true", help="Do not fetch AniList immediately after an accepted Jikan row.")
    parser.add_argument("--skip-characters", action="store_true", help="Do not fetch Jikan character/voice-actor payloads during the Jikan pass.")
    parser.add_argument("--skip-recommendations", action="store_true", help="Do not fetch Jikan /anime/{id}/recommendations during the Jikan pass.")
    parser.add_argument("--refresh-seasonal-recommendations", action="store_true", help="During seasonal refresh, also clear/refetch Jikan recommendation caches. Slow and Jikan 504-prone.")
    parser.add_argument("--refresh-characters", action="store_true", help="Refresh Jikan character/voice-actor payloads even when cached.")
    parser.add_argument("--refresh-shoko", action="store_true", help="Download Shoko Anime_HTTP.zip and rebuild the AniDB cache before mirroring it.")
    parser.add_argument("--retry-failed", action="store_true", help="Requeue retryable Jikan failures.")
    parser.add_argument("--jikan-sleep", type=float, default=1.0)
    parser.add_argument("--anilist-sleep", type=float, default=2.2)
    parser.add_argument("--anidb-sleep", type=float, default=ANIDB_REQUEST_DELAY_SECONDS)
    parser.add_argument("--anidb-since", default=ANIDB_RECENT_BACKFILL_SINCE, help="Only live-fetch AniDB ids whose MAL air date is on/after this YYYY-MM-DD date.")
    parser.add_argument("--anidb-refresh-existing", action="store_true", help="Refresh even AniDB ids already fetched from the live HTTP API.")
    parser.add_argument("--anidb-continue-after-ban", action="store_true", help="Do not stop the live AniDB stage after a ban/rate-limit error.")
    args = parser.parse_args()

    ensure_dirs()
    only_ids = parse_id_list(args.ids)
    refresh_ids = parse_id_list(args.refresh_ids)
    seasonal_refresh_ids: list[int] = []
    discovered_refresh_ids: list[int] = []
    if args.seasonal_refresh:
        reference_date = parse_date_arg(args.seasonal_refresh_date) if args.seasonal_refresh_date else datetime.now()
        discovered_rows: list[dict[str, Any]] = []
        mal_discovered_rows: list[dict[str, Any]] = []
        if not args.skip_seasonal_discovery:
            discovered_rows = discover_jikan_season_ids(
                reference_date,
                include_previous=args.include_previous_season,
                include_current=True,
                include_next=False,
                page_limit=args.seasonal_discovery_page_limit,
            )
            if not args.skip_mal_season_scrape:
                mal_discovered_rows = discover_mal_season_page_ids(
                    reference_date,
                    include_previous=args.include_previous_season,
                )
            combined_discovered_rows = discovered_rows + mal_discovered_rows
            discovered_refresh_ids = stale_or_missing_jikan_ids(
                combined_discovered_rows,
                reference_date,
                max_age_hours=float(args.seasonal_refresh_max_age_hours),
            )
            if only_ids:
                only_id_set = set(only_ids)
                discovered_rows = [row for row in discovered_rows if int(row["mal_id"]) in only_id_set]
                mal_discovered_rows = [row for row in mal_discovered_rows if int(row["mal_id"]) in only_id_set]
                discovered_refresh_ids = [mal_id for mal_id in discovered_refresh_ids if mal_id in only_id_set]
            pd.DataFrame(discovered_rows).to_csv(SEASONAL_DISCOVERY_CSV, index=False)
            pd.DataFrame(mal_discovered_rows).to_csv(SEASONAL_MAL_DISCOVERY_CSV, index=False)
            write_log(
                "SEASONAL_DISCOVERY_COMPLETE | "
                f"reference_date={reference_date.date().isoformat()} | "
                f"discovered={len(discovered_rows)} | "
                f"mal_html_discovered={len(mal_discovered_rows)} | "
                f"accepted_stale_or_missing={len(discovered_refresh_ids)} | "
                f"csv={SEASONAL_DISCOVERY_CSV} | "
                f"mal_csv={SEASONAL_MAL_DISCOVERY_CSV}"
            )
        candidates = seasonal_refresh_candidates(
            reference_date,
            max_age_hours=float(args.seasonal_refresh_max_age_hours),
            include_previous=args.include_previous_season,
        )
        if only_ids:
            only_id_set = set(only_ids)
            candidates = [row for row in candidates if int(row["mal_id"]) in only_id_set]
        if args.limit is not None:
            candidates = candidates[: int(args.limit)]
        seasonal_refresh_ids = sorted(set(int(row["mal_id"]) for row in candidates) | set(discovered_refresh_ids))
        pd.DataFrame(candidates).to_csv(SEASONAL_REFRESH_CANDIDATES_CSV, index=False)
        write_log(
            "SEASONAL_REFRESH_CANDIDATES | "
            f"reference_date={reference_date.date().isoformat()} | "
            f"max_age_hours={args.seasonal_refresh_max_age_hours} | "
            f"include_previous={args.include_previous_season} | "
            f"cache_selected={len(candidates)} | "
            f"discovery_selected={len(discovered_refresh_ids)} | "
            f"selected={len(seasonal_refresh_ids)} | "
            f"csv={SEASONAL_REFRESH_CANDIDATES_CSV}"
        )
        if seasonal_refresh_ids:
            anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
            rec_cache = load_cache(JIKAN_RECOMMENDATION_CACHE_FILE)
            anilist_cache = load_anilist_cache()
            for mal_id in seasonal_refresh_ids:
                key = str(int(mal_id))
                anime_cache.get("items", {}).pop(key, None)
                if args.refresh_seasonal_recommendations:
                    rec_cache.get("items", {}).pop(key, None)
                anilist_cache.get("items", {}).pop(key, None)
            save_cache(JIKAN_ANIME_CACHE_FILE, anime_cache)
            if args.refresh_seasonal_recommendations:
                save_cache(JIKAN_RECOMMENDATION_CACHE_FILE, rec_cache)
            save_anilist_cache(anilist_cache)
            mirror_anilist_cache()
            write_log(
                "SEASONAL_REFRESH_CACHES_CLEARED | "
                f"ids={seasonal_refresh_ids[:50]} | total={len(seasonal_refresh_ids)} | "
                f"recommendations_cleared={args.refresh_seasonal_recommendations}"
            )
        only_ids = seasonal_refresh_ids

    if refresh_ids:
        anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
        rec_cache = load_cache(JIKAN_RECOMMENDATION_CACHE_FILE)
        character_cache = load_cache(JIKAN_CHARACTER_CACHE_FILE)
        anilist_cache = load_anilist_cache()
        for mal_id in refresh_ids:
            anime_cache.get("items", {}).pop(str(mal_id), None)
            rec_cache.get("items", {}).pop(str(mal_id), None)
            character_cache.get("items", {}).pop(str(mal_id), None)
            anilist_cache.get("items", {}).pop(str(mal_id), None)
        save_cache(JIKAN_ANIME_CACHE_FILE, anime_cache)
        save_cache(JIKAN_RECOMMENDATION_CACHE_FILE, rec_cache)
        save_cache(JIKAN_CHARACTER_CACHE_FILE, character_cache)
        save_anilist_cache(anilist_cache)
        mirror_anilist_cache()
        write_log(f"REFRESH_IDS_CLEARED | {refresh_ids}")

    if args.refresh_missing_anidb_id:
        anime_cache = load_cache(JIKAN_ANIME_CACHE_FILE)
        candidate_refresh_ids = only_ids or [
            int(key)
            for key, item in anime_cache.get("items", {}).items()
            if str(key).isdigit() and parse_anidb_id(((item.get("response") or {}).get("data") or {})) is None
        ]
        if args.limit is not None:
            candidate_refresh_ids = candidate_refresh_ids[: int(args.limit)]
        cleared = 0
        for mal_id in candidate_refresh_ids:
            key = str(int(mal_id))
            item = anime_cache.get("items", {}).get(key) or {}
            data = (item.get("response") or {}).get("data") or {}
            if parse_anidb_id(data) is None:
                anime_cache.get("items", {}).pop(key, None)
                cleared += 1
        save_cache(JIKAN_ANIME_CACHE_FILE, anime_cache)
        write_log(
            "REFRESH_MISSING_ANIDB_ID_CLEARED | "
            f"cleared={cleared} | scope={'ids' if only_ids else 'all_cached'} | "
            "rerun with --jikan to fetch fresh external links"
        )

    summary: dict[str, Any] = {"updated_at": now_iso()}
    if args.seasonal_refresh:
        summary["seasonal_refresh"] = {
            "selected_count": len(seasonal_refresh_ids),
            "discovered_refresh_count": len(discovered_refresh_ids),
            "candidate_csv": str(SEASONAL_REFRESH_CANDIDATES_CSV),
            "discovery_csv": str(SEASONAL_DISCOVERY_CSV),
            "mal_html_discovery_csv": str(SEASONAL_MAL_DISCOVERY_CSV),
            "max_age_hours": args.seasonal_refresh_max_age_hours,
            "include_previous_season": args.include_previous_season,
            "reference_date": args.seasonal_refresh_date or datetime.now().date().isoformat(),
        }
    if args.refresh_shoko:
        summary["shoko"] = refresh_shoko_anidb_cache()
    if (args.all or args.jikan) and not only_ids:
        download_mal_candidate_ids(force=not args.skip_mal_id_download)
    if args.all or args.jikan or args.seasonal_refresh:
        summary["jikan"] = gather_jikan(
            limit=args.limit,
            retry_failed=args.retry_failed,
            sleep_seconds=args.jikan_sleep,
            anilist_sleep_seconds=args.anilist_sleep,
            only_ids=only_ids or None,
            refresh_ids=refresh_ids or None,
            with_anilist=not args.skip_anilist_with_jikan,
            with_characters=(not args.skip_characters and (not args.seasonal_refresh or args.refresh_characters)),
            with_recommendations=not args.skip_recommendations,
        )
    if args.jikan_characters:
        summary["jikan_characters"] = gather_jikan_characters(
            limit=args.limit,
            retry_failed=args.retry_failed,
            sleep_seconds=args.jikan_sleep,
            only_ids=only_ids or None,
            refresh_existing=args.refresh_characters,
        )
    if args.all or args.anilist:
        if only_ids:
            cache = load_anilist_cache()
            cached_ids = {int(key) for key in cache.get("items", {}) if str(key).isdigit()}
            missing = [mal_id for mal_id in only_ids if mal_id not in cached_ids]
            updated = update_anilist_cache_for_mal_ids(missing, cache, limit=args.limit, sleep_seconds=args.anilist_sleep)
            summary["anilist"] = {
                "accepted_mal_ids": len(only_ids),
                "anilist_missing_before_limit": len(missing),
                "anilist_live_updates": updated,
                "anilist_cache_entries": mirror_anilist_cache(),
            }
        else:
            summary["anilist"] = gather_anilist(limit=args.limit, sleep_seconds=args.anilist_sleep)
    if args.anidb_live_recent:
        summary["anidb_live_recent"] = gather_anidb_recent_live(
            since=args.anidb_since,
            limit=args.limit,
            sleep_seconds=args.anidb_sleep,
            refresh_existing=args.anidb_refresh_existing,
            stop_on_ban=not args.anidb_continue_after_ban,
        )
    if args.all or args.anidb:
        summary["anidb"] = mirror_anidb_cache()

    index_df = build_raw_source_index()
    summary["raw_source_index_rows"] = int(len(index_df))
    summary["raw_source_index_csv"] = str(RAW_SOURCE_INDEX_CSV)
    summary["raw_cache_files"] = {
        "jikan_anime": str(JIKAN_ANIME_CACHE_FILE),
        "jikan_characters": str(JIKAN_CHARACTER_CACHE_FILE),
        "jikan_recommendations": str(JIKAN_RECOMMENDATION_CACHE_FILE),
        "anilist": str(ANILIST_RAW_CACHE_FILE),
        "anidb": str(ANIDB_RAW_CACHE_FILE),
    }
    atomic_write_json(RAW_GATHER_SUMMARY_FILE, summary)
    write_log(f"RAW_GATHER_COMPLETE | {json.dumps(summary, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
