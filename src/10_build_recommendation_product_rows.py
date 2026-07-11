from __future__ import annotations

import json
import math
import argparse
import os
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import pandas as pd
import requests
from pandas.errors import EmptyDataError


ROOT = Path(__file__).resolve().parents[1]
CATALOG_CSV = ROOT / "data" / "processed" / "anime_dataset.csv"
MYLIST_XML = ROOT / "data" / "raw" / "MyList.xml"
SECRETS_DIR = ROOT / "secrets"
VA_EDGE_CSV = ROOT / "data" / "processed" / "anime_voice_actor_edges.csv"
STAFF_EDGE_CSV = ROOT / "data" / "processed" / "anime_staff_edges.csv"
OUT_DIR = ROOT / "artifacts" / "recommendation"
ROWS_CSV = OUT_DIR / "product_recommendation_rows.csv"
SUMMARY_JSON = OUT_DIR / "product_recommendation_summary.json"
MYLIST_GUARDED_RECOMMENDATIONS_CSV = OUT_DIR / "mylist_guarded_recommendation_example.csv"
MYLIST_RECOMMENDATIONS_CSV = OUT_DIR / "mylist_recommendation_example.csv"
EVALUATION_METRICS_CSV = OUT_DIR / "evaluation_metrics.csv"
ADVANCED_METRICS_CSV = OUT_DIR / "advanced_ranker_metrics.csv"
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
JIKAN_API_BASE = "https://api.jikan.moe/v4"
REQUEST_HEADERS = {
    "User-Agent": "anime-recommender-product-demo/1.0",
    "Accept": "application/json",
}


def project_path(value: object) -> object:
    if not isinstance(value, (str, Path)):
        return value
    try:
        path = Path(str(value))
        if path.is_absolute():
            return str(path.relative_to(ROOT))
    except (ValueError, OSError):
        pass
    text = str(value)
    root_text = str(ROOT)
    if text.startswith(root_text):
        return text[len(root_text) :].lstrip("\\/")
    return text


def relativize_payload(value: object) -> object:
    if isinstance(value, dict):
        return {key: relativize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_payload(item) for item in value]
    return project_path(value)

MAX_ROW_ITEMS = 12
POSITIVE_SCORE = 7
STRONG_SCORE = 8
CURRENT_YEAR = 2026
CURRENT_SEASON = "Summer"
LOW_VALUE_TITLE_TERMS = ("recap", "recaps", "manner movie", "digest")
SUMMARY_TITLE_TERMS = ("summary", "soushuuhen", "digest", "recap", "beginning")

ALLOWED_GENERAL_TYPES = {"TV", "Movie", "OVA", "ONA", "Special"}
STRONG_CONTINUE_RELATIONS = {
    "Sequel",
    "Prequel",
    "Parent Story",
    "Full Story",
    "Side Story",
    "Alternative Version",
}
WEAK_CONTINUE_RELATIONS = {"Alternative Setting", "Other"}
CONTINUE_RELATIONS = STRONG_CONTINUE_RELATIONS | WEAK_CONTINUE_RELATIONS
PREREQUISITE_RELATIONS = {"Sequel", "Side Story", "Alternative Version"}
BLOCKING_PREREQUISITE_RELATIONS = {"Prequel", "Parent Story", "Full Story"}
GENERAL_CURRENT_ITEMS = 3
GENERAL_POPULAR_ITEMS = 3

PERSON_NAME_ALIASES = {
    "oohara, sayaka": "Ohara, Sayaka",
}
TITLE_STOPWORDS = {
    "the",
    "a",
    "an",
    "no",
    "ni",
    "to",
    "wo",
    "ga",
    "de",
    "season",
    "movie",
    "ova",
    "ona",
    "special",
    "part",
    "hen",
    "tv",
}
SEASON_ORDER = ["Winter", "Spring", "Summer", "Fall"]

LEVEL_GENERAL_WEIGHTS = {
    "Beginner": {
        "affinity": 0.28,
        "score": 0.26,
        "members": 0.26,
        "recent": 0.05,
        "runtime": 0.15,
        "novelty": 0.00,
    },
    "Casual": {
        "affinity": 0.38,
        "score": 0.22,
        "members": 0.18,
        "recent": 0.08,
        "runtime": 0.09,
        "novelty": 0.05,
    },
    "Fan": {
        "affinity": 0.40,
        "score": 0.17,
        "members": 0.13,
        "recent": 0.14,
        "runtime": 0.04,
        "novelty": 0.12,
    },
    "Veteran": {
        "affinity": 0.32,
        "score": 0.13,
        "members": 0.05,
        "recent": 0.13,
        "runtime": 0.02,
        "novelty": 0.35,
    },
}


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    text = str(value).strip()
    return text == "" or text.lower() in {"nan", "none", "null"}


def split_pipe(value: Any) -> list[str]:
    if is_missing(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def clean_token_text(value: Any) -> list[str]:
    text = re.sub(r"[^a-z0-9]+", " ", str(value or "").lower())
    return [token for token in text.split() if token and token not in TITLE_STOPWORDS and len(token) > 1]


def same_franchise_hint(left: Any, right: Any) -> bool:
    left_tokens = set(clean_token_text(left))
    right_tokens = set(clean_token_text(right))
    if not left_tokens or not right_tokens:
        return False
    shared = left_tokens & right_tokens
    if len(shared) >= 2:
        return True
    # Distinctive first tokens are useful for entries like K, Naruto, Gundam,
    # or Fate, but one-letter tokens are ignored to avoid false positives.
    left_first = next(iter(clean_token_text(left)), "")
    right_first = next(iter(clean_token_text(right)), "")
    return bool(left_first and left_first == right_first and len(left_first) > 2)


def canonical_person_name(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    key = text.lower()
    return PERSON_NAME_ALIASES.get(key, text)


def canonical_studio_name(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    return text.lower().replace(".", "").replace("studio ", "").replace(" studios", "")


def season_window(current_year: int = CURRENT_YEAR, current_season: str = CURRENT_SEASON, back: int = 2) -> set[tuple[int, str]]:
    if current_season not in SEASON_ORDER:
        return {(current_year, current_season)}
    year = current_year
    index = SEASON_ORDER.index(current_season)
    window: set[tuple[int, str]] = set()
    for _ in range(back + 1):
        window.add((year, SEASON_ORDER[index]))
        index -= 1
        if index < 0:
            index = len(SEASON_ORDER) - 1
            year -= 1
    return window


def is_current_window(row: pd.Series) -> bool:
    season = str(row.get("season") or "").strip().title()
    try:
        year = int(float(row.get("aired_year")))
    except (TypeError, ValueError):
        return False
    return (year, season) in season_window()


def parse_weighted_edges(value: Any) -> list[tuple[int, float]]:
    edges: list[tuple[int, float]] = []
    for part in split_pipe(value):
        if ":" not in part:
            continue
        left, right = part.split(":", 1)
        try:
            edges.append((int(float(left)), float(right)))
        except ValueError:
            continue
    return edges


def parse_relation_edges(value: Any) -> list[tuple[int, str]]:
    edges: list[tuple[int, str]] = []
    for part in split_pipe(value):
        if ":" not in part:
            continue
        left, right = part.split(":", 1)
        try:
            edges.append((int(float(left)), right.strip()))
        except ValueError:
            try:
                edges.append((int(float(right)), left.strip()))
            except ValueError:
                continue
    return edges


def load_catalog() -> pd.DataFrame:
    df = pd.read_csv(CATALOG_CSV)
    df["mal_id"] = df["mal_id"].astype(int)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df["members"] = pd.to_numeric(df["members"], errors="coerce").fillna(0)
    df["popularity"] = pd.to_numeric(df["popularity"], errors="coerce")
    df["aired_year"] = pd.to_numeric(df["aired_year"], errors="coerce")
    df["total_watch_minutes"] = pd.to_numeric(df["total_watch_minutes"], errors="coerce")
    return df


def parse_mylist(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["mal_id", "title", "score", "status"])
    root = ET.parse(path).getroot()
    rows = []
    for anime in root.findall("anime"):
        def text(name: str) -> str:
            node = anime.find(name)
            return "" if node is None or node.text is None else node.text.strip()

        try:
            mal_id = int(text("series_animedb_id"))
        except ValueError:
            continue
        try:
            score = int(text("my_score") or 0)
        except ValueError:
            score = 0
        rows.append(
            {
                "mal_id": mal_id,
                "title": text("series_title"),
                "score": score,
                "status": text("my_status"),
            }
        )
    return pd.DataFrame(rows)


def parse_secret_file(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    parsed: dict[str, str] = {}
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    bare_index = 0
    for line in lines:
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        if "=" in text:
            key, value = text.split("=", 1)
            parsed[key.strip().upper().replace(" ", "_")] = value.strip().strip('"').strip("'")
        else:
            parsed[f"BARE_{bare_index}"] = text.strip().strip('"').strip("'")
            bare_index += 1
    return parsed


def read_secret(*names: str, filename: str | None = None, bare_index: int | None = None) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip()
    if filename:
        path = SECRETS_DIR / filename
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    payload = parse_secret_file(SECRETS_DIR / "secret.txt")
    for name in names:
        value = payload.get(name.upper().replace(" ", "_"))
        if value:
            return value
    if bare_index is not None:
        return payload.get(f"BARE_{bare_index}")
    return None


def normalize_list_status(value: Any) -> str:
    text = str(value or "").strip().lower().replace("_", " ").replace("-", " ")
    return {
        "completed": "completed",
        "watching": "watching",
        "on hold": "on_hold",
        "dropped": "dropped",
        "plan to watch": "plan_to_watch",
        "planning": "plan_to_watch",
        "current": "watching",
        "repeating": "watching",
        "paused": "on_hold",
    }.get(text, text or "unknown")


def request_json(url: str, *, headers: dict[str, str] | None = None, params: dict[str, Any] | None = None, timeout: int = 45) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(url, headers=headers or REQUEST_HEADERS, params=params, timeout=timeout)
            if response.status_code in {429, 500, 502, 503, 504}:
                retry_after = response.headers.get("Retry-After")
                wait = float(retry_after) if retry_after and retry_after.isdigit() else 2.0 * attempt
                time.sleep(wait)
                continue
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"Request failed for {url}: {last_error}")


def request_anilist(query: str, variables: dict[str, Any], *, timeout: int = 45) -> dict[str, Any]:
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.post(
                ANILIST_GRAPHQL_URL,
                json={"query": query, "variables": variables},
                headers=REQUEST_HEADERS,
                timeout=timeout,
            )
            if response.status_code == 429:
                wait = float(response.headers.get("Retry-After") or 3 * attempt)
                time.sleep(wait)
                continue
            response.raise_for_status()
            payload = response.json()
            if payload.get("errors"):
                raise RuntimeError(payload["errors"])
            return payload.get("data") or {}
        except (requests.RequestException, RuntimeError) as exc:
            last_error = exc
            time.sleep(2.0 * attempt)
    raise RuntimeError(f"AniList request failed: {last_error}")


def fetch_anilist_user_list(username: str, catalog_ids: set[int]) -> pd.DataFrame:
    query = """
    query UserAnimeList($userName: String, $chunk: Int) {
      MediaListCollection(userName: $userName, type: ANIME, chunk: $chunk, perChunk: 500, forceSingleCompletedList: false) {
        lists {
          name
          status
          entries {
            score(format: POINT_10)
            status
            media { idMal title { romaji english } }
          }
        }
      }
    }
    """
    rows: list[dict[str, Any]] = []
    for chunk in range(1, 40):
        data = request_anilist(query, {"userName": username, "chunk": chunk})
        collection = data.get("MediaListCollection") or {}
        lists = collection.get("lists") or []
        entry_count = 0
        for list_payload in lists:
            for entry in list_payload.get("entries") or []:
                entry_count += 1
                media = entry.get("media") or {}
                mal_id = media.get("idMal")
                if mal_id is None or (catalog_ids and int(mal_id) not in catalog_ids):
                    continue
                title_payload = media.get("title") or {}
                rows.append(
                    {
                        "mal_id": int(mal_id),
                        "title": title_payload.get("romaji") or title_payload.get("english") or "",
                        "score": int(float(entry.get("score") or 0)),
                        "status": normalize_list_status(entry.get("status") or list_payload.get("status")),
                    }
                )
        if entry_count < 500:
            break
        time.sleep(0.8)
    return pd.DataFrame(rows).drop_duplicates(subset=["mal_id"], keep="last")


def fetch_jikan_favorites(username: str, catalog_ids: set[int]) -> dict[str, set[Any]]:
    encoded = quote(username, safe="")
    payload = request_json(f"{JIKAN_API_BASE}/users/{encoded}/favorites", headers=REQUEST_HEADERS)
    data = ((payload or {}).get("data") or {})
    anime = data.get("anime") or []
    people = data.get("people") or []
    favorite_anime_ids = {
        int(item["mal_id"])
        for item in anime
        if item.get("mal_id") is not None and (not catalog_ids or int(item["mal_id"]) in catalog_ids)
    }
    favorite_mal_people = {int(item["mal_id"]) for item in people if item.get("mal_id") is not None}
    favorite_people_names = {canonical_person_name(item.get("name")) for item in people if item.get("name")}
    favorite_va_ids: set[int] = set()
    if favorite_mal_people and VA_EDGE_CSV.exists():
        va_ids = set(pd.read_csv(VA_EDGE_CSV, usecols=["voice_actor_id_mal"])["voice_actor_id_mal"].dropna().astype(int))
        favorite_va_ids = favorite_mal_people & va_ids
    return {
        "favorite_anime_ids": favorite_anime_ids,
        "favorite_voice_actor_ids": favorite_va_ids,
        "favorite_staff_ids": set(),
        "favorite_people_names": favorite_people_names,
        "favorite_studio_names": set(),
        "favorite_studio_ids": set(),
    }


def fetch_anilist_favorites(username: str, catalog_ids: set[int]) -> dict[str, set[Any]]:
    query = """
    query UserFavorites($name: String) {
      User(name: $name) {
        favourites {
          anime(page: 1, perPage: 50) { nodes { idMal } }
          characters(page: 1, perPage: 50) { nodes { id name { full userPreferred } } }
          staff(page: 1, perPage: 50) { nodes { id name { full userPreferred } } }
          studios(page: 1, perPage: 50) { nodes { id name } }
        }
      }
    }
    """
    data = request_anilist(query, {"name": username})
    fav = (((data.get("User") or {}).get("favourites") or {}))
    anime_nodes = ((fav.get("anime") or {}).get("nodes") or [])
    staff_nodes = ((fav.get("staff") or {}).get("nodes") or [])
    studio_nodes = ((fav.get("studios") or {}).get("nodes") or [])
    favorite_anime_ids = {
        int(node["idMal"])
        for node in anime_nodes
        if node.get("idMal") is not None and (not catalog_ids or int(node["idMal"]) in catalog_ids)
    }
    favorite_staff_ids = {int(node["id"]) for node in staff_nodes if node.get("id") is not None}
    favorite_people_names = {
        canonical_person_name((node.get("name") or {}).get("full") or (node.get("name") or {}).get("userPreferred"))
        for node in staff_nodes
        if node.get("name")
    }
    favorite_studio_names = {str(node.get("name")).strip() for node in studio_nodes if node.get("name")}
    favorite_studio_ids = {int(node["id"]) for node in studio_nodes if node.get("id") is not None}
    return {
        "favorite_anime_ids": favorite_anime_ids,
        "favorite_voice_actor_ids": set(),
        "favorite_staff_ids": favorite_staff_ids,
        "favorite_people_names": favorite_people_names,
        "favorite_studio_names": favorite_studio_names,
        "favorite_studio_ids": favorite_studio_ids,
    }


def load_user_list_for_demo(source: str, username: str | None, catalog_ids: set[int]) -> tuple[pd.DataFrame, dict[str, Any]]:
    source = source.lower().strip()
    if source == "xml":
        favorites: dict[str, set[Any]] = {}
        if username:
            try:
                favorites = fetch_jikan_favorites(username, catalog_ids)
            except RuntimeError as exc:
                print(f"Could not fetch Jikan favorites for MAL username {username}: {exc}")
        return parse_mylist(MYLIST_XML), {
            "profile_source": "mylist_xml_with_mal_favorites" if username else "mylist_xml",
            "username": username,
            "favorite_anime_ids": favorites.get("favorite_anime_ids", set()),
            "favorite_voice_actor_ids": favorites.get("favorite_voice_actor_ids", set()),
            "favorite_staff_ids": favorites.get("favorite_staff_ids", set()),
            "favorite_people_names": favorites.get("favorite_people_names", set()),
            "favorite_studio_names": favorites.get("favorite_studio_names", set()),
            "favorite_studio_ids": favorites.get("favorite_studio_ids", set()),
        }
    if source == "anilist":
        if not username:
            raise ValueError("AniList source requires a username.")
        user_list = fetch_anilist_user_list(username, catalog_ids)
        try:
            favorites = fetch_anilist_favorites(username, catalog_ids)
        except RuntimeError as exc:
            print(f"Could not fetch AniList favorites for {username}: {exc}")
            favorites = {}
        return user_list, {
            "profile_source": "anilist_username",
            "username": username,
            "favorite_anime_ids": favorites.get("favorite_anime_ids", set()),
            "favorite_voice_actor_ids": favorites.get("favorite_voice_actor_ids", set()),
            "favorite_staff_ids": favorites.get("favorite_staff_ids", set()),
            "favorite_people_names": favorites.get("favorite_people_names", set()),
            "favorite_studio_names": favorites.get("favorite_studio_names", set()),
            "favorite_studio_ids": favorites.get("favorite_studio_ids", set()),
        }
    raise ValueError(f"Unknown source {source!r}. Use 'xml' or 'anilist'.")


def profile_band(scored_count: int) -> str:
    if scored_count <= 0:
        return "Cold Starter"
    if scored_count < 50:
        return "Beginner"
    if scored_count < 150:
        return "Casual"
    if scored_count < 500:
        return "Fan"
    return "Veteran"


def normalized_series(series: pd.Series, higher_is_better: bool = True) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo = values.quantile(0.02)
    hi = values.quantile(0.98)
    clipped = values.clip(lo, hi)
    denom = hi - lo
    if not denom or pd.isna(denom):
        out = pd.Series(0.0, index=series.index)
    else:
        out = (clipped - lo) / denom
    if not higher_is_better:
        out = 1 - out
    return out.fillna(0)


def catalog_signals(catalog: pd.DataFrame) -> pd.DataFrame:
    df = catalog.copy()
    df["score_signal"] = normalized_series(df["score"])
    df["member_signal"] = normalized_series(np.log1p(df["members"]))
    df["popularity_signal"] = normalized_series(df["popularity"], higher_is_better=False)
    df["recent_signal"] = np.where(df["aired_year"].fillna(0) >= CURRENT_YEAR - 1, 1.0, 0.0)
    df["safe_runtime_signal"] = np.where(df["total_watch_minutes"].fillna(10_000) <= 900, 1.0, 0.4)
    return df


def profile_vectors(
    catalog: pd.DataFrame,
    user_list: pd.DataFrame,
    *,
    favorite_anime_ids: set[int] | None = None,
    favorite_voice_actor_ids: set[int] | None = None,
    favorite_staff_ids: set[int] | None = None,
    favorite_people_names: set[str] | None = None,
    favorite_studio_names: set[str] | None = None,
    favorite_studio_ids: set[int] | None = None,
    profile_source: str = "mylist_xml",
    username: str | None = None,
) -> dict[str, Any]:
    favorite_anime_ids = favorite_anime_ids or set()
    merged = user_list.merge(catalog[["mal_id", "genres", "tags", "rating", "studios"]], on="mal_id", how="inner")
    missing_favorites = sorted(favorite_anime_ids - set(merged["mal_id"].astype(int)))
    if missing_favorites:
        favorite_rows = catalog[catalog["mal_id"].isin(missing_favorites)][["mal_id", "title", "genres", "tags", "rating", "studios"]].copy()
        favorite_rows["score"] = 10
        favorite_rows["status"] = "favorite"
        merged = pd.concat([merged, favorite_rows], ignore_index=True, sort=False)
    scored = merged[merged["score"].gt(0)].copy()
    positives = scored[(scored["score"].ge(POSITIVE_SCORE)) & (~scored["status"].str.lower().eq("dropped"))].copy()
    strong = scored[(scored["score"].ge(STRONG_SCORE)) & (~scored["status"].str.lower().eq("dropped"))].copy()
    negatives = scored[(scored["score"].lt(POSITIVE_SCORE)) | (scored["status"].str.lower().eq("dropped"))].copy()
    genre_counter: Counter[str] = Counter()
    tag_counter: Counter[str] = Counter()
    negative_genre_counter: Counter[str] = Counter()
    negative_tag_counter: Counter[str] = Counter()
    for row in positives.itertuples(index=False):
        weight = max(float(row.score) - 6, 1)
        if int(row.mal_id) in favorite_anime_ids:
            weight += 3
        genre_counter.update({genre: weight for genre in split_pipe(row.genres)})
        tag_counter.update({tag: weight for tag in split_pipe(row.tags)})
    for row in negatives.itertuples(index=False):
        weight = max(7 - float(row.score), 1)
        if str(row.status).lower() == "dropped":
            weight += 2
        negative_genre_counter.update({genre: weight for genre in split_pipe(row.genres)})
        negative_tag_counter.update({tag: weight for tag in split_pipe(row.tags)})
    hentai_seen = merged["genres"].fillna("").str.contains("Hentai|Erotica", case=False, regex=True).sum()
    hentai_majority = bool(len(merged) and hentai_seen / max(len(merged), 1) >= 0.35)
    return {
        "scored": scored,
        "positives": positives,
        "strong": strong if not strong.empty else positives,
        "negatives": negatives,
        "known_ids": set(user_list["mal_id"].astype(int)) | favorite_anime_ids,
        "favorite_anime_ids": favorite_anime_ids,
        "favorite_voice_actor_ids": favorite_voice_actor_ids or set(),
        "favorite_staff_ids": favorite_staff_ids or set(),
        "favorite_people_names": {canonical_person_name(name) for name in (favorite_people_names or set()) if name},
        "favorite_studio_names": {str(name).strip() for name in (favorite_studio_names or set()) if str(name).strip()},
        "favorite_studio_ids": favorite_studio_ids or set(),
        "profile_source": profile_source,
        "username": username,
        "genre_counter": genre_counter,
        "tag_counter": tag_counter,
        "negative_genre_counter": negative_genre_counter,
        "negative_tag_counter": negative_tag_counter,
        "hentai_majority": hentai_majority,
        "profile_band": profile_band(int(scored["mal_id"].nunique())),
    }


def weighted_label_score(labels: list[str], counter: Counter[str]) -> float:
    total = sum(counter.values()) or 1
    return float(sum(counter.get(label, 0) for label in labels) / total)


def content_affinity(row: pd.Series, profile: dict[str, Any]) -> float:
    genres = split_pipe(row.get("genres"))
    tags = split_pipe(row.get("tags"))
    genre_score = weighted_label_score(genres, profile["genre_counter"])
    tag_score = weighted_label_score(tags, profile["tag_counter"])
    disliked_genre_score = weighted_label_score(genres, profile["negative_genre_counter"])
    disliked_tag_score = weighted_label_score(tags, profile["negative_tag_counter"])
    positive = 0.65 * genre_score + 0.35 * tag_score
    negative = 0.60 * disliked_genre_score + 0.40 * disliked_tag_score
    return float(max(positive - 0.75 * negative, 0.0))


def negative_affinity(row: pd.Series, profile: dict[str, Any]) -> float:
    genres = split_pipe(row.get("genres"))
    tags = split_pipe(row.get("tags"))
    return float(
        0.60 * weighted_label_score(genres, profile["negative_genre_counter"])
        + 0.40 * weighted_label_score(tags, profile["negative_tag_counter"])
    )


def has_unmet_prerequisite(row: pd.Series, known_ids: set[int]) -> bool:
    for target_id, relation in parse_relation_edges(row.get("relations")):
        if relation in BLOCKING_PREREQUISITE_RELATIONS and target_id not in known_ids:
            return True
    return False


def is_continuation_candidate(row: pd.Series, known_ids: set[int], catalog_by_id: dict[int, pd.Series] | None = None) -> bool:
    """True when an item belongs in continue_your_journey, not the main row."""
    title = str(row.get("title") or "")
    title_lower = title.lower()
    if re.search(r"\b(2nd|3rd|4th|5th|second|third|fourth|season\s*[2-9]|part\s*[2-9])\b", title_lower):
        return True
    for target_id, relation in parse_relation_edges(row.get("relations")):
        if relation in {"Prequel", "Parent Story", "Full Story"}:
            return True
        if relation in {"Sequel", "Side Story", "Alternative Version"} and target_id in known_ids:
            return True
        if relation in WEAK_CONTINUE_RELATIONS and catalog_by_id and target_id in catalog_by_id:
            if same_franchise_hint(title, catalog_by_id[target_id].get("title")):
                return True
    return False


def eligible_catalog(catalog: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    df = catalog[~catalog["mal_id"].isin(profile["known_ids"])].copy()
    df = df[df["type"].isin(ALLOWED_GENERAL_TYPES)]
    if not profile["hentai_majority"]:
        text = (
            df["genres"].fillna("")
            + "|"
            + df["explicit_tags"].fillna("")
            + "|"
            + df["rating"].fillna("")
        )
        df = df[~text.str.contains("Hentai|Erotica|Rx - Hentai", case=False, regex=True)]
    return df


def primary_genre(row: pd.Series) -> str:
    genres = split_pipe(row.get("genres"))
    return genres[0] if genres else "Unknown"


def add_row(
    catalog_by_id: dict[int, pd.Series],
    rows: list[dict[str, Any]],
    row_name: str,
    items: list[dict[str, Any]],
    description: str,
    profile: dict[str, Any],
    enforce_prerequisites: bool = True,
    global_seen: set[int] | None = None,
) -> None:
    clean_items = []
    seen: set[int] = set()
    global_seen = global_seen if global_seen is not None else set()
    genre_counts: Counter[str] = Counter()
    anchor_counts: Counter[str] = Counter()
    for item in items:
        mal_id = int(item["mal_id"])
        if mal_id in seen or mal_id in global_seen or mal_id not in catalog_by_id:
            continue
        seen.add(mal_id)
        source = catalog_by_id[mal_id]
        if is_low_value_related_entry(source):
            continue
        if enforce_prerequisites and has_unmet_prerequisite(source, profile["known_ids"]):
            continue
        genre = primary_genre(source)
        anchor = str(item.get("anchor", "") or "")
        if row_name not in {"continue_your_journey"}:
            if genre_counts[genre] >= 4:
                continue
            anchor_limit = 1 if row_name == "because_you_liked" else 3
            if (
                row_name != "general_recommendations"
                and anchor
                and not item.get("allow_anchor_repeat")
                and anchor_counts[anchor] >= anchor_limit
            ):
                continue
        genre_counts[genre] += 1
        if anchor:
            anchor_counts[anchor] += 1
        clean_items.append(
            {
                "row": row_name,
                "rank": len(clean_items) + 1,
                "mal_id": mal_id,
                "title": source.get("title"),
                "score": round(float(item.get("score", 0)), 4),
                "reason": item.get("reason", ""),
                "anchor": item.get("anchor", ""),
                "primary_genre": genre,
                "mal_url": source.get("url"),
                "anilist_id": int(source["anilist_id"]) if not is_missing(source.get("anilist_id")) else None,
            }
        )
        if len(clean_items) >= MAX_ROW_ITEMS:
            break
    if clean_items:
        rows.append({"row": row_name, "description": description, "items": clean_items})
        global_seen.update(int(item["mal_id"]) for item in clean_items)


def is_low_value_related_entry(row: pd.Series) -> bool:
    title = str(row.get("title") or "").lower()
    if not any(term in title for term in LOW_VALUE_TITLE_TERMS):
        return False
    entry_type = str(row.get("type") or "").lower()
    duration = float(row.get("total_watch_minutes") or 0)
    return not (entry_type == "movie" and duration >= 50)


def is_weak_people_target(row: pd.Series) -> bool:
    """Avoid tiny/weak related entries in people rows unless they are substantial."""
    duration = float(row.get("total_watch_minutes") or 0)
    members = float(row.get("members") or 0)
    score = float(row.get("score") or 0)
    relations = parse_relation_edges(row.get("relations"))
    if duration and duration < 30 and (members < 25_000 or score < 7.0):
        return True
    if is_missing(row.get("genres")) and members < 25_000:
        return True
    if duration and duration < 60 and relations and all(rel == "Other" for _, rel in relations):
        return True
    if str(row.get("type") or "") == "Special" and duration and duration < 30 and members < 25_000:
        return True
    return False


def is_minor_continuation_target(row: pd.Series) -> bool:
    duration = float(row.get("total_watch_minutes") or 0)
    score = float(row.get("score") or 0)
    entry_type = str(row.get("type") or "")
    return bool(duration and duration < 30 and entry_type != "Movie" and score < 7.5)


def is_bad_exploration_entry(row: pd.Series) -> bool:
    title = str(row.get("title") or "").lower()
    entry_type = str(row.get("type") or "")
    duration = float(row.get("total_watch_minutes") or 0)
    episodes = float(row.get("episodes") or 0)
    members = float(row.get("members") or 0)
    score = float(row.get("score") or 0)
    if any(term in title for term in SUMMARY_TITLE_TERMS):
        return True
    if is_low_value_related_entry(row) or is_minor_continuation_target(row):
        return True
    if duration and duration < 60 and entry_type != "TV":
        return True
    if episodes and episodes <= 1 and entry_type in {"Special", "OVA", "ONA"} and duration < 60:
        return True
    if members < 30_000 and score < 8.5:
        return True
    if is_continuation_candidate(row, set()):
        return True
    return False


def general_recommendations(candidates: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    df = candidates.copy()
    df["affinity"] = df.apply(lambda row: content_affinity(row, profile), axis=1)
    df["negative_affinity"] = df.apply(lambda row: negative_affinity(row, profile), axis=1)
    df["novelty_signal"] = 1 - df["member_signal"].clip(0, 1)
    band = profile["profile_band"]
    weights = LEVEL_GENERAL_WEIGHTS.get(band, LEVEL_GENERAL_WEIGHTS["Fan"])
    df["product_score"] = (
        weights["affinity"] * df["affinity"]
        + weights["score"] * df["score_signal"]
        + weights["members"] * df["member_signal"]
        + weights["recent"] * df["recent_signal"]
        + weights["runtime"] * df["safe_runtime_signal"]
        + weights["novelty"] * df["novelty_signal"]
        - 0.25 * df["negative_affinity"]
    )
    return [
        {
            "mal_id": int(row.mal_id),
            "score": row.product_score,
            "reason": f"{band} blend: profile match + quality/popularity/recency with disliked-tag penalty",
        }
        for row in df.sort_values("product_score", ascending=False).head(MAX_ROW_ITEMS * 2).itertuples(index=False)
    ]


def learned_general_recommendations(profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Use the learned guarded hybrid as the main row when available.

    The hand-built content formula is a fallback. For a real imported list, the
    general row should come from the collaborative/product hybrid because that
    is the model validated in the offline evaluator. This also avoids the
    failure mode where a Veteran profile gets over-exposed to obscure recent
    titles just because they are novel.
    """
    source_path = MYLIST_GUARDED_RECOMMENDATIONS_CSV if MYLIST_GUARDED_RECOMMENDATIONS_CSV.exists() else MYLIST_RECOMMENDATIONS_CSV
    if not source_path.exists():
        return []
    try:
        df = pd.read_csv(source_path)
    except EmptyDataError:
        return []
    if "mal_id" not in df.columns or "mylist_hybrid_score" not in df.columns:
        return []
    rows = []
    for item in df.head(MAX_ROW_ITEMS * 3).itertuples(index=False):
        mal_id = int(getattr(item, "mal_id"))
        if mal_id in profile["known_ids"]:
            continue
        rows.append(
            {
                "mal_id": mal_id,
                "score": float(getattr(item, "mylist_hybrid_score", 0.0) or 0.0),
                "reason": "learned guarded hybrid: collaborative profile score + popularity prior + franchise guardrails",
                "anchor": "MyList hybrid",
            }
        )
    return rows


def selected_recommender_models() -> dict[str, Any]:
    """Read the latest evaluation artifacts and choose the product model set.

    The product UI is row-based, but the main ranked row still needs a default
    learned ranker. We keep the strongest classical/product model as the safe,
    interpretable backbone and the top two advanced models as the learned
    reranking candidates for final comparison.
    """
    selected: dict[str, Any] = {
        "classical_backbone": None,
        "advanced_candidates": [],
        "final_general_ranker": None,
        "selection_metric_order": [
            "balanced_profile_hit_at_12",
            "hit_rate_at_12",
            "ndcg_at_12",
            "map_at_12",
        ],
    }
    metric_cols = selected["selection_metric_order"]
    if EVALUATION_METRICS_CSV.exists():
        classical = pd.read_csv(EVALUATION_METRICS_CSV)
        classical = classical[[col for col in ["model_layer", "method", *metric_cols] if col in classical.columns]].copy()
        if "model_layer" in classical.columns:
            classical = classical[classical["model_layer"].fillna("").str.contains("classical|product", case=False, regex=True)]
        if not classical.empty:
            sort_cols = [col for col in metric_cols if col in classical.columns]
            best = classical.sort_values(sort_cols, ascending=False).iloc[0].to_dict()
            selected["classical_backbone"] = {
                key: (round(float(value), 6) if isinstance(value, (float, np.floating)) else value)
                for key, value in best.items()
            }
    if ADVANCED_METRICS_CSV.exists():
        advanced = pd.read_csv(ADVANCED_METRICS_CSV)
        advanced = advanced[[col for col in ["method", *metric_cols] if col in advanced.columns]].copy()
        if not advanced.empty:
            sort_cols = [col for col in metric_cols if col in advanced.columns]
            selected_rows = advanced.sort_values(sort_cols, ascending=False).head(2)
            selected["advanced_candidates"] = [
                {
                    key: (round(float(value), 6) if isinstance(value, (float, np.floating)) else value)
                    for key, value in row.items()
                }
                for row in selected_rows.to_dict(orient="records")
            ]
    if selected["advanced_candidates"]:
        selected["final_general_ranker"] = selected["advanced_candidates"][0]["method"]
    elif selected["classical_backbone"]:
        selected["final_general_ranker"] = selected["classical_backbone"]["method"]
    return selected


def balanced_general_recommendations(
    candidates: pd.DataFrame,
    catalog_by_id: dict[int, pd.Series],
    profile: dict[str, Any],
    learned_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build the main row as current + proven catalog anchors + learned matches."""
    selected: list[dict[str, Any]] = []
    seen: set[int] = set()

    def add_item(item: dict[str, Any]) -> None:
        mal_id = int(item["mal_id"])
        if mal_id in seen or mal_id not in catalog_by_id:
            return
        row = catalog_by_id[mal_id]
        if is_continuation_candidate(row, profile["known_ids"], catalog_by_id):
            return
        if is_low_value_related_entry(row):
            return
        selected.append(item)
        seen.add(mal_id)

    pool = candidates.copy()
    pool["affinity"] = pool.apply(lambda row: content_affinity(row, profile), axis=1)
    pool["negative_affinity"] = pool.apply(lambda row: negative_affinity(row, profile), axis=1)
    pool["main_quality_score"] = (
        0.28 * pool["affinity"]
        + 0.28 * pool["score_signal"]
        + 0.28 * pool["member_signal"]
        + 0.16 * pool["recent_signal"]
        - 0.30 * pool["negative_affinity"]
    )

    recent = pool[pool.apply(is_current_window, axis=1)].copy()
    recent = recent[recent["score"].fillna(0).ge(7.0)]
    for row in recent.sort_values("main_quality_score", ascending=False).head(40).itertuples(index=False):
        add_item(
            {
                "mal_id": int(row.mal_id),
                "score": float(row.main_quality_score),
                "reason": f"{CURRENT_SEASON} {CURRENT_YEAR} window: current/recent show with profile and quality fit",
                "anchor": "current season window",
            }
        )
        if sum(1 for item in selected if item.get("anchor") == "current season window") >= GENERAL_CURRENT_ITEMS:
            break

    popular = pool[pool["score"].fillna(0).ge(8.0)].copy()
    popular["popular_quality_score"] = (
        0.40 * popular["score_signal"]
        + 0.38 * popular["member_signal"]
        + 0.15 * popular["affinity"]
        + 0.07 * popular["safe_runtime_signal"]
        - 0.25 * popular["negative_affinity"]
    )
    for row in popular.sort_values("popular_quality_score", ascending=False).head(80).itertuples(index=False):
        add_item(
            {
                "mal_id": int(row.mal_id),
                "score": float(row.popular_quality_score),
                "reason": "popular high-score anchor with mild profile fit",
                "anchor": "popular high-score",
            }
        )
        if sum(1 for item in selected if item.get("anchor") == "popular high-score") >= GENERAL_POPULAR_ITEMS:
            break

    for item in learned_items:
        copied = dict(item)
        copied["reason"] = "learned guarded hybrid profile match"
        add_item(copied)
        if len(selected) >= MAX_ROW_ITEMS * 2:
            break

    if len(selected) < MAX_ROW_ITEMS:
        for item in general_recommendations(candidates, profile):
            copied = dict(item)
            copied["reason"] = "content fallback: profile match + catalog quality"
            copied["anchor"] = "content fallback"
            add_item(copied)
            if len(selected) >= MAX_ROW_ITEMS * 2:
                break

    return selected


def because_you_liked(catalog: pd.DataFrame, catalog_by_id: dict[int, pd.Series], profile: dict[str, Any]) -> list[dict[str, Any]]:
    """Return one strong similar-title pick per liked anchor before repeats.

    This row is meant to read like a streaming-service carousel:
    "because you liked X, try Y". Repeating the same anchor several times makes
    the row feel less personal, so the first pass keeps the best unseen target
    for each favorite/high-rated seed. A second pass is allowed only as a fill
    strategy when the profile does not have enough distinct anchors with valid
    recommendation edges.
    """
    seeds = profile["strong"].sort_values(["score"], ascending=False).head(50).copy()
    favorite_ids = set(profile.get("favorite_anime_ids", set()))
    favorite_extra = catalog[catalog["mal_id"].isin(favorite_ids - set(seeds["mal_id"].astype(int)))].copy()
    if not favorite_extra.empty:
        favorite_extra["score"] = 10
        seeds = pd.concat([seeds, favorite_extra], ignore_index=True, sort=False)

    seeds = seeds.drop_duplicates("mal_id", keep="first")
    per_anchor: list[dict[str, Any]] = []
    overflow: list[dict[str, Any]] = []
    chosen_targets: set[int] = set()

    for seed in seeds.itertuples(index=False):
        if int(seed.mal_id) not in catalog_by_id:
            continue
        seed_row = catalog_by_id[int(seed.mal_id)]
        anchor_candidates: list[dict[str, Any]] = []
        for target_id, weight in parse_weighted_edges(seed_row.get("recommendations")):
            if target_id in profile["known_ids"] or target_id not in catalog_by_id:
                continue
            target_row = catalog_by_id[target_id]
            penalty = 1 - min(negative_affinity(target_row, profile), 0.85)
            score = float(weight) * (float(seed.score) / 10) * penalty
            anchor_candidates.append(
                {
                    "mal_id": target_id,
                    "score": score,
                    "anchor": seed_row.get("title"),
                    "reason": f"similar to {seed_row.get('title')} through recommendation edges",
                },
            )
        if not anchor_candidates:
            continue
        anchor_candidates = sorted(anchor_candidates, key=lambda item: item["score"], reverse=True)
        for candidate in anchor_candidates:
            if int(candidate["mal_id"]) not in chosen_targets:
                per_anchor.append(candidate)
                chosen_targets.add(int(candidate["mal_id"]))
                break
        for candidate in anchor_candidates[1:]:
            if int(candidate["mal_id"]) in chosen_targets:
                continue
            copied = dict(candidate)
            copied["allow_anchor_repeat"] = True
            overflow.append(copied)

    if len(per_anchor) >= MAX_ROW_ITEMS:
        return per_anchor
    overflow = sorted(overflow, key=lambda item: item["score"], reverse=True)
    return per_anchor + overflow


def continue_journey(catalog: pd.DataFrame, catalog_by_id: dict[int, pd.Series], profile: dict[str, Any]) -> list[dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}
    known = profile["known_ids"]
    liked = set(profile["positives"]["mal_id"].astype(int)) | set(profile.get("favorite_anime_ids", set()))
    relation_weights = {
        "Sequel": 5.0,
        "Prequel": 3.0,
        "Parent Story": 3.0,
        "Full Story": 3.0,
        "Alternative Version": 2.8,
        "Side Story": 2.0,
        "Alternative Setting": 0.8,
        "Other": 0.4,
    }
    for seed_id in liked:
        if seed_id not in catalog_by_id:
            continue
        seed_row = catalog_by_id[seed_id]
        for target_id, rel in parse_relation_edges(seed_row.get("relations")):
            if rel not in CONTINUE_RELATIONS or target_id in known or target_id not in catalog_by_id:
                continue
            target_row = catalog_by_id[target_id]
            if is_minor_continuation_target(target_row):
                continue
            if rel in WEAK_CONTINUE_RELATIONS and not same_franchise_hint(seed_row.get("title"), target_row.get("title")):
                continue
            weight = relation_weights.get(rel, 1.0)
            current = scores.setdefault(
                target_id,
                {
                    "mal_id": target_id,
                    "score": 0.0,
                    "anchor": seed_row.get("title"),
                    "reason": f"{rel} connected to a liked title",
                },
            )
            current["score"] += weight
    for source in catalog.itertuples(index=False):
        if int(source.mal_id) in known:
            continue
        source_row = catalog_by_id[int(source.mal_id)]
        if is_minor_continuation_target(source_row):
            continue
        for target_id, rel in parse_relation_edges(getattr(source, "relations", "")):
            if target_id in liked and rel in STRONG_CONTINUE_RELATIONS:
                current = scores.setdefault(
                    int(source.mal_id),
                    {
                        "mal_id": int(source.mal_id),
                        "score": 0.0,
                        "anchor": catalog_by_id[target_id].get("title"),
                        "reason": f"relation path from {catalog_by_id[target_id].get('title')}",
                    },
                )
                current["score"] += 2.0 if rel != "Side Story" else 1.2
    return sorted(scores.values(), key=lambda item: item["score"], reverse=True)


def people_you_like(profile: dict[str, Any], catalog_by_id: dict[int, pd.Series]) -> list[dict[str, Any]]:
    scores: dict[int, dict[str, Any]] = {}
    liked = set(profile["positives"]["mal_id"].astype(int))
    known = profile["known_ids"]
    for edge_path, person_col, fav_col, role_col, label, id_col in [
        (VA_EDGE_CSV, "voice_actor_name", "voice_actor_favorites", "character_role", "voice actor", "voice_actor_id_mal"),
        (STAFF_EDGE_CSV, "staff_name", "staff_favorites", "staff_role_group", "staff", "staff_id_anilist"),
    ]:
        if not edge_path.exists():
            continue
        edges = pd.read_csv(edge_path)
        edges["_person_name"] = edges[person_col].map(canonical_person_name)
        edges["_person_key"] = edges["_person_name"].str.lower()
        edges["_person_favorites"] = pd.to_numeric(edges[fav_col], errors="coerce").fillna(0)
        edges["_role"] = edges[role_col].fillna("").astype(str)
        if label == "voice actor":
            edges["_character_favorites"] = pd.to_numeric(edges.get("character_favorites", 0), errors="coerce").fillna(0)
            character_name = edges.get("character_name", pd.Series("", index=edges.index)).fillna("").astype(str).str.lower()
            generic_narrator = character_name.str.fullmatch(r"narrator|narration|voice|announcer", na=False)
            role_upper = edges["_role"].str.upper()
            useful_role = role_upper.isin({"MAIN", "SUPPORTING"})
            useful_background = edges["_character_favorites"].ge(750)
            edges = edges[(useful_role | useful_background) & ~generic_narrator].copy()
        else:
            role_key = edges["_role"].str.lower().str.replace(" ", "_", regex=False)
            edges = edges[role_key.isin({"director", "original_creator", "original_character_design"})].copy()

        explicit_ids = profile["favorite_voice_actor_ids"] if label == "voice actor" else profile["favorite_staff_ids"]
        explicit_names = {canonical_person_name(name).lower() for name in profile.get("favorite_people_names", set())}
        source_edges = edges[edges["mal_id"].isin(liked)].copy()
        if explicit_ids and id_col in edges.columns:
            explicit_source = edges[pd.to_numeric(edges[id_col], errors="coerce").isin(explicit_ids)].copy()
            source_edges = pd.concat([source_edges, explicit_source], ignore_index=True).drop_duplicates()
        if explicit_names:
            explicit_name_source = edges[edges["_person_key"].isin(explicit_names)].copy()
            source_edges = pd.concat([source_edges, explicit_name_source], ignore_index=True).drop_duplicates()
        if source_edges.empty:
            continue
        people = (
            source_edges.groupby("_person_key")
            .agg(
                favorites=("_person_favorites", "max"),
                liked_anime=("mal_id", "nunique"),
                display_name=("_person_name", "first"),
            )
            .sort_values(["liked_anime", "favorites"], ascending=False)
            .head(30)
        )
        target_edges = edges[edges["_person_key"].isin(people.index) & ~edges["mal_id"].isin(known)].copy()
        for _, row in target_edges.iterrows():
            mal_id = int(row["mal_id"])
            if mal_id not in catalog_by_id:
                continue
            target_row = catalog_by_id[mal_id]
            if is_weak_people_target(target_row):
                continue
            if not profile["hentai_majority"]:
                text = (
                    str(target_row.get("genres") or "")
                    + "|"
                    + str(target_row.get("explicit_tags") or "")
                    + "|"
                    + str(target_row.get("rating") or "")
                )
                if re.search(r"Hentai|Erotica|Rx - Hentai", text, flags=re.IGNORECASE):
                    continue
            person_key = row["_person_key"]
            person = str(people.loc[person_key, "display_name"])
            favorites = float(row.get("_person_favorites", 0) or 0)
            role = str(row.get("_role", ""))
            role_norm = role.upper().replace(" ", "_")
            if label == "voice actor":
                boost = 1.5 if role_norm == "MAIN" else 1.15 if role_norm == "SUPPORTING" else 0.85
                character_fav = float(row.get("_character_favorites", 0) or 0)
                boost += min(math.log1p(character_fav) / 12, 0.6)
            else:
                boost = 2.35 if role_norm in {"DIRECTOR", "ORIGINAL_CREATOR"} else 1.75
            penalty = 1 - min(negative_affinity(target_row, profile), 0.80)
            source_strength = float(people.loc[person_key, "liked_anime"])
            score = (math.log1p(favorites) * boost + source_strength) * penalty
            current = scores.setdefault(
                mal_id,
                {
                    "mal_id": mal_id,
                    "score": 0.0,
                    "anchor": person,
                    "reason": f"shared {label}: {person} ({role})",
                    "people_category": label,
                },
            )
            current["score"] += score
    studio_counter: Counter[str] = Counter()
    studio_display: dict[str, str] = {}
    explicit_studios = {canonical_studio_name(name) for name in profile.get("favorite_studio_names", set())}
    for row in profile["positives"].itertuples(index=False):
        weight = max(float(row.score) - 6, 1)
        if int(row.mal_id) in profile.get("favorite_anime_ids", set()):
            weight += 3
        for studio in split_pipe(getattr(row, "studios", "")):
            key = canonical_studio_name(studio)
            if not key:
                continue
            studio_counter[key] += weight
            studio_display.setdefault(key, studio)
    for studio in profile.get("favorite_studio_names", set()):
        key = canonical_studio_name(studio)
        if key:
            studio_counter[key] += 12
            studio_display.setdefault(key, studio)
    if studio_counter:
        top_studios = {key for key, _ in studio_counter.most_common(20)} | explicit_studios
        for mal_id, target_row in catalog_by_id.items():
            if mal_id in known:
                continue
            if is_weak_people_target(target_row):
                continue
            if not profile["hentai_majority"]:
                text = (
                    str(target_row.get("genres") or "")
                    + "|"
                    + str(target_row.get("explicit_tags") or "")
                    + "|"
                    + str(target_row.get("rating") or "")
                )
                if re.search(r"Hentai|Erotica|Rx - Hentai", text, flags=re.IGNORECASE):
                    continue
            target_studios = {canonical_studio_name(studio) for studio in split_pipe(target_row.get("studios"))}
            matched = [key for key in target_studios if key in top_studios]
            if not matched:
                continue
            best_key = max(matched, key=lambda key: studio_counter[key])
            studio_name = studio_display.get(best_key, best_key)
            score = (
                math.log1p(studio_counter[best_key])
                + 1.8 * float(target_row.get("score_signal") or 0)
                + 1.4 * float(target_row.get("member_signal") or 0)
            )
            current = scores.setdefault(
                mal_id,
                {
                    "mal_id": mal_id,
                    "score": 0.0,
                    "anchor": studio_name,
                    "reason": f"shared studio: {studio_name}",
                    "people_category": "studio",
                },
            )
            current["score"] += score
    ordered = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
    staff_items = [item for item in ordered if item.get("people_category") == "staff"]
    studio_items = [item for item in ordered if item.get("people_category") == "studio"]
    va_items = [item for item in ordered if item.get("people_category") == "voice actor"]
    mixed: list[dict[str, Any]] = []
    used: set[int] = set()
    for item in staff_items[:3] + studio_items[:3] + va_items[:6] + ordered:
        mal_id = int(item["mal_id"])
        if mal_id in used:
            continue
        used.add(mal_id)
        mixed.append(item)
    return mixed


def give_it_a_try(candidates: pd.DataFrame, profile: dict[str, Any]) -> list[dict[str, Any]]:
    df = candidates.copy()
    df["affinity"] = df.apply(lambda row: content_affinity(row, profile), axis=1)
    df["negative_affinity"] = df.apply(lambda row: negative_affinity(row, profile), axis=1)
    df = df[~df.apply(is_bad_exploration_entry, axis=1)].copy()
    df = df[df["score"].fillna(0).ge(7.2)]
    df = df[df["members"].fillna(0).ge(30_000)]
    df["genre_distance"] = 1 - df["affinity"].clip(0, 1)
    df["retro_signal"] = np.where(df["aired_year"].fillna(CURRENT_YEAR) <= CURRENT_YEAR - 15, 1.0, 0.0)
    df["moderate_novelty"] = (1 - (df["member_signal"] - 0.45).abs() / 0.55).clip(0, 1)
    genre_counts = profile["genre_counter"]
    common_genres = {genre for genre, _ in genre_counts.most_common(5)}

    def genre_shift(row: pd.Series) -> float:
        genres = set(split_pipe(row.get("genres")))
        if not genres:
            return 0.0
        if genres.isdisjoint(common_genres):
            return 1.0
        return 0.45 if len(genres - common_genres) >= 2 else 0.0

    df["genre_shift_signal"] = df.apply(genre_shift, axis=1)
    df["try_score"] = (
        0.27 * df["score_signal"]
        + 0.18 * df["member_signal"]
        + 0.18 * df["moderate_novelty"]
        + 0.17 * df["genre_shift_signal"]
        + 0.13 * df["retro_signal"]
        + 0.12 * df["genre_distance"]
        - 0.38 * df["negative_affinity"]
    )
    df = df[df["affinity"].le(max(df["affinity"].quantile(0.55), 0.08))]
    return [
        {
            "mal_id": int(row.mal_id),
            "score": row.try_score,
            "reason": "controlled exploration: different genre/era/style with enough audience signal",
        }
        for row in df.sort_values("try_score", ascending=False).head(MAX_ROW_ITEMS * 5).itertuples(index=False)
    ]


def flatten_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        for item in row["items"]:
            records.append({**item, "row_description": row["description"]})
    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build streaming-style recommendation product rows.")
    parser.add_argument("--source", choices=["xml", "anilist"], default="xml", help="Use a MAL XML export or an AniList username.")
    parser.add_argument(
        "--username",
        default="",
        help="For --source xml, optional MAL username for Jikan favorites. For --source anilist, required AniList username.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog = catalog_signals(load_catalog())
    catalog_by_id = {int(row.mal_id): pd.Series(row._asdict()) for row in catalog.itertuples(index=False)}
    catalog_ids = set(catalog["mal_id"].astype(int))
    user_list, source_info = load_user_list_for_demo(args.source, args.username.strip() or None, catalog_ids)
    if user_list.empty:
        raise RuntimeError(f"No usable list entries found for source={args.source}, username={args.username!r}.")
    profile = profile_vectors(
        catalog,
        user_list,
        favorite_anime_ids=source_info["favorite_anime_ids"],
        favorite_voice_actor_ids=source_info["favorite_voice_actor_ids"],
        favorite_staff_ids=source_info["favorite_staff_ids"],
        favorite_people_names=source_info["favorite_people_names"],
        favorite_studio_names=source_info["favorite_studio_names"],
        favorite_studio_ids=source_info["favorite_studio_ids"],
        profile_source=source_info["profile_source"],
        username=source_info["username"],
    )
    candidates = eligible_catalog(catalog, profile)
    selected_models = selected_recommender_models()
    learned_general = learned_general_recommendations(profile)
    if args.source != "xml":
        learned_general = []
    general_items = balanced_general_recommendations(candidates, catalog_by_id, profile, learned_general)

    rows: list[dict[str, Any]] = []
    global_seen: set[int] = set()
    add_row(
        catalog_by_id,
        rows,
        "general_recommendations",
        general_items,
        (
            "Main row: current/recent season picks, popular high-score anchors, and learned guarded hybrid matches."
            if learned_general
            else "Fallback main row: current/recent season picks, popular high-score anchors, and content profile matches."
        ),
        profile,
        global_seen=global_seen,
    )
    add_row(
        catalog_by_id,
        rows,
        "because_you_liked",
        because_you_liked(catalog, catalog_by_id, profile),
        "Similar-anime row anchored on highly rated titles.",
        profile,
        global_seen=global_seen,
    )
    add_row(
        catalog_by_id,
        rows,
        "continue_your_journey",
        continue_journey(catalog, catalog_by_id, profile),
        "Relation-aware franchise row for sequels, side stories, parent stories, specials, and related entries.",
        profile,
        enforce_prerequisites=False,
        global_seen=global_seen,
    )
    add_row(
        catalog_by_id,
        rows,
        "people_you_like",
        people_you_like(profile, catalog_by_id),
        "Voice actor, director, original creator, original character design, and studio row.",
        profile,
        global_seen=global_seen,
    )
    add_row(
        catalog_by_id,
        rows,
        "give_it_a_try",
        give_it_a_try(candidates, profile),
        "Controlled exploration row: strong titles that deviate from the dominant taste sphere.",
        profile,
        global_seen=global_seen,
    )

    flat = flatten_rows(rows)
    flat.to_csv(ROWS_CSV, index=False)
    summary = {
        "profile_band": profile["profile_band"],
        "profile_source": profile["profile_source"],
        "username": profile["username"],
        "scored_count": int(len(profile["scored"])),
        "positive_count": int(len(profile["positives"])),
        "known_count": int(len(profile["known_ids"])),
        "favorite_anime_ids_used": len(profile["favorite_anime_ids"]),
        "favorite_voice_actor_ids_used": len(profile["favorite_voice_actor_ids"]),
        "favorite_staff_ids_used": len(profile["favorite_staff_ids"]),
        "favorite_people_names_used": len(profile["favorite_people_names"]),
        "favorite_studio_names_used": len(profile["favorite_studio_names"]),
        "hentai_majority": bool(profile["hentai_majority"]),
        "general_recommendation_source": "balanced_current_popular_learned" if learned_general else "balanced_current_popular_content",
        "selected_recommender_models": selected_models,
        "models_used_for_rows": {
            "general_recommendations": [
                selected_models.get("final_general_ranker") or "level_tuned_product_hybrid",
                "current_season_quality_prior",
                "popular_high_score_prior",
            ],
            "because_you_liked": ["item-to-item recommendation graph", "anchor-level deduplication"],
            "continue_your_journey": ["franchise relation graph", "prerequisite/status guardrails"],
            "people_you_like": ["voice actor affinity", "director/original creator/original story affinity", "studio affinity"],
            "give_it_a_try": ["metadata-content distance", "quality floor", "novelty and anti-recap guardrails"],
        },
        "rows": {row["row"]: len(row["items"]) for row in rows},
        "architecture": {
            "product_shape": "hybrid row-based recommender, not one universal ranked list",
            "pipeline": [
                "profile band detection",
                "candidate generators: collaborative/product hybrid, metadata, graph relations, people/staff/studios, popularity/current season",
                "hard filters: known titles, explicit defaults, low-value recaps/shorts, unmet prerequisites",
                "row-specific rankers",
                "cross-row deduplication and light diversity caps",
                "user-facing explanations and source links",
            ],
            "global_hard_filters": [
                "remove already known titles outside continuation context",
                "exclude hentai/explicit entries unless the profile indicates they are wanted",
                "avoid low-value recaps, manner movies, tiny shorts, and summary-only entries",
                "block sequels/specials with unmet prerequisites outside continue_your_journey",
            ],
        },
        "row_policy": {
            "max_items_per_row": MAX_ROW_ITEMS,
            "general_row_mix": {
                "current_recent_items": GENERAL_CURRENT_ITEMS,
                "popular_high_score_items": GENERAL_POPULAR_ITEMS,
                "remaining_slots": "learned guarded hybrid profile matches, with content fallback",
            },
            "profile_band_cutoffs": {
                "Beginner": "1-49 scored entries",
                "Casual": "50-149 scored entries",
                "Fan": "150-499 scored entries",
                "Veteran": "500+ scored entries",
            },
            "negative_feedback": "Low scores and dropped entries reduce content, similar-title, people/staff, and exploration scores.",
            "sequencing_guardrail": "Rows other than continue_your_journey block titles with unmet Prequel, Parent Story, or Full Story prerequisites.",
            "weak_relation_guardrail": "Other and Alternative Setting relations only feed continue_your_journey when title tokens suggest the same franchise.",
            "diversity_guardrail": "Rows cap repeated primary genres and repeated anchors so one franchise/person cannot fill the whole row.",
            "cross_row_guardrail": "A title can appear in only one row; earlier rows claim it first.",
            "exploration_guardrail": "give_it_a_try avoids recaps, summary movies, tiny shorts, low-audience score spikes, and direct continuation entries.",
        },
        "outputs": {
            "rows_csv": project_path(ROWS_CSV),
        },
    }
    summary = relativize_payload(summary)
    SUMMARY_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
