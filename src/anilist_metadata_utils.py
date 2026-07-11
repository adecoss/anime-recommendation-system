from __future__ import annotations

import json
import shutil
import time
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover - live mode is optional
    requests = None


ROOT = Path(__file__).resolve().parents[1]
ANILIST_DIR = ROOT / "data" / "reference" / "anilist"
ANILIST_SAMPLE_RESPONSE_PATH = ANILIST_DIR / "response.json"
ANILIST_CACHE_FILE = ROOT / "data" / "raw_sources" / "anilist" / "anilist_media_cache.json"
ANILIST_GRAPHQL_URL = "https://graphql.anilist.co"
ANILIST_MIN_REQUEST_DELAY_SECONDS = 2.2
ANILIST_MAX_RETRIES = 4
ANILIST_MEDIA_CACHE_VERSION = 3

DROP_TAGS = {"LGBTQ+ Themes"}
TAG_RENAMES = {
    "Boys' Love": "Yaoi",
    "Boys Love": "Yaoi",
    "Association Football": "Football",
}
DEMOGRAPHIC_TAGS = {"Josei", "Seinen", "Shoujo", "Shounen"}
MAL_EXPLICIT_GENRES = {"Ecchi", "Erotica", "Hentai"}
STRICT_EXPLICIT_RATINGS = {"Rx - Hentai"}
EXPLICIT_RATINGS = {"R+ - Mild Nudity", "Rx - Hentai"}
ANIDB_CONTENT_INDICATORS = {"nudity", "sex", "violence"}

# AniList marks many sexual/romance-adjacent tags as Sexual Content. For this
# project only Rx/Hentai titles move the full sexual-content tag set into
# explicit_tags. R+ titles get only direct content indicators such as Ecchi or
# Nudity, so a single fanservice episode does not make broad tags like Incest
# part of the explicit profile.
RATING_GATED_EXPLICIT_TAGS = {"Ecchi", "Nudity", "Sex", "Violence"}

MEDIA_QUERY = """
query MediaByMalId($idMal: Int) {
  Media(idMal: $idMal, type: ANIME) {
    id
    idMal
    title {
      romaji
    }
    type
    format
    status
    startDate {
      year
      month
      day
    }
    endDate {
      year
      month
      day
    }
    season
    seasonYear
    episodes
    duration
    countryOfOrigin
    source
    studios(isMain: true, sort: [FAVOURITES_DESC, ID]) {
      nodes {
        name
        isAnimationStudio
      }
    }
    nextAiringEpisode {
      id
      airingAt
      timeUntilAiring
      episode
      mediaId
    }
    genres
    tags {
      id
      rank
    }
    isAdult
    recommendations(sort: RATING_DESC, page: 1, perPage: 25) {
      nodes {
        rating
        mediaRecommendation {
          id
          idMal
        }
      }
    }
    mainCharacters: characters(role: MAIN, page: 1, perPage: 25, sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
      edges {
        role
        node {
          id
          favourites
          name {
            full
          }
        }
        voiceActors(sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
          id
          favourites
          name {
            full
          }
          languageV2
        }
      }
    }
    supportingCharacters: characters(role: SUPPORTING, page: 1, perPage: 25, sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
      edges {
        role
        node {
          id
          favourites
          name {
            full
          }
        }
        voiceActors(sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
          id
          favourites
          name {
            full
          }
          languageV2
        }
      }
    }
    backgroundCharacters: characters(role: BACKGROUND, page: 1, perPage: 25, sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
      edges {
        role
        node {
          id
          favourites
          name {
            full
          }
        }
        voiceActors(sort: [RELEVANCE, FAVOURITES_DESC, ID]) {
          id
          favourites
          name {
            full
          }
          languageV2
        }
      }
    }
    staff(sort: [RELEVANCE, ID]) {
      edges {
        role
        node {
          id
          favourites
          languageV2
          name {
            full
          }
        }
      }
    }
  }
}
"""


def now_iso() -> str:
    return datetime.now().replace(microsecond=0).isoformat()


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


def load_anilist_cache(path: Path = ANILIST_CACHE_FILE) -> dict[str, Any]:
    try:
        payload = load_json(path, {"updated_at": None, "items": {}})
    except (JSONDecodeError, UnicodeDecodeError) as exc:
        corrupt_path = path.with_suffix(path.suffix + f".corrupt_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        try:
            path.replace(corrupt_path)
            print(f"[WARN] AniList cache was corrupt and was moved to {corrupt_path}: {exc}", flush=True)
        except OSError as move_exc:
            print(f"[WARN] AniList cache is corrupt and could not be moved: {move_exc}", flush=True)

        for candidate in sorted(path.parent.glob(path.name + ".*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if candidate == corrupt_path or ".corrupt_" in candidate.name:
                continue
            try:
                payload = load_json(candidate, {"updated_at": None, "items": {}})
            except (JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            if isinstance(payload, dict) and isinstance(payload.get("items"), dict):
                print(f"[WARN] Recovered AniList cache from {candidate}", flush=True)
                try:
                    shutil.copy2(candidate, path)
                except OSError as copy_exc:
                    print(f"[WARN] Could not copy recovered AniList cache back to {path}: {copy_exc}", flush=True)
                payload.setdefault("items", {})
                return payload

        print("[WARN] No valid AniList cache backup was found; starting with an empty cache.", flush=True)
        payload = {"updated_at": None, "items": {}}
    payload.setdefault("items", {})
    return payload


def save_anilist_cache(payload: dict[str, Any], path: Path = ANILIST_CACHE_FILE) -> None:
    payload["updated_at"] = now_iso()
    atomic_write_json(path, payload)


def load_anilist_tag_lookup() -> dict[int, dict[str, Any]]:
    tags = load_json(ANILIST_DIR / "tags.json", [])
    lookup: dict[int, dict[str, Any]] = {}
    if not isinstance(tags, list):
        return lookup
    for tag in tags:
        if not isinstance(tag, dict) or tag.get("id") is None:
            continue
        try:
            lookup[int(tag["id"])] = {
                "id": int(tag["id"]),
                "name": tag.get("name") or "",
                "category": tag.get("category") or "",
                "description": tag.get("description") or "",
                "isAdult": bool(tag.get("isAdult")),
            }
        except (TypeError, ValueError):
            continue
    return lookup


ANILIST_TAG_LOOKUP = load_anilist_tag_lookup()


def compact_anilist_media(media: dict[str, Any] | None) -> dict[str, Any] | None:
    """Keep only fields used by the dataset pipeline.

    AniList media descriptions, country fields, and full dates are intentionally
    excluded. Tag names/categories/descriptions live in the local
    ``data/reference/anilist/tags.json`` reference file.
    """
    if not isinstance(media, dict):
        return None

    start = media.get("startDate") or {}
    end = media.get("endDate") or {}
    compact_tags: list[dict[str, int]] = []
    for tag in media.get("tags") or []:
        if not isinstance(tag, dict) or tag.get("id") is None:
            continue
        try:
            compact_tags.append(
                {
                    "id": int(tag["id"]),
                    "rank": int(float(tag.get("rank") or 0)),
                }
            )
        except (TypeError, ValueError):
            continue

    compact_recommendations: list[dict[str, int]] = []
    raw_recommendations = media.get("recommendations") or {}
    if isinstance(raw_recommendations, dict):
        nodes = raw_recommendations.get("nodes") or []
    elif isinstance(raw_recommendations, list):
        nodes = raw_recommendations
    else:
        nodes = []
    for node in nodes:
        if not isinstance(node, dict):
            continue
        if "mal_id" in node:
            rec_mal_id = node.get("mal_id")
            rating = node.get("rating")
        else:
            rec_media = node.get("mediaRecommendation") or {}
            rec_mal_id = rec_media.get("idMal")
            rating = node.get("rating")
        try:
            rec_mal_int = int(float(rec_mal_id))
            rating_int = int(float(rating))
        except (TypeError, ValueError):
            continue
        if rec_mal_int > 0 and rating_int > 0:
            compact_recommendations.append({"mal_id": rec_mal_int, "rating": rating_int})

    compact_characters: list[dict[str, Any]] = []
    seen_character_ids: set[int] = set()
    for connection_name in ["mainCharacters", "supportingCharacters", "backgroundCharacters", "characters"]:
        raw_characters = media.get(connection_name) or {}
        character_edges = raw_characters.get("edges") if isinstance(raw_characters, dict) else []
        for edge in character_edges or []:
            if not isinstance(edge, dict):
                continue
            character = edge.get("node") or {}
            if not isinstance(character, dict) or character.get("id") is None:
                continue
            character_id = int(character["id"])
            if character_id in seen_character_ids:
                continue
            seen_character_ids.add(character_id)
            character_name = ((character.get("name") or {}).get("full") or "").strip()
            voice_actors: list[dict[str, Any]] = []
            for actor in edge.get("voiceActors") or []:
                if not isinstance(actor, dict) or actor.get("id") is None:
                    continue
                language = str(actor.get("languageV2") or "").strip()
                actor_name = ((actor.get("name") or {}).get("full") or "").strip()
                voice_actors.append(
                    {
                        "id": int(actor["id"]),
                        "name": actor_name,
                        "favorites": actor.get("favourites"),
                        "language": language,
                    }
                )
            compact_characters.append(
                {
                    "id": character_id,
                    "name": character_name,
                    "role": edge.get("role"),
                    "favorites": character.get("favourites"),
                    "voice_actors": voice_actors,
                }
            )

    compact_staff: list[dict[str, Any]] = []
    raw_staff = media.get("staff") or {}
    staff_edges = raw_staff.get("edges") if isinstance(raw_staff, dict) else []
    seen_staff_edges: set[tuple[int, str]] = set()
    for edge in staff_edges or []:
        if not isinstance(edge, dict):
            continue
        staff = edge.get("node") or {}
        if not isinstance(staff, dict) or staff.get("id") is None:
            continue
        role = str(edge.get("role") or "").strip()
        staff_id = int(staff["id"])
        edge_key = (staff_id, role.casefold())
        if edge_key in seen_staff_edges:
            continue
        seen_staff_edges.add(edge_key)
        compact_staff.append(
            {
                "id": staff_id,
                "name": ((staff.get("name") or {}).get("full") or "").strip(),
                "role": role,
                "favorites": staff.get("favourites"),
                "language": str(staff.get("languageV2") or "").strip(),
            }
        )

    title = media.get("title") or {}
    next_airing = media.get("nextAiringEpisode") or {}
    raw_studios = media.get("studios") or {}
    if isinstance(raw_studios, dict):
        studio_nodes = raw_studios.get("nodes") or []
    else:
        studio_nodes = []
    studios = [
        str(node.get("name")).strip()
        for node in studio_nodes
        if (
            isinstance(node, dict)
            and node.get("isAnimationStudio") is not False
            and str(node.get("name") or "").strip()
        )
    ]

    return {
        "id": media.get("id"),
        "idMal": media.get("idMal"),
        "title_romaji": media.get("title_romaji") or title.get("romaji"),
        "type": media.get("type"),
        "format": media.get("format"),
        "status": media.get("status"),
        "startDate": {
            "year": start.get("year"),
            "month": start.get("month"),
            "day": start.get("day"),
        },
        "endDate": {
            "year": end.get("year"),
            "month": end.get("month"),
            "day": end.get("day"),
        },
        "season": media.get("season"),
        "seasonYear": media.get("seasonYear"),
        "episodes": media.get("episodes"),
        "duration": media.get("duration"),
        "countryOfOrigin": media.get("countryOfOrigin"),
        "source": media.get("source"),
        "studios": studios,
        "next_airing_episode": media.get("next_airing_episode") or next_airing or None,
        "genres": media.get("genres") or [],
        "tags": compact_tags,
        "isAdult": bool(media.get("isAdult")),
        "recommendations": compact_recommendations,
        "characters": compact_characters,
        "staff": compact_staff,
    }


def seed_anilist_cache_from_sample(payload: dict[str, Any]) -> bool:
    if not ANILIST_SAMPLE_RESPONSE_PATH.exists():
        return False
    sample = load_json(ANILIST_SAMPLE_RESPONSE_PATH, {})
    media = sample.get("data", {}).get("Media") if isinstance(sample.get("data"), dict) else sample
    if not isinstance(media, dict) or not media.get("idMal"):
        return False
    key = str(int(media["idMal"]))
    if key in payload["items"]:
        return False
    payload["items"][key] = {
        "fetched_at": now_iso(),
        "source": str(ANILIST_SAMPLE_RESPONSE_PATH),
        "query_version": ANILIST_MEDIA_CACHE_VERSION,
        "media": compact_anilist_media(media),
    }
    return True


def split_pipe(value: Any) -> list[str]:
    if pd.isna(value):
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def merge_pipe(values: list[str]) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        clean = str(value or "").strip()
        key = clean.casefold()
        if clean and key not in seen:
            out.append(clean)
            seen.add(key)
    return "|".join(out)


def parse_weight_map(value: Any) -> dict[str, int]:
    weights: dict[str, int] = {}
    for part in split_pipe(value):
        if ":" not in part:
            continue
        name, raw_weight = part.rsplit(":", 1)
        try:
            weights[name.strip().casefold()] = int(float(raw_weight))
        except ValueError:
            continue
    return weights


def parse_recommendation_edges(value: Any) -> dict[int, list[int]]:
    edges: dict[int, list[int]] = {}
    for item in split_pipe(value):
        if ":" in item:
            target, raw_weight = item.split(":", 1)
        else:
            target, raw_weight = item, "1"
        try:
            target_id = int(float(target))
            weight = max(1, int(float(raw_weight)))
        except ValueError:
            continue
        edges.setdefault(target_id, []).append(weight)
    return edges


def format_average_recommendations(edge_sources: dict[int, list[int]]) -> str:
    averaged: list[tuple[int, int]] = []
    for target_id, weights in edge_sources.items():
        if not weights:
            continue
        averaged.append((target_id, max(1, round(sum(weights) / len(weights)))))
    averaged.sort(key=lambda item: item[1], reverse=True)
    return "|".join(f"{target}:{weight}" for target, weight in averaged)


def normalize_tag_name(name: str) -> str:
    return TAG_RENAMES.get(str(name or "").strip(), str(name or "").strip())


def has_strict_explicit_rating(row: pd.Series) -> bool:
    return str(row.get("rating", "")).strip() in STRICT_EXPLICIT_RATINGS


def has_explicit_rating(row: pd.Series) -> bool:
    return str(row.get("rating", "")).strip() in EXPLICIT_RATINGS


def anidb_content_indicator_tags(row: pd.Series) -> set[str]:
    tags = {tag.casefold() for tag in split_pipe(row.get("tags"))}
    tags |= {tag.casefold() for tag in split_pipe(row.get("explicit_tags"))}
    return tags & ANIDB_CONTENT_INDICATORS


def anidb_loli_weight(row: pd.Series) -> int | None:
    weights = parse_weight_map(row.get("explicit_tag_weights"))
    if "loli" in weights:
        return max(1, min(100, round(weights["loli"] / 6)))
    if "loli" in {tag.casefold() for tag in split_pipe(row.get("explicit_tags"))}:
        return 100
    return None


def is_hentai(row: pd.Series, media: dict[str, Any] | None = None) -> bool:
    mal_genres = set(split_pipe(row.get("genres")))
    anilist_genres = set((media or {}).get("genres") or [])
    return (
        "Hentai" in mal_genres
        or "Hentai" in anilist_genres
        or str(row.get("rating", "")).strip() == "Rx - Hentai"
    )


def classify_anilist_media(row: pd.Series, media: dict[str, Any] | None) -> dict[str, str]:
    if not media:
        return {
            "anilist_id": "",
            "genres": "",
            "tags": "",
            "tag_weights": "",
            "explicit_tags": "",
            "explicit_tag_weights": "",
            "demographics": "",
            "recommendations": "",
        }

    genres: list[str] = []
    tags: list[str] = []
    tag_weights: list[str] = []
    explicit_tags: list[str] = []
    explicit_weights: list[str] = []
    demographics: list[str] = []
    strict_explicit = has_strict_explicit_rating(row) or bool(media.get("isAdult"))
    explicit_rating = has_explicit_rating(row)
    content_indicators = anidb_content_indicator_tags(row)

    for genre in media.get("genres") or []:
        genre = str(genre)
        if genre == "Ecchi" and explicit_rating:
            explicit_tags.append("Ecchi")
            explicit_weights.append("Ecchi:100")
        else:
            genres.append(genre)

    for genre in split_pipe(row.get("genres")):
        if genre in {"Erotica", "Hentai"}:
            genres.append(genre)

    for tag in media.get("tags") or []:
        tag_meta = {}
        if isinstance(tag, dict) and tag.get("id") is not None:
            try:
                tag_meta = ANILIST_TAG_LOOKUP.get(int(tag["id"]), {})
            except (TypeError, ValueError):
                tag_meta = {}
        name = normalize_tag_name(str(tag.get("name") or tag_meta.get("name") or ""))
        if not name or name in DROP_TAGS:
            continue
        rank = int(tag.get("rank") or 0)
        category = str(tag.get("category") or tag_meta.get("category") or "")

        if name in DEMOGRAPHIC_TAGS or category == "Demographic":
            demographics.append(name)
            continue

        is_sexual_category = bool(tag.get("isAdult") or tag_meta.get("isAdult")) or category == "Sexual Content"
        is_rating_gated = name in RATING_GATED_EXPLICIT_TAGS and explicit_rating
        is_anidb_indicator = name.casefold() in content_indicators and explicit_rating

        if (strict_explicit and is_sexual_category) or is_rating_gated or is_anidb_indicator:
            explicit_tags.append(name)
            explicit_weights.append(f"{name}:{rank}")
        else:
            tags.append(name)
            tag_weights.append(f"{name}:{rank}")

    if is_hentai(row, media):
        loli_weight = anidb_loli_weight(row)
        if loli_weight is not None:
            explicit_tags.append("Loli")
            explicit_weights.append(f"Loli:{loli_weight}")

    recommendation_edges: list[str] = []
    raw_recommendations = media.get("recommendations") or {}
    nodes = raw_recommendations.get("nodes") if isinstance(raw_recommendations, dict) else raw_recommendations
    for node in nodes or []:
        if "mal_id" in node:
            rating = node.get("rating")
            rec_mal_id = node.get("mal_id")
        else:
            rating = node.get("rating")
            rec_media = node.get("mediaRecommendation") or {}
            rec_mal_id = rec_media.get("idMal")
        if rec_mal_id is None or rating is None:
            continue
        try:
            rating_int = int(float(rating))
            rec_mal_int = int(float(rec_mal_id))
        except ValueError:
            continue
        if rating_int <= 0:
            continue
        recommendation_edges.append(f"{rec_mal_int}:{rating_int}")

    return {
        "anilist_id": str(media.get("id") or ""),
        "genres": merge_pipe(genres),
        "tags": merge_pipe(tags),
        "tag_weights": merge_pipe(tag_weights),
        "explicit_tags": merge_pipe(explicit_tags),
        "explicit_tag_weights": merge_pipe(explicit_weights),
        "demographics": merge_pipe(demographics),
        "recommendations": merge_pipe(recommendation_edges),
    }


def anilist_retry_wait(response: Any, fallback_seconds: float) -> float:
    retry_after = response.headers.get("Retry-After")
    if retry_after:
        try:
            return max(fallback_seconds, float(retry_after) + 1)
        except ValueError:
            pass

    reset = response.headers.get("X-RateLimit-Reset")
    if reset and reset.isdigit():
        return max(fallback_seconds, int(reset) - int(time.time()) + 1)

    return fallback_seconds


def request_anilist_media(mal_id: int, sleep_seconds: float = 1.0) -> dict[str, Any] | None:
    if requests is None:
        raise RuntimeError("requests is not installed; live AniList calls are unavailable")

    delay = max(float(sleep_seconds or 0), ANILIST_MIN_REQUEST_DELAY_SECONDS)
    last_error = None
    for attempt in range(1, ANILIST_MAX_RETRIES + 1):
        response = requests.post(
            ANILIST_GRAPHQL_URL,
            json={"query": MEDIA_QUERY, "variables": {"idMal": int(mal_id)}},
            timeout=30,
        )
        remaining = response.headers.get("X-RateLimit-Remaining")

        if response.status_code == 429:
            wait = anilist_retry_wait(response, fallback_seconds=max(60.0, delay))
            last_error = f"AniList rate limited; waited {wait:.0f}s"
            print(
                f"AniList 429 for MAL {mal_id} on attempt {attempt}/{ANILIST_MAX_RETRIES}; "
                f"sleeping {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
            continue

        if response.status_code == 404:
            print(f"AniList has no media match for MAL {mal_id}; caching as missing", flush=True)
            time.sleep(delay)
            return None

        if response.status_code >= 500:
            wait = max(10.0, delay * attempt)
            last_error = f"AniList server error {response.status_code}; waited {wait:.0f}s"
            print(
                f"AniList {response.status_code} for MAL {mal_id} on attempt "
                f"{attempt}/{ANILIST_MAX_RETRIES}; sleeping {wait:.1f}s",
                flush=True,
            )
            time.sleep(wait)
            continue

        try:
            payload = response.json()
        except ValueError:
            payload = {}

        if response.status_code >= 400:
            errors = payload.get("errors") if isinstance(payload, dict) else None
            if errors:
                error_text = str(errors)[:500]
                if response.status_code == 400:
                    raise RuntimeError(f"AniList GraphQL bad request for MAL {mal_id}: {error_text}")
                raise RuntimeError(f"AniList HTTP {response.status_code} for MAL {mal_id}: {error_text}")
            response.raise_for_status()

        errors = payload.get("errors") or []
        if errors:
            statuses = {str(error.get("status")) for error in errors if isinstance(error, dict)}
            if "429" in statuses:
                wait = anilist_retry_wait(response, fallback_seconds=max(60.0, delay))
                last_error = f"AniList GraphQL rate limited; waited {wait:.0f}s"
                print(
                    f"AniList GraphQL 429 for MAL {mal_id} on attempt {attempt}/{ANILIST_MAX_RETRIES}; "
                    f"sleeping {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)
                continue
            if "404" in statuses:
                print(f"AniList GraphQL has no media match for MAL {mal_id}; caching as missing", flush=True)
                time.sleep(delay)
                return None
            raise RuntimeError(str(errors)[:500])

        if remaining is not None:
            try:
                if int(remaining) <= 2:
                    wait = anilist_retry_wait(response, fallback_seconds=max(10.0, delay))
                    print(f"AniList rate-limit remaining={remaining}; sleeping {wait:.1f}s", flush=True)
                    time.sleep(wait)
            except ValueError:
                pass

        time.sleep(delay)
        return compact_anilist_media(payload.get("data", {}).get("Media"))

    raise RuntimeError(last_error or "AniList request failed after retries")


def get_anilist_media(
    mal_id: int,
    cache: dict[str, Any],
    live: bool = False,
    sleep_seconds: float = 1.0,
) -> tuple[dict[str, Any] | None, str]:
    key = str(int(mal_id))
    cached_item = cache.get("items", {}).get(key)
    if cached_item and cached_item.get("query_version") == ANILIST_MEDIA_CACHE_VERSION:
        return cached_item.get("media"), "cache"
    if not live:
        return (cached_item or {}).get("media"), "stale_cache" if cached_item else "missing_cache"

    media = request_anilist_media(int(mal_id), sleep_seconds=sleep_seconds)
    cache.setdefault("items", {})[key] = {
        "fetched_at": now_iso(),
        "source": "anilist_graphql" if media else "anilist_graphql_missing",
        "query_version": ANILIST_MEDIA_CACHE_VERSION,
        "media": media,
    }
    save_anilist_cache(cache)
    return media, "live"


def merge_recommendation_sources(*values: Any) -> str:
    edge_sources: dict[int, list[int]] = {}
    for value in values:
        for target_id, weights in parse_recommendation_edges(value).items():
            edge_sources.setdefault(target_id, []).extend(weights)
    return format_average_recommendations(edge_sources)


def update_anilist_cache_for_mal_ids(
    mal_ids: list[int],
    cache: dict[str, Any],
    limit: int | None = None,
    sleep_seconds: float = 1.0,
) -> int:
    updated = 0
    selected = mal_ids if limit is None else mal_ids[:limit]
    total = len(selected)
    for index, mal_id in enumerate(selected, start=1):
        try:
            media, source = get_anilist_media(int(mal_id), cache, live=True, sleep_seconds=sleep_seconds)
        except Exception as exc:
            print(
                f"[{index}/{total}] AniList MAL {int(mal_id)} | failed | "
                f"{type(exc).__name__}: {str(exc)[:220]}",
                flush=True,
            )
            save_anilist_cache(cache)
            continue
        if source == "live":
            updated += 1
        print(
            f"[{index}/{total}] AniList MAL {int(mal_id)} | {source} | "
            f"anilist_id={(media or {}).get('id')}",
            flush=True,
        )
    save_anilist_cache(cache)
    return updated
