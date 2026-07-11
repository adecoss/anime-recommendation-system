from __future__ import annotations

import argparse
import base64
import hashlib
import hmac
import json
import os
import re
import shutil
import time
from datetime import datetime, timezone
from json import JSONDecodeError
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

import pandas as pd

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None


BASE_DIR = Path(__file__).resolve().parents[1]
BUILD_DIR = BASE_DIR / "data" / "build"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
SECRETS_DIR = BASE_DIR / "secrets"

CURRENT_RATINGS_FILE = PROCESSED_DIR / "current_user_ratings.csv"
CURRENT_PROFILE_FEATURES_FILE = PROCESSED_DIR / "current_user_profile_features.csv"
CURRENT_RATINGS_SUMMARY_FILE = BUILD_DIR / "current_user_ratings_summary.json"
CURRENT_RATINGS_CHECKPOINT_FILE = BUILD_DIR / "current_user_ratings_checkpoint.json"
CURRENT_RATINGS_FAILED_FILE = BUILD_DIR / "current_user_ratings_failed_users.json"
USERNAME_QUEUE_FILE = BUILD_DIR / "current_usernames_encrypted_queue.json"
USERNAME_KEY_FILE = SECRETS_DIR / "current_usernames_encryption_key.txt"
ANIME_DATASET_FILE = PROCESSED_DIR / "anime_dataset.csv"
VOICE_ACTOR_INDEX_FILE = PROCESSED_DIR / "voice_actor_index.csv"

MAL_API_BASE = "https://api.myanimelist.net/v2"
MAL_WEB_BASE = "https://myanimelist.net"
JIKAN_API_BASE = "https://api.jikan.moe/v4"

EXPECTED_RATING_COLUMNS = ["userID", "animeID", "rating", "status"]
EXPECTED_PROFILE_FEATURE_COLUMNS = [
    "userID",
    "scored_count",
    "completed_count",
    "watching_count",
    "on_hold_count",
    "dropped_count",
    "plan_to_watch_count",
    "mean_score",
    "favorite_anime_ids",
    "favorite_voice_actor_ids",
    "account_age_years",
    "activity_recency_month",
]

REQUEST_HEADERS = {"User-Agent": "anime-recommender-course-project/1.0 (+local research notebook)"}
PROFILE_RE = re.compile(r"/profile/([A-Za-z0-9_\-]+)")
CLUB_ID_RE = re.compile(r"(?:clubid=|cid=|/club/)(\d+)")

CLUB_INDEX_PAGES_BY_SORT = {
    "largest": {"sort": 5, "pages": 25},
    "recent_comment": {"sort": 2, "pages": 25},
}
FORUM_SPECS = [
    {"kind": "board", "id": 5, "max_show": 700},
    {"kind": "subboard", "id": 2, "max_show": 2000},
    {"kind": "subboard", "id": 3, "max_show": 1450},
    {"kind": "subboard", "id": 5, "max_show": 2200},
    {"kind": "board", "id": 3, "max_show": 14000},
    {"kind": "board", "id": 4, "max_show": 9700},
    {"kind": "board", "id": 13, "max_show": 2100},
    {"kind": "board", "id": 15, "max_show": 15600},
    {"kind": "board", "id": 16, "max_show": 114400},
    {"kind": "board", "id": 8, "max_show": 33200},
    {"kind": "board", "id": 6, "max_show": 17600},
    {"kind": "board", "id": 9, "max_show": 22100},
    {"kind": "board", "id": 12, "max_show": 10250},
    {"kind": "board", "id": 10, "max_show": 8000},
    {"kind": "board", "id": 7, "max_show": 18100},
    {"kind": "board", "id": 1, "max_show": 75700},
    {"kind": "board", "id": 2, "max_show": 7100},
]

DEFAULT_SOURCE_ORDER = ["clubs", "recommendations", "reviews", "forums"]
DEFAULT_MAX_USERNAMES = 1_000_000
DEFAULT_MAX_USERS_TO_PROCESS = 1_000_000
DEFAULT_RECOMMENDATION_PAGES = 50
DEFAULT_REVIEW_PAGES = 50
DEFAULT_MIN_RATINGS_PER_USER = 10
MAL_API_PAGE_LIMIT = 1000
MAX_RETRIES = 4
COLLECTION_PROGRESS_SAVE_EVERY = 25


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def project_path(value: object) -> object:
    if not isinstance(value, (str, Path)):
        return value
    try:
        path = Path(str(value))
        if path.is_absolute():
            return str(path.relative_to(BASE_DIR))
    except (ValueError, OSError):
        pass
    text = str(value)
    base_text = str(BASE_DIR)
    if text.startswith(base_text):
        return text[len(base_text) :].lstrip("\\/")
    return text


def relativize_payload(value: object) -> object:
    if isinstance(value, dict):
        return {key: relativize_payload(item) for key, item in value.items()}
    if isinstance(value, list):
        return [relativize_payload(item) for item in value]
    return project_path(value)


def ensure_dirs() -> None:
    for directory in [BUILD_DIR, PROCESSED_DIR, SECRETS_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    last_error: PermissionError | None = None
    for attempt in range(1, 6):
        try:
            tmp.replace(path)
            return
        except PermissionError as exc:
            last_error = exc
            time.sleep(0.2 * attempt)
    raise last_error or PermissionError(f"Could not replace {path}")


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
        for candidate in sorted(path.parent.glob(path.name + ".*"), key=lambda item: item.stat().st_mtime, reverse=True):
            if candidate == corrupt_path or ".corrupt_" in candidate.name:
                continue
            try:
                payload = json.loads(candidate.read_text(encoding="utf-8-sig"))
            except (JSONDecodeError, UnicodeDecodeError, OSError):
                continue
            print(f"[WARN] Recovered JSON for {path.name} from {candidate.name}", flush=True)
            try:
                shutil.copy2(candidate, path)
            except OSError as copy_exc:
                print(f"[WARN] Could not copy recovered JSON back to {path}: {copy_exc}", flush=True)
            return payload
        return default


def normalize_secret_key(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")


def parse_secret_file(path: Path) -> tuple[dict[str, str], list[str]]:
    parsed: dict[str, str] = {}
    bare_values: list[str] = []
    if not path.exists():
        return parsed, bare_values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" in line:
            key, value = line.split("=", 1)
            parsed[normalize_secret_key(key)] = value.strip().strip('"').strip("'")
        elif ":" in line:
            key, value = line.split(":", 1)
            parsed[normalize_secret_key(key)] = value.strip().strip('"').strip("'")
        else:
            bare_values.append(line)
    return parsed, bare_values


SECRET_CACHE, SECRET_BARE_VALUES = parse_secret_file(SECRETS_DIR / "secret.txt")


def read_secret(*names: str, default: str | None = None, filename: str | None = None, bare_index: int | None = None) -> str | None:
    for name in names:
        env_value = os.getenv(name)
        if env_value:
            return env_value.strip()
    if filename:
        path = SECRETS_DIR / filename
        if path.exists():
            value = path.read_text(encoding="utf-8").strip()
            if value:
                return value
    for name in names:
        value = SECRET_CACHE.get(normalize_secret_key(name))
        if value:
            return value
    if bare_index is not None and len(SECRET_BARE_VALUES) > bare_index:
        return SECRET_BARE_VALUES[bare_index]
    return default


MAL_ACCESS_TOKEN = read_secret("MAL_ACCESS_TOKEN", filename="mal_access_token.txt")
MAL_CLIENT_ID = read_secret(
    "MAL_CLIENT_ID",
    "MAL_CLIENTID",
    "MAL_CLIENT",
    "CLIENT_ID",
    filename="mal_client_id.txt",
    bare_index=0,
)
CURRENT_USER_RATINGS_SALT = read_secret(
    "CURRENT_USER_RATINGS_SALT",
    filename="current_user_ratings_salt.txt",
    default="anime_recommender_current_ratings_v1",
) or "anime_recommender_current_ratings_v1"


def mal_headers() -> dict[str, str]:
    headers = dict(REQUEST_HEADERS)
    if MAL_ACCESS_TOKEN:
        headers["Authorization"] = f"Bearer {MAL_ACCESS_TOKEN}"
    elif MAL_CLIENT_ID:
        headers["X-MAL-CLIENT-ID"] = MAL_CLIENT_ID
    else:
        raise RuntimeError("Missing MAL credentials. Set MAL_CLIENT_ID or MAL_ACCESS_TOKEN.")
    return headers


def encryption_key() -> bytes:
    ensure_dirs()
    if USERNAME_KEY_FILE.exists():
        return base64.urlsafe_b64decode(USERNAME_KEY_FILE.read_text(encoding="utf-8").strip().encode("ascii"))
    key = os.urandom(32)
    USERNAME_KEY_FILE.write_text(base64.urlsafe_b64encode(key).decode("ascii"), encoding="utf-8")
    return key


def xor_stream(key: bytes, length: int) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < length:
        out.extend(hashlib.sha256(key + counter.to_bytes(8, "big")).digest())
        counter += 1
    return bytes(out[:length])


def encrypt_username(username: str) -> str:
    raw = username.encode("utf-8")
    key = encryption_key()
    cipher = bytes(a ^ b for a, b in zip(raw, xor_stream(key, len(raw))))
    return base64.urlsafe_b64encode(cipher).decode("ascii")


def decrypt_username(token: str) -> str:
    cipher = base64.urlsafe_b64decode(token.encode("ascii"))
    key = encryption_key()
    raw = bytes(a ^ b for a, b in zip(cipher, xor_stream(key, len(cipher))))
    return raw.decode("utf-8")


def username_hash(username: str) -> str:
    text = f"{CURRENT_USER_RATINGS_SALT}:{username.strip().casefold()}"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def anonymized_user_id(username: str) -> int:
    return int(username_hash(username)[:15], 16) % 2_147_483_647


def queue_key(username: str) -> str:
    key = encryption_key()
    return hmac.new(key, username.strip().casefold().encode("utf-8"), hashlib.sha256).hexdigest()


def load_username_queue() -> dict[str, Any]:
    payload = load_json(USERNAME_QUEUE_FILE, {"updated_at": None, "pending": {}, "completed_hashes": {}, "rejected_hashes": {}})
    payload.setdefault("pending", {})
    payload.setdefault("completed_hashes", {})
    payload.setdefault("rejected_hashes", {})
    return payload


def save_username_queue(payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    atomic_write_json(USERNAME_QUEUE_FILE, payload)


def reconcile_queue_with_existing_profiles(queue: dict[str, Any]) -> dict[str, int]:
    if not CURRENT_PROFILE_FEATURES_FILE.exists() or CURRENT_PROFILE_FEATURES_FILE.stat().st_size == 0:
        return {"pending_removed_existing_profiles": 0}
    try:
        profile_ids = set(pd.read_csv(CURRENT_PROFILE_FEATURES_FILE, usecols=["userID"])["userID"].dropna().astype(int))
    except (ValueError, pd.errors.EmptyDataError):
        return {"pending_removed_existing_profiles": 0}
    pending = queue.setdefault("pending", {})
    completed = queue.setdefault("completed_hashes", {})
    removed = 0
    for qk, entry in list(pending.items()):
        try:
            user_id = int(entry.get("userID"))
        except (TypeError, ValueError):
            continue
        if user_id not in profile_ids:
            continue
        h = str(entry.get("username_hash") or "")
        pending.pop(qk, None)
        if h:
            completed[h] = {
                "userID": user_id,
                "ratings_written": "already_in_profile_features",
                "recorded_at": now_iso(),
                "recovered_after_interrupted_run": True,
            }
        removed += 1
    return {"pending_removed_existing_profiles": removed}


def load_checkpoint() -> dict[str, Any]:
    payload = load_json(
        CURRENT_RATINGS_CHECKPOINT_FILE,
        {
            "updated_at": None,
            "completed_user_hashes": {},
            "failed_user_hashes": {},
            "source_state": {},
        },
    )
    payload.setdefault("completed_user_hashes", {})
    payload.setdefault("failed_user_hashes", {})
    payload.setdefault("source_state", {})
    return payload


def save_checkpoint(payload: dict[str, Any]) -> None:
    payload["updated_at"] = now_iso()
    atomic_write_json(CURRENT_RATINGS_CHECKPOINT_FILE, payload)


def save_failed_user_registry(payload: dict[str, Any]) -> None:
    atomic_write_json(
        CURRENT_RATINGS_FAILED_FILE,
        {
            "updated_at": now_iso(),
            "failed_user_hashes": payload.get("failed_user_hashes", {}),
        },
    )


def load_catalog_ids() -> set[int]:
    if not ANIME_DATASET_FILE.exists():
        return set()
    df = pd.read_csv(ANIME_DATASET_FILE, usecols=["mal_id"])
    return {int(value) for value in df["mal_id"].dropna().astype(int).tolist()}


def load_voice_actor_person_ids() -> set[int]:
    if not VOICE_ACTOR_INDEX_FILE.exists():
        return set()
    try:
        df = pd.read_csv(VOICE_ACTOR_INDEX_FILE, usecols=["voice_actor_id_mal"])
    except ValueError:
        return set()
    ids: set[int] = set()
    for value in df["voice_actor_id_mal"]:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "<na>"}:
            continue
        try:
            ids.add(int(float(text)))
        except (TypeError, ValueError):
            continue
    return ids


def request_text(url: str, *, delay_seconds: float) -> str:
    if requests is None:
        raise RuntimeError("requests is not installed")
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=REQUEST_HEADERS, timeout=45)
        except requests.RequestException as exc:
            last_error = f"request error: {type(exc).__name__}: {exc}"
            sleep_seconds = min(120, (2**attempt) * delay_seconds)
            print(f"Retryable public request error {last_error}; sleeping {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
            continue
        if response.status_code == 200:
            return response.text
        last_error = f"HTTP {response.status_code}: {response.text[:200]}"
        if response.status_code in {429, 500, 502, 503, 504}:
            sleep_seconds = min(120, (2**attempt) * delay_seconds)
            print(f"Retryable public page error {last_error}; sleeping {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
            continue
        raise RuntimeError(last_error)
    raise RuntimeError(last_error or "public page request failed")


def extract_profile_usernames(html: str) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for match in PROFILE_RE.finditer(html or ""):
        username = unquote(match.group(1)).strip()
        key = username.casefold()
        if username and key not in seen:
            seen.add(key)
            names.append(username)
    return names


def extract_club_ids(html: str) -> list[int]:
    ids: list[int] = []
    seen: set[int] = set()
    for match in CLUB_ID_RE.finditer(html or ""):
        club_id = int(match.group(1))
        if club_id not in seen:
            seen.add(club_id)
            ids.append(club_id)
    return ids


def add_usernames_to_queue(
    usernames: list[str],
    queue: dict[str, Any],
    *,
    source: str,
    source_ref: str,
    max_usernames: int,
) -> int:
    added = 0
    pending = queue.setdefault("pending", {})
    completed = queue.setdefault("completed_hashes", {})
    rejected = queue.setdefault("rejected_hashes", {})
    for username in usernames:
        if len(pending) >= max_usernames:
            break
        h = username_hash(username)
        qk = queue_key(username)
        if qk in pending or h in completed or h in rejected:
            continue
        pending[qk] = {
            "encrypted_username": encrypt_username(username),
            "username_hash": h,
            "userID": anonymized_user_id(username),
            "source": source,
            "source_ref": source_ref,
            "discovered_at": now_iso(),
            "attempts": 0,
            "last_error": None,
        }
        added += 1
    return added


def discover_club_ids(checkpoint: dict[str, Any], *, delay_seconds: float, max_usernames: int) -> list[int]:
    source_state = checkpoint.setdefault("source_state", {})
    discovered = list(source_state.get("club_ids_discovered", []))
    seen = set(discovered)
    completed_by_sort = source_state.setdefault("club_index_pages_completed_by_sort", {})
    for sort_name, config in CLUB_INDEX_PAGES_BY_SORT.items():
        completed_pages = set(completed_by_sort.get(sort_name, []))
        for page in range(1, int(config["pages"]) + 1):
            if len(discovered) >= max_usernames:
                break
            if page in completed_pages:
                continue
            url = f"{MAL_WEB_BASE}/clubs.php?sort={config['sort']}&p={page}"
            print(f"discover club index sort={sort_name} page={page}: {url}", flush=True)
            html = request_text(url, delay_seconds=delay_seconds)
            before = len(discovered)
            for club_id in extract_club_ids(html):
                if club_id not in seen:
                    seen.add(club_id)
                    discovered.append(club_id)
            completed_pages.add(page)
            completed_by_sort[sort_name] = sorted(completed_pages)
            source_state["club_ids_discovered"] = discovered
            save_checkpoint(checkpoint)
            print(f"  club ids added={len(discovered) - before:,}; total={len(discovered):,}", flush=True)
            time.sleep(delay_seconds)
    return discovered


def discover_club_members(queue: dict[str, Any], checkpoint: dict[str, Any], *, delay_seconds: float, max_usernames: int) -> None:
    source_state = checkpoint.setdefault("source_state", {})
    club_ids = discover_club_ids(checkpoint, delay_seconds=delay_seconds, max_usernames=max_usernames)
    completed_offsets = source_state.setdefault("club_member_offsets_completed", {})
    completed_clubs = set(source_state.get("club_member_completed_club_ids", []))
    skipped_clubs = source_state.setdefault("club_member_skipped_club_ids", {})
    for club_position, club_id in enumerate(club_ids, start=1):
        if len(queue.get("pending", {})) >= max_usernames:
            break
        if club_id in completed_clubs or str(club_id) in skipped_clubs:
            continue
        done_offsets = set(completed_offsets.get(str(club_id), []))
        page_index = 0
        while len(queue.get("pending", {})) < max_usernames:
            offset = page_index * 36
            page_index += 1
            if offset in done_offsets:
                continue
            url = f"{MAL_WEB_BASE}/clubs.php?action=view&t=members&id={club_id}&show={offset}"
            print(f"discover club {club_position:,}/{len(club_ids):,} id={club_id} offset={offset}: {url}", flush=True)
            try:
                html = request_text(url, delay_seconds=delay_seconds)
            except RuntimeError as exc:
                reason = str(exc)[:300]
                if offset == 0:
                    skipped_clubs[str(club_id)] = {"reason": reason, "failed_at_offset": offset, "url": url}
                else:
                    completed_clubs.add(club_id)
                    source_state["club_member_completed_club_ids"] = sorted(completed_clubs)
                save_checkpoint(checkpoint)
                print(f"  stopped club {club_id}: {reason}", flush=True)
                break
            usernames = extract_profile_usernames(html)
            added = add_usernames_to_queue(usernames, queue, source="clubs", source_ref=url, max_usernames=max_usernames)
            save_username_queue(queue)
            done_offsets.add(offset)
            completed_offsets[str(club_id)] = sorted(done_offsets)
            save_checkpoint(checkpoint)
            print(f"  page usernames={len(usernames):,}; added={added:,}; queue={len(queue['pending']):,}", flush=True)
            time.sleep(delay_seconds)
            if not usernames:
                completed_clubs.add(club_id)
                source_state["club_member_completed_club_ids"] = sorted(completed_clubs)
                save_checkpoint(checkpoint)
                break


def discover_recommendations(queue: dict[str, Any], checkpoint: dict[str, Any], *, pages: int, delay_seconds: float, max_usernames: int) -> None:
    source_state = checkpoint.setdefault("source_state", {})
    completed_offsets = set(source_state.get("recommendation_offsets_completed", []))
    for offset in range(0, pages * 50, 50):
        if len(queue.get("pending", {})) >= max_usernames:
            break
        if offset in completed_offsets:
            continue
        url = f"{MAL_WEB_BASE}/recommendations.php?s=recentrecs&t=anime&show={offset}"
        print(f"discover recommendations offset={offset}: {url}", flush=True)
        html = request_text(url, delay_seconds=delay_seconds)
        usernames = extract_profile_usernames(html)
        added = add_usernames_to_queue(usernames, queue, source="recommendations", source_ref=url, max_usernames=max_usernames)
        save_username_queue(queue)
        completed_offsets.add(offset)
        source_state["recommendation_offsets_completed"] = sorted(completed_offsets)
        save_checkpoint(checkpoint)
        print(f"  usernames={len(usernames):,}; added={added:,}; queue={len(queue['pending']):,}", flush=True)
        time.sleep(delay_seconds)


def discover_reviews(queue: dict[str, Any], checkpoint: dict[str, Any], *, pages: int, delay_seconds: float, max_usernames: int) -> None:
    source_state = checkpoint.setdefault("source_state", {})
    completed_pages = set(source_state.get("review_pages_completed", []))
    for page in range(1, pages + 1):
        if len(queue.get("pending", {})) >= max_usernames:
            break
        if page in completed_pages:
            continue
        url = f"{MAL_WEB_BASE}/reviews.php?t=anime&p={page}"
        print(f"discover reviews page={page}: {url}", flush=True)
        html = request_text(url, delay_seconds=delay_seconds)
        usernames = extract_profile_usernames(html)
        added = add_usernames_to_queue(usernames, queue, source="reviews", source_ref=url, max_usernames=max_usernames)
        save_username_queue(queue)
        completed_pages.add(page)
        source_state["review_pages_completed"] = sorted(completed_pages)
        save_checkpoint(checkpoint)
        print(f"  usernames={len(usernames):,}; added={added:,}; queue={len(queue['pending']):,}", flush=True)
        time.sleep(delay_seconds)


def forum_url(spec: dict[str, Any], offset: int) -> str:
    return f"{MAL_WEB_BASE}/forum/?{spec['kind']}={spec['id']}&show={offset}"


def discover_forums(queue: dict[str, Any], checkpoint: dict[str, Any], *, delay_seconds: float, max_usernames: int) -> None:
    source_state = checkpoint.setdefault("source_state", {})
    completed = source_state.setdefault("forum_offsets_completed", {})
    for spec in FORUM_SPECS:
        spec_key = f"{spec['kind']}:{spec['id']}"
        completed_offsets = set(completed.get(spec_key, []))
        for offset in range(0, int(spec["max_show"]) + 1, 50):
            if len(queue.get("pending", {})) >= max_usernames:
                break
            if offset in completed_offsets:
                continue
            url = forum_url(spec, offset)
            print(f"discover forum {spec_key} offset={offset}: {url}", flush=True)
            html = request_text(url, delay_seconds=delay_seconds)
            usernames = extract_profile_usernames(html)
            added = add_usernames_to_queue(usernames, queue, source="forums", source_ref=url, max_usernames=max_usernames)
            save_username_queue(queue)
            completed_offsets.add(offset)
            completed[spec_key] = sorted(completed_offsets)
            save_checkpoint(checkpoint)
            print(f"  usernames={len(usernames):,}; added={added:,}; queue={len(queue['pending']):,}", flush=True)
            time.sleep(delay_seconds)
        if len(queue.get("pending", {})) >= max_usernames:
            break


def discover_usernames(args: argparse.Namespace) -> dict[str, Any]:
    queue = load_username_queue()
    checkpoint = load_checkpoint()
    for source in args.source_order:
        if len(queue.get("pending", {})) >= args.max_usernames:
            break
        if source == "clubs":
            discover_club_members(queue, checkpoint, delay_seconds=args.public_sleep, max_usernames=args.max_usernames)
        elif source == "recommendations":
            discover_recommendations(
                queue,
                checkpoint,
                pages=args.recommendation_pages,
                delay_seconds=args.public_sleep,
                max_usernames=args.max_usernames,
            )
        elif source == "reviews":
            discover_reviews(queue, checkpoint, pages=args.review_pages, delay_seconds=args.public_sleep, max_usernames=args.max_usernames)
        elif source == "forums":
            discover_forums(queue, checkpoint, delay_seconds=args.public_sleep, max_usernames=args.max_usernames)
        else:
            raise ValueError(f"Unknown discovery source: {source}")
    save_username_queue(queue)
    return {
        "pending_encrypted_usernames": len(queue.get("pending", {})),
        "completed_hashes": len(queue.get("completed_hashes", {})),
        "rejected_hashes": len(queue.get("rejected_hashes", {})),
    }


def request_json(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, delay_seconds: float) -> dict[str, Any]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    headers = headers or mal_headers()
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=45)
        except requests.RequestException as exc:
            last_error = f"request error: {type(exc).__name__}: {exc}"
            sleep_seconds = min(120, (2**attempt) * delay_seconds)
            print(f"Retryable JSON request error {last_error}; sleeping {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
            continue
        if response.status_code == 200:
            return response.json()
        if response.status_code in {401, 403}:
            raise RuntimeError(f"MAL auth/access error HTTP {response.status_code}. Public list may be private.")
        if response.status_code == 404:
            raise RuntimeError("MAL user not found or animelist unavailable")
        last_error = f"HTTP {response.status_code}: {response.text[:300]}"
        if response.status_code in {429, 500, 502, 503, 504}:
            sleep_seconds = min(120, (2**attempt) * delay_seconds)
            print(f"Retryable JSON error {last_error}; sleeping {sleep_seconds:.1f}s", flush=True)
            time.sleep(sleep_seconds)
            continue
        raise RuntimeError(last_error)
    raise RuntimeError(last_error or "JSON request failed")


def request_json_optional(url: str, *, params: dict[str, Any] | None = None, headers: dict[str, str] | None = None, delay_seconds: float) -> dict[str, Any] | None:
    try:
        return request_json(url, params=params, headers=headers or REQUEST_HEADERS, delay_seconds=delay_seconds)
    except RuntimeError:
        return None


def normalize_list_status(value: Any) -> str:
    text = str(value or "unknown").strip().lower()
    return {
        "completed": "completed",
        "watching": "watching",
        "on_hold": "on_hold",
        "on-hold": "on_hold",
        "on hold": "on_hold",
        "dropped": "dropped",
        "plan_to_watch": "plan_to_watch",
        "plan-to-watch": "plan_to_watch",
        "plan to watch": "plan_to_watch",
        "ptw": "plan_to_watch",
    }.get(text, text or "unknown")


def fetch_scored_ratings_for_user(username: str, *, catalog_ids: set[int], mal_sleep: float) -> tuple[list[dict[str, Any]], int]:
    encoded_username = quote(username, safe="@")
    url = f"{MAL_API_BASE}/users/{encoded_username}/animelist"
    params: dict[str, Any] | None = {"fields": "list_status", "limit": MAL_API_PAGE_LIMIT, "offset": 0}
    user_id = anonymized_user_id(username)
    rows: list[dict[str, Any]] = []
    page_count = 0
    while url:
        payload = request_json(url, params=params, delay_seconds=mal_sleep)
        page_count += 1
        for item in payload.get("data", []):
            node = item.get("node") or {}
            list_status = item.get("list_status") or {}
            anime_id = node.get("id")
            rating = list_status.get("score") or 0
            status = normalize_list_status(list_status.get("status"))
            if not rating or rating <= 0 or anime_id is None:
                continue
            if catalog_ids and int(anime_id) not in catalog_ids:
                continue
            rows.append({"userID": user_id, "animeID": int(anime_id), "rating": int(rating), "status": status})
        url = (payload.get("paging") or {}).get("next")
        params = None
        if url:
            time.sleep(mal_sleep)
    return rows, page_count


def parse_mal_datetime(value: Any) -> datetime | None:
    if not value:
        return None
    text = str(value).strip()
    for candidate in [text, text.replace("Z", "+00:00")]:
        try:
            parsed = datetime.fromisoformat(candidate)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            pass
    for fmt in ["%Y-%m-%d", "%b %d, %Y", "%B %d, %Y"]:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
        except ValueError:
            pass
    return None


def account_age_years(joined_at: Any) -> float | None:
    joined = parse_mal_datetime(joined_at)
    if joined is None:
        return None
    years = (datetime.now(timezone.utc) - joined).days / 365.25
    return round(max(years, 0), 1)


def activity_recency_month(last_online: Any) -> str | None:
    seen = parse_mal_datetime(last_online)
    if seen is None:
        return None
    return f"{seen.month:02d}.{seen.year:04d}"


def fetch_jikan_user_full_profile(username: str, *, jikan_sleep: float) -> dict[str, Any]:
    encoded_username = quote(username, safe="")
    url = f"{JIKAN_API_BASE}/users/{encoded_username}/full"
    payload = request_json_optional(url, headers=REQUEST_HEADERS, delay_seconds=jikan_sleep)
    time.sleep(jikan_sleep)
    return (payload or {}).get("data") or {}


def fetch_jikan_user_profile(username: str, *, jikan_sleep: float) -> dict[str, Any]:
    encoded_username = quote(username, safe="")
    url = f"{JIKAN_API_BASE}/users/{encoded_username}"
    payload = request_json_optional(url, headers=REQUEST_HEADERS, delay_seconds=jikan_sleep)
    time.sleep(jikan_sleep)
    return (payload or {}).get("data") or {}


def fetch_jikan_favorites(
    username: str,
    *,
    catalog_ids: set[int],
    voice_actor_person_ids: set[int],
    jikan_sleep: float,
) -> tuple[list[int], list[int]]:
    encoded_username = quote(username, safe="")
    url = f"{JIKAN_API_BASE}/users/{encoded_username}/favorites"
    payload = request_json_optional(url, headers=REQUEST_HEADERS, delay_seconds=jikan_sleep)
    time.sleep(jikan_sleep)
    anime_ids: list[int] = []
    seen_anime: set[int] = set()
    people_ids: list[int] = []
    seen_people: set[int] = set()
    for item in ((payload or {}).get("data") or {}).get("anime") or []:
        anime_id = item.get("mal_id")
        if anime_id is None:
            continue
        anime_id = int(anime_id)
        if catalog_ids and anime_id not in catalog_ids:
            continue
        if anime_id not in seen_anime:
            seen_anime.add(anime_id)
            anime_ids.append(anime_id)
    if not voice_actor_person_ids:
        return anime_ids, people_ids
    for item in ((payload or {}).get("data") or {}).get("people") or []:
        person_id = item.get("mal_id")
        if person_id is None:
            continue
        person_id = int(person_id)
        if person_id not in voice_actor_person_ids:
            continue
        if person_id not in seen_people:
            seen_people.add(person_id)
            people_ids.append(person_id)
    return anime_ids, people_ids


def build_profile_features(
    username: str,
    rows: list[dict[str, Any]],
    *,
    catalog_ids: set[int],
    voice_actor_person_ids: set[int],
    jikan_sleep: float,
) -> dict[str, Any]:
    status_counts = {"completed": 0, "watching": 0, "on_hold": 0, "dropped": 0, "plan_to_watch": 0}
    scores: list[float] = []
    for row in rows:
        status = normalize_list_status(row.get("status"))
        if status in status_counts:
            status_counts[status] += 1
        score = row.get("rating")
        if score is not None and score > 0:
            scores.append(float(score))
    mean_score = round(sum(scores) / len(scores), 4) if scores else None
    jikan_full_profile = fetch_jikan_user_full_profile(username, jikan_sleep=jikan_sleep)
    favorite_ids, favorite_voice_actor_ids = fetch_jikan_favorites(
        username,
        catalog_ids=catalog_ids,
        voice_actor_person_ids=voice_actor_person_ids,
        jikan_sleep=jikan_sleep,
    )
    joined_at = jikan_full_profile.get("joined")
    last_online = jikan_full_profile.get("last_online")
    if not joined_at or not last_online:
        jikan_profile = fetch_jikan_user_profile(username, jikan_sleep=jikan_sleep)
        joined_at = joined_at or jikan_profile.get("joined")
        last_online = last_online or jikan_profile.get("last_online")
    return {
        "userID": anonymized_user_id(username),
        "scored_count": int(len(rows)),
        "completed_count": int(status_counts["completed"]),
        "watching_count": int(status_counts["watching"]),
        "on_hold_count": int(status_counts["on_hold"]),
        "dropped_count": int(status_counts["dropped"]),
        "plan_to_watch_count": int(status_counts["plan_to_watch"]),
        "mean_score": float(mean_score) if mean_score is not None else None,
        "favorite_anime_ids": "|".join(str(anime_id) for anime_id in favorite_ids),
        "favorite_voice_actor_ids": "|".join(str(person_id) for person_id in favorite_voice_actor_ids),
        "account_age_years": account_age_years(joined_at),
        "activity_recency_month": activity_recency_month(last_online),
    }


def append_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ensure_csv_columns(path, columns)
    first_write = not path.exists() or path.stat().st_size == 0
    pd.DataFrame(rows, columns=columns).to_csv(path, mode="w" if first_write else "a", header=first_write, index=False)


def ensure_csv_columns(path: Path, columns: list[str]) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return
    try:
        df = pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return
    missing = [column for column in columns if column not in df.columns]
    if not missing:
        return
    for column in missing:
        df[column] = ""
    df = df[[column for column in columns if column in df.columns] + [column for column in df.columns if column not in columns]]
    df.to_csv(path, index=False)
    print(f"Updated CSV schema for {path}: added {missing}", flush=True)


def load_existing_profile_signatures() -> set[str]:
    if not CURRENT_PROFILE_FEATURES_FILE.exists() or CURRENT_PROFILE_FEATURES_FILE.stat().st_size == 0:
        return set()
    ensure_csv_columns(CURRENT_PROFILE_FEATURES_FILE, EXPECTED_PROFILE_FEATURE_COLUMNS)
    df = pd.read_csv(CURRENT_PROFILE_FEATURES_FILE)
    signatures = set()
    for _, row in df.iterrows():
        try:
            signatures.add(profile_signature(row.to_dict()))
        except (TypeError, ValueError):
            continue
    return signatures


def safe_profile_int(value: Any) -> int:
    if pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def safe_profile_float(value: Any) -> float:
    if pd.isna(value):
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def safe_profile_pipe_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value or "").strip()
    if text.casefold() in {"nan", "none", "<na>"}:
        return ""
    return text


def profile_signature(profile: dict[str, Any]) -> str:
    favorite_text = safe_profile_pipe_text(profile.get("favorite_anime_ids"))
    favorite_ids = "|".join(sorted(part for part in favorite_text.split("|") if part.strip()))
    favorite_voice_actor_text = safe_profile_pipe_text(profile.get("favorite_voice_actor_ids"))
    favorite_voice_actor_ids = "|".join(sorted(part for part in favorite_voice_actor_text.split("|") if part.strip()))
    parts = [
        str(safe_profile_int(profile.get("scored_count"))),
        str(safe_profile_int(profile.get("completed_count"))),
        str(round(safe_profile_float(profile.get("mean_score")), 2)),
        favorite_ids,
        favorite_voice_actor_ids,
    ]
    return hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()


def collect_ratings(args: argparse.Namespace) -> dict[str, Any]:
    queue = load_username_queue()
    reconciliation_counts = reconcile_queue_with_existing_profiles(queue)
    if reconciliation_counts.get("pending_removed_existing_profiles", 0):
        save_username_queue(queue)
        print(
            f"Removed {reconciliation_counts['pending_removed_existing_profiles']:,} already-collected users from pending queue.",
            flush=True,
        )
    checkpoint = load_checkpoint()
    catalog_ids = load_catalog_ids()
    voice_actor_person_ids = load_voice_actor_person_ids()
    print(
        f"Loaded catalog_ids={len(catalog_ids):,}; known_voice_actor_person_ids={len(voice_actor_person_ids):,}",
        flush=True,
    )
    profile_signatures = load_existing_profile_signatures()
    pending_items = list(queue.get("pending", {}).items())
    if args.user_limit is not None:
        pending_items = pending_items[: args.user_limit]
    processed = written_users = rejected = duplicate_similarity = failures = rows_written = 0
    dirty_progress = 0

    def mark_progress_dirty() -> None:
        nonlocal dirty_progress
        dirty_progress += 1

    def flush_progress(*, force: bool = False) -> None:
        nonlocal dirty_progress
        if not force and dirty_progress < COLLECTION_PROGRESS_SAVE_EVERY:
            return
        save_username_queue(queue)
        save_checkpoint(checkpoint)
        save_failed_user_registry(checkpoint)
        dirty_progress = 0

    for qk, entry in pending_items:
        username = decrypt_username(entry["encrypted_username"])
        h = entry["username_hash"]
        user_id = int(entry["userID"])
        print(f"fetching queued user hash={h[:10]} userID={user_id} source={entry.get('source')}", flush=True)
        try:
            rows, page_count = fetch_scored_ratings_for_user(username, catalog_ids=catalog_ids, mal_sleep=args.mal_sleep)
            profile = build_profile_features(
                username,
                rows,
                catalog_ids=catalog_ids,
                voice_actor_person_ids=voice_actor_person_ids,
                jikan_sleep=args.jikan_sleep,
            )
        except Exception as exc:
            failures += 1
            entry["attempts"] = int(entry.get("attempts") or 0) + 1
            entry["last_error"] = str(exc)[:500]
            entry["last_attempt_at"] = now_iso()
            if entry["attempts"] >= args.max_failed_attempts:
                queue["pending"].pop(qk, None)
                queue.setdefault("rejected_hashes", {})[h] = {
                    "userID": user_id,
                    "reason": f"failed_{entry['attempts']}_attempts",
                    "error": str(exc)[:500],
                    "source": entry.get("source"),
                    "recorded_at": now_iso(),
                }
                checkpoint.setdefault("failed_user_hashes", {}).pop(h, None)
                print(f"  failed {entry['attempts']}/{args.max_failed_attempts}; removed from queue: {exc}", flush=True)
            else:
                checkpoint.setdefault("failed_user_hashes", {})[h] = {
                    "userID": user_id,
                    "error": str(exc)[:500],
                    "attempts": entry["attempts"],
                    "last_attempt_at": now_iso(),
                    "source": entry.get("source"),
                }
                print(f"  failed {entry['attempts']}/{args.max_failed_attempts}: {exc}", flush=True)
            mark_progress_dirty()
            flush_progress()
            continue

        processed += 1
        if len(rows) < args.min_ratings_per_user:
            rejected += 1
            queue["pending"].pop(qk, None)
            queue.setdefault("rejected_hashes", {})[h] = {
                "userID": user_id,
                "reason": f"low_signal_less_than_{args.min_ratings_per_user}_ratings",
                "ratings_seen": len(rows),
                "recorded_at": now_iso(),
            }
            checkpoint.setdefault("completed_user_hashes", {})[h] = {
                "userID": user_id,
                "ratings_seen": len(rows),
                "ratings_written": 0,
                "rejected": True,
                "recorded_at": now_iso(),
            }
            mark_progress_dirty()
            flush_progress()
            print(f"  rejected low signal: ratings_seen={len(rows):,}", flush=True)
            continue

        sig = profile_signature(profile)
        if sig in profile_signatures:
            duplicate_similarity += 1
            queue["pending"].pop(qk, None)
            queue.setdefault("rejected_hashes", {})[h] = {
                "userID": user_id,
                "reason": "duplicate_similarity_existing_profile",
                "ratings_seen": len(rows),
                "recorded_at": now_iso(),
            }
            mark_progress_dirty()
            flush_progress()
            print(f"  skipped likely duplicate profile: ratings_seen={len(rows):,}", flush=True)
            continue

        append_csv(CURRENT_RATINGS_FILE, rows, EXPECTED_RATING_COLUMNS)
        append_csv(CURRENT_PROFILE_FEATURES_FILE, [profile], EXPECTED_PROFILE_FEATURE_COLUMNS)
        profile_signatures.add(sig)
        rows_written += len(rows)
        written_users += 1
        queue["pending"].pop(qk, None)
        queue.setdefault("completed_hashes", {})[h] = {"userID": user_id, "ratings_written": len(rows), "recorded_at": now_iso()}
        checkpoint.setdefault("completed_user_hashes", {})[h] = {
            "userID": user_id,
            "ratings_seen": len(rows),
            "ratings_written": len(rows),
            "pages": page_count,
            "recorded_at": now_iso(),
        }
        checkpoint.setdefault("failed_user_hashes", {}).pop(h, None)
        mark_progress_dirty()
        flush_progress()
        favorite_va_count = len([part for part in str(profile.get("favorite_voice_actor_ids") or "").split("|") if part.strip()])
        print(
            f"  written user ratings={len(rows):,} pages={page_count} "
            f"favorite_vas={favorite_va_count:,} remaining_queue={len(queue['pending']):,}",
            flush=True,
        )
    flush_progress(force=True)
    return {
        "processed_successfully": processed,
        "written_users": written_users,
        "ratings_rows_written": rows_written,
        "rejected_low_signal": rejected,
        "skipped_duplicate_similarity": duplicate_similarity,
        "failed_users_still_queued": failures,
        "pending_queue": len(queue.get("pending", {})),
        **reconciliation_counts,
    }


def write_summary(stage: str, details: dict[str, Any]) -> None:
    summary = {
        "updated_at": now_iso(),
        "stage": stage,
        "ratings_file": str(CURRENT_RATINGS_FILE),
        "profile_features_file": str(CURRENT_PROFILE_FEATURES_FILE),
        "encrypted_username_queue": str(USERNAME_QUEUE_FILE),
        "username_key_file": str(USERNAME_KEY_FILE),
        "privacy_note": (
            "Discovered usernames are stored only in a reversible local encrypted queue. "
            "Successful, low-signal, and likely duplicate users are removed from the pending queue; "
            "failed users remain encrypted for retry."
        ),
        **details,
    }
    atomic_write_json(CURRENT_RATINGS_SUMMARY_FILE, relativize_payload(summary))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover encrypted MAL usernames and collect current public ratings.")
    parser.add_argument("--discover-users", action="store_true", help="Gather usernames into the encrypted retry queue.")
    parser.add_argument("--collect-ratings", action="store_true", help="Consume encrypted username queue into ratings/profile CSVs.")
    parser.add_argument("--run", action="store_true", help="Run discovery, then collection.")
    parser.add_argument("--source-order", nargs="*", default=DEFAULT_SOURCE_ORDER, choices=DEFAULT_SOURCE_ORDER)
    parser.add_argument("--max-usernames", type=int, default=DEFAULT_MAX_USERNAMES)
    parser.add_argument("--user-limit", type=int, default=None, help="Limit queued users consumed by collection.")
    parser.add_argument("--recommendation-pages", type=int, default=DEFAULT_RECOMMENDATION_PAGES)
    parser.add_argument("--review-pages", type=int, default=DEFAULT_REVIEW_PAGES)
    parser.add_argument("--min-ratings-per-user", type=int, default=DEFAULT_MIN_RATINGS_PER_USER)
    parser.add_argument("--max-failed-attempts", type=int, default=3, help="Remove a queued encrypted username after this many failed collection attempts.")
    parser.add_argument("--public-sleep", type=float, default=2.5)
    parser.add_argument("--mal-sleep", type=float, default=1.0)
    parser.add_argument("--jikan-sleep", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    ensure_dirs()
    args = parse_args()
    if not (args.discover_users or args.collect_ratings or args.run):
        print(f"Encrypted username queue: {project_path(USERNAME_QUEUE_FILE)}")
        print(f"Ratings output: {project_path(CURRENT_RATINGS_FILE)}")
        print(f"Profile output: {project_path(CURRENT_PROFILE_FEATURES_FILE)}")
        print("Use --discover-users, --collect-ratings, or --run.")
        return
    details: dict[str, Any] = {}
    if args.run or args.discover_users:
        details["discovery"] = discover_usernames(args)
    if args.run or args.collect_ratings:
        details["collection"] = collect_ratings(args)
    write_summary("run" if args.run else "partial", details)
    print(json.dumps(details, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
