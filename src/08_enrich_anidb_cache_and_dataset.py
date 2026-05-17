from __future__ import annotations

import argparse
import importlib.util
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd
import requests

from anidb_metadata_utils import atomic_write_json, extract_anidb_payload


ROOT = Path(__file__).resolve().parents[1]
ANIDB_XML_CACHE_URL = "https://files.shokoanime.com/files/shoko-server/other/Anime_HTTP.zip"
ANIDB_ZIP_PATH = ROOT / "data" / "raw" / "Anime_HTTP.zip"
ANIDB_CACHE_FILE = ROOT / "data" / "caches" / "anidb_metadata_cache.json"
DATASET_CSV = ROOT / "data" / "processed" / "anime_dataset.csv"
DATASET_JSON = ROOT / "data" / "processed" / "anime_dataset.json"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def zip_info_datetime(zip_info: zipfile.ZipInfo) -> str | None:
    try:
        return datetime(*zip_info.date_time).isoformat(timespec="seconds")
    except (TypeError, ValueError):
        return None


def load_cache() -> dict:
    if not ANIDB_CACHE_FILE.exists():
        return {"updated_at": None, "items": {}}
    with ANIDB_CACHE_FILE.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    payload.setdefault("items", {})
    return payload


def compact_episode_summary(summary: dict | None) -> dict:
    summary = summary or {}
    keep_keys = [
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
    return {
        key: summary[key]
        for key in keep_keys
        if key in summary and summary[key] is not None
    }


def compact_similar_anime(items: list | None) -> list[dict]:
    compacted = []
    for item in items or []:
        anidb_id = item.get("anidb_id") or item.get("aid") or item.get("id")
        if anidb_id is None:
            continue
        compact = {"anidb_id": int(anidb_id)}
        if item.get("approval") is not None:
            compact["approval"] = int(item["approval"])
        if item.get("total") is not None:
            compact["total"] = int(item["total"])
        compacted.append(compact)
    return compacted


def compact_cache_payload(cache: dict) -> dict:
    for entry in cache.get("items", {}).values():
        if "episode_summary" in entry:
            entry["episode_summary"] = compact_episode_summary(entry.get("episode_summary"))
        if "similar_anime" in entry:
            entry["similar_anime"] = compact_similar_anime(entry.get("similar_anime"))
        if "animation_work_creators" in entry:
            studios = [
                creator.get("name")
                for creator in entry.get("animation_work_creators", []) or []
                if creator.get("name")
            ]
            if studios and not entry.get("animation_work_studios"):
                entry["animation_work_studios"] = studios
            entry.pop("animation_work_creators", None)
        entry.pop("extra_metadata", None)
    cache["updated_at"] = now_iso()
    return cache


def download_shoko_zip() -> None:
    ANIDB_ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {ANIDB_XML_CACHE_URL}")
    tmp_path = ANIDB_ZIP_PATH.with_suffix(ANIDB_ZIP_PATH.suffix + ".tmp")

    with requests.get(
        ANIDB_XML_CACHE_URL,
        stream=True,
        timeout=120,
        headers={"User-Agent": "Mozilla/5.0"},
    ) as response:
        response.raise_for_status()
        with tmp_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    tmp_path.replace(ANIDB_ZIP_PATH)
    print(f"Downloaded {ANIDB_ZIP_PATH}")


def merge_cache_entry(existing: dict | None, payload: dict, source_downloaded_at: str) -> dict:
    existing = dict(existing or {})
    live_episode_count = (
        existing.get("episode_count")
        if existing.get("source") == "live_http" and existing.get("episode_count")
        else None
    )

    merged = {
        **existing,
        **payload,
        "cached_at": existing.get("cached_at") or payload.get("xml_file_modified_at") or source_downloaded_at,
        "source": existing.get("source") if existing.get("source") == "live_http" else "shoko_xml_cache",
        "source_url": ANIDB_XML_CACHE_URL,
        "source_downloaded_at": source_downloaded_at,
    }

    if live_episode_count:
        merged["episode_count"] = live_episode_count

    return merged


def enrich_cache_from_shoko_zip(delete_zip_after_import: bool = True) -> dict:
    if not ANIDB_ZIP_PATH.exists():
        download_shoko_zip()

    cache = load_cache()
    items = cache.setdefault("items", {})
    source_downloaded_at = now_iso()
    anime_doc_pattern = re.compile(r"AnimeDoc_(\d+)\.xml$", re.IGNORECASE)

    imported = 0
    parse_errors = 0

    try:
        with zipfile.ZipFile(ANIDB_ZIP_PATH) as zf:
            xml_infos = [
                info
                for info in zf.infolist()
                if anime_doc_pattern.search(info.filename)
            ]

            for position, info in enumerate(xml_infos, start=1):
                match = anime_doc_pattern.search(info.filename)
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
                payload["xml_file_modified_at"] = zip_info_datetime(info)

                items[anidb_id] = merge_cache_entry(
                    items.get(anidb_id),
                    payload,
                    source_downloaded_at=source_downloaded_at,
                )
                imported += 1

                if imported % 500 == 0:
                    cache = compact_cache_payload(cache)
                    atomic_write_json(ANIDB_CACHE_FILE, cache)
                    print(
                        f"Enriched {imported}/{len(xml_infos)} XML payloads "
                        f"(parse_errors={parse_errors})",
                        flush=True,
                    )

        cache = compact_cache_payload(cache)
        atomic_write_json(ANIDB_CACHE_FILE, cache)
    finally:
        if delete_zip_after_import and ANIDB_ZIP_PATH.exists():
            ANIDB_ZIP_PATH.unlink()
            print(f"Deleted disposable zip: {ANIDB_ZIP_PATH}")

    print(
        f"AniDB cache enriched: imported={imported}, "
        f"parse_errors={parse_errors}, total_entries={len(items)}"
    )
    return cache


def load_improvements_module():
    module_path = ROOT / "src" / "04_apply_dataset_improvements.py"
    spec = importlib.util.spec_from_file_location("dataset_improvements", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def apply_dataset_improvements(cache_payload: dict) -> dict:
    improve = load_improvements_module()
    df = pd.read_csv(DATASET_CSV)
    improved_df, summary = improve.apply_improvements(df, cache_payload)
    improve.save_dataset(improved_df, csv_path=DATASET_CSV, json_path=DATASET_JSON)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich or compact the AniDB metadata cache.")
    parser.add_argument("--compact-only", action="store_true", help="Compact the current cache without downloading Shoko XML.")
    args = parser.parse_args()

    if args.compact_only:
        cache_payload = compact_cache_payload(load_cache())
        atomic_write_json(ANIDB_CACHE_FILE, cache_payload)
    else:
        cache_payload = enrich_cache_from_shoko_zip()

    summary = apply_dataset_improvements(cache_payload)
    compact_summary = {
        key: value
        for key, value in summary.items()
        if not key.startswith("remaining_")
    }
    print(json.dumps(compact_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
