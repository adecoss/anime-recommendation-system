from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
OUT_FILE = APP_DIR / "assets" / "anireco-data.js"

CATALOG = ROOT / "data" / "processed" / "anime_dataset.csv"
ROWS = ROOT / "artifacts" / "recommendation" / "product_recommendation_rows.csv"
SUMMARY = ROOT / "artifacts" / "recommendation" / "product_recommendation_summary.json"
TAGS_JSON = ROOT / "data" / "reference" / "anilist" / "tags.json"
PROFILES = ROOT / "data" / "processed" / "current_user_profile_features.csv"
RATINGS = ROOT / "data" / "processed" / "current_user_ratings.csv"
MYLIST_XML = ROOT / "data" / "raw" / "MyList.xml"
VA_INDEX = ROOT / "data" / "processed" / "voice_actor_index.csv"
STAFF_INDEX = ROOT / "data" / "processed" / "staff_index.csv"
CHARACTER_INDEX = ROOT / "data" / "processed" / "character_index.csv"
STAFF_EDGES = ROOT / "data" / "processed" / "anime_staff_edges.csv"


def clean(value):
    if pd.isna(value):
        return None
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def split_pipe(value) -> list[str]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    return [part.strip() for part in str(value).split("|") if part.strip()]


def row_to_item(row, staff_by_anime: dict[int, list[str]] | None = None) -> dict:
    mal_id = int(row.mal_id)
    tags = sorted(set(split_pipe(getattr(row, "tags", "")) + split_pipe(getattr(row, "explicit_tags", ""))))
    return {
        "mal_id": mal_id,
        "anilist_id": clean(getattr(row, "anilist_id", None)),
        "title": clean(row.title),
        "title_english": clean(getattr(row, "title_english", "")),
        "image_url": clean(getattr(row, "image_url", "")),
        "mal_url": clean(getattr(row, "url", "")),
        "type": clean(getattr(row, "type", "")),
        "source": clean(getattr(row, "source", "")),
        "status": clean(getattr(row, "status", "")),
        "rating": clean(getattr(row, "rating", "")),
        "score": clean(getattr(row, "score", None)),
        "members": clean(getattr(row, "members", None)),
        "favorites": clean(getattr(row, "favorites", None)),
        "episodes": clean(getattr(row, "episodes", None)),
        "duration": clean(getattr(row, "duration", None)),
        "total_watch_minutes": clean(getattr(row, "total_watch_minutes", None)),
        "aired_year": clean(getattr(row, "aired_year", None)),
        "season": clean(getattr(row, "season", "")),
        "genres": split_pipe(getattr(row, "genres", "")),
        "tags": tags,
        "demographics": split_pipe(getattr(row, "demographics", "")),
        "studios": split_pipe(getattr(row, "studios", "")),
        "voice_actors": split_pipe(getattr(row, "voice_actors", ""))[:12],
        "staff": (staff_by_anime or {}).get(mal_id, [])[:12],
        "relations": clean(getattr(row, "relations", "")),
        "recommendations": clean(getattr(row, "recommendations", "")),
    }


def parse_id_list(value) -> list[int]:
    if pd.isna(value) or str(value).strip() == "":
        return []
    ids = []
    for part in str(value).split("|"):
        try:
            ids.append(int(float(part.strip())))
        except ValueError:
            continue
    return ids


def is_plan_to_watch_status(value) -> bool:
    status = str(value or "").strip().lower().replace("_", " ")
    return status in {"plan to watch", "plantowatch", "planning"}


def hydrate_demo_profiles_with_ratings(profiles_by_level: dict[str, dict], catalog_ids: set[int]) -> None:
    user_ids = {
        int(profile["userID"])
        for profile in profiles_by_level.values()
        if profile.get("source") == "current_user_profile_features" and profile.get("userID") is not None
    }
    if not user_ids or not RATINGS.exists():
        return

    ratings_by_user = {user_id: {"known": set(), "scored": []} for user_id in user_ids}
    for chunk in pd.read_csv(RATINGS, usecols=["userID", "animeID", "rating", "status"], chunksize=750_000):
        chunk = chunk[chunk["userID"].isin(user_ids)]
        if chunk.empty:
            continue
        for row in chunk.itertuples(index=False):
            try:
                user_id = int(row.userID)
                mal_id = int(row.animeID)
                score = int(float(row.rating or 0))
            except (TypeError, ValueError):
                continue
            if mal_id not in catalog_ids or is_plan_to_watch_status(row.status):
                continue
            ratings_by_user[user_id]["known"].add(mal_id)
            if score > 0:
                ratings_by_user[user_id]["scored"].append({"mal_id": mal_id, "score": score})

    for profile in profiles_by_level.values():
        user_id = profile.get("userID")
        if user_id is None:
            continue
        bucket = ratings_by_user.get(int(user_id))
        if not bucket or not bucket["known"]:
            continue
        profile["known_ids"] = sorted(bucket["known"])
        profile["scored"] = sorted(bucket["scored"], key=lambda item: (item["mal_id"], item["score"]))


def load_lookup(path: Path, id_cols: list[str], name_col: str, favorite_col: str) -> list[dict]:
    if not path.exists():
        return []
    df = pd.read_csv(path, low_memory=False)
    rows = []
    for row in df.head(5000).itertuples(index=False):
        ids = []
        for col in id_cols:
            if col in df.columns:
                value = getattr(row, col)
                if not pd.isna(value):
                    try:
                        ids.append(int(float(value)))
                    except ValueError:
                        pass
        rows.append(
            {
                "ids": sorted(set(ids)),
                "name": str(getattr(row, name_col)),
                "favorites": int(float(getattr(row, favorite_col, 0) or 0)),
            }
        )
    return rows


def load_staff_by_anime() -> dict[int, list[str]]:
    if not STAFF_EDGES.exists():
        return {}
    usecols = ["mal_id", "staff_name", "staff_role_group", "staff_favorites"]
    df = pd.read_csv(STAFF_EDGES, usecols=lambda col: col in usecols, low_memory=False)
    if df.empty or "mal_id" not in df.columns or "staff_name" not in df.columns:
        return {}
    if "staff_role_group" in df.columns:
        priority = {"director": 3, "original_creator": 2, "original_story": 2}
        df["_role_priority"] = df["staff_role_group"].map(priority).fillna(1)
    else:
        df["_role_priority"] = 1
    if "staff_favorites" not in df.columns:
        df["staff_favorites"] = 0
    df["staff_favorites"] = pd.to_numeric(df["staff_favorites"], errors="coerce").fillna(0)
    df = df.sort_values(["mal_id", "_role_priority", "staff_favorites"], ascending=[True, False, False])
    output: dict[int, list[str]] = {}
    for mal_id, part in df.groupby("mal_id", sort=False):
        names = []
        seen = set()
        for name in part["staff_name"].dropna().astype(str):
            key = name.casefold()
            if key in seen:
                continue
            seen.add(key)
            names.append(name)
            if len(names) >= 12:
                break
        output[int(mal_id)] = names
    return output


def parse_mylist_profile(catalog_ids: set[int]) -> dict | None:
    if not MYLIST_XML.exists():
        return None
    root = ET.parse(MYLIST_XML).getroot()
    known = []
    scored = []
    favorite_candidates = []
    completed_count = 0
    for anime in root.findall("anime"):
        try:
            mal_id = int(anime.findtext("series_animedb_id") or 0)
            score = int(float(anime.findtext("my_score") or 0))
        except ValueError:
            continue
        if mal_id not in catalog_ids:
            continue
        status = (anime.findtext("my_status") or "").strip().lower()
        if status in {"plan to watch", "plantowatch", "plan_to_watch"}:
            continue
        if status == "completed":
            completed_count += 1
        known.append(mal_id)
        if score > 0:
            scored.append({"mal_id": mal_id, "score": score})
        if score >= 10:
            favorite_candidates.append(mal_id)
    return {
        "id": "champux",
        "label": "Champux local XML",
        "source": "local_xml",
        "level": "Veteran",
        "completed_count": completed_count or len(scored),
        "known_ids": known,
        "scored": scored,
        "favorite_anime_ids": favorite_candidates[:24],
        "favorite_voice_actor_ids": [],
    }


def build_demo_profiles(catalog_ids: set[int]) -> list[dict]:
    profiles_by_level: dict[str, dict] = {}
    if PROFILES.exists():
        df = pd.read_csv(PROFILES, low_memory=False)
        df = df[df["completed_count"].notna()].copy()
        level_specs = [
            ("Beginner", 1, 49),
            ("Casual", 50, 149),
            ("Fan", 150, 499),
            ("Veteran", 500, 10**9),
        ]
        for level, low, high in level_specs:
            candidates = df[df["completed_count"].between(low, high)].copy()
            if candidates.empty:
                continue
            candidates["favorite_signal_count"] = candidates.apply(
                lambda row: len(parse_id_list(row.get("favorite_anime_ids", "")))
                + len(parse_id_list(row.get("favorite_voice_actor_ids", ""))),
                axis=1,
            )
            candidates = candidates.sort_values(
                ["favorite_signal_count", "completed_count"],
                ascending=[False, False],
                na_position="last",
            )
            for row in candidates.itertuples(index=False):
                anime_ids = [mal_id for mal_id in parse_id_list(getattr(row, "favorite_anime_ids", "")) if mal_id in catalog_ids]
                va_ids = parse_id_list(getattr(row, "favorite_voice_actor_ids", ""))
                completed = int(getattr(row, "completed_count", 0) or 0)
                profiles_by_level[level] = {
                    "id": f"demo_{level.lower()}_{int(float(row.userID))}",
                    "label": f"{level} demo - collected profile ({completed:,} completed)",
                    "source": "current_user_profile_features",
                    "level": level,
                    "userID": int(float(row.userID)),
                    "completed_count": completed,
                    "mean_score": float(getattr(row, "mean_score", 0) or 0),
                    "known_ids": anime_ids,
                    "scored": [{"mal_id": mal_id, "score": 10} for mal_id in anime_ids],
                    "favorite_anime_ids": anime_ids,
                    "favorite_voice_actor_ids": va_ids,
                }
                break
    hydrate_demo_profiles_with_ratings(profiles_by_level, catalog_ids)
    return [profiles_by_level[level] for level in ["Beginner", "Casual", "Fan", "Veteran"] if level in profiles_by_level]


def main() -> None:
    catalog = pd.read_csv(CATALOG)
    catalog = catalog.sort_values(["members", "score"], ascending=False)
    staff_by_anime = load_staff_by_anime()
    items = [row_to_item(row, staff_by_anime) for row in catalog.itertuples(index=False)]

    product_rows = []
    if ROWS.exists():
        rows = pd.read_csv(ROWS)
        for row_name, part in rows.groupby("row", sort=False):
            product_rows.append(
                {
                    "row": row_name,
                    "description": str(part["row_description"].iloc[0]) if "row_description" in part.columns else "",
                    "items": [
                        {
                            "mal_id": int(item.mal_id),
                            "reason": clean(getattr(item, "reason", "")),
                            "anchor": clean(getattr(item, "anchor", "")),
                            "score": clean(getattr(item, "score", None)),
                        }
                        for item in part.itertuples(index=False)
                    ],
                }
            )

    summary = {}
    if SUMMARY.exists():
        summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

    genres = sorted({label for item in items for label in item["genres"]})
    tags = sorted({label for item in items for label in item["tags"]})
    studios = sorted({label for item in items for label in item["studios"]})
    demographics = sorted({label for item in items for label in item["demographics"]})
    tag_categories: dict[str, list[dict]] = {}
    if TAGS_JSON.exists():
        reference_tags = json.loads(TAGS_JSON.read_text(encoding="utf-8"))
        catalog_tag_set = set(tags)
        categorized_tags = set()
        for tag in reference_tags:
            name = tag.get("name")
            if name not in catalog_tag_set:
                continue
            categorized_tags.add(name)
            category = tag.get("category") or "Other"
            tag_categories.setdefault(category, []).append(
                {
                    "id": tag.get("id"),
                    "name": name,
                    "description": tag.get("description") or "",
                    "isAdult": bool(tag.get("isAdult")),
                }
            )
        uncategorized = sorted(catalog_tag_set - categorized_tags)
        if uncategorized:
            tag_categories["Other catalog tags"] = [
                {"id": None, "name": name, "description": "Catalog tag without an AniList reference category.", "isAdult": False}
                for name in uncategorized
            ]
        tag_categories = {key: sorted(value, key=lambda item: item["name"]) for key, value in sorted(tag_categories.items())}

    payload = {
        "generated_from": {
            "catalog": "data/processed/anime_dataset.csv",
            "rows": "artifacts/recommendation/product_recommendation_rows.csv",
        },
        "catalog": items,
        "product_rows": product_rows,
        "summary": summary,
        "facets": {
            "genres": genres,
            "tags": tags,
            "tag_categories": tag_categories,
            "demographics": demographics,
            "studios": studios[:400],
            "types": sorted({str(item["type"]) for item in items if item["type"]}),
            "ratings": sorted({str(item["rating"]) for item in items if item["rating"]}),
            "seasons": sorted({str(item["season"]) for item in items if item["season"]}),
        },
        "lookups": {
            "voice_actors": load_lookup(VA_INDEX, ["voice_actor_id_mal", "voice_actor_id_anilist"], "voice_actor_name", "voice_actor_favorites"),
            "staff": load_lookup(STAFF_INDEX, ["staff_id_anilist"], "staff_name", "staff_favorites"),
            "characters": load_lookup(CHARACTER_INDEX, ["character_id_mal", "character_id_anilist"], "character_name", "character_favorites"),
        },
        "demo_profiles": build_demo_profiles({int(item["mal_id"]) for item in items}),
    }
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(
        "window.ANIRECO_DATA = " + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUT_FILE.relative_to(ROOT)} with {len(items):,} catalog items.")


if __name__ == "__main__":
    main()
