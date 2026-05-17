from __future__ import annotations

import json
import random
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def parse_int(value: Any, default: int | None = None) -> int | None:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return int(float(text))
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        text = str(value).strip()
        if text == "":
            return default
        return float(text)
    except (TypeError, ValueError):
        return default


def compact_text(value: str | None) -> str | None:
    if value is None:
        return None
    text = re.sub(r"\s+", " ", value).strip()
    return text or None


def scalar_or_text(value: str | None) -> int | float | str | None:
    text = compact_text(value)
    if text is None:
        return None
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    if re.fullmatch(r"-?\d+\.\d+", text):
        return float(text)
    return text


def extract_episode_summary(root: ET.Element) -> dict[str, Any]:
    episode_nodes = root.findall(".//episodes/episode")
    lengths = []
    regular_lengths = []
    regular_records = []

    for episode in episode_nodes:
        epno_node = episode.find("epno")
        episode_type = (
            episode.findtext("type")
            or episode.get("type")
            or (epno_node.get("type") if epno_node is not None else None)
            or "unknown"
        )
        episode_type = compact_text(episode_type) or "unknown"

        titles = [
            compact_text(title_node.text) or ""
            for title_node in episode.findall("title")
        ]
        length = parse_float(episode.findtext("length"), default=None)
        record = {
            "type": episode_type,
            "epno": compact_text(epno_node.text if epno_node is not None else None),
            "length": length,
            "titles": titles,
        }

        if length is not None and length > 0:
            lengths.append(length)
            if episode_type == "1":
                regular_lengths.append(length)
        if episode_type == "1":
            regular_records.append(record)

    part_records = [
        record
        for record in regular_records
        if record.get("length")
        and any(re.search(r"\bpart\s+\d+\s+of\s+\d+\b", title, re.IGNORECASE) for title in record.get("titles", []))
    ]
    compilation_records = [
        record
        for record in regular_records
        if record.get("length")
        and any(
            re.search(r"\b(complete|full)\s+(movie|version|edition)\b", title, re.IGNORECASE)
            or re.search(r"\b(movie|complete)\b", title, re.IGNORECASE)
            for title in record.get("titles", [])
        )
    ]

    preferred_records = regular_records
    preferred_basis = "regular_type_1"

    if len(part_records) >= 2:
        preferred_records = part_records
        preferred_basis = "part_episodes"
    elif compilation_records and len(regular_records) > 1:
        compilation_ids = {id(record) for record in compilation_records}
        non_compilation = [
            record for record in regular_records
            if id(record) not in compilation_ids and record.get("length")
        ]
        if non_compilation:
            preferred_records = non_compilation
            preferred_basis = "regular_type_1_without_compilation"

    preferred_lengths = [
        record["length"]
        for record in preferred_records
        if record.get("length") is not None and record["length"] > 0
    ]

    summary: dict[str, Any] = {
        "listed_episode_count": len(episode_nodes),
        "length_count": len(lengths),
        "regular_episode_count": len(regular_records),
        "preferred_episode_count": len(preferred_records),
        "preferred_length_count": len(preferred_lengths),
        "preferred_basis": preferred_basis,
        "compilation_episode_count": len(compilation_records),
        "part_episode_count": len(part_records),
    }

    if lengths:
        total = sum(lengths)
        if not preferred_lengths:
            preferred_lengths = regular_lengths or lengths
        preferred_total = sum(preferred_lengths)
        summary.update(
            {
                "average_length_minutes": round(preferred_total / len(preferred_lengths), 3),
                "total_length_minutes": round(total, 3),
                "preferred_average_length_minutes": round(preferred_total / len(preferred_lengths), 3),
                "preferred_total_length_minutes": round(preferred_total, 3),
                "regular_length_count": len(regular_lengths),
                "regular_average_length_minutes": (
                    round(sum(regular_lengths) / len(regular_lengths), 3)
                    if regular_lengths
                    else None
                ),
                "regular_total_length_minutes": (
                    round(sum(regular_lengths), 3)
                    if regular_lengths
                    else None
                ),
            }
        )

    return summary


def extract_raw_tags(root: ET.Element) -> list[dict[str, Any]]:
    raw_tags = []
    for tag_node in root.findall(".//tags/tag"):
        name_node = tag_node.find("name")
        tag_id = parse_int(tag_node.get("id"), default=None)
        if name_node is None or not name_node.text or tag_id is None:
            continue
        raw_tags.append(
            {
                "id": tag_id,
                "parent_id": parse_int(tag_node.get("parentid"), default=None),
                "name": name_node.text.strip(),
                "weight": parse_int(tag_node.get("weight"), default=0) or 0,
            }
        )
    return raw_tags


def extract_animation_work_creators(root: ET.Element) -> list[dict[str, Any]]:
    creators = []
    for name_node in root.findall(".//creators/name"):
        if name_node.get("type") != "Animation Work":
            continue
        name = compact_text(name_node.text)
        if not name:
            continue
        creators.append(
            {
                "id": parse_int(name_node.get("id"), default=None),
                "type": name_node.get("type"),
                "name": name,
            }
        )
    return creators


def extract_similar_anime(root: ET.Element) -> list[dict[str, Any]]:
    similar = []
    for node in root.findall(".//similaranime/anime"):
        anidb_id = parse_int(node.get("id") or node.get("aid"), default=None)
        if anidb_id is None:
            continue
        item = {"anidb_id": anidb_id}
        approval = parse_int(node.get("approval"), default=None)
        total = parse_int(node.get("total"), default=None)
        if approval is not None:
            item["approval"] = approval
        if total is not None:
            item["total"] = total
        similar.append(item)
    return similar


def extract_anidb_payload(root: ET.Element) -> dict[str, Any]:
    episode_count = parse_int(root.findtext("episodecount"), default=None)
    animation_work_creators = extract_animation_work_creators(root)

    payload = {
        "episode_count": episode_count,
        "episode_summary": extract_episode_summary(root),
        "raw_tags": extract_raw_tags(root),
        "animation_work_studios": [
            item["name"]
            for item in animation_work_creators
            if item.get("name")
        ],
        "similar_anime": extract_similar_anime(root),
    }

    return payload


def atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(
        f"{path.name}.{int(time.time() * 1000)}.{random.randint(1000, 9999)}.tmp"
    )
    tmp_path.write_text(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    json.loads(tmp_path.read_text(encoding="utf-8"))
    tmp_path.replace(path)
