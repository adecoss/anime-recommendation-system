from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
BUILD_DIR = ROOT / "data" / "build"
PROCESSED_DIR = ROOT / "data" / "processed"

RAW_SCRIPT = SRC_DIR / "01_gather_raw_sources.py"
BUILD_SCRIPT = SRC_DIR / "02_build_anime_dataset.py"
IMPROVE_SCRIPT = SRC_DIR / "04_improve_anime_dataset.py"

SEASONAL_CANDIDATES_CSV = BUILD_DIR / "seasonal_refresh_candidates.csv"
SEASONAL_DISCOVERY_CSV = BUILD_DIR / "seasonal_discovered_jikan_ids.csv"
SEASONAL_MAL_DISCOVERY_CSV = BUILD_DIR / "seasonal_discovered_mal_html_ids.csv"
SEASONAL_RUN_SUMMARY = BUILD_DIR / "seasonal_refresh_run_summary.json"
DATASET_CSV = PROCESSED_DIR / "anime_dataset.csv"


def now_iso() -> str:
    return datetime.now().isoformat(timespec="seconds")


def project_path(path: Path | str) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except Exception:
        return str(path)


def display_command(command: list[str]) -> list[str]:
    if not command:
        return []
    rendered = ["python" if Path(str(command[0])).name.lower().startswith("python") else str(command[0])]
    if len(command) > 1:
        rendered.append(project_path(command[1]))
    rendered.extend(str(part) for part in command[2:])
    return rendered


def parse_ids(values: list[str] | None) -> list[int]:
    ids: list[int] = []
    for value in values or []:
        for part in str(value).split(","):
            text = part.strip()
            if text.isdigit():
                ids.append(int(text))
    return sorted(set(ids))


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f".{time.time_ns()}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


def run_stream(command: list[str], *, label: str, dry_run: bool = False) -> dict[str, Any]:
    printable = display_command(command)
    print(f"\n=== {label} ===", flush=True)
    print("Running:", " ".join(str(part) for part in printable if part), flush=True)
    started = time.perf_counter()
    if dry_run:
        return {
            "label": label,
            "command": printable,
            "return_code": 0,
            "elapsed_seconds": 0.0,
            "dry_run": True,
        }

    process = subprocess.Popen(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    return_code = process.wait()
    elapsed = time.perf_counter() - started
    print(f"{label} finished in {elapsed:.1f}s with code {return_code}", flush=True)
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)
    return {
        "label": label,
        "command": printable,
        "return_code": int(return_code),
        "elapsed_seconds": round(float(elapsed), 3),
        "dry_run": False,
    }


def load_dataset_status(ids: list[int]) -> list[dict[str, Any]]:
    if not DATASET_CSV.exists() or not ids:
        return []
    df = pd.read_csv(DATASET_CSV)
    if "mal_id" not in df.columns:
        return []
    subset = df[df["mal_id"].astype(str).isin({str(item) for item in ids})].copy()
    columns = [
        col
        for col in [
            "mal_id",
            "anilist_id",
            "anidb_id",
            "title",
            "type",
            "status",
            "score",
            "episodes",
            "duration",
            "season",
            "aired_year",
            "members",
        ]
        if col in subset.columns
    ]
    return subset[columns].to_dict(orient="records")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "One-command seasonal refresh: update Jikan/AniList seasonal caches, "
            "optionally refresh explicit ids, rebuild the dataset, and rebuild VA/staff tables."
        )
    )
    parser.add_argument("--date", default=None, help="Reference date for seasonal selection, YYYY-MM-DD. Defaults to today.")
    parser.add_argument("--max-age-hours", type=float, default=6.0, help="Refresh cached seasonal rows older than this many hours.")
    parser.add_argument("--include-previous-season", action=argparse.BooleanOptionalAction, default=True, help="Also refresh the previous two seasons alongside the current season.")
    parser.add_argument("--ids", nargs="*", default=None, help="Force refresh specific MAL ids, e.g. --ids 59193 or --ids 59193,60000.")
    parser.add_argument("--limit", type=int, default=None, help="Limit seasonal live calls for testing.")
    parser.add_argument("--jikan-sleep", type=float, default=1.0)
    parser.add_argument("--anilist-sleep", type=float, default=2.2)
    parser.add_argument("--refresh-characters", action=argparse.BooleanOptionalAction, default=True, help="Refresh Jikan character/VA payloads for seasonal/targeted ids.")
    parser.add_argument("--refresh-recommendations", action=argparse.BooleanOptionalAction, default=False, help="Also refetch Jikan recommendation edges. Disabled by default because the endpoint is slow/504-prone.")
    parser.add_argument("--run-anidb-recent", action=argparse.BooleanOptionalAction, default=False, help="Also refresh recent AniDB ids after Jikan/AniList.")
    parser.add_argument("--anidb-since", default="2025-09-14")
    parser.add_argument("--anidb-limit", type=int, default=None)
    parser.add_argument("--anidb-sleep", type=float, default=4.0)
    parser.add_argument("--skip-seasonal", action="store_true", help="Only run targeted ids / rebuild steps.")
    parser.add_argument("--skip-build", action="store_true", help="Skip rebuilding anime_dataset.csv/json.")
    parser.add_argument("--skip-improvements", action="store_true", help="Skip dataset improvements and VA/staff table rebuild.")
    parser.add_argument("--skip-va-staff", action="store_true", help="Run dataset improvements but skip VA/character/staff enrichment table rebuilds.")
    parser.add_argument("--drop-auxiliary-columns", action="store_true", help="Pass --drop-auxiliary-columns to the improvement script.")
    parser.add_argument("--refresh-top-favorites", action="store_true", help="Refresh Jikan/AniList top character/person favorite caches before rebuilding people tables. Slow.")
    parser.add_argument("--jikan-top-character-pages", type=int, default=1000)
    parser.add_argument("--jikan-top-people-pages", type=int, default=1000)
    parser.add_argument("--anilist-top-pages", type=int, default=100)
    parser.add_argument("--anilist-top-per-page", type=int, default=50)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    forced_ids = parse_ids(args.ids)
    steps: list[dict[str, Any]] = []
    started_at = time.perf_counter()

    if not args.skip_seasonal:
        cmd = [
            sys.executable,
            str(RAW_SCRIPT),
            "--seasonal-refresh",
            "--seasonal-refresh-max-age-hours",
            str(args.max_age_hours),
            "--skip-mal-id-download",
            "--jikan-sleep",
            str(args.jikan_sleep),
            "--anilist-sleep",
            str(args.anilist_sleep),
        ]
        if args.include_previous_season:
            cmd.append("--include-previous-season")
        if args.date:
            cmd.extend(["--seasonal-refresh-date", args.date])
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        if args.refresh_characters:
            cmd.append("--refresh-characters")
        if not args.refresh_recommendations:
            cmd.append("--skip-recommendations")
        else:
            cmd.append("--refresh-seasonal-recommendations")
        steps.append(run_stream(cmd, label="Seasonal Jikan/AniList refresh", dry_run=args.dry_run))

    if forced_ids:
        id_args = [str(item) for item in forced_ids]
        cmd = [
            sys.executable,
            str(RAW_SCRIPT),
            "--jikan",
            "--ids",
            *id_args,
            "--refresh-ids",
            *id_args,
            "--skip-mal-id-download",
            "--jikan-sleep",
            str(args.jikan_sleep),
            "--anilist-sleep",
            str(args.anilist_sleep),
        ]
        if not args.refresh_characters:
            cmd.append("--skip-characters")
        if not args.refresh_recommendations:
            cmd.append("--skip-recommendations")
        steps.append(run_stream(cmd, label=f"Targeted MAL id refresh ({','.join(id_args)})", dry_run=args.dry_run))

    if args.run_anidb_recent:
        cmd = [
            sys.executable,
            str(RAW_SCRIPT),
            "--anidb-live-recent",
            "--anidb-since",
            args.anidb_since,
            "--anidb-sleep",
            str(args.anidb_sleep),
        ]
        if args.anidb_limit is not None:
            cmd.extend(["--limit", str(args.anidb_limit)])
        steps.append(run_stream(cmd, label="Recent AniDB live refresh", dry_run=args.dry_run))

    if not args.skip_build:
        steps.append(run_stream([sys.executable, str(BUILD_SCRIPT)], label="Rebuild anime dataset", dry_run=args.dry_run))

    if not args.skip_improvements:
        cmd = [sys.executable, str(IMPROVE_SCRIPT)]
        if not args.skip_va_staff:
            cmd.extend(
                [
                    "--enrich-character-favorites",
                    "--build-va-character-tables",
                    "--max-va-character-edges-per-anime",
                    "30",
                    "--max-dynamic-va-character-edges-per-anime",
                    "150",
                    "--build-staff-tables",
                ]
            )
        if args.drop_auxiliary_columns:
            cmd.append("--drop-auxiliary-columns")
        if args.refresh_top_favorites:
            cmd.extend(
                [
                    "--refresh-jikan-top-favorites",
                    "--jikan-top-character-pages",
                    str(args.jikan_top_character_pages),
                    "--jikan-top-people-pages",
                    str(args.jikan_top_people_pages),
                    "--refresh-anilist-top-favorites",
                    "--anilist-top-pages",
                    str(args.anilist_top_pages),
                    "--anilist-top-per-page",
                    str(args.anilist_top_per_page),
                ]
            )
        steps.append(run_stream(cmd, label="Apply dataset improvements and rebuild people tables", dry_run=args.dry_run))

    seasonal_rows = 0
    discovered_rows = 0
    mal_discovered_rows = 0
    if SEASONAL_CANDIDATES_CSV.exists():
        try:
            seasonal_rows = int(len(pd.read_csv(SEASONAL_CANDIDATES_CSV)))
        except Exception:
            seasonal_rows = 0
    if SEASONAL_DISCOVERY_CSV.exists():
        try:
            discovered_rows = int(len(pd.read_csv(SEASONAL_DISCOVERY_CSV)))
        except Exception:
            discovered_rows = 0
    if SEASONAL_MAL_DISCOVERY_CSV.exists():
        try:
            mal_discovered_rows = int(len(pd.read_csv(SEASONAL_MAL_DISCOVERY_CSV)))
        except Exception:
            mal_discovered_rows = 0

    summary = {
        "updated_at": now_iso(),
        "reference_date": args.date or datetime.now().date().isoformat(),
        "max_age_hours": args.max_age_hours,
        "include_previous_season": bool(args.include_previous_season),
        "forced_mal_ids": forced_ids,
        "seasonal_discovered_rows": discovered_rows,
        "seasonal_mal_html_discovered_rows": mal_discovered_rows,
        "seasonal_candidate_rows": seasonal_rows,
        "refresh_characters": bool(args.refresh_characters),
        "run_anidb_recent": bool(args.run_anidb_recent),
        "dataset_csv": project_path(DATASET_CSV),
        "seasonal_candidates_csv": project_path(SEASONAL_CANDIDATES_CSV),
        "seasonal_discovery_csv": project_path(SEASONAL_DISCOVERY_CSV),
        "seasonal_mal_html_discovery_csv": project_path(SEASONAL_MAL_DISCOVERY_CSV),
        "forced_id_dataset_rows": load_dataset_status(forced_ids),
        "elapsed_seconds": round(float(time.perf_counter() - started_at), 3),
        "steps": steps,
    }
    atomic_write_json(SEASONAL_RUN_SUMMARY, summary)
    print("\nSeasonal refresh pipeline complete.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
