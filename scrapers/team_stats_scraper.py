# -*- coding: utf-8 -*-
"""
NHL Team Stats Scraper

Pulls per-game team-level stats from the NHL stats REST API.
Uses monthly chunking to stay under the 10k row API limit.

Reports scraped:
  1. team/summary     — Goals, shots, faceoff %, wins/losses
  2. team/percentages — Corsi%, Fenwick%, PDO, 5v5 shooting/save %
  3. team/powerplay   — PP%, PP goals, PP opportunities
  4. team/penaltykill  — PK%, shorthanded goals against
  5. team/penalties   — Penalties taken/drawn, PIM
  6. team/faceoffpercentages — Faceoff % by zone
  7. team/realtime    — Hits, blocks, giveaways, takeaways
  8. team/goalsbyperiod — Goals for/against by period (incl. OT)

Outputs:
  - data/raw/team_summary.csv
  - data/raw/team_percentages.csv
  - data/raw/team_powerplay.csv
  - data/raw/team_penaltykill.csv
  - data/raw/team_penalties.csv
  - data/raw/team_faceoffs.csv
  - data/raw/team_realtime.csv
  - data/raw/team_goalsbyperiod.csv

@author: chazf
"""

import requests
import pandas as pd
import time
import os
from datetime import datetime, timedelta

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://api.nhle.com/stats/rest"
PROJECT_DIR = r"D:\Python\NHL Overtime Model"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
DELAY = 0.5
PAGE_SIZE = 100

# Season date ranges
SEASONS = {
    "20202021": ("2021-01-13", "2021-05-19"),   # COVID-shortened (56 games)
    "20212022": ("2021-10-12", "2022-04-29"),
    "20222023": ("2022-10-07", "2023-04-14"),
    "20232024": ("2023-10-10", "2024-04-19"),
    "20242025": ("2024-10-04", "2025-04-18"),
    "20252026": ("2025-10-07", "2026-04-17"),
}

PROGRESS_FILE = os.path.join(DATA_DIR, "team_stats_progress.txt")

# =============================================================================
# Report Definitions
# =============================================================================
REPORTS = [
    {
        "name": "team_summary",
        "entity": "team",
        "endpoint": "summary",
        "output_file": os.path.join(DATA_DIR, "team_summary.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "goalsFor", "goalsAgainst", "shotsForPerGame", "shotsAgainstPerGame",
            "faceoffWinPct", "wins", "losses", "otLosses", "points",
        ],
    },
    {
        "name": "team_percentages",
        "entity": "team",
        "endpoint": "percentages",
        "output_file": os.path.join(DATA_DIR, "team_percentages.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "satPct", "satPctAhead", "satPctBehind", "satPctClose",
            "usatPct",  # Fenwick
            "shootingPct5v5", "savePct5v5",
            "PDO",  # shooting% + save% at 5v5
            "satForPer60", "satAgainstPer60",
        ],
    },
    {
        "name": "team_powerplay",
        "entity": "team",
        "endpoint": "powerplay",
        "output_file": os.path.join(DATA_DIR, "team_powerplay.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "ppPct", "ppGoalsFor", "ppOpportunities",
            "shNumTimes", "ppTimeOnIce",
        ],
    },
    {
        "name": "team_penaltykill",
        "entity": "team",
        "endpoint": "penaltykill",
        "output_file": os.path.join(DATA_DIR, "team_penaltykill.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "pkPct", "shGoalsAgainst", "shNumTimes",
            "shTimeOnIce",
        ],
    },
    {
        "name": "team_penalties",
        "entity": "team",
        "endpoint": "penalties",
        "output_file": os.path.join(DATA_DIR, "team_penalties.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "penalties", "penaltyMinutes", "minors", "majors",
            "totalPenaltiesDrawn", "benchMinorPenalties",
        ],
    },
    {
        "name": "team_faceoffs",
        "entity": "team",
        "endpoint": "faceoffpercentages",
        "output_file": os.path.join(DATA_DIR, "team_faceoffs.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "faceoffWinPct",
            "zoneFaceoffPctDefensive", "zoneFaceoffPctNeutral",
            "zoneFaceoffPctOffensive",
            "totalFaceoffs",
        ],
    },
    {
        "name": "team_realtime",
        "entity": "team",
        "endpoint": "realtime",
        "output_file": os.path.join(DATA_DIR, "team_realtime.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "hits", "blockedShots", "giveaways", "takeaways",
            "missedShots",
        ],
    },
    {
        "name": "team_goalsbyperiod",
        "entity": "team",
        "endpoint": "goalsbyperiod",
        "output_file": os.path.join(DATA_DIR, "team_goalsbyperiod.csv"),
        "fields": [
            "teamId", "teamFullName", "teamAbbrev", "gameId", "gameDate",
            "opponentTeamAbbrev", "homeRoad",
            "goalsFor1st", "goalsAgainst1st",
            "goalsFor2nd", "goalsAgainst2nd",
            "goalsFor3rd", "goalsAgainst3rd",
            "goalsForOt", "goalsAgainstOt",
        ],
    },
]


# =============================================================================
# Progress Tracking
# =============================================================================

def load_progress():
    """Load set of completed progress keys."""
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
    return completed


def mark_progress(key):
    """Mark a chunk as complete."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{key}\n")


# =============================================================================
# API Functions
# =============================================================================

def generate_monthly_chunks(start_str, end_str):
    """Split a date range into monthly chunks."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    chunks = []
    current = start

    while current <= end:
        if current.month == 12:
            month_end = datetime(current.year + 1, 1, 1) - timedelta(days=1)
        else:
            month_end = datetime(current.year, current.month + 1, 1) - timedelta(days=1)

        chunk_end = min(month_end, end)
        chunks.append((current.strftime("%Y-%m-%d"), chunk_end.strftime("%Y-%m-%d")))
        current = chunk_end + timedelta(days=1)

    return chunks


def fetch_report_chunk(entity, endpoint, season, date_start, date_end):
    """Fetch all pages for a single report + season + date chunk."""
    all_records = []
    start = 0

    while True:
        cayenne = (
            f"seasonId={season} and gameTypeId=2"
            f' and gameDate>="{date_start}"'
            f' and gameDate<="{date_end}"'
        )

        url = (
            f"{BASE_URL}/en/{entity}/{endpoint}"
            f"?isGame=true&limit={PAGE_SIZE}&start={start}"
            f"&cayenneExp={cayenne}"
        )

        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.exceptions.RequestException as e:
            print(f"    !! API error: {e}")
            time.sleep(2)
            try:
                resp = requests.get(url, timeout=30)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e2:
                print(f"    !! Retry failed: {e2}")
                break

        records = data.get("data", [])
        total = data.get("total", 0)

        if not records:
            break

        all_records.extend(records)
        start += PAGE_SIZE
        time.sleep(DELAY)

        if start >= total or len(records) < PAGE_SIZE:
            break

    return all_records


def append_to_csv(records, filepath, fields):
    """Append records to CSV, keeping only specified fields."""
    if not records:
        return 0
    df = pd.DataFrame(records)
    available = [f for f in fields if f in df.columns]
    df = df[available]

    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    return len(df)


# =============================================================================
# Main Scraper
# =============================================================================

def scrape_all():
    """Scrape all team stat reports for all seasons, chunked by month."""
    completed = load_progress()
    os.makedirs(DATA_DIR, exist_ok=True)

    for report in REPORTS:
        report_name = report["name"]
        entity = report["entity"]
        endpoint = report["endpoint"]
        output_file = report["output_file"]
        fields = report["fields"]

        print(f"\n{'='*60}")
        print(f"Report: {report_name} ({entity}/{endpoint})")
        print(f"Output: {output_file}")
        print(f"{'='*60}")

        for season, (season_start, season_end) in SEASONS.items():
            print(f"\n  Season {season}:")
            chunks = generate_monthly_chunks(season_start, season_end)

            for chunk_start, chunk_end in chunks:
                progress_key = f"{report_name}|{season}|{chunk_start}|{chunk_end}"

                if progress_key in completed:
                    print(f"    {chunk_start} to {chunk_end}: SKIP (already done)")
                    continue

                print(f"    {chunk_start} to {chunk_end}: ", end="", flush=True)

                records = fetch_report_chunk(entity, endpoint, season,
                                             chunk_start, chunk_end)
                n_saved = append_to_csv(records, output_file, fields)
                print(f"{n_saved:,} rows")

                mark_progress(progress_key)

        # Summary
        if os.path.exists(output_file):
            df = pd.read_csv(output_file)
            print(f"\n  Total rows in {os.path.basename(output_file)}: {len(df):,}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NHL Team Stats Scraper")
    print("=" * 60)

    scrape_all()

    print(f"\n{'='*60}")
    print("SCRAPING COMPLETE")
    print(f"{'='*60}")

    for report in REPORTS:
        filepath = report["output_file"]
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  {os.path.basename(filepath)}: {len(df):,} rows")
        else:
            print(f"  {os.path.basename(filepath)}: NOT YET CREATED")
