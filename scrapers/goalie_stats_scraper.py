# -*- coding: utf-8 -*-
"""
NHL Goalie Stats Scraper

Pulls per-game goalie stats from the NHL stats REST API.
Complements the goalie_boxscores.csv from pbp_boxscore_scraper.py
with additional fields available from the stats endpoint.

Output:
  - data/raw/goalie_summary.csv

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

SEASONS = {
    "20202021": ("2021-01-13", "2021-05-19"),   # COVID-shortened (56 games)
    "20212022": ("2021-10-12", "2022-04-29"),
    "20222023": ("2022-10-07", "2023-04-14"),
    "20232024": ("2023-10-10", "2024-04-19"),
    "20242025": ("2024-10-04", "2025-04-18"),
    "20252026": ("2025-10-07", "2026-04-17"),
}

PROGRESS_FILE = os.path.join(DATA_DIR, "goalie_stats_progress.txt")
OUTPUT_FILE = os.path.join(DATA_DIR, "goalie_summary.csv")

FIELDS = [
    "playerId", "goalieFullName", "gameId", "gameDate",
    "teamAbbrev", "opponentTeamAbbrev", "homeRoad",
    "gamesPlayed", "gamesStarted", "wins", "losses", "otLosses",
    "savePct", "goalsAgainst", "shotsAgainst",
    "goalsAgainstAverage", "timeOnIce",
    "shutouts",
]


# =============================================================================
# Progress Tracking
# =============================================================================

def load_progress():
    completed = set()
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    completed.add(line)
    return completed


def mark_progress(key):
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{key}\n")


# =============================================================================
# API Functions
# =============================================================================

def generate_monthly_chunks(start_str, end_str):
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


def fetch_chunk(season, date_start, date_end):
    all_records = []
    start = 0

    while True:
        cayenne = (
            f"seasonId={season} and gameTypeId=2"
            f' and gameDate>="{date_start}"'
            f' and gameDate<="{date_end}"'
        )

        url = (
            f"{BASE_URL}/en/goalie/summary"
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


def append_to_csv(records, filepath):
    if not records:
        return 0
    df = pd.DataFrame(records)
    available = [f for f in FIELDS if f in df.columns]
    df = df[available]

    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    return len(df)


# =============================================================================
# Main
# =============================================================================

def scrape_all():
    completed = load_progress()
    os.makedirs(DATA_DIR, exist_ok=True)

    print(f"\nReport: goalie/summary")
    print(f"Output: {OUTPUT_FILE}\n")

    for season, (season_start, season_end) in SEASONS.items():
        print(f"  Season {season}:")
        chunks = generate_monthly_chunks(season_start, season_end)

        for chunk_start, chunk_end in chunks:
            progress_key = f"goalie_summary|{season}|{chunk_start}|{chunk_end}"

            if progress_key in completed:
                print(f"    {chunk_start} to {chunk_end}: SKIP (already done)")
                continue

            print(f"    {chunk_start} to {chunk_end}: ", end="", flush=True)

            records = fetch_chunk(season, chunk_start, chunk_end)
            n_saved = append_to_csv(records, OUTPUT_FILE)
            print(f"{n_saved:,} rows")

            mark_progress(progress_key)

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        print(f"\n  Total rows: {len(df):,}")


if __name__ == "__main__":
    print("=" * 60)
    print("NHL Goalie Stats Scraper")
    print("=" * 60)

    scrape_all()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
