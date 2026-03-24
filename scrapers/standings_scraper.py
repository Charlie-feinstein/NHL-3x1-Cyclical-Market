# -*- coding: utf-8 -*-
"""
NHL Standings Scraper

Pulls standings snapshots from the NHL API for context features:
points%, home/road records, L10, streaks, goal differential.

Strategy:
  - Historical seasons: sample every 7 days (standings change slowly)
  - Current season: every 7 days (can re-run for daily if needed)

Output:
  - data/raw/standings_daily.csv

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
BASE_URL = "https://api-web.nhle.com"
PROJECT_DIR = r"D:\Python\NHL Overtime Model"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
DELAY = 0.5

# Season date ranges
SEASONS = {
    "20202021": ("2021-01-18", "2021-05-19"),   # COVID-shortened (56 games)
    "20212022": ("2021-10-17", "2022-04-29"),
    "20222023": ("2022-10-12", "2023-04-14"),
    "20232024": ("2023-10-15", "2024-04-19"),
    "20242025": ("2024-10-08", "2025-04-18"),
    "20252026": ("2025-10-12", "2026-04-17"),
}

# How often to sample standings (days)
SAMPLE_INTERVAL = 7

OUTPUT_FILE = os.path.join(DATA_DIR, "standings_daily.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "standings_progress.txt")


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
# API
# =============================================================================

def api_get(endpoint):
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                time.sleep(2)
            else:
                print(f"  !! FAILED: {e}")
                return None


def parse_standings(data, date_str):
    """Parse standings response into flat rows (one per team)."""
    rows = []
    if not data or "standings" not in data:
        return rows

    for team in data["standings"]:
        row = {
            "date": date_str,
            "team_abbrev": team.get("teamAbbrev", {}).get("default", ""),
            "team_name": team.get("teamName", {}).get("default", ""),
            "conference": team.get("conferenceName", ""),
            "division": team.get("divisionName", ""),
            "games_played": team.get("gamesPlayed", 0),
            "wins": team.get("wins", 0),
            "losses": team.get("losses", 0),
            "ot_losses": team.get("otLosses", 0),
            "points": team.get("points", 0),
            "point_pct": team.get("pointPctg"),
            "reg_wins": team.get("regulationWins", 0),
            "reg_plus_ot_wins": team.get("regulationPlusOtWins", 0),
            "goals_for": team.get("goalFor", 0),
            "goals_against": team.get("goalAgainst", 0),
            "goal_diff": team.get("goalDifferential", 0),
            "home_wins": team.get("homeWins", 0),
            "home_losses": team.get("homeLosses", 0),
            "home_ot_losses": team.get("homeOtLosses", 0),
            "road_wins": team.get("roadWins", 0),
            "road_losses": team.get("roadLosses", 0),
            "road_ot_losses": team.get("roadOtLosses", 0),
            "l10_wins": team.get("l10Wins", 0),
            "l10_losses": team.get("l10Losses", 0),
            "l10_ot_losses": team.get("l10OtLosses", 0),
            "streak_code": team.get("streakCode", ""),
            "streak_count": team.get("streakCount", 0),
            "wildcard_sequence": team.get("wildcardSequence"),
        }
        rows.append(row)

    return rows


# =============================================================================
# Main
# =============================================================================

def scrape_standings():
    completed = load_progress()
    os.makedirs(DATA_DIR, exist_ok=True)

    today = datetime.now()

    for season, (start_str, end_str) in SEASONS.items():
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = min(datetime.strptime(end_str, "%Y-%m-%d"), today)
        current = start_date

        print(f"\nSeason {season}:")

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")

            if date_str in completed:
                current += timedelta(days=SAMPLE_INTERVAL)
                continue

            data = api_get(f"/v1/standings/{date_str}")
            rows = parse_standings(data, date_str)

            if rows:
                df = pd.DataFrame(rows)
                if os.path.exists(OUTPUT_FILE):
                    df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
                else:
                    df.to_csv(OUTPUT_FILE, index=False)
                print(f"  {date_str}: {len(rows)} teams")
            else:
                print(f"  {date_str}: no data")

            mark_progress(date_str)
            time.sleep(DELAY)
            current += timedelta(days=SAMPLE_INTERVAL)

    if os.path.exists(OUTPUT_FILE):
        df = pd.read_csv(OUTPUT_FILE)
        n_dates = df["date"].nunique()
        print(f"\nTotal: {len(df):,} rows across {n_dates} dates")


if __name__ == "__main__":
    print("=" * 60)
    print("NHL Standings Scraper")
    print("=" * 60)

    scrape_standings()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
