# -*- coding: utf-8 -*-
"""
NHL Game ID Scraper - Schedule Data with Outcome Types

Pulls all regular season game IDs from the NHL schedule API,
including final scores and outcome types (REG/OT/SO) for
the Overtime Edge Model.

Output:
  - data/raw/game_ids.csv

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
OUTPUT_DIR = os.path.join(PROJECT_DIR, "data", "raw")

# Season date ranges (regular season start -> approximate end)
# End dates extended slightly to ensure we capture all games
SEASONS = {
    "20202021": ("2021-01-13", "2021-05-19"),   # COVID-shortened (56 games)
    "20212022": ("2021-10-12", "2022-04-29"),
    "20222023": ("2022-10-07", "2023-04-14"),
    "20232024": ("2023-10-10", "2024-04-19"),
    "20242025": ("2024-10-04", "2025-04-18"),
    "20252026": ("2025-10-07", "2026-04-17"),
}

# Request pacing
DELAY = 0.5

# Output file
GAME_IDS_FILE = os.path.join(OUTPUT_DIR, "game_ids.csv")

# =============================================================================
# Helper Functions
# =============================================================================

def api_get(endpoint):
    """Make a GET request to the NHL API with error handling and retries."""
    url = f"{BASE_URL}{endpoint}"
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt < 2:
                print(f"  !! Retry {attempt+1} for {endpoint}: {e}")
                time.sleep(2)
            else:
                print(f"  !! FAILED after 3 attempts for {endpoint}: {e}")
                return None


def parse_game(game, date_str):
    """
    Extract relevant fields from a schedule game entry.

    Returns a dict with game metadata, teams, scores, and outcome type,
    or None if the game is not a completed regular season game.
    """
    # Only regular season (gameType=2)
    if game.get("gameType") != 2:
        return None

    game_state = game.get("gameState", "")
    away = game.get("awayTeam", {})
    home = game.get("homeTeam", {})
    period_desc = game.get("periodDescriptor", {})
    outcome = game.get("gameOutcome", {})

    return {
        "game_id": game["id"],
        "season": game.get("season", ""),
        "game_date": date_str,
        "away_team_id": away.get("id"),
        "away_team": away.get("abbrev", ""),
        "home_team_id": home.get("id"),
        "home_team": home.get("abbrev", ""),
        "away_score": away.get("score"),
        "home_score": home.get("score"),
        "game_state": game_state,
        "venue": game.get("venue", {}).get("default", ""),
        "period_count": period_desc.get("number"),
        "last_period_type": period_desc.get("periodType", ""),
        "game_outcome_type": outcome.get("lastPeriodType", ""),
    }


# =============================================================================
# Main Scraper
# =============================================================================

def scrape_game_ids(force_refresh=False):
    """
    Iterate through each week in each season and collect all
    regular season game IDs from the schedule endpoint.

    The schedule endpoint returns a full gameWeek (7 days) per call,
    so we jump by 7 days to avoid redundant requests.

    Args:
        force_refresh: If True, re-scrape even if file exists.

    Returns:
        DataFrame with all game IDs and metadata.
    """
    if os.path.exists(GAME_IDS_FILE) and not force_refresh:
        print(f"Game IDs file already exists: {GAME_IDS_FILE}")
        existing = pd.read_csv(GAME_IDS_FILE)
        print(f"  {len(existing)} games loaded from cache.")
        print(f"  Outcome breakdown:")
        if "game_outcome_type" in existing.columns:
            for otype, count in existing["game_outcome_type"].value_counts().items():
                print(f"    {otype}: {count}")
        return existing

    all_games = []
    seen_ids = set()

    for season, (start_str, end_str) in SEASONS.items():
        start_date = datetime.strptime(start_str, "%Y-%m-%d")
        end_date = datetime.strptime(end_str, "%Y-%m-%d")
        current = start_date

        print(f"\nCollecting game IDs for {season}...")
        season_count = 0

        while current <= end_date:
            date_str = current.strftime("%Y-%m-%d")
            data = api_get(f"/v1/schedule/{date_str}")

            if data and "gameWeek" in data:
                for week in data["gameWeek"]:
                    week_date = week.get("date", date_str)
                    for game in week.get("games", []):
                        parsed = parse_game(game, week_date)
                        if parsed and parsed["game_id"] not in seen_ids:
                            all_games.append(parsed)
                            seen_ids.add(parsed["game_id"])
                            season_count += 1

                # Jump by 7 days (schedule returns a full week)
                current += timedelta(days=7)
            else:
                # Fallback: advance by 1 day if response is unexpected
                current += timedelta(days=1)

            time.sleep(DELAY)

        print(f"  -> {season_count} games found for {season}")

    # Build DataFrame, sort by date
    games_df = pd.DataFrame(all_games)
    games_df = games_df.sort_values(["game_date", "game_id"]).reset_index(drop=True)

    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    games_df.to_csv(GAME_IDS_FILE, index=False)

    # Summary
    print(f"\nTotal: {len(games_df)} unique regular season games saved to {GAME_IDS_FILE}")

    completed = games_df[games_df["game_state"].isin(["FINAL", "OFF"])]
    print(f"  Completed games: {len(completed)}")

    if "game_outcome_type" in games_df.columns:
        print(f"  Outcome breakdown (completed games):")
        for otype, count in completed["game_outcome_type"].value_counts().items():
            pct = 100 * count / len(completed)
            print(f"    {otype}: {count} ({pct:.1f}%)")

    return games_df


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NHL Game ID Scraper")
    print("=" * 60)

    games_df = scrape_game_ids()

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)
