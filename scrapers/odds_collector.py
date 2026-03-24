# -*- coding: utf-8 -*-
"""
NHL DraftKings Odds Collector

Collects moneyline odds from the DraftKings partner API.
This endpoint only provides current/upcoming game odds (not historical),
so run this daily to build a historical odds dataset over time.

Recommended: Schedule via Windows Task Scheduler to run once per day
around 10am ET (after lines are posted, before games start).

Output:
  - data/raw/odds_snapshots.csv (appended each run)

@author: chazf
"""

import requests
import pandas as pd
import os
from datetime import datetime

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://api-web.nhle.com"
PROJECT_DIR = r"D:\Python\NHL Overtime Model"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")
OUTPUT_FILE = os.path.join(DATA_DIR, "odds_snapshots.csv")


# =============================================================================
# API
# =============================================================================

def fetch_odds():
    """
    Fetch current DraftKings odds from the NHL partner endpoint.
    Returns list of dicts with game odds info.
    """
    url = f"{BASE_URL}/v1/partner-game/US/now"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except requests.exceptions.RequestException as e:
        print(f"!! API error: {e}")
        return []

    snapshot_time = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    # The partner endpoint returns games with odds
    for game in data:
        # Extract game info
        game_id = game.get("id")
        game_date = game.get("gameDate", "")
        away = game.get("awayTeam", {})
        home = game.get("homeTeam", {})

        # Extract odds from the odds array
        odds_list = game.get("odds", [])
        for odds_entry in odds_list:
            provider = odds_entry.get("providerId", "")

            # Look for moneyline odds
            home_odds = odds_entry.get("homeTeamOdds", {})
            away_odds = odds_entry.get("awayTeamOdds", {})

            row = {
                "snapshot_time": snapshot_time,
                "game_id": game_id,
                "game_date": game_date,
                "away_team": away.get("abbrev", ""),
                "home_team": home.get("abbrev", ""),
                "provider_id": provider,
                "away_ml_odds": away_odds.get("moneyLine"),
                "home_ml_odds": home_odds.get("moneyLine"),
            }
            rows.append(row)

    return rows


def save_odds(rows):
    """Append odds rows to the output CSV."""
    if not rows:
        print("No odds data to save.")
        return

    os.makedirs(DATA_DIR, exist_ok=True)
    df = pd.DataFrame(rows)

    if os.path.exists(OUTPUT_FILE):
        df.to_csv(OUTPUT_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(OUTPUT_FILE, index=False)

    print(f"Saved {len(rows)} odds entries to {OUTPUT_FILE}")


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print(f"NHL Odds Collector — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 60)

    rows = fetch_odds()
    save_odds(rows)

    if rows:
        df = pd.DataFrame(rows)
        print(f"\nGames with odds today: {df['game_id'].nunique()}")
        for _, r in df.drop_duplicates("game_id").iterrows():
            print(f"  {r['away_team']} @ {r['home_team']} | "
                  f"Away ML: {r['away_ml_odds']} | Home ML: {r['home_ml_odds']}")

    print("\nDone.")
