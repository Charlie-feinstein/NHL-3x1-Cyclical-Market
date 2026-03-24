# -*- coding: utf-8 -*-
"""
NHL PBP & Boxscore Scraper

For each completed game in game_ids.csv, pulls:
  1. Play-by-play data — shot/goal events with x/y coordinates (for xG model),
     plus faceoff and penalty events (for OT features).
  2. Boxscores — team-level game stats and goalie-level game stats.

Outputs:
  - data/raw/pbp_events.csv        : Shot/goal/faceoff/penalty events (~2.4M rows)
  - data/raw/boxscores.csv         : Team-level per-game stats (~7,800 rows)
  - data/raw/goalie_boxscores.csv  : Per-goalie per-game stats (~9,000 rows)

Supports resuming — tracks completed games in a progress file.
Saves to disk every 50 games to avoid memory issues.

@author: chazf
"""

import requests
import pandas as pd
import time
import os

# =============================================================================
# Configuration
# =============================================================================
BASE_URL = "https://api-web.nhle.com"
PROJECT_DIR = r"D:\Python\NHL Overtime Model"
DATA_DIR = os.path.join(PROJECT_DIR, "data", "raw")

# Request pacing
DELAY = 0.2
BATCH_SIZE = 50  # save to disk every N games

# Files
GAME_IDS_FILE = os.path.join(DATA_DIR, "game_ids.csv")
PBP_FILE = os.path.join(DATA_DIR, "pbp_events.csv")
BOXSCORES_FILE = os.path.join(DATA_DIR, "boxscores.csv")
GOALIE_BOX_FILE = os.path.join(DATA_DIR, "goalie_boxscores.csv")
PROGRESS_FILE = os.path.join(DATA_DIR, "pbp_scrape_progress.txt")

# Event type codes from PBP
# 502 = faceoff, 503 = hit, 504 = giveaway, 505 = goal,
# 506 = shot-on-goal, 507 = missed-shot, 508 = blocked-shot,
# 509 = penalty, 516 = takeaway
EVENT_CODES = {
    502: "faceoff",
    505: "goal",
    506: "shot-on-goal",
    507: "missed-shot",
    508: "blocked-shot",
    509: "penalty",
}

# =============================================================================
# Helper Functions
# =============================================================================

def api_get(endpoint):
    """Make a GET request to the NHL API with retries."""
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
                print(f"  !! FAILED after 3 attempts: {e}")
                return None


def parse_situation_code(code_str):
    """
    Parse the 4-digit situation code into strength state.
    Format: [away_goalie][away_skaters][home_skaters][home_goalie]
    e.g. '1551' = 5v5, '1541' = away PP (5v4), '0550' = empty net both
    """
    if not code_str or len(str(code_str)) != 4:
        return "unknown", 0, 0, False
    s = str(code_str)
    away_goalie = int(s[0])
    away_skaters = int(s[1])
    home_skaters = int(s[2])
    home_goalie = int(s[3])
    strength = f"{away_skaters}v{home_skaters}"
    is_empty_net = (away_goalie == 0 or home_goalie == 0)
    return strength, away_skaters, home_skaters, is_empty_net


def load_completed_games():
    """Load set of game IDs already scraped."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r") as f:
            return set(int(line.strip()) for line in f if line.strip())
    return set()


def mark_game_complete(game_id):
    """Append a game ID to the progress file."""
    with open(PROGRESS_FILE, "a") as f:
        f.write(f"{game_id}\n")


def append_to_csv(records, filepath, label):
    """Append records to CSV, creating if it doesn't exist."""
    if not records:
        return
    df = pd.DataFrame(records)
    if os.path.exists(filepath):
        df.to_csv(filepath, mode="a", header=False, index=False)
    else:
        df.to_csv(filepath, index=False)
    print(f"    -> Saved {len(records)} {label}")


# =============================================================================
# PBP Event Extraction
# =============================================================================

def extract_pbp_events(pbp_data, game_id, game_date, away_team, home_team):
    """
    Extract shot, goal, faceoff, and penalty events from play-by-play.

    Shot/goal events include x/y coordinates for the xG model.
    Faceoff events capture zone and winner for faceoff features.
    Penalty events capture type and duration for PP/PK context.
    """
    events = []
    if not pbp_data or "plays" not in pbp_data:
        return events

    away_team_id = pbp_data.get("awayTeam", {}).get("id")
    home_team_id = pbp_data.get("homeTeam", {}).get("id")

    for play in pbp_data["plays"]:
        type_code = play.get("typeCode")
        if type_code not in EVENT_CODES:
            continue

        details = play.get("details", {})
        period = play.get("periodDescriptor", {})
        sit_code = play.get("situationCode", "")
        strength, away_sk, home_sk, is_empty_net = parse_situation_code(sit_code)

        # Determine event team
        event_team_id = details.get("eventOwnerTeamId")
        is_home_event = (event_team_id == home_team_id)

        # Shooter/scorer (for shots/goals)
        shooter_id = (details.get("shootingPlayerId")
                      or details.get("scoringPlayerId"))

        event = {
            "game_id": game_id,
            "game_date": game_date,
            "event_id": play.get("eventId"),
            "period": period.get("number"),
            "period_type": period.get("periodType", ""),
            "time_in_period": play.get("timeInPeriod", ""),
            "time_remaining": play.get("timeRemaining", ""),
            "event_type": EVENT_CODES[type_code],
            "type_code": type_code,
            "event_team_id": event_team_id,
            "is_home_event": is_home_event,
            "away_team": away_team,
            "home_team": home_team,
            "situation_code": sit_code,
            "strength": strength,
            "away_skaters": away_sk,
            "home_skaters": home_sk,
            "is_empty_net": is_empty_net,
            # Shot/goal fields
            "x_coord": details.get("xCoord"),
            "y_coord": details.get("yCoord"),
            "zone_code": details.get("zoneCode", ""),
            "shot_type": details.get("shotType", ""),
            "shooter_id": shooter_id,
            "goalie_id": details.get("goalieInNetId"),
            "blocker_id": details.get("blockingPlayerId"),
            # Score state at time of event
            "away_score": details.get("awayScore"),
            "home_score": details.get("homeScore"),
            # Faceoff fields
            "winning_player_id": details.get("winningPlayerId"),
            "losing_player_id": details.get("losingPlayerId"),
            # Penalty fields
            "penalty_type": details.get("descKey", ""),
            "penalty_minutes": details.get("duration"),
        }
        events.append(event)

    return events


# =============================================================================
# Boxscore Extraction
# =============================================================================

def extract_team_boxscore(box_data, game_id, game_date, away_team, home_team):
    """
    Extract team-level stats from boxscore for both teams.
    Returns 2 rows (one per team).
    """
    rows = []
    if not box_data:
        return rows

    # The boxscore has separate team stat blocks
    for side, team_abbrev, is_home in [("awayTeam", away_team, False),
                                        ("homeTeam", home_team, True)]:
        team_data = box_data.get(side, {})
        opp_team = home_team if not is_home else away_team

        row = {
            "game_id": game_id,
            "game_date": game_date,
            "team": team_abbrev,
            "opponent": opp_team,
            "is_home": is_home,
            "score": team_data.get("score"),
            "sog": team_data.get("sog"),
            "faceoff_pct": team_data.get("faceoffWinningPctg"),
            "pp_fraction": team_data.get("powerPlay", ""),
            "pim": team_data.get("pim"),
            "hits": team_data.get("hits"),
            "blocked_shots": team_data.get("blocks"),
            "giveaways": team_data.get("giveaways"),
            "takeaways": team_data.get("takeaways"),
        }

        # Parse power play fraction (e.g. "2/4" -> pp_goals=2, pp_opps=4)
        pp = team_data.get("powerPlay", "")
        if pp and "/" in str(pp):
            parts = str(pp).split("/")
            try:
                row["pp_goals"] = int(parts[0])
                row["pp_opps"] = int(parts[1])
            except (ValueError, IndexError):
                row["pp_goals"] = None
                row["pp_opps"] = None
        else:
            row["pp_goals"] = None
            row["pp_opps"] = None

        rows.append(row)

    return rows


def extract_goalie_boxscore(box_data, game_id, game_date, away_team, home_team):
    """
    Extract per-goalie stats from boxscore.
    Returns one row per goalie who appeared in the game.
    """
    goalies = []
    if not box_data or "playerByGameStats" not in box_data:
        return goalies

    stats = box_data["playerByGameStats"]

    for side, team_abbrev, is_home in [("awayTeam", away_team, False),
                                        ("homeTeam", home_team, True)]:
        team_stats = stats.get(side, {})
        opp_team = home_team if not is_home else away_team

        for g in team_stats.get("goalies", []):
            goalie = {
                "game_id": game_id,
                "game_date": game_date,
                "goalie_id": g.get("playerId"),
                "goalie_name": g.get("name", {}).get("default", ""),
                "team": team_abbrev,
                "opponent": opp_team,
                "is_home": is_home,
                "starter": g.get("starter", False),
                "decision": g.get("decision", ""),
                "toi": g.get("toi", "0:00"),
                "saves": g.get("saves"),
                "shots_against": g.get("shotsAgainst"),
                "save_pct": g.get("savePctg"),
                "goals_against": g.get("goalsAgainst"),
                "ev_shots_against": g.get("evenStrengthShotsAgainst"),
                "pp_shots_against": g.get("powerPlayShotsAgainst"),
                "sh_shots_against": g.get("shorthandedShotsAgainst"),
                "ev_goals_against": g.get("evenStrengthGoalsAgainst"),
                "pp_goals_against": g.get("powerPlayGoalsAgainst"),
                "sh_goals_against": g.get("shorthandedGoalsAgainst"),
            }
            goalies.append(goalie)

    return goalies


# =============================================================================
# Main Scraping Loop
# =============================================================================

def scrape_all_games():
    """
    For each completed game, pull PBP and boxscore data.
    Saves incrementally every BATCH_SIZE games and supports resuming.
    """
    # Load game IDs
    if not os.path.exists(GAME_IDS_FILE):
        print(f"ERROR: game_ids.csv not found at {GAME_IDS_FILE}")
        print("Run game_scraper.py first.")
        return

    games_df = pd.read_csv(GAME_IDS_FILE)
    completed = load_completed_games()

    # Filter to completed games not yet scraped
    games_to_scrape = games_df[
        games_df["game_state"].isin(["FINAL", "OFF"])
        & ~games_df["game_id"].isin(completed)
    ].copy()

    total = len(games_to_scrape)
    print(f"\n{len(completed)} games already scraped.")
    print(f"{total} games remaining to scrape.\n")

    if total == 0:
        print("All games already scraped!")
        return

    # Buffers for batch saving
    pbp_buffer = []
    box_buffer = []
    goalie_buffer = []

    for i, (_, game) in enumerate(games_to_scrape.iterrows()):
        gid = game["game_id"]
        gdate = game["game_date"]
        away = game["away_team"]
        home = game["home_team"]

        print(f"  [{i+1}/{total}] Game {gid} | {away} @ {home} | {gdate}")

        # Pull play-by-play
        pbp = api_get(f"/v1/gamecenter/{gid}/play-by-play")
        time.sleep(DELAY)

        # Pull boxscore
        box = api_get(f"/v1/gamecenter/{gid}/boxscore")
        time.sleep(DELAY)

        # Extract data
        if pbp:
            events = extract_pbp_events(pbp, gid, gdate, away, home)
            pbp_buffer.extend(events)

        if box:
            team_rows = extract_team_boxscore(box, gid, gdate, away, home)
            goalie_rows = extract_goalie_boxscore(box, gid, gdate, away, home)
            box_buffer.extend(team_rows)
            goalie_buffer.extend(goalie_rows)

        mark_game_complete(gid)

        # Periodic save to disk
        if (i + 1) % BATCH_SIZE == 0 or (i + 1) == total:
            print(f"\n  -- Batch save at game {i+1}/{total} --")
            append_to_csv(pbp_buffer, PBP_FILE, "PBP events")
            append_to_csv(box_buffer, BOXSCORES_FILE, "boxscore rows")
            append_to_csv(goalie_buffer, GOALIE_BOX_FILE, "goalie boxscores")
            pbp_buffer = []
            box_buffer = []
            goalie_buffer = []
            print()


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("NHL PBP & Boxscore Scraper")
    print("=" * 60)

    scrape_all_games()

    # Summary
    print("\n" + "=" * 60)
    print("SCRAPING COMPLETE")
    print("=" * 60)

    for filepath, label in [(PBP_FILE, "PBP events"),
                             (BOXSCORES_FILE, "Boxscores"),
                             (GOALIE_BOX_FILE, "Goalie boxscores")]:
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            print(f"  {label}: {len(df):,} rows")
        else:
            print(f"  {label}: NOT YET CREATED")

    print(f"\nFiles saved in: {DATA_DIR}")
