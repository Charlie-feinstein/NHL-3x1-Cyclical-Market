# -*- coding: utf-8 -*-
"""
Goalie Starter Scraper — Daily Faceoff

Scrapes confirmed goalie starters from Daily Faceoff's starting goalies page.
Uses embedded __NEXT_DATA__ JSON (Next.js SSG page) — no HTML parsing needed.

Goalie names from Daily Faceoff (full names like "Arturs Silovs") are matched
to goalie IDs in goalie_boxscores.csv (abbreviated "A. Silovs") via fuzzy matching.

Usage:
    from scrapers.goalie_starter_scraper import scrape_starters
    starters = scrape_starters()  # Returns DataFrame

@author: chazf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import re
import json
import requests
import pandas as pd
from difflib import SequenceMatcher

from config import RAW_DIR
from models.poisson_combiner import TEAM_NAME_TO_ABBREV

DAILY_FACEOFF_URL = "https://www.dailyfaceoff.com/starting-goalies/"
GOALIE_BOX_FILE = os.path.join(RAW_DIR, "goalie_boxscores.csv")
OUTPUT_FILE = os.path.join(RAW_DIR, "goalie_starters.csv")

# Status values we trust enough to use
USABLE_STATUSES = {"Confirmed", "Likely", "Expected"}


def _fetch_page():
    """Fetch the Daily Faceoff starting goalies page."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
    }
    resp = requests.get(DAILY_FACEOFF_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.text


def _extract_json(html):
    """Extract __NEXT_DATA__ JSON from the page source."""
    pattern = r'<script\s+id="__NEXT_DATA__"\s+type="application/json">(.*?)</script>'
    match = re.search(pattern, html, re.DOTALL)
    if not match:
        raise ValueError("Could not find __NEXT_DATA__ in page source")
    return json.loads(match.group(1))


def _parse_games(data):
    """Parse game/goalie data from the __NEXT_DATA__ structure.

    Returns list of dicts with game and goalie info.
    """
    try:
        games = data["props"]["pageProps"]["data"]
    except (KeyError, TypeError):
        raise ValueError("Unexpected __NEXT_DATA__ structure — "
                         "Daily Faceoff may have changed their page format")

    if not isinstance(games, list):
        raise ValueError(f"Expected list of games, got {type(games)}")

    results = []
    for g in games:
        results.append({
            "game_date": g.get("date", ""),
            "home_team_full": g.get("homeTeamName", ""),
            "away_team_full": g.get("awayTeamName", ""),
            "home_goalie_name": g.get("homeGoalieName", ""),
            "away_goalie_name": g.get("awayGoalieName", ""),
            "home_status": g.get("homeNewsStrengthName", "Unconfirmed"),
            "away_status": g.get("awayNewsStrengthName", "Unconfirmed"),
            "game_time": g.get("time", ""),
        })

    return results


def _build_goalie_roster():
    """Build name-to-ID mapping from goalie_boxscores.csv.

    Returns dict: normalized_last_name → list of (goalie_id, full_abbrev_name)
    Also returns a flat dict: abbrev_name → goalie_id (latest season preferred)
    """
    if not os.path.exists(GOALIE_BOX_FILE):
        print(f"  WARNING: {GOALIE_BOX_FILE} not found — cannot match goalie IDs")
        return {}, {}

    gb = pd.read_csv(GOALIE_BOX_FILE)
    gb = gb[["goalie_id", "goalie_name"]].drop_duplicates()

    # Build lastname → [(goalie_id, abbrev_name)] for fuzzy matching
    by_lastname = {}
    name_to_id = {}
    for _, row in gb.iterrows():
        gid = int(row["goalie_id"])
        abbrev = row["goalie_name"]  # e.g. "A. Silovs"
        name_to_id[abbrev] = gid

        # Extract last name
        parts = abbrev.split(". ", 1)
        if len(parts) == 2:
            lastname = parts[1].strip().lower()
        else:
            lastname = abbrev.strip().lower()

        if lastname not in by_lastname:
            by_lastname[lastname] = []
        by_lastname[lastname].append((gid, abbrev))

    return by_lastname, name_to_id


def _match_goalie_id(full_name, by_lastname, name_to_id):
    """Match a full goalie name (e.g. "Arturs Silovs") to a goalie_id.

    Strategy:
    1. Extract last name from full name, look up in by_lastname dict
    2. If multiple matches for same last name, check first initial
    3. Fallback: fuzzy match against all abbreviated names
    """
    if not full_name or full_name.strip() == "":
        return None, None

    parts = full_name.strip().split()
    if len(parts) < 2:
        return None, None

    first_initial = parts[0][0].upper()
    # Handle hyphenated/multi-word last names
    lastname = " ".join(parts[1:]).lower()

    # Direct last name match
    candidates = by_lastname.get(lastname, [])

    if len(candidates) == 1:
        return candidates[0]  # (goalie_id, abbrev_name)

    if len(candidates) > 1:
        # Multiple goalies with same last name — match by first initial
        for gid, abbrev in candidates:
            if abbrev.startswith(first_initial + "."):
                return gid, abbrev
        # Still ambiguous, return first match
        return candidates[0]

    # Try hyphenated variations (e.g. "Ukko-Pekka Luukkonen" → lastname "luukkonen")
    for ln_candidate, entries in by_lastname.items():
        if lastname.endswith(ln_candidate) or ln_candidate.endswith(lastname):
            for gid, abbrev in entries:
                if abbrev.startswith(first_initial + "."):
                    return gid, abbrev
            if entries:
                return entries[0]

    # Fuzzy fallback: match against all abbreviated names
    best_score = 0.0
    best_match = (None, None)
    target = f"{first_initial}. {parts[-1]}"  # "A. Silovs"
    for abbrev, gid in name_to_id.items():
        score = SequenceMatcher(None, target.lower(), abbrev.lower()).ratio()
        if score > best_score:
            best_score = score
            best_match = (gid, abbrev)

    if best_score >= 0.7:
        return best_match

    return None, None


def scrape_starters(date_str=None, save=True):
    """Scrape today's goalie starters from Daily Faceoff.

    Args:
        date_str: Optional date string (YYYY-MM-DD) for filtering/labeling.
                  Daily Faceoff always shows today's games regardless.
        save: Whether to save output to CSV.

    Returns:
        DataFrame with columns:
            game_date, home_team, away_team, home_goalie, away_goalie,
            home_goalie_id, away_goalie_id, home_status, away_status
    """
    print(f"\n{'='*60}")
    print("Goalie Starter Scraper — Daily Faceoff")
    print(f"{'='*60}")

    # Fetch and parse
    print("  Fetching Daily Faceoff starting goalies page...")
    html = _fetch_page()
    data = _extract_json(html)
    games = _parse_games(data)
    print(f"  Found {len(games)} games")

    if not games:
        print("  No games found.")
        return pd.DataFrame()

    # Build goalie roster for ID matching
    by_lastname, name_to_id = _build_goalie_roster()

    # Process each game
    rows = []
    for g in games:
        home_abbrev = TEAM_NAME_TO_ABBREV.get(g["home_team_full"], g["home_team_full"])
        away_abbrev = TEAM_NAME_TO_ABBREV.get(g["away_team_full"], g["away_team_full"])

        home_gid, home_matched = _match_goalie_id(
            g["home_goalie_name"], by_lastname, name_to_id)
        away_gid, away_matched = _match_goalie_id(
            g["away_goalie_name"], by_lastname, name_to_id)

        game_date = date_str or g["game_date"]

        status_h = g["home_status"] or "Unconfirmed"
        status_a = g["away_status"] or "Unconfirmed"

        rows.append({
            "game_date": game_date,
            "home_team": home_abbrev,
            "away_team": away_abbrev,
            "home_goalie": g["home_goalie_name"],
            "away_goalie": g["away_goalie_name"],
            "home_goalie_id": home_gid,
            "away_goalie_id": away_gid,
            "home_goalie_matched": home_matched,
            "away_goalie_matched": away_matched,
            "home_status": status_h,
            "away_status": status_a,
        })

    df = pd.DataFrame(rows)

    # Report
    n_confirmed_h = (df["home_status"].isin(USABLE_STATUSES)).sum()
    n_confirmed_a = (df["away_status"].isin(USABLE_STATUSES)).sum()
    n_matched_h = df["home_goalie_id"].notna().sum()
    n_matched_a = df["away_goalie_id"].notna().sum()
    n_total = len(df)

    print(f"\n  Results:")
    for _, row in df.iterrows():
        h_flag = "✓" if row["home_status"] in USABLE_STATUSES else "?"
        a_flag = "✓" if row["away_status"] in USABLE_STATUSES else "?"
        h_id = f"#{int(row['home_goalie_id'])}" if pd.notna(row["home_goalie_id"]) else "NO MATCH"
        a_id = f"#{int(row['away_goalie_id'])}" if pd.notna(row["away_goalie_id"]) else "NO MATCH"
        print(f"    {row['away_team']}@{row['home_team']}: "
              f"{row['home_goalie']} ({h_flag} {row['home_status']}, {h_id}) vs "
              f"{row['away_goalie']} ({a_flag} {row['away_status']}, {a_id})")

    print(f"\n  Home: {n_confirmed_h}/{n_total} confirmed/likely, "
          f"{n_matched_h}/{n_total} ID matched")
    print(f"  Away: {n_confirmed_a}/{n_total} confirmed/likely, "
          f"{n_matched_a}/{n_total} ID matched")

    if save:
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        df.to_csv(OUTPUT_FILE, index=False)
        print(f"  Saved to {OUTPUT_FILE}")

    return df


def get_usable_starters(starters_df):
    """Filter to only games where both starters are confirmed/likely AND matched.

    Returns DataFrame with columns: game_date, team, goalie_id, goalie_name
    (one row per team, suitable for starters_override in build_goalie_features)
    """
    rows = []
    for _, game in starters_df.iterrows():
        h_usable = (game["home_status"] in USABLE_STATUSES
                     and pd.notna(game["home_goalie_id"]))
        a_usable = (game["away_status"] in USABLE_STATUSES
                     and pd.notna(game["away_goalie_id"]))

        if h_usable:
            rows.append({
                "game_date": game["game_date"],
                "team": game["home_team"],
                "goalie_id": int(game["home_goalie_id"]),
                "goalie_name": game["home_goalie_matched"] or game["home_goalie"],
                "is_home": 1,
                "opponent": game["away_team"],
            })
        if a_usable:
            rows.append({
                "game_date": game["game_date"],
                "team": game["away_team"],
                "goalie_id": int(game["away_goalie_id"]),
                "goalie_name": game["away_goalie_matched"] or game["away_goalie"],
                "is_home": 0,
                "opponent": game["home_team"],
            })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    from datetime import date
    today = date.today().strftime("%Y-%m-%d")
    df = scrape_starters(date_str=today)
    if len(df) > 0:
        usable = get_usable_starters(df)
        print(f"\n  Usable starters (confirmed/likely + matched): {len(usable)}")
