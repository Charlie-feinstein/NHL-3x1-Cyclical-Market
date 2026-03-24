# -*- coding: utf-8 -*-
"""
NHL 3-Way Historical Odds Scraper

Fetches historical regulation-time 3-way odds (home/draw/away) from
the-odds-api.com using the historical EVENT odds endpoint with the
h2h_3_way market key.

The regular historical odds endpoint only supports h2h/spreads/totals.
The per-event endpoint supports ANY market but costs 10 tokens per event.

Usage:
  - Test mode:  python three_way_odds_scraper.py --test
  - Full run:   python three_way_odds_scraper.py
  - 2025-26 only: python three_way_odds_scraper.py --season 2025

Token budget:
  - Historical events listing:  1 token per date
  - Historical event odds:     10 tokens per game per region (1 market)
  - Estimated: ~1500 games × 10 = ~15,000 tokens (1 region)

Output:
  - data/raw/three_way_odds.csv

@author: chazf
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import requests
import pandas as pd
import time
import argparse
from datetime import datetime, timedelta

from config import RAW_DIR, GAME_IDS_FILE, ODDS_API_KEY, THREE_WAY_ODDS_FILE
from models.poisson_combiner import TEAM_NAME_TO_ABBREV

# =============================================================================
# Configuration
# =============================================================================
API_BASE = "https://api.the-odds-api.com/v4"
SPORT = "icehockey_nhl"
MARKET = "h2h_3_way"
REGIONS = "us,eu"       # US + EU (Pinnacle, Matchbook)
ODDS_FORMAT = "decimal"
REQUEST_DELAY = 0.1     # Seconds between requests
TOKEN_BUDGET = 47000    # Stop before exceeding this

# Season date ranges — start from Jan 2025 to save tokens
SEASON_RANGES = {
    "2024": ("2024-10-01", "2025-07-30"),  # 2024-25 season (Jan onwards)
    "2025": ("2025-10-01", "2026-07-30"),  # 2025-26 season (full)
}


# =============================================================================
# Team Name Mapping
# =============================================================================

def normalize_team(name):
    """Map full team name to 3-letter abbreviation."""
    if name in TEAM_NAME_TO_ABBREV:
        return TEAM_NAME_TO_ABBREV[name]
    for full_name, abbrev in TEAM_NAME_TO_ABBREV.items():
        if name.lower() == full_name.lower():
            return abbrev
    return name


# =============================================================================
# API Functions
# =============================================================================

def get_remaining_tokens(resp):
    """Extract remaining tokens from response headers."""
    val = resp.headers.get("x-requests-remaining")
    return int(val) if val is not None else None


def fetch_event_ids(date_str, api_key):
    """Fetch event IDs for a given date. Costs 1 token.

    Returns: (list of event dicts, remaining_tokens)
    Each event dict: {id, home_team, away_team, commence_time}
    """
    iso_date = f"{date_str}T19:00:00Z"
    url = f"{API_BASE}/historical/sports/{SPORT}/events"
    params = {
        "apiKey": api_key,
        "date": iso_date,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\n  !! Events API error for {date_str}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"     Status: {e.response.status_code}")
            try:
                print(f"     Body: {e.response.text[:300]}")
            except Exception:
                pass
        return [], None

    remaining = get_remaining_tokens(resp)
    data = resp.json()

    # Historical events endpoint wraps in "data" key
    events = data.get("data", data)
    if isinstance(events, dict):
        events = events.get("data", [events])
    if not isinstance(events, list):
        events = [events]

    result = []
    for ev in events:
        if not isinstance(ev, dict):
            continue
        result.append({
            "id": ev.get("id", ""),
            "home_team": ev.get("home_team", ""),
            "away_team": ev.get("away_team", ""),
            "commence_time": ev.get("commence_time", ""),
        })

    return result, remaining


def filter_events_for_date(events, target_date_str):
    """Filter events to only those whose commence_time falls on the target date.

    NHL games are scheduled by ET date but commence_time is UTC. Evening ET
    games (7pm+) have UTC times on the next calendar day. So a game on the
    ET date "2025-01-06" might have commence_time "2025-01-07T00:00:00Z".

    We match events whose UTC commence_time falls in the window:
      target_date 10:00 UTC  to  target_date+1 09:59 UTC
    This covers 5am-5am ET, capturing all possible NHL start times for the ET date.
    """
    from datetime import datetime, timedelta
    try:
        target = datetime.strptime(target_date_str, "%Y-%m-%d")
    except ValueError:
        return []

    window_start = target + timedelta(hours=10)   # target_date 10:00 UTC = 5am ET
    window_end = target + timedelta(days=1, hours=10)  # next day 10:00 UTC = 5am ET

    filtered = []
    for ev in events:
        ct = ev.get("commence_time", "")
        if not ct:
            continue
        try:
            event_dt = datetime.strptime(ct, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            continue
        if window_start <= event_dt < window_end:
            filtered.append(ev)
    return filtered


def fetch_event_odds(event_id, date_str, api_key, regions=None, commence_time=None):
    """Fetch 3-way odds for a specific event from ALL bookmakers.

    Cost: 10 tokens per region (us + eu = 20 tokens).

    Uses commence_time - 1 hour as the snapshot time to ensure we get
    pre-game odds (not live in-play odds for matinee games).

    Returns: (list of odds_dicts, remaining_tokens)
    Each odds_dict: {home, draw, away, bookmaker}
    Returns empty list if no odds found.
    """
    if regions is None:
        regions = REGIONS

    # Use 1 hour before game start to get pre-game odds
    if commence_time:
        try:
            ct = datetime.strptime(commence_time, "%Y-%m-%dT%H:%M:%SZ")
            snapshot = ct - timedelta(hours=1)
            iso_date = snapshot.strftime("%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            iso_date = f"{date_str}T19:00:00Z"
    else:
        iso_date = f"{date_str}T19:00:00Z"
    url = f"{API_BASE}/historical/sports/{SPORT}/events/{event_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": regions,
        "markets": MARKET,
        "date": iso_date,
        "oddsFormat": ODDS_FORMAT,
    }

    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        status = None
        if hasattr(e, 'response') and e.response is not None:
            status = e.response.status_code
        if status in (422, 404):
            return [], None
        print(f"\n  !! Odds API error for event {event_id}: {e}")
        return [], None

    remaining = get_remaining_tokens(resp)
    data = resp.json()

    # Parse response — may be wrapped in "data" key
    event_data = data.get("data", data)
    if not isinstance(event_data, dict):
        return [], remaining

    home_team_raw = event_data.get("home_team", "")
    bookmakers = event_data.get("bookmakers", [])

    if not bookmakers:
        return [], remaining

    # Extract odds from ALL bookmakers that have h2h_3_way
    all_odds = []
    for bk in bookmakers:
        result = extract_3way_odds(bk, home_team_raw)
        if result:
            result["bookmaker"] = bk.get("key", "unknown")
            all_odds.append(result)

    return all_odds, remaining


def extract_3way_odds(bookmaker, home_team_name):
    """Extract home/draw/away decimal odds from a bookmaker's h2h_3_way market."""
    markets = bookmaker.get("markets", [])
    for market in markets:
        if market.get("key") != MARKET:
            continue
        outcomes = market.get("outcomes", [])
        if len(outcomes) < 3:
            continue

        result = {}
        for o in outcomes:
            name = o.get("name", "")
            price = o.get("price")
            if price is None:
                continue
            if name == "Draw":
                result["draw"] = price
            elif name == home_team_name:
                result["home"] = price
            else:
                result["away"] = price

        if "home" in result and "draw" in result and "away" in result:
            return result

    return None


def count_regions(regions_str):
    """Count how many regions are in the comma-separated string."""
    return len([r.strip() for r in regions_str.split(",") if r.strip()])


# =============================================================================
# Date Collection
# =============================================================================

def get_game_dates(seasons=None):
    """Load unique game dates from game_ids.csv for specified seasons."""
    games = pd.read_csv(GAME_IDS_FILE, parse_dates=["game_date"])
    dates = set()
    for season_key, (start, end) in SEASON_RANGES.items():
        if seasons and season_key not in seasons:
            continue
        mask = (games["game_date"] >= start) & (games["game_date"] <= end)
        season_dates = games.loc[mask, "game_date"].dt.strftime("%Y-%m-%d").unique()
        dates.update(season_dates)
    return sorted(dates)


def get_fetched_event_ids():
    """Load event IDs already fetched from output CSV.

    Returns set of event_ids that have at least one row.
    Since we now save all books per event, any event_id present
    means we've already fetched ALL books for that event.
    """
    if not os.path.exists(THREE_WAY_ODDS_FILE):
        return set()
    df = pd.read_csv(THREE_WAY_ODDS_FILE)
    if "event_id" in df.columns:
        return set(df["event_id"].dropna().unique())
    return set()


def get_fetched_game_keys():
    """Load (date, home_team) keys already fetched."""
    if not os.path.exists(THREE_WAY_ODDS_FILE):
        return set()
    df = pd.read_csv(THREE_WAY_ODDS_FILE)
    return set(zip(df["game_date"].astype(str), df["home_team"]))


# =============================================================================
# Main
# =============================================================================

def run_test(api_key):
    """Make a single test call to verify h2h_3_way works via event endpoint."""
    print("=" * 60)
    print("TEST MODE — Event endpoint test")
    print("=" * 60)

    test_date = "2025-11-15"
    print(f"\n1. Fetching event IDs for {test_date}...")
    events, remaining = fetch_event_ids(test_date, api_key)
    print(f"   Found {len(events)} events total. Tokens remaining: {remaining}")

    # Filter to just this date
    day_events = filter_events_for_date(events, test_date)
    print(f"   Events on {test_date}: {len(day_events)}")

    if not day_events:
        # Try without filtering if no exact match
        print(f"   No exact date match. Showing all events:")
        for ev in events[:5]:
            ct = ev.get("commence_time", "")[:10]
            print(f"     {ev['away_team']} @ {ev['home_team']} (date: {ct}, id: {ev['id'][:12]}...)")
        if events:
            day_events = events[:1]  # test with first event regardless
        else:
            print("   No events found at all.")
            return False

    # Show events
    for ev in day_events[:5]:
        print(f"   {ev['away_team']} @ {ev['home_team']} (id: {ev['id'][:12]}...)")

    # Test one event's odds
    test_event = day_events[0]
    print(f"\n2. Fetching h2h_3_way for event {test_event['id'][:12]}... (regions={REGIONS})")
    all_odds, remaining = fetch_event_odds(test_event["id"], test_date, api_key)
    print(f"   Tokens remaining: {remaining}")

    if all_odds:
        print(f"\n   3-WAY ODDS FOUND — {len(all_odds)} bookmaker(s)!")
        for odds in all_odds:
            print(f"   Home: {odds['home']:.2f} | Draw: {odds['draw']:.2f} | "
                  f"Away: {odds['away']:.2f} | Book: {odds['bookmaker']}")
        return True
    else:
        print(f"\n   No h2h_3_way odds returned for regions={REGIONS}.")
        print(f"   Try --regions us,eu if US books don't carry this market.")
        return False


def run_full(api_key, seasons=None):
    """Fetch historical 3-way odds for all game dates via event endpoint."""
    print("=" * 60)
    print(f"NHL 3-Way Odds Scraper — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Using per-event historical endpoint")
    print(f"Regions: {REGIONS}")
    print("=" * 60)

    n_regions = count_regions(REGIONS)
    tokens_per_event = 10 * n_regions

    all_dates = get_game_dates(seasons)
    fetched_event_ids = get_fetched_event_ids()
    fetched_keys = get_fetched_game_keys()

    print(f"\nTotal game dates: {len(all_dates)}")
    print(f"Games already fetched: {len(fetched_keys)}")
    print(f"Tokens per event odds call: {tokens_per_event} ({n_regions} region(s))")
    print(f"Token budget: {TOKEN_BUDGET}")

    os.makedirs(RAW_DIR, exist_ok=True)
    tokens_used = 0
    total_games_fetched = 0
    total_games_skipped = 0
    total_events_no_odds = 0
    consecutive_failures = 0
    budget_exceeded = False

    for date_idx, date_str in enumerate(all_dates):
        if budget_exceeded:
            break

        if tokens_used > TOKEN_BUDGET:
            print(f"\n!! TOKEN BUDGET REACHED ({tokens_used} used). Stopping.")
            break

        print(f"\n[{date_idx+1}/{len(all_dates)}] {date_str}", end="")

        # Step 1: Get event IDs for this date (1 token)
        events, remaining = fetch_event_ids(date_str, api_key)
        tokens_used += 1

        if not events:
            print(f" — no events", end="")
            if remaining is not None:
                print(f" | Remaining: {remaining}", end="")
            print()
            continue

        # Filter to only events that START on this date (ET-aware)
        day_events = filter_events_for_date(events, date_str)

        print(f" — {len(day_events)} games (of {len(events)} total events)", end="")
        if remaining is not None:
            print(f" | Remaining: {remaining}", end="")
        print()

        if not day_events:
            continue

        # Step 2: Fetch odds for each event on this date
        date_rows = []
        for ev in day_events:
            home_abbrev = normalize_team(ev["home_team"])
            away_abbrev = normalize_team(ev["away_team"])
            event_id = ev["id"]

            # Skip already fetched (by event_id or by key)
            if event_id in fetched_event_ids:
                total_games_skipped += 1
                continue
            if (date_str, home_abbrev) in fetched_keys:
                total_games_skipped += 1
                continue

            # Check token budget
            if tokens_used + tokens_per_event > TOKEN_BUDGET:
                print(f"  !! Token budget would be exceeded. Stopping.")
                budget_exceeded = True
                break

            all_odds, remaining = fetch_event_odds(
                event_id, date_str, api_key,
                commence_time=ev.get("commence_time")
            )
            tokens_used += tokens_per_event

            if all_odds:
                books_found = []
                for odds in all_odds:
                    row = {
                        "game_date": date_str,
                        "home_team": home_abbrev,
                        "away_team": away_abbrev,
                        "home_dec": odds["home"],
                        "draw_dec": odds["draw"],
                        "away_dec": odds["away"],
                        "bookmaker": odds["bookmaker"],
                        "event_id": event_id,
                        "timestamp": ev["commence_time"],
                    }
                    date_rows.append(row)
                    books_found.append(odds["bookmaker"])
                consecutive_failures = 0
                # Show first book's odds + total count
                o = all_odds[0]
                print(f"  {away_abbrev}@{home_abbrev}: "
                      f"H={o['home']:.2f} D={o['draw']:.2f} A={o['away']:.2f} "
                      f"[{', '.join(books_found)}]")
            else:
                total_events_no_odds += 1
                consecutive_failures += 1
                print(f"  {away_abbrev}@{home_abbrev}: no 3-way odds")

            # If 20 consecutive failures, market probably doesn't exist
            if consecutive_failures >= 20:
                print(f"\n!! 20 consecutive events with no h2h_3_way odds.")
                print(f"   The market may not be available for NHL.")
                print(f"   Tokens used: {tokens_used}")
                if date_rows:
                    _save_rows(date_rows)
                return

            time.sleep(REQUEST_DELAY)

        # Save this date's results
        if date_rows:
            _save_rows(date_rows)
            total_games_fetched += len(date_rows)
            fetched_keys.update(
                (r["game_date"], r["home_team"]) for r in date_rows
            )
            fetched_event_ids.update(r["event_id"] for r in date_rows)

    print(f"\n{'=' * 60}")
    print(f"Done!")
    print(f"  Games fetched:    {total_games_fetched}")
    print(f"  Games skipped:    {total_games_skipped} (already in CSV)")
    print(f"  Events no odds:   {total_events_no_odds}")
    print(f"  Tokens used:      ~{tokens_used}")
    print(f"{'=' * 60}")


def _save_rows(rows):
    """Append rows to the output CSV."""
    df = pd.DataFrame(rows)
    if os.path.exists(THREE_WAY_ODDS_FILE):
        df.to_csv(THREE_WAY_ODDS_FILE, mode="a", header=False, index=False)
    else:
        df.to_csv(THREE_WAY_ODDS_FILE, index=False)


# =============================================================================
# Live Odds Fetcher (for predict_today.py)
# =============================================================================

def fetch_today_odds(date_str=None, api_key=None, regions=None):
    """Fetch live 3-way regulation odds for today's games.

    Uses the LIVE (non-historical) odds API endpoint — current pre-game lines.
    Cost: ~1 token for events list + ~10 tokens per game (per region).
    Typical daily cost: ~150 tokens for a full NHL slate.

    Args:
        date_str: Date string (YYYY-MM-DD) for labeling. Defaults to today.
        api_key: The Odds API key. Defaults to config.
        regions: API regions string. Defaults to "us".

    Returns:
        dict[(date_str, home_abbrev)] → list of {
            home_dec, draw_dec, away_dec,
            home_fair, draw_fair, away_fair,
            bookmaker
        }
        Same format as load_three_way_odds() in walk_forward_3way.py.
    """
    from models.predict_3way import power_devig_3way

    if api_key is None:
        api_key = ODDS_API_KEY
    if regions is None:
        regions = REGIONS
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")

    print(f"\n  Fetching live 3-way odds (regions={regions})...")

    # Step 1: Get today's events from LIVE endpoint (not historical)
    url = f"{API_BASE}/sports/{SPORT}/events"
    params = {"apiKey": api_key}
    try:
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"  !! Events API error: {e}")
        return {}

    remaining = get_remaining_tokens(resp)
    events = resp.json()
    if not isinstance(events, list):
        events = events.get("data", [])

    # Filter to today's games (ET-aware)
    day_events = filter_events_for_date(events, date_str)
    print(f"  Found {len(day_events)} events for {date_str} "
          f"(tokens remaining: {remaining})")

    if not day_events:
        return {}

    # Step 2: Fetch odds for each event from LIVE endpoint
    result = {}
    for ev in day_events:
        event_id = ev["id"]
        home_abbrev = normalize_team(ev["home_team"])
        away_abbrev = normalize_team(ev["away_team"])

        odds_url = f"{API_BASE}/sports/{SPORT}/events/{event_id}/odds"
        params = {
            "apiKey": api_key,
            "regions": regions,
            "markets": MARKET,
            "oddsFormat": ODDS_FORMAT,
        }
        try:
            resp = requests.get(odds_url, params=params, timeout=30)
            resp.raise_for_status()
        except requests.exceptions.RequestException as e:
            status = None
            if hasattr(e, 'response') and e.response is not None:
                status = e.response.status_code
            if status in (422, 404):
                print(f"    {away_abbrev}@{home_abbrev}: no 3-way market")
                continue
            print(f"    {away_abbrev}@{home_abbrev}: API error: {e}")
            continue

        remaining = get_remaining_tokens(resp)
        data = resp.json()

        # Parse bookmakers
        bookmakers = data.get("bookmakers", [])
        if not bookmakers:
            print(f"    {away_abbrev}@{home_abbrev}: no bookmakers")
            continue

        home_team_raw = data.get("home_team", ev["home_team"])
        key = (date_str, home_abbrev)
        if key not in result:
            result[key] = []

        books_found = []
        for bk in bookmakers:
            odds_dict = extract_3way_odds(bk, home_team_raw)
            if odds_dict:
                # Sanity filter: reject live/stale odds
                if (odds_dict["home"] < 1.10 or odds_dict["away"] < 1.10
                        or odds_dict["draw"] > 15):
                    continue

                home_imp = 1.0 / odds_dict["home"]
                draw_imp = 1.0 / odds_dict["draw"]
                away_imp = 1.0 / odds_dict["away"]
                fair_h, fair_d, fair_a = power_devig_3way(
                    home_imp, draw_imp, away_imp)

                result[key].append({
                    "home_dec": odds_dict["home"],
                    "draw_dec": odds_dict["draw"],
                    "away_dec": odds_dict["away"],
                    "home_fair": fair_h,
                    "draw_fair": fair_d,
                    "away_fair": fair_a,
                    "bookmaker": bk.get("key", "unknown"),
                })
                books_found.append(bk.get("key", "?"))

        if books_found:
            o = result[key][0]
            print(f"    {away_abbrev}@{home_abbrev}: "
                  f"H={o['home_dec']:.2f} D={o['draw_dec']:.2f} "
                  f"A={o['away_dec']:.2f} [{', '.join(books_found)}]")
        else:
            print(f"    {away_abbrev}@{home_abbrev}: no valid 3-way odds")

        time.sleep(REQUEST_DELAY)

    n_games = len(result)
    n_books = sum(len(v) for v in result.values())
    print(f"  Live odds: {n_games} games, {n_books} book-rows "
          f"(tokens remaining: {remaining})")

    return result


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch historical 3-way NHL odds")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: verify h2h_3_way works via event endpoint")
    parser.add_argument("--season", default=None,
                        help="Season to fetch: '2024', '2025', or omit for both")
    parser.add_argument("--regions", default=REGIONS,
                        help="API regions (default: us)")
    parser.add_argument("--budget", type=int, default=TOKEN_BUDGET,
                        help=f"Max tokens to use (default: {TOKEN_BUDGET})")
    args = parser.parse_args()

    REGIONS = args.regions
    TOKEN_BUDGET = args.budget

    api_key = ODDS_API_KEY
    if api_key == "YOUR_API_KEY_HERE":
        api_key = input("Enter your the-odds-api.com API key: ").strip()

    if args.test:
        success = run_test(api_key)
        if success:
            print("\nTest passed! Run without --test to fetch all dates.")
        else:
            print("\nTest failed. h2h_3_way may not be available for NHL.")
    else:
        seasons = [args.season] if args.season else None
        run_full(api_key, seasons=seasons)
