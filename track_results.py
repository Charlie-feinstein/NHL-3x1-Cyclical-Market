# -*- coding: utf-8 -*-
"""
track_results.py — Daily P&L Tracker for NHL 3-Way Model

Grades predictions from predict_today.py against actual game results.
Run each morning to see how yesterday's bets performed.

Tracks 3 staking methods (matching backtest format):
  - Flat:  1u per qualifying bet
  - Blend: edge-proportional staking (draw priority, then home)
  - Kelly: quarter-Kelly sizing

Usage:
    Spyder:  Set RUN_DATE / SKIP_SCRAPING below, then run the file.
    CLI:     python track_results.py [--date 2026-03-05] [--skip-scraping]

@author: chazf
"""

import sys
import os
import argparse
from datetime import date, timedelta

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
RUN_DATE = None        # Date to grade (YYYY-MM-DD), or None = yesterday
SKIP_SCRAPING = False  # Skip game_ids refresh

# ── Blend staking parameters (must match predict_3way.py / dashboard.py) ────
DRAW_MIN_EDGE = 0.025
HOME_MIN_EDGE = 0.025
HOME_MAX_EDGE = 0.05
DRAW_STAKE_MULT = 25
DRAW_STAKE_CAP = 1.5
HOME_STAKE_MULT = 20
HOME_STAKE_CAP = 1.0
KELLY_FRACTION = 0.25   # Quarter-Kelly

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import RAW_DIR, GAME_IDS_FILE

PREDICTIONS_DIR = os.path.join(PROJECT_DIR, "data", "predictions")
PREDICTIONS_LOG = os.path.join(PREDICTIONS_DIR, "predictions_log.csv")
TRACKING_FILE = os.path.join(PREDICTIONS_DIR, "tracking_results.csv")

# Use local paths (config.py has Windows paths; fall back for WSL)
_LOCAL_GAME_IDS = os.path.join(PROJECT_DIR, "data", "raw", "game_ids.csv")
if os.path.exists(_LOCAL_GAME_IDS) and not os.path.exists(GAME_IDS_FILE):
    GAME_IDS_FILE = _LOCAL_GAME_IDS


# =============================================================================
# Step 1: Refresh game results (current season only)
# =============================================================================

def _refresh_current_season():
    """Re-scrape only the current season and merge with existing game_ids.csv."""
    from scrapers.game_scraper import SEASONS, api_get, parse_game
    from datetime import datetime
    import time as _time

    # Determine current season (Oct 2025 → Apr 2026 = "20252026")
    today = date.today()
    if today.month >= 10:
        season_key = f"{today.year}{today.year + 1}"
    else:
        season_key = f"{today.year - 1}{today.year}"

    if season_key not in SEASONS:
        print(f"  WARNING: Season {season_key} not in SEASONS dict. Running full refresh.")
        from scrapers.game_scraper import scrape_game_ids
        scrape_game_ids(force_refresh=True)
        return

    # Load existing data
    existing = pd.read_csv(GAME_IDS_FILE)
    old_season = existing[existing["season"] != season_key].copy() if "season" in existing.columns else existing.copy()

    # Scrape current season
    start_str, end_str = SEASONS[season_key]
    start_date = datetime.strptime(start_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_str, "%Y-%m-%d")
    current = start_date

    print(f"  Re-scraping {season_key} ({start_str} to {end_str})...")
    new_games = []
    seen_ids = set()

    while current <= end_date:
        date_str = current.strftime("%Y-%m-%d")
        data = api_get(f"/v1/schedule/{date_str}")

        if data and "gameWeek" in data:
            for week in data["gameWeek"]:
                week_date = week.get("date", date_str)
                for game in week.get("games", []):
                    parsed = parse_game(game, week_date)
                    if parsed and parsed["game_id"] not in seen_ids:
                        new_games.append(parsed)
                        seen_ids.add(parsed["game_id"])
            current += timedelta(days=7)
        else:
            current += timedelta(days=1)

        _time.sleep(0.3)

    new_df = pd.DataFrame(new_games)
    print(f"  -> {len(new_df)} games found for {season_key}")

    # Merge: keep old seasons + fresh current season
    if "season" in existing.columns:
        combined = pd.concat([old_season, new_df], ignore_index=True)
    else:
        # Old file doesn't have season column — replace entirely with new scrape
        combined = pd.concat([
            existing[~existing["game_id"].isin(new_df["game_id"])],
            new_df
        ], ignore_index=True)

    combined = combined.drop_duplicates(subset=["game_id"], keep="last")
    combined = combined.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    combined.to_csv(GAME_IDS_FILE, index=False)

    completed = combined[combined["game_state"].isin(["FINAL", "OFF"])]
    print(f"  Total: {len(combined)} games ({len(completed)} completed)")


def step_refresh_results():
    """Refresh game results — only re-scrape current season, not all history."""
    print("=" * 70)
    print("STEP 1: Refreshing game results (current season only)")
    print("=" * 70)

    if not os.path.exists(GAME_IDS_FILE):
        # First run — need full scrape
        from scrapers.game_scraper import scrape_game_ids
        print("  No game_ids.csv found, running full scrape...")
        scrape_game_ids(force_refresh=True)
    else:
        # Incremental: only re-scrape the current season
        _refresh_current_season()

    print()


# =============================================================================
# Step 2: Load predictions for a specific date
# =============================================================================

def load_predictions(target_date):
    """Load predictions for a specific date from predictions_log.csv.

    Deduplicates on (game_date, game_id, bookmaker), keeping the last entry.
    """
    if not os.path.exists(PREDICTIONS_LOG):
        print(f"  ERROR: {PREDICTIONS_LOG} not found. Run predict_today.py first.")
        return pd.DataFrame()

    df = pd.read_csv(PREDICTIONS_LOG)

    # Coerce numeric columns that may have been saved as strings
    _numeric_cols = [
        "p_home_reg", "p_away_reg", "p_ot", "draw_edge", "home_edge", "away_edge",
        "dec_draw", "dec_home", "dec_away", "fair_home", "fair_draw", "fair_away",
        "bet_odds_dec", "kelly_stake", "lam_home", "lam_away",
    ]
    for col in _numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    # Filter to target date only
    df = df[df["game_date"] == target_date].copy()
    if df.empty:
        print(f"  No predictions found for {target_date}")
        return pd.DataFrame()

    n_before = len(df)
    df = df.drop_duplicates(subset=["game_date", "game_id", "bookmaker"], keep="last")
    n_after = len(df)

    if n_before > n_after:
        print(f"  Loaded {target_date}: {n_before} → {n_after} rows ({n_before - n_after} dupes removed)")
    else:
        print(f"  Loaded {target_date}: {n_after} prediction rows")

    return df


# =============================================================================
# Step 3: Load game results
# =============================================================================

def load_game_results():
    """Load game_ids.csv and compute actual 3-way outcome."""
    df = pd.read_csv(GAME_IDS_FILE)

    # Only completed games
    df = df[df["game_state"] == "OFF"].copy()

    # Compute actual 3-way outcome
    def get_actual_3way(row):
        if row["game_outcome_type"] in ("OT", "SO"):
            return "draw"
        elif row["home_score"] > row["away_score"]:
            return "home"
        else:
            return "away"

    df["actual_3way"] = df.apply(get_actual_3way, axis=1)

    # Deduplicate on game_id (safety net)
    df = df.drop_duplicates(subset=["game_id"], keep="last")

    df.rename(columns={
        "home_score": "home_score_result",
        "away_score": "away_score_result",
        "game_outcome_type": "game_outcome_type_result",
    }, inplace=True)

    return df[["game_id", "home_score_result", "away_score_result",
               "game_outcome_type_result", "actual_3way"]].copy()


# =============================================================================
# Step 4: Grade predictions — compute all 3 staking methods
# =============================================================================

def _compute_blend(row):
    """Compute blend staking: both sides bet independently (matches backtest).

    Returns: (draw_stake, home_stake) — either can be 0.
    """
    d_edge = row.get("draw_edge", 0) or 0
    h_edge = row.get("home_edge", 0) or 0

    d_stake = round(min(d_edge * DRAW_STAKE_MULT, DRAW_STAKE_CAP), 4) if d_edge >= DRAW_MIN_EDGE else 0.0
    h_stake = round(min(h_edge * HOME_STAKE_MULT, HOME_STAKE_CAP), 4) if HOME_MIN_EDGE <= h_edge <= HOME_MAX_EDGE else 0.0
    return d_stake, h_stake


def _compute_kelly(edge, model_prob, odds_dec):
    """Compute quarter-Kelly stake.

    Returns stake in units (0 if no edge).
    """
    if edge <= 0 or odds_dec <= 1.0 or model_prob <= 0:
        return 0.0
    b = odds_dec - 1.0
    p = model_prob
    q = 1.0 - p
    kelly_full = (b * p - q) / b if b > 0 else 0.0
    return max(0.0, kelly_full * KELLY_FRACTION)


def _get_draw_odds(row):
    """Extract actual book draw odds from prediction row."""
    # Best source: raw book odds column (saved by predict_3way.py)
    try:
        dec = float(row.get("dec_draw", 0) or 0)
    except (ValueError, TypeError):
        dec = 0.0
    if dec > 1.0:
        return dec
    # If original bet_side was draw, bet_odds_dec has the actual book draw odds
    orig_side = row.get("_orig_bet_side", "")
    try:
        bet_dec = float(row.get("bet_odds_dec", 0) or 0)
    except (ValueError, TypeError):
        bet_dec = 0.0
    if orig_side == "draw" and bet_dec > 1.0:
        return bet_dec
    # Last resort: devigged fair odds (slightly inflated vs real book price)
    try:
        fair_d = float(row.get("fair_draw", 0) or 0)
    except (ValueError, TypeError):
        fair_d = 0.0
    return (1.0 / fair_d) if fair_d > 0 else 0.0


def _get_home_odds(row):
    """Extract actual book home odds from prediction row."""
    try:
        dec = float(row.get("dec_home", 0) or 0)
    except (ValueError, TypeError):
        dec = 0.0
    if dec > 1.0:
        return dec
    orig_side = row.get("_orig_bet_side", "")
    try:
        bet_dec = float(row.get("bet_odds_dec", 0) or 0)
    except (ValueError, TypeError):
        bet_dec = 0.0
    if orig_side == "home" and bet_dec > 1.0:
        return bet_dec
    try:
        fair_h = float(row.get("fair_home", 0) or 0)
    except (ValueError, TypeError):
        fair_h = 0.0
    return (1.0 / fair_h) if fair_h > 0 else 0.0


def grade_predictions(preds, results):
    """Join predictions with results and compute all staking methods."""
    graded = preds.merge(results, on="game_id", how="left")

    # Save original bet_side before resetting (needed to know what bet_odds_dec means)
    graded["_orig_bet_side"] = graded["bet_side"].copy()

    # Reset blend columns — recompute from blend logic
    # Use old names (bet_side, kelly_stake, etc.) for dashboard compatibility
    graded["bet_side"] = None
    graded["kelly_stake"] = 0.0
    graded["bet_won"] = np.nan
    graded["bet_profit"] = 0.0
    graded["is_bet"] = 0

    # Draw-side columns (flat + Kelly)
    graded["draw_won"] = np.nan
    graded["draw_flat_profit"] = 0.0
    graded["draw_kelly_stake"] = 0.0
    graded["draw_kelly_profit"] = 0.0

    # Home-side columns (flat + Kelly)
    graded["home_won"] = np.nan
    graded["home_flat_profit"] = 0.0
    graded["home_kelly_stake"] = 0.0
    graded["home_kelly_profit"] = 0.0

    for idx, row in graded.iterrows():
        actual = row.get("actual_3way")
        if pd.isna(actual):
            continue  # pending game

        d_edge = row.get("draw_edge", 0) or 0
        h_edge = row.get("home_edge", 0) or 0
        draw_odds = _get_draw_odds(row)
        home_odds = _get_home_odds(row)

        # --- Draw side ---
        if d_edge > 0 and draw_odds > 1.0:
            draw_won = int(actual == "draw")
            graded.at[idx, "draw_won"] = draw_won
            graded.at[idx, "draw_flat_profit"] = round(
                (draw_odds - 1.0) if draw_won else -1.0, 4
            )
            k_stake = _compute_kelly(d_edge, row.get("p_ot", 0), draw_odds)
            graded.at[idx, "draw_kelly_stake"] = round(k_stake, 4)
            graded.at[idx, "draw_kelly_profit"] = round(
                k_stake * (draw_odds - 1.0) if draw_won else -k_stake, 4
            )

        # --- Home side ---
        if h_edge > 0 and home_odds > 1.0:
            home_won = int(actual == "home")
            graded.at[idx, "home_won"] = home_won
            graded.at[idx, "home_flat_profit"] = round(
                (home_odds - 1.0) if home_won else -1.0, 4
            )
            k_stake = _compute_kelly(h_edge, row.get("p_home_reg", 0), home_odds)
            graded.at[idx, "home_kelly_stake"] = round(k_stake, 4)
            graded.at[idx, "home_kelly_profit"] = round(
                k_stake * (home_odds - 1.0) if home_won else -k_stake, 4
            )

        # --- Blend (what we actually bet — both sides independent) ---
        d_blend, h_blend = _compute_blend(row)
        blend_sides = []
        blend_total_stake = 0.0
        blend_total_profit = 0.0
        blend_wins = 0
        blend_bets = 0

        if d_blend > 0 and draw_odds > 1.0:
            d_won = int(actual == "draw")
            d_profit = round(d_blend * (draw_odds - 1.0) if d_won else -d_blend, 4)
            blend_sides.append("draw")
            blend_total_stake += d_blend
            blend_total_profit += d_profit
            blend_wins += d_won
            blend_bets += 1

        if h_blend > 0 and home_odds > 1.0:
            h_won = int(actual == "home")
            h_profit = round(h_blend * (home_odds - 1.0) if h_won else -h_blend, 4)
            blend_sides.append("home")
            blend_total_stake += h_blend
            blend_total_profit += h_profit
            blend_wins += h_won
            blend_bets += 1

        if blend_bets > 0:
            graded.at[idx, "bet_side"] = "+".join(blend_sides)
            graded.at[idx, "kelly_stake"] = round(blend_total_stake, 4)
            graded.at[idx, "is_bet"] = 1
            graded.at[idx, "bet_won"] = float(blend_wins > 0)
            graded.at[idx, "bet_profit"] = round(blend_total_profit, 4)

    # Clean up temp column
    graded.drop(columns=["_orig_bet_side"], inplace=True)
    graded["is_bet"] = graded["is_bet"].astype(int)
    return graded


# =============================================================================
# Step 5: Save — append to tracking_results.csv
# =============================================================================

def save_tracking_results(graded, target_date):
    """Append new graded results to tracking_results.csv.

    If the target_date already exists, replace those rows (idempotent re-run).
    """
    os.makedirs(PREDICTIONS_DIR, exist_ok=True)

    if os.path.exists(TRACKING_FILE):
        existing = pd.read_csv(TRACKING_FILE)
        # Remove any existing rows for this date (allows re-running)
        existing = existing[existing["game_date"] != target_date]
        combined = pd.concat([existing, graded], ignore_index=True)
        combined.to_csv(TRACKING_FILE, index=False)
        n_existing = len(existing)
        print(f"  Appended {len(graded)} rows for {target_date} (existing: {n_existing} rows from other dates)")
    else:
        graded.to_csv(TRACKING_FILE, index=False)
        print(f"  Created {TRACKING_FILE} with {len(graded)} rows")

    print(f"  Total tracking file: {len(pd.read_csv(TRACKING_FILE))} rows")


# =============================================================================
# Step 6: Print summary
# =============================================================================

def print_summary(graded, target_date):
    """Print performance summary for the graded date + cumulative."""
    print()
    print("=" * 70)
    print("NHL 3-WAY MODEL — RESULTS TRACKER")
    print("=" * 70)

    completed = graded[graded["actual_3way"].notna()].copy()
    pending = graded[graded["actual_3way"].isna()].copy()

    if completed.empty:
        print(f"\n  No completed games for {target_date}.")
        if not pending.empty:
            print(f"  {pending.drop_duplicates('game_id').shape[0]} games pending results.")
        return

    # ── Game summary ──
    games = completed.drop_duplicates("game_id")
    n_games = len(games)
    print(f"\n  {target_date}: {n_games} games completed")

    # Outcome breakdown
    n_home = (games["actual_3way"] == "home").sum()
    n_away = (games["actual_3way"] == "away").sum()
    n_draw = (games["actual_3way"] == "draw").sum()
    print(f"  Results: {n_home} Home, {n_away} Away, {n_draw} OT/SO")

    # ── BLEND STAKING (what we actually bet) — per book ──
    blend_bets = completed[completed["is_bet"] == 1].copy()
    print(f"\n── Blend Strategy (ACTUAL BETS) " + "─" * 38)

    if blend_bets.empty:
        print("  No blend bets qualified.")
    else:
        # Per-book summary (each book has unique edges → unique blend bets)
        print(f"  {'Book':<16} {'Bets':>5} {'W-L':>7} {'Staked':>8} {'Profit':>8} {'ROI':>7}")
        print(f"  {'─'*16} {'─'*5} {'─'*7} {'─'*8} {'─'*8} {'─'*7}")

        for bk in sorted(blend_bets["bookmaker"].unique()):
            bk_rows = blend_bets[blend_bets["bookmaker"] == bk]
            n_b = len(bk_rows)
            n_w = int(bk_rows["bet_won"].sum())
            stk = bk_rows["kelly_stake"].sum()
            pnl = bk_rows["bet_profit"].sum()
            roi = pnl / stk * 100 if stk > 0 else 0
            wl = f"{n_w}-{n_b - n_w}"
            print(f"  {bk:<16} {n_b:>5} {wl:>7} {stk:>8.3f} {pnl:>+8.3f} {roi:>+6.1f}%")

        # Totals per side across all books
        print(f"  {'─'*55}")
        for side in ["draw", "home"]:
            s = blend_bets[blend_bets["bet_side"] == side]
            if len(s) > 0:
                sw = int(s["bet_won"].sum())
                sp = s["bet_profit"].sum()
                ss = s["kelly_stake"].sum()
                sr = sp / ss * 100 if ss > 0 else 0
                print(f"  {side.title():<16} {len(s):>5} {sw}-{len(s)-sw:>3} "
                      f"{ss:>8.3f} {sp:>+8.3f} {sr:>+6.1f}%")

        n_total = len(blend_bets)
        n_wins = int(blend_bets["bet_won"].sum())
        t_staked = blend_bets["kelly_stake"].sum()
        t_profit = blend_bets["bet_profit"].sum()
        t_roi = t_profit / t_staked * 100 if t_staked > 0 else 0
        print(f"  {'TOTAL':<16} {n_total:>5} {n_wins}-{n_total-n_wins:>3} "
              f"{t_staked:>8.3f} {t_profit:>+8.3f} {t_roi:>+6.1f}%")

        # Detail: every bet
        print(f"\n  {'Book':<16} {'Game':<12} {'Side':<6} {'Stake':>7} {'W/L':>5} {'Profit':>8}")
        print(f"  {'─'*16} {'─'*12} {'─'*6} {'─'*7} {'─'*5} {'─'*8}")
        for _, r in blend_bets.sort_values(["bookmaker", "game_id"]).iterrows():
            wl = "W" if r["bet_won"] == 1 else "L"
            game = f"{r['away_team']}@{r['home_team']}"
            print(f"  {r['bookmaker']:<16} {game:<12} {r['bet_side']:<6} "
                  f"{r['kelly_stake']:>7.3f} {wl:>5} {r['bet_profit']:>+8.3f}")

    # ── FLAT STAKING (1u per +edge, all games × all books) ──
    print(f"\n── Flat Staking (1u per +edge) " + "─" * 39)

    # Draw flat
    draw_flat = completed[completed["draw_won"].notna()].copy()
    if not draw_flat.empty:
        n_d = len(draw_flat)
        w_d = int(draw_flat["draw_won"].sum())
        p_d = draw_flat["draw_flat_profit"].sum()
        print(f"  Draw:  {n_d} bets, {w_d}-{n_d-w_d} ({w_d/n_d*100:.1f}%), "
              f"{p_d:+.2f}u, {p_d/n_d*100:+.1f}% ROI")
    else:
        print("  Draw: no +edge bets")

    # Home flat
    home_flat = completed[completed["home_won"].notna()].copy()
    if not home_flat.empty:
        n_h = len(home_flat)
        w_h = int(home_flat["home_won"].sum())
        p_h = home_flat["home_flat_profit"].sum()
        print(f"  Home:  {n_h} bets, {w_h}-{n_h-w_h} ({w_h/n_h*100:.1f}%), "
              f"{p_h:+.2f}u, {p_h/n_h*100:+.1f}% ROI")
    else:
        print("  Home: no +edge bets")

    # ── KELLY STAKING ──
    print(f"\n── Kelly Staking (¼K) " + "─" * 48)

    draw_kelly = completed[completed["draw_kelly_stake"] > 0].copy()
    if not draw_kelly.empty:
        n_dk = len(draw_kelly)
        w_dk = int(draw_kelly["draw_won"].sum())
        s_dk = draw_kelly["draw_kelly_stake"].sum()
        p_dk = draw_kelly["draw_kelly_profit"].sum()
        r_dk = p_dk / s_dk * 100 if s_dk > 0 else 0
        print(f"  Draw:  {n_dk} bets, {w_dk}-{n_dk-w_dk}, "
              f"staked {s_dk:.3f}u, {p_dk:+.3f}u, {r_dk:+.1f}% ROI")
    else:
        print("  Draw: no Kelly bets")

    home_kelly = completed[completed["home_kelly_stake"] > 0].copy()
    if not home_kelly.empty:
        n_hk = len(home_kelly)
        w_hk = int(home_kelly["home_won"].sum())
        s_hk = home_kelly["home_kelly_stake"].sum()
        p_hk = home_kelly["home_kelly_profit"].sum()
        r_hk = p_hk / s_hk * 100 if s_hk > 0 else 0
        print(f"  Home:  {n_hk} bets, {w_hk}-{n_hk-w_hk}, "
              f"staked {s_hk:.3f}u, {p_hk:+.3f}u, {r_hk:+.1f}% ROI")
    else:
        print("  Home: no Kelly bets")

    # ── OUTCOME CALIBRATION ──
    print(f"\n── Outcome Calibration " + "─" * 47)
    cal_games = completed.drop_duplicates("game_id")
    n_cal = len(cal_games)
    avg_p_home = cal_games["p_home_reg"].mean()
    avg_p_away = cal_games["p_away_reg"].mean()
    avg_p_ot = cal_games["p_ot"].mean()
    act_home = (cal_games["actual_3way"] == "home").mean()
    act_away = (cal_games["actual_3way"] == "away").mean()
    act_draw = (cal_games["actual_3way"] == "draw").mean()

    print(f"  {'Outcome':<10} {'Model':>8} {'Actual':>8} {'Diff':>8}  (n={n_cal} games)")
    print(f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}")
    print(f"  {'Home':<10} {avg_p_home:>7.1%} {act_home:>7.1%} {act_home-avg_p_home:>+7.1%}")
    print(f"  {'Away':<10} {avg_p_away:>7.1%} {act_away:>7.1%} {act_away-avg_p_away:>+7.1%}")
    print(f"  {'Draw/OT':<10} {avg_p_ot:>7.1%} {act_draw:>7.1%} {act_draw-avg_p_ot:>+7.1%}")

    # ── CUMULATIVE (if tracking file has multiple dates) ──
    if os.path.exists(TRACKING_FILE):
        all_data = pd.read_csv(TRACKING_FILE)
        all_dates = all_data["game_date"].nunique()
        if all_dates > 1:
            print(f"\n── Cumulative ({all_dates} dates) " + "─" * 40)
            all_completed = all_data[all_data["actual_3way"].notna()]
            all_bets = all_completed[all_completed["is_bet"] == 1]
            if not all_bets.empty:
                # Per-book cumulative
                print(f"  {'Book':<16} {'Bets':>5} {'W-L':>7} {'Staked':>8} {'Profit':>8} {'ROI':>7}")
                print(f"  {'─'*16} {'─'*5} {'─'*7} {'─'*8} {'─'*8} {'─'*7}")
                for bk in sorted(all_bets["bookmaker"].unique()):
                    bk_rows = all_bets[all_bets["bookmaker"] == bk]
                    n_b = len(bk_rows)
                    n_w = int(bk_rows["bet_won"].sum())
                    t_s = bk_rows["kelly_stake"].sum()
                    t_p = bk_rows["bet_profit"].sum()
                    r = t_p / t_s * 100 if t_s > 0 else 0
                    wl = f"{n_w}-{n_b - n_w}"
                    print(f"  {bk:<16} {n_b:>5} {wl:>7} {t_s:>8.3f} {t_p:>+8.3f} {r:>+6.1f}%")

    # ── Pending ──
    if not pending.empty:
        pending_games = pending.drop_duplicates("game_id")
        print(f"\n── Pending ({len(pending_games)} games) " + "─" * 40)
        for _, g in pending_games.iterrows():
            print(f"  {g['away_team']}@{g['home_team']}  (ID: {g['game_id']})")

    print()


# =============================================================================
# Main
# =============================================================================

def main(run_date=None):
    """Run the tracking pipeline for a specific date."""
    # Default to yesterday
    if not run_date:
        run_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    print()
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║             NHL 3-WAY MODEL — RESULTS TRACKER                      ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Tracking date: {run_date}")
    print()

    # Step 1: Refresh results
    if not SKIP_SCRAPING:
        step_refresh_results()
    else:
        print("  Skipping scraping (SKIP_SCRAPING=True)")
        print()

    # Step 2: Load predictions for this date
    print("=" * 70)
    print("STEP 2: Loading predictions")
    print("=" * 70)
    preds = load_predictions(run_date)
    if preds.empty:
        return
    n_games = preds["game_id"].nunique()
    n_books = preds["bookmaker"].nunique() if "bookmaker" in preds.columns else 1
    print(f"  {n_games} games × {n_books} books")
    print()

    # Step 3: Load game results
    print("=" * 70)
    print("STEP 3: Loading game results")
    print("=" * 70)
    results = load_game_results()
    # Show only relevant games
    relevant = results[results["game_id"].isin(preds["game_id"])]
    n_completed = len(relevant)
    n_total = preds["game_id"].nunique()
    print(f"  {n_completed}/{n_total} games completed")
    print()

    # Step 4: Grade predictions
    print("=" * 70)
    print("STEP 4: Grading predictions")
    print("=" * 70)
    graded = grade_predictions(preds, results)
    n_graded = graded["actual_3way"].notna().sum()
    n_pending = graded["actual_3way"].isna().sum()
    bet_rows = graded[graded["is_bet"] == 1]
    n_bet_games = bet_rows["game_id"].nunique() if len(bet_rows) > 0 else 0
    print(f"  {n_graded} graded, {n_pending} pending")
    print(f"  {n_bet_games} games with blend bets ({len(bet_rows)} book-rows)")
    print()

    # Step 5: Save (append)
    print("=" * 70)
    print("STEP 5: Saving tracking results")
    print("=" * 70)
    save_tracking_results(graded, run_date)
    print()

    # Step 6: Print summary
    print_summary(graded, run_date)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL 3-Way Model Results Tracker")
    parser.add_argument("--date", type=str, default=None,
                        help="Date to grade (YYYY-MM-DD). Default: yesterday.")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip game_ids refresh")
    args = parser.parse_args()

    # CLI args override module-level defaults
    if args.date:
        RUN_DATE = args.date
    if args.skip_scraping:
        SKIP_SCRAPING = True

    main(run_date=RUN_DATE)
