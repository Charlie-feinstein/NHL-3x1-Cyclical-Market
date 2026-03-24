# -*- coding: utf-8 -*-
"""
Walk-Forward Backtest — 3-Way Markets (Regulation-Time)

Runs a strict walk-forward backtest across multiple NHL seasons using:
  - Layer 1: Regulation quantile predictions (OOS: per-season cutoff)
  - Layer 2: OT edge predictions (OOS: per-season cutoff)
  - Layer 3: Poisson combiner with rolling calibration

Compares model 3-way probabilities (home reg / draw / away reg) against
3-way regulation-time market odds from the-odds-api.com.

Usage: Run in Spyder (Windows venv). Outputs:
  - backtest/backtest_results_3way.csv (game-by-game, all seasons)
  - Console: per-season + combined diagnostic & betting reports

@author: chazf
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root so we can import config and models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROJECT_DIR, PROCESSED_DIR, GAME_IDS_FILE, THREE_WAY_ODDS_FILE
sys.path.insert(0, PROJECT_DIR)

from models.poisson_combiner import (
    quantiles_to_lambda,
    compute_game_probabilities,
    apply_tie_calibration,
    compute_scoring_anchor,
    compute_home_ice_shift,
    estimate_poisson_rho,
    fit_platt_scaling,
    apply_platt_calibration,
    QUANTILE_COLS,
    QUANTILE_PROBS,
    ANCHOR_LOOKBACK_DAYS,
    ANCHOR_MIN_GAMES,
    ANCHOR_CLIP,
    TIE_LOOKBACK_DAYS,
    MAX_GOALS,
    LAMBDA_CLIP,
    DEFAULT_P_HOME_OT,
    TEAM_NAME_TO_ABBREV,
)

# Shared 3-way prediction logic (also used by predict_today.py)
from models.predict_3way import power_devig_3way

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_FILE = os.path.join(PROJECT_DIR, "backtest", "backtest_results_3way.csv")

# Test seasons: each entry defines one walk-forward season
TEST_SEASONS = [
    {
        "cutoff": "2024-10-01",
        "next_cutoff": "2025-10-01",
        "reg_file": os.path.join(PROCESSED_DIR, "regulation_predictions_2024.csv"),
        "ot_file": os.path.join(PROCESSED_DIR, "ot_predictions_2024.csv"),
        "label": "2024-25",
    },
    {
        "cutoff": "2025-10-01",
        "next_cutoff": None,
        "reg_file": os.path.join(PROCESSED_DIR, "regulation_predictions_2025.csv"),
        "ot_file": os.path.join(PROCESSED_DIR, "ot_predictions_2025.csv"),
        "label": "2025-26",
    },
]

PLATT_MIN_GAMES = 200
PLATT_ENABLED = False
HOME_ICE_ENABLED = True
LAMBDA_CAL_ENABLED = False
BIVARIATE_RHO_ENABLED = True
KELLY_FRACTION = 10.00
STARTING_BANKROLL = 1000.0

# OT model blending: 0.0 = pure constant (base rate), 1.0 = pure model
OT_BLEND_WEIGHT = 0.0

# Tie inflation mode:
#   "fixed"    = hard-coded constant (1.35)
#   "rolling"  = 120-day rolling window (ratio of actual OT to Poisson P(tie))
#   "bayesian" = season-to-date rate, shrunk toward historical prior
TIE_INFLATION_MODE = "bayesian"
TIE_INFLATION_FIXED = 1.35  # ~0.22 actual / ~0.163 raw Poisson (5-season avg)

# Bayesian prior for tie inflation: computed from pre-cutoff data at runtime
# (no more hardcoded constants)


# =============================================================================
# 3-Way Odds Loading
# =============================================================================

def load_three_way_odds():
    """Load 3-way regulation-time odds from ALL bookmakers, devig each.

    Returns: dict[(date_str, home_abbrev)] → list of {
        home_dec, draw_dec, away_dec,
        home_fair, draw_fair, away_fair,
        bookmaker
    }
    Each game key maps to a list (one entry per bookmaker).
    """
    if not os.path.exists(THREE_WAY_ODDS_FILE):
        print("  WARNING: three_way_odds.csv not found")
        return {}

    odds = pd.read_csv(THREE_WAY_ODDS_FILE)
    print(f"  3-way odds rows loaded: {len(odds):,}")

    # Ensure numeric
    for col in ["home_dec", "draw_dec", "away_dec"]:
        odds[col] = pd.to_numeric(odds[col], errors="coerce")
    odds = odds.dropna(subset=["home_dec", "draw_dec", "away_dec"])

    # Filter out live/in-play odds (matinee games caught mid-game)
    # Real pre-game 3-way odds: home/away typically 1.5-15, draw 3-8
    live_mask = (odds["home_dec"] < 1.10) | (odds["away_dec"] < 1.10) | (odds["draw_dec"] > 15)
    n_live = live_mask.sum()
    if n_live > 0:
        print(f"  Filtered {n_live} rows with live/in-play odds")
        odds = odds[~live_mask]

    # Ensure bookmaker column exists
    if "bookmaker" not in odds.columns:
        odds["bookmaker"] = "unknown"

    # Deduplicate: keep one row per (date, home_team, bookmaker)
    odds = odds.drop_duplicates(
        subset=["game_date", "home_team", "bookmaker"], keep="first"
    )

    # Convert decimal odds to implied probs and devig
    odds["home_imp"] = 1.0 / odds["home_dec"]
    odds["draw_imp"] = 1.0 / odds["draw_dec"]
    odds["away_imp"] = 1.0 / odds["away_dec"]

    result = {}
    for _, row in odds.iterrows():
        fair_h, fair_d, fair_a = power_devig_3way(
            row["home_imp"], row["draw_imp"], row["away_imp"]
        )
        key = (str(row["game_date"]), row["home_team"])
        entry = {
            "home_dec": row["home_dec"],
            "draw_dec": row["draw_dec"],
            "away_dec": row["away_dec"],
            "home_fair": fair_h,
            "draw_fair": fair_d,
            "away_fair": fair_a,
            "bookmaker": row["bookmaker"],
        }
        if key not in result:
            result[key] = []
        result[key].append(entry)

    n_books = sum(len(v) for v in result.values())
    books = odds["bookmaker"].value_counts()
    print(f"  3-way odds: {len(result):,} games, {n_books:,} book-rows")
    print(f"  Books: {dict(books)}")
    return result


# =============================================================================
# Walk-Forward Engine (Single Season)
# =============================================================================

def run_season_walk_forward(season_cfg, completed, odds_3way):
    """Run walk-forward for a single test season (3-way markets).

    Args:
        season_cfg: dict with cutoff, next_cutoff, reg_file, ot_file, label
        completed: DataFrame of completed games with outcomes
        odds_3way: dict of 3-way odds by (date, home_abbrev)

    Returns:
        DataFrame of per-game results for this season
    """
    label = season_cfg["label"]
    cutoff = pd.Timestamp(season_cfg["cutoff"])
    next_cutoff = pd.Timestamp(season_cfg["next_cutoff"]) if season_cfg["next_cutoff"] else None

    print(f"\n{'='*60}")
    print(f"SEASON: {label}  (cutoff={cutoff.date()}, "
          f"end={'end of data' if next_cutoff is None else next_cutoff.date()})")
    print(f"{'='*60}")

    # --- Load regulation predictions for this cutoff ---
    reg_file = season_cfg["reg_file"]
    if not os.path.exists(reg_file):
        print(f"  WARNING: {reg_file} not found — skipping {label}")
        return pd.DataFrame()

    reg = pd.read_csv(reg_file)
    reg["game_date"] = pd.to_datetime(reg["game_date"])
    print(f"  Regulation predictions: {len(reg):,} rows from {reg_file}")

    # --- Load OT predictions for this cutoff ---
    ot_file = season_cfg["ot_file"]
    ot_map = {}
    if os.path.exists(ot_file):
        ot = pd.read_csv(ot_file)
        ot["game_date"] = pd.to_datetime(ot["game_date"])
        ot_map = dict(zip(ot["game_id"], ot["p_home_win_ot"]))
        print(f"  OT predictions: {len(ot):,} rows from {ot_file}")
    else:
        print(f"  WARNING: {ot_file} not found — using constant P(home|OT)")

    # --- Compute lambdas ---
    reg["lam_raw"] = reg[QUANTILE_COLS].apply(
        lambda row: quantiles_to_lambda(row.values), axis=1
    )

    # --- Pair home/away ---
    home = reg[reg["is_home"] == 1][
        ["game_id", "game_date", "team", "opponent", "season", "reg_gf", "lam_raw"]
    ].copy()
    home.columns = ["game_id", "game_date", "home_team", "away_team",
                     "season", "home_reg_gf", "lam_home_raw"]

    away = reg[reg["is_home"] == 0][
        ["game_id", "reg_gf", "lam_raw"]
    ].copy()
    away.columns = ["game_id", "away_reg_gf", "lam_away_raw"]

    paired = home.merge(away, on="game_id", how="inner")
    print(f"  Paired games: {len(paired):,}")

    # --- Merge outcomes ---
    outcome_cols = ["game_id", "game_outcome_type", "is_ot", "home_won_actual"]
    paired = paired.merge(completed[outcome_cols], on="game_id", how="inner")

    # --- Merge OT predictions ---
    paired["p_home_win_ot_model"] = paired["game_id"].map(ot_map)
    paired = paired.sort_values("game_date").reset_index(drop=True)
    print(f"  Paired with outcomes: {len(paired):,}")

    # --- Split history vs test ---
    history = paired[paired["game_date"] < cutoff].copy()
    if next_cutoff is not None:
        test = paired[(paired["game_date"] >= cutoff) &
                      (paired["game_date"] < next_cutoff)].copy()
    else:
        test = paired[paired["game_date"] >= cutoff].copy()

    print(f"  History (pre-{cutoff.date()}): {len(history):,} games")
    print(f"  Test ({label}):               {len(test):,} games")

    if len(test) == 0:
        print(f"  WARNING: No test games for {label}")
        return pd.DataFrame()

    # --- Compute OT base rate from history ---
    base_rate_ot = history["home_won_actual"][
        history["game_outcome_type"].isin(["OT", "SO"])
    ].mean() if len(history[history["game_outcome_type"].isin(["OT", "SO"])]) > 0 else DEFAULT_P_HOME_OT

    print(f"  OT home win base rate (history): {base_rate_ot:.3f}")
    print(f"  OT blend weight: {OT_BLEND_WEIGHT} "
          f"({'pure constant' if OT_BLEND_WEIGHT == 0.0 else 'pure model' if OT_BLEND_WEIGHT == 1.0 else 'blended'})")

    # --- Build anchor DataFrames from history ---
    home_anchor_records = []
    away_anchor_records = []

    for _, row in history.iterrows():
        home_anchor_records.append({
            "game_date": row["game_date"],
            "reg_gf": row["home_reg_gf"],
            "lam_raw": row["lam_home_raw"],
        })
        away_anchor_records.append({
            "game_date": row["game_date"],
            "reg_gf": row["away_reg_gf"],
            "lam_raw": row["lam_away_raw"],
        })

    # --- Bivariate Poisson rho ---
    poisson_rho = 0.0
    if BIVARIATE_RHO_ENABLED and len(history) > 100:
        poisson_rho = estimate_poisson_rho(history)
        print(f"  Bivariate Poisson rho: {poisson_rho:.4f} "
              f"(from {len(history)} history games)")

    # --- Build tie-inflation history ---
    tie_records = []
    for _, row in history.iterrows():
        _, _, p_tie_raw = compute_game_probabilities(
            row["lam_home_raw"], row["lam_away_raw"], rho=poisson_rho)
        tie_records.append({
            "game_date": row["game_date"],
            "is_ot": row["is_ot"],
            "p_tie_raw": p_tie_raw,
        })

    # --- Bayesian prior: computed from pre-cutoff data ---
    # Cap at MAX_PRIOR_GAMES so the season retains meaningful influence
    from models.predict_3way import MAX_PRIOR_GAMES
    if tie_records:
        tie_df_prior = pd.DataFrame(tie_records)
        bayesian_prior_ot_rate = tie_df_prior["is_ot"].mean()
        bayesian_prior_ptie = tie_df_prior["p_tie_raw"].mean()
        bayesian_prior_games = min(len(tie_df_prior), MAX_PRIOR_GAMES)
        print(f"  Bayesian prior (from {len(tie_df_prior)} games, capped at {bayesian_prior_games}): "
              f"OT rate={bayesian_prior_ot_rate:.4f}, "
              f"raw P(tie)={bayesian_prior_ptie:.4f}")
    else:
        bayesian_prior_ot_rate = 0.224
        bayesian_prior_ptie = 0.163
        bayesian_prior_games = 0

    # --- OOS Platt records (still tracked for consistency) ---
    platt_records = []

    # --- Home-ice shift for 3-way reg split (from reg-only outcomes) ---
    home_ice_shift = 0.0
    if HOME_ICE_ENABLED and len(history) > 100:
        reg_only = history[history["game_outcome_type"] == "REG"]
        if len(reg_only) > 50:
            # Actual home share of regulation wins
            actual_home_share = np.clip(reg_only["home_won_actual"].mean(), 0.01, 0.99)

            # Model's predicted home share of regulation (from raw Poisson)
            model_shares = []
            for _, row in reg_only.iterrows():
                p_hr, p_ar, _ = compute_game_probabilities(
                    row["lam_home_raw"], row["lam_away_raw"], rho=poisson_rho)
                reg_total = p_hr + p_ar
                model_shares.append(p_hr / reg_total if reg_total > 0 else 0.5)
            pred_home_share = np.clip(np.mean(model_shares), 0.01, 0.99)

            home_ice_shift = (
                np.log(actual_home_share / (1 - actual_home_share))
                - np.log(pred_home_share / (1 - pred_home_share))
            )
            print(f"  Home-ice logit shift (3-way reg): {home_ice_shift:+.4f} "
                  f"(actual={actual_home_share:.4f}, model={pred_home_share:.4f}, "
                  f"from {len(reg_only)} reg games)")

    # --- Lambda calibration (disabled by default) ---
    lam_cal_alpha_h, lam_cal_beta_h = 0.0, 1.0
    lam_cal_alpha_a, lam_cal_beta_a = 0.0, 1.0
    if LAMBDA_CAL_ENABLED and len(history) > 100:
        x_h = history["lam_home_raw"].values
        y_h = history["home_reg_gf"].values
        lam_cal_beta_h = np.cov(x_h, y_h)[0, 1] / np.var(x_h)
        lam_cal_alpha_h = np.mean(y_h) - lam_cal_beta_h * np.mean(x_h)

        x_a = history["lam_away_raw"].values
        y_a = history["away_reg_gf"].values
        lam_cal_beta_a = np.cov(x_a, y_a)[0, 1] / np.var(x_a)
        lam_cal_alpha_a = np.mean(y_a) - lam_cal_beta_a * np.mean(x_a)

    # --- Process each test game ---
    results = []
    test_dates = sorted(test["game_date"].unique())
    n_dates = len(test_dates)

    for date_idx, game_date in enumerate(test_dates):
        day_games = test[test["game_date"] == game_date]

        if (date_idx + 1) % 50 == 0 or date_idx == 0:
            print(f"  [{label}] Processing date {date_idx + 1}/{n_dates}: "
                  f"{game_date.date()} ({len(day_games)} games, "
                  f"{len(platt_records)} OOS games so far)")

        # Build anchor DataFrames for this date
        home_anchor_df = pd.DataFrame(home_anchor_records)
        away_anchor_df = pd.DataFrame(away_anchor_records)

        # Scoring anchor (60-day lookback)
        anchor_home = compute_scoring_anchor(home_anchor_df, game_date)
        anchor_away = compute_scoring_anchor(away_anchor_df, game_date)

        # Tie inflation
        if TIE_INFLATION_MODE == "fixed":
            tie_inflation = TIE_INFLATION_FIXED
        elif TIE_INFLATION_MODE == "bayesian":
            # Season-to-date OT rate, shrunk toward data-driven prior.
            # Prior = pre-cutoff history. Likelihood = current season.
            tie_df = pd.DataFrame(tie_records)
            season_games = tie_df[
                (tie_df["game_date"] >= cutoff) & (tie_df["game_date"] < game_date)
            ]
            n_season = len(season_games)
            if n_season >= 20:
                season_ot_rate = season_games["is_ot"].mean()
                season_ptie_mean = season_games["p_tie_raw"].mean()
            else:
                season_ot_rate = bayesian_prior_ot_rate
                season_ptie_mean = bayesian_prior_ptie

            # Bayesian posterior: blend season rate with prior
            n_ot = season_ot_rate * n_season
            shrunk_ot_rate = (
                (bayesian_prior_ot_rate * bayesian_prior_games + n_ot)
                / (bayesian_prior_games + n_season)
            )
            # Convert shrunk OT rate to tie inflation ratio
            tie_inflation = shrunk_ot_rate / season_ptie_mean if season_ptie_mean > 0 else TIE_INFLATION_FIXED
        else:
            # Rolling window mode
            tie_df = pd.DataFrame(tie_records)
            cutoff_tie = game_date - pd.Timedelta(days=TIE_LOOKBACK_DAYS)
            tie_window = tie_df[
                (tie_df["game_date"] >= cutoff_tie) & (tie_df["game_date"] < game_date)
            ]
            if len(tie_window) >= 50:
                actual_ot_rate = tie_window["is_ot"].mean()
                pred_tie_mean = tie_window["p_tie_raw"].mean()
                tie_inflation = actual_ot_rate / pred_tie_mean if pred_tie_mean > 0 else 1.0
            else:
                all_ot_rate = tie_df["is_ot"].mean()
                all_tie_mean = tie_df["p_tie_raw"].mean()
                tie_inflation = all_ot_rate / all_tie_mean if all_tie_mean > 0 else 1.0

        # Platt calibration (OOS-only, tracked but unused in 3-way)
        platt_a, platt_b = 1.0, 0.0
        use_platt = PLATT_ENABLED and len(platt_records) >= PLATT_MIN_GAMES
        if use_platt:
            platt_arr = np.array(platt_records)
            p_raw_hist = np.clip(platt_arr[:, 0], 0.01, 0.99)
            y_hist = platt_arr[:, 1]
            logit_p = np.log(p_raw_hist / (1 - p_raw_hist))
            a, b = fit_platt_scaling(logit_p, y_hist)
            if 0.5 <= a <= 5.0:
                platt_a, platt_b = a, b

        # Process each game on this date
        for _, game in day_games.iterrows():
            game_id = game["game_id"]

            # Adjust lambdas
            lam_home = game["lam_home_raw"] * anchor_home
            lam_away = game["lam_away_raw"] * anchor_away

            # Lambda calibration (regression-based shrinkage)
            if LAMBDA_CAL_ENABLED:
                lam_home = lam_cal_alpha_h + lam_cal_beta_h * lam_home
                lam_away = lam_cal_alpha_a + lam_cal_beta_a * lam_away

            lam_home = np.clip(lam_home, *LAMBDA_CLIP)
            lam_away = np.clip(lam_away, *LAMBDA_CLIP)

            # Poisson probabilities
            p_home_reg, p_away_reg, p_tie_raw = compute_game_probabilities(
                lam_home, lam_away, rho=poisson_rho)

            # Tie inflation → 3-way model probabilities
            p_home_reg_cal, p_away_reg_cal, p_ot = apply_tie_calibration(
                p_home_reg, p_away_reg, p_tie_raw, tie_inflation
            )

            # Home-ice correction: shift the reg home/away split in logit space
            # (same shift used in 2-way, applied to reg-only share here)
            if home_ice_shift != 0.0:
                reg_total = p_home_reg_cal + p_away_reg_cal
                if reg_total > 0:
                    home_share = p_home_reg_cal / reg_total
                    logit_share = np.log(max(home_share, 1e-6) / max(1 - home_share, 1e-6))
                    home_share_adj = 1.0 / (1.0 + np.exp(-(logit_share + home_ice_shift)))
                    p_home_reg_cal = reg_total * home_share_adj
                    p_away_reg_cal = reg_total * (1.0 - home_share_adj)

            # --- 3-Way: Model probabilities ---
            # p_home_reg_cal = P(home wins in regulation)
            # p_away_reg_cal = P(away wins in regulation)
            # p_ot = P(draw / game goes to OT or SO)
            # These sum to ~1.0

            # --- Actual 3-way outcome (same for all books) ---
            home_won = game["home_won_actual"]
            outcome_type = game["game_outcome_type"]

            if outcome_type in ("OT", "SO"):
                actual_3way = "draw"
            elif home_won == 1:
                actual_3way = "home"
            else:
                actual_3way = "away"

            # --- Match 3-way odds: one result row per book ---
            date_str = game_date.strftime("%Y-%m-%d")
            key = (date_str, game["home_team"])
            book_list = odds_3way.get(key, [])

            # If no odds at all, emit one row with NaN odds
            if not book_list:
                book_list = [None]

            for o in book_list:
                if o is not None:
                    fair_home = o["home_fair"]
                    fair_draw = o["draw_fair"]
                    fair_away = o["away_fair"]
                    dec_home = o["home_dec"]
                    dec_draw = o["draw_dec"]
                    dec_away = o["away_dec"]
                    bookmaker = o["bookmaker"]
                else:
                    fair_home = fair_draw = fair_away = np.nan
                    dec_home = dec_draw = dec_away = np.nan
                    bookmaker = ""

                # Compute 3-way edges
                home_edge = p_home_reg_cal - fair_home if not np.isnan(fair_home) else np.nan
                draw_edge = p_ot - fair_draw if not np.isnan(fair_draw) else np.nan
                away_edge = p_away_reg_cal - fair_away if not np.isnan(fair_away) else np.nan

                # Bet selection: draw only when +edge
                bet_side = None
                bet_edge = np.nan
                bet_odds_dec = np.nan
                bet_model_prob = np.nan
                kelly_stake = 0.0

                if not np.isnan(draw_edge) and draw_edge > 0:
                    bet_side = "draw"
                    bet_edge = draw_edge
                    bet_odds_dec = dec_draw
                    bet_model_prob = p_ot

                    # Kelly sizing
                    b = bet_odds_dec - 1.0
                    p = bet_model_prob
                    q = 1.0 - p
                    kelly_full = (b * p - q) / b if b > 0 else 0.0
                    kelly_stake = max(0.0, kelly_full * KELLY_FRACTION)

                bet_won = np.nan
                bet_profit = 0.0
                if bet_side is not None:
                    bet_won = int(bet_side == actual_3way)
                    bet_profit = kelly_stake * (bet_odds_dec - 1.0) if bet_won else -kelly_stake

                results.append({
                    "game_id": game_id,
                    "game_date": game_date,
                    "season_label": label,
                    "home_team": game["home_team"],
                    "away_team": game["away_team"],
                    "lam_home": lam_home,
                    "lam_away": lam_away,
                    "anchor_home": anchor_home,
                    "anchor_away": anchor_away,
                    "p_home_reg_win": p_home_reg_cal,
                    "p_away_reg_win": p_away_reg_cal,
                    "p_ot": p_ot,
                    "home_ice_shift": home_ice_shift,
                    "poisson_rho": poisson_rho,
                    "tie_inflation": tie_inflation,
                    # 3-way market
                    "fair_home_3w": fair_home,
                    "fair_draw_3w": fair_draw,
                    "fair_away_3w": fair_away,
                    "dec_home_3w": dec_home,
                    "dec_draw_3w": dec_draw,
                    "dec_away_3w": dec_away,
                    "bookmaker": bookmaker,
                    # Edges
                    "home_edge": home_edge,
                    "draw_edge": draw_edge,
                    "away_edge": away_edge,
                    # Betting
                    "bet_side": bet_side,
                    "bet_edge": bet_edge,
                    "bet_odds_dec": bet_odds_dec,
                    "bet_model_prob": bet_model_prob,
                    "kelly_stake": kelly_stake,
                    # Outcomes
                    "actual_3way": actual_3way,
                    "home_won_actual": home_won,
                    "game_outcome_type": outcome_type,
                    "bet_won": bet_won,
                    "bet_profit": bet_profit,
                })

            # Update rolling records (once per game, not per book)
            home_anchor_records.append({
                "game_date": game_date,
                "reg_gf": game["home_reg_gf"],
                "lam_raw": game["lam_home_raw"],
            })
            away_anchor_records.append({
                "game_date": game_date,
                "reg_gf": game["away_reg_gf"],
                "lam_raw": game["lam_away_raw"],
            })
            tie_records.append({
                "game_date": game_date,
                "is_ot": game["is_ot"],
                "p_tie_raw": p_tie_raw,
            })
            # Track Platt for 2-way consistency (p_home_win used for diagnostics)
            p_home_win_raw = p_home_reg_cal + p_ot * base_rate_ot
            platt_records.append([p_home_win_raw, home_won])

    results_df = pd.DataFrame(results)
    print(f"\n  [{label}] Walk-forward complete: {len(results_df)} games processed")

    return results_df


# =============================================================================
# Walk-Forward Engine (Multi-Season)
# =============================================================================

def run_walk_forward():
    """Main walk-forward backtest pipeline across all test seasons (3-way)."""

    print(f"\n{'='*60}")
    print("STEP 1: Load Shared Data")
    print(f"{'='*60}")

    # Game IDs (completed only)
    games = pd.read_csv(GAME_IDS_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    completed = games[games["game_state"] == "OFF"].copy()
    completed["is_ot"] = completed["game_outcome_type"].isin(["OT", "SO"]).astype(int)
    completed["home_won_actual"] = (completed["home_score"] > completed["away_score"]).astype(int)
    print(f"  Completed games: {len(completed):,}")

    # Load 3-way odds
    odds_3way = load_three_way_odds()

    # Run walk-forward for each test season
    all_results = []
    for season_cfg in TEST_SEASONS:
        season_df = run_season_walk_forward(season_cfg, completed, odds_3way)
        if len(season_df) > 0:
            all_results.append(season_df)

    if not all_results:
        print("\nWARNING: No results from any season!")
        return pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)
    results_df = results_df.sort_values("game_date").reset_index(drop=True)
    n_unique_games = results_df["game_id"].nunique()
    n_books = results_df["bookmaker"].nunique() if "bookmaker" in results_df.columns else 1
    print(f"\n{'='*60}")
    print(f"COMBINED: {len(results_df)} rows ({n_unique_games} unique games, "
          f"{n_books} books) across {len(all_results)} season(s)")
    print(f"{'='*60}")

    return results_df


# =============================================================================
# Reporting — 3-Way
# =============================================================================

def print_report(df, label="Combined"):
    """Diagnostic report for 3-way backtest."""

    print(f"\n{'='*60}")
    print(f"3-WAY BACKTEST DIAGNOSTICS — {label}")
    print(f"{'='*60}")
    n_unique = df["game_id"].nunique()
    print(f"  Total rows: {len(df)} ({n_unique} unique games)")
    if "bookmaker" in df.columns:
        books = df[df["bookmaker"] != ""]["bookmaker"].nunique()
        print(f"  Bookmakers: {books}")
    print(f"  Date range: {df['game_date'].min().date()} to "
          f"{df['game_date'].max().date()}")

    # --- Outcome distribution (unique games only) ---
    unique_games = df.drop_duplicates(subset=["game_id"])
    print(f"\n  Actual 3-way outcomes ({len(unique_games)} games):")
    for outcome in ["home", "draw", "away"]:
        n = (unique_games["actual_3way"] == outcome).sum()
        pct = n / len(unique_games) * 100
        print(f"    {outcome:>5}: {n:4d} ({pct:.1f}%)")

    # --- Model probability calibration ---
    print(f"\n  Model probability calibration (mean):")
    with_odds = df.dropna(subset=["fair_home_3w"])
    for col, fair_col, actual_val, name in [
        ("p_home_reg_win", "fair_home_3w", "home", "Home reg"),
        ("p_ot", "fair_draw_3w", "draw", "Draw (OT)"),
        ("p_away_reg_win", "fair_away_3w", "away", "Away reg"),
    ]:
        model_mean = df[col].mean()
        actual_rate = (df["actual_3way"] == actual_val).mean()
        if len(with_odds) > 0:
            market_mean = with_odds[fair_col].mean()
            print(f"    {name:>10}: model={model_mean:.3f}, actual={actual_rate:.3f}, "
                  f"market={market_mean:.3f}, edge={model_mean - market_mean:+.3f}")
        else:
            print(f"    {name:>10}: model={model_mean:.3f}, actual={actual_rate:.3f}")

    # --- Lambda bias ---
    print(f"\n  Lambda bias:")
    print(f"    Home mean: {df['lam_home'].mean():.3f}")
    print(f"    Away mean: {df['lam_away'].mean():.3f}")

    # --- Tie inflation ---
    print(f"\n  Tie inflation:")
    print(f"    Mean: {df['tie_inflation'].mean():.3f}")
    print(f"    Range: [{df['tie_inflation'].min():.3f}, {df['tie_inflation'].max():.3f}]")

    # --- Edge distribution ---
    with_edges = df.dropna(subset=["home_edge"])
    if len(with_edges) > 0:
        print(f"\n  Edge distribution ({len(with_edges)} games with odds):")
        for col, name in [("home_edge", "Home reg"), ("draw_edge", "Draw"), ("away_edge", "Away reg")]:
            e = with_edges[col]
            print(f"    {name:>10}: mean={e.mean():+.4f}, std={e.std():.4f}, "
                  f"|>3%|={( e.abs() > 0.03).sum()}")

    # --- Monthly breakdown ---
    print(f"\n  Monthly P(OT) vs actual:")
    df_copy = df.copy()
    df_copy["month"] = df_copy["game_date"].dt.to_period("M")
    df_copy["is_draw"] = (df_copy["actual_3way"] == "draw").astype(int)
    for month, grp in df_copy.groupby("month"):
        pred_ot = grp["p_ot"].mean()
        actual_ot = grp["is_draw"].mean()
        print(f"    {month}: pred={pred_ot:.3f}, actual={actual_ot:.3f}, "
              f"diff={actual_ot - pred_ot:+.3f}, n={len(grp)}")


def print_betting_report(df, label="Combined", thresholds=None):
    """Betting simulation report for 3-way markets."""
    if thresholds is None:
        thresholds = [0.0, 0.02, 0.05]

    print(f"\n{'='*60}")
    print(f"3-WAY BETTING SIMULATION — {label}")
    print(f"{'='*60}")
    print(f"  Kelly fraction: {KELLY_FRACTION}")
    print(f"  Starting bankroll: {STARTING_BANKROLL} units")

    with_odds = df.dropna(subset=["fair_home_3w"]).copy()
    n_unique = with_odds["game_id"].nunique() if len(with_odds) > 0 else 0
    print(f"  Rows with 3-way odds: {len(with_odds)} ({n_unique} unique games)")
    if "bookmaker" in with_odds.columns and len(with_odds) > 0:
        print(f"  Books: {dict(with_odds['bookmaker'].value_counts())}")

    has_bet = with_odds[with_odds["bet_side"].notna()]
    has_stake = has_bet[has_bet["kelly_stake"] > 0]
    print(f"  Games with any bet: {len(has_bet)}")
    print(f"  Games with positive Kelly stake: {len(has_stake)}")

    for threshold in thresholds:
        print(f"\n  {'─'*50}")
        print(f"  Edge threshold: {threshold*100:.0f}%")
        print(f"  {'─'*50}")

        bets = with_odds[
            (with_odds["bet_side"].notna()) &
            (with_odds["bet_edge"] >= threshold) &
            (with_odds["kelly_stake"] > 0)
        ].copy()

        if len(bets) == 0:
            print(f"    No bets placed at this threshold")
            continue

        n_bets = len(bets)
        n_wins = int(bets["bet_won"].sum())
        win_rate = n_wins / n_bets

        total_staked = bets["kelly_stake"].sum()
        total_profit = bets["bet_profit"].sum()
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0

        avg_edge = bets["bet_edge"].mean()
        avg_odds = bets["bet_odds_dec"].mean()
        avg_stake = bets["kelly_stake"].mean()

        # Max drawdown
        cum_pnl = bets["bet_profit"].cumsum()
        running_max = cum_pnl.cummax()
        max_dd = (cum_pnl - running_max).min()

        final_bankroll = STARTING_BANKROLL + total_profit

        print(f"    Bets placed:    {n_bets}")
        print(f"    Win rate:       {win_rate:.3f} ({n_wins}/{n_bets})")
        print(f"    Avg edge:       {avg_edge:.4f} ({avg_edge*100:.1f}%)")
        print(f"    Avg odds (dec): {avg_odds:.3f}")
        print(f"    Avg stake:      {avg_stake:.4f} units")
        print(f"    Total staked:   {total_staked:.2f} units")
        print(f"    Net profit:     {total_profit:+.2f} units")
        print(f"    ROI:            {roi:+.1f}%")
        print(f"    Max drawdown:   {max_dd:.2f} units")
        print(f"    Final bankroll: {final_bankroll:.2f} units")

        # Bet side breakdown
        print(f"\n    By bet side:")
        for side in ["home", "draw", "away"]:
            side_bets = bets[bets["bet_side"] == side]
            if len(side_bets) == 0:
                print(f"      {side:>5}: 0 bets")
                continue
            s_n = len(side_bets)
            s_w = int(side_bets["bet_won"].sum())
            s_staked = side_bets["kelly_stake"].sum()
            s_profit = side_bets["bet_profit"].sum()
            s_roi = s_profit / s_staked * 100 if s_staked > 0 else 0
            s_avg_edge = side_bets["bet_edge"].mean()
            s_avg_odds = side_bets["bet_odds_dec"].mean()
            print(f"      {side:>5}: {s_n} bets, {s_w}W ({s_w/s_n:.3f}), "
                  f"ROI={s_roi:+.1f}%, edge={s_avg_edge:.3f}, "
                  f"odds={s_avg_odds:.2f}, P&L={s_profit:+.2f}")

        # Monthly P&L
        print(f"\n    Monthly P&L:")
        bets["month"] = bets["game_date"].dt.to_period("M")
        for month, grp in bets.groupby("month"):
            m_bets = len(grp)
            m_wins = int(grp["bet_won"].sum())
            m_staked = grp["kelly_stake"].sum()
            m_profit = grp["bet_profit"].sum()
            m_roi = m_profit / m_staked * 100 if m_staked > 0 else 0
            # Side counts
            m_home = (grp["bet_side"] == "home").sum()
            m_draw = (grp["bet_side"] == "draw").sum()
            m_away = (grp["bet_side"] == "away").sum()
            print(f"      {month}: {m_bets} bets ({m_home}H/{m_draw}D/{m_away}A), "
                  f"{m_wins}W, P&L={m_profit:+.2f}, ROI={m_roi:+.1f}%")

        # --- Flat staking ---
        flat_bets = with_odds[
            (with_odds["bet_side"].notna()) &
            (with_odds["bet_edge"] >= threshold)
        ].copy()

        if len(flat_bets) > 0:
            flat_bets["flat_profit"] = flat_bets.apply(
                lambda row: (row["bet_odds_dec"] - 1.0) if row["bet_won"] == 1
                else -1.0, axis=1
            )

            flat_n = len(flat_bets)
            flat_wins = int(flat_bets["bet_won"].sum())
            flat_profit = flat_bets["flat_profit"].sum()
            flat_roi = flat_profit / flat_n * 100

            flat_cum = flat_bets["flat_profit"].cumsum()
            flat_max_dd = (flat_cum - flat_cum.cummax()).min()

            print(f"\n    Flat staking (1 unit/bet):")
            print(f"      Bets: {flat_n}, Wins: {flat_wins} ({flat_wins/flat_n:.3f})")
            print(f"      P&L: {flat_profit:+.2f}, ROI: {flat_roi:+.2f}%")
            print(f"      Max DD: {flat_max_dd:.2f}")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Walk-Forward Backtest — 3-Way Markets (NHL)")
    print("=" * 60)

    results_df = run_walk_forward()

    if len(results_df) == 0:
        print("\nNo results to report.")
        sys.exit(0)

    # --- Per-season reports ---
    for season_label in results_df["season_label"].unique():
        season_df = results_df[results_df["season_label"] == season_label]
        print_report(season_df, label=season_label)
        print_betting_report(season_df, label=season_label, thresholds=[0.0, 0.02, 0.05])

    # --- Combined report ---
    if results_df["season_label"].nunique() > 1:
        print_report(results_df, label="COMBINED")
        print_betting_report(results_df, label="COMBINED", thresholds=[0.0, 0.02, 0.05])

    # --- Save ---
    print(f"\n{'='*60}")
    print("SAVING RESULTS")
    print(f"{'='*60}")

    output = results_df.copy()
    output["game_date"] = output["game_date"].dt.strftime("%Y-%m-%d")
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"  Saved {len(output)} rows to {OUTPUT_FILE}")

    # --- Verification ---
    print(f"\n{'='*60}")
    print("VERIFICATION")
    print(f"{'='*60}")

    n_rows = len(results_df)
    n_unique = results_df["game_id"].nunique()
    n_with_odds = results_df["fair_home_3w"].notna().sum()
    n_no_odds = n_rows - n_with_odds

    print(f"  Total rows: {n_rows} ({n_unique} unique games)")
    if "bookmaker" in results_df.columns:
        book_counts = results_df[results_df["bookmaker"] != ""]["bookmaker"].value_counts()
        print(f"  Books: {dict(book_counts)}")
    for sl in results_df["season_label"].unique():
        mask = results_df["season_label"] == sl
        n_s = results_df.loc[mask, "game_id"].nunique()
        n_r = mask.sum()
        n_o = results_df.loc[mask, "fair_home_3w"].notna().sum()
        print(f"    {sl}: {n_s} games, {n_r} rows, {n_o} with 3-way odds")
    print(f"  Rows with 3-way odds: {n_with_odds}")
    print(f"  Rows without odds: {n_no_odds}")

    # Check 3-way prob sums
    prob_sum = results_df["p_home_reg_win"] + results_df["p_away_reg_win"] + results_df["p_ot"]
    print(f"  Model prob sum: mean={prob_sum.mean():.4f}, "
          f"range=[{prob_sum.min():.4f}, {prob_sum.max():.4f}]")

    with_odds = results_df.dropna(subset=["fair_home_3w"])
    if len(with_odds) > 0:
        fair_sum = with_odds["fair_home_3w"] + with_odds["fair_draw_3w"] + with_odds["fair_away_3w"]
        print(f"  Market fair prob sum: mean={fair_sum.mean():.4f}")

    # Bet side distribution
    bet_counts = results_df["bet_side"].value_counts()
    print(f"\n  Bet side distribution:")
    for side in ["home", "draw", "away"]:
        n = bet_counts.get(side, 0)
        print(f"    {side:>5}: {n}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
