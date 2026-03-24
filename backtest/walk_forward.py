# -*- coding: utf-8 -*-
"""
Walk-Forward Backtest — Multi-Season (Leakage-Free)

Runs a strict walk-forward backtest across multiple NHL seasons using:
  - Layer 1: Regulation quantile predictions (OOS: per-season cutoff)
  - Layer 2: OT edge predictions (OOS: per-season cutoff)
  - Layer 3: Poisson combiner with rolling calibration

Each test season uses its own model (trained on data before that season's
cutoff), so all predictions are truly out-of-sample.

Usage: Run in Spyder (Windows venv). Outputs:
  - backtest/backtest_results.csv (game-by-game, all seasons)
  - Console: per-season + combined diagnostic & betting reports

@author: chazf
"""

import pandas as pd
import numpy as np
import os
import sys

# Add project root so we can import config and models/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROJECT_DIR, PROCESSED_DIR, GAME_IDS_FILE, ODDS_2026_FILE, ODDS_HIST_FILE
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
    american_to_implied,
    power_devig,
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

# =============================================================================
# Configuration
# =============================================================================
OUTPUT_FILE = os.path.join(PROJECT_DIR, "backtest", "backtest_results.csv")

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
        "next_cutoff": None,  # None = through end of available data
        "reg_file": os.path.join(PROCESSED_DIR, "regulation_predictions_2025.csv"),
        "ot_file": os.path.join(PROCESSED_DIR, "ot_predictions_2025.csv"),
        "label": "2025-26",
    },
]

PLATT_MIN_GAMES = 200           # Min OOS games before enabling Platt
PLATT_ENABLED = False           # Disabled: Platt compresses spread by 34% and hurts LL by -1.1 mb
HOME_ICE_ENABLED = True         # Fixed home-ice logit shift from training data
LAMBDA_CAL_ENABLED = False      # Disabled: β>1 from training widens spread, hurts OOS
BIVARIATE_RHO_ENABLED = True    # Estimate rho from training data
KELLY_FRACTION = 10.00           # Quarter-Kelly
STARTING_BANKROLL = 1000.0      # Units

# OT model blending: 0.0 = pure constant (base rate), 1.0 = pure model
OT_BLEND_WEIGHT = 0.0


# =============================================================================
# Helpers
# =============================================================================

def american_to_decimal(odds):
    """Convert American odds to decimal odds.

    -205 → 1.488, +170 → 2.70
    """
    odds = float(odds)
    if odds > 0:
        return 1.0 + odds / 100.0
    else:
        return 1.0 + 100.0 / (-odds)


def load_odds_2026():
    """Load nhl_odds_2026.csv, devig, return dict: game_id → odds info."""
    if not os.path.exists(ODDS_2026_FILE):
        print("  WARNING: nhl_odds_2026.csv not found")
        return {}

    odds = pd.read_csv(ODDS_2026_FILE)

    # Drop unnamed index column
    unnamed = [c for c in odds.columns if c.startswith("Unnamed") or c == ""]
    if unnamed:
        odds = odds.drop(columns=unnamed)

    # Deduplicate: one row per game_id
    odds = odds.drop_duplicates(subset="game_id", keep="first")

    # Convert and devig
    odds["dk_home_imp"] = odds["home_american"].apply(american_to_implied)
    odds["dk_away_imp"] = odds["away_american"].apply(american_to_implied)

    devigged = odds.apply(
        lambda row: power_devig(row["dk_home_imp"], row["dk_away_imp"]),
        axis=1, result_type="expand"
    )
    odds["dk_home_fair"] = devigged[0]
    odds["dk_away_fair"] = devigged[1]
    odds["dk_home_dec"] = odds["home_american"].apply(american_to_decimal)
    odds["dk_away_dec"] = odds["away_american"].apply(american_to_decimal)

    result = {}
    for _, row in odds.iterrows():
        result[row["game_id"]] = {
            "dk_home_fair": row["dk_home_fair"],
            "dk_away_fair": row["dk_away_fair"],
            "dk_home_dec": row["dk_home_dec"],
            "dk_away_dec": row["dk_away_dec"],
            "home_american": row["home_american"],
            "away_american": row["away_american"],
        }

    return result


def load_odds_historical():
    """Load odds_collected.csv, devig, return dict: (date_str, home_abbrev) → odds info."""
    if not os.path.exists(ODDS_HIST_FILE):
        print("  WARNING: odds_collected.csv not found")
        return {}

    odds = pd.read_csv(ODDS_HIST_FILE)
    odds["home_abbrev"] = odds["home_team_odds"].map(TEAM_NAME_TO_ABBREV)
    odds = odds.dropna(subset=["home_abbrev"])

    # Date string
    odds["odds_date"] = pd.to_datetime(odds["odds_date"]).dt.strftime("%Y-%m-%d")

    # Deduplicate: latest odds per (date, home_team)
    odds = odds.sort_values("odds_timestamp_requested", ascending=False)
    odds = odds.drop_duplicates(subset=["odds_date", "home_abbrev"], keep="first")

    # Convert
    odds["home_odds_american"] = pd.to_numeric(odds["home_odds_american"], errors="coerce")
    odds["away_odds_american"] = pd.to_numeric(odds["away_odds_american"], errors="coerce")
    odds = odds.dropna(subset=["home_odds_american", "away_odds_american"])

    odds["dk_home_imp"] = odds["home_odds_american"].apply(american_to_implied)
    odds["dk_away_imp"] = odds["away_odds_american"].apply(american_to_implied)

    devigged = odds.apply(
        lambda row: power_devig(row["dk_home_imp"], row["dk_away_imp"]),
        axis=1, result_type="expand"
    )
    odds["dk_home_fair"] = devigged[0]
    odds["dk_away_fair"] = devigged[1]
    odds["dk_home_dec"] = odds["home_odds_american"].apply(american_to_decimal)
    odds["dk_away_dec"] = odds["away_odds_american"].apply(american_to_decimal)

    result = {}
    for _, row in odds.iterrows():
        key = (row["odds_date"], row["home_abbrev"])
        result[key] = {
            "dk_home_fair": row["dk_home_fair"],
            "dk_away_fair": row["dk_away_fair"],
            "dk_home_dec": row["dk_home_dec"],
            "dk_away_dec": row["dk_away_dec"],
            "home_american": row["home_odds_american"],
            "away_american": row["away_odds_american"],
        }

    return result


# =============================================================================
# Walk-Forward Engine (Single Season)
# =============================================================================

def run_season_walk_forward(season_cfg, completed, odds_2026, odds_hist):
    """Run walk-forward for a single test season.

    Args:
        season_cfg: dict with cutoff, next_cutoff, reg_file, ot_file, label
        completed: DataFrame of completed games with outcomes
        odds_2026: dict of 2025-26 odds by game_id
        odds_hist: dict of historical odds by (date, home_abbrev)

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

    # --- OOS Platt records ---
    platt_records = []

    # --- Home-ice shift (fixed from history) ---
    home_ice_shift = 0.0
    if HOME_ICE_ENABLED and len(history) > 100:
        home_ice_records = []
        global_ot_rate = history["is_ot"].mean()
        for _, row in history.iterrows():
            p_hr, p_ar, p_tr = compute_game_probabilities(
                row["lam_home_raw"], row["lam_away_raw"], rho=poisson_rho)
            tie_inf = global_ot_rate / p_tr if p_tr > 0 else 1.0
            p_hr_c, _, p_ot = apply_tie_calibration(p_hr, p_ar, p_tr, tie_inf)
            p_home_raw = p_hr_c + p_ot * DEFAULT_P_HOME_OT
            home_ice_records.append({
                "home_won": row["home_won_actual"],
                "p_home_win_raw": p_home_raw,
            })
        home_ice_df = pd.DataFrame(home_ice_records)
        home_ice_shift = compute_home_ice_shift(home_ice_df)
        print(f"  Home-ice logit shift: {home_ice_shift:+.4f} "
              f"(from {len(home_ice_df)} history games)")

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

        # Tie inflation (120-day lookback)
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

        # Platt calibration (OOS-only)
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

            # Tie inflation
            p_home_reg_cal, p_away_reg_cal, p_ot = apply_tie_calibration(
                p_home_reg, p_away_reg, p_tie_raw, tie_inflation
            )

            # OT blend: blend model prediction toward constant base rate
            model_ot = game["p_home_win_ot_model"]
            if np.isnan(model_ot):
                p_home_ot = base_rate_ot
            else:
                p_home_ot = OT_BLEND_WEIGHT * model_ot + (1 - OT_BLEND_WEIGHT) * base_rate_ot

            # Raw P(home win)
            p_home_win_raw = p_home_reg_cal + p_ot * p_home_ot

            # Home-ice correction (fixed from training data)
            if home_ice_shift != 0.0:
                logit_raw = np.log(max(p_home_win_raw, 1e-6) / max(1 - p_home_win_raw, 1e-6))
                p_home_win_raw = 1.0 / (1.0 + np.exp(-(logit_raw + home_ice_shift)))

            # Rolling Platt calibration (OOS)
            if use_platt:
                p_home_win = apply_platt_calibration(p_home_win_raw, platt_a, platt_b)
            else:
                p_home_win = p_home_win_raw
            p_away_win = 1.0 - p_home_win

            # Match DK odds
            dk_home_fair = np.nan
            dk_away_fair = np.nan
            dk_home_dec = np.nan
            dk_away_dec = np.nan

            # Try 2026 odds first (by game_id)
            if game_id in odds_2026:
                o = odds_2026[game_id]
                dk_home_fair = o["dk_home_fair"]
                dk_away_fair = o["dk_away_fair"]
                dk_home_dec = o["dk_home_dec"]
                dk_away_dec = o["dk_away_dec"]
            else:
                # Try historical odds (by date + home team)
                date_str = game_date.strftime("%Y-%m-%d")
                key = (date_str, game["home_team"])
                if key in odds_hist:
                    o = odds_hist[key]
                    dk_home_fair = o["dk_home_fair"]
                    dk_away_fair = o["dk_away_fair"]
                    dk_home_dec = o["dk_home_dec"]
                    dk_away_dec = o["dk_away_dec"]

            # Compute edges
            home_edge = p_home_win - dk_home_fair if not np.isnan(dk_home_fair) else np.nan
            away_edge = p_away_win - dk_away_fair if not np.isnan(dk_away_fair) else np.nan

            # Kelly sizing — bet best side only
            bet_side = None
            bet_edge = np.nan
            bet_odds_dec = np.nan
            kelly_stake = 0.0

            if not np.isnan(home_edge) and not np.isnan(away_edge):
                if home_edge > away_edge and home_edge > 0:
                    bet_side = "home"
                    bet_edge = home_edge
                    bet_odds_dec = dk_home_dec
                elif away_edge > 0:
                    bet_side = "away"
                    bet_edge = away_edge
                    bet_odds_dec = dk_away_dec

                if bet_side is not None:
                    b = bet_odds_dec - 1.0
                    p = p_home_win if bet_side == "home" else p_away_win
                    q = 1.0 - p
                    kelly_full = (b * p - q) / b if b > 0 else 0.0
                    kelly_stake = max(0.0, kelly_full * KELLY_FRACTION)

            # Actual outcome
            home_won = game["home_won_actual"]
            outcome_type = game["game_outcome_type"]

            bet_won = np.nan
            bet_profit = 0.0
            if bet_side is not None:
                if bet_side == "home":
                    bet_won = int(home_won == 1)
                else:
                    bet_won = int(home_won == 0)
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
                "p_home_ot": p_home_ot,
                "p_home_win_raw": p_home_win_raw,
                "home_ice_shift": home_ice_shift,
                "poisson_rho": poisson_rho,
                "platt_a": platt_a,
                "platt_b": platt_b,
                "p_home_win": p_home_win,
                "p_away_win": p_away_win,
                "dk_home_fair": dk_home_fair,
                "dk_away_fair": dk_away_fair,
                "dk_home_dec": dk_home_dec,
                "dk_away_dec": dk_away_dec,
                "home_edge": home_edge,
                "away_edge": away_edge,
                "bet_side": bet_side,
                "bet_edge": bet_edge,
                "bet_odds_dec": bet_odds_dec,
                "kelly_stake": kelly_stake,
                "home_won_actual": home_won,
                "game_outcome_type": outcome_type,
                "bet_won": bet_won,
                "bet_profit": bet_profit,
            })

            # Update rolling records for future games
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
            platt_records.append([p_home_win_raw, home_won])

    results_df = pd.DataFrame(results)
    print(f"\n  [{label}] Walk-forward complete: {len(results_df)} games processed")

    return results_df


# =============================================================================
# Walk-Forward Engine (Multi-Season)
# =============================================================================

def run_walk_forward():
    """Main walk-forward backtest pipeline across all test seasons."""

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

    # Load odds (shared across seasons)
    odds_2026 = load_odds_2026()
    odds_hist = load_odds_historical()
    print(f"  2025-26 odds entries: {len(odds_2026):,}")
    print(f"  Historical odds entries: {len(odds_hist):,}")

    # Run walk-forward for each test season
    all_results = []
    for season_cfg in TEST_SEASONS:
        season_df = run_season_walk_forward(
            season_cfg, completed, odds_2026, odds_hist)
        if len(season_df) > 0:
            all_results.append(season_df)

    if not all_results:
        print("\nWARNING: No results from any season!")
        return pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)
    results_df = results_df.sort_values("game_date").reset_index(drop=True)
    print(f"\n{'='*60}")
    print(f"COMBINED: {len(results_df)} games across "
          f"{len(all_results)} season(s)")
    print(f"{'='*60}")

    return results_df


# =============================================================================
# Reporting
# =============================================================================

def print_report(df, label="Combined"):
    """Full diagnostic report to console."""

    print(f"\n{'='*60}")
    print(f"BACKTEST DIAGNOSTICS — {label} (Walk-Forward)")
    print(f"{'='*60}")
    print(f"  Total games: {len(df)}")
    print(f"  Date range: {df['game_date'].min().date()} to "
          f"{df['game_date'].max().date()}")

    # --- Lambda bias ---
    print(f"\n  Lambda bias (predicted vs actual):")
    pred_home = df["lam_home"].mean()
    pred_away = df["lam_away"].mean()
    print(f"    Lambda home mean: {pred_home:.3f}")
    print(f"    Lambda away mean: {pred_away:.3f}")

    # --- Scoring anchor ---
    print(f"\n  Scoring anchor:")
    print(f"    Home: mean={df['anchor_home'].mean():.4f}, "
          f"range=[{df['anchor_home'].min():.4f}, {df['anchor_home'].max():.4f}]")
    print(f"    Away: mean={df['anchor_away'].mean():.4f}, "
          f"range=[{df['anchor_away'].min():.4f}, {df['anchor_away'].max():.4f}]")

    # --- P(OT) calibration by quintile ---
    print(f"\n  P(OT) calibration by quintile:")
    valid = df.dropna(subset=["p_ot"])
    if len(valid) > 100:
        valid = valid.copy()
        valid["ot_actual"] = (valid["game_outcome_type"].isin(["OT", "SO"])).astype(int)
        valid["ot_bin"] = pd.qcut(valid["p_ot"], 5, labels=False, duplicates="drop")
        for b, grp in valid.groupby("ot_bin"):
            pred = grp["p_ot"].mean()
            actual = grp["ot_actual"].mean()
            print(f"    Bin {b}: pred={pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - pred:+.3f}, n={len(grp)}")

    # --- P(home win) calibration by quintile ---
    print(f"\n  P(home win) calibration by quintile:")
    valid = df.dropna(subset=["p_home_win", "home_won_actual"]).copy()
    if len(valid) > 100:
        valid["win_bin"] = pd.qcut(valid["p_home_win"], 5, labels=False, duplicates="drop")
        for b, grp in valid.groupby("win_bin"):
            pred = grp["p_home_win"].mean()
            actual = grp["home_won_actual"].mean()
            print(f"    Bin {b}: pred={pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - pred:+.3f}, n={len(grp)}")

    # --- Log-loss, Brier, AUC ---
    valid = df.dropna(subset=["p_home_win", "home_won_actual"])
    if len(valid) > 0:
        y_true = valid["home_won_actual"].values.astype(float)
        y_pred = np.clip(valid["p_home_win"].values, 0.001, 0.999)

        ll = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)).mean()
        brier = ((y_pred - y_true) ** 2).mean()

        base_rate = y_true.mean()
        base_pred = np.full_like(y_pred, base_rate)
        base_ll = -(y_true * np.log(base_pred) + (1 - y_true) * np.log(1 - base_pred)).mean()
        base_brier = ((base_pred - y_true) ** 2).mean()

        # AUC
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        if len(pos) > 0 and len(neg) > 0:
            auc = np.mean(pos[:, None] > neg[None, :]) + \
                  0.5 * np.mean(pos[:, None] == neg[None, :])
        else:
            auc = 0.5

        print(f"\n  Overall metrics ({len(valid)} games):")
        print(f"    Log-loss:    {ll:.5f}  (baseline: {base_ll:.5f}, "
              f"improvement: {(1 - ll/base_ll)*100:+.1f}%)")
        print(f"    Brier score: {brier:.5f}  (baseline: {base_brier:.5f})")
        print(f"    AUC-ROC:     {auc:.4f}")
        print(f"    Home win rate: {base_rate:.3f}")

    # --- DK odds comparison ---
    with_odds = df.dropna(subset=["dk_home_fair", "home_won_actual"])
    if len(with_odds) > 50:
        y_true = with_odds["home_won_actual"].values.astype(float)
        model_pred = np.clip(with_odds["p_home_win"].values, 0.001, 0.999)
        dk_pred = np.clip(with_odds["dk_home_fair"].values, 0.001, 0.999)

        model_ll = -(y_true * np.log(model_pred) + (1 - y_true) * np.log(1 - model_pred)).mean()
        dk_ll = -(y_true * np.log(dk_pred) + (1 - y_true) * np.log(1 - dk_pred)).mean()

        print(f"\n  Model vs DK ({len(with_odds)} games with odds):")
        print(f"    Model log-loss: {model_ll:.5f}")
        print(f"    DK log-loss:    {dk_ll:.5f}")
        print(f"    Difference:     {model_ll - dk_ll:+.5f} "
              f"({'model better' if model_ll < dk_ll else 'DK better'})")
        print(f"    Millibits:      {(model_ll - dk_ll) * 1000:+.1f} mb")

    # --- Edge distribution ---
    edges = df.dropna(subset=["home_edge"])
    if len(edges) > 0:
        best_edge = edges[["home_edge", "away_edge"]].max(axis=1)

        print(f"\n  Edge distribution (best side per game):")
        print(f"    Mean:   {best_edge.mean():+.4f}")
        print(f"    Std:    {best_edge.std():.4f}")
        print(f"    Median: {best_edge.median():+.4f}")
        print(f"    |home_edge| > 3%: {(edges['home_edge'].abs() > 0.03).sum()} "
              f"({(edges['home_edge'].abs() > 0.03).mean()*100:.1f}%)")
        print(f"    Home edge mean: {edges['home_edge'].mean():+.4f}")
        print(f"    Away edge mean: {edges['away_edge'].mean():+.4f}")

    # --- Monthly breakdown ---
    print(f"\n  Monthly metrics:")
    df_with_pred = df.dropna(subset=["p_home_win", "home_won_actual"]).copy()
    df_with_pred["month"] = df_with_pred["game_date"].dt.to_period("M")
    for month, grp in df_with_pred.groupby("month"):
        y = grp["home_won_actual"].values.astype(float)
        p = np.clip(grp["p_home_win"].values, 0.001, 0.999)
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        base = y.mean()
        base_ll = -(y * np.log(base) + (1 - y) * np.log(1 - base)).mean() if 0 < base < 1 else ll
        n = len(grp)

        platt_active = grp["platt_a"].iloc[0] != 1.0 or grp["platt_b"].iloc[0] != 0.0

        print(f"    {month}: LL={ll:.4f} (base={base_ll:.4f}, "
              f"imp={((1 - ll/base_ll)*100) if base_ll > 0 else 0:+.1f}%) "
              f"n={n} {'[Platt ON]' if platt_active else '[raw probs]'}")


def print_betting_report(df, label="Combined", thresholds=None):
    """Betting simulation report at multiple edge thresholds."""
    if thresholds is None:
        thresholds = [0.0, 0.02, 0.05]

    print(f"\n{'='*60}")
    print(f"BETTING SIMULATION — {label} Walk-Forward")
    print(f"{'='*60}")
    print(f"  Kelly fraction: {KELLY_FRACTION}")
    print(f"  Starting bankroll: {STARTING_BANKROLL} units")

    with_odds = df.dropna(subset=["dk_home_fair"]).copy()
    print(f"  Games with odds: {len(with_odds)}")

    has_edge = with_odds[
        (with_odds["bet_side"].notna()) & (with_odds["bet_edge"] > 0)
    ]
    has_stake = has_edge[has_edge["kelly_stake"] > 0]
    vig_killed = len(has_edge) - len(has_stake)
    print(f"  Games with positive edge: {len(has_edge)}")
    print(f"  Games with positive Kelly stake: {len(has_stake)}")
    print(f"  Edge killed by vig (edge>0 but Kelly=0): {vig_killed}")

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

        edge_only = with_odds[
            (with_odds["bet_side"].notna()) &
            (with_odds["bet_edge"] >= threshold)
        ]
        n_vig_killed = len(edge_only) - len(bets)

        n_bets = len(bets)
        n_wins = int(bets["bet_won"].sum())
        win_rate = n_wins / n_bets

        total_staked = bets["kelly_stake"].sum()
        total_profit = bets["bet_profit"].sum()
        total_returned = total_staked + total_profit
        roi = total_profit / total_staked * 100 if total_staked > 0 else 0

        avg_edge = bets["bet_edge"].mean()
        avg_odds = bets["bet_odds_dec"].mean()
        avg_stake = bets["kelly_stake"].mean()

        # Max drawdown
        cum_pnl = bets["bet_profit"].cumsum()
        running_max = cum_pnl.cummax()
        drawdown = cum_pnl - running_max
        max_dd = drawdown.min()

        final_bankroll = STARTING_BANKROLL + total_profit

        bets["ev_edge"] = bets.apply(
            lambda row: (row["p_home_win"] if row["bet_side"] == "home"
                         else row["p_away_win"]) * row["bet_odds_dec"] - 1.0,
            axis=1
        )

        print(f"    Bets placed:    {n_bets} ({n_vig_killed} more had edge but vig killed Kelly)")
        print(f"    Win rate:       {win_rate:.3f} ({n_wins}/{n_bets})")
        print(f"    Avg edge vs fair: {avg_edge:.4f} ({avg_edge*100:.1f}%)")
        print(f"    Avg EV at odds:   {bets['ev_edge'].mean():+.4f} ({bets['ev_edge'].mean()*100:+.1f}%)")
        print(f"    Avg odds (dec): {avg_odds:.3f}")
        print(f"    Avg stake:      {avg_stake:.4f} units")
        print(f"    Total staked:   {total_staked:.2f} units")
        print(f"    Total returned: {total_returned:.2f} units")
        print(f"    Net profit:     {total_profit:+.2f} units")
        print(f"    ROI:            {roi:+.1f}%")
        print(f"    Max drawdown:   {max_dd:.2f} units")
        print(f"    Final bankroll: {final_bankroll:.2f} units")

        # Monthly P&L
        print(f"\n    Monthly P&L:")
        bets["month"] = bets["game_date"].dt.to_period("M")
        for month, grp in bets.groupby("month"):
            m_bets = len(grp)
            m_wins = int(grp["bet_won"].sum())
            m_staked = grp["kelly_stake"].sum()
            m_profit = grp["bet_profit"].sum()
            m_roi = m_profit / m_staked * 100 if m_staked > 0 else 0
            print(f"      {month}: {m_bets} bets, {m_wins}W, "
                  f"staked={m_staked:.2f}, P&L={m_profit:+.2f}, ROI={m_roi:+.1f}%")

        # Home vs away split
        home_bets = bets[bets["bet_side"] == "home"]
        away_bets = bets[bets["bet_side"] == "away"]
        if len(home_bets) > 0:
            h_roi = home_bets["bet_profit"].sum() / home_bets["kelly_stake"].sum() * 100
        else:
            h_roi = 0
        if len(away_bets) > 0:
            a_roi = away_bets["bet_profit"].sum() / away_bets["kelly_stake"].sum() * 100
        else:
            a_roi = 0
        print(f"\n    Side split (Kelly):")
        print(f"      Home: {len(home_bets)} bets, ROI={h_roi:+.1f}%")
        print(f"      Away: {len(away_bets)} bets, ROI={a_roi:+.1f}%")

        # --- Flat staking (1 unit per bet) ---
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
            flat_total_profit = flat_bets["flat_profit"].sum()
            flat_roi = flat_total_profit / flat_n * 100

            flat_cum = flat_bets["flat_profit"].cumsum()
            flat_max_dd = (flat_cum - flat_cum.cummax()).min()

            print(f"\n    Flat staking (1 unit/bet):")
            print(f"      Bets:       {flat_n} (incl. {n_vig_killed} vig-killed)")
            print(f"      Win rate:   {flat_wins/flat_n:.3f} ({flat_wins}/{flat_n})")
            print(f"      Staked:     {flat_n} units")
            print(f"      P&L:        {flat_total_profit:+.2f} units")
            print(f"      ROI:        {flat_roi:+.2f}%")
            print(f"      Max DD:     {flat_max_dd:.2f} units")

            print(f"      Monthly:")
            flat_bets["month"] = flat_bets["game_date"].dt.to_period("M")
            for month, grp in flat_bets.groupby("month"):
                fm_n = len(grp)
                fm_w = int(grp["bet_won"].sum())
                fm_pnl = grp["flat_profit"].sum()
                fm_roi = fm_pnl / fm_n * 100
                print(f"        {month}: {fm_n} bets, {fm_w}W, "
                      f"P&L={fm_pnl:+.2f}, ROI={fm_roi:+.1f}%")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Walk-Forward Backtest — Multi-Season NHL")
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
    print(f"  Saved {len(output)} games to {OUTPUT_FILE}")

    # --- Verification checks ---
    print(f"\n{'='*60}")
    print("VERIFICATION CHECKS")
    print(f"{'='*60}")

    n_games = len(results_df)
    n_with_odds = results_df["dk_home_fair"].notna().sum()

    edges = results_df["home_edge"].dropna()
    edge_mean = edges.mean()

    print(f"  Total games: {n_games}")
    for sl in results_df["season_label"].unique():
        n_s = (results_df["season_label"] == sl).sum()
        n_o = results_df.loc[results_df["season_label"] == sl, "dk_home_fair"].notna().sum()
        print(f"    {sl}: {n_s} games, {n_o} with odds")
    print(f"  Games with DK odds: {n_with_odds}")
    print(f"  Home edge mean: {edge_mean:+.4f} "
          f"({'OK - near zero' if abs(edge_mean) < 0.02 else 'WARNING - biased'})")
    print(f"  OT blend weight: {OT_BLEND_WEIGHT}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
