# -*- coding: utf-8 -*-
"""
predict_today.py — Daily Live NHL 3-Way Predictions

Replicates the exact walk-forward backtest logic for live daily use.
Same features, same models, same calibration, same edge calculation.
Run each morning to get tonight's bet recommendations.

Usage:
    Spyder:  Set RUN_DATE below, then run the file.
    CLI:     python predict_today.py --date 2026-03-05

Steps:
    1. Scrape latest data (incremental)
    2. Get tonight's confirmed starters (Daily Faceoff)
    3. Run feature pipeline (re-uses existing model artifacts)
    4. Load today's features + trained models
    5. Predict regulation quantiles → lambdas
    6. Compute calibration params from history
    7. Bivariate Poisson → 3-way probabilities
    8. Fetch live 3-way odds
    9. Calculate edges + bet recommendations
   10. Output formatted recommendations

@author: chazf
"""

import sys
import os
import time
import argparse
import pickle
from datetime import date, datetime

import pandas as pd
import numpy as np

# ── Configuration ────────────────────────────────────────────────────────────
# Set this when running in Spyder (overridden by --date CLI arg)
RUN_DATE = None  # e.g., "2026-03-05" or None for today

# Skip scraping (for testing/speed — assumes data is already up to date)
SKIP_SCRAPING = False

# Skip odds fetching (for testing without spending API tokens)
SKIP_ODDS = False

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from config import PROCESSED_DIR, RAW_DIR, MODEL_DIR, GAME_IDS_FILE

# Model cutoff year for 2025-26 season
CUTOFF_YEAR = "2025"
CUTOFF_DATE = pd.Timestamp("2025-10-01")


# =============================================================================
# Step 1: Scrape Latest Data
# =============================================================================

def step_scrape_data():
    """Refresh raw data files (incremental — only fetches new games)."""
    from scrapers.game_scraper import scrape_game_ids
    from scrapers.pbp_boxscore_scraper import scrape_all_games as scrape_pbp
    from scrapers.goalie_stats_scraper import scrape_all as scrape_goalie_stats
    from scrapers.team_stats_scraper import scrape_all as scrape_team_stats
    from scrapers.standings_scraper import scrape_standings

    print(f"\n{'='*60}")
    print("STEP 1: Scrape Latest Data")
    print(f"{'='*60}")

    print("\n  [1a] Game schedule...")
    scrape_game_ids(force_refresh=True)

    print("\n  [1b] Play-by-play + boxscores...")
    scrape_pbp()

    print("\n  [1c] Goalie stats...")
    scrape_goalie_stats()

    print("\n  [1d] Team stats...")
    scrape_team_stats()

    print("\n  [1e] Standings...")
    scrape_standings()


# =============================================================================
# Step 2: Get Tonight's Starters
# =============================================================================

def step_get_starters(run_date):
    """Scrape confirmed goalie starters from Daily Faceoff.

    Returns:
        starters_df: Full starter info (for display)
        starters_override: DataFrame suitable for goalie feature injection
    """
    from scrapers.goalie_starter_scraper import scrape_starters, get_usable_starters

    print(f"\n{'='*60}")
    print("STEP 2: Get Tonight's Starters")
    print(f"{'='*60}")

    starters_df = scrape_starters(date_str=run_date)

    if len(starters_df) == 0:
        print("  No games found on Daily Faceoff.")
        return starters_df, pd.DataFrame()

    usable = get_usable_starters(starters_df)
    print(f"\n  Usable starters (confirmed/likely + ID matched): {len(usable)}")

    # To create starters_override, we need game_ids for today's games
    games = pd.read_csv(GAME_IDS_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    today_games = games[games["game_date"].dt.strftime("%Y-%m-%d") == run_date]

    if len(today_games) == 0:
        print(f"  WARNING: No games in game_ids.csv for {run_date}")
        return starters_df, pd.DataFrame()

    # Match starters to game_ids by team
    override_rows = []
    for _, starter in usable.iterrows():
        team = starter["team"]
        # Find game where this team is home or away
        home_match = today_games[today_games["home_team"] == team]
        away_match = today_games[today_games["away_team"] == team]

        if len(home_match) > 0:
            game_id = home_match.iloc[0]["game_id"]
            opponent = home_match.iloc[0]["away_team"]
            is_home = 1
        elif len(away_match) > 0:
            game_id = away_match.iloc[0]["game_id"]
            opponent = away_match.iloc[0]["home_team"]
            is_home = 0
        else:
            continue

        override_rows.append({
            "game_id": game_id,
            "game_date": run_date,
            "team": team,
            "goalie_id": starter["goalie_id"],
            "goalie_name": starter["goalie_name"],
            "is_home": is_home,
            "opponent": opponent,
        })

    starters_override = pd.DataFrame(override_rows)
    print(f"  Starters override rows: {len(starters_override)}")

    return starters_df, starters_override


# =============================================================================
# Step 3: Run Feature Pipeline
# =============================================================================

def step_run_features(starters_override=None):
    """Re-run feature engineering to ensure features match backtest exactly.

    Does NOT retrain models — uses existing .pkl artifacts.
    """
    from features.xg_features import build_xg_features
    from features.goalie_features import build_goalie_features
    from features.regulation_features import build_regulation_features
    from features.ot_features import build_ot_features

    print(f"\n{'='*60}")
    print("STEP 3: Run Feature Pipeline")
    print(f"{'='*60}")

    print("\n  [3a] xG feature engineering...")
    build_xg_features()

    # Skip xG model training — use existing artifact

    print(f"\n  [3b] Goalie feature engineering (cutoff={CUTOFF_YEAR})...")
    build_goalie_features(cutoff_year=CUTOFF_YEAR,
                          starters_override=starters_override)

    # Skip goalie model training — use existing artifact

    print(f"\n  [3c] Regulation feature engineering (cutoff={CUTOFF_YEAR})...")
    build_regulation_features(cutoff_year=CUTOFF_YEAR)

    print(f"\n  [3d] OT feature engineering (cutoff={CUTOFF_YEAR})...")
    build_ot_features(cutoff_year=CUTOFF_YEAR)


# =============================================================================
# Step 4-6: Load Features + Models, Predict Lambdas
# =============================================================================

def step_predict_lambdas(run_date):
    """Load features, load models, predict regulation quantiles → lambdas.

    Uses TWO data sources to guarantee exact backtest parity:
      - Pre-computed predictions file (regulation_predictions_*.csv) for calibration
        history. This matches the backtest's NaN-filtered game set exactly.
      - Fresh features (regulation_features_*.csv) for new games not in the
        predictions file (games played since last model run + today's games).

    Returns:
        today_paired: DataFrame of today's games with lambdas
        history_paired: DataFrame of all completed paired games (for calibration)
    """
    from models.regulation_model import load_regulation_model, predict_regulation_quantiles
    from models.ot_model import load_ot_model, predict_ot_winner
    from models.poisson_combiner import quantiles_to_lambda, QUANTILE_COLS

    print(f"\n{'='*60}")
    print("STEP 4-6: Load Models + Predict Lambdas")
    print(f"{'='*60}")

    # ── A: Load pre-computed predictions (matches backtest's row set exactly) ──
    pred_file = os.path.join(PROCESSED_DIR, f"regulation_predictions_{CUTOFF_YEAR}.csv")
    reg_pred = pd.read_csv(pred_file, low_memory=False)
    reg_pred["game_date"] = pd.to_datetime(reg_pred["game_date"])
    print(f"  Pre-computed predictions: {len(reg_pred):,} rows")

    # Compute lambdas from pre-computed quantiles
    reg_pred["lam_raw"] = reg_pred[QUANTILE_COLS].apply(
        lambda row: quantiles_to_lambda(row.values), axis=1)

    # ── B: Load fresh features for games NOT in predictions file ──
    reg_file = os.path.join(PROCESSED_DIR, f"regulation_features_{CUTOFF_YEAR}.csv")
    reg_feat = pd.read_csv(reg_file, low_memory=False)
    reg_feat["game_date"] = pd.to_datetime(reg_feat["game_date"])

    pred_game_ids = set(reg_pred["game_id"].unique())
    new_rows = reg_feat[~reg_feat["game_id"].isin(pred_game_ids)].copy()
    print(f"  New games needing fresh prediction: {len(new_rows)//2}")

    if len(new_rows) > 0:
        # Load model and predict on new games
        print(f"  Loading regulation model ({CUTOFF_YEAR})...")
        reg_models, reg_calibrator, reg_feature_cols, reg_quantiles = \
            load_regulation_model(cutoff_year=CUTOFF_YEAR)

        preds = predict_regulation_quantiles(new_rows, reg_models, reg_calibrator)
        for q_name, q_values in preds.items():
            q_col = f"q{int(q_name*100):02d}"
            new_rows[q_col] = q_values

        new_rows["lam_raw"] = new_rows[QUANTILE_COLS].apply(
            lambda row: quantiles_to_lambda(row.values), axis=1)

    # ── C: Combine pre-computed + new predictions ──
    common_cols = ["game_id", "game_date", "team", "opponent", "is_home",
                   "season", "reg_gf", "lam_raw"]
    reg_all = pd.concat(
        [reg_pred[common_cols], new_rows[common_cols]] if len(new_rows) > 0
        else [reg_pred[common_cols]],
        ignore_index=True,
    )
    print(f"  Combined rows: {len(reg_all):,}")

    # ── D: OT predictions (informational — OT_BLEND_WEIGHT=0.0) ──
    ot_file = os.path.join(PROCESSED_DIR, f"ot_features_{CUTOFF_YEAR}.csv")
    ot = pd.read_csv(ot_file, low_memory=False)
    ot["game_date"] = pd.to_datetime(ot["game_date"])

    print(f"  Loading OT model ({CUTOFF_YEAR})...")
    ot_artifact = load_ot_model(cutoff_year=CUTOFF_YEAR)

    ot_valid = ot.dropna(subset=ot_artifact["feature_cols"]).copy()
    if len(ot_valid) > 0:
        ot_valid["p_home_win_ot"] = predict_ot_winner(ot_valid, ot_artifact)
        ot_map = dict(zip(ot_valid["game_id"], ot_valid["p_home_win_ot"]))
    else:
        ot_map = {}

    # ── E: Pair home/away ──
    home = reg_all[reg_all["is_home"] == 1][
        ["game_id", "game_date", "team", "opponent", "season", "reg_gf", "lam_raw"]
    ].copy()
    home.columns = ["game_id", "game_date", "home_team", "away_team",
                     "season", "home_reg_gf", "lam_home_raw"]

    away = reg_all[reg_all["is_home"] == 0][
        ["game_id", "reg_gf", "lam_raw"]
    ].copy()
    away.columns = ["game_id", "away_reg_gf", "lam_away_raw"]

    paired = home.merge(away, on="game_id", how="inner")
    print(f"  Paired games total: {len(paired):,}")

    # Add OT predictions
    paired["p_home_win_ot_model"] = paired["game_id"].map(ot_map)

    # ── F: Split history vs today ──
    games = pd.read_csv(GAME_IDS_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    completed = games[games["game_state"] == "OFF"].copy()
    completed["is_ot"] = completed["game_outcome_type"].isin(["OT", "SO"]).astype(int)
    completed["home_won_actual"] = (
        completed["home_score"] > completed["away_score"]).astype(int)

    run_ts = pd.Timestamp(run_date)

    history_paired = paired.merge(
        completed[["game_id", "game_outcome_type", "is_ot", "home_won_actual"]],
        on="game_id", how="inner"
    )
    history_paired = history_paired[history_paired["game_date"] < run_ts].copy()
    history_paired = history_paired.sort_values("game_date").reset_index(drop=True)

    today_paired = paired[paired["game_date"].dt.strftime("%Y-%m-%d") == run_date].copy()

    print(f"  History (completed, before {run_date}): {len(history_paired):,} games")
    print(f"  Today ({run_date}): {len(today_paired):,} games")

    if len(today_paired) == 0:
        print(f"  WARNING: No games found for {run_date} in features!")
        print(f"  Check if game_ids.csv includes today's games.")

    return today_paired, history_paired


# =============================================================================
# Step 7: Compute Calibration Parameters
# =============================================================================

def step_calibrate(history_paired):
    """Compute calibration parameters from completed history."""
    from models.predict_3way import compute_calibration_params

    print(f"\n{'='*60}")
    print("STEP 7: Compute Calibration Parameters")
    print(f"{'='*60}")

    calib = compute_calibration_params(history_paired, CUTOFF_DATE)

    print(f"  Poisson rho:      {calib['poisson_rho']:.4f}")
    print(f"  Tie inflation:    {calib['tie_inflation']:.4f}")
    print(f"  Home-ice shift:   {calib['home_ice_shift']:+.4f}")
    print(f"  Scoring anchor H: {calib['anchor_home']:.4f}")
    print(f"  Scoring anchor A: {calib['anchor_away']:.4f}")

    return calib


# =============================================================================
# Step 8: Predict 3-Way Probabilities
# =============================================================================

def step_predict_3way(today_paired, calib_params):
    """Compute 3-way probabilities for today's games."""
    from models.predict_3way import predict_game_3way
    from models.poisson_combiner import DEFAULT_P_HOME_OT

    print(f"\n{'='*60}")
    print("STEP 8: Predict 3-Way Probabilities")
    print(f"{'='*60}")

    games = []
    for _, game in today_paired.iterrows():
        probs = predict_game_3way(
            game["lam_home_raw"], game["lam_away_raw"], calib_params)

        games.append({
            "game_id": game["game_id"],
            "home_team": game["home_team"],
            "away_team": game["away_team"],
            "lam_home": probs["lam_home"],
            "lam_away": probs["lam_away"],
            "p_home_reg": probs["p_home_reg"],
            "p_away_reg": probs["p_away_reg"],
            "p_ot": probs["p_ot"],
        })

    for g in games:
        prob_sum = g["p_home_reg"] + g["p_away_reg"] + g["p_ot"]
        print(f"  {g['away_team']:>3}@{g['home_team']:<3}: "
              f"H={g['p_home_reg']:.3f} D={g['p_ot']:.3f} A={g['p_away_reg']:.3f} "
              f"(sum={prob_sum:.4f}) "
              f"λH={g['lam_home']:.2f} λA={g['lam_away']:.2f}")

    return games


# =============================================================================
# Step 9: Fetch Live 3-Way Odds
# =============================================================================

def step_fetch_odds(run_date):
    """Fetch live 3-way regulation odds from The Odds API."""
    from scrapers.three_way_odds_scraper import fetch_today_odds

    print(f"\n{'='*60}")
    print("STEP 9: Fetch Live 3-Way Odds")
    print(f"{'='*60}")

    odds_3way = fetch_today_odds(date_str=run_date)
    return odds_3way


# =============================================================================
# Step 10: Calculate Edges + Recommendations
# =============================================================================

def step_calculate_edges(games, odds_3way, run_date):
    """Calculate edges and format bet recommendations."""
    from models.predict_3way import calculate_edges, format_recommendations

    print(f"\n{'='*60}")
    print("STEP 10: Calculate Edges + Recommendations")
    print(f"{'='*60}")

    for g in games:
        key = (run_date, g["home_team"])
        book_list = odds_3way.get(key, [])

        # Sort DraftKings first — primary comparison book
        book_list.sort(key=lambda x: (0 if x.get("bookmaker") == "draftkings" else 1))

        g["odds_results"] = []
        for odds_row in book_list:
            edge_result = calculate_edges(g, odds_row)
            g["odds_results"].append(edge_result)

        if not book_list:
            print(f"  {g['away_team']}@{g['home_team']}: no odds available")

    # Format and print
    output = format_recommendations(games, run_date)
    print(output)

    return games


# =============================================================================
# Save Predictions
# =============================================================================

def save_predictions(games, run_date):
    """Save predictions to CSV for record-keeping."""
    predictions_dir = os.path.join(PROJECT_DIR, "data", "predictions")
    os.makedirs(predictions_dir, exist_ok=True)

    rows = []
    for g in games:
        base = {
            "game_date": run_date,
            "game_id": g["game_id"],
            "home_team": g["home_team"],
            "away_team": g["away_team"],
            "lam_home": g["lam_home"],
            "lam_away": g["lam_away"],
            "p_home_reg": g["p_home_reg"],
            "p_away_reg": g["p_away_reg"],
            "p_ot": g["p_ot"],
        }

        if g.get("odds_results"):
            for edge_result in g["odds_results"]:
                row = {**base, **edge_result}
                rows.append(row)
        else:
            rows.append(base)

    df = pd.DataFrame(rows)
    output_file = os.path.join(predictions_dir, f"predictions_{run_date}.csv")
    df.to_csv(output_file, index=False)
    print(f"\n  Saved {len(df)} rows to {output_file}")

    # Also append to running log (handle column changes gracefully)
    log_file = os.path.join(predictions_dir, "predictions_log.csv")
    if os.path.exists(log_file):
        existing = pd.read_csv(log_file, nrows=0)  # header only
        if set(df.columns) != set(existing.columns):
            # Columns changed — re-read, concat, re-write with unified schema
            existing = pd.read_csv(log_file)
            combined = pd.concat([existing, df], ignore_index=True)
            combined.to_csv(log_file, index=False)
        else:
            # Reorder df columns to match log header before appending
            df = df[existing.columns]
            df.to_csv(log_file, mode="a", header=False, index=False)
    else:
        df.to_csv(log_file, index=False)


# =============================================================================
# Main Pipeline
# =============================================================================

def main(run_date=None):
    """Run the full daily prediction pipeline."""
    if run_date is None:
        run_date = date.today().strftime("%Y-%m-%d")

    print("=" * 65)
    print(f"  NHL 3-Way Daily Predictions — {run_date}")
    print("=" * 65)
    t0 = time.time()

    # Step 1: Scrape latest data
    if not SKIP_SCRAPING:
        step_scrape_data()
    else:
        print("\n  [SKIP] Scraping disabled (SKIP_SCRAPING=True)")

    # Step 2: Get tonight's starters
    starters_df, starters_override = step_get_starters(run_date)
    if len(starters_override) == 0:
        print("\n  WARNING: No usable starters found. Proceeding without goalie override.")
        starters_override = None

    # Step 3: Run feature pipeline
    step_run_features(starters_override=starters_override)

    # Steps 4-6: Load features + models, predict lambdas
    today_paired, history_paired = step_predict_lambdas(run_date)

    if len(today_paired) == 0:
        print(f"\n  No games to predict for {run_date}. Exiting.")
        return

    # Step 7: Compute calibration parameters
    calib_params = step_calibrate(history_paired)

    # Step 8: Predict 3-way probabilities
    games = step_predict_3way(today_paired, calib_params)

    # Step 9: Fetch live odds
    if not SKIP_ODDS:
        odds_3way = step_fetch_odds(run_date)
    else:
        print("\n  [SKIP] Odds fetching disabled (SKIP_ODDS=True)")
        odds_3way = {}

    # Step 10: Calculate edges + recommendations
    games = step_calculate_edges(games, odds_3way, run_date)

    # Save predictions
    save_predictions(games, run_date)

    elapsed = time.time() - t0
    print(f"\n{'='*65}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"{'='*65}")

    return games


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NHL 3-Way Daily Predictions")
    parser.add_argument("--date", default=None,
                        help="Date to predict (YYYY-MM-DD). Default: today")
    parser.add_argument("--skip-scraping", action="store_true",
                        help="Skip data scraping step")
    parser.add_argument("--skip-odds", action="store_true",
                        help="Skip live odds fetching")
    args, _ = parser.parse_known_args()

    if args.skip_scraping:
        SKIP_SCRAPING = True
    if args.skip_odds:
        SKIP_ODDS = True

    # CLI --date overrides RUN_DATE
    target_date = args.date or RUN_DATE
    main(run_date=target_date)
