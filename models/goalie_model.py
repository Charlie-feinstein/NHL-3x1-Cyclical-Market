# -*- coding: utf-8 -*-
"""
Goalie Deployment Model (Layer 0 v3)

Ridge regression predicting team GA residual from goalie deployment context
+ rolling GSAx quality features.

Target: team_reg_ga - team_ga_ewm_20 (regulation goals-against residual)
  - Positive = team allowed more goals than their recent average
  - team_ga_ewm_20 uses shift(1) — no leakage

Output: goalie_ga_rate = team_ga_ewm_20 + Ridge_prediction
  - Reconstituted GA rate in the 2-4 range (meaningful scale)
  - Baseline carries most variance; Ridge adds goalie+deployment adjustment

Features: 13 features (9 deployment context + 4 rolling GSAx)

Outputs:
  - model_artifacts/goalie_model.pkl : Trained Ridge + scaler + feature list
  - data/processed/goalie_deployment_predictions.csv : GA rate + features

@author: chazf
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =============================================================================
# Configuration
# =============================================================================
from config import PROCESSED_DIR, RAW_DIR, MODEL_DIR

GAME_FILE = os.path.join(RAW_DIR, "game_ids.csv")

# Train cutoffs to generate OOS predictions for each test season
TRAIN_CUTOFFS = ["2024-10-01", "2025-10-01"]

# Features (all from goalie_game_features.csv)
FEATURE_COLS = [
    # Deployment context (situation)
    "starter_role_share",
    "start_share_trend",
    "goalie_switch",
    "consecutive_starts",
    "was_pulled_last_game",
    "days_rest",
    "is_back_to_back",
    "season_games_started",
    "games_started_last_14d",
    # Goalie quality (rolling GSAx — captures skill signal)
    "gsax_ewm_20",          # Long-term GSAx (quality baseline)
    "gsax_ewm_5",           # Short-term GSAx (recent form)
    "save_pct_ewm_20",      # Save percentage trend
    "hd_gsax_ewm_10",       # High-danger saves (most relevant for OT)
]

TARGET_COL = "team_ga_residual"

# Ridge alphas to search
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

# EWM span for team GA baseline
TEAM_GA_EWM_SPAN = 20


# =============================================================================
# Target Computation
# =============================================================================

def compute_team_ga_target(goalie_df, shots, games):
    """
    Compute team regulation GA residual for each goalie-game row.

    Target = team_reg_ga (this game) - team_ga_ewm_20 (rolling avg, shift(1))

    Steps:
    1. Compute regulation GA per team per game from shot data
    2. Compute rolling EWM(20) of team GA with shift(1)
    3. Residual = actual - rolling average
    4. Merge onto goalie feature rows
    """
    # --- Step 1: Regulation GA per team per game ---
    # Use shots from periods 1-3 only
    reg_shots = shots[shots["period"].astype(int) <= 3].copy()

    # Goals against each team: opponent's goals scored
    # Home team GA = away team goals
    home_ga = reg_shots[reg_shots["is_home_event"] == False].groupby("game_id").agg(
        reg_ga=("is_goal", "sum")
    )
    home_ga["is_home"] = 1

    away_ga = reg_shots[reg_shots["is_home_event"] == True].groupby("game_id").agg(
        reg_ga=("is_goal", "sum")
    )
    away_ga["is_home"] = 0

    team_ga = pd.concat([home_ga.reset_index(), away_ga.reset_index()])

    # Add team name and date
    game_info = games[["game_id", "game_date", "home_team", "away_team"]].copy()
    game_info["game_date"] = pd.to_datetime(game_info["game_date"])

    home_info = game_info[["game_id", "game_date", "home_team"]].rename(
        columns={"home_team": "team"})
    home_info["is_home"] = 1

    away_info = game_info[["game_id", "game_date", "away_team"]].rename(
        columns={"away_team": "team"})
    away_info["is_home"] = 0

    team_info = pd.concat([home_info, away_info])
    team_ga = team_ga.merge(team_info, on=["game_id", "is_home"], how="left")

    # --- Step 2: Rolling EWM of team GA with shift(1) ---
    team_ga = team_ga.sort_values(["team", "game_date", "game_id"])
    team_ga["team_ga_ewm_20"] = team_ga.groupby("team")["reg_ga"].transform(
        lambda x: x.shift(1).ewm(span=TEAM_GA_EWM_SPAN, min_periods=3).mean()
    )

    # --- Step 3: Residual ---
    team_ga["team_ga_residual"] = team_ga["reg_ga"] - team_ga["team_ga_ewm_20"]

    # --- Step 4: Merge onto goalie rows ---
    merge_cols = ["game_id", "team", "reg_ga", "team_ga_ewm_20", "team_ga_residual"]
    goalie_df = goalie_df.merge(
        team_ga[merge_cols].drop_duplicates(subset=["game_id", "team"]),
        on=["game_id", "team"], how="left"
    )

    return goalie_df


# =============================================================================
# Model Training
# =============================================================================

def train_goalie_model(train_cutoff="2025-10-01"):
    """
    Train Ridge regression to predict team GA residual from deployment features.
    """
    cutoff_year = train_cutoff[:4]
    goalie_features_file = os.path.join(PROCESSED_DIR, f"goalie_game_features_{cutoff_year}.csv")
    shot_xg_file = os.path.join(PROCESSED_DIR, f"shot_xg_{cutoff_year}.csv")
    model_file = os.path.join(MODEL_DIR, f"goalie_model_{cutoff_year}.pkl")
    output_file = os.path.join(PROCESSED_DIR, f"goalie_deployment_predictions_{cutoff_year}.csv")

    print(f"\n{'#'*60}")
    print(f"  Train cutoff: {train_cutoff}")
    print(f"  Model file:   {model_file}")
    print(f"  Output file:  {output_file}")
    print(f"{'#'*60}")

    print("Loading data...")
    df = pd.read_csv(goalie_features_file)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Goalie-game rows: {len(df):,}")

    shots = pd.read_csv(shot_xg_file)
    shots["game_date"] = pd.to_datetime(shots["game_date"])

    games = pd.read_csv(GAME_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])

    # ----- Compute target -----
    print("\nComputing team GA residual target...")
    df = compute_team_ga_target(df, shots, games)
    print(f"  Target coverage: {df[TARGET_COL].notna().mean():.1%}")
    print(f"  Target stats: mean={df[TARGET_COL].mean():.3f}, "
          f"std={df[TARGET_COL].std():.3f}")

    # ----- Drop rows with NaN features or target -----
    df_valid = df.dropna(subset=FEATURE_COLS + [TARGET_COL]).copy()
    print(f"  After dropping NaN: {len(df_valid):,} rows "
          f"(dropped {len(df) - len(df_valid):,})")

    # ----- Temporal split -----
    train = df_valid[df_valid["game_date"] < train_cutoff].copy()
    test = df_valid[df_valid["game_date"] >= train_cutoff].copy()

    print(f"\n  Training:  {len(train):,} rows "
          f"({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"  Test:      {len(test):,} rows "
          f"({test['game_date'].min().date()} to {test['game_date'].max().date()})")

    # ----- Prepare data -----
    X_train = train[FEATURE_COLS].values
    y_train = train[TARGET_COL].values
    X_test = test[FEATURE_COLS].values
    y_test = test[TARGET_COL].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ----- Train RidgeCV -----
    print("\nTraining RidgeCV...")
    model = RidgeCV(alphas=RIDGE_ALPHAS, cv=5, scoring="neg_mean_squared_error")
    model.fit(X_train_scaled, y_train)
    print(f"  Best alpha: {model.alpha_:.1f}")

    # ----- Evaluate on training set -----
    print("\n--- Training Set ---")
    y_train_pred = model.predict(X_train_scaled)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    train_corr = np.corrcoef(y_train, y_train_pred)[0, 1]
    print(f"  MAE:  {train_mae:.4f}")
    print(f"  RMSE: {train_rmse:.4f}")
    print(f"  Corr: {train_corr:.4f}")

    # ----- Evaluate on test set -----
    print("\n--- Test Set (OOS) ---")
    if len(test) > 0:
        y_test_pred = model.predict(X_test_scaled)
        test_mae = mean_absolute_error(y_test, y_test_pred)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_corr = np.corrcoef(y_test, y_test_pred)[0, 1]
        print(f"  MAE:  {test_mae:.4f}")
        print(f"  RMSE: {test_rmse:.4f}")
        print(f"  Corr: {test_corr:.4f}")

        # Naive baseline: predict 0 (mean residual is ~0 by definition)
        naive_mae = mean_absolute_error(y_test, np.zeros_like(y_test))
        naive_rmse = np.sqrt(mean_squared_error(y_test, np.zeros_like(y_test)))
        print(f"\n  Naive baseline (predict 0):")
        print(f"    MAE:  {naive_mae:.4f}")
        print(f"    RMSE: {naive_rmse:.4f}")
        print(f"  Model improvement: MAE {(1 - test_mae / naive_mae) * 100:.1f}%, "
              f"RMSE {(1 - test_rmse / naive_rmse) * 100:.1f}%")

        print(f"\n  Prediction distribution:")
        print(f"    Mean: {y_test_pred.mean():.4f}")
        print(f"    Std:  {y_test_pred.std():.4f}")
        print(f"    Range: [{y_test_pred.min():.4f}, {y_test_pred.max():.4f}]")
    else:
        print("  No test data available")

    # ----- Feature importance -----
    print(f"\n  Feature coefficients (standardized):")
    coefs = sorted(zip(FEATURE_COLS, model.coef_),
                   key=lambda x: abs(x[1]), reverse=True)
    for feat, coef in coefs:
        print(f"    {feat:25s}: {coef:+.4f}")
    print(f"  Intercept: {model.intercept_:.4f}")

    # ----- Save model -----
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    artifact = {
        "model": model,
        "scaler": scaler,
        "feature_cols": FEATURE_COLS,
        "alpha": model.alpha_,
        "train_cutoff": train_cutoff,
    }
    with open(model_file, "wb") as f:
        pickle.dump(artifact, f)
    print("  Model saved.")

    # ----- Generate predictions for ALL valid games -----
    print(f"\nGenerating goalie GA rate for all goalie-game rows...")
    X_all = df_valid[FEATURE_COLS].values
    X_all_scaled = scaler.transform(X_all)
    df_valid = df_valid.copy()

    # Raw Ridge output (zero-centered adjustment)
    df_valid["goalie_deployment_adjustment"] = model.predict(X_all_scaled)

    # Reconstitute to GA rate: baseline + adjustment → 2-4 range
    df_valid["team_ga_baseline"] = df_valid["team_ga_ewm_20"]
    df_valid["goalie_ga_rate"] = (
        df_valid["team_ga_baseline"] + df_valid["goalie_deployment_adjustment"]
    )

    # Output columns: identifiers + features + GA rate outputs
    output_cols = [
        "game_id", "game_date", "goalie_id", "goalie_name",
        "team", "opponent", "is_home", "season",
    ] + FEATURE_COLS + [
        "goalie_ga_rate",                # Final usable prediction (2-4 range)
        "team_ga_baseline",              # team_ga_ewm_20 (for transparency)
        "goalie_deployment_adjustment",  # Raw Ridge output (for diagnostics)
        "team_ga_residual",              # Actual target (for diagnostics)
        "reg_ga",                        # Actual regulation GA (for calibration)
    ]
    output = df_valid[output_cols].copy()

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.to_csv(output_file, index=False)
    print(f"  Saved {len(output):,} predictions to {output_file}")

    # ----- Summary -----
    print(f"\n--- Goalie GA Rate Summary ---")
    print(f"  goalie_ga_rate:  mean={output['goalie_ga_rate'].mean():.3f}, "
          f"std={output['goalie_ga_rate'].std():.3f}, "
          f"range=[{output['goalie_ga_rate'].min():.3f}, {output['goalie_ga_rate'].max():.3f}]")
    print(f"  team_ga_baseline: mean={output['team_ga_baseline'].mean():.3f}")
    print(f"  adjustment:       mean={output['goalie_deployment_adjustment'].mean():.4f}, "
          f"std={output['goalie_deployment_adjustment'].std():.4f}")
    print(f"  actual reg_ga:    mean={output['reg_ga'].mean():.3f}")

    # Calibration check: GA rate vs actual reg GA
    valid_cal = output.dropna(subset=["goalie_ga_rate", "reg_ga"])
    if len(valid_cal) > 50:
        cal_corr = valid_cal["goalie_ga_rate"].corr(valid_cal["reg_ga"])
        print(f"\n  Calibration: goalie_ga_rate vs actual reg GA:")
        print(f"    Correlation: {cal_corr:.3f}")

    # GA rate by deployment context
    print(f"\n  GA rate by goalie_switch:")
    for val in [0, 1]:
        subset = output[output["goalie_switch"] == val]
        print(f"    switch={val}: n={len(subset):,}, "
              f"mean_ga_rate={subset['goalie_ga_rate'].mean():.3f}, "
              f"mean_actual={subset['reg_ga'].mean():.3f}")

    print(f"\n  GA rate by is_back_to_back:")
    for val in [0, 1]:
        subset = output[output["is_back_to_back"] == val]
        print(f"    b2b={val}: n={len(subset):,}, "
              f"mean_ga_rate={subset['goalie_ga_rate'].mean():.3f}, "
              f"mean_actual={subset['reg_ga'].mean():.3f}")

    print(f"\n  GA rate by starter role (primary vs backup):")
    primary = output[output["starter_role_share"] >= 0.55]
    backup = output[output["starter_role_share"] < 0.55]
    print(f"    Primary (share>=0.55): n={len(primary):,}, "
          f"mean_ga_rate={primary['goalie_ga_rate'].mean():.3f}, "
          f"mean_actual={primary['reg_ga'].mean():.3f}")
    print(f"    Backup  (share<0.55):  n={len(backup):,}, "
          f"mean_ga_rate={backup['goalie_ga_rate'].mean():.3f}, "
          f"mean_actual={backup['reg_ga'].mean():.3f}")

    return model, scaler


# =============================================================================
# Prediction Helper (for use by other modules)
# =============================================================================

def load_goalie_model(cutoff_year="2025"):
    """Load trained goalie deployment model and scaler for a given cutoff year."""
    model_file = os.path.join(MODEL_DIR, f"goalie_model_{cutoff_year}.pkl")
    with open(model_file, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["scaler"], artifact["feature_cols"]


def predict_goalie_ga_rate(features_df, team_ga_baseline, model=None, scaler=None):
    """
    Predict goalie-adjusted GA rate for a DataFrame.

    Parameters:
        features_df: DataFrame with required feature columns
        team_ga_baseline: Series/array of team_ga_ewm_20 values
        model, scaler: optional, loads from disk if not provided

    Returns:
        goalie_ga_rate: team_ga_baseline + Ridge adjustment (2-4 range)
    """
    if model is None or scaler is None:
        model, scaler, feature_cols = load_goalie_model()
    else:
        feature_cols = FEATURE_COLS

    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)
    adjustment = model.predict(X_scaled)
    return team_ga_baseline + adjustment


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    for cutoff in TRAIN_CUTOFFS:
        print("=" * 60)
        print(f"Goalie Deployment Model (Layer 0 v3) — cutoff {cutoff}")
        print("=" * 60)

        train_goalie_model(train_cutoff=cutoff)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
