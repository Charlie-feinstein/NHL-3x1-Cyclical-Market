# -*- coding: utf-8 -*-
"""
Goalie Feature Engineering (v2 — Deployment Context)

Computes per-goalie-per-game GSAx from shot-level data (for diagnostics),
then builds DEPLOYMENT CONTEXT features that capture the goalie situation
without relying on goalie skill metrics.

Why deployment context instead of GSAx rolling stats?
  - Per-game GSAx is ~95% noise (std=1.60, lag-1 autocorrelation ≈ 0)
  - Rolling GSAx averages compress to near-zero (range [-0.15, +0.15])
  - OOS prediction correlation was -0.007 (dead)
  - Deployment features (who's starting, rest, workload, coaching signals)
    capture systematic patterns without trying to predict goalie skill

Deployment features (all shift(1) anti-leakage):
  - starter_role_share: fraction of team's last 30 games this goalie started
  - start_share_trend: 10-game share minus 30-game share (rising/falling)
  - goalie_switch: team started a different goalie than last game
  - consecutive_starts: how many starts in a row for this goalie
  - was_pulled_last_game: goalie was replaced mid-game in previous start
  - days_rest, is_back_to_back, season_games_started, games_started_last_14d

Output:
  - data/processed/goalie_game_features.csv

@author: chazf
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =============================================================================
# Configuration
# =============================================================================
from config import PROCESSED_DIR, RAW_DIR

GOALIE_BOX_FILE = os.path.join(RAW_DIR, "goalie_boxscores.csv")
GAME_FILE = os.path.join(RAW_DIR, "game_ids.csv")

# Cutoff years to process (one per backtest season)
CUTOFF_YEARS = ["2024", "2025"]

# High-danger shot threshold (xG >= this = high-danger)
HD_XG_THRESHOLD = 0.10

# Starter role share lookback windows (in team games)
ROLE_WINDOW_LONG = 30
ROLE_WINDOW_SHORT = 10


# =============================================================================
# Per-Game GSAx Computation (kept for diagnostics, NOT model features)
# =============================================================================

def compute_goalie_game_stats(shots):
    """
    Compute per-goalie-per-game xG stats from shot-level data.

    xGA uses ALL unblocked shots (SOG + goals + missed) because:
      - The xG model was trained/calibrated on this full set
      - sum(xG, all unblocked) ≈ total goals (by calibration)
      - This is the standard approach (MoneyPuck, Evolving Hockey)

    Save percentage uses only SOG + goals (traditional definition).
    Excludes empty-net shots (no goalie to evaluate).
    """
    unblocked = shots.copy()
    unblocked = unblocked[unblocked["is_empty_net"] == 0]
    unblocked = unblocked.dropna(subset=["goalie_id"])
    unblocked["goalie_id"] = unblocked["goalie_id"].astype(int)
    print(f"  All unblocked shots (non-empty-net): {len(unblocked):,}")

    unblocked["is_sog"] = unblocked["event_type"].isin(
        ["shot-on-goal", "goal"]
    ).astype(int)
    unblocked["is_hd"] = (unblocked["xg"] >= HD_XG_THRESHOLD).astype(int)
    unblocked["is_hd_goal"] = (
        (unblocked["is_hd"] == 1) & (unblocked["is_goal"] == 1)
    ).astype(int)
    unblocked["hd_xg"] = unblocked["xg"] * unblocked["is_hd"]

    stats = unblocked.groupby(["game_id", "goalie_id"]).agg(
        unblocked_against=("xg", "count"),
        shots_faced=("is_sog", "sum"),
        goals_against=("is_goal", "sum"),
        xga=("xg", "sum"),
        hd_shots=("is_hd", "sum"),
        hd_goals_against=("is_hd_goal", "sum"),
        hd_xga=("hd_xg", "sum"),
    ).reset_index()

    stats["gsax"] = stats["xga"] - stats["goals_against"]
    stats["save_pct"] = np.where(
        stats["shots_faced"] > 0,
        1 - stats["goals_against"] / stats["shots_faced"],
        np.nan
    )
    stats["hd_gsax"] = stats["hd_xga"] - stats["hd_goals_against"]

    return stats


# =============================================================================
# Pulled Goalie Detection
# =============================================================================

def detect_pulled_starters(goalie_box):
    """
    Detect games where the starter was pulled (replaced mid-game).

    Logic: if >1 goalie appeared for the same team in a game, and the
    non-starter had actual TOI, the starter was pulled.

    Returns DataFrame with (game_id, team, starter_was_pulled).
    """
    # Count goalies per team per game who actually played
    played = goalie_box[goalie_box["toi"] != "00:00"].copy()
    goalie_counts = played.groupby(["game_id", "team"]).size().reset_index(name="n_goalies")
    goalie_counts["starter_was_pulled"] = (goalie_counts["n_goalies"] > 1).astype(int)

    return goalie_counts[["game_id", "team", "starter_was_pulled"]]


# =============================================================================
# Deployment Feature Computation
# =============================================================================

def compute_deployment_features(df, pulled_df):
    """
    Compute deployment context features from prior games only.

    All features use shift(1) or lookback logic to prevent leakage.
    Includes:
      - Deployment context (situation): rest, workload, role, coaching signals
      - Rolling GSAx (goalie quality): EWM averages of per-game GSAx stats
    """
    # === 0. Rolling GSAx features (goalie quality) ===
    # Per-game GSAx is noisy, but EWM averages carry real signal.
    # shift(1) = only past starts, no leakage.
    print("  Computing rolling GSAx features...")
    df["gsax_ewm_20"] = df.groupby("goalie_id")["gsax"].transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=5).mean()
    )
    df["gsax_ewm_5"] = df.groupby("goalie_id")["gsax"].transform(
        lambda x: x.shift(1).ewm(span=5, min_periods=3).mean()
    )
    df["save_pct_ewm_20"] = df.groupby("goalie_id")["save_pct"].transform(
        lambda x: x.shift(1).ewm(span=20, min_periods=5).mean()
    )
    df["hd_gsax_ewm_10"] = df.groupby("goalie_id")["hd_gsax"].transform(
        lambda x: x.shift(1).ewm(span=10, min_periods=3).mean()
    )

    # === 1. Days rest + back-to-back (goalie-specific, not team-level) ===
    print("  Computing rest/workload features...")
    df["prev_game_date"] = df.groupby("goalie_id")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(float)
    df["days_rest"] = df["days_rest"].clip(upper=30).fillna(7.0)
    df["is_back_to_back"] = df["is_back_to_back"].fillna(0).astype(int)
    df.drop(columns=["prev_game_date"], inplace=True)

    # === 2. Season games started (before current game) ===
    df["season_games_started"] = df.groupby(["goalie_id", "season"]).cumcount()

    # === 3. Games started in last 14 days ===
    print("  Computing games started in last 14 days...")
    starts_14d = []
    for goalie_id, group in df.groupby("goalie_id"):
        dates = group["game_date"].values
        counts = np.zeros(len(dates), dtype=int)
        for i in range(len(dates)):
            cutoff = dates[i] - np.timedelta64(14, 'D')
            counts[i] = np.sum(dates[:i] > cutoff)
        starts_14d.append(pd.Series(counts, index=group.index))
    df["games_started_last_14d"] = pd.concat(starts_14d)

    # === 4. Was pulled last game ===
    print("  Computing was_pulled_last_game...")
    df = df.merge(pulled_df, on=["game_id", "team"], how="left")
    df["starter_was_pulled"] = df["starter_was_pulled"].fillna(0).astype(int)
    # Shift: was THIS goalie pulled in their PREVIOUS start?
    df["was_pulled_last_game"] = df.groupby("goalie_id")["starter_was_pulled"].shift(1)
    df["was_pulled_last_game"] = df["was_pulled_last_game"].fillna(0).astype(int)
    df.drop(columns=["starter_was_pulled"], inplace=True)

    # === 5. Goalie switch (team started different goalie than last game) ===
    # Need team-level view: for each team's game, who started?
    print("  Computing goalie_switch...")
    team_starters = df[["game_id", "game_date", "team", "goalie_id"]].copy()
    team_starters = team_starters.sort_values(["team", "game_date", "game_id"])
    team_starters["prev_team_goalie"] = team_starters.groupby("team")["goalie_id"].shift(1)
    team_starters["goalie_switch"] = (
        (team_starters["goalie_id"] != team_starters["prev_team_goalie"]) &
        team_starters["prev_team_goalie"].notna()
    ).astype(int)
    # First game of team has NaN prev_team_goalie → goalie_switch = 0
    team_starters.loc[team_starters["prev_team_goalie"].isna(), "goalie_switch"] = 0

    df = df.merge(
        team_starters[["game_id", "team", "goalie_switch"]],
        on=["game_id", "team"], how="left"
    )

    # === 6. Consecutive starts ===
    print("  Computing consecutive_starts...")
    # For each row, count how many consecutive games this goalie started
    # for this team (going backwards). Resets when team used a different goalie.
    consec = []
    for team, group in df.groupby("team"):
        group = group.sort_values(["game_date", "game_id"])
        streaks = np.zeros(len(group), dtype=int)
        current_goalie = None
        current_streak = 0
        for i, (_, row) in enumerate(group.iterrows()):
            if row["goalie_id"] == current_goalie:
                current_streak += 1
            else:
                current_streak = 1
                current_goalie = row["goalie_id"]
            # Shift: use streak count BEFORE this game (i.e., how many
            # consecutive starts they had coming into this game)
            streaks[i] = current_streak - 1  # 0 = first start in streak
        consec.append(pd.Series(streaks, index=group.index))
    df["consecutive_starts"] = pd.concat(consec)

    # === 7. Starter role share (fraction of team's last N games) ===
    print("  Computing starter_role_share (30-game and 10-game windows)...")
    role_shares_long = []
    role_shares_short = []
    for team, group in df.groupby("team"):
        group = group.sort_values(["game_date", "game_id"])
        goalie_ids = group["goalie_id"].values
        shares_long = np.full(len(group), np.nan)
        shares_short = np.full(len(group), np.nan)

        for i in range(len(group)):
            # Long window (30 games)
            start_long = max(0, i - ROLE_WINDOW_LONG)
            lookback_long = goalie_ids[start_long:i]  # excludes current game
            if len(lookback_long) > 0:
                shares_long[i] = np.mean(lookback_long == goalie_ids[i])

            # Short window (10 games)
            start_short = max(0, i - ROLE_WINDOW_SHORT)
            lookback_short = goalie_ids[start_short:i]
            if len(lookback_short) > 0:
                shares_short[i] = np.mean(lookback_short == goalie_ids[i])

        role_shares_long.append(pd.Series(shares_long, index=group.index))
        role_shares_short.append(pd.Series(shares_short, index=group.index))

    df["starter_role_share"] = pd.concat(role_shares_long)
    starter_role_share_short = pd.concat(role_shares_short)

    # === 8. Start share trend (short - long) ===
    # Positive = rising (taking over), negative = falling (losing job)
    df["start_share_trend"] = starter_role_share_short - df["starter_role_share"]
    # NaN when not enough games for both windows
    df["start_share_trend"] = df["start_share_trend"].fillna(0.0)

    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def build_goalie_features(cutoff_year="2025", starters_override=None):
    """
    Load data, compute GSAx (diagnostics), build deployment features, save.

    Args:
        cutoff_year: Season cutoff year (e.g., "2025" for 2025-26 season).
        starters_override: Optional DataFrame with columns:
            game_id, game_date, team, goalie_id, goalie_name, is_home, opponent
            Used for today's unplayed games where goalie_boxscores has no
            starter yet. Past completed games still use actual boxscore data.
    """
    shot_xg_file = os.path.join(PROCESSED_DIR, f"shot_xg_{cutoff_year}.csv")
    output_file = os.path.join(PROCESSED_DIR, f"goalie_game_features_{cutoff_year}.csv")

    print(f"\n{'#'*60}")
    print(f"  Cutoff year:  {cutoff_year}")
    print(f"  Shot xG file: {shot_xg_file}")
    print(f"  Output file:  {output_file}")
    if starters_override is not None:
        print(f"  Starters override: {len(starters_override)} rows")
    print(f"{'#'*60}")

    print("Loading data...")
    shots = pd.read_csv(shot_xg_file)
    shots["game_date"] = pd.to_datetime(shots["game_date"])
    print(f"  Shots: {len(shots):,}")

    goalie_box = pd.read_csv(GOALIE_BOX_FILE)
    goalie_box["game_date"] = pd.to_datetime(goalie_box["game_date"])
    print(f"  Goalie boxscores: {len(goalie_box):,}")

    games = pd.read_csv(GAME_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    print(f"  Games: {len(games):,}")

    # ----- Step 1: Per-goalie-per-game xG stats (for diagnostics) -----
    print("\nComputing per-goalie-per-game GSAx...")
    goalie_stats = compute_goalie_game_stats(shots)
    print(f"  Goalie-game rows from shots: {len(goalie_stats):,}")

    # ----- Step 2: Detect pulled starters -----
    print("Detecting pulled starters...")
    pulled_df = detect_pulled_starters(goalie_box)
    n_pulled = pulled_df["starter_was_pulled"].sum()
    print(f"  Games with pulled starter: {n_pulled:,} "
          f"({n_pulled / len(pulled_df) * 100:.1f}%)")

    # ----- Step 3: Get starters from goalie_boxscores -----
    starters = goalie_box[goalie_box["starter"].astype(str) == "True"].copy()
    starters = starters[["game_id", "game_date", "goalie_id", "goalie_name",
                          "team", "opponent", "is_home"]].copy()
    print(f"  Starters from boxscores: {len(starters):,}")

    # ----- Step 3b: Inject override starters for unplayed games -----
    if starters_override is not None and len(starters_override) > 0:
        override = starters_override.copy()
        override["game_date"] = pd.to_datetime(override["game_date"])
        # Only inject for game_ids NOT already in boxscore starters
        existing_ids = set(starters["game_id"].unique())
        new_rows = override[~override["game_id"].isin(existing_ids)]
        if len(new_rows) > 0:
            # Ensure columns match
            inject_cols = ["game_id", "game_date", "goalie_id", "goalie_name",
                           "team", "opponent", "is_home"]
            new_rows = new_rows[inject_cols].copy()
            starters = pd.concat([starters, new_rows], ignore_index=True)
            print(f"  Injected {len(new_rows)} override starters for unplayed games")

    print(f"  Total starters: {len(starters):,}")

    # ----- Step 4: Join starters with xG stats -----
    df = starters.merge(goalie_stats, on=["game_id", "goalie_id"], how="left")

    fill_cols = ["unblocked_against", "shots_faced", "goals_against", "xga", "gsax",
                 "hd_shots", "hd_goals_against", "hd_xga", "hd_gsax"]
    for col in fill_cols:
        df[col] = df[col].fillna(0)

    df = df.merge(games[["game_id", "season"]], on="game_id", how="left")
    df = df.sort_values(["goalie_id", "game_date", "game_id"]).reset_index(drop=True)

    print(f"\n  Starter-game rows: {len(df):,}")
    print(f"  Unique goalies: {df['goalie_id'].nunique()}")
    print(f"  GSAx summary (diagnostic only):")
    print(f"    Mean:  {df['gsax'].mean():.4f}")
    print(f"    Std:   {df['gsax'].std():.4f}")

    # ----- Step 5: Deployment features -----
    print("\nComputing deployment features...")
    df = compute_deployment_features(df, pulled_df)

    # ----- Step 6: Save -----
    output_cols = [
        # Identifiers
        "game_id", "game_date", "goalie_id", "goalie_name",
        "team", "opponent", "is_home", "season",

        # Raw stats (for diagnostics, NOT model features)
        "unblocked_against", "shots_faced", "goals_against", "xga", "gsax",
        "hd_shots", "hd_goals_against", "hd_gsax",
        "save_pct",

        # Rolling GSAx features (goalie quality — EWM of per-game stats)
        "gsax_ewm_20",
        "gsax_ewm_5",
        "save_pct_ewm_20",
        "hd_gsax_ewm_10",

        # Deployment features (model inputs)
        "starter_role_share",
        "start_share_trend",
        "goalie_switch",
        "consecutive_starts",
        "was_pulled_last_game",
        "days_rest",
        "is_back_to_back",
        "season_games_started",
        "games_started_last_14d",
    ]

    output = df[output_cols].copy()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.to_csv(output_file, index=False)

    print(f"\nSaved {len(output):,} goalie-game rows to {output_file}")

    # ----- Sanity checks -----
    print(f"\n--- Sanity Checks ---")
    print(f"  Mean GSAx: {df['gsax'].mean():.4f} (should be ~0)")

    games_per_game = df.groupby("game_id").size()
    print(f"  Starters per game: mean={games_per_game.mean():.2f} (should be ~2.0)")

    starts_per = df.groupby("goalie_id").size()
    print(f"  Starts per goalie: median={starts_per.median():.0f}, "
          f"mean={starts_per.mean():.1f}, max={starts_per.max()}")

    # Deployment feature distributions
    print(f"\n  Deployment feature distributions:")
    print(f"    starter_role_share: mean={df['starter_role_share'].mean():.3f}, "
          f"std={df['starter_role_share'].std():.3f}")
    print(f"    goalie_switch rate: {df['goalie_switch'].mean():.3f} "
          f"({df['goalie_switch'].sum():,} switches)")
    print(f"    was_pulled_last_game rate: {df['was_pulled_last_game'].mean():.3f} "
          f"({df['was_pulled_last_game'].sum():,} games)")
    print(f"    consecutive_starts: mean={df['consecutive_starts'].mean():.1f}, "
          f"max={df['consecutive_starts'].max()}")
    print(f"    days_rest: mean={df['days_rest'].mean():.1f}, "
          f"median={df['days_rest'].median():.0f}")

    # GSAx rolling feature distributions
    print(f"\n  Rolling GSAx feature distributions:")
    gsax_cols = ["gsax_ewm_20", "gsax_ewm_5", "save_pct_ewm_20", "hd_gsax_ewm_10"]
    for col in gsax_cols:
        valid = df[col].dropna()
        print(f"    {col:20s}: mean={valid.mean():.4f}, std={valid.std():.4f}, "
              f"range=[{valid.min():.4f}, {valid.max():.4f}], coverage={len(valid)/len(df)*100:.1f}%")

    # Feature coverage
    deploy_cols = ["gsax_ewm_20", "gsax_ewm_5", "save_pct_ewm_20", "hd_gsax_ewm_10",
                   "starter_role_share", "start_share_trend", "goalie_switch",
                   "consecutive_starts", "was_pulled_last_game",
                   "days_rest", "is_back_to_back",
                   "season_games_started", "games_started_last_14d"]
    print(f"\n  Feature coverage (% non-null):")
    for col in deploy_cols:
        pct = df[col].notna().mean() * 100
        print(f"    {col:25s}: {pct:.1f}%")

    # Top 10 goalies by starts
    print(f"\n  Top 10 goalies by starts:")
    top = df.groupby(["goalie_id", "goalie_name"]).agg(
        starts=("game_id", "count"),
        mean_gsax=("gsax", "mean"),
        mean_role_share=("starter_role_share", "mean"),
    ).sort_values("starts", ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"    {row.name[1]:20s}: {int(row['starts']):3d} starts, "
              f"GSAx={row['mean_gsax']:+.3f}, role_share={row['mean_role_share']:.2f}")

    return output


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    for year in CUTOFF_YEARS:
        print("=" * 60)
        print(f"Goalie Feature Engineering (v2) — cutoff year {year}")
        print("=" * 60)

        build_goalie_features(cutoff_year=year)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
