# -*- coding: utf-8 -*-
"""
Regulation Feature Engineering

Builds per-team-per-game features for predicting regulation goals (P1-3).
Each game produces 2 rows (home team + away team). All rolling features
use .shift(1) to prevent leakage.

Feature categories:
  - Scoring: goals, xG, shooting efficiency (EWM + tail stats)
  - Defense: goals against, xGA, save %
  - Possession: Corsi%, Fenwick%, faceoff %
  - Discipline: penalties, PIM, drawn
  - Physical: hits, blocks, turnover differential
  - Period scoring: P1/P3 tendencies (late push → OT proxy)
  - Distribution: variance, max, min, high/low scoring rates
  - Luck regression: xG conversion, PDO
  - Opponent mirrors: all key features for the opposing team
  - Goalie matchup: deployment context scores for both starters
  - Context: home/away, rest, back-to-back, standings

Output:
  - data/processed/regulation_features.csv

@author: chazf
"""

import pandas as pd
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROCESSED_DIR, RAW_DIR

# =============================================================================
# Configuration
# =============================================================================
# Input files (raw — single files, no per-cutoff needed)
GAME_FILE = os.path.join(RAW_DIR, "game_ids.csv")
STANDINGS_FILE = os.path.join(RAW_DIR, "standings_daily.csv")
PBP_FILE = os.path.join(RAW_DIR, "pbp_events.csv")

# REST API team stats
TEAM_SUMMARY_FILE = os.path.join(RAW_DIR, "team_summary.csv")
TEAM_PCT_FILE = os.path.join(RAW_DIR, "team_percentages.csv")
TEAM_PENALTIES_FILE = os.path.join(RAW_DIR, "team_penalties.csv")
TEAM_FACEOFFS_FILE = os.path.join(RAW_DIR, "team_faceoffs.csv")
TEAM_REALTIME_FILE = os.path.join(RAW_DIR, "team_realtime.csv")
TEAM_PP_FILE = os.path.join(RAW_DIR, "team_powerplay.csv")

# Cutoff years to process (one per backtest season)
CUTOFF_YEARS = ["2024", "2025"]


# =============================================================================
# Rolling Helpers
# =============================================================================

def add_ewm(df, group, src, span, name):
    """EWM feature with shift(1) — no leakage."""
    # Group by season + team so EWM resets each season (rosters change)
    if isinstance(group, str):
        ewm_group = ["season", group]
    else:
        ewm_group = ["season"] + list(group)
    df[name] = df.groupby(ewm_group)[src].transform(
        lambda x: x.shift(1).ewm(span=span, min_periods=max(3, span // 4)).mean()
    )

def add_rolling(df, group, src, window, name, func="mean"):
    """Rolling stat with shift(1) — supports mean/std/max/min."""
    df[name] = df.groupby(group)[src].transform(
        lambda x: x.shift(1).rolling(window, min_periods=max(3, window // 3)).agg(func)
    )


# =============================================================================
# 1. Base Table
# =============================================================================

def build_base_table(games):
    """
    Create one row per team-game from game_ids.csv.
    Each game → 2 rows (home + away).
    """
    home = games[["game_id", "game_date", "season",
                   "home_team_id", "home_team", "away_team",
                   "period_count", "game_outcome_type"]].copy()
    home.rename(columns={"home_team_id": "team_id",
                         "home_team": "team",
                         "away_team": "opponent"}, inplace=True)
    home["is_home"] = 1

    away = games[["game_id", "game_date", "season",
                   "away_team_id", "away_team", "home_team",
                   "period_count", "game_outcome_type"]].copy()
    away.rename(columns={"away_team_id": "team_id",
                         "away_team": "team",
                         "home_team": "opponent"}, inplace=True)
    away["is_home"] = 0

    base = pd.concat([home, away], ignore_index=True)
    base = base.sort_values(["game_date", "game_id", "is_home"],
                            ascending=[True, True, False]).reset_index(drop=True)
    return base


# =============================================================================
# 2. Shot-Based Stats (regulation-only from shot_xg.csv)
# =============================================================================

def compute_shot_game_stats(shots):
    """
    Aggregate shot_xg.csv to per-team-per-game stats.
    Computes regulation (P1-3) goals, xG, and period breakdowns.
    """
    shots = shots.copy()
    shots["period"] = shots["period"].astype(int)

    results = []
    for is_home_val in [True, False]:
        team_shots = shots[shots["is_home_event"] == is_home_val]
        opp_shots = shots[shots["is_home_event"] != is_home_val]

        # --- Regulation stats (P1-3) ---
        t_reg = team_shots[team_shots["period"] <= 3]
        o_reg = opp_shots[opp_shots["period"] <= 3]

        t_agg = t_reg.groupby("game_id").agg(
            reg_gf=("is_goal", "sum"),
            reg_xgf=("xg", "sum"),
            reg_shots_for=("xg", "count"),
        )
        # SOG only (exclude missed shots for traditional SOG)
        t_sog = t_reg[t_reg["event_type"].isin(["shot-on-goal", "goal"])]
        t_sog_ct = t_sog.groupby("game_id").size().rename("reg_sog_for")
        t_agg = t_agg.join(t_sog_ct, how="left")

        o_agg = o_reg.groupby("game_id").agg(
            reg_ga=("is_goal", "sum"),
            reg_xga=("xg", "sum"),
            reg_shots_against=("xg", "count"),
        )

        # --- Period-specific goals ---
        for p in [1, 2, 3]:
            p_gf = team_shots[team_shots["period"] == p].groupby("game_id")["is_goal"].sum()
            p_ga = opp_shots[opp_shots["period"] == p].groupby("game_id")["is_goal"].sum()
            t_agg[f"p{p}_gf"] = p_gf
            o_agg[f"p{p}_ga"] = p_ga

        combined = t_agg.join(o_agg, how="outer").reset_index()
        combined["is_home"] = 1 if is_home_val else 0
        results.append(combined)

    out = pd.concat(results, ignore_index=True)
    # Fill NaN period goals with 0
    period_cols = [f"p{p}_{s}" for p in [1, 2, 3] for s in ["gf", "ga"]]
    for col in period_cols + ["reg_sog_for"]:
        if col in out.columns:
            out[col] = out[col].fillna(0).astype(int)
    return out


# =============================================================================
# 2b. Special Teams Stats (from PBP)
# =============================================================================

def compute_special_teams_game_stats(pbp, games):
    """
    Compute per-team-per-game PP/PK stats from PBP data.
    PP = team has man advantage (more skaters on ice).
    PK = opponent has man advantage (fewer skaters on ice).
    Regulation periods only (P1-3).
    """
    reg = pbp[pbp["period"].astype(int) <= 3].copy()
    reg["home_sk"] = pd.to_numeric(reg["home_skaters"], errors="coerce")
    reg["away_sk"] = pd.to_numeric(reg["away_skaters"], errors="coerce")

    # Shot and goal events
    shot_types = ["shot-on-goal", "goal", "missed-shot"]
    reg_shots = reg[reg["event_type"].isin(shot_types)].copy()
    reg_shots["is_goal"] = (reg_shots["event_type"] == "goal").astype(int)

    results = []
    for is_home_val in [True, False]:
        # PP offense: own team has more skaters and is shooting
        if is_home_val:
            pp_mask = (reg_shots["home_sk"] > reg_shots["away_sk"]) & \
                      (reg_shots["is_home_event"] == True)
            pk_mask = (reg_shots["away_sk"] > reg_shots["home_sk"]) & \
                      (reg_shots["is_home_event"] == False)
        else:
            pp_mask = (reg_shots["away_sk"] > reg_shots["home_sk"]) & \
                      (reg_shots["is_home_event"] == False)
            pk_mask = (reg_shots["home_sk"] > reg_shots["away_sk"]) & \
                      (reg_shots["is_home_event"] == True)

        pp_agg = reg_shots[pp_mask].groupby("game_id").agg(
            pp_goals_pbp=("is_goal", "sum"),
            pp_shots_pbp=("is_goal", "count"),
        )
        pk_agg = reg_shots[pk_mask].groupby("game_id").agg(
            pk_goals_against_pbp=("is_goal", "sum"),
            pk_shots_against_pbp=("is_goal", "count"),
        )

        combined = pp_agg.join(pk_agg, how="outer").fillna(0).reset_index()
        combined["is_home"] = 1 if is_home_val else 0
        results.append(combined)

    out = pd.concat(results, ignore_index=True)

    # Derived rates
    out["pp_conversion_rate_pbp"] = np.where(
        out["pp_shots_pbp"] > 0,
        out["pp_goals_pbp"] / out["pp_shots_pbp"],
        0
    )
    out["pk_save_rate_pbp"] = np.where(
        out["pk_shots_against_pbp"] > 0,
        1 - out["pk_goals_against_pbp"] / out["pk_shots_against_pbp"],
        1.0
    )

    # Ensure all team-games have a row
    game_info = games[["game_id", "game_date", "home_team", "away_team"]].copy()
    home_info = game_info.rename(columns={"home_team": "team"})[["game_id", "game_date", "team"]]
    home_info["is_home"] = 1
    away_info = game_info.rename(columns={"away_team": "team"})[["game_id", "game_date", "team"]]
    away_info["is_home"] = 0
    all_team_games = pd.concat([home_info, away_info])

    out = all_team_games.merge(out, on=["game_id", "is_home"], how="left")
    for col in ["pp_goals_pbp", "pp_shots_pbp", "pk_goals_against_pbp",
                "pk_shots_against_pbp", "pp_conversion_rate_pbp", "pk_save_rate_pbp"]:
        out[col] = out[col].fillna(0)

    return out[["game_id", "is_home", "pp_goals_pbp", "pp_shots_pbp",
                "pk_goals_against_pbp", "pk_shots_against_pbp",
                "pp_conversion_rate_pbp", "pk_save_rate_pbp"]]


# =============================================================================
# 2c. Score-State Stats (from PBP)
# =============================================================================

def compute_score_state_game_stats(pbp, games):
    """
    Compute per-team-per-game goals by score state from PBP.
    - trailing_gf: goals scored when trailing (comeback ability)
    - leading_ga: goals allowed when leading (turtle tendency)
    - scored_first: binary, did team score first?
    - tied_gf: goals scored when score is tied
    """
    reg = pbp[pbp["period"].astype(int) <= 3].copy()
    goals = reg[reg["event_type"] == "goal"].copy()

    # Score is AFTER the goal, so adjust to get state BEFORE the goal
    goals["score_diff_before"] = np.where(
        goals["is_home_event"] == True,
        goals["home_score"].fillna(0) - goals["away_score"].fillna(0) - 1,
        goals["home_score"].fillna(0) - goals["away_score"].fillna(0) + 1,
    )

    # Determine first goal per game
    goals = goals.sort_values(["game_id", "period", "time_in_period"])
    first_goals = goals.groupby("game_id").first().reset_index()

    results = []
    for is_home_val in [True, False]:
        # From home perspective: positive diff = home leading
        # For home team: trailing when score_diff_before < 0
        # For away team: trailing when score_diff_before > 0
        t_goals = goals[goals["is_home_event"] == is_home_val]
        o_goals = goals[goals["is_home_event"] != is_home_val]

        if is_home_val:
            trailing = t_goals[t_goals["score_diff_before"] < 0]
            leading_opp = o_goals[o_goals["score_diff_before"] > 0]  # opp scores when home leads
            tied = t_goals[t_goals["score_diff_before"] == 0]
        else:
            trailing = t_goals[t_goals["score_diff_before"] > 0]
            leading_opp = o_goals[o_goals["score_diff_before"] < 0]  # opp scores when away leads
            tied = t_goals[t_goals["score_diff_before"] == 0]

        trailing_agg = trailing.groupby("game_id")["event_type"].count().rename("trailing_gf")
        leading_ga_agg = leading_opp.groupby("game_id")["event_type"].count().rename("leading_ga")
        tied_agg = tied.groupby("game_id")["event_type"].count().rename("tied_gf")

        # Scored first
        first_mask = first_goals["is_home_event"] == is_home_val
        scored_first_games = set(first_goals[first_mask]["game_id"])

        combined = pd.DataFrame({"game_id": goals["game_id"].unique()})
        combined = combined.merge(trailing_agg, on="game_id", how="left")
        combined = combined.merge(leading_ga_agg, on="game_id", how="left")
        combined = combined.merge(tied_agg, on="game_id", how="left")
        combined["scored_first"] = combined["game_id"].isin(scored_first_games).astype(float)
        combined["is_home"] = 1 if is_home_val else 0
        results.append(combined)

    out = pd.concat(results, ignore_index=True)
    for col in ["trailing_gf", "leading_ga", "tied_gf", "scored_first"]:
        out[col] = out[col].fillna(0)

    # Ensure all team-games have a row
    game_info = games[["game_id", "game_date", "home_team", "away_team"]].copy()
    home_info = game_info.rename(columns={"home_team": "team"})[["game_id", "game_date", "team"]]
    home_info["is_home"] = 1
    away_info = game_info.rename(columns={"away_team": "team"})[["game_id", "game_date", "team"]]
    away_info["is_home"] = 0
    all_team_games = pd.concat([home_info, away_info])

    out = all_team_games.merge(out, on=["game_id", "is_home"], how="left")
    for col in ["trailing_gf", "leading_ga", "tied_gf", "scored_first"]:
        out[col] = out[col].fillna(0)

    return out[["game_id", "is_home", "trailing_gf", "leading_ga", "scored_first", "tied_gf"]]


# =============================================================================
# 3. REST API Team Stats
# =============================================================================

def load_team_game_stats():
    """
    Load REST API team stats, merge into one wide table.
    Returns DataFrame keyed by (game_id, team_id).
    """
    KEY = ["game_id", "team_id"]

    def _load_dedup(path, col_map):
        """Load CSV, rename columns, deduplicate on (game_id, team_id)."""
        raw = pd.read_csv(path)
        renamed = raw.rename(columns=col_map)[[v for v in col_map.values()]]
        n_before = len(renamed)
        renamed = renamed.drop_duplicates(subset=KEY)
        n_dropped = n_before - len(renamed)
        if n_dropped > 0:
            print(f"    {os.path.basename(path)}: dropped {n_dropped:,} duplicate rows")
        return renamed

    # --- Team Summary ---
    merged = _load_dedup(TEAM_SUMMARY_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "goalsFor": "total_gf", "goalsAgainst": "total_ga",
        "shotsForPerGame": "sog_for_api", "shotsAgainstPerGame": "sog_against_api",
        "faceoffWinPct": "faceoff_pct",
        "wins": "game_win", "losses": "game_loss",
        "otLosses": "game_otl", "points": "game_points"}).copy()

    # --- Percentages (Corsi, PDO) ---
    pcts = _load_dedup(TEAM_PCT_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "satPct": "corsi_pct", "usatPct": "fenwick_pct",
        "shootingPct5v5": "shooting_pct_5v5",
        "savePct5v5": "save_pct_5v5",
        "satPctAhead": "corsi_ahead", "satPctBehind": "corsi_behind",
        "satPctClose": "corsi_close"})
    merged = merged.merge(pcts, on=KEY, how="left")

    # --- Penalties ---
    pen = _load_dedup(TEAM_PENALTIES_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "penalties": "penalties", "penaltyMinutes": "pim",
        "totalPenaltiesDrawn": "penalties_drawn",
        "majors": "major_penalties"})
    merged = merged.merge(pen, on=KEY, how="left")

    # --- Faceoffs ---
    fo = _load_dedup(TEAM_FACEOFFS_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "totalFaceoffs": "total_faceoffs"})
    merged = merged.merge(fo, on=KEY, how="left")

    # --- Realtime (physical play) ---
    rt = _load_dedup(TEAM_REALTIME_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "hits": "hits", "blockedShots": "blocks",
        "giveaways": "giveaways", "takeaways": "takeaways",
        "missedShots": "missed_shots"})
    merged = merged.merge(rt, on=KEY, how="left")

    # --- Power Play ---
    pp = _load_dedup(TEAM_PP_FILE, {
        "gameId": "game_id", "teamId": "team_id",
        "ppOpportunities": "pp_opportunities"})
    merged = merged.merge(pp, on=KEY, how="left")

    return merged


# =============================================================================
# 4. Rolling Features
# =============================================================================

def compute_all_rolling(df):
    """
    Compute rolling features from per-game raw stats.
    All use shift(1) to prevent leakage.
    """
    g = "team"  # group column

    # --- Derived per-game stats (before rolling) ---
    df["turnover_diff"] = df["takeaways"] - df["giveaways"]
    df["pdo_5v5"] = df["shooting_pct_5v5"] + df["save_pct_5v5"]
    df["xg_conversion"] = np.where(
        df["reg_xgf"] > 0, df["reg_gf"] / df["reg_xgf"], 1.0
    )
    df["reg_goal_diff"] = df["reg_gf"] - df["reg_ga"]
    df["p3_goal_diff"] = df["p3_gf"] - df["p3_ga"]

    # Binary flags for rate features
    df["is_high_scoring"] = (df["reg_gf"] >= 4).astype(float)
    df["is_shutout"] = (df["reg_gf"] == 0).astype(float)
    df["is_one_goal_game"] = (df["reg_goal_diff"].abs() <= 1).astype(float)
    df["is_blowout_win"] = ((df["reg_goal_diff"] >= 3)).astype(float)
    df["is_blowout_loss"] = ((df["reg_goal_diff"] <= -3)).astype(float)

    print("  Computing EWM features...")

    # ===================== EWM FEATURES =====================
    # --- Scoring (core) ---
    add_ewm(df, g, "reg_gf",     10, "gf_ewm_10")
    add_ewm(df, g, "reg_gf",     20, "gf_ewm_20")
    add_ewm(df, g, "reg_ga",     10, "ga_ewm_10")
    add_ewm(df, g, "reg_ga",     20, "ga_ewm_20")
    add_ewm(df, g, "reg_xgf",    10, "xgf_ewm_10")
    add_ewm(df, g, "reg_xgf",    20, "xgf_ewm_20")
    add_ewm(df, g, "reg_xga",    10, "xga_ewm_10")
    add_ewm(df, g, "reg_xga",    20, "xga_ewm_20")
    add_ewm(df, g, "reg_sog_for", 10, "sog_ewm_10")

    # --- Efficiency ---
    add_ewm(df, g, "shooting_pct_5v5", 20, "shooting_pct_ewm_20")
    add_ewm(df, g, "save_pct_5v5",     20, "save_pct_ewm_20")
    add_ewm(df, g, "xg_conversion",    20, "xg_conversion_ewm_20")
    add_ewm(df, g, "pdo_5v5",          20, "pdo_ewm_20")

    # --- Possession ---
    add_ewm(df, g, "corsi_pct",   20, "corsi_ewm_20")
    add_ewm(df, g, "fenwick_pct", 20, "fenwick_ewm_20")
    add_ewm(df, g, "faceoff_pct", 20, "faceoff_ewm_20")

    # --- Score effects (creative: how team plays when leading/trailing) ---
    add_ewm(df, g, "corsi_ahead",  20, "corsi_ahead_ewm_20")
    add_ewm(df, g, "corsi_behind", 20, "corsi_behind_ewm_20")
    add_ewm(df, g, "corsi_close",  20, "corsi_close_ewm_20")

    # --- Discipline ---
    add_ewm(df, g, "penalties",      10, "penalties_ewm_10")
    add_ewm(df, g, "pim",            10, "pim_ewm_10")
    add_ewm(df, g, "penalties_drawn", 10, "pen_drawn_ewm_10")
    add_ewm(df, g, "pp_opportunities", 10, "pp_opps_ewm_10")

    # --- Physical ---
    add_ewm(df, g, "hits",          10, "hits_ewm_10")
    add_ewm(df, g, "blocks",        10, "blocks_ewm_10")
    add_ewm(df, g, "turnover_diff", 10, "turnover_diff_ewm_10")

    # --- Period scoring (P3 = late push, OT predictor) ---
    add_ewm(df, g, "p1_gf", 10, "p1_gf_ewm_10")
    add_ewm(df, g, "p3_gf", 10, "p3_gf_ewm_10")
    add_ewm(df, g, "p3_ga", 10, "p3_ga_ewm_10")
    add_ewm(df, g, "p3_goal_diff", 10, "p3_diff_ewm_10")

    # --- Points / win tendency ---
    add_ewm(df, g, "game_points", 20, "points_ewm_20")
    add_ewm(df, g, "reg_goal_diff", 20, "goal_diff_ewm_20")

    # --- Longer-horizon stability features (less reactive, span=40 ~ 28-game half-life) ---
    add_ewm(df, g, "reg_gf",           40, "gf_ewm_40")
    add_ewm(df, g, "reg_ga",           40, "ga_ewm_40")
    add_ewm(df, g, "reg_xgf",          40, "xgf_ewm_40")
    add_ewm(df, g, "reg_xga",          40, "xga_ewm_40")
    add_ewm(df, g, "reg_goal_diff",    40, "goal_diff_ewm_40")
    add_ewm(df, g, "shooting_pct_5v5", 40, "shooting_pct_ewm_40")
    add_ewm(df, g, "save_pct_5v5",     40, "save_pct_ewm_40")
    add_ewm(df, g, "corsi_pct",        40, "corsi_ewm_40")
    add_ewm(df, g, "fenwick_pct",      40, "fenwick_ewm_40")
    add_ewm(df, g, "game_points",      40, "points_ewm_40")

    print("  Computing tail/distribution features...")

    # ===================== TAIL / DISTRIBUTION FEATURES =====================
    # Scoring variance — captures boom/bust tendency
    add_rolling(df, g, "reg_gf", 10, "gf_std_10", "std")
    add_rolling(df, g, "reg_ga", 10, "ga_std_10", "std")

    # Scoring ceiling and floor
    add_rolling(df, g, "reg_gf", 10, "gf_max_10", "max")
    add_rolling(df, g, "reg_gf", 10, "gf_min_10", "min")
    add_rolling(df, g, "reg_ga", 10, "ga_max_10", "max")
    add_rolling(df, g, "reg_ga", 10, "ga_min_10", "min")

    # Scoring range (ceiling - floor)
    df["gf_range_10"] = df["gf_max_10"] - df["gf_min_10"]

    # Rate features — how often extreme outcomes happen
    add_ewm(df, g, "is_high_scoring",  20, "high_scoring_rate_20")
    add_ewm(df, g, "is_shutout",       20, "shutout_rate_20")
    add_ewm(df, g, "is_one_goal_game", 20, "one_goal_rate_20")
    add_ewm(df, g, "is_blowout_win",   20, "blowout_win_rate_20")
    add_ewm(df, g, "is_blowout_loss",  20, "blowout_loss_rate_20")

    # xG variance — how consistent is shot quality generated?
    add_rolling(df, g, "reg_xgf", 10, "xgf_std_10", "std")
    add_rolling(df, g, "reg_xga", 10, "xga_std_10", "std")

    # --- Special teams (PBP-based) ---
    add_ewm(df, g, "pp_goals_pbp",           10, "pp_goals_ewm_10")
    add_ewm(df, g, "pp_conversion_rate_pbp",  20, "pp_conv_rate_ewm_20")
    add_ewm(df, g, "pk_goals_against_pbp",   10, "pk_ga_ewm_10")
    add_ewm(df, g, "pk_save_rate_pbp",       20, "pk_save_rate_ewm_20")

    # --- Score-state tendencies ---
    add_ewm(df, g, "trailing_gf",  20, "trailing_gf_ewm_20")
    add_ewm(df, g, "leading_ga",   20, "leading_ga_ewm_20")
    add_ewm(df, g, "scored_first", 20, "first_goal_rate_ewm_20")
    add_ewm(df, g, "tied_gf",      20, "tied_gf_ewm_20")

    return df


# =============================================================================
# 5. Opponent Features
# =============================================================================

# Columns to mirror for the opponent
OPPONENT_FEATURES = [
    "gf_ewm_10", "gf_ewm_20", "ga_ewm_10", "ga_ewm_20",
    "xgf_ewm_10", "xgf_ewm_20", "xga_ewm_10", "xga_ewm_20",
    "sog_ewm_10",
    "shooting_pct_ewm_20", "save_pct_ewm_20",
    "xg_conversion_ewm_20", "pdo_ewm_20",
    "corsi_ewm_20", "fenwick_ewm_20", "faceoff_ewm_20",
    "corsi_behind_ewm_20",
    "penalties_ewm_10", "pim_ewm_10", "pen_drawn_ewm_10",
    "pp_opps_ewm_10",
    "hits_ewm_10", "blocks_ewm_10", "turnover_diff_ewm_10",
    "p3_gf_ewm_10", "p3_ga_ewm_10",
    "points_ewm_20", "goal_diff_ewm_20",
    "gf_std_10", "ga_std_10",
    "gf_max_10", "gf_min_10",
    "high_scoring_rate_20", "shutout_rate_20", "one_goal_rate_20",
    # Special teams
    "pp_goals_ewm_10", "pp_conv_rate_ewm_20",
    "pk_ga_ewm_10", "pk_save_rate_ewm_20",
    # Score-state
    "trailing_gf_ewm_20", "leading_ga_ewm_20", "first_goal_rate_ewm_20",
    # Travel
    "cumulative_travel_7d",
    # Longer-horizon stability (span=40)
    "gf_ewm_40", "ga_ewm_40", "xgf_ewm_40", "xga_ewm_40",
    "goal_diff_ewm_40", "shooting_pct_ewm_40", "save_pct_ewm_40",
    "corsi_ewm_40", "fenwick_ewm_40", "points_ewm_40",
]

def add_opponent_features(df):
    """Mirror key rolling features for the opponent team."""
    opp_df = df[["game_id", "team"] + OPPONENT_FEATURES].copy()
    opp_rename = {col: f"opp_{col}" for col in OPPONENT_FEATURES}
    opp_rename["team"] = "_opp_team"
    opp_df = opp_df.rename(columns=opp_rename)

    df = df.merge(opp_df, left_on=["game_id", "opponent"],
                  right_on=["game_id", "_opp_team"], how="left")
    df.drop(columns=["_opp_team"], inplace=True)
    return df


# =============================================================================
# 6. Goalie Matchup
# =============================================================================

def add_goalie_features(df, goalie_deploy_file=None):
    """
    Add goalie features for tonight's starters.

    Primary feature: goalie_ga_rate (2-4 range, team's expected GA rate
    given tonight's goalie + deployment context).
    Also includes direct deployment features and interaction features.
    """
    gdeploy = pd.read_csv(goalie_deploy_file)

    # Columns to merge from deployment predictions
    deploy_cols = [
        "goalie_ga_rate",
        "starter_role_share",
        "goalie_switch",
        "consecutive_starts",
        "was_pulled_last_game",
        "is_back_to_back",
    ]

    # One row per team-game (starters only, deduplicate defensively)
    gdeploy_slim = gdeploy[["game_id", "team"] + deploy_cols].copy()
    gdeploy_slim = gdeploy_slim.drop_duplicates(subset=["game_id", "team"])

    # Rename for own team
    own_rename = {col: f"goalie_{col}" if col != "goalie_ga_rate"
                  else "goalie_ga_rate" for col in deploy_cols}
    gdeploy_own = gdeploy_slim.rename(columns=own_rename)

    # Own goalie
    df = df.merge(gdeploy_own, on=["game_id", "team"], how="left")

    # Opponent goalie
    opp_rename = {col: f"opp_goalie_{col}" if col != "goalie_ga_rate"
                  else "opp_goalie_ga_rate" for col in deploy_cols}
    opp_rename["team"] = "_opp_team"
    gdeploy_opp = gdeploy_slim.rename(columns=opp_rename)

    df = df.merge(gdeploy_opp, left_on=["game_id", "opponent"],
                  right_on=["game_id", "_opp_team"], how="left")
    df.drop(columns=["_opp_team"], inplace=True)

    # --- Differentials ---
    df["goalie_ga_rate_diff"] = (
        df["goalie_ga_rate"].fillna(0) -
        df["opp_goalie_ga_rate"].fillna(0)
    )

    # --- Interaction features ---
    # Goalie vulnerability vs opponent attack quality
    df["goalie_rate_x_opp_xgf"] = (
        df["goalie_ga_rate"].fillna(0) * df["opp_xgf_ewm_10"].fillna(0)
    )
    # Opponent goalie vs own attack quality
    df["opp_goalie_rate_x_xgf"] = (
        df["opp_goalie_ga_rate"].fillna(0) * df["xgf_ewm_10"].fillna(0)
    )
    # Emergency start: backup on B2B = worst case
    df["goalie_switch_x_b2b"] = (
        df["goalie_goalie_switch"].fillna(0) *
        df["goalie_is_back_to_back"].fillna(0)
    )
    # Workload interaction: primary starter with long streak
    df["role_share_x_consecutive"] = (
        df["goalie_starter_role_share"].fillna(0) *
        df["goalie_consecutive_starts"].fillna(0)
    )

    return df


# =============================================================================
# 7. Standings Context
# =============================================================================

def add_standings_features(df):
    """
    Add standings-based features via asof merge.
    Uses standings from BEFORE the game date (offset by +1 day to avoid leakage).
    """
    standings = pd.read_csv(STANDINGS_FILE)
    standings["date"] = pd.to_datetime(standings["date"])

    # Offset: standings from date X include X's games, so they're valid from X+1
    standings["effective_date"] = standings["date"] + pd.Timedelta(days=1)

    # Select features
    st_cols = ["effective_date", "team_abbrev",
               "point_pct", "goals_for", "goals_against", "games_played",
               "home_wins", "home_losses", "home_ot_losses",
               "road_wins", "road_losses", "road_ot_losses",
               "l10_wins", "l10_losses", "l10_ot_losses",
               "streak_code", "streak_count",
               "reg_wins", "wildcard_sequence"]
    standings = standings[st_cols].copy()

    # Derived
    standings["goal_diff_per_gp"] = np.where(
        standings["games_played"] > 0,
        (standings["goals_for"] - standings["goals_against"]) / standings["games_played"],
        0
    )
    home_gp = standings["home_wins"] + standings["home_losses"] + standings["home_ot_losses"]
    standings["home_win_pct"] = np.where(home_gp > 0, standings["home_wins"] / home_gp, 0.5)
    road_gp = standings["road_wins"] + standings["road_losses"] + standings["road_ot_losses"]
    standings["road_win_pct"] = np.where(road_gp > 0, standings["road_wins"] / road_gp, 0.5)
    l10_gp = standings["l10_wins"] + standings["l10_losses"] + standings["l10_ot_losses"]
    standings["l10_pct"] = np.where(l10_gp > 0, standings["l10_wins"] / l10_gp, 0.5)
    standings["streak_value"] = np.where(
        standings["streak_code"] == "W", standings["streak_count"],
        -standings["streak_count"]
    )
    standings["reg_win_pct"] = np.where(
        standings["games_played"] > 0,
        standings["reg_wins"] / standings["games_played"],
        0.5
    )
    standings["wildcard_sequence"] = standings["wildcard_sequence"].fillna(0)

    keep_cols = ["effective_date", "team_abbrev",
                 "point_pct", "goal_diff_per_gp", "games_played",
                 "home_win_pct", "road_win_pct", "l10_pct", "streak_value",
                 "reg_win_pct", "wildcard_sequence"]
    standings = standings[keep_cols].sort_values("effective_date")

    # Asof merge — for each team-game, get most recent standings BEFORE game
    df = df.sort_values("game_date")
    df = pd.merge_asof(
        df, standings.rename(columns={"team_abbrev": "team",
                                       "effective_date": "game_date"}),
        on="game_date", by="team", direction="backward",
        suffixes=("", "_standings")
    )

    # Opponent standings
    opp_standings = standings.rename(columns={
        "team_abbrev": "opponent",
        "point_pct": "opp_point_pct",
        "goal_diff_per_gp": "opp_goal_diff_per_gp",
        "l10_pct": "opp_l10_pct",
        "reg_win_pct": "opp_reg_win_pct",
        "wildcard_sequence": "opp_wildcard_sequence",
        "effective_date": "game_date",
    })
    opp_keep = ["game_date", "opponent", "opp_point_pct",
                "opp_goal_diff_per_gp", "opp_l10_pct",
                "opp_reg_win_pct", "opp_wildcard_sequence"]
    df = pd.merge_asof(
        df, opp_standings[opp_keep].sort_values("game_date"),
        on="game_date", by="opponent", direction="backward"
    )

    # Cross features
    df["point_pct_diff"] = df["point_pct"].fillna(0.5) - df["opp_point_pct"].fillna(0.5)

    # Location-adjusted win rate
    df["location_win_pct"] = np.where(
        df["is_home"] == 1, df["home_win_pct"], df["road_win_pct"]
    )

    # Regulation win pct differential
    df["reg_win_pct_diff"] = df["reg_win_pct"].fillna(0.5) - df["opp_reg_win_pct"].fillna(0.5)
    # Playoff position (lower wildcard_sequence = better; 0 = unset)
    df["playoff_position_diff"] = (
        df["opp_wildcard_sequence"].fillna(0) - df["wildcard_sequence"].fillna(0)
    )

    return df


# =============================================================================
# 8. Travel & Timezone Features
# =============================================================================

# Venue → (latitude, longitude, UTC_offset_hours)
VENUE_COORDS = {
    # --- Active primary arenas ---
    "Amalie Arena": (27.94, -82.45, -5),
    "Amerant Bank Arena": (26.16, -80.33, -5),
    "American Airlines Center": (32.79, -96.81, -6),
    "Ball Arena": (39.75, -105.01, -7),
    "Bridgestone Arena": (36.16, -86.78, -6),
    "Canada Life Centre": (49.89, -97.14, -6),
    "Canadian Tire Centre": (45.30, -75.93, -5),
    "Capital One Arena": (38.90, -77.02, -5),
    "Centre Bell": (45.50, -73.57, -5),
    "Climate Pledge Arena": (47.62, -122.35, -8),
    "Crypto.com Arena": (34.04, -118.27, -8),
    "Delta Center": (40.77, -111.90, -7),
    "Enterprise Center": (38.63, -90.20, -6),
    "Honda Center": (33.81, -117.88, -8),
    "KeyBank Center": (42.88, -78.88, -5),
    "Lenovo Center": (35.80, -78.72, -5),
    "Little Caesars Arena": (42.34, -83.06, -5),
    "Madison Square Garden": (40.75, -73.99, -5),
    "Nationwide Arena": (39.97, -83.01, -5),
    "PPG Paints Arena": (40.44, -79.99, -5),
    "Prudential Center": (40.73, -74.17, -5),
    "Rogers Arena": (49.28, -123.11, -8),
    "Rogers Place": (53.55, -113.50, -7),
    "SAP Center at San Jose": (37.33, -121.90, -8),
    "Scotiabank Arena": (43.64, -79.38, -5),
    "Scotiabank Saddledome": (51.04, -114.05, -7),
    "T-Mobile Arena": (36.10, -115.18, -8),
    "TD Garden": (42.37, -71.06, -5),
    "UBS Arena": (40.72, -73.73, -5),
    "United Center": (41.88, -87.67, -6),
    "Wells Fargo Center": (39.90, -75.17, -5),
    "Xcel Energy Center": (44.94, -93.10, -6),
    # --- Historical / renamed arenas ---
    "BB&T Center": (26.16, -80.33, -5),
    "Bell MTS Place": (49.89, -97.14, -6),
    "FLA Live Arena": (26.16, -80.33, -5),
    "Gila River Arena": (33.53, -112.23, -7),
    "Mullett Arena": (33.43, -111.93, -7),
    "Nassau Veterans Memorial Coliseum": (40.72, -73.59, -5),
    "PNC Arena": (35.80, -78.72, -5),
    "STAPLES Center": (34.04, -118.27, -8),
    # --- Renamed arenas (same physical location) ---
    "Benchmark International Arena": (33.43, -111.93, -7),
    "Grand Casino Arena": (33.43, -111.93, -7),
    "Xfinity Mobile Arena": (39.90, -75.17, -5),
    # --- International ---
    "Avicii Arena": (59.29, 18.08, 1),
    "Nokia Arena": (61.49, 23.78, 2),
    "O2 Czech Republic": (50.10, 14.47, 1),
    # --- Outdoor / neutral site ---
    "Carter-Finley Stadium": (35.79, -78.72, -5),
    "Commonwealth Stadium, Edmonton": (53.56, -113.52, -7),
    "Edgewood Tahoe Resort": (38.95, -119.98, -8),
    "Fenway Park": (42.35, -71.10, -5),
    "loanDepot park": (25.78, -80.22, -5),
    "MetLife Stadium": (40.81, -74.07, -5),
    "Nissan Stadium": (36.17, -86.77, -6),
    "Ohio Stadium": (40.00, -83.02, -5),
    "Raymond James Stadium": (27.98, -82.50, -5),
    "T-Mobile Park": (47.59, -122.33, -8),
    "Target Field": (44.98, -93.28, -6),
    "Tim Hortons Field": (43.25, -79.83, -5),
    "Wrigley Field": (41.95, -87.66, -6),
}


def _haversine_km(lat1, lon1, lat2, lon2):
    """Haversine distance in km between two (lat, lon) points."""
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def add_travel_features(df, games):
    """
    Add per-team travel/timezone features based on venue coordinates.
    Requires df sorted by (team, game_date).
    """
    # Join venue from games
    venue_map = games[["game_id", "venue"]].drop_duplicates(subset=["game_id"])
    df = df.merge(venue_map, on="game_id", how="left")

    # Map venue to coordinates
    df["_lat"] = df["venue"].map(lambda v: VENUE_COORDS.get(v, (None, None, None))[0])
    df["_lon"] = df["venue"].map(lambda v: VENUE_COORDS.get(v, (None, None, None))[1])
    df["_tz"] = df["venue"].map(lambda v: VENUE_COORDS.get(v, (None, None, None))[2])

    # Per-team: distance from previous game venue
    df["_prev_lat"] = df.groupby("team")["_lat"].shift(1)
    df["_prev_lon"] = df.groupby("team")["_lon"].shift(1)
    df["_prev_tz"] = df.groupby("team")["_tz"].shift(1)

    # Travel distance (haversine km)
    valid = df["_lat"].notna() & df["_prev_lat"].notna()
    df["travel_distance"] = 0.0
    df.loc[valid, "travel_distance"] = _haversine_km(
        df.loc[valid, "_prev_lat"].values, df.loc[valid, "_prev_lon"].values,
        df.loc[valid, "_lat"].values, df.loc[valid, "_lon"].values,
    )
    df["travel_distance"] = df["travel_distance"].clip(upper=5000)

    # Timezone change
    df["tz_change"] = 0.0
    tz_valid = df["_tz"].notna() & df["_prev_tz"].notna()
    df.loc[tz_valid, "tz_change"] = (df.loc[tz_valid, "_tz"] - df.loc[tz_valid, "_prev_tz"]).abs()

    # Road trip game number (consecutive away games)
    road_trip_nums = []
    for _, group in df.groupby("team"):
        count = 0
        for h in group["is_home"].values:
            if h == 1:
                count = 0
            else:
                count += 1
            road_trip_nums.append(count)
    df["road_trip_game_num"] = road_trip_nums
    df["road_trip_game_num"] = df["road_trip_game_num"].clip(upper=6)
    df["is_road_trip"] = (df["road_trip_game_num"] >= 2).astype(float)

    # Cumulative travel in last 7 calendar days (per team)
    def _cum_travel_7d(group):
        dates = group["game_date"].values
        dists = group["travel_distance"].values
        result = np.zeros(len(dates))
        for i in range(len(dates)):
            cutoff = dates[i] - np.timedelta64(7, "D")
            mask = (dates[:i+1] >= cutoff) & (dates[:i+1] <= dates[i])
            result[i] = np.nansum(dists[:i+1][mask])
        return pd.Series(result, index=group.index)

    df["cumulative_travel_7d"] = df.groupby("team", group_keys=False).apply(_cum_travel_7d)

    # Clean up temp columns
    df.drop(columns=["_lat", "_lon", "_tz", "_prev_lat", "_prev_lon", "_prev_tz", "venue"],
            inplace=True)

    return df


# =============================================================================
# 9. Rest / Context
# =============================================================================

def add_rest_features(df):
    """Compute rest days, back-to-back flags, and schedule fatigue."""
    # Days since last game for each team
    df["prev_game_date"] = df.groupby("team")["game_date"].shift(1)
    df["days_rest"] = (df["game_date"] - df["prev_game_date"]).dt.days
    df["is_back_to_back"] = (df["days_rest"] <= 1).astype(float)
    df["days_rest"] = df["days_rest"].clip(upper=14).fillna(5.0)
    df["is_back_to_back"] = df["is_back_to_back"].fillna(0)
    df.drop(columns=["prev_game_date"], inplace=True)

    # Games in last 7 calendar days (per team, excludes current game)
    def _count_recent_games(dates, days):
        arr = dates.values
        counts = []
        for i in range(len(arr)):
            cutoff = arr[i] - np.timedelta64(days, "D")
            counts.append(int(((arr[:i] >= cutoff) & (arr[:i] < arr[i])).sum()))
        return pd.Series(counts, index=dates.index)

    df["games_last_7d"] = df.groupby("team", group_keys=False)["game_date"].transform(
        lambda x: _count_recent_games(x, 7)
    )
    # 3-in-4: at least 2 games in last 4 days before this one
    df["is_3_in_4"] = df.groupby("team", group_keys=False)["game_date"].transform(
        lambda x: _count_recent_games(x, 4)
    )
    df["is_3_in_4"] = (df["is_3_in_4"] >= 2).astype(float)

    # Opponent rest and fatigue (join from opponent's perspective)
    rest_df = df[["game_id", "team", "days_rest", "is_back_to_back",
                  "games_last_7d", "is_3_in_4"]].copy()
    rest_df.rename(columns={
        "team": "_opp",
        "days_rest": "opp_days_rest",
        "is_back_to_back": "opp_is_b2b",
        "games_last_7d": "opp_games_last_7d",
        "is_3_in_4": "opp_is_3_in_4",
    }, inplace=True)
    df = df.merge(rest_df, left_on=["game_id", "opponent"],
                  right_on=["game_id", "_opp"], how="left")
    df.drop(columns=["_opp"], inplace=True)

    df["rest_advantage"] = df["days_rest"] - df["opp_days_rest"].fillna(3.0)
    df["fatigue_advantage"] = df["opp_games_last_7d"].fillna(0) - df["games_last_7d"]

    # Season game number (proxy for sample size / fatigue)
    df["season_game_num"] = df.groupby(["team", "season"]).cumcount()

    return df


# =============================================================================
# Main Pipeline
# =============================================================================

def build_regulation_features(cutoff_year="2025"):
    shot_xg_file = os.path.join(PROCESSED_DIR, f"shot_xg_{cutoff_year}.csv")
    goalie_deploy_file = os.path.join(PROCESSED_DIR, f"goalie_deployment_predictions_{cutoff_year}.csv")
    output_file = os.path.join(PROCESSED_DIR, f"regulation_features_{cutoff_year}.csv")

    print(f"\n{'#'*60}")
    print(f"  Cutoff year:  {cutoff_year}")
    print(f"  Shot xG file: {shot_xg_file}")
    print(f"  Goalie file:  {goalie_deploy_file}")
    print(f"  Output file:  {output_file}")
    print(f"{'#'*60}")

    print("Loading data...")
    games = pd.read_csv(GAME_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    print(f"  Games: {len(games):,}")

    shots = pd.read_csv(shot_xg_file)
    shots["game_date"] = pd.to_datetime(shots["game_date"])
    print(f"  Shots: {len(shots):,}")

    pbp = pd.read_csv(PBP_FILE)
    pbp["game_date"] = pd.to_datetime(pbp["game_date"])
    print(f"  PBP events: {len(pbp):,}")

    # --- Step 1: Base table ---
    print("\nBuilding base table...")
    base = build_base_table(games)
    print(f"  Team-game rows: {len(base):,}")

    # --- Step 2: Shot-based regulation stats ---
    print("Computing regulation stats from shot data...")
    shot_stats = compute_shot_game_stats(shots)
    n_before = len(base)
    base = base.merge(shot_stats, on=["game_id", "is_home"], how="left")
    assert len(base) == n_before, f"Shot merge row explosion: {n_before} → {len(base)}"
    # Fill any games with no shot data
    for col in ["reg_gf", "reg_ga", "reg_xgf", "reg_xga",
                "reg_shots_for", "reg_shots_against", "reg_sog_for"]:
        base[col] = base[col].fillna(0)

    # --- Step 2b: Special teams stats (PBP-based) ---
    print("Computing special teams stats from PBP...")
    st_stats = compute_special_teams_game_stats(pbp, games)
    n_before = len(base)
    base = base.merge(st_stats, on=["game_id", "is_home"], how="left")
    assert len(base) == n_before, f"ST merge row explosion: {n_before} → {len(base)}"
    for col in ["pp_goals_pbp", "pp_shots_pbp", "pk_goals_against_pbp",
                "pk_shots_against_pbp", "pp_conversion_rate_pbp", "pk_save_rate_pbp"]:
        base[col] = base[col].fillna(0)

    # --- Step 2c: Score-state stats (PBP-based) ---
    print("Computing score-state stats from PBP...")
    ss_stats = compute_score_state_game_stats(pbp, games)
    n_before = len(base)
    base = base.merge(ss_stats, on=["game_id", "is_home"], how="left")
    assert len(base) == n_before, f"Score-state merge row explosion: {n_before} → {len(base)}"
    for col in ["trailing_gf", "leading_ga", "scored_first", "tied_gf"]:
        base[col] = base[col].fillna(0)

    # --- Step 3: REST API team stats ---
    print("Loading REST API team stats...")
    team_stats = load_team_game_stats()
    print(f"  REST API rows: {len(team_stats):,}")
    n_before = len(base)
    base = base.merge(team_stats, on=["game_id", "team_id"], how="left")
    assert len(base) == n_before, f"REST API merge row explosion: {n_before} → {len(base)}"

    # --- Step 4: Sort by team + date for rolling ---
    base = base.sort_values(["team", "game_date", "game_id"]).reset_index(drop=True)

    # --- Step 4b: Travel features ---
    print("Computing travel/timezone features...")
    n_before = len(base)
    base = add_travel_features(base, games)
    assert len(base) == n_before, f"Travel merge row explosion: {n_before} → {len(base)}"

    # --- Step 5: Rolling features ---
    print("\nComputing rolling features...")
    base = compute_all_rolling(base)

    # --- Step 6: Cross features (team vs opponent matchup) ---
    print("Computing cross features...")
    base["xg_matchup"] = base["xgf_ewm_20"] - base["xga_ewm_20"]

    # --- Step 7: Opponent features ---
    print("Adding opponent features...")
    n_before = len(base)
    base = add_opponent_features(base)
    assert len(base) == n_before, f"Opponent merge row explosion: {n_before} → {len(base)}"

    # Matchup cross features (after opponent join)
    base["scoring_matchup"] = base["gf_ewm_20"] - base["opp_ga_ewm_20"]
    base["defense_matchup"] = base["ga_ewm_20"] - base["opp_gf_ewm_20"]
    base["pp_vs_pk"] = base["pp_opps_ewm_10"] - base["opp_pen_drawn_ewm_10"]

    # --- Group 1: PP/PK matchup interactions ---
    base["pp_pk_matchup"] = (
        base["pp_conv_rate_ewm_20"] * (1 - base["opp_pk_save_rate_ewm_20"].fillna(0.85))
    )
    base["opp_pp_pk_matchup"] = (
        base["opp_pp_conv_rate_ewm_20"] * (1 - base["pk_save_rate_ewm_20"].fillna(0.85))
    )

    # --- Group 2: Matchup interaction features ---
    base["xg_offense_x_defense"] = base["xgf_ewm_10"] * base["opp_xga_ewm_10"]
    base["defense_x_opp_offense"] = base["ga_ewm_10"] * base["opp_gf_ewm_10"]
    base["efficiency_matchup"] = (
        base["shooting_pct_ewm_20"] * (1 - base["opp_save_pct_ewm_20"].fillna(0.91))
    )
    base["gf_vs_opp_ga_ratio"] = np.where(
        base["opp_ga_ewm_10"] > 0.5, base["gf_ewm_10"] / base["opp_ga_ewm_10"], 1.0
    )
    base["xg_net_matchup"] = (
        (base["xgf_ewm_10"] - base["xga_ewm_10"]) -
        (base["opp_xgf_ewm_10"] - base["opp_xga_ewm_10"])
    )
    base["corsi_matchup_x"] = base["corsi_ewm_20"] * (100 - base["opp_corsi_ewm_20"].fillna(50))

    # --- Travel advantage (after opponent mirror) ---
    base["travel_advantage"] = (
        base["opp_cumulative_travel_7d"].fillna(0) - base["cumulative_travel_7d"].fillna(0)
    )

    # --- Step 8: Goalie matchup ---
    print("Adding goalie features...")
    n_before = len(base)
    base = add_goalie_features(base, goalie_deploy_file=goalie_deploy_file)
    assert len(base) == n_before, f"Goalie merge row explosion: {n_before} → {len(base)}"

    # --- Step 9: Standings ---
    print("Adding standings features...")
    n_before = len(base)
    base = add_standings_features(base)
    assert len(base) == n_before, f"Standings merge row explosion: {n_before} → {len(base)}"

    # --- Step 10: Rest / context ---
    print("Adding rest/context features...")
    n_before = len(base)
    base = add_rest_features(base)
    assert len(base) == n_before, f"Rest merge row explosion: {n_before} → {len(base)}"

    # --- Identify feature columns ---
    id_cols = ["game_id", "game_date", "season", "team", "opponent",
               "team_id", "is_home", "period_count", "game_outcome_type"]
    target_col = "reg_gf"

    # Raw per-game stats (not features — used to build rolling)
    raw_cols = ["reg_ga", "reg_xgf", "reg_xga", "reg_shots_for",
                "reg_shots_against", "reg_sog_for",
                "p1_gf", "p1_ga", "p2_gf", "p2_ga", "p3_gf", "p3_ga",
                "total_gf", "total_ga", "sog_for_api", "sog_against_api",
                "corsi_pct", "fenwick_pct", "shooting_pct_5v5", "save_pct_5v5",
                "penalties", "pim", "penalties_drawn", "pp_opportunities",
                "hits", "blocks", "giveaways", "takeaways",
                "faceoff_pct", "game_win", "game_loss", "game_otl", "game_points",
                "turnover_diff", "pdo_5v5", "xg_conversion", "reg_goal_diff",
                "p3_goal_diff", "major_penalties", "total_faceoffs", "missed_shots",
                "corsi_ahead", "corsi_behind", "corsi_close",
                "is_high_scoring", "is_shutout", "is_one_goal_game",
                "is_blowout_win", "is_blowout_loss",
                # PBP-based raw stats
                "pp_goals_pbp", "pp_shots_pbp", "pk_goals_against_pbp",
                "pk_shots_against_pbp", "pp_conversion_rate_pbp", "pk_save_rate_pbp",
                "trailing_gf", "leading_ga", "scored_first", "tied_gf"]

    # All other columns are features
    feature_cols = [c for c in base.columns
                    if c not in id_cols + [target_col] + raw_cols
                    and c != "games_played"]  # standings artifact

    print(f"\n--- Feature Summary ---")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  ID columns: {len(id_cols)}")
    print(f"  Target: {target_col}")

    # --- Save ---
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    base.to_csv(output_file, index=False)
    print(f"\nSaved {len(base):,} rows to {output_file}")

    # --- Sanity checks ---
    print(f"\n--- Sanity Checks ---")
    print(f"  Mean reg goals: {base['reg_gf'].mean():.2f} (expect ~2.8-3.0)")
    print(f"  Rows per game: {base.groupby('game_id').size().mean():.1f} (expect 2.0)")
    print(f"  OT games: {(base['game_outcome_type'] != 'REG').sum() // 2:,}")

    # Feature coverage
    print(f"\n  Feature coverage (% non-null):")
    low_coverage = []
    for col in sorted(feature_cols):
        pct = base[col].notna().mean() * 100
        if pct < 80:
            low_coverage.append((col, pct))
    if low_coverage:
        for col, pct in low_coverage:
            print(f"    {col:35s}: {pct:.1f}%")
    else:
        print(f"    All features > 80% coverage")

    # Feature correlation with target
    print(f"\n  Top feature correlations with reg_gf:")
    corrs = base[feature_cols + [target_col]].corr()[target_col].drop(target_col)
    top_corr = corrs.abs().sort_values(ascending=False).head(15)
    for feat in top_corr.index:
        print(f"    {feat:35s}: {corrs[feat]:+.3f}")

    return base


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    for year in CUTOFF_YEARS:
        print("=" * 60)
        print(f"Regulation Feature Engineering — cutoff year {year}")
        print("=" * 60)

        build_regulation_features(cutoff_year=year)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
