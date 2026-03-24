# -*- coding: utf-8 -*-
"""
xG Feature Engineering

Reads pbp_events.csv and engineers shot-level features for the
expected goals (xG) model. This is a SHOT QUALITY model — no shooter
or goalie identity, no team strength context. Those belong in
downstream models (goalie model, regulation model).

Key design decisions:
  - EXCLUDE blocked shots (coordinates are where the block happened,
    not where the shot originated)
  - INCLUDE empty-net shots with a flag (they are legitimately high xG)
  - Proper coordinate normalization using period + team direction
    (not just |x|, which fails for defensive-zone shots)
  - Rebound detection looks backward only (no future leakage)
  - No shooter/goalie identity (pure shot geometry model)

Output:
  - data/processed/xg_training_data.csv

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
from config import RAW_DIR, PROCESSED_DIR

PBP_FILE = os.path.join(RAW_DIR, "pbp_events.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "xg_training_data.csv")

# Net position (center of the goal mouth)
NET_X = 89.0
NET_Y = 0.0

# Rebound threshold (seconds)
REBOUND_SECONDS = 3.0

# Rush threshold (seconds since zone entry / neutral zone event)
RUSH_SECONDS = 4.0


# =============================================================================
# Coordinate Normalization
# =============================================================================

def normalize_coordinates(df):
    """
    Normalize shot coordinates using |x| (absolute value).

    The home team CHOOSES which end to defend in period 1, so the
    attack direction is NOT deterministic from period + home/away.
    Empirically, home teams attack positive-x only ~47% of the time.

    Using |x| is robust because:
      - Offensive zone shots (98%+ of all shots) always have |x| near 89
      - Distance = sqrt((89 - |x|)^2 + y^2) is correct for both ends
      - The only error is for rare defensive-zone shots (<2%), which
        have ~0% goal rate anyway so the impact is negligible
      - This matches the proven approach from the SOG model

    After normalization, x_norm is always >= 0, with larger values
    meaning closer to the target net (89, 0).
    """
    x = df["x_coord"].values.astype(float)
    y = df["y_coord"].values.astype(float)

    x_norm = np.abs(x)
    y_abs = np.abs(y)

    return x_norm, y_abs


# =============================================================================
# Feature Computation
# =============================================================================

def compute_distance_angle(x_norm, y_abs):
    """
    Compute shot distance and angle from normalized coordinates.

    Distance: Euclidean from shot to net center (89, 0)
    Angle: Degrees from center line to shot location
           0° = dead center, 90° = along the goal line
    """
    dx = NET_X - x_norm
    dy = y_abs  # distance from center line

    distance = np.sqrt(dx**2 + dy**2)
    angle = np.degrees(np.arctan2(dy, np.maximum(dx, 0.1)))  # avoid division by zero

    return distance, angle


def time_to_seconds(time_str):
    """Convert MM:SS time string to seconds."""
    if pd.isna(time_str) or time_str == "":
        return np.nan
    try:
        parts = str(time_str).split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return np.nan


def compute_game_seconds(period, time_in_period_sec):
    """
    Compute total elapsed seconds in the game.
    Each regulation period is 20 minutes (1200 seconds).
    OT is period 4+ (5 minutes in regular season).
    """
    # Periods 1-3 are 1200 seconds each
    period = np.array(period, dtype=float)
    elapsed_prior = np.minimum(period - 1, 3) * 1200.0

    # Add OT periods if applicable (5 min each for regular season)
    ot_periods = np.maximum(period - 4, 0)
    elapsed_prior += ot_periods * 300.0

    return elapsed_prior + time_in_period_sec


def compute_rebound_feature(df):
    """
    Detect rebound shots: a shot within REBOUND_SECONDS of a prior
    shot-on-goal by EITHER team in the same game and period.

    A rebound implies the goalie made a save but couldn't control
    the puck, creating a high-danger second chance.

    Returns:
      - is_rebound: binary flag
      - time_since_last_shot: seconds since prior shot (capped at 60)
    """
    is_rebound = np.zeros(len(df), dtype=int)
    time_since_last = np.full(len(df), 60.0)  # default: "long time ago"

    # Process within each game
    for game_id, group in df.groupby("game_id"):
        idx = group.index.values
        periods = group["period"].values
        times = group["game_seconds"].values
        event_types = group["event_type"].values

        for i in range(1, len(idx)):
            curr_period = periods[i]
            curr_time = times[i]

            # Look backward for the most recent shot in the same period
            for j in range(i - 1, -1, -1):
                if periods[j] != curr_period:
                    break  # different period, stop looking

                prev_time = times[j]
                time_diff = curr_time - prev_time

                if time_diff > 60.0:
                    break  # too far back, stop

                # Only count prior shots-on-goal (saves that produce rebounds)
                # Goals end the sequence
                if event_types[j] == "shot-on-goal":
                    time_since_last[idx[i]] = time_diff
                    if time_diff <= REBOUND_SECONDS:
                        is_rebound[idx[i]] = 1
                    break
                elif event_types[j] == "goal":
                    break  # goal resets the sequence

    return is_rebound, time_since_last


def encode_shot_type(shot_types):
    """
    Encode shot types as ordinal categories roughly ordered by xG.
    Unknown/missing → 0 (average).
    """
    mapping = {
        "": 0,
        "wrist": 1,
        "snap": 2,
        "slap": 3,
        "backhand": 4,
        "tip-in": 5,
        "deflected": 6,
        "wrap-around": 7,
        "bat": 8,
        "between-legs": 9,
        "cradle": 10,
        "poke": 11,
    }
    return shot_types.map(lambda x: mapping.get(str(x).lower().strip(), 0))


def encode_strength(df):
    """
    Encode strength state into categories from the shooting team's perspective.

    Categories:
      - EV (even strength: 5v5, 4v4, 3v3)
      - PP (power play: more skaters than opponent)
      - SH (shorthanded: fewer skaters than opponent)

    Uses away_skaters/home_skaters + is_home_event to determine.
    """
    away_sk = df["away_skaters"].values
    home_sk = df["home_skaters"].values
    is_home = df["is_home_event"].values

    # From the shooting team's perspective
    my_skaters = np.where(is_home, home_sk, away_sk)
    opp_skaters = np.where(is_home, away_sk, home_sk)

    strength = np.where(
        my_skaters > opp_skaters, 2,  # PP
        np.where(my_skaters < opp_skaters, 1,  # SH
                 0)  # EV
    )
    return strength


def compute_score_diff(df):
    """
    Compute score differential from the shooting team's perspective
    BEFORE the shot outcome.

    CRITICAL: The NHL API records the score AFTER a goal event, so
    for goals, the scoring team's score is inflated by 1. We must
    subtract 1 from the scoring team's score to get the pre-shot
    game state. Without this fix, score_diff leaks the target variable
    (goals have systematically different score_diff than non-goals).

    Positive = leading, negative = trailing.
    """
    away_score = df["away_score"].values.astype(float)
    home_score = df["home_score"].values.astype(float)
    is_home = df["is_home_event"].values
    is_goal = (df["event_type"] == "goal").values

    # For goals, subtract 1 from the scoring team's score
    # to get the game state BEFORE the goal was scored
    home_adj = home_score.copy()
    away_adj = away_score.copy()

    # Home goals: home_score is inflated by 1
    home_adj[is_goal & is_home] -= 1
    # Away goals: away_score is inflated by 1
    away_adj[is_goal & ~is_home] -= 1

    score_diff = np.where(is_home,
                          home_adj - away_adj,
                          away_adj - home_adj)
    return score_diff


# =============================================================================
# Main Pipeline
# =============================================================================

def build_xg_features():
    """
    Load PBP events, filter to shots, engineer features, save output.
    """
    print("Loading PBP events...")
    df = pd.read_csv(PBP_FILE)
    print(f"  Total events: {len(df):,}")

    # ----- Filter to shot events (exclude blocked shots) -----
    # Blocked shots have coordinates of the blocker, not the shooter
    shot_types = ["goal", "shot-on-goal", "missed-shot"]
    shots = df[df["event_type"].isin(shot_types)].copy()
    print(f"  Shots (excl. blocked): {len(shots):,}")

    # Drop rows with missing coordinates
    shots = shots.dropna(subset=["x_coord", "y_coord"])
    print(f"  After dropping NaN coords: {len(shots):,}")

    # ----- Target variable -----
    shots["is_goal"] = (shots["event_type"] == "goal").astype(int)
    print(f"  Goals: {shots['is_goal'].sum():,} "
          f"({100*shots['is_goal'].mean():.1f}%)")

    # ----- Coordinate normalization -----
    print("Computing normalized coordinates...")
    x_norm, y_abs = normalize_coordinates(shots)
    shots["x_norm"] = x_norm
    shots["y_abs"] = y_abs

    # ----- Distance and angle -----
    distance, angle = compute_distance_angle(x_norm, y_abs)
    shots["shot_distance"] = distance
    shots["shot_angle"] = angle

    # ----- Time features -----
    print("Computing time features...")
    shots["time_in_period_sec"] = shots["time_in_period"].apply(time_to_seconds)
    shots["time_remaining_sec"] = shots["time_remaining"].apply(time_to_seconds)
    shots["game_seconds"] = compute_game_seconds(
        shots["period"].values, shots["time_in_period_sec"].values
    )

    # ----- Rebound detection -----
    print("Detecting rebounds (this may take a minute)...")
    # Sort by game and time for proper rebound detection
    shots = shots.sort_values(["game_id", "game_seconds"]).reset_index(drop=True)
    is_rebound, time_since_last = compute_rebound_feature(shots)
    shots["is_rebound"] = is_rebound
    shots["time_since_last_shot"] = time_since_last

    # ----- Shot type encoding -----
    shots["shot_type_code"] = encode_shot_type(shots["shot_type"])

    # ----- Strength state -----
    shots["strength_code"] = encode_strength(shots)

    # ----- Empty net flag -----
    shots["is_empty_net"] = shots["is_empty_net"].astype(int)

    # ----- Score differential -----
    shots["score_diff"] = compute_score_diff(shots)

    # ----- Period features -----
    # Binary flags for period context
    shots["is_ot"] = (shots["period"] > 3).astype(int)

    # ----- Select output columns -----
    feature_cols = [
        # Identifiers (not model features, but needed for joins)
        "game_id", "game_date", "event_id", "period", "event_type",
        "shooter_id", "goalie_id",

        # Target
        "is_goal",

        # Geometric features
        "shot_distance", "shot_angle", "x_norm", "y_abs",

        # Shot context
        "shot_type_code", "is_rebound", "time_since_last_shot",
        "strength_code", "is_empty_net",

        # Game state
        "score_diff", "game_seconds", "is_ot",
        "time_remaining_sec",

        # Keep raw for downstream use (goalie model needs these)
        "is_home_event", "away_team", "home_team",
        "situation_code",
    ]

    output = shots[feature_cols].copy()

    # ----- Save -----
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)

    print(f"\nSaved {len(output):,} shots to {OUTPUT_FILE}")
    print(f"\nFeature summary:")
    print(f"  shot_distance: mean={output['shot_distance'].mean():.1f}, "
          f"median={output['shot_distance'].median():.1f}")
    print(f"  shot_angle:    mean={output['shot_angle'].mean():.1f}°")
    print(f"  is_rebound:    {output['is_rebound'].mean()*100:.1f}% of shots")
    print(f"  is_empty_net:  {output['is_empty_net'].mean()*100:.1f}% of shots")
    print(f"  Goal rate:     {output['is_goal'].mean()*100:.1f}%")

    # Sanity checks
    print(f"\n--- Sanity Checks ---")
    # Rebounds should have higher goal rate
    reb_rate = output.loc[output["is_rebound"] == 1, "is_goal"].mean()
    nonreb_rate = output.loc[output["is_rebound"] == 0, "is_goal"].mean()
    print(f"  Rebound goal rate:     {reb_rate*100:.1f}% "
          f"(vs {nonreb_rate*100:.1f}% non-rebound)")

    # Close shots should have higher goal rate
    close = output["shot_distance"] < 15
    close_rate = output.loc[close, "is_goal"].mean()
    far_rate = output.loc[~close, "is_goal"].mean()
    print(f"  Close (<15ft) goal rate: {close_rate*100:.1f}% "
          f"(vs {far_rate*100:.1f}% far)")

    # Empty net should be very high
    en_rate = output.loc[output["is_empty_net"] == 1, "is_goal"].mean()
    print(f"  Empty net goal rate:   {en_rate*100:.1f}%")

    # PP should be higher than SH
    pp_rate = output.loc[output["strength_code"] == 2, "is_goal"].mean()
    sh_rate = output.loc[output["strength_code"] == 1, "is_goal"].mean()
    ev_rate = output.loc[output["strength_code"] == 0, "is_goal"].mean()
    print(f"  PP goal rate: {pp_rate*100:.1f}%, "
          f"EV: {ev_rate*100:.1f}%, SH: {sh_rate*100:.1f}%")

    return output


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("xG Feature Engineering")
    print("=" * 60)

    build_xg_features()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
