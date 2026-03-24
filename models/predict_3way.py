# -*- coding: utf-8 -*-
"""
Shared 3-Way Prediction Logic

Extracted from walk_forward_3way.py so that both the backtest and live
predict_today.py use identical prediction math. Any change here is
automatically reflected in both pipelines.

Functions:
  - compute_calibration_params: rho, tie_inflation, home_ice_shift from history
  - predict_game_3way: single-game 3-way probabilities
  - power_devig_3way: remove vig from 3-way odds
  - calculate_edges: model probs vs market probs
  - format_recommendations: pretty-print bet recommendations

@author: chazf
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd

from models.poisson_combiner import (
    quantiles_to_lambda,
    compute_game_probabilities,
    apply_tie_calibration,
    compute_scoring_anchor,
    estimate_poisson_rho,
    QUANTILE_COLS,
    LAMBDA_CLIP,
    DEFAULT_P_HOME_OT,
    ANCHOR_LOOKBACK_DAYS,
)

# Calibration defaults
OT_BLEND_WEIGHT = 0.0

# Blend staking parameters
DRAW_MIN_EDGE = 0.025
HOME_MIN_EDGE = 0.025
HOME_MAX_EDGE = 0.05
DRAW_STAKE_MULT = 25
DRAW_STAKE_CAP = 1.5
HOME_STAKE_MULT = 20
HOME_STAKE_CAP = 1.0

# Bayesian prior strength cap: limits how much historical data anchors
# the posterior, so the current season's OT rate has meaningful influence.
# ~1500 = ~2 full NHL seasons of effective prior weight.
MAX_PRIOR_GAMES = 1500


# =============================================================================
# 3-Way Power Devig
# =============================================================================

def power_devig_3way(p1, p2, p3):
    """Remove vig from 3-way market using power method.

    Finds k such that p1^k + p2^k + p3^k = 1.
    Returns (fair_1, fair_2, fair_3).
    """
    total = p1 + p2 + p3
    if total <= 1.0:
        return p1 / total, p2 / total, p3 / total

    lo, hi = 1.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        val = p1 ** mid + p2 ** mid + p3 ** mid
        if val > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-10:
            break

    k = (lo + hi) / 2.0
    fair_1 = p1 ** k
    fair_2 = p2 ** k
    fair_3 = p3 ** k

    s = fair_1 + fair_2 + fair_3
    return fair_1 / s, fair_2 / s, fair_3 / s


# =============================================================================
# Calibration Parameters
# =============================================================================

def compute_calibration_params(history_paired, cutoff_date, poisson_rho=None):
    """Compute all calibration parameters from historical games.

    Replicates the exact calibration logic from walk_forward_3way.py:
      - rho and home_ice_shift: computed from PRE-CUTOFF history only
        (fixed for the season, matching the backtest)
      - tie inflation (Bayesian): uses season-to-date games (cutoff → today)
      - scoring anchor: uses all completed games (disabled when ANCHOR_STRENGTH=0)

    Args:
        history_paired: DataFrame of ALL completed paired games before today,
            with columns: game_date, lam_home_raw, lam_away_raw, home_reg_gf,
            away_reg_gf, home_won_actual, game_outcome_type, is_ot
        cutoff_date: pd.Timestamp — start of current test season (e.g. 2025-10-01)
        poisson_rho: If None, estimate from data. If float, use directly.

    Returns dict with:
        poisson_rho, tie_inflation, home_ice_shift,
        anchor_home, anchor_away
    """
    history = history_paired.copy()

    # Pre-cutoff history: used for rho and home_ice_shift (matches backtest)
    pre_cutoff = history[history["game_date"] < cutoff_date]

    # --- Bivariate Poisson rho (from pre-cutoff only) ---
    if poisson_rho is None:
        if len(pre_cutoff) > 100:
            poisson_rho = estimate_poisson_rho(pre_cutoff)
        else:
            poisson_rho = 0.0

    # --- Bayesian tie inflation (season-to-date) ---
    # Build tie records for ALL history (pre-cutoff + current season)
    tie_records = []
    for _, row in history.iterrows():
        _, _, p_tie_raw = compute_game_probabilities(
            row["lam_home_raw"], row["lam_away_raw"], rho=poisson_rho)
        tie_records.append({
            "game_date": row["game_date"],
            "is_ot": row["is_ot"],
            "p_tie_raw": p_tie_raw,
        })
    tie_df = pd.DataFrame(tie_records)

    # Prior: computed from pre-cutoff data, capped at MAX_PRIOR_GAMES
    # to keep appropriate prior strength (~2 seasons, not 5+)
    pre_cutoff_tie = tie_df[tie_df["game_date"] < cutoff_date]
    prior_ot_rate = pre_cutoff_tie["is_ot"].mean() if len(pre_cutoff_tie) > 0 else 0.224
    prior_games = min(len(pre_cutoff_tie), MAX_PRIOR_GAMES)
    prior_ptie_mean = pre_cutoff_tie["p_tie_raw"].mean() if len(pre_cutoff_tie) > 0 else 0.163

    # Season games: from cutoff to latest available
    season_games = tie_df[tie_df["game_date"] >= cutoff_date]
    n_season = len(season_games)
    if n_season >= 20:
        season_ot_rate = season_games["is_ot"].mean()
        season_ptie_mean = season_games["p_tie_raw"].mean()
    else:
        season_ot_rate = prior_ot_rate
        season_ptie_mean = prior_ptie_mean

    # Bayesian posterior: shrink season rate toward prior
    n_ot = season_ot_rate * n_season
    shrunk_ot_rate = (
        (prior_ot_rate * prior_games + n_ot)
        / (prior_games + n_season)
    )
    tie_inflation = shrunk_ot_rate / season_ptie_mean if season_ptie_mean > 0 else 1.35

    # --- Home-ice shift (from pre-cutoff only, matches backtest) ---
    home_ice_shift = 0.0
    if len(pre_cutoff) > 100:
        reg_only = pre_cutoff[pre_cutoff["game_outcome_type"] == "REG"]
        if len(reg_only) > 50:
            actual_home_share = np.clip(reg_only["home_won_actual"].mean(), 0.01, 0.99)
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

    # --- Scoring anchors (all completed games, lookback from today) ---
    latest_date = history["game_date"].max() + pd.Timedelta(days=1)
    home_anchor_df = history[["game_date", "home_reg_gf", "lam_home_raw"]].rename(
        columns={"home_reg_gf": "reg_gf", "lam_home_raw": "lam_raw"})
    away_anchor_df = history[["game_date", "away_reg_gf", "lam_away_raw"]].rename(
        columns={"away_reg_gf": "reg_gf", "lam_away_raw": "lam_raw"})

    anchor_home = compute_scoring_anchor(home_anchor_df, latest_date)
    anchor_away = compute_scoring_anchor(away_anchor_df, latest_date)

    return {
        "poisson_rho": poisson_rho,
        "tie_inflation": tie_inflation,
        "home_ice_shift": home_ice_shift,
        "anchor_home": anchor_home,
        "anchor_away": anchor_away,
        "tie_records": tie_records,  # for incremental updates if needed
    }


# =============================================================================
# Single-Game 3-Way Prediction
# =============================================================================

def predict_game_3way(lam_home_raw, lam_away_raw, calib_params):
    """Predict 3-way probabilities for a single game.

    Args:
        lam_home_raw: Raw expected goals for home team (from quantile integration)
        lam_away_raw: Raw expected goals for away team
        calib_params: Dict from compute_calibration_params

    Returns:
        dict with p_home_reg, p_away_reg, p_ot, lam_home, lam_away
    """
    rho = calib_params["poisson_rho"]
    tie_inflation = calib_params["tie_inflation"]
    home_ice_shift = calib_params["home_ice_shift"]
    anchor_home = calib_params["anchor_home"]
    anchor_away = calib_params["anchor_away"]

    # Apply scoring anchor
    lam_home = np.clip(lam_home_raw * anchor_home, *LAMBDA_CLIP)
    lam_away = np.clip(lam_away_raw * anchor_away, *LAMBDA_CLIP)

    # Raw Poisson probabilities
    p_home_reg, p_away_reg, p_tie_raw = compute_game_probabilities(
        lam_home, lam_away, rho=rho)

    # Tie inflation → 3-way model probabilities
    p_home_reg_cal, p_away_reg_cal, p_ot = apply_tie_calibration(
        p_home_reg, p_away_reg, p_tie_raw, tie_inflation)

    # Home-ice correction in logit space
    if home_ice_shift != 0.0:
        reg_total = p_home_reg_cal + p_away_reg_cal
        if reg_total > 0:
            home_share = p_home_reg_cal / reg_total
            logit_share = np.log(max(home_share, 1e-6) / max(1 - home_share, 1e-6))
            home_share_adj = 1.0 / (1.0 + np.exp(-(logit_share + home_ice_shift)))
            p_home_reg_cal = reg_total * home_share_adj
            p_away_reg_cal = reg_total * (1.0 - home_share_adj)

    return {
        "p_home_reg": p_home_reg_cal,
        "p_away_reg": p_away_reg_cal,
        "p_ot": p_ot,
        "lam_home": lam_home,
        "lam_away": lam_away,
    }


# =============================================================================
# Edge Calculation
# =============================================================================

def calculate_edges(model_probs, odds_row):
    """Calculate edges for a single game against a single bookmaker.

    Args:
        model_probs: dict with p_home_reg, p_away_reg, p_ot
        odds_row: dict with home_dec, draw_dec, away_dec, bookmaker

    Returns:
        dict with edges, fair probs, and bet recommendation
    """
    # Devig
    home_imp = 1.0 / odds_row["home_dec"]
    draw_imp = 1.0 / odds_row["draw_dec"]
    away_imp = 1.0 / odds_row["away_dec"]

    fair_home, fair_draw, fair_away = power_devig_3way(home_imp, draw_imp, away_imp)

    # Edges
    home_edge = model_probs["p_home_reg"] - fair_home
    draw_edge = model_probs["p_ot"] - fair_draw
    away_edge = model_probs["p_away_reg"] - fair_away

    # Blend strategy: both sides bet independently (matches backtest)
    draw_bet = draw_edge >= DRAW_MIN_EDGE
    home_bet = HOME_MIN_EDGE <= home_edge <= HOME_MAX_EDGE
    draw_stake = min(draw_edge * DRAW_STAKE_MULT, DRAW_STAKE_CAP) if draw_bet else 0.0
    home_stake = min(home_edge * HOME_STAKE_MULT, HOME_STAKE_CAP) if home_bet else 0.0

    # Legacy single-bet columns: pick the primary side for backwards compat
    if draw_bet:
        bet_side = "draw"
        bet_edge = draw_edge
        bet_odds_dec = odds_row["draw_dec"]
        bet_model_prob = model_probs["p_ot"]
        kelly_stake = draw_stake
    elif home_bet:
        bet_side = "home"
        bet_edge = home_edge
        bet_odds_dec = odds_row["home_dec"]
        bet_model_prob = model_probs["p_home_reg"]
        kelly_stake = home_stake
    else:
        bet_side = None
        bet_edge = 0.0
        bet_odds_dec = 0.0
        bet_model_prob = 0.0
        kelly_stake = 0.0

    return {
        "fair_home": fair_home,
        "fair_draw": fair_draw,
        "fair_away": fair_away,
        "dec_home": odds_row["home_dec"],
        "dec_draw": odds_row["draw_dec"],
        "dec_away": odds_row["away_dec"],
        "home_edge": home_edge,
        "draw_edge": draw_edge,
        "away_edge": away_edge,
        "draw_bet": int(draw_bet),
        "draw_stake": round(draw_stake, 4),
        "home_bet": int(home_bet),
        "home_stake": round(home_stake, 4),
        "bet_side": bet_side,
        "bet_edge": bet_edge,
        "bet_odds_dec": bet_odds_dec,
        "bet_model_prob": bet_model_prob,
        "kelly_stake": kelly_stake,
        "bookmaker": odds_row.get("bookmaker", "unknown"),
    }


# =============================================================================
# Formatting
# =============================================================================

def format_recommendations(games, date_str):
    """Format predictions and bet recommendations for console output.

    Args:
        games: list of dicts, each with:
            home_team, away_team, p_home_reg, p_ot, p_away_reg,
            lam_home, lam_away, odds_results (list of edge dicts)
        date_str: Date string for header

    Returns:
        Formatted string
    """
    lines = []
    lines.append("")
    lines.append(f"{'='*78}")
    lines.append(f"  NHL 3-Way Predictions — {date_str}")
    lines.append(f"{'='*78}")
    lines.append("")

    # Identify primary book (first in odds_results, should be DraftKings)
    primary_book = "Book"
    for g in games:
        if g.get("odds_results") and len(g["odds_results"]) > 0:
            bk = g["odds_results"][0].get("bookmaker", "")
            if bk:
                primary_book = bk.replace("_", " ").title()
                if "draftkings" in bk.lower():
                    primary_book = "DK"
                break

    # Header
    lines.append(f"  {'Game':<16} {'Mdl H%':>7} {'Mdl OT%':>8} {'Mdl A%':>7}"
                  f" │ {primary_book+' H%':>8} {primary_book+' OT%':>9} {primary_book+' A%':>8}"
                  f" │ {'D.Edge':>7} {'H.Edge':>7}")
    lines.append(f"  {'─'*16} {'─'*7} {'─'*8} {'─'*7}"
                  f" │ {'─'*8} {'─'*9} {'─'*8}"
                  f" │ {'─'*7} {'─'*7}")

    for g in games:
        matchup = f"{g['away_team']}@{g['home_team']}"
        mdl_h = f"{g['p_home_reg']*100:.1f}"
        mdl_d = f"{g['p_ot']*100:.1f}"
        mdl_a = f"{g['p_away_reg']*100:.1f}"

        # Primary book odds (first in list = DraftKings after sorting)
        if g.get("odds_results") and len(g["odds_results"]) > 0:
            best = g["odds_results"][0]
            bk_h = f"{best['fair_home']*100:.1f}"
            bk_d = f"{best['fair_draw']*100:.1f}"
            bk_a = f"{best['fair_away']*100:.1f}"
            d_edge = f"{best['draw_edge']*100:+.1f}%"
            h_edge = f"{best['home_edge']*100:+.1f}%"
        else:
            bk_h = bk_d = bk_a = "  ---"
            d_edge = h_edge = "   ---"

        lines.append(f"  {matchup:<16} {mdl_h:>7} {mdl_d:>8} {mdl_a:>7}"
                      f" │ {bk_h:>8} {bk_d:>9} {bk_a:>8}"
                      f" │ {d_edge:>7} {h_edge:>7}")

    # Bet recommendations
    bets = []
    for g in games:
        for edge_result in g.get("odds_results", []):
            if edge_result.get("bet_side") and edge_result["kelly_stake"] > 0:
                bets.append({
                    "matchup": f"{g['away_team']}@{g['home_team']}",
                    **edge_result,
                })

    lines.append("")
    if bets:
        lines.append(f"  ── Bet Recommendations {'─'*40}")
        for bet in sorted(bets, key=lambda x: -x["bet_edge"]):
            side_label = bet["bet_side"].upper()
            lines.append(
                f"  {side_label:<5} {bet['matchup']:<16} "
                f"Edge {bet['bet_edge']*100:+.1f}%   "
                f"Odds {bet['bet_odds_dec']:.2f}   "
                f"{bet['bookmaker']:<15} "
                f"{bet['kelly_stake']:.2f}u"
            )
    else:
        lines.append("  No bets recommended today.")

    lines.append("")
    return "\n".join(lines)
