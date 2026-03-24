# -*- coding: utf-8 -*-
"""
Poisson Combiner (Layer 3)

Combines Layer 1 (regulation quantile XGBoost) and Layer 2 (OT edge model)
into final win probabilities. Compares to DraftKings odds to find edges.

Pipeline:
  1. Load regulation predictions, OT predictions, game outcomes
  2. Quantile → Lambda via trapezoidal integration
  3. Pair home/away by game_id
  4. Merge game outcomes
  5. Merge OT predictions
  6. Scoring anchor (60-day rolling)
  7. Compute raw Poisson game probabilities
  8. Calibrate P(tie) with rolling tie-inflation
  9. Load odds + power devig → compute edges
  10. Diagnostics + save

Outputs:
  - data/processed/combined_predictions.csv

@author: chazf
"""

import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from config import PROJECT_DIR, PROCESSED_DIR, RAW_DIR, GAME_IDS_FILE, ODDS_2026_FILE, ODDS_HIST_FILE

# =============================================================================
# Configuration
# =============================================================================
REG_PREDS_FILE = os.path.join(PROCESSED_DIR, "regulation_predictions.csv")
OT_PREDS_FILE = os.path.join(PROCESSED_DIR, "ot_predictions.csv")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "combined_predictions.csv")

# Quantile columns and their probability levels
QUANTILE_COLS = ["q05", "q10", "q15", "q20", "q30", "q40", "q50", "q60",
                 "q70", "q80", "q85", "q90", "q95"]
QUANTILE_PROBS = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60,
                  0.70, 0.80, 0.85, 0.90, 0.95]

# Scoring anchor
ANCHOR_LOOKBACK_DAYS = 60
ANCHOR_MIN_GAMES = 100
ANCHOR_CLIP = (0.8, 1.2)
ANCHOR_STRENGTH = 0.0  # 0.0 = disabled, 1.0 = full correction. Waterfall shows anchor hurts LL.

# Tie calibration
TIE_LOOKBACK_DAYS = 120

# Poisson
MAX_GOALS = 12
LAMBDA_CLIP = (0.1, 15.0)

# OT
DEFAULT_P_HOME_OT = 0.518

# Bivariate Poisson correlation parameter
# Estimated from training data; 0.0 = independent (current), positive = correlated
BIVARIATE_RHO = None  # None = estimate from data, float = fixed value

# Team name mapping (odds_collected.csv full names → abbreviations)
TEAM_NAME_TO_ABBREV = {
    "Anaheim Ducks": "ANA",
    "Arizona Coyotes": "ARI",
    "Boston Bruins": "BOS",
    "Buffalo Sabres": "BUF",
    "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR",
    "Chicago Blackhawks": "CHI",
    "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ",
    "Dallas Stars": "DAL",
    "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM",
    "Florida Panthers": "FLA",
    "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN",
    "Montréal Canadiens": "MTL",
    "Montreal Canadiens": "MTL",
    "Nashville Predators": "NSH",
    "New Jersey Devils": "NJD",
    "New York Islanders": "NYI",
    "New York Rangers": "NYR",
    "Ottawa Senators": "OTT",
    "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT",
    "San Jose Sharks": "SJS",
    "Seattle Kraken": "SEA",
    "St Louis Blues": "STL",
    "St. Louis Blues": "STL",
    "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR",
    "Utah Hockey Club": "UTA",
    "Utah Mammoth": "UTA",
    "Vancouver Canucks": "VAN",
    "Vegas Golden Knights": "VGK",
    "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}


# =============================================================================
# Helper Functions
# =============================================================================

def quantiles_to_lambda(q_values, quantiles=QUANTILE_PROBS):
    """Convert quantile predictions to expected goals via trapezoidal integration.

    Extrapolates tails: Q(0) = max(0, Q(0.05) - slope), Q(1) = Q(0.95) + slope
    15-point grid: [0, 0.05, 0.10, ..., 0.95, 1.0]
    """
    q_arr = np.asarray(q_values, dtype=float)

    # Extrapolate tails
    slope_low = q_arr[1] - q_arr[0]    # Q(0.10) - Q(0.05)
    slope_high = q_arr[-1] - q_arr[-2]  # Q(0.95) - Q(0.90)

    q0 = max(0.0, q_arr[0] - slope_low)
    q1 = q_arr[-1] + slope_high

    # Build full grid
    full_probs = np.concatenate([[0.0], quantiles, [1.0]])
    full_values = np.concatenate([[q0], q_arr, [q1]])

    # Trapezoidal integration: E[X] = integral_0^1 Q(p) dp
    # np.trapz removed in numpy 2.0, renamed to np.trapezoid
    _trapz = getattr(np, 'trapezoid', getattr(np, 'trapz', None))
    lam = _trapz(full_values, full_probs)

    return np.clip(lam, LAMBDA_CLIP[0], LAMBDA_CLIP[1])


def poisson_pmf(lam, max_k=MAX_GOALS):
    """Poisson PMF for k=0..max_k via iterative computation (no scipy)."""
    pmf = np.zeros(max_k + 1)
    pmf[0] = np.exp(-lam)
    for k in range(1, max_k + 1):
        pmf[k] = pmf[k - 1] * lam / k
    return pmf


def compute_game_probabilities(lam_h, lam_a, rho=0.0):
    """Compute Poisson game outcome probabilities.

    When rho=0: independent Poisson (original behavior).
    When rho>0: bivariate Poisson with shared component.

    Shared component model:
      X_h = Y + Z_h, X_a = Y + Z_a
      Y ~ Poisson(lam_shared), Z_h ~ Poisson(lam_h - lam_shared), Z_a ~ Poisson(lam_a - lam_shared)
      lam_shared = rho * min(lam_h, lam_a)

    Returns: (p_home_reg_win, p_away_reg_win, p_tie_raw)
    """
    if rho <= 0.0:
        # Independent case (original behavior)
        pmf_h = poisson_pmf(lam_h)
        pmf_a = poisson_pmf(lam_a)
        joint = np.outer(pmf_h, pmf_a)
    else:
        # Bivariate Poisson
        lam_shared = rho * min(lam_h, lam_a)
        lam_h_indep = max(lam_h - lam_shared, 0.01)
        lam_a_indep = max(lam_a - lam_shared, 0.01)

        pmf_shared = poisson_pmf(lam_shared)
        pmf_h_indep = poisson_pmf(lam_h_indep)
        pmf_a_indep = poisson_pmf(lam_a_indep)

        # joint[h, a] = sum_k P(Y=k) * P(Z_h=h-k) * P(Z_a=a-k)
        n = MAX_GOALS + 1
        joint = np.zeros((n, n))
        for k in range(n):
            if pmf_shared[k] < 1e-15:
                continue
            for h in range(k, n):
                for a in range(k, n):
                    joint[h, a] += pmf_shared[k] * pmf_h_indep[h - k] * pmf_a_indep[a - k]

    h_idx, a_idx = np.meshgrid(np.arange(MAX_GOALS + 1),
                                np.arange(MAX_GOALS + 1), indexing='ij')

    p_home_reg_win = joint[h_idx > a_idx].sum()
    p_away_reg_win = joint[h_idx < a_idx].sum()
    p_tie_raw = np.diag(joint).sum()

    return p_home_reg_win, p_away_reg_win, p_tie_raw


def apply_tie_calibration(p_home_reg, p_away_reg, p_tie_raw, tie_inflation):
    """Apply tie-inflation and redistribute to maintain sum = 1."""
    p_tie = p_tie_raw * tie_inflation

    # Redistribute excess from regulation wins proportionally
    reg_total = p_home_reg + p_away_reg
    if reg_total > 0:
        scale = (1.0 - p_tie) / reg_total
    else:
        scale = 1.0

    return p_home_reg * scale, p_away_reg * scale, p_tie


def compute_home_ice_shift(history_df):
    """Compute fixed home-ice logit shift from training history.

    Compares actual home win rate to model's mean predicted P(home)
    across all history games. Returns additive logit correction.
    Positive = boost home probability.
    """
    actual_home_wr = np.clip(history_df["home_won"].mean(), 0.01, 0.99)
    pred_home_mean = np.clip(history_df["p_home_win_raw"].mean(), 0.01, 0.99)

    shift = (np.log(actual_home_wr / (1 - actual_home_wr))
             - np.log(pred_home_mean / (1 - pred_home_mean)))
    return shift


def estimate_poisson_rho(history_df):
    """Estimate shared Poisson component from historical game data.

    Uses method of moments: Cov(X_h, X_a) = lam_shared = rho * min(lam_h, lam_a).
    Estimates rho as sample_cov / mean(min(lam_h, lam_a)).
    Clips to [0, 0.15] — hockey correlation is small but real.
    """
    cov = np.cov(history_df["home_reg_gf"], history_df["away_reg_gf"])[0, 1]
    mean_min_lam = np.minimum(
        history_df["lam_home_raw"], history_df["lam_away_raw"]
    ).mean()

    if mean_min_lam <= 0:
        return 0.0

    rho = cov / mean_min_lam
    rho = np.clip(rho, 0.0, 0.15)  # Small positive range
    return float(rho)


def american_to_implied(odds):
    """Convert American odds to raw implied probability."""
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    else:
        return (-odds) / (-odds + 100.0)


def power_devig(home_imp, away_imp):
    """Remove vig using power method.

    Finds k such that home_imp^k + away_imp^k = 1.
    Returns (fair_home, fair_away).
    """
    if home_imp <= 0 or away_imp <= 0:
        return np.nan, np.nan

    total = home_imp + away_imp
    if total <= 1.0:
        return home_imp / total, away_imp / total

    # Bisection to find k where p_h^k + p_a^k = 1
    lo, hi = 1.0, 20.0
    for _ in range(100):
        mid = (lo + hi) / 2.0
        val = home_imp ** mid + away_imp ** mid
        if val > 1.0:
            lo = mid
        else:
            hi = mid

    k = (lo + hi) / 2.0
    fair_h = home_imp ** k
    fair_a = away_imp ** k

    # Normalize for numerical stability
    s = fair_h + fair_a
    return fair_h / s, fair_a / s


def compute_scoring_anchor(anchor_df, game_date,
                           lookback_days=ANCHOR_LOOKBACK_DAYS):
    """Compute multiplicative scoring correction for a given date.

    Uses lookback window of completed games (no lookahead).
    Returns correction factor shrunk toward 1.0 by ANCHOR_STRENGTH.
    """
    if ANCHOR_STRENGTH == 0.0:
        return 1.0

    cutoff = game_date - pd.Timedelta(days=lookback_days)
    mask = (anchor_df["game_date"] >= cutoff) & (anchor_df["game_date"] < game_date)
    window = anchor_df.loc[mask]

    if len(window) < ANCHOR_MIN_GAMES:
        return 1.0

    actual_mean = window["reg_gf"].mean()
    pred_mean = window["lam_raw"].mean()

    if pred_mean <= 0:
        return 1.0

    raw_correction = actual_mean / pred_mean
    raw_correction = np.clip(raw_correction, ANCHOR_CLIP[0], ANCHOR_CLIP[1])

    # Shrink toward 1.0: strength=0 → no correction, strength=1 → full correction
    correction = 1.0 + ANCHOR_STRENGTH * (raw_correction - 1.0)
    return correction


def calibrate_tie_probability(df, lookback_days=TIE_LOOKBACK_DAYS):
    """Compute rolling tie-inflation factor per game date.

    tie_inflation = actual_ot_rate / mean(p_tie_raw) in lookback window.
    """
    dates = np.sort(df["game_date"].unique())
    inflation_map = {}

    # Precompute global fallback
    global_ot_rate = df["is_ot"].mean()
    global_tie_raw = df["p_tie_raw"].mean()
    global_inflation = global_ot_rate / global_tie_raw if global_tie_raw > 0 else 1.0

    for d in dates:
        cutoff = d - pd.Timedelta(days=lookback_days)
        mask = (df["game_date"] >= cutoff) & (df["game_date"] < d)
        window = df.loc[mask]

        if len(window) < 50:
            inflation_map[d] = global_inflation
        else:
            actual_ot_rate = window["is_ot"].mean()
            pred_tie_mean = window["p_tie_raw"].mean()
            if pred_tie_mean > 0:
                inflation_map[d] = actual_ot_rate / pred_tie_mean
            else:
                inflation_map[d] = global_inflation

    return inflation_map


def fit_platt_scaling(logit_pred, y, max_iter=50):
    """Fit Platt scaling via Newton-Raphson (no sklearn dependency).

    logit(p_calibrated) = a * logit(p_raw) + b
    Returns (a, b). a > 1 means model is compressed, b > 0 shifts toward home.
    """
    a, b = 1.0, 0.0
    for _ in range(max_iter):
        z = a * logit_pred + b
        z = np.clip(z, -20, 20)
        p = 1.0 / (1.0 + np.exp(-z))

        r = y - p
        w = p * (1 - p)

        # Gradient of log-likelihood
        g = np.array([np.dot(r, logit_pred), r.sum()])

        # Hessian (negative definite)
        H = np.array([
            [-np.dot(w * logit_pred, logit_pred), -np.dot(w, logit_pred)],
            [-np.dot(w, logit_pred), -w.sum()]
        ])

        # Newton step: delta = -H^{-1} @ g
        det = H[0, 0] * H[1, 1] - H[0, 1] * H[1, 0]
        if abs(det) < 1e-10:
            break
        delta = np.array([
            -(H[1, 1] * g[0] - H[0, 1] * g[1]) / det,
            -(H[0, 0] * g[1] - H[1, 0] * g[0]) / det,
        ])

        a += delta[0]
        b += delta[1]

        if np.abs(delta).max() < 1e-6:
            break

    return a, b


def calibrate_win_probabilities(df, lookback_days=TIE_LOOKBACK_DAYS):
    """Rolling Platt calibration for P(home win).

    Fits logit(p_cal) = a * logit(p_raw) + b on lookback window.
    Returns dict mapping game_date -> (a, b).
    """
    dates = np.sort(df["game_date"].unique())
    cal_map = {}

    # Global fallback: fit on all data
    p_all = np.clip(df["p_home_win_raw"].values, 0.01, 0.99)
    logit_all = np.log(p_all / (1 - p_all))
    y_all = df["home_won_actual"].values
    global_a, global_b = fit_platt_scaling(logit_all, y_all)

    for d in dates:
        cutoff = d - pd.Timedelta(days=lookback_days)
        mask = (df["game_date"] >= cutoff) & (df["game_date"] < d)
        window = df.loc[mask]

        if len(window) < 200:
            cal_map[d] = (global_a, global_b)
        else:
            p_raw = np.clip(window["p_home_win_raw"].values, 0.01, 0.99)
            logit_p = np.log(p_raw / (1 - p_raw))
            y = window["home_won_actual"].values
            a, b = fit_platt_scaling(logit_p, y)
            # Sanity: a should be positive and not extreme
            if a < 0.5 or a > 5.0:
                cal_map[d] = (global_a, global_b)
            else:
                cal_map[d] = (a, b)

    return cal_map


def apply_platt_calibration(p_raw, a, b):
    """Apply Platt scaling to probability."""
    p_clipped = np.clip(p_raw, 0.001, 0.999)
    logit_p = np.log(p_clipped / (1 - p_clipped))
    logit_cal = a * logit_p + b
    logit_cal = np.clip(logit_cal, -10, 10)
    return 1.0 / (1.0 + np.exp(-logit_cal))


# =============================================================================
# Odds Loading
# =============================================================================

def load_and_merge_odds(games_df):
    """Load both odds files, match to games, apply power devig."""
    print("\n  Loading odds...")

    odds_cols = ["dk_home_imp", "dk_away_imp", "dk_home_fair", "dk_away_fair"]
    for col in odds_cols:
        games_df[col] = np.nan

    # --- 2025-26 odds (direct game_id match) ---
    if os.path.exists(ODDS_2026_FILE):
        odds_2026 = pd.read_csv(ODDS_2026_FILE)

        # Drop unnamed index column if present
        unnamed = [c for c in odds_2026.columns if c.startswith("Unnamed") or c == ""]
        if unnamed:
            odds_2026 = odds_2026.drop(columns=unnamed)

        # Deduplicate: one row per game_id
        odds_2026 = odds_2026.drop_duplicates(subset="game_id", keep="first")

        # Convert odds
        odds_2026["dk_home_imp"] = odds_2026["home_american"].apply(american_to_implied)
        odds_2026["dk_away_imp"] = odds_2026["away_american"].apply(american_to_implied)

        # Power devig
        devigged = odds_2026.apply(
            lambda row: power_devig(row["dk_home_imp"], row["dk_away_imp"]),
            axis=1, result_type="expand"
        )
        odds_2026["dk_home_fair"] = devigged[0]
        odds_2026["dk_away_fair"] = devigged[1]

        # Merge by game_id
        odds_map = odds_2026.set_index("game_id")[odds_cols]
        matched_mask = games_df["game_id"].isin(odds_map.index)
        matched_ids = games_df.loc[matched_mask, "game_id"]
        for col in odds_cols:
            games_df.loc[matched_mask, col] = matched_ids.map(odds_map[col]).values

        n_2026 = matched_mask.sum()
        print(f"    2025-26 odds: matched {n_2026} games")

    # --- Historical odds (date + home team match) ---
    if os.path.exists(ODDS_HIST_FILE):
        odds_hist = pd.read_csv(ODDS_HIST_FILE)

        # Map team names to abbreviations
        odds_hist["home_abbrev"] = odds_hist["home_team_odds"].map(TEAM_NAME_TO_ABBREV)

        # Drop unmapped teams
        unmapped = odds_hist["home_abbrev"].isna()
        if unmapped.any():
            unmapped_names = odds_hist.loc[unmapped, "home_team_odds"].unique()
            print(f"    WARNING: unmapped team names: {unmapped_names}")
        odds_hist = odds_hist[~unmapped].copy()

        # Convert date
        odds_hist["odds_date"] = pd.to_datetime(
            odds_hist["odds_date"]
        ).dt.strftime("%Y-%m-%d")

        # Deduplicate: latest odds per (date, home_team)
        odds_hist = odds_hist.sort_values("odds_timestamp_requested",
                                          ascending=False)
        odds_hist = odds_hist.drop_duplicates(
            subset=["odds_date", "home_abbrev"], keep="first"
        )

        # Convert odds
        odds_hist["home_odds_american"] = pd.to_numeric(
            odds_hist["home_odds_american"], errors="coerce"
        )
        odds_hist["away_odds_american"] = pd.to_numeric(
            odds_hist["away_odds_american"], errors="coerce"
        )
        odds_hist = odds_hist.dropna(
            subset=["home_odds_american", "away_odds_american"]
        )

        odds_hist["dk_home_imp"] = odds_hist["home_odds_american"].apply(
            american_to_implied
        )
        odds_hist["dk_away_imp"] = odds_hist["away_odds_american"].apply(
            american_to_implied
        )

        devigged = odds_hist.apply(
            lambda row: power_devig(row["dk_home_imp"], row["dk_away_imp"]),
            axis=1, result_type="expand"
        )
        odds_hist["dk_home_fair"] = devigged[0]
        odds_hist["dk_away_fair"] = devigged[1]

        # Create match key
        odds_hist["match_key"] = (
            odds_hist["odds_date"] + "_" + odds_hist["home_abbrev"]
        )
        odds_hist_map = odds_hist.set_index("match_key")[odds_cols]

        # Match games that don't already have odds
        needs_odds = games_df["dk_home_fair"].isna()
        games_df["match_key"] = (
            games_df["game_date"].dt.strftime("%Y-%m-%d") + "_"
            + games_df["home_team"]
        )

        hist_mask = needs_odds & games_df["match_key"].isin(odds_hist_map.index)
        hist_keys = games_df.loc[hist_mask, "match_key"]
        for col in odds_cols:
            games_df.loc[hist_mask, col] = hist_keys.map(
                odds_hist_map[col]
            ).values

        n_hist = hist_mask.sum()
        print(f"    Historical odds: matched {n_hist} additional games")

        games_df.drop(columns=["match_key"], inplace=True)

    total_with_odds = games_df["dk_home_fair"].notna().sum()
    print(f"    Total games with odds: {total_with_odds} / {len(games_df)}")

    return games_df


# =============================================================================
# Diagnostics
# =============================================================================

def run_diagnostics(df, label="Full"):
    """Print calibration tables, metrics, and edge analysis."""
    print(f"\n{'='*60}")
    print(f"DIAGNOSTICS: {label}")
    print(f"{'='*60}")

    # --- Lambda bias ---
    print(f"\n  Lambda bias (predicted vs actual regulation goals):")
    for season, grp in df.groupby("season"):
        pred_home = grp["lam_home"].mean()
        pred_away = grp["lam_away"].mean()
        act_home = grp["home_reg_gf"].mean()
        act_away = grp["away_reg_gf"].mean()
        print(f"    {season}: pred_h={pred_home:.2f} act_h={act_home:.2f} "
              f"({pred_home - act_home:+.2f}) | "
              f"pred_a={pred_away:.2f} act_a={act_away:.2f} "
              f"({pred_away - act_away:+.2f})")

    # --- Scoring anchor correction by season ---
    print(f"\n  Scoring anchor by season (home / away):")
    for season, grp in df.groupby("season"):
        mean_h = grp["anchor_home"].mean()
        mean_a = grp["anchor_away"].mean()
        print(f"    {season}: home={mean_h:.4f}, away={mean_a:.4f}")

    # --- P(OT) calibration by quintile ---
    print(f"\n  P(OT) calibration by quintile:")
    valid = df.dropna(subset=["p_ot", "is_ot"])
    if len(valid) > 100:
        valid = valid.copy()
        valid["ot_bin"] = pd.qcut(valid["p_ot"], 5, labels=False,
                                   duplicates="drop")
        for b, grp in valid.groupby("ot_bin"):
            pred = grp["p_ot"].mean()
            actual = grp["is_ot"].mean()
            print(f"    Bin {b}: pred={pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - pred:+.3f}, n={len(grp)}")

    # --- Win probability calibration by quintile ---
    print(f"\n  P(home win) calibration by quintile:")
    valid = df.dropna(subset=["p_home_win", "home_won_actual"])
    if len(valid) > 100:
        valid = valid.copy()
        valid["win_bin"] = pd.qcut(valid["p_home_win"], 5, labels=False,
                                    duplicates="drop")
        for b, grp in valid.groupby("win_bin"):
            pred = grp["p_home_win"].mean()
            actual = grp["home_won_actual"].mean()
            print(f"    Bin {b}: pred={pred:.3f}, actual={actual:.3f}, "
                  f"diff={actual - pred:+.3f}, n={len(grp)}")

    # --- Log-loss, Brier, AUC ---
    valid = df.dropna(subset=["p_home_win", "home_won_actual"])
    if len(valid) > 0:
        y_true = valid["home_won_actual"].values
        y_pred = np.clip(valid["p_home_win"].values, 0.001, 0.999)

        ll = -(y_true * np.log(y_pred)
               + (1 - y_true) * np.log(1 - y_pred)).mean()
        brier = ((y_pred - y_true) ** 2).mean()

        base_rate = y_true.mean()
        base_pred = np.full_like(y_pred, base_rate)
        base_ll = -(y_true * np.log(base_pred)
                    + (1 - y_true) * np.log(1 - base_pred)).mean()
        base_brier = ((base_pred - y_true) ** 2).mean()

        # AUC — manual: P(pred_pos > pred_neg)
        pos = y_pred[y_true == 1]
        neg = y_pred[y_true == 0]
        auc = np.mean(pos[:, None] > neg[None, :]) + \
              0.5 * np.mean(pos[:, None] == neg[None, :])

        print(f"\n  Overall metrics ({len(valid)} games):")
        print(f"    Log-loss:    {ll:.5f}  (baseline: {base_ll:.5f}, "
              f"improvement: {(1 - ll/base_ll)*100:+.1f}%)")
        print(f"    Brier score: {brier:.5f}  (baseline: {base_brier:.5f})")
        print(f"    AUC-ROC:     {auc:.4f}")
        print(f"    Home win rate: {base_rate:.3f}")

    # --- DK odds comparison ---
    with_odds = df.dropna(subset=["dk_home_fair", "home_won_actual"])
    if len(with_odds) > 50:
        y_true = with_odds["home_won_actual"].values
        model_pred = np.clip(with_odds["p_home_win"].values, 0.001, 0.999)
        dk_pred = np.clip(with_odds["dk_home_fair"].values, 0.001, 0.999)

        model_ll = -(y_true * np.log(model_pred)
                     + (1 - y_true) * np.log(1 - model_pred)).mean()
        dk_ll = -(y_true * np.log(dk_pred)
                  + (1 - y_true) * np.log(1 - dk_pred)).mean()

        print(f"\n  DK odds comparison ({len(with_odds)} games with odds):")
        print(f"    Model log-loss: {model_ll:.5f}")
        print(f"    DK log-loss:    {dk_ll:.5f}")
        print(f"    Difference:     {model_ll - dk_ll:+.5f}")

        # Edge distribution
        edges = with_odds["home_edge"].dropna()
        if len(edges) > 0:
            print(f"\n  Edge distribution (home):")
            print(f"    Mean:   {edges.mean():+.4f}")
            print(f"    Std:    {edges.std():.4f}")
            print(f"    Median: {edges.median():+.4f}")
            print(f"    |edge| > 3%: {(edges.abs() > 0.03).sum()} "
                  f"({(edges.abs() > 0.03).mean()*100:.1f}%)")
            print(f"    |edge| > 5%: {(edges.abs() > 0.05).sum()} "
                  f"({(edges.abs() > 0.05).mean()*100:.1f}%)")

        # Accuracy when edge > 3%
        big_edges = with_odds[with_odds["home_edge"].abs() > 0.03].copy()
        if len(big_edges) > 20:
            model_pick = big_edges["p_home_win"] > 0.5
            actual = big_edges["home_won_actual"] == 1
            accuracy = (model_pick == actual).mean()
            print(f"\n  Accuracy when |edge| > 3%: {accuracy:.3f} "
                  f"({len(big_edges)} games)")


# =============================================================================
# Main Pipeline
# =============================================================================

def run_poisson_combiner():
    """Full pipeline: combine regulation + OT predictions into win probs."""

    # =====================================================================
    # Step 1: Load data
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 1: Load Data")
    print(f"{'='*60}")

    reg = pd.read_csv(REG_PREDS_FILE)
    reg["game_date"] = pd.to_datetime(reg["game_date"])
    print(f"  Regulation predictions: {len(reg):,} rows "
          f"({reg['game_date'].min().date()} to "
          f"{reg['game_date'].max().date()})")

    ot = pd.read_csv(OT_PREDS_FILE)
    ot["game_date"] = pd.to_datetime(ot["game_date"])
    print(f"  OT predictions: {len(ot):,} rows")

    games = pd.read_csv(GAME_IDS_FILE)
    games["game_date"] = pd.to_datetime(games["game_date"])
    print(f"  Game IDs: {len(games):,} rows")

    # =====================================================================
    # Step 2: Quantile → Lambda
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 2: Quantile -> Lambda (Trapezoidal Integration)")
    print(f"{'='*60}")

    reg["lam_raw"] = reg[QUANTILE_COLS].apply(
        lambda row: quantiles_to_lambda(row.values), axis=1
    )

    print(f"  Lambda stats:")
    print(f"    Mean:   {reg['lam_raw'].mean():.3f}")
    print(f"    Median: {reg['lam_raw'].median():.3f}")
    print(f"    Std:    {reg['lam_raw'].std():.3f}")
    print(f"    Range:  [{reg['lam_raw'].min():.3f}, "
          f"{reg['lam_raw'].max():.3f}]")
    print(f"    Actual reg_gf mean: {reg['reg_gf'].mean():.3f}")

    # =====================================================================
    # Step 3: Pair home/away by game_id
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 3: Pair Home/Away by game_id")
    print(f"{'='*60}")

    home = reg[reg["is_home"] == 1][
        ["game_id", "game_date", "team", "opponent", "season",
         "reg_gf", "lam_raw"]
    ].copy()
    home.columns = ["game_id", "game_date", "home_team", "away_team",
                     "season", "home_reg_gf", "lam_home_raw"]

    away = reg[reg["is_home"] == 0][
        ["game_id", "reg_gf", "lam_raw"]
    ].copy()
    away.columns = ["game_id", "away_reg_gf", "lam_away_raw"]

    paired = home.merge(away, on="game_id", how="inner")
    print(f"  Paired games: {len(paired):,}")

    # =====================================================================
    # Step 4: Merge game outcomes
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 4: Merge Game Outcomes")
    print(f"{'='*60}")

    # Filter to completed games only (exclude FUT/scheduled games)
    completed = games[games["game_state"] == "OFF"].copy()
    n_excluded = len(games) - len(completed)
    print(f"  Completed games: {len(completed):,} "
          f"(excluded {n_excluded} unplayed)")

    games_slim = completed[
        ["game_id", "game_outcome_type", "home_score", "away_score"]
    ].copy()
    games_slim["is_ot"] = games_slim["game_outcome_type"].isin(
        ["OT", "SO"]
    ).astype(int)
    games_slim["home_won_actual"] = (
        games_slim["home_score"] > games_slim["away_score"]
    ).astype(int)

    paired = paired.merge(
        games_slim[["game_id", "game_outcome_type", "is_ot",
                     "home_won_actual"]],
        on="game_id", how="inner"
    )

    ot_rate = paired["is_ot"].mean()
    home_win_rate = paired["home_won_actual"].mean()
    print(f"  Paired after filter: {len(paired):,} games")
    print(f"  OT rate: {ot_rate:.3f} "
          f"({paired['is_ot'].sum()} / {len(paired)})")
    print(f"  Home win rate: {home_win_rate:.3f}")

    # =====================================================================
    # Step 5: Merge OT predictions
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 5: Merge OT Predictions")
    print(f"{'='*60}")

    ot_slim = ot[["game_id", "p_home_win_ot"]].copy()
    paired = paired.merge(ot_slim, on="game_id", how="left")

    n_with_ot = paired["p_home_win_ot"].notna().sum()
    n_without_ot = paired["p_home_win_ot"].isna().sum()
    paired["p_home_win_ot_given"] = paired["p_home_win_ot"].fillna(
        DEFAULT_P_HOME_OT
    )

    print(f"  Games with OT predictions: {n_with_ot}")
    print(f"  Games using default ({DEFAULT_P_HOME_OT}): {n_without_ot}")

    # =====================================================================
    # Step 6: Scoring anchor (60-day rolling, split home/away)
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 6: Scoring Anchor (60-day Rolling, Split Home/Away)")
    print(f"{'='*60}")

    # Separate home and away anchor DataFrames
    home_anchor_df = paired[["game_date", "home_reg_gf", "lam_home_raw"]].rename(
        columns={"home_reg_gf": "reg_gf", "lam_home_raw": "lam_raw"})
    away_anchor_df = paired[["game_date", "away_reg_gf", "lam_away_raw"]].rename(
        columns={"away_reg_gf": "reg_gf", "lam_away_raw": "lam_raw"})

    # Compute per-date corrections separately
    paired = paired.sort_values("game_date").reset_index(drop=True)
    corrections_home = {}
    corrections_away = {}
    for d in paired["game_date"].unique():
        corrections_home[d] = compute_scoring_anchor(home_anchor_df, d)
        corrections_away[d] = compute_scoring_anchor(away_anchor_df, d)

    paired["anchor_home"] = paired["game_date"].map(corrections_home)
    paired["anchor_away"] = paired["game_date"].map(corrections_away)
    paired["lam_home"] = paired["lam_home_raw"] * paired["anchor_home"]
    paired["lam_away"] = paired["lam_away_raw"] * paired["anchor_away"]

    # Clip lambdas
    paired["lam_home"] = paired["lam_home"].clip(*LAMBDA_CLIP)
    paired["lam_away"] = paired["lam_away"].clip(*LAMBDA_CLIP)

    print(f"  Home anchor: mean={paired['anchor_home'].mean():.4f}, "
          f"range=[{paired['anchor_home'].min():.4f}, "
          f"{paired['anchor_home'].max():.4f}]")
    print(f"  Away anchor: mean={paired['anchor_away'].mean():.4f}, "
          f"range=[{paired['anchor_away'].min():.4f}, "
          f"{paired['anchor_away'].max():.4f}]")
    print(f"  Adjusted lambda home: {paired['lam_home'].mean():.3f} "
          f"(raw: {paired['lam_home_raw'].mean():.3f}, "
          f"actual: {paired['home_reg_gf'].mean():.3f})")
    print(f"  Adjusted lambda away: {paired['lam_away'].mean():.3f} "
          f"(raw: {paired['lam_away_raw'].mean():.3f}, "
          f"actual: {paired['away_reg_gf'].mean():.3f})")

    # =====================================================================
    # Step 7: Raw Poisson game probabilities
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 7: Compute Raw Poisson Game Probabilities")
    print(f"{'='*60}")

    probs = paired.apply(
        lambda row: compute_game_probabilities(
            row["lam_home"], row["lam_away"]
        ),
        axis=1, result_type="expand"
    )
    paired["p_home_reg_win_raw"] = probs[0]
    paired["p_away_reg_win_raw"] = probs[1]
    paired["p_tie_raw"] = probs[2]

    print(f"  Raw probabilities:")
    print(f"    P(home reg win): {paired['p_home_reg_win_raw'].mean():.3f}")
    print(f"    P(away reg win): {paired['p_away_reg_win_raw'].mean():.3f}")
    print(f"    P(tie raw):      {paired['p_tie_raw'].mean():.3f}")
    print(f"    Actual OT rate:  {paired['is_ot'].mean():.3f}")

    # =====================================================================
    # Step 8: Calibrate P(tie) — rolling tie inflation
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 8: Calibrate P(tie) -- Rolling Tie Inflation")
    print(f"{'='*60}")

    inflation_map = calibrate_tie_probability(paired)
    paired["tie_inflation"] = paired["game_date"].map(inflation_map)

    # Apply tie calibration
    calibrated = paired.apply(
        lambda row: apply_tie_calibration(
            row["p_home_reg_win_raw"], row["p_away_reg_win_raw"],
            row["p_tie_raw"], row["tie_inflation"]
        ),
        axis=1, result_type="expand"
    )
    paired["p_home_reg_win"] = calibrated[0]
    paired["p_away_reg_win"] = calibrated[1]
    paired["p_ot"] = calibrated[2]

    # Raw win probabilities (before Platt calibration)
    paired["p_home_win_raw"] = (
        paired["p_home_reg_win"]
        + paired["p_ot"] * paired["p_home_win_ot_given"]
    )
    paired["p_away_win_raw"] = 1.0 - paired["p_home_win_raw"]

    print(f"  Tie inflation stats:")
    print(f"    Mean:  {paired['tie_inflation'].mean():.3f}")
    print(f"    Range: [{paired['tie_inflation'].min():.3f}, "
          f"{paired['tie_inflation'].max():.3f}]")
    print(f"\n  Pre-calibration probabilities:")
    print(f"    P(home reg win): {paired['p_home_reg_win'].mean():.3f}")
    print(f"    P(away reg win): {paired['p_away_reg_win'].mean():.3f}")
    print(f"    P(OT):           {paired['p_ot'].mean():.3f}")
    print(f"    P(home win raw): {paired['p_home_win_raw'].mean():.3f} "
          f"(actual: {paired['home_won_actual'].mean():.3f})")

    # =====================================================================
    # Step 8b: Rolling Platt calibration
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 8b: Rolling Platt Calibration")
    print(f"{'='*60}")

    cal_map = calibrate_win_probabilities(paired)
    paired["platt_a"] = paired["game_date"].map(
        {d: ab[0] for d, ab in cal_map.items()}
    )
    paired["platt_b"] = paired["game_date"].map(
        {d: ab[1] for d, ab in cal_map.items()}
    )

    # Apply calibration
    paired["p_home_win"] = paired.apply(
        lambda row: apply_platt_calibration(
            row["p_home_win_raw"], row["platt_a"], row["platt_b"]
        ),
        axis=1
    )
    paired["p_away_win"] = 1.0 - paired["p_home_win"]

    print(f"  Platt parameters:")
    print(f"    a (stretch): mean={paired['platt_a'].mean():.3f}, "
          f"range=[{paired['platt_a'].min():.3f}, "
          f"{paired['platt_a'].max():.3f}]")
    print(f"    b (bias):    mean={paired['platt_b'].mean():.3f}, "
          f"range=[{paired['platt_b'].min():.3f}, "
          f"{paired['platt_b'].max():.3f}]")
    print(f"\n  After Platt calibration:")
    print(f"    P(home win): {paired['p_home_win'].mean():.3f} "
          f"(actual: {paired['home_won_actual'].mean():.3f})")
    print(f"    P(home win) std: {paired['p_home_win'].std():.3f} "
          f"(raw std: {paired['p_home_win_raw'].std():.3f})")
    print(f"    P(home win) range: [{paired['p_home_win'].min():.3f}, "
          f"{paired['p_home_win'].max():.3f}]")

    # =====================================================================
    # Step 9: Load odds + compute edges
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 9: Load Odds + Compute Edges")
    print(f"{'='*60}")

    paired = load_and_merge_odds(paired)

    # Compute edges
    paired["home_edge"] = paired["p_home_win"] - paired["dk_home_fair"]
    paired["away_edge"] = paired["p_away_win"] - paired["dk_away_fair"]

    with_edges = paired.dropna(subset=["home_edge"])
    if len(with_edges) > 0:
        print(f"\n  Edge summary ({len(with_edges)} games):")
        print(f"    Home edge mean: {with_edges['home_edge'].mean():+.4f}")
        print(f"    Away edge mean: {with_edges['away_edge'].mean():+.4f}")

    # =====================================================================
    # Step 10: Diagnostics + Save
    # =====================================================================
    print(f"\n{'='*60}")
    print("STEP 10: Diagnostics + Save")
    print(f"{'='*60}")

    # Full dataset diagnostics
    run_diagnostics(paired, label="All Seasons")

    # 2025-26 only diagnostics
    current = paired[paired["season"] == 20252026]
    if len(current) > 50:
        run_diagnostics(current, label="2025-26 Season")

    # --- Save ---
    output_cols = [
        "game_id", "game_date", "home_team", "away_team", "season",
        "lam_home", "lam_away", "anchor_home", "anchor_away",
        "p_home_reg_win", "p_away_reg_win", "p_ot",
        "p_home_win_raw", "p_home_win", "p_away_win",
        "p_home_win_ot_given", "platt_a", "platt_b",
        "dk_home_fair", "dk_away_fair", "home_edge", "away_edge",
        "home_won_actual", "game_outcome_type",
    ]

    output = paired[output_cols].copy()
    output["game_date"] = output["game_date"].dt.strftime("%Y-%m-%d")

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    output.to_csv(OUTPUT_FILE, index=False)
    print(f"\n  Saved {len(output):,} games to {OUTPUT_FILE}")

    return paired


# =============================================================================
# Daily Prediction Helper
# =============================================================================

def predict_today(reg_preds_df, ot_preds_df=None):
    """Generate predictions for today's games from fresh Layer 1+2 output.

    Args:
        reg_preds_df: DataFrame with columns matching regulation_predictions.csv
                      (game_id, team, is_home, q05..q95)
        ot_preds_df: Optional DataFrame with game_id, p_home_win_ot

    Returns:
        DataFrame with per-game win probabilities
    """
    # Compute lambdas
    reg_preds_df = reg_preds_df.copy()
    reg_preds_df["lam"] = reg_preds_df[QUANTILE_COLS].apply(
        lambda row: quantiles_to_lambda(row.values), axis=1
    )

    # Pair home/away
    home = reg_preds_df[reg_preds_df["is_home"] == 1][
        ["game_id", "team", "opponent", "lam"]
    ].copy()
    home.columns = ["game_id", "home_team", "away_team", "lam_home"]

    away = reg_preds_df[reg_preds_df["is_home"] == 0][
        ["game_id", "lam"]
    ].copy()
    away.columns = ["game_id", "lam_away"]

    today = home.merge(away, on="game_id")

    # Load calibration from historical output
    anchor_home = 1.0
    anchor_away = 1.0
    tie_inflation = 1.38
    platt_a = 1.0
    platt_b = 0.0
    if os.path.exists(OUTPUT_FILE):
        hist = pd.read_csv(OUTPUT_FILE)
        anchor_home = hist["anchor_home"].iloc[-1]
        anchor_away = hist["anchor_away"].iloc[-1]
        platt_a = hist["platt_a"].iloc[-1]
        platt_b = hist["platt_b"].iloc[-1]

        # Recompute tie inflation from recent history
        recent = hist.tail(500)
        actual_ot_rate = recent["game_outcome_type"].isin(
            ["OT", "SO"]
        ).mean()
        p_ties = []
        for _, row in recent.iterrows():
            _, _, pt = compute_game_probabilities(
                row["lam_home"], row["lam_away"]
            )
            p_ties.append(pt)
        mean_p_tie_raw = np.mean(p_ties)
        if mean_p_tie_raw > 0:
            tie_inflation = actual_ot_rate / mean_p_tie_raw

    today["lam_home"] = (today["lam_home"] * anchor_home).clip(*LAMBDA_CLIP)
    today["lam_away"] = (today["lam_away"] * anchor_away).clip(*LAMBDA_CLIP)

    # Merge OT predictions
    if ot_preds_df is not None:
        ot_slim = ot_preds_df[["game_id", "p_home_win_ot"]].copy()
        today = today.merge(ot_slim, on="game_id", how="left")

    if "p_home_win_ot" not in today.columns:
        today["p_home_win_ot"] = np.nan
    today["p_home_win_ot_given"] = today["p_home_win_ot"].fillna(
        DEFAULT_P_HOME_OT
    )

    # Compute probabilities
    results = []
    for _, row in today.iterrows():
        p_h_reg, p_a_reg, p_tie_raw = compute_game_probabilities(
            row["lam_home"], row["lam_away"]
        )
        p_h_adj, p_a_adj, p_ot = apply_tie_calibration(
            p_h_reg, p_a_reg, p_tie_raw, tie_inflation
        )
        p_home_win_raw = p_h_adj + p_ot * row["p_home_win_ot_given"]
        p_home_win = apply_platt_calibration(p_home_win_raw, platt_a, platt_b)
        p_away_win = 1.0 - p_home_win

        results.append({
            "game_id": row["game_id"],
            "home_team": row["home_team"],
            "away_team": row["away_team"],
            "lam_home": row["lam_home"],
            "lam_away": row["lam_away"],
            "p_home_reg_win": p_h_adj,
            "p_away_reg_win": p_a_adj,
            "p_ot": p_ot,
            "p_home_win_raw": p_home_win_raw,
            "p_home_win": p_home_win,
            "p_away_win": p_away_win,
            "p_home_win_ot_given": row["p_home_win_ot_given"],
        })

    return pd.DataFrame(results)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Poisson Combiner (Layer 3)")
    print("=" * 60)

    run_poisson_combiner()

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
