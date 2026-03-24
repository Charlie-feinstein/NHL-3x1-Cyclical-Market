# -*- coding: utf-8 -*-
"""
Regulation Model (Layer 1)

13 Quantile XGBoost models predicting the distribution of regulation
goals (periods 1-3) for each team in each game.

Pipeline:
  1. Feature selection via permutation importance (105 → ~30 features)
  2. Optuna hyperparameter tuning (mean pinball loss across 13 quantiles)
  3. Final training with best params + selected features
  4. Conformal calibration + monotonicity enforcement

Outputs:
  - model_artifacts/regulation_model.pkl  : 13 models + calibration
  - data/processed/regulation_predictions.csv : quantile predictions

@author: chazf
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.inspection import permutation_importance
import optuna
import pickle
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import PROCESSED_DIR, RAW_DIR, MODEL_DIR, GAME_IDS_FILE

# =============================================================================
# Configuration
# =============================================================================

# Train cutoffs to generate OOS predictions for each test season
TRAIN_CUTOFFS = ["2024-10-01", "2025-10-01"]

# Quantiles to predict
QUANTILES = [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

TARGET_COL = "reg_gf"

# Temporal split
CAL_HOLDOUT_DAYS = 30   # conformal calibration holdout
TUNE_HOLDOUT_DAYS = 90  # tuning validation holdout (before conformal)

# Feature selection
N_SUPPLEMENTARY = 20          # extra features beyond core (via perm importance)
PERM_IMP_REPEATS = 50

# Locked features — set to None to enable permutation importance selection
# Exact 42 features from the best run (λ corr=0.142, ROI=+4.7%)
LOCKED_FEATURES = [
    # --- Core (23) ---
    "is_home",
    "gf_ewm_10", "ga_ewm_10",
    "xgf_ewm_10", "xga_ewm_10",
    "goal_diff_ewm_20",
    "shooting_pct_ewm_20", "save_pct_ewm_20",
    "opp_gf_ewm_10", "opp_ga_ewm_10",
    "opp_xgf_ewm_10", "opp_xga_ewm_10",
    "opp_save_pct_ewm_20", "opp_shooting_pct_ewm_20",
    "points_ewm_20", "opp_point_pct",
    "days_rest", "opp_days_rest", "rest_advantage",
    "is_back_to_back", "opp_is_b2b",
    "opp_goalie_ga_rate",
    "scoring_matchup",
    # --- Supplementary (20) ---
    "opp_pk_save_rate_ewm_20", "opp_fenwick_ewm_20",
    "opp_p3_ga_ewm_10", "road_trip_game_num",
    "efficiency_matchup", "shutout_rate_20",
    "opp_ga_std_10", "opp_corsi_ewm_20",
    "p3_ga_ewm_10", "opp_trailing_gf_ewm_20",
    "streak_value", "fatigue_advantage",
    "opp_pdo_ewm_20", "opp_pp_goals_ewm_10",
    "opp_corsi_behind_ewm_20", "opp_one_goal_rate_20",
    "opp_pim_ewm_10", "opp_pk_ga_ewm_10",
    "opp_blocks_ewm_10", "opp_goalie_starter_role_share",
]

# Core features: used for permutation importance when LOCKED_FEATURES is None
CORE_FEATURES = [
    "is_home",
    # --- Own scoring & defense ---
    "gf_ewm_10", "ga_ewm_10",
    "xgf_ewm_10", "xga_ewm_10",
    "goal_diff_ewm_20",
    "shooting_pct_ewm_20", "save_pct_ewm_20",
    # --- Opponent scoring & defense ---
    "opp_gf_ewm_10", "opp_ga_ewm_10",
    "opp_xgf_ewm_10", "opp_xga_ewm_10",
    "opp_save_pct_ewm_20", "opp_shooting_pct_ewm_20",
    # --- Quality & form ---
    "points_ewm_20", "opp_point_pct",
    # --- Schedule & rest ---
    "days_rest", "opp_days_rest", "rest_advantage",
    "is_back_to_back", "opp_is_b2b",
    # --- Goalie ---
    "opp_goalie_ga_rate",
    # --- Existing matchup interaction ---
    "scoring_matchup",
]

# Optuna
N_OPTUNA_TRIALS = 50

# Locked hyperparams — set to None to enable Optuna tuning
# These are from the best previous run (trial #41, pinball=0.4379, λ corr=0.142, ROI=+4.7%)
LOCKED_PARAMS = {
    "colsample_bytree": 0.43420835263970586,
    "learning_rate": 0.016992712273408404,
    "max_depth": 2,
    "min_child_weight": 133,
    "n_estimators": 800,
    "reg_alpha": 0.11406068290600342,
    "reg_lambda": 0.2887364155966318,
    "subsample": 0.9122223024459617,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

# Features — exclude identifiers, target, raw per-game stats
ID_COLS = ["game_id", "game_date", "season", "team", "opponent",
           "team_id", "period_count", "game_outcome_type"]

RAW_COLS = ["reg_ga", "reg_xgf", "reg_xga", "reg_shots_for",
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
            "is_blowout_win", "is_blowout_loss", "games_played",
            # PBP-based raw stats
            "pp_goals_pbp", "pp_shots_pbp", "pk_goals_against_pbp",
            "pk_shots_against_pbp", "pp_conversion_rate_pbp", "pk_save_rate_pbp",
            "trailing_gf", "leading_ga", "scored_first", "tied_gf"]

# Conservative defaults for feature selection model
FEATURE_SELECT_PARAMS = {
    "max_depth": 3,
    "learning_rate": 0.05,
    "n_estimators": 300,
    "min_child_weight": 50,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 1.0,
    "reg_lambda": 1.0,
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}

# Jitter scale for discrete targets
JITTER_SCALE = 0.5

# Time weighting: exponential decay by season distance from test season
SEASON_WEIGHT_HALF_LIFE = 2.0  # in seasons; 2 seasons ago gets weight 0.5


# =============================================================================
# Helpers
# =============================================================================

def compute_season_weights(seasons):
    """Compute exponential decay sample weights by season distance.

    More recent seasons weighted higher (roster/style relevance).
    Returns array of weights, one per row.
    """
    season_ordinals = seasons.apply(lambda s: int(str(s)[:4]))
    max_ordinal = season_ordinals.max()
    seasons_ago = max_ordinal - season_ordinals
    return np.power(0.5, seasons_ago / SEASON_WEIGHT_HALF_LIFE)


def pinball_loss(y_true, y_pred, q):
    """Quantile (pinball) loss for a single quantile."""
    residual = y_true - y_pred
    return np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual))


def mean_pinball_loss(y_true, quantile_preds, quantiles):
    """Mean pinball loss across all quantiles."""
    return np.mean([pinball_loss(y_true, quantile_preds[q], q) for q in quantiles])


def enforce_monotonicity(quantile_preds, quantiles):
    """Enforce q_lo <= q_hi for all adjacent quantile pairs."""
    sorted_qs = sorted(quantiles)
    adjusted = {sorted_qs[0]: quantile_preds[sorted_qs[0]].copy()}
    for i in range(1, len(sorted_qs)):
        adjusted[sorted_qs[i]] = np.maximum(
            quantile_preds[sorted_qs[i]], adjusted[sorted_qs[i - 1]]
        )
    return adjusted


# =============================================================================
# Conformal Calibration
# =============================================================================

class ConformalCalibrator:
    """
    Conformal calibration for quantile predictions.

    For each quantile q, finds an additive adjustment c such that
    P(y < predicted_q + c) ≈ q on the calibration set.
    """

    def __init__(self):
        self.adjustments = {}

    def fit(self, y_true, quantile_preds):
        for q, preds in quantile_preds.items():
            residuals = y_true - preds
            self.adjustments[q] = np.quantile(residuals, q)

        print(f"\n  Conformal calibration adjustments:")
        for q in sorted(self.adjustments.keys()):
            raw_coverage = (y_true <= quantile_preds[q]).mean()
            adj_coverage = (y_true <= quantile_preds[q] + self.adjustments[q]).mean()
            print(f"    q={q:.2f}: raw_cov={raw_coverage:.3f}, "
                  f"adj={self.adjustments[q]:+.3f}, adj_cov={adj_coverage:.3f}")

    def transform(self, quantile_preds):
        adjusted = {}
        for q, preds in quantile_preds.items():
            adjusted[q] = preds + self.adjustments.get(q, 0)
        return adjusted


# =============================================================================
# Feature Selection
# =============================================================================

def select_features(X_train, y_train, X_val, y_val, feature_cols,
                    n_supplementary=None, sample_weight=None):
    """
    Core + supplementary feature selection.

    Core features (CORE_FEATURES) are always included — they form the
    scoring/defense backbone.  Permutation importance then ranks all
    remaining features and selects the top N_SUPPLEMENTARY that have
    positive importance.  This prevents new/experimental features from
    displacing fundamentals while still allowing them to add signal.
    """
    if n_supplementary is None:
        n_supplementary = N_SUPPLEMENTARY

    print(f"\n{'='*60}")
    print("PHASE 1: Feature Selection (core + supplementary)")
    print(f"{'='*60}")

    # --- Identify core features present in this dataset ---
    core_present = [f for f in CORE_FEATURES if f in feature_cols]
    core_missing = [f for f in CORE_FEATURES if f not in feature_cols]
    if core_missing:
        print(f"  WARNING: {len(core_missing)} core features missing from data: "
              f"{core_missing}")
    print(f"  Core features locked in: {len(core_present)}")

    # --- Non-core candidates ---
    core_set = set(core_present)
    candidate_cols = [f for f in feature_cols if f not in core_set]
    print(f"  Supplementary candidates: {len(candidate_cols)}")

    # Jitter for training
    rng = np.random.RandomState(42)
    y_jittered = y_train + rng.uniform(-JITTER_SCALE, JITTER_SCALE, len(y_train))

    # Train median model with conservative params (uses ALL features)
    params = FEATURE_SELECT_PARAMS.copy()
    params["objective"] = "reg:quantileerror"
    params["quantile_alpha"] = 0.5
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_jittered, sample_weight=sample_weight, verbose=False)

    val_mae = mean_absolute_error(y_val, model.predict(X_val))
    print(f"  Preliminary median model MAE (val): {val_mae:.3f}")

    # Permutation importance on validation set
    print(f"  Computing permutation importance ({PERM_IMP_REPEATS} repeats)...")
    result = permutation_importance(
        model, X_val, y_val,
        n_repeats=PERM_IMP_REPEATS,
        random_state=42,
        scoring="neg_mean_absolute_error"
    )

    imp_means = result.importances_mean
    imp_stds = result.importances_std

    # --- Print core feature importances ---
    print(f"\n  Core features ({len(core_present)}):")
    core_indices = [feature_cols.index(f) for f in core_present]
    core_sorted = sorted(core_indices, key=lambda i: imp_means[i], reverse=True)
    for rank, idx in enumerate(core_sorted):
        snr = imp_means[idx] / imp_stds[idx] if imp_stds[idx] > 0 else float("inf")
        tag = " *" if imp_means[idx] <= 0 else ""
        print(f"    {rank+1:3d}. {feature_cols[idx]:35s}  "
              f"imp={imp_means[idx]:+.4f} +/- {imp_stds[idx]:.4f}  SNR={snr:.1f}{tag}")

    # --- Select top supplementary features ---
    cand_indices = [feature_cols.index(f) for f in candidate_cols]
    cand_sorted = sorted(cand_indices, key=lambda i: imp_means[i], reverse=True)

    # Take top N with positive importance
    supp_idx = [i for i in cand_sorted[:n_supplementary] if imp_means[i] > 0]
    supp_features = [feature_cols[i] for i in supp_idx]

    print(f"\n  Supplementary features selected ({len(supp_features)}/{len(candidate_cols)}, "
          f"top {n_supplementary} by importance):")
    for rank, idx in enumerate(supp_idx):
        snr = imp_means[idx] / imp_stds[idx] if imp_stds[idx] > 0 else float("inf")
        print(f"    {rank+1:3d}. {feature_cols[idx]:35s}  "
              f"imp={imp_means[idx]:+.4f} +/- {imp_stds[idx]:.4f}  SNR={snr:.1f}")

    # --- Combine ---
    selected_features = core_present + supp_features
    print(f"\n  Total: {len(core_present)} core + {len(supp_features)} supplementary "
          f"= {len(selected_features)} features")

    # Show dropped supplementary
    n_cand_dropped = len(candidate_cols) - len(supp_features)
    n_neg = sum(1 for i in cand_indices if imp_means[i] < 0)
    print(f"  Dropped {n_cand_dropped} supplementary candidates "
          f"({n_neg} negative importance)")

    return selected_features


# =============================================================================
# Hyperparameter Tuning
# =============================================================================

def tune_hyperparams(X_train, y_train, X_val, y_val, n_trials=50, sample_weight=None):
    """
    Optuna tuning of XGBoost params.
    Optimizes mean pinball loss across all 13 quantiles on validation set.
    """
    print(f"\n{'='*60}")
    print("PHASE 2: Hyperparameter Tuning")
    print(f"{'='*60}")

    rng = np.random.RandomState(42)
    y_jittered = y_train + rng.uniform(-JITTER_SCALE, JITTER_SCALE, len(y_train))

    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 5),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 200, 800, step=100),
            "min_child_weight": trial.suggest_int("min_child_weight", 30, 200),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 10.0, log=True),
            "tree_method": "hist",
            "random_state": 42,
            "verbosity": 0,
        }

        # Train all 13 quantile models
        preds = {}
        for q in QUANTILES:
            p = params.copy()
            p["objective"] = "reg:quantileerror"
            p["quantile_alpha"] = q
            model = xgb.XGBRegressor(**p)
            model.fit(X_train, y_jittered, sample_weight=sample_weight, verbose=False)
            preds[q] = model.predict(X_val)

        return mean_pinball_loss(y_val, preds, QUANTILES)

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    print(f"  Running {n_trials} trials (13 quantile models each)...")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best mean pinball loss: {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in sorted(best.items()):
        print(f"    {k:25s}: {v}")

    # Add fixed params
    best["tree_method"] = "hist"
    best["random_state"] = 42
    best["verbosity"] = 0

    return best


# =============================================================================
# Model Training
# =============================================================================

def train_regulation_model(train_cutoff="2025-10-01"):
    cutoff_year = train_cutoff[:4]
    input_file = os.path.join(PROCESSED_DIR, f"regulation_features_{cutoff_year}.csv")
    model_file = os.path.join(MODEL_DIR, f"regulation_model_{cutoff_year}.pkl")
    output_file = os.path.join(PROCESSED_DIR, f"regulation_predictions_{cutoff_year}.csv")

    print(f"\n{'#'*60}")
    print(f"  Train cutoff: {train_cutoff}")
    print(f"  Input file:   {input_file}")
    print(f"  Model file:   {model_file}")
    print(f"  Output file:  {output_file}")
    print(f"{'#'*60}")

    print("Loading regulation features...")
    df = pd.read_csv(input_file, low_memory=False)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Total rows: {len(df):,}")

    # --- Exclude unplayed (FUT) games ---
    game_ids = pd.read_csv(GAME_IDS_FILE, usecols=["game_id", "game_state"])
    df = df.merge(game_ids, on="game_id", how="left")
    n_fut = (df["game_state"] == "FUT").sum()
    df = df[df["game_state"] != "FUT"].copy()
    df.drop(columns=["game_state"], inplace=True)
    print(f"  Excluded {n_fut} unplayed (FUT) game rows")
    print(f"  Completed rows: {len(df):,}")

    # --- Identify feature columns ---
    all_feature_cols = [c for c in df.columns
                        if c not in ID_COLS + [TARGET_COL] + RAW_COLS]
    print(f"  Feature columns: {len(all_feature_cols)}")

    # --- Drop rows with too many NaN features (early season) ---
    min_features = len(all_feature_cols) * 0.5
    valid_mask = df[all_feature_cols].notna().sum(axis=1) >= min_features
    df_valid = df[valid_mask].copy()
    print(f"  After dropping sparse rows: {len(df_valid):,}")

    # --- Temporal split (3-way: tune_train / tune_val / test) ---
    # Conformal holdout removed — raw model is well-calibrated (+0.014 lambda error).
    # Those 472 rows now go back into final training.
    train_pool = df_valid[df_valid["game_date"] < train_cutoff].copy()
    test = df_valid[df_valid["game_date"] >= train_cutoff].copy()

    # Tuning validation: last 90 days of training pool
    tune_val_cutoff = train_pool["game_date"].max() - pd.Timedelta(days=TUNE_HOLDOUT_DAYS)
    tune_val = train_pool[train_pool["game_date"] >= tune_val_cutoff].copy()

    # Tuning train: everything before tune_val
    tune_train = train_pool[train_pool["game_date"] < tune_val_cutoff].copy()

    # Final train = entire training pool (no conformal holdout)
    final_train = train_pool.copy()

    print(f"\n  Tune train:  {len(tune_train):,} rows "
          f"({tune_train['game_date'].min().date()} to {tune_train['game_date'].max().date()})")
    print(f"  Tune val:    {len(tune_val):,} rows "
          f"({tune_val['game_date'].min().date()} to {tune_val['game_date'].max().date()})")
    print(f"  Final train: {len(final_train):,} rows "
          f"({final_train['game_date'].min().date()} to {final_train['game_date'].max().date()})")
    print(f"  Test (OOS):  {len(test):,} rows "
          f"({test['game_date'].min().date()} to {test['game_date'].max().date()})")

    print(f"\n  Target distribution (tune_train):")
    y_tune_train = tune_train[TARGET_COL].values
    for k in range(8):
        pct = (y_tune_train == k).mean() * 100
        print(f"    {k} goals: {pct:.1f}%")

    # --- Compute season-based sample weights ---
    sample_weights_tune = compute_season_weights(tune_train["season"])
    sample_weights_final = compute_season_weights(final_train["season"])

    print(f"\n  Sample weights by season:")
    for s in sorted(final_train["season"].unique()):
        ordinal = int(str(s)[:4])
        max_ordinal = int(str(final_train["season"].max())[:4])
        w = 0.5 ** ((max_ordinal - ordinal) / SEASON_WEIGHT_HALF_LIFE)
        n = (final_train["season"] == s).sum()
        print(f"    {s}: weight={w:.3f}, n={n}")

    # =================================================================
    # PHASE 1: Feature Selection
    # =================================================================
    if LOCKED_FEATURES is not None:
        print(f"\n{'='*60}")
        print("PHASE 1: Using LOCKED Features (permutation importance skipped)")
        print(f"{'='*60}")
        # Verify all locked features exist in the data
        missing = [f for f in LOCKED_FEATURES if f not in all_feature_cols]
        if missing:
            print(f"  WARNING: {len(missing)} locked features missing from data: {missing}")
        selected_features = [f for f in LOCKED_FEATURES if f in all_feature_cols]
        print(f"  Locked features: {len(selected_features)}")
    else:
        selected_features = select_features(
            tune_train[all_feature_cols], y_tune_train,
            tune_val[all_feature_cols], tune_val[TARGET_COL].values,
            all_feature_cols,
            sample_weight=sample_weights_tune.values,
        )

    # =================================================================
    # PHASE 2: Hyperparameter Tuning (on selected features only)
    # =================================================================
    if LOCKED_PARAMS is not None:
        print(f"\n{'='*60}")
        print("PHASE 2: Using LOCKED Hyperparameters (Optuna skipped)")
        print(f"{'='*60}")
        best_params = LOCKED_PARAMS.copy()
        print(f"  Params:")
        for k, v in sorted(best_params.items()):
            print(f"    {k:25s}: {v}")
    else:
        best_params = tune_hyperparams(
            tune_train[selected_features], y_tune_train,
            tune_val[selected_features], tune_val[TARGET_COL].values,
            n_trials=N_OPTUNA_TRIALS,
            sample_weight=sample_weights_tune.values,
        )

    # =================================================================
    # PHASE 3: Final Training
    # =================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: Final Training")
    print(f"{'='*60}")

    X_train = final_train[selected_features]
    y_train = final_train[TARGET_COL].values
    X_test = test[selected_features]
    y_test = test[TARGET_COL].values

    print(f"  Training on {len(final_train):,} rows with {len(selected_features)} features")
    print(f"  Using optimized params: depth={best_params['max_depth']}, "
          f"lr={best_params['learning_rate']:.3f}, "
          f"n_est={best_params['n_estimators']}, "
          f"mcw={best_params['min_child_weight']}")

    # Jitter targets
    rng = np.random.RandomState(42)
    y_train_jittered = y_train + rng.uniform(-JITTER_SCALE, JITTER_SCALE, len(y_train))

    # Train 13 quantile models
    print(f"\n  Training {len(QUANTILES)} quantile XGBoost models...")
    models = {}
    for q in QUANTILES:
        params = best_params.copy()
        params["objective"] = "reg:quantileerror"
        params["quantile_alpha"] = q
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train_jittered, sample_weight=sample_weights_final.values, verbose=False)
        models[q] = model
        print(f"    q={q:.2f} trained")

    # --- Conformal calibration DISABLED ---
    # Raw model lambda error is +0.014 goals (well calibrated).
    # Conformal was adding +0.132 to lambda via upper-tail inflation (q70-q95).
    # This inflated lambdas by ~5%, creating artificial underdog bias via Poisson convexity.
    calibrator = ConformalCalibrator()  # Identity (no adjustments)

    # =================================================================
    # PHASE 4: OOS Evaluation
    # =================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Test Set (OOS) Evaluation")
    print(f"{'='*60}")

    if len(test) > 0:
        test_preds_raw = {q: model.predict(X_test) for q, model in models.items()}
        test_preds = enforce_monotonicity(test_preds_raw, QUANTILES)

        # Median (q=0.50) performance
        y_median = test_preds[0.50]
        mae = mean_absolute_error(y_test, y_median)
        rmse = np.sqrt(mean_squared_error(y_test, y_median))
        corr = np.corrcoef(y_test, y_median)[0, 1]
        print(f"  Median (q=0.50):")
        print(f"    MAE:  {mae:.3f}")
        print(f"    RMSE: {rmse:.3f}")
        print(f"    Corr: {corr:.3f}")

        # Naive baseline
        naive_pred = y_train.mean()
        naive_mae = mean_absolute_error(y_test, np.full_like(y_test, naive_pred, dtype=float))
        print(f"\n  Naive baseline (predict {naive_pred:.2f}):")
        print(f"    MAE:  {naive_mae:.3f}")
        print(f"  Model improvement: {(1 - mae / naive_mae) * 100:.1f}%")

        # Mean pinball loss
        mpl = mean_pinball_loss(y_test, test_preds, QUANTILES)
        print(f"\n  Mean pinball loss (OOS): {mpl:.4f}")

        # Quantile calibration check
        print(f"\n  Quantile coverage (target: q value):")
        for q in QUANTILES:
            coverage = (y_test <= test_preds[q]).mean()
            diff = coverage - q
            flag = " *" if abs(diff) > 0.05 else ""
            print(f"    q={q:.2f}: coverage={coverage:.3f} (diff={diff:+.3f}){flag}")

        # Monotonicity check
        violations = 0
        for i in range(len(QUANTILES) - 1):
            q_lo, q_hi = QUANTILES[i], QUANTILES[i + 1]
            violations += (test_preds[q_hi] < test_preds[q_lo]).sum()
        total_pairs = len(test) * (len(QUANTILES) - 1)
        print(f"\n  Quantile crossing violations: {violations}/{total_pairs} "
              f"({violations / total_pairs * 100:.2f}%)")

        # Prediction spread
        spread = test_preds[0.90] - test_preds[0.10]
        print(f"\n  80% prediction interval (q10-q90):")
        print(f"    Mean width: {spread.mean():.2f} goals")
        print(f"    Range: [{spread.min():.2f}, {spread.max():.2f}]")

    # --- Feature importance (from median model) ---
    print(f"\n  Top 15 feature importance (gain, median model):")
    importance = models[0.50].get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:15]
    for feat, gain in sorted_imp:
        if feat.startswith("f") and feat[1:].isdigit():
            feat_idx = int(feat[1:])
            feat_name = selected_features[feat_idx] if feat_idx < len(selected_features) else feat
        else:
            feat_name = feat
        print(f"    {feat_name:35s}: {gain:.1f}")

    # --- Save model ---
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    artifact = {
        "models": models,
        "calibrator": calibrator,
        "feature_cols": selected_features,
        "quantiles": QUANTILES,
        "best_params": best_params,
        "train_cutoff": train_cutoff,
        "train_date_range": (str(final_train["game_date"].min().date()),
                             str(final_train["game_date"].max().date())),
    }
    with open(model_file, "wb") as f:
        pickle.dump(artifact, f)
    print("  Model saved.")

    # --- Generate predictions for ALL valid rows ---
    print(f"\nGenerating predictions for all rows...")
    X_all = df_valid[selected_features]
    all_preds_raw = {q: model.predict(X_all) for q, model in models.items()}
    all_preds = enforce_monotonicity(all_preds_raw, QUANTILES)

    output = df_valid[["game_id", "game_date", "team", "opponent",
                        "is_home", "season", TARGET_COL]].copy()
    for q in QUANTILES:
        output[f"q{int(q*100):02d}"] = all_preds[q]

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.to_csv(output_file, index=False)
    print(f"  Saved {len(output):,} predictions to {output_file}")

    print(f"\n--- Prediction Summary ---")
    print(f"  Median prediction mean: {output['q50'].mean():.2f} "
          f"(actual: {output[TARGET_COL].mean():.2f})")
    print(f"  q10 mean: {output['q10'].mean():.2f}, "
          f"q90 mean: {output['q90'].mean():.2f}")

    return models, calibrator


# =============================================================================
# Prediction Helper
# =============================================================================

def load_regulation_model(cutoff_year="2025"):
    """Load the trained regulation model for a given cutoff year."""
    model_file = os.path.join(MODEL_DIR, f"regulation_model_{cutoff_year}.pkl")
    with open(model_file, "rb") as f:
        artifact = pickle.load(f)
    return (artifact["models"], artifact["calibrator"],
            artifact["feature_cols"], artifact["quantiles"])


def predict_regulation_quantiles(features_df, models=None, calibrator=None):
    """
    Predict regulation goal quantiles for a features DataFrame.
    Returns dict of {quantile: predictions}.
    """
    if models is None or calibrator is None:
        models, calibrator, feature_cols, quantiles = load_regulation_model()
    else:
        feature_cols = list(models.values())[0].get_booster().feature_names
        quantiles = sorted(models.keys())

    X = features_df[feature_cols]
    raw_preds = {q: model.predict(X) for q, model in models.items()}
    calibrated = calibrator.transform(raw_preds)
    return enforce_monotonicity(calibrated, quantiles)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    # Accept cutoffs from command line, or use TRAIN_CUTOFFS
    cutoffs = sys.argv[1:] if len(sys.argv) > 1 else TRAIN_CUTOFFS

    for cutoff in cutoffs:
        print("=" * 60)
        print(f"Regulation Model Training (Layer 1) — cutoff {cutoff}")
        print("=" * 60)

        train_regulation_model(train_cutoff=cutoff)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
