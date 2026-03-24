# -*- coding: utf-8 -*-
"""
Expected Goals (xG) Model

XGBoost binary classifier that assigns P(goal) to every unblocked shot
based purely on shot geometry and context. No shooter/goalie identity.

This is a STATIC model — trained once on historical data, not retrained
in the walk-forward backtest. Shot physics don't change season to season.

Key lessons from SOG model applied here:
  1. Conservative hyperparameters (shallow trees, high min_child_weight)
  2. Platt calibration on OOS holdout (don't trust raw probabilities)
  3. Proper train/OOS split (temporal, not random)
  4. Class imbalance handling via scale_pos_weight
  5. Comprehensive validation (AUC, calibration curve, feature importance)

Outputs:
  - model_artifacts/xg_model.pkl       : Trained model + calibration params
  - data/processed/shot_xg.csv         : xG predictions for ALL shots

@author: chazf
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import (roc_auc_score, brier_score_loss,
                              log_loss, confusion_matrix)
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
import pickle
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# =============================================================================
# Configuration
# =============================================================================
from config import PROCESSED_DIR, MODEL_DIR

INPUT_FILE = os.path.join(PROCESSED_DIR, "xg_training_data.csv")

# Train cutoffs to generate OOS predictions for each test season
TRAIN_CUTOFFS = ["2024-10-01", "2025-10-01"]

# Features used by the model (no identifiers, no target)
FEATURE_COLS = [
    "shot_distance",
    "shot_angle",
    "x_norm",
    "y_abs",
    "shot_type_code",
    "is_rebound",
    "time_since_last_shot",
    "strength_code",
    "is_empty_net",
    # score_diff REMOVED — NHL API only records scores on goal events,
    # so non-goals have NaN → filled with 0 → trivial leakage.
    # Can add back later with properly computed running game score.
    "game_seconds",
    "is_ot",
    "time_remaining_sec",
]

TARGET_COL = "is_goal"

# Within training, hold out last 30 days for Platt calibration
CAL_HOLDOUT_DAYS = 30

# XGBoost hyperparameters (conservative — avoid overfitting)
XGB_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 4,               # slightly deeper than SOG (3) — more data
    "learning_rate": 0.03,
    "n_estimators": 500,
    "min_child_weight": 80,       # high — prevents overfitting to small leaf nodes
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "reg_alpha": 0.1,             # L1 regularization
    "reg_lambda": 1.0,            # L2 regularization
    "tree_method": "hist",
    "random_state": 42,
    "verbosity": 0,
}


# =============================================================================
# Platt Calibration
# =============================================================================

class PlattCalibrator:
    """
    Platt scaling: calibrated_p = sigmoid(a * logit(p) + b)

    Fits two parameters (a, b) on OOS data to correct any
    systematic bias in probability estimates.

    From SOG model: a < 1 softens overconfident predictions,
    b shifts the overall level up/down.
    """

    def __init__(self):
        self.a = 1.0  # slope (1.0 = no change)
        self.b = 0.0  # intercept (0.0 = no change)

    def fit(self, y_true, y_pred_raw):
        """
        Fit Platt parameters on calibration holdout.
        Minimizes log-loss of calibrated probabilities.
        """
        # Clip predictions to avoid log(0)
        p = np.clip(y_pred_raw, 1e-6, 1 - 1e-6)
        logit_p = logit(p)
        y = np.array(y_true, dtype=float)

        def neg_log_loss(params):
            a, b = params
            calibrated = expit(a * logit_p + b)
            calibrated = np.clip(calibrated, 1e-6, 1 - 1e-6)
            return -np.mean(y * np.log(calibrated) +
                            (1 - y) * np.log(1 - calibrated))

        # Grid search for good starting point, then optimize
        best_loss = np.inf
        best_params = (1.0, 0.0)

        for a_init in [0.7, 0.8, 0.9, 1.0, 1.1]:
            for b_init in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                from scipy.optimize import minimize
                result = minimize(neg_log_loss, [a_init, b_init],
                                  method="Nelder-Mead",
                                  options={"maxiter": 500})
                if result.fun < best_loss:
                    best_loss = result.fun
                    best_params = result.x

        self.a, self.b = best_params
        print(f"  Platt params: a={self.a:.4f}, b={self.b:.4f}")

        # Show calibration improvement
        raw_ll = log_loss(y, y_pred_raw)
        cal_p = self.transform(y_pred_raw)
        cal_ll = log_loss(y, cal_p)
        print(f"  Log-loss: raw={raw_ll:.5f} -> calibrated={cal_ll:.5f}")

    def transform(self, y_pred_raw):
        """Apply Platt calibration."""
        p = np.clip(y_pred_raw, 1e-6, 1 - 1e-6)
        return expit(self.a * logit(p) + self.b)


# =============================================================================
# Model Training
# =============================================================================

def train_xg_model(train_cutoff="2025-10-01"):
    """
    Train the xG model with proper temporal split and calibration.
    """
    cutoff_year = train_cutoff[:4]
    model_file = os.path.join(MODEL_DIR, f"xg_model_{cutoff_year}.pkl")
    output_file = os.path.join(PROCESSED_DIR, f"shot_xg_{cutoff_year}.csv")

    print(f"\n{'#'*60}")
    print(f"  Train cutoff: {train_cutoff}")
    print(f"  Model file:   {model_file}")
    print(f"  Output file:  {output_file}")
    print(f"{'#'*60}")

    print("Loading xG training data...")
    df = pd.read_csv(INPUT_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])
    print(f"  Total shots: {len(df):,}")

    # ----- Handle NaNs in features -----
    # time_remaining_sec can be NaN for some events
    df["time_remaining_sec"] = df["time_remaining_sec"].fillna(600.0)
    # time_since_last_shot: default is 60 (set in feature engineering)
    df["time_since_last_shot"] = df["time_since_last_shot"].fillna(60.0)

    # ----- Temporal split -----
    train_pool = df[df["game_date"] < train_cutoff].copy()
    test = df[df["game_date"] >= train_cutoff].copy()

    # Within train pool, hold out last 30 days for Platt calibration
    cal_cutoff = train_pool["game_date"].max() - pd.Timedelta(days=CAL_HOLDOUT_DAYS)
    train = train_pool[train_pool["game_date"] < cal_cutoff].copy()
    cal = train_pool[train_pool["game_date"] >= cal_cutoff].copy()

    print(f"\n  Training:    {len(train):,} shots "
          f"({train['game_date'].min().date()} to {train['game_date'].max().date()})")
    print(f"  Calibration: {len(cal):,} shots "
          f"({cal['game_date'].min().date()} to {cal['game_date'].max().date()})")
    print(f"  Test (OOS):  {len(test):,} shots "
          f"({test['game_date'].min().date()} to {test['game_date'].max().date()})")

    # ----- Class balance -----
    goal_rate = train[TARGET_COL].mean()
    scale_pos = (1 - goal_rate) / goal_rate
    print(f"\n  Training goal rate: {goal_rate*100:.1f}%")
    print(f"  scale_pos_weight: {scale_pos:.2f}")

    # ----- Train XGBoost -----
    print("\nTraining XGBoost...")
    params = XGB_PARAMS.copy()
    params["scale_pos_weight"] = scale_pos

    X_train = train[FEATURE_COLS]
    y_train = train[TARGET_COL]

    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train, verbose=False)

    # ----- Platt calibration on holdout -----
    print("\nCalibrating on holdout...")
    X_cal = cal[FEATURE_COLS]
    y_cal = cal[TARGET_COL]
    y_cal_pred_raw = model.predict_proba(X_cal)[:, 1]

    calibrator = PlattCalibrator()
    calibrator.fit(y_cal, y_cal_pred_raw)

    # ----- Evaluate on test set -----
    print("\n--- Test Set (OOS) Evaluation ---")
    if len(test) > 0:
        X_test = test[FEATURE_COLS]
        y_test = test[TARGET_COL]
        y_test_raw = model.predict_proba(X_test)[:, 1]
        y_test_cal = calibrator.transform(y_test_raw)

        auc = roc_auc_score(y_test, y_test_cal)
        brier = brier_score_loss(y_test, y_test_cal)
        ll = log_loss(y_test, y_test_cal)
        print(f"  AUC-ROC:     {auc:.4f}")
        print(f"  Brier Score: {brier:.5f}")
        print(f"  Log-Loss:    {ll:.5f}")

        # Calibration by decile
        print(f"\n  Calibration by decile:")
        test_with_pred = test.copy()
        test_with_pred["xg"] = y_test_cal
        test_with_pred["decile"] = pd.qcut(test_with_pred["xg"], 10,
                                            labels=False, duplicates="drop")
        cal_table = test_with_pred.groupby("decile").agg(
            mean_xg=("xg", "mean"),
            actual_rate=("is_goal", "mean"),
            n_shots=("is_goal", "count"),
        ).reset_index()
        for _, row in cal_table.iterrows():
            diff = row["actual_rate"] - row["mean_xg"]
            print(f"    Decile {int(row['decile'])}: "
                  f"pred={row['mean_xg']:.3f}, "
                  f"actual={row['actual_rate']:.3f}, "
                  f"diff={diff:+.3f}, "
                  f"n={int(row['n_shots']):,}")
    else:
        print("  No test data available (2025-26 games not yet scraped)")

    # ----- Feature importance -----
    print(f"\n  Feature importance (gain):")
    importance = model.get_booster().get_score(importance_type="gain")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feat, gain in sorted_imp:
        # XGBoost may use feature names directly or f0/f1/... format
        if feat.startswith("f") and feat[1:].isdigit():
            feat_idx = int(feat[1:])
            feat_name = FEATURE_COLS[feat_idx] if feat_idx < len(FEATURE_COLS) else feat
        else:
            feat_name = feat
        print(f"    {feat_name:25s}: {gain:.1f}")

    # ----- Evaluate on calibration set (in-sample check) -----
    print(f"\n--- Calibration Set Check ---")
    y_cal_cal = calibrator.transform(y_cal_pred_raw)
    auc_cal = roc_auc_score(y_cal, y_cal_cal)
    print(f"  AUC: {auc_cal:.4f} (should be similar to test AUC)")

    # ----- Save model -----
    print(f"\nSaving model to {model_file}...")
    os.makedirs(os.path.dirname(model_file), exist_ok=True)
    artifact = {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": FEATURE_COLS,
        "xgb_params": params,
        "train_cutoff": train_cutoff,
        "train_date_range": (str(train["game_date"].min().date()),
                             str(train["game_date"].max().date())),
    }
    with open(model_file, "wb") as f:
        pickle.dump(artifact, f)
    print("  Model saved.")

    # ----- Generate xG for ALL shots -----
    print(f"\nGenerating xG predictions for all {len(df):,} shots...")
    X_all = df[FEATURE_COLS]
    y_all_raw = model.predict_proba(X_all)[:, 1]
    y_all_cal = calibrator.transform(y_all_raw)

    # Build output: identifiers + xG
    output = df[["game_id", "game_date", "event_id", "period",
                  "event_type", "shooter_id", "goalie_id",
                  "is_home_event", "away_team", "home_team",
                  "is_goal", "is_empty_net"]].copy()
    output["xg"] = y_all_cal

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    output.to_csv(output_file, index=False)
    print(f"  Saved {len(output):,} shots with xG to {output_file}")

    # ----- Summary stats -----
    print(f"\n--- xG Summary ---")
    print(f"  Mean xG:  {output['xg'].mean():.4f} "
          f"(actual goal rate: {output['is_goal'].mean():.4f})")
    print(f"  xG range: [{output['xg'].min():.4f}, {output['xg'].max():.4f}]")

    # Per-game xG totals (sanity: should average ~2.8-3.0 per team)
    game_xg = output.groupby(["game_id", "is_home_event"]).agg(
        total_xg=("xg", "sum"),
        total_goals=("is_goal", "sum"),
        n_shots=("xg", "count"),
    )
    print(f"  Mean xG per team per game: {game_xg['total_xg'].mean():.2f}")
    print(f"  Mean goals per team per game: {game_xg['total_goals'].mean():.2f}")

    return model, calibrator


# =============================================================================
# Prediction Helper (for use by other modules)
# =============================================================================

def load_xg_model(cutoff_year="2025"):
    """Load the trained xG model and calibrator for a given cutoff year."""
    model_file = os.path.join(MODEL_DIR, f"xg_model_{cutoff_year}.pkl")
    with open(model_file, "rb") as f:
        artifact = pickle.load(f)
    return artifact["model"], artifact["calibrator"], artifact["feature_cols"]


def predict_xg(shots_df, model=None, calibrator=None):
    """
    Predict xG for a DataFrame of shots with the required feature columns.
    If model/calibrator not provided, loads from disk.
    """
    if model is None or calibrator is None:
        model, calibrator, feature_cols = load_xg_model()
    else:
        feature_cols = FEATURE_COLS

    X = shots_df[feature_cols]
    y_raw = model.predict_proba(X)[:, 1]
    return calibrator.transform(y_raw)


# =============================================================================
# Run
# =============================================================================

if __name__ == "__main__":
    for cutoff in TRAIN_CUTOFFS:
        print("=" * 60)
        print(f"xG Model Training — cutoff {cutoff}")
        print("=" * 60)

        train_xg_model(train_cutoff=cutoff)

        print(f"\n{'='*60}")
        print("DONE")
        print(f"{'='*60}")
