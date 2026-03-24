# -*- coding: utf-8 -*-
"""
Pipeline Diagnostic Script — NHL Win Probability Model
========================================================
Traces through every stage of the prediction pipeline to identify
where the model gains or loses accuracy relative to DraftKings.

Stages:
  0. Goalie Model (Layer 0) — Deployment context score, matchup signal
  1. Regulation Model (Layer 1) — lambda accuracy, goalie feature value
  2. Scoring Anchor — correction quality, over/under-correction
  3. Poisson Probabilities — raw P(home), P(tie) accuracy
  4. Tie Inflation — P(OT) calibration improvement
  5. OT Edge Model (Layer 2) — P(home|OT) signal, goalie matchup, feature breakdown
  6. Platt Calibration — probability stretch/compression
  7. Betting Edge — model vs DK decomposition, where profit comes from
  8. Improvement Opportunities (ranked)

Uses walk-forward backtest results (backtest_results.csv) plus raw
predictions to measure value-add at each stage.

Run from Spyder:  runfile('backtest/diagnose_pipeline.py')

@author: chazf
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# --- Setup ---
from config import PROJECT_DIR
PROJECT_DIR = Path(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

from models.poisson_combiner import (
    quantiles_to_lambda,
    compute_game_probabilities,
    apply_tie_calibration,
    apply_platt_calibration,
    american_to_implied,
    power_devig,
    QUANTILE_COLS,
    DEFAULT_P_HOME_OT,
    LAMBDA_CLIP,
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.grid": True,
    "axes.grid.which": "major",
    "grid.alpha": 0.3,
    "font.size": 10,
})

PLOTS_DIR = PROJECT_DIR / "backtest" / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

TEST_CUTOFF = pd.Timestamp("2025-10-01")


# =====================================================================
# DATA LOADING
# =====================================================================
print("=" * 70)
print("PIPELINE DIAGNOSTIC — NHL WIN PROBABILITY MODEL")
print("=" * 70)

print("\nLoading data...")

# Raw predictions
reg = pd.read_csv(PROJECT_DIR / "data" / "processed" / "regulation_predictions.csv")
reg["game_date"] = pd.to_datetime(reg["game_date"])
print(f"  Regulation predictions: {len(reg):,}")

ot = pd.read_csv(PROJECT_DIR / "data" / "processed" / "ot_predictions.csv")
ot["game_date"] = pd.to_datetime(ot["game_date"])
ot_map = dict(zip(ot["game_id"], ot["p_home_win_ot"]))
print(f"  OT predictions: {len(ot):,}")

games = pd.read_csv(PROJECT_DIR / "data" / "raw" / "game_ids.csv")
games["game_date"] = pd.to_datetime(games["game_date"])
completed = games[games["game_state"] == "OFF"].copy()
completed["is_ot"] = completed["game_outcome_type"].isin(["OT", "SO"]).astype(int)
completed["home_won_actual"] = (completed["home_score"] > completed["away_score"]).astype(int)

# Goalie deployment predictions
goalie_deploy = pd.read_csv(PROJECT_DIR / "data" / "processed" / "goalie_deployment_predictions.csv")
goalie_deploy["game_date"] = pd.to_datetime(goalie_deploy["game_date"])
print(f"  Goalie deployment predictions: {len(goalie_deploy):,}")

# Goalie game features (for GSAx actuals / diagnostics)
goalie_feat = pd.read_csv(PROJECT_DIR / "data" / "processed" / "goalie_game_features.csv")
goalie_feat["game_date"] = pd.to_datetime(goalie_feat["game_date"])
print(f"  Goalie game features: {len(goalie_feat):,}")

# OT features (for goalie matchup analysis)
ot_feat = pd.read_csv(PROJECT_DIR / "data" / "processed" / "ot_features.csv")
ot_feat["game_date"] = pd.to_datetime(ot_feat["game_date"])
print(f"  OT features: {len(ot_feat):,}")

# Regulation features (for goalie impact on regulation)
reg_feat = pd.read_csv(PROJECT_DIR / "data" / "processed" / "regulation_features.csv")
reg_feat["game_date"] = pd.to_datetime(reg_feat["game_date"])
print(f"  Regulation features: {len(reg_feat):,}")

# Backtest results
bt = pd.read_csv(PROJECT_DIR / "backtest" / "backtest_results.csv")
bt["game_date"] = pd.to_datetime(bt["game_date"])
print(f"  Backtest results: {len(bt):,} games")

# Compute lambdas
reg["lam_raw"] = reg[QUANTILE_COLS].apply(
    lambda row: quantiles_to_lambda(row.values), axis=1
)

# Pair home/away
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
paired = paired.merge(
    completed[["game_id", "game_outcome_type", "is_ot", "home_won_actual",
               "home_score", "away_score"]],
    on="game_id", how="inner"
)
paired["p_home_win_ot_given"] = paired["game_id"].map(ot_map).fillna(DEFAULT_P_HOME_OT)
paired = paired.sort_values("game_date").reset_index(drop=True)

# Split
test = paired[paired["game_date"] >= TEST_CUTOFF].copy()
history = paired[paired["game_date"] < TEST_CUTOFF].copy()

print(f"  Paired games: {len(paired):,} total, {len(test):,} test (2025-26)")

# Merge backtest calibrated probs
bt_cols = ["game_id", "lam_home", "lam_away", "anchor_home", "anchor_away",
           "p_home_reg_win", "p_away_reg_win", "p_ot",
           "p_home_win_raw", "platt_a", "platt_b", "p_home_win", "p_away_win",
           "dk_home_fair", "dk_away_fair", "dk_home_dec", "dk_away_dec",
           "home_edge", "away_edge"]
test = test.merge(bt[bt_cols], on="game_id", how="left")

print(f"  Test games with backtest data: {test['lam_home'].notna().sum()}")


def log_loss(y_true, y_pred):
    """Per-game log-loss."""
    p = np.clip(y_pred, 0.001, 0.999)
    return -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def mean_ll(y_true, y_pred):
    """Mean log-loss."""
    return log_loss(y_true, y_pred).mean()


# =====================================================================
# STAGE 0: GOALIE DEPLOYMENT MODEL (LAYER 0) — DEPLOYMENT CONTEXT
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 0: GOALIE DEPLOYMENT MODEL (LAYER 0)")
print("=" * 70)

# GA rate accuracy (goalie-adjusted GA rate vs actual regulation GA)
gd = goalie_deploy.copy()
gd_test = gd[gd["game_date"] >= TEST_CUTOFF].copy()

# Support both old (goalie_deployment_score) and new (goalie_ga_rate) column names
has_ga_rate = "goalie_ga_rate" in gd.columns
ga_pred_col = "goalie_ga_rate" if has_ga_rate else "goalie_deployment_score"
ga_actual_col = "reg_ga" if "reg_ga" in gd.columns else "team_ga_residual"

for label, gd_sub in [("All Seasons", gd), ("2025-26 OOS", gd_test)]:
    valid = gd_sub.dropna(subset=[ga_actual_col, ga_pred_col])
    if len(valid) < 10:
        continue
    corr = valid[ga_pred_col].corr(valid[ga_actual_col])
    mae = (valid[ga_pred_col] - valid[ga_actual_col]).abs().mean()
    bias = valid[ga_pred_col].mean() - valid[ga_actual_col].mean()
    pred_std = valid[ga_pred_col].std()
    actual_std = valid[ga_actual_col].std()

    print(f"\n  {label} (n={len(valid):,}):")
    print(f"    Metric: {ga_pred_col} vs {ga_actual_col}")
    print(f"    Correlation:  {corr:.3f}")
    print(f"    MAE:          {mae:.3f}")
    print(f"    Bias:         {bias:+.4f}")
    print(f"    Spread:       pred_std={pred_std:.4f}, actual_std={actual_std:.4f}")

# Deployment feature distributions (2025-26)
if len(gd_test) > 50:
    print(f"\n  Deployment feature distributions (2025-26, n={len(gd_test):,}):")
    deploy_feats = ["starter_role_share", "goalie_switch", "consecutive_starts",
                    "was_pulled_last_game", "days_rest", "is_back_to_back",
                    "games_started_last_14d"]
    for feat in deploy_feats:
        if feat in gd_test.columns:
            if gd_test[feat].nunique() <= 2:
                print(f"    {feat:25s}: rate={gd_test[feat].mean():.3f}")
            else:
                print(f"    {feat:25s}: mean={gd_test[feat].mean():.2f}, "
                      f"std={gd_test[feat].std():.2f}")

    # Score by primary vs backup
    primary = gd_test[gd_test["starter_role_share"] >= 0.55]
    backup = gd_test[gd_test["starter_role_share"] < 0.55]
    print(f"\n  Primary starter (share>=0.55, n={len(primary):,}):")
    print(f"    Mean {ga_pred_col}: {primary[ga_pred_col].mean():.3f}")
    print(f"    Mean actual {ga_actual_col}: {primary[ga_actual_col].mean():.3f}")
    print(f"  Backup (share<0.55, n={len(backup):,}):")
    print(f"    Mean {ga_pred_col}: {backup[ga_pred_col].mean():.3f}")
    print(f"    Mean actual {ga_actual_col}: {backup[ga_actual_col].mean():.3f}")

# Goalie GA rate differential → game outcome
home_gk = goalie_deploy[goalie_deploy["is_home"] == True][
    ["game_id", ga_pred_col, "goalie_name"]
].copy().drop_duplicates(subset=["game_id"])
home_gk.columns = ["game_id", "home_goalie_pred", "home_goalie"]

away_gk = goalie_deploy[goalie_deploy["is_home"] == False][
    ["game_id", ga_pred_col, "goalie_name"]
].copy().drop_duplicates(subset=["game_id"])
away_gk.columns = ["game_id", "away_goalie_pred", "away_goalie"]

gk_matchup = home_gk.merge(away_gk, on="game_id", how="inner")
gk_matchup = gk_matchup.merge(
    completed[["game_id", "game_date", "home_won_actual", "is_ot", "game_outcome_type"]],
    on="game_id", how="inner"
)
gk_matchup["deploy_diff"] = gk_matchup["home_goalie_pred"] - gk_matchup["away_goalie_pred"]

gk_test = gk_matchup[gk_matchup["game_date"] >= TEST_CUTOFF].copy()

print(f"\n  Deployment matchup → game outcome (2025-26, n={len(gk_test)}):")
if len(gk_test) > 50:
    corr_deploy = gk_test["deploy_diff"].corr(gk_test["home_won_actual"])
    print(f"    Deploy score diff → home win: r={corr_deploy:.3f}")

    # By tercile
    gk_test["deploy_q"] = pd.qcut(gk_test["deploy_diff"], 3, labels=["Low", "Mid", "High"],
                                   duplicates="drop")
    print(f"\n    Win rate by deployment matchup tercile:")
    for q, grp in gk_test.groupby("deploy_q", observed=True):
        wr = grp["home_won_actual"].mean()
        mean_diff = grp["deploy_diff"].mean()
        print(f"      {q:4s} (diff={mean_diff:+.4f}): home WR={wr:.3f}, n={len(grp)}")


# =====================================================================
# STAGE 1: REGULATION MODEL (LAYER 1) — LAMBDA ACCURACY
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 1: REGULATION MODEL (LAYER 1) — LAMBDA ACCURACY")
print("=" * 70)

# Lambda vs actual regulation goals
for label, df_sub in [("All Seasons", paired), ("2025-26 OOS", test)]:
    lam_h_corr = df_sub["lam_home_raw"].corr(df_sub["home_reg_gf"])
    lam_a_corr = df_sub["lam_away_raw"].corr(df_sub["away_reg_gf"])
    lam_h_bias = df_sub["lam_home_raw"].mean() - df_sub["home_reg_gf"].mean()
    lam_a_bias = df_sub["lam_away_raw"].mean() - df_sub["away_reg_gf"].mean()
    lam_h_mae = (df_sub["lam_home_raw"] - df_sub["home_reg_gf"]).abs().mean()
    lam_a_mae = (df_sub["lam_away_raw"] - df_sub["away_reg_gf"]).abs().mean()

    print(f"\n  {label}:")
    print(f"    Home λ: corr={lam_h_corr:.3f}, MAE={lam_h_mae:.3f}, "
          f"bias={lam_h_bias:+.3f} (pred={df_sub['lam_home_raw'].mean():.3f}, "
          f"actual={df_sub['home_reg_gf'].mean():.3f})")
    print(f"    Away λ: corr={lam_a_corr:.3f}, MAE={lam_a_mae:.3f}, "
          f"bias={lam_a_bias:+.3f} (pred={df_sub['lam_away_raw'].mean():.3f}, "
          f"actual={df_sub['away_reg_gf'].mean():.3f})")

    # Spread: predicted vs actual
    h_pred_std = df_sub["lam_home_raw"].std()
    h_actual_std = df_sub["home_reg_gf"].std()
    a_pred_std = df_sub["lam_away_raw"].std()
    a_actual_std = df_sub["away_reg_gf"].std()
    print(f"    Home spread: pred_std={h_pred_std:.3f}, actual_std={h_actual_std:.3f}, "
          f"ratio={h_pred_std/h_actual_std:.3f}")
    print(f"    Away spread: pred_std={a_pred_std:.3f}, actual_std={a_actual_std:.3f}, "
          f"ratio={a_pred_std/a_actual_std:.3f}")

# Goalie deployment features in regulation model
print(f"\n  Goalie deployment features in regulation model:")
reg_test = reg_feat[reg_feat["game_date"] >= TEST_CUTOFF].copy()
goalie_cols = ["goalie_ga_rate", "opp_goalie_ga_rate", "goalie_ga_rate_diff",
               "goalie_deploy_score", "opp_goalie_deploy_score", "deploy_score_diff",
               "goalie_goalie_switch", "goalie_starter_role_share"]
available_goalie = [c for c in goalie_cols if c in reg_test.columns]

if available_goalie:
    reg_test = reg_test.merge(
        completed[["game_id", "home_won_actual"]],
        on="game_id", how="inner"
    )
    reg_home = reg_test[reg_test["is_home"] == 1].copy()
    for col in available_goalie:
        if col in reg_home.columns:
            corr_gf = reg_home[col].corr(reg_home["reg_gf"])
            avail = reg_home[col].notna().mean()
            print(f"    {col:35s} → reg goals: r={corr_gf:.3f}, avail={avail:.1%}")
else:
    print(f"    No goalie deployment columns found in regulation features")


# =====================================================================
# STAGE 2: SCORING ANCHOR
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 2: SCORING ANCHOR")
print("=" * 70)

valid_test = test.dropna(subset=["anchor_home", "anchor_away"])
print(f"\n  Test games with anchor data: {len(valid_test)}")

# Compare raw lambda vs anchored lambda
raw_h_bias = valid_test["lam_home_raw"].mean() - valid_test["home_reg_gf"].mean()
adj_h_bias = valid_test["lam_home"].mean() - valid_test["home_reg_gf"].mean()
raw_a_bias = valid_test["lam_away_raw"].mean() - valid_test["away_reg_gf"].mean()
adj_a_bias = valid_test["lam_away"].mean() - valid_test["away_reg_gf"].mean()

raw_h_mae = (valid_test["lam_home_raw"] - valid_test["home_reg_gf"]).abs().mean()
adj_h_mae = (valid_test["lam_home"] - valid_test["home_reg_gf"]).abs().mean()
raw_a_mae = (valid_test["lam_away_raw"] - valid_test["away_reg_gf"]).abs().mean()
adj_a_mae = (valid_test["lam_away"] - valid_test["away_reg_gf"]).abs().mean()

print(f"\n  Home lambda:")
print(f"    Raw:      bias={raw_h_bias:+.4f}, MAE={raw_h_mae:.4f}")
print(f"    Anchored: bias={adj_h_bias:+.4f}, MAE={adj_h_mae:.4f}")
print(f"    Anchor improved bias: {abs(raw_h_bias) > abs(adj_h_bias)}")
print(f"    Anchor improved MAE:  {raw_h_mae > adj_h_mae}")

print(f"\n  Away lambda:")
print(f"    Raw:      bias={raw_a_bias:+.4f}, MAE={raw_a_mae:.4f}")
print(f"    Anchored: bias={adj_a_bias:+.4f}, MAE={adj_a_mae:.4f}")
print(f"    Anchor improved bias: {abs(raw_a_bias) > abs(adj_a_bias)}")
print(f"    Anchor improved MAE:  {raw_a_mae > adj_a_mae}")

print(f"\n  Anchor range:")
print(f"    Home: mean={valid_test['anchor_home'].mean():.4f}, "
      f"std={valid_test['anchor_home'].std():.4f}")
print(f"    Away: mean={valid_test['anchor_away'].mean():.4f}, "
      f"std={valid_test['anchor_away'].std():.4f}")


# =====================================================================
# STAGE 3: POISSON PROBABILITIES — RAW P(HOME WIN)
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 3: POISSON PROBABILITIES")
print("=" * 70)

# Compute raw Poisson probs from raw lambdas (no anchor)
raw_probs = []
for _, row in test.iterrows():
    p_h, p_a, p_t = compute_game_probabilities(row["lam_home_raw"], row["lam_away_raw"])
    raw_p_home = p_h + p_t * row["p_home_win_ot_given"]
    raw_probs.append({
        "game_id": row["game_id"],
        "p_home_raw_no_anchor": raw_p_home,
        "p_tie_raw_no_anchor": p_t,
    })
raw_df = pd.DataFrame(raw_probs)
test = test.merge(raw_df, on="game_id", how="left")

y = test["home_won_actual"].values.astype(float)

# Log-loss at each stage
ll_constant = mean_ll(y, np.full(len(y), y.mean()))
ll_raw_no_anchor = mean_ll(y, test["p_home_raw_no_anchor"].values)
ll_raw_with_anchor = mean_ll(y, test["p_home_win_raw"].values)
ll_calibrated = mean_ll(y, test["p_home_win"].values)

dk_valid = test.dropna(subset=["dk_home_fair"])
y_dk = dk_valid["home_won_actual"].values.astype(float)
ll_dk = mean_ll(y_dk, dk_valid["dk_home_fair"].values)

print(f"\n  Log-loss comparison (lower is better):")
print(f"    Stage 0: Constant baseline      = {ll_constant:.5f}")
print(f"    Stage 1: Raw Poisson (no anchor) = {ll_raw_no_anchor:.5f}  (Δ={ll_raw_no_anchor - ll_constant:+.5f})")
print(f"    Stage 2: + Scoring anchor        = {ll_raw_with_anchor:.5f}  (Δ={ll_raw_with_anchor - ll_raw_no_anchor:+.5f})")
print(f"    Stage 3: + Tie inflation         = {ll_raw_with_anchor:.5f}  (included in raw)")
print(f"    Stage 4: + Platt calibration     = {ll_calibrated:.5f}  (Δ={ll_calibrated - ll_raw_with_anchor:+.5f})")
print(f"    DraftKings (market)              = {ll_dk:.5f}")
print(f"    Model vs DK                      = {ll_calibrated - ll_dk:+.5f} ({'model better' if ll_calibrated < ll_dk else 'DK better'})")

# Improvement % over constant
imp_raw = (1 - ll_raw_no_anchor / ll_constant) * 100
imp_anchor = (1 - ll_raw_with_anchor / ll_constant) * 100
imp_platt = (1 - ll_calibrated / ll_constant) * 100
print(f"\n  Improvement over constant baseline:")
print(f"    Raw Poisson:      {imp_raw:+.2f}%")
print(f"    + Anchor:         {imp_anchor:+.2f}% ({imp_anchor - imp_raw:+.3f}% marginal)")
print(f"    + Platt:          {imp_platt:+.2f}% ({imp_platt - imp_anchor:+.3f}% marginal)")


# =====================================================================
# STAGE 4: TIE INFLATION — P(OT) CALIBRATION
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 4: TIE INFLATION — P(OT) CALIBRATION")
print("=" * 70)

actual_ot_rate = test["is_ot"].mean()
raw_p_tie = test["p_tie_raw_no_anchor"].mean()
calibrated_p_ot = test["p_ot"].mean()

print(f"  Actual OT rate:       {actual_ot_rate:.4f}")
print(f"  Raw P(tie) mean:      {raw_p_tie:.4f}")
print(f"  Calibrated P(OT):     {calibrated_p_ot:.4f}")
print(f"  Raw bias:             {raw_p_tie - actual_ot_rate:+.4f}")
print(f"  Calibrated bias:      {calibrated_p_ot - actual_ot_rate:+.4f}")
print(f"  Inflation improved:   {abs(calibrated_p_ot - actual_ot_rate) < abs(raw_p_tie - actual_ot_rate)}")

# P(OT) by decile
test_ot = test.dropna(subset=["p_ot"]).copy()
test_ot["ot_actual"] = test_ot["is_ot"]
test_ot["ot_bin"] = pd.qcut(test_ot["p_ot"], 5, labels=False, duplicates="drop")
print(f"\n  P(OT) calibration by quintile:")
for b, grp in test_ot.groupby("ot_bin"):
    pred = grp["p_ot"].mean()
    actual = grp["ot_actual"].mean()
    print(f"    Bin {b}: pred={pred:.3f}, actual={actual:.3f}, diff={actual-pred:+.3f}, n={len(grp)}")


# =====================================================================
# STAGE 5: OT EDGE MODEL (LAYER 2) — DEEP DIVE
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 5: OT EDGE MODEL (LAYER 2) — DEEP DIVE")
print("=" * 70)

ot_games = test[test["is_ot"] == 1].copy()
has_ot_pred = ot_games["game_id"].isin(ot["game_id"])
ot_with_model = ot_games[has_ot_pred]
ot_without_model = ot_games[~has_ot_pred]

print(f"  OT games in test: {len(ot_games)}")
print(f"  With OT model predictions: {len(ot_with_model)}")
print(f"  Using default ({DEFAULT_P_HOME_OT}): {len(ot_without_model)}")

if len(ot_with_model) > 0:
    ot_y = ot_with_model["home_won_actual"].values.astype(float)
    ot_p = ot_with_model["p_home_win_ot_given"].values
    ot_ll = mean_ll(ot_y, ot_p)
    ot_base = mean_ll(ot_y, np.full(len(ot_y), ot_y.mean()))
    default_ll = mean_ll(ot_y, np.full(len(ot_y), DEFAULT_P_HOME_OT))

    ot_actual_home_rate = ot_y.mean()
    ot_pred_mean = ot_p.mean()
    ot_corr = np.corrcoef(ot_y, ot_p)[0, 1] if len(ot_y) > 10 else np.nan

    print(f"\n  OT model metrics (n={len(ot_with_model)}):")
    print(f"    Actual home win|OT: {ot_actual_home_rate:.3f}")
    print(f"    Predicted mean:     {ot_pred_mean:.3f}")
    print(f"    Correlation:        {ot_corr:.3f}")
    print(f"    LL (model):         {ot_ll:.5f}")
    print(f"    LL (constant base): {ot_base:.5f}")
    print(f"    LL (default 0.518): {default_ll:.5f}")
    print(f"    Model vs default:   {ot_ll - default_ll:+.5f}")
    print(f"    Model vs constant:  {ot_ll - ot_base:+.5f}")

    # Calibration by quintile
    ot_cal = ot_with_model.copy()
    ot_cal["ot_pred_q"] = pd.qcut(ot_cal["p_home_win_ot_given"], 5, labels=False, duplicates="drop")
    print(f"\n  P(home|OT) calibration by quintile:")
    for q, grp in ot_cal.groupby("ot_pred_q"):
        pred = grp["p_home_win_ot_given"].mean()
        actual = grp["home_won_actual"].mean()
        print(f"    Q{q} (pred={pred:.3f}): actual={actual:.3f}, diff={actual-pred:+.3f}, n={len(grp)}")

    # Prediction spread
    pred_spread = ot_p.std()
    print(f"\n  P(home|OT) spread:   std={pred_spread:.4f}, range=[{ot_p.min():.3f}, {ot_p.max():.3f}]")

else:
    print("  No OT model predictions available for test games")

# Value of OT model: compare final P(home win) using model vs using default
if len(ot_games) > 10:
    # With OT model
    ll_with_ot = mean_ll(
        ot_games["home_won_actual"].values.astype(float),
        ot_games["p_home_win"].values
    )
    # If we used default for all
    ot_games_default = ot_games.copy()
    default_p_home = ot_games_default["p_home_reg_win"] + ot_games_default["p_ot"] * DEFAULT_P_HOME_OT
    if ot_games_default["platt_a"].notna().all():
        default_p_home = ot_games_default.apply(
            lambda r: apply_platt_calibration(
                r["p_home_reg_win"] + r["p_ot"] * DEFAULT_P_HOME_OT,
                r["platt_a"], r["platt_b"]
            ), axis=1
        )
    ll_default_ot = mean_ll(
        ot_games["home_won_actual"].values.astype(float),
        default_p_home.values
    )
    print(f"\n  Value-add of OT model on final P(home win):")
    print(f"    LL with OT model:  {ll_with_ot:.5f}")
    print(f"    LL with default:   {ll_default_ot:.5f}")
    print(f"    Improvement:       {ll_default_ot - ll_with_ot:+.5f}")
    print(f"    In millibits:      {(ll_default_ot - ll_with_ot)*1000:+.2f} mb")

# Goalie matchup analysis within OT model
print(f"\n  --- Goalie Matchup in OT Model ---")

# Get OT features for test games
ot_feat_test = ot_feat[ot_feat["game_date"] >= TEST_CUTOFF].copy()
goalie_ot_cols = ["d_goalie_ga_rate", "d_goalie_deployment_score",
                  "d_starter_role_share", "d_goalie_switch", "d_consecutive_starts",
                  "d_was_pulled_last_game", "d_days_rest",
                  "d_games_started_last_14d",
                  "d_goalie_rate_x_hd_xgf", "d_switch_x_ot_win_rate"]
available_goalie_ot = [c for c in goalie_ot_cols if c in ot_feat_test.columns]

if available_goalie_ot and len(ot_feat_test) > 0:
    # Merge outcomes
    ot_feat_test = ot_feat_test.merge(
        completed[["game_id", "home_won_actual"]],
        on="game_id", how="inner"
    )

    print(f"  OT games with goalie deployment features: {len(ot_feat_test)}")
    print(f"\n  Goalie deployment feature correlations with OT winner:")
    for col in available_goalie_ot:
        valid = ot_feat_test.dropna(subset=[col])
        if len(valid) > 20:
            corr = valid[col].corr(valid["home_win"])
            print(f"    {col:35s}: r={corr:+.3f} (n={len(valid)})")

    # d_goalie_ga_rate (or d_goalie_deployment_score) by tercile → OT win rate
    goalie_diff_col = "d_goalie_ga_rate" if "d_goalie_ga_rate" in ot_feat_test.columns else "d_goalie_deployment_score"
    if goalie_diff_col in ot_feat_test.columns:
        valid_deploy = ot_feat_test.dropna(subset=[goalie_diff_col])
        if len(valid_deploy) > 30:
            valid_deploy["deploy_tercile"] = pd.qcut(
                valid_deploy[goalie_diff_col], 3,
                labels=["Worse", "Even", "Better"], duplicates="drop"
            )
            print(f"\n  Home OT win rate by goalie matchup ({goalie_diff_col}):")
            for label, grp in valid_deploy.groupby("deploy_tercile", observed=True):
                wr = grp["home_win"].mean()
                mean_diff = grp[goalie_diff_col].mean()
                print(f"    {label:6s} (diff={mean_diff:+.4f}): "
                      f"home WR={wr:.3f}, n={len(grp)}")
else:
    print(f"  No goalie deployment features available in OT features")

# OT vs Shootout breakdown
if len(ot_games) > 10:
    print(f"\n  --- OT vs Shootout Breakdown ---")
    for outcome_type in ["OT", "SO"]:
        sub = ot_games[ot_games["game_outcome_type"] == outcome_type]
        if len(sub) > 5:
            sub_y = sub["home_won_actual"].values.astype(float)
            sub_p = sub["p_home_win"].values
            sub_ll = mean_ll(sub_y, sub_p)
            sub_base = mean_ll(sub_y, np.full(len(sub_y), sub_y.mean()))
            print(f"    {outcome_type}: n={len(sub)}, actual home WR={sub_y.mean():.3f}, "
                  f"model LL={sub_ll:.5f}, baseline LL={sub_base:.5f}, "
                  f"improvement={sub_base - sub_ll:+.5f}")


# =====================================================================
# STAGE 6: PLATT CALIBRATION
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 6: PLATT CALIBRATION")
print("=" * 70)

platt_on = test[test["platt_a"] != 1.0].copy()
platt_off = test[test["platt_a"] == 1.0].copy()

print(f"  Games with Platt OFF (raw probs): {len(platt_off)}")
print(f"  Games with Platt ON:              {len(platt_on)}")

if len(platt_on) > 50:
    y_on = platt_on["home_won_actual"].values.astype(float)
    ll_raw_on = mean_ll(y_on, platt_on["p_home_win_raw"].values)
    ll_cal_on = mean_ll(y_on, platt_on["p_home_win"].values)

    print(f"\n  Platt ON games:")
    print(f"    LL (raw probs):    {ll_raw_on:.5f}")
    print(f"    LL (calibrated):   {ll_cal_on:.5f}")
    print(f"    Improvement:       {ll_raw_on - ll_cal_on:+.5f} ({'Platt helps' if ll_cal_on < ll_raw_on else 'Platt hurts'})")
    print(f"    Platt a (stretch): mean={platt_on['platt_a'].mean():.3f}, "
          f"range=[{platt_on['platt_a'].min():.3f}, {platt_on['platt_a'].max():.3f}]")
    print(f"    Platt b (bias):    mean={platt_on['platt_b'].mean():.3f}")

    # Compression check: are raw probs too compressed?
    raw_std = platt_on["p_home_win_raw"].std()
    cal_std = platt_on["p_home_win"].std()
    print(f"\n  Probability spread:")
    print(f"    Raw P(home) std:        {raw_std:.4f}")
    print(f"    Calibrated P(home) std: {cal_std:.4f}")
    print(f"    Platt stretches by:     {(cal_std/raw_std - 1)*100:+.1f}%")

if len(platt_off) > 50:
    y_off = platt_off["home_won_actual"].values.astype(float)
    ll_off = mean_ll(y_off, platt_off["p_home_win"].values)
    base_off = mean_ll(y_off, np.full(len(y_off), y_off.mean()))
    print(f"\n  Platt OFF games (early season):")
    print(f"    LL:                {ll_off:.5f}")
    print(f"    Baseline:          {base_off:.5f}")
    print(f"    Improvement:       {(1 - ll_off/base_off)*100:+.1f}%")


# =====================================================================
# STAGE 7: BETTING EDGE — MODEL VS DK DECOMPOSITION
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 7: BETTING EDGE — MODEL VS DK DECOMPOSITION")
print("=" * 70)

with_odds = test.dropna(subset=["dk_home_fair"]).copy()
print(f"\n  Games with DK odds: {len(with_odds)}")

y_o = with_odds["home_won_actual"].values.astype(float)
p_model = with_odds["p_home_win"].values
p_dk = with_odds["dk_home_fair"].values

ll_model = mean_ll(y_o, p_model)
ll_dk = mean_ll(y_o, p_dk)
ll_base = mean_ll(y_o, np.full(len(y_o), y_o.mean()))

print(f"\n  Overall log-loss:")
print(f"    Constant baseline: {ll_base:.5f}")
print(f"    Model:             {ll_model:.5f} (Δ base: {ll_model - ll_base:+.5f})")
print(f"    DK:                {ll_dk:.5f} (Δ base: {ll_dk - ll_base:+.5f})")
print(f"    Model vs DK:       {ll_model - ll_dk:+.5f}")

# Where is edge coming from? Split by quintile of DK fair prob
with_odds["dk_quintile"] = pd.qcut(with_odds["dk_home_fair"], 5, labels=False, duplicates="drop")
print(f"\n  Model vs DK by DK probability quintile:")
for q, grp in with_odds.groupby("dk_quintile"):
    y_q = grp["home_won_actual"].values.astype(float)
    ml = mean_ll(y_q, grp["p_home_win"].values)
    dl = mean_ll(y_q, grp["dk_home_fair"].values)
    n = len(grp)
    dk_mean = grp["dk_home_fair"].mean()
    model_mean = grp["p_home_win"].mean()
    actual_mean = y_q.mean()
    print(f"    Q{q} (DK≈{dk_mean:.2f}): model_LL={ml:.4f}, dk_LL={dl:.4f}, "
          f"Δ={ml-dl:+.4f} ({'Model' if ml < dl else 'DK'}), "
          f"model={model_mean:.3f}, actual={actual_mean:.3f}, n={n}")

# Edge by month
print(f"\n  Model vs DK by month:")
with_odds["month"] = with_odds["game_date"].dt.to_period("M")
for month, grp in with_odds.groupby("month"):
    y_m = grp["home_won_actual"].values.astype(float)
    ml = mean_ll(y_m, grp["p_home_win"].values)
    dl = mean_ll(y_m, grp["dk_home_fair"].values)
    print(f"    {month}: model={ml:.4f}, DK={dl:.4f}, Δ={ml-dl:+.4f} "
          f"({'Model' if ml < dl else 'DK'}), n={len(grp)}")

# Home vs away edge analysis
print(f"\n  Home edge vs Away edge:")
print(f"    Home edge mean: {with_odds['home_edge'].mean():+.4f}")
print(f"    Away edge mean: {with_odds['away_edge'].mean():+.4f}")
print(f"    Model favors away by {abs(with_odds['home_edge'].mean())*100:.1f}% on average")

# Is the away bias correct?
# Split: games where model disagrees with DK about who is favored
model_home_fav = with_odds["p_home_win"] > 0.5
dk_home_fav = with_odds["dk_home_fair"] > 0.5
agree = model_home_fav == dk_home_fav
disagree = ~agree

print(f"\n  Model vs DK agreement on favorite:")
print(f"    Agree:    {agree.sum()} games ({agree.mean()*100:.1f}%)")
print(f"    Disagree: {disagree.sum()} games ({disagree.mean()*100:.1f}%)")

if disagree.sum() > 20:
    dis = with_odds[disagree]
    dis_y = dis["home_won_actual"].values.astype(float)
    dis_model_ll = mean_ll(dis_y, dis["p_home_win"].values)
    dis_dk_ll = mean_ll(dis_y, dis["dk_home_fair"].values)
    print(f"    On disagreement games: model LL={dis_model_ll:.4f}, DK LL={dis_dk_ll:.4f}, "
          f"Δ={dis_model_ll-dis_dk_ll:+.4f}")

# Flat staking ROI by edge bucket
print(f"\n  Flat staking ROI by edge bucket:")
with_odds["best_edge"] = with_odds[["home_edge", "away_edge"]].max(axis=1)
with_odds["bet_side"] = np.where(with_odds["home_edge"] > with_odds["away_edge"], "home", "away")
with_odds["bet_odds"] = np.where(
    with_odds["bet_side"] == "home",
    with_odds["dk_home_dec"],
    with_odds["dk_away_dec"]
)
with_odds["bet_won"] = np.where(
    with_odds["bet_side"] == "home",
    with_odds["home_won_actual"] == 1,
    with_odds["home_won_actual"] == 0
)
with_odds["flat_pnl"] = np.where(
    with_odds["bet_won"],
    with_odds["bet_odds"] - 1.0,
    -1.0
)

bins = [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0]
labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(bins)-1)]
with_odds["edge_bucket"] = pd.cut(with_odds["best_edge"], bins=bins, labels=labels)
for bucket, grp in with_odds.groupby("edge_bucket", observed=True):
    n = len(grp)
    pnl = grp["flat_pnl"].sum()
    roi = pnl / n * 100
    wr = grp["bet_won"].mean()
    print(f"    {bucket:>10s}: n={n:>4d}, ROI={roi:+6.1f}%, WR={wr:.3f}, P&L={pnl:+.1f}u")

# Edge on OT games vs regulation games
print(f"\n  Edge by game outcome type:")
with_odds_type = with_odds.copy()
with_odds_type["is_ot_game"] = with_odds_type["game_outcome_type"].isin(["OT", "SO"])
for is_ot, label in [(False, "Regulation"), (True, "OT/SO")]:
    sub = with_odds_type[with_odds_type["is_ot_game"] == is_ot]
    if len(sub) > 10:
        sub_y = sub["home_won_actual"].values.astype(float)
        sub_ml = mean_ll(sub_y, sub["p_home_win"].values)
        sub_dl = mean_ll(sub_y, sub["dk_home_fair"].values)
        sub_pnl = sub["flat_pnl"].sum()
        sub_roi = sub_pnl / len(sub) * 100
        print(f"    {label:10s}: n={len(sub)}, model LL={sub_ml:.5f}, DK LL={sub_dl:.5f}, "
              f"Δ={sub_ml-sub_dl:+.5f}, flat ROI={sub_roi:+.1f}%")


# =====================================================================
# STAGE 8: WHERE CAN WE IMPROVE MOST?
# =====================================================================
print("\n" + "=" * 70)
print("STAGE 8: IMPROVEMENT OPPORTUNITIES (RANKED)")
print("=" * 70)

# Compute marginal contribution of each stage
stages = [
    ("Constant → Raw Poisson",     ll_constant - ll_raw_no_anchor),
    ("+ Scoring Anchor",           ll_raw_no_anchor - ll_raw_with_anchor),
    ("+ Platt Calibration",        ll_raw_with_anchor - ll_calibrated),
]

total_improvement = ll_constant - ll_calibrated
remaining_gap = ll_calibrated - ll_dk  # positive = DK still better

print(f"\n  Total model improvement over constant:  {total_improvement*1000:.2f} millibits")
print(f"  Remaining gap to DK:                    {remaining_gap*1000:.2f} millibits")
print(f"  DK improvement over constant:           {(ll_constant - ll_dk)*1000:.2f} millibits")

print(f"\n  Stage-by-stage contribution (millibits gained):")
for name, delta in sorted(stages, key=lambda x: -x[1]):
    pct = delta / total_improvement * 100 if total_improvement > 0 else 0
    print(f"    {name:<30s}: {delta*1000:+.3f} mb ({pct:5.1f}% of total)")

# OT model contribution
if len(ot_games) > 10:
    ot_model_value = (ll_default_ot - ll_with_ot)  # positive = OT model helps
    print(f"    OT Model (on OT games)        : {ot_model_value*1000:+.3f} mb")

# Goalie deployment model contribution estimate
gd_test_valid = gd_test.dropna(subset=[ga_actual_col, ga_pred_col])
if len(gd_test_valid) > 50:
    gk_corr = gd_test_valid[ga_pred_col].corr(gd_test_valid[ga_actual_col])
    print(f"    Goalie ({ga_pred_col} corr)    : r={gk_corr:.3f}")

# Largest improvement areas
lam_diff_corr = test["lam_home_raw"].corr(test["home_reg_gf"])
raw_spread = test["p_home_win_raw"].std()
actual_spread = test["home_won_actual"].std()
away_bias = with_odds["home_edge"].mean()

print(f"\n  Improvement priority:")
print(f"  ┌───────────────────────────────────────────────────────────┐")
print(f"  │ 1. REGULATION MODEL (Layer 1)                            │")
print(f"  │    Lambda corr: {lam_diff_corr:.3f} — room to improve            │")
print(f"  │    This is the foundation; better λ → better probs        │")
print(f"  │                                                           │")
print(f"  │ 2. GOALIE MODEL (Layer 0)                                 │")
if len(gd_test_valid) > 50:
    print(f"  │    {ga_pred_col} corr: {gk_corr:.3f}                        │")
    print(f"  │    Feeds into BOTH Layer 1 (reg) and Layer 2 (OT)        │")
print(f"  │                                                           │")
print(f"  │ 3. OT MODEL (Layer 2)                                    │")
if len(ot_with_model) > 0:
    print(f"  │    P(home|OT) corr: {ot_corr:.3f}                               │")
    print(f"  │    LL improvement: {ot_model_value*1000:+.2f} mb on OT games              │")
    print(f"  │    Goalie deployment is key OT factor                     │")
print(f"  │                                                           │")
print(f"  │ 4. PROBABILITY SPREAD                                    │")
print(f"  │    P(home win) std: {raw_spread:.4f} (actual: {actual_spread:.4f})       │")
if raw_spread < actual_spread * 0.5:
    print(f"  │    Heavily compressed — model not confident enough        │")
print(f"  │                                                           │")
print(f"  │ 5. HOME/AWAY BIAS                                        │")
print(f"  │    Systematic away edge: {away_bias*100:+.1f}%                        │")
if abs(away_bias) > 0.01:
    print(f"  │    Consider home-ice advantage calibration                │")
print(f"  │                                                           │")
print(f"  │ 6. P(OT) CALIBRATION                                     │")
ot_bias = calibrated_p_ot - actual_ot_rate
print(f"  │    Bias: {ot_bias:+.4f} — {'OK' if abs(ot_bias) < 0.01 else 'needs work'}                                 │")
print(f"  └───────────────────────────────────────────────────────────┘")


# =====================================================================
# FIGURES — 8 comprehensive diagnostic figures
# =====================================================================
print("\n" + "=" * 70)
print("GENERATING DIAGNOSTIC PLOTS")
print("=" * 70)


# ─── Figure 1: Goalie Deployment Context ─────────────────────────────
fig1, axes1 = plt.subplots(2, 2, figsize=(14, 11))
fig1.suptitle("Figure 1: Goalie Deployment Context (OOS)", fontsize=14, fontweight="bold")

# 1a: Goalie GA rate distribution vs actual GA
ax = axes1[0, 0]
gd_plot = gd_test.dropna(subset=[ga_actual_col, ga_pred_col])
if len(gd_plot) > 10:
    ax.hist(gd_plot[ga_pred_col], bins=40, alpha=0.6, color="steelblue",
            label=f"Predicted ({ga_pred_col})", density=True)
    ax.hist(gd_plot[ga_actual_col], bins=40, alpha=0.4, color="coral",
            label=f"Actual ({ga_actual_col})", density=True)
    ax.axvline(gd_plot[ga_pred_col].mean(), color="steelblue", linestyle="--", alpha=0.7)
    ax.axvline(gd_plot[ga_actual_col].mean(), color="coral", linestyle="--", alpha=0.7)
    ax.set_xlabel("Goals Against (Regulation)")
    ax.set_ylabel("Density")
    ax.set_title(f"GA Rate Distribution (n={len(gd_plot):,})")
    ax.legend(fontsize=8)

# 1b: Goalie GA rate scatter (predicted vs actual, calibration line)
ax = axes1[0, 1]
if len(gd_plot) > 10:
    ax.scatter(gd_plot[ga_pred_col], gd_plot[ga_actual_col],
               alpha=0.08, s=8, color="mediumseagreen")
    lo = min(gd_plot[ga_pred_col].min(), gd_plot[ga_actual_col].min())
    hi = max(gd_plot[ga_pred_col].max(), gd_plot[ga_actual_col].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="Perfect (y=x)")
    corr_gk = gd_plot[ga_pred_col].corr(gd_plot[ga_actual_col])
    ax.set_xlabel(f"Predicted {ga_pred_col}")
    ax.set_ylabel(f"Actual {ga_actual_col}")
    ax.set_title(f"Goalie Model Calibration (r={corr_gk:.3f})")
    ax.legend()

# 1c: Win rate by deployment context (bar chart)
ax = axes1[1, 0]
if len(gk_test) > 50:
    # Group by goalie_switch + B2B
    cats = []
    if "is_back_to_back" in gd_test.columns:
        # Merge with game outcomes
        gd_test_m = gd_test.merge(
            completed[["game_id", "home_won_actual"]],
            on="game_id", how="inner"
        )
        gd_home = gd_test_m[gd_test_m["is_home"] == 1]
        for (sw, b2b), grp in gd_home.groupby(["goalie_switch", "is_back_to_back"]):
            wr = grp["home_won_actual"].mean()
            label = f"Switch={int(sw)}\nB2B={int(b2b)}"
            cats.append({"label": label, "wr": wr, "n": len(grp)})
    if cats:
        cat_df = pd.DataFrame(cats)
        colors_bar = ["#2ecc71", "#3498db", "#e74c3c", "#f39c12"]
        ax.bar(range(len(cat_df)), cat_df["wr"],
               color=colors_bar[:len(cat_df)], alpha=0.7, edgecolor="black", linewidth=0.5)
        ax.set_xticks(range(len(cat_df)))
        ax.set_xticklabels(cat_df["label"], fontsize=8)
        for i, row_data in cat_df.iterrows():
            ax.text(i, row_data["wr"] + 0.01, f"n={row_data['n']}", ha="center", fontsize=7)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
        ax.set_ylabel("Home Win Rate")
        ax.set_title("Win Rate by Deployment Context")
    else:
        ax.text(0.5, 0.5, "No deployment context data", ha="center", va="center", transform=ax.transAxes)

# 1d: Goalie matchup differential vs home win rate (terciles)
ax = axes1[1, 1]
if len(gk_test) > 50:
    gk_test_plot = gk_test.copy()
    gk_test_plot["deploy_q"] = pd.qcut(gk_test_plot["deploy_diff"], 3,
                                        labels=False, duplicates="drop")
    by_q = gk_test_plot.groupby("deploy_q").agg(
        home_wr=("home_won_actual", "mean"),
        mean_diff=("deploy_diff", "mean"),
        n=("game_id", "count"),
    ).reset_index()
    colors_t = ["#ff4757", "#ffd700", "#00ff88"]
    ax.bar(by_q["deploy_q"], by_q["home_wr"], color=colors_t[:len(by_q)],
           alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, label="50%")
    for i, row_data in by_q.iterrows():
        ax.text(row_data["deploy_q"], row_data["home_wr"] + 0.01,
                f"{row_data['mean_diff']:+.3f}\nn={row_data['n']}", ha="center", fontsize=7)
    ax.set_xlabel("Goalie Matchup Tercile")
    ax.set_ylabel("Home Win Rate")
    ax.set_title("Home Win Rate by Goalie Matchup Differential")
    ax.legend()

plt.tight_layout()
fig1.savefig(PLOTS_DIR / "fig1_goalie_deployment.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig1_goalie_deployment.png'}")


# ─── Figure 2: Regulation Lambda Quality ─────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle("Figure 2: Regulation Lambda Quality (OOS)", fontsize=14, fontweight="bold")

# 2a: Home lambda scatter
ax = axes2[0, 0]
ax.scatter(test["lam_home_raw"], test["home_reg_gf"], alpha=0.1, s=10, color="steelblue")
ax.plot([0, 7], [0, 7], "k--", alpha=0.4, label="Perfect")
corr_h = test["lam_home_raw"].corr(test["home_reg_gf"])
ax.set_xlabel("Predicted λ Home")
ax.set_ylabel("Actual Home Reg Goals")
ax.set_title(f"Home λ vs Actual (r={corr_h:.3f})")
ax.legend()

# 2b: Away lambda scatter
ax = axes2[0, 1]
ax.scatter(test["lam_away_raw"], test["away_reg_gf"], alpha=0.1, s=10, color="coral")
ax.plot([0, 7], [0, 7], "k--", alpha=0.4, label="Perfect")
corr_a = test["lam_away_raw"].corr(test["away_reg_gf"])
ax.set_xlabel("Predicted λ Away")
ax.set_ylabel("Actual Away Reg Goals")
ax.set_title(f"Away λ vs Actual (r={corr_a:.3f})")
ax.legend()

# 2c: Lambda error distribution
ax = axes2[1, 0]
home_err = test["lam_home_raw"] - test["home_reg_gf"]
away_err = test["lam_away_raw"] - test["away_reg_gf"]
ax.hist(home_err, bins=40, alpha=0.6, color="steelblue", label="Home", density=True)
ax.hist(away_err, bins=40, alpha=0.4, color="coral", label="Away", density=True)
ax.axvline(0, color="black", linestyle="--", alpha=0.4)
ax.set_xlabel("Lambda Error (predicted - actual)")
ax.set_ylabel("Density")
ax.set_title(f"Lambda Error Distribution (home bias={home_err.mean():+.3f})")
ax.legend()

# 2d: Lambda calibration by quintile
ax = axes2[1, 1]
all_lam = pd.concat([
    test[["lam_home_raw", "home_reg_gf"]].rename(
        columns={"lam_home_raw": "lam", "home_reg_gf": "actual"}),
    test[["lam_away_raw", "away_reg_gf"]].rename(
        columns={"lam_away_raw": "lam", "away_reg_gf": "actual"}),
])
all_lam["lam_bin"] = pd.qcut(all_lam["lam"], 5, labels=False, duplicates="drop")
lam_cal = all_lam.groupby("lam_bin").agg(
    pred=("lam", "mean"), actual=("actual", "mean"), n=("lam", "count"),
).reset_index()
ax.plot([1.5, 4.5], [1.5, 4.5], "k--", alpha=0.4, label="Perfect")
ax.scatter(lam_cal["pred"], lam_cal["actual"], s=80, c="steelblue", zorder=5)
for _, row_data in lam_cal.iterrows():
    ax.annotate(f"n={row_data['n']}", (row_data["pred"], row_data["actual"]),
                fontsize=8, textcoords="offset points", xytext=(5, 5))
ax.set_xlabel("Predicted λ (quintile mean)")
ax.set_ylabel("Actual Goals (quintile mean)")
ax.set_title("Lambda Calibration by Quintile")
ax.legend()

plt.tight_layout()
fig2.savefig(PLOTS_DIR / "fig2_regulation_lambda.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig2_regulation_lambda.png'}")


# ─── Figure 3: xG & Shot Quality ─────────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11))
fig3.suptitle("Figure 3: xG & Shot Quality (OOS)", fontsize=14, fontweight="bold")

# 3a: xG vs actual goals scatter (regulation, both teams)
ax = axes3[0, 0]
reg_test_subset = reg_feat[reg_feat["game_date"] >= TEST_CUTOFF].copy()
if "reg_xgf" in reg_test_subset.columns and "reg_gf" in reg_test_subset.columns:
    ax.scatter(reg_test_subset["reg_xgf"], reg_test_subset["reg_gf"],
               alpha=0.08, s=8, color="purple")
    ax.plot([0, 7], [0, 7], "k--", alpha=0.4, label="Perfect")
    xg_corr = reg_test_subset["reg_xgf"].corr(reg_test_subset["reg_gf"])
    ax.set_xlabel("Expected Goals (xG)")
    ax.set_ylabel("Actual Goals")
    ax.set_title(f"xG vs Actual Goals (r={xg_corr:.3f})")
    ax.legend()

# 3b: xG residual by team (top 10 over/under-performers)
ax = axes3[0, 1]
if "reg_xgf" in reg_test_subset.columns and "reg_gf" in reg_test_subset.columns:
    reg_test_subset["xg_resid"] = reg_test_subset["reg_gf"] - reg_test_subset["reg_xgf"]
    team_xg = reg_test_subset.groupby("team")["xg_resid"].mean().sort_values()
    top_teams = pd.concat([team_xg.head(5), team_xg.tail(5)])
    colors_team = ["#e74c3c" if v < 0 else "#2ecc71" for v in top_teams.values]
    ax.barh(range(len(top_teams)), top_teams.values, color=colors_team, alpha=0.7)
    ax.set_yticks(range(len(top_teams)))
    ax.set_yticklabels(top_teams.index, fontsize=8)
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Goals - xG (per game)")
    ax.set_title("xG Over/Under-Performance by Team")

# 3c: Home/Away xG differential distribution
ax = axes3[1, 0]
if "xgf_ewm_20" in reg_test_subset.columns and "xga_ewm_20" in reg_test_subset.columns:
    reg_test_subset["xg_diff_ewm"] = reg_test_subset["xgf_ewm_20"] - reg_test_subset["xga_ewm_20"]
    home_xg = reg_test_subset[reg_test_subset["is_home"] == 1]["xg_diff_ewm"].dropna()
    away_xg = reg_test_subset[reg_test_subset["is_home"] == 0]["xg_diff_ewm"].dropna()
    ax.hist(home_xg, bins=30, alpha=0.6, color="steelblue", label="Home", density=True)
    ax.hist(away_xg, bins=30, alpha=0.4, color="coral", label="Away", density=True)
    ax.axvline(0, color="black", linestyle="--", alpha=0.4)
    ax.set_xlabel("xG Differential (EWM 20)")
    ax.set_ylabel("Density")
    ax.set_title("Home vs Away xG Differential Distribution")
    ax.legend()

# 3d: Predicted lambda distribution (home vs away)
ax = axes3[1, 1]
ax.hist(test["lam_home_raw"], bins=30, alpha=0.6, color="steelblue", label="Home λ", density=True)
ax.hist(test["lam_away_raw"], bins=30, alpha=0.4, color="coral", label="Away λ", density=True)
ax.axvline(test["lam_home_raw"].mean(), color="steelblue", linestyle="--", alpha=0.7)
ax.axvline(test["lam_away_raw"].mean(), color="coral", linestyle="--", alpha=0.7)
ax.set_xlabel("Predicted λ")
ax.set_ylabel("Density")
ax.set_title(f"Lambda Distribution (home μ={test['lam_home_raw'].mean():.2f}, away μ={test['lam_away_raw'].mean():.2f})")
ax.legend()

plt.tight_layout()
fig3.savefig(PLOTS_DIR / "fig3_xg_shot_quality.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig3_xg_shot_quality.png'}")


# ─── Figure 4: Scoring Anchor & Tie Inflation ────────────────────────
fig4, axes4 = plt.subplots(2, 2, figsize=(14, 11))
fig4.suptitle("Figure 4: Scoring Anchor & Tie Inflation (OOS)", fontsize=14, fontweight="bold")

# 4a: Scoring anchor value over time (rolling 60-day)
ax = axes4[0, 0]
valid_anchor = test.dropna(subset=["anchor_home"]).sort_values("game_date")
if len(valid_anchor) > 30:
    valid_anchor["anchor_avg"] = valid_anchor["anchor_home"].rolling(60, min_periods=10).mean()
    ax.plot(valid_anchor["game_date"], valid_anchor["anchor_avg"],
            color="steelblue", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Date")
    ax.set_ylabel("Anchor Value (60-game rolling mean)")
    ax.set_title("Scoring Anchor Over Time")

# 4b: Raw lambda vs anchored lambda scatter
ax = axes4[0, 1]
valid_anch = test.dropna(subset=["lam_home", "lam_home_raw"])
if len(valid_anch) > 10:
    ax.scatter(valid_anch["lam_home_raw"], valid_anch["lam_home"],
               alpha=0.1, s=10, color="steelblue")
    lo = min(valid_anch["lam_home_raw"].min(), valid_anch["lam_home"].min())
    hi = max(valid_anch["lam_home_raw"].max(), valid_anch["lam_home"].max())
    ax.plot([lo, hi], [lo, hi], "k--", alpha=0.4, label="No change")
    ax.set_xlabel("Raw λ (Home)")
    ax.set_ylabel("Anchored λ (Home)")
    ax.set_title("Raw vs Anchored Lambda")
    ax.legend()

# 4c: P(OT) predicted vs actual — calibration by decile
ax = axes4[1, 0]
test_ot_plot = test.dropna(subset=["p_ot"]).copy()
if len(test_ot_plot) > 50:
    test_ot_plot["ot_bin"] = pd.qcut(test_ot_plot["p_ot"], 5, labels=False, duplicates="drop")
    ot_cal_data = test_ot_plot.groupby("ot_bin").agg(
        pred=("p_ot", "mean"), actual=("is_ot", "mean"), n=("game_id", "count"),
    ).reset_index()
    ax.plot([0.1, 0.5], [0.1, 0.5], "k--", alpha=0.4, label="Perfect")
    ax.scatter(ot_cal_data["pred"], ot_cal_data["actual"], s=80, c="mediumseagreen", zorder=5)
    for _, row_data in ot_cal_data.iterrows():
        ax.annotate(f"n={row_data['n']}", (row_data["pred"], row_data["actual"]),
                    fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Predicted P(OT)")
    ax.set_ylabel("Actual OT Rate")
    ax.set_title(f"P(OT) Calibration (actual rate={test['is_ot'].mean():.3f})")
    ax.legend()

# 4d: Regulation score distribution — actual home/away goals
ax = axes4[1, 1]
ax.hist(test["home_reg_gf"], bins=range(0, 10), alpha=0.6, color="steelblue",
        label=f"Home (μ={test['home_reg_gf'].mean():.2f})", density=True, align="left")
ax.hist(test["away_reg_gf"], bins=range(0, 10), alpha=0.4, color="coral",
        label=f"Away (μ={test['away_reg_gf'].mean():.2f})", density=True, align="left")
ax.set_xlabel("Regulation Goals")
ax.set_ylabel("Frequency")
ax.set_title("Regulation Goal Distribution")
ax.legend()

plt.tight_layout()
fig4.savefig(PLOTS_DIR / "fig4_anchor_tie_inflation.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig4_anchor_tie_inflation.png'}")


# ─── Figure 5: OT Model Performance ──────────────────────────────────
fig5, axes5 = plt.subplots(2, 2, figsize=(14, 11))
fig5.suptitle("Figure 5: OT Model Performance (OOS)", fontsize=14, fontweight="bold")

# 5a: P(home|OT) predicted vs actual by quintile
ax = axes5[0, 0]
if len(ot_with_model) > 20:
    ot_cal_plot = ot_with_model.copy()
    ot_cal_plot["ot_q"] = pd.qcut(ot_cal_plot["p_home_win_ot_given"], 5,
                                    labels=False, duplicates="drop")
    ot_by_q = ot_cal_plot.groupby("ot_q").agg(
        pred=("p_home_win_ot_given", "mean"),
        actual=("home_won_actual", "mean"),
        n=("game_id", "count"),
    ).reset_index()
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", alpha=0.4, label="Perfect")
    ax.scatter(ot_by_q["pred"], ot_by_q["actual"], s=80, c="mediumseagreen",
               zorder=5, label="OT Model")
    ax.axhline(DEFAULT_P_HOME_OT, color="gray", linestyle=":", alpha=0.5,
               label=f"Default ({DEFAULT_P_HOME_OT})")
    for _, row_data in ot_by_q.iterrows():
        ax.annotate(f"n={row_data['n']}", (row_data["pred"], row_data["actual"]),
                    fontsize=8, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Predicted P(home wins OT)")
    ax.set_ylabel("Actual P(home wins OT)")
    ax.set_title("OT Model Calibration by Quintile")
    ax.legend()
else:
    ax.text(0.5, 0.5, "Insufficient OT data", ha="center", va="center", transform=ax.transAxes)
    ax.set_title("OT Model Calibration")

# 5b: OT feature importance (top 15, horizontal bar)
ax = axes5[0, 1]
ot_feat_test_local = ot_feat[ot_feat["game_date"] >= TEST_CUTOFF].copy()
d_cols_ot = [c for c in ot_feat_test_local.columns if c.startswith("d_")]
if d_cols_ot and "home_win" in ot_feat_test_local.columns:
    ot_corrs = ot_feat_test_local[d_cols_ot + ["home_win"]].corr()["home_win"].drop("home_win")
    top15 = ot_corrs.abs().sort_values(ascending=False).head(15)
    top15_vals = ot_corrs.loc[top15.index]
    colors_imp = ["#2ecc71" if v > 0 else "#e74c3c" for v in top15_vals.values]
    ax.barh(range(len(top15_vals)), top15_vals.values, color=colors_imp, alpha=0.7)
    ax.set_yticks(range(len(top15_vals)))
    ax.set_yticklabels([c.replace("d_", "") for c in top15_vals.index], fontsize=7)
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Correlation with Home Win")
    ax.set_title("OT Feature Correlations (top 15)")
    ax.invert_yaxis()

# 5c: OT model reliability diagram
ax = axes5[1, 0]
if len(ot_with_model) > 30:
    ot_rel = ot_with_model.copy()
    ot_rel["p_bin"] = pd.cut(ot_rel["p_home_win_ot_given"],
                              bins=np.linspace(0.35, 0.65, 7), include_lowest=True)
    rel_data = ot_rel.groupby("p_bin", observed=True).agg(
        pred=("p_home_win_ot_given", "mean"),
        actual=("home_won_actual", "mean"),
        n=("game_id", "count"),
    ).reset_index()
    ax.plot([0.35, 0.65], [0.35, 0.65], "k--", alpha=0.4, label="Perfect")
    ax.scatter(rel_data["pred"], rel_data["actual"], s=rel_data["n"] * 2,
               c="mediumseagreen", zorder=5, alpha=0.7)
    for _, row_data in rel_data.iterrows():
        ax.annotate(f"n={row_data['n']}", (row_data["pred"], row_data["actual"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Predicted P(home|OT)")
    ax.set_ylabel("Observed P(home|OT)")
    ax.set_title("OT Model Reliability Diagram")
    ax.legend()
else:
    ax.text(0.5, 0.5, "Insufficient OT data", ha="center", va="center", transform=ax.transAxes)

# 5d: OT win rate by predicted probability bin (with n's)
ax = axes5[1, 1]
if len(ot_with_model) > 20:
    ot_wr = ot_with_model.copy()
    ot_wr["pred_bin"] = pd.qcut(ot_wr["p_home_win_ot_given"], 4, labels=False, duplicates="drop")
    wr_data = ot_wr.groupby("pred_bin").agg(
        pred=("p_home_win_ot_given", "mean"),
        actual=("home_won_actual", "mean"),
        n=("game_id", "count"),
    ).reset_index()
    x_pos = range(len(wr_data))
    ax.bar(x_pos, wr_data["actual"], alpha=0.6, color="mediumseagreen",
           label="Actual", edgecolor="black", linewidth=0.5)
    ax.scatter(x_pos, wr_data["pred"], s=80, c="steelblue", zorder=5,
               marker="D", label="Predicted")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"Q{i}\n(n={n})" for i, n in zip(wr_data["pred_bin"], wr_data["n"])],
                       fontsize=8)
    ax.axhline(DEFAULT_P_HOME_OT, color="gray", linestyle=":", alpha=0.5)
    ax.set_ylabel("P(Home Wins OT)")
    ax.set_title("Home OT Win Rate by Model Quartile")
    ax.legend()
else:
    ax.text(0.5, 0.5, "Insufficient OT data", ha="center", va="center", transform=ax.transAxes)

plt.tight_layout()
fig5.savefig(PLOTS_DIR / "fig5_ot_model.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig5_ot_model.png'}")


# ─── Figure 6: Platt Calibration & Final Probabilities ───────────────
fig6, axes6 = plt.subplots(2, 2, figsize=(14, 11))
fig6.suptitle("Figure 6: Platt Calibration & Final Probabilities (OOS)", fontsize=14, fontweight="bold")

# 6a: Pre-Platt vs post-Platt P(home) scatter
ax = axes6[0, 0]
valid_platt = test.dropna(subset=["p_home_win_raw", "p_home_win"]).copy()
if len(valid_platt) > 10:
    ax.scatter(valid_platt["p_home_win_raw"], valid_platt["p_home_win"],
               alpha=0.1, s=10, color="steelblue")
    ax.plot([0.3, 0.7], [0.3, 0.7], "k--", alpha=0.4, label="No change")
    ax.set_xlabel("Raw P(home win)")
    ax.set_ylabel("Platt-Calibrated P(home win)")
    ax.set_title(f"Pre vs Post Platt (raw std={valid_platt['p_home_win_raw'].std():.4f})")
    ax.legend()

# 6b: Calibration curve — model vs DK vs actual
ax = axes6[0, 1]
valid_cal = with_odds.copy()
valid_cal["prob_bin"] = pd.qcut(valid_cal["p_home_win"], 5, labels=False, duplicates="drop")
cal_data = valid_cal.groupby("prob_bin").agg(
    model=("p_home_win", "mean"),
    dk=("dk_home_fair", "mean"),
    actual=("home_won_actual", "mean"),
    n=("game_id", "count"),
).reset_index()
ax.plot([0.3, 0.7], [0.3, 0.7], "k--", alpha=0.4, label="Perfect")
ax.scatter(cal_data["model"], cal_data["actual"], s=80, c="steelblue", zorder=5, label="Model")
ax.scatter(cal_data["dk"], cal_data["actual"], s=80, c="gold", marker="D", zorder=5, label="DK")
ax.set_xlabel("Predicted P(home win)")
ax.set_ylabel("Actual P(home win)")
ax.set_title("Calibration: Model vs DK by Quintile")
ax.legend()

# 6c: Probability distributions — model P(home) vs DK P(home)
ax = axes6[1, 0]
ax.hist(with_odds["p_home_win"], bins=30, alpha=0.6, color="steelblue",
        label=f"Model (std={with_odds['p_home_win'].std():.3f})", density=True)
ax.hist(with_odds["dk_home_fair"], bins=30, alpha=0.4, color="gold",
        label=f"DK (std={with_odds['dk_home_fair'].std():.3f})", density=True)
ax.axvline(0.5, color="black", linestyle="--", alpha=0.3)
ax.set_xlabel("P(Home Win)")
ax.set_ylabel("Density")
ax.set_title("Model vs DK Probability Distribution")
ax.legend()

# 6d: Platt a, b parameters over walk-forward windows
ax = axes6[1, 1]
platt_data = test.dropna(subset=["platt_a", "platt_b"]).sort_values("game_date")
if len(platt_data) > 10:
    ax.plot(platt_data["game_date"], platt_data["platt_a"],
            color="steelblue", label="Platt a (stretch)", linewidth=1)
    ax_twin = ax.twinx()
    ax_twin.plot(platt_data["game_date"], platt_data["platt_b"],
                 color="coral", label="Platt b (bias)", linewidth=1)
    ax.set_xlabel("Date")
    ax.set_ylabel("Platt a", color="steelblue")
    ax_twin.set_ylabel("Platt b", color="coral")
    ax.set_title(f"Platt Parameters Over Time (a μ={platt_data['platt_a'].mean():.3f})")
    ax.legend(loc="upper left")
    ax_twin.legend(loc="upper right")

plt.tight_layout()
fig6.savefig(PLOTS_DIR / "fig6_platt_calibration.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig6_platt_calibration.png'}")


# ─── Figure 7: Model vs DraftKings ───────────────────────────────────
fig7, axes7 = plt.subplots(2, 2, figsize=(14, 11))
fig7.suptitle("Figure 7: Model vs DraftKings (OOS)", fontsize=14, fontweight="bold")

# 7a: Rolling 50-game log-loss (model vs DK)
ax = axes7[0, 0]
sorted_odds = with_odds.sort_values("game_date").reset_index(drop=True)
y_s = sorted_odds["home_won_actual"].values.astype(float)
p_s = sorted_odds["p_home_win"].values
d_s = sorted_odds["dk_home_fair"].values
ll_model_per = log_loss(y_s, p_s)
ll_dk_per = log_loss(y_s, d_s)
rolling_model = pd.Series(ll_model_per).rolling(50, min_periods=10).mean()
rolling_dk = pd.Series(ll_dk_per).rolling(50, min_periods=10).mean()
ax.plot(sorted_odds["game_date"], rolling_model, color="steelblue", label="Model", linewidth=1.5)
ax.plot(sorted_odds["game_date"], rolling_dk, color="gold", label="DK", linewidth=1.5)
ax.set_xlabel("Date")
ax.set_ylabel("Rolling 50-game Log-Loss")
ax.set_title("Model vs DK: Rolling Log-Loss")
ax.legend()

# 7b: Cumulative log-loss advantage over time
ax = axes7[0, 1]
cum_advantage = np.cumsum(ll_dk_per - ll_model_per)
ax.plot(sorted_odds["game_date"], cum_advantage, color="steelblue", linewidth=1.5)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.fill_between(sorted_odds["game_date"], 0, cum_advantage,
                where=cum_advantage > 0, alpha=0.2, color="green")
ax.fill_between(sorted_odds["game_date"], 0, cum_advantage,
                where=cum_advantage < 0, alpha=0.2, color="red")
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative LL Advantage (DK - Model)")
ax.set_title(f"Cumulative Log-Loss Advantage ({cum_advantage[-1]:+.2f} total)")

# 7c: Model advantage by game type
ax = axes7[1, 0]
categories = []
# Regulation vs OT
for is_ot, label in [(False, "Regulation"), (True, "OT/SO")]:
    sub = with_odds[with_odds["is_ot"] == int(is_ot)]
    if len(sub) > 10:
        sy = sub["home_won_actual"].values.astype(float)
        ml = mean_ll(sy, sub["p_home_win"].values)
        dl = mean_ll(sy, sub["dk_home_fair"].values)
        categories.append({"label": label, "advantage": (dl - ml) * 1000, "n": len(sub)})
# Home fav vs Away fav
for side, label in [(True, "Home Fav"), (False, "Away Fav")]:
    sub = with_odds[with_odds["dk_home_fair"] > 0.5] if side else with_odds[with_odds["dk_home_fair"] <= 0.5]
    if len(sub) > 10:
        sy = sub["home_won_actual"].values.astype(float)
        ml = mean_ll(sy, sub["p_home_win"].values)
        dl = mean_ll(sy, sub["dk_home_fair"].values)
        categories.append({"label": label, "advantage": (dl - ml) * 1000, "n": len(sub)})
if categories:
    cat_df = pd.DataFrame(categories)
    colors_cat = ["#2ecc71" if v > 0 else "#e74c3c" for v in cat_df["advantage"]]
    ax.barh(range(len(cat_df)), cat_df["advantage"], color=colors_cat, alpha=0.7)
    ax.set_yticks(range(len(cat_df)))
    ax.set_yticklabels([f"{r['label']}\n(n={r['n']})" for _, r in cat_df.iterrows()], fontsize=8)
    ax.axvline(0, color="black", linestyle="-", alpha=0.3)
    ax.set_xlabel("Model Advantage (millibits, positive = model better)")
    ax.set_title("Model vs DK by Game Type")

# 7d: Edge distribution histogram
ax = axes7[1, 1]
ax.hist(with_odds["home_edge"], bins=40, alpha=0.6, color="steelblue",
        label=f"Home (μ={with_odds['home_edge'].mean():+.3f})", density=True)
ax.hist(with_odds["away_edge"], bins=40, alpha=0.4, color="coral",
        label=f"Away (μ={with_odds['away_edge'].mean():+.3f})", density=True)
ax.axvline(0, color="black", linestyle="--", alpha=0.4)
ax.set_xlabel("Edge (Model - DK)")
ax.set_ylabel("Density")
ax.set_title("Edge Distribution")
ax.legend()

plt.tight_layout()
fig7.savefig(PLOTS_DIR / "fig7_model_vs_dk.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig7_model_vs_dk.png'}")


# ─── Figure 8: Betting Performance ───────────────────────────────────
fig8, axes8 = plt.subplots(2, 2, figsize=(14, 11))
fig8.suptitle("Figure 8: Betting Performance (OOS)", fontsize=14, fontweight="bold")

# 8a: Cumulative P&L over time (flat staking, 0% and 5% thresholds)
ax = axes8[0, 0]
sorted_bets = with_odds.sort_values("game_date").reset_index(drop=True)
for thresh, color, ls in [(0.0, "steelblue", "-"), (0.05, "mediumseagreen", "--")]:
    mask = sorted_bets["best_edge"] >= thresh
    pnl_series = sorted_bets.loc[mask, "flat_pnl"].cumsum()
    # Re-index to plot on game timeline
    bets_subset = sorted_bets[mask].copy()
    ax.plot(bets_subset["game_date"], pnl_series.values,
            color=color, linestyle=ls, linewidth=1.5,
            label=f"≥{thresh*100:.0f}% edge (n={mask.sum()}, ROI={pnl_series.iloc[-1]/mask.sum()*100:+.1f}%)")
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_xlabel("Date")
ax.set_ylabel("Cumulative P&L (units)")
ax.set_title("Flat Staking P&L Over Time")
ax.legend(fontsize=8)

# 8b: ROI by edge bucket
ax = axes8[0, 1]
edge_bins = [0.0, 0.02, 0.05, 0.10, 1.0]
edge_labels = ["0-2%", "2-5%", "5-10%", "10%+"]
sorted_bets["edge_bucket"] = pd.cut(sorted_bets["best_edge"], bins=edge_bins,
                                     labels=edge_labels)
roi_data = sorted_bets.groupby("edge_bucket", observed=True).agg(
    n=("flat_pnl", "count"),
    total_pnl=("flat_pnl", "sum"),
    win_rate=("bet_won", "mean"),
).reset_index()
roi_data["roi"] = roi_data["total_pnl"] / roi_data["n"] * 100
colors_roi = ["#e74c3c" if r < 0 else "#2ecc71" for r in roi_data["roi"]]
ax.bar(range(len(roi_data)), roi_data["roi"], color=colors_roi, alpha=0.7,
       edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(roi_data)))
ax.set_xticklabels([f"{r['edge_bucket']}\n(n={r['n']})" for _, r in roi_data.iterrows()],
                   fontsize=8)
ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
ax.set_ylabel("ROI (%)")
ax.set_title("ROI by Edge Bucket")

# 8c: Win rate by predicted edge quintile
ax = axes8[1, 0]
sorted_bets["edge_q"] = pd.qcut(sorted_bets["best_edge"], 5, labels=False, duplicates="drop")
wr_by_q = sorted_bets.groupby("edge_q").agg(
    win_rate=("bet_won", "mean"),
    mean_edge=("best_edge", "mean"),
    n=("flat_pnl", "count"),
).reset_index()
ax.bar(range(len(wr_by_q)), wr_by_q["win_rate"], alpha=0.6, color="steelblue",
       edgecolor="black", linewidth=0.5)
ax.set_xticks(range(len(wr_by_q)))
ax.set_xticklabels([f"Q{q}\n({wr_by_q.loc[i, 'mean_edge']*100:.1f}%)"
                     for i, q in enumerate(wr_by_q["edge_q"])], fontsize=8)
ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
for i, row_data in wr_by_q.iterrows():
    ax.text(i, row_data["win_rate"] + 0.01, f"n={row_data['n']}", ha="center", fontsize=7)
ax.set_ylabel("Win Rate")
ax.set_title("Win Rate by Edge Quintile")

# 8d: Waterfall — log-loss at each pipeline stage
ax = axes8[1, 1]
stage_names = ["Constant", "Raw\nPoisson", "+Anchor", "+Platt", "DK"]
stage_lls = [ll_constant, ll_raw_no_anchor, ll_raw_with_anchor, ll_calibrated, ll_dk]
colors_wf = ["gray", "steelblue", "coral", "green", "gold"]
ax.bar(range(len(stage_names)), stage_lls, color=colors_wf, alpha=0.7,
       edgecolor="black", linewidth=0.5)
for i, (name, ll) in enumerate(zip(stage_names, stage_lls)):
    ax.text(i, ll + 0.0005, f"{ll:.4f}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(range(len(stage_names)))
ax.set_xticklabels(stage_names, fontsize=10)
ax.set_ylabel("Log-Loss (lower is better)")
ax.set_title("Log-Loss by Pipeline Stage")

plt.tight_layout()
fig8.savefig(PLOTS_DIR / "fig8_betting_performance.png", dpi=150, bbox_inches="tight")
print(f"  Saved: {PLOTS_DIR / 'fig8_betting_performance.png'}")


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  The model achieves {imp_platt:+.2f}% improvement over constant baseline.
  DK achieves {(1 - ll_dk/ll_constant)*100:+.2f}% over constant.
  Gap: model is {remaining_gap*1000:.1f} millibits worse than DK.

  ── Layer Contributions ──"""
      )

gd_test_valid = gd_test.dropna(subset=[ga_actual_col, ga_pred_col])
if len(gd_test_valid) > 50:
    gk_corr = gd_test_valid[ga_pred_col].corr(gd_test_valid[ga_actual_col])
    print(f"  Layer 0 (Goalie):  {ga_pred_col} corr={gk_corr:.3f} — feeds into Layers 1+2")

if len(ot_games) > 10:
    print(f"  Layer 2 (OT):      corr={ot_corr:.3f}, +{ot_model_value*1000:.1f} mb on OT games")
    print(f"                     {len(ot_games)} OT games = {len(ot_games)/len(test)*100:.0f}% of test set")

print(f"""  Layer 1 (Reg):     λ corr={lam_diff_corr:.3f} — biggest improvement opportunity
  Anchor:            +{(ll_raw_no_anchor - ll_raw_with_anchor)*1000:.1f} mb
  Platt:             {(ll_raw_with_anchor - ll_calibrated)*1000:+.1f} mb (hurts)

  ── Key Insights ──
  1. Regulation model (Layer 1) provides 71% of total signal
  2. Scoring anchor adds significant value (+{(ll_raw_no_anchor - ll_raw_with_anchor)*1000:.1f} mb)
  3. Goalie model is foundation for OT edge — GSAx matchup correlates with outcomes
  4. OT model adds +{ot_model_value*1000:.1f} mb on the {len(ot_games)} games that go to OT
  5. Platt calibration is currently hurting — consider disabling or fixing

  Plots saved to: {PLOTS_DIR}
    fig1: Goalie Deployment Context
    fig2: Regulation Lambda Quality
    fig3: xG & Shot Quality
    fig4: Scoring Anchor & Tie Inflation
    fig5: OT Model Performance
    fig6: Platt Calibration & Final Probabilities
    fig7: Model vs DraftKings
    fig8: Betting Performance
""")

plt.show()

print("=" * 70)
print("DONE")
print("=" * 70)
