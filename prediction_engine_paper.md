# A Bayesian-Calibrated Poisson Model for NHL 3-Way Regulation Markets
### Technical Specification of the Prediction Engine

**Charles Feinstein**
Whizard Analytics

March 2026

---

## Abstract

This paper documents the end-to-end prediction engine behind the NHL 3-way regulation betting model introduced in *Probability Redistribution in 3-Way Sports Betting Markets*. The system is a 4-layer architecture: (1) an XGBoost expected goals model assigns shot-level probabilities from play-by-play geometry; (2) a Ridge regression goalie deployment model adjusts team goals-against baselines for starter quality and fatigue; (3) a 13-quantile XGBoost regression ensemble predicts the full distribution of regulation goals per team, integrating scoring form, opponent strength, schedule context, and goalie matchups across 42 features; and (4) a bivariate Poisson combiner converts predicted goal distributions into 3-way probabilities, with Bayesian tie inflation and home-ice correction. Power devigging extracts fair probabilities from 10+ bookmakers, and edge-weighted staking completes the pipeline. Every parameter value, feature definition, and formula presented here is drawn directly from the production codebase. A walk-forward backtest across 1,593 games with strict temporal separation produces +8.1% ROI on draw bets and +12.5 units of profit. This paper is a companion to the market theory paper; that paper establishes *why* draws are mispriced, this one documents *how* the model exploits it.

---

## 1. Introduction

The companion paper (*Probability Redistribution in 3-Way Sports Betting Markets*) establishes three empirical facts: (1) NHL 3-way regulation markets systematically underprice the draw outcome by 1-4 percentage points after devigging, (2) the mispricing is structural, driven by bettor neglect and the probability conservation constraint, and (3) the magnitude of the mispricing varies over time, particularly during regime changes like the high-parity 2025-26 season.

That paper deliberately treats the prediction engine as a black box. This paper opens the box.

### Architecture Overview

The prediction engine has four layers, each with a distinct role:

| Layer | Model | Input | Output |
|-------|-------|-------|--------|
| **xG Model** | XGBoost binary classifier | Shot-level PBP features | Per-shot P(goal) |
| **Goalie Model (Layer 0)** | Ridge regression | Deployment context + GSAx | Adjusted team GA rate |
| **Regulation Model (Layer 1)** | 13× XGBoost quantile regression | 42 game-level features | Full goal distribution per team |
| **Poisson Combiner (Layer 3)** | Bivariate Poisson + Bayesian calibration | Predicted λ per team | 3-way probabilities |

The xG model and goalie model are upstream feature generators: their outputs feed into the regulation model as inputs. The regulation model is the core predictor. The Poisson combiner converts predictions into probabilities and applies calibration corrections that can only be computed from historical data.

A final edge engine compares model probabilities to devigged market probabilities and generates bet recommendations.

---

## 2. Data Pipeline

### Data Sources

**NHL API** (`api-web.nhle.com`): The primary data source. Three endpoints provide the foundation:

- **Play-by-play events**: Every shot, goal, penalty, faceoff, and stoppage. Each event includes coordinates (x, y), period, time, players involved, and game state (skater counts, score). ~2.4 million events across 7,142 games.
- **Boxscores**: Per-game team summaries including shots, goals, hits, blocks, takeaways, giveaways, faceoff percentages, and penalty minutes.
- **Game schedule**: Game IDs, dates, teams, venue, and game state (final, live, future).

**Team stats** (NHL API, 6 stat categories): Summary, percentages (shooting/save %), penalties, faceoffs, realtime (hits/blocks), and powerplay. These are per-game records aggregated into rolling features.

**Goalie stats**: Per-game boxscores (saves, goals against, shots faced, time on ice) plus season-rolling metrics: GSAx (goals saved above expected), save percentage, and high-danger save percentage. All computed with `shift(1)` to prevent leakage.

**Confirmed starters** (Daily Faceoff): Scraped daily to identify which goalie starts each game. Critical for the goalie deployment model — a backup goalie starting has a measurable effect on goals against.

**3-way odds** (The Odds API): Home regulation win, draw, and away regulation win odds from 10+ US sportsbooks in decimal format. The `h2h_3_way` market key specifically targets regulation-time outcomes. Odds are filtered to exclude in-play lines (home or away decimal < 1.10, or draw decimal > 15.0).

**Standings**: Daily snapshots from the NHL API, providing point percentages and win/loss records used as team quality proxies.

### Data Volume

| Dataset | Records | Date Range |
|---------|---------|------------|
| Play-by-play events | ~2.4M | 2018-2026 |
| Games | 7,142 | 2018-2026 |
| Shot-level xG predictions | ~570K | 2018-2026 |
| Team-game feature rows | ~14,200 | 2018-2026 |
| 3-way odds snapshots | ~85,000 | 2023-2026 |

### Temporal Integrity

Every feature uses `shift(1)` — the rolling average for game *n* is computed from games *1* through *n-1* only. Season-level aggregations (EWM) reset at season boundaries because rosters change. The walk-forward backtest enforces strict temporal separation: models trained on `[start, cutoff)` are tested on `[cutoff, next_cutoff)`, with no information from the test period leaking into training, feature computation, or calibration.

---

## 3. Expected Goals (xG) Model

### Purpose

The xG model assigns a probability of scoring to every unblocked shot based on shot geometry and context. It answers: "Given this shot from this location in this situation, how likely is it to be a goal?" The output is a real number between 0 and 1, typically in the range 0.02-0.20.

This is a **static** model. Shot physics (distance, angle, rebound dynamics) don't change season to season, so the model is trained once and applied to all data. It is not retrained in the walk-forward backtest.

### Training Data

Input: All unblocked shots (goals, shots-on-goal, missed shots) from play-by-play data. Blocked shots are excluded because their recorded coordinates represent where the block occurred, not where the shot originated.

Class imbalance: ~9:1 ratio of non-goals to goals. Handled via `scale_pos_weight = (1 - goal_rate) / goal_rate` in XGBoost.

### Features

The model uses 12 features, all derived from shot geometry and game state. No shooter or goalie identity is included — this is a pure shot quality model.

| # | Feature | Description |
|---|---------|-------------|
| 1 | `shot_distance` | Euclidean distance from shot to net center (89, 0) |
| 2 | `shot_angle` | Degrees from center line to shot location (0° = dead center) |
| 3 | `x_norm` | Absolute value of x-coordinate (robust to attack direction) |
| 4 | `y_abs` | Absolute value of y-coordinate (distance from center line) |
| 5 | `shot_type_code` | Ordinal encoding: wrist=1, snap=2, slap=3, backhand=4, tip-in=5, deflected=6, wrap-around=7 |
| 6 | `is_rebound` | Binary: shot within 3.0 seconds of a prior shot-on-goal |
| 7 | `time_since_last_shot` | Seconds since prior shot in same period (capped at 60) |
| 8 | `strength_code` | 0=even strength, 1=shorthanded, 2=power play |
| 9 | `is_empty_net` | Binary: goalie pulled |
| 10 | `game_seconds` | Total elapsed seconds (periods 1-3 = 1200s each, OT = 300s) |
| 11 | `is_ot` | Binary: period > 3 |
| 12 | `time_remaining_sec` | Seconds remaining in the current period |

**Coordinate normalization**: The home team chooses which end to defend in period 1, so attack direction is not deterministic from period + home/away. Empirically, home teams attack positive-x only ~47% of the time. Using `|x|` is robust: offensive-zone shots (98%+ of all shots) always have |x| near 89, so distance and angle calculations are correct regardless of which end is being attacked. The only error is for rare defensive-zone shots (<2%), which have ~0% goal rate and negligible impact.

### Model Specification

```
XGBoost binary classifier
  objective:        binary:logistic
  max_depth:        4
  learning_rate:    0.03
  n_estimators:     500
  min_child_weight: 80
  subsample:        0.8
  colsample_bytree: 0.7
  reg_alpha:        0.1 (L1)
  reg_lambda:       1.0 (L2)
  scale_pos_weight: ~9.0 (computed from data)
```

### Platt Calibration

Raw XGBoost probabilities are overconfident near the extremes. A Platt scaling layer corrects this:

**calibrated_p = sigmoid(a × logit(p_raw) + b)**

Parameters `a` and `b` are fit on a 30-day holdout at the end of the training period by minimizing log-loss via Nelder-Mead optimization. Typical values: `a ≈ 0.95` (slight softening), `b ≈ 0.0` (minimal shift).

### Output

The model produces `shot_xg.csv`: every unblocked shot in the dataset with a calibrated xG value. These per-shot xG values are aggregated to per-team-per-game xGF (expected goals for) and xGA (expected goals against), which become rolling features in the regulation model.

---

## 4. Goalie Deployment Model (Layer 0)

### Purpose

Different goalies allow different numbers of goals. A team starting their backup on a back-to-back will likely concede more than the same team with their starter on full rest. The goalie model quantifies this adjustment.

### Architecture

Ridge regression predicting **team GA residual**: the difference between a team's actual regulation goals-against in a game and their rolling EWM(span=20) baseline.

**target = reg_GA_actual − team_GA_ewm_20**

The EWM baseline uses `shift(1)` (no leakage) with `min_periods=3`. A positive residual means the team allowed more goals than their recent average.

The final output reconstitutes to a meaningful GA rate:

**goalie_GA_rate = team_GA_ewm_20 + Ridge_prediction**

This produces values in the 2-4 range (typical regulation goals-against), which feeds directly into the regulation model as the `opp_goalie_ga_rate` feature.

### Features

13 features in two categories:

**Deployment context** (9 features): `starter_role_share` (fraction of team starts this season), `start_share_trend` (recent vs season-long share), `goalie_switch` (different starter than last game), `consecutive_starts`, `was_pulled_last_game`, `days_rest`, `is_back_to_back`, `season_games_started`, `games_started_last_14d`.

**Goalie quality** (4 features): `gsax_ewm_20` (20-game rolling goals saved above expected), `gsax_ewm_5` (5-game form), `save_pct_ewm_20` (rolling save percentage), `hd_gsax_ewm_10` (high-danger saves, 10-game window).

### Model Specification

```
RidgeCV with StandardScaler
  alphas searched: [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
  cv: 5-fold
  scoring: neg_mean_squared_error
```

Features are standardized before fitting. The model selects the best alpha via cross-validation.

### Role in the Pipeline

The goalie model's output (`opp_goalie_ga_rate`) appears as a feature in the regulation model. It replaces the naive assumption that every goalie is average. When a backup goalie starts on the second night of a back-to-back, the regulation model sees a higher expected GA rate for the opposing team and adjusts its goal predictions accordingly.

---

## 5. Regulation Goal Model (Layer 1)

This is the core of the prediction engine. Everything upstream (xG, goalie model) feeds into it; everything downstream (Poisson combiner, edge engine) consumes its output.

### Architecture: 13-Quantile XGBoost Regression

Rather than predicting a single number (mean goals), the model predicts 13 points on the **cumulative distribution function** of regulation goals for each team in each game.

**Quantile grid**: [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95]

Each quantile gets its own XGBoost model with objective `reg:quantileerror` and the corresponding `quantile_alpha`. This means 13 separate models are trained per cutoff.

**Why quantile regression?** The Poisson combiner needs a single λ (expected goals) per team. A mean regression model would provide this directly, but quantile regression provides the full distribution shape. Teams with the same mean goals can have very different variance profiles. The quantile approach captures this, and the trapezoidal integration (Section 6) converts the distribution back to λ while preserving information about tail behavior.

### Feature Set

42 features, divided into 23 locked core features and ~20 supplementary features selected via permutation importance.

**Core features** (23, always included):

| Category | Features |
|----------|----------|
| **Own scoring** | `gf_ewm_10`, `ga_ewm_10`, `xgf_ewm_10`, `xga_ewm_10`, `goal_diff_ewm_20`, `shooting_pct_ewm_20`, `save_pct_ewm_20` |
| **Opponent scoring** | `opp_gf_ewm_10`, `opp_ga_ewm_10`, `opp_xgf_ewm_10`, `opp_xga_ewm_10`, `opp_save_pct_ewm_20`, `opp_shooting_pct_ewm_20` |
| **Quality / form** | `points_ewm_20`, `opp_point_pct` |
| **Schedule / rest** | `days_rest`, `opp_days_rest`, `rest_advantage`, `is_back_to_back`, `opp_is_b2b` |
| **Goalie matchup** | `opp_goalie_ga_rate` |
| **Interaction** | `scoring_matchup` |
| **Context** | `is_home` |

**Supplementary features** (20, selected from ~80 candidates via permutation importance):

`opp_pk_save_rate_ewm_20`, `opp_fenwick_ewm_20`, `opp_p3_ga_ewm_10`, `road_trip_game_num`, `efficiency_matchup`, `shutout_rate_20`, `opp_ga_std_10`, `opp_corsi_ewm_20`, `p3_ga_ewm_10`, `opp_trailing_gf_ewm_20`, `streak_value`, `fatigue_advantage`, `opp_pdo_ewm_20`, `opp_pp_goals_ewm_10`, `opp_corsi_behind_ewm_20`, `opp_one_goal_rate_20`, `opp_pim_ewm_10`, `opp_pk_ga_ewm_10`, `opp_blocks_ewm_10`, `opp_goalie_starter_role_share`.

**Feature engineering details**:

- All rolling features use exponentially weighted means (EWM) with `shift(1)` to prevent leakage
- EWM resets at season boundaries (rosters change between seasons)
- Opponent features are the opponent's own rolling stats, not relative to the predicting team
- `scoring_matchup = gf_ewm_10 + opp_ga_ewm_10` (how much the team scores vs how much the opponent allows)

### Hyperparameters

Tuned via Optuna (50 trials, TPE sampler) optimizing mean pinball loss across all 13 quantiles on a 90-day validation holdout:

```
XGBoost quantile regression (per quantile)
  objective:        reg:quantileerror
  quantile_alpha:   [0.05, 0.10, ..., 0.95]
  max_depth:        2
  learning_rate:    0.017
  n_estimators:     800
  min_child_weight: 133
  subsample:        0.912
  colsample_bytree: 0.434
  reg_alpha:        0.114 (L1)
  reg_lambda:       0.289 (L2)
```

Key observations: `max_depth=2` (extremely shallow trees) and `min_child_weight=133` (large minimum leaf size) indicate the model benefits from heavy regularization. Goal counts are noisy; deep trees overfit to random variation.

### Training Protocol

1. **Temporal split**: Train on `[start, cutoff)`, test on `[cutoff, next_cutoff)`. Two cutoffs: 2024-10-01 and 2025-10-01.
2. **Season weight decay**: Recent seasons weighted higher. Weight = 0.5^(seasons_ago / 2.0). A season 2 years old gets weight 0.5; 4 years old gets 0.25.
3. **Jitter**: Discrete goal counts are jittered by ±0.5 uniform noise during training to help the quantile models produce smoother distributions.
4. **Feature selection**: If LOCKED_FEATURES is set (production mode), permutation importance is skipped and the exact 42 features are used. Otherwise, 23 core features are always included and the top 20 supplementary features (by permutation importance, 50 repeats) are added.
5. **Conformal calibration**: Implemented but **disabled** in production. The raw model's lambda bias is only +0.014 goals, meaning it is already well-calibrated. The 30-day conformal holdout was returned to the training set for more data.
6. **Monotonicity enforcement**: After prediction, quantile values are post-processed so that Q(0.05) ≤ Q(0.10) ≤ ... ≤ Q(0.95). Violations are rare but possible because each quantile model is trained independently.

### Output

The model produces `regulation_predictions.csv`: one row per team-game with 13 quantile columns (`q05` through `q95`) and metadata (game_id, team, date, actual goals).

---

## 6. Poisson Combiner (Layer 3)

The Poisson combiner converts per-team goal distributions into 3-way game probabilities. This is where the model meets the market.

### Step 1: Quantile → Lambda

Each team's 13 quantile predictions are converted to a single expected goals parameter λ via trapezoidal integration:

**λ = ∫₀¹ Q(p) dp**

The integration uses a 15-point grid. Since the model only predicts quantiles from 0.05 to 0.95, the tails are extrapolated:

- **Q(0)** = max(0, Q(0.05) − [Q(0.10) − Q(0.05)])
- **Q(1)** = Q(0.95) + [Q(0.95) − Q(0.90)]

The full grid is [0, 0.05, 0.10, ..., 0.95, 1.0] with corresponding Q values. Trapezoidal integration over this grid yields E[X], the expected regulation goals.

Lambda is clipped to [0.1, 15.0] as a safety bound.

### Step 2: Scoring Anchor (Disabled)

A multiplicative scoring anchor was implemented to correct for league-wide scoring environment shifts (e.g., if goals are trending higher than predicted across all games). It computes `actual_mean / predicted_mean` over a 60-day lookback window of at least 100 games.

In practice, the model's lambda bias is only +0.014, so the anchor provides negligible correction and introduces noise. It is **disabled** in production (`ANCHOR_STRENGTH = 0.0`). All games use an anchor of 1.0.

### Step 3: Bivariate Poisson Joint Distribution

Given λ_home and λ_away, the model computes the joint probability of every possible scoreline (0-0 through 12-12) using a bivariate Poisson distribution.

**Independent case** (ρ = 0): The joint PMF is the outer product of two independent Poisson distributions:

**P(H=h, A=a) = P_H(h) × P_A(a)**

where P_H(h) = e^(-λ_h) × λ_h^h / h!

**Bivariate case** (ρ > 0): A shared component model introduces positive correlation between the two teams' goal totals:

- X_home = Y + Z_home
- X_away = Y + Z_away
- Y ~ Poisson(λ_shared), Z_home ~ Poisson(λ_h − λ_shared), Z_away ~ Poisson(λ_a − λ_shared)
- λ_shared = ρ × min(λ_home, λ_away)

**P(H=h, A=a) = Σ_k P(Y=k) × P(Z_h = h−k) × P(Z_a = a−k)**

The correlation parameter ρ is estimated from historical data via method of moments:

**ρ = Cov(X_h, X_a) / mean(min(λ_h, λ_a))**

clipped to [0, 0.15]. Hockey games have a small but real positive correlation — high-event games tend to produce more goals for both teams.

From the 13×13 joint PMF grid:

- **P(home reg win)** = Σ P(H=h, A=a) for h > a
- **P(away reg win)** = Σ P(H=h, A=a) for h < a
- **P(tie)_raw** = Σ P(H=h, A=h) = trace of joint matrix

### Step 4: Bayesian Tie Inflation

Raw Poisson P(tie) systematically underestimates the actual overtime rate. Across the dataset, raw P(tie) averages ~16.3% while the actual OT rate is ~22-26%. The tie inflation factor corrects this gap.

The model uses a **Bayesian approach** that balances historical priors with current-season evidence:

**Prior**: Computed from all pre-cutoff games. The prior OT rate and prior P(tie)_raw mean establish the baseline inflation ratio. The prior is capped at 1,500 effective games (~2 NHL seasons) to prevent over-anchoring to distant history.

**Posterior**: As the current season accumulates games (minimum 20 to use season data), the Bayesian update shrinks toward the prior:

**shrunk_OT_rate = (prior_rate × prior_games + season_OT_count) / (prior_games + n_season)**

**tie_inflation = shrunk_OT_rate / season_P(tie)_raw_mean**

This produces inflation factors typically in the range 1.30-1.45. Critically, the factor adapts within a season: the 2025-26 season's higher parity (25.8% OT rate) is captured as games accumulate, pushing the inflation from ~1.31x toward ~1.40x.

After inflation:

**P(tie)_calibrated = P(tie)_raw × tie_inflation**

The excess probability mass is subtracted from P(home reg) and P(away reg) proportionally:

**scale = (1 − P(tie)_calibrated) / (P(home reg) + P(away reg))**
**P(home reg)_final = P(home reg) × scale**
**P(away reg)_final = P(away reg) × scale**

All three probabilities sum to exactly 1.0.

### Step 5: Home-Ice Correction

The model applies a fixed home-ice adjustment in logit space. Computed from pre-cutoff regulation-decided games only:

1. Compute actual home win share among regulation-decided games
2. Compute model's mean predicted home share (from raw Poisson, before tie inflation)
3. **home_ice_shift = logit(actual) − logit(predicted)**

Applied per-game:

**logit(home_share_adj) = logit(home_share_raw) + home_ice_shift**

The adjusted home share redistributes the regulation probability mass (1 − P(tie)) between home and away. This is a small correction (typically 0.02-0.05 logit units) but improves calibration of the home/away split.

---

## 7. Power Devigging & Edge Calculation

### Power Devig

Bookmaker odds include a margin (vig). To compare model probabilities to market probabilities, the vig must be removed. The model uses the **power method**, which respects the relative pricing structure across all three outcomes.

Given implied probabilities p_1, p_2, p_3 (where p_i = 1/decimal_odds_i and p_1 + p_2 + p_3 > 1):

**Find k such that p_1^k + p_2^k + p_3^k = 1**

Solved via bisection search: 200 iterations, tolerance 1e-10, search range k ∈ [1.0, 50.0].

Fair probabilities: **fair_i = p_i^k / Σ p_j^k**

The power method is preferred over alternatives (multiplicative, additive, Shin) because it maps longer odds to proportionally larger devig adjustments, matching the empirical observation that bookmakers load more vig onto longshot outcomes.

### Edge Calculation

For each game-bookmaker pair:

**edge_i = model_prob_i − fair_prob_i**

A positive edge means the model believes the outcome is more likely than the market does.

### Staking Rules

The model uses a blend strategy with independent thresholds for draw and home bets:

| Side | Min Edge | Stake Formula | Cap |
|------|----------|---------------|-----|
| **Draw** | ≥ 2.5% | edge × 25 | 1.5 units |
| **Home** | 2.5% - 5.0% | edge × 20 | 1.0 units |
| **Away** | Not bet | — | — |

Away regulation bets are not taken because the model's edge on away wins is a mirror of draw underpricing — when draws are underpriced, the away win is typically overpriced by a similar amount, making away bets negative-EV after vig.

The draw stake multiplier (25×) is aggressive by Kelly criterion standards but reflects the structural nature of the edge: this is not a noisy signal that could flip sign, but a persistent structural mispricing with a well-estimated magnitude.

---

## 8. Walk-Forward Backtest

### Protocol

The backtest enforces strict temporal separation at every level:

1. **Model training**: Models trained on `[start, cutoff)`, tested on `[cutoff, next_cutoff)`. Two test windows:
   - **2024-25**: cutoff 2024-10-01 → 2025-10-01
   - **2025-26**: cutoff 2025-10-01 → present

2. **Feature computation**: All rolling features (EWM, rolling windows) use `shift(1)`. No game's own outcome appears in its features.

3. **Calibration parameters**: Bivariate ρ and home-ice shift are computed from pre-cutoff data only (fixed for the test season). Tie inflation uses the Bayesian update with pre-cutoff prior + season-to-date evidence — the same per-date loop that would run in live deployment.

4. **Per-date loop**: For each test date, the model recomputes:
   - Scoring anchors (60-day lookback of completed games, currently disabled)
   - Bayesian tie inflation (prior + season-to-date OT rate)
   - These are the only parameters that update within a test season

5. **Odds**: Matched by game_id and bookmaker. Live/in-play odds filtered out. Multiple bookmakers per game; edges computed against each independently.

### What Is NOT Retrained

The xG model is static (trained once). The goalie model and regulation model are trained per-cutoff but not updated within a test season. Only the calibration parameters (tie inflation, scoring anchor) update daily, using only historical information available on each date.

---

## 9. Results & Diagnostics

### Lambda Accuracy

The regulation model's predicted λ (expected goals) is evaluated against actual regulation goals:

- **Mean lambda bias**: +0.014 (model predicts 0.014 more goals than actually scored, on average)
- This is negligible — less than 0.5% of the mean goal count (~2.8 per team per game)
- The near-zero bias is why the scoring anchor and conformal calibration are disabled: they add noise without improving accuracy

### Probability Calibration

The model's P(OT) predictions are compared against actual overtime frequency in quintile bins. A well-calibrated model has predicted ≈ actual in each bin. The Bayesian tie inflation ensures that the overall P(OT) level tracks the actual OT rate, while the per-game variation comes from the Poisson model (teams with similar λ values produce more ties).

### Tie Inflation Adaptation

The Bayesian tie inflation factor adapts as the season progresses:

| Season | Early (Nov) | Mid (Jan) | Late (Mar) | Actual OT% |
|--------|-------------|-----------|------------|-------------|
| 2024-25 | ~1.31x | ~1.31x | ~1.31x | 22.3% |
| 2025-26 | ~1.31x | ~1.37x | ~1.40x | 25.8% |

The 2025-26 season's higher OT rate is a regime change — more parity means more close games reaching overtime. The Bayesian update detects this shift and inflates the tie probability accordingly, without overreacting to small samples early in the season.

### Backtest Performance

Across 1,593 post-calibration games (2024-25 and 2025-26 combined):

- **Total profit**: +12.5 units
- **ROI**: +8.1% on edge-weighted bets
- **Draw bet win rate**: ~25% (at average odds of ~4.5, breakeven is ~22%)

These numbers are from the companion paper's backtest. The methodology is identical to what is described in Section 8 above.

---

## 10. Worked Example

To make the pipeline concrete, here is an end-to-end prediction chain for a single hypothetical game: **NYR @ BOS, March 2026**.

### Step 1: Feature Computation

The regulation feature pipeline computes 42 features for each team. Selected values for Boston (home):

| Feature | Value | Description |
|---------|-------|-------------|
| `is_home` | 1 | Home team |
| `gf_ewm_10` | 3.12 | 10-game scoring average |
| `ga_ewm_10` | 2.45 | 10-game goals-against average |
| `xgf_ewm_10` | 2.89 | Expected goals for |
| `opp_ga_ewm_10` | 2.78 | NYR's goals-against average |
| `opp_goalie_ga_rate` | 2.61 | NYR starter's adjusted GA rate |
| `days_rest` | 2 | Days since last game |
| `is_back_to_back` | 0 | Not a back-to-back |
| `scoring_matchup` | 5.90 | BOS_gf + NYR_ga = offensive matchup |

### Step 2: Quantile Predictions

The 13 XGBoost models each predict one quantile for Boston's regulation goals:

| Q(0.05) | Q(0.10) | Q(0.20) | Q(0.30) | Q(0.50) | Q(0.70) | Q(0.80) | Q(0.90) | Q(0.95) |
|---------|---------|---------|---------|---------|---------|---------|---------|---------|
| 0.82 | 1.15 | 1.58 | 1.92 | 2.61 | 3.35 | 3.72 | 4.28 | 4.71 |

### Step 3: Quantile → Lambda

Trapezoidal integration with tail extrapolation:
- Q(0) = max(0, 0.82 − (1.15 − 0.82)) = max(0, 0.49) = 0.49
- Q(1) = 4.71 + (4.71 − 4.28) = 5.14
- **λ_BOS = ∫₀¹ Q(p) dp ≈ 2.72**

Similarly for NYR: **λ_NYR ≈ 2.48**

### Step 4: Bivariate Poisson

With ρ ≈ 0.03 (estimated from pre-cutoff data):

- P(BOS reg win) = 0.396
- P(NYR reg win) = 0.299
- P(tie)_raw = 0.169

### Step 5: Tie Inflation

Current tie_inflation = 1.38 (Bayesian posterior, late March 2026):

- P(tie)_calibrated = 0.169 × 1.38 = 0.233
- Remaining regulation mass = 1 − 0.233 = 0.767
- Raw regulation total = 0.396 + 0.299 = 0.695
- scale = 0.767 / 0.695 = 1.104
- P(BOS reg) = 0.396 × 1.104 = 0.437
- P(NYR reg) = 0.299 × 1.104 = 0.330
- P(OT) = 0.233

### Step 6: Home-Ice Correction

With home_ice_shift ≈ 0.03 logit units:

- home_share_raw = 0.437 / (0.437 + 0.330) = 0.570
- logit(0.570) = 0.281
- logit_adj = 0.281 + 0.03 = 0.311
- home_share_adj = sigmoid(0.311) = 0.577
- P(BOS reg)_final = 0.767 × 0.577 = 0.443
- P(NYR reg)_final = 0.767 × 0.423 = 0.324

**Final 3-way: BOS 44.3% | OT 23.3% | NYR 32.4%**

### Step 7: Edge Calculation

Suppose DraftKings offers: BOS −145 / Draw +370 / NYR +260 (decimal: 1.690 / 4.700 / 3.600).

Implied: 0.592 / 0.213 / 0.278 (sum = 1.083, overround = 8.3%)

Power devig (find k ≈ 1.18): fair = 0.547 / 0.196 / 0.257

Edges:
- Home: 0.443 − 0.547 = −0.104 (no bet)
- Draw: 0.233 − 0.196 = +0.037 (3.7% edge → bet)
- Away: 0.324 − 0.257 = +0.067 (away bets disabled)

Draw stake: min(0.037 × 25, 1.5) = 0.925 units at decimal 4.700.

---

## 11. Limitations & Future Work

### Disabled Components

Several model components were implemented, tested, and deliberately disabled because they hurt or failed to help performance:

- **Scoring anchor** (`ANCHOR_STRENGTH = 0.0`): The model's lambda bias is only +0.014. The anchor adds noise without improving accuracy.
- **OT model blend** (`OT_BLEND_WEIGHT = 0.0`): A separate OT prediction model was built but a constant 0.518 home OT win probability outperforms the model's predictions. The constant is used.
- **Conformal calibration**: Removed from the training pipeline. The raw model is well-calibrated, and the conformal holdout data is better used for training.
- **Platt scaling on win probabilities** (`PLATT_ENABLED = False`): Implemented but disabled. The home-ice logit shift provides the same correction with fewer parameters.
- **Lambda calibration** (`LAMBDA_CAL_ENABLED = False`): A multiplicative per-game lambda correction was tested and disabled.

### Known Limitations

- **Bivariate correlation**: ρ is estimated and used (typically 0.02-0.05) but its effect on final probabilities is small. The independent Poisson (ρ = 0) produces nearly identical results.
- **No in-season retraining**: Models are trained at season cutoffs and not updated. A team that undergoes a major trade-deadline transformation in February is still being evaluated with October-trained weights.
- **Goalie injury uncertainty**: The model requires a confirmed starter. If the starter is announced after the features are computed, the model uses a default (team average) goalie quality.
- **Away bets not exploited**: The model identifies positive away edges (mirror of draw underpricing) but does not bet them. This is conservative — the away edge is correlated with the draw edge, so betting both would introduce correlation risk.

### Future Work

- **In-season model updates**: Retrain the regulation model monthly or after major roster changes.
- **Live line movement**: Track odds changes from open to close to identify sharp money patterns.
- **Player-level features**: Individual skater impacts (injuries, line combinations) beyond the goalie deployment model.
- **Expanded markets**: Apply the same framework to other 3-way markets (e.g., soccer) where the draw is also structurally underpriced.

---

## References

1. NHL API documentation: `api-web.nhle.com`
2. The Odds API: `the-odds-api.com`
3. Daily Faceoff starting goalies: `dailyfaceoff.com`
4. XGBoost: Chen, T. and Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *Proc. 22nd ACM SIGKDD*.
5. Power devigging methodology: Štrumbelj, E. (2014). "On determining probability forecasts from betting odds." *International Journal of Forecasting*.
6. Bivariate Poisson: Karlis, D. and Ntzoufras, I. (2003). "Analysis of sports data by using bivariate Poisson models." *Journal of the Royal Statistical Society: Series D*.

---

## Disclaimer

This paper documents a quantitative model for analytical purposes. Sports betting involves financial risk. Past backtest performance does not guarantee future results. The model's edge depends on market conditions that can change. The author makes no guarantees about the profitability of any betting strategy derived from this work.

---

*Companion paper: "Probability Redistribution in 3-Way Sports Betting Markets: Structural Mispricing of the Draw Outcome in NHL Regulation Markets" (Feinstein, 2026)*
