# NHL 3-Way Regulation Market: Draw Edge Model

**+19.1% live ROI across 137 bets · +8.1% backtest ROI across 1,593 games · March 2026**

A full-stack quantitative sports betting system that identifies and exploits structural mispricing of the draw outcome in NHL 3-way regulation markets. Built from the ground up: play-by-play data collection, expected goals modeling, team goal prediction, Bayesian probability calibration, live deployment, and cross-bookmaker edge extraction.

---

## The Core Idea

In a 3-way betting market, probabilities must sum to 100%. When bookmakers misprice one outcome, the error is absorbed by the others, and the draw is the one nobody is correcting. Recreational bettors pick teams. Sharp money flows into home and away. The draw sits neglected, underpriced by 1-4 percentage points on every game, on every book, every night.

This project quantifies that mispricing, builds a model that prices draws more accurately than the market, and deploys it live against 20+ bookmakers simultaneously.

---

## Results

### Backtest (Walk-Forward, 1,593 Games)

| Season | Games | ROI |
|--------|-------|-----|
| 2024-25 OOS | ~800 | +6.8% |
| 2025-26 OOS | ~800 | +9.3% |
| **Combined** | **1,593** | **+8.1%** |

Strict temporal separation: models trained on data before each season's cutoff, tested on games after. No lookahead in features, calibration, or staking.

### Live Deployment (March 5–22, 2026)

| Metric | Value |
|--------|-------|
| Games graded | 143 |
| Draw bets placed | 137 |
| Win rate | 28.5% (breakeven: 23.8%) |
| Average decimal odds | 4.21 |
| **Flat ROI** | **+19.1%** |
| **Flat profit** | **+26.15 units** |

Bets placed before puck drop. Results graded the following morning. No post-hoc filtering.

---

## Architecture

The prediction engine has four layers:

| Layer | Model | Input | Output |
|-------|-------|-------|--------|
| **xG Model** | XGBoost binary classifier | Shot coordinates, strength, rebound, shot type | P(goal) per shot |
| **Goalie Model** | Ridge regression | GSAx, deployment context, rest, consecutive starts | Adjusted team GA rate |
| **Regulation Model** | 13× XGBoost quantile regression | 42 game-level features including xG, goalie matchup, schedule | Full goal distribution per team |
| **Poisson Combiner** | Bivariate Poisson + Bayesian calibration | Predicted λ per team | 3-way fair probabilities |

A power devig engine extracts fair probabilities from raw bookmaker odds across 20+ books. The edge filter flags draw bets where model probability exceeds fair market probability by 2.5%+.

---

## Papers

This project is documented in three companion papers, written in sequence as the system was built and deployed.

### Paper 1: [Probability Redistribution in 3-Way Sports Betting Markets](probability_redistribution_paper.md)

The market theory paper. Establishes why NHL draw outcomes are structurally mispriced: probability mass conservation, bettor neglect, market illiquidity, and the inability of bookmakers relying on long-run averages to adapt to regime changes like the 2025-26 parity shift. Quantifies the mispricing across 7,142 games and 7 sportsbooks.

> *"The draw acts as the residual absorber of probability mass that nobody is actively trading."*

### Paper 2: [A Bayesian-Calibrated Poisson Model for NHL 3-Way Regulation Markets](prediction_engine_paper.md)

The technical paper. Opens the black box: every feature, every hyperparameter, every formula in the production system. xG model (AUC 0.776), goalie deployment model, 13-quantile regulation model, Bayesian tie inflation, home-ice correction, power devigging, and staking methodology. Includes a worked end-to-end prediction example.

> *"That paper establishes why draws are mispriced. This one documents how the model exploits it."*

### Paper 3: [Live Deployment Validation](live_deployment_paper.md)

The validation paper. First-month out-of-sample results: 143 games, 137 draw bets, +19.1% ROI. Includes a cross-bookmaker analysis showing sharp books (FanDuel: -15.6%) have tightened draw pricing while recreational and international books remain materially behind. Discusses model calibration, the OT rate regime continuation, and what the data says the model actually gets right and wrong.

> *"The draw is still being neglected. The market has not closed the gap. The strategy works."*

---

## Selected Figures

<table>
<tr>
<td><img src="figures/fig01_probability_vessels.png" width="380"/><br><sub>Probability constraint: market vs. actual draw rate</sub></td>
<td><img src="figures/fig02_market_vs_actual_vs_model.png" width="380"/><br><sub>Model, market, and actual draw probabilities</sub></td>
</tr>
<tr>
<td><img src="figures/fig06_bookmaker_draw_pricing.png" width="380"/><br><sub>Draw pricing dispersion across sportsbooks</sub></td>
<td><img src="figures/fig08_equity_curve.png" width="380"/><br><sub>Cumulative backtest equity curve</sub></td>
</tr>
<tr>
<td><img src="figures/fig05_tie_inflation.png" width="380"/><br><sub>Bayesian tie inflation tracking the 2025-26 OT shift</sub></td>
<td><img src="figures/fig11_seasonal_shift.png" width="380"/><br><sub>OT rate regime change: 2024-25 vs. 2025-26</sub></td>
</tr>
</table>

---

## Key Methodology Points

**Expected Goals**: Shot-level xG model trained on 2.4M+ play-by-play events. 12 features from shot geometry and game state, with no shooter or goalie identity. Static model applied across all seasons.

**Goalie Deployment**: Ridge regression predicting GA residual from 13 context and quality features. Captures backup starters, fatigue, and consecutive-start fatigue. Output feeds the regulation model as `opp_goalie_ga_rate`.

**Quantile Regression**: 13 separate XGBoost models per cutoff, each trained on a different quantile. Trapezoidal integration converts the distribution to a single λ. Preserves tail behavior that mean regression discards.

**Bayesian Tie Inflation**: Raw Poisson underestimates overtime probability by 5-7 percentage points. The Bayesian updater balances a historical prior (capped at 1,500 effective games) against current-season evidence, adapting within a season without overreacting to small samples.

**Power Devigging**: Finds the exponent k such that p₁ᵏ + p₂ᵏ + p₃ᵏ = 1. Preferred over multiplicative or additive methods because it proportionally reduces vig on longshot outcomes, matching how bookmakers actually distribute their margin.

**Line Shopping**: The model computes edges against each bookmaker independently. The same game can clear the edge threshold at 4% on one book and miss entirely at another. Cross-bookmaker ROI shows a 35+ percentage point spread in this period; sharp books have adapted, soft books haven't.

---

## Data Sources

- **NHL API** (`api-web.nhle.com`): Play-by-play events, boxscores, team stats, standings, goalie stats
- **Daily Faceoff**: Confirmed starting goalies
- **The Odds API**: 3-way regulation odds from 20+ US and international sportsbooks

---

## Repository Structure

```
├── probability_redistribution_paper.md   # Paper 1: market theory
├── prediction_engine_paper.md            # Paper 2: model specification
├── live_deployment_paper.md              # Paper 3: live validation
├── figures/                              # All paper figures
├── generate_paper_figures.py             # Figure generation script
├── scrapers/                             # NHL API + odds data collection
├── features/                             # Feature engineering (xG, goalie, regulation, OT)
├── models/                               # Model classes (xG, goalie, regulation, combiner)
├── backtest/                             # Walk-forward backtest engine
├── data/predictions/                     # Live prediction logs + tracking results
├── predict_today.py                      # Daily production pipeline
└── track_results.py                      # Results grading and P&L tracking
```

---

## Setup

```bash
pip install -r requirements.txt
python results_dashboard.py   # → http://127.0.0.1:8060
```

**Note on paths**: The pipeline scripts (`scrapers/`, `features/`, `models/`, `predict_today.py`, etc.) use `PROJECT_DIR` from `config.py`, which is set to a local Windows path. If you want to run the full pipeline, update `PROJECT_DIR` in `config.py` to point to your local clone. The results dashboard (`results_dashboard.py`) uses relative paths and runs out of the box without any configuration.

---

## Disclaimer

This project is presented for analytical and portfolio purposes. Sports betting involves financial risk. Documented performance, whether from a backtest or live deployment, does not guarantee future results. The edge quantified here depends on market conditions, bookmaker pricing behavior, and the NHL's current overtime rate, all of which are subject to change.

---

*Whizard Analytics · Charles Feinstein · 2026*
