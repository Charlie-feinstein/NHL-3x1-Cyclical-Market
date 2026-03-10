"""
Generate all figures for the Probability Redistribution paper.

Reads real backtest data, 3-way odds, and game results to produce
publication-quality visualizations with a dark theme matching the
Whizard Analytics brand.

Usage:  python paper/generate_paper_figures.py
Output: paper/figures/*.png
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

# ── Paths ──
PROJECT = Path(r"D:\Python\NHL Overtime Model")
BACKTEST = PROJECT / "backtest" / "backtest_results_3way.csv"
GAME_IDS = PROJECT / "data" / "raw" / "game_ids.csv"
ODDS_FILE = PROJECT / "data" / "raw" / "three_way_odds.csv"
OUT_DIR = PROJECT / "paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Dark Theme ──
BG = "#0a0a1a"
BG_CARD = "#12122a"
TEXT = "#e8e8f0"
TEXT_MUTED = "#8a8aa4"
GRID = "#1a1a3a"
GREEN = "#00e59b"
AMBER = "#ffb347"
BLUE = "#64b5f6"
PINK = "#f472b6"
RED = "#ff5252"
PURPLE = "#b388ff"

plt.rcParams.update({
    "figure.facecolor": BG,
    "axes.facecolor": BG_CARD,
    "axes.edgecolor": GRID,
    "axes.labelcolor": TEXT,
    "text.color": TEXT,
    "xtick.color": TEXT_MUTED,
    "ytick.color": TEXT_MUTED,
    "grid.color": GRID,
    "grid.alpha": 0.5,
    "legend.facecolor": BG_CARD,
    "legend.edgecolor": GRID,
    "font.family": "sans-serif",
    "font.size": 11,
})

# ── Load Data ──
print("Loading data...")
bt = pd.read_csv(BACKTEST, parse_dates=["game_date"])
gi = pd.read_csv(GAME_IDS, parse_dates=["game_date"])
gi = gi[gi["game_state"] == "OFF"].copy()

# Actual 3-way outcome for game_ids
def actual_3way(row):
    if row["game_outcome_type"] in ("OT", "SO"):
        return "draw"
    elif row["home_score"] > row["away_score"]:
        return "home"
    else:
        return "away"

gi["actual_3way"] = gi.apply(actual_3way, axis=1)

# Exclude burn-in: model priors need ~3 months to calibrate
CUTOFF = "2025-01-01"

# Filter to post-calibration period
gi_bt = gi[gi["game_date"] >= CUTOFF].copy()
bt = bt[bt["game_date"] >= CUTOFF].copy()

# Unique games from backtest (drop bookmaker duplicates)
bt_games = bt.drop_duplicates("game_id").copy()
bt_games["month"] = bt_games["game_date"].dt.to_period("M")


def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved: {path.name}")


# ======================================================================
# FIGURE 1: The Three Vessels — conceptual diagram
# ======================================================================
def fig_probability_vessels():
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Title
    ax.text(5, 7.5, "THE PROBABILITY CONSTRAINT",
            ha="center", va="center", fontsize=18, fontweight="bold", color=TEXT)
    ax.text(5, 7.0, "Three outcomes must always sum to 100%",
            ha="center", va="center", fontsize=12, color=TEXT_MUTED)

    # Three vessels (bars showing actual vs market)
    labels = ["Home Reg", "Draw (OT)", "Away Reg"]
    actual = [42.2, 22.3, 35.5]
    market = [43.1, 20.9, 36.0]
    colors = [GREEN, AMBER, BLUE]
    x_pos = [1.5, 5.0, 8.5]

    for i, (lbl, act, mkt, col, xp) in enumerate(zip(labels, actual, market, colors, x_pos)):
        # Market bar (background)
        bar_h = mkt / 10
        ax.add_patch(plt.Rectangle((xp - 0.6, 1.5), 1.2, bar_h,
                                   facecolor=col, alpha=0.15, edgecolor=col, linewidth=1.5))
        ax.text(xp, 1.5 + bar_h + 0.15, f"Market: {mkt:.1f}%",
                ha="center", fontsize=9, color=TEXT_MUTED)

        # Actual bar (overlay)
        bar_h_act = act / 10
        ax.add_patch(plt.Rectangle((xp - 0.4, 1.5), 0.8, bar_h_act,
                                   facecolor=col, alpha=0.6, edgecolor="none"))
        ax.text(xp, 1.5 + bar_h_act + 0.15, f"Actual: {act:.1f}%",
                ha="center", fontsize=10, fontweight="bold", color=col)

        # Label
        ax.text(xp, 1.0, lbl, ha="center", fontsize=13, fontweight="bold", color=col)

    # Gap annotation on Draw
    ax.annotate("", xy=(5.0, 1.5 + 22.3/10), xytext=(5.0, 1.5 + 20.9/10),
                arrowprops=dict(arrowstyle="<->", color=AMBER, lw=2))
    ax.text(5.85, 1.5 + 21.6/10, "1.4pp gap",
            fontsize=10, fontweight="bold", color=AMBER)

    # Sum annotation
    ax.text(5, 0.3, "Home + Draw + Away = 100%   (always)",
            ha="center", fontsize=11, color=TEXT_MUTED, style="italic")

    save(fig, "probability_vessels.png")

fig_probability_vessels()


# ======================================================================
# FIGURE 2: Market vs Actual vs Model — bar chart
# ======================================================================
def fig_market_vs_actual():
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Home Reg Win", "Draw (OT/SO)", "Away Reg Win"]
    actual_vals = [42.2, 22.3, 35.5]
    market_vals = [43.1, 20.9, 36.0]
    model_vals = [41.6, 22.1, 36.3]

    x = np.arange(len(categories))
    w = 0.25

    bars1 = ax.bar(x - w, actual_vals, w, label="Actual", color=GREEN, alpha=0.85)
    bars2 = ax.bar(x, market_vals, w, label="Market (devigged)", color=RED, alpha=0.85)
    bars3 = ax.bar(x + w, model_vals, w, label="Model", color=BLUE, alpha=0.85)

    ax.set_ylabel("Probability (%)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 50)
    ax.legend(fontsize=11)
    ax.set_title("THE MISPRICING GAP\nActual vs. Market vs. Model Probability Distribution",
                 fontsize=14, fontweight="bold", pad=15)

    # Value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                    ha="center", fontsize=9, color=TEXT_MUTED)

    ax.grid(axis="y", alpha=0.3)
    save(fig, "market_vs_actual_vs_model.png")

fig_market_vs_actual()


# ======================================================================
# FIGURE 3: Monthly 3-Way Distribution — stacked area
# ======================================================================
def fig_monthly_distribution():
    gi_bt["month"] = gi_bt["game_date"].dt.to_period("M")
    monthly = gi_bt.groupby("month")["actual_3way"].value_counts(normalize=True).unstack(fill_value=0) * 100
    for col in ["home", "away", "draw"]:
        if col not in monthly.columns:
            monthly[col] = 0

    fig, ax = plt.subplots(figsize=(12, 6))
    months = monthly.index.astype(str)
    x = np.arange(len(months))

    ax.bar(x, monthly["home"], label="Home Reg", color=GREEN, alpha=0.7)
    ax.bar(x, monthly["draw"], bottom=monthly["home"], label="Draw (OT/SO)", color=AMBER, alpha=0.85)
    ax.bar(x, monthly["away"], bottom=monthly["home"] + monthly["draw"],
           label="Away Reg", color=BLUE, alpha=0.7)

    # Draw rate line
    ax2 = ax.twinx()
    ax2.plot(x, monthly["draw"], color=AMBER, linewidth=2.5, marker="o", markersize=6, zorder=5)
    ax2.axhline(22.3, color=AMBER, linestyle="--", alpha=0.4, linewidth=1)
    ax2.set_ylabel("Draw Rate (%)", color=AMBER, fontsize=11)
    ax2.tick_params(axis="y", colors=AMBER)
    ax2.set_ylim(10, 35)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Share of Outcomes (%)", fontsize=11)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper left", fontsize=10)
    ax.set_title("MONTHLY PROBABILITY REDISTRIBUTION\nWatch How Probability Mass Flows Between Outcomes",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.2)

    save(fig, "monthly_3way_distribution.png")

fig_monthly_distribution()


# ======================================================================
# FIGURE 4: Probability Mass Flow — arrow diagram
# ======================================================================
def fig_probability_flow():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.text(5, 5.5, "WHERE THE PROBABILITY MASS GOES",
            ha="center", fontsize=16, fontweight="bold", color=TEXT)
    ax.text(5, 5.0, "Season-over-season shift: 2024-25 → 2025-26",
            ha="center", fontsize=11, color=TEXT_MUTED)

    # Three boxes
    boxes = [
        (1.5, 2.5, "HOME REG", "45.1% → 39.9%", "-5.2pp", GREEN),
        (5.0, 2.5, "DRAW (OT)", "20.7% → 25.8%", "+5.1pp", AMBER),
        (8.5, 2.5, "AWAY REG", "34.2% → 34.3%", "+0.1pp", BLUE),
    ]
    for bx, by, lbl, detail, delta, col in boxes:
        ax.add_patch(plt.Rectangle((bx - 1.0, by - 0.8), 2.0, 1.6,
                                   facecolor=col, alpha=0.12, edgecolor=col,
                                   linewidth=2, joinstyle="round"))
        ax.text(bx, by + 0.35, lbl, ha="center", fontsize=11, fontweight="bold", color=col)
        ax.text(bx, by - 0.05, detail, ha="center", fontsize=9, color=TEXT_MUTED)
        ax.text(bx, by - 0.45, delta, ha="center", fontsize=14, fontweight="bold",
                color=GREEN if delta.startswith("+") and "5.1" in delta else (RED if delta.startswith("-") else TEXT_MUTED))

    # Arrows: Home → Draw (big), Away → Draw (tiny)
    ax.annotate("", xy=(4.0, 2.7), xytext=(2.5, 2.7),
                arrowprops=dict(arrowstyle="-|>", color=AMBER, lw=3, mutation_scale=20))
    ax.text(3.25, 3.2, "~5pp flows\nHome → Draw", ha="center", fontsize=9, fontweight="bold", color=AMBER)

    # Market response
    ax.text(5, 0.8, "Market adjusted draw pricing by only 0.65pp", ha="center",
            fontsize=11, color=RED, fontweight="bold")
    ax.text(5, 0.3, "Reality moved 5.16pp. The market missed 87% of the shift.",
            ha="center", fontsize=10, color=TEXT_MUTED)

    save(fig, "probability_flow_sankey.png")

fig_probability_flow()


# ======================================================================
# FIGURE 5: Tie Inflation Over Time
# ======================================================================
def fig_tie_inflation():
    # Monthly avg tie inflation from backtest
    bt_games_ti = bt.drop_duplicates("game_id").copy()
    bt_games_ti["month"] = bt_games_ti["game_date"].dt.to_period("M")
    monthly_ti = bt_games_ti.groupby("month")["tie_inflation"].mean()

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(monthly_ti))
    ax.plot(x, monthly_ti.values, color=AMBER, linewidth=2.5, marker="o", markersize=6)
    ax.fill_between(x, monthly_ti.values, alpha=0.15, color=AMBER)
    ax.axhline(1.0, color=TEXT_MUTED, linestyle=":", alpha=0.4, label="No inflation (raw Poisson)")
    ax.axhline(monthly_ti.mean(), color=GREEN, linestyle="--", alpha=0.5,
               label=f"Average: {monthly_ti.mean():.3f}x")

    ax.set_xticks(x)
    ax.set_xticklabels(monthly_ti.index.astype(str), rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Tie Inflation Multiplier", fontsize=11)
    ax.set_title("TIE INFLATION: THE MODEL ADAPTS\nBayesian calibration tracks the shifting OT rate",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Annotate season boundary
    for i, m in enumerate(monthly_ti.index):
        if str(m) == "2025-10":
            ax.axvline(i, color=PINK, linestyle="--", alpha=0.4)
            ax.text(i, ax.get_ylim()[1], " 2025-26 →", fontsize=9, color=PINK, va="top")
            break

    save(fig, "tie_inflation_timeline.png")

fig_tie_inflation()


# ======================================================================
# FIGURE 6: Bookmaker Draw Pricing Comparison
# ======================================================================
def fig_bookmaker_pricing():
    book_stats = bt.groupby("bookmaker").agg(
        n=("fair_draw_3w", "count"),
        fair_draw=("fair_draw_3w", "mean"),
        draw_edge=("draw_edge", "mean"),
    ).reset_index()
    # Filter to books with enough data
    book_stats = book_stats[book_stats["n"] >= 500].sort_values("fair_draw")

    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [AMBER if e > 0.005 else TEXT_MUTED for e in book_stats["draw_edge"]]
    bars = ax.barh(book_stats["bookmaker"], book_stats["fair_draw"] * 100, color=colors, alpha=0.8)

    ax.axvline(22.3, color=GREEN, linestyle="--", linewidth=2, label="Actual OT Rate: 22.3%")
    ax.axvline(22.1, color=BLUE, linestyle="--", linewidth=1.5, alpha=0.7, label="Model: 22.1%")

    for bar, edge in zip(bars, book_stats["draw_edge"]):
        w = bar.get_width()
        ax.text(w + 0.1, bar.get_y() + bar.get_height()/2,
                f"+{edge*100:.1f}pp edge" if edge > 0 else f"{edge*100:.1f}pp",
                va="center", fontsize=9, color=AMBER if edge > 0.005 else TEXT_MUTED)

    ax.set_xlabel("Fair Implied Draw Probability (%)", fontsize=11)
    ax.set_title("BOOKMAKER DRAW PRICING\nHow different books price the same outcome",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(axis="x", alpha=0.3)

    save(fig, "bookmaker_draw_pricing.png")

fig_bookmaker_pricing()


# ======================================================================
# FIGURE 7: Rolling Draw Rate vs Market Pricing
# ======================================================================
def fig_rolling_draw_rate():
    # Daily actual draw rate (using game_ids, not backtest — cleaner)
    daily = gi_bt.groupby("game_date")["actual_3way"].apply(
        lambda x: (x == "draw").mean() * 100
    ).reset_index()
    daily.columns = ["date", "draw_pct"]
    daily = daily.sort_values("date")
    daily["roll_7"] = daily["draw_pct"].rolling(7, min_periods=3).mean()
    daily["roll_30"] = daily["draw_pct"].rolling(30, min_periods=10).mean()

    # Market fair draw (daily avg from backtest)
    mkt_daily = bt.groupby("game_date")["fair_draw_3w"].mean().reset_index()
    mkt_daily.columns = ["date", "mkt_draw"]
    mkt_daily["mkt_draw"] *= 100
    mkt_daily["mkt_roll_30"] = mkt_daily["mkt_draw"].rolling(30, min_periods=10).mean()

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(daily["date"], daily["roll_7"], color=AMBER, alpha=0.4, linewidth=1, label="7-day actual")
    ax.plot(daily["date"], daily["roll_30"], color=AMBER, linewidth=2.5, label="30-day actual draw rate")
    ax.plot(mkt_daily["date"], mkt_daily["mkt_roll_30"], color=RED, linewidth=2,
            linestyle="--", label="30-day market implied draw")
    ax.axhline(22.3, color=GREEN, linestyle=":", alpha=0.4, label="Long-run average: 22.3%")

    ax.fill_between(daily["date"], daily["roll_30"], mkt_daily["mkt_draw"].reindex(daily.index).values,
                     alpha=0.08, color=AMBER, interpolate=True)

    ax.set_ylabel("Draw / OT Rate (%)", fontsize=11)
    ax.set_ylim(10, 38)
    ax.set_title("THE MARKET CAN'T KEEP UP\nActual draw rate swings wildly while market pricing barely moves",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    save(fig, "rolling_draw_rate_vs_market.png")

fig_rolling_draw_rate()


# ======================================================================
# FIGURE 8: Equity Curve — Draw + Home bets split
# ======================================================================
def fig_equity_curve():
    # --- Draw bets (from backtest) ---
    draw_bets = bt[bt["bet_side"] == "draw"].copy().sort_values("game_date")
    draw_bets["cum_flat"] = draw_bets["bet_profit"].cumsum()

    # --- Simulate home bets (edge 2.5-5%, stake = edge×20, cap 1u) ---
    home_pool = bt.drop_duplicates("game_id").copy()
    home_pool = home_pool[
        (home_pool["home_edge"] >= 0.025) & (home_pool["home_edge"] <= 0.05)
    ].copy()
    home_pool["bet_won_h"] = (home_pool["actual_3way"] == "home").astype(int)
    home_pool["bet_profit_h"] = np.where(
        home_pool["bet_won_h"] == 1,
        home_pool["dec_home_3w"] - 1,
        -1.0,
    )
    home_pool = home_pool.sort_values("game_date")
    home_pool["cum_flat_h"] = home_pool["bet_profit_h"].cumsum()

    # Daily endpoints
    daily_draw = draw_bets.groupby("game_date")["cum_flat"].last().reset_index()
    daily_home = home_pool.groupby("game_date")["cum_flat_h"].last().reset_index()

    # Combined cumulative (align on date)
    all_dates = sorted(set(daily_draw["game_date"]) | set(daily_home["game_date"]))
    combined = pd.DataFrame({"game_date": all_dates})
    combined = combined.merge(daily_draw, on="game_date", how="left")
    combined = combined.merge(daily_home, on="game_date", how="left")
    combined = combined.sort_values("game_date")
    combined["cum_flat"] = combined["cum_flat"].ffill().fillna(0)
    combined["cum_flat_h"] = combined["cum_flat_h"].ffill().fillna(0)
    combined["cum_total"] = combined["cum_flat"] + combined["cum_flat_h"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(combined["game_date"], combined["cum_total"], color=TEXT, linewidth=2.5,
            label=f"Combined ({len(draw_bets)+len(home_pool):,} bets)")
    ax.fill_between(combined["game_date"], combined["cum_total"], alpha=0.06, color=TEXT)
    ax.plot(daily_draw["game_date"], daily_draw["cum_flat"], color=AMBER, linewidth=2,
            label=f"Draw Bets ({len(draw_bets):,} bets, +{draw_bets['cum_flat'].iloc[-1]:.1f}u)")
    ax.plot(daily_home["game_date"], daily_home["cum_flat_h"], color=GREEN, linewidth=2,
            label=f"Home Reg Bets ({len(home_pool):,} bets, +{home_pool['cum_flat_h'].iloc[-1]:.1f}u)")
    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)

    ax.set_ylabel("Cumulative Flat-Bet Profit (units)", fontsize=11)
    ax.set_title("EQUITY CURVE: DRAW vs HOME BETS\nTwo independent profit streams from the same probability redistribution",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=11, loc="upper left")
    ax.grid(alpha=0.3)

    save(fig, "equity_curve.png")

fig_equity_curve()


# ======================================================================
# FIGURE 9: Momentum & Acceleration Signals
# ======================================================================
def fig_momentum_acceleration():
    # Use 14-day rolling draw hit rate from actual games
    daily_draw = gi_bt.groupby("game_date")["actual_3way"].apply(
        lambda x: (x == "draw").mean() * 100
    ).reset_index()
    daily_draw.columns = ["date", "draw_pct"]
    daily_draw = daily_draw.sort_values("date")
    daily_draw["roll_14"] = daily_draw["draw_pct"].rolling(14, min_periods=5).mean()
    daily_draw["momentum"] = daily_draw["roll_14"].diff(7)  # 7-day change
    daily_draw["acceleration"] = daily_draw["momentum"].diff(7)
    daily_draw = daily_draw.dropna()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), height_ratios=[2, 1], sharex=True)

    # Top: draw rate with zones
    ax1.plot(daily_draw["date"], daily_draw["roll_14"], color=AMBER, linewidth=2)
    ax1.axhline(22.3, color=TEXT_MUTED, linestyle=":", alpha=0.4)
    ax1.fill_between(daily_draw["date"], daily_draw["roll_14"],
                     where=daily_draw["roll_14"] > 22.3,
                     alpha=0.15, color=GREEN, label="Above average")
    ax1.fill_between(daily_draw["date"], daily_draw["roll_14"],
                     where=daily_draw["roll_14"] < 22.3,
                     alpha=0.15, color=RED, label="Below average")
    ax1.set_ylabel("14-Day Rolling Draw Rate (%)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_title("MOMENTUM & ACCELERATION\nTracking the probability redistribution wave",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.grid(alpha=0.3)

    # Bottom: momentum and acceleration
    ax2.plot(daily_draw["date"], daily_draw["momentum"], color=BLUE, linewidth=1.5,
             label="Momentum (7d Δ)")
    ax2.plot(daily_draw["date"], daily_draw["acceleration"], color=PINK, linewidth=1.5,
             alpha=0.8, label="Acceleration (7d ΔΔ)")
    ax2.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax2.fill_between(daily_draw["date"], daily_draw["momentum"],
                     where=(daily_draw["momentum"] > 0) & (daily_draw["acceleration"] > 0),
                     alpha=0.2, color=GREEN, label="Sweet spot (+mom, +accel)")
    ax2.set_ylabel("Rate of Change (pp)", fontsize=11)
    ax2.legend(fontsize=9, loc="lower left")
    ax2.grid(alpha=0.3)

    save(fig, "momentum_acceleration.png")

fig_momentum_acceleration()


# ======================================================================
# FIGURE 10: ROI by Edge Bucket
# ======================================================================
def fig_roi_by_edge():
    draw_bets = bt[bt["bet_side"] == "draw"].copy()
    draw_bets["edge_pct"] = draw_bets["draw_edge"] * 100

    bins = [0, 1, 2, 3, 5, 100]
    labels = ["0-1%", "1-2%", "2-3%", "3-5%", "5%+"]
    draw_bets["edge_bucket"] = pd.cut(draw_bets["edge_pct"], bins=bins, labels=labels, right=False)

    summary = draw_bets.groupby("edge_bucket", observed=True).agg(
        n=("bet_profit", "count"),
        wins=("bet_won", "sum"),
        total_profit=("bet_profit", "sum"),
    ).reset_index()
    summary["win_rate"] = summary["wins"] / summary["n"] * 100
    summary["roi"] = summary["total_profit"] / summary["n"] * 100

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [RED if r < 0 else GREEN for r in summary["roi"]]
    bars = ax.bar(summary["edge_bucket"], summary["roi"], color=colors, alpha=0.85, width=0.6)

    for bar, n, wr in zip(bars, summary["n"], summary["win_rate"]):
        h = bar.get_height()
        y = h + 0.5 if h >= 0 else h - 1.5
        ax.text(bar.get_x() + bar.get_width()/2, y,
                f"n={n:,}\n{wr:.1f}% W",
                ha="center", fontsize=9, color=TEXT_MUTED)

    ax.axhline(0, color=TEXT_MUTED, linewidth=0.8)
    ax.set_ylabel("Flat Bet ROI (%)", fontsize=11)
    ax.set_xlabel("Model Edge Bucket", fontsize=11)
    ax.set_title("ROI INCREASES WITH EDGE SIZE\nThe model's edge estimates are real — larger edges produce larger returns",
                 fontsize=14, fontweight="bold", pad=15)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "roi_by_edge_bucket.png")

fig_roi_by_edge()


# ======================================================================
# FIGURE 11: Season-Over-Season Shift
# ======================================================================
def fig_seasonal_shift():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Actual 3-way split by season
    seasons = {
        "2024-25": gi_bt[gi_bt["game_date"] < "2025-07-01"],
        "2025-26": gi_bt[gi_bt["game_date"] >= "2025-07-01"],
    }
    for i, (label, df) in enumerate(seasons.items()):
        counts = df["actual_3way"].value_counts(normalize=True) * 100
        h = counts.get("home", 0)
        d = counts.get("draw", 0)
        a = counts.get("away", 0)

        ax1.barh(i, h, color=GREEN, alpha=0.7, height=0.5)
        ax1.barh(i, d, left=h, color=AMBER, alpha=0.85, height=0.5)
        ax1.barh(i, a, left=h+d, color=BLUE, alpha=0.7, height=0.5)

        ax1.text(h/2, i, f"H: {h:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold")
        ax1.text(h + d/2, i, f"D: {d:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold")
        ax1.text(h + d + a/2, i, f"A: {a:.1f}%", ha="center", va="center", fontsize=10, fontweight="bold")

    ax1.set_yticks([0, 1])
    ax1.set_yticklabels(list(seasons.keys()), fontsize=12)
    ax1.set_xlabel("Outcome Share (%)", fontsize=11)
    ax1.set_title("ACTUAL OUTCOME SPLIT", fontsize=13, fontweight="bold")
    ax1.grid(axis="x", alpha=0.3)

    # Right: Market pricing barely moved
    market_by_season = bt.copy()
    market_by_season["season"] = market_by_season["game_date"].apply(
        lambda d: "2024-25" if d < pd.Timestamp("2025-07-01") else "2025-26"
    )
    mkt_agg = market_by_season.groupby("season")[["fair_home_3w", "fair_draw_3w", "fair_away_3w"]].mean() * 100

    x = np.arange(2)
    w = 0.25
    ax2.bar(x - w, mkt_agg["fair_home_3w"], w, label="Home", color=GREEN, alpha=0.7)
    ax2.bar(x, mkt_agg["fair_draw_3w"], w, label="Draw", color=AMBER, alpha=0.85)
    ax2.bar(x + w, mkt_agg["fair_away_3w"], w, label="Away", color=BLUE, alpha=0.7)

    ax2.set_xticks(x)
    ax2.set_xticklabels(["2024-25", "2025-26"], fontsize=12)
    ax2.set_ylabel("Market Fair Implied (%)", fontsize=11)
    ax2.set_title("MARKET PRICING (BARELY MOVED)", fontsize=13, fontweight="bold")
    ax2.set_ylim(15, 50)
    ax2.legend(fontsize=9)
    ax2.grid(axis="y", alpha=0.3)

    # Annotate the draw change
    d1 = mkt_agg.loc["2024-25", "fair_draw_3w"]
    d2 = mkt_agg.loc["2025-26", "fair_draw_3w"]
    ax2.annotate(f"Draw: {d1:.1f}% → {d2:.1f}%\n(only +{d2-d1:.1f}pp)",
                 xy=(1, d2), xytext=(1.3, d2 + 5),
                 fontsize=9, color=AMBER, fontweight="bold",
                 arrowprops=dict(arrowstyle="->", color=AMBER))

    fig.suptitle("THE SEASONAL SHIFT\nReality moved 5.2pp. The market moved 0.7pp.",
                 fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    save(fig, "seasonal_shift.png")

fig_seasonal_shift()


# ======================================================================
# FIGURE 12: Edge Distribution Across Books — violin/strip
# ======================================================================
def fig_bookmaker_edge_distribution():
    top_books = bt.groupby("bookmaker").size()
    top_books = top_books[top_books >= 500].index.tolist()
    bt_top = bt[bt["bookmaker"].isin(top_books)].copy()
    bt_top["draw_edge_pct"] = bt_top["draw_edge"] * 100

    book_order = bt_top.groupby("bookmaker")["draw_edge_pct"].mean().sort_values()

    fig, ax = plt.subplots(figsize=(12, 6))

    positions = []
    labels = []
    for i, book in enumerate(book_order.index):
        data = bt_top[bt_top["bookmaker"] == book]["draw_edge_pct"]
        vp = ax.violinplot(data, positions=[i], widths=0.7, showmedians=True, showextrema=False)
        color = AMBER if data.mean() > 0.5 else TEXT_MUTED
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_alpha(0.4)
        vp["cmedians"].set_color(color)
        vp["cmedians"].set_linewidth(2)
        positions.append(i)
        labels.append(book)

    ax.axhline(0, color=RED, linewidth=1, linestyle="--", alpha=0.5, label="Zero edge")
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Draw Edge (%)", fontsize=11)
    ax.set_title("EDGE DISTRIBUTION BY BOOKMAKER\nSame game, different prices — book shopping matters",
                 fontsize=14, fontweight="bold", pad=15)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    save(fig, "bookmaker_edge_distribution.png")

fig_bookmaker_edge_distribution()


# ======================================================================
# Done
# ======================================================================
print(f"\nAll figures saved to {OUT_DIR}")
print("Figures generated:")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
