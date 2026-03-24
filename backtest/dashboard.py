# -*- coding: utf-8 -*-
"""
NHL Win Probability Model — Backtest Dashboard

Dark-themed, Whizard-inspired analytics dashboard for the Poisson combiner
moneyline model. Evaluates walk-forward backtest performance on 2025-26 season.

Run: python dashboard.py
Open: http://127.0.0.1:8051

@author: chazf
"""

import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import io
import os
from math import erfc

# =============================================================================
# Configuration
# =============================================================================
PROJECT_DIR = r"D:\Python\NHL Overtime Model"
BT_FILE = os.path.join(PROJECT_DIR, "backtest", "backtest_results.csv")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
ODDS_2PM_FILE = os.path.join(RAW_DIR, "three_way_odds_2pm_est.csv")
ODDS_1HR_FILE = os.path.join(RAW_DIR, "three_way_odds.csv")
PORT = 8051

# =============================================================================
# Color Palette — Whizard-inspired dark theme
# =============================================================================
COLORS = {
    "bg_primary": "#020219",
    "bg_secondary": "#0a0a2e",
    "bg_card": "rgba(255, 255, 255, 0.025)",
    "bg_card_hover": "rgba(255, 255, 255, 0.04)",
    "bg_input": "rgba(255, 255, 255, 0.06)",
    "border": "rgba(255, 255, 255, 0.08)",
    "border_accent": "rgba(234, 103, 255, 0.3)",
    "text_primary": "#fafafa",
    "text_secondary": "rgba(255, 255, 255, 0.6)",
    "text_muted": "rgba(255, 255, 255, 0.35)",
    "accent_magenta": "#ea67ff",
    "accent_cyan": "#00a3ff",
    "accent_green": "#00ff88",
    "accent_gold": "#ffd700",
    "accent_red": "#ff4757",
    "accent_orange": "#ff9f43",
    "profit_green": "#00ff88",
    "loss_red": "#ff4757",
    "chart_grid": "rgba(255, 255, 255, 0.04)",
    "chart_line": "#ea67ff",
}

# Chart layout template
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text_primary"], size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(
        gridcolor=COLORS["chart_grid"],
        zerolinecolor=COLORS["chart_grid"],
        tickfont=dict(size=11, color=COLORS["text_secondary"]),
    ),
    yaxis=dict(
        gridcolor=COLORS["chart_grid"],
        zerolinecolor=COLORS["chart_grid"],
        tickfont=dict(size=11, color=COLORS["text_secondary"]),
    ),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(size=11, color=COLORS["text_secondary"]),
    ),
    hoverlabel=dict(
        bgcolor=COLORS["bg_secondary"],
        bordercolor=COLORS["border"],
        font=dict(family="JetBrains Mono, monospace", size=12, color=COLORS["text_primary"]),
    ),
)


# =============================================================================
# Data Loading
# =============================================================================
def load_data():
    """Load backtest results and compute derived columns."""
    if not os.path.exists(BT_FILE):
        return pd.DataFrame(columns=[
            "game_id", "game_date", "home_team", "away_team",
            "lam_home", "lam_away", "anchor_home", "anchor_away",
            "p_home_reg_win", "p_away_reg_win", "p_ot",
            "p_home_win_raw", "platt_a", "platt_b",
            "p_home_win", "p_away_win",
            "dk_home_fair", "dk_away_fair", "dk_home_dec", "dk_away_dec",
            "home_edge", "away_edge",
            "bet_side", "bet_edge", "bet_odds_dec", "kelly_stake",
            "home_won_actual", "game_outcome_type", "bet_won", "bet_profit",
        ])

    df = pd.read_csv(BT_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Derived columns
    df["day_of_week"] = df["game_date"].dt.day_name()
    df["is_ot"] = df["game_outcome_type"].isin(["OT", "SO"]).astype(int)
    df["best_edge"] = df[["home_edge", "away_edge"]].max(axis=1)

    # Flat staking profit (1 unit on bet_side)
    df["flat_profit"] = np.nan
    has_bet = df["bet_side"].notna() & df["bet_won"].notna()
    df.loc[has_bet, "flat_profit"] = df.loc[has_bet].apply(
        lambda r: (r["bet_odds_dec"] - 1.0) if r["bet_won"] == 1 else -1.0, axis=1
    )

    return df


# =============================================================================
# KPI Computation
# =============================================================================
def compute_kpis(df):
    """Compute headline KPIs from backtest results."""
    with_odds = df.dropna(subset=["dk_home_fair", "home_won_actual"])

    if len(with_odds) == 0:
        return {k: "—" for k in [
            "total_games", "games_with_odds", "win_rate", "roi_flat",
            "avg_edge", "log_loss", "ll_vs_dk", "sharpe", "max_drawdown",
            "best_day", "worst_day", "auc",
        ]}

    n = len(with_odds)
    y = with_odds["home_won_actual"].values.astype(float)
    p = np.clip(with_odds["p_home_win"].values, 0.001, 0.999)

    # Log-loss
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
    dk_p = np.clip(with_odds["dk_home_fair"].values, 0.001, 0.999)
    dk_ll = -(y * np.log(dk_p) + (1 - y) * np.log(1 - dk_p)).mean()

    # Flat staking
    flat_bets = with_odds[with_odds["flat_profit"].notna()]
    flat_pnl = flat_bets["flat_profit"].sum() if len(flat_bets) > 0 else 0
    flat_n = len(flat_bets) if len(flat_bets) > 0 else 1
    flat_roi = flat_pnl / flat_n * 100

    flat_wins = (flat_bets["bet_won"] == 1).sum() if len(flat_bets) > 0 else 0

    # Sharpe (daily P&L)
    daily = flat_bets.groupby("game_date")["flat_profit"].sum() if len(flat_bets) > 0 else pd.Series(dtype=float)
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if len(daily) > 1 and daily.std() > 0 else 0

    # Max drawdown
    if len(daily) > 0:
        cum = daily.cumsum()
        max_dd = (cum - cum.cummax()).min()
    else:
        max_dd = 0

    # AUC
    pos = p[y == 1]
    neg = p[y == 0]
    if len(pos) > 0 and len(neg) > 0:
        auc = np.mean(pos[:, None] > neg[None, :]) + 0.5 * np.mean(pos[:, None] == neg[None, :])
    else:
        auc = 0.5

    # Statistical significance
    if len(flat_bets) > 1 and flat_bets["flat_profit"].std() > 0:
        mean_pnl = flat_bets["flat_profit"].mean()
        std_pnl = flat_bets["flat_profit"].std()
        se_pnl = std_pnl / np.sqrt(flat_n)
        t_stat = mean_pnl / se_pnl
        p_value = erfc(abs(t_stat) / np.sqrt(2))
        ci_lo = (mean_pnl - 1.96 * se_pnl) / 1.0 * 100  # as ROI %
        ci_hi = (mean_pnl + 1.96 * se_pnl) / 1.0 * 100
        n_needed = int((1.96 * std_pnl / abs(mean_pnl)) ** 2) if mean_pnl != 0 else 99999
    else:
        t_stat, p_value, ci_lo, ci_hi, n_needed = 0, 1.0, 0, 0, 99999

    # Streak analysis
    max_win_streak = 0
    max_loss_streak = 0
    if len(flat_bets) > 0:
        sorted_bets = flat_bets.sort_values("game_date")
        cur = 0
        for w in sorted_bets["bet_won"].values:
            if w == 1:
                cur = max(1, cur + 1) if cur > 0 else 1
                max_win_streak = max(max_win_streak, cur)
            else:
                cur = min(-1, cur - 1) if cur < 0 else -1
                max_loss_streak = max(max_loss_streak, abs(cur))

    # Fav vs underdog
    fav_roi_str = "—"
    dog_roi_str = "—"
    if len(flat_bets) > 0:
        fb = flat_bets.copy()
        fb["is_fav"] = np.where(
            fb["bet_side"] == "home", fb["p_home_win"] > 0.5, fb["p_away_win"] > 0.5
        )
        fav = fb[fb["is_fav"]]
        dog = fb[~fb["is_fav"]]
        if len(fav) > 0:
            fav_roi_str = f"{fav['flat_profit'].sum()/len(fav)*100:+.1f}%"
        if len(dog) > 0:
            dog_roi_str = f"{dog['flat_profit'].sum()/len(dog)*100:+.1f}%"

    return {
        "total_games": f"{len(df):,}",
        "games_with_odds": f"{n:,}",
        "total_bets": f"{flat_n:,}",
        "win_rate": f"{flat_wins/flat_n:.1%}" if flat_n > 0 else "—",
        "roi_flat": f"{flat_roi:+.1f}%",
        "avg_edge": f"{with_odds['best_edge'].mean()*100:.1f}%",
        "log_loss": f"{ll:.4f}",
        "ll_vs_dk": f"{(ll - dk_ll)*1000:+.1f}‰",
        "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{max_dd:.1f}u",
        "best_day": f"+{daily.max():.1f}u" if len(daily) > 0 else "—",
        "worst_day": f"{daily.min():.1f}u" if len(daily) > 0 else "—",
        "auc": f"{auc:.3f}",
        "t_stat": f"{t_stat:.2f}",
        "p_value": f"{p_value:.3f}",
        "ci_95": f"[{ci_lo:+.1f}, {ci_hi:+.1f}]%",
        "n_needed": f"{n_needed:,}",
        "n_remaining": f"{max(0, n_needed - flat_n):,}",
        "significant": "Yes" if p_value < 0.05 else "No",
        "max_win_streak": f"{max_win_streak}",
        "max_loss_streak": f"{max_loss_streak}",
        "flat_pnl": f"{flat_pnl:+.1f}u",
        "fav_roi": fav_roi_str,
        "dog_roi": dog_roi_str,
        "home_bets": f"{len(flat_bets[flat_bets['bet_side'] == 'home']) if len(flat_bets) > 0 else 0}",
        "away_bets": f"{len(flat_bets[flat_bets['bet_side'] == 'away']) if len(flat_bets) > 0 else 0}",
    }


# =============================================================================
# Chart Builders
# =============================================================================
def build_cumulative_pnl(df, staking="flat"):
    """Cumulative P&L for selected staking mode."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No graded bets")

    fig = go.Figure()

    staking_label = "Flat (1u)" if staking == "flat" else "Kelly (¼K)"
    color = COLORS["accent_magenta"] if staking == "flat" else COLORS["accent_cyan"]
    fill_rgb = "234, 103, 255" if staking == "flat" else "0, 163, 255"

    daily = graded.groupby("game_date")["flat_profit"].sum().reset_index()
    daily["cumulative"] = daily["flat_profit"].cumsum()
    fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["cumulative"],
        fill="tozeroy", fillcolor=f"rgba({fill_rgb}, 0.08)",
        line=dict(color=color, width=2.5),
        hovertemplate="<b>%{x|%b %d}</b><br>" + staking_label + ": %{y:+.1f}u<extra></extra>",
        name=staking_label,
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"CUMULATIVE P&L ({staking_label.upper()})",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Units",
        height=360,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["text_secondary"])))
    return fig


def build_split_pnl(df):
    """Cumulative P&L split by Home/Away side."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No graded bets")

    fig = go.Figure()

    # Combined
    daily_all = graded.groupby("game_date")["flat_profit"].sum().reset_index()
    daily_all["cumulative"] = daily_all["flat_profit"].cumsum()
    fig.add_trace(go.Scatter(
        x=daily_all["game_date"], y=daily_all["cumulative"],
        line=dict(color=COLORS["accent_magenta"], width=2.5, dash="dash"),
        hovertemplate="<b>%{x|%b %d}</b><br>Combined: %{y:+.1f}u<extra></extra>",
        name="Combined",
    ))

    # Home bets
    home = graded[graded["bet_side"] == "home"]
    if len(home) > 0:
        daily_h = home.groupby("game_date")["flat_profit"].sum().reset_index()
        daily_h["cumulative"] = daily_h["flat_profit"].cumsum()
        fig.add_trace(go.Scatter(
            x=daily_h["game_date"], y=daily_h["cumulative"],
            fill="tozeroy", fillcolor="rgba(0,255,136,0.06)",
            line=dict(color=COLORS["accent_green"], width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>Home: %{y:+.1f}u<extra></extra>",
            name="Home",
        ))

    # Away bets
    away = graded[graded["bet_side"] == "away"]
    if len(away) > 0:
        daily_a = away.groupby("game_date")["flat_profit"].sum().reset_index()
        daily_a["cumulative"] = daily_a["flat_profit"].cumsum()
        fig.add_trace(go.Scatter(
            x=daily_a["game_date"], y=daily_a["cumulative"],
            fill="tozeroy", fillcolor="rgba(0,163,255,0.06)",
            line=dict(color=COLORS["accent_cyan"], width=2),
            hovertemplate="<b>%{x|%b %d}</b><br>Away: %{y:+.1f}u<extra></extra>",
            name="Away",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="CUMULATIVE P&L BY SIDE", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Units",
        height=360,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["text_secondary"])))
    return fig


def build_daily_pnl_bars(df):
    """Daily P&L bar chart (flat staking)."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No graded bets")

    daily = graded.groupby("game_date").agg(
        profit=("flat_profit", "sum"),
        n_bets=("flat_profit", "count"),
    ).reset_index()

    colors = [COLORS["profit_green"] if p >= 0 else COLORS["loss_red"] for p in daily["profit"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["game_date"], y=daily["profit"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x|%b %d}</b><br>P&L: %{y:+.2f}u<br>Bets: %{customdata}<extra></extra>",
        customdata=daily["n_bets"],
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DAILY P&L (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Units", showlegend=False, height=300,
    )
    return fig


def build_calibration_chart(df):
    """P(home win) calibration — model probability vs actual win rate."""
    valid = df.dropna(subset=["p_home_win", "home_won_actual"])
    if len(valid) < 50:
        return empty_chart("Not enough data")

    bins = np.arange(0.30, 0.75, 0.04)
    valid = valid.copy()
    valid["prob_bin"] = pd.cut(valid["p_home_win"], bins=bins)
    cal = valid.groupby("prob_bin", observed=True).agg(
        actual=("home_won_actual", "mean"),
        model=("p_home_win", "mean"),
        n=("home_won_actual", "count"),
    ).reset_index()
    cal = cal[cal["n"] >= 5]

    fig = go.Figure()

    # Perfect line
    fig.add_trace(go.Scatter(
        x=[0.3, 0.7], y=[0.3, 0.7],
        mode="lines", line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=cal["model"], y=cal["actual"],
        mode="markers+lines",
        marker=dict(
            size=cal["n"].clip(upper=200) / 5 + 6,
            color=COLORS["accent_cyan"],
            line=dict(width=1, color="rgba(0,163,255,0.5)"),
        ),
        line=dict(color=COLORS["accent_cyan"], width=2),
        hovertemplate="Model: %{x:.1%}<br>Actual: %{y:.1%}<br>n=%{text}<extra></extra>",
        text=cal["n"],
        name="Model",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="P(HOME WIN) CALIBRATION", font=dict(size=14, color=COLORS["text_secondary"])),
        showlegend=False, height=360,
    )
    fig.update_layout(
        xaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%", range=[0.3, 0.7],
                   title="Model Probability", tickfont=dict(size=11, color=COLORS["text_secondary"])),
        yaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%", range=[0.3, 0.7],
                   title="Actual Win Rate", tickfont=dict(size=11, color=COLORS["text_secondary"])),
    )
    return fig


def build_ot_calibration(df):
    """P(OT) calibration by quintile."""
    valid = df.dropna(subset=["p_ot"]).copy()
    if len(valid) < 50:
        return empty_chart("Not enough data")

    valid["ot_actual"] = valid["game_outcome_type"].isin(["OT", "SO"]).astype(int)
    valid["ot_bin"] = pd.qcut(valid["p_ot"], 5, labels=False, duplicates="drop")

    cal = valid.groupby("ot_bin").agg(
        model=("p_ot", "mean"),
        actual=("ot_actual", "mean"),
        n=("ot_actual", "count"),
    ).reset_index()

    fig = go.Figure()

    # Perfect line
    fig.add_trace(go.Scatter(
        x=[0.20, 0.32], y=[0.20, 0.32],
        mode="lines", line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
    ))

    fig.add_trace(go.Scatter(
        x=cal["model"], y=cal["actual"],
        mode="markers+lines",
        marker=dict(size=10, color=COLORS["accent_gold"],
                    line=dict(width=1, color="rgba(255,215,0,0.5)")),
        line=dict(color=COLORS["accent_gold"], width=2),
        hovertemplate="Model: %{x:.1%}<br>Actual: %{y:.1%}<br>n=%{text}<extra></extra>",
        text=cal["n"], name="P(OT)",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="P(OT) CALIBRATION", font=dict(size=14, color=COLORS["text_secondary"])),
        showlegend=False, height=360,
    )
    fig.update_layout(
        xaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%",
                   title="Model P(OT)", tickfont=dict(size=11, color=COLORS["text_secondary"])),
        yaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%",
                   title="Actual OT Rate", tickfont=dict(size=11, color=COLORS["text_secondary"])),
    )
    return fig


def build_edge_roi_chart(df):
    """ROI by edge bucket — monotonicity check (flat staking)."""
    graded = df.dropna(subset=["flat_profit", "best_edge"])
    if len(graded) == 0:
        return empty_chart("No data")

    bins = [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.15, 1.0]
    labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(bins)-1)]
    graded = graded.copy()
    graded["edge_bucket"] = pd.cut(graded["best_edge"], bins=bins, labels=labels)

    by_edge = graded.groupby("edge_bucket", observed=True).agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
    ).reset_index()

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_edge["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_edge["edge_bucket"], y=by_edge["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="Edge: %{x}<br>ROI: %{y:+.1f}%<br>Win: %{customdata:.1%}<extra></extra>",
        customdata=by_edge["win_rate"],
        text=by_edge["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY EDGE BUCKET (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_pot_roi_chart(df):
    """ROI by P(OT) bucket."""
    graded = df.dropna(subset=["flat_profit", "p_ot"])
    if len(graded) == 0:
        return empty_chart("No data")

    bins = [0.0, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 1.0]
    labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(bins)-1)]
    graded = graded.copy()
    graded["pot_bucket"] = pd.cut(graded["p_ot"], bins=bins, labels=labels)

    by_pot = graded.groupby("pot_bucket", observed=True).agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
        avg_pot=("p_ot", "mean"),
    ).reset_index()

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_pot["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_pot["pot_bucket"], y=by_pot["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="P(OT): %{x}<br>ROI: %{y:+.1f}%<br>Win: %{customdata[0]:.1%}<br>Avg P(OT): %{customdata[1]:.1%}<extra></extra>",
        customdata=by_pot[["win_rate", "avg_pot"]].values,
        text=by_pot["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY P(OT) BUCKET", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_home_away_chart(df):
    """Home vs Away side ROI comparison."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    by_side = graded.groupby("bet_side").agg(
        n=("flat_profit", "count"),
        wins=("bet_won", "sum"),
        profit=("flat_profit", "sum"),
        avg_edge=("bet_edge", "mean"),
    ).reset_index()
    by_side["win_rate"] = by_side["wins"] / by_side["n"]
    by_side["roi"] = by_side["profit"] / by_side["n"] * 100

    side_colors = {"home": COLORS["accent_green"], "away": COLORS["accent_cyan"]}

    fig = go.Figure()
    for _, row in by_side.iterrows():
        fig.add_trace(go.Bar(
            x=[row["bet_side"].title()],
            y=[row["roi"]],
            marker_color=side_colors.get(row["bet_side"], COLORS["text_muted"]),
            opacity=0.85, name=row["bet_side"].title(),
            hovertemplate=(
                f"<b>{row['bet_side'].title()}</b><br>"
                f"ROI: {row['roi']:+.1f}%<br>Win: {row['win_rate']:.1%}<br>"
                f"n={int(row['n'])}<extra></extra>"
            ),
            text=[f"{row['roi']:+.1f}%"],
            textposition="outside", textfont=dict(size=12, color=COLORS["text_primary"]),
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="HOME vs AWAY (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_outcome_type_chart(df):
    """ROI by game outcome type (REG / OT / SO)."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    by_type = graded.groupby("game_outcome_type").agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
    ).reset_index()

    type_colors = {
        "REG": COLORS["accent_magenta"],
        "OT": COLORS["accent_cyan"],
        "SO": COLORS["accent_gold"],
    }

    fig = go.Figure()
    for _, row in by_type.iterrows():
        fig.add_trace(go.Bar(
            x=[row["game_outcome_type"]],
            y=[row["roi"]],
            marker_color=type_colors.get(row["game_outcome_type"], COLORS["text_muted"]),
            opacity=0.85, name=row["game_outcome_type"],
            hovertemplate=(
                f"<b>{row['game_outcome_type']}</b><br>"
                f"ROI: {row['roi']:+.1f}%<br>Win: {row['win_rate']:.1%}<br>"
                f"n={int(row['n'])}<extra></extra>"
            ),
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY OUTCOME TYPE (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_lambda_scatter(df):
    """Lambda home vs away scatter — color by outcome."""
    valid = df.dropna(subset=["lam_home", "lam_away", "home_won_actual"])
    if len(valid) == 0:
        return empty_chart("No data")

    fig = go.Figure()

    for won, label, color in [
        (1, "Home Win", COLORS["accent_green"]),
        (0, "Away Win", COLORS["accent_cyan"]),
    ]:
        subset = valid[valid["home_won_actual"] == won]
        fig.add_trace(go.Scatter(
            x=subset["lam_home"], y=subset["lam_away"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.4),
            name=label,
            hovertemplate=(
                "<b>%{customdata[0]} vs %{customdata[1]}</b><br>"
                "λ Home: %{x:.2f}<br>λ Away: %{y:.2f}<extra></extra>"
            ),
            customdata=subset[["home_team", "away_team"]].values,
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[1.5, 4.5], y=[1.5, 4.5],
        mode="lines", line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="LAMBDA HOME vs AWAY", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="λ Home", yaxis_title="λ Away", height=360,
    )
    return fig


def build_edge_distribution(df):
    """Histogram of home edge (model - DK fair)."""
    valid = df.dropna(subset=["home_edge"])
    if len(valid) == 0:
        return empty_chart("No data")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid["home_edge"] * 100,
        nbinsx=40,
        marker_color=COLORS["accent_magenta"],
        opacity=0.7,
        hovertemplate="Edge: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))

    mean_edge = valid["home_edge"].mean() * 100
    fig.add_vline(x=mean_edge, line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
                  annotation_text=f"Mean={mean_edge:.1f}%", annotation_font_color=COLORS["accent_cyan"])
    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="EDGE DISTRIBUTION (HOME)", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Home Edge %", yaxis_title="Count",
        showlegend=False, height=300,
    )
    return fig


def build_dow_roi_chart(df):
    """ROI by day of week."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    graded = graded.copy()
    if "day_of_week" not in graded.columns:
        graded["day_of_week"] = graded["game_date"].dt.day_name()

    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    by_dow = graded.groupby("day_of_week").agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
    ).reindex(day_order).dropna().reset_index()

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_dow["roi"]]
    short_names = [d[:3] for d in by_dow["day_of_week"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=short_names, y=by_dow["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x}</b><br>ROI: %{y:+.1f}%<br>Win: %{customdata[0]:.1%}<br>n=%{customdata[1]}<extra></extra>",
        customdata=by_dow[["win_rate", "n"]].values,
        text=by_dow["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY DAY OF WEEK", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_roi_heatmap(df):
    """ROI heatmap: Bet Side x Outcome Type."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    sides = ["home", "away"]
    outcomes = ["REG", "OT", "SO"]

    z_vals = []
    annotations = []
    for i, side in enumerate(sides):
        row = []
        for j, oc in enumerate(outcomes):
            subset = graded[(graded["bet_side"] == side) & (graded["game_outcome_type"] == oc)]
            n = len(subset)
            if n > 0:
                roi = subset["flat_profit"].sum() / n * 100
                row.append(roi)
                annotations.append(dict(
                    x=oc, y=side.title(),
                    text=f"{roi:+.1f}%<br>n={n}",
                    showarrow=False,
                    font=dict(size=12, color="white", family="JetBrains Mono, monospace"),
                ))
            else:
                row.append(0)
                annotations.append(dict(
                    x=oc, y=side.title(), text="—", showarrow=False,
                    font=dict(size=12, color=COLORS["text_muted"]),
                ))
        z_vals.append(row)

    fig = go.Figure(data=go.Heatmap(
        z=z_vals, x=outcomes, y=[s.title() for s in sides],
        colorscale=[[0, COLORS["loss_red"]], [0.5, COLORS["bg_secondary"]], [1, COLORS["profit_green"]]],
        zmid=0, showscale=True,
        colorbar=dict(title=dict(text="ROI%", font=dict(size=11, color=COLORS["text_secondary"])),
                      tickfont=dict(size=10, color=COLORS["text_secondary"])),
        hovertemplate="Side: %{y}<br>Outcome: %{x}<br>ROI: %{z:+.1f}%<extra></extra>",
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text_primary"], size=12),
        margin=dict(l=50, r=20, t=40, b=40),
        title=dict(text="ROI: SIDE x OUTCOME", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis=dict(tickfont=dict(size=12, color=COLORS["text_secondary"])),
        yaxis=dict(tickfont=dict(size=12, color=COLORS["text_secondary"])),
        annotations=annotations, height=300,
    )
    return fig


def build_team_roi_chart(df):
    """Top/bottom teams by ROI (home team)."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    # Combine home and away team perspectives
    rows = []
    for _, r in graded.iterrows():
        if r["bet_side"] == "home":
            rows.append({"team": r["home_team"], "profit": r["flat_profit"], "won": r["bet_won"]})
        elif r["bet_side"] == "away":
            rows.append({"team": r["away_team"], "profit": r["flat_profit"], "won": r["bet_won"]})

    if not rows:
        return empty_chart("No data")

    team_df = pd.DataFrame(rows)
    by_team = team_df.groupby("team").agg(
        n=("profit", "count"),
        roi=("profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("won", "mean"),
    ).reset_index()
    by_team = by_team[by_team["n"] >= 5]
    if len(by_team) == 0:
        return empty_chart("No teams with 5+ bets")

    by_team = by_team.sort_values("roi")
    if len(by_team) > 20:
        by_team = pd.concat([by_team.head(10), by_team.tail(10)])

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_team["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=by_team["team"], x=by_team["roi"],
        orientation="h", marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{y}</b><br>ROI: %{x:+.1f}%<br>Win: %{customdata[0]:.1%}<br>n=%{customdata[1]}<extra></extra>",
        customdata=by_team[["win_rate", "n"]].values,
        text=by_team.apply(lambda r: f"{r['roi']:+.1f}% (n={int(r['n'])})", axis=1),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_secondary"]),
    ))

    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY TEAM BET ON", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="ROI %", showlegend=False,
        height=max(300, len(by_team) * 25 + 80),
    )
    fig.update_layout(margin=dict(l=60, r=100, t=40, b=40))
    return fig


def build_rolling_ll(df):
    """Rolling 50-game log-loss: model vs DK."""
    valid = df.dropna(subset=["p_home_win", "dk_home_fair", "home_won_actual"]).copy().sort_values("game_date")
    if len(valid) < 50:
        return empty_chart("Not enough data")

    y = valid["home_won_actual"].values.astype(float)
    p_model = np.clip(valid["p_home_win"].values, 0.001, 0.999)
    p_dk = np.clip(valid["dk_home_fair"].values, 0.001, 0.999)

    ll_model = -(y * np.log(p_model) + (1 - y) * np.log(1 - p_model))
    ll_dk = -(y * np.log(p_dk) + (1 - y) * np.log(1 - p_dk))

    rolling_model = pd.Series(ll_model).rolling(50).mean()
    rolling_dk = pd.Series(ll_dk).rolling(50).mean()

    dates = valid["game_date"].values

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_model,
        line=dict(color=COLORS["accent_magenta"], width=2),
        name="Model",
        hovertemplate="<b>%{x|%b %d}</b><br>Model LL: %{y:.4f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=rolling_dk,
        line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
        name="DK",
        hovertemplate="<b>%{x|%b %d}</b><br>DK LL: %{y:.4f}<extra></extra>",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROLLING 50-GAME LOG-LOSS", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Log-Loss (lower is better)",
        height=360,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["text_secondary"])))
    return fig


def build_monthly_pnl_chart(df):
    """Monthly P&L bar chart with 95% confidence interval error bars."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No graded bets")

    graded = graded.copy()
    graded["month"] = graded["game_date"].dt.to_period("M").astype(str)

    monthly = graded.groupby("month").agg(
        n=("flat_profit", "count"),
        wins=("bet_won", "sum"),
        profit=("flat_profit", "sum"),
        std=("flat_profit", "std"),
    ).reset_index()
    monthly["roi"] = monthly["profit"] / monthly["n"] * 100
    monthly["win_rate"] = monthly["wins"] / monthly["n"]
    monthly["se"] = monthly["std"] / np.sqrt(monthly["n"]) * 100
    monthly["ci_lo"] = monthly["roi"] - 1.96 * monthly["se"]
    monthly["ci_hi"] = monthly["roi"] + 1.96 * monthly["se"]

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in monthly["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly["month"], y=monthly["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        error_y=dict(type="data", symmetric=False,
                     array=(monthly["ci_hi"] - monthly["roi"]).values,
                     arrayminus=(monthly["roi"] - monthly["ci_lo"]).values,
                     color=COLORS["text_muted"], thickness=1.5, width=4),
        hovertemplate=(
            "<b>%{x}</b><br>ROI: %{y:+.1f}%<br>"
            "P&L: %{customdata[0]:+.1f}u<br>"
            "n=%{customdata[1]}, WR=%{customdata[2]:.1%}<br>"
            "95% CI: [%{customdata[3]:+.1f}, %{customdata[4]:+.1f}]%<extra></extra>"
        ),
        customdata=monthly[["profit", "n", "win_rate", "ci_lo", "ci_hi"]].values,
        text=monthly["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=9, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="MONTHLY ROI WITH 95% CI", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=340,
    )
    return fig


def build_fav_underdog_chart(df):
    """ROI split by favorite vs underdog bets."""
    graded = df.dropna(subset=["flat_profit"]).copy()
    if len(graded) == 0:
        return empty_chart("No data")

    graded["is_fav"] = np.where(
        graded["bet_side"] == "home",
        graded["p_home_win"] > 0.5,
        graded["p_away_win"] > 0.5,
    )

    cats = []
    for is_fav, label in [(True, "Favorite"), (False, "Underdog")]:
        sub = graded[graded["is_fav"] == is_fav]
        if len(sub) > 0:
            roi = sub["flat_profit"].sum() / len(sub) * 100
            wr = sub["bet_won"].mean()
            avg_odds = sub["bet_odds_dec"].mean()
            cats.append({"label": label, "roi": roi, "n": len(sub), "wr": wr, "avg_odds": avg_odds})

    if not cats:
        return empty_chart("No data")

    cat_df = pd.DataFrame(cats)
    cat_colors = {"Favorite": COLORS["accent_gold"], "Underdog": COLORS["accent_cyan"]}

    fig = go.Figure()
    for _, row in cat_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["label"]], y=[row["roi"]],
            marker_color=cat_colors.get(row["label"], COLORS["text_muted"]),
            opacity=0.85, name=row["label"],
            hovertemplate=(
                f"<b>{row['label']}</b><br>"
                f"ROI: {row['roi']:+.1f}%<br>"
                f"Win Rate: {row['wr']:.1%}<br>"
                f"Avg Odds: {row['avg_odds']:.3f}<br>"
                f"n={int(row['n'])}<extra></extra>"
            ),
            text=[f"{row['roi']:+.1f}%\nn={int(row['n'])}"],
            textposition="outside", textfont=dict(size=11, color=COLORS["text_primary"]),
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="FAVORITE vs UNDERDOG (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_drawdown_chart(df):
    """Drawdown over time (distance from peak P&L)."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No data")

    daily = graded.groupby("game_date")["flat_profit"].sum().reset_index()
    daily["cumulative"] = daily["flat_profit"].cumsum()
    daily["peak"] = daily["cumulative"].cummax()
    daily["drawdown"] = daily["cumulative"] - daily["peak"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["drawdown"],
        fill="tozeroy", fillcolor="rgba(255, 71, 87, 0.15)",
        line=dict(color=COLORS["accent_red"], width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>Drawdown: %{y:.1f}u<extra></extra>",
        name="Drawdown",
    ))

    max_dd = daily["drawdown"].min()
    max_dd_date = daily.loc[daily["drawdown"].idxmin(), "game_date"]
    fig.add_annotation(
        x=max_dd_date, y=max_dd,
        text=f"Max: {max_dd:.1f}u", showarrow=True, arrowhead=2,
        font=dict(size=11, color=COLORS["accent_red"]),
        arrowcolor=COLORS["accent_red"], ax=30, ay=-30,
    )

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DRAWDOWN FROM PEAK", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Units from peak", showlegend=False, height=300,
    )
    return fig


def load_clv_data():
    """Load 2pm EST and 1hr pre-game odds, join, and compute CLV."""
    if not os.path.exists(ODDS_2PM_FILE) or not os.path.exists(ODDS_1HR_FILE):
        return pd.DataFrame()

    early = pd.read_csv(ODDS_2PM_FILE)
    close = pd.read_csv(ODDS_1HR_FILE)

    # Filter out live/in-play odds from the 2pm scrape
    live_mask = (
        (early["home_dec"] < 1.10) |
        (early["away_dec"] < 1.10) |
        (early["draw_dec"] > 15)
    )
    early = early[~live_mask].copy()

    # Join on game_date + home_team + bookmaker
    join_cols = ["game_date", "home_team", "bookmaker"]
    merged = early.merge(
        close, on=join_cols, suffixes=("_2pm", "_1hr"), how="inner",
    )

    if len(merged) == 0:
        return pd.DataFrame()

    # CLV: positive = better odds at 2pm (early)
    merged["draw_clv_raw"] = merged["draw_dec_2pm"] - merged["draw_dec_1hr"]
    merged["draw_clv_pct"] = (merged["draw_dec_2pm"] / merged["draw_dec_1hr"] - 1) * 100
    merged["home_clv_raw"] = merged["home_dec_2pm"] - merged["home_dec_1hr"]
    merged["away_clv_raw"] = merged["away_dec_2pm"] - merged["away_dec_1hr"]

    # Implied prob CLV (positive = closing line moved toward draw)
    merged["draw_ip_2pm"] = 1 / merged["draw_dec_2pm"]
    merged["draw_ip_1hr"] = 1 / merged["draw_dec_1hr"]
    merged["draw_ip_clv"] = (merged["draw_ip_1hr"] - merged["draw_ip_2pm"]) * 100

    merged["game_date"] = pd.to_datetime(merged["game_date"])
    return merged


def build_clv_scatter(clv):
    """Scatter: 2pm draw odds vs 1hr draw odds."""
    if len(clv) == 0:
        return empty_chart("No CLV data — re-scrape odds first")

    fig = go.Figure()

    # home_team is a join key (no suffix), away_team gets suffixes
    away_col = "away_team_2pm" if "away_team_2pm" in clv.columns else "away_team"
    hover_df = clv[["game_date", "home_team", away_col, "draw_clv_pct"]].copy()
    hover_df.columns = ["date", "home", "away", "clv"]
    hover_df["date"] = hover_df["date"].dt.strftime("%b %d")

    fig.add_trace(go.Scatter(
        x=clv["draw_dec_2pm"], y=clv["draw_dec_1hr"],
        mode="markers",
        marker=dict(size=5, color=COLORS["accent_gold"], opacity=0.5),
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]} vs %{customdata[2]}<br>"
            "2pm: %{x:.2f} → 1hr: %{y:.2f}<br>"
            "CLV: %{customdata[3]:+.1f}%<extra></extra>"
        ),
        customdata=hover_df.values,
    ))

    # Perfect correlation line
    lo, hi = 3.0, 6.0
    fig.add_trace(go.Scatter(
        x=[lo, hi], y=[lo, hi],
        mode="lines", line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DRAW ODDS: 2PM EST vs 1HR PRE-GAME",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Draw Decimal (2pm EST)",
        yaxis_title="Draw Decimal (1hr Pre-Game)",
        height=360,
    )
    return fig


def build_clv_distribution(clv):
    """Histogram of draw CLV percentage."""
    if len(clv) == 0:
        return empty_chart("No CLV data — re-scrape odds first")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=clv["draw_clv_pct"],
        nbinsx=50,
        marker_color=COLORS["accent_gold"],
        opacity=0.7,
        hovertemplate="CLV: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))

    mean_clv = clv["draw_clv_pct"].mean()
    median_clv = clv["draw_clv_pct"].median()
    fig.add_vline(x=mean_clv, line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
                  annotation_text=f"Mean={mean_clv:+.2f}%",
                  annotation_font_color=COLORS["accent_cyan"])
    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DRAW CLV DISTRIBUTION (2PM vs 1HR)",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="CLV % (positive = better odds at 2pm)",
        yaxis_title="Count",
        showlegend=False, height=360,
    )
    return fig


def build_clv_by_book(clv):
    """Mean draw CLV by bookmaker."""
    if len(clv) == 0:
        return empty_chart("No CLV data — re-scrape odds first")

    by_book = clv.groupby("bookmaker").agg(
        n=("draw_clv_pct", "count"),
        mean_clv=("draw_clv_pct", "mean"),
        median_clv=("draw_clv_pct", "median"),
        mean_raw=("draw_clv_raw", "mean"),
    ).reset_index()
    by_book = by_book[by_book["n"] >= 10].sort_values("mean_clv")

    if len(by_book) == 0:
        return empty_chart("Not enough data per book")

    colors = [COLORS["accent_green"] if c >= 0 else COLORS["accent_red"] for c in by_book["mean_clv"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=by_book["bookmaker"], x=by_book["mean_clv"],
        orientation="h", marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Mean CLV: %{x:+.2f}%<br>"
            "Median: %{customdata[0]:+.2f}%<br>"
            "Avg raw: %{customdata[1]:+.3f}<br>"
            "n=%{customdata[2]}<extra></extra>"
        ),
        customdata=by_book[["median_clv", "mean_raw", "n"]].values,
        text=by_book.apply(lambda r: f"{r['mean_clv']:+.2f}% (n={int(r['n'])})", axis=1),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_secondary"]),
    ))

    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="MEAN DRAW CLV BY BOOKMAKER",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Mean CLV %",
        showlegend=False,
        height=max(250, len(by_book) * 40 + 80),
    )
    fig.update_layout(margin=dict(l=100, r=100, t=40, b=40))
    return fig


def build_clv_over_time(clv):
    """Rolling mean CLV over time (by game date)."""
    if len(clv) == 0:
        return empty_chart("No CLV data — re-scrape odds first")

    daily = clv.groupby("game_date").agg(
        mean_clv=("draw_clv_pct", "mean"),
        n=("draw_clv_pct", "count"),
    ).reset_index().sort_values("game_date")

    # 14-day rolling average
    daily["rolling_clv"] = daily["mean_clv"].rolling(14, min_periods=3).mean()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["mean_clv"],
        mode="markers",
        marker=dict(size=4, color=COLORS["accent_gold"], opacity=0.4),
        name="Daily Avg",
        hovertemplate="<b>%{x|%b %d}</b><br>CLV: %{y:+.2f}%<br>n=%{customdata}<extra></extra>",
        customdata=daily["n"],
    ))
    fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["rolling_clv"],
        mode="lines",
        line=dict(color=COLORS["accent_magenta"], width=2.5),
        name="14-Day Rolling",
        hovertemplate="<b>%{x|%b %d}</b><br>Rolling CLV: %{y:+.2f}%<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DRAW CLV OVER TIME (2PM vs 1HR)",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="CLV %",
        height=360,
    )
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["text_secondary"])))
    return fig


def empty_chart(msg="No data available"):
    """Return an empty chart with a message."""
    fig = go.Figure()
    fig.add_annotation(
        text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
        font=dict(size=16, color=COLORS["text_muted"]), showarrow=False,
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False), yaxis=dict(visible=False), height=300,
    )
    return fig


# =============================================================================
# CSS Styles
# =============================================================================
STYLESHEET = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    background: #020219;
    color: #fafafa;
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    -webkit-font-smoothing: antialiased;
    min-height: 100vh;
}

body::before {
    content: '';
    position: fixed;
    top: -50%; left: -50%;
    width: 200%; height: 200%;
    background: radial-gradient(ellipse at 20% 50%, rgba(234,103,255,0.04) 0%, transparent 50%),
                radial-gradient(ellipse at 80% 20%, rgba(0,163,255,0.03) 0%, transparent 50%),
                radial-gradient(ellipse at 50% 80%, rgba(0,255,136,0.02) 0%, transparent 50%);
    z-index: -1;
    pointer-events: none;
}

.dashboard-container { max-width: 1440px; margin: 0 auto; padding: 24px 32px; }

.header {
    display: flex; align-items: center; justify-content: space-between;
    padding: 20px 0 32px 0;
    border-bottom: 1px solid rgba(255,255,255,0.06);
    margin-bottom: 28px;
}
.header-left { display: flex; align-items: baseline; gap: 16px; }
.header-title {
    font-size: 28px; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #ea67ff 0%, #00a3ff 50%, #00ff88 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.header-subtitle {
    font-size: 13px; color: rgba(255,255,255,0.35); font-weight: 400;
    font-family: 'JetBrains Mono', monospace;
}
.header-badge {
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    background: rgba(234,103,255,0.1); border: 1px solid rgba(234,103,255,0.2);
    color: #ea67ff; padding: 4px 12px; border-radius: 20px; font-weight: 500;
}

.filter-bar {
    display: flex; gap: 16px; align-items: center; flex-wrap: wrap;
    margin-bottom: 28px; padding: 16px 20px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06); border-radius: 12px;
}
.filter-group { display: flex; flex-direction: column; gap: 4px; }
.filter-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: rgba(255,255,255,0.35);
}
.filter-range-pair { display: flex; align-items: center; gap: 4px; }
.filter-range-sep { color: rgba(255,255,255,0.25); font-size: 12px; padding-top: 2px; }

.kpi-grid {
    display: grid; grid-template-columns: repeat(8, 1fr);
    gap: 12px; margin-bottom: 28px;
}
.kpi-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px; padding: 16px 18px;
    position: relative; overflow: hidden;
    transition: border-color 0.2s, background 0.2s;
}
.kpi-card:hover {
    border-color: rgba(234,103,255,0.2); background: rgba(255,255,255,0.04);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(234,103,255,0.3), transparent);
    opacity: 0; transition: opacity 0.2s;
}
.kpi-card:hover::before { opacity: 1; }
.kpi-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: rgba(255,255,255,0.35); margin-bottom: 8px;
}
.kpi-value {
    font-size: 22px; font-weight: 700;
    font-family: 'JetBrains Mono', monospace; letter-spacing: -0.5px;
}
.kpi-positive { color: #00ff88; }
.kpi-negative { color: #ff4757; }
.kpi-neutral { color: #fafafa; }

.chart-grid-2 {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 16px; margin-bottom: 20px;
}
.chart-grid-3 {
    display: grid; grid-template-columns: 1fr 1fr 1fr;
    gap: 16px; margin-bottom: 20px;
}
.chart-grid-2-1 {
    display: grid; grid-template-columns: 2fr 1fr;
    gap: 16px; margin-bottom: 20px;
}
.chart-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 20px;
    transition: border-color 0.2s;
}
.chart-card:hover { border-color: rgba(255,255,255,0.12); }
.chart-card-full {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 20px;
    transition: border-color 0.2s; margin-bottom: 20px;
}
.chart-card-full:hover { border-color: rgba(255,255,255,0.12); }

.section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.5px; color: rgba(255,255,255,0.3);
    margin-bottom: 16px; padding-left: 2px;
}

.table-container {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 14px; padding: 20px;
    margin-bottom: 20px; overflow: hidden;
}
.table-title {
    font-size: 14px; font-weight: 600; color: rgba(255,255,255,0.6);
    text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 16px;
}

.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner table {
    border-collapse: separate; border-spacing: 0;
}
.dash-table-container .dash-header {
    background: rgba(255,255,255,0.04) !important;
    border-bottom: 1px solid rgba(255,255,255,0.08) !important;
}

.Select-menu-outer { background-color: #0a0a2e !important; border: 1px solid rgba(255,255,255,0.12) !important; }
.VirtualizedSelectOption { background-color: #0a0a2e !important; color: #fafafa !important; }
.VirtualizedSelectFocusedOption { background-color: rgba(234,103,255,0.2) !important; color: #fafafa !important; }
.Select-value-label { color: #fafafa !important; }
.Select-placeholder { color: rgba(255,255,255,0.35) !important; }
.Select-input input { color: #fafafa !important; }
.Select--multi .Select-value { background-color: rgba(234,103,255,0.2) !important; border-color: rgba(234,103,255,0.3) !important; }
.Select--multi .Select-value-label { color: #fafafa !important; }
.Select--multi .Select-value-icon { border-color: rgba(234,103,255,0.3) !important; color: rgba(255,255,255,0.6) !important; }
.Select--multi .Select-value-icon:hover { background-color: rgba(255,71,87,0.3) !important; color: #ff4757 !important; }
.Select-control { background-color: rgba(255,255,255,0.06) !important; border-color: rgba(255,255,255,0.08) !important; }
.Select-arrow-zone .Select-arrow { border-top-color: rgba(255,255,255,0.4) !important; }
.Select-clear { color: rgba(255,255,255,0.4) !important; }
"""

# Dropdown style
DROPDOWN_STYLE = {
    "backgroundColor": COLORS["bg_input"],
    "color": COLORS["text_primary"],
    "border": f"1px solid {COLORS['border']}",
    "borderRadius": "8px",
    "fontSize": "13px",
    "minWidth": "140px",
}


# =============================================================================
# Layout Components
# =============================================================================
def kpi_card(label, value, value_class="kpi-neutral"):
    return html.Div(className="kpi-card", children=[
        html.Div(label, className="kpi-label"),
        html.Div(value, className=f"kpi-value {value_class}"),
    ])


def section_label(text):
    return html.Div(text, className="section-label")


def _stat_row(label, value, color=None):
    val_color = color or COLORS["text_primary"]
    return html.Div(style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "6px 0",
    }, children=[
        html.Span(label, style={
            "fontSize": "12px", "color": COLORS["text_secondary"], "fontWeight": "400",
        }),
        html.Span(value, style={
            "fontSize": "14px", "fontFamily": "'JetBrains Mono', monospace",
            "fontWeight": "600", "color": val_color,
        }),
    ])


# =============================================================================
# App Layout
# =============================================================================
app = dash.Dash(
    __name__,
    title="NHL ML Model | Backtest Dashboard",
    update_title=None,
    suppress_callback_exceptions=True,
)

app.index_string = """<!DOCTYPE html>
<html>
<head>
    {%metas%}
    <title>{%title%}</title>
    {%favicon%}
    {%css%}
    <style>""" + STYLESHEET + """</style>
</head>
<body>
    {%app_entry%}
    <footer>{%config%}{%scripts%}{%renderer%}</footer>
</body>
</html>"""


def _build_clv_section():
    """Build the CLV comparison section (static, not filtered)."""
    clv = load_clv_data()
    if len(clv) == 0:
        return [
            section_label("Closing Line Value — 2pm EST vs 1hr Pre-Game"),
            html.Div(className="chart-card-full", children=[
                html.Div(
                    "No CLV data available. Need both three_way_odds_2pm_est.csv and three_way_odds.csv.",
                    style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"},
                ),
            ]),
        ]

    # Compute summary stats
    n_games = clv[["game_date", "home_team"]].drop_duplicates().shape[0]
    n_rows = len(clv)
    mean_clv = clv["draw_clv_pct"].mean()
    median_clv = clv["draw_clv_pct"].median()
    pct_positive = (clv["draw_clv_pct"] > 0).mean() * 100
    mean_ip_clv = clv["draw_ip_clv"].mean()

    # Build charts
    scatter = build_clv_scatter(clv)
    dist = build_clv_distribution(clv)
    by_book = build_clv_by_book(clv)
    over_time = build_clv_over_time(clv)

    return [
        section_label("Closing Line Value — 2pm EST vs 1hr Pre-Game"),

        # CLV KPI row
        html.Div(className="kpi-grid", style={"gridTemplateColumns": "repeat(6, 1fr)", "marginBottom": "20px"}, children=[
            kpi_card("Games Matched", f"{n_games:,}", "kpi-neutral"),
            kpi_card("Odds Rows", f"{n_rows:,}", "kpi-neutral"),
            kpi_card("Mean Draw CLV", f"{mean_clv:+.2f}%",
                     "kpi-positive" if mean_clv > 0 else "kpi-negative"),
            kpi_card("Median Draw CLV", f"{median_clv:+.2f}%",
                     "kpi-positive" if median_clv > 0 else "kpi-negative"),
            kpi_card("% Better at 2pm", f"{pct_positive:.1f}%",
                     "kpi-positive" if pct_positive > 50 else "kpi-negative"),
            kpi_card("Avg IP Shift", f"{mean_ip_clv:+.2f}pp",
                     "kpi-positive" if mean_ip_clv > 0 else "kpi-negative"),
        ]),

        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(figure=scatter, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(figure=dist, config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(figure=over_time, config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(figure=by_book, config={"displayModeBar": False}),
            ]),
        ]),
    ]


def serve_layout():
    """Build the full dashboard layout."""
    df = load_data()

    teams = sorted(set(
        df["home_team"].dropna().unique().tolist() +
        df["away_team"].dropna().unique().tolist()
    )) if len(df) > 0 else []

    min_date = df["game_date"].min() if len(df) > 0 else pd.Timestamp("2025-10-07")
    max_date = df["game_date"].max() if len(df) > 0 else pd.Timestamp("2026-02-05")

    range_input_style = {**DROPDOWN_STYLE, "width": "65px", "padding": "6px 8px"}

    return html.Div(className="dashboard-container", children=[

        # === HEADER ===
        html.Div(className="header", children=[
            html.Div(className="header-left", children=[
                html.Div("NHL WIN PROB", className="header-title"),
                html.Div("walk-forward backtest 2025-26", className="header-subtitle"),
            ]),
            html.Div("POISSON + OT EDGE + PLATT", className="header-badge"),
        ]),

        # === FILTER BAR ===
        html.Div(className="filter-bar", children=[
            html.Div(className="filter-group", children=[
                html.Div("DATE RANGE", className="filter-label"),
                dcc.DatePickerRange(
                    id="date-filter",
                    start_date=min_date, end_date=max_date,
                    display_format="MMM D", style={"fontSize": "12px"},
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("BET SIDE", className="filter-label"),
                dcc.Dropdown(
                    id="side-filter",
                    options=[{"label": "Home", "value": "home"}, {"label": "Away", "value": "away"}],
                    multi=True, placeholder="All", style=DROPDOWN_STYLE,
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("OUTCOME", className="filter-label"),
                dcc.Dropdown(
                    id="outcome-filter",
                    options=[{"label": t, "value": t} for t in ["REG", "OT", "SO"]],
                    multi=True, placeholder="All", style=DROPDOWN_STYLE,
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("EDGE %", className="filter-label"),
                html.Div(className="filter-range-pair", children=[
                    dcc.Input(
                        id="min-edge-filter", type="number", value=0,
                        step=1, min=0, max=50,
                        style=range_input_style, placeholder="Min",
                    ),
                    html.Span("–", className="filter-range-sep"),
                    dcc.Input(
                        id="max-edge-filter", type="number", value=None,
                        step=1, min=0, max=50,
                        style=range_input_style, placeholder="Max",
                    ),
                ]),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("TEAM", className="filter-label"),
                dcc.Dropdown(
                    id="team-filter",
                    options=[{"label": t, "value": t} for t in teams],
                    multi=True, placeholder="All",
                    style={**DROPDOWN_STYLE, "minWidth": "120px"},
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("DAY", className="filter-label"),
                dcc.Dropdown(
                    id="dow-filter",
                    options=[{"label": d[:3], "value": d} for d in
                             ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]],
                    multi=True, placeholder="All",
                    style={**DROPDOWN_STYLE, "minWidth": "100px"},
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("ODDS", className="filter-label"),
                html.Div(className="filter-range-pair", children=[
                    dcc.Input(
                        id="min-odds-filter", type="number", value=None,
                        step=0.05, min=1.0, max=10.0,
                        style=range_input_style, placeholder="Min",
                    ),
                    html.Span("–", className="filter-range-sep"),
                    dcc.Input(
                        id="max-odds-filter", type="number", value=None,
                        step=0.05, min=1.0, max=10.0,
                        style=range_input_style, placeholder="Max",
                    ),
                ]),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("P(OT) %", className="filter-label"),
                html.Div(className="filter-range-pair", children=[
                    dcc.Input(
                        id="min-pot-filter", type="number", value=None,
                        step=1, min=0, max=100,
                        style=range_input_style, placeholder="Min",
                    ),
                    html.Span("–", className="filter-range-sep"),
                    dcc.Input(
                        id="max-pot-filter", type="number", value=None,
                        step=1, min=0, max=100,
                        style=range_input_style, placeholder="Max",
                    ),
                ]),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("STAKING", className="filter-label"),
                dcc.Dropdown(
                    id="staking-filter",
                    options=[
                        {"label": "Flat (1u)", "value": "flat"},
                        {"label": "Kelly (¼K)", "value": "kelly"},
                    ],
                    value="flat", clearable=False,
                    style={**DROPDOWN_STYLE, "minWidth": "120px"},
                ),
            ]),
        ]),

        # === KPI CARDS ===
        html.Div(id="kpi-row"),

        # === PERFORMANCE ===
        section_label("Performance"),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="cumulative-pnl", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="rolling-ll", config={"displayModeBar": False}),
            ]),
        ]),

        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="daily-pnl", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="edge-roi", config={"displayModeBar": False}),
            ]),
        ]),

        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="pot-roi", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="split-pnl", config={"displayModeBar": False}),
            ]),
        ]),

        # === BETTING CONFIDENCE ===
        section_label("Betting Confidence"),
        html.Div(className="chart-grid-2-1", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="monthly-pnl-chart", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", id="confidence-panel"),
        ]),
        html.Div(className="chart-grid-3", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="fav-underdog-chart", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="drawdown-chart", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="outcome-type-chart", config={"displayModeBar": False}),
            ]),
        ]),

        # === CALIBRATION ===
        section_label("Calibration"),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="calibration-chart", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="ot-calibration", config={"displayModeBar": False}),
            ]),
        ]),

        # === SUBSET ANALYSIS ===
        section_label("Subset Analysis"),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="roi-heatmap", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="home-away-chart", config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card-full", children=[
            dcc.Graph(id="dow-roi-chart", config={"displayModeBar": False}),
        ]),

        # === MODEL DETAIL ===
        section_label("Model Detail"),
        html.Div(className="chart-grid-2-1", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="lambda-scatter", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", id="model-stats-panel"),
        ]),
        html.Div(className="chart-card-full", children=[
            dcc.Graph(id="edge-distribution", config={"displayModeBar": False}),
        ]),

        # === TEAM TABLE ===
        section_label("Team Breakdown"),
        html.Div(className="chart-card-full", children=[
            dcc.Graph(id="team-roi-chart", config={"displayModeBar": False}),
        ]),

        # === BET LOG ===
        section_label("Bet Log"),
        html.Div(className="table-container", children=[
            html.Div("RECENT BETS", className="table-title"),
            html.Div(id="bets-table-container"),
        ]),

        # === CLV ANALYSIS (static — not affected by filters) ===
        *_build_clv_section(),

        # Hidden data store
        dcc.Store(id="filtered-data-store"),
    ])


app.layout = serve_layout


# =============================================================================
# Callbacks
# =============================================================================
@app.callback(
    Output("filtered-data-store", "data"),
    [
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("side-filter", "value"),
        Input("outcome-filter", "value"),
        Input("min-edge-filter", "value"),
        Input("max-edge-filter", "value"),
        Input("team-filter", "value"),
        Input("dow-filter", "value"),
        Input("min-odds-filter", "value"),
        Input("max-odds-filter", "value"),
        Input("min-pot-filter", "value"),
        Input("max-pot-filter", "value"),
    ],
)
def filter_data(start_date, end_date, sides, outcomes, min_edge, max_edge,
                teams, dow, min_odds, max_odds, min_pot, max_pot):
    """Apply all filters and store filtered data."""
    df = load_data()
    if len(df) == 0:
        return df.to_json(date_format="iso", orient="split")

    if start_date:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    if sides:
        df = df[df["bet_side"].isin(sides)]

    if outcomes:
        df = df[df["game_outcome_type"].isin(outcomes)]

    if min_edge and min_edge > 0:
        df = df[df["best_edge"] >= min_edge / 100.0]
    if max_edge and max_edge > 0:
        df = df[df["best_edge"] <= max_edge / 100.0]

    if teams:
        df = df[(df["home_team"].isin(teams)) | (df["away_team"].isin(teams))]

    if dow:
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["game_date"].dt.day_name()
        df = df[df["day_of_week"].isin(dow)]

    if min_odds is not None:
        df = df[df["bet_odds_dec"] >= min_odds]
    if max_odds is not None:
        df = df[df["bet_odds_dec"] <= max_odds]

    if min_pot is not None:
        df = df[df["p_ot"] >= min_pot / 100.0]
    if max_pot is not None:
        df = df[df["p_ot"] <= max_pot / 100.0]

    return df.to_json(date_format="iso", orient="split")


@app.callback(
    [
        Output("kpi-row", "children"),
        Output("cumulative-pnl", "figure"),
        Output("rolling-ll", "figure"),
        Output("daily-pnl", "figure"),
        Output("edge-roi", "figure"),
        Output("pot-roi", "figure"),
        Output("monthly-pnl-chart", "figure"),
        Output("confidence-panel", "children"),
        Output("fav-underdog-chart", "figure"),
        Output("drawdown-chart", "figure"),
        Output("calibration-chart", "figure"),
        Output("ot-calibration", "figure"),
        Output("split-pnl", "figure"),
        Output("roi-heatmap", "figure"),
        Output("home-away-chart", "figure"),
        Output("outcome-type-chart", "figure"),
        Output("dow-roi-chart", "figure"),
        Output("lambda-scatter", "figure"),
        Output("model-stats-panel", "children"),
        Output("edge-distribution", "figure"),
        Output("team-roi-chart", "figure"),
        Output("bets-table-container", "children"),
    ],
    Input("filtered-data-store", "data"),
    Input("staking-filter", "value"),
)
def update_all(json_data, staking):
    """Update all dashboard components from filtered data."""
    df = pd.read_json(io.StringIO(json_data), orient="split") if json_data else pd.DataFrame()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    # Recompute derived columns
    if len(df) > 0:
        df["day_of_week"] = df["game_date"].dt.day_name()
        df["is_ot"] = df["game_outcome_type"].isin(["OT", "SO"]).astype(int)
        df["best_edge"] = df[["home_edge", "away_edge"]].max(axis=1)
        has_bet = df["bet_side"].notna() & df["bet_won"].notna()
        df["flat_profit"] = np.nan
        df.loc[has_bet, "flat_profit"] = df.loc[has_bet].apply(
            lambda r: (r["bet_odds_dec"] - 1.0) if r["bet_won"] == 1 else -1.0, axis=1
        )

        # Apply staking mode — overwrite flat_profit with kelly profit
        if staking == "kelly":
            df["flat_profit"] = np.nan
            df.loc[has_bet, "flat_profit"] = df.loc[has_bet, "bet_profit"]

    # KPIs
    kpis = compute_kpis(df)

    def val_class(val_str):
        if "+" in str(val_str):
            return "kpi-positive"
        if str(val_str).startswith("-"):
            return "kpi-negative"
        return "kpi-neutral"

    roi_label = "Flat ROI" if staking != "kelly" else "Kelly ROI"
    kpi_row = html.Div(className="kpi-grid", children=[
        kpi_card(roi_label, kpis.get("roi_flat", "—"), val_class(kpis.get("roi_flat", "—"))),
        kpi_card("Total Bets", kpis.get("total_bets", "—"), "kpi-neutral"),
        kpi_card("Win Rate", kpis.get("win_rate", "—"), "kpi-neutral"),
        kpi_card("vs DK", kpis.get("ll_vs_dk", "—"), val_class(kpis.get("ll_vs_dk", "—"))),
        kpi_card("Sharpe", kpis.get("sharpe", "—"), val_class(kpis.get("sharpe", "—"))),
        kpi_card("Max DD", kpis.get("max_drawdown", "—"),
                 "kpi-negative" if kpis.get("max_drawdown", "—") != "—" else "kpi-neutral"),
        kpi_card("t-stat", kpis.get("t_stat", "—"),
                 "kpi-positive" if kpis.get("significant") == "Yes" else "kpi-neutral"),
        kpi_card("p-value", kpis.get("p_value", "—"),
                 "kpi-positive" if kpis.get("significant") == "Yes" else "kpi-negative"),
    ])

    # Charts
    cum_pnl = build_cumulative_pnl(df, staking=staking or "flat")
    roll_ll = build_rolling_ll(df)
    daily_pnl = build_daily_pnl_bars(df)
    edge_roi = build_edge_roi_chart(df)
    pot_roi = build_pot_roi_chart(df)
    monthly_pnl = build_monthly_pnl_chart(df)
    fav_dog = build_fav_underdog_chart(df)
    dd_chart = build_drawdown_chart(df)
    cal_chart = build_calibration_chart(df)
    ot_cal = build_ot_calibration(df)
    split_pnl = build_split_pnl(df)
    roi_heatmap = build_roi_heatmap(df)
    ha_chart = build_home_away_chart(df)
    oc_chart = build_outcome_type_chart(df)
    dow_roi = build_dow_roi_chart(df)
    lam_scatter = build_lambda_scatter(df)
    edge_dist = build_edge_distribution(df)
    team_roi = build_team_roi_chart(df)

    # Update chart titles with staking mode
    staking_label = "FLAT" if staking != "kelly" else "KELLY"
    title_font = dict(size=14, color=COLORS["text_secondary"])
    daily_pnl.update_layout(title=dict(text=f"DAILY P&L ({staking_label})", font=title_font))
    edge_roi.update_layout(title=dict(text=f"ROI BY EDGE BUCKET ({staking_label})", font=title_font))
    pot_roi.update_layout(title=dict(text=f"ROI BY P(OT) BUCKET ({staking_label})", font=title_font))
    ha_chart.update_layout(title=dict(text=f"HOME vs AWAY ({staking_label})", font=title_font))
    oc_chart.update_layout(title=dict(text=f"ROI BY OUTCOME TYPE ({staking_label})", font=title_font))
    fav_dog.update_layout(title=dict(text=f"FAVORITE vs UNDERDOG ({staking_label})", font=title_font))

    # Confidence panel
    sig_color = COLORS["accent_green"] if kpis.get("significant") == "Yes" else COLORS["accent_red"]
    confidence_panel = html.Div(style={"padding": "10px 0"}, children=[
        html.Div("SAMPLE CONFIDENCE", style={
            "fontSize": "14px", "fontWeight": "600", "color": COLORS["text_secondary"],
            "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "20px",
        }),
        _stat_row("Total Bets", kpis.get("total_bets", "—")),
        _stat_row("Home / Away", f"{kpis.get('home_bets', '—')} / {kpis.get('away_bets', '—')}"),
        _stat_row("Flat P&L" if staking != "kelly" else "Kelly P&L", kpis.get("flat_pnl", "—"),
                  COLORS["accent_green"] if kpis.get("flat_pnl", "—").startswith("+") else COLORS["accent_red"]),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("t-statistic", kpis.get("t_stat", "—"), sig_color),
        _stat_row("p-value", kpis.get("p_value", "—"), sig_color),
        _stat_row("Significant (5%)", kpis.get("significant", "—"), sig_color),
        _stat_row("95% CI (ROI)", kpis.get("ci_95", "—")),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("Games to Signif.", kpis.get("n_remaining", "—"), COLORS["accent_gold"]),
        _stat_row("Fav ROI", kpis.get("fav_roi", "—")),
        _stat_row("Underdog ROI", kpis.get("dog_roi", "—")),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("Max Win Streak", kpis.get("max_win_streak", "—"), COLORS["accent_green"]),
        _stat_row("Max Loss Streak", kpis.get("max_loss_streak", "—"), COLORS["accent_red"]),
        _stat_row("Avg Edge", kpis.get("avg_edge", "—")),
    ])

    # Model stats panel
    if len(df) > 0:
        with_odds = df.dropna(subset=["dk_home_fair"])
        n_total = len(df)
        n_odds = len(with_odds)
        ot_rate = df["is_ot"].mean() if "is_ot" in df.columns else 0
        platt_on = (df["platt_a"] != 1.0).sum() if "platt_a" in df.columns else 0

        y = df["home_won_actual"].values.astype(float)
        p = np.clip(df["p_home_win"].values, 0.001, 0.999)
        base_rate = y.mean()
        ll = -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean()
        base_ll = -(y * np.log(base_rate) + (1 - y) * np.log(1 - base_rate)).mean() if 0 < base_rate < 1 else ll
        ll_imp = (1 - ll / base_ll) * 100 if base_ll > 0 else 0

        model_panel = html.Div(style={"padding": "10px 0"}, children=[
            html.Div("MODEL STATS", style={
                "fontSize": "14px", "fontWeight": "600", "color": COLORS["text_secondary"],
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "20px",
            }),
            _stat_row("Total Games", f"{n_total}"),
            _stat_row("With DK Odds", f"{n_odds}"),
            _stat_row("Home Win Rate", f"{base_rate:.3f}"),
            _stat_row("OT Rate", f"{ot_rate:.3f}"),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
            _stat_row("LL Improvement", f"{ll_imp:+.1f}%", COLORS["accent_cyan"]),
            _stat_row("Platt Active", f"{platt_on} games", COLORS["accent_magenta"]),
            _stat_row("Avg λ Home", f"{df['lam_home'].mean():.3f}", COLORS["accent_gold"]),
            _stat_row("Avg λ Away", f"{df['lam_away'].mean():.3f}", COLORS["accent_gold"]),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
            _stat_row("Anchor Home", f"{df['anchor_home'].mean():.4f}"),
            _stat_row("Anchor Away", f"{df['anchor_away'].mean():.4f}"),
            _stat_row("Best Day", kpis.get("best_day", "—")),
            _stat_row("Worst Day", kpis.get("worst_day", "—")),
        ])
    else:
        model_panel = html.Div("No data", style={"color": COLORS["text_muted"], "padding": "40px", "textAlign": "center"})

    # Bets table
    bets_table = _build_bets_table(df, staking=staking or "flat")

    return (
        kpi_row, cum_pnl, roll_ll, daily_pnl, edge_roi,
        pot_roi, monthly_pnl, confidence_panel, fav_dog, dd_chart,
        cal_chart, ot_cal,
        split_pnl, roi_heatmap, ha_chart, oc_chart, dow_roi,
        lam_scatter, model_panel, edge_dist, team_roi,
        bets_table,
    )


def _build_bets_table(df, staking="flat"):
    """Build the bet log DataTable."""
    if len(df) == 0:
        return html.Div("No data", style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})

    display = df.copy()
    display["game_date_str"] = display["game_date"].dt.strftime("%b %d")
    display["home_edge_pct"] = (display["home_edge"] * 100).round(1)
    display["away_edge_pct"] = (display["away_edge"] * 100).round(1)
    display["p_home_pct"] = (display["p_home_win"] * 100).round(1)
    display["dk_home_pct"] = (display["dk_home_fair"] * 100).round(1)
    display["lam_home"] = display["lam_home"].round(2)
    display["lam_away"] = display["lam_away"].round(2)
    display["p_ot_pct"] = (display["p_ot"] * 100).round(1)
    display["flat_profit"] = display["flat_profit"].round(2)
    if staking == "kelly":
        display["stake"] = display["kelly_stake"].round(3)
    else:
        display["stake"] = np.where(display["bet_side"].notna(), 1.0, np.nan)

    display_cols = [
        {"name": "Date", "id": "game_date_str"},
        {"name": "Home", "id": "home_team"},
        {"name": "Away", "id": "away_team"},
        {"name": "λH", "id": "lam_home"},
        {"name": "λA", "id": "lam_away"},
        {"name": "P(OT)", "id": "p_ot_pct"},
        {"name": "Model%", "id": "p_home_pct"},
        {"name": "DK%", "id": "dk_home_pct"},
        {"name": "H Edge", "id": "home_edge_pct"},
        {"name": "A Edge", "id": "away_edge_pct"},
        {"name": "Bet", "id": "bet_side"},
        {"name": "Stake", "id": "stake"},
        {"name": "Result", "id": "game_outcome_type"},
        {"name": "P&L" if staking == "flat" else "P&L (K)", "id": "flat_profit"},
    ]

    return dash_table.DataTable(
        data=display.sort_values("game_date", ascending=False).head(200).to_dict("records"),
        columns=display_cols,
        sort_action="native", sort_mode="single", page_size=20,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "rgba(255,255,255,0.04)",
            "color": COLORS["text_secondary"],
            "fontWeight": "600", "fontSize": "10px",
            "textTransform": "uppercase", "letterSpacing": "0.5px",
            "border": "none", "borderBottom": f"1px solid {COLORS['border']}",
            "padding": "8px 10px", "fontFamily": "Inter, sans-serif",
        },
        style_cell={
            "backgroundColor": "transparent",
            "color": COLORS["text_primary"],
            "border": "none",
            "borderBottom": "1px solid rgba(255,255,255,0.04)",
            "padding": "8px 10px", "fontSize": "12px",
            "fontFamily": "'JetBrains Mono', monospace",
            "textAlign": "right", "minWidth": "55px", "maxWidth": "120px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "home_team"}, "textAlign": "center", "fontFamily": "Inter, sans-serif", "fontWeight": "500"},
            {"if": {"column_id": "away_team"}, "textAlign": "center", "fontFamily": "Inter, sans-serif", "fontWeight": "500"},
            {"if": {"column_id": "game_date_str"}, "textAlign": "left", "minWidth": "70px"},
            {"if": {"column_id": "bet_side"}, "textAlign": "center"},
            {"if": {"column_id": "game_outcome_type"}, "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{flat_profit} > 0", "column_id": "flat_profit"}, "color": COLORS["profit_green"]},
            {"if": {"filter_query": "{flat_profit} < 0", "column_id": "flat_profit"}, "color": COLORS["loss_red"]},
            {"if": {"filter_query": '{bet_side} = "home"', "column_id": "bet_side"}, "color": COLORS["accent_green"]},
            {"if": {"filter_query": '{bet_side} = "away"', "column_id": "bet_side"}, "color": COLORS["accent_cyan"]},
            {"if": {"state": "active"}, "backgroundColor": "rgba(234,103,255,0.05)", "border": "none"},
        ],
        style_as_list_view=True,
    )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NHL WIN PROBABILITY — Backtest Dashboard")
    print("=" * 60)

    if os.path.exists(BT_FILE):
        df = pd.read_csv(BT_FILE)
        print(f"Loaded {len(df):,} games from {BT_FILE}")
    else:
        print(f"WARNING: {BT_FILE} not found. Dashboard will show empty state.")
        print("Run walk_forward.py first to generate data.")

    print(f"\nStarting dashboard at http://127.0.0.1:{PORT}")
    app.run(debug=False, port=PORT, host="127.0.0.1")
