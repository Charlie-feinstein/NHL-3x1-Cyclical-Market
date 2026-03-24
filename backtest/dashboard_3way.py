# -*- coding: utf-8 -*-
"""
NHL 3-Way Regulation Model — Backtest Dashboard

Dark-themed, Whizard-inspired analytics dashboard for the Poisson combiner
3-way regulation market (draw-only betting). Evaluates walk-forward backtest
performance across 2024-25 and 2025-26 seasons.

Run: python dashboard_3way.py
Open: http://127.0.0.1:8052

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
BT_FILE = os.path.join(PROJECT_DIR, "backtest", "backtest_results_3way.csv")
RAW_DIR = os.path.join(PROJECT_DIR, "data", "raw")
ODDS_2PM_FILE = os.path.join(RAW_DIR, "three_way_odds_2pm_est.csv")
ODDS_1HR_FILE = os.path.join(RAW_DIR, "three_way_odds.csv")
PORT = 8056

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
    """Load 3-way backtest results and compute derived columns."""
    if not os.path.exists(BT_FILE):
        return pd.DataFrame()

    df = pd.read_csv(BT_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Derived columns
    df["day_of_week"] = df["game_date"].dt.day_name()
    df["is_ot"] = df["game_outcome_type"].isin(["OT", "SO"]).astype(int)

    # Ensure bookmaker column exists
    if "bookmaker" not in df.columns:
        df["bookmaker"] = "unknown"
    df["bookmaker"] = df["bookmaker"].fillna("unknown").replace("", "unknown")

    # Flat staking profit (1 unit on draw when bet placed)
    df["flat_profit"] = np.nan
    has_bet = df["bet_side"].notna() & df["bet_won"].notna()
    df.loc[has_bet, "flat_profit"] = df.loc[has_bet].apply(
        lambda r: (r["bet_odds_dec"] - 1.0) if r["bet_won"] == 1 else -1.0, axis=1
    )

    # Matchup closeness: abs(lam_home - lam_away) — closer = more likely draw
    df["lam_diff"] = (df["lam_home"] - df["lam_away"]).abs()

    return df


# =============================================================================
# KPI Computation
# =============================================================================
def compute_kpis(df):
    """Compute headline KPIs from 3-way backtest results."""
    with_odds = df.dropna(subset=["fair_draw_3w"])

    if len(with_odds) == 0:
        return {k: "—" for k in [
            "total_games", "games_with_odds", "win_rate", "roi_flat",
            "avg_edge", "sharpe", "max_drawdown", "best_day", "worst_day",
        ]}

    n = len(with_odds)

    # Flat staking
    flat_bets = df[df["flat_profit"].notna()]
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

    # Draw calibration
    actual_draw_rate = (with_odds["actual_3way"] == "draw").mean()
    model_draw_mean = with_odds["p_ot"].mean()
    market_draw_mean = with_odds["fair_draw_3w"].mean()

    # Statistical significance
    if len(flat_bets) > 1 and flat_bets["flat_profit"].std() > 0:
        mean_pnl = flat_bets["flat_profit"].mean()
        std_pnl = flat_bets["flat_profit"].std()
        se_pnl = std_pnl / np.sqrt(flat_n)
        t_stat = mean_pnl / se_pnl
        p_value = erfc(abs(t_stat) / np.sqrt(2))
        ci_lo = (mean_pnl - 1.96 * se_pnl) / 1.0 * 100
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

    # Avg draw odds for bets
    avg_draw_odds = flat_bets["bet_odds_dec"].mean() if len(flat_bets) > 0 else 0

    return {
        "total_games": f"{len(df):,}",
        "games_with_odds": f"{n:,}",
        "total_bets": f"{flat_n:,}",
        "win_rate": f"{flat_wins/flat_n:.1%}" if flat_n > 0 else "—",
        "roi_flat": f"{flat_roi:+.1f}%",
        "avg_edge": f"{flat_bets['bet_edge'].mean()*100:.1f}%" if len(flat_bets) > 0 else "—",
        "sharpe": f"{sharpe:.2f}",
        "max_drawdown": f"{max_dd:.1f}u",
        "best_day": f"+{daily.max():.1f}u" if len(daily) > 0 else "—",
        "worst_day": f"{daily.min():.1f}u" if len(daily) > 0 else "—",
        "t_stat": f"{t_stat:.2f}",
        "p_value": f"{p_value:.3f}",
        "ci_95": f"[{ci_lo:+.1f}, {ci_hi:+.1f}]%",
        "n_needed": f"{n_needed:,}",
        "n_remaining": f"{max(0, n_needed - flat_n):,}",
        "significant": "Yes" if p_value < 0.05 else "No",
        "max_win_streak": f"{max_win_streak}",
        "max_loss_streak": f"{max_loss_streak}",
        "flat_pnl": f"{flat_pnl:+.1f}u",
        "actual_draw_rate": f"{actual_draw_rate:.1%}",
        "model_draw_mean": f"{model_draw_mean:.1%}",
        "market_draw_mean": f"{market_draw_mean:.1%}",
        "avg_draw_odds": f"{avg_draw_odds:.3f}",
        "draw_bets": f"{flat_n}",
    }


# =============================================================================
# Chart Builders
# =============================================================================
def build_all_markets_pnl(df, staking="flat", min_edge=None, max_edge=None):
    """Overlay cumulative P&L for Draw, Home Reg, and Away Reg (+edge bets only).

    Each market independently applies its own edge threshold so the chart
    is consistent regardless of which single-market view is selected.
    """
    if len(df) == 0:
        return empty_chart("No data")

    min_e = (min_edge / 100.0) if min_edge and min_edge > 0 else 0.0
    max_e = (max_edge / 100.0) if max_edge and max_edge > 0 else None

    market_config = [
        ("draw", "draw_edge", "dec_draw_3w", "p_ot",
         lambda d: d["actual_3way"].isin(["OT", "SO", "draw"]),
         "Draw (OT)", COLORS["accent_gold"], "255,215,0"),
        ("home", "home_edge", "dec_home_3w", "p_home_reg_win",
         lambda d: d["actual_3way"] == "home",
         "Home Reg", COLORS["accent_green"], "0,255,136"),
        ("away", "away_edge", "dec_away_3w", "p_away_reg_win",
         lambda d: d["actual_3way"] == "away",
         "Away Reg", COLORS["accent_cyan"], "0,163,255"),
    ]

    fig = go.Figure()
    summary_lines = []

    for mkt, edge_col, dec_col, prob_col, win_fn, label, color, rgb in market_config:
        if edge_col not in df.columns or dec_col not in df.columns:
            continue

        has_edge = df[edge_col].notna() & (df[edge_col] >= max(min_e, 1e-9))
        if max_e is not None:
            has_edge = has_edge & (df[edge_col] <= max_e)
        bets = df[has_edge].copy()
        if len(bets) == 0:
            continue

        win_cond = win_fn(bets)

        if staking == "kelly" and prob_col in bets.columns:
            b = bets[dec_col] - 1.0
            p = bets[prob_col]
            q = 1.0 - p
            kelly_f = ((b * p - q) / b).clip(lower=0)
            stake = kelly_f * 0.25
            bets["mkt_profit"] = np.where(win_cond, stake * b, -stake)
        else:
            bets["mkt_profit"] = np.where(win_cond, bets[dec_col] - 1.0, -1.0)

        daily = bets.groupby("game_date")["mkt_profit"].sum().reset_index()
        daily["cumulative"] = daily["mkt_profit"].cumsum()

        n_bets = len(bets)
        pnl = bets["mkt_profit"].sum()
        roi = pnl / n_bets * 100 if n_bets > 0 else 0
        summary_lines.append(f"{label}: {n_bets} bets, {pnl:+.1f}u, {roi:+.1f}% ROI")

        fig.add_trace(go.Scatter(
            x=daily["game_date"], y=daily["cumulative"],
            fill="tozeroy", fillcolor=f"rgba({rgb}, 0.06)",
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>%{{x|%b %d}}</b><br>{label}: %{{y:+.1f}}u<extra></extra>"
            ),
            name=f"{label} ({n_bets}, {roi:+.1f}%)",
        ))

    # Add combined Blend line (Draw + Home only)
    blend_parts = []
    for mkt, edge_col, dec_col, prob_col, win_fn, label, color, rgb in market_config:
        if mkt == "away" or edge_col not in df.columns:
            continue
        if mkt == "draw":
            has_e = df[edge_col].notna() & (df[edge_col] >= max(min_e, 1e-9))
        else:  # home — also cap at 5%
            has_e = df[edge_col].notna() & (df[edge_col] >= max(min_e, 1e-9)) & (df[edge_col] <= 0.05)
        if max_e is not None:
            has_e = has_e & (df[edge_col] <= max_e)
        bets = df[has_e].copy()
        if len(bets) == 0:
            continue
        wc = win_fn(bets)
        # Edge-proportional staking for blend
        if mkt == "draw":
            bets["stake"] = (bets[edge_col] * 25).clip(upper=1.5)
        else:
            bets["stake"] = (bets[edge_col] * 20).clip(upper=1.0)
        bets["mkt_profit"] = np.where(wc, bets["stake"] * (bets[dec_col] - 1.0), -bets["stake"])
        blend_parts.append(bets[["game_date", "mkt_profit"]])

    if blend_parts:
        blend_all = pd.concat(blend_parts)
        blend_daily = blend_all.groupby("game_date")["mkt_profit"].sum().reset_index()
        blend_daily["cumulative"] = blend_daily["mkt_profit"].cumsum()
        blend_n = len(blend_all)
        blend_pnl = blend_all["mkt_profit"].sum()
        blend_staked = sum(p["mkt_profit"].abs().sum() for p in blend_parts)  # approx
        blend_roi = blend_pnl / blend_n * 100 if blend_n > 0 else 0
        fig.add_trace(go.Scatter(
            x=blend_daily["game_date"], y=blend_daily["cumulative"],
            line=dict(color=COLORS["accent_magenta"], width=3, dash="dash"),
            hovertemplate="<b>%{x|%b %d}</b><br>Blend: %{y:+.1f}u<extra></extra>",
            name=f"Blend ({blend_n}, {blend_roi:+.1f}%)",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    staking_label = "FLAT" if staking != "kelly" else "KELLY"
    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text=f"ALL MARKETS — CUMULATIVE P&L (EDGE-PROP)",
                   font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="Units",
        height=400,
    )
    fig.update_layout(legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=COLORS["text_primary"]),
    ))
    return fig


def build_cumulative_pnl(df, staking="flat"):
    """Cumulative P&L for selected staking mode."""
    graded = df.dropna(subset=["flat_profit"]).sort_values("game_date")
    if len(graded) == 0:
        return empty_chart("No graded bets")

    fig = go.Figure()

    staking_label = "Flat (1u)" if staking == "flat" else "Kelly"
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


def build_season_split_pnl(df):
    """Cumulative P&L split by season."""
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

    # By season
    season_colors = {
        "2024-25": (COLORS["accent_green"], "rgba(0,255,136,0.06)"),
        "2025-26": (COLORS["accent_cyan"], "rgba(0,163,255,0.06)"),
    }
    for season in sorted(graded["season_label"].unique()):
        s = graded[graded["season_label"] == season]
        if len(s) > 0:
            daily_s = s.groupby("game_date")["flat_profit"].sum().reset_index()
            daily_s["cumulative"] = daily_s["flat_profit"].cumsum()
            color, fill = season_colors.get(season, (COLORS["accent_gold"], "rgba(255,215,0,0.06)"))
            fig.add_trace(go.Scatter(
                x=daily_s["game_date"], y=daily_s["cumulative"],
                fill="tozeroy", fillcolor=fill,
                line=dict(color=color, width=2),
                hovertemplate=f"<b>%{{x|%b %d}}</b><br>{season}: %{{y:+.1f}}u<extra></extra>",
                name=season,
            ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="CUMULATIVE P&L BY SEASON", font=dict(size=14, color=COLORS["text_secondary"])),
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


def build_draw_calibration(df):
    """P(Draw/OT) calibration — model probability vs actual draw rate."""
    valid = df.dropna(subset=["p_ot"]).copy()
    if len(valid) < 50:
        return empty_chart("Not enough data")

    valid["is_draw"] = (valid["actual_3way"] == "draw").astype(int)
    bins = np.arange(0.16, 0.34, 0.02)
    valid["prob_bin"] = pd.cut(valid["p_ot"], bins=bins)
    cal = valid.groupby("prob_bin", observed=True).agg(
        actual=("is_draw", "mean"),
        model=("p_ot", "mean"),
        n=("is_draw", "count"),
    ).reset_index()
    cal = cal[cal["n"] >= 5]

    fig = go.Figure()

    # Perfect line
    fig.add_trace(go.Scatter(
        x=[0.16, 0.34], y=[0.16, 0.34],
        mode="lines", line=dict(color=COLORS["text_muted"], dash="dash", width=1),
        showlegend=False,
    ))

    # Actual
    fig.add_trace(go.Scatter(
        x=cal["model"], y=cal["actual"],
        mode="markers+lines",
        marker=dict(
            size=cal["n"].clip(upper=200) / 5 + 6,
            color=COLORS["accent_gold"],
            line=dict(width=1, color="rgba(255,215,0,0.5)"),
        ),
        line=dict(color=COLORS["accent_gold"], width=2),
        hovertemplate="Model: %{x:.1%}<br>Actual: %{y:.1%}<br>n=%{text}<extra></extra>",
        text=cal["n"],
        name="P(Draw)",
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="P(DRAW) CALIBRATION", font=dict(size=14, color=COLORS["text_secondary"])),
        showlegend=False, height=360,
    )
    fig.update_layout(
        xaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%", range=[0.16, 0.34],
                   title="Model P(Draw)", tickfont=dict(size=11, color=COLORS["text_secondary"])),
        yaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%", range=[0.16, 0.34],
                   title="Actual Draw Rate", tickfont=dict(size=11, color=COLORS["text_secondary"])),
    )
    return fig


def build_edge_vs_market(df):
    """Model P(draw) vs Market P(draw) scatter — each game is a point."""
    valid = df.dropna(subset=["fair_draw_3w", "p_ot"]).copy()
    if len(valid) < 10:
        return empty_chart("Not enough data")

    valid["is_draw"] = (valid["actual_3way"] == "draw").astype(int)

    fig = go.Figure()

    for drawn, label, color in [
        (1, "Draw (OT/SO)", COLORS["accent_gold"]),
        (0, "Regulation", COLORS["text_muted"]),
    ]:
        subset = valid[valid["is_draw"] == drawn]
        fig.add_trace(go.Scatter(
            x=subset["fair_draw_3w"], y=subset["p_ot"],
            mode="markers",
            marker=dict(size=5, color=color, opacity=0.5 if drawn == 0 else 0.8),
            name=label,
            hovertemplate=(
                "<b>%{customdata[0]} vs %{customdata[1]}</b><br>"
                "Market: %{x:.1%}<br>Model: %{y:.1%}<extra></extra>"
            ),
            customdata=subset[["home_team", "away_team"]].values,
        ))

    # Diagonal
    fig.add_trace(go.Scatter(
        x=[0.15, 0.35], y=[0.15, 0.35],
        mode="lines", line=dict(color=COLORS["accent_red"], dash="dash", width=1),
        showlegend=False,
    ))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="MODEL vs MARKET P(DRAW)", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Market Fair P(Draw)", yaxis_title="Model P(Draw)", height=360,
    )
    fig.update_layout(
        xaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%",
                   tickfont=dict(size=11, color=COLORS["text_secondary"])),
        yaxis=dict(gridcolor=COLORS["chart_grid"], tickformat=".0%",
                   tickfont=dict(size=11, color=COLORS["text_secondary"])),
    )
    return fig


def build_edge_roi_chart(df):
    """ROI by draw edge bucket — monotonicity check."""
    graded = df.dropna(subset=["flat_profit", "bet_edge"])
    if len(graded) == 0:
        return empty_chart("No data")

    bins = [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 1.0]
    labels = [f"{bins[i]*100:.0f}-{bins[i+1]*100:.0f}%" for i in range(len(bins)-1)]
    graded = graded.copy()
    graded["edge_bucket"] = pd.cut(graded["bet_edge"], bins=bins, labels=labels)

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
        title=dict(text="ROI BY DRAW EDGE BUCKET (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
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


def build_closeness_roi_chart(df):
    """ROI by matchup closeness — |λ_home - λ_away| buckets."""
    graded = df.dropna(subset=["flat_profit", "lam_diff"])
    if len(graded) == 0:
        return empty_chart("No data")

    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 5.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    graded = graded.copy()
    graded["close_bucket"] = pd.cut(graded["lam_diff"], bins=bins, labels=labels)

    by_close = graded.groupby("close_bucket", observed=True).agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
        avg_pot=("p_ot", "mean"),
    ).reset_index()

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_close["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_close["close_bucket"], y=by_close["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="|Δλ|: %{x}<br>ROI: %{y:+.1f}%<br>Win: %{customdata[0]:.1%}<br>Avg P(OT): %{customdata[1]:.1%}<extra></extra>",
        customdata=by_close[["win_rate", "avg_pot"]].values,
        text=by_close["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY MATCHUP CLOSENESS (|Δλ|)", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="|λ Home − λ Away|",
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_outcome_type_chart(df):
    """ROI by game outcome type (REG / OT / SO) when we bet draw."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    by_type = graded.groupby("game_outcome_type").agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
    ).reset_index()

    type_colors = {
        "REG": COLORS["accent_red"],
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
    """Lambda home vs away scatter — color by outcome (draw highlighted)."""
    valid = df.dropna(subset=["lam_home", "lam_away"])
    if len(valid) == 0:
        return empty_chart("No data")

    fig = go.Figure()

    for outcome, label, color, opacity in [
        ("home", "Home Reg Win", COLORS["accent_green"], 0.25),
        ("away", "Away Reg Win", COLORS["accent_cyan"], 0.25),
        ("draw", "Draw (OT/SO)", COLORS["accent_gold"], 0.7),
    ]:
        subset = valid[valid["actual_3way"] == outcome]
        fig.add_trace(go.Scatter(
            x=subset["lam_home"], y=subset["lam_away"],
            mode="markers",
            marker=dict(size=5 if outcome != "draw" else 7, color=color, opacity=opacity),
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
    """Histogram of draw edge (model P(draw) - market fair P(draw))."""
    valid = df.dropna(subset=["draw_edge"])
    if len(valid) == 0:
        return empty_chart("No data")

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=valid["draw_edge"] * 100,
        nbinsx=40,
        marker_color=COLORS["accent_gold"],
        opacity=0.7,
        hovertemplate="Edge: %{x:.1f}%<br>Count: %{y}<extra></extra>",
    ))

    mean_edge = valid["draw_edge"].mean() * 100
    fig.add_vline(x=mean_edge, line=dict(color=COLORS["accent_cyan"], width=2, dash="dash"),
                  annotation_text=f"Mean={mean_edge:.1f}%", annotation_font_color=COLORS["accent_cyan"])
    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="DRAW EDGE DISTRIBUTION", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Draw Edge %", yaxis_title="Count",
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


def build_season_comparison(df):
    """Season-by-season ROI comparison bar chart."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    by_season = graded.groupby("season_label").agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
        avg_edge=("bet_edge", "mean"),
        profit=("flat_profit", "sum"),
    ).reset_index()

    season_colors = {
        "2024-25": COLORS["accent_green"],
        "2025-26": COLORS["accent_cyan"],
    }

    fig = go.Figure()
    for _, row in by_season.iterrows():
        fig.add_trace(go.Bar(
            x=[row["season_label"]],
            y=[row["roi"]],
            marker_color=season_colors.get(row["season_label"], COLORS["accent_magenta"]),
            opacity=0.85, name=row["season_label"],
            hovertemplate=(
                f"<b>{row['season_label']}</b><br>"
                f"ROI: {row['roi']:+.1f}%<br>Win: {row['win_rate']:.1%}<br>"
                f"Avg Edge: {row['avg_edge']*100:.1f}%<br>"
                f"P&L: {row['profit']:+.1f}u<br>"
                f"n={int(row['n'])}<extra></extra>"
            ),
            text=[f"{row['roi']:+.1f}%\nn={int(row['n'])}"],
            textposition="outside", textfont=dict(size=11, color=COLORS["text_primary"]),
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY SEASON (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        yaxis_title="ROI %", showlegend=False, height=300,
    )
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


def build_odds_roi_chart(df):
    """ROI by odds bucket — bins adapt to draw vs regulation odds."""
    graded = df.dropna(subset=["flat_profit", "bet_odds_dec"])
    if len(graded) == 0:
        return empty_chart("No data")

    # Use different bins for draw (3-10) vs regulation (1.5-4) odds ranges
    median_odds = graded["bet_odds_dec"].median()
    if median_odds > 3.0:
        bins = [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 10.0]
    else:
        bins = [1.3, 1.6, 1.8, 2.0, 2.3, 2.7, 3.2, 5.0]
    labels = [f"{bins[i]:.1f}-{bins[i+1]:.1f}" for i in range(len(bins)-1)]
    graded = graded.copy()
    graded["odds_bucket"] = pd.cut(graded["bet_odds_dec"], bins=bins, labels=labels)

    by_odds = graded.groupby("odds_bucket", observed=True).agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
    ).reset_index()

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_odds["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=by_odds["odds_bucket"], y=by_odds["roi"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="Odds: %{x}<br>ROI: %{y:+.1f}%<br>Win: %{customdata:.1%}<extra></extra>",
        customdata=by_odds["win_rate"],
        text=by_odds["n"].apply(lambda n: f"n={n}"),
        textposition="outside", textfont=dict(size=10, color=COLORS["text_muted"]),
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY DRAW ODDS", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="Draw Decimal Odds",
        yaxis_title="ROI %", showlegend=False, height=300,
    )
    return fig


def build_book_roi_chart(df):
    """ROI by bookmaker."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    if "bookmaker" not in graded.columns:
        return empty_chart("No bookmaker data")

    by_book = graded.groupby("bookmaker").agg(
        n=("flat_profit", "count"),
        roi=("flat_profit", lambda x: x.sum() / len(x) * 100),
        win_rate=("bet_won", "mean"),
        avg_edge=("bet_edge", "mean"),
        avg_odds=("bet_odds_dec", "mean"),
        profit=("flat_profit", "sum"),
    ).reset_index()
    by_book = by_book[by_book["n"] >= 3]
    by_book = by_book.sort_values("roi", ascending=True)

    if len(by_book) == 0:
        return empty_chart("No bookmaker data")

    colors = [COLORS["profit_green"] if r >= 0 else COLORS["loss_red"] for r in by_book["roi"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=by_book["bookmaker"], x=by_book["roi"],
        orientation="h", marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate=(
            "<b>%{y}</b><br>ROI: %{x:+.1f}%<br>"
            "Win: %{customdata[0]:.1%}<br>"
            "Avg Edge: %{customdata[1]:.1%}<br>"
            "Avg Odds: %{customdata[2]:.2f}<br>"
            "P&L: %{customdata[3]:+.1f}u<br>"
            "n=%{customdata[4]}<extra></extra>"
        ),
        customdata=by_book[["win_rate", "avg_edge", "avg_odds", "profit", "n"]].values,
        text=by_book.apply(lambda r: f"{r['roi']:+.1f}% (n={int(r['n'])})", axis=1),
        textposition="outside", textfont=dict(size=11, color=COLORS["text_secondary"]),
    ))

    fig.add_vline(x=0, line=dict(color=COLORS["text_muted"], width=1))

    fig.update_layout(
        **CHART_LAYOUT,
        title=dict(text="ROI BY BOOKMAKER (FLAT)", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="ROI %", showlegend=False,
        height=max(250, len(by_book) * 45 + 80),
    )
    fig.update_layout(margin=dict(l=100, r=100, t=40, b=40))
    return fig


def build_team_roi_chart(df):
    """Top/bottom teams by ROI (home team in games we bet draw on)."""
    graded = df.dropna(subset=["flat_profit"])
    if len(graded) == 0:
        return empty_chart("No data")

    # Combine: credit both teams when we bet their game as draw
    rows = []
    for _, r in graded.iterrows():
        rows.append({"team": r["home_team"], "profit": r["flat_profit"], "won": r["bet_won"]})
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
        title=dict(text="ROI BY TEAM (DRAW BETS)", font=dict(size=14, color=COLORS["text_secondary"])),
        xaxis_title="ROI %", showlegend=False,
        height=max(300, len(by_team) * 25 + 80),
    )
    fig.update_layout(margin=dict(l=60, r=100, t=40, b=40))
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
# CLV Analysis — 2pm EST vs 1hr Pre-Game
# =============================================================================
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
    merged = early.merge(
        close, on=["game_date", "home_team", "bookmaker"],
        suffixes=("_2pm", "_1hr"), how="inner",
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
    background: radial-gradient(ellipse at 20% 50%, rgba(255,215,0,0.04) 0%, transparent 50%),
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
    background: linear-gradient(135deg, #ffd700 0%, #ea67ff 50%, #00a3ff 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.header-subtitle {
    font-size: 13px; color: rgba(255,255,255,0.35); font-weight: 400;
    font-family: 'JetBrains Mono', monospace;
}
.header-badge {
    font-size: 11px; font-family: 'JetBrains Mono', monospace;
    background: rgba(255,215,0,0.1); border: 1px solid rgba(255,215,0,0.2);
    color: #ffd700; padding: 4px 12px; border-radius: 20px; font-weight: 500;
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
    border-color: rgba(255,215,0,0.2); background: rgba(255,255,255,0.04);
}
.kpi-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, transparent, rgba(255,215,0,0.3), transparent);
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
.VirtualizedSelectFocusedOption { background-color: rgba(255,215,0,0.2) !important; color: #fafafa !important; }
.Select-value-label { color: #fafafa !important; }
.Select-placeholder { color: rgba(255,255,255,0.35) !important; }
.Select-input input { color: #fafafa !important; }
.Select--multi .Select-value { background-color: rgba(255,215,0,0.2) !important; border-color: rgba(255,215,0,0.3) !important; }
.Select--multi .Select-value-label { color: #fafafa !important; }
.Select--multi .Select-value-icon { border-color: rgba(255,215,0,0.3) !important; color: rgba(255,255,255,0.6) !important; }
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
    title="NHL 3-Way Reg | Backtest Dashboard",
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


def serve_layout():
    """Build the full dashboard layout."""
    df = load_data()

    teams = sorted(set(
        df["home_team"].dropna().unique().tolist() +
        df["away_team"].dropna().unique().tolist()
    )) if len(df) > 0 else []

    seasons = sorted(df["season_label"].unique().tolist()) if len(df) > 0 and "season_label" in df.columns else []
    books = sorted(df["bookmaker"].unique().tolist()) if len(df) > 0 and "bookmaker" in df.columns else []
    books = [b for b in books if b and b != "unknown"]

    min_date = df["game_date"].min() if len(df) > 0 else pd.Timestamp("2024-10-07")
    max_date = df["game_date"].max() if len(df) > 0 else pd.Timestamp("2026-02-05")

    range_input_style = {**DROPDOWN_STYLE, "width": "65px", "padding": "6px 8px"}

    return html.Div(className="dashboard-container", children=[

        # === HEADER ===
        html.Div(className="header", children=[
            html.Div(className="header-left", children=[
                html.Div("NHL 3-WAY REG", className="header-title"),
                html.Div("draw-only backtest | 2024-26", className="header-subtitle"),
            ]),
            html.Div("POISSON + FIXED TIE INFLATION", className="header-badge"),
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
                html.Div("SEASON", className="filter-label"),
                dcc.Dropdown(
                    id="season-filter",
                    options=[{"label": s, "value": s} for s in seasons],
                    multi=True, placeholder="All", style=DROPDOWN_STYLE,
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("BOOK", className="filter-label"),
                dcc.Dropdown(
                    id="book-filter",
                    options=[{"label": b.title(), "value": b} for b in books],
                    multi=True, placeholder="All",
                    style={**DROPDOWN_STYLE, "minWidth": "130px"},
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
                html.Div("MARKET", className="filter-label"),
                dcc.Dropdown(
                    id="market-filter",
                    options=[
                        {"label": "Draw (OT)", "value": "draw"},
                        {"label": "Home Reg", "value": "home"},
                        {"label": "Away Reg", "value": "away"},
                    ],
                    value="draw", clearable=False,
                    style={**DROPDOWN_STYLE, "minWidth": "120px"},
                ),
            ]),
            html.Div(className="filter-group", children=[
                html.Div("STAKING", className="filter-label"),
                dcc.Dropdown(
                    id="staking-filter",
                    options=[
                        {"label": "Flat (1u)", "value": "flat"},
                        {"label": "Edge-Prop", "value": "edge_prop"},
                        {"label": "Kelly", "value": "kelly"},
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
        html.Div(className="chart-card", style={"marginBottom": "20px"}, children=[
            dcc.Graph(id="all-markets-pnl", config={"displayModeBar": False}),
        ]),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="cumulative-pnl", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="season-split-pnl", config={"displayModeBar": False}),
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
                dcc.Graph(id="closeness-roi", config={"displayModeBar": False}),
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
                dcc.Graph(id="season-comparison", config={"displayModeBar": False}),
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
                dcc.Graph(id="draw-calibration", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="edge-vs-market", config={"displayModeBar": False}),
            ]),
        ]),

        # === SUBSET ANALYSIS ===
        section_label("Subset Analysis"),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="odds-roi-chart", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="dow-roi-chart", config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-card-full", children=[
            dcc.Graph(id="book-roi-chart", config={"displayModeBar": False}),
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
            html.Div("RECENT DRAW BETS", className="table-title"),
            html.Div(id="bets-table-container"),
        ]),

        # === CLV ANALYSIS ===
        section_label("Closing Line Value — 2pm EST vs 1hr Pre-Game"),
        html.Div(id="clv-kpi-row"),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="clv-scatter", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="clv-distribution", config={"displayModeBar": False}),
            ]),
        ]),
        html.Div(className="chart-grid-2", children=[
            html.Div(className="chart-card", children=[
                dcc.Graph(id="clv-over-time", config={"displayModeBar": False}),
            ]),
            html.Div(className="chart-card", children=[
                dcc.Graph(id="clv-by-book", config={"displayModeBar": False}),
            ]),
        ]),

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
        Input("season-filter", "value"),
        Input("book-filter", "value"),
        Input("outcome-filter", "value"),
        Input("team-filter", "value"),
        Input("dow-filter", "value"),
        Input("min-pot-filter", "value"),
        Input("max-pot-filter", "value"),
    ],
)
def filter_data(start_date, end_date, seasons, books, outcomes,
                teams, dow, min_pot, max_pot):
    """Apply all filters EXCEPT edge (edge filtering handled in update_all)."""
    df = load_data()
    if len(df) == 0:
        return df.to_json(date_format="iso", orient="split")

    if start_date:
        df = df[df["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df["game_date"] <= pd.to_datetime(end_date)]

    if seasons:
        df = df[df["season_label"].isin(seasons)]

    if books:
        df = df[df["bookmaker"].isin(books)]

    if outcomes:
        df = df[df["game_outcome_type"].isin(outcomes)]

    if teams:
        df = df[(df["home_team"].isin(teams)) | (df["away_team"].isin(teams))]

    if dow:
        if "day_of_week" not in df.columns:
            df["day_of_week"] = df["game_date"].dt.day_name()
        df = df[df["day_of_week"].isin(dow)]

    if min_pot is not None:
        df = df[df["p_ot"] >= min_pot / 100.0]
    if max_pot is not None:
        df = df[df["p_ot"] <= max_pot / 100.0]

    return df.to_json(date_format="iso", orient="split")


@app.callback(
    [
        Output("kpi-row", "children"),
        Output("all-markets-pnl", "figure"),
        Output("cumulative-pnl", "figure"),
        Output("season-split-pnl", "figure"),
        Output("daily-pnl", "figure"),
        Output("edge-roi", "figure"),
        Output("pot-roi", "figure"),
        Output("closeness-roi", "figure"),
        Output("monthly-pnl-chart", "figure"),
        Output("confidence-panel", "children"),
        Output("season-comparison", "figure"),
        Output("drawdown-chart", "figure"),
        Output("outcome-type-chart", "figure"),
        Output("draw-calibration", "figure"),
        Output("edge-vs-market", "figure"),
        Output("odds-roi-chart", "figure"),
        Output("dow-roi-chart", "figure"),
        Output("book-roi-chart", "figure"),
        Output("lambda-scatter", "figure"),
        Output("model-stats-panel", "children"),
        Output("edge-distribution", "figure"),
        Output("team-roi-chart", "figure"),
        Output("bets-table-container", "children"),
        Output("clv-kpi-row", "children"),
        Output("clv-scatter", "figure"),
        Output("clv-distribution", "figure"),
        Output("clv-over-time", "figure"),
        Output("clv-by-book", "figure"),
    ],
    Input("filtered-data-store", "data"),
    Input("staking-filter", "value"),
    Input("market-filter", "value"),
    Input("min-edge-filter", "value"),
    Input("max-edge-filter", "value"),
)
def update_all(json_data, staking, market, min_edge, max_edge):
    """Update all dashboard components from filtered data."""
    df = pd.read_json(io.StringIO(json_data), orient="split") if json_data else pd.DataFrame()
    if "game_date" in df.columns:
        df["game_date"] = pd.to_datetime(df["game_date"])

    # Recompute derived columns based on selected market side
    if not market:
        market = "draw"

    if len(df) > 0:
        df["day_of_week"] = df["game_date"].dt.day_name()
        df["is_ot"] = df["game_outcome_type"].isin(["OT", "SO"]).astype(int)
        df["lam_diff"] = (df["lam_home"] - df["lam_away"]).abs()

        # Build all-markets chart BEFORE edge filtering (needs all 3 original edge columns)
        all_markets = build_all_markets_pnl(df, staking=staking or "flat",
                                            min_edge=min_edge, max_edge=max_edge)

        # Map market selection to edge/odds/win columns
        if market == "draw":
            edge_col, dec_col = "draw_edge", "dec_draw_3w"
        elif market == "home":
            edge_col, dec_col = "home_edge", "dec_home_3w"
        else:  # away
            edge_col, dec_col = "away_edge", "dec_away_3w"

        # Apply edge filter for selected market
        if min_edge and min_edge > 0:
            df = df[df[edge_col] >= min_edge / 100.0]
        if max_edge and max_edge > 0:
            df = df[df[edge_col] <= max_edge / 100.0]

        # Compute bet columns for the selected market
        if market == "draw":
            win_cond = df["actual_3way"].isin(["OT", "SO", "draw"])
        elif market == "home":
            win_cond = df["actual_3way"] == "home"
        else:
            win_cond = df["actual_3way"] == "away"

        has_edge = df[edge_col].notna() & (df[edge_col] > 0)
        df["bet_side"] = np.where(has_edge, market, None)
        df["bet_edge"] = np.where(has_edge, df[edge_col], np.nan)
        df["bet_odds_dec"] = np.where(has_edge, df[dec_col], np.nan)
        df["bet_model_prob"] = np.nan
        if market == "draw":
            df.loc[has_edge, "bet_model_prob"] = df.loc[has_edge, "p_ot"]
        elif market == "home":
            df.loc[has_edge, "bet_model_prob"] = df.loc[has_edge, "p_home_reg_win"]
        else:
            df.loc[has_edge, "bet_model_prob"] = df.loc[has_edge, "p_away_reg_win"]
        df["bet_won"] = np.where(has_edge, win_cond.astype(float), np.nan)

        # Alias selected market's edge to draw_edge for chart builders
        df["draw_edge"] = df[edge_col]

        has_bet = has_edge & df["bet_won"].notna()
        df["flat_profit"] = np.nan

        if staking == "edge_prop":
            # Edge-proportional: draw=edge*25 cap 1.5, home/away=edge*20 cap 1.0
            if market == "draw":
                stake = (df.loc[has_bet, edge_col] * 25).clip(upper=1.5)
            else:
                stake = (df.loc[has_bet, edge_col] * 20).clip(upper=1.0)
            df.loc[has_bet, "kelly_stake"] = stake.values
            b = df.loc[has_bet, dec_col] - 1.0
            df.loc[has_bet, "flat_profit"] = np.where(
                df.loc[has_bet, "bet_won"] == 1,
                stake.values * b.values,
                -stake.values,
            )
        elif staking == "kelly":
            b = df.loc[has_bet, dec_col] - 1.0
            p = df.loc[has_bet, "bet_model_prob"]
            q = 1.0 - p
            kelly_f = ((b * p - q) / b).clip(lower=0)
            quarter_kelly = kelly_f * 0.25
            df.loc[has_bet, "kelly_stake"] = quarter_kelly.values
            df.loc[has_bet, "flat_profit"] = np.where(
                df.loc[has_bet, "bet_won"] == 1,
                quarter_kelly.values * b.values,
                -quarter_kelly.values,
            )
        else:  # flat
            df.loc[has_bet, "flat_profit"] = np.where(
                df.loc[has_bet, "bet_won"] == 1,
                df.loc[has_bet, dec_col] - 1.0,
                -1.0,
            )

    # KPIs
    kpis = compute_kpis(df)

    def val_class(val_str):
        if "+" in str(val_str):
            return "kpi-positive"
        if str(val_str).startswith("-"):
            return "kpi-negative"
        return "kpi-neutral"

    roi_label = "Flat ROI" if staking != "kelly" else "Kelly ROI"
    market_label = {"draw": "Draw", "home": "Home Reg", "away": "Away Reg"}.get(market, "Draw")
    kpi_row = html.Div(className="kpi-grid", children=[
        kpi_card(roi_label, kpis.get("roi_flat", "—"), val_class(kpis.get("roi_flat", "—"))),
        kpi_card(f"{market_label} Bets", kpis.get("draw_bets", "—"), "kpi-neutral"),
        kpi_card("Win Rate", kpis.get("win_rate", "—"), "kpi-neutral"),
        kpi_card("Avg Edge", kpis.get("avg_edge", "—"), val_class(kpis.get("avg_edge", "—"))),
        kpi_card("Sharpe", kpis.get("sharpe", "—"), val_class(kpis.get("sharpe", "—"))),
        kpi_card("Max DD", kpis.get("max_drawdown", "—"),
                 "kpi-negative" if kpis.get("max_drawdown", "—") != "—" else "kpi-neutral"),
        kpi_card("t-stat", kpis.get("t_stat", "—"),
                 "kpi-positive" if kpis.get("significant") == "Yes" else "kpi-neutral"),
        kpi_card("p-value", kpis.get("p_value", "—"),
                 "kpi-positive" if kpis.get("significant") == "Yes" else "kpi-negative"),
    ])

    # Charts (all_markets built earlier, before draw_edge aliasing)
    if len(df) == 0:
        all_markets = empty_chart("No data")
    cum_pnl = build_cumulative_pnl(df, staking=staking or "flat")
    season_split = build_season_split_pnl(df)
    daily_pnl = build_daily_pnl_bars(df)
    edge_roi = build_edge_roi_chart(df)
    pot_roi = build_pot_roi_chart(df)
    close_roi = build_closeness_roi_chart(df)
    monthly_pnl = build_monthly_pnl_chart(df)
    season_comp = build_season_comparison(df)
    dd_chart = build_drawdown_chart(df)
    oc_chart = build_outcome_type_chart(df)
    draw_cal = build_draw_calibration(df)
    edge_market = build_edge_vs_market(df)
    odds_roi = build_odds_roi_chart(df)
    dow_roi = build_dow_roi_chart(df)
    book_roi = build_book_roi_chart(df)
    lam_scatter = build_lambda_scatter(df)
    edge_dist = build_edge_distribution(df)
    team_roi = build_team_roi_chart(df)

    # Update chart titles with staking mode and market
    staking_label = "FLAT" if staking != "kelly" else "KELLY"
    edge_label = {"draw": "DRAW EDGE", "home": "HOME REG EDGE", "away": "AWAY REG EDGE"}.get(market, "DRAW EDGE")
    odds_label = {"draw": "DRAW ODDS", "home": "HOME REG ODDS", "away": "AWAY REG ODDS"}.get(market, "DRAW ODDS")
    title_font = dict(size=14, color=COLORS["text_secondary"])
    daily_pnl.update_layout(title=dict(text=f"DAILY P&L ({staking_label})", font=title_font))
    edge_roi.update_layout(title=dict(text=f"ROI BY {edge_label} BUCKET ({staking_label})", font=title_font))
    pot_roi.update_layout(title=dict(text=f"ROI BY P(OT) BUCKET ({staking_label})", font=title_font))
    oc_chart.update_layout(title=dict(text=f"ROI BY OUTCOME TYPE ({staking_label})", font=title_font))
    season_comp.update_layout(title=dict(text=f"ROI BY SEASON ({staking_label})", font=title_font))
    odds_roi.update_layout(title=dict(text=f"ROI BY {odds_label} ({staking_label})", font=title_font))
    book_roi.update_layout(title=dict(text=f"ROI BY BOOKMAKER ({staking_label})", font=title_font))
    close_roi.update_layout(title=dict(text=f"ROI BY MATCHUP CLOSENESS ({staking_label})", font=title_font))
    edge_dist.update_layout(title=dict(text=f"{edge_label} DISTRIBUTION", font=title_font))

    # Confidence panel
    sig_color = COLORS["accent_green"] if kpis.get("significant") == "Yes" else COLORS["accent_red"]
    confidence_panel = html.Div(style={"padding": "10px 0"}, children=[
        html.Div("SAMPLE CONFIDENCE", style={
            "fontSize": "14px", "fontWeight": "600", "color": COLORS["text_secondary"],
            "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "20px",
        }),
        _stat_row("Total Bets", kpis.get("total_bets", "—")),
        _stat_row(f"Avg {market_label} Odds", kpis.get("avg_draw_odds", "—")),
        _stat_row("Flat P&L" if staking != "kelly" else "Kelly P&L", kpis.get("flat_pnl", "—"),
                  COLORS["accent_green"] if kpis.get("flat_pnl", "—").startswith("+") else COLORS["accent_red"]),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("t-statistic", kpis.get("t_stat", "—"), sig_color),
        _stat_row("p-value", kpis.get("p_value", "—"), sig_color),
        _stat_row("Significant (5%)", kpis.get("significant", "—"), sig_color),
        _stat_row("95% CI (ROI)", kpis.get("ci_95", "—")),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("Games to Signif.", kpis.get("n_remaining", "—"), COLORS["accent_gold"]),
        _stat_row("Actual OT Rate", kpis.get("actual_draw_rate", "—")),
        _stat_row("Model P(Draw)", kpis.get("model_draw_mean", "—")),
        _stat_row("Market P(Draw)", kpis.get("market_draw_mean", "—")),
        html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
        _stat_row("Max Win Streak", kpis.get("max_win_streak", "—"), COLORS["accent_green"]),
        _stat_row("Max Loss Streak", kpis.get("max_loss_streak", "—"), COLORS["accent_red"]),
    ])

    # Model stats panel
    if len(df) > 0:
        with_odds = df.dropna(subset=["fair_draw_3w"])
        n_total = len(df)
        n_odds = len(with_odds)
        ot_rate = df["is_ot"].mean() if "is_ot" in df.columns else 0

        model_panel = html.Div(style={"padding": "10px 0"}, children=[
            html.Div("MODEL STATS", style={
                "fontSize": "14px", "fontWeight": "600", "color": COLORS["text_secondary"],
                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "20px",
            }),
            _stat_row("Total Games", f"{n_total}"),
            _stat_row("With 3W Odds", f"{n_odds}"),
            _stat_row("OT Rate", f"{ot_rate:.3f}"),
            _stat_row("Tie Inflation", f"{df['tie_inflation'].mean():.3f}"),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
            _stat_row("Avg λ Home", f"{df['lam_home'].mean():.3f}", COLORS["accent_gold"]),
            _stat_row("Avg λ Away", f"{df['lam_away'].mean():.3f}", COLORS["accent_gold"]),
            _stat_row("Avg |Δλ|", f"{df['lam_diff'].mean():.3f}", COLORS["accent_cyan"]),
            html.Hr(style={"border": "none", "borderTop": f"1px solid {COLORS['border']}", "margin": "16px 0"}),
            _stat_row("Anchor Home", f"{df['anchor_home'].mean():.4f}"),
            _stat_row("Anchor Away", f"{df['anchor_away'].mean():.4f}"),
            _stat_row("Poisson ρ", f"{df['poisson_rho'].mean():.4f}"),
            _stat_row("Best Day", kpis.get("best_day", "—")),
            _stat_row("Worst Day", kpis.get("worst_day", "—")),
        ])
    else:
        model_panel = html.Div("No data", style={"color": COLORS["text_muted"], "padding": "40px", "textAlign": "center"})

    # Bets table
    bets_table = _build_bets_table(df, staking=staking or "flat")

    # CLV analysis — filter to match the same games in the filtered df
    clv = load_clv_data()
    if len(clv) > 0 and len(df) > 0:
        # Filter CLV data to match the filtered backtest games
        bt_games = df[["game_date", "home_team"]].drop_duplicates()
        bt_games["game_date"] = bt_games["game_date"].dt.strftime("%Y-%m-%d")
        clv["game_date_str"] = clv["game_date"].dt.strftime("%Y-%m-%d")
        clv = clv.merge(
            bt_games.rename(columns={"game_date": "game_date_str"}),
            on=["game_date_str", "home_team"], how="inner",
        )
        clv = clv.drop(columns=["game_date_str"])

    if len(clv) > 0:
        n_games = clv[["game_date", "home_team"]].drop_duplicates().shape[0]
        n_rows = len(clv)
        mean_clv = clv["draw_clv_pct"].mean()
        median_clv = clv["draw_clv_pct"].median()
        pct_positive = (clv["draw_clv_pct"] > 0).mean() * 100
        mean_ip_clv = clv["draw_ip_clv"].mean()

        def vc(v):
            return "kpi-positive" if v > 0 else "kpi-negative"

        clv_kpi_row = html.Div(className="kpi-grid",
            style={"gridTemplateColumns": "repeat(6, 1fr)", "marginBottom": "20px"}, children=[
            kpi_card("Games Matched", f"{n_games:,}", "kpi-neutral"),
            kpi_card("Odds Rows", f"{n_rows:,}", "kpi-neutral"),
            kpi_card("Mean Draw CLV", f"{mean_clv:+.2f}%", vc(mean_clv)),
            kpi_card("Median Draw CLV", f"{median_clv:+.2f}%", vc(median_clv)),
            kpi_card("% Better at 2pm", f"{pct_positive:.1f}%",
                     "kpi-positive" if pct_positive > 50 else "kpi-negative"),
            kpi_card("Avg IP Shift", f"{mean_ip_clv:+.2f}pp", vc(mean_ip_clv)),
        ])
        clv_scatter = build_clv_scatter(clv)
        clv_dist = build_clv_distribution(clv)
        clv_time = build_clv_over_time(clv)
        clv_book = build_clv_by_book(clv)
    else:
        clv_kpi_row = html.Div(
            "No CLV data available. Need both three_way_odds_2pm_est.csv and three_way_odds.csv.",
            style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"},
        )
        clv_scatter = empty_chart("No CLV data")
        clv_dist = empty_chart("No CLV data")
        clv_time = empty_chart("No CLV data")
        clv_book = empty_chart("No CLV data")

    return (
        kpi_row, all_markets, cum_pnl, season_split, daily_pnl, edge_roi,
        pot_roi, close_roi, monthly_pnl, confidence_panel,
        season_comp, dd_chart, oc_chart,
        draw_cal, edge_market,
        odds_roi, dow_roi, book_roi,
        lam_scatter, model_panel, edge_dist, team_roi,
        bets_table,
        clv_kpi_row, clv_scatter, clv_dist, clv_time, clv_book,
    )


def _build_bets_table(df, staking="flat"):
    """Build the bet log DataTable for 3-way draw bets."""
    if len(df) == 0:
        return html.Div("No data", style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})

    # Only show rows where a bet was placed
    display = df[df["bet_side"].notna()].copy() if df["bet_side"].notna().any() else df.copy()

    display["game_date_str"] = display["game_date"].dt.strftime("%b %d")
    display["draw_edge_pct"] = (display["draw_edge"] * 100).round(1)
    display["p_ot_pct"] = (display["p_ot"] * 100).round(1)
    display["fair_draw_pct"] = (display["fair_draw_3w"] * 100).round(1)
    display["lam_home"] = display["lam_home"].round(2)
    display["lam_away"] = display["lam_away"].round(2)
    display["bet_odds_dec"] = display["bet_odds_dec"].round(2)
    display["flat_profit"] = display["flat_profit"].round(2)
    if staking == "kelly":
        display["stake"] = display["kelly_stake"].round(3)
    else:
        display["stake"] = np.where(display["bet_side"].notna(), 1.0, np.nan)

    display_cols = [
        {"name": "Date", "id": "game_date_str"},
        {"name": "Season", "id": "season_label"},
        {"name": "Home", "id": "home_team"},
        {"name": "Away", "id": "away_team"},
        {"name": "λH", "id": "lam_home"},
        {"name": "λA", "id": "lam_away"},
        {"name": "P(OT)", "id": "p_ot_pct"},
        {"name": "Mkt%", "id": "fair_draw_pct"},
        {"name": "Edge", "id": "draw_edge_pct"},
        {"name": "Book", "id": "bookmaker"},
        {"name": "Odds", "id": "bet_odds_dec"},
        {"name": "Stake", "id": "stake"},
        {"name": "Result", "id": "actual_3way"},
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
            {"if": {"column_id": "season_label"}, "textAlign": "center", "fontSize": "11px"},
            {"if": {"column_id": "bookmaker"}, "textAlign": "center", "fontSize": "10px"},
            {"if": {"column_id": "actual_3way"}, "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{flat_profit} > 0", "column_id": "flat_profit"}, "color": COLORS["profit_green"]},
            {"if": {"filter_query": "{flat_profit} < 0", "column_id": "flat_profit"}, "color": COLORS["loss_red"]},
            {"if": {"filter_query": '{actual_3way} = "draw"', "column_id": "actual_3way"}, "color": COLORS["accent_gold"]},
            {"if": {"filter_query": '{actual_3way} = "home"', "column_id": "actual_3way"}, "color": COLORS["accent_green"]},
            {"if": {"filter_query": '{actual_3way} = "away"', "column_id": "actual_3way"}, "color": COLORS["accent_cyan"]},
            {"if": {"state": "active"}, "backgroundColor": "rgba(255,215,0,0.05)", "border": "none"},
        ],
        style_as_list_view=True,
    )


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("NHL 3-WAY REG — Backtest Dashboard")
    print("=" * 60)

    if os.path.exists(BT_FILE):
        df = pd.read_csv(BT_FILE)
        print(f"Loaded {len(df):,} games from {BT_FILE}")
        n_bets = df["bet_side"].notna().sum()
        print(f"Draw bets: {n_bets}")
    else:
        print(f"WARNING: {BT_FILE} not found. Dashboard will show empty state.")
        print("Run walk_forward_3way.py first to generate data.")

    print(f"\nStarting dashboard at http://127.0.0.1:{PORT}")
    app.run(debug=False, port=PORT, host="127.0.0.1")
