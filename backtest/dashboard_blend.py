# -*- coding: utf-8 -*-
"""
NHL 3-Way Blend Portfolio Dashboard

Dedicated dashboard for the Draw + Home Reg blended portfolio strategy.
- Draw 2.5%+ edge: stake = edge*25, cap 1.5u
- Home 2.5-5% edge: stake = edge*20, cap 1.0u
Edge-proportional staking is the default. Shows both streams individually
and combined, with portfolio-specific analytics.

Run: python dashboard_blend.py
Open: http://127.0.0.1:8057

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
PORT = 8057

# Blend strategy parameters
DRAW_MIN_EDGE = 0.025
HOME_MIN_EDGE = 0.025
HOME_MAX_EDGE = 0.05
DRAW_STAKE_MULT = 25
DRAW_STAKE_CAP = 1.5
HOME_STAKE_MULT = 20
HOME_STAKE_CAP = 1.0

# =============================================================================
# Color Palette
# =============================================================================
COLORS = {
    "bg_primary": "#020219",
    "bg_secondary": "#0a0a2e",
    "bg_card": "rgba(255, 255, 255, 0.025)",
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
    "profit_green": "#00ff88",
    "loss_red": "#ff4757",
    "chart_grid": "rgba(255, 255, 255, 0.04)",
    "draw_color": "#ffd700",
    "home_color": "#00ff88",
    "blend_color": "#ea67ff",
}

CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, system-ui, sans-serif", color=COLORS["text_primary"], size=12),
    margin=dict(l=50, r=20, t=40, b=40),
    xaxis=dict(gridcolor=COLORS["chart_grid"], zerolinecolor=COLORS["chart_grid"],
               tickfont=dict(size=11, color=COLORS["text_secondary"])),
    yaxis=dict(gridcolor=COLORS["chart_grid"], zerolinecolor=COLORS["chart_grid"],
               tickfont=dict(size=11, color=COLORS["text_secondary"])),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=COLORS["text_secondary"])),
    hoverlabel=dict(bgcolor=COLORS["bg_secondary"], bordercolor=COLORS["border"],
                    font=dict(family="JetBrains Mono, monospace", size=12, color=COLORS["text_primary"])),
)


# =============================================================================
# Data Loading
# =============================================================================
def load_data():
    """Load backtest and compute both bet streams."""
    if not os.path.exists(BT_FILE):
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    raw = pd.read_csv(BT_FILE)
    raw["game_date"] = pd.to_datetime(raw["game_date"])
    if "bookmaker" not in raw.columns:
        raw["bookmaker"] = "unknown"

    # Draw bets (permissive — edge filtering happens in filter_bets callback)
    d_mask = raw["draw_edge"].notna() & (raw["draw_edge"] > 0)
    d = raw[d_mask].copy()
    d["side"] = "draw"
    d["edge"] = d["draw_edge"]
    d["odds"] = d["dec_draw_3w"]
    d["model_prob"] = d["p_ot"]
    d["won"] = d["actual_3way"].isin(["OT", "SO", "draw"]).astype(int)
    d["stake"] = (d["draw_edge"] * DRAW_STAKE_MULT).clip(upper=DRAW_STAKE_CAP)
    d["profit"] = np.where(d["won"] == 1, d["stake"] * (d["odds"] - 1.0), -d["stake"])
    d["flat_profit"] = np.where(d["won"] == 1, d["odds"] - 1.0, -1.0)
    d["stake_tw"] = d["stake"] / (d["odds"] - 1.0)
    d["profit_tw"] = np.where(d["won"] == 1, d["stake"], -d["stake_tw"])
    d["flat_stake_tw"] = 1.0 / (d["odds"] - 1.0)
    d["flat_profit_tw"] = np.where(d["won"] == 1, 1.0, -d["flat_stake_tw"])

    # Home bets (permissive — edge filtering happens in filter_bets callback)
    h_mask = raw["home_edge"].notna() & (raw["home_edge"] > 0)
    h = raw[h_mask].copy()
    h["side"] = "home"
    h["edge"] = h["home_edge"]
    h["odds"] = h["dec_home_3w"]
    h["model_prob"] = h["p_home_reg_win"]
    h["won"] = (h["actual_3way"] == "home").astype(int)
    h["stake"] = (h["home_edge"] * HOME_STAKE_MULT).clip(upper=HOME_STAKE_CAP)
    h["profit"] = np.where(h["won"] == 1, h["stake"] * (h["odds"] - 1.0), -h["stake"])
    h["flat_profit"] = np.where(h["won"] == 1, h["odds"] - 1.0, -1.0)
    h["stake_tw"] = h["stake"] / (h["odds"] - 1.0)
    h["profit_tw"] = np.where(h["won"] == 1, h["stake"], -h["stake_tw"])
    h["flat_stake_tw"] = 1.0 / (h["odds"] - 1.0)
    h["flat_profit_tw"] = np.where(h["won"] == 1, 1.0, -h["flat_stake_tw"])

    # Combined (one row per bet)
    keep_cols = ["game_id", "game_date", "season_label", "home_team", "away_team",
                 "lam_home", "lam_away", "actual_3way", "game_outcome_type", "bookmaker",
                 "side", "edge", "odds", "model_prob", "won", "stake", "profit", "flat_profit",
                 "stake_tw", "profit_tw", "flat_stake_tw", "flat_profit_tw"]
    bets = pd.concat([d[keep_cols], h[keep_cols]], ignore_index=True).sort_values("game_date")

    return raw, bets, bets  # raw for filters, bets for everything else


def empty_chart(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper",
                       font=dict(size=16, color=COLORS["text_muted"]), showarrow=False)
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      xaxis=dict(visible=False), yaxis=dict(visible=False), height=300)
    return fig


# =============================================================================
# Chart Builders
# =============================================================================
def build_equity_curves(bets, staking="edge_prop"):
    """Three overlaid equity curves: Draw, Home, Combined."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"

    fig = go.Figure()

    for side, label, color, rgb in [
        ("draw", "Draw (OT)", COLORS["draw_color"], "255,215,0"),
        ("home", "Home Reg", COLORS["home_color"], "0,255,136"),
    ]:
        s = bets[bets["side"] == side]
        if len(s) == 0:
            continue
        daily = s.groupby("game_date")[pnl_col].sum().reset_index()
        daily["cum"] = daily[pnl_col].cumsum()
        n = len(s)
        roi = s[pnl_col].sum() / s["stake"].sum() * 100
        fig.add_trace(go.Scatter(
            x=daily["game_date"], y=daily["cum"],
            fill="tozeroy", fillcolor=f"rgba({rgb}, 0.05)",
            line=dict(color=color, width=2),
            name=f"{label} ({n}, {roi:+.1f}%)",
            hovertemplate=f"<b>%{{x|%b %d}}</b><br>{label}: %{{y:+.1f}}u<extra></extra>",
        ))

    # Combined
    daily_all = bets.groupby("game_date")[pnl_col].sum().reset_index()
    daily_all["cum"] = daily_all[pnl_col].cumsum()
    n_all = len(bets)
    roi_all = bets[pnl_col].sum() / bets["stake"].sum() * 100
    fig.add_trace(go.Scatter(
        x=daily_all["game_date"], y=daily_all["cum"],
        line=dict(color=COLORS["blend_color"], width=3, dash="dash"),
        name=f"Blend ({n_all}, {roi_all:+.1f}%)",
        hovertemplate="<b>%{x|%b %d}</b><br>Blend: %{y:+.1f}u<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"PORTFOLIO EQUITY CURVES ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="Units", height=420)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                                  bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=COLORS["text_primary"])))
    return fig


def build_monthly_contribution(bets, staking="edge_prop"):
    """Monthly P&L stacked bars showing draw vs home contribution."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    bets = bets.copy()
    bets["month"] = bets["game_date"].dt.to_period("M").astype(str)

    monthly = bets.groupby(["month", "side"])[pnl_col].sum().unstack(fill_value=0)
    if "draw" not in monthly.columns:
        monthly["draw"] = 0
    if "home" not in monthly.columns:
        monthly["home"] = 0

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["draw"],
        name="Draw", marker_color=COLORS["draw_color"], opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Draw: %{y:+.1f}u<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=monthly.index, y=monthly["home"],
        name="Home", marker_color=COLORS["home_color"], opacity=0.85,
        hovertemplate="<b>%{x}</b><br>Home: %{y:+.1f}u<extra></extra>",
    ))

    # Combo line
    combo = monthly["draw"] + monthly["home"]
    fig.add_trace(go.Scatter(
        x=monthly.index, y=combo,
        mode="markers+lines",
        line=dict(color=COLORS["blend_color"], width=2.5),
        marker=dict(size=7, color=COLORS["blend_color"]),
        name="Combo",
        hovertemplate="<b>%{x}</b><br>Combo: %{y:+.1f}u<extra></extra>",
    ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"
    fig.update_layout(**CHART_LAYOUT, barmode="relative",
                      title=dict(text=f"MONTHLY P&L BY SIDE ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="Units", height=360)
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig


def build_drawdown(bets, staking="edge_prop"):
    """Portfolio drawdown chart."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    daily = bets.groupby("game_date")[pnl_col].sum().reset_index()
    daily["cum"] = daily[pnl_col].cumsum()
    daily["peak"] = daily["cum"].cummax()
    daily["dd"] = daily["cum"] - daily["peak"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["dd"],
        fill="tozeroy", fillcolor="rgba(255,71,87,0.15)",
        line=dict(color=COLORS["accent_red"], width=2),
        hovertemplate="<b>%{x|%b %d}</b><br>DD: %{y:.1f}u<extra></extra>",
    ))

    max_dd = daily["dd"].min()
    max_dd_date = daily.loc[daily["dd"].idxmin(), "game_date"]
    fig.add_annotation(x=max_dd_date, y=max_dd, text=f"Max: {max_dd:.1f}u",
                       showarrow=True, arrowhead=2, font=dict(size=11, color=COLORS["accent_red"]),
                       arrowcolor=COLORS["accent_red"], ax=30, ay=-30)

    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text="PORTFOLIO DRAWDOWN", font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="Units from peak", height=300)
    return fig


def build_edge_roi(bets, staking="edge_prop"):
    """ROI by edge bucket, split by side."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    bets = bets.copy()
    bins = [0.0, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.10]
    labels = [f"{bins[i]*100:.1f}-{bins[i+1]*100:.1f}%" for i in range(len(bins)-1)]
    bets["edge_bucket"] = pd.cut(bets["edge"], bins=bins, labels=labels)

    fig = go.Figure()
    for side, color in [("draw", COLORS["draw_color"]), ("home", COLORS["home_color"])]:
        s = bets[bets["side"] == side]
        by_edge = s.groupby("edge_bucket", observed=True).agg(
            n=(pnl_col, "count"),
            roi=(pnl_col, lambda x: x.sum() / s.loc[x.index, "stake"].sum() * 100),
        ).reset_index()
        by_edge = by_edge[by_edge["n"] >= 10]

        fig.add_trace(go.Bar(
            x=by_edge["edge_bucket"], y=by_edge["roi"],
            name=side.title(), marker_color=color, opacity=0.8,
            text=by_edge["n"].apply(lambda n: f"n={n}"),
            textposition="outside", textfont=dict(size=9, color=COLORS["text_muted"]),
            hovertemplate=f"{side.title()}<br>Edge: %{{x}}<br>ROI: %{{y:+.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"
    fig.update_layout(**CHART_LAYOUT, barmode="group",
                      title=dict(text=f"ROI BY EDGE BUCKET ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="ROI %", height=340)
    return fig


def build_season_comparison(bets, staking="edge_prop"):
    """Side-by-side season comparison."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    fig = go.Figure()

    for side, color in [("draw", COLORS["draw_color"]), ("home", COLORS["home_color"])]:
        s = bets[bets["side"] == side]
        by_season = s.groupby("season_label").agg(
            n=(pnl_col, "count"),
            pnl=(pnl_col, "sum"),
            staked=("stake", "sum"),
        ).reset_index()
        by_season["roi"] = by_season["pnl"] / by_season["staked"] * 100

        fig.add_trace(go.Bar(
            x=by_season["season_label"], y=by_season["roi"],
            name=side.title(), marker_color=color, opacity=0.85,
            text=by_season.apply(lambda r: f"{r['pnl']:+.0f}u\nn={int(r['n'])}", axis=1),
            textposition="outside", textfont=dict(size=10, color=COLORS["text_primary"]),
            hovertemplate=f"{side.title()}<br>%{{x}}<br>ROI: %{{y:+.1f}}%<extra></extra>",
        ))

    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"
    fig.update_layout(**CHART_LAYOUT, barmode="group",
                      title=dict(text=f"ROI BY SEASON & SIDE ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="ROI %", height=340)
    return fig


def build_daily_bars(bets, staking="edge_prop"):
    """Daily P&L bars colored by net result."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    daily = bets.groupby("game_date").agg(
        total=(pnl_col, "sum"), n=(pnl_col, "count"),
    ).reset_index()
    colors = [COLORS["profit_green"] if p >= 0 else COLORS["loss_red"] for p in daily["total"]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=daily["game_date"], y=daily["total"],
        marker_color=colors, marker_line_width=0, opacity=0.85,
        hovertemplate="<b>%{x|%b %d}</b><br>P&L: %{y:+.2f}u<br>Bets: %{customdata}<extra></extra>",
        customdata=daily["n"],
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1))
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"DAILY P&L ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="Units", showlegend=False, height=300)
    return fig


def build_overlap_roi(bets, staking="edge_prop"):
    """ROI by overlap category: draw-only, both, home-only.
    For 'both' games, at most one side wins per game — naturally handled."""
    if len(bets) == 0:
        return empty_chart("No bets")

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    d_games = set(bets[bets["side"] == "draw"]["game_id"])
    h_games = set(bets[bets["side"] == "home"]["game_id"])
    both = d_games & h_games
    d_only = d_games - h_games
    h_only = h_games - d_games

    results = []
    for label, game_set, color in [
        ("Draw Only", d_only, COLORS["draw_color"]),
        ("Both Sides", both, COLORS["blend_color"]),
        ("Home Only", h_only, COLORS["home_color"]),
    ]:
        sub = bets[bets["game_id"].isin(game_set)]
        if len(sub) == 0:
            results.append((label, 0, 0, 0, 0, color))
            continue
        pnl = sub[pnl_col].sum()
        staked = sub["stake"].sum()
        roi = pnl / staked * 100 if staked > 0 else 0
        results.append((label, roi, pnl, len(game_set), len(sub), color))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[r[0] for r in results],
        y=[r[1] for r in results],
        marker_color=[r[5] for r in results],
        opacity=0.85,
        text=[f"{r[3]} games / {r[4]} bets\n{r[2]:+.1f}u" for r in results],
        textposition="outside",
        textfont=dict(size=10, color=COLORS["text_primary"]),
        hovertemplate="%{x}<br>ROI: %{y:+.1f}%<br>%{text}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=COLORS["text_muted"], width=1, dash="dot"))
    staking_label = "EDGE-PROP" if staking == "edge_prop" else "FLAT"
    fig.update_layout(**CHART_LAYOUT,
                      title=dict(text=f"ROI BY OVERLAP CATEGORY ({staking_label})",
                                 font=dict(size=14, color=COLORS["text_secondary"])),
                      yaxis_title="ROI %", showlegend=False, height=300)
    return fig


def build_bets_table(bets, staking="edge_prop"):
    """Bet log showing both draw and home bets."""
    if len(bets) == 0:
        return html.Div("No bets", style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"
    display = bets.copy()
    display["date_str"] = display["game_date"].dt.strftime("%b %d")
    display["edge_pct"] = (display["edge"] * 100).round(2)
    display["odds"] = display["odds"].round(2)
    display["stake"] = display["stake"].round(3)
    display["pnl"] = display[pnl_col].round(2)
    display["result"] = display["actual_3way"]

    cols = [
        {"name": "Date", "id": "date_str"},
        {"name": "Season", "id": "season_label"},
        {"name": "Side", "id": "side"},
        {"name": "Home", "id": "home_team"},
        {"name": "Away", "id": "away_team"},
        {"name": "Edge %", "id": "edge_pct"},
        {"name": "Book", "id": "bookmaker"},
        {"name": "Odds", "id": "odds"},
        {"name": "Stake", "id": "stake"},
        {"name": "Result", "id": "result"},
        {"name": "P&L", "id": "pnl"},
    ]

    return dash_table.DataTable(
        data=display.sort_values("game_date", ascending=False).head(300).to_dict("records"),
        columns=cols, sort_action="native", sort_mode="single", page_size=25,
        filter_action="native",
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "rgba(255,255,255,0.04)", "color": COLORS["text_secondary"],
            "fontWeight": "600", "fontSize": "10px", "textTransform": "uppercase",
            "letterSpacing": "0.5px", "border": "none",
            "borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 10px",
        },
        style_cell={
            "backgroundColor": "transparent", "color": COLORS["text_primary"],
            "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.04)",
            "padding": "8px 10px", "fontSize": "12px",
            "fontFamily": "'JetBrains Mono', monospace",
            "textAlign": "right", "minWidth": "55px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "home_team"}, "textAlign": "center", "fontFamily": "Inter, sans-serif"},
            {"if": {"column_id": "away_team"}, "textAlign": "center", "fontFamily": "Inter, sans-serif"},
            {"if": {"column_id": "date_str"}, "textAlign": "left", "minWidth": "70px"},
            {"if": {"column_id": "side"}, "textAlign": "center", "fontWeight": "600"},
            {"if": {"column_id": "bookmaker"}, "textAlign": "center", "fontSize": "10px"},
            {"if": {"column_id": "result"}, "textAlign": "center"},
        ],
        style_data_conditional=[
            {"if": {"filter_query": "{pnl} > 0", "column_id": "pnl"}, "color": COLORS["profit_green"]},
            {"if": {"filter_query": "{pnl} < 0", "column_id": "pnl"}, "color": COLORS["loss_red"]},
            {"if": {"filter_query": '{side} = "draw"', "column_id": "side"}, "color": COLORS["draw_color"]},
            {"if": {"filter_query": '{side} = "home"', "column_id": "side"}, "color": COLORS["home_color"]},
            {"if": {"state": "active"}, "backgroundColor": "rgba(255,215,0,0.05)", "border": "none"},
        ],
        style_as_list_view=True,
    )


def build_game_grid(raw, bets, staking="edge_prop"):
    """Game-by-game grid: all games on betting days, bets highlighted.
    Uses the bet's actual edge (correct bookmaker) for bet rows."""
    if len(bets) == 0 or len(raw) == 0:
        return html.Div("No games to display",
                        style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"

    # All games on days we placed bets
    bet_dates = bets["game_date"].dt.normalize().unique()
    games = raw[raw["game_date"].dt.normalize().isin(bet_dates)].copy()
    games = games.drop_duplicates("game_id").sort_values("game_date", ascending=False)

    if len(games) == 0:
        return html.Div("No games to display",
                        style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})

    # Bet info — include edge from the actual bet (correct bookmaker)
    d_info = (bets[bets["side"] == "draw"]
              .drop_duplicates("game_id")[["game_id", "stake", "edge", "odds", pnl_col]]
              .rename(columns={"stake": "d_stake", "edge": "d_bet_edge",
                                "odds": "d_odds", pnl_col: "d_pnl"}))
    h_info = (bets[bets["side"] == "home"]
              .drop_duplicates("game_id")[["game_id", "stake", "edge", "odds", pnl_col]]
              .rename(columns={"stake": "h_stake", "edge": "h_bet_edge",
                                "odds": "h_odds", pnl_col: "h_pnl"}))

    g = games.merge(d_info, on="game_id", how="left").merge(h_info, on="game_id", how="left")

    # Use bet's edge (correct bookmaker) when we bet, raw edge otherwise
    g["d_edge_show"] = np.where(g["d_bet_edge"].notna(), g["d_bet_edge"], g["draw_edge"])
    g["h_edge_show"] = np.where(g["h_bet_edge"].notna(), g["h_bet_edge"], g["home_edge"])

    # Game-level P&L
    g["game_pnl"] = g[["d_pnl", "h_pnl"]].sum(axis=1, min_count=1)

    display = pd.DataFrame({
        "date": g["game_date"].dt.strftime("%a %b %d").values,
        "away": g["away_team"].values,
        "home": g["home_team"].values,
        "mdl_away": (g["p_away_reg_win"] * 100).round(1).values,
        "mdl_home": (g["p_home_reg_win"] * 100).round(1).values,
        "mdl_ot": (g["p_ot"] * 100).round(1).values,
        "h_edge": (g["h_edge_show"] * 100).round(1).values,
        "h_odds": g["h_odds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "").values,
        "h_bet": g["h_stake"].apply(lambda x: f"{x:.2f}u" if pd.notna(x) else "").values,
        "d_edge": (g["d_edge_show"] * 100).round(1).values,
        "d_odds": g["d_odds"].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "").values,
        "d_bet": g["d_stake"].apply(lambda x: f"{x:.2f}u" if pd.notna(x) else "").values,
        "result": g["actual_3way"].values,
        "pnl": g["game_pnl"].apply(lambda x: f"{x:+.2f}" if pd.notna(x) else "").values,
        "_hd": g["d_stake"].notna().astype(int).values,
        "_hh": g["h_stake"].notna().astype(int).values,
        "_has_bet": (g["d_stake"].notna() | g["h_stake"].notna()).astype(int).values,
    })

    display = display.head(500)

    cols = [
        {"name": "Date", "id": "date"},
        {"name": "Away", "id": "away"},
        {"name": "Home", "id": "home"},
        # ── Model probabilities ──
        {"name": "Mdl A%", "id": "mdl_away"},
        {"name": "Mdl H%", "id": "mdl_home"},
        {"name": "Mdl OT%", "id": "mdl_ot"},
        # ── Home reg bet ──
        {"name": "H Edge%", "id": "h_edge"},
        {"name": "H Odds", "id": "h_odds"},
        {"name": "H Bet", "id": "h_bet"},
        # ── Draw (OT) bet ──
        {"name": "D Edge%", "id": "d_edge"},
        {"name": "D Odds", "id": "d_odds"},
        {"name": "D Bet", "id": "d_bet"},
        # ── Outcome ──
        {"name": "Result", "id": "result"},
        {"name": "P&L", "id": "pnl"},
        {"name": "", "id": "_hd"},
        {"name": "", "id": "_hh"},
        {"name": "", "id": "_has_bet"},
    ]

    return dash_table.DataTable(
        data=display.to_dict("records"),
        columns=cols,
        hidden_columns=["_hd", "_hh", "_has_bet"],
        sort_action="native",
        filter_action="native",
        page_size=30,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "rgba(255,255,255,0.04)", "color": COLORS["text_secondary"],
            "fontWeight": "600", "fontSize": "10px", "textTransform": "uppercase",
            "letterSpacing": "0.5px", "border": "none",
            "borderBottom": f"1px solid {COLORS['border']}", "padding": "8px 10px",
        },
        style_cell={
            "backgroundColor": "transparent", "color": COLORS["text_primary"],
            "border": "none", "borderBottom": "1px solid rgba(255,255,255,0.04)",
            "padding": "7px 10px", "fontSize": "12px",
            "fontFamily": "'JetBrains Mono', monospace",
            "textAlign": "center", "minWidth": "48px",
        },
        style_cell_conditional=[
            {"if": {"column_id": "date"}, "textAlign": "left", "minWidth": "100px",
             "fontFamily": "Inter, sans-serif", "fontSize": "11px",
             "color": COLORS["text_secondary"]},
            {"if": {"column_id": "away"}, "fontFamily": "Inter, sans-serif",
             "fontWeight": "500", "minWidth": "50px"},
            {"if": {"column_id": "home"}, "fontFamily": "Inter, sans-serif",
             "fontWeight": "500", "minWidth": "50px"},
            # Model prob columns — slightly muted
            {"if": {"column_id": "mdl_away"}, "color": COLORS["text_secondary"],
             "fontSize": "11px"},
            {"if": {"column_id": "mdl_home"}, "color": COLORS["text_secondary"],
             "fontSize": "11px"},
            {"if": {"column_id": "mdl_ot"}, "color": COLORS["text_secondary"],
             "fontSize": "11px"},
            # Bet columns
            {"if": {"column_id": "h_bet"}, "minWidth": "58px"},
            {"if": {"column_id": "d_bet"}, "minWidth": "58px"},
            {"if": {"column_id": "h_odds"}, "fontSize": "11px",
             "color": COLORS["text_secondary"]},
            {"if": {"column_id": "d_odds"}, "fontSize": "11px",
             "color": COLORS["text_secondary"]},
            {"if": {"column_id": "pnl"}, "minWidth": "58px", "fontWeight": "600"},
            {"if": {"column_id": "result"}, "fontSize": "11px"},
        ],
        style_data_conditional=[
            # ── Draw bet highlights (gold) ──
            {"if": {"filter_query": "{_hd} = 1", "column_id": "d_bet"},
             "backgroundColor": "rgba(255,215,0,0.22)", "color": COLORS["draw_color"],
             "fontWeight": "700"},
            {"if": {"filter_query": "{_hd} = 1", "column_id": "d_edge"},
             "backgroundColor": "rgba(255,215,0,0.12)", "color": COLORS["draw_color"]},
            {"if": {"filter_query": "{_hd} = 1", "column_id": "d_odds"},
             "backgroundColor": "rgba(255,215,0,0.06)", "color": COLORS["draw_color"]},
            {"if": {"filter_query": "{_hd} = 1", "column_id": "mdl_ot"},
             "backgroundColor": "rgba(255,215,0,0.06)", "color": COLORS["draw_color"]},
            # ── Home bet highlights (green) ──
            {"if": {"filter_query": "{_hh} = 1", "column_id": "h_bet"},
             "backgroundColor": "rgba(0,255,136,0.22)", "color": COLORS["home_color"],
             "fontWeight": "700"},
            {"if": {"filter_query": "{_hh} = 1", "column_id": "h_edge"},
             "backgroundColor": "rgba(0,255,136,0.12)", "color": COLORS["home_color"]},
            {"if": {"filter_query": "{_hh} = 1", "column_id": "h_odds"},
             "backgroundColor": "rgba(0,255,136,0.06)", "color": COLORS["home_color"]},
            {"if": {"filter_query": "{_hh} = 1", "column_id": "mdl_home"},
             "backgroundColor": "rgba(0,255,136,0.06)", "color": COLORS["home_color"]},
            # ── P&L coloring ──
            {"if": {"filter_query": '{pnl} contains "+"', "column_id": "pnl"},
             "color": COLORS["profit_green"]},
            {"if": {"filter_query": '{pnl} contains "-"', "column_id": "pnl"},
             "color": COLORS["loss_red"]},
            # ── Result coloring ──
            {"if": {"filter_query": '{result} = "draw"', "column_id": "result"},
             "color": COLORS["draw_color"]},
            {"if": {"filter_query": '{result} = "home"', "column_id": "result"},
             "color": COLORS["home_color"]},
            {"if": {"filter_query": '{result} = "away"', "column_id": "result"},
             "color": COLORS["accent_red"]},
            # ── Dim non-bet rows ──
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "away"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "home"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "mdl_away"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "mdl_home"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "mdl_ot"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "h_edge"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "d_edge"},
             "color": COLORS["text_muted"]},
            {"if": {"filter_query": "{_has_bet} = 0", "column_id": "result"},
             "color": COLORS["text_muted"]},
            # Active cell
            {"if": {"state": "active"}, "backgroundColor": "rgba(255,215,0,0.05)",
             "border": "none"},
        ],
        style_as_list_view=True,
    )


# =============================================================================
# App
# =============================================================================
app = dash.Dash(__name__)

DROPDOWN_STYLE = {
    "backgroundColor": "rgba(255,255,255,0.06)", "color": "#fafafa",
    "border": "1px solid rgba(255,255,255,0.08)", "borderRadius": "6px",
    "fontSize": "12px", "minWidth": "100px",
}


def kpi_card(title, value, css_class="kpi-neutral"):
    color = (COLORS["accent_green"] if css_class == "kpi-positive"
             else COLORS["accent_red"] if css_class == "kpi-negative"
             else COLORS["text_primary"])
    return html.Div(style={
        "background": COLORS["bg_card"], "borderRadius": "8px",
        "padding": "16px 20px", "border": f"1px solid {COLORS['border']}",
    }, children=[
        html.Div(title, style={"fontSize": "10px", "color": COLORS["text_muted"],
                                "textTransform": "uppercase", "letterSpacing": "0.5px", "marginBottom": "6px"}),
        html.Div(value, style={"fontSize": "22px", "fontWeight": "700", "color": color,
                                "fontFamily": "'JetBrains Mono', monospace"}),
    ])


def _stat_row(label, value, color=None):
    return html.Div(style={"display": "flex", "justifyContent": "space-between",
                            "padding": "6px 0", "fontSize": "13px"}, children=[
        html.Span(label, style={"color": COLORS["text_secondary"]}),
        html.Span(value, style={"color": color or COLORS["text_primary"],
                                 "fontFamily": "'JetBrains Mono', monospace", "fontWeight": "500"}),
    ])


def serve_layout():
    raw, bets, _ = load_data()

    min_date = bets["game_date"].min() if len(bets) > 0 else None
    max_date = bets["game_date"].max() if len(bets) > 0 else None
    seasons = sorted(bets["season_label"].unique()) if len(bets) > 0 else []
    books = sorted(bets["bookmaker"].unique()) if len(bets) > 0 else []

    return html.Div(style={
        "backgroundColor": COLORS["bg_primary"], "minHeight": "100vh",
        "fontFamily": "Inter, system-ui, sans-serif", "color": COLORS["text_primary"],
        "padding": "20px 30px",
    }, children=[
        # Title
        html.Div(style={"marginBottom": "25px"}, children=[
            html.H1("BLEND PORTFOLIO", style={
                "fontSize": "28px", "fontWeight": "800", "letterSpacing": "2px",
                "background": f"linear-gradient(135deg, {COLORS['blend_color']}, {COLORS['accent_cyan']})",
                "WebkitBackgroundClip": "text", "WebkitTextFillColor": "transparent",
                "margin": "0 0 4px 0",
            }),
            html.Div(f"Draw {DRAW_MIN_EDGE*100:.1f}%+ (e×{DRAW_STAKE_MULT}, cap {DRAW_STAKE_CAP}u) + "
                      f"Home {HOME_MIN_EDGE*100:.1f}–{HOME_MAX_EDGE*100:.0f}% (e×{HOME_STAKE_MULT}, cap {HOME_STAKE_CAP}u)",
                      style={"fontSize": "12px", "color": COLORS["text_muted"], "letterSpacing": "0.5px"}),
        ]),

        # Filters
        html.Div(style={
            "display": "flex", "gap": "20px", "marginBottom": "12px", "flexWrap": "wrap",
            "padding": "15px 20px", "background": COLORS["bg_card"],
            "borderRadius": "10px", "border": f"1px solid {COLORS['border']}",
        }, children=[
            html.Div(children=[
                html.Div("DATE RANGE", style={"fontSize": "10px", "color": COLORS["text_muted"],
                                               "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.DatePickerRange(id="date-filter", start_date=min_date, end_date=max_date,
                                   display_format="MMM D", style={"fontSize": "12px"}),
            ]),
            html.Div(children=[
                html.Div("SEASON", style={"fontSize": "10px", "color": COLORS["text_muted"],
                                           "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Dropdown(id="season-filter",
                             options=[{"label": s, "value": s} for s in seasons],
                             multi=True, placeholder="All", style=DROPDOWN_STYLE),
            ]),
            html.Div(children=[
                html.Div("BOOK", style={"fontSize": "10px", "color": COLORS["text_muted"],
                                         "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Dropdown(id="book-filter",
                             options=[{"label": b, "value": b} for b in books],
                             multi=True, placeholder="All Books",
                             style={**DROPDOWN_STYLE, "minWidth": "140px"}),
            ]),
            html.Div(children=[
                html.Div("STAKING", style={"fontSize": "10px", "color": COLORS["text_muted"],
                                            "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Dropdown(id="staking-filter",
                             options=[
                                 {"label": "Edge-Prop", "value": "edge_prop"},
                                 {"label": "Flat (1u)", "value": "flat"},
                             ],
                             value="edge_prop", clearable=False,
                             style={**DROPDOWN_STYLE, "minWidth": "120px"}),
            ]),
            html.Div(children=[
                html.Div("STAKE MODE", style={"fontSize": "10px", "color": COLORS["text_muted"],
                                               "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Dropdown(id="stake-mode",
                             options=[
                                 {"label": "To Risk", "value": "to_risk"},
                                 {"label": "To Win", "value": "to_win"},
                             ],
                             value="to_risk", clearable=False,
                             style={**DROPDOWN_STYLE, "minWidth": "100px"}),
            ]),
        ]),

        # Edge threshold filters
        html.Div(style={
            "display": "flex", "gap": "20px", "marginBottom": "25px", "flexWrap": "wrap",
            "padding": "12px 20px", "background": COLORS["bg_card"],
            "borderRadius": "10px", "border": f"1px solid {COLORS['border']}",
            "alignItems": "flex-end",
        }, children=[
            html.Div(style={"fontSize": "11px", "color": COLORS["text_muted"],
                             "fontWeight": "600", "textTransform": "uppercase",
                             "letterSpacing": "0.5px", "alignSelf": "center"},
                     children="EDGE THRESHOLDS"),
            html.Div(children=[
                html.Div("DRAW MIN %", style={"fontSize": "10px", "color": COLORS["draw_color"],
                                               "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Input(id="draw-min-edge", type="number", value=DRAW_MIN_EDGE * 100,
                          step=0.1, min=0, max=20, debounce=True,
                          style={"backgroundColor": "rgba(255,255,255,0.06)", "color": "#fafafa",
                                 "border": f"1px solid {COLORS['border']}", "borderRadius": "6px",
                                 "padding": "6px 10px", "fontSize": "13px", "width": "80px",
                                 "fontFamily": "'JetBrains Mono', monospace"}),
            ]),
            html.Div(children=[
                html.Div("HOME MIN %", style={"fontSize": "10px", "color": COLORS["home_color"],
                                               "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Input(id="home-min-edge", type="number", value=HOME_MIN_EDGE * 100,
                          step=0.1, min=0, max=20, debounce=True,
                          style={"backgroundColor": "rgba(255,255,255,0.06)", "color": "#fafafa",
                                 "border": f"1px solid {COLORS['border']}", "borderRadius": "6px",
                                 "padding": "6px 10px", "fontSize": "13px", "width": "80px",
                                 "fontFamily": "'JetBrains Mono', monospace"}),
            ]),
            html.Div(children=[
                html.Div("HOME MAX %", style={"fontSize": "10px", "color": COLORS["home_color"],
                                               "textTransform": "uppercase", "marginBottom": "4px"}),
                dcc.Input(id="home-max-edge", type="number", value=HOME_MAX_EDGE * 100,
                          step=0.1, min=0, max=20, debounce=True,
                          style={"backgroundColor": "rgba(255,255,255,0.06)", "color": "#fafafa",
                                 "border": f"1px solid {COLORS['border']}", "borderRadius": "6px",
                                 "padding": "6px 10px", "fontSize": "13px", "width": "80px",
                                 "fontFamily": "'JetBrains Mono', monospace"}),
            ]),
        ]),

        # KPIs
        html.Div(id="kpi-row"),

        # Equity curves (full width)
        html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                         "border": f"1px solid {COLORS['border']}", "padding": "10px",
                         "marginBottom": "20px"}, children=[
            dcc.Graph(id="equity-curves", config={"displayModeBar": False}),
        ]),

        # Monthly + Drawdown
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                         "marginBottom": "20px"}, children=[
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="monthly-contrib", config={"displayModeBar": False}),
            ]),
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="drawdown-chart", config={"displayModeBar": False}),
            ]),
        ]),

        # Edge ROI + Season + Daily
        html.Div(style={"display": "grid", "gridTemplateColumns": "1fr 1fr", "gap": "20px",
                         "marginBottom": "20px"}, children=[
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="edge-roi-chart", config={"displayModeBar": False}),
            ]),
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="season-chart", config={"displayModeBar": False}),
            ]),
        ]),

        html.Div(style={"display": "grid", "gridTemplateColumns": "2fr 1fr", "gap": "20px",
                         "marginBottom": "20px"}, children=[
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="daily-bars", config={"displayModeBar": False}),
            ]),
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "10px"}, children=[
                dcc.Graph(id="overlap-chart", config={"displayModeBar": False}),
            ]),
        ]),

        # Stats panel
        html.Div(id="stats-panel", style={"marginBottom": "20px"}),

        # Bet table
        html.Div(style={"fontSize": "14px", "fontWeight": "600", "color": COLORS["text_secondary"],
                         "textTransform": "uppercase", "letterSpacing": "1px", "marginBottom": "12px"},
                 children="Bet Log"),
        html.Div(id="bets-table", style={"background": COLORS["bg_card"], "borderRadius": "10px",
                                          "border": f"1px solid {COLORS['border']}", "padding": "10px",
                                          "marginBottom": "20px"}),

        # Game grid
        html.Div(style={"marginBottom": "12px"}, children=[
            html.Div("Game Grid", style={"fontSize": "14px", "fontWeight": "600",
                     "color": COLORS["text_secondary"], "textTransform": "uppercase",
                     "letterSpacing": "1px", "marginBottom": "4px"}),
            html.Div("Mdl = model probability · Edge/Odds/Bet = from the book we bet · "
                     "Highlighted rows = bet placed · Dimmed = no bet",
                     style={"fontSize": "11px", "color": COLORS["text_muted"]}),
        ]),
        html.Div(id="game-grid", style={"background": COLORS["bg_card"], "borderRadius": "10px",
                                         "border": f"1px solid {COLORS['border']}", "padding": "10px",
                                         "marginBottom": "20px"}),

        dcc.Store(id="filtered-bets"),
    ])


app.layout = serve_layout


# =============================================================================
# Callbacks
# =============================================================================
@app.callback(
    Output("filtered-bets", "data"),
    [
        Input("date-filter", "start_date"),
        Input("date-filter", "end_date"),
        Input("season-filter", "value"),
        Input("book-filter", "value"),
        Input("draw-min-edge", "value"),
        Input("home-min-edge", "value"),
        Input("home-max-edge", "value"),
    ],
)
def filter_bets(start_date, end_date, seasons, books, draw_min, home_min, home_max):
    _, bets, _ = load_data()
    if len(bets) == 0:
        return bets.to_json(date_format="iso", orient="split")

    # Edge thresholds (inputs are in %, convert to decimal)
    draw_min_dec = (draw_min if draw_min is not None else DRAW_MIN_EDGE * 100) / 100
    home_min_dec = (home_min if home_min is not None else HOME_MIN_EDGE * 100) / 100
    home_max_dec = (home_max if home_max is not None else HOME_MAX_EDGE * 100) / 100

    draw_mask = (bets["side"] == "draw") & (bets["edge"] >= draw_min_dec)
    home_mask = ((bets["side"] == "home")
                 & (bets["edge"] >= home_min_dec)
                 & (bets["edge"] <= home_max_dec))
    bets = bets[draw_mask | home_mask]

    if start_date:
        bets = bets[bets["game_date"] >= pd.to_datetime(start_date)]
    if end_date:
        bets = bets[bets["game_date"] <= pd.to_datetime(end_date)]
    if seasons:
        bets = bets[bets["season_label"].isin(seasons)]
    if books:
        bets = bets[bets["bookmaker"].isin(books)]
    return bets.to_json(date_format="iso", orient="split")


@app.callback(
    [
        Output("kpi-row", "children"),
        Output("equity-curves", "figure"),
        Output("monthly-contrib", "figure"),
        Output("drawdown-chart", "figure"),
        Output("edge-roi-chart", "figure"),
        Output("season-chart", "figure"),
        Output("daily-bars", "figure"),
        Output("overlap-chart", "figure"),
        Output("stats-panel", "children"),
        Output("bets-table", "children"),
        Output("game-grid", "children"),
    ],
    [
        Input("filtered-bets", "data"),
        Input("staking-filter", "value"),
        Input("stake-mode", "value"),
    ],
)
def update_all(json_data, staking, stake_mode):
    bets = pd.read_json(io.StringIO(json_data), orient="split") if json_data else pd.DataFrame()
    if "game_date" in bets.columns:
        bets["game_date"] = pd.to_datetime(bets["game_date"])

    raw, _, _ = load_data()
    staking = staking or "edge_prop"
    stake_mode = stake_mode or "to_risk"

    # Normalize stake/pnl columns based on staking + mode
    if stake_mode == "to_win" and len(bets) > 0:
        bets["profit"] = bets["profit_tw"]
        bets["flat_profit"] = bets["flat_profit_tw"]
        if staking == "edge_prop":
            bets["stake"] = bets["stake_tw"]
        else:
            bets["stake"] = bets["flat_stake_tw"]
    elif staking != "edge_prop" and len(bets) > 0:
        bets["stake"] = 1.0

    pnl_col = "profit" if staking == "edge_prop" else "flat_profit"

    # KPIs
    if len(bets) > 0:
        d = bets[bets["side"] == "draw"]
        h = bets[bets["side"] == "home"]

        total_pnl = bets[pnl_col].sum()
        total_staked = bets["stake"].sum()
        total_roi = total_pnl / total_staked * 100 if total_staked > 0 else 0

        d_pnl = d[pnl_col].sum()
        d_staked = d["stake"].sum()
        d_roi = d_pnl / d_staked * 100 if d_staked > 0 else 0

        h_pnl = h[pnl_col].sum()
        h_staked = h["stake"].sum()
        h_roi = h_pnl / h_staked * 100 if h_staked > 0 else 0

        daily = bets.groupby("game_date")[pnl_col].sum()
        sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0
        cum = daily.cumsum()
        max_dd = (cum - cum.cummax()).min()

        # Correlation
        d_daily = d.groupby("game_date")[pnl_col].sum()
        h_daily = h.groupby("game_date")[pnl_col].sum()
        corr_df = pd.DataFrame({"d": d_daily, "h": h_daily}).fillna(0)
        corr = corr_df["d"].corr(corr_df["h"])

        # Significance
        if len(daily) > 1 and daily.std() > 0:
            se = daily.std() / np.sqrt(len(daily))
            t_stat = daily.mean() / se
            p_value = erfc(abs(t_stat) / np.sqrt(2))
        else:
            t_stat, p_value = 0, 1.0

        # Max losing streak
        streak = 0
        max_streak = 0
        for v in daily.values:
            if v < 0:
                streak += 1
                max_streak = max(max_streak, streak)
            else:
                streak = 0

        def vc(v):
            if isinstance(v, str):
                return "kpi-positive" if "+" in v else ("kpi-negative" if v.startswith("-") else "kpi-neutral")
            return "kpi-positive" if v > 0 else "kpi-negative"

        kpi_row = html.Div(style={
            "display": "grid", "gridTemplateColumns": "repeat(8, 1fr)",
            "gap": "12px", "marginBottom": "20px",
        }, children=[
            kpi_card("Blend ROI", f"{total_roi:+.1f}%", vc(total_roi)),
            kpi_card("Total P&L", f"{total_pnl:+.1f}u", vc(total_pnl)),
            kpi_card("Total Bets", f"{len(bets):,}", "kpi-neutral"),
            kpi_card("Sharpe", f"{sharpe:.2f}", vc(sharpe)),
            kpi_card("Max DD", f"{max_dd:.1f}u", "kpi-negative"),
            kpi_card("Draw P&L", f"{d_pnl:+.1f}u ({d_roi:+.1f}%)", vc(d_pnl)),
            kpi_card("Home P&L", f"{h_pnl:+.1f}u ({h_roi:+.1f}%)", vc(h_pnl)),
            kpi_card("Correlation", f"{corr:+.3f}", "kpi-neutral"),
        ])

        # Stats panel
        stats_panel = html.Div(style={
            "display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "20px",
        }, children=[
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "16px"}, children=[
                html.Div("DRAW STREAM", style={"fontSize": "13px", "fontWeight": "600",
                         "color": COLORS["draw_color"], "marginBottom": "12px"}),
                _stat_row("Bets", f"{len(d):,}"),
                _stat_row("Win Rate", f"{d['won'].mean():.1%}" if len(d) > 0 else "—"),
                _stat_row("Avg Edge", f"{d['edge'].mean()*100:.2f}%" if len(d) > 0 else "—"),
                _stat_row("Avg Odds", f"{d['odds'].mean():.2f}" if len(d) > 0 else "—"),
                _stat_row("Avg Stake", f"{d['stake'].mean():.3f}u" if len(d) > 0 else "—"),
                _stat_row("P&L", f"{d_pnl:+.1f}u", COLORS["accent_green"] if d_pnl > 0 else COLORS["accent_red"]),
            ]),
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "16px"}, children=[
                html.Div("HOME STREAM", style={"fontSize": "13px", "fontWeight": "600",
                         "color": COLORS["home_color"], "marginBottom": "12px"}),
                _stat_row("Bets", f"{len(h):,}"),
                _stat_row("Win Rate", f"{h['won'].mean():.1%}" if len(h) > 0 else "—"),
                _stat_row("Avg Edge", f"{h['edge'].mean()*100:.2f}%" if len(h) > 0 else "—"),
                _stat_row("Avg Odds", f"{h['odds'].mean():.2f}" if len(h) > 0 else "—"),
                _stat_row("Avg Stake", f"{h['stake'].mean():.3f}u" if len(h) > 0 else "—"),
                _stat_row("P&L", f"{h_pnl:+.1f}u", COLORS["accent_green"] if h_pnl > 0 else COLORS["accent_red"]),
            ]),
            html.Div(style={"background": COLORS["bg_card"], "borderRadius": "10px",
                             "border": f"1px solid {COLORS['border']}", "padding": "16px"}, children=[
                html.Div("PORTFOLIO STATS", style={"fontSize": "13px", "fontWeight": "600",
                         "color": COLORS["blend_color"], "marginBottom": "12px"}),
                _stat_row("t-statistic", f"{t_stat:.2f}",
                          COLORS["accent_green"] if p_value < 0.05 else COLORS["text_secondary"]),
                _stat_row("p-value", f"{p_value:.3f}",
                          COLORS["accent_green"] if p_value < 0.05 else COLORS["accent_red"]),
                _stat_row("Daily Corr", f"{corr:+.3f}"),
                _stat_row("Max Lose Streak", f"{max_streak} days", COLORS["accent_red"]),
                _stat_row("Betting Days", f"{len(daily)}"),
                _stat_row("Avg Daily P&L", f"{daily.mean():+.2f}u"),
            ]),
        ])
    else:
        kpi_row = html.Div("No data", style={"color": COLORS["text_muted"], "textAlign": "center", "padding": "40px"})
        stats_panel = html.Div()

    # Charts
    equity = build_equity_curves(bets, staking)
    monthly = build_monthly_contribution(bets, staking)
    dd = build_drawdown(bets, staking)
    edge_roi = build_edge_roi(bets, staking)
    season = build_season_comparison(bets, staking)
    daily_b = build_daily_bars(bets, staking)
    overlap = build_overlap_roi(bets, staking)
    table = build_bets_table(bets, staking)
    grid = build_game_grid(raw, bets, staking)

    return kpi_row, equity, monthly, dd, edge_roi, season, daily_b, overlap, stats_panel, table, grid


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    app.run(debug=False, port=PORT)
