# -*- coding: utf-8 -*-
"""
NHL 3-Way Model — Portfolio Results Dashboard

Self-contained interactive dashboard showing live deployment performance,
cross-bookmaker market analysis, model calibration, and backtest comparison.
Reads only from committed CSV files — no external dependencies beyond the
packages listed below.

Run:   python results_dashboard.py
Open:  http://127.0.0.1:8060

@author: chazf
"""

import os
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =============================================================================
# Paths — relative to this script so the repo is portable
# =============================================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TRACKING_FILE  = os.path.join(SCRIPT_DIR, "data", "predictions", "tracking_results.csv")
BACKTEST_FILE  = os.path.join(SCRIPT_DIR, "backtest", "backtest_results_3way.csv")
PORT = 8060

# =============================================================================
# Color palette — Whizard dark theme
# =============================================================================
C = {
    "bg":          "#020219",
    "card":        "rgba(255,255,255,0.03)",
    "border":      "rgba(255,255,255,0.08)",
    "border_acc":  "rgba(234,103,255,0.35)",
    "text":        "#fafafa",
    "muted":       "rgba(255,255,255,0.45)",
    "magenta":     "#ea67ff",
    "cyan":        "#00a3ff",
    "green":       "#00ff88",
    "gold":        "#ffd700",
    "red":         "#ff4757",
    "orange":      "#ff9f43",
}

# =============================================================================
# Data loading
# =============================================================================

def load_live():
    """Load and pre-process the live tracking results."""
    df = pd.read_csv(TRACKING_FILE)
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Graded rows only (results are filled in)
    df = df[df["bet_won"].notna() & (df["bet_won"] != "")]
    df["bet_won"]           = pd.to_numeric(df["bet_won"],           errors="coerce")
    df["draw_flat_profit"]  = pd.to_numeric(df["draw_flat_profit"],  errors="coerce")
    df["draw_kelly_profit"] = pd.to_numeric(df["draw_kelly_profit"], errors="coerce")
    df["draw_kelly_stake"]  = pd.to_numeric(df["draw_kelly_stake"],  errors="coerce")
    df["bet_odds_dec"]      = pd.to_numeric(df["bet_odds_dec"],      errors="coerce")
    df["bet_edge"]          = pd.to_numeric(df["bet_edge"],          errors="coerce")
    df["p_ot"]              = pd.to_numeric(df["p_ot"],              errors="coerce")
    return df


def unique_draw_bets(df):
    """One row per game where a draw bet was flagged (first occurrence)."""
    draw = df[df["bet_side"] == "draw"].copy()
    return draw.drop_duplicates(subset="game_id", keep="first")


def load_backtest():
    """Load backtest results — only games where a bet was actually flagged (kelly_stake > 0)."""
    df = pd.read_csv(BACKTEST_FILE, parse_dates=["game_date"])
    df = df[df["bet_side"] == "draw"].copy()
    df["bet_profit"]   = pd.to_numeric(df["bet_profit"],   errors="coerce")
    df["bet_odds_dec"] = pd.to_numeric(df["bet_odds_dec"], errors="coerce")
    df["kelly_stake"]  = pd.to_numeric(df["kelly_stake"],  errors="coerce")
    df["bet_won"]      = pd.to_numeric(df["bet_won"],      errors="coerce")
    df["p_ot"]         = pd.to_numeric(df["p_ot"],         errors="coerce")
    # Only games where the edge filter triggered a bet
    df = df[df["kelly_stake"] > 0]
    # One row per game — best-edge bookmaker line
    df = df.sort_values("bet_edge", ascending=False).drop_duplicates(subset="game_id", keep="first").copy()
    df["bet_profit"] = df.apply(
        lambda r: (r["bet_odds_dec"] - 1) if r["bet_won"] == 1 else -1, axis=1
    )
    return df


# =============================================================================
# Chart helpers
# =============================================================================

def empty_fig(msg="No data"):
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False,
                       font=dict(color=C["muted"], size=14),
                       xref="paper", yref="paper", x=0.5, y=0.5)
    _style(fig)
    return fig


def _style(fig, title=None):
    fig.update_layout(
        paper_bgcolor=C["bg"],
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=C["text"], size=12),
        title=dict(text=title or "", font=dict(size=15, color=C["text"]), x=0.01) if title else None,
        margin=dict(l=50, r=30, t=40 if title else 20, b=40),
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.1)", zeroline=False),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.05)",
                   linecolor="rgba(255,255,255,0.1)", zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(color=C["muted"], size=11)),
        hovermode="x unified",
    )


def kpi_card(label, value, color=None):
    return html.Div([
        html.Div(label, style={"color": C["muted"], "fontSize": "11px",
                                "textTransform": "uppercase", "letterSpacing": "0.08em",
                                "marginBottom": "6px"}),
        html.Div(value, style={"color": color or C["text"], "fontSize": "28px",
                               "fontWeight": "700", "lineHeight": "1"}),
    ], style={
        "background": C["card"],
        "border": f"1px solid {C['border']}",
        "borderRadius": "10px",
        "padding": "18px 22px",
        "flex": "1",
        "minWidth": "140px",
    })


# =============================================================================
# Tab 1 — Live Performance
# =============================================================================

def build_live_tab(df_raw):
    bets = unique_draw_bets(df_raw)
    if bets.empty:
        return html.Div("No live data available.", style={"color": C["muted"], "padding": "40px"})

    n         = len(bets)
    wins      = int(bets["bet_won"].sum())
    wr        = wins / n
    flat_pl   = bets["draw_flat_profit"].sum()
    roi       = flat_pl / n
    avg_odds  = bets["bet_odds_dec"].mean()
    breakeven = 1 / avg_odds

    # Equity curve
    daily = (bets.groupby("game_date")
                 .agg(n_bets=("bet_won","count"),
                      wins=("bet_won","sum"),
                      flat_pl=("draw_flat_profit","sum"))
                 .reset_index()
                 .sort_values("game_date"))
    daily["cum_pl"] = daily["flat_pl"].cumsum()
    daily["roi_pct"] = daily["cum_pl"] / daily["n_bets"].cumsum() * 100

    equity_fig = go.Figure()
    # Zero reference
    equity_fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.15)", dash="dot"), layer="below")
    # Fill area
    equity_fig.add_trace(go.Scatter(
        x=daily["game_date"], y=daily["cum_pl"],
        fill="tozeroy",
        fillcolor="rgba(0,255,136,0.06)",
        line=dict(color=C["green"], width=2.5),
        name="Cumulative Profit (units)",
        hovertemplate="%{x|%b %d}: <b>%{y:+.2f}u</b><extra></extra>",
    ))
    _style(equity_fig, title="Live Equity Curve — Flat Staking (1u per bet)")

    # Daily table
    daily["date_str"] = daily["game_date"].dt.strftime("%b %d")
    daily["wl"]       = daily["wins"].astype(int).astype(str) + "W / " + (daily["n_bets"] - daily["wins"]).astype(int).astype(str) + "L"
    daily["pl_str"]   = daily["flat_pl"].map(lambda x: f"{x:+.2f}")
    daily["cum_str"]  = daily["cum_pl"].map(lambda x: f"{x:+.2f}")
    daily["roi_str"]  = daily["roi_pct"].map(lambda x: f"{x:+.1f}%")

    tbl_data = daily[["date_str","n_bets","wl","pl_str","cum_str"]].rename(columns={
        "date_str": "Date", "n_bets": "Bets", "wl": "W / L",
        "pl_str": "Daily P&L", "cum_str": "Cumulative",
    }).to_dict("records")

    table = html.Div(
        html.Table([
            html.Thead(html.Tr([
                html.Th(c, style={"color": C["muted"], "fontWeight": "600",
                                  "padding": "8px 14px", "fontSize": "11px",
                                  "textTransform": "uppercase", "letterSpacing": "0.07em",
                                  "borderBottom": f"1px solid {C['border']}", "textAlign": "left"})
                for c in ["Date","Bets","W / L","Daily P&L","Cumulative"]
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(row[c], style={
                        "padding": "7px 14px",
                        "color": (C["green"] if "+" in str(row.get("Daily P&L",""))
                                  else C["red"] if row.get("Daily P&L","").startswith("-")
                                  else C["text"]) if c == "Daily P&L" else C["text"],
                        "fontSize": "13px",
                        "borderBottom": f"1px solid rgba(255,255,255,0.04)",
                    })
                    for c in ["Date","Bets","W / L","Daily P&L","Cumulative"]
                ])
                for row in tbl_data
            ]),
        ], style={"width": "100%", "borderCollapse": "collapse"}),
        style={"background": C["card"], "border": f"1px solid {C['border']}",
               "borderRadius": "10px", "overflow": "hidden", "marginTop": "20px"}
    )

    return html.Div([
        # KPI row
        html.Div([
            kpi_card("Total Bets",    str(n)),
            kpi_card("Win Rate",      f"{wr:.1%}",   C["cyan"]),
            kpi_card("Breakeven",     f"{breakeven:.1%}", C["muted"]),
            kpi_card("Avg Odds",      f"{avg_odds:.2f}"),
            kpi_card("Flat Profit",   f"{flat_pl:+.2f}u", C["green"] if flat_pl > 0 else C["red"]),
            kpi_card("Flat ROI",      f"{roi:.1%}",  C["green"] if roi > 0 else C["red"]),
        ], style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "24px"}),

        html.Div(dcc.Graph(figure=equity_fig, config={"displayModeBar": False}),
                 style={"background": C["card"], "border": f"1px solid {C['border']}",
                        "borderRadius": "10px", "padding": "8px", "marginBottom": "20px"}),
        table,
    ], style={"padding": "24px"})


# =============================================================================
# Tab 2 — Bookmaker Analysis
# =============================================================================

def build_books_tab(df_raw):
    draw = df_raw[df_raw["bet_side"] == "draw"].copy()
    if draw.empty:
        return html.Div("No data.", style={"color": C["muted"], "padding": "40px"})

    # Per-bookmaker stats
    stats = (draw.groupby("bookmaker")
                 .agg(n=("draw_flat_profit","count"),
                      wins=("bet_won","sum"),
                      flat_pl=("draw_flat_profit","sum"),
                      avg_odds=("bet_odds_dec","mean"))
                 .reset_index())
    stats["roi"]     = stats["flat_pl"] / stats["n"] * 100
    stats["win_rate"] = stats["wins"] / stats["n"] * 100
    stats = stats[stats["n"] >= 20].sort_values("roi", ascending=True)

    # Clean up bookmaker names
    NAME_MAP = {
        "draftkings": "DraftKings", "fanduel": "FanDuel", "betmgm": "BetMGM",
        "williamhill_us": "William Hill", "fanatics": "Fanatics", "bovada": "Bovada",
        "betrivers": "BetRivers", "pinnacle": "Pinnacle", "matchbook": "Matchbook",
        "unibet_nl": "Unibet NL", "unibet_se": "Unibet SE", "unibet_fr": "Unibet FR",
        "winamax_fr": "Winamax FR", "winamax_de": "Winamax DE",
        "marathonbet": "Marathonbet", "onexbet": "1xBet", "coolbet": "Coolbet",
        "leovegas_se": "LeoVegas", "codere_it": "Codere", "tipico_de": "Tipico",
        "betclic_fr": "Betclic", "parionssport_fr": "ParionsSport",
    }
    stats["book_label"] = stats["bookmaker"].map(lambda x: NAME_MAP.get(x, x))
    colors = [C["red"] if r < 0 else C["green"] for r in stats["roi"]]

    roi_fig = go.Figure(go.Bar(
        x=stats["roi"],
        y=stats["book_label"],
        orientation="h",
        marker=dict(color=colors, opacity=0.85),
        text=[f"{r:+.1f}%" for r in stats["roi"]],
        textposition="outside",
        textfont=dict(size=11, color=C["text"]),
        hovertemplate="<b>%{y}</b><br>ROI: %{x:.1f}%<extra></extra>",
    ))
    roi_fig.add_vline(x=0, line=dict(color="rgba(255,255,255,0.2)", width=1))
    _style(roi_fig, title="Draw Bet ROI by Bookmaker (flat 1u, min 20 bets)")
    roi_fig.update_layout(
        height=max(350, len(stats) * 28 + 80),
        yaxis=dict(tickfont=dict(size=11)),
        xaxis=dict(ticksuffix="%"),
        bargap=0.35,
        hovermode="y unified",
    )

    # Odds distribution per book (sorted by avg odds descending)
    odds_stats = stats.sort_values("avg_odds", ascending=False)
    odds_fig = go.Figure(go.Bar(
        x=odds_stats["book_label"],
        y=odds_stats["avg_odds"],
        marker=dict(color=C["cyan"], opacity=0.75),
        text=[f"{o:.2f}" for o in odds_stats["avg_odds"]],
        textposition="outside",
        textfont=dict(size=10, color=C["text"]),
        hovertemplate="<b>%{x}</b><br>Avg odds: %{y:.3f}<extra></extra>",
    ))
    odds_fig.add_hline(y=1/0.238, line=dict(color=C["muted"], dash="dot"),
                       annotation_text="Breakeven (23.8%)",
                       annotation_font=dict(color=C["muted"], size=10))
    _style(odds_fig, title="Average Draw Odds by Bookmaker")
    odds_fig.update_layout(height=320, xaxis_tickangle=-35, bargap=0.3)

    callout = html.Div([
        html.Div("Market Efficiency Note", style={
            "color": C["magenta"], "fontWeight": "700", "fontSize": "13px", "marginBottom": "10px"
        }),
        html.P([
            html.B("FanDuel"), " is the only book showing negative ROI (−15.6%). "
            "It has tightened draw pricing more aggressively than peers, posting lines 15–20 cents "
            "lower than the field on the same game. A bettor limited to FanDuel alone would lose money. "
            "The same model, shopping all 20+ books, returns ", html.B("+19.1% ROI"), "."
        ], style={"color": C["muted"], "fontSize": "13px", "lineHeight": "1.7", "margin": "0 0 10px"}),
        html.P([
            "The 35+ percentage point spread in ROI across books reflects a structural pricing gradient: "
            "US-licensed sharp books have adapted their draw models to the 2025-26 parity regime. "
            "Recreational and international books have not. Line shopping is not optional — it is "
            "the mechanism by which structural edge is converted into realized profit."
        ], style={"color": C["muted"], "fontSize": "13px", "lineHeight": "1.7", "margin": "0"}),
    ], style={
        "background": "rgba(234,103,255,0.06)",
        "border": f"1px solid {C['border_acc']}",
        "borderRadius": "10px",
        "padding": "18px 22px",
        "marginBottom": "24px",
    })

    return html.Div([
        callout,
        html.Div([
            html.Div(dcc.Graph(figure=roi_fig,  config={"displayModeBar": False}),
                     style={"flex": "1.4", "background": C["card"],
                            "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "8px"}),
            html.Div(dcc.Graph(figure=odds_fig, config={"displayModeBar": False}),
                     style={"flex": "1",   "background": C["card"],
                            "border": f"1px solid {C['border']}", "borderRadius": "10px", "padding": "8px"}),
        ], style={"display": "flex", "gap": "16px", "alignItems": "flex-start"}),
    ], style={"padding": "24px"})


# =============================================================================
# Tab 3 — Model Calibration
# =============================================================================

def build_calibration_tab(df_raw):
    bets = unique_draw_bets(df_raw)
    if bets.empty:
        return html.Div("No data.", style={"color": C["muted"], "padding": "40px"})

    # Edge bucket analysis
    def edge_bucket(e):
        if e < 0.030: return "2.5–3.0%"
        if e < 0.035: return "3.0–3.5%"
        if e < 0.040: return "3.5–4.0%"
        return "4.0%+"

    bets = bets.copy()
    bets["edge_bucket"] = bets["bet_edge"].apply(edge_bucket)
    bucket_order = ["2.5–3.0%", "3.0–3.5%", "3.5–4.0%", "4.0%+"]
    grp = (bets.groupby("edge_bucket")
               .agg(n=("bet_won","count"),
                    wins=("bet_won","sum"),
                    flat_pl=("draw_flat_profit","sum"),
                    avg_odds=("bet_odds_dec","mean"))
               .reindex(bucket_order).dropna().reset_index())
    grp["roi"]     = grp["flat_pl"] / grp["n"] * 100
    grp["win_rate"] = grp["wins"] / grp["n"] * 100
    grp["breakeven"] = 100 / grp["avg_odds"]

    edge_fig = go.Figure()
    edge_fig.add_trace(go.Bar(
        x=grp["edge_bucket"], y=grp["roi"],
        name="Realized ROI (%)",
        marker=dict(color=[C["green"] if r > 0 else C["red"] for r in grp["roi"]], opacity=0.8),
        text=[f"{r:+.1f}%" for r in grp["roi"]],
        textposition="outside",
        textfont=dict(size=11),
        hovertemplate="<b>%{x}</b><br>ROI: %{y:.1f}%<extra></extra>",
    ))
    edge_fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.2)"))
    _style(edge_fig, title="Realized ROI by Model-Claimed Edge Bucket")
    edge_fig.update_layout(height=300, showlegend=False, bargap=0.4)

    # Win rate vs breakeven per bucket
    wr_fig = go.Figure()
    wr_fig.add_trace(go.Bar(
        x=grp["edge_bucket"], y=grp["win_rate"],
        name="Actual Win Rate",
        marker=dict(color=C["cyan"], opacity=0.75),
        text=[f"{w:.1f}%" for w in grp["win_rate"]],
        textposition="outside",
        textfont=dict(size=11),
    ))
    wr_fig.add_trace(go.Scatter(
        x=grp["edge_bucket"], y=grp["breakeven"],
        mode="markers+lines",
        name="Breakeven Win Rate",
        line=dict(color=C["gold"], dash="dot", width=1.5),
        marker=dict(size=7, color=C["gold"]),
    ))
    _style(wr_fig, title="Actual Win Rate vs Breakeven by Edge Bucket")
    wr_fig.update_layout(height=300, bargap=0.4,
                         legend=dict(orientation="h", y=1.12, x=0))

    # P(OT) calibration — model vs actual
    all_games = df_raw.drop_duplicates(subset="game_id", keep="first").copy()
    all_games["actual_ot"] = all_games["game_outcome_type_result"].isin(["OT","SO"]).astype(int)
    # Daily avg
    daily_cal = (all_games.groupby("game_date")
                           .agg(model_pot=("p_ot","mean"),
                                actual_ot=("actual_ot","mean"),
                                n=("game_id","count"))
                           .reset_index().sort_values("game_date"))
    daily_cal["model_pot_pct"]  = daily_cal["model_pot"]  * 100
    daily_cal["actual_ot_pct"]  = daily_cal["actual_ot"]  * 100
    daily_cal["model_cum"]  = daily_cal["model_pot"].expanding().mean() * 100
    daily_cal["actual_cum"] = daily_cal["actual_ot"].expanding().mean() * 100

    cal_fig = go.Figure()
    cal_fig.add_trace(go.Scatter(
        x=daily_cal["game_date"], y=daily_cal["actual_cum"],
        name="Actual OT Rate (cumulative)",
        line=dict(color=C["green"], width=2.5),
        hovertemplate="%{x|%b %d}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    cal_fig.add_trace(go.Scatter(
        x=daily_cal["game_date"], y=daily_cal["model_cum"],
        name="Model P(OT) (cumulative avg)",
        line=dict(color=C["cyan"], width=2, dash="dash"),
        hovertemplate="%{x|%b %d}: <b>%{y:.1f}%</b><extra></extra>",
    ))
    cal_fig.add_hline(y=22.3, line=dict(color=C["muted"], dash="dot"),
                      annotation_text="Historical avg (22.3%)",
                      annotation_font=dict(color=C["muted"], size=10))
    _style(cal_fig, title="Model P(OT) vs Actual OT Rate — Cumulative, March 2026")
    cal_fig.update_layout(height=300, yaxis_ticksuffix="%",
                          legend=dict(orientation="h", y=1.12, x=0))

    note = html.Div([
        html.Span("Calibration note: ", style={"color": C["magenta"], "fontWeight": "700"}),
        html.Span(
            "The model's average P(OT) in the live period is 24.2% while the actual OT rate "
            "is 28.7% — a 4.5pp gap. Both the model and the market underprice overtime in "
            "the 2025-26 high-parity season. The edge is captured because the model is "
            "systematically closer to reality than book pricing, not because it is perfectly "
            "calibrated in absolute terms.",
            style={"color": C["muted"], "fontSize": "12px", "lineHeight": "1.6"}
        )
    ], style={"background": C["card"], "border": f"1px solid {C['border']}",
              "borderRadius": "8px", "padding": "14px 18px", "marginBottom": "16px"})

    return html.Div([
        note,
        html.Div([
            html.Div(dcc.Graph(figure=edge_fig, config={"displayModeBar": False}),
                     style={"flex":"1","background":C["card"],"border":f"1px solid {C['border']}","borderRadius":"10px","padding":"8px"}),
            html.Div(dcc.Graph(figure=wr_fig, config={"displayModeBar": False}),
                     style={"flex":"1","background":C["card"],"border":f"1px solid {C['border']}","borderRadius":"10px","padding":"8px"}),
        ], style={"display":"flex","gap":"16px","marginBottom":"16px"}),
        html.Div(dcc.Graph(figure=cal_fig, config={"displayModeBar": False}),
                 style={"background":C["card"],"border":f"1px solid {C['border']}","borderRadius":"10px","padding":"8px"}),
    ], style={"padding": "24px"})


# =============================================================================
# Tab 4 — Backtest vs Live
# =============================================================================

def build_backtest_tab(bt, live_bets):
    if bt.empty:
        return html.Div("No backtest data.", style={"color": C["muted"], "padding": "40px"})

    # Season stats
    def season_stats(df):
        n     = len(df)
        wins  = df["bet_won"].sum()
        pl    = df["bet_profit"].sum()
        roi   = pl / n * 100 if n > 0 else 0
        odds  = df["bet_odds_dec"].mean()
        return {"n": n, "wins": int(wins), "roi": roi, "pl": pl, "avg_odds": odds}

    s_2425  = season_stats(bt[bt["season_label"] == "2024-25"])
    s_2526  = season_stats(bt[bt["season_label"] == "2025-26"])
    s_live  = {
        "n":        len(live_bets),
        "wins":     int(live_bets["bet_won"].sum()),
        "roi":      live_bets["draw_flat_profit"].sum() / len(live_bets) * 100,
        "pl":       live_bets["draw_flat_profit"].sum(),
        "avg_odds": live_bets["bet_odds_dec"].mean(),
    }

    # Combined equity curve
    bt_daily = (bt.groupby("game_date")["bet_profit"].sum()
                  .reset_index().sort_values("game_date"))
    bt_daily["cum"] = bt_daily["bet_profit"].cumsum()

    live_daily = (live_bets.groupby("game_date")["draw_flat_profit"].sum()
                            .reset_index().sort_values("game_date"))
    live_daily["cum"] = live_daily["draw_flat_profit"].cumsum()

    combo_fig = go.Figure()
    combo_fig.add_hline(y=0, line=dict(color="rgba(255,255,255,0.1)", dash="dot"))
    combo_fig.add_trace(go.Scatter(
        x=bt_daily["game_date"], y=bt_daily["cum"],
        name="Backtest (2024-26)",
        line=dict(color=C["cyan"], width=2, dash="dash"),
        fill="tozeroy", fillcolor="rgba(0,163,255,0.05)",
        hovertemplate="%{x|%b %d %Y}: <b>%{y:+.1f}u</b><extra></extra>",
    ))
    combo_fig.add_trace(go.Scatter(
        x=live_daily["game_date"], y=live_daily["cum"],
        name="Live (Mar 2026)",
        line=dict(color=C["green"], width=2.5),
        fill="tozeroy", fillcolor="rgba(0,255,136,0.06)",
        hovertemplate="%{x|%b %d}: <b>%{y:+.2f}u</b><extra></extra>",
    ))
    _style(combo_fig, title="Backtest vs Live — Cumulative Flat P&L")
    combo_fig.update_layout(height=360, legend=dict(orientation="h", y=1.12, x=0))

    # Summary table
    rows = [
        ("Season",     "2024-25 OOS",    "2025-26 OOS",    "Live (Mar 2026)"),
        ("Bets",       s_2425["n"],       s_2526["n"],       s_live["n"]),
        ("Win Rate",   f"{s_2425['wins']/s_2425['n']:.1%}", f"{s_2526['wins']/s_2526['n']:.1%}", f"{s_live['wins']/s_live['n']:.1%}"),
        ("Avg Odds",   f"{s_2425['avg_odds']:.2f}", f"{s_2526['avg_odds']:.2f}", f"{s_live['avg_odds']:.2f}"),
        ("Flat ROI",   f"{s_2425['roi']:+.1f}%",   f"{s_2526['roi']:+.1f}%",   f"{s_live['roi']:+.1f}%"),
        ("Flat Profit",f"{s_2425['pl']:+.1f}u",    f"{s_2526['pl']:+.1f}u",    f"{s_live['pl']:+.2f}u"),
    ]

    def cell_color(val):
        if isinstance(val, str) and val.startswith("+"):  return C["green"]
        if isinstance(val, str) and val.startswith("-"):  return C["red"]
        return C["text"]

    tbl = html.Table([
        html.Tbody([
            html.Tr([
                html.Td(cell, style={
                    "padding": "9px 20px",
                    "borderBottom": f"1px solid rgba(255,255,255,0.05)",
                    "color": C["muted"] if i == 0 else cell_color(cell),
                    "fontWeight": "600" if i == 0 else "400",
                    "fontSize": "13px",
                    "textAlign": "left" if i == 0 else "center",
                })
                for i, cell in enumerate(row)
            ]) for row in rows
        ])
    ], style={"width": "100%", "borderCollapse": "collapse"})

    return html.Div([
        html.Div(dcc.Graph(figure=combo_fig, config={"displayModeBar": False}),
                 style={"background": C["card"], "border": f"1px solid {C['border']}",
                        "borderRadius": "10px", "padding": "8px", "marginBottom": "20px"}),
        html.Div([
            html.Div("Performance Summary", style={"color": C["muted"], "fontSize": "11px",
                                                    "textTransform": "uppercase",
                                                    "letterSpacing": "0.08em",
                                                    "padding": "14px 20px 8px",
                                                    "borderBottom": f"1px solid {C['border']}"}),
            tbl,
        ], style={"background": C["card"], "border": f"1px solid {C['border']}",
                  "borderRadius": "10px", "overflow": "hidden"}),
    ], style={"padding": "24px"})


# =============================================================================
# App layout
# =============================================================================

app = dash.Dash(__name__, title="NHL 3-Way Model — Results")
app.config.suppress_callback_exceptions = True

TAB_STYLE = {
    "background": "transparent",
    "border": "none",
    "color": C["muted"],
    "fontFamily": "Inter, sans-serif",
    "fontSize": "13px",
    "fontWeight": "500",
    "padding": "10px 20px",
    "borderBottom": f"2px solid transparent",
}
TAB_SELECTED = {
    **TAB_STYLE,
    "color": C["text"],
    "borderBottom": f"2px solid {C['magenta']}",
    "background": "transparent",
}

app.layout = html.Div([
    # Header
    html.Div([
        html.Div([
            html.Div("NHL 3-Way Regulation Model", style={
                "fontSize": "18px", "fontWeight": "700", "color": C["text"],
                "letterSpacing": "0.02em",
            }),
            html.Div("Whizard Analytics · Draw Edge · March 2026", style={
                "fontSize": "12px", "color": C["muted"], "marginTop": "3px",
            }),
        ]),
        html.Div([
            html.Span("+19.1% Live ROI", style={"color": C["green"], "fontWeight": "700", "fontSize": "14px"}),
            html.Span("  ·  ", style={"color": C["border"]}),
            html.Span("+8.1% Backtest ROI", style={"color": C["cyan"], "fontSize": "14px"}),
            html.Span("  ·  ", style={"color": C["border"]}),
            html.Span("137 Live Bets", style={"color": C["muted"], "fontSize": "13px"}),
        ]),
    ], style={
        "display": "flex", "justifyContent": "space-between", "alignItems": "center",
        "padding": "18px 28px",
        "borderBottom": f"1px solid {C['border']}",
        "background": "rgba(0,0,0,0.3)",
    }),

    # Tabs
    dcc.Tabs(id="tabs", value="live",
             style={"background": C["bg"], "borderBottom": f"1px solid {C['border']}",
                    "padding": "0 20px"},
             children=[
                 dcc.Tab(label="Live Performance",    value="live",      style=TAB_STYLE, selected_style=TAB_SELECTED),
                 dcc.Tab(label="Bookmaker Analysis",  value="books",     style=TAB_STYLE, selected_style=TAB_SELECTED),
                 dcc.Tab(label="Model Calibration",   value="calib",     style=TAB_STYLE, selected_style=TAB_SELECTED),
                 dcc.Tab(label="Backtest vs Live",    value="backtest",  style=TAB_STYLE, selected_style=TAB_SELECTED),
             ]),

    html.Div(id="tab-content"),

], style={"background": C["bg"], "minHeight": "100vh", "fontFamily": "Inter, sans-serif"})


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    df_raw    = load_live()
    live_bets = unique_draw_bets(df_raw)
    bt        = load_backtest()

    if tab == "live":     return build_live_tab(df_raw)
    if tab == "books":    return build_books_tab(df_raw)
    if tab == "calib":    return build_calibration_tab(df_raw)
    if tab == "backtest": return build_backtest_tab(bt, live_bets)
    return html.Div()


if __name__ == "__main__":
    print(f"\n  NHL 3-Way Results Dashboard")
    print(f"  Open: http://127.0.0.1:{PORT}\n")
    app.run(debug=False, port=PORT)
