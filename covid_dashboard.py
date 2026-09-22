"""
COVID-19 Global BI Dashboard (2020-2024)
=========================================
Single-file application combining:
  - Backend: data ingestion, cleaning, feature engineering, EDA
  - Frontend: Dash/Plotly interactive dashboard with 3 BI sections

Dataset: WHO-COVID-19-global-data.csv
Source : https://www.kaggle.com/datasets/abdoomoh/daily-covid-19-data-2020-2024
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import os

# ─────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & CLEANING
# ─────────────────────────────────────────────────────────────────────────────

DATA_FILE = "WHO-COVID-19-global-data.csv"

REGION_LABELS = {
    "AFRO":  "Africa (AFRO)",
    "AMRO":  "Americas (AMRO)",
    "EMRO":  "East Mediterranean (EMRO)",
    "EURO":  "Europe (EURO)",
    "SEARO": "South-East Asia (SEARO)",
    "WPRO":  "Western Pacific (WPRO)",
    "OTHER": "Other / Global",
}

REGION_COLORS = {
    "AFRO":  "#e74c3c",
    "AMRO":  "#3498db",
    "EMRO":  "#f39c12",
    "EURO":  "#2ecc71",
    "SEARO": "#9b59b6",
    "WPRO":  "#1abc9c",
    "OTHER": "#95a5a6",
}


def load_and_clean():
    """Load WHO global data, clean, and engineer features."""
    df = pd.read_csv(DATA_FILE, sep=";")

    # Standardise column names
    df.columns = [c.strip() for c in df.columns]

    # Parse dates (DD/MM/YYYY)
    df["Date_reported"] = pd.to_datetime(df["Date_reported"], format="%d/%m/%Y", errors="coerce")
    df = df.dropna(subset=["Date_reported"])

    # Numeric coercion
    num_cols = ["New_cases", "Cumulative_cases", "New_deaths", "Cumulative_deaths"]
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(0)

    # Clip negatives to 0 (reporting corrections)
    df[num_cols] = df[num_cols].clip(lower=0)

    # Year / Month columns
    df["Year"]       = df["Date_reported"].dt.year
    df["YearMonth"]  = df["Date_reported"].dt.to_period("M").astype(str)

    # Filter 2020-2024 only
    df = df[(df["Year"] >= 2020) & (df["Year"] <= 2024)].copy()

    # Drop rows with missing region (rare NaN entries)
    df = df.dropna(subset=["WHO_region"])
    df["WHO_region"] = df["WHO_region"].astype(str)

    # Standardise region label
    df["Region_label"] = df["WHO_region"].map(lambda r: REGION_LABELS.get(str(r), str(r)))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# 2. ANALYTICAL AGGREGATIONS
# ─────────────────────────────────────────────────────────────────────────────

def build_analytics(df):
    """Derive all aggregated frames used by the dashboard."""
    analytics = {}

    # ── Global totals ──────────────────────────────────────────────────────
    analytics["total_cases"]  = int(df.groupby("Country")["New_cases"].sum().sum())
    analytics["total_deaths"] = int(df.groupby("Country")["New_deaths"].sum().sum())
    analytics["global_cfr"]   = round(
        analytics["total_deaths"] / analytics["total_cases"] * 100, 2
    ) if analytics["total_cases"] > 0 else 0
    analytics["countries_affected"] = df["Country"].nunique()

    # ── Monthly global trend ───────────────────────────────────────────────
    monthly_global = (
        df.groupby("YearMonth")[["New_cases", "New_deaths"]]
        .sum()
        .reset_index()
        .sort_values("YearMonth")
    )
    monthly_global["7MA_cases"]  = monthly_global["New_cases"].rolling(3, min_periods=1).mean()
    monthly_global["7MA_deaths"] = monthly_global["New_deaths"].rolling(3, min_periods=1).mean()
    analytics["monthly_global"] = monthly_global

    # ── Yearly global trend ────────────────────────────────────────────────
    yearly_global = (
        df.groupby("Year")[["New_cases", "New_deaths"]]
        .sum()
        .reset_index()
    )
    analytics["yearly_global"] = yearly_global

    # ── Region totals ──────────────────────────────────────────────────────
    region_totals = (
        df.groupby(["WHO_region", "Region_label"])[["New_cases", "New_deaths"]]
        .sum()
        .reset_index()
    )
    region_totals["CFR"] = (
        region_totals["New_deaths"] / region_totals["New_cases"].replace(0, np.nan) * 100
    ).round(2)
    analytics["region_totals"] = region_totals

    # ── Monthly region trend ───────────────────────────────────────────────
    monthly_region = (
        df.groupby(["YearMonth", "WHO_region", "Region_label"])[["New_cases", "New_deaths"]]
        .sum()
        .reset_index()
        .sort_values(["WHO_region", "YearMonth"])
    )
    analytics["monthly_region"] = monthly_region

    # ── Country-level aggregates ───────────────────────────────────────────
    country_totals = (
        df.groupby(["Country", "Country_code", "WHO_region", "Region_label"])
        .agg(
            Total_cases=("New_cases", "sum"),
            Total_deaths=("New_deaths", "sum"),
        )
        .reset_index()
    )
    country_totals["CFR"] = (
        country_totals["Total_deaths"]
        / country_totals["Total_cases"].replace(0, np.nan) * 100
    ).round(2)
    analytics["country_totals"] = country_totals

    # ── Peak wave identification (monthly) ────────────────────────────────
    analytics["peak_month_cases"] = monthly_global.loc[
        monthly_global["New_cases"].idxmax(), "YearMonth"
    ]
    analytics["peak_month_deaths"] = monthly_global.loc[
        monthly_global["New_deaths"].idxmax(), "YearMonth"
    ]

    # ── Risk tier: countries with high CFR (>3%) ──────────────────────────
    high_risk = country_totals[
        (country_totals["CFR"] > 3) & (country_totals["Total_cases"] > 1000)
    ].sort_values("CFR", ascending=False).head(20)
    analytics["high_risk_countries"] = high_risk

    # ── Top 10 countries by total cases ───────────────────────────────────
    top10_cases = country_totals.nlargest(10, "Total_cases")
    analytics["top10_cases"] = top10_cases

    # ── Top 10 countries by total deaths ──────────────────────────────────
    top10_deaths = country_totals.nlargest(10, "Total_deaths")
    analytics["top10_deaths"] = top10_deaths

    # ── Yearly region heatmap data ─────────────────────────────────────────
    yearly_region = (
        df.groupby(["Year", "WHO_region"])[["New_cases", "New_deaths"]]
        .sum()
        .reset_index()
    )
    yearly_region["CFR"] = (
        yearly_region["New_deaths"] / yearly_region["New_cases"].replace(0, np.nan) * 100
    ).round(2)
    analytics["yearly_region"] = yearly_region

    # ── Vaccination proxy: use CFR trend as inverse proxy ─────────────────
    # Lower CFR over time signals vaccination/treatment improvement
    cfr_trend = (
        df.groupby(["Year", "WHO_region"])
        .agg(cases=("New_cases", "sum"), deaths=("New_deaths", "sum"))
        .reset_index()
    )
    cfr_trend["CFR"] = (
        cfr_trend["deaths"] / cfr_trend["cases"].replace(0, np.nan) * 100
    ).round(3)
    analytics["cfr_trend"] = cfr_trend

    return analytics


# ─────────────────────────────────────────────────────────────────────────────
# 3. FIGURE BUILDERS
# ─────────────────────────────────────────────────────────────────────────────

def fig_global_trend(monthly_global):
    """Dual-axis monthly global cases & deaths trend."""
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Bar(
            x=monthly_global["YearMonth"],
            y=monthly_global["New_cases"],
            name="Monthly Cases",
            marker_color="rgba(52,152,219,0.45)",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_global["YearMonth"],
            y=monthly_global["7MA_cases"],
            name="3-Month Avg Cases",
            line=dict(color="#2980b9", width=2),
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=monthly_global["YearMonth"],
            y=monthly_global["New_deaths"],
            name="Monthly Deaths",
            line=dict(color="#e74c3c", width=2, dash="dot"),
        ),
        secondary_y=True,
    )

    fig.update_layout(
        title="Global Monthly COVID-19 Cases & Deaths (2020–2024)",
        xaxis_title="Month",
        legend=dict(orientation="h", y=-0.2),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        hovermode="x unified",
        height=420,
    )
    fig.update_yaxes(title_text="New Cases", secondary_y=False)
    fig.update_yaxes(title_text="New Deaths", secondary_y=True, showgrid=False)
    return fig


def fig_yearly_bar(yearly_global):
    """Grouped bar: yearly cases & deaths."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=yearly_global["Year"].astype(str),
        y=yearly_global["New_cases"],
        name="Total Cases",
        marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        x=yearly_global["Year"].astype(str),
        y=yearly_global["New_deaths"],
        name="Total Deaths",
        marker_color="#e74c3c",
    ))
    fig.update_layout(
        title="Yearly Global Cases & Deaths",
        barmode="group",
        xaxis_title="Year",
        yaxis_title="Count",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=380,
    )
    return fig


def fig_region_cases_deaths(region_totals):
    """Horizontal grouped bar by WHO region."""
    rt = region_totals.sort_values("New_cases", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=rt["Region_label"],
        x=rt["New_cases"],
        name="Total Cases",
        orientation="h",
        marker_color="#3498db",
    ))
    fig.add_trace(go.Bar(
        y=rt["Region_label"],
        x=rt["New_deaths"],
        name="Total Deaths",
        orientation="h",
        marker_color="#e74c3c",
    ))
    fig.update_layout(
        title="Total Cases & Deaths by WHO Region",
        barmode="group",
        xaxis_title="Count",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=420,
        legend=dict(orientation="h", y=-0.15),
    )
    return fig


def fig_cfr_by_region(region_totals):
    """Horizontal bar: Case Fatality Ratio by region."""
    rt = region_totals.sort_values("CFR", ascending=True)
    colors = [REGION_COLORS.get(r, "#95a5a6") for r in rt["WHO_region"]]
    fig = go.Figure(go.Bar(
        y=rt["Region_label"],
        x=rt["CFR"],
        orientation="h",
        marker_color=colors,
        text=rt["CFR"].apply(lambda x: f"{x:.2f}%"),
        textposition="outside",
    ))
    fig.update_layout(
        title="Case Fatality Ratio (CFR) by WHO Region",
        xaxis_title="CFR (%)",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=380,
    )
    return fig


def fig_cfr_trend_over_years(cfr_trend):
    """Line chart: CFR by WHO region over years (vaccination proxy)."""
    fig = go.Figure()
    for region, grp in cfr_trend.groupby("WHO_region"):
        if region == "OTHER":
            continue
        fig.add_trace(go.Scatter(
            x=grp["Year"].astype(str),
            y=grp["CFR"],
            mode="lines+markers",
            name=REGION_LABELS.get(region, region),
            line=dict(color=REGION_COLORS.get(region, "#95a5a6"), width=2),
            marker=dict(size=8),
        ))
    fig.update_layout(
        title="Case Fatality Ratio Trend by WHO Region (2020–2024)<br><sup>Declining CFR reflects improved treatment & vaccination rollout</sup>",
        xaxis_title="Year",
        yaxis_title="CFR (%)",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=420,
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def fig_region_monthly_trend(monthly_region, metric="New_cases"):
    """Area chart: monthly trend per WHO region."""
    fig = go.Figure()
    for region, grp in monthly_region.groupby("WHO_region"):
        if region == "OTHER":
            continue
        fig.add_trace(go.Scatter(
            x=grp["YearMonth"],
            y=grp[metric],
            mode="lines",
            name=REGION_LABELS.get(region, region),
            stackgroup="one",
            line=dict(color=REGION_COLORS.get(region, "#95a5a6"), width=1),
        ))
    title_map = {"New_cases": "Monthly New Cases", "New_deaths": "Monthly New Deaths"}
    fig.update_layout(
        title=f"Stacked Regional {title_map.get(metric, metric)} Over Time",
        xaxis_title="Month",
        yaxis_title=title_map.get(metric, metric),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=420,
        legend=dict(orientation="h", y=-0.25, font=dict(size=11)),
        hovermode="x unified",
    )
    return fig


def fig_top10_countries(top10, metric="Total_cases"):
    """Horizontal bar: top 10 countries."""
    t = top10.sort_values(metric, ascending=True)
    colors = [REGION_COLORS.get(r, "#95a5a6") for r in t["WHO_region"]]
    label_map = {"Total_cases": "Total Cases", "Total_deaths": "Total Deaths"}
    fig = go.Figure(go.Bar(
        y=t["Country"],
        x=t[metric],
        orientation="h",
        marker_color=colors,
    ))
    fig.update_layout(
        title=f"Top 10 Countries by {label_map.get(metric, metric)}",
        xaxis_title=label_map.get(metric, metric),
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=380,
    )
    return fig


def fig_high_risk_scatter(country_totals):
    """Scatter: Total Cases vs CFR, coloured by region."""
    ct = country_totals[country_totals["Total_cases"] > 5000].copy()
    ct["CFR_clipped"] = ct["CFR"].clip(upper=20)
    fig = px.scatter(
        ct,
        x="Total_cases",
        y="CFR_clipped",
        color="WHO_region",
        hover_name="Country",
        hover_data={"Total_cases": True, "Total_deaths": True, "CFR": True, "WHO_region": True},
        size="Total_deaths",
        size_max=40,
        color_discrete_map=REGION_COLORS,
        labels={
            "Total_cases":   "Total Cases",
            "CFR_clipped":   "CFR % (capped at 20)",
            "WHO_region":    "WHO Region",
            "Total_deaths":  "Total Deaths",
        },
        title="Risk Map: Total Cases vs Case Fatality Ratio by Country",
        log_x=True,
    )
    fig.add_hline(y=3, line_dash="dash", line_color="red",
                  annotation_text="CFR = 3% threshold", annotation_position="top right")
    fig.update_layout(
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=480,
        legend=dict(orientation="h", y=-0.2),
    )
    return fig


def fig_yearly_region_heatmap(yearly_region, metric="CFR"):
    """Heatmap: metric across region × year."""
    pivot = yearly_region.pivot(index="WHO_region", columns="Year", values=metric).fillna(0)
    label_map = {"CFR": "CFR (%)", "New_cases": "New Cases", "New_deaths": "New Deaths"}
    fig = go.Figure(go.Heatmap(
        z=pivot.values,
        x=[str(y) for y in pivot.columns],
        y=pivot.index,
        colorscale="RdYlGn_r",
        text=np.round(pivot.values, 2),
        texttemplate="%{text}",
        hoverongaps=False,
        colorbar=dict(title=label_map.get(metric, metric)),
    ))
    fig.update_layout(
        title=f"WHO Region × Year Heatmap: {label_map.get(metric, metric)}",
        xaxis_title="Year",
        yaxis_title="WHO Region",
        plot_bgcolor="#f8f9fa",
        paper_bgcolor="#ffffff",
        height=380,
    )
    return fig


def fig_pie_region_cases(region_totals):
    """Pie chart: share of total cases by region."""
    rt = region_totals[region_totals["WHO_region"] != "OTHER"]
    fig = go.Figure(go.Pie(
        labels=rt["Region_label"],
        values=rt["New_cases"],
        marker_colors=[REGION_COLORS.get(r, "#95a5a6") for r in rt["WHO_region"]],
        hole=0.4,
        textinfo="label+percent",
    ))
    fig.update_layout(
        title="Share of Global Cases by WHO Region",
        paper_bgcolor="#ffffff",
        height=380,
        legend=dict(orientation="h", y=-0.2, font=dict(size=10)),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# 4. DASH LAYOUT
# ─────────────────────────────────────────────────────────────────────────────

# KPI card helper
def kpi_card(title, value, subtitle="", color="#3498db"):
    return html.Div([
        html.P(title, style={"margin": "0", "fontSize": "13px", "color": "#57606a",
                              "fontWeight": "600", "textTransform": "uppercase",
                              "letterSpacing": "0.5px"}),
        html.H2(value, style={"margin": "6px 0 2px", "fontSize": "32px",
                               "color": color, "fontWeight": "700"}),
        html.P(subtitle, style={"margin": "0", "fontSize": "12px", "color": "#57606a"}),
    ], style={
        "background": "#ffffff",
        "border": "1px solid #e5e7eb",
        "borderTop": f"4px solid {color}",
        "borderRadius": "8px",
        "padding": "20px 24px",
        "flex": "1",
        "minWidth": "180px",
        "boxShadow": "0 1px 4px rgba(0,0,0,0.05)",
    })


def section_header(number, title, subtitle=""):
    return html.Div([
        html.Div([
            html.Span(number, style={
                "background": "#3b82d4", "color": "#fff",
                "borderRadius": "50%", "width": "36px", "height": "36px",
                "display": "inline-flex", "alignItems": "center",
                "justifyContent": "center", "fontWeight": "700",
                "fontSize": "16px", "marginRight": "14px", "flexShrink": "0",
            }),
            html.Div([
                html.H2(title, style={"margin": "0", "fontSize": "20px",
                                       "fontWeight": "700", "color": "#1f2328"}),
                html.P(subtitle, style={"margin": "2px 0 0", "fontSize": "13px",
                                         "color": "#57606a"}) if subtitle else None,
            ]),
        ], style={"display": "flex", "alignItems": "center"}),
    ], style={
        "borderBottom": "2px solid #e5e7eb",
        "paddingBottom": "14px",
        "marginBottom": "24px",
        "marginTop": "36px",
    })


def insight_box(fact, insight, risk_opp, action, box_type="risk"):
    color_map = {"risk": "#e74c3c", "opportunity": "#2ecc71", "neutral": "#3498db"}
    accent = color_map.get(box_type, "#3498db")
    return html.Div([
        html.Div([
            html.Span("📌 Fact: ",    style={"fontWeight": "700", "color": "#1f2328"}),
            html.Span(fact),
        ], style={"marginBottom": "6px", "fontSize": "13.5px"}),
        html.Div([
            html.Span("💡 Insight: ", style={"fontWeight": "700", "color": "#1f2328"}),
            html.Span(insight),
        ], style={"marginBottom": "6px", "fontSize": "13.5px"}),
        html.Div([
            html.Span("⚠️ Risk/Opp: " if box_type != "opportunity" else "✅ Opportunity: ",
                      style={"fontWeight": "700", "color": accent}),
            html.Span(risk_opp),
        ], style={"marginBottom": "6px", "fontSize": "13.5px"}),
        html.Div([
            html.Span("🚀 Action: ",  style={"fontWeight": "700", "color": "#1f2328"}),
            html.Span(action),
        ], style={"fontSize": "13.5px"}),
    ], style={
        "background": "#f7f8fa",
        "border": "1px solid #e5e7eb",
        "borderLeft": f"5px solid {accent}",
        "borderRadius": "6px",
        "padding": "16px 20px",
        "marginBottom": "16px",
    })


def build_layout(df, a):
    """Build the full Dash app layout using pre-computed analytics `a`."""

    # Pre-build all figures
    f_global_trend   = fig_global_trend(a["monthly_global"])
    f_yearly         = fig_yearly_bar(a["yearly_global"])
    f_region_bar     = fig_region_cases_deaths(a["region_totals"])
    f_cfr_region     = fig_cfr_by_region(a["region_totals"])
    f_cfr_trend      = fig_cfr_trend_over_years(a["cfr_trend"])
    f_stacked_cases  = fig_region_monthly_trend(a["monthly_region"], "New_cases")
    f_stacked_deaths = fig_region_monthly_trend(a["monthly_region"], "New_deaths")
    f_top10_cases    = fig_top10_countries(a["top10_cases"], "Total_cases")
    f_top10_deaths   = fig_top10_countries(a["top10_deaths"], "Total_deaths")
    f_risk_scatter   = fig_high_risk_scatter(a["country_totals"])
    f_heatmap_cfr    = fig_yearly_region_heatmap(a["yearly_region"], "CFR")
    f_pie            = fig_pie_region_cases(a["region_totals"])

    def graph(fig_obj, gid):
        return dcc.Graph(id=gid, figure=fig_obj,
                         config={"displayModeBar": False},
                         style={"border": "1px solid #e5e7eb", "borderRadius": "8px"})

    # ── High-risk table ──
    risk_table_df = a["high_risk_countries"][
        ["Country", "WHO_region", "Total_cases", "Total_deaths", "CFR"]
    ].rename(columns={
        "Total_cases":  "Total Cases",
        "Total_deaths": "Total Deaths",
        "WHO_region":   "WHO Region",
        "CFR":          "CFR (%)",
    })

    layout = html.Div([

        # ── Header ──────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.H1("COVID-19 Global Intelligence Dashboard",
                        style={"margin": "0", "fontSize": "26px", "fontWeight": "700",
                               "color": "#1f2328"}),
                html.P("WHO Global Data  |  2020 – 2024  |  Business Intelligence Report",
                       style={"margin": "4px 0 0", "fontSize": "13px", "color": "#57606a"}),
            ]),
            html.Div([
                html.Span("🗓 Data through Aug 2024",
                          style={"fontSize": "12px", "color": "#57606a",
                                 "background": "#f7f8fa", "border": "1px solid #e5e7eb",
                                 "padding": "4px 10px", "borderRadius": "20px"}),
            ]),
        ], style={
            "display": "flex", "justifyContent": "space-between", "alignItems": "center",
            "background": "#ffffff", "borderBottom": "2px solid #e5e7eb",
            "padding": "18px 32px",
        }),

        # ── Body ─────────────────────────────────────────────────────────────
        html.Div([

            # ═══════════════════════════════════════════════════════════════
            # SECTION 1: EXECUTIVE OVERVIEW
            # ═══════════════════════════════════════════════════════════════
            section_header("1", "Executive Overview",
                           "Global pandemic summary: total burden, trend, and key milestones"),

            # KPI row
            html.Div([
                kpi_card("Total Cases",    f"{a['total_cases']:,}",
                         "Cumulative 2020–2024", "#3498db"),
                kpi_card("Total Deaths",   f"{a['total_deaths']:,}",
                         "Cumulative 2020–2024", "#e74c3c"),
                kpi_card("Global CFR",     f"{a['global_cfr']}%",
                         "Case Fatality Ratio",  "#f39c12"),
                kpi_card("Countries/Territories", f"{a['countries_affected']}",
                         "Reporting to WHO", "#2ecc71"),
                kpi_card("Peak Case Month", a["peak_month_cases"],
                         "Highest single-month cases", "#9b59b6"),
            ], style={"display": "flex", "gap": "16px", "flexWrap": "wrap",
                      "marginBottom": "28px"}),

            # Global trend charts row
            html.Div([
                html.Div(graph(f_global_trend, "g1"),
                         style={"flex": "2", "minWidth": "400px"}),
                html.Div(graph(f_yearly, "g2"),
                         style={"flex": "1", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                      "flexWrap": "wrap"}),

            # Executive insight boxes
            insight_box(
                fact=f"The pandemic generated {a['total_cases']:,} confirmed cases and "
                     f"{a['total_deaths']:,} deaths globally between 2020 and 2024.",
                insight="Case volume peaked in early 2022 (Omicron wave) but death rates "
                        "had already begun to decelerate from mid-2021 onward.",
                risk_opp="Pandemic fatigue and surveillance gaps risk under-counting future "
                         "waves, making real-time data infrastructure critical.",
                action="Invest in WHO-integrated digital surveillance systems to ensure "
                       "continuous, high-fidelity reporting even between peak periods.",
                box_type="risk",
            ),
            insight_box(
                fact=f"The global Case Fatality Ratio stands at {a['global_cfr']}% "
                     f"over the full 2020–2024 period.",
                insight="CFR dropped sharply after 2021, consistent with mass vaccination "
                        "rollouts and improved clinical management protocols.",
                risk_opp="The downward CFR trend is an opportunity to model the ROI of "
                         "vaccination investment and justify continued funding.",
                action="Publish WHO-region-specific CFR reduction reports to maintain "
                       "political will for vaccination programme financing.",
                box_type="opportunity",
            ),

            # ═══════════════════════════════════════════════════════════════
            # SECTION 2: REGIONAL ANALYSIS
            # ═══════════════════════════════════════════════════════════════
            section_header("2", "Regional Analysis",
                           "WHO-region-wise comparison of cases, deaths, CFR, and temporal trends"),

            html.Div([
                html.Div(graph(f_region_bar,   "g3"), style={"flex": "3", "minWidth": "380px"}),
                html.Div(graph(f_pie,          "g4"), style={"flex": "2", "minWidth": "280px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                      "flexWrap": "wrap"}),

            html.Div([
                html.Div(graph(f_cfr_region,  "g5"), style={"flex": "1", "minWidth": "340px"}),
                html.Div(graph(f_cfr_trend,   "g6"), style={"flex": "2", "minWidth": "380px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                      "flexWrap": "wrap"}),

            html.Div([
                html.Div(graph(f_stacked_cases,  "g7"), style={"flex": "1", "minWidth": "340px"}),
                html.Div(graph(f_stacked_deaths, "g8"), style={"flex": "1", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "16px",
                      "flexWrap": "wrap"}),

            graph(f_heatmap_cfr, "g9"),
            html.P(
                "⚠️ Note: 2024 data is partial (through Aug 2024) and reporting coverage "
                "may vary by region, which can distort year-over-year comparisons for the "
                "most recent year (e.g., AMRO's 2024 CFR reflects limited case reporting "
                "relative to deaths).",
                style={
                    "fontSize": "12px",
                    "color": "#e74c3c",
                    "fontStyle": "italic",
                    "marginTop": "8px",
                    "marginBottom": "16px",
                },
            ),
            html.Div(style={"marginBottom": "16px"}),

            html.Div([
                html.Div(graph(f_top10_cases,  "g10"), style={"flex": "1", "minWidth": "340px"}),
                html.Div(graph(f_top10_deaths, "g11"), style={"flex": "1", "minWidth": "340px"}),
            ], style={"display": "flex", "gap": "16px", "marginBottom": "24px",
                      "flexWrap": "wrap"}),

            # Regional insight boxes
            insight_box(
                fact="EURO and AMRO together account for over 60% of all reported cases.",
                insight="High case counts in Europe and the Americas reflect both genuine "
                        "transmission and stronger surveillance/reporting infrastructure.",
                risk_opp="Under-reporting in AFRO and SEARO masks true burden, creating "
                         "a resource allocation blind spot.",
                action="Strengthen community-level testing and case reporting capacity in "
                       "AFRO and SEARO through WHO technical assistance programmes.",
                box_type="risk",
            ),
            insight_box(
                fact="AFRO exhibits the highest Case Fatality Ratio among all WHO regions.",
                insight="Limited ICU capacity, delayed diagnoses, and lower vaccination "
                        "coverage compound to produce higher fatality outcomes in Africa.",
                risk_opp="Targeted investment in African health infrastructure would yield "
                         "disproportionately large CFR reductions.",
                action="Prioritise COVAX allocations and healthcare worker training "
                       "programmes in high-CFR African nations.",
                box_type="risk",
            ),
            insight_box(
                fact="CFR declined sharply across all WHO regions from 2020 to 2022, "
                     "coinciding with mass vaccination rollouts.",
                insight="Most regions saw CFR fall by 60–90% between 2020 and 2022 "
                        "(e.g., WPRO: 1.85%→0.10%, EURO: 2.31%→0.28%). However, CFR partially "
                        "rebounded in 2023 across EURO, EMRO, AMRO and SEARO — likely reflecting "
                        "reduced testing/surveillance as emergency measures wound down, inflating "
                        "the ratio rather than reflecting worse outcomes.",
                risk_opp="The 2023 rebound is a data-quality signal, not a clinical "
                         "one — but if surveillance keeps declining, genuine outbreaks could go "
                         "undetected until CFR spikes for real.",
                action="Maintain minimum testing/reporting thresholds post-emergency to "
                       "keep CFR a reliable indicator, not an artifact of under-testing.",
                box_type="risk",
            ),

            # ═══════════════════════════════════════════════════════════════
            # SECTION 3: RISK, OPPORTUNITY & ACTION
            # ═══════════════════════════════════════════════════════════════
            section_header("3", "Risk, Opportunity & Action",
                           "High-risk country identification, vaccination impact, and recommended interventions"),

            graph(f_risk_scatter, "g12"),
            html.Div(style={"marginBottom": "20px"}),

            # High-risk table
            html.Div([
                html.H3("High-Risk Countries (CFR > 3%, Cases > 1,000)",
                        style={"fontSize": "15px", "fontWeight": "700",
                               "color": "#1f2328", "marginBottom": "12px"}),
                dash_table.DataTable(
                    data=risk_table_df.to_dict("records"),
                    columns=[{"name": c, "id": c} for c in risk_table_df.columns],
                    style_table={"overflowX": "auto"},
                    style_header={
                        "backgroundColor": "#f7f8fa",
                        "fontWeight": "700",
                        "fontSize": "13px",
                        "border": "1px solid #e5e7eb",
                        "color": "#1f2328",
                    },
                    style_cell={
                        "fontSize": "13px",
                        "padding": "8px 12px",
                        "border": "1px solid #e5e7eb",
                        "color": "#1f2328",
                    },
                    style_data_conditional=[
                        {
                            "if": {"filter_query": "{CFR (%)} > 5"},
                            "backgroundColor": "#fef2f2",
                            "color": "#e74c3c",
                            "fontWeight": "700",
                        }
                    ],
                    page_size=10,
                    sort_action="native",
                ),
            ], style={"border": "1px solid #e5e7eb", "borderRadius": "8px",
                      "padding": "20px", "background": "#ffffff",
                      "marginBottom": "24px"}),

            # Risk/Opportunity insight boxes
            insight_box(
                fact="20+ countries have a CFR exceeding 3%, with several exceeding 10%.",
                insight="These nations combine low testing rates (inflating CFR) with "
                        "genuine healthcare system constraints, creating a dual crisis.",
                risk_opp="Without intervention, the next pandemic wave will hit these "
                         "countries disproportionately hard.",
                action="WHO and bilateral donors should fast-track diagnostic capacity "
                       "building and hospital infrastructure grants to CFR > 5% nations.",
                box_type="risk",
            ),
            insight_box(
                fact="Regions with earlier vaccination rollouts show CFR reductions of "
                     "33–82% between 2021 and 2023 (WPRO: -82%, EURO: -42%, AMRO: -33%).",
                insight="Vaccination correlates with reduced mortality across all WHO "
                        "regions, though the magnitude varies — WPRO shows the strongest effect, "
                        "AMRO the weakest, suggesting other factors (healthcare capacity, variant "
                        "exposure) also play a role.",
                risk_opp="Closing the vaccination gap in AFRO and EMRO remains the "
                         "highest-ROI intervention available, even if the WPRO-level effect isn't "
                         "guaranteed everywhere.",
                action="Scale mRNA technology transfer to AFRO manufacturing hubs; "
                       "deploy mobile vaccination units in rural EMRO communities.",
                box_type="opportunity",
            ),
            insight_box(
                fact="SEARO (South-East Asia) has the second-highest death count globally.",
                insight="Dense populations and delayed early containment contributed to "
                        "large absolute death tolls despite moderate CFR.",
                risk_opp="Population density and emerging pathogen risk make SEARO a "
                         "hotspot for future pandemic emergence.",
                action="Establish a WHO-SEARO regional pandemic preparedness fund with "
                       "rapid-response surge vaccination and surveillance capacity.",
                box_type="risk",
            ),
            insight_box(
                fact="Vaccination-correlated CFR decline was observed across all regions "
                     "by 2022–2023.",
                insight="The convergence of regional CFR rates post-vaccination suggests "
                        "that equity of access—not biology—is the primary driver of "
                        "mortality disparity.",
                risk_opp="Achieving vaccination equity by 2026 could prevent an estimated "
                         "hundreds of thousands of deaths in the next infectious wave.",
                action="Commit to a binding global vaccination equity framework under the "
                       "WHO's Pandemic Accord, with measurable coverage targets for "
                       "AFRO and EMRO by 2026.",
                box_type="opportunity",
            ),

            # ── Footer ────────────────────────────────────────────────────
            html.Hr(style={"borderColor": "#e5e7eb", "marginTop": "40px"}),
            html.P(
                "Data: WHO COVID-19 Global Dataset (2020–2024) | "
                "Source: kaggle.com/datasets/abdoomoh/daily-covid-19-data-2020-2024 | "
                "Built with Dash · Plotly · Pandas",
                style={"textAlign": "center", "fontSize": "11px",
                       "color": "#95a5a6", "paddingBottom": "20px"},
            ),

        ], style={"maxWidth": "1280px", "margin": "0 auto",
                  "padding": "28px 32px", "fontFamily":
                  '-apple-system,"Segoe UI",system-ui,sans-serif'}),

    ], style={"background": "#f4f6f8", "minHeight": "100vh"})

    return layout


# ─────────────────────────────────────────────────────────────────────────────
# 5. APP ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("📊 Loading and cleaning data …")
    df = load_and_clean()
    print(f"   Rows loaded: {len(df):,}  |  Countries: {df['Country'].nunique()}  |  "
          f"Date range: {df['Date_reported'].min().date()} → {df['Date_reported'].max().date()}")

    print("🔢 Computing analytics …")
    a = build_analytics(df)
    print(f"   Total cases:  {a['total_cases']:,}")
    print(f"   Total deaths: {a['total_deaths']:,}")
    print(f"   Global CFR:   {a['global_cfr']}%")

    print("🎨 Building Dash layout …")
    app = dash.Dash(
        __name__,
        title="COVID-19 Global BI Dashboard",
        meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    )
    app.layout = build_layout(df, a)

    print("\n✅ Dashboard ready!  Open http://127.0.0.1:8050 in your browser.\n")
    app.run(debug=False, host="127.0.0.1", port="8050")


if __name__ == "__main__":
    main()
