"""
app.py
======
Indian Flight Price Analysis — Streamlit Dashboard
===================================================
A self-contained, professional data analytics dashboard built on the
Indian flight price dataset.  No external project modules required.

Run with:
    streamlit run app.py
"""

import os
import io
import re
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


# ============================================================================
# CONSTANTS
# ============================================================================

STOPS_LABEL_MAP = {
    0: "0 Stops (Non-stop)",
    1: "1 Stop",
    2: "2 Stops",
    3: "3 Stops",
    4: "4 Stops",
}

MONTH_NAME_MAP = {
    1: "January", 2: "February", 3: "March", 4: "April",
    5: "May", 6: "June", 7: "July", 8: "August",
    9: "September", 10: "October", 11: "November", 12: "December",
}

DAY_NAME_MAP = {
    0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday",
    4: "Friday", 5: "Saturday", 6: "Sunday",
}


# ============================================================================
# DATA PROCESSING — load, clean, feature-engineer, filter
# ============================================================================

def parse_duration_to_minutes(duration_str: str) -> float:
    """Convert '2h 50m', '19h', '45m' → total minutes.  NaN on failure."""
    if not isinstance(duration_str, str):
        return np.nan
    duration_str = duration_str.strip()
    match = re.fullmatch(r'(?:(\d+)h)?\s*(?:(\d+)m)?', duration_str)
    if match and (match.group(1) or match.group(2)):
        hours = int(match.group(1)) if match.group(1) else 0
        minutes = int(match.group(2)) if match.group(2) else 0
        return hours * 60 + minutes
    return np.nan


def classify_departure_period(hour: int) -> str:
    """
    Map an hour (0-23) to a departure period label.
    Early Morning 0-5 | Morning 6-11 | Afternoon 12-16 | Evening 17-20 | Night 21-23
    """
    if 0 <= hour <= 5:
        return "Early Morning"
    elif 6 <= hour <= 11:
        return "Morning"
    elif 12 <= hour <= 16:
        return "Afternoon"
    elif 17 <= hour <= 20:
        return "Evening"
    else:
        return "Night"


def load_raw_data(filepath: str):
    """Load CSV; return None on any error so the app can show a friendly message."""
    if not os.path.isfile(filepath):
        return None
    try:
        return pd.read_csv(filepath)
    except Exception:
        return None


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and validate the raw dataframe.
    - Strip whitespace from string columns
    - Drop full duplicates
    - Parse Date_of_Journey (DD/MM/YYYY → datetime)
    - Coerce Price and Total_Stops to numeric; drop unparsable rows
    Original raw data is never modified (operates on a copy).
    """
    df = df.copy()

    # Strip leading/trailing whitespace from all string columns
    for col in df.select_dtypes(include=["object", "string"]).columns:
        df[col] = df[col].str.strip()

    # Drop full duplicate rows
    df.drop_duplicates(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Parse journey date
    df["Date_of_Journey"] = pd.to_datetime(
        df["Date_of_Journey"], format="%d/%m/%Y", errors="coerce"
    )
    df.dropna(subset=["Date_of_Journey"], inplace=True)

    # Ensure Price is integer
    df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
    df.dropna(subset=["Price"], inplace=True)
    df["Price"] = df["Price"].astype(int)

    # Ensure Total_Stops is integer
    df["Total_Stops"] = pd.to_numeric(df["Total_Stops"], errors="coerce")
    df.dropna(subset=["Total_Stops"], inplace=True)
    df["Total_Stops"] = df["Total_Stops"].astype(int)

    df.reset_index(drop=True, inplace=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add derived analytical columns to the cleaned dataframe.
    New columns: Journey_Year, Journey_Month, Journey_Month_Name,
    Journey_Day, Journey_Day_Name, Departure_Hour, Departure_Period,
    Arrival_Hour, Duration_Minutes, Stops_Label, Route_Short.
    The original Duration column is kept intact.
    """
    df = df.copy()

    # Date features
    df["Journey_Year"]       = df["Date_of_Journey"].dt.year
    df["Journey_Month"]      = df["Date_of_Journey"].dt.month
    df["Journey_Month_Name"] = df["Journey_Month"].map(MONTH_NAME_MAP)
    df["Journey_Day"]        = df["Date_of_Journey"].dt.day
    df["Journey_Day_Name"]   = df["Date_of_Journey"].dt.dayofweek.map(DAY_NAME_MAP)

    # Departure time features
    df["Departure_Hour"] = (
        df["Dep_Time"].str.extract(r'^(\d{1,2}):').iloc[:, 0]
        .astype(float).astype("Int64")
    )
    df["Departure_Period"] = df["Departure_Hour"].apply(
        lambda h: classify_departure_period(int(h)) if pd.notna(h) else "Unknown"
    )

    # Arrival time features
    df["Arrival_Hour"] = (
        df["Arrival_Time"].str.extract(r'^(\d{1,2}):').iloc[:, 0]
        .astype(float).astype("Int64")
    )

    # Duration in total minutes (original Duration column unchanged)
    df["Duration_Minutes"] = df["Duration"].apply(parse_duration_to_minutes)

    # Human-readable stop labels
    df["Stops_Label"] = df["Total_Stops"].map(STOPS_LABEL_MAP).fillna("Unknown")

    # Short route label for charts
    df["Route_Short"] = df["Source"] + " → " + df["Destination"]

    return df


def get_processed_data(filepath: str):
    """
    Full pipeline: load → clean → engineer features.
    Returns (dataframe, error_string).  On success error_string is "".
    """
    raw = load_raw_data(filepath)
    if raw is None:
        return None, (
            f"❌ Dataset not found at `{filepath}`.  "
            "Please ensure the file exists in the `data/` folder."
        )
    try:
        cleaned   = clean_data(raw)
        processed = engineer_features(cleaned)
        return processed, ""
    except Exception as exc:
        return None, f"❌ Error processing data: {exc}"


def apply_filters(
    df: pd.DataFrame,
    airlines: list,
    sources: list,
    destinations: list,
    stops: list,
    months: list,
    periods: list,
) -> pd.DataFrame:
    """
    Return a filtered copy of df based on sidebar selections.
    An empty list for any parameter means 'select all'.
    """
    mask = pd.Series([True] * len(df), index=df.index)
    if airlines:     mask &= df["Airline"].isin(airlines)
    if sources:      mask &= df["Source"].isin(sources)
    if destinations: mask &= df["Destination"].isin(destinations)
    if stops:        mask &= df["Total_Stops"].isin(stops)
    if months:       mask &= df["Journey_Month"].isin(months)
    if periods:      mask &= df["Departure_Period"].isin(periods)
    return df[mask].copy()

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Indian Flight Price Analysis",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# CUSTOM CSS — clean, professional, readable
# ============================================================================
st.markdown(
    """
    <style>
        /* Main background */
        .block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

        /* KPI card */
        .kpi-card {
            background: #f7f8fa;
            border: 1px solid #e5e7eb;
            border-radius: 10px;
            padding: 18px 20px;
            text-align: center;
        }
        .kpi-label {
            font-size: 13px;
            color: #57606a;
            font-weight: 500;
            margin-bottom: 6px;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .kpi-value {
            font-size: 26px;
            font-weight: 700;
            color: #1f2328;
            line-height: 1.2;
        }
        .kpi-sub {
            font-size: 12px;
            color: #8b949e;
            margin-top: 4px;
        }

        /* Section heading */
        .section-title {
            font-size: 18px;
            font-weight: 600;
            color: #1f2328;
            border-left: 4px solid #3b82d4;
            padding-left: 10px;
            margin-top: 10px;
            margin-bottom: 6px;
        }

        /* Insight box */
        .insight-box {
            background: #f0f6ff;
            border: 1px solid #c8deff;
            border-radius: 8px;
            padding: 14px 18px;
            margin-bottom: 8px;
            font-size: 14px;
            color: #1f2328;
            line-height: 1.6;
        }

        /* Remove Streamlit branding footer */
        footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# HELPERS
# ============================================================================

def fmt_inr(value: float) -> str:
    """Format a number as Indian Rupees with ₹ symbol and comma separator."""
    return f"₹{int(round(value)):,}"


def fmt_minutes(minutes: float) -> str:
    """Convert total minutes to 'Xh Ym' string."""
    if pd.isna(minutes):
        return "N/A"
    h = int(minutes) // 60
    m = int(minutes) % 60
    return f"{h}h {m}m" if m else f"{h}h"


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="kpi-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="kpi-card">'
        f'  <div class="kpi-label">{label}</div>'
        f'  <div class="kpi-value">{value}</div>'
        f'  {sub_html}'
        f'</div>'
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def insight_box(text: str):
    st.markdown(f'<div class="insight-box">💡 {text}</div>', unsafe_allow_html=True)


# ── Plotly theme defaults ────────────────────────────────────────────────────
CHART_COLORS = px.colors.qualitative.Set2
CHART_HEIGHT = 420
CHART_MARGIN = dict(l=60, r=30, t=50, b=60)

PLOTLY_LAYOUT = dict(
    margin=CHART_MARGIN,
    paper_bgcolor="#ffffff",
    plot_bgcolor="#f7f8fa",
    font=dict(family="-apple-system, Segoe UI, sans-serif", size=13, color="#1f2328"),
    title_font_size=15,
    hoverlabel=dict(bgcolor="white", font_size=13),
)


def apply_layout(fig, **kwargs):
    # height defaults to CHART_HEIGHT but can be overridden by callers
    kwargs.setdefault("height", CHART_HEIGHT)
    fig.update_layout(**PLOTLY_LAYOUT, **kwargs)
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb", zeroline=False)
    return fig


# ============================================================================
# DATA LOADING (cached)
# ============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    """Load and process the dataset once; cache the result."""
    base_dir = os.path.dirname(__file__)
    filepath = os.path.join(base_dir, "data", "Indian Flight Price.csv")
    df, error = get_processed_data(filepath)
    return df, error


# ============================================================================
# SIDEBAR FILTERS
# ============================================================================

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """Render sidebar filters and return the filtered dataframe."""

    st.sidebar.markdown(
        """
        <div style="
            background: linear-gradient(135deg, #1a3a5c 0%, #2563a8 60%, #3b82d4 100%);
            border-radius: 10px;
            padding: 18px 12px 14px 12px;
            text-align: center;
            margin-bottom: 4px;
        ">
            <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64"
                 width="52" height="52" style="display:block;margin:0 auto 8px auto;">
                <!-- fuselage -->
                <ellipse cx="32" cy="32" rx="22" ry="7" fill="#e8f0fe" transform="rotate(-30 32 32)"/>
                <!-- left wing -->
                <polygon points="32,32 10,44 18,34" fill="#93c5fd"/>
                <!-- right wing -->
                <polygon points="32,32 54,20 46,30" fill="#93c5fd"/>
                <!-- tail fin -->
                <polygon points="32,32 28,18 36,22" fill="#60a5fa"/>
                <!-- cockpit window -->
                <ellipse cx="44" cy="24" rx="3" ry="2"
                         fill="#bfdbfe" transform="rotate(-30 44 24)"/>
            </svg>
            <div style="color:#ffffff;font-size:17px;font-weight:700;letter-spacing:0.03em;
                        line-height:1.3;">Indian Flight<br>Price Analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.title("✈️ Filters")
    st.sidebar.markdown("---")

    # ── Airline ──────────────────────────────────────────────────────────────
    all_airlines = sorted(df["Airline"].unique())
    selected_airlines = st.sidebar.multiselect(
        "Airline",
        options=all_airlines,
        default=[],
        placeholder="All Airlines",
    )

    # ── Source ───────────────────────────────────────────────────────────────
    all_sources = sorted(df["Source"].unique())
    selected_sources = st.sidebar.multiselect(
        "Source City",
        options=all_sources,
        default=[],
        placeholder="All Sources",
    )

    # ── Destination ──────────────────────────────────────────────────────────
    all_destinations = sorted(df["Destination"].unique())
    selected_destinations = st.sidebar.multiselect(
        "Destination",
        options=all_destinations,
        default=[],
        placeholder="All Destinations",
    )

    # ── Stops ────────────────────────────────────────────────────────────────
    all_stops = sorted(df["Total_Stops"].unique())
    stop_labels = {k: STOPS_LABEL_MAP.get(k, str(k)) for k in all_stops}
    selected_stop_labels = st.sidebar.multiselect(
        "Number of Stops",
        options=list(stop_labels.values()),
        default=[],
        placeholder="All Stop Counts",
    )
    label_to_num = {v: k for k, v in stop_labels.items()}
    selected_stops = [label_to_num[l] for l in selected_stop_labels]

    # ── Month ────────────────────────────────────────────────────────────────
    all_months = sorted(df["Journey_Month"].unique())
    month_options = [MONTH_NAME_MAP[m] for m in all_months]
    selected_month_names = st.sidebar.multiselect(
        "Journey Month",
        options=month_options,
        default=[],
        placeholder="All Months",
    )
    name_to_num = {v: k for k, v in MONTH_NAME_MAP.items()}
    selected_months = [name_to_num[n] for n in selected_month_names]

    # ── Departure Period ─────────────────────────────────────────────────────
    period_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
    all_periods = [p for p in period_order if p in df["Departure_Period"].unique()]
    selected_periods = st.sidebar.multiselect(
        "Departure Period",
        options=all_periods,
        default=[],
        placeholder="All Periods",
    )

    # ── Apply filters ─────────────────────────────────────────────────────────
    filtered = apply_filters(
        df,
        airlines=selected_airlines,
        sources=selected_sources,
        destinations=selected_destinations,
        stops=selected_stops,
        months=selected_months,
        periods=selected_periods,
    )

    st.sidebar.markdown("---")
    st.sidebar.metric("Flights Shown", f"{len(filtered):,}")
    st.sidebar.caption(
        "Select one or more options to filter. Leave blank to include all."
    )

    return filtered


# ============================================================================
# TAB 1 — OVERVIEW
# ============================================================================

def tab_overview(df: pd.DataFrame):
    st.markdown("### 📊 Dashboard Overview")
    st.caption(
        "Key performance indicators and top-level trends for the filtered dataset."
    )

    # ── KPI Cards ─────────────────────────────────────────────────────────────
    total_flights = len(df)
    avg_price = df["Price"].mean()
    min_price = df["Price"].min()
    max_price = df["Price"].max()
    median_price = df["Price"].median()
    avg_duration = df["Duration_Minutes"].mean()
    avg_stops = df["Total_Stops"].mean()

    cols = st.columns(6)
    cards = [
        ("Total Flights", f"{total_flights:,}", "records in selection"),
        ("Avg Price", fmt_inr(avg_price), f"Median: {fmt_inr(median_price)}"),
        ("Min Price", fmt_inr(min_price), "lowest fare"),
        ("Max Price", fmt_inr(max_price), "highest fare"),
        ("Avg Duration", fmt_minutes(avg_duration), "per flight"),
        ("Avg Stops", f"{avg_stops:.1f}", "stops per flight"),
    ]
    for col, (label, value, sub) in zip(cols, cards):
        col.markdown(kpi_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Avg Price by Airline ──────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        section_title("Average Price by Airline")
        airline_avg = (
            df.groupby("Airline")["Price"]
            .mean()
            .reset_index()
            .sort_values("Price", ascending=True)
        )
        airline_avg["Price_fmt"] = airline_avg["Price"].apply(fmt_inr)
        fig = px.bar(
            airline_avg,
            x="Price",
            y="Airline",
            orientation="h",
            text="Price_fmt",
            color="Airline",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"Price": "Avg Price (₹)", "Airline": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(showlegend=False)
        apply_layout(fig, title="Average Ticket Price by Airline")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Average Price by Month")
        # Use only months present in the filtered data
        month_avg = (
            df.groupby(["Journey_Month", "Journey_Month_Name"])["Price"]
            .mean()
            .reset_index()
            .sort_values("Journey_Month")
        )
        month_avg["Price_fmt"] = month_avg["Price"].apply(fmt_inr)
        fig2 = px.line(
            month_avg,
            x="Journey_Month_Name",
            y="Price",
            markers=True,
            text="Price_fmt",
            labels={"Journey_Month_Name": "Month", "Price": "Avg Price (₹)"},
        )
        fig2.update_traces(
            line_color="#3b82d4",
            marker=dict(size=8, color="#3b82d4"),
            textposition="top center",
            textfont_size=11,
        )
        apply_layout(fig2, title="Average Ticket Price by Month")
        st.plotly_chart(fig2, use_container_width=True)


# ============================================================================
# TAB 2 — AIRLINE ANALYSIS
# ============================================================================

def tab_airline(df: pd.DataFrame):
    st.markdown("### ✈️ Airline Analysis")

    col1, col2 = st.columns(2)

    with col1:
        section_title("Number of Flights by Airline")
        counts = (
            df["Airline"].value_counts().reset_index()
        )
        counts.columns = ["Airline", "Count"]
        fig = px.bar(
            counts,
            x="Count",
            y="Airline",
            orientation="h",
            text="Count",
            color="Airline",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"Count": "Number of Flights", "Airline": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(showlegend=False)
        apply_layout(fig, title="Flight Count by Airline")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Average Price by Airline")
        airline_avg = (
            df.groupby("Airline")["Price"]
            .mean()
            .reset_index()
            .sort_values("Price", ascending=False)
        )
        airline_avg["Price_fmt"] = airline_avg["Price"].apply(fmt_inr)
        fig2 = px.bar(
            airline_avg,
            x="Airline",
            y="Price",
            text="Price_fmt",
            color="Airline",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"Price": "Avg Price (₹)", "Airline": ""},
        )
        fig2.update_traces(textposition="outside", textfont_size=11)
        fig2.update_layout(showlegend=False)
        fig2.update_xaxes(tickangle=-30)
        apply_layout(fig2, title="Average Ticket Price by Airline")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Price Distribution by Airline (box plot) ─────────────────────────────
    section_title("Price Distribution by Airline")
    airline_order = (
        df.groupby("Airline")["Price"].median().sort_values(ascending=False).index.tolist()
    )
    fig3 = px.box(
        df,
        x="Airline",
        y="Price",
        color="Airline",
        category_orders={"Airline": airline_order},
        labels={"Price": "Ticket Price (₹)", "Airline": ""},
        color_discrete_sequence=CHART_COLORS,
    )
    fig3.update_xaxes(tickangle=-30)
    fig3.update_layout(showlegend=False)
    apply_layout(fig3, height=460, title="Price Distribution by Airline (Box Plot)")
    st.plotly_chart(fig3, use_container_width=True)

    # ── Avg Duration by Airline ───────────────────────────────────────────────
    section_title("Average Flight Duration by Airline")
    dur_avg = (
        df.groupby("Airline")["Duration_Minutes"]
        .mean()
        .reset_index()
        .sort_values("Duration_Minutes", ascending=False)
    )
    dur_avg["Duration_fmt"] = dur_avg["Duration_Minutes"].apply(fmt_minutes)
    fig4 = px.bar(
        dur_avg,
        x="Airline",
        y="Duration_Minutes",
        text="Duration_fmt",
        color="Airline",
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={"Duration_Minutes": "Avg Duration (mins)", "Airline": ""},
    )
    fig4.update_traces(textposition="outside", textfont_size=11)
    fig4.update_layout(showlegend=False)
    fig4.update_xaxes(tickangle=-30)
    apply_layout(fig4, title="Average Flight Duration by Airline")
    st.plotly_chart(fig4, use_container_width=True)


# ============================================================================
# TAB 3 — ROUTE ANALYSIS
# ============================================================================

def tab_route(df: pd.DataFrame):
    st.markdown("### 🗺️ Route Analysis")

    col1, col2 = st.columns(2)

    with col1:
        section_title("Top 10 Most Common Routes")
        top_routes = (
            df["Route_Short"].value_counts().head(10).reset_index()
        )
        top_routes.columns = ["Route", "Count"]
        fig = px.bar(
            top_routes.sort_values("Count"),
            x="Count",
            y="Route",
            orientation="h",
            text="Count",
            color="Route",
            color_discrete_sequence=px.colors.qualitative.Vivid,
            labels={"Count": "Number of Flights", "Route": ""},
        )
        fig.update_traces(textposition="outside", textfont_size=11)
        fig.update_layout(showlegend=False)
        apply_layout(fig, title="Top 10 Most Common Routes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Top 10 Most Expensive Routes (Avg Price)")
        exp_routes = (
            df.groupby("Route_Short")["Price"]
            .mean()
            .reset_index()
            .sort_values("Price", ascending=False)
            .head(10)
        )
        exp_routes["Price_fmt"] = exp_routes["Price"].apply(fmt_inr)
        fig2 = px.bar(
            exp_routes.sort_values("Price"),
            x="Price",
            y="Route_Short",
            orientation="h",
            text="Price_fmt",
            color="Route_Short",
            color_discrete_sequence=px.colors.qualitative.Alphabet,
            labels={"Price": "Avg Price (₹)", "Route_Short": ""},
        )
        fig2.update_traces(textposition="outside", textfont_size=11)
        fig2.update_layout(showlegend=False)
        apply_layout(fig2, title="Top 10 Most Expensive Routes")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Source & Destination ──────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        section_title("Average Price by Source City")
        src_avg = (
            df.groupby("Source")["Price"]
            .mean()
            .reset_index()
            .sort_values("Price", ascending=False)
        )
        src_avg["Price_fmt"] = src_avg["Price"].apply(fmt_inr)
        fig3 = px.bar(
            src_avg,
            x="Source",
            y="Price",
            text="Price_fmt",
            color="Source",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"Price": "Avg Price (₹)", "Source": "Source City"},
        )
        fig3.update_traces(textposition="outside", textfont_size=11)
        fig3.update_layout(showlegend=False)
        apply_layout(fig3, title="Average Price by Source City")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section_title("Average Price by Destination")
        dest_avg = (
            df.groupby("Destination")["Price"]
            .mean()
            .reset_index()
            .sort_values("Price", ascending=False)
        )
        dest_avg["Price_fmt"] = dest_avg["Price"].apply(fmt_inr)
        fig4 = px.bar(
            dest_avg,
            x="Destination",
            y="Price",
            text="Price_fmt",
            color="Destination",
            color_discrete_sequence=px.colors.qualitative.Safe,
            labels={"Price": "Avg Price (₹)", "Destination": "Destination"},
        )
        fig4.update_traces(textposition="outside", textfont_size=11)
        fig4.update_layout(showlegend=False)
        apply_layout(fig4, title="Average Price by Destination")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Most Common Source-Destination Pairs ──────────────────────────────────
    section_title("Most Common Source → Destination Combinations")
    pairs = (
        df.groupby(["Source", "Destination"])
        .size()
        .reset_index(name="Count")
        .sort_values("Count", ascending=False)
        .head(15)
    )
    pairs["Pair"] = pairs["Source"] + " → " + pairs["Destination"]
    fig5 = px.bar(
        pairs.sort_values("Count"),
        x="Count",
        y="Pair",
        orientation="h",
        text="Count",
        color="Pair",
        color_discrete_sequence=px.colors.qualitative.Alphabet,
        labels={"Count": "Number of Flights", "Pair": ""},
    )
    fig5.update_traces(textposition="outside", textfont_size=11)
    fig5.update_layout(showlegend=False)
    apply_layout(fig5, height=500, title="Top 15 Source → Destination Combinations")
    st.plotly_chart(fig5, use_container_width=True)


# ============================================================================
# TAB 4 — PRICE ANALYSIS
# ============================================================================

def tab_price(df: pd.DataFrame):
    st.markdown("### 💰 Price Analysis")

    col1, col2 = st.columns(2)

    with col1:
        section_title("Price Distribution")
        fig = px.histogram(
            df,
            x="Price",
            nbins=60,
            color_discrete_sequence=["#3b82d4"],
            labels={"Price": "Ticket Price (₹)", "count": "Number of Flights"},
        )
        fig.update_traces(marker_line_width=0.4, marker_line_color="white")
        apply_layout(fig, title="Flight Price Distribution")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        section_title("Average Price by Number of Stops")
        stops_order = [STOPS_LABEL_MAP[k] for k in sorted(STOPS_LABEL_MAP.keys()) if k in df["Total_Stops"].unique()]
        stops_avg = (
            df.groupby("Stops_Label")["Price"]
            .mean()
            .reset_index()
        )
        # Sort by the defined order
        stops_avg["_order"] = stops_avg["Stops_Label"].apply(
            lambda x: list(STOPS_LABEL_MAP.values()).index(x) if x in STOPS_LABEL_MAP.values() else 99
        )
        stops_avg = stops_avg.sort_values("_order")
        stops_avg["Price_fmt"] = stops_avg["Price"].apply(fmt_inr)
        fig2 = px.bar(
            stops_avg,
            x="Stops_Label",
            y="Price",
            text="Price_fmt",
            color="Stops_Label",
            color_discrete_sequence=px.colors.qualitative.Bold,
            labels={"Price": "Avg Price (₹)", "Stops_Label": "Number of Stops"},
        )
        fig2.update_traces(textposition="outside", textfont_size=11)
        fig2.update_layout(showlegend=False)
        apply_layout(fig2, title="Average Price by Number of Stops")
        st.plotly_chart(fig2, use_container_width=True)

    # ── Price by Month ────────────────────────────────────────────────────────
    col3, col4 = st.columns(2)

    with col3:
        section_title("Average Price by Month")
        month_avg = (
            df.groupby(["Journey_Month", "Journey_Month_Name"])["Price"]
            .mean()
            .reset_index()
            .sort_values("Journey_Month")
        )
        month_avg["Price_fmt"] = month_avg["Price"].apply(fmt_inr)
        fig3 = px.line(
            month_avg,
            x="Journey_Month_Name",
            y="Price",
            markers=True,
            text="Price_fmt",
            labels={"Journey_Month_Name": "Month", "Price": "Avg Price (₹)"},
        )
        fig3.update_traces(
            line_color="#7c5cd8",
            marker=dict(size=9, color="#7c5cd8"),
            textposition="top center",
            textfont_size=11,
        )
        apply_layout(fig3, title="Average Price by Month")
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        section_title("Average Price by Departure Period")
        period_order = ["Early Morning", "Morning", "Afternoon", "Evening", "Night"]
        period_avg = (
            df.groupby("Departure_Period")["Price"]
            .mean()
            .reset_index()
        )
        period_avg["_order"] = period_avg["Departure_Period"].apply(
            lambda x: period_order.index(x) if x in period_order else 99
        )
        period_avg = period_avg.sort_values("_order")
        period_avg["Price_fmt"] = period_avg["Price"].apply(fmt_inr)
        fig4 = px.bar(
            period_avg,
            x="Departure_Period",
            y="Price",
            text="Price_fmt",
            color="Departure_Period",
            color_discrete_sequence=CHART_COLORS,
            labels={"Price": "Avg Price (₹)", "Departure_Period": "Departure Period"},
        )
        fig4.update_traces(textposition="outside", textfont_size=11)
        fig4.update_layout(showlegend=False)
        apply_layout(fig4, title="Average Price by Departure Period")
        st.plotly_chart(fig4, use_container_width=True)

    # ── Duration vs Price scatter ─────────────────────────────────────────────
    section_title("Flight Duration vs Ticket Price")
    scatter_df = df.dropna(subset=["Duration_Minutes"]).copy()
    scatter_df["Duration_fmt"] = scatter_df["Duration_Minutes"].apply(fmt_minutes)
    fig5 = px.scatter(
        scatter_df,
        x="Duration_Minutes",
        y="Price",
        color="Airline",
        hover_data={"Airline": True, "Duration_fmt": True, "Price": True, "Duration_Minutes": False},
        opacity=0.65,
        labels={
            "Duration_Minutes": "Flight Duration (minutes)",
            "Price": "Ticket Price (₹)",
            "Duration_fmt": "Duration",
        },
        color_discrete_sequence=px.colors.qualitative.Pastel,
    )
    fig5.update_traces(marker=dict(size=6))
    apply_layout(fig5, height=480, title="Flight Duration vs Ticket Price")
    st.plotly_chart(fig5, use_container_width=True)

    # ── Avg Duration by Stops ─────────────────────────────────────────────────
    section_title("Average Duration by Number of Stops")
    dur_stops = (
        df.groupby("Stops_Label")["Duration_Minutes"]
        .mean()
        .reset_index()
    )
    dur_stops["_order"] = dur_stops["Stops_Label"].apply(
        lambda x: list(STOPS_LABEL_MAP.values()).index(x) if x in STOPS_LABEL_MAP.values() else 99
    )
    dur_stops = dur_stops.sort_values("_order")
    dur_stops["Duration_fmt"] = dur_stops["Duration_Minutes"].apply(fmt_minutes)
    fig6 = px.bar(
        dur_stops,
        x="Stops_Label",
        y="Duration_Minutes",
        text="Duration_fmt",
        color="Stops_Label",
        color_discrete_sequence=px.colors.qualitative.Safe,
        labels={"Duration_Minutes": "Avg Duration (mins)", "Stops_Label": "Number of Stops"},
    )
    fig6.update_traces(textposition="outside", textfont_size=11)
    fig6.update_layout(showlegend=False)
    apply_layout(fig6, title="Average Flight Duration by Number of Stops")
    st.plotly_chart(fig6, use_container_width=True)


# ============================================================================
# TAB 5 — DATA EXPLORER
# ============================================================================

def tab_data_explorer(df: pd.DataFrame):
    st.markdown("### 🔍 Data Explorer")
    st.caption(
        "Browse, search, and download the filtered dataset. "
        "All filters from the sidebar are already applied."
    )

    # ── Quick text search ─────────────────────────────────────────────────────
    search_query = st.text_input(
        "🔎 Search (searches Airline, Source, Destination, Route)",
        placeholder="e.g. IndiGo  or  Delhi  or  BLR → DEL",
    )

    display_df = df.copy()
    if search_query.strip():
        q = search_query.strip().lower()
        mask = (
            display_df["Airline"].str.lower().str.contains(q, na=False)
            | display_df["Source"].str.lower().str.contains(q, na=False)
            | display_df["Destination"].str.lower().str.contains(q, na=False)
            | display_df["Route"].str.lower().str.contains(q, na=False)
        )
        display_df = display_df[mask]

    # ── Column selection ──────────────────────────────────────────────────────
    default_cols = [
        "Airline", "Date_of_Journey", "Source", "Destination",
        "Dep_Time", "Arrival_Time", "Duration", "Total_Stops", "Price",
        "Departure_Period", "Journey_Month_Name",
    ]
    show_cols = st.multiselect(
        "Columns to display",
        options=list(df.columns),
        default=default_cols,
    )

    st.markdown(f"**{len(display_df):,} records** match current filters.")

    # ── Table ──────────────────────────────────────────────────────────────────
    show_df = display_df[show_cols] if show_cols else display_df
    st.dataframe(
        show_df.reset_index(drop=True),
        use_container_width=True,
        height=420,
    )

    # ── Download ──────────────────────────────────────────────────────────────
    st.markdown("---")
    csv_buffer = io.BytesIO()
    display_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue()

    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name="indian_flight_price_filtered.csv",
        mime="text/csv",
        help="Downloads all filtered records (all columns) as a CSV file.",
    )


# ============================================================================
# KEY BUSINESS INSIGHTS (dynamic)
# ============================================================================

def render_insights(df: pd.DataFrame):
    st.markdown("---")
    st.markdown("### 🔎 Key Business Insights")
    st.caption("Dynamically calculated from the currently filtered dataset.")

    if df.empty:
        st.warning("No data available for the current filters — insights cannot be generated.")
        return

    # Compute
    airline_avg   = df.groupby("Airline")["Price"].mean()
    best_airline  = airline_avg.idxmax()
    cheap_airline = airline_avg.idxmin()

    route_counts  = df["Route_Short"].value_counts()
    top_route     = route_counts.index[0]
    top_route_cnt = route_counts.iloc[0]

    route_avg     = df.groupby("Route_Short")["Price"].mean()
    exp_route     = route_avg.idxmax()
    exp_route_val = route_avg.max()

    month_avg     = df.groupby("Journey_Month_Name")["Price"].mean()
    peak_month    = month_avg.idxmax()
    off_month     = month_avg.idxmin()

    stops_avg     = df.groupby("Total_Stops")["Price"].mean()
    non_stop_avg  = stops_avg.get(0, None)
    one_stop_avg  = stops_avg.get(1, None)

    src_avg       = df.groupby("Source")["Price"].mean()
    top_src       = src_avg.idxmax()

    insights = [
        f"Among the filtered records, <b>{best_airline}</b> has the highest average ticket price "
        f"at <b>{fmt_inr(airline_avg[best_airline])}</b>.",

        f"<b>{cheap_airline}</b> offers the lowest average ticket price among the filtered airlines "
        f"at <b>{fmt_inr(airline_avg[cheap_airline])}</b>.",

        f"The most frequent route is <b>{top_route}</b> with <b>{top_route_cnt:,}</b> flight records.",

        f"The most expensive route (by average price) is <b>{exp_route}</b> "
        f"with an average fare of <b>{fmt_inr(exp_route_val)}</b>.",

        f"<b>{peak_month}</b> records the highest average ticket price "
        f"(<b>{fmt_inr(month_avg[peak_month])}</b>), "
        f"while <b>{off_month}</b> is the most affordable month "
        f"(<b>{fmt_inr(month_avg[off_month])}</b>).",

        (
            f"Non-stop flights average <b>{fmt_inr(non_stop_avg)}</b>, "
            f"while 1-stop flights average <b>{fmt_inr(one_stop_avg)}</b> — "
            f"adding one stop {'increases' if one_stop_avg > non_stop_avg else 'decreases'} "
            f"fares by <b>{fmt_inr(abs(one_stop_avg - non_stop_avg))}</b> on average."
            if (non_stop_avg is not None and one_stop_avg is not None) else
            "Insufficient data to compare non-stop and 1-stop prices for the current selection."
        ),

        f"Flights departing from <b>{top_src}</b> have the highest average ticket price "
        f"at <b>{fmt_inr(src_avg[top_src])}</b>.",
    ]

    for insight in insights:
        st.markdown(
            f'<div class="insight-box">💡 {insight}</div>',
            unsafe_allow_html=True,
        )


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # ── Header ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding: 10px 0 4px 0;">
            <h1 style="font-size:2rem; font-weight:700; color:#1f2328; margin-bottom:4px;">
                ✈️ Indian Flight Price Analysis
            </h1>
            <p style="color:#57606a; font-size:15px; margin:0;">
                An interactive data analytics dashboard exploring airline fares,
                routes, and travel patterns across Indian airports.
            </p>
        </div>
        <hr style="border:none; border-top:1px solid #e5e7eb; margin: 12px 0 18px 0;">
        """,
        unsafe_allow_html=True,
    )

    # ── Load Data ─────────────────────────────────────────────────────────────
    with st.spinner("Loading and processing dataset …"):
        df, error = load_data()

    if df is None:
        st.error(error)
        st.info(
            "Please ensure `Indian Flight Price.csv` is placed inside the `data/` folder "
            "relative to `app.py`."
        )
        st.stop()

    # ── Sidebar filters ───────────────────────────────────────────────────────
    filtered_df = render_sidebar(df)

    if filtered_df.empty:
        st.warning(
            "⚠️ No records match the current filter combination. "
            "Please adjust the sidebar filters."
        )
        st.stop()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Overview",
        "✈️ Airline Analysis",
        "🗺️ Route Analysis",
        "💰 Price Analysis",
        "🔍 Data Explorer",
    ])

    with tab1:
        tab_overview(filtered_df)

    with tab2:
        tab_airline(filtered_df)

    with tab3:
        tab_route(filtered_df)

    with tab4:
        tab_price(filtered_df)

    with tab5:
        tab_data_explorer(filtered_df)

    # ── Insights (below tabs, full width) ─────────────────────────────────────
    render_insights(filtered_df)

    # ── Footer ────────────────────────────────────────────────────────────────
    st.markdown(
        """
        <div style="text-align:center; padding:24px 0 8px 0;
                    border-top:1px solid #e5e7eb; margin-top:24px;">
            <span style="color:#8b949e; font-size:12px;">
                Indian Flight Price Analysis · IBM Data Analytics Portfolio ·
                Dataset: <em>Indian Flight Price.csv</em> · Built with Streamlit &amp; Plotly
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
