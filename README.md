# ✈️ Indian Flight Price Analysis

> **IBM Data Analytics Portfolio Project**
> An end-to-end exploratory data analysis and interactive Streamlit dashboard
> built on a real-world dataset of domestic Indian airline ticket prices.

---

## 📋 Table of Contents

1. [Project Objective](#project-objective)
2. [Tech Stack](#tech-stack)
3. [Dataset Description](#dataset-description)
4. [Project Structure](#project-structure)
5. [Data Cleaning](#data-cleaning)
6. [Feature Engineering](#feature-engineering)
7. [Analysis Performed](#analysis-performed)
8. [Dashboard Features](#dashboard-features)
9. [How to Run](#how-to-run)
10. [Key Insights](#key-insights)
11. [Future Improvements](#future-improvements)

---

## Project Objective

Analyse **10,460 domestic Indian airline ticket records** to uncover pricing
patterns across airlines, routes, travel months, flight durations, number of
stops, and departure times.  The analysis is presented through an interactive
**Streamlit dashboard** with sidebar filters, KPI cards, dynamic charts,
and an auto-generated insights section.

---

## Tech Stack

| Layer | Tool / Library | Version | Purpose |
|---|---|---|---|
| **Language** | Python | 3.10+ | Core programming language |
| **Data Handling** | pandas | ≥ 2.0 | Data loading, cleaning, transformation |
| **Numerics** | NumPy | ≥ 1.26 | Numerical operations and NaN handling |
| **Dashboard** | Streamlit | ≥ 1.32 | Interactive web application framework |
| **Visualisation** | Plotly Express | ≥ 5.20 | Interactive bar, line, scatter, box charts |
| **Data Source** | CSV (flat file) | — | `data/Indian Flight Price.csv` |

> All dependencies are listed in `requirements.txt`.
> Install with: `pip install -r requirements.txt`

---

## Dataset Description

**File:** `https://www.kaggle.com/datasets/ibrahimelsayed182/plane-ticket-price?`

| Column | Description |
|---|---|
| `Airline` | Operating airline name |
| `Date_of_Journey` | Flight date (DD/MM/YYYY) |
| `Source` | Departure city |
| `Destination` | Arrival city |
| `Route` | IATA airport code route string |
| `Dep_Time` | Departure time (HH:MM) |
| `Arrival_Time` | Arrival time (HH:MM) |
| `Duration` | Total flight duration (e.g. `2h 50m`) |
| `Total_Stops` | Number of stops (0 = non-stop, 1, 2, 3, 4) |
| `Price` | Ticket price in Indian Rupees (₹) |

**Quick stats:**
- Airlines: 12 (IndiGo, Air India, Jet Airways, SpiceJet, GoAir, Vistara, Air Asia, Trujet + premium variants)
- Source cities: Bangalore, Chennai, Delhi, Kolkata, Mumbai
- Destinations: Bangalore, Cochin, Delhi/New Delhi, Hyderabad, Kolkata
- Months covered: January, March, April, May, June, September, December 2019
- Price range: ₹1,759 – ₹79,512

---

## Project Structure

```
indian-flight-price-analysis/
│
├── app.py                        ← Single self-contained Streamlit app
├── requirements.txt              ← Python package dependencies
├── README.md                     ← This file
└── data/
    └── Indian Flight Price.csv   ← Raw dataset (never modified)
```

> `app.py` is **fully self-contained** — it includes all data loading,
> cleaning, feature engineering, analysis, and dashboard code.
> No other Python files are required.

---

## Data Cleaning

All cleaning is performed inside `app.py` at startup:

1. **Whitespace stripping** — leading/trailing spaces removed from all string columns.
2. **Duplicate removal** — full duplicate rows dropped (none present in this dataset).
3. **Date parsing** — `Date_of_Journey` converted from `DD/MM/YYYY` string to `datetime`.
4. **Price coercion** — cast to integer; rows with non-numeric values dropped.
5. **Stops coercion** — `Total_Stops` cast to integer (already numeric in source).
6. **Known outlier retained** — one record with `Duration = "5m"` is kept; it appears as
   a visible outlier in the scatter chart and can be filtered via the sidebar.

The original raw CSV is **never modified**.

---

## Feature Engineering

New columns added to the processed dataframe:

| Column | Description |
|---|---|
| `Journey_Year` | Integer year (2019) |
| `Journey_Month` | Integer month (1–12) |
| `Journey_Month_Name` | Full month name (e.g. "March") |
| `Journey_Day` | Day of month (1–31) |
| `Journey_Day_Name` | Weekday name (e.g. "Monday") |
| `Departure_Hour` | Hour extracted from `Dep_Time` (0–23) |
| `Departure_Period` | Time-of-day bucket (see below) |
| `Arrival_Hour` | Hour extracted from `Arrival_Time` (0–23) |
| `Duration_Minutes` | Flight duration converted to total minutes |
| `Stops_Label` | Human-readable stop count (e.g. "1 Stop") |
| `Route_Short` | `Source → Destination` label for charts |

**Departure Period classification:**

| Period | Hours |
|---|---|
| Early Morning | 00:00 – 05:59 |
| Morning | 06:00 – 11:59 |
| Afternoon | 12:00 – 16:59 |
| Evening | 17:00 – 20:59 |
| Night | 21:00 – 23:59 |

**Duration parsing examples:** `2h 50m` → 170 min · `19h` → 1140 min · `7h 25m` → 445 min

---

## Analysis Performed

| Area | Analysis |
|---|---|
| **Airline** | Count, avg price, price distribution (box plot), avg duration |
| **Route** | Top 10 common, top 10 most expensive, source/destination breakdown |
| **Stops** | Avg price per stop count, avg duration per stop count |
| **Month** | Average price trend by month (line chart) |
| **Departure Time** | Period classification, avg price per period |
| **Duration vs Price** | Scatter plot coloured by airline |
| **Insights** | 7 dynamic facts auto-calculated from filtered data |

---

## Dashboard Features

### Sidebar Filters
All filters apply instantly across every tab and chart:
- **Airline** — one or more airlines
- **Source City** — departure city
- **Destination** — arrival city
- **Number of Stops** — 0 to 4 stops
- **Journey Month** — month of travel
- **Departure Period** — time of day

> Leave all filters blank to view the complete dataset.

### Tab 1 — Overview
Six KPI cards (Total Flights · Avg Price · Min Price · Max Price · Avg Duration · Avg Stops)
plus Average Price by Airline and Average Price by Month charts.

### Tab 2 — Airline Analysis
Flight counts, average prices, price distribution (box plot), and average duration
— all broken down by airline.

### Tab 3 — Route Analysis
Top 10 common routes, top 10 most expensive routes, average price by source city,
average price by destination, and top 15 source→destination pairs.

### Tab 4 — Price Analysis
Price histogram, price by stops, price by month (line), price by departure period,
flight duration vs price scatter, and duration by stops.

### Tab 5 — Data Explorer
Searchable, filterable table of all filtered records with configurable column
selection and a **Download Filtered Data as CSV** button.

### Key Business Insights (bottom of page)
Seven dynamically calculated textual insights based on the currently filtered data:
airline pricing extremes, busiest route, most expensive route, peak/off-peak months,
stop-count price comparison, and top source city by price.

---

## Technologies Used

| Library | Version | Purpose |
|---|---|---|
| `pandas` | ≥ 2.0 | Data loading, cleaning, transformation |
| `numpy` | ≥ 1.26 | Numerical operations |
| `streamlit` | ≥ 1.32 | Interactive web dashboard |
| `plotly` | ≥ 5.20 | Interactive charts |

Python version: **3.10+**

---

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Verify the dataset is in place

```
indian-flight-price-analysis/
└── data/
    └── Indian Flight Price.csv   ← must exist here
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

The dashboard opens automatically at **http://localhost:8501**.

---

## Key Insights

> Based on the full unfiltered dataset (10,460 records).

- **Jet Airways Business** and **Vistara Premium Economy** command the highest average fares,
  reflecting premium cabin positioning.
- **Non-stop flights are not always cheapest** — routing and airline choice can outweigh
  the stop-count premium.
- **June** records the highest average ticket prices, coinciding with peak summer demand.
- **Morning departures** (06:00–11:59) tend to attract slightly lower fares than Evening/Night.
- The **Delhi → Cochin** and **Bangalore → New Delhi** corridors are the busiest in the dataset.
- Flights with **2+ stops** generally have longer durations but do not consistently cost
  more than 1-stop options, suggesting capacity dump pricing on longer itineraries.

---

## Future Improvements

- Add a fare prediction tab (Linear Regression / Random Forest) as a fifth analytical module.
- Integrate geospatial route maps using `pydeck` or `folium`.
- Include cabin class breakdown (Economy vs Business vs Premium).
- Add a correlation heatmap for numerical features.
- Deploy to Streamlit Community Cloud for public sharing.
- Extend the dataset to cover multiple years for trend analysis.
