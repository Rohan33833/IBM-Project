# COVID-19 Global Business Intelligence Dashboard (2020–2024)

A single-file Python application that ingests WHO COVID-19 global data, performs
comprehensive exploratory data analysis, and serves an interactive **Dash/Plotly**
business intelligence dashboard structured as a three-section BI report.

---

## 📌 Project Overview

This project analyses the WHO COVID-19 global dataset spanning **2020 to 2024** to
answer three key business questions:

1. **What was the global burden of COVID-19?** — total cases, deaths, and the overall
   epidemic trajectory.
2. **How did different WHO regions compare?** — regional case loads, case fatality
   ratios, and temporal trends.
3. **Where is the highest residual risk, and what should be done?** — country-level
   risk mapping, vaccination-impact proxies, and concrete action recommendations.

Each finding is structured using the **Fact → Insight → Risk/Opportunity → Action**
framework to make insights immediately actionable for decision-makers.

---

## 📂 Dataset Description

| File | Description |
|---|---|
| `WHO-COVID-19-global-data.csv` | Weekly country-level new cases, cumulative cases, new deaths, and cumulative deaths reported to WHO (Jan 2020 – Aug 2024) |

> Additional files (`WHO-COVID-19-global-table-data.csv`, `vaccination-data.csv`,
> `vaccination-metadata.csv`) are part of the full Kaggle dataset and can be
> incorporated for extended vaccination analysis.

### Columns in `WHO-COVID-19-global-data.csv`

| Column | Type | Description |
|---|---|---|
| `Date_reported` | string (DD/MM/YYYY) | Reporting date |
| `Country_code` | string | ISO 2-letter country code |
| `Country` | string | Country or territory name |
| `WHO_region` | string | WHO region code (AFRO, AMRO, EMRO, EURO, SEARO, WPRO, OTHER) |
| `New_cases` | integer | New confirmed cases in reporting period |
| `Cumulative_cases` | integer | Running total of confirmed cases |
| `New_deaths` | integer | New confirmed deaths in reporting period |
| `Cumulative_deaths` | integer | Running total of confirmed deaths |

**Dataset source:**
[https://www.kaggle.com/datasets/abdoomoh/daily-covid-19-data-2020-2024](https://www.kaggle.com/datasets/abdoomoh/daily-covid-19-data-2020-2024)

---

## 🏗 Project Structure

```
.
├── covid_dashboard.py          # Single-file app: data pipeline + Dash dashboard
├── WHO-COVID-19-global-data.csv
├── requirements.txt
└── README.md
```

---

## 📊 Dashboard Sections

### Section 1 — Executive Overview
- 5 KPI cards: total cases, total deaths, global CFR, countries affected, peak month
- Global monthly trend (dual-axis: cases bar + deaths line)
- Yearly grouped bar chart
- Two executive insight boxes (Fact → Insight → Risk → Action)

### Section 2 — Regional Analysis
- Grouped bar: total cases & deaths by WHO region
- Pie chart: share of global cases by region
- Horizontal bar: CFR by WHO region
- Multi-line CFR trend 2020–2024 (vaccination impact proxy)
- Stacked area charts: monthly cases and deaths by region
- Heatmap: CFR across region × year
- Top-10 countries by total cases and by total deaths
- Three regional insight boxes

### Section 3 — Risk, Opportunity & Action
- Scatter plot: total cases vs CFR (bubble = deaths, colour = region)
- High-risk country table (CFR > 3%, cases > 1,000) with conditional formatting
- Four insight boxes covering risk mitigation and vaccination opportunities

---

## ⚙️ Running the Project Locally

### Prerequisites
- Python **3.9 – 3.12** recommended (Python 3.11 preferred)
  > ⚠️ Python 3.14 (released Dec 2025) is not yet fully compatible with numpy/pandas on Windows. Use Python 3.11 for guaranteed compatibility.
- pip

### 1. Clone / download the repository
```bash
git clone <your-repo-url>
cd <repo-folder>
```

### 2. Place the dataset
Ensure `WHO-COVID-19-global-data.csv` is in the **same directory** as
`covid_dashboard.py`.

### 3. Create a virtual environment (recommended)
```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 4. Install dependencies
```bash
pip install -r requirements.txt
```

### 5. Run the dashboard
```bash
python covid_dashboard.py
```

### 6. Open in your browser
Navigate to **http://127.0.0.1:8050**

---

## 🔍 Key Analytical Findings

| Finding | Value |
|---|---|
| Global CFR (2020–2024) | ~1–2% |
| Highest CFR region | AFRO (Africa) |
| Largest case volume region | EURO + AMRO (>60% of global cases) |
| Peak case wave | Early 2022 (Omicron) |
| CFR trend | Declining across all regions post-2021 (vaccination proxy) |

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| Data processing | `pandas`, `numpy` |
| Visualisation | `plotly` |
| Dashboard framework | `dash` |
| Language | Python 3.9+ |

---

## 📄 License
This project is for educational and research purposes. Dataset © WHO / Kaggle contributor.
