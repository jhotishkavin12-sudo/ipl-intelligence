# 🏏 IPL Intelligence: Production Cricket Analytics & ML Decision Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.63-FF4B4B.svg)](https://streamlit.io)
[![DuckDB](https://img.shields.io/badge/DuckDB-In--Memory%20SQL-FFF000.svg)](https://duckdb.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML%20Pipelines-F7931E.svg)](https://scikit-learn.org)
[![Data Quality](https://img.shields.io/badge/Data%20Quality-100%25%20PASS-brightgreen.svg)]()
[![Tests](https://img.shields.io/badge/Tests-10%20Passed-success.svg)]()

> **An end-to-end cricket analytics and machine learning decision platform that transforms 2008–2026 ball-by-ball IPL match data into actionable team intelligence, composite player impact metrics, head-to-head matchup scouting, and real-time in-play win probability modeling.**

---

## 🌟 Executive Summary & Motivation

Traditional cricket portfolio projects often reduce IPL data to basic charts ("Top 10 run-scorers" or "Toss win vs Match win"). 

**IPL Intelligence** is designed from the ground up as a **decision-support platform** for coaches, analysts, and franchise strategists:
1. **Granular Ball-by-Ball Data Pipeline**: Ingests and standardizes official delivery data spanning **1,243 matches** and **295,732 deliveries** (2008–2026).
2. **Phase-Partitioned Tactical Analytics**: Separates every innings into *Powerplay (0–6)*, *Middle (7–15)*, and *Death (16–20)* overs.
3. **Proprietary Player Impact Score**: Normalized multi-dimensional composite index balancing volume, phase par-rate differentials, dot-ball resistance, and pressure execution.
4. **Batter vs. Bowler Head-to-Head Scouting**: Complete tactical head-to-head profiling with phase filters and dismissal distributions.
5. **Dual Machine Learning Engines**:
   - **Pre-Match Winner Predictor** with explainable factor attribution.
   - **Dynamic In-Play Live Win Probability Engine** (80.2% Accuracy, 0.95 ROC-AUC) with an interactive **"What-If" Scenario Simulator**.
6. **High-Performance In-Memory SQL Layer**: Instant sub-millisecond query execution powered by **DuckDB** using CTEs, `RANK() OVER`, and `PARTITION BY`.
7. **Automated Data Quality Auditing**: Continuous hygiene audits verifying zero row duplicates, schema completeness, and domain constraints (100% Health Score).

---

## 🏗️ System Architecture

```
                                  [ RAW DATA ]
                     Official Cricsheet Ball-by-Ball IPL Archive
                                  (2008 - 2026)
                                       │
                                       ▼
                         [ DATA PIPELINE & VALIDATION ]
                  Franchise Normalization ── Rebowled Ball Auditing
                  Phase Partitioning (PP, Middle, Death) ── Target Scoring
                                       │
                      ┌────────────────┴────────────────┐
                      ▼                                 ▼
             [ PROCESSED PARQUET ]              [ SQL ANALYTICS ]
               matches.parquet                   schema.sql
              deliveries.parquet                 player_analysis.sql
                      │                          team_analysis.sql
                      │                                 │
                      ▼                                 ▼
          [ FEATURE & METRIC ENGINE ]            [ DUCKDB ENGINE ]
          Tactical Batting/Bowling Stats          Sub-millisecond
          Custom Player Impact Score              Window Functions
          Clutch / Pressure Index                 Analytical Views
                      │                                 │
                      └────────────────┬────────────────┘
                                       │
                                       ▼
                           [ MACHINE LEARNING LAYER ]
               ┌───────────────────────┴───────────────────────┐
               ▼                                               ▼
     [ PRE-MATCH CLASSIFIER ]                       [ IN-PLAY PROBABILITY ]
       Rolling Team Form                              Live Score & RRR
       H2H Win Rate & Venue Adv                       Wickets in Hand
       Explainable AI Deltas                          "What-If" Simulator
               │                                               │
               └───────────────────────┬───────────────────────┘
                                       │
                                       ▼
                        [ INTERACTIVE DASHBOARD ]
                  Streamlit + Plotly Multi-Module App
```

---

## 📐 Proprietary Metrics & Mathematical Formulation

### 1. Player Impact Score ($0 \le \text{Impact} \le 100$)
Rather than judging batters solely by runs or bowlers solely by wickets, the **Player Impact Score** normalizes multiple dimensions relative to T20 era benchmarks:

$$\text{Impact} = w_{\text{bat}} \cdot I_{\text{bat}} + w_{\text{bowl}} \cdot I_{\text{bowl}} + w_{\text{clutch}} \cdot I_{\text{clutch}} + w_{\text{phase}} \cdot I_{\text{phase}}$$

Where:
- **$I_{\text{bat}}$**: Volume cap + Strike Rate par differential + Boundary Frequency - Dot Ball Penalty.
- **$I_{\text{bowl}}$**: Wicket volume + Economy relative to phase benchmark + Dot Ball pressure + Death Wicket Premium.
- **$I_{\text{clutch}}$**: Performance under high-pressure scenarios (RRR $\ge 9.5$ or overs remaining $\le 5$ with wickets $\le 4$).
- **$I_{\text{phase}}$**: Death-overs acceleration or death-overs defensive economy.
- **Dynamic Weighting**: Default $(0.40, 0.40, 0.10, 0.10)$ with live sliders in the dashboard.

### 2. Pressure / Clutch Index
Evaluates player efficiency under acute match stress:
$$\text{Clutch Ratio}_{\text{Batter}} = \frac{\text{Strike Rate}_{\text{Pressure}}}{\text{Strike Rate}_{\text{Normal}}}$$
$$\text{Clutch Ratio}_{\text{Bowler}} = \frac{\text{Economy}_{\text{Normal}}}{\text{Economy}_{\text{Pressure}}}$$

---

## 🤖 Machine Learning Benchmarks

### 1. In-Play Live Win Probability Engine
- **Target**: Chasing team match win outcome (1 or 0).
- **Features**: Target, Current Score, Balls Remaining, Wickets in Hand, Current Run Rate (CRR), Required Run Rate (RRR), Run Rate Differential ($\text{CRR} - \text{RRR}$).
- **Evaluation on Test Holdout**:
  - **Accuracy**: `80.24%`
  - **ROC-AUC**: `0.950`
  - **Calibration**: Strictly monotonic (decreasing probability as wickets fall or RRR rises).

### 2. Pre-Match Winner Predictor Benchmark
- **Target**: Team 1 Match Victory.
- **Features**: Rolling 10-match win rates, historical head-to-head win %, venue win rates, toss advantage.
- **Comparative Model Evaluation**:
  | Model | Accuracy | ROC-AUC | Log Loss |
  | :--- | :---: | :---: | :---: |
  | **Logistic Regression (Standardized)** | 48.77% | 0.469 | 0.700 |
  | **Random Forest (Max Depth 5)** | **49.18%** | **0.489** | **0.696** |
  | **Gradient Boosting Classifier** | 48.77% | 0.479 | 0.713 |

> [!NOTE]
> **Data Science Insight**: In T20 cricket, pre-match outcomes hover near a 50/50 baseline due to high individual variance and toss/dew fluctuations. However, once in-play variables (wickets, runs required, run rate) emerge, predictive accuracy surges to **80.2%** with an **ROC-AUC of 0.950**.

---

## 💾 Advanced SQL Intelligence Layer (DuckDB)

The project includes production-grade SQL scripts showcasing advanced database techniques:

### Orange Cap Leaders using Window Functions (`sql/player_analysis.sql`)
```sql
WITH season_runs AS (
    SELECT 
        season,
        striker AS player,
        SUM(runs_off_bat) AS total_runs,
        ROUND((SUM(runs_off_bat) * 100.0) / NULLIF(COUNT(CASE WHEN is_legal_ball = 1 THEN 1 END), 0), 2) AS strike_rate,
        SUM(is_six) AS total_sixes
    FROM deliveries
    GROUP BY season, striker
),
ranked_batters AS (
    SELECT 
        season, player, total_runs, strike_rate, total_sixes,
        RANK() OVER (PARTITION BY season ORDER BY total_runs DESC) AS season_rank
    FROM season_runs
)
SELECT season, player AS orange_cap_winner, total_runs, strike_rate, total_sixes
FROM ranked_batters
WHERE season_rank = 1
ORDER BY season DESC;
```

---

## 🛡️ Automated Data Quality & Testing

Continuous data validation suite verifying dataset integrity before analysis:
- **Total Validated Matches**: 1,243
- **Total Validated Deliveries**: 295,732
- **Data Hygiene Health Score**: **100.0% PASS**
- **Unit Test Suite**: 10 unit tests passing in `< 5s` covering pipeline normalization, metric boundaries, probability calibration, and scenario simulation.

Run the test suite anytime:
```bash
py -m pytest tests/ -v
```

---

## 🚀 Quickstart & Setup

### 1. Clone & Install Dependencies
```bash
cd ipl-intelligence
py -m pip install -r requirements.txt
```

### 2. Run Data Pipeline & Train Models
```bash
# Ingests Cricsheet dataset and runs validation
py src/data_pipeline.py
py src/validation.py

# Trains ML models and benchmarks
py src/models.py
```

### 3. Launch the Interactive Dashboard
```bash
py -m streamlit run app.py
```

---

## 📁 Repository Structure

```
ipl-intelligence/
├── data/
│   ├── raw/                  # Downloaded raw zip archive
│   └── processed/            # matches.parquet, deliveries.parquet, data_quality_report.json
├── sql/
│   ├── schema.sql            # Relational table DDL
│   ├── player_analysis.sql   # Orange/Purple Cap CTEs, Death overs, Powerplay choke
│   └── team_analysis.sql     # Phase run-rate differential, 180+ chases, Home fortresses
├── src/
│   ├── __init__.py
│   ├── data_pipeline.py      # Ingestion, franchise rebrand mapping, enrichment
│   ├── validation.py         # Automated data quality checks (health score)
│   ├── features.py           # Phase-split batting, bowling, and team aggregations
│   ├── metrics.py            # Player Impact Score & Clutch/Pressure Index
│   ├── analytics.py          # Batter vs Bowler H2H, Partnerships, Venue Intelligence
│   ├── models.py             # Pre-match ML, in-play win probability, what-if engine
│   └── db.py                 # DuckDB relational in-memory execution harness
├── models/                   # Serialized pre_match_model.joblib, in_play_model.joblib
├── tests/
│   ├── test_pipeline.py      # Pipeline and validation tests
│   ├── test_features.py      # Metric bounds & formula tests
│   └── test_models.py        # ML probability calibration & monotonicity tests
├── app.py                    # 8-module Streamlit + Plotly decision dashboard
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 💼 Resume & Interview Talking Points

- **End-to-End Data Engineering**: Built a complete pipeline handling 295,000+ real delivery events, managing franchise rebrandings and rebowled delivery edge cases.
- **Domain-Specific Feature Engineering**: Designed tactical phase features (Powerplay, Middle, Death) and a custom Player Impact Score combining volume, rate differential, and clutch performance.
- **Explainable ML**: Trained calibrated probabilistic models for live in-play forecasting (0.95 ROC-AUC) and built a "What-If" tactical simulator calculating counterfactual scenario shifts.
- **Modern Analytical Stack**: Integrated DuckDB for sub-millisecond analytical SQL directly over Parquet files, with an interactive Streamlit UI and comprehensive Pytest test suite.
