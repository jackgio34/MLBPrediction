# MLB Breakout Predictor

This project uses Statcast data to identify potential breakout Major League Baseball (MLB) players by evaluating offensive metrics and comparing them to elite-level performance benchmarks. It compares offensive skill sets to historical superstar trends and projects future performance over a 4-year horizon.

# MLB Breakout Predictor

## Project Overview

The pipeline performs the following:

- Loads and merges **expected offensive metrics** and **exit velocity data**.
- Applies **supplemental player age data** from 2023 and 2024 to fill in gaps.
- Filters for breakout candidates who debuted **in 2023 or 2024**.
- Calculates a **superstar similarity score** and a **breakout index**.
- Projects performance both historically (via player analogs) and statistically (via regression).
- Visualizes individual projections in an interactive Streamlit dashboard.

## Included Files

### Scripts

- `main.py` – Entry point to run the full data pipeline and generate all outputs.
- `data_loader.py` – Loads and merges all Statcast and age data; cleans and aligns required metrics.
- `candidate_filter.py` – Filters valid breakout candidates based on debut years and data completeness.
- `weighted_metrics.py` – Computes weighted multi-year stat averages with recent seasons prioritized.
- `similarity_and_breakout.py` – Computes breakout scores and superstar similarity.
- `projections.py` – Generates historical and regression-based stat projections.
- `app.py` – Streamlit dashboard for exploring player stats and projections.

### CSV Data

- `batting.csv` – Core batting data including player stats and identifiers.
- `mlb-player-stats-Batters2023.csv`, `mlb-player-stats-Batters2024.csv` – Supplemental files for 2023 and 2024 player age.
- `expected_stats 23.csv`, `expected_stats 24.csv` – Yearly expected offensive metrics (xwOBA, xBA, xSLG, etc.).
- `exit_velocity 23.csv`, `exit_velocity 24.csv` – Exit velocity and launch angle stats.
- `breakout_candidate_metrics.csv` – Final list of candidate players with scores and raw metrics.
- `historic_projected_breakouts.csv`, `linear_reg_projected_breakouts.csv` – Future stat projections by method.

## Key Metrics

The following advanced metrics are calculated and compared:

| Metric                | Description                                             |
|-----------------------|---------------------------------------------------------|
| `exit_velocity_avg`   | Average exit velocity of batted balls                   |
| `launch_angle_avg`    | Average launch angle                                    |
| `barrel_batted_rate`  | % of batted balls that are barrels                      |
| `hard_hit_percent`    | % of batted balls hit 95+ mph                           |
| `xwoba`               | Expected weighted on-base average                       |
| `xba`                 | Expected batting average                                |
| `xslg`                | Expected slugging percentage                            |
| `player_age`          | Derived or merged player age by season                  |
| `breakout_score`      | Overall breakout potential score                        |
| `superstar_similarity`| Similarity score to historical superstar performance    |

## How to Run

### Step 1: Install Required Libraries

Ensure Python 3.8+ is installed. Then install dependencies:

``` pip install pandas numpy scikit-learn matplotlib plotly streamlit ```

### Step 2: Generate Breakout Candidates
Run the data processing script to generate breakout candidates:

``` python main.py ```

This will output linear_reg_projected_breakouts, historic_projected_breakouts, and breakout_candidate_metrics.csv.

### Step 3: Launch the Streamlit App

Start the interactive dashboard with:

``` streamlit run app.py ```

You can then explore:

Stat trends for breakout candidates

Historical analogs with projections

Stat-by-stat visual comparisons
