# MLB Breakout Predictor

This project uses Statcast data to identify potential breakout Major League Baseball (MLB) players by evaluating offensive metrics and comparing them to elite-level performance benchmarks.

## Project Overview

The program:
- Loads and merges expected offensive stats and exit velocity data from multiple seasons.
- Computes weighted averages favoring more recent years (2024 and 2025).
- Filters for breakout candidates who did **not** appear in 2022 or 2023.
- Calculates a **superstar similarity score** and a **breakout index** for each candidate.
- Outputs a ranked list of breakout candidates and visualizes the results in an interactive Plotly scatter plot.

## Features

- Dynamic player filtering by name with an autocomplete dropdown.
- Interactive visualizations using Plotly and Streamlit.
- Stat comparison to elite performers in 2025.
- Player similarity matching for contextual analysis.
- Side-by-side comparisons of year-over-year stat changes (e.g., 2024 to 2025).

## Files Included

- `mlb_breakout_model.py` – Core script for calculating breakout scores and generating output.
- `InteractiveDropdown.py` – Streamlit application for interactive exploration of breakout candidates.
- `breakout_candidates.csv` – Model-generated output including breakout index and similarity metrics.
- `full_stats_combined.csv` – Combined player stat history used for comparative lookups.
- `expected_stats 22.csv` to `expected_stats 25.csv` – Expected offensive statistics by year.
- `exit_velocity22.csv` to `exit_velocity25.csv` – Exit velocity and batted ball data by year.

## Metrics Used

Metrics computed and analyzed include:

- `exit_velocity`: Average exit velocity of batted balls.
- `launch_angle`: Average launch angle of batted balls.
- `barrel_pct`: Percentage of batted balls classified as "barrels".
- `hard_hit_pct`: Percentage of batted balls hit 95+ mph.
- `xwOBA`: Expected weighted on-base average.
- `xBA`: Expected batting average.
- `xSLG`: Expected slugging percentage.

## How to Run the Project

### Step 1: Install Required Libraries

Ensure Python 3.8+ is installed. Then install dependencies:

  pip install pandas numpy scikit-learn matplotlib plotly streamlit

### Step 2: Generate Breakout Candidates
Run the data processing script to generate breakout candidates:

  python mlb_breakout_model.py

This will output breakout_candidates.csv and generate the interactive plot.

### Step 3: LAunch the Streamlit App

Start the interactive dashboard with:

  streamlit run InteractiveDropdown.py

This will open a web browser where you can:
- Select a breakout candidate from a dropdown menu (sorted by breakout index).
- View their similarity to elite performers.
- Compare their 2024 and 2025 metrics if available.
- Explore how similar non-candidates relate to elite performance metrics.

### Notes

- All CSV data files must be placed in the same directory as the scripts.
- Only players who did not appear in 2022 or 2023 are considered breakout candidates.
- Established players from other seasons are used as comparison references only.

