# MLB Breakout Predictor

This project uses advanced Statcast data to identify potential breakout Major League Baseball (MLB) players by evaluating offensive metrics and comparing them to elite-level performance benchmarks.

## Project Overview

The program:
- Loads and merges expected offensive stats and exit velocity data from multiple seasons.
- Computes weighted averages favoring more recent years (2024 and 2025).
- Filters for breakout candidates who did **not** appear in 2022 or 2023.
- Calculates a **superstar similarity score** and a **breakout index** for each candidate.
- Outputs a ranked list of breakout candidates and visualizes the results in an interactive Plotly scatter plot.

## Files

- `mlb_breakout_model.py`: Main executable script with modular functions.
- `expected_stats 22.csv` to `expected_stats 25.csv`: Expected offensive statistics for each year.
- `exit_velocity22.csv` to `exit_velocity25.csv`: Exit velocity and batted ball profile data.
- `breakout_candidates.csv`: Output file with breakout metrics and similarity scores.

## Metrics Used

- `exit_velocity`
- `launch_angle`
- `barrel_pct`
- `hard_hit_pct`
- `xwOBA`
- `xBA`
- `xSLG`
