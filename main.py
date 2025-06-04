from data_loader import load_local_batting_data
from candidate_filter import filter_breakout_candidates
from weighted_metrics import calculate_weighted_averages
from similarity_and_breakout import compute_similarity_and_breakout
from projections import (
    match_and_project,
    build_projection_reference
    )

import pandas as pd

# Load Data
print("Loading Statcast data...")
batting_df = load_local_batting_data()

# Filter for Breakout Candidates
print("Filtering breakout candidates (players who debuted in 2023 or 2024)...")
breakout_candidates_df = filter_breakout_candidates(batting_df)

# Calculate Weighted Averages
print("Calculating weighted averages for recent seasons...")
weighted_df = calculate_weighted_averages(breakout_candidates_df)

# Filter for reference players
print("Preparing reference dataset of established players...")
reference_df = build_projection_reference(batting_df)

# Compute Breakout Scores
print("Computing similarity and breakout scores...")
scored_df = compute_similarity_and_breakout(weighted_df, reference_df)

# Match and Project
print("Matching breakout candidates to historical comps and projecting future performance...")
hist_proj_df, reg_proj_df = match_and_project(scored_df, reference_df, batting_df)

# Export Results
print("Exporting results to CSV for use in Streamlit app...")

scored_df.to_csv("breakout_candidate_metrics.csv", index=False)
hist_proj_df.to_csv("historic_projected_breakouts.csv", index=False)
reg_proj_df.to_csv("linear_reg_projected_breakouts.csv", index=False)

print("All files successfully saved.")
