import streamlit as st
import pandas as pd
import numpy as np

import os
print("Current working directory:", os.getcwd())

import os
import pandas as pd

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the full path to the CSV file
bc_csv_path = os.path.join(script_dir, "breakout_candidates.csv")
fs_csv_path = os.path.join(script_dir, "full_stats_combined.csv")

# Load the CSV
breakout_df = pd.read_csv(bc_csv_path)
full_df = pd.read_csv(fs_csv_path)

# Metrics to display
metrics = ['exit_velocity', 'launch_angle', 'barrel_pct', 'hard_hit_pct', 'xwOBA', 'xBA', 'xSLG']

# Sort breakout players by breakout_index
sorted_breakouts = (
    breakout_df.sort_values("breakout_index", ascending=False)[["name", "player_id", "breakout_index"]]
    .drop_duplicates(subset=["name"])
    .reset_index(drop=True)
)

# Streamlit UI
st.title("MLB Breakout Player Comparison Tool")
selected_name = st.selectbox("Select a Breakout Player", sorted_breakouts["name"])

# Filter for selected breakout player
player_data = breakout_df[breakout_df["name"] == selected_name].iloc[0]
player_id = player_data["player_id"]

# Most similar established player
similar_df = full_df[(full_df["year"] < 2024) & (full_df["player_id"] != player_id)]
scaled_metrics = full_df[metrics].apply(lambda x: (x - x.mean()) / x.std())
target_vector = scaled_metrics[full_df["player_id"] == player_id].values[0]
similar_df["similarity"] = scaled_metrics.loc[similar_df.index].apply(lambda row: -((row - target_vector) ** 2).sum(), axis=1)
most_similar = similar_df.sort_values("similarity", ascending=False).iloc[0]

# Display Section: Similarity
st.subheader("Most Similar Established Player")
st.markdown(f"**{player_data['name']}** is most similar offensively to **{most_similar['name']}**.")

# Display Section: Player Stats
st.subheader(f"Breakout Stats for {player_data['name']}")
for metric in metrics:
    st.write(f"**{metric}**: {player_data[metric]:.3f}")

# Breakout leader comparison
leader_data = breakout_df.sort_values("breakout_index", ascending=False).iloc[0]

st.markdown("---")
st.subheader("Comparison to Breakout Leader")
for metric in metrics:
    player_val = player_data[metric]
    leader_val = leader_data[metric]
    diff_pct = 100 * (player_val - leader_val) / leader_val
    st.write(f"**{metric}**: {player_val:.3f} vs Leader {leader_val:.3f} ({diff_pct:+.1f}%)")

# Veteran comparison dropdown
st.markdown("---")
veteran_names = full_df[(full_df["year"] == 2023) & (~full_df["name"].isin(sorted_breakouts["name"]))]["name"].unique()
veteran_name = st.selectbox("Select a Veteran Player for Comparison", sorted(veteran_names))

# Pull selected veteran stats
veteran_data = full_df[(full_df["name"] == veteran_name) & (full_df["year"] == 2023)].iloc[0]

st.subheader(f"Comparison to Veteran: {veteran_name}")
for metric in metrics:
    player_val = player_data[metric]
    vet_val = veteran_data[metric]
    diff_pct = 100 * (player_val - vet_val) / vet_val
    st.write(f"**{metric}**: {player_val:.3f} vs {veteran_name} {vet_val:.3f} ({diff_pct:+.1f}%)")
