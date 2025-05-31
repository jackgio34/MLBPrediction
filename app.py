import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import plotly.express as px

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_data
def load_data():
    # Construct full paths to the CSV files
    bc_csv_path = os.path.join(script_dir, "breakout_candidate_metrics.csv")
    hist_csv_path = os.path.join(script_dir, "historic_projected_breakouts.csv")
    reg_csv_path = os.path.join(script_dir, "linear_reg_projected_breakouts.csv")

    # Load the CSVs
    candidates = pd.read_csv(bc_csv_path)
    projections_hist = pd.read_csv(hist_csv_path)
    projections_reg = pd.read_csv(reg_csv_path)

    # Clean up names to First Last format
    candidates['first_last_name'] = candidates['last_name, first_name'].apply(lambda x: " ".join(x.split(", ")[::-1]))
    projections_hist['first_last_name'] = projections_hist['last_name, first_name'].apply(lambda x: " ".join(x.split(", ")[::-1]))
    projections_reg['first_last_name'] = projections_reg['last_name, first_name'].apply(lambda x: " ".join(x.split(", ")[::-1]))
    projections_hist['match_name'] = projections_hist['match_name'].apply(lambda x: " ".join(x.split(", ")[::-1]))

    return candidates, projections_hist, projections_reg

candidates_df, hist_proj_df, reg_proj_df = load_data()

# Create a mapping from first_last_name to last_name, first_name
name_map = dict(zip(candidates_df['first_last_name'], candidates_df['last_name, first_name']))

# Streamlit UI
st.title("MLB Breakout Candidates Viewer")

# Interactive Scatter Plot
st.subheader("Breakout Profile: Similarity vs. Breakout Score")
fig = px.scatter(
    candidates_df,
    x='superstar_similarity',
    y='breakout_score',
    hover_name='first_last_name',
    labels={
        'superstar_similarity': 'Superstar Similarity',
        'breakout_score': 'Breakout Score'
    },
    title="Breakout Candidate Map"
)
st.plotly_chart(fig, use_container_width=True)

# Sort players by breakout score (highest first)
sorted_candidates = candidates_df.sort_values(by='breakout_score', ascending=False)
sorted_names = sorted_candidates['first_last_name'].tolist()

# Update name map for consistent lookups
name_map = dict(zip(sorted_candidates['first_last_name'], sorted_candidates['last_name, first_name']))

# Display dropdown sorted by breakout score
selected_first_last = st.selectbox(
    "Select a breakout candidate (ranked by breakout score):",
    sorted_names,
    key="candidate_dropdown"
)

# Convert back to original name format for filtering
selected_name = name_map[selected_first_last]

# --- Display Key Stats ---
st.subheader("Key Statcast Metrics")
metrics_to_show = {
    'breakout_score': 'Breakout Score',
    'superstar_similarity': 'Similarity Similarity',
    'exit_velocity_avg': 'Avg Exit Velocity (mph)',
    'launch_angle_avg': 'Avg Launch Angle (°)',
    'barrel_batted_rate': 'Barrel Rate (%)',
    'hard_hit_percent': 'Hard Hit %',
    'xwoba': 'xwOBA',
    'xba': 'xBA',
    'xslg': 'xSLG'
}

player_row = candidates_df[candidates_df['last_name, first_name'] == selected_name]
if not player_row.empty:
    for stat, label in metrics_to_show.items():
        value = player_row.iloc[0][stat]
        st.metric(label, f"{value:.3f}")
else:
    st.warning("Player data not found.")

st.subheader("Comparison Player & Projection Context")

# Pull comparison match name from projections
match_row = hist_proj_df[hist_proj_df['first_last_name'] == selected_first_last]
if not match_row.empty:
    match_name = match_row['match_name'].iloc[0]
    st.markdown(f"""
    **Historical Comparison:** {match_name}  
    This projection assumes that **{selected_first_last}** progresses similarly to how **{match_name}** did over their career.  
    The historical projection you're seeing below are what we'd expect if this trend continues year-over-year.
    """)
else:
    st.warning("No comparison player found for this candidate.")


# Projection Chart
st.subheader("Stat Projection Over Time")
proj_type = st.radio("Select projection type:", ["Historical", "Regression"], horizontal=True)
proj_df = hist_proj_df if proj_type == "Historical" else reg_proj_df
player_proj = proj_df[proj_df['last_name, first_name'] == selected_name]

# Display name to internal stat mapping, excluding breakout and similarity scores
display_to_stat = {
    v: k for k, v in metrics_to_show.items()
    if k not in ['breakout_score', 'superstar_similarity']
}

# Stat options for dropdown
stat_display = st.selectbox("Select a stat to plot:", list(display_to_stat.keys()), key="stat_selector")
stat_to_plot = display_to_stat[stat_display]

if not player_proj.empty:
    fig, ax = plt.subplots()
    ax.plot(player_proj['year'], player_proj[stat_to_plot], marker='o')
    ax.set_title(f"{stat_display} Projection for {selected_first_last}")
    ax.set_xlabel("Year")
    ax.set_ylabel(stat_display)
    ax.grid(True)
    st.pyplot(fig)
else:
    st.warning("No projection data available for this player.")
