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

st.markdown("""
### Methodology Overview

#### 1. Historical Player Comparison (Similarity Matching)
This method compares each breakout candidate's weighted average offensive metrics (e.g., `exit_velocity`, `xwOBA`, `xBA`) from their first two seasons to those of historical players using Euclidean distance. A candidate is "matched" to the most statistically similar player who also had two qualifying early seasons before 2023.

**Purpose**: Project the breakout candidate’s future performance by averaging the matched historical player's production over the next 3–5 years.

#### 2. Linear Regression Forecasting
In parallel with the similarity method, a linear regression model is fitted using historical players’ early stats and their future production growth. This regression is then applied to the breakout candidates to predict their trajectory based on trends learned from historical outcomes.

**Purpose**: Provide an alternate, generalized projection using statistical modeling rather than individual comparisons.

#### 3. Superstar Similarity & Breakout Score
- **Superstar Similarity Score**: Measures how close a player’s stats are to the top 5% of all 2025 players using z-scored Euclidean distance. The higher (less negative) the score, the more "superstar-like" the player is.
- **Breakout Score**: The weighted average of all key performance metrics.
- **Breakout Index**: A composite metric calculated as a blend of breakout score and superstar similarity.
""")

# Interactive Scatter Plot
st.subheader("Breakout Profile")
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
sorted_candidates = candidates_df.sort_values(by='first_last_name', ascending=False)
sorted_names = sorted_candidates['first_last_name'].tolist()

# Update name map for consistent lookups
name_map = dict(zip(sorted_candidates['first_last_name'], sorted_candidates['last_name, first_name']))

# Display dropdown sorted by first name
selected_first_last = st.selectbox(
    "Select a breakout candidate:",
    sorted_names,
    key="candidate_dropdown"
)

# Convert back to original name format for filtering
selected_name = name_map[selected_first_last]

# Display Key Stats
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

    player_index = player_row.index[0]

    # Breakout/Similarity/Index rank
    breakout_sorted = candidates_df.sort_values(by='breakout_score', ascending=False).reset_index()
    breakout_rank = breakout_sorted[breakout_sorted['last_name, first_name'] == selected_name].index[0] + 1
    breakout_total = len(breakout_sorted)

    similarity_sorted = candidates_df.sort_values(by='superstar_similarity', ascending=False).reset_index()
    similarity_rank = similarity_sorted[similarity_sorted['last_name, first_name'] == selected_name].index[0] + 1
    similarity_total = len(similarity_sorted)

    index_sorted = candidates_df.sort_values(by='breakout_index', ascending=False).reset_index()
    index_rank = index_sorted[similarity_sorted['last_name, first_name'] == selected_name].index[0] + 1
    index_total = len(index_sorted)

    # Row 1: 4 metrics
    row1 = st.columns(4)
    row1[0].metric("Avg Exit Velocity (mph)", f"{player_row.iloc[0]['exit_velocity_avg']:.3f}")
    row1[1].metric("Avg Launch Angle (°)", f"{player_row.iloc[0]['launch_angle_avg']:.3f}")
    row1[2].metric("Barrel Rate (%)", f"{player_row.iloc[0]['barrel_batted_rate']:.3f}")
    row1[3].metric("Hard Hit %", f"{player_row.iloc[0]['hard_hit_percent']:.3f}")

    # Row 2: 4 columns, 3 metrics + 1 empty for alignment
    row2 = st.columns(4)
    row2[0].metric("xBA", f"{player_row.iloc[0]['xba']:.3f}")
    row2[1].metric("xSLG", f"{player_row.iloc[0]['xslg']:.3f}")
    row2[2].metric("xwOBA", f"{player_row.iloc[0]['xwoba']:.3f}")
    row2[3].empty()  # Padding for alignment

    # Row 3 (Ranked Scores)
    row3 = st.columns(4)
    row3[0].metric(
        "Breakout Score",
        f"{player_row.iloc[0]['breakout_score']:.3f}",
        f"Rank: {breakout_rank} / {breakout_total}"
    )
    row3[1].metric(
        "Superstar Similarity",
        f"{player_row.iloc[0]['superstar_similarity']:.3f}",
        f"Rank: {similarity_rank} / {similarity_total}"
    )
    row3[2].metric(
        "Breakout Index",
        f"{player_row.iloc[0]['breakout_index']:.3f}",
        f"Rank: {index_rank} / {index_total}"
    )
    row3[3].empty()
    
else:
    st.warning("Player data not found.")


st.subheader("Comparison Player & Projection Context")

# Pull comparison match name from projections
match_row = hist_proj_df[hist_proj_df['first_last_name'] == selected_first_last]
if not match_row.empty:
    match_name = match_row['match_name'].iloc[0]
    st.markdown(f"""
    **Historical Comparison:** {match_name}  
    This projection assumes that **{selected_first_last}** progresses similarly to how **{match_name}** did **OFFENSIVELY** over their career.  
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


# Define keys for a stat based leaderboard
stat_buttons = {
    "Breakout Score": "breakout_score",
    "Superstar Similarity": "superstar_similarity",
    "Avg Exit Velocity": "exit_velocity_avg",
    "Barrel Rate": "barrel_batted_rate",
    "Hard Hit %": "hard_hit_percent",
    "xwOBA": "xwoba",
    "xBA": "xba",
    "xSLG": "xslg"
}

# Split buttons into two rows
row1_labels = list(stat_buttons.keys())[:4]
row2_labels = list(stat_buttons.keys())[4:]

st.markdown("### Stat-Based Leaderboards")

# Render first row of buttons
cols1 = st.columns(len(row1_labels))
for i, label in enumerate(row1_labels):
    if cols1[i].button(label, key=f"stat_button_1_{label}"):
        st.session_state.selected_leaderboard_stat = stat_buttons[label]
        st.session_state.selected_label = label

# Render second row of buttons
cols2 = st.columns(len(row2_labels))
for i, label in enumerate(row2_labels):
    if cols2[i].button(label, key=f"stat_button_2_{label}"):
        st.session_state.selected_leaderboard_stat = stat_buttons[label]
        st.session_state.selected_label = label

# Set defaults if none selected yet
if "selected_leaderboard_stat" not in st.session_state:
    st.session_state.selected_leaderboard_stat = "breakout_score"
    st.session_state.selected_label = "Breakout Score"

selected_stat = st.session_state.selected_leaderboard_stat
selected_label = st.session_state.selected_label

# Compute proper rankings using rank()
candidates_df['rank'] = candidates_df[selected_stat].rank(method="min", ascending=False)
candidates_df_sorted = candidates_df.sort_values(by=selected_stat, ascending=False)


st.markdown(f"#### Top Players by **{selected_label}**")
st.dataframe(
    candidates_df_sorted[['first_last_name', selected_stat]]
    .rename(columns={
        'first_last_name': 'Player',
        selected_stat: selected_label,
    })
    .reset_index(drop=True)
    .style.format({selected_label: "{:.3f}", 'Rank': '{:.0f}'})
    .hide(axis="index"),
    use_container_width=True
)
