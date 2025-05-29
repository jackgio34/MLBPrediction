import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import plotly.express as px
import os

# Configuration
EXPECTED_FILES = ["expected_stats 22.csv", "expected_stats 23.csv", "expected_stats 24.csv", "expected_stats 25.csv"]
EV_FILES = ["exit_velocity22.csv", "exit_velocity 23.csv", "exit_velocity 24.csv", "exit_velocity25.csv"]
METRICS = ['exit_velocity', 'launch_angle', 'barrel_pct', 'hard_hit_pct', 'xwOBA', 'xBA', 'xSLG']
WEIGHTS = {2024: 0.4, 2025: 0.6}

# Functions
def load_and_merge_data(expected_files, ev_files):
    
     """
    Load and merge expected stats and exit velocity data across multiple years.
    - Renames columns to standard metric names.
    - Dynamically assigns a 'name' column.
    - Cleans and combines all yearly data into a single DataFrame.
    Returns:
        pd.DataFrame: Combined and cleaned player stat data across all years.
    """
     
     all_data = []
     for expected, ev in zip(expected_files, ev_files):
        expected_df = pd.read_csv(expected)
        ev_df = pd.read_csv(ev)
        merged = pd.merge(expected_df, ev_df, on='player_id', suffixes=('_est', '_ev'))

        merged = merged.rename(columns={
            'avg_hit_speed': 'exit_velocity',
            'avg_hit_angle': 'launch_angle',
            'brl_percent': 'barrel_pct',
            'ev95percent': 'hard_hit_pct',
            'est_woba': 'xwOBA',
            'est_ba': 'xBA',
            'est_slg': 'xSLG'
        })

        name_col = next((col for col in merged.columns if 'name' in col.lower()), None)
        merged['name'] = merged[name_col] if name_col else 'Unknown'

        merged = merged[['player_id', 'name', 'year', *METRICS]].dropna()
        merged['year'] = merged['year'].astype(int)
        all_data.append(merged)
     return pd.concat(all_data, ignore_index=True)

def calculate_weighted_averages(df):

    """
    Filters players who only appeared in 2024 or both 2024 and 2025, excluding anyone in 2022 or 2023.
    - Includes players who have not qualified for leaderboards in 2025 but started their career in 2024.
    - Assigns weights based on year (higher for more recent).
    - Calculates weighted metric averages for each eligible player.
    Returns:
        grouped (pd.DataFrame): Weighted average metrics for breakout candidates.
        df (pd.DataFrame): Original combined dataset.
        eligible_ids (List): List of player_ids considered eligible for breakout analysis.
    """
    
    df['year'] = df['year'].astype(int)
    player_years = df.groupby('player_id')['year'].agg(set)

    # Players who appeared did not appear prior to 2023

    
    eligible_ids = [pid for pid, years in player_years.items() if (2022 not in years and 2023 not in years and (2024 in years or 2025 in years))]

    df_recent = df[df['player_id'].isin(eligible_ids) & df['year'].isin([2024, 2025])].copy()
    df_recent['weight'] = df_recent['year'].map(WEIGHTS)

    for metric in METRICS:
        df_recent[f'{metric}_w'] = df_recent[metric] * df_recent['weight']

    grouped = df_recent.groupby(['player_id', 'name'])[[f'{m}_w' for m in METRICS]].sum()
    grouped.columns = METRICS
    grouped = grouped.reset_index()
    grouped['player_id'] = grouped['player_id'].astype(str)

    return grouped, df, eligible_ids

def compute_similarity_and_breakout(df, grouped):

    """
    Calculates superstar similarity and a composite breakout index for each player.
    - Similarity is computed using negative Euclidean distance from superstar thresholds (95th percentile).
    - Breakout index is a weighted sum of breakout score and similarity score.
    Returns:
        pd.DataFrame: Final results with similarity and breakout scores.
    """
    
    latest_year_data = df[df['year'] == 2025]
    superstar_thresholds = latest_year_data[METRICS].quantile(0.95)

    scaler = StandardScaler()
    scaled_metrics = scaler.fit_transform(grouped[METRICS])
    similarity = -np.linalg.norm(scaled_metrics - scaler.transform([superstar_thresholds])[0], axis=1)
    grouped['superstar_similarity'] = similarity

    grouped['breakout_score'] = grouped[METRICS].mean(axis=1)
    grouped['breakout_index'] = 0.7 * grouped['breakout_score'] + 0.3 * grouped['superstar_similarity']

     # --- Compare to established players ---
    pre2024_ids = df[df['year'].isin([2022, 2023])]['player_id'].unique()
    established_df = df[df['player_id'].isin(pre2024_ids) & df['year'].isin([2024, 2025])].copy()
    established_df['weight'] = established_df['year'].map(WEIGHTS)

    for metric in METRICS:
        established_df[f'{metric}_w'] = established_df[metric] * established_df['weight']

    established_grouped = established_df.groupby(['player_id', 'name'])[[f'{m}_w' for m in METRICS]].sum()
    established_grouped.columns = METRICS
    established_grouped = established_grouped.reset_index()

    # Normalize both sets
    all_metrics = pd.concat([grouped[METRICS], established_grouped[METRICS]], ignore_index=True)
    scaler_all = StandardScaler().fit(all_metrics)

    breakout_scaled = scaler_all.transform(grouped[METRICS])
    established_scaled = scaler_all.transform(established_grouped[METRICS])

    similarities = cosine_similarity(breakout_scaled, established_scaled)
    most_similar_indices = np.argmax(similarities, axis=1)
    grouped['most_similar_established_player'] = [
        established_grouped.iloc[idx]['name'] for idx in most_similar_indices
    ]
    

    return grouped.reset_index()

def generate_interactive_plot(df):
    
    """
    Generates an interactive Plotly scatter plot.
    - X-axis: Superstar Similarity
    - Y-axis: Breakout Index
    - Hover text: Player names
    Displays the figure in a browser or notebook.
    """

    fig = px.scatter(
        df,
        x='superstar_similarity',
        y='breakout_index',
        text='name',
        title='MLB Breakout Candidates: Breakout Index vs Superstar Similarity',
        labels={"superstar_similarity": "Superstar Similarity", "breakout_index": "Breakout Index"}
    )
    fig.update_traces(textposition='top center')
    fig.show()

def compute_established_similarity(breakout_df, established_df):
    scaler = StandardScaler()
    breakout_scaled = scaler.fit_transform(breakout_df[METRICS])
    established_scaled = scaler.transform(established_df[METRICS])

    similarities = cosine_similarity(breakout_scaled, established_scaled)
    most_similar_indices = similarities.argmax(axis=1)
    best_scores = similarities[np.arange(len(breakout_df)), most_similar_indices]

    breakout_df['most_similar_established'] = established_df.iloc[most_similar_indices]['name'].values
    breakout_df['similarity_to_established'] = best_scores

    return breakout_df

# Main Script
if __name__ == "__main__":
    combined_data = load_and_merge_data(EXPECTED_FILES, EV_FILES)
    grouped_data, raw_data, eligible_ids = calculate_weighted_averages(combined_data)
    final_results = compute_similarity_and_breakout(raw_data, grouped_data)

    # Save results to CSV
    final_results.to_csv("breakout_candidates.csv", index=False)
    print("Top Breakout Candidates:\n")
    print(final_results[['name', 'superstar_similarity', 'breakout_index']])

    # Save full stats data to CSV for use in Streamlit
    combined_data.to_csv("full_stats_combined.csv", index=False)

    # Interactive plot
    generate_interactive_plot(final_results)
