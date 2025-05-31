import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

def compute_similarity_and_breakout(candidate_df: pd.DataFrame, reference_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the similarity of each breakout candidate to established players based on key metrics.
    Generates a similarity score, breakout score, and overall breakout index.
    """
    metrics = ['exit_velocity_avg', 'launch_angle_avg', 'barrel_batted_rate',
               'hard_hit_percent', 'xwoba', 'xba', 'xslg']

    # Compute reference superstar threshold
    superstar_thresholds = reference_df[metrics].quantile(0.95)

    scaler = StandardScaler()
    scaler.fit(reference_df[metrics])

    # Ensure candidate_df is transformed with a DataFrame having matching column names
    candidate_scaled = scaler.transform(candidate_df[metrics])
    superstar_scaled = scaler.transform(pd.DataFrame([superstar_thresholds], columns=metrics))[0]

    # Similarity = negative euclidean distance
    similarity_scores = -np.linalg.norm(candidate_scaled - superstar_scaled, axis=1)
    candidate_df['superstar_similarity'] = similarity_scores

    # Breakout score = average of metrics
    candidate_df['breakout_score'] = candidate_df[metrics].mean(axis=1)

    # Final breakout index
    candidate_df['breakout_index'] = (
        0.7 * candidate_df['breakout_score'] +
        0.3 * candidate_df['superstar_similarity']
    )

    return candidate_df
