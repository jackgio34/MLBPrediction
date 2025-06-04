from sklearn.linear_model import LinearRegression
import pandas as pd
import numpy as np

def build_projection_reference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds a reference dataset of established players who had their first four seasons
    within any 5-year span before 2024. These players are used for comparison against breakout candidates.
    """
    df_pre_2024 = df[df["year"] < 2024].copy()

    eligible_ids = []
    for player_id, group in df_pre_2024.groupby("player_id"):
        years = sorted(group["year"].unique())
        # Look for any span of 5 years containing at least 4 seasons
        for i in range(len(years)):
            window = [y for y in years if years[i] <= y < years[i] + 6]
            if len(window) >= 4:
                eligible_ids.append(player_id)
                break

    reference_df = df_pre_2024[df_pre_2024["player_id"].isin(eligible_ids)].copy()
    reference_df = reference_df.sort_values(by=["last_name, first_name", "year"]).reset_index(drop=True)

    return reference_df


def match_and_project(candidate_df: pd.DataFrame, reference_df: pd.DataFrame, full_data: pd.DataFrame) -> pd.DataFrame:
    """
    Matches breakout candidates to the most similar player from the reference group
    using rank-based metric similarity. Projects their next 4 years using:
    1. Historical stat progression of the matched player
    2. Linear regression trend from the matched player's career
    Works even if candidate has only one year of data.
    """
    metrics = ['exit_velocity_avg', 'launch_angle_avg', 'barrel_batted_rate',
               'hard_hit_percent', 'xwoba', 'xba', 'xslg']

    projections_hist = []
    projections_reg = []

    full_data = full_data.drop_duplicates(subset=['player_id', 'year'])
    
    for _, candidate in candidate_df.iterrows():
        try:
            # Stat-by-stat similarity ranking
            rank_scores = []
            for metric in metrics:
                diffs = (reference_df[metric] - candidate[metric]).abs()
                ranks = diffs.rank(method='min')
                rank_scores.append(ranks)

            total_rank = sum(rank_scores)
            best_match_idx = total_rank.idxmin()
            best_match_id = reference_df.loc[best_match_idx, 'player_id']
            match_name = reference_df.loc[best_match_idx, 'last_name, first_name']

            # Get full career data for the best match
            match_history = full_data[full_data['player_id'] == best_match_id].sort_values('year')
            match_metrics = match_history[metrics + ['year']].copy()

            # Check for duplicated index or year values
            if match_metrics.index.duplicated().any():
                print(f"Duplicate index detected for match: {match_name} (player_id: {best_match_id})")
            if match_metrics['year'].duplicated().any():
                print(f"Duplicate year rows found for match: {match_name} (player_id: {best_match_id})")

            #Remove duplicate years before any indexing
            match_metrics = match_metrics.drop_duplicates(subset='year').reset_index(drop=True)


            # HISTORICAL PROJECTION
            base_year = match_metrics['year'].min() + 1
            base_metrics = match_metrics[match_metrics['year'] == base_year]
            future_metrics = match_metrics[match_metrics['year'] > base_year].copy()

            if not base_metrics.empty and not future_metrics.empty:
                diffs = future_metrics[metrics].reset_index(drop=True) - base_metrics[metrics].values[0]
                if len(diffs) < 4:
                    last_diff = diffs.iloc[-1]
                    extension = pd.DataFrame([last_diff] * (4 - len(diffs)))
                    diffs = pd.concat([diffs, extension], ignore_index=True)
                diffs = diffs.head(4)
            else:
                # fallback: use full-career year-to-year diffs
                diffs = match_metrics[metrics].diff().dropna().reset_index(drop=True)
                if not diffs.empty:
                    last_diff = diffs.iloc[-1]
                    extension = pd.DataFrame([last_diff] * 4)
                    diffs = extension
                else:
                    diffs = pd.DataFrame(np.zeros((4, len(metrics))), columns=metrics)

            # Build the projection
            hist_proj = pd.DataFrame()
            hist_proj['year'] = [2025 + i for i in range(4)]
            for metric in metrics:
                hist_proj[metric] = candidate[metric] + diffs[metric].cumsum()

            # Add identifying info
            hist_proj.insert(0, 'last_name, first_name', candidate['last_name, first_name'])
            hist_proj.insert(1, 'match_name', match_name)
            projections_hist.append(hist_proj)

            # REGRESSION PROJECTION
            if len(match_metrics) >= 2:
                reg_proj = pd.DataFrame()
                reg_proj['year'] = [2025 + i for i in range(4)]
                reg_proj['last_name, first_name'] = candidate['last_name, first_name']
                reg_proj['match_name'] = match_name

                for metric in metrics:
                    lr = LinearRegression()
                    X = match_metrics[['year']]
                    y = match_metrics[[metric]]
                    lr.fit(X, y)
                    future_years = pd.DataFrame({'year': [2024 + i for i in range(4)]})
                    reg_proj[metric] = lr.predict(future_years).flatten()

                projections_reg.append(reg_proj)
        except Exception as e:
            continue

    hist_proj_df = pd.concat(projections_hist, ignore_index=True) if projections_hist else pd.DataFrame()
    reg_proj_df = pd.concat(projections_reg, ignore_index=True) if projections_reg else pd.DataFrame()
    return hist_proj_df, reg_proj_df
