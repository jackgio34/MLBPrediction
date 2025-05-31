import pandas as pd

def filter_breakout_candidates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters for breakout candidates who:
    - Had their first MLB season in 2023 or later
    - Are under age 27 across all seasons (or have missing age)
    - Never ranked in the top 5% for xwOBA, xBA, or xSLG in any single season
    """
    # Step 1: Filter by first recorded season (must be 2023 or later)
    first_years = df.groupby('player_id')['year'].min()
    recent_starts = first_years[first_years >= 2023].index
    df = df[df['player_id'].isin(recent_starts)].copy()

    # Step 2: Filter by age (under 27 in all known seasons or all missing)
    def age_ok(ages):
        return ages.dropna().lt(27).all() or ages.isna().all()

    age_filtered = df.groupby('player_id')['player_age'].apply(age_ok)
    eligible_ids = age_filtered[age_filtered].index
    filtered_df = df[df['player_id'].isin(eligible_ids)].copy()

    # Step 3: Disqualify players who had elite metrics in any year
    disqualify_ids = set()
    for year in filtered_df['year'].unique():
        year_df = filtered_df[filtered_df['year'] == year]
        for metric in ['xwoba', 'xba', 'xslg']:
            threshold = year_df[metric].quantile(0.95)
            elite_players = year_df[year_df[metric] >= threshold]['player_id']
            disqualify_ids.update(elite_players)

    # Step 4: Final filter
    return filtered_df[~filtered_df['player_id'].isin(disqualify_ids)].copy()
