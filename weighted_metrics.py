import pandas as pd

def calculate_weighted_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates weighted averages of offensive metrics for each breakout candidate,
    prioritizing the most recent year (2024 > 2023). Accepts players with data in only one year.
    """
    weights = {2023: 0.4, 2024: 0.6}
    df = df[df['year'].isin(weights.keys())].copy()
    df['weight'] = df['year'].map(weights)

    metrics = ['exit_velocity_avg', 'launch_angle_avg', 'barrel_batted_rate',
               'hard_hit_percent', 'xwoba', 'xba', 'xslg']

    for metric in metrics:
        df[f'{metric}_w'] = df[metric] * df['weight']

    sum_weights = df.groupby(['player_id', 'last_name, first_name'])['weight'].sum().reset_index(name='total_weight')
    weighted_sums = df.groupby(['player_id', 'last_name, first_name'])[[f'{m}_w' for m in metrics]].sum().reset_index()

    result = pd.merge(weighted_sums, sum_weights, on=['player_id', 'last_name, first_name'])
    for metric in metrics:
        result[metric] = result[f'{metric}_w'] / result['total_weight']
        result.drop(columns=[f'{metric}_w'], inplace=True)

    result.drop(columns=['total_weight'], inplace=True)
    return result
