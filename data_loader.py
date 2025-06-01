import pandas as pd
import numpy as np

def load_local_batting_data(batting_path: str = "batting.csv") -> pd.DataFrame:
    """ Loads core batting data from batting.csv, merges Statcast expected stats (xwOBA, xBA, xSLG, etc.) and exit velocity data from separate CSVs for 2023 and 2024.
        fills missing Statcast ages for newer players from external supplemental CSVs (mlb-player-stats-Batters2023.csv and 2024.csv).
    """
    # Load primary batting data
    df = pd.read_csv("batting.csv")
    df["player_id"] = df["player_id"].astype(str)

    # Load expected stats 2023 and 2024
    expected_23 = pd.read_csv("expected_stats 23.csv")
    expected_24 = pd.read_csv("expected_stats 24.csv")
    ev_23 = pd.read_csv("exit_velocity 23.csv")
    ev_24 = pd.read_csv("exit_velocity 24.csv")

    for df_ in [expected_23, expected_24, ev_23, ev_24]:
        df_["player_id"] = df_["player_id"].astype(str)

    # Standardize and combine Statcast supplemental files
    def standardize_and_merge(ev, estats, year):
        df_ev = ev.copy()
        df_estats = estats.copy()

        df_ev["year"] = year
        df_estats["year"] = year

        # For exit_velocity
        df_ev = df_ev.rename(columns={
            "avg_hit_speed": "exit_velocity_avg",
            "avg_hit_angle": "launch_angle_avg",
            "brl_percent": "barrel_batted_rate",
            "ev95percent": "hard_hit_percent"
        })

        # For expected_stats
        df_estats = df_estats.rename(columns={
            "est_woba": "xwoba",
            "est_ba": "xba",
            "est_slg": "xslg"
        })

        merged = pd.merge(df_ev, df_estats, on=["player_id", "last_name, first_name", "year"], how="outer")
        return merged

    merged_23 = standardize_and_merge(ev_23, expected_23, 2023)
    merged_24 = standardize_and_merge(ev_24, expected_24, 2024)
    statcast_combined = pd.concat([merged_23, merged_24], ignore_index=True)

    # Merge with primary batting data
    df = pd.merge(df, statcast_combined, on=["player_id", "last_name, first_name", "year"], how="outer")
    merged_df = df.copy()

    # Sort and reset
    merged_df.sort_values(by=["player_id", "year"], inplace=True)
    merged_df.reset_index(drop=True, inplace=True)

    #Fill missing player_age using supplemental CSVs
    print("Loading supplemental player age data...")

    # Load and combine 2023 and 2024 age data
    supp_2023 = pd.read_csv("mlb-player-stats-Batters2023.csv")
    supp_2024 = pd.read_csv("mlb-player-stats-Batters2024.csv")

    supp_2023["year"] = 2023
    supp_2024["year"] = 2024

    supp_all = pd.concat([supp_2023, supp_2024], ignore_index=True)
    supp_all = supp_all.rename(columns={"Player": "first_last_name", "Age": "age_supplement"})
    supp_all = supp_all[["first_last_name", "year", "age_supplement"]]

    # Create name format to match supplemental
    merged_df["first_last_name"] = merged_df["last_name, first_name"].apply(lambda x: " ".join(x.split(", ")[::-1]))

    # Merge supplemental age info
    merged_df = pd.merge(
        merged_df,
        supp_all,
        on=["first_last_name", "year"],
        how="left"
    )

    # Fill in missing player_age from supplemental
    merged_df["player_age"] = merged_df.apply(
        lambda row: row["age_supplement"] if pd.isna(row["player_age"]) and not pd.isna(row["age_supplement"]) else row["player_age"],
        axis=1
    )

    # Cleanup
    merged_df.drop(columns=["age_supplement", "first_last_name"], inplace=True)

    # merge of _x and _y columns
    for col in ['xba', 'exit_velocity_avg', 'launch_angle_avg', 'xwoba', 'xslg', 'barrel_batted_rate', 'hard_hit_percent']:
        x_col = f"{col}_x"
        y_col = f"{col}_y"
        if x_col in merged_df.columns and y_col in merged_df.columns:
            merged_df[col] = merged_df.apply(
             lambda row: row[y_col] if pd.isna(row[x_col]) or row[x_col] in ['', ' ', 'nan', 'NaN', np.nan] else row[x_col],
                axis=1
         )
        merged_df.drop(columns=[x_col, y_col], inplace=True)
    
    # Drop rows that are missing any key Statcast metric
    required_metrics = [
        'player_age',
        'exit_velocity_avg',
        'launch_angle_avg',
        'barrel_batted_rate',
        'hard_hit_percent',
        'xwoba',
        'xba',
        'xslg',
    ]
    
    merged_df.dropna(subset=required_metrics, inplace=True)

    # Remove duplicate (player_id, year) entries — keep first
    merged_df.sort_values(by=["player_id", "year"], inplace=True)
    merged_df = merged_df.drop_duplicates(subset=["player_id", "year"], keep="first")
    
    return merged_df
