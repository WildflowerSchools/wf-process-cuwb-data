def combine_features_with_ground_truth_data(
    df_features,
    df_ground_truth,
    baseline_state='Not carried',
    inplace=False
):
    if not inplace:
        df_features = df_features.copy()
    df_features['ground_truth_state'] = baseline_state
    for index, row in df_ground_truth.iterrows():
        if row['ground_truth_state'] != baseline_state:
            df_features.loc[
                (
                    (df_features['device_id'] == row['device_id']) &
                    (df_features.index >= row['start_datetime']) &
                    (df_features.index <= row['end_datetime'])
                ),
                'ground_truth_state'
            ] = row['ground_truth_state']
    if not inplace:
        return df_features
