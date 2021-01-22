import pandas as pd

from .uwb_motion_carry_categories import CarryCategory


def validate_ground_truth(df_groundtruth):
    required_columns = ['device_id', 'ground_truth_state', 'start_datetime', 'end_datetime']

    # Verify required columns exist
    missing_columns = []
    for rcolumn in required_columns:
        if rcolumn not in df_groundtruth.columns:
            missing_columns.append(rcolumn)

    if len(missing_columns) > 0:
        return False, "Groundtruth data missing column(s) {}".format(missing_columns)

    for index, row in df_groundtruth.iterrows():
        if CarryCategory(row['ground_truth_state']) is None:
            msg = "Invalid ground_truth_state '{}', valid options include {}".format(
                row['ground_truth_state'], CarryCategory.as_name_list())
            return False, msg

    return True, ""


def combine_features_with_ground_truth_data(
    df_features,
    df_groundtruth,
    baseline_state=CarryCategory.NOT_CARRIED.name
):
    if CarryCategory(baseline_state) is None:
        raise Exception(
            "Invalid baseline_state '{}', valid options include {}".format(
                baseline_state, CarryCategory.as_name_list()))

    df_features_filtered = pd.DataFrame(columns=df_features.columns)

    df_features['ground_truth_state'] = baseline_state
    for index, row in df_groundtruth.iterrows():
        valid, msg = validate_ground_truth(df_groundtruth)
        if not valid:
            raise Exception(msg)

        mask = (
            (df_features['device_id'] == row['device_id']) &
            (df_features.index >= row['start_datetime']) &
            (df_features.index <= row['end_datetime'])
        )

        df_features_masked = df_features.loc[mask].copy()
        df_features_masked['ground_truth_state'] = row['ground_truth_state']
        df_features_filtered = df_features_filtered.append(df_features_masked)

        # if CarryCategory(row['ground_truth_state']) != CarryCategory(baseline_state):
        #     df_features.loc[
        #         (
        #             (df_features['device_id'] == row['device_id']) &
        #             (df_features.index >= row['start_datetime']) &
        #             (df_features.index <= row['end_datetime'])
        #         ),
        #         'ground_truth_state'
        #     ] = row['ground_truth_state']

    return df_features_filtered
