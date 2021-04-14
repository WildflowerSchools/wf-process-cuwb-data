import pandas as pd

from process_cuwb_data.uwb_motion_enum_carry_categories import CarryCategory
from process_cuwb_data.uwb_motion_enum_groundtruth_data_source import GroundtruthDataSource
from process_cuwb_data.uwb_motion_enum_human_activities import HumanActivity


GROUNDTRUTH_TYPE_TRAY_CARRY = 'tray_carry'
GROUNDTRUTH_TYPE_HUMAN_ACTIVITY = 'human_activity'


def validate_ground_truth(df_groundtruth, groundtruth_type):
    required_columns = ['device_id', 'ground_truth_state', 'start_datetime', 'end_datetime']

    if groundtruth_type == GROUNDTRUTH_TYPE_TRAY_CARRY:
        category_enum_class = CarryCategory
    elif groundtruth_type == GROUNDTRUTH_TYPE_HUMAN_ACTIVITY:
        category_enum_class = HumanActivity
    else:
        return False, "Unknown groundtruth_type ('{}') supplied to validate_ground_truth".format(groundtruth_type)

    # Verify required columns exist
    missing_columns = []
    for rcolumn in required_columns:
        if rcolumn not in df_groundtruth.columns:
            missing_columns.append(rcolumn)

    if len(missing_columns) > 0:
        return False, "Groundtruth data missing column(s) {}".format(missing_columns)

    if 'data_source' not in df_groundtruth.columns:
        df_groundtruth['data_source'] = GroundtruthDataSource.IMU_TABLES.name

    for index, row in df_groundtruth.iterrows():
        try:
            category_enum_class(row['ground_truth_state'])
        except ValueError:
            msg = "Invalid ground_truth_state '{}', valid options include {}".format(
                row['ground_truth_state'], category_enum_class.as_name_list())
            return False, msg

        try:
            GroundtruthDataSource(row['data_source'])
        except ValueError:
            msg = "Invalid data_source '{}', valid options include {}".format(
                row['data_source'], GroundtruthDataSource.as_name_list())
            return False, msg
    return True, ""


def combine_features_with_ground_truth_data(
    df_features,
    df_groundtruth,
    groundtruth_type
    # baseline_state=CarryCategory.NOT_CARRIED.name
):
    # if CarryCategory(baseline_state) is None:
    #     raise Exception(
    #         "Invalid baseline_state '{}', valid options include {}".format(
    #             baseline_state, CarryCategory.as_name_list()))

    df_features_filtered = pd.DataFrame(columns=df_features.columns)

    #df_features['ground_truth_state'] = baseline_state
    for index, row in df_groundtruth.iterrows():
        valid, msg = validate_ground_truth(df_groundtruth, groundtruth_type=groundtruth_type)
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


def combine_features_with_tray_carry_ground_truth_data(df_features, df_groundtruth):
    return combine_features_with_ground_truth_data(
        df_features=df_features, df_groundtruth=df_groundtruth, groundtruth_type=GROUNDTRUTH_TYPE_TRAY_CARRY)


def combine_features_with_human_activity_ground_truth_data(df_features, df_groundtruth):
    return combine_features_with_ground_truth_data(
        df_features=df_features, df_groundtruth=df_groundtruth, groundtruth_type=GROUNDTRUTH_TYPE_HUMAN_ACTIVITY)
