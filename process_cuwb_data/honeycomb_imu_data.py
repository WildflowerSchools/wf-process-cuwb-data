import pandas as pd

from honeycomb_io import (
    fetch_cuwb_position_data,
    fetch_cuwb_accelerometer_data,
    fetch_cuwb_gyroscope_data,
    fetch_cuwb_magnetometer_data,
    add_device_assignment_info,
    add_device_entity_assignment_info,
    add_tray_material_assignment_info,
)

from .uwb_motion_filters import TrayMotionButterFiltFiltFilter
from .utils.util import filter_entity_type


def fetch_imu_data(imu_type, environment_name, start, end, device_ids=None, entity_type="all"):
    if imu_type == "position":
        fetch = fetch_cuwb_position_data
    elif imu_type == "accelerometer":
        fetch = fetch_cuwb_accelerometer_data
    elif imu_type == "gyroscope":
        fetch = fetch_cuwb_gyroscope_data
    elif imu_type == "magnetometer":
        fetch = fetch_cuwb_magnetometer_data
    else:
        raise ValueError(f"Unexpected IMU type: {imu_type}")

    df = fetch(
        start=start,
        end=end,
        device_ids=device_ids,
        environment_id=None,
        environment_name=environment_name,
        device_types=["UWBTAG"],
        output_format="dataframe",
        sort_arguments={"field": "timestamp"},
        chunk_size=20000,
    )
    if len(df) == 0:
        return None

    # Add metadata
    df = add_device_assignment_info(df)
    df = add_device_entity_assignment_info(df)
    df = add_tray_material_assignment_info(df)
    # Filter on entity type
    df = filter_entity_type(df, entity_type=entity_type)

    df["type"] = imu_type
    df.reset_index(drop=True, inplace=True)
    df.set_index("timestamp", inplace=True)

    return df


def smooth_imu_position_data(df_position):
    position_filter = TrayMotionButterFiltFiltFilter(useSosFiltFilt=True)
    df_position_smoothed = pd.DataFrame(data=None, columns=df_position.columns)
    for device_id in df_position["device_id"].unique().tolist():
        df_positions_for_device = df_position.loc[df_position["device_id"] == device_id].copy().sort_index()

        df_positions_for_device["x"] = position_filter.filter(series=df_positions_for_device["x"])
        df_positions_for_device["y"] = position_filter.filter(series=df_positions_for_device["y"])
        df_position_smoothed = pd.concat([df_position_smoothed, df_positions_for_device])
    return df_position_smoothed
