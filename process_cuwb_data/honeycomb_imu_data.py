import datetime
import hashlib
import json
import pathlib

import pandas as pd
from platformdirs import user_cache_dir

from honeycomb_io import (
    fetch_cuwb_position_data,
    fetch_cuwb_accelerometer_data,
    fetch_cuwb_gyroscope_data,
    fetch_cuwb_magnetometer_data,
    add_device_assignment_info,
    add_device_entity_assignment_info,
    add_tray_material_assignment_info,
)

from .honeycomb_service import HoneycombCachingClient
from .utils.log import logger
from .uwb_motion_filters import TrayMotionButterFiltFiltFilter
from .utils import const
from .utils.util import filter_by_entity_type


def fetch_imu_data(
    imu_type,
    environment_name,
    start,
    end,
    device_ids=None,
    entity_type="all",
    use_cache: bool = True,
    cache_directory="/".join([user_cache_dir(appname=const.APP_NAME, appauthor=const.APP_AUTHOR), "uwb_data"]),
):
    file_path = None
    if use_cache:
        file_path = generate_imu_file_path(
            filename_prefix=f"{imu_type}_data",
            start=start,
            end=end,
            device_ids=device_ids,
            environment_name=environment_name,
            entity_type=entity_type,
            cache_directory=cache_directory,
        )
        if file_path.is_file():
            imu_data = pd.read_pickle(file_path)
            logger.info(f"File {file_path} exists locally. Fetching from local")
            return imu_data

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
        logger.warning(f"No IMU {imu_type} data found for {environment_name} between {start} and {end}")
        return None

    # Add metadata
    df = add_device_assignment_info(df)
    df = add_device_entity_assignment_info(df)
    df = add_tray_material_assignment_info(df)

    # Filter on entity type
    df = filter_by_entity_type(df, entity_type=entity_type)

    df["type"] = imu_type
    df.reset_index(drop=True, inplace=True)
    df.set_index("timestamp", inplace=True)

    if use_cache and file_path is not None:
        df.to_pickle(file_path)

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


def generate_imu_file_path(
    filename_prefix,
    start,
    end,
    device_ids=None,
    part_numbers=None,
    serial_numbers=None,
    environment_id=None,
    environment_name=None,
    entity_type=None,
    cache_directory="/".join([user_cache_dir(appname=const.APP_NAME, appauthor=const.APP_AUTHOR), "uwb_data"]),
):
    honeycomb_caching_client = HoneycombCachingClient()

    if environment_id is None:
        if environment_name is None:
            raise ValueError("Must specify either environment ID or environment name")
        environment_id = honeycomb_caching_client.fetch_environment_id(environment_name=environment_name)
    start_string = start.astimezone(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    end_string = end.astimezone(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    if device_ids is None:
        device_ids = honeycomb_caching_client.fetch_device_ids(
            device_types=tuple("UWBTAG"),
            part_numbers=tuple(part_numbers) if part_numbers else None,
            serial_numbers=tuple(serial_numbers) if serial_numbers else None,
            environment_id=environment_id,
            environment_name=None,
            start=start,
            end=end,
        )
    arguments_hash = generate_imu_arguments_hash(
        start=start, end=end, environment_id=environment_id, device_ids=device_ids, entity_type=entity_type
    )
    file_path = pathlib.Path(cache_directory) / ".".join(
        ["_".join([filename_prefix, environment_id, start_string, end_string, arguments_hash]), "pkl"]
    )
    return file_path


def generate_imu_arguments_hash(start, end, environment_id, device_ids, entity_type):
    arguments_normalized = (start.timestamp(), end.timestamp(), environment_id, tuple(sorted(device_ids)), entity_type)
    arguments_serialized = json.dumps(arguments_normalized)
    arguments_hash = hashlib.sha256(arguments_serialized.encode()).hexdigest()
    return arguments_hash
